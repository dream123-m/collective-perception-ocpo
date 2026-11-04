"""Sequential, matched MADDPG scenario-grid training and evaluation."""
from __future__ import annotations

import copy
import csv
import json
import random
import statistics
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from major.cross_scene import legacy_layout_experiment as main
from major.cross_scene import legacy_scenario_grid as grid
from major.cross_scene.layout_only import FixedFadingEnv
from major.cross_scene.run import now, sha, state_hash, write_csv, write_json
from major.maddpg import basic_continuous as basic
from major.maddpg.adapter import JointActors
from major.maddpg.original.maddpg_network import CriticNetwork

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'major/maddpg/results/scenario_grid'
SEED = 123
ORDER = [(50, .5), (50, .3), (50, .7), (25, .5), (75, .5),
         (25, .3), (25, .7), (75, .3), (75, .7)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_one(density, penetration, target, smoke_dir=None, *, coverage_first=False,
            output_root=None, episodes=1500, keep_checkpoint=False):
    primary = (density, penetration) == (50, .5)
    source = grid.source_info(density, penetration)
    data = main.DATA if primary else grid.DATA / grid.key(density, penetration)
    bank = main.load_bank(data) if primary else grid.load_bank(data)
    env_cfg = grid.config_for(main, source, target)
    destination = OUT if output_root is None else Path(output_root).resolve()
    if not destination.is_relative_to(ROOT / 'major'):
        raise ValueError('Outputs must remain inside major.')
    folder = (destination if smoke_dir is None else Path(smoke_dir)) / grid.key(density, penetration) / f'target_{target:.2f}'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'checkpoint.pt'
    rec_path = folder / 'run.json'
    result = folder / 'test_summary.json'
    if result.exists():
        row = read(result)
        if (row['status'] != 'completed' or row['manifest_sha256'] != sha(data / 'manifest.json')
                or row.get('coverage_first', False) != coverage_first):
            raise ValueError(f'Incompatible completed result: {folder}')
        return row

    torch.set_num_threads(1)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    cfg = dict(episodes=episodes, episode_steps=env_cfg['episode_steps'], validation_every=100,
               batch_size=32, replay_capacity=2048, warmup_steps=1000,
               updates_per_window=1, gamma=.95, tau=.01, actor_lr=1e-4, critic_lr=1e-3,
               gaussian_noise=.1, noise_decay=.9995, min_noise=.01,
               action_threshold=.5, target_coverage=target)
    if smoke_dir is not None:
        cfg.update(episodes=4, validation_every=2, batch_size=4, warmup_steps=0)
    train = bank['splits']['train']; val = bank['splits']['validation']
    manager = main.adapted() if primary else grid.context(data, source)
    with manager:
        env_config = main.env_config(env_cfg) if primary else main.runner.environment_config(env_cfg)
        envs = [FixedFadingEnv(data / r['path'], seed=0, config=env_config) for r in train]
        env = envs[0]
        actor = JointActors(env.state_dim, env.num_agents, env.max_targets_per_cav)
        critic = CriticNetwork(env.state_dim, env.max_targets_per_cav, env.num_agents, [128, 128])
        target_actor, target_critic = copy.deepcopy(actor), copy.deepcopy(critic)
        ao = torch.optim.Adam(actor.parameters(), lr=cfg['actor_lr'])
        co = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'])
        scene_rng = np.random.default_rng(SEED + 700000)
        replay_rng = np.random.default_rng(SEED + 910000)
        noise_rng = np.random.default_rng(SEED + 300000)
        replay = deque(maxlen=cfg['replay_capacity'])
        identity = dict(seed=SEED, density=density, penetration=penetration, config=cfg,
            manifest_sha256=sha(data / 'manifest.json'), basic_source_sha256=sha(Path(basic.__file__)),
            grid_source_sha256=sha(Path(__file__)), environment_sha256=sha(ROOT / 'major/cross_scene/environment.py'),
            initial_actor_sha256=state_hash(actor), initial_critic_sha256=state_hash(critic))
        if coverage_first:
            identity['selection'] = 'Validation coverage >= target first, then original validation reward; otherwise maximum coverage.'
        histories, validations = [], []
        best, best_ep, steps, updates, noise, start, elapsed = -float('inf'), None, 0, 0, cfg['gaussian_noise'], 1, 0.
        best_key, best_coverage = None, None
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise ValueError(f'Checkpoint identity mismatch: {folder}')
            for obj, key in ((actor, 'actor'), (target_actor, 'target_actor'), (critic, 'critic'),
                             (target_critic, 'target_critic'), (ao, 'ao'), (co, 'co')):
                obj.load_state_dict(saved[key])
            torch.set_rng_state(saved['torch_rng'])
            scene_rng.bit_generator.state = saved['scene_rng']; replay_rng.bit_generator.state = saved['replay_rng']
            noise_rng.bit_generator.state = saved['noise_rng']; replay.extend(saved['replay'])
            histories, validations = saved['histories'], saved['validations']
            best, best_ep, steps, updates, noise = (saved[k] for k in ('best', 'best_ep', 'steps', 'updates', 'noise'))
            best_key = saved.get('best_key'); best_coverage = saved.get('best_coverage')
            start, elapsed = saved['episode'] + 1, saved['elapsed']
        elif rec_path.exists():
            raise RuntimeError(f'Incomplete run has no checkpoint: {folder}')
        record = dict(identity, status='running', started_at_utc=now())
        write_json(rec_path, record)
        started = time.perf_counter()

        def validate(episode):
            nonlocal best, best_ep, best_key, best_coverage
            rows = main.runner.evaluate_frozen(actor, val, data, env_cfg, namespace='validation', channels=1)
            scores = main.runner.summary(rows)
            validations.append(dict(episode=episode, **scores))
            feasible = scores['coverage'] >= target
            candidate = ((1, scores['reward'], scores['coverage']) if feasible else
                         (0, scores['coverage'], scores['reward']))
            better = (best_key is None or candidate > best_key) if coverage_first else scores['reward'] > best
            if updates and better:
                best, best_ep = scores['reward'], episode
                best_key, best_coverage = candidate, scores['coverage']
                torch.save(dict(policy=actor.state_dict(), episode=episode,
                                validation_reward=best, validation_coverage=best_coverage,
                                coverage_first=coverage_first), folder / 'best_trained.pt')

        if start == 1:
            validate(0)
        for episode in range(start, cfg['episodes'] + 1):
            index = int(scene_rng.choice(len(envs), size=1, replace=False)[0]); env = envs[index]
            obs = env.reset(seed=0); rewards, coverage, payload = [], [], []
            for _ in range(cfg['episode_steps']):
                previous = env.previous_actions.copy()
                with torch.no_grad():
                    probs = actor.probabilities(torch.from_numpy(obs)[None])[0].numpy()
                continuous = np.clip(probs + noise_rng.normal(0, noise, probs.shape), 0, 1).astype(np.float32)
                continuous *= env.mask
                nxt, reward, done, info = env.step((continuous > .5) & env.mask)
                replay.append((index, np.packbits(previous.ravel()), continuous, float(np.mean(reward))))
                obs = nxt; steps += 1
                rewards.append(float(np.mean(reward))); coverage.append(float(info['coverage_rate'])); payload.append(float(info['payload']))
            if not done:
                raise AssertionError('Expected collection window to end.')
            if steps >= cfg['warmup_steps'] and len(replay) >= cfg['batch_size']:
                batch = basic.sample_batch(replay, replay_rng, envs, cfg['batch_size'])
                basic.update(actor, target_actor, critic, target_critic, ao, co, batch, cfg['gamma'], cfg['tau'])
                updates += 1
            noise = max(cfg['min_noise'], noise * cfg['noise_decay'])
            histories.append(dict(episode=episode, scene_id=train[index]['scene_id'],
                reward=float(np.mean(rewards)), coverage=float(np.mean(coverage)), payload=float(np.mean(payload))))
            if episode % cfg['validation_every'] == 0 or episode == cfg['episodes']:
                validate(episode)
                elapsed_now = elapsed + time.perf_counter() - started
                record.update(completed_episodes=episode, joint_steps=steps, optimizer_updates=updates,
                    selected_episode=best_ep, best_validation_reward=best if best_ep else None,
                    selected_validation_coverage=best_coverage, coverage_first=coverage_first, elapsed_seconds=elapsed_now)
                write_csv(folder / 'training.csv', histories); write_csv(folder / 'validation.csv', validations)
                saved = dict(identity=identity, episode=episode, actor=actor.state_dict(), target_actor=target_actor.state_dict(),
                    critic=critic.state_dict(), target_critic=target_critic.state_dict(), ao=ao.state_dict(), co=co.state_dict(),
                    torch_rng=torch.get_rng_state(), scene_rng=scene_rng.bit_generator.state,
                    replay_rng=replay_rng.bit_generator.state, noise_rng=noise_rng.bit_generator.state,
                    replay=list(replay), histories=histories, validations=validations, best=best, best_ep=best_ep,
                    best_key=best_key, best_coverage=best_coverage, steps=steps, updates=updates, noise=noise, elapsed=elapsed_now)
                temp = folder / 'checkpoint.pending.pt'; torch.save(saved, temp); temp.replace(checkpoint)
                write_json(rec_path, record)
                print(f'{density}/{penetration}/{target}: {episode}/{cfg["episodes"]}, {elapsed_now:.1f}s', flush=True)
        if best_ep is None:
            raise AssertionError('No trained model was selected.')
        selected = torch.load(folder / 'best_trained.pt', map_location='cpu', weights_only=False)
        actor.load_state_dict(selected['policy'], strict=True)
        rows = main.runner.evaluate_frozen(actor, bank['splits']['test'], data, env_cfg, namespace='grid', channels=1)
        if len(rows) != 32:
            raise AssertionError('Expected 32 test layouts.')
        reference_path = grid.BASE / 'single_seed/exploratory_test/per_layout.csv'
        with reference_path.open(encoding='utf-8-sig', newline='') as f:
            refs = [r for r in csv.DictReader(f) if int(r['density']) == density and float(r['penetration']) == penetration
                    and float(r['target']) == target and r['method'] == 'mappo']
        fields = ('scene_id', 'scene_sha256', 'channel_sha256', 'unit_fading_sha256', 'role_sha256', 'steps')
        if len(refs) != 32 or any(any(str(a[k]) != str(b[k]) for k in fields) for a, b in zip(rows, refs)):
            raise AssertionError('Test layout, fading, roles or steps differ from MAPPO.')
        reference_run = (main.OUTPUT / f'mappo_seed_{SEED}' if primary and target == .95
                         else grid.TRAIN / grid.key(density, penetration) / f'target_{target:.2f}' / f'mappo_seed_{SEED}')
        with (reference_run / 'training.csv').open(encoding='utf-8-sig', newline='') as f:
            expected = [r['scene_id'] for r in csv.DictReader(f)]
        shared_length = min(len(histories), len(expected))
        if [r['scene_id'] for r in histories[:shared_length]] != expected[:shared_length]:
            raise AssertionError('Training layout sequence differs from MAPPO.')
        write_csv(folder / 'test.csv', rows)
        record.update(status='completed', finished_at_utc=now(), checkpoint_sha256=sha(folder / 'best_trained.pt'))
        write_json(rec_path, record)
        result_row = dict(status='completed', density=density, penetration=penetration, target=target, seed=SEED,
            manifest_sha256=identity['manifest_sha256'], selected_episode=best_ep, elapsed_seconds=record['elapsed_seconds'],
            coverage_first=coverage_first, selected_validation_coverage=best_coverage,
            validation_feasible=best_coverage >= target, training_episodes=cfg['episodes'],
            **{key: statistics.mean(float(r[key]) for r in rows) for key in ('reward', 'coverage', 'payload')})
        write_json(result, result_row)
        if not keep_checkpoint:
            checkpoint.unlink()
        return result_row


def execute():
    OUT.mkdir(parents=True, exist_ok=True)
    # Reuse the main-scenario test generated by the identical continuous baseline.
    ref = read(basic.OUT / 'test_seed_123_summary.json')
    main_run = read(basic.OUT / 'run.json')
    if main_run['source_sha256'] != sha(Path(basic.__file__)) or not ref['same_training_layout_sequence']:
        raise ValueError('Main MADDPG run is incompatible.')
    rows = [dict(status='completed', density=50, penetration=.5, target=.95, seed=SEED,
                 reused=True, **ref['maddpg'])]
    for density, penetration in ORDER:
        for target in (.95, .90):
            if (density, penetration, target) == (50, .5, .95):
                continue
            write_json(OUT / 'status.json', dict(status='running', completed=len(rows), total=18,
                active=dict(density=density, penetration=penetration, target=target), updated_at_utc=now()))
            rows.append(run_one(density, penetration, target))
            write_json(OUT / 'summary.json', dict(status='running', results=rows, updated_at_utc=now()))
    write_json(OUT / 'summary.json', dict(status='completed', results=rows, finished_at_utc=now()))
    write_json(OUT / 'status.json', dict(status='completed', completed=len(rows), total=18, finished_at_utc=now()))


if __name__ == '__main__':
    if sys.argv[1:] == ['--smoke']:
        with tempfile.TemporaryDirectory(prefix='grid_smoke_', dir=ROOT / 'major/maddpg') as temporary:
            for d, p, target in ((50, .5, .9), (25, .3, .95), (75, .7, .95)):
                row = run_one(d, p, target, smoke_dir=temporary)
                assert row['status'] == 'completed'
        print('Primary, sparse, and dense scenario checks passed.', flush=True)
    elif not sys.argv[1:]:
        execute()
    else:
        raise SystemExit('Usage: python -m major.maddpg.scenario_grid [--smoke]')
