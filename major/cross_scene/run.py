"""Matched MAPPO/OCPO training with disjoint layouts and sealed final testing.

Commands: prepare, check, train, freeze, test, batch. Old experiments are never
loaded as initialization. Test layouts are only evaluated after all six runs
are complete and their validation-selected checkpoint hashes are frozen.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import scipy
import torch
from torch.distributions import Bernoulli

from OCPO.update import clipped_actor_loss, factor_logp, groups_for
from MAPPO.mappo_network import CriticNetwork
from major.model import JointPolicy
from major.cross_scene.data import generate_bank, load_manifest
from major.cross_scene.environment import CrossSceneEnv, DEFAULT_CONFIG as ENV_DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEEDS = (42, 123, 456)
METHODS = ('mappo', 'ocpo')
SOURCE_NAMES = (
    'major/cross_scene/data.py', 'major/cross_scene/environment.py',
    'major/cross_scene/run.py', 'major/model.py', 'MAPPO/mappo_network.py',
    'OCPO/update.py',
)
DEFAULT_CONFIG = dict(
    episodes=1500, episode_steps=8, validation_every=100,
    validation_channels_per_layout=2, test_channels_per_layout=4,
    gamma=.95, gae_lambda=.95, clip=.1, epochs=10,
    actor_lr=.0005, critic_lr=.0005, entropy=.01, max_grad_norm=.5,
    actor_hidden=[64, 64], critic_hidden=[128, 128],
    target_coverage=.95, coverage_reward_sigma=.04,
    cpu_threads=1,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(policy):
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = Path(path).with_name(Path(path).name + '.pending')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def channel_seed(namespace, scene_id, replicate, run_seed=None):
    # No Python hash(), mutable global RNG, method name or action history.
    key = json.dumps([20261025, namespace, scene_id, replicate, run_seed])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & ((1 << 63) - 1)


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_NAMES}


def environment_config(config):
    return {key: config[key] for key in ('episode_steps', 'target_coverage', 'coverage_reward_sigma')}


def feasibility_check(manifest, data, config):
    """Audit achievable coverage on train/validation only; never filter scenes."""
    result = {}
    for split in ('train', 'validation'):
        coverage, success = [], []
        for scene in manifest['splits'][split]:
            env = CrossSceneEnv(data / scene['path'], seed=0, config=environment_config(config))
            for replicate in range(2):
                env.reset(seed=channel_seed('preflight_' + split, scene['scene_id'], replicate))
                coverage.append(env.all_on_coverage())
                success.append(sum(env.communication_success.values()) / env.num_agents)
        result[split] = dict(layouts=len(manifest['splits'][split]), channel_episodes=len(coverage),
                             all_on_mean=float(np.mean(coverage)), all_on_min=float(np.min(coverage)),
                             fraction_all_on_ge_target=float(np.mean(np.array(coverage) >= config['target_coverage'])),
                             sender_success_mean=float(np.mean(success)))
    return result


def prepare_protocol(data, output, config):
    manifest = load_manifest(data, verify=True)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        schema_version=1, data_root=str(data.resolve()),
        data_manifest_sha256=sha(data / 'manifest.json'),
        split_counts=manifest['split_counts'], config=config,
        environment={**ENV_DEFAULT_CONFIG, **environment_config(config)},
        methods=list(METHODS), seeds=list(SEEDS), code_sha256=source_hashes(),
        checkpoint_selection='Maximum validation mean reward; ties retain earlier checkpoint; test never used.',
        actor='Independent per-CAV MLPs, identical input encoding and architecture across methods.',
        roles='Fixed CAV lane/rank roles across independent layouts; evaluates unseen geometry at fixed population.',
        input='Geometry-ordered local objects, fixed padding, relative coordinates and co-perception fraction; no channel outcomes.',
        objective='Original information, coverage and communication reward with self-targets excluded.',
        mappo='Individual GAE advantages and whole-vehicle clipped likelihood ratio.',
        ocpo='Mean-team GAE advantage and disjoint same-object pair/singleton clipped ratios.',
        channels='Independent directed block fading per episode; shared link matrix for desired/interference; matched methods.',
        test='One frozen evaluation over whole layouts excluded from training and validation; no optimizer, no adaptation.',
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     torch=torch.__version__, platform=platform.platform()),
    )
    path = output / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        for key in ('data_manifest_sha256', 'config', 'code_sha256', 'seeds', 'methods'):
            if old[key] != protocol[key]:
                raise ValueError(f'Refusing to mix incompatible experiment: {key}')
        return manifest, old
    protocol['preflight_train_validation_only'] = feasibility_check(manifest, data, config)
    write_json(path, protocol)
    for name in SOURCE_NAMES:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return manifest, protocol


def make_policy(env, config):
    return JointPolicy(env.state_dim, env.num_agents, env.max_targets_per_cav,
                       config['actor_hidden'], use_editor=False)


def rollout_probabilities(policy, observations):
    """Replay the exact single-transition forward used by action collection.

    A larger float32 einsum batch can select a different reduction kernel.
    Near a saturated Bernoulli probability, a tiny probability difference is
    amplified in log likelihood. Preserve the behavior-policy numerical path
    rather than relaxing the PPO likelihood consistency check.
    """
    if observations.ndim != 3 or len(observations) == 0:
        raise ValueError('Expected nonempty [steps, agents, features] observations.')
    return torch.cat([policy.probabilities(state[None]) for state in observations], dim=0)


def gae(reward, values, bootstrap, dones, gamma, lam):
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(reward[0])
    following = bootstrap
    for index in reversed(range(len(reward))):
        live = (~dones[index]).float()
        delta = reward[index] + gamma * live * following - values[index]
        running = delta + gamma * lam * live * running
        advantage[index] = running
        following = values[index]
    return advantage, advantage + values


def normalized(advantage):
    return (advantage - advantage.mean(0)) / (advantage.std(0, unbiased=False) + 1e-8)


def evaluate_frozen(policy, records, data, config, *, namespace, channels):
    before = state_hash(policy)
    rng_before = torch.get_rng_state().clone()
    was_training = policy.training
    rows = []
    policy.eval()
    try:
        with torch.inference_mode():
            for record in records:
                env = CrossSceneEnv(data / record['path'], seed=0, config=environment_config(config))
                if env.state_dim != policy.weights[0].shape[-1] or env.num_agents != policy.num_agents:
                    raise ValueError('Cross-layout policy interface mismatch.')
                mask = torch.from_numpy(env.mask)
                for replicate in range(channels):
                    seed = channel_seed(namespace, record['scene_id'], replicate)
                    obs = torch.from_numpy(env.reset(seed=seed))
                    rewards, coverage, payload = [], [], []
                    bound = float(env.all_on_coverage())
                    channel_hash = env.channel_hash
                    for step in range(config['episode_steps']):
                        actions = (policy.probabilities(obs[None])[0] > .5) & mask
                        nxt, reward, done, info = env.step(actions.numpy())
                        rewards.append(float(np.mean(reward)))
                        coverage.append(float(info['coverage_rate']))
                        payload.append(float(actions.sum()) / env.num_agents)
                        obs = torch.from_numpy(nxt)
                        if done:
                            break
                    rows.append(dict(scene_id=record['scene_id'], scene_sha256=record['scene_sha256'],
                                     channel_seed=seed, channel_sha256=channel_hash, replicate=replicate,
                                     reward=float(np.mean(rewards)), coverage=float(np.mean(coverage)),
                                     payload=float(np.mean(payload)), all_on_coverage=bound, steps=len(rewards)))
    finally:
        policy.train(was_training)
    if before != state_hash(policy) or not torch.equal(rng_before, torch.get_rng_state()):
        raise AssertionError('Frozen evaluation changed model parameters/buffers or training RNG.')
    return rows


def summary(rows):
    # Each layout receives equal weight, regardless of its number of objects.
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ('reward', 'coverage', 'payload', 'all_on_coverage')}


def run_training(method, seed, data, output, config, *, resume=False):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError('Unknown method or undeclared training seed.')
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    manifest, protocol = prepare_protocol(data, output, config)
    if (output / 'freeze.json').exists():
        raise RuntimeError('Experiment is frozen; training is forbidden.')
    destination = output / f'{method}_seed_{seed}'
    destination.mkdir(exist_ok=True)
    if (destination / 'run.json').exists() and not resume:
        raise FileExistsError(f'Run exists: {destination}; use --resume.')
    envs = [CrossSceneEnv(data / r['path'], seed=0, config=environment_config(config))
            for r in manifest['splits']['train']]
    if len({(e.state_dim, e.num_agents, e.max_targets_per_cav) for e in envs}) != 1:
        raise ValueError('Training scenes have inconsistent fixed interfaces.')
    template = envs[0]
    policy = make_policy(template, config)
    initial = state_hash(policy)
    critic = CriticNetwork(template.total_vehicles, config['critic_hidden'])
    actor_optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=config['actor_lr'])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config['critic_lr'])
    scene_rng = np.random.default_rng(seed + 700000)
    pair_rng = np.random.default_rng(seed + 800000)
    history, validations, validation_rows = [], [], []
    best_reward, best_episode, start = -float('inf'), None, 1
    record = dict(method=method, seed=seed, status='running', pid=os.getpid(),
                  started_at_utc=now(), config=config, data_manifest_sha256=protocol['data_manifest_sha256'],
                  code_sha256=protocol['code_sha256'], initial_actor_sha256=initial,
                  initialization='Random, from scratch; no old checkpoint loaded.',
                  agents=template.num_agents, objects=template.total_vehicles,
                  state_dim=template.state_dim, action_dim=template.max_targets_per_cav)
    checkpoint_path = destination / 'checkpoint.pt'
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        for key in ('method', 'seed', 'config', 'data_manifest_sha256', 'code_sha256', 'initial_actor_sha256'):
            if saved['record'][key] != record[key]:
                raise ValueError(f'Resume mismatch: {key}')
        policy.load_state_dict(saved['policy']); critic.load_state_dict(saved['critic'])
        actor_optimizer.load_state_dict(saved['actor_optimizer']); critic_optimizer.load_state_dict(saved['critic_optimizer'])
        torch.set_rng_state(saved['torch_rng']); scene_rng.bit_generator.state = saved['scene_rng']
        pair_rng.bit_generator.state = saved['pair_rng']
        history, validations, validation_rows = saved['history'], saved['validations'], saved['validation_rows']
        best_reward, best_episode = saved['best_reward'], saved['best_episode']
        start = saved['episode'] + 1
        record['resumed_from_episode'] = saved['episode']
    elif resume and (destination / 'run.json').exists():
        raise FileNotFoundError('Resume requested but no resumable checkpoint exists.')
    write_json(destination / 'run.json', record)
    started = time.perf_counter()
    cfg = config

    def validate(episode):
        nonlocal best_reward, best_episode, validation_rows
        validation_rows = evaluate_frozen(policy, manifest['splits']['validation'], data, cfg,
                                         namespace='validation', channels=cfg['validation_channels_per_layout'])
        scores = summary(validation_rows)
        validations.append(dict(episode=episode, **scores))
        if episode > 0 and scores['reward'] > best_reward:
            best_reward, best_episode = scores['reward'], episode
            temporary = destination / 'best.pending.pt'
            torch.save(dict(policy=policy.state_dict(), episode=episode,
                            validation_reward=best_reward, record=record), temporary)
            temporary.replace(destination / 'best.pt')
        print(f'{method} seed={seed} validation episode={episode} reward={scores["reward"]:.5f}', flush=True)

    if start == 1:
        validate(0)
    for episode in range(start, cfg['episodes'] + 1):
        index = int(scene_rng.integers(len(envs)))
        env, scene = envs[index], manifest['splits']['train'][index]
        cseed = channel_seed('train', scene['scene_id'], episode, seed)
        obs = torch.from_numpy(env.reset(seed=cseed))
        mask = torch.from_numpy(env.mask)
        rows = []
        with torch.no_grad():
            for step in range(cfg['episode_steps']):
                broadcast, coverage = env.global_state()
                broadcast = torch.from_numpy(broadcast)
                coverage = torch.tensor([coverage], dtype=torch.float32)
                value = critic(broadcast[None], coverage[None]).reshape(())
                probabilities = policy.probabilities(obs[None])[0]
                actions = Bernoulli(probabilities).sample().bool() & mask
                bits = Bernoulli(probabilities.double()).log_prob(actions.double()) * mask
                groups = groups_for(env, pair_rng) if method == 'ocpo' else None
                old = factor_logp(bits[None], groups[None])[0] if groups is not None else bits.sum(-1)
                nxt, rewards, done, info = env.step(actions.numpy())
                rows.append(dict(obs=obs, action=actions, old=old, value=value,
                                 reward=torch.from_numpy(np.asarray(rewards, dtype=np.float32)),
                                 done=torch.tensor(done), broadcast=broadcast, coverage=coverage,
                                 groups=groups, observed_coverage=float(info['coverage_rate']),
                                 payload=float(actions.sum()) / env.num_agents))
                obs = torch.from_numpy(nxt)
                if done:
                    break
            # The configured rollout is a complete finite-horizon episode.
            if not done:
                raise AssertionError('Rollout ended before the declared episode horizon.')
            batch = {key: torch.stack([row[key] for row in rows])
                     for key in ('obs', 'action', 'old', 'value', 'reward', 'done', 'broadcast', 'coverage')}
            if method == 'ocpo':
                groups = torch.stack([row['groups'] for row in rows])
                advantage, returns = gae(batch['reward'].mean(-1), batch['value'], torch.tensor(0.),
                                         batch['done'], cfg['gamma'], cfg['gae_lambda'])
            else:
                advantage, individual_returns = gae(batch['reward'], batch['value'][:, None],
                    torch.zeros(env.num_agents), batch['done'], cfg['gamma'], cfg['gae_lambda'])
                returns = individual_returns.mean(-1)
            advantage = normalized(advantage)
        for epoch in range(cfg['epochs']):
            probabilities = rollout_probabilities(policy, batch['obs'])
            distribution = Bernoulli(probabilities.double())
            bits = distribution.log_prob(batch['action'].double()) * mask
            new = factor_logp(bits, groups) if method == 'ocpo' else bits.sum(-1)
            if epoch == 0:
                torch.testing.assert_close(new, batch['old'], atol=1e-10, rtol=1e-10)
            ratio = (new - batch['old']).exp()
            if method == 'ocpo':
                actor_loss = clipped_actor_loss(ratio, advantage, cfg['clip'])
            else:
                actor_loss = -torch.minimum(ratio * advantage,
                    ratio.clamp(1 - cfg['clip'], 1 + cfg['clip']) * advantage).mean(0).sum()
            entropy = (distribution.entropy() * mask).sum(-1) / mask.sum(-1).clamp_min(1)
            actor_loss -= cfg['entropy'] * entropy.mean(0).sum()
            actor_optimizer.zero_grad(); actor_loss.backward()
            policy.clip_gradients(cfg['max_grad_norm']); actor_optimizer.step()
            critic_loss = .5 * (critic(batch['broadcast'], batch['coverage']).flatten() - returns).square().mean()
            critic_optimizer.zero_grad(); critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg['max_grad_norm']); critic_optimizer.step()
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                raise FloatingPointError('Nonfinite PPO update.')
        history.append(dict(episode=episode, scene_id=scene['scene_id'], channel_seed=cseed,
                            reward=float(batch['reward'].mean()),
                            coverage=float(np.mean([r['observed_coverage'] for r in rows])),
                            payload=float(np.mean([r['payload'] for r in rows])),
                            all_on_coverage=float(env.all_on_coverage()),
                            actor_loss=float(actor_loss.detach() / env.num_agents),
                            critic_loss=float(critic_loss.detach())))
        if episode % cfg['validation_every'] == 0 or episode == cfg['episodes']:
            validate(episode)
            saved = dict(policy=policy.state_dict(), critic=critic.state_dict(),
                         actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                         record=record, episode=episode, history=history, validations=validations,
                         validation_rows=validation_rows, best_reward=best_reward, best_episode=best_episode,
                         torch_rng=torch.get_rng_state(), scene_rng=scene_rng.bit_generator.state,
                         pair_rng=pair_rng.bit_generator.state)
            temporary = destination / 'checkpoint.pending.pt'; torch.save(saved, temporary)
            temporary.replace(checkpoint_path)
            write_csv(destination / 'training.csv', history)
            write_csv(destination / 'validation.csv', validations)
            write_csv(destination / 'validation_channels.csv', validation_rows)
            record.update(completed_episodes=episode, best_validation_reward=best_reward,
                          selected_episode=best_episode, elapsed_seconds=time.perf_counter() - started,
                          final_training_reward=float(np.mean([r['reward'] for r in history[-100:]])))
            write_json(destination / 'run.json', record)
            print(f'{method} seed={seed} {episode}/{cfg["episodes"]} elapsed={record["elapsed_seconds"]:.1f}s', flush=True)
    record.update(status='completed', completed_episodes=len(history),
                  best_validation_reward=best_reward, selected_episode=best_episode,
                  final_training_reward=float(np.mean([r['reward'] for r in history[-100:]])),
                  finished_at_utc=now(), selected_checkpoint_sha256=sha(destination / 'best.pt'))
    write_json(destination / 'run.json', record)
    return record


def freeze(output, data, config):
    _, protocol = prepare_protocol(data, output, config)
    runs = []
    for seed in SEEDS:
        initial = []
        for method in METHODS:
            folder = output / f'{method}_seed_{seed}'
            record = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            if record['status'] != 'completed' or record['completed_episodes'] != config['episodes']:
                raise RuntimeError('All six full training runs must finish before freezing.')
            initial.append(record['initial_actor_sha256'])
            checkpoint_sha = sha(folder / 'best.pt')
            if checkpoint_sha != record['selected_checkpoint_sha256']:
                raise ValueError('Selected checkpoint was modified.')
            runs.append(dict(method=method, seed=seed, checkpoint=str((folder / 'best.pt').relative_to(output)),
                             checkpoint_sha256=checkpoint_sha, selected_episode=record['selected_episode']))
        if len(set(initial)) != 1:
            raise AssertionError('Matched methods did not start from identical actor initialization.')
    frozen = dict(data_manifest_sha256=protocol['data_manifest_sha256'], code_sha256=protocol['code_sha256'],
                  config=config, runs=runs)
    path = output / 'freeze.json'
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError('Existing freeze differs; cannot alter a frozen experiment.')
    write_json(path, frozen)
    return frozen


def final_test(data, output, config, *, resume=False):
    manifest, protocol = prepare_protocol(data, output, config)
    frozen = json.loads((output / 'freeze.json').read_text(encoding='utf-8'))
    if (frozen['code_sha256'] != source_hashes()
            or frozen['data_manifest_sha256'] != sha(data / 'manifest.json')
            or frozen['config'] != config):
        raise ValueError('Frozen source/data hashes changed.')
    marker = output / 'test_started.json'
    if marker.exists() and not resume:
        raise RuntimeError('Final test was already started; refusing repeated adaptive testing.')
    # Validate every checkpoint BEFORE recording the start of the sealed test.
    for run in frozen['runs']:
        if sha(output / run['checkpoint']) != run['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint hash mismatch.')
    if marker.exists():
        if json.loads(marker.read_text())['freeze_sha256'] != sha(output / 'freeze.json'):
            raise ValueError('Cannot resume a changed frozen test.')
    else:
        write_json(marker, dict(started_at_utc=now(), freeze_sha256=sha(output / 'freeze.json')))
    aggregates, reference_channels = [], None
    for run in frozen['runs']:
        folder = output / f'{run["method"]}_seed_{run["seed"]}'
        if resume and (folder / 'test_summary.json').exists():
            record = json.loads((folder / 'test_summary.json').read_text())
            if record['checkpoint_sha256'] != run['checkpoint_sha256'] or record['test_csv_sha256'] != sha(folder / 'test.csv'):
                raise ValueError('Previously completed test artifact changed.')
            with (folder / 'test.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row['channel_seed'] = int(row['channel_seed'])
        else:
            saved = torch.load(output / run['checkpoint'], map_location='cpu', weights_only=False)
            env = CrossSceneEnv(data / manifest['splits']['test'][0]['path'], seed=0, config=environment_config(config))
            policy = make_policy(env, config); policy.load_state_dict(saved['policy'])
            rows = evaluate_frozen(policy, manifest['splits']['test'], data, config,
                                   namespace='test', channels=config['test_channels_per_layout'])
            write_csv(folder / 'test.csv', rows)
            record = dict(**run, **summary(rows), policy_sha256=state_hash(policy),
                          evaluation='Frozen local deterministic decisions; no adaptation.', test_rows=len(rows),
                          test_csv_sha256=sha(folder / 'test.csv'))
            write_json(folder / 'test_summary.json', record)
        conditions = [(r['scene_sha256'], r['channel_seed'], r['channel_sha256']) for r in rows]
        if reference_channels is None:
            reference_channels = conditions
        elif conditions != reference_channels:
            raise AssertionError('Methods/seeds did not use identical test conditions.')
        aggregates.append(record)
    means = {}
    for method in METHODS:
        records = [r for r in aggregates if r['method'] == method]
        means[method] = {key: dict(mean=float(np.mean([r[key] for r in records])),
                                   sample_sd=float(np.std([r[key] for r in records], ddof=1)))
                         for key in ('reward', 'coverage', 'payload')}
    result = dict(status='completed', finished_at_utc=now(), runs=aggregates, aggregate=means,
                  reward_change=means['ocpo']['reward']['mean'] - means['mappo']['reward']['mean'],
                  paired_conditions_identical=True, test_layouts=len(manifest['splits']['test']))
    write_json(output / 'test_summary.json', result)
    return result


def acquire_batch_lock(output):
    """OS-backed lock releases automatically even if a worker crashes."""
    handle = (output / 'batch.lock').open('a+b')
    if handle.seek(0, 2) == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('Another worker holds this experiment batch lock.') from None
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'train', 'freeze', 'test', 'batch'))
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=HERE / 'results' / 'main')
    parser.add_argument('--method', choices=METHODS, default='mappo')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    config = copy.deepcopy(DEFAULT_CONFIG); config['episodes'] = args.episodes
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if not args.data.is_relative_to(HERE) or not args.output.is_relative_to(HERE):
        parser.error('Data and outputs must stay under major/cross_scene.')
    if args.episodes != 1500 and args.output == HERE / 'results' / 'main':
        parser.error('Short verification runs require a separate temporary output directory.')
    torch.set_num_threads(config['cpu_threads'])
    if args.command == 'prepare':
        manifest = generate_bank(args.data)
        print(json.dumps({'split_counts': manifest['split_counts']}), flush=True)
    elif args.command == 'check':
        manifest = load_manifest(args.data, verify=True)
        print(json.dumps({'verified_splits': manifest['split_counts']}), flush=True)
    elif args.command == 'train':
        print(json.dumps(run_training(args.method, args.seed, args.data, args.output, config, resume=args.resume)), flush=True)
    elif args.command == 'freeze':
        print(json.dumps(freeze(args.output, args.data, config)), flush=True)
    elif args.command == 'test':
        print(json.dumps(final_test(args.data, args.output, config, resume=args.resume)), flush=True)
    else:
        if args.episodes != 1500:
            parser.error('Formal batch requires 1500 episodes.')
        prepare_protocol(args.data, args.output, config)
        lock = acquire_batch_lock(args.output)
        status = dict(status='running', pid=os.getpid(), started_at_utc=now(), completed=[])
        try:
            for seed in SEEDS:
                for method in METHODS:
                    name = f'{method}_seed_{seed}'
                    run_path = args.output / name / 'run.json'
                    if run_path.exists() and json.loads(run_path.read_text())['status'] == 'completed':
                        status['completed'].append(name); continue
                    status['active'] = name
                    write_json(args.output / 'status.json', status)
                    run_training(method, seed, args.data, args.output, config, resume=run_path.exists())
                    status['completed'].append(name)
            freeze(args.output, args.data, config)
            status['active'] = 'sealed_test'; write_json(args.output / 'status.json', status)
            final_test(args.data, args.output, config, resume=(args.output / 'test_started.json').exists())
            status.update(status='completed', active=None, finished_at_utc=now())
            write_json(args.output / 'status.json', status)
        except BaseException:
            status.update(status='failed', error=traceback.format_exc(), finished_at_utc=now())
            write_json(args.output / 'status.json', status)
            raise
        finally:
            lock.close()


if __name__ == '__main__':
    main()
