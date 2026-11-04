"""Controlled layout-context critic experiment; actor and PPO updates are unchanged.

The original runner and its frozen results stay intact. This variant adds four
geometry-only features per object to BOTH methods' centralized critics, retains
the same initial value predictions, and completes training/validation only.
Final testing requires a separate, previously unused layout bank.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
import torch
from torch import nn
from torch.distributions import Bernoulli

from MAPPO.mappo_network import CriticNetwork
from .environment import CrossSceneEnv, DEFAULT_CONFIG as ENV_DEFAULT_CONFIG
from .data import load_manifest
from .run import (
    HERE, ROOT, METHODS, SEEDS, SOURCE_NAMES as BASE_SOURCE_NAMES,
    DEFAULT_CONFIG as BASE_CONFIG, now, sha, state_hash, write_json, write_csv,
    channel_seed, environment_config, feasibility_check, make_policy,
    rollout_probabilities, gae, normalized, groups_for, factor_logp,
    evaluate_frozen, summary, acquire_batch_lock,
)

DEFAULT_CONFIG = copy.deepcopy(BASE_CONFIG)
DEFAULT_OUTPUT = HERE / 'results' / 'layout_critic'
SOURCE_NAMES = (*BASE_SOURCE_NAMES, 'major/cross_scene/layout_critic.py')
FEATURE_NAMES = (
    'perceiving_CAV_count / CAV_count',
    'nearest_perceiving_CAV_distance / perception_range',
    'mean_perceiving_CAV_distance / perception_range',
    'mean_local_co_perception_fraction_across_perceiving_CAVs',
)


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_NAMES}


def object_layout_features(env):
    """Static [objects,4] geometry; zero rows for objects with no perceiver.

    No channel powers, delivery outcomes, selected actions or reward labels are
    used. Count distinguishes absent objects from zero-valued distance features.
    The final feature averages the co-perception fraction already available in
    each candidate sender's local observation.
    """
    visible = env.target_bool
    count = visible.sum(axis=0)
    distance = env.distance_matrix[env.cav_indices] / env.perception_distance
    nearest = np.where(visible, distance, np.inf).min(axis=0)
    nearest[count == 0] = 0.
    mean_distance = np.where(visible, distance, 0.).sum(axis=0) / np.maximum(count, 1)
    overlap = np.zeros(env.total_vehicles, dtype=np.float64)
    for agent in range(env.num_agents):
        valid = env.mask[agent]
        overlap[env.slot_targets[agent, valid]] += env.co_perception[agent, valid]
    overlap /= np.maximum(count, 1)
    features = np.column_stack((count / env.num_agents, nearest, mean_distance, overlap))
    features = features.astype(np.float32)
    if not np.isfinite(features).all() or np.any(features < 0) or np.any(features > 1):
        raise AssertionError('Invalid normalized layout features.')
    return features


class LayoutCriticEnv(CrossSceneEnv):
    """The simulator is unchanged; only the centralized critic state is extended."""

    def __init__(self, mat_path, seed, config=None):
        super().__init__(mat_path, seed, config)
        self.layout_features = object_layout_features(self)
        self.layout_features.setflags(write=False)

    def global_state(self):
        broadcasts, coverage = super().global_state()
        return np.concatenate((broadcasts, self.layout_features.ravel())), coverage


class LayoutCritic(CriticNetwork):
    """Same hidden layers; add a zero-initialized geometry input projection.

    Preserve all original weights and the Torch RNG state. Consequently the
    initial value predictions and the first actor update are identical to the
    original critic experiment for the same seed and trajectory. New context
    columns are trainable from the first critic update onward.
    """

    def __init__(self, num_targets, hidden_dims):
        super().__init__(num_targets, hidden_dims)
        self.num_targets = num_targets
        # Keep the original matrix multiplication intact, including its
        # reduction shape. Adding zero columns to that matrix would select a
        # different float32 kernel and needlessly perturb the initial values.
        with torch.random.fork_rng(devices=[]):
            self.layout_projection = nn.Linear(4 * num_targets, hidden_dims[0], bias=False)
        nn.init.zeros_(self.layout_projection.weight)

    def forward(self, broadcast_state, coverage_rate):
        counts = broadcast_state[..., :2 * self.num_targets]
        context = broadcast_state[..., 2 * self.num_targets:]
        original_input = torch.cat((counts, coverage_rate), dim=-1)
        hidden = self.shared_net[0](original_input) + self.layout_projection(context)
        for layer in self.shared_net[1:]:
            hidden = layer(hidden)
        return self.value_head(hidden)


def prepare_protocol(data, output, config):
    data, output = Path(data).resolve(), Path(output).resolve()
    if not data.is_relative_to(HERE) or not output.is_relative_to(HERE):
        raise ValueError('Data and outputs must stay under major/cross_scene.')
    manifest = load_manifest(data, verify=True)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        schema_version=1, variant='layout_critic', data_root=str(data),
        data_manifest_sha256=sha(data / 'manifest.json'),
        split_counts=manifest['split_counts'], config=config,
        environment={**ENV_DEFAULT_CONFIG, **environment_config(config)},
        methods=list(METHODS), seeds=list(SEEDS), code_sha256=source_hashes(),
        only_change='Geometry context added to both centralized critics; simulator, actor and PPO update unchanged.',
        actor='Independent local actor MLPs, identical to the original cross_scene runner.',
        critic=dict(features=list(FEATURE_NAMES), per_object_feature_count=4,
                    ordering='Original report presence/counts, object-major geometry features, coverage scalar.',
                    hidden_widths=config['critic_hidden'],
                    initialization='Original weights and RNG preserved; added geometry columns initialized to zero.',
                    information='Geometry only, centralized training only; no channel outcomes or reward labels.'),
        checkpoint_selection='Maximum fixed validation mean reward; ties retain earlier trained checkpoint.',
        evaluation='Training and validation only. The exposed old test bank is never evaluated by this runner.',
        channels='Same corrected Nakagami simulator, parameters and seeds as results/main.',
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     torch=torch.__version__, platform=platform.platform()),
    )
    path = output / 'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        for key in ('variant', 'data_manifest_sha256', 'config', 'code_sha256', 'methods', 'seeds'):
            if previous.get(key) != protocol[key]:
                raise ValueError(f'Refusing incompatible experiment: {key}')
        return manifest, previous
    if any(output.iterdir()):
        # Shell-redirected logs and an OS lock can precede protocol creation.
        allowed = {'batch.stdout.log', 'batch.stderr.log', 'batch.lock'}
        if any(p.name not in allowed for p in output.iterdir()):
            raise FileExistsError('Refusing a nonempty output without a matching protocol.')
    protocol['preflight_train_validation_only'] = feasibility_check(manifest, data, config)
    write_json(path, protocol)
    for name in SOURCE_NAMES:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return manifest, protocol


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
    envs = [LayoutCriticEnv(data / r['path'], seed=0, config=environment_config(config))
            for r in manifest['splits']['train']]
    if len({(e.state_dim, e.num_agents, e.max_targets_per_cav) for e in envs}) != 1:
        raise ValueError('Training scenes have inconsistent fixed interfaces.')
    template = envs[0]
    policy = make_policy(template, config)
    initial = state_hash(policy)
    critic = LayoutCritic(template.total_vehicles, config['critic_hidden'])
    actor_optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=config['actor_lr'])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config['critic_lr'])
    scene_rng = np.random.default_rng(seed + 700000)
    pair_rng = np.random.default_rng(seed + 800000)
    history, validations, validation_rows = [], [], []
    best_reward, best_episode, start = -float('inf'), None, 1
    record = dict(method=method, seed=seed, status='running', pid=os.getpid(),
                  started_at_utc=now(), config=config, data_manifest_sha256=protocol['data_manifest_sha256'],
                  code_sha256=protocol['code_sha256'], initial_actor_sha256=initial, initial_critic_sha256=state_hash(critic),
                  initialization='Random, from scratch; no old checkpoint loaded.',
                  agents=template.num_agents, objects=template.total_vehicles,
                  state_dim=template.state_dim, action_dim=template.max_targets_per_cav,
                  critic_state_dim=6 * template.total_vehicles + 1,
                  layout_feature_dim=4 * template.total_vehicles)
    checkpoint_path = destination / 'checkpoint.pt'
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        for key in ('method', 'seed', 'config', 'data_manifest_sha256', 'code_sha256', 'initial_actor_sha256', 'initial_critic_sha256'):
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
            torch.save(dict(policy=policy.state_dict(), critic=critic.state_dict(), episode=episode,
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
            adv = advantage[:, None] if method == 'ocpo' else advantage
            actor_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - cfg['clip'], 1 + cfg['clip']) * adv).mean(0).sum()
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

def freeze_training(data, output, config):
    _, protocol = prepare_protocol(data, output, config)
    runs = []
    for seed in SEEDS:
        initials = []
        for method in METHODS:
            folder = output / f'{method}_seed_{seed}'
            record = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            if record['status'] != 'completed' or record['completed_episodes'] != config['episodes']:
                raise RuntimeError('All six full runs must finish before freezing.')
            if record['code_sha256'] != protocol['code_sha256']:
                raise ValueError('Completed run source does not match the protocol.')
            if sha(folder / 'best.pt') != record['selected_checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed.')
            initials.append((record['initial_actor_sha256'], record['initial_critic_sha256']))
            runs.append(dict(
                method=method, seed=seed, selected_episode=record['selected_episode'],
                validation_reward=record['best_validation_reward'],
                training_last100=record['final_training_reward'],
                checkpoint=str((folder / 'best.pt').relative_to(output)),
                checkpoint_sha256=record['selected_checkpoint_sha256'],
                training_csv_sha256=sha(folder / 'training.csv'),
                validation_csv_sha256=sha(folder / 'validation.csv'),
            ))
        if len(set(initials)) != 1:
            raise AssertionError('Paired methods must share actor and critic initialization.')
    frozen = dict(phase='training_and_validation_only',
                  data_manifest_sha256=protocol['data_manifest_sha256'],
                  code_sha256=protocol['code_sha256'], config=config, runs=runs)
    path = output / 'freeze.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != frozen:
        raise ValueError('Cannot modify a frozen experiment.')
    write_json(path, frozen)
    return frozen


def validation_summary(data, output, config, reference=None):
    frozen = freeze_training(data, output, config)
    aggregates = {}
    for method in METHODS:
        runs = [r for r in frozen['runs'] if r['method'] == method]
        aggregates[method] = {
            key: dict(mean=float(np.mean([r[key] for r in runs])),
                      sample_sd=float(np.std([r[key] for r in runs], ddof=1)))
            for key in ('validation_reward', 'training_last100')
        }
    for seed in SEEDS:
        train_conditions, validation_conditions = [], []
        for method in METHODS:
            folder = output / f'{method}_seed_{seed}'
            with (folder / 'training.csv').open(newline='', encoding='utf-8') as handle:
                train_conditions.append([(r['scene_id'], r['channel_seed']) for r in csv.DictReader(handle)])
            with (folder / 'validation_channels.csv').open(newline='', encoding='utf-8') as handle:
                validation_conditions.append([
                    (r['scene_sha256'], r['channel_seed'], r['channel_sha256'])
                    for r in csv.DictReader(handle)
                ])
        if train_conditions[0] != train_conditions[1] or validation_conditions[0] != validation_conditions[1]:
            raise AssertionError('Paired methods did not receive the same layouts and channels.')
    result = dict(
        status='completed', phase='validation_only', finished_at_utc=now(),
        final_test_performed=False, runs=frozen['runs'], aggregate=aggregates,
        ocpo_minus_mappo_validation=(
            aggregates['ocpo']['validation_reward']['mean'] - aggregates['mappo']['validation_reward']['mean']),
        paired_conditions_identical=True,
    )
    if reference is not None:
        # Read validation records only; never open exposed test rewards.
        reference = Path(reference).resolve()
        previous_protocol = json.loads((reference / 'protocol.json').read_text(encoding='utf-8'))
        if (previous_protocol['data_manifest_sha256'] != frozen['data_manifest_sha256']
                or previous_protocol['config'] != config):
            raise ValueError('Reference training data/config differs.')
        previous = []
        for seed in SEEDS:
            for method in METHODS:
                record = json.loads((reference / f'{method}_seed_{seed}' / 'run.json').read_text(encoding='utf-8'))
                previous.append(dict(method=method, seed=seed,
                                     validation_reward=record['best_validation_reward'],
                                     training_last100=record['final_training_reward']))
        result['previous_critic_validation'] = previous
        result['change_from_previous_critic'] = {
            method: aggregates[method]['validation_reward']['mean']
                    - float(np.mean([r['validation_reward'] for r in previous if r['method'] == method]))
            for method in METHODS
        }
    write_json(output / 'validation_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('batch', 'train', 'summary'))
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--method', choices=METHODS, default='mappo')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['episodes'] = args.episodes
    if args.episodes < 1:
        parser.error('episodes must be positive')
    if args.episodes != 1500 and (args.command == 'batch' or args.output == DEFAULT_OUTPUT.resolve()):
        parser.error('Formal batch requires 1500 episodes; short checks need a temporary output.')
    torch.set_num_threads(config['cpu_threads'])
    if args.command == 'train':
        print(json.dumps(run_training(args.method, args.seed, args.data, args.output,
                                      config, resume=args.resume)), flush=True)
        return
    if args.command == 'summary':
        print(json.dumps(validation_summary(args.data, args.output, config,
                                            reference=HERE / 'results' / 'main')), flush=True)
        return
    # Only the batch worker owns the output lock and sequential scheduling.
    args.output.mkdir(parents=True, exist_ok=True)
    lock = acquire_batch_lock(args.output)
    status = dict(status='running', phase='training_and_validation_only',
                  pid=os.getpid(), started_at_utc=now(), completed=[],
                  final_test_scheduled=False)
    try:
        prepare_protocol(args.data, args.output, config)
        for seed in SEEDS:
            for method in METHODS:
                name = f'{method}_seed_{seed}'
                run_path = args.output / name / 'run.json'
                if run_path.exists() and json.loads(run_path.read_text(encoding='utf-8'))['status'] == 'completed':
                    status['completed'].append(name)
                    continue
                status['active'] = name
                write_json(args.output / 'status.json', status)
                run_training(method, seed, args.data, args.output, config, resume=run_path.exists())
                status['completed'].append(name)
        validation_summary(args.data, args.output, config, reference=HERE / 'results' / 'main')
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
