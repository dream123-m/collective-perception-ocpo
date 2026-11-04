"""Matched component ablations for the fixed-fading, layout-only experiment.

Run team advantage with vehicle clipping, then individual advantage with paired
clipping. Reuse completed MAPPO/OCPO references; stop after the declared seed.
The final validation layouts are never loaded or evaluated by this entry point.
"""
from __future__ import annotations

import argparse
import copy
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
from torch.distributions import Bernoulli

from . import layout_only as baseline
from .data import load_manifest
from .environment import DEFAULT_CONFIG as ENV_DEFAULT_CONFIG
from .layout_critic import FEATURE_NAMES
from .layout_only import FixedFadingEnv, FIXED_FADING_SEED, evaluate_frozen, feasibility_check
from .value_scale import ScaledValueCritic, critic_diagnostics
from .multi_layout_update import balanced_minibatches, padded_factor_logp
from .run_helpers import _inside, _read, _rows, _completed_run
from .run import (
    HERE, ROOT, METHODS, SEEDS, now, sha, state_hash, write_json, write_csv,
    environment_config, make_policy, groups_for, factor_logp, summary,
    rollout_probabilities, gae, normalized, acquire_batch_lock,
)

CONTROL_METHODS = ('team_vehicle', 'individual_pair')
DEFAULT_CONFIG = copy.deepcopy(baseline.DEFAULT_CONFIG)
DEFAULT_OUTPUT = HERE / 'results' / 'ablation_layout_only'
DEFAULT_REFERENCE = baseline.DEFAULT_OUTPUT
SOURCE_NAMES = (*baseline.SOURCE_NAMES, 'major/cross_scene/ablation_layout_only.py',
                'major/cross_scene/run_helpers.py')
CONTROL_DEFINITIONS = {
    'team_vehicle': {
        'advantage': 'Team-mean reward, scalar GAE and pooled scalar normalization.',
        'ratio': 'Product of all valid Bernoulli decisions of one vehicle.',
        'clipping': 'One PPO clipped term per vehicle.',
    },
    'individual_pair': {
        'advantage': 'Own vehicle reward, per-vehicle GAE and pooled per-vehicle normalization.',
        'ratio': 'Disjoint same-object pairs and singletons, unchanged from OCPO.',
        'clipping': 'Each endpoint receives its own advantage through the joint ratio; peer log ratio detached.',
        'scaling': 'Endpoint terms summed, no pair averaging or factor-count normalization.',
    },
}


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_NAMES}


def check_config(config):
    baseline.check_config(config)


def update_loss(new_bits, old_bits, groups, advantage, clip, mode):
    """Control objectives; detaching peer credit preserves endpoint-owned gradients."""
    def surrogate(ratio, credit):
        return torch.minimum(ratio * credit, ratio.clamp(1 - clip, 1 + clip) * credit)

    difference = new_bits - old_bits
    if mode == 'team_vehicle':
        return -surrogate(difference.sum(-1).exp(), advantage[:, None]).mean(0).sum()
    if mode != 'individual_pair':
        raise ValueError('Unknown ablation control.')
    flat = difference.flatten(-2)
    first, second = groups[..., 0], groups[..., 1]
    valid, paired = first >= 0, (first >= 0) & (second >= 0)
    da = flat.gather(-1, first.clamp_min(0))
    db = torch.where(paired, flat.gather(-1, second.clamp_min(0)), 0.)
    slots = difference.shape[-1]
    aa = advantage.gather(-1, first.clamp_min(0) // slots)
    ab = advantage.gather(-1, second.clamp_min(0) // slots)
    a = surrogate((da + db.detach()).exp(), aa) * valid
    b = surrogate((da.detach() + db).exp(), ab) * paired
    return -(a + b).mean(0).sum()


def prepare_protocol(data, output, config):
    check_config(config)
    data, output = Path(data).resolve(), Path(output).resolve()
    if not data.is_relative_to(HERE) or not output.is_relative_to(HERE):
        raise ValueError('Data and outputs must stay under major/cross_scene.')
    manifest = load_manifest(data, verify=True)
    if len(manifest['splits']['train']) < config['layouts_per_update']:
        raise ValueError('Not enough distinct training layouts for a pooled update.')
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        schema_version=1, variant='ablation_layout_only', data_root=str(data),
        data_manifest_sha256=sha(data / 'manifest.json'), split_counts=manifest['split_counts'],
        config=config, environment={**ENV_DEFAULT_CONFIG, **environment_config(config)},
        methods=list(CONTROL_METHODS), controls=CONTROL_DEFINITIONS, seeds=list(SEEDS), code_sha256=source_hashes(),
        only_change='Cross the existing team/individual advantage and paired/vehicle clipping components; all physical and training settings unchanged.',
        rollout_semantics=dict(task='Continuing discounted broadcast decisions.',
                               window_steps=config['episode_steps'],
                               window_end='External sampling truncation, not an absorbing task terminal.',
                               target='Per-window GAE uses V(last state) on the same layout and channel.',
                               evaluation='Mean of actual rewards only; bootstrap is never added to evaluation reward.'),
        fading=dict(seed=FIXED_FADING_SEED, normalization='Unit-mean Gamma(m, 1/m).',
                    assignment='Fixed ordered CAV lane/rank roles; same matrix across layouts and methods.',
                    geometry='Original distance, LOS, interference and SINR retained.',
                    evaluation='One rollout per layout; no additional random fading repetitions.',
                    selection='Seed zero fixed before training, without evaluating alternatives.'),
        sampling='Uniform distinct layouts without replacement inside an update; independent draws across updates.',
        advantages='Window-local bootstrapped GAE and pooled normalization; team for team_vehicle, individual for individual_pair.',
        minibatches='Shuffle time within each episode, interleave layouts, then split into episode_steps-sized minibatches.',
        budgets=dict(training_episodes=config['episodes'],
                     joint_steps=config['episodes'] * config['episode_steps'],
                     optimizer_steps=config['episodes'] * config['epochs'],
                     sample_passes=config['epochs']),
        critic=dict(features=list(FEATURE_NAMES), implementation='Unchanged ScaledValueCritic; FixedFadingEnv fixes only the sampler seed.',
                    normalization='Fixed scale c=1-gamma; no running moments and no PopArt.',
                    value_scale=1. - config['gamma'], network_output='z=cV',
                    training_target='c times the original-unit GAE return',
                    collection_and_bootstrap='V=z/c in original reward units',
                    initialization='Output head multiplied by c; initial real-unit values preserved within float32 precision.',
                    diagnostics='Training-only value/target moments, RMSE, explained variance and hidden saturation.'),
        actor='Unchanged local actor observations, MLPs, action masks and deployment.',
        checkpoint_selection='Maximum fixed validation mean reward; ties retain earlier trained checkpoint.',
        evaluation='Development training/validation only; no final test or other seeds are automatically launched.',
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     torch=torch.__version__, platform=platform.platform()),
    )
    path = output / 'protocol.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        for name in ('variant', 'data_manifest_sha256', 'config', 'code_sha256', 'methods', 'seeds', 'controls'):
            if existing.get(name) != protocol[name]:
                raise ValueError(f'Refusing incompatible experiment: {name}')
        return manifest, existing
    allowed = {'batch.lock', 'stage.stdout.log', 'stage.stderr.log'}
    if any(path.name not in allowed for path in output.iterdir()):
        raise FileExistsError('Refusing a nonempty output without a matching protocol.')
    protocol['preflight_train_validation_only'] = feasibility_check(manifest, data, config)
    write_json(path, protocol)
    for name in SOURCE_NAMES:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return manifest, protocol


def pack_rollouts(rollouts, method, config):
    """Bootstrap each sampling window separately; never carry traces across layouts."""
    if method not in CONTROL_METHODS:
        raise ValueError('Unknown ablation control.')
    fields = ('obs', 'action', 'old_bits', 'value', 'reward', 'done', 'window_end', 'broadcast', 'coverage', 'mask')
    episodes = [{key: torch.stack([row[key] for row in rollout]) for key in fields}
                for rollout in rollouts]
    batch = {key: torch.cat([episode[key] for episode in episodes]) for key in fields}
    advantages, returns = [], []
    for episode_index, episode in enumerate(episodes):
        if (not bool(episode['window_end'][-1]) or bool(episode['window_end'][:-1].any())
                or bool(episode['done'][:-1].any())):
            raise AssertionError('Expected independent completed sampling windows.')
        bootstrap = torch.as_tensor(rollouts[episode_index][-1]['bootstrap'],
                                    dtype=episode['value'].dtype).reshape(())
        if bool(episode['done'][-1]):
            bootstrap = torch.zeros_like(bootstrap)
        if method == 'team_vehicle':
            advantage, target = gae(episode['reward'].mean(-1), episode['value'], bootstrap,
                                    episode['done'], config['gamma'], config['gae_lambda'])
        else:
            advantage, individual = gae(episode['reward'], episode['value'][:, None],
                bootstrap.expand(episode['reward'].shape[-1]), episode['done'],
                config['gamma'], config['gae_lambda'])
            target = individual.mean(-1)
        advantages.append(advantage)
        returns.append(target)
    batch['raw_advantage'] = torch.cat(advantages)
    batch['advantage'] = normalized(batch['raw_advantage'])
    batch['returns'] = torch.cat(returns)
    rows = [row for rollout in rollouts for row in rollout]
    if method == 'individual_pair':
        width = max(len(row['groups']) for row in rows)
        groups = torch.full((len(rows), width, 2), -1, dtype=torch.long)
        old = torch.zeros((len(rows), width), dtype=torch.float64)
        for index, row in enumerate(rows):
            count = len(row['groups'])
            groups[index, :count] = row['groups']
            old[index, :count] = row['old']
        batch['groups'], batch['old'] = groups, old
        batch['factor_mask'] = groups[..., 0] >= 0
    else:
        batch['old'] = torch.stack([row['old'] for row in rows])
    return batch


def optimize(policy, critic, actor_optimizer, critic_optimizer, batch, method, lengths, rng, config):
    mask = batch['mask']
    with torch.no_grad():
        replay = rollout_probabilities(policy, batch['obs'])
        bits = Bernoulli(replay.double()).log_prob(batch['action'].double()) * mask
        torch.testing.assert_close(bits, batch['old_bits'], atol=1e-10, rtol=1e-10)
        replay_logp = padded_factor_logp(bits, batch['groups']) if method == 'individual_pair' else bits.sum(-1)
        torch.testing.assert_close(replay_logp, batch['old'], atol=1e-10, rtol=1e-10)
    actor_sum, critic_sum, weight, steps = 0., 0., 0, 0
    for epoch in range(config['epochs']):
        for indices in balanced_minibatches(lengths, config['minibatch_joint_steps'], rng):
            probabilities = rollout_probabilities(policy, batch['obs'][indices])
            distribution = Bernoulli(probabilities.double())
            local_mask = mask[indices]
            bits = distribution.log_prob(batch['action'][indices].double()) * local_mask
            groups = batch['groups'][indices] if method == 'individual_pair' else None
            actor_loss = update_loss(bits, batch['old_bits'][indices], groups,
                                     batch['advantage'][indices], config['clip'], method)
            entropy = (distribution.entropy() * local_mask).sum(-1) / local_mask.sum(-1).clamp_min(1)
            actor_loss -= config['entropy'] * entropy.mean(0).sum()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            policy.clip_gradients(config['max_grad_norm'])
            actor_optimizer.step()
            value = critic.normalized_value(batch['broadcast'][indices], batch['coverage'][indices]).flatten()
            critic_loss = .5 * (value - batch['returns'][indices] * critic.value_scale).square().mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config['max_grad_norm'])
            critic_optimizer.step()
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                raise FloatingPointError('Nonfinite PPO update.')
            count = len(indices)
            actor_sum += float(actor_loss.detach()) * count
            critic_sum += float(critic_loss.detach()) * count
            weight += count
            steps += 1
    metrics = dict(actor_loss=actor_sum / weight / policy.num_agents,
                   normalized_critic_loss=critic_sum / weight,
                   critic_loss=critic_sum / weight / float(critic.value_scale) ** 2,
                   optimizer_steps=steps)
    metrics.update(critic_diagnostics(critic, batch))
    return metrics


def run_training(method, seed, data, output, config, *, resume=False):
    if method not in CONTROL_METHODS or seed not in SEEDS:
        raise ValueError('Unknown method or undeclared seed.')
    data, output = _inside(data), _inside(output)
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    manifest, protocol = prepare_protocol(data, output, config)
    destination = output / f'{method}_seed_{seed}'
    destination.mkdir(exist_ok=True)
    if (output / 'freeze.json').exists():
        raise RuntimeError('Experiment is frozen.')
    if (destination / 'run.json').exists() and not resume:
        raise FileExistsError('Run exists; resume explicitly.')
    envs = [FixedFadingEnv(data / record['path'], seed=FIXED_FADING_SEED, config=environment_config(config))
            for record in manifest['splits']['train']]
    template = envs[0]
    if len({(e.state_dim, e.num_agents, e.max_targets_per_cav) for e in envs}) != 1:
        raise ValueError('Training scene interfaces differ.')
    policy = make_policy(template, config)
    initial = state_hash(policy)
    critic = ScaledValueCritic(template.total_vehicles, config['critic_hidden'], 1. - config['gamma'])
    initial_broadcast, initial_coverage = template.global_state()
    with torch.no_grad():
        initial_raw_value = float(critic(torch.from_numpy(initial_broadcast)[None],
                                        torch.tensor([[initial_coverage]], dtype=torch.float32)))
    actor_optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=config['actor_lr'])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config['critic_lr'])
    scene_rng = np.random.default_rng(seed + 700000)
    pair_rng = np.random.default_rng(seed + 800000)
    minibatch_rng = np.random.default_rng(seed + 900000)
    record = dict(method=method, seed=seed, status='running', pid=os.getpid(), started_at_utc=now(),
                  config=config, data_manifest_sha256=protocol['data_manifest_sha256'],
                  code_sha256=protocol['code_sha256'], initial_actor_sha256=initial,
                  initial_critic_sha256=state_hash(critic),
                  initial_unscaled_critic_sha256=critic.initial_unscaled_sha256,
                  initial_raw_critic_value=initial_raw_value,
                  critic_value_scale=float(critic.value_scale), agents=template.num_agents,
                  objects=template.total_vehicles, state_dim=template.state_dim,
                  action_dim=template.max_targets_per_cav, critic_state_dim=6 * template.total_vehicles + 1,
                  fixed_fading_seed=FIXED_FADING_SEED,
                  unit_fading_sha256=template.unit_fading_sha256, role_sha256=template.role_sha256,
                  initialization='Random from scratch; no old checkpoint used.')
    history, validations, validation_rows = [], [], []
    best_reward, best_episode, start, updates, optimizer_steps = -float('inf'), None, 1, 0, 0
    elapsed_before = 0.
    checkpoint = destination / 'checkpoint.pt'
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError('Requested resume has no durable checkpoint.')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for key in ('method', 'seed', 'config', 'data_manifest_sha256', 'code_sha256',
                    'initial_actor_sha256', 'initial_critic_sha256', 'initial_unscaled_critic_sha256',
                    'initial_raw_critic_value', 'critic_value_scale', 'fixed_fading_seed',
                    'unit_fading_sha256', 'role_sha256'):
            if saved['record'][key] != record[key]:
                raise ValueError(f'Resume mismatch: {key}')
        policy.load_state_dict(saved['policy']); critic.load_state_dict(saved['critic'])
        if float(critic.value_scale) != record['critic_value_scale']:
            raise ValueError('Checkpoint critic scale differs from its declared configuration.')
        actor_optimizer.load_state_dict(saved['actor_optimizer']); critic_optimizer.load_state_dict(saved['critic_optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        scene_rng.bit_generator.state = saved['scene_rng']
        pair_rng.bit_generator.state = saved['pair_rng']
        minibatch_rng.bit_generator.state = saved['minibatch_rng']
        history, validations, validation_rows = saved['history'], saved['validations'], saved['validation_rows']
        best_reward, best_episode = saved['best_reward'], saved['best_episode']
        updates, optimizer_steps = saved['updates'], saved['optimizer_steps']
        start = saved['episode'] + 1
        elapsed_before = saved['record']['elapsed_seconds']
        record.update(started_at_utc=saved['record']['started_at_utc'], resumed_from_episode=saved['episode'])
    write_json(destination / 'run.json', record)
    started = time.perf_counter()

    def validate(episode):
        nonlocal best_reward, best_episode, validation_rows
        validation_rows = evaluate_frozen(policy, manifest['splits']['validation'], data, config,
                                         namespace='validation', channels=config['validation_channels_per_layout'])
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
    while start <= config['episodes']:
        count = min(config['layouts_per_update'], config['episodes'] - start + 1)
        scene_indices = scene_rng.choice(len(envs), size=count, replace=False)
        rollouts, episode_metrics = [], []
        with torch.no_grad():
            for offset, index in enumerate(scene_indices):
                episode = start + offset
                env, scene = envs[int(index)], manifest['splits']['train'][int(index)]
                cseed = FIXED_FADING_SEED
                obs = torch.from_numpy(env.reset(seed=cseed))
                mask = torch.from_numpy(env.mask)
                rows = []
                for step in range(config['episode_steps']):
                    broadcast, coverage = env.global_state()
                    broadcast = torch.from_numpy(broadcast)
                    coverage = torch.tensor([coverage], dtype=torch.float32)
                    value = critic(broadcast[None], coverage[None]).reshape(())
                    probabilities = policy.probabilities(obs[None])[0]
                    actions = Bernoulli(probabilities).sample().bool() & mask
                    bits = Bernoulli(probabilities.double()).log_prob(actions.double()) * mask
                    groups = groups_for(env, pair_rng) if method == 'individual_pair' else None
                    old = factor_logp(bits[None], groups[None])[0] if groups is not None else bits.sum(-1)
                    nxt, rewards, done, info = env.step(actions.numpy())
                    rows.append(dict(obs=obs, action=actions, old=old, old_bits=bits, value=value,
                                     reward=torch.from_numpy(np.asarray(rewards, dtype=np.float32)),
                                     done=torch.tensor(info['terminated']), window_end=torch.tensor(done),
                                     broadcast=broadcast, coverage=coverage,
                                     mask=mask, groups=groups, observed_coverage=float(info['coverage_rate']),
                                     payload=float(actions.sum()) / env.num_agents))
                    obs = torch.from_numpy(nxt)
                    if done:
                        break
                if not done:
                    raise AssertionError('Incomplete episode.')
                # Evaluate the actual final state before any layout/channel reset.
                if info['terminated']:
                    bootstrap = torch.tensor(0.)
                else:
                    final_broadcast, final_coverage = env.global_state()
                    bootstrap = critic(torch.from_numpy(final_broadcast)[None],
                        torch.tensor([[final_coverage]], dtype=torch.float32)).reshape(())
                rows[-1]['bootstrap'] = bootstrap
                rollouts.append(rows)
                episode_metrics.append(dict(
                    episode=episode, scene_id=scene['scene_id'], channel_seed=env.channel_seed,
                    channel_sha256=env.channel_hash, unit_fading_sha256=env.unit_fading_sha256,
                    role_sha256=env.role_sha256,
                    reward=float(torch.stack([row['reward'] for row in rows]).mean()),
                    coverage=float(np.mean([row['observed_coverage'] for row in rows])),
                    payload=float(np.mean([row['payload'] for row in rows])), all_on_coverage=float(env.all_on_coverage()),
                    bootstrap_value=float(bootstrap), terminated=bool(info['terminated']),
                    truncated=bool(info['truncated'])))
            batch = pack_rollouts(rollouts, method, config)
        metrics = optimize(policy, critic, actor_optimizer, critic_optimizer, batch, method,
                           [len(rollout) for rollout in rollouts], minibatch_rng, config)
        updates += 1
        optimizer_steps += metrics.pop('optimizer_steps')
        episode = start + count - 1
        for row in episode_metrics:
            row.update(**metrics, ppo_update=updates, batch_end_episode=episode,
                       batch_distinct_layouts=count, optimizer_steps_total=optimizer_steps)
            history.append(row)
        if episode % config['validation_every'] == 0 or episode == config['episodes']:
            validate(episode)
            record.update(completed_episodes=episode, completed_updates=updates,
                          optimizer_steps=optimizer_steps, joint_steps=episode * config['episode_steps'],
                          best_validation_reward=best_reward, selected_episode=best_episode,
                          elapsed_seconds=elapsed_before + time.perf_counter() - started,
                          final_training_reward=float(np.mean([row['reward'] for row in history[-100:]])))
            saved = dict(policy=policy.state_dict(), critic=critic.state_dict(),
                         actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                         record=record, episode=episode, history=history, validations=validations,
                         validation_rows=validation_rows, best_reward=best_reward, best_episode=best_episode,
                         updates=updates, optimizer_steps=optimizer_steps, torch_rng=torch.get_rng_state(),
                         scene_rng=scene_rng.bit_generator.state, pair_rng=pair_rng.bit_generator.state,
                         minibatch_rng=minibatch_rng.bit_generator.state)
            temporary = destination / 'checkpoint.pending.pt'
            torch.save(saved, temporary)
            temporary.replace(checkpoint)
            write_csv(destination / 'training.csv', history)
            write_csv(destination / 'validation.csv', validations)
            write_csv(destination / 'validation_channels.csv', validation_rows)
            write_json(destination / 'run.json', record)
            print(f'{method} seed={seed} {episode}/{config["episodes"]} updates={updates} optimizer_steps={optimizer_steps} elapsed={record["elapsed_seconds"]:.1f}s', flush=True)
        start = episode + 1
    if optimizer_steps != config['episodes'] * config['epochs']:
        raise AssertionError('Optimizer-step budget changed.')
    record.update(status='completed', finished_at_utc=now(),
                  selected_checkpoint_sha256=sha(destination / 'best.pt'))
    write_json(destination / 'run.json', record)
    return record



def paired_conditions(folder):
    """Physical conditions, excluding policy-dependent action/reward metrics."""
    train_keys = ('episode', 'scene_id', 'channel_seed', 'channel_sha256',
                  'unit_fading_sha256', 'role_sha256')
    validation_keys = ('scene_id', 'replicate', 'channel_seed', 'channel_sha256',
                       'unit_fading_sha256', 'role_sha256', 'steps')
    return ([tuple(row[key] for key in train_keys) for row in _rows(folder / 'training.csv')],
            [tuple(row[key] for key in validation_keys)
             for row in _rows(folder / 'validation_channels.csv')])


def run_development(data, output, reference, seed=42):
    """Train the two fixed controls sequentially; no final validation or seed gate."""
    data, output, reference = _inside(data), _inside(output), _inside(reference)
    if seed not in SEEDS or output == reference:
        raise ValueError('Use a declared seed and a separate ablation output.')
    # Reference preparation verifies its ten immutable learner sources and data.
    reference_protocol = _read(reference / 'protocol.json')
    config = copy.deepcopy(reference_protocol['config'])
    _, reference_protocol = baseline.prepare_protocol(data, reference, config)
    records, reference_best = {}, {}
    condition_reference = None
    for method in METHODS:
        record, _, _ = _completed_run(data, reference, config, reference_protocol, method, seed)
        folder = reference / f'{method}_seed_{seed}'
        conditions = paired_conditions(folder)
        if condition_reference is None:
            condition_reference = conditions
        elif conditions != condition_reference:
            raise AssertionError('Main references have different physical conditions.')
        records[method] = record
        reference_best[method] = record['selected_checkpoint_sha256']
    if any(records['mappo'][key] != records['ocpo'][key]
           for key in ('initial_actor_sha256', 'initial_critic_sha256')):
        raise AssertionError('Main references did not share initialization.')
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_batch_lock(output)
    status = dict(status='running', phase='component_ablation_development', seed=seed,
                  pid=os.getpid(), started_at_utc=now(), completed=[], queued_seeds=[],
                  final_test_scheduled=False, stop_after_this_seed=True)
    try:
        _, protocol = prepare_protocol(data, output, config)
        identity = dict(schema_version=1, seed=seed, controls=CONTROL_DEFINITIONS,
                        reference_output=str(reference),
                        reference_protocol_sha256=sha(reference / 'protocol.json'),
                        reference_best_sha256=reference_best,
                        protocol_sha256=sha(output / 'protocol.json'),
                        data_manifest_sha256=protocol['data_manifest_sha256'],
                        code_sha256=protocol['code_sha256'],
                        selection='Same fixed validation layouts, maximal deterministic reward; earlier checkpoint on ties.',
                        decision='Report both controls without a reward gate or retuning; no final layouts accessed.',
                        queued_seeds=[], final_test_scheduled=False)
        stage_path = output / 'stage.json'
        if stage_path.exists():
            if _read(stage_path) != identity:
                raise ValueError('Ablation stage references or definitions changed.')
        else:
            write_json(stage_path, identity)
        for method in CONTROL_METHODS:
            name = f'{method}_seed_{seed}'
            folder = output / name
            path = folder / 'run.json'
            status['active'] = name
            write_json(output / 'status.json', status)
            prior = _read(path) if path.exists() else None
            if prior and prior['status'] == 'completed':
                print(f'Reusing completed {name}.', flush=True)
            else:
                run_training(method, seed, data, output, config, resume=path.exists())
            record, _, _ = _completed_run(data, output, config, protocol, method, seed)
            for key in ('initial_actor_sha256', 'initial_critic_sha256',
                        'initial_unscaled_critic_sha256', 'fixed_fading_seed',
                        'unit_fading_sha256', 'role_sha256'):
                if record[key] != records['mappo'][key]:
                    raise AssertionError(f'Ablation initialization/physics differ: {key}')
            if paired_conditions(folder) != condition_reference:
                raise AssertionError('Ablation physical sampling differs from the main reference.')
            records[method] = record
            status['completed'].append(name)
            write_json(output / 'status.json', status)
        # Verify reused checkpoints and provenance again after the long-running stage.
        if sha(reference / 'protocol.json') != identity['reference_protocol_sha256']:
            raise ValueError('Reference protocol changed while ablations ran.')
        for method in METHODS:
            current, _, _ = _completed_run(data, reference, config, reference_protocol, method, seed)
            if current['selected_checkpoint_sha256'] != reference_best[method]:
                raise ValueError('Reused main checkpoint changed.')
        comparison = dict(seed=seed, phase='development_validation',
                          final_test_performed=False, paired_conditions_identical=True,
                          validation_rewards={name: row['best_validation_reward'] for name, row in records.items()},
                          training_last100={name: row['final_training_reward'] for name, row in records.items()},
                          selected_episodes={name: row['selected_episode'] for name, row in records.items()})
        write_json(output / 'comparison.json', comparison)
        status.update(status='stage_completed', active=None, result=comparison, finished_at_utc=now())
        write_json(output / 'status.json', status)
        print(json.dumps(comparison), flush=True)
        return comparison
    except BaseException:
        status.update(status='failed', error=traceback.format_exc(), finished_at_utc=now())
        write_json(output / 'status.json', status)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    args = parser.parse_args()
    run_development(args.data, args.output, args.reference, args.seed)


if __name__ == '__main__':
    main()
