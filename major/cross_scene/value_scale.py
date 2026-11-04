"""Fixed-scale critic learning for the matched continuing-task experiment.

The critic predicts z=(1-gamma)*V internally and regresses to identically
scaled return targets. Rollout values, GAE and final-state bootstraps remain
in original reward units. Actor updates and physical evaluation are unchanged.
Only one matched development seed is scheduled; no final test is performed.
"""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from torch import nn
from torch.distributions import Bernoulli

from .data import load_manifest
from .environment import DEFAULT_CONFIG as ENV_DEFAULT_CONFIG
from .layout_critic import LayoutCritic, FEATURE_NAMES
from . import truncation_update as baseline
from .multi_layout_update import optimize as base_optimize
from .truncation_update import TruncatedLayoutEnv, pack_rollouts
from .run import (
    HERE, ROOT, METHODS, SEEDS, now, sha, state_hash, write_json, write_csv,
    channel_seed, environment_config, feasibility_check, make_policy,
    groups_for, factor_logp,
    evaluate_frozen, summary,
)

DEFAULT_CONFIG = copy.deepcopy(baseline.DEFAULT_CONFIG)
DEFAULT_CONFIG['value_normalization'] = 'fixed_one_minus_gamma'
DEFAULT_OUTPUT = HERE / 'results' / 'value_scale'
SOURCE_NAMES = (*baseline.SOURCE_NAMES, 'major/cross_scene/value_scale.py')


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_NAMES}


def check_config(config):
    baseline.check_config(config)
    if config.get('value_normalization') != 'fixed_one_minus_gamma':
        raise ValueError('This experiment requires fixed (1-gamma) critic scaling.')
    if not 0. <= config['gamma'] < 1.:
        raise ValueError('Fixed continuing-value scaling requires 0 <= gamma < 1.')


class ScaledValueCritic(LayoutCritic):
    """Expose raw V to collection while training a smaller internal z=cV.

    Scaling the output head preserves initial real-unit predictions within
    float32 precision. No extra parameters, random draws or moving statistics
    are introduced. The scale is saved as a buffer in every checkpoint.
    """

    def __init__(self, num_targets, hidden_dims, value_scale):
        if not np.isfinite(value_scale) or not 0. < value_scale <= 1.:
            raise ValueError('value_scale must be finite and in (0, 1].')
        super().__init__(num_targets, hidden_dims)
        self.initial_unscaled_sha256 = state_hash(self)
        self.register_buffer('value_scale', torch.tensor(float(value_scale), dtype=torch.float32))
        with torch.no_grad():
            self.value_head.weight.mul_(self.value_scale)
            self.value_head.bias.mul_(self.value_scale)

    def normalized_value(self, broadcast_state, coverage_rate):
        return super().forward(broadcast_state, coverage_rate)

    def forward(self, broadcast_state, coverage_rate):
        return self.normalized_value(broadcast_state, coverage_rate) / self.value_scale


class CriticTrainingView(nn.Module):
    """Share all critic parameters, exposing z only to the unchanged PPO loop."""

    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, broadcast_state, coverage_rate):
        return self.critic.normalized_value(broadcast_state, coverage_rate)


def critic_diagnostics(critic, batch):
    """Training-only scale/calibration diagnostics; never inspect validation targets."""
    with torch.no_grad():
        broadcasts, coverage = batch['broadcast'], batch['coverage']
        n = critic.num_targets
        hidden = (critic.shared_net[0](torch.cat((broadcasts[..., :2*n], coverage), -1))
                  + critic.layout_projection(broadcasts[..., 2*n:]))
        saturation = []
        for layer in critic.shared_net[1:]:
            hidden = layer(hidden)
            if isinstance(layer, nn.Tanh):
                saturation.append(float((hidden.abs() > .99).float().mean()))
        normalized_value = critic.value_head(hidden).flatten()
        value = normalized_value / critic.value_scale
        target = batch['returns']
        variance = target.var(unbiased=False)
        error = value - target
        explained = float(1 - error.var(unbiased=False) / variance) if float(variance) > 1e-12 else None
        result = dict(value_scale=float(critic.value_scale),
                      critic_value_mean=float(value.mean()),
                      critic_value_std=float(value.std(unbiased=False)),
                      critic_target_mean=float(target.mean()),
                      critic_target_std=float(target.std(unbiased=False)),
                      critic_raw_rmse=float(error.square().mean().sqrt()),
                      critic_explained_variance=explained)
        result.update({f'critic_hidden_{i+1}_saturation_099': x for i, x in enumerate(saturation)})
        return result


def optimize(policy, critic, actor_optimizer, critic_optimizer, batch, method, lengths, rng, config):
    """Only critic regression is rescaled; actor tensors and PPO operations are reused."""
    normalized_batch = {**batch, 'returns': batch['returns'] * critic.value_scale}
    metrics = base_optimize(policy, CriticTrainingView(critic), actor_optimizer, critic_optimizer,
                            normalized_batch, method, lengths, rng, config)
    # Preserve the original-unit loss field for comparison with earlier logs.
    metrics['normalized_critic_loss'] = metrics['critic_loss']
    metrics['critic_loss'] /= float(critic.value_scale) ** 2
    metrics.update(critic_diagnostics(critic, batch))
    return metrics


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
        schema_version=1, variant='value_scale', data_root=str(data),
        data_manifest_sha256=sha(data / 'manifest.json'), split_counts=manifest['split_counts'],
        config=config, environment={**ENV_DEFAULT_CONFIG, **environment_config(config)},
        methods=list(METHODS), seeds=list(SEEDS), code_sha256=source_hashes(),
        only_change='Fixed (1-gamma) scaling of critic output parameterization and regression targets, shared by both methods.',
        rollout_semantics=dict(task='Continuing discounted broadcast decisions.',
                               window_steps=config['episode_steps'],
                               window_end='External sampling truncation, not an absorbing task terminal.',
                               target='Per-window GAE uses V(last state) on the same layout and channel.',
                               evaluation='Mean of actual rewards only; bootstrap is never added to evaluation reward.'),
        sampling='Uniform distinct layouts without replacement inside an update; independent draws across updates.',
        advantages='Window-local bootstrapped GAE, concatenated and normalized over the same pooled rollout; individual for MAPPO, team for OCPO.',
        minibatches='Shuffle time within each episode, interleave layouts, then split into episode_steps-sized minibatches.',
        budgets=dict(training_episodes=config['episodes'],
                     joint_steps=config['episodes'] * config['episode_steps'],
                     optimizer_steps=config['episodes'] * config['epochs'],
                     sample_passes=config['epochs']),
        critic=dict(features=list(FEATURE_NAMES), implementation='Same LayoutCritic layers with a scaled output head; same TruncatedLayoutEnv.',
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
        for name in ('variant', 'data_manifest_sha256', 'config', 'code_sha256', 'methods', 'seeds'):
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


def run_training(method, seed, data, output, config, *, resume=False):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError('Unknown method or undeclared seed.')
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    manifest, protocol = prepare_protocol(data, output, config)
    destination = output / f'{method}_seed_{seed}'
    destination.mkdir(exist_ok=True)
    if (output / 'freeze.json').exists():
        raise RuntimeError('Experiment is frozen.')
    if (destination / 'run.json').exists() and not resume:
        raise FileExistsError('Run exists; resume explicitly.')
    envs = [TruncatedLayoutEnv(data / record['path'], seed=0, config=environment_config(config))
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
                    'initial_raw_critic_value', 'critic_value_scale'):
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
                cseed = channel_seed('train', scene['scene_id'], episode, seed)
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
                    groups = groups_for(env, pair_rng) if method == 'ocpo' else None
                    old = factor_logp(bits[None], groups[None])[0] if groups is not None else bits.sum(-1)
                    nxt, rewards, done, info = env.step(actions.numpy())
                    rows.append(dict(obs=obs, action=actions, old=old, value=value,
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
                    episode=episode, scene_id=scene['scene_id'], channel_seed=cseed,
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

