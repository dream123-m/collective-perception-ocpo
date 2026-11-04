"""MADDPG-style baseline using the current conditional-layout simulator.

The untouched imported implementation is in ``original/``.  This adapter keeps
its independent ReLU actors, centralized team critic, replay, target networks,
and soft updates.  Because the simulator accepts binary actions, interaction
uses Bernoulli exploration and deterministic thresholding at evaluation; the
actor gradient uses a straight-through binary relaxation.  These are explicit
adaptations, not a byte-for-byte reproduction of the historical baseline.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
import time
from collections import deque
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch
from torch import nn

from major.maddpg.original.maddpg_network import ActorNetwork, CriticNetwork
from major.cross_scene import legacy_layout_experiment as main
from major.cross_scene.layout_only import FixedFadingEnv
from major.cross_scene.run import ROOT, now, sha, state_hash, write_csv, write_json

OUTPUT = ROOT / 'major/maddpg/results'
SEEDS = (42, 123, 456)


class JointActors(nn.Module):
    """Independent original MADDPG actors, stacked for efficient evaluation."""
    def __init__(self, state_dim, n, slots):
        super().__init__()
        actors = [ActorNetwork(state_dim, slots, [64, 64]) for _ in range(n)]
        states = [actor.state_dict() for actor in actors]
        keys = ('shared_net.0', 'shared_net.2', 'action_head')
        self.weights = nn.ParameterList([nn.Parameter(torch.stack([s[k + '.weight'] for s in states])) for k in keys])
        self.biases = nn.ParameterList([nn.Parameter(torch.stack([s[k + '.bias'] for s in states])) for k in keys])
        self.num_agents = n

    def probabilities(self, states):
        x = states
        for j, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum('bni,noi->bno', x, w) + b[None]
            if j < 2:
                x = torch.relu(x)
        return x.sigmoid()


def reconstruct(envs, indices, packed):
    """Restore exactly the common simulator observations using saved actions."""
    obs = np.stack([envs[int(i)]._template for i in indices]).copy()
    n, slots = envs[0].mask.shape
    actions = np.unpackbits(packed, axis=1, count=n * slots).reshape(-1, n, slots)
    obs[:, :, 2] = actions.sum(axis=-1) / slots
    obs[:, :, 3:].reshape(-1, n, slots, 5)[:, :, :, 4] = actions
    return torch.from_numpy(obs), torch.from_numpy(actions.astype(np.float32))


def sample_batch(replay, rng, envs, size):
    chosen = rng.choice(len(replay), size=size, replace=False)
    rows = [replay[int(i)] for i in chosen]
    indices = np.asarray([r[0] for r in rows], dtype=np.int64)
    prev = np.stack([r[1] for r in rows]); current = np.stack([r[2] for r in rows])
    states, _ = reconstruct(envs, indices, prev)
    next_states, actions = reconstruct(envs, indices, current)
    masks = torch.from_numpy(np.stack([envs[int(i)].mask for i in indices]))
    rewards = torch.tensor([r[3] for r in rows], dtype=torch.float32).reshape(-1, 1)
    return states, actions, rewards, next_states, masks


def update(actor, target_actor, critic, target_critic, ao, co, batch, gamma, tau):
    states, actions, rewards, next_states, masks = batch
    with torch.no_grad():
        next_action = (target_actor.probabilities(next_states) > .5).float() * masks
        # The seven-step collection boundary truncates a continuing task.
        target = rewards + gamma * target_critic(next_states, next_action)
    predicted = critic(states, actions)
    critic_loss = nn.functional.mse_loss(predicted, target)
    co.zero_grad(); critic_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(), .5); co.step()
    for param in critic.parameters():
        param.requires_grad_(False)
    probs = actor.probabilities(states)
    hard = (probs > .5).float()
    straight_through = (hard + probs - probs.detach()) * masks
    actor_loss = -critic(states, straight_through).mean()
    ao.zero_grad(); actor_loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), .5); ao.step()
    for param in critic.parameters():
        param.requires_grad_(True)
    with torch.no_grad():
        for target_net, net in ((target_actor, actor), (target_critic, critic)):
            for t, p in zip(target_net.parameters(), net.parameters()):
                t.lerp_(p, tau)
    if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
        raise FloatingPointError('Non-finite MADDPG loss.')
    return float(actor_loss.detach()), float(critic_loss.detach())


def settings(episodes=1500, smoke=False):
    return dict(episodes=episodes, episode_steps=7, validation_every=100,
        batch_size=4 if smoke else 32, replay_capacity=2048, warmup_steps=0 if smoke else 1000,
        updates_per_window=1, gamma=.95, tau=.01, actor_lr=1e-4, critic_lr=1e-3,
        target_coverage=.95, action='Bernoulli sample in training; threshold >0.5 in evaluation; straight-through actor gradient',
        smoke=smoke)


def train(seed, output, cfg, *, resume=False, stop_after=None):
    if seed not in SEEDS:
        raise ValueError('Undeclared seed.')
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'major'):
        raise ValueError('Outputs must be under major.')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run.json').exists() and not resume:
        raise FileExistsError('Run exists; use --resume rather than overwriting it.')
    bank = main.load_bank(main.DATA)
    envcfg = main.env_config(main.config())
    train_records = bank['splits']['train']
    val_records = bank['splits']['validation'][:2] if cfg['smoke'] else bank['splits']['validation']
    envs = [FixedFadingEnv(main.DATA / r['path'], seed=0, config=envcfg) for r in train_records]
    env = envs[0]; actor = JointActors(env.state_dim, env.num_agents, env.max_targets_per_cav)
    critic = CriticNetwork(env.state_dim, env.max_targets_per_cav, env.num_agents, [128, 128])
    target_actor, target_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    ao = torch.optim.Adam(actor.parameters(), lr=cfg['actor_lr'])
    co = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'])
    scene_rng = np.random.default_rng(seed + 700000)
    replay_rng = np.random.default_rng(seed + 910000)
    replay = deque(maxlen=cfg['replay_capacity'])
    files = [Path(__file__), ROOT / 'major/maddpg/original/maddpg_network.py',
             ROOT / 'major/cross_scene/environment.py', ROOT / 'major/cross_scene/layout_only.py',
             ROOT / 'major/cross_scene/legacy_layout_experiment.py']
    record = dict(status='running', method='maddpg_binary_straight_through', seed=seed,
        config=cfg, data_manifest_sha256=sha(main.DATA / 'manifest.json'),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        initial_actor_sha256=state_hash(actor), initial_critic_sha256=state_hash(critic),
        started_at_utc=now(), agents=env.num_agents, state_dim=env.state_dim,
        action_dim=env.max_targets_per_cav,
        note='Original centralized team critic; all current actors jointly optimize team Q. Fixed-layout fading and reward use the shared simulator. Exploration and binary straight-through are adaptations.')
    history, validations = [], []
    best, best_episode, steps, updates = -float('inf'), None, 0, 0
    start, elapsed_before = 1, 0.
    checkpoint = output / 'checkpoint.pt'
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError('No saved MADDPG checkpoint to resume.')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for key in ('seed', 'config', 'data_manifest_sha256', 'code_sha256',
                    'initial_actor_sha256', 'initial_critic_sha256'):
            if saved['record'][key] != record[key]:
                raise ValueError(f'Checkpoint mismatch: {key}')
        if saved['episode'] >= cfg['episodes']:
            raise ValueError('This run has already reached its requested training budget.')
        actor.load_state_dict(saved['actor']); target_actor.load_state_dict(saved['target_actor'])
        critic.load_state_dict(saved['critic']); target_critic.load_state_dict(saved['target_critic'])
        ao.load_state_dict(saved['actor_optimizer']); co.load_state_dict(saved['critic_optimizer'])
        torch.set_rng_state(saved['torch_rng']); scene_rng.bit_generator.state = saved['scene_rng']
        replay_rng.bit_generator.state = saved['replay_rng']; replay.extend(saved['replay'])
        history, validations = saved['history'], saved['validations']
        best, best_episode, steps, updates = saved['best'], saved['best_episode'], saved['steps'], saved['updates']
        start, elapsed_before = saved['episode'] + 1, saved['record']['elapsed_seconds']
        record.update(started_at_utc=saved['record']['started_at_utc'], resumed_from_episode=saved['episode'])
    write_json(output / 'run.json', record)
    started = time.perf_counter()
    with main.adapted():
        def validate(episode):
            nonlocal best, best_episode
            rows = main.runner.evaluate_frozen(actor, val_records, main.DATA, main.config(), namespace='validation', channels=1)
            scores = main.runner.summary(rows); validations.append(dict(episode=episode, **scores))
            if episode and scores['reward'] > best:
                best, best_episode = scores['reward'], episode
                torch.save(dict(policy=actor.state_dict(), episode=episode, validation_reward=best), output / 'best.pt')
            print(f'MADDPG seed={seed} validation episode={episode} reward={scores["reward"]:.5f}', flush=True)

        if start == 1:
            validate(0)
        for episode in range(start, cfg['episodes'] + 1):
            index = int(scene_rng.choice(len(envs), size=1, replace=False)[0]); env = envs[index]
            obs = env.reset(seed=0); rewards, coverage, payload = [], [], []
            for _ in range(cfg['episode_steps']):
                previous = env.previous_actions.copy()
                with torch.no_grad():
                    probs = actor.probabilities(torch.from_numpy(obs)[None])[0]
                    action = (torch.bernoulli(probs).bool().numpy() & env.mask)
                nxt, reward, done, info = env.step(action)
                replay.append((index, np.packbits(previous.ravel()), np.packbits(action.ravel()), float(np.mean(reward))))
                obs = nxt; steps += 1; rewards.append(float(np.mean(reward)))
                coverage.append(info['coverage_rate']); payload.append(info['payload'])
            if not done:
                raise AssertionError('Expected collection window to end.')
            aloss = closs = None
            if steps >= cfg['warmup_steps'] and len(replay) >= cfg['batch_size']:
                for _ in range(cfg['updates_per_window']):
                    aloss, closs = update(actor, target_actor, critic, target_critic, ao, co,
                        sample_batch(replay, replay_rng, envs, cfg['batch_size']), cfg['gamma'], cfg['tau'])
                    updates += 1
            history.append(dict(episode=episode, scene_id=train_records[index]['scene_id'],
                reward=float(np.mean(rewards)), coverage=float(np.mean(coverage)), payload=float(np.mean(payload)),
                joint_steps=steps, optimizer_updates=updates, actor_loss=aloss, critic_loss=closs))
            if episode % cfg['validation_every'] == 0 or episode == cfg['episodes']:
                validate(episode)
                write_csv(output / 'training.csv', history); write_csv(output / 'validation.csv', validations)
                record.update(completed_episodes=episode, joint_steps=steps, optimizer_updates=updates,
                    best_validation_reward=best, selected_episode=best_episode, elapsed_seconds=elapsed_before+time.perf_counter()-started)
                saved = dict(record=record.copy(), episode=episode, actor=actor.state_dict(), target_actor=target_actor.state_dict(),
                    critic=critic.state_dict(), target_critic=target_critic.state_dict(), actor_optimizer=ao.state_dict(),
                    critic_optimizer=co.state_dict(), torch_rng=torch.get_rng_state(), scene_rng=scene_rng.bit_generator.state,
                    replay_rng=replay_rng.bit_generator.state, replay=list(replay), history=history, validations=validations,
                    best=best, best_episode=best_episode, steps=steps, updates=updates)
                temporary = output / 'checkpoint.pending.pt'
                torch.save(saved, temporary); temporary.replace(checkpoint)
                write_json(output / 'run.json', record)
                print(f'MADDPG seed={seed} {episode}/{cfg["episodes"]} joint_steps={steps} updates={updates}', flush=True)
                if stop_after is not None and episode >= stop_after and episode < cfg['episodes']:
                    return record
    record.update(status='completed', finished_at_utc=now(), selected_checkpoint_sha256=sha(output / 'best.pt'))
    write_json(output / 'run.json', record)
    return record


def smoke():
    with tempfile.TemporaryDirectory(prefix='maddpg_smoke_', dir=ROOT / 'major/maddpg') as tmp:
        root = Path(tmp); cfg = settings(2, smoke=True); cfg['validation_every'] = 1
        reference = train(42, root / 'reference', cfg)
        partial = train(42, root / 'resumed', cfg, stop_after=1)
        assert partial['completed_episodes'] == 1
        result = train(42, root / 'resumed', cfg, resume=True)
        assert result['status'] == 'completed' and result['joint_steps'] == 14 and result['optimizer_updates'] == 2
        a = torch.load(root / 'reference/checkpoint.pt', map_location='cpu', weights_only=False)
        b = torch.load(root / 'resumed/checkpoint.pt', map_location='cpu', weights_only=False)
        for key in ('actor', 'target_actor', 'critic', 'target_critic'):
            for tensor in a[key]:
                torch.testing.assert_close(a[key][tensor], b[key][tensor], atol=0, rtol=0)
        assert a['history'] == b['history'] and reference['best_validation_reward'] == result['best_validation_reward']
    print('MADDPG shared-environment smoke passed; temporary results removed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('smoke', 'train'))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'smoke':
        smoke()
    else:
        train(args.seed, OUTPUT / f'seed_{args.seed}', settings(), resume=args.resume)
