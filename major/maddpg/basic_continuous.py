"""Original-source-style deterministic actor baseline on the binary environment.

Independent continuous actors, Gaussian exploration, one centralized team-Q,
replay and target networks follow the supplied F:/maddpg source. Only the
environment interface thresholds continuous actions to binary decisions.
The critic sees the continuous actions that generated those decisions. This
is a continuous-proxy adaptation, not an exact reproduction of the old
environment or a canonical per-agent-critic implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from major.cross_scene import legacy_layout_experiment as main
from major.cross_scene.layout_only import FixedFadingEnv
from major.cross_scene.run import now, sha, state_hash, write_csv, write_json
from major.maddpg.adapter import JointActors
from major.maddpg.original.maddpg_network import CriticNetwork
from major.maddpg.simple_baseline import clip_per_actor, focal_q

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'major/maddpg/results/basic_continuous'


def sample_batch(replay, rng, envs, size):
    samples = [replay[int(i)] for i in rng.choice(len(replay), size=size, replace=False)]
    indices = np.asarray([r[0] for r in samples], dtype=np.int64)
    n, slots = envs[0].mask.shape
    previous = np.unpackbits(np.stack([r[1] for r in samples]), axis=1,
                             count=n * slots).reshape(-1, n, slots)
    continuous = np.stack([r[2] for r in samples]).astype(np.float32)
    executed = continuous > .5
    states = np.stack([envs[int(i)]._template for i in indices]).copy()
    following = states.copy()
    for obs, actions in ((states, previous), (following, executed)):
        obs[:, :, 2] = actions.sum(axis=-1) / slots
        obs[:, :, 3:].reshape(-1, n, slots, 5)[:, :, :, 4] = actions
    masks = np.stack([envs[int(i)].mask for i in indices])
    rewards = np.asarray([r[3] for r in samples], np.float32).reshape(-1, 1)
    return tuple(torch.from_numpy(x) for x in (states, continuous, rewards, following, masks))


def update(actor, target_actor, critic, target_critic, ao, co, batch, gamma, tau):
    states, actions, rewards, following, masks = batch
    with torch.no_grad():
        peer = target_actor.probabilities(following) * masks
        target = rewards + gamma * target_critic(following, peer)
    critic_loss = nn.functional.mse_loss(critic(states, actions), target)
    co.zero_grad(); critic_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(), .5); co.step()
    for p in critic.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        peer_actions = target_actor.probabilities(states) * masks
    focal_actions = actor.probabilities(states) * masks
    actor_loss = -focal_q(critic, states, peer_actions, focal_actions).mean(0).sum()
    ao.zero_grad(); actor_loss.backward(); clip_per_actor(actor, .5); ao.step()
    for p in critic.parameters(): p.requires_grad_(True)
    with torch.no_grad():
        for target_net, net in ((target_actor, actor), (target_critic, critic)):
            for t, p in zip(target_net.parameters(), net.parameters()): t.lerp_(p, tau)
    if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
        raise FloatingPointError('Non-finite update loss.')
    return float(actor_loss.detach() / actor.num_agents), float(critic_loss.detach())


def run(out, seed=123, smoke=False):
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT / 'major'): raise ValueError('Outputs must remain in major.')
    if (out / 'run.json').exists(): raise FileExistsError('Existing run is protected.')
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(episodes=4 if smoke else 1500, episode_steps=7,
               validation_every=2 if smoke else 100, batch_size=4 if smoke else 32,
               replay_capacity=2048, warmup_steps=0 if smoke else 1000,
               updates_per_window=1, gamma=.95, tau=.01, actor_lr=1e-4, critic_lr=1e-3,
               gaussian_noise=.1, noise_decay=.9995, min_noise=.01,
               action_threshold=.5, target_coverage=.95, smoke=smoke)
    bank = main.load_bank(main.DATA); envcfg = main.env_config(main.config())
    train = bank['splits']['train']; val = bank['splits']['validation'][:2] if smoke else bank['splits']['validation']
    envs = [FixedFadingEnv(main.DATA / r['path'], seed=0, config=envcfg) for r in train]
    env = envs[0]
    actor = JointActors(env.state_dim, env.num_agents, env.max_targets_per_cav)
    critic = CriticNetwork(env.state_dim, env.max_targets_per_cav, env.num_agents, [128,128])
    target_actor, target_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    ao = torch.optim.Adam(actor.parameters(), lr=cfg['actor_lr'])
    co = torch.optim.Adam(critic.parameters(), lr=cfg['critic_lr'])
    scene_rng = np.random.default_rng(seed + 700000)
    replay_rng = np.random.default_rng(seed + 910000)
    noise_rng = np.random.default_rng(seed + 300000)
    replay = deque(maxlen=cfg['replay_capacity'])
    record = dict(status='running', method='continuous_proxy_shared_team_critic', seed=seed,
                  config=cfg, source_sha256=sha(Path(__file__)),
                  data_manifest_sha256=sha(main.DATA / 'manifest.json'),
                  environment_sha256=sha(ROOT / 'major/cross_scene/environment.py'),
                  initial_actor_sha256=state_hash(actor), initial_critic_sha256=state_hash(critic),
                  started_at_utc=now(),
                  definition='Original-source continuous actor and target-peer update; Gaussian exploration; '
                             'binary threshold for environment only. Same training interaction budget and '
                             'computational replay settings as the existing binary adapter.',
                  selection='Best validation reward among checkpoints after at least one optimizer update.')
    write_json(out / 'run.json', record)
    histories, validations = [], []
    best, best_ep, steps, updates, noise = -float('inf'), None, 0, 0, cfg['gaussian_noise']
    started = time.perf_counter()
    with main.adapted():
        def validate(episode):
            nonlocal best, best_ep
            scores = main.runner.summary(main.runner.evaluate_frozen(actor, val, main.DATA, main.config(),
                                                                      namespace='validation', channels=1))
            validations.append(dict(episode=episode, **scores))
            if updates and scores['reward'] > best:
                best, best_ep = scores['reward'], episode
                torch.save(dict(policy=actor.state_dict(), episode=episode,
                                validation_reward=best), out / 'best_trained.pt')
        validate(0)
        for episode in range(1, cfg['episodes'] + 1):
            index = int(scene_rng.choice(len(envs), size=1, replace=False)[0]); env = envs[index]
            obs = env.reset(seed=0); rewards, coverage, payload = [], [], []
            for _ in range(cfg['episode_steps']):
                previous = env.previous_actions.copy()
                with torch.no_grad(): probs = actor.probabilities(torch.from_numpy(obs)[None])[0].numpy()
                continuous = np.clip(probs + noise_rng.normal(0, noise, probs.shape), 0, 1).astype(np.float32)
                continuous *= env.mask
                binary = (continuous > .5) & env.mask
                nxt, reward, done, info = env.step(binary)
                replay.append((index, np.packbits(previous.ravel()), continuous, float(np.mean(reward))))
                obs = nxt; steps += 1; rewards.append(float(np.mean(reward)))
                coverage.append(info['coverage_rate']); payload.append(info['payload'])
            if not done: raise AssertionError('Expected collection window to end.')
            aloss = closs = None
            if steps >= cfg['warmup_steps'] and len(replay) >= cfg['batch_size']:
                for _ in range(cfg['updates_per_window']):
                    batch = sample_batch(replay, replay_rng, envs, cfg['batch_size'])
                    aloss, closs = update(actor, target_actor, critic, target_critic, ao, co,
                                          batch, cfg['gamma'], cfg['tau'])
                    updates += 1
            noise = max(cfg['min_noise'], noise * cfg['noise_decay'])
            histories.append(dict(episode=episode, scene_id=train[index]['scene_id'], reward=float(np.mean(rewards)),
                                  coverage=float(np.mean(coverage)), payload=float(np.mean(payload)),
                                  joint_steps=steps, optimizer_updates=updates, actor_loss=aloss, critic_loss=closs))
            if episode % cfg['validation_every'] == 0 or episode == cfg['episodes']:
                validate(episode)
                write_csv(out/'training.csv', histories); write_csv(out/'validation.csv', validations)
                record.update(completed_episodes=episode, joint_steps=steps, optimizer_updates=updates,
                              best_validation_reward=best if best_ep else None, selected_episode=best_ep,
                              elapsed_seconds=time.perf_counter()-started)
                write_json(out/'run.json', record)
                print(f'episode={episode} validation={validations[-1]["reward"]:.5f}', flush=True)
    if best_ep is None: raise AssertionError('No trained model was selected.')
    record.update(status='completed', finished_at_utc=now(), checkpoint_sha256=sha(out/'best_trained.pt'))
    write_json(out/'run.json', record)
    print(json.dumps(record, indent=2), flush=True)


def smoke():
    with tempfile.TemporaryDirectory(prefix='basic_maddpg_smoke_', dir=ROOT/'major/maddpg') as tmp:
        run(tmp, smoke=True)
        rec = json.loads((Path(tmp)/'run.json').read_text(encoding='utf-8'))
        assert rec['completed_episodes'] == 4 and rec['optimizer_updates'] == 4
    print('Four-episode integration check passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('smoke', 'train'))
    args = parser.parse_args()
    smoke() if args.command == 'smoke' else run(OUT)
