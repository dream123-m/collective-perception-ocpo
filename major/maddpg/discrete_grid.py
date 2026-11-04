"""ST Gumbel-Softmax binary-action adaptation of the existing MADDPG grid.

Each object has two logits [0, z] (do not send, send). Interaction, replay,
critic fitting, critic targets and actor-loss forward passes all use binary
actions. Only the actor derivative uses the temperature-1 soft relaxation.
The existing independent actors and shared cooperative team critic are retained;
this is an action-adapter comparison, not a per-agent-Q reproduction of MADDPG.
All 18 cases use the common MAPPO reward, archived layouts and seed 123.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import statistics
import tempfile
import time
import traceback
from pathlib import Path

for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import numpy as np
import torch
from torch import nn

from major.maddpg import adapter, scenario_grid as previous
from major.maddpg.simple_baseline import clip_per_actor, focal_q
from major.maddpg.original.maddpg_network import CriticNetwork
from major.cross_scene.layout_only import FixedFadingEnv, FIXED_FADING_SEED
from major.cross_scene.run import now, sha, state_hash, write_csv, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'major/maddpg/results/discrete_grid'
SEED = 123
CASES = [(d, p, target) for d, p in previous.ORDER for target in (.95, .90)]
MATCH_FIELDS = ('scene_id', 'scene_sha256', 'channel_sha256', 'unit_fading_sha256', 'role_sha256', 'steps')


class BinaryActors(adapter.JointActors):
    """Same networks/initial parameters; one logit encodes two action scores."""

    def scores(self, states):
        x = states
        for j, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum('bni,noi->bno', x, weight) + bias[None]
            if j < 2:
                x = torch.relu(x)
        return torch.stack((torch.zeros_like(x), x), dim=-1)

    def deterministic(self, states, masks):
        return self.scores(states).argmax(dim=-1).to(states.dtype) * masks

    def sample(self, states, masks, generator, temperature=1.0):
        scores = self.scores(states)
        eps = torch.finfo(scores.dtype).eps
        uniforms = torch.rand(scores.shape, dtype=scores.dtype, device=scores.device,
                              generator=generator).clamp(eps, 1 - eps)
        perturbed = scores - torch.log(-torch.log(uniforms))
        soft = torch.softmax(perturbed / temperature, dim=-1)
        hard = nn.functional.one_hot(perturbed.argmax(dim=-1), num_classes=2).to(soft.dtype)
        # Parentheses give exactly binary forward values; derivative is soft's.
        straight_through = hard + (soft - soft.detach())
        return straight_through[..., 1] * masks


def binary(tensor):
    return bool(torch.all((tensor == 0) | (tensor == 1)))


def update(actor, target_actor, critic, target_critic, actor_optimizer,
           critic_optimizer, batch, generator, gamma=.95, tau=.01):
    states, actions, rewards, following, masks = batch
    if not binary(actions):
        raise AssertionError('The replay contains a nonbinary action.')
    with torch.no_grad():
        next_actions = target_actor.sample(following, masks, generator)
        # Collection boundaries truncate a continuing task and bootstrap.
        targets = rewards + gamma * target_critic(following, next_actions)
    critic_loss = nn.functional.mse_loss(critic(states, actions), targets)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), .5)
    critic_optimizer.step()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    own_actions = actor.sample(states, masks, generator)
    if not binary(own_actions) or not binary(next_actions):
        raise AssertionError('A forward action is not binary.')
    # The focal agent changes its action; peers stay at their replayed actions.
    actor_loss = -focal_q(critic, states, actions, own_actions).mean(0).sum()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    clip_per_actor(actor, .5)
    actor_optimizer.step()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        for target, model in ((target_actor, actor), (target_critic, critic)):
            for destination, parameter in zip(target.parameters(), model.parameters()):
                destination.lerp_(parameter, tau)
    if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
        raise FloatingPointError('Nonfinite loss.')
    return float(actor_loss.detach() / actor.num_agents), float(critic_loss.detach())


def evaluate(actor, records, data, config):
    """Same frozen layout evaluation as MAPPO, with explicit binary argmax."""
    runner = previous.main.runner
    old_hash, rng = state_hash(actor), torch.get_rng_state().clone()
    was_training = actor.training
    actor.eval()
    rows = []
    try:
        with torch.inference_mode():
            for record in records:
                env = FixedFadingEnv(data / record['path'], seed=FIXED_FADING_SEED,
                                     config=runner.environment_config(config))
                masks = torch.from_numpy(env.mask)
                obs = torch.from_numpy(env.reset(seed=FIXED_FADING_SEED))
                rewards, coverage, payload = [], [], []
                bound, channel_hash = env.all_on_coverage(), env.channel_hash
                for _ in range(config['episode_steps']):
                    actions = actor.deterministic(obs[None], masks)[0]
                    nxt, reward, done, info = env.step(actions.numpy())
                    rewards.append(float(np.mean(reward)))
                    coverage.append(float(info['coverage_rate']))
                    payload.append(float(actions.sum()) / env.num_agents)
                    obs = torch.from_numpy(nxt)
                    if done:
                        break
                rows.append(dict(scene_id=record['scene_id'], scene_sha256=record['scene_sha256'],
                    channel_seed=env.channel_seed, channel_sha256=channel_hash,
                    fixed_fading_seed=FIXED_FADING_SEED, unit_fading_sha256=env.unit_fading_sha256,
                    role_sha256=env.role_sha256, replicate=0, reward=statistics.mean(rewards),
                    coverage=statistics.mean(coverage), payload=statistics.mean(payload),
                    all_on_coverage=float(bound), steps=len(rewards)))
    finally:
        actor.train(was_training)
    if old_hash != state_hash(actor) or not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError('Evaluation changed the policy or training RNG.')
    return rows


def source_hashes():
    paths = [Path(__file__), Path(adapter.__file__),
             ROOT / 'major/maddpg/simple_baseline.py',
             ROOT / 'major/maddpg/original/maddpg_network.py',
             Path(previous.__file__), ROOT / 'major/cross_scene/environment.py',
             ROOT / 'major/cross_scene/layout_only.py',
             ROOT / 'major/cross_scene/legacy_layout_experiment.py',
             ROOT / 'major/cross_scene/legacy_scenario_grid.py']
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def run_one(density, penetration, target, *, output=OUT, smoke=False, stop_after=None,
            warmup_check=False):
    from collections import deque

    primary = (density, penetration) == (50, .5)
    source = previous.grid.source_info(density, penetration)
    data = previous.main.DATA if primary else previous.grid.DATA / previous.grid.key(density, penetration)
    bank = previous.main.load_bank(data) if primary else previous.grid.load_bank(data)
    env_config = previous.grid.config_for(previous.main, source, target)
    folder = Path(output).resolve() / previous.grid.key(density, penetration) / f'target_{target:.2f}'
    if not folder.is_relative_to(ROOT / 'major'):
        raise ValueError('Output must be inside major.')
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    config = dict(episodes=4 if smoke else 1500, episode_steps=env_config['episode_steps'],
        validation_every=2 if smoke else 100, batch_size=4 if smoke else 32,
        replay_capacity=2048, warmup_steps=0 if smoke and not warmup_check else 1000,
        updates_per_window=1, gamma=.95, tau=.01, actor_lr=1e-4, critic_lr=1e-3,
        gumbel_temperature=1., deterministic_action='argmax([no-send, send])',
        target_coverage=target, penalty_exponent=1.4, coverage_reward_sigma=.04,
        reward_weights=[.2, .6, .2, 1.4, 6.], smoke=smoke)
    train, validation = bank['splits']['train'], bank['splits']['validation']
    context = previous.main.adapted() if primary else previous.grid.context(data, source)
    with context:
        effective = previous.main.runner.environment_config(env_config)
        envs = [FixedFadingEnv(data / row['path'], seed=0, config=effective) for row in train]
        env = envs[0]
        for name, expected in [('penalty_exponent',1.4), ('coverage_reward_sigma',.04),
                               ('w1',.2), ('w2',.6), ('w3',.2), ('w5',6.)]:
            if env.config[name] != expected:
                raise AssertionError(f'Common reward differs: {name}')
        actor = BinaryActors(env.state_dim, env.num_agents, env.max_targets_per_cav)
        critic = CriticNetwork(env.state_dim, env.max_targets_per_cav, env.num_agents, [128,128])
        target_actor, target_critic = copy.deepcopy(actor), copy.deepcopy(critic)
        ao = torch.optim.Adam(actor.parameters(), lr=config['actor_lr'])
        co = torch.optim.Adam(critic.parameters(), lr=config['critic_lr'])
        scene_rng = np.random.default_rng(SEED + 700000)
        replay_rng = np.random.default_rng(SEED + 910000)
        action_rng = torch.Generator().manual_seed(SEED + 300000)
        update_rng = torch.Generator().manual_seed(SEED + 920000)
        replay = deque(maxlen=config['replay_capacity'])
        identity = dict(seed=SEED, density=density, penetration=penetration, target=target,
            config=config, environment_config=env.config, code_sha256=source_hashes(),
            manifest_sha256=sha(data / 'manifest.json'),
            initial_actor_sha256=state_hash(actor), initial_critic_sha256=state_hash(critic),
            method='MADDPG_STGS_shared_team_critic',
            selection='Maximum validation reward after at least one optimizer update; earliest on tie.',
            action='ST Gumbel-Softmax at temperature 1; binary interaction/replay/Q inputs; deterministic argmax test.',
            adaptation='Independent original actors; one existing shared team-Q critic; peers fixed to replay during actor updates.')
        result_path = folder / 'test_summary.json'
        if result_path.exists():
            rec = previous.read(folder / 'run.json')
            if any(rec[key] != value for key, value in identity.items()):
                raise ValueError('Completed case has a different protocol.')
            if sha(folder / 'best_trained.pt') != rec['checkpoint_sha256']:
                raise ValueError('Completed checkpoint hash differs.')
            return previous.read(result_path)
        histories, validations = [], []
        best, best_episode, steps, updates, start, elapsed = -float('inf'), None, 0, 0, 1, 0.
        checkpoint = folder / 'checkpoint.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('Resume protocol, data or source hash differs.')
            for model, key in ((actor,'actor'), (critic,'critic'), (target_actor,'target_actor'),
                               (target_critic,'target_critic'), (ao,'ao'), (co,'co')):
                model.load_state_dict(saved[key])
            torch.set_rng_state(saved['torch_rng'])
            action_rng.set_state(saved['action_rng'])
            update_rng.set_state(saved['update_rng'])
            scene_rng.bit_generator.state = saved['scene_rng']
            replay_rng.bit_generator.state = saved['replay_rng']
            replay.extend(saved['replay'])
            histories, validations = saved['histories'], saved['validations']
            best, best_episode = saved['best'], saved['best_episode']
            steps, updates, start, elapsed = saved['steps'], saved['updates'], saved['episode'] + 1, saved['elapsed']
        elif (folder / 'run.json').exists():
            raise RuntimeError('Incomplete case has no recoverable checkpoint.')
        record = dict(identity, status='running', pid=os.getpid(), started_at_utc=now())
        write_json(folder / 'run.json', record)
        started = time.perf_counter()

        def validate(episode):
            nonlocal best, best_episode
            scores = previous.main.runner.summary(evaluate(actor, validation, data, env_config))
            validations.append(dict(episode=episode, **scores))
            if updates and scores['reward'] > best:
                best, best_episode = scores['reward'], episode
                torch.save(dict(policy=actor.state_dict(), episode=episode,
                    validation_reward=best, method=identity['method'], config=config), folder / 'best_trained.pt')

        if start == 1:
            validate(0)
        for episode in range(start, config['episodes'] + 1):
            index = int(scene_rng.choice(len(envs), size=1, replace=False)[0])
            env = envs[index]
            obs = env.reset(seed=0)
            masks = torch.from_numpy(env.mask)
            rewards, coverage, payload = [], [], []
            for _ in range(config['episode_steps']):
                prior_actions = env.previous_actions.copy()
                with torch.no_grad():
                    actions = actor.sample(torch.from_numpy(obs)[None], masks, action_rng)[0].numpy()
                nxt, reward, done, info = env.step(actions)
                replay.append((index, np.packbits(prior_actions.ravel()), np.packbits(actions.astype(bool).ravel()),
                               float(np.mean(reward))))
                if smoke and len(replay) == 1:
                    rebuilt = adapter.sample_batch([replay[-1]], np.random.default_rng(0), envs, 1)
                    np.testing.assert_array_equal(rebuilt[0][0].numpy(), obs)
                    np.testing.assert_array_equal(rebuilt[1][0].numpy(), actions)
                    np.testing.assert_array_equal(rebuilt[3][0].numpy(), nxt)
                    np.testing.assert_array_equal(rebuilt[4][0].numpy(), env.mask)
                obs = nxt
                steps += 1
                rewards.append(float(np.mean(reward)))
                coverage.append(float(info['coverage_rate']))
                payload.append(float(info['payload']))
            if not done:
                raise AssertionError('Collection window did not end.')
            actor_loss = critic_loss = None
            if steps >= config['warmup_steps'] and len(replay) >= config['batch_size']:
                batch = adapter.sample_batch(replay, replay_rng, envs, config['batch_size'])
                actor_loss, critic_loss = update(actor, target_actor, critic, target_critic,
                    ao, co, batch, update_rng, config['gamma'], config['tau'])
                updates += 1
            histories.append(dict(episode=episode, scene_id=train[index]['scene_id'],
                reward=statistics.mean(rewards), coverage=statistics.mean(coverage),
                payload=statistics.mean(payload), joint_steps=steps, optimizer_updates=updates,
                actor_loss=actor_loss, critic_loss=critic_loss))
            if episode % config['validation_every'] == 0 or episode == config['episodes']:
                validate(episode)
                elapsed_now = elapsed + time.perf_counter() - started
                record.update(completed_episodes=episode, joint_steps=steps, optimizer_updates=updates,
                    selected_episode=best_episode,
                    best_validation_reward=best if best_episode is not None else None,
                    elapsed_seconds=elapsed_now)
                write_csv(folder / 'training.csv', histories)
                write_csv(folder / 'validation.csv', validations)
                saved = dict(identity=identity, episode=episode, actor=actor.state_dict(), critic=critic.state_dict(),
                    target_actor=target_actor.state_dict(), target_critic=target_critic.state_dict(),
                    ao=ao.state_dict(), co=co.state_dict(), torch_rng=torch.get_rng_state(),
                    action_rng=action_rng.get_state(), update_rng=update_rng.get_state(),
                    scene_rng=scene_rng.bit_generator.state, replay_rng=replay_rng.bit_generator.state,
                    replay=list(replay), histories=histories, validations=validations, best=best,
                    best_episode=best_episode, steps=steps, updates=updates, elapsed=elapsed_now)
                pending = folder / 'checkpoint.pending.pt'
                torch.save(saved, pending)
                pending.replace(checkpoint)
                write_json(folder / 'run.json', record)
                print(f'{density}/{penetration}/{target}: {episode}/{config["episodes"]}; {elapsed_now:.1f}s', flush=True)
                if stop_after is not None and episode >= stop_after and episode < config['episodes']:
                    return None
        if best_episode is None:
            raise AssertionError('No trained checkpoint was selected.')
        selected = torch.load(folder / 'best_trained.pt', map_location='cpu', weights_only=False)
        actor.load_state_dict(selected['policy'])
        rows = evaluate(actor, bank['splits']['test'], data, env_config)
        if len(rows) != 32:
            raise AssertionError('Expected 32 test layouts.')
        ref_path = previous.grid.BASE / 'single_seed/exploratory_test/per_layout.csv'
        refs = [r for r in read_csv(ref_path) if int(r['density']) == density
                and float(r['penetration']) == penetration and float(r['target']) == target and r['method'] == 'mappo']
        if len(refs) != 32 or any(any(str(a[key]) != str(b[key]) for key in MATCH_FIELDS) for a,b in zip(rows,refs)):
            raise AssertionError('Test layouts, fading, roles or interaction budget differ from MAPPO.')
        baseline = (previous.main.OUTPUT / f'mappo_seed_{SEED}' if primary and target == .95 else
                    previous.grid.TRAIN / previous.grid.key(density, penetration) / f'target_{target:.2f}' / f'mappo_seed_{SEED}')
        expected = read_csv(baseline / 'training.csv')
        if [r['scene_id'] for r in histories] != [r['scene_id'] for r in expected[:config['episodes']]]:
            raise AssertionError('Training layout sequence differs from MAPPO.')
        write_csv(folder / 'test.csv', rows)
        record.update(status='completed', finished_at_utc=now(), checkpoint_sha256=sha(folder/'best_trained.pt'),
                      final_actor_sha256=state_hash(actor), test_csv_sha256=sha(folder/'test.csv'))
        write_json(folder / 'run.json', record)
        result = dict(status='completed', density=density, penetration=penetration, target=target, seed=SEED,
            method=identity['method'], training_episodes=config['episodes'], selected_episode=best_episode,
            manifest_sha256=identity['manifest_sha256'], checkpoint_sha256=record['checkpoint_sha256'],
            test_csv_sha256=record['test_csv_sha256'], elapsed_seconds=record['elapsed_seconds'],
            **{key: statistics.mean(float(row[key]) for row in rows) for key in ('reward','coverage','payload')})
        write_json(result_path, result)
        if not smoke:
            checkpoint.unlink()  # Completed case keeps its selected model and all curves.
        return result


def checks():
    """Action gradients, replay consistency, actual environments, exact resume."""
    from major.maddpg.simple_baseline import check as check_focal_gradient

    check_focal_gradient()
    torch.set_num_threads(1)
    torch.manual_seed(91)
    actor = BinaryActors(7, 3, 4)
    states = torch.randn(2,3,7)
    masks = torch.ones(2,3,4)
    masks[:,0,0] = 0
    generator = torch.Generator().manual_seed(11)
    values = actor.sample(states, masks, generator)
    assert binary(values) and torch.equal(values[:,0,0], torch.zeros(2))
    values.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    assert sum(float(p.grad.abs().sum()) for p in actor.parameters()) > 0
    saved_rng = generator.get_state().clone()
    assert binary(actor.deterministic(states,masks))
    assert torch.equal(saved_rng,generator.get_state())
    with tempfile.TemporaryDirectory(prefix='discrete_check_',dir=ROOT/'major/maddpg') as temporary:
        root = Path(temporary)
        run_one(50,.5,.95,output=root/'warmup',smoke=True,stop_after=2,warmup_check=True)
        warmup = previous.read(root/'warmup'/previous.grid.key(50,.5)/'target_0.95/run.json')
        assert warmup['optimizer_updates'] == 0 and warmup['best_validation_reward'] is None
        for d,p,t in ((50,.5,.95),(25,.3,.90),(75,.7,.95)):
            result = run_one(d,p,t,output=root/'full',smoke=True)
            assert result['training_episodes'] == 4
        run_one(50,.5,.95,output=root/'resumed',smoke=True,stop_after=2)
        run_one(50,.5,.95,output=root/'resumed',smoke=True)
        name = Path(previous.grid.key(50,.5))/'target_0.95'
        first = torch.load(root/'full'/name/'checkpoint.pt',weights_only=False)
        second = torch.load(root/'resumed'/name/'checkpoint.pt',weights_only=False)
        def tensor_digest(value):
            h = hashlib.sha256()
            def visit(x):
                if torch.is_tensor(x):
                    h.update(str((x.dtype,tuple(x.shape))).encode()); h.update(x.numpy().tobytes())
                elif isinstance(x,dict):
                    for key in sorted(x,key=str): h.update(str(key).encode()); visit(x[key])
                elif isinstance(x,(list,tuple)):
                    for item in x: visit(item)
                else: h.update(str(x).encode())
            visit(value)
            return h.hexdigest()
        for key in ('actor','critic','target_actor','target_critic','ao','co','action_rng','update_rng','torch_rng'):
            assert tensor_digest(first[key]) == tensor_digest(second[key]), key
        assert first['histories'] == second['histories'] and first['validations'] == second['validations']
        assert (root/'full'/name/'test.csv').read_bytes() == (root/'resumed'/name/'test.csv').read_bytes()
    print('Passed: binary forward actions and gradients; primary/sparse/dense integrations; exact checkpoint resume.',flush=True)


def comparison(results):
    reference = read_csv(previous.grid.BASE/'single_seed/exploratory_test/per_layout.csv')
    old = previous.read(ROOT/'major/maddpg/results/reward_shape_grid/summary.json')['results']
    values = []
    for result in results:
        d,p,t = (result[key] for key in ('density','penetration','target'))
        base = dict(density=d,penetration=p,target=t)
        values.append(dict(**base,method='MADDPG_STGS',**{k:result[k] for k in ('reward','coverage','payload')}))
        for method in ('mappo','ocpo'):
            rows = [r for r in reference if int(r['density'])==d and float(r['penetration'])==p
                    and float(r['target'])==t and r['method']==method]
            assert len(rows)==32
            values.append(dict(**base,method=method.upper(),**{k:statistics.mean(float(r[k]) for r in rows) for k in ('reward','coverage','payload')}))
        prior = next(row for row in old if row['density']==d and row['penetration']==p and row['target']==t)
        values.append(dict(**base,method='MADDPG_previous_figures',**{k:prior[k] for k in ('reward','coverage','payload')}))
    write_csv(OUT/'comparison.csv',values)


def execute():
    OUT.mkdir(parents=True,exist_ok=True)
    identity = dict(seed=SEED,episodes=1500,cases=CASES,code_sha256=source_hashes(),
        common_reward=dict(w1=.2,w2=.6,w3=.2,penalty_exponent=1.4,w5=6.,coverage_reward_sigma=.04),
        action='Per-object two-class ST Gumbel-Softmax, temperature 1, argmax evaluation.',
        critic='Existing shared cooperative team critic, not per-agent critics.',
        other_changes='Binary replay and critic targets; actor updates replace only the focal action in replay.',
        selection='Maximum 16-layout validation reward after an update; no coverage-first selection.',
        evaluation='Same 32 archived test layouts/fixed fading/budget as MAPPO and OCPO; all 18 cases reported.')
    # JSON normalizes tuple entries in the frozen case list.
    identity=json.loads(json.dumps(identity))
    protocol=OUT/'protocol.json'
    if protocol.exists():
        old=previous.read(protocol)
        if old['identity']!=identity: raise ValueError('Existing grid protocol differs.')
    else:
        write_json(protocol,dict(created_at_utc=now(),identity=identity))
    lock=OUT/'driver.lock'
    handle=lock.open('x',encoding='utf-8')
    handle.write(str(os.getpid())); handle.flush()
    results=[]
    active=None
    def publish(status,**extra):
        record=dict(status=status,pid=os.getpid(),completed=len(results),total=len(CASES),
                    active=active,updated_at_utc=now(),protocol_sha256=sha(protocol),**extra)
        write_json(OUT/'status.json',record)
        write_json(OUT/'summary.json',dict(**record,results=results))
    try:
        for d,p,t in CASES:
            active=dict(density=d,penetration=p,target=t)
            publish('running')
            results.append(run_one(d,p,t))
            comparison(results)
            publish('running')
        active=None
        publish('completed',finished_at_utc=now())
    except Exception as exc:
        publish('failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        handle.close()
        lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    checks() if args.check else execute()
