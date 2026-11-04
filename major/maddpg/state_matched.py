"""Control critic state information while retaining the saved STGS MADDPG.

The Q input is exactly MAPPO's 2T+1 broadcast-state fields plus joint actions.
Actor observations, discrete action estimation, peer replay actions, optimizers,
reward, layout sequence, validation rule and interaction budget are unchanged.
The saved full-observation version in discrete_grid/ is never modified.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from major.maddpg import discrete_grid as retained
from major.maddpg.original.maddpg_network import CriticNetwork
from major.cross_scene.run import now, sha, write_json, write_csv

ROOT = retained.ROOT
OUT = ROOT / 'major/maddpg/results/state_matched'
CASE = (50, .3, .95)  # Fixed before this control; a large-gap low-penetration case.


class SharedStateQ(CriticNetwork):
    def __init__(self, targets, agents, slots, hidden):
        self.global_dim = 2 * targets + 1
        self.agents, self.slots = agents, slots
        # Reuse the original hidden layers, activations and initialization.
        super().__init__(self.global_dim, agents * slots, 1, hidden)

    def forward(self, global_state, actions):
        if global_state.ndim != 2 or global_state.shape[1] != self.global_dim:
            raise ValueError('Q must receive only the MAPPO global-state fields.')
        return super().forward(global_state, actions)


def broadcast_state(env, actions):
    """Reconstruct pre/post state from actual replay actions, without geometry features."""
    counts = np.zeros(env.total_vehicles, dtype=np.int64)
    for sender in np.flatnonzero(env._success):
        targets = env.slot_targets[sender, actions[sender].astype(bool)]
        np.add.at(counts, targets, 1)
    covered = counts > 0
    return np.concatenate((covered.astype(np.float32),
        counts.astype(np.float32) / env.num_agents,
        np.asarray([covered.sum() / env.total_targets], dtype=np.float32)))


def sample_batch(replay, rng, envs, size):
    indices = rng.choice(len(replay), size=size, replace=False)
    rows = [replay[int(i)] for i in indices]
    layouts = np.asarray([row[0] for row in rows], dtype=np.int64)
    before = np.stack([row[1] for row in rows])
    after = np.stack([row[2] for row in rows])
    states, previous = retained.adapter.reconstruct(envs, layouts, before)
    following, actions = retained.adapter.reconstruct(envs, layouts, after)
    masks = torch.from_numpy(np.stack([envs[int(i)].mask for i in layouts]))
    rewards = torch.tensor([row[3] for row in rows], dtype=torch.float32).reshape(-1,1)
    global_before = np.stack([broadcast_state(envs[int(i)], a.numpy()) for i,a in zip(layouts,previous)])
    global_after = np.stack([broadcast_state(envs[int(i)], a.numpy()) for i,a in zip(layouts,actions)])
    return states, actions, rewards, following, masks, torch.from_numpy(global_before), torch.from_numpy(global_after)


def focal_q(critic, global_state, replay_actions, own_actions):
    batch, agents, slots = replay_actions.shape
    first = critic.shared_net[0]
    joint = torch.cat((global_state, replay_actions.reshape(batch,-1)), dim=1)
    base = first(joint)
    action_weights = first.weight[:,critic.global_dim:].reshape(first.out_features,agents,slots)
    replacement = base[:,None,:] + torch.einsum('bna,hna->bnh',own_actions-replay_actions,action_weights)
    return critic.q_head(critic.shared_net[1:](replacement)).squeeze(-1)


def update(actor, target_actor, critic, target_critic, ao, co, batch, generator, gamma=.95, tau=.01):
    states, actions, rewards, following, masks, global_state, next_global = batch
    if not retained.binary(actions):
        raise AssertionError('Replay actions are not binary.')
    with torch.no_grad():
        next_actions = target_actor.sample(following,masks,generator)
        targets = rewards + gamma * target_critic(next_global,next_actions)
    critic_loss = nn.functional.mse_loss(critic(global_state,actions),targets)
    co.zero_grad(); critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(),.5); co.step()
    for p in critic.parameters(): p.requires_grad_(False)
    own_actions = actor.sample(states,masks,generator)
    if not retained.binary(own_actions) or not retained.binary(next_actions):
        raise AssertionError('A critic action is not binary.')
    actor_loss = -focal_q(critic,global_state,actions,own_actions).mean(0).sum()
    ao.zero_grad(); actor_loss.backward(); retained.clip_per_actor(actor,.5); ao.step()
    for p in critic.parameters(): p.requires_grad_(True)
    with torch.no_grad():
        for target,model in ((target_actor,actor),(target_critic,critic)):
            for destination,p in zip(target.parameters(),model.parameters()): destination.lerp_(p,tau)
    if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
        raise FloatingPointError('Nonfinite loss.')
    return float(actor_loss.detach()/actor.num_agents),float(critic_loss.detach())


@contextlib.contextmanager
def matched_runner(targets):
    """Explicitly scoped reuse of the frozen training loop; old files stay intact."""
    original_critic, original_update = retained.CriticNetwork, retained.update
    original_sampler, original_hashes = retained.adapter.sample_batch, retained.source_hashes
    def hashes():
        return {**original_hashes(),Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))}
    retained.CriticNetwork = lambda state_dim,slots,agents,hidden: SharedStateQ(targets,agents,slots,hidden)
    retained.update, retained.adapter.sample_batch, retained.source_hashes = update, sample_batch, hashes
    try:
        yield
    finally:
        retained.CriticNetwork, retained.update = original_critic, original_update
        retained.adapter.sample_batch, retained.source_hashes = original_sampler, original_hashes


def run_case(case, *, output=OUT, smoke=False, stop_after=None):
    density,penetration,target = case
    source = retained.previous.grid.source_info(density,penetration)
    with matched_runner(source['vehicles']):
        result = retained.run_one(density,penetration,target,output=output,smoke=smoke,stop_after=stop_after)
    return result


def checks():
    torch.set_num_threads(1); torch.manual_seed(17)
    critic = SharedStateQ(5,3,4,[8,8])
    state = torch.rand(2,11); replay = torch.randint(0,2,(2,3,4)).float()
    own = torch.rand(2,3,4,requires_grad=True)
    fast = focal_q(critic,state,replay,own)
    direct=[]
    for i in range(3):
        actions=torch.stack([own[:,j] if i==j else replay[:,j] for j in range(3)],dim=1)
        direct.append(critic(state,actions).squeeze(-1))
    exact=torch.stack(direct,dim=1)
    torch.testing.assert_close(fast,exact,atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(torch.autograd.grad(fast.sum(),own,retain_graph=True)[0],
                               torch.autograd.grad(exact.sum(),own)[0],atol=1e-7,rtol=1e-5)
    # Check every actual state field against the environment/MAPPO input.
    source=retained.previous.grid.source_info(50,.3)
    data=retained.previous.grid.DATA/retained.previous.grid.key(50,.3)
    bank=retained.previous.grid.load_bank(data)
    config=retained.previous.grid.config_for(retained.previous.main,source,.95)
    with retained.previous.grid.context(data,source):
        env=retained.FixedFadingEnv(data/bank['splits']['train'][0]['path'],0,
                                   config=retained.previous.main.runner.environment_config(config))
        env.reset(seed=0)
        def compare(actions):
            full,coverage=env.global_state()
            expected=np.concatenate((full[:2*env.total_vehicles],np.asarray([coverage],np.float32)))
            np.testing.assert_array_equal(broadcast_state(env,actions),expected)
        previous=np.zeros_like(env.mask)
        compare(previous)
        rng=np.random.default_rng(12)
        for _ in range(3):
            actions=(rng.random(env.mask.shape)<.6)&env.mask
            env.step(actions); compare(actions)
    with tempfile.TemporaryDirectory(prefix='state_matched_check_',dir=ROOT/'major/maddpg') as temp:
        root=Path(temp)
        run_case(CASE,output=root/'full',smoke=True)
        run_case(CASE,output=root/'resumed',smoke=True,stop_after=2)
        run_case(CASE,output=root/'resumed',smoke=True)
        name=Path(retained.previous.grid.key(50,.3))/'target_0.95'
        a=torch.load(root/'full'/name/'checkpoint.pt',weights_only=False)
        b=torch.load(root/'resumed'/name/'checkpoint.pt',weights_only=False)
        for key in ('actor','critic','target_actor','target_critic'):
            for parameter in a[key]: torch.testing.assert_close(a[key][parameter],b[key][parameter],atol=0,rtol=0)
        assert a['histories']==b['histories'] and a['validations']==b['validations']
        assert (root/'full'/name/'test.csv').read_bytes()==(root/'resumed'/name/'test.csv').read_bytes()
    print('Passed: exact MAPPO state reconstruction, focal Q/gradient equivalence, training integration and exact resume.',flush=True)


def execute():
    OUT.mkdir(parents=True,exist_ok=True)
    density,penetration,target=CASE
    old_folder=retained.OUT/retained.previous.grid.key(density,penetration)/f'target_{target:.2f}'
    preserved=[retained.OUT/'protocol.json',retained.OUT/'summary.json',retained.OUT/'comparison.csv',
               Path(retained.__file__),old_folder/'run.json',old_folder/'best_trained.pt',old_folder/'test.csv']
    old_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in preserved}
    protocol=dict(case=CASE,seed=123,episodes=1500,source_sha256=sha(Path(__file__)),retained_files=old_hashes,
        only_change='Critic state is MAPPO broadcast flags, normalized report counts and coverage; joint actions retained for Q.',
        state_excludes='Vehicle positions, local observations, distances, visibility/co-perception features and layout context.',
        unchanged='Actor inputs/initialization, STGS actions, replay peers, reward, optimizers, training/test layouts, budget and validation selection.',
        global_state_dimensions='2 * number_of_objects + 1; Q additionally receives all binary action bits.',
        comparison='State-information control, not a test of universally better or worse algorithms.')
    protocol=json.loads(json.dumps(protocol))
    path=OUT/'protocol.json'
    if path.exists():
        if retained.previous.read(path)['identity']!=protocol: raise ValueError('State-control protocol changed.')
    else: write_json(path,dict(created_at_utc=now(),identity=protocol))
    lock=OUT/'driver.lock'; handle=lock.open('x',encoding='utf-8')
    handle.write(str(os.getpid())); handle.flush()
    write_json(OUT/'status.json',dict(status='running',pid=os.getpid(),case=CASE,started_at_utc=now()))
    try:
        result=run_case(CASE)
        new_folder=OUT/retained.previous.grid.key(density,penetration)/f'target_{target:.2f}'
        old_run=retained.previous.read(old_folder/'run.json')
        new_run=retained.previous.read(new_folder/'run.json')
        assert old_run['initial_actor_sha256']==new_run['initial_actor_sha256']
        assert old_run['config']==new_run['config']
        assert old_run['environment_config']==new_run['environment_config']
        assert old_run['manifest_sha256']==new_run['manifest_sha256']
        old_training=retained.read_csv(old_folder/'training.csv')
        new_training=retained.read_csv(new_folder/'training.csv')
        assert [r['scene_id'] for r in old_training]==[r['scene_id'] for r in new_training]
        old_test,new_test=retained.read_csv(old_folder/'test.csv'),retained.read_csv(new_folder/'test.csv')
        assert all(all(a[k]==b[k] for k in retained.MATCH_FIELDS) for a,b in zip(old_test,new_test))
        assert all(sha(ROOT/name)==digest for name,digest in old_hashes.items())
        rows=[]
        reference=retained.read_csv(retained.OUT/'comparison.csv')
        for method in ('MAPPO','OCPO','MADDPG_STGS'):
            row=next(r for r in reference if int(r['density'])==density and float(r['penetration'])==penetration
                     and float(r['target'])==target and r['method']==method)
            rows.append(dict(method=method,**{k:float(row[k]) for k in ('reward','coverage','payload')}))
        rows.append(dict(method='MADDPG_STGS_MAPPO_state',**{k:result[k] for k in ('reward','coverage','payload')}))
        write_csv(OUT/'comparison.csv',rows)
        write_json(OUT/'summary.json',dict(status='completed',result=result,comparison=rows,
            identical_actor_initialization=True,identical_training_layout_sequence=True,
            identical_reward_and_training_settings=True,identical_test_conditions=True,
            retained_files_unchanged=True,finished_at_utc=now()))
        write_json(OUT/'status.json',dict(status='completed',case=CASE,finished_at_utc=now()))
    except Exception as exc:
        write_json(OUT/'status.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        handle.close(); lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    checks() if args.check else execute()
