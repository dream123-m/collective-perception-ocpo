"""Simple cooperative MADDPG baseline for the common binary-action task.

Keep the user's original single team-Q critic. For each actor, evaluate its
own binary decision with other actors' actions fixed to the replayed actions,
as in the original MADDPG actor-update structure. A straight-through binary
derivative connects actor probabilities to actual 0/1 simulator decisions.
There is no overlap module, search, teacher, or added reward. The common team
critic is a cooperative simplification, not a claim of independent Q_i critics.
"""
from __future__ import annotations
import argparse
import copy
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from major.cross_scene import legacy_layout_experiment as main
from major.cross_scene.run import acquire_batch_lock, now, sha, write_json
from major.maddpg import adapter
from major.maddpg.original.maddpg_network import CriticNetwork

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'major/maddpg/results/simple_baseline'


def focal_q(critic, states, replay_actions, focal_actions):
    """Evaluate Q once per focal actor with all other actions held at replay."""
    batch,n,slots=replay_actions.shape
    joint=torch.cat((states.reshape(batch,-1),replay_actions.reshape(batch,-1)),dim=1)
    first=critic.shared_net[0]
    base=first(joint)
    # This is algebraically the same as constructing N separate joint-action
    # vectors and replacing just the focal actor's entry in each one.
    action_weights=first.weight[:,n*states.shape[-1]:].reshape(first.out_features,n,slots)
    delta=focal_actions-replay_actions
    replaced=base[:,None,:]+torch.einsum('bna,hna->bnh',delta,action_weights)
    return critic.q_head(critic.shared_net[1:](replaced)).squeeze(-1)


def clip_per_actor(actor,max_norm):
    parameters=list(actor.weights)+list(actor.biases)
    norms=torch.zeros(actor.num_agents)
    for p in parameters:
        if p.grad is not None:norms+=p.grad.flatten(1).pow(2).sum(1)
    scale=(max_norm/(norms.sqrt()+1e-6)).clamp_max(1)
    for p in parameters:
        if p.grad is not None:p.grad.mul_(scale.reshape((-1,)+(1,)*(p.ndim-1)))


def update(actor,target_actor,critic,target_critic,ao,co,batch,gamma,tau):
    states,actions,rewards,next_states,masks=batch
    with torch.no_grad():
        next_action=(target_actor.probabilities(next_states)>.5).float()*masks
        target=rewards+gamma*target_critic(next_states,next_action)
    value=critic(states,actions)
    critic_loss=nn.functional.mse_loss(value,target)
    co.zero_grad();critic_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),.5);co.step()
    for p in critic.parameters():p.requires_grad_(False)
    probs=actor.probabilities(states); hard=(probs>.5).float()
    own=(hard+probs-probs.detach())*masks
    loss=-focal_q(critic,states,actions,own).mean(0).sum()
    ao.zero_grad();loss.backward();clip_per_actor(actor,.5);ao.step()
    for p in critic.parameters():p.requires_grad_(True)
    with torch.no_grad():
        for target_model,model in ((target_actor,actor),(target_critic,critic)):
            for target_p,p in zip(target_model.parameters(),model.parameters()):target_p.lerp_(p,tau)
    if not torch.isfinite(loss) or not torch.isfinite(critic_loss):raise FloatingPointError('Non-finite loss.')
    return float(loss.detach()/actor.num_agents),float(critic_loss.detach())


def check():
    """Check focal-action substitution and gradients against a direct loop."""
    torch.set_num_threads(1);torch.manual_seed(17)
    batch,n,dim,slots=3,4,7,5
    critic=CriticNetwork(dim,slots,n,[8,8]);states=torch.randn(batch,n,dim)
    replay=torch.randint(0,2,(batch,n,slots)).float();own=torch.rand(batch,n,slots,requires_grad=True)
    fast=focal_q(critic,states,replay,own)
    direct=[]
    for i in range(n):
        a=torch.stack([own[:,j] if j==i else replay[:,j] for j in range(n)],dim=1)
        direct.append(critic(states,a).squeeze(-1))
    reference=torch.stack(direct,dim=1)
    torch.testing.assert_close(fast,reference,atol=1e-7,rtol=1e-5)
    ga=torch.autograd.grad(fast.sum(),own,retain_graph=True)[0]
    gb=torch.autograd.grad(reference.sum(),own)[0]
    torch.testing.assert_close(ga,gb,atol=1e-7,rtol=1e-5)
    print('Focal-action Q and gradients match the direct independent-actor update.',flush=True)


def run(seed):
    if seed not in adapter.SEEDS:raise ValueError('Unknown seed.')
    OUT.mkdir(parents=True,exist_ok=True); destination=OUT/f'seed_{seed}'
    lock=acquire_batch_lock(OUT);status=OUT/'status.json'
    try:
        if destination.exists():raise FileExistsError('Existing run is protected.')
        check()
        cfg=adapter.settings();seed_dir=destination
        plan=dict(seed=seed,started_at_utc=now(),
            implementation='Independent actors, one team-Q critic, replay, target networks, focal actor replacement using replay actions for peers.',
            action='Binary threshold with straight-through actor derivative; Bernoulli training exploration.',
            config=cfg,source_sha256=sha(Path(__file__)),adapter_sha256=sha(Path(adapter.__file__)),
            data_manifest_sha256=sha(main.DATA/'manifest.json'),
            selection='Highest validation reward among checkpoints after at least one optimizer update; earliest on tie.',
            scope='Single-seed development on previously used train/validation layouts; no test results opened.')
        write_json(OUT/f'plan_seed_{seed}.json',plan)
        write_json(status,dict(status='running',seed=seed,started_at_utc=now()))
        old_update=adapter.update;old_eval=main.runner.evaluate_frozen
        capture={'index':0,'best':float('-inf'),'episode':None}
        def validation(policy,records,data,config,*,namespace,channels):
            rows=old_eval(policy,records,data,config,namespace=namespace,channels=channels)
            if namespace=='validation':
                episode=capture['index']*cfg['validation_every'];capture['index']+=1
                score=main.runner.summary(rows)['reward']
                if episode*cfg['episode_steps']>=cfg['warmup_steps'] and score>capture['best']:
                    capture.update(best=score,episode=episode)
                    seed_dir.mkdir(parents=True,exist_ok=True)
                    tmp=seed_dir/'best_trained.pending.pt'
                    torch.save(dict(policy=policy.state_dict(),episode=episode,validation_reward=score,
                                    variant_source_sha256=sha(Path(__file__))),tmp)
                    tmp.replace(seed_dir/'best_trained.pt')
            return rows
        adapter.update=update;main.runner.evaluate_frozen=validation
        try: record=adapter.train(seed,seed_dir,cfg)
        finally: adapter.update=old_update;main.runner.evaluate_frozen=old_eval
        if record['status']!='completed' or capture['episode'] is None:
            raise AssertionError('Training or post-update model selection incomplete.')
        # The reused generic runner embeds its default method note. Record the
        # active injected update explicitly so a reader cannot confuse the two.
        record.update(method='maddpg_focal_team_q_binary',
            note='Focal actor update with peer actions fixed to replay; one shared team critic; binary straight-through.',
            variant_source_sha256=sha(Path(__file__)))
        write_json(seed_dir/'run.json',record)
        prior=json.loads((adapter.OUTPUT/f'seed_{seed}'/'run.json').read_text(encoding='utf8'))
        result=dict(status='completed',seed=seed,finished_at_utc=now(),
            selected_episode=capture['episode'],validation_reward=capture['best'],
            trained_actor_sha256=sha(seed_dir/'best_trained.pt'),
            original_adaptation_selected_episode=prior['selected_episode'],
            original_adaptation_validation_reward=prior['best_validation_reward'],
            joint_steps=record['joint_steps'],optimizer_updates=record['optimizer_updates'],
            interpretation='Single-seed development result on previously inspected validation layouts; no test or performance conclusion.')
        write_json(OUT/f'summary_seed_{seed}.json',result);write_json(status,result)
        print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        write_json(status,dict(status='failed',error=str(exc),traceback=traceback.format_exc()));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('check','run'));p.add_argument('--seed',type=int,default=123)
    args=p.parse_args();check() if args.command=='check' else run(args.seed)
