"""Three matched OCPO component controls on the conditional original layouts.

Controls are trained from scratch with exactly the main experiment's seed,
data, common actor/critic, optimizer, single-window GAE and checkpoint rule.
The already inspected held-out split is reused for component comparison, so
its result is an exploratory follow-up, not a new untouched confirmation set.
"""
from __future__ import annotations
import copy
import json
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import statistics
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.distributions import Bernoulli

from . import legacy_layout_experiment as main
from . import legacy_update_layout as legacy
from . import ablation_layout_only as controls
from .layout_only import FixedFadingEnv
from .multi_layout_update import balanced_minibatches, padded_factor_logp
from .run import ROOT,HERE,now,sha,state_hash,write_csv,write_json,acquire_batch_lock,rollout_probabilities,groups_for

OUTPUT=HERE/'results/legacy_vehicle/ablation'
SELF=Path(__file__).resolve()
SEEDS=(42,123,456)
METHODS=('team_vehicle','individual_pair','target_shuffle')


def protocol(data,output,config):
    data,output=Path(data).resolve(),Path(output).resolve()
    if not data.is_relative_to(HERE) or not output.is_relative_to(HERE): raise ValueError('Path outside workspace.')
    legacy.check_config(config)
    bank=main.load_bank(data,verify=True)
    reference=json.loads((main.OUTPUT/'protocol.json').read_text(encoding='utf8'))
    if (sha(data/'manifest.json')!=reference['data_manifest_sha256'] or
        any(config[k]!=v for k,v in reference['config'].items() if k!='episodes')):
        raise ValueError('Ablation data/config differ from the original comparison.')
    code=copy.deepcopy(reference['code_sha256'])
    code[SELF.relative_to(ROOT).as_posix()]=sha(SELF)
    code[Path(controls.__file__).resolve().relative_to(ROOT).as_posix()]=sha(controls.__file__)
    entry=dict(variant='legacy_layout_component_ablation',config=config,
        data_manifest_sha256=sha(data/'manifest.json'),reference_protocol_sha256=sha(main.OUTPUT/'protocol.json'),
        code_sha256=code,methods=list(METHODS),seeds=list(SEEDS),
        controls={
          'team_vehicle':'Team reward/advantage with original per-vehicle clipping.',
          'individual_pair':'Per-agent advantages with same-object pair clipping and endpoint-owned gradient.',
          'target_shuffle':'Team advantage and same sender-pair structure with independently permuted local target slots.'},
        common='Same physical layouts, fixed fading, actor/critic architecture, one-layout rollout, sample-SD normalization, optimizers and budget as the main comparison.',
        test='Reuses the same already-inspected 32 held-out layouts; component comparison is exploratory.',
        created_at_utc=now())
    output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf8'))
        for k in ('variant','config','data_manifest_sha256','reference_protocol_sha256','code_sha256','methods','seeds'):
            if previous[k]!=entry[k]: raise ValueError(f'Incompatible ablation protocol: {k}')
        return bank,previous
    allowed={'batch.lock','stage.stdout.log','stage.stderr.log','status.json'}
    if any(p.name not in allowed for p in output.iterdir()): raise FileExistsError('Refusing to overwrite output.')
    write_json(path,entry)
    for name in code:
        dst=output/'source'/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dst)
    return bank,entry


def pack(rollouts,method,config):
    if len(rollouts)!=1: raise ValueError('Expected one layout per update.')
    batch=controls_original_pack(rollouts,method,config)
    x=batch['raw_advantage']
    if len(x)>1: batch['advantage']=(x-x.mean(0))/(x.std(0)+1e-8)
    return batch


controls_original_pack=controls.pack_rollouts


def optimize(policy,critic,actor_optimizer,critic_optimizer,batch,method,lengths,rng,config):
    mask=batch['mask']
    with torch.no_grad():
        replay=rollout_probabilities(policy,batch['obs'])
        bits=Bernoulli(replay.double()).log_prob(batch['action'].double())*mask
        torch.testing.assert_close(bits,batch['old_bits'],atol=1e-10,rtol=1e-10)
        old=padded_factor_logp(bits,batch['groups']) if method=='individual_pair' else bits.sum(-1)
        torch.testing.assert_close(old,batch['old'],atol=1e-10,rtol=1e-10)
    actor_total=critic_total=weight=steps=0
    for _ in range(config['epochs']):
        for ix in balanced_minibatches(lengths,config['minibatch_joint_steps'],rng):
            probs=rollout_probabilities(policy,batch['obs'][ix])
            dist=Bernoulli(probs.double()); local_mask=mask[ix]
            bits=dist.log_prob(batch['action'][ix].double())*local_mask
            groups=batch['groups'][ix] if method=='individual_pair' else None
            actor_loss=controls.update_loss(bits,batch['old_bits'][ix],groups,batch['advantage'][ix],config['clip'],method)
            entropy=(dist.entropy()*local_mask).sum(-1)/local_mask.sum(-1).clamp_min(1)
            actor_loss-=config['entropy']*entropy.mean(0).sum()
            actor_optimizer.zero_grad(); actor_loss.backward(); policy.clip_gradients(config['max_grad_norm']); actor_optimizer.step()
            value=critic(batch['broadcast'][ix],batch['coverage'][ix]).flatten()
            critic_loss=.5*(value-batch['returns'][ix]).square().mean()
            critic_optimizer.zero_grad(); critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),config['max_grad_norm']); critic_optimizer.step()
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss): raise FloatingPointError('Nonfinite control loss.')
            n=len(ix); actor_total+=float(actor_loss.detach())*n; critic_total+=float(critic_loss.detach())*n
            weight+=n; steps+=1
    return dict(actor_loss=actor_total/weight/policy.num_agents,critic_loss=critic_total/weight,optimizer_steps=steps)


def shuffled_groups(seed):
    shuffle_rng=np.random.default_rng(seed+950000)
    def groups(env,matching_rng):
        original=groups_for(env,matching_rng)
        mapping=np.arange(env.num_agents*env.max_targets_per_cav,dtype=np.int64)
        for i in range(env.num_agents):
            ids=i*env.max_targets_per_cav+np.flatnonzero(env.mask[i])
            mapping[ids]=shuffle_rng.permutation(ids)
        result=original.clone(); valid=result>=0
        result[valid]=torch.from_numpy(mapping[result[valid].numpy()])
        if not torch.equal((original//env.max_targets_per_cav)[valid],(result//env.max_targets_per_cav)[valid]):
            raise AssertionError('Sender pair structure changed.')
        legal=np.flatnonzero(env.mask.ravel())
        if not np.array_equal(np.sort(result[valid].numpy()),legal): raise AssertionError('Target shuffle lost a decision.')
        return result
    return groups


@contextmanager
def control_runner():
    with main.adapted():
        names=('pack_rollouts','optimize','prepare_protocol','ScaledValueCritic','environment_config','evaluate_frozen')
        old={name:getattr(controls,name) for name in names}
        controls.pack_rollouts=pack; controls.optimize=optimize; controls.prepare_protocol=protocol
        controls.ScaledValueCritic=legacy.LegacyCritic; controls.environment_config=main.env_config
        controls.evaluate_frozen=main.runner.evaluate_frozen
        try: yield
        finally:
            for name,value in old.items(): setattr(controls,name,value)


@contextmanager
def shuffle_runner(seed):
    with main.adapted():
        old=(main.runner.prepare_protocol,main.runner.groups_for)
        main.runner.prepare_protocol=protocol; main.runner.groups_for=shuffled_groups(seed)
        try: yield
        finally: main.runner.prepare_protocol,main.runner.groups_for=old


def check():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='ablation_check_',dir=HERE) as name:
        tmp=Path(name); actors=[]
        with control_runner():
            for method in ('team_vehicle','individual_pair'):
                result=controls.run_training(method,42,main.DATA,tmp/'components',main.config(2))
                if result['completed_episodes']!=2 or result['optimizer_steps']!=20: raise AssertionError('Control check failed.')
                actors.append(result['initial_actor_sha256'])
        with shuffle_runner(42):
            result=main.runner.run_training('ocpo',42,main.DATA,tmp/'shuffled',main.config(2))
            actors.append(result['initial_actor_sha256'])
            if result['completed_episodes']!=2 or result['optimizer_steps']!=20: raise AssertionError('Shuffle check failed.')
        if len(set(actors))!=1: raise AssertionError('Controls have different initial actor models.')
    print('All three matched controls passed preflight.',flush=True)


def finish_test():
    bank=main.load_bank(main.DATA); models=[]
    for seed in SEEDS:
        reference=json.loads((main.OUTPUT/f'mappo_seed_{seed}'/'run.json').read_text(encoding='utf8'))
        for name in METHODS:
            folder=(OUTPUT/f'{name}_seed_{seed}' if name!='target_shuffle' else OUTPUT/'shuffled'/f'ocpo_seed_{seed}')
            record=json.loads((folder/'run.json').read_text(encoding='utf8')); best=folder/'best.pt'
            if (record['status']!='completed' or record['completed_episodes']!=1500 or
                record['initial_actor_sha256']!=reference['initial_actor_sha256'] or
                sha(best)!=record['selected_checkpoint_sha256']): raise ValueError('Invalid control checkpoint.')
            models.append(dict(method=name,seed=seed,checkpoint=str(best),sha256=sha(best),episode=record['selected_episode']))
    out=OUTPUT/'exploratory_test'; out.mkdir(exist_ok=True)
    if (out/'started.json').exists(): raise RuntimeError('Ablation test already started.')
    write_json(out/'freeze.json',dict(models=models,source_sha256=sha(SELF),data_manifest_sha256=sha(main.DATA/'manifest.json'),
               reference=sha(main.OUTPUT/'final_test/summary.json'),created_at_utc=now()))
    write_json(out/'started.json',dict(started_at_utc=now(),freeze_sha256=sha(out/'freeze.json')))
    records=[]; cfg=main.config()
    with main.adapted():
        template=FixedFadingEnv(main.DATA/bank['splits']['train'][0]['path'],0,main.env_config(cfg))
        for model in models:
            policy=main.runner.make_policy(template,cfg)
            ckpt=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
            policy.load_state_dict(ckpt['policy'],strict=True)
            rows=main.runner.evaluate_frozen(policy,bank['splits']['test'],main.DATA,cfg,namespace='component_control',channels=1)
            if len(rows)!=32: raise ValueError('Expected 32 layouts.')
            reference_csv=main.OUTPUT/'final_test'/f"mappo_seed_{model['seed']}.csv"
            import csv
            with reference_csv.open(encoding='utf-8-sig',newline='') as f: baseline=list(csv.DictReader(f))
            if [(r['scene_id'],r['channel_sha256']) for r in rows] != [(r['scene_id'],r['channel_sha256']) for r in baseline]:
                raise AssertionError('Test layouts or fading mismatch.')
            fname=f"{model['method']}_seed_{model['seed']}.csv"; write_csv(out/fname,rows)
            records.append({**model,**main.runner.summary(rows),'layouts':len(rows),
                            'policy_sha256':state_hash(policy),'csv_sha256':sha(out/fname)})
    aggregates={}
    for method in METHODS:
        subset=[r for r in records if r['method']==method]
        aggregates[method]={k:dict(mean=statistics.mean(r[k] for r in subset),sample_sd=statistics.stdev(r[k] for r in subset))
                            for k in ('reward','coverage','payload')}
    result=dict(status='completed',finished_at_utc=now(),records=records,aggregate=aggregates,
                interpretation='Exploratory component comparison on the previously inspected 32 layouts.')
    write_json(out/'summary.json',result)
    return aggregates


def run():
    torch.set_num_threads(1); OUTPUT.mkdir(parents=True,exist_ok=True); lock=acquire_batch_lock(OUTPUT)
    state=dict(status='running',seeds=list(SEEDS),methods=list(METHODS),completed=[],started_at_utc=now())
    try:
        write_json(OUTPUT/'status.json',state)
        for seed in SEEDS:
            for method in METHODS:
                state.update(active_method=method,active_seed=seed); write_json(OUTPUT/'status.json',state)
                if method=='target_shuffle':
                    with shuffle_runner(seed): main.runner.run_training('ocpo',seed,main.DATA,OUTPUT/'shuffled',main.config())
                else:
                    with control_runner(): controls.run_training(method,seed,main.DATA,OUTPUT,main.config())
                state['completed'].append(dict(method=method,seed=seed)); write_json(OUTPUT/'status.json',state)
        state.update(phase='exploratory_test'); write_json(OUTPUT/'status.json',state)
        state.update(status='completed',finished_at_utc=now(),aggregate=finish_test())
        write_json(OUTPUT/'status.json',state)
    except Exception as exc:
        state.update(status='failed',error=str(exc),traceback=traceback.format_exc()); write_json(OUTPUT/'status.json',state); raise
    finally: lock.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('check','run'))
    args=parser.parse_args(); check() if args.mode=='check' else run()
