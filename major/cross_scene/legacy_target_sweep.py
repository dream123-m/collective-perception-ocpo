"""Sequential OCPO target sweep using the same conditional-layout protocol.

New targets .80/.85/.90 are trained from scratch; .95 reuses the existing
three frozen policies. The original test layouts have already been inspected,
so this is a sensitivity follow-up, not an additional untouched test set.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import tempfile
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'results/legacy_vehicle/target_sweep'
TARGETS=(.80,.85,.90,.95)
SEEDS=(42,123,456)


def wait_for_process(pid):
    if not pid: return
    if os.name!='nt': raise RuntimeError('The dependency wait is implemented for Windows.')
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if handle:
        try:
            if kernel.WaitForSingleObject(handle,0xffffffff)!=0: raise OSError('Dependency wait failed.')
        finally: kernel.CloseHandle(handle)
    # A missing process is acceptable only if its durable completion record exists.
    status=json.loads((HERE/'results/legacy_vehicle/ablation/status.json').read_text(encoding='utf8'))
    if status['status']!='completed': raise RuntimeError('Ablation did not complete; target sweep was not started.')


def cfg_for(base,target,episodes=1500):
    c=base.config(episodes); c['target_coverage']=float(target); return c


def evaluate_cavs(base,policy,records,data,cfg):
    import numpy as np
    import torch
    from .layout_only import FixedFadingEnv
    from .run import state_hash
    before=state_hash(policy); rng=torch.get_rng_state().clone(); mode=policy.training
    layout=[]; counts=np.zeros(65,dtype=np.float64); total_steps=0
    policy.eval()
    try:
        with torch.inference_mode():
            for record in records:
                env=FixedFadingEnv(data/record['path'],0,base.env_config(cfg))
                obs=torch.from_numpy(env.reset(seed=0)); mask=torch.from_numpy(env.mask)
                metrics=[]
                for _ in range(cfg['episode_steps']):
                    acts=(policy.probabilities(obs[None])[0]>.5)&mask
                    counts+=acts.sum(-1).numpy(); total_steps+=1
                    nxt,reward,done,info=env.step(acts.numpy())
                    metrics.append((float(np.mean(reward)),float(info['coverage_rate']),float(acts.sum())/env.num_agents))
                    obs=torch.from_numpy(nxt)
                    if done: break
                average=np.asarray(metrics).mean(0)
                layout.append(dict(scene_id=record['scene_id'],scene_sha256=record['scene_sha256'],
                    channel_sha256=env.channel_hash,unit_fading_sha256=env.unit_fading_sha256,
                    reward=float(average[0]),coverage=float(average[1]),payload=float(average[2]),steps=len(metrics)))
    finally: policy.train(mode)
    if before!=state_hash(policy) or not torch.equal(rng,torch.get_rng_state()):
        raise AssertionError('Evaluation changed policy or RNG.')
    if len(layout)!=32 or total_steps!=32*cfg['episode_steps']: raise AssertionError('Missing test layouts or steps.')
    return layout,(counts/total_steps).tolist()


def check():
    import torch
    from . import legacy_layout_experiment as base
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='target_check_',dir=HERE) as tmp:
        with base.adapted():
            rec=base.runner.run_training('ocpo',42,base.DATA,Path(tmp),cfg_for(base,.80,2))
            if rec['status']!='completed' or rec['optimizer_steps']!=20: raise AssertionError('Target preflight failed.')
    print('Target-specific OCPO preflight passed.',flush=True)


def run(wait_pid):
    from datetime import datetime,timezone
    OUTPUT.mkdir(parents=True,exist_ok=True)
    def write_status(value):
        from .run import write_json
        write_json(OUTPUT/'status.json',value)
    try:
        write_status(dict(status='waiting' if wait_pid else 'starting',wait_for_pid=wait_pid,targets=list(TARGETS),seeds=list(SEEDS)))
        wait_for_process(wait_pid)
        import torch
        from . import legacy_layout_experiment as base
        from .layout_only import FixedFadingEnv
        from .run import now,sha,write_json,write_csv,acquire_batch_lock
        torch.set_num_threads(1)
        lock=acquire_batch_lock(OUTPUT)
        try:
            state=dict(status='running',started_at_utc=now(),completed=[],targets=list(TARGETS),seeds=list(SEEDS))
            write_json(OUTPUT/'status.json',state)
            bank=base.load_bank(base.DATA,verify=True)
            protocol=dict(source_sha256=sha(Path(__file__)),data_manifest_sha256=sha(base.DATA/'manifest.json'),
                reference_protocol_sha256=sha(base.OUTPUT/'protocol.json'),
                original_final_test_sha256=sha(base.OUTPUT/'final_test/summary.json'),
                targets=list(TARGETS),new_training_targets=[.80,.85,.90],seeds=list(SEEDS),
                settings='Same original-count conditional layouts, fixed fading, MAPPO backbone and OCPO update; target only changes.',
                selection='Best 16-layout validation reward per target and seed.',
                final_test='Reuse 32 previously inspected held-out layouts, exploratory sensitivity analysis.')
            write_json(OUTPUT/'protocol.json',protocol)
            for target in TARGETS[:-1]:
                output=OUTPUT/f'target_{target:.2f}'
                with base.adapted():
                    for seed in SEEDS:
                        state.update(active_target=target,active_seed=seed,phase='training'); write_json(OUTPUT/'status.json',state)
                        base.runner.run_training('ocpo',seed,base.DATA,output,cfg_for(base,target))
                        state['completed'].append(dict(target=target,seed=seed)); write_json(OUTPUT/'status.json',state)
            state.update(phase='freeze_and_exploratory_test'); write_json(OUTPUT/'status.json',state)
            models=[]
            for target in TARGETS:
                folder=base.OUTPUT if target==.95 else OUTPUT/f'target_{target:.2f}'
                for seed in SEEDS:
                    run_dir=folder/f'ocpo_seed_{seed}'; rec=json.loads((run_dir/'run.json').read_text(encoding='utf8'))
                    best=run_dir/'best.pt'; reference=json.loads((base.OUTPUT/f'ocpo_seed_{seed}'/'run.json').read_text(encoding='utf8'))
                    if (rec['status']!='completed' or rec['completed_episodes']!=1500 or rec['config']!=cfg_for(base,target)
                        or rec['initial_actor_sha256']!=reference['initial_actor_sha256'] or sha(best)!=rec['selected_checkpoint_sha256']):
                        raise ValueError('Incomplete or unmatched target model.')
                    models.append(dict(target=target,seed=seed,checkpoint=str(best),
                                       checkpoint_sha256=sha(best),selected_episode=rec['selected_episode']))
            test=OUTPUT/'exploratory_test'; test.mkdir(exist_ok=True)
            if (test/'started.json').exists(): raise RuntimeError('Target test already started.')
            write_json(test/'freeze.json',dict(models=models,protocol_sha256=sha(OUTPUT/'protocol.json'),created_at_utc=now()))
            write_json(test/'started.json',dict(started_at_utc=now(),freeze_sha256=sha(test/'freeze.json')))
            layout_rows=[]; cav_rows=[]; per_seed=[]
            with base.adapted():
                template=FixedFadingEnv(base.DATA/bank['splits']['train'][0]['path'],0,base.env_config(base.config()))
                for model in models:
                    target,seed=model['target'],model['seed']; c=cfg_for(base,target)
                    policy=base.runner.make_policy(template,c)
                    saved=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
                    policy.load_state_dict(saved['policy'],strict=True)
                    rows,cavs=evaluate_cavs(base,policy,bank['splits']['test'],base.DATA,c)
                    if target==.95:
                        path=base.OUTPUT/'final_test'/f'ocpo_seed_{seed}.csv'
                        with path.open(encoding='utf-8-sig',newline='') as f: expected=list(csv.DictReader(f))
                        for a,b in zip(rows,expected):
                            if a['scene_id']!=b['scene_id'] or a['channel_sha256']!=b['channel_sha256']:
                                raise AssertionError('Target .95 scene does not match existing test.')
                            for key in ('reward','coverage','payload'):
                                if abs(a[key]-float(b[key]))>1e-7: raise AssertionError('Target .95 does not reproduce saved test.')
                    layout_rows.extend(dict(target=target,seed=seed,**r) for r in rows)
                    cav_rows.extend(dict(target=target,seed=seed,cav_role=i,payload=x) for i,x in enumerate(cavs))
                    per_seed.append(dict(target=target,seed=seed,selected_episode=model['selected_episode'],
                        **{key:statistics.mean(r[key] for r in rows) for key in ('reward','coverage','payload')}))
            write_csv(test/'per_layout.csv',layout_rows); write_csv(test/'per_cav.csv',cav_rows); write_csv(test/'per_seed.csv',per_seed)
            aggregate=[]
            for target in TARGETS:
                subset=[r for r in per_seed if r['target']==target]; row=dict(target=target)
                for key in ('reward','coverage','payload'):
                    row[key+'_mean']=statistics.mean(r[key] for r in subset)
                    row[key+'_sample_sd']=statistics.stdev(r[key] for r in subset)
                aggregate.append(row)
            write_json(test/'summary.json',dict(status='completed',finished_at_utc=now(),aggregate=aggregate,
                       per_seed=per_seed,target095_reproduced=True,interpretation=protocol['final_test']))
            state.update(status='completed',finished_at_utc=now(),aggregate=aggregate); write_json(OUTPUT/'status.json',state)
            print('Completed target sweep and paired layout evaluation.',flush=True)
        finally: lock.close()
    except Exception as error:
        # Failure is recorded even if the dependency or initialization failed.
        try: write_status(dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        except Exception: pass
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('check','run')); p.add_argument('--wait-for-pid',type=int,default=0)
    args=p.parse_args(); check() if args.mode=='check' else run(args.wait_for_pid)
