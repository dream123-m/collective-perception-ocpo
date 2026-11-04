"""Sequential local-actor inference timing on archived vehicle scenarios.

For five population sizes, fresh actor networks have the current local model's
exact layer widths and scenario-dependent input/output dimensions. These are
architecture timing controls, not policies trained on those populations. A
separate 65-CAV measurement uses the three actual trained OCPO checkpoints.
Inference only: it excludes perception construction, training and radio delay.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'results/legacy_vehicle/inference_latency'
SCENARIOS=((25,.3),(25,.5),(50,.5),(50,.7),(75,.7))
SEEDS=(42,123,456)
WARMUP=20
REPEATS=200


def wait_process(pid):
    if not pid: return
    if os.name!='nt': raise RuntimeError('Dependency wait requires Windows.')
    import ctypes
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD); k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD); k.WaitForSingleObject.restype=wintypes.DWORD
    k.CloseHandle.argtypes=(wintypes.HANDLE,)
    h=k.OpenProcess(0x00100000,False,pid)
    if h:
        try:
            if k.WaitForSingleObject(h,0xffffffff)!=0: raise OSError('Wait failed.')
        finally: k.CloseHandle(h)


def atomic(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.pending'); tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8'); tmp.replace(p)


def run(pid):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'protocol.json').exists(): raise FileExistsError('Latency evidence already exists.')
    atomic(OUTPUT/'status.json',dict(status='waiting' if pid else 'starting',wait_for_pid=pid))
    try:
        wait_process(pid)
        import numpy as np
        import torch
        from scipy.io import loadmat
        from MAPPO.mappo_network import ActorNetwork
        from . import legacy_layout_experiment as experiment
        from .data import _geometric_los,_save_mat
        from .environment import CrossSceneEnv
        from .run import now,sha,write_csv,write_json
        torch.set_num_threads(1)
        previous=HERE/'results/legacy_vehicle/target_sweep/status.json'
        previous_status=json.loads(previous.read_text(encoding='utf8'))['status'] if previous.exists() else 'unknown'
        model_freeze=json.loads((experiment.OUTPUT/'final_test/freeze.json').read_text(encoding='utf8'))
        model_by_seed={x['seed']:x for x in model_freeze['models'] if x['method']=='ocpo'}
        for seed in SEEDS:
            m=model_by_seed[seed]
            if sha(experiment.OUTPUT/m['checkpoint'])!=m['checkpoint_sha256']:
                raise ValueError('Trained actor checkpoint changed.')
        source=[]
        for density,penetration in SCENARIOS:
            path=ROOT/f'MAPPO/vehicles_density{density}_penetration{penetration:.1f}.mat'
            raw=loadmat(path); source.append(dict(density=density,penetration=penetration,
                file=path.relative_to(ROOT).as_posix(),sha256=sha(path),
                vehicles=len(raw['positions']),cavs=int(np.asarray(raw['is_cav']).sum())))
        protocol=dict(code_sha256=sha(Path(__file__)),trained_freeze_sha256=sha(experiment.OUTPUT/'final_test/freeze.json'),
            scenarios=source,seeds=list(SEEDS),warmup=WARMUP,repeats=REPEATS,device='cpu',threads=1,
            processor=platform.processor(),platform=platform.platform(),torch=torch.__version__,
            population_weights='Freshly initialized actor weights for all five scenario sizes; seed labels initialization and timing repeats, not training.',
            trained_reference='Separate reference on the source 122-vehicle/65-CAV scene with three trained OCPO actor checkpoints.',
            network='Current local actor: two Tanh layers (64,64), sigmoid outputs; input/output width follows each scenario.',
            operation='Sequential local forwards, observation conversion, valid-action mask, 0.5 threshold and NumPy output; excludes observation construction, environment stepping, radio and training.',
            statistics='Mean over 200 timed repetitions for each seed; aggregate mean and sample SD across the three initialization or training seeds.',
            source_los='LOS is recomputed from archived positions with the current geometry code; self excluded.',
            previous_target_sweep_status=previous_status)
        write_json(OUTPUT/'protocol.json',protocol)
        write_json(OUTPUT/'status.json',dict(status='running',started_at_utc=now(),completed=[]))
        samples=[]; per_seed=[]
        with tempfile.TemporaryDirectory(prefix='latency_scene_',dir=HERE) as tmp:
            for scenario in source:
                path=ROOT/scenario['file']; raw=loadmat(path); pos=raw['positions']; labels=raw['is_cav']; n=len(pos)
                los,comm=_geometric_los(pos); target=Path(tmp)/f"scenario_{scenario['density']}_{scenario['penetration']:.1f}.mat"
                _save_mat(target,dict(positions=pos,is_cav=labels,vehicle_ids=raw['vehicle_ids'],los_matrix=los,comm_los_matrix=comm))
                env=CrossSceneEnv(target,0,dict(required_vehicle_count=n,required_cav_count=scenario['cavs']))
                obs=env.reset(seed=0); masks=env.mask; n_agents=env.num_agents
                kinds=('initialized','trained') if (scenario['density'],scenario['penetration'])==(50,.5) else ('initialized',)
                for kind in kinds:
                    for seed in SEEDS:
                        torch.manual_seed(seed)
                        actors=[ActorNetwork(env.state_dim,env.max_targets_per_cav,[64,64]).eval() for _ in range(n_agents)]
                        checkpoint_hash=None
                        if kind=='trained':
                            m=model_by_seed[seed]; ckpt=torch.load(experiment.OUTPUT/m['checkpoint'],map_location='cpu',weights_only=False)
                            from major.model import JointPolicy
                            policy=JointPolicy(env.state_dim,n_agents,env.max_targets_per_cav,[64,64],use_editor=False)
                            policy.load_state_dict(ckpt['policy'],strict=True); policy.eval(); checkpoint_hash=m['checkpoint_sha256']
                            with torch.no_grad():
                                for i,actor in enumerate(actors):
                                    for j,layer in enumerate((actor.shared_net[0],actor.shared_net[2],actor.action_head)):
                                        layer.weight.copy_(policy.weights[j][i]); layer.bias.copy_(policy.biases[j][i])
                                batch=policy.probabilities(torch.from_numpy(obs)[None])[0]
                                local=torch.cat([actor(torch.from_numpy(obs[i])[None]) for i,actor in enumerate(actors)])
                                torch.testing.assert_close(local.clamp(1e-6,1-1e-6),batch,rtol=1e-5,atol=1e-6)
                                if not torch.equal((local>.5)&torch.from_numpy(masks),(batch>.5)&torch.from_numpy(masks)):
                                    raise AssertionError('Local and batched decisions differ.')
                        timing=[]
                        with torch.inference_mode():
                            for iteration in range(WARMUP+REPEATS):
                                start=time.perf_counter_ns()
                                for i,actor in enumerate(actors):
                                    probabilities=actor(torch.from_numpy(obs[i])[None]).clamp(1e-6,1-1e-6)
                                    actions=((probabilities[0]>.5)&torch.from_numpy(masks[i])).numpy()
                                ms=(time.perf_counter_ns()-start)/1e6
                                if iteration>=WARMUP: timing.append(ms)
                        for rep,value in enumerate(timing):
                            samples.append(dict(density=scenario['density'],penetration=scenario['penetration'],
                                                kind=kind,seed=seed,iteration=rep,agents=n_agents,decision_time_ms=value))
                        per_seed.append(dict(density=scenario['density'],penetration=scenario['penetration'],kind=kind,seed=seed,
                            vehicles=n,agents=n_agents,state_dim=env.state_dim,action_dim=env.max_targets_per_cav,
                            decision_time_ms=statistics.mean(timing),per_cav_time_ms=statistics.mean(timing)/n_agents,
                            checkpoint_sha256=checkpoint_hash))
                        print(f"Latency {scenario['density']}/{scenario['penetration']} {kind} seed={seed} done.",flush=True)
        write_csv(OUTPUT/'samples.csv',samples); write_csv(OUTPUT/'per_seed.csv',per_seed)
        aggregate=[]
        for density,penetration in SCENARIOS:
            for kind in (('initialized','trained') if (density,penetration)==(50,.5) else ('initialized',)):
                group=[r for r in per_seed if r['density']==density and r['penetration']==penetration and r['kind']==kind]
                aggregate.append(dict(density=density,penetration=penetration,kind=kind,agents=group[0]['agents'],
                    vehicles=group[0]['vehicles'],state_dim=group[0]['state_dim'],action_dim=group[0]['action_dim'],
                    decision_time_ms_mean=statistics.mean(r['decision_time_ms'] for r in group),
                    decision_time_ms_sample_sd=statistics.stdev(r['decision_time_ms'] for r in group),
                    per_cav_time_ms_mean=statistics.mean(r['per_cav_time_ms'] for r in group)))
        result=dict(status='completed',finished_at_utc=now(),aggregate=aggregate,
                    protocol_sha256=sha(OUTPUT/'protocol.json'),samples_sha256=sha(OUTPUT/'samples.csv'),
                    interpretation='Initialized actor timings measure architecture scaling; only the separate 65-CAV reference uses trained weights.')
        write_json(OUTPUT/'summary.json',result); write_json(OUTPUT/'status.json',dict(status='completed',finished_at_utc=now()))
        print(json.dumps(result,indent=2),flush=True)
    except Exception as error:
        atomic(OUTPUT/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc())); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--wait-for-pid',type=int,default=0); args=p.parse_args(); run(args.wait_for_pid)
