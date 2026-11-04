"""Traffic scenario generation and experiment configuration."""








from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import traceback
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=HERE/'results/legacy_vehicle/scenario_grid'
DATA=BASE/'data'
TRAIN=BASE/'training'
MASTER=20261030
COUNTS={'train':64,'validation':16,'test':32}
SCENARIOS=tuple((d,p) for d in (25,50,75) for p in (.3,.5,.7))
TARGETS=(.90,.95)
SEEDS=(42,123,456)


def atomic(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.pending'); t.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8'); t.replace(p)


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
            if k.WaitForSingleObject(h,0xffffffff)!=0: raise OSError('Dependency wait failed.')
        finally: k.CloseHandle(h)


def key(density,penetration): return f'density{density}_penetration{penetration:.1f}'


def source_info(density,penetration):
    import numpy as np
    from scipy.io import loadmat
    from .run import sha
    path=ROOT/f'MAPPO/vehicles_density{density}_penetration{penetration:.1f}.mat'
    raw=loadmat(path); pos=np.asarray(raw['positions'],dtype=np.float64)
    ids=np.asarray(raw['vehicle_ids']).reshape(-1); labels=np.asarray(raw['is_cav']).reshape(-1).astype(np.uint8)
    if not np.array_equal(ids,np.arange(1,len(pos)+1)): raise ValueError('Source IDs not canonical.')
    lanes=np.unique(pos[:,1]); sizes=[]; roles=[]
    for y in lanes:
        ix=np.flatnonzero(pos[:,1]==y); ordered=ix[np.argsort(pos[ix,0])]
        if not np.array_equal(ix,ordered): raise ValueError('Source rows are not in lane/rank order.')
        sizes.append(len(ix)); roles.extend(labels[ix].tolist())
    if len(lanes)!=4 or sum(roles)==0: raise ValueError('Unexpected source topology.')
    return dict(file=str(path.relative_to(ROOT)),sha256=sha(path),vehicles=len(pos),cavs=int(sum(roles)),
                lanes=list(map(float,lanes)),lane_counts=sizes,roles=roles)


def scene_hash(positions,roles):
    import numpy as np
    h=hashlib.sha256(); h.update(np.asarray(positions,dtype='<f8').tobytes()); h.update(np.asarray(roles,dtype=np.uint8).tobytes())
    return h.hexdigest()


def draw_lane(rng,intensity,count):
    import numpy as np
    for _ in range(10000):
        candidates=np.sort(rng.uniform(0,1000,rng.poisson(intensity))); kept=[]
        for x in candidates:
            if not kept or x-kept[-1]>=10.2: kept.append(float(x))
        if len(kept)==count: return np.asarray(kept)
    raise RuntimeError(f'Failed conditional lane draw: intensity={intensity}, count={count}')


def load_bank(data,verify=True):
    import numpy as np
    from scipy.io import loadmat
    from .run import sha
    data=Path(data).resolve(); m=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    source=source_info(m['density'],m['penetration'])
    if m['source']['sha256']!=source['sha256'] or m['split_counts']!=COUNTS: raise ValueError('Scenario source changed.')
    seen=set()
    for split,total in COUNTS.items():
        if len(m['splits'][split])!=total: raise ValueError('Split count changed.')
        for r in m['splits'][split]:
            path=(data/r['path']).resolve()
            if not path.is_relative_to(data) or sha(path)!=r['file_sha256']: raise ValueError('Scenario MAT hash mismatch.')
            if verify:
                x=loadmat(path); pos=np.asarray(x['positions']); labels=np.asarray(x['is_cav']).reshape(-1)
                if len(pos)!=source['vehicles'] or not np.array_equal(labels,source['roles']): raise ValueError('Role mismatch.')
                offset=0
                for y,n in zip(source['lanes'],source['lane_counts']):
                    z=pos[offset:offset+n]; offset+=n
                    if not np.all(z[:,1]==y) or np.any(np.diff(z[:,0])<10.2-1e-9): raise ValueError('Invalid lane geometry.')
                h=scene_hash(pos,labels)
                if h in seen or h!=r['scene_sha256']: raise ValueError('Duplicate or changed scene.')
                seen.add(h)
    return dict(splits=m['splits'],split_counts=COUNTS,source=source)


def generate():
    import numpy as np
    from .data import _geometric_los,_save_mat
    from .run import sha,write_json
    for density,penetration in SCENARIOS:
        if (density,penetration)==(50,.5): continue
        data=DATA/key(density,penetration)
        if (data/'manifest.json').exists(): load_bank(data); continue
        if data.exists() and any(data.iterdir()): raise FileExistsError(f'Incomplete dataset: {data}')
        data.mkdir(parents=True,exist_ok=True); source=source_info(density,penetration)
        manifest=dict(density=density,penetration=penetration,source=source,split_counts=COUNTS,
            generator_source_sha256=sha(Path(__file__)),master_seed=MASTER,
            generation='Independent Poisson candidates, sequential 10.2 m thinning, conditional on archived per-lane counts; archived CAV lane/rank roles fixed.',
            splits={name:[] for name in COUNTS})
        for split_index,(split,total) in enumerate(COUNTS.items(),1):
            folder=data/split; folder.mkdir()
            for index in range(total):
                positions=[]
                for lane,(y,n) in enumerate(zip(source['lanes'],source['lane_counts'])):
                    rng=np.random.default_rng(np.random.SeedSequence([MASTER,density,int(penetration*10),split_index,index,lane]))
                    x=draw_lane(rng,density,n); positions.append(np.column_stack((x,np.full(n,y))))
                pos=np.concatenate(positions); los,comm=_geometric_los(pos)
                path=folder/f'layout_{index:03d}.mat'
                _save_mat(path,dict(positions=pos,is_cav=np.asarray(source['roles'])[:,None],
                    vehicle_ids=np.arange(1,source['vehicles']+1)[:,None],los_matrix=los,comm_los_matrix=comm))
                manifest['splits'][split].append(dict(scene_id=f'{split}_{index:03d}',path=path.relative_to(data).as_posix(),
                    file_sha256=sha(path),scene_sha256=scene_hash(pos,source['roles'])))
        write_json(data/'manifest.json',manifest); load_bank(data)
        print(f'Verified scenario {key(density,penetration)}: {sum(COUNTS.values())} layouts.',flush=True)


def config_for(main,source,target,episodes=1500):
    c=main.config(episodes); steps=max(1,512//source['cavs'])
    c.update(target_coverage=float(target),episode_steps=steps,minibatch_joint_steps=steps)
    return c


def context(data,source):
    from . import legacy_layout_experiment as main
    from .run import environment_config,sha,write_json
    import shutil
    @contextmanager
    def managed():
        with main.adapted():
            old=(main.runner.load_manifest,main.runner.environment_config,main.runner.prepare_protocol)
            def envcfg(c): return {**environment_config(c),'required_vehicle_count':source['vehicles'],'required_cav_count':source['cavs']}
            def prepare(d,out,c):
                d,out=Path(d).resolve(),Path(out).resolve()
                if not d.is_relative_to(HERE) or not out.is_relative_to(HERE): raise ValueError('Path outside workspace.')
                main.legacy.check_config(c); bank=load_bank(d)
                code=main.runner.source_hashes()
                for p in (Path(__file__).resolve(),Path(main.__file__).resolve(),Path(main.legacy.__file__).resolve()):
                    code[p.relative_to(ROOT).as_posix()]=sha(p)
                item=dict(variant='conditional_scenario_grid',config=c,data_manifest_sha256=sha(d/'manifest.json'),
                          source_mat_sha256=source['sha256'],code_sha256=code,methods=['mappo','ocpo'],seeds=list(SEEDS),
                          source_snapshot='scenario_grid/source (shared across scenarios)',
                          description='One-layout MAPPO/OCPO update on disjoint conditional vehicle layouts; fixed fading.',created_at_utc=main.now())
                out.mkdir(parents=True,exist_ok=True); path=out/'protocol.json'
                if path.exists():
                    oldp=json.loads(path.read_text(encoding='utf8'))
                    for k in ('variant','config','data_manifest_sha256','source_mat_sha256','code_sha256','methods','seeds'):
                        if oldp[k]!=item[k]: raise ValueError(f'Incompatible grid run: {k}')
                    return bank,oldp
                if any(p.name not in ('batch.lock','status.json','stage.stdout.log','stage.stderr.log') for p in out.iterdir()):
                    raise FileExistsError('Nonempty grid output.')
                write_json(path,item)
                for name in code:
                    dst=BASE/'source'/name; dst.parent.mkdir(parents=True,exist_ok=True)
                    if dst.exists():
                        if sha(dst)!=code[name]: raise ValueError(f'Shared source snapshot changed: {name}')
                    else: shutil.copy2(ROOT/name,dst)
                return bank,item
            main.runner.load_manifest=load_bank; main.runner.environment_config=envcfg; main.runner.prepare_protocol=prepare
            try: yield envcfg
            finally: main.runner.load_manifest,main.runner.environment_config,main.runner.prepare_protocol=old
    return managed()


def check():
    import torch
    from . import legacy_layout_experiment as main
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='grid_check_',dir=HERE) as tmp:
        for density,penetration in ((25,.3),(75,.7)):
            data=DATA/key(density,penetration); source=source_info(density,penetration)
            c=config_for(main,source,.95,2); init=[]
            with context(data,source):
                for method in ('mappo','ocpo'):
                    r=main.runner.run_training(method,42,data,Path(tmp)/key(density,penetration),c)
                    if r['status']!='completed' or r['optimizer_steps']!=20 or r['agents']!=source['cavs']:
                        raise AssertionError('Grid preflight failed.')
                    init.append(r['initial_actor_sha256'])
            if len(set(init))!=1: raise AssertionError('Grid method initialization mismatch.')
    print('Sparse/dense MAPPO/OCPO preflight passed.',flush=True)


def train_one(main,density,penetration,target,method,seed):
    from .run import sha,write_json,now
    is_main=(density,penetration)==(50,.5)
    if is_main and target==.95:
        return main.OUTPUT/f'{method}_seed_{seed}'
    if is_main and target==.90 and method=='ocpo':
        return HERE/f'results/legacy_vehicle/target_sweep/target_0.90/ocpo_seed_{seed}'
    if is_main:
        data=main.DATA; source=source_info(density,penetration); c=main.config(); c['target_coverage']=target
        out=TRAIN/key(density,penetration)/f'target_{target:.2f}'
        manager=main.adapted()
    else:
        data=DATA/key(density,penetration); source=source_info(density,penetration)
        out=TRAIN/key(density,penetration)/f'target_{target:.2f}'
        c=config_for(main,source,target); manager=context(data,source)
    folder=out/f'{method}_seed_{seed}'; record_path=folder/'run.json'; checkpoint=folder/'checkpoint.pt'
    resume=False
    if record_path.exists():
        record=json.loads(record_path.read_text(encoding='utf8'))
        if (record['method']!=method or record['seed']!=seed or record['config']!=c or
            record['data_manifest_sha256']!=sha(data/'manifest.json')):
            raise ValueError('Existing grid run has a different identity or data.')
        if record['status']=='completed':
            if sha(folder/'best.pt')!=record['selected_checkpoint_sha256']:
                raise ValueError('Completed grid checkpoint was changed.')
            return folder
        if not checkpoint.exists():
            raise RuntimeError(f'Interrupted run lacks a checkpoint: {folder}')
        
        
        
        import torch
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if int(saved['episode'])==c['episodes']:
            restored=saved['record']
            if (restored['config']!=c or restored['completed_episodes']!=c['episodes'] or
                restored['method']!=method or restored['seed']!=seed or not (folder/'best.pt').exists()):
                raise ValueError('Final checkpoint metadata cannot be restored safely.')
            restored.update(status='completed',finished_at_utc=now(),
                            selected_checkpoint_sha256=sha(folder/'best.pt'))
            write_json(record_path,restored)
            return folder
        resume=True
    with manager: main.runner.run_training(method,seed,data,out,c,resume=resume)
    return folder


def run(wait_pid):
    BASE.mkdir(parents=True,exist_ok=True)
    atomic(BASE/'status.json',dict(status='waiting' if wait_pid else 'starting',wait_for_pid=wait_pid))
    try:
        wait_process(wait_pid)
        import numpy as np
        import torch
        from . import legacy_layout_experiment as main
        from .layout_only import FixedFadingEnv
        from .run import sha,now,write_json,write_csv,acquire_batch_lock,environment_config
        torch.set_num_threads(1)
        for dep in ('ablation','target_sweep'):
            status=json.loads((HERE/f'results/legacy_vehicle/{dep}/status.json').read_text(encoding='utf8'))
            if status['status']!='completed': raise RuntimeError(f'Dependency {dep} did not complete.')
        lock=acquire_batch_lock(BASE)
        try:
            state=dict(status='running',phase='training',completed=[],started_at_utc=now()); atomic(BASE/'status.json',state)
            models=[]
            
            ordered=[(50,.5),(50,.3),(50,.7),(25,.5),(75,.5),(25,.3),(25,.7),(75,.3),(75,.7)]
            for density,penetration in ordered:
                for target in (.95,.90):
                    for seed in SEEDS:
                        for method in ('mappo','ocpo'):
                            state.update(active_scenario=key(density,penetration),active_target=target,active_seed=seed,active_method=method)
                            atomic(BASE/'status.json',state)
                            folder=train_one(main,density,penetration,target,method,seed)
                            record=json.loads((folder/'run.json').read_text(encoding='utf8')); ckpt=folder/'best.pt'
                            if record['status']!='completed' or record['completed_episodes']!=1500 or sha(ckpt)!=record['selected_checkpoint_sha256']:
                                raise ValueError('Invalid grid checkpoint.')
                            models.append(dict(density=density,penetration=penetration,target=target,seed=seed,
                                method=method,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),selected_episode=record['selected_episode']))
                            state['completed'].append(dict(density=density,penetration=penetration,target=target,seed=seed,method=method))
                            atomic(BASE/'status.json',state)
            
            
            for density,penetration in ordered:
                for target in TARGETS:
                    for seed in SEEDS:
                        pair=[x for x in models if (x['density'],x['penetration'],x['target'],x['seed'])==(density,penetration,target,seed)]
                        if len(pair)!=2: raise AssertionError('Missing matched method.')
                        metadata=[]; sequences=[]
                        for item in pair:
                            folder=Path(item['checkpoint']).parent
                            metadata.append(json.loads((folder/'run.json').read_text(encoding='utf8')))
                            with (folder/'training.csv').open(encoding='utf-8-sig',newline='') as f:
                                sequences.append([(r['scene_id'],r['channel_sha256']) for r in csv.DictReader(f)])
                        for k in ('initial_actor_sha256','config','data_manifest_sha256'):
                            if metadata[0][k]!=metadata[1][k]: raise AssertionError(f'Matched methods differ: {k}')
                        if len(sequences[0])!=1500 or sequences[0]!=sequences[1]:
                            raise AssertionError('Training geometry sequence differs across methods.')
            state['phase']='exploratory_test'; atomic(BASE/'status.json',state)
            out=BASE/'exploratory_test'; out.mkdir(exist_ok=True)
            if (out/'started.json').exists(): raise RuntimeError('Grid test already started.')
            freeze=dict(models=models,code_sha256=sha(Path(__file__)),created_at_utc=now(),
                interpretation='Full retained grid; original 50/.5 test layouts were already inspected; exploratory comparison.')
            write_json(out/'freeze.json',freeze); write_json(out/'started.json',dict(started_at_utc=now(),freeze_sha256=sha(out/'freeze.json')))
            rows=[]; naive=[]
            for density,penetration in ordered:
                primary=(density,penetration)==(50,.5); source=source_info(density,penetration)
                data=main.DATA if primary else DATA/key(density,penetration)
                bank=main.load_bank(data) if primary else load_bank(data)
                for target in TARGETS:
                    cfg=config_for(main,source,target)
                    mgr=main.adapted() if primary else context(data,source)
                    with mgr:
                        if primary: envcfg=main.env_config
                        else: envcfg=lambda c:{**environment_config(c),'required_vehicle_count':source['vehicles'],'required_cav_count':source['cavs']}
                        template=FixedFadingEnv(data/bank['splits']['train'][0]['path'],0,envcfg(cfg))
                        for model in [x for x in models if (x['density'],x['penetration'],x['target'])==(density,penetration,target)]:
                            policy=main.runner.make_policy(template,cfg)
                            policy.load_state_dict(torch.load(model['checkpoint'],map_location='cpu',weights_only=False)['policy'],strict=True)
                            records=main.runner.evaluate_frozen(policy,bank['splits']['test'],data,cfg,namespace='grid',channels=1)
                            if len(records)!=32: raise AssertionError('Expected 32 grid layouts.')
                            rows.extend(dict(density=density,penetration=penetration,target=target,seed=model['seed'],method=model['method'],**r) for r in records)
                        for r in bank['splits']['test']:
                            env=FixedFadingEnv(data/r['path'],0,envcfg(cfg)); env.reset(seed=0); samples=[]
                            for _ in range(cfg['episode_steps']):
                                _,reward,_,info=env.step(env.mask); samples.append((float(np.mean(reward)),float(info['coverage_rate']),float(env.mask.sum())/env.num_agents))
                            a=np.asarray(samples).mean(0)
                            naive.append(dict(density=density,penetration=penetration,target=target,scene_id=r['scene_id'],
                                scene_sha256=r['scene_sha256'],channel_sha256=env.channel_hash,reward=float(a[0]),coverage=float(a[1]),payload=float(a[2])))
            for density,penetration in ordered:
                for target in TARGETS:
                    expected=[r for r in naive if (r['density'],r['penetration'],r['target'])==(density,penetration,target)]
                    conditions=[(r['scene_id'],r['scene_sha256'],r['channel_sha256']) for r in expected]
                    if len(conditions)!=32: raise AssertionError('Naive grid evaluation incomplete.')
                    for seed in SEEDS:
                        for method in ('mappo','ocpo'):
                            actual=[r for r in rows if (r['density'],r['penetration'],r['target'],r['seed'],r['method'])==(density,penetration,target,seed,method)]
                            if [(r['scene_id'],r['scene_sha256'],r['channel_sha256']) for r in actual]!=conditions:
                                raise AssertionError('Grid test layouts or fading differ across methods.')
                            if (density,penetration,target)==(50,.5,.95):
                                oldpath=main.OUTPUT/'final_test'/f'{method}_seed_{seed}.csv'
                                with oldpath.open(encoding='utf-8-sig',newline='') as f: baseline=list(csv.DictReader(f))
                                for a,b in zip(actual,baseline):
                                    if a['scene_id']!=b['scene_id'] or any(abs(a[k]-float(b[k]))>1e-7 for k in ('reward','coverage','payload')):
                                        raise AssertionError('Primary grid did not reproduce the saved test.')
            write_csv(out/'per_layout.csv',rows); write_csv(out/'naive_per_layout.csv',naive)
            aggregate=[]
            for density,penetration in ordered:
                for target in TARGETS:
                    for method in ('mappo','ocpo'):
                        subset=[r for r in rows if (r['density'],r['penetration'],r['target'],r['method'])==(density,penetration,target,method)]
                        item=dict(density=density,penetration=penetration,target=target,method=method)
                        for metric in ('reward','coverage','payload'):
                            averages=[statistics.mean(r[metric] for r in subset if r['seed']==s) for s in SEEDS]
                            item[metric+'_mean']=statistics.mean(averages); item[metric+'_sample_sd']=statistics.stdev(averages)
                        aggregate.append(item)
                    subset=[r for r in naive if (r['density'],r['penetration'],r['target'])==(density,penetration,target)]
                    item=dict(density=density,penetration=penetration,target=target,method='naive')
                    for metric in ('reward','coverage','payload'):
                        item[metric+'_mean']=statistics.mean(r[metric] for r in subset); item[metric+'_sample_sd']=None
                    aggregate.append(item)
            write_csv(out/'summary.csv',aggregate)
            result=dict(status='completed',finished_at_utc=now(),aggregate=aggregate,freeze_sha256=sha(out/'freeze.json'),
                interpretation='Conditional fixed-role layouts; previously inspected primary test split reused; no Adaptive or MADDPG models available.')
            write_json(out/'summary.json',result); state.update(status='completed',finished_at_utc=now()); atomic(BASE/'status.json',state)
        finally: lock.close()
    except Exception as error:
        atomic(BASE/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc())); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('generate','check','run')); p.add_argument('--wait-for-pid',type=int,default=0)
    a=p.parse_args(); {'generate':generate,'check':check,'run':lambda:run(a.wait_for_pid)}[a.mode]()
