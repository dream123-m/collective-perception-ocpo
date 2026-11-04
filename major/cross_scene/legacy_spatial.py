"""Spatial layout policy evaluation."""






from __future__ import annotations
import json
import statistics
import tempfile
import traceback
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from . import legacy_layout_experiment as e
from .data import _geometric_los,_save_mat
from .layout_only import FixedFadingEnv
from .run import HERE,now,sha,state_hash,write_csv,write_json

OUTPUT=HERE/'results/legacy_vehicle/spatial_layout'
SOURCE={
 'baseline':e.OLD,
 'headway':e.ROOT/'MAPPO/supplemental_experiments/spatial_distribution_sensitivity/data/vehicles_density50_penetration0.5_headway.mat',
}


def run():
    torch.set_num_threads(1)
    bank=e.load_bank(e.DATA,verify=True); cfg=e.config()
    freeze=json.loads((e.OUTPUT/'final_test/freeze.json').read_text(encoding='utf8'))
    if freeze['data_manifest_sha256']!=sha(e.DATA/'manifest.json'): raise ValueError('Frozen bank changed.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'status.json').exists(): raise FileExistsError('Spatial assessment already started.')
    raw={name:loadmat(path) for name,path in SOURCE.items()}
    for key in ('vehicle_ids','is_cav'):
        if not np.array_equal(raw['baseline'][key],raw['headway'][key]): raise ValueError('Vehicle roles differ.')
    if np.array_equal(raw['baseline']['positions'],raw['headway']['positions']): raise ValueError('Layouts are identical.')
    lanes,sizes,roles=e.source_roles()
    if not np.array_equal(raw['baseline']['is_cav'].reshape(-1),roles): raise ValueError('CAV roles do not match training.')
    protocol=dict(code_sha256=sha(Path(__file__)),main_freeze_sha256=sha(e.OUTPUT/'final_test/freeze.json'),
        data_manifest_sha256=sha(e.DATA/'manifest.json'),source_sha256={k:sha(v) for k,v in SOURCE.items()},
        design='Two archived vehicle layouts with the same vehicles/CAV labels; fixed policies trained on the conditional layout bank.',
        inference='One shared fixed fading draw, seven deterministic steps per layout, three training seeds.',
        caveat='One baseline and one headway geometry, not a statistical sample of headway layouts; policies are not independently retrained.',
        baseline='Naive broadcasts every visible non-self object using the same physical environment.',
        geometry='LOS recalculated from positions and vehicle rectangles using the current simulator geometry.')
    write_json(OUTPUT/'protocol.json',protocol); write_json(OUTPUT/'status.json',dict(status='running',started_at_utc=now()))
    try:
        with tempfile.TemporaryDirectory(prefix='spatial_eval_',dir=HERE) as tmp:
            root=Path(tmp); files={}
            for name,raw_scene in raw.items():
                los,comm=_geometric_los(raw_scene['positions'])
                path=root/f'{name}.mat'
                _save_mat(path,dict(positions=raw_scene['positions'],is_cav=raw_scene['is_cav'],
                    vehicle_ids=raw_scene['vehicle_ids'],los_matrix=los,comm_los_matrix=comm))
                files[name]=path
            rows=[]
            with e.adapted():
                template=FixedFadingEnv(e.DATA/bank['splits']['train'][0]['path'],0,e.env_config(cfg))
                for name,path in files.items():
                    scene=FixedFadingEnv(path,0,e.env_config(cfg))
                    for model in freeze['models']:
                        checkpoint=e.OUTPUT/model['checkpoint']
                        if sha(checkpoint)!=model['checkpoint_sha256']: raise ValueError('Frozen checkpoint changed.')
                        policy=e.runner.make_policy(template,cfg)
                        policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'],strict=True)
                        before=state_hash(policy); obs=torch.from_numpy(scene.reset(seed=0)); mask=torch.from_numpy(scene.mask)
                        step=[]
                        with torch.inference_mode():
                            for _ in range(cfg['episode_steps']):
                                action=(policy.probabilities(obs[None])[0]>.5)&mask
                                nxt,reward,done,info=scene.step(action.numpy())
                                step.append((float(np.mean(reward)),float(info['coverage_rate']),float(action.sum())/scene.num_agents))
                                obs=torch.from_numpy(nxt)
                        if before!=state_hash(policy): raise AssertionError('Evaluation changed the actor.')
                        avg=np.asarray(step).mean(0)
                        rows.append(dict(layout=name,method=model['method'],seed=model['seed'],
                            reward=float(avg[0]),coverage=float(avg[1]),payload=float(avg[2]),
                            channel_sha256=scene.channel_hash,unit_fading_sha256=scene.unit_fading_sha256,
                            source_sha256=sha(SOURCE[name]),policy_sha256=before))
                    scene.reset(seed=0); steps=[]
                    for _ in range(cfg['episode_steps']):
                        _,reward,_,info=scene.step(scene.mask)
                        steps.append((float(np.mean(reward)),float(info['coverage_rate']),float(scene.mask.sum())/scene.num_agents))
                    avg=np.asarray(steps).mean(0)
                    rows.append(dict(layout=name,method='naive',seed=0,reward=float(avg[0]),coverage=float(avg[1]),
                        payload=float(avg[2]),channel_sha256=scene.channel_hash,unit_fading_sha256=scene.unit_fading_sha256,
                        source_sha256=sha(SOURCE[name]),policy_sha256='not_applicable'))
            if len({r['unit_fading_sha256'] for r in rows})!=1: raise AssertionError('Fixed fading draw changed.')
            for name in SOURCE:
                if len({r['channel_sha256'] for r in rows if r['layout']==name})!=1: raise AssertionError('Unmatched layout channels.')
            write_csv(OUTPUT/'per_seed.csv',rows)
            aggregate=[]
            for name in SOURCE:
                for method in ('mappo','ocpo','naive'):
                    subset=[r for r in rows if r['layout']==name and r['method']==method]
                    item=dict(layout=name,method=method,n_training_seeds=len(subset) if method!='naive' else 0)
                    for key in ('reward','coverage','payload'):
                        item[key+'_mean']=statistics.mean(r[key] for r in subset)
                        item[key+'_sample_sd']=statistics.stdev(r[key] for r in subset) if len(subset)>1 else None
                    aggregate.append(item)
            result=dict(status='completed',finished_at_utc=now(),aggregate=aggregate,
                        protocol_sha256=sha(OUTPUT/'protocol.json'),per_seed_sha256=sha(OUTPUT/'per_seed.csv'),
                        interpretation=protocol['caveat'])
            write_json(OUTPUT/'summary.json',result)
        write_json(OUTPUT/'status.json',dict(status='completed',finished_at_utc=now()))
        print(json.dumps(result,indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc())); raise


if __name__=='__main__': run()
