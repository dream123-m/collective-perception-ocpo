"""Vehicle motion and policy evaluation."""








from __future__ import annotations
import csv
import hashlib
import json
import statistics
import tempfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch

from . import legacy_layout_experiment as experiment
from .data import _geometric_los, _save_mat
from .layout_only import FixedFadingEnv
from .run import ROOT,HERE,now,sha,state_hash,write_csv,write_json

OUTPUT=HERE/'results/legacy_vehicle/mobility'
SPEEDS=(0,60,90,120)
INTERVAL=.5
MOBILITY_SEED=20261030
VEHICLE_LENGTH=5.2


def moved_scene(raw,index,speed):
    base=np.asarray(raw['positions'],dtype=np.float64)
    rng=np.random.default_rng(np.random.SeedSequence([MOBILITY_SEED,index]))
    ratio=np.clip(rng.normal(1.,.2,len(base)),.8,1.2)
    moved=base.copy(); moved[:,0]+=(speed/3.6)*ratio*INTERVAL
    corrections=0
    
    
    for y in np.unique(base[:,1]):
        lane=np.flatnonzero(base[:,1]==y)
        lane=lane[np.argsort(base[lane,0])]
        for j in range(len(lane)-2,-1,-1):
            follower,leader=lane[j],lane[j+1]
            limit=moved[leader,0]-VEHICLE_LENGTH
            if moved[follower,0]>limit:
                moved[follower,0]=limit; corrections+=1
        if np.any(np.diff(moved[lane,0])<VEHICLE_LENGTH-1e-9):
            raise AssertionError('Vehicle rectangles overlap within a lane.')
    if np.any(moved[:,0] < base[:,0]-1e-9): raise AssertionError('A vehicle moved backward.')
    los,comm=_geometric_los(moved)
    return dict(positions=moved,is_cav=np.asarray(raw['is_cav']),vehicle_ids=np.asarray(raw['vehicle_ids']),
                los_matrix=los,comm_los_matrix=comm),dict(follower_corrections=corrections,
                    outside_road=int(np.sum((moved[:,0]<0)|(moved[:,0]>1000))),
                    mean_displacement_m=float(np.mean(moved[:,0]-base[:,0])))


def digest_scene(scene):
    h=hashlib.sha256(); h.update(np.asarray(scene['positions'],dtype='<f8').tobytes())
    h.update(np.asarray(scene['is_cav'],dtype=np.uint8).reshape(-1).tobytes()); return h.hexdigest()


def frozen_models():
    folder=experiment.OUTPUT/'final_test'
    freeze=json.loads((folder/'freeze.json').read_text(encoding='utf8'))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    if freeze['data_manifest_sha256']!=sha(experiment.DATA/'manifest.json') or summary['status']!='completed':
        raise ValueError('Baseline split or test record changed.')
    if len(freeze['models'])!=6: raise ValueError('Expected six frozen policies.')
    for model in freeze['models']:
        if sha(experiment.OUTPUT/model['checkpoint'])!=model['checkpoint_sha256']:
            raise ValueError('Frozen policy checkpoint changed.')
    return freeze,summary


def run():
    torch.set_num_threads(1)
    bank=experiment.load_bank(experiment.DATA,verify=True)
    freeze,summary=frozen_models(); cfg=experiment.config()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'status.json').exists(): raise FileExistsError('Mobility result already exists.')
    protocol=dict(code_sha256=sha(Path(__file__)),data_manifest_sha256=sha(experiment.DATA/'manifest.json'),
        original_freeze_sha256=sha(experiment.OUTPUT/'final_test/freeze.json'),
        baseline_test_sha256=sha(experiment.OUTPUT/'final_test/summary.json'),
        speeds_kmh=list(SPEEDS),interval_seconds=INTERVAL,mobility_seed=MOBILITY_SEED,
        speeds='Per vehicle Normal(mean speed, 0.2*mean), clipped to [0.8,1.2] times mean. The same standardized draws are reused across speeds.',
        following='Same lane, positive x, no overtaking; follower movement constrained to preserve a 5.2 m minimum center gap.',
        road_boundary='Displaced vehicles may exit the original 1000 m segment; none are wrapped or removed.',
        geometry='Recompute LOS, sensing and local geometric ordering using the displaced positions.',
        fading='Same unit fading matrix, fixed seed 0, across all layouts, methods and speeds; geometry changes link powers.',
        policy='Frozen best validation checkpoints; no further training.',
        inference='Deterministic local actors; seven-step average reward/coverage/payload.',
        split='Transforms the 32 already-inspected held-out layouts; results are exploratory, not new independent layouts.')
    write_json(OUTPUT/'protocol.json',protocol)
    write_json(OUTPUT/'status.json',dict(status='running',started_at_utc=now(),phase='construct_geometry'))
    try:
        with tempfile.TemporaryDirectory(prefix='mobility_eval_',dir=HERE) as name:
            root=Path(name); scenes={}; geometry=[]
            for speed in SPEEDS:
                folder=root/f'speed_{speed}'; folder.mkdir(); records=[]
                for index,record in enumerate(bank['splits']['test']):
                    raw=loadmat(experiment.DATA/record['path'])
                    scene,details=moved_scene(raw,index,speed)
                    if speed==0:
                        if not np.array_equal(scene['positions'],raw['positions']) or not np.array_equal(scene['los_matrix'],raw['los_matrix']):
                            raise AssertionError('Zero-speed layout differs from saved source.')
                    path=folder/f'layout_{index:03d}.mat'; _save_mat(path,scene)
                    records.append(dict(scene_id=record['scene_id'],path=path.relative_to(root).as_posix(),
                                        scene_sha256=digest_scene(scene)))
                    geometry.append(dict(speed_kmh=speed,scene_id=record['scene_id'],
                                         source_scene_sha256=record['scene_sha256'],moved_scene_sha256=digest_scene(scene),
                                         moved_mat_sha256=sha(path),**details))
                scenes[speed]=records
            write_csv(OUTPUT/'geometry.csv',geometry)
            write_json(OUTPUT/'status.json',dict(status='running',started_at_utc=now(),phase='zero_speed_verification'))
            all_rows=[]; per_seed=[]; model_list=freeze['models']
            with experiment.adapted():
                template=FixedFadingEnv(experiment.DATA/bank['splits']['train'][0]['path'],0,experiment.env_config(cfg))
                policies=[]
                for model in model_list:
                    policy=experiment.runner.make_policy(template,cfg)
                    saved=torch.load(experiment.OUTPUT/model['checkpoint'],map_location='cpu',weights_only=False)
                    policy.load_state_dict(saved['policy'],strict=True); policies.append((model,policy))
                
                for speed in SPEEDS:
                    for model,policy in policies:
                        rows=experiment.runner.evaluate_frozen(policy,scenes[speed],root,cfg,namespace='motion',channels=1)
                        if speed==0:
                            path=experiment.OUTPUT/'final_test'/f"{model['method']}_seed_{model['seed']}.csv"
                            with path.open(encoding='utf-8-sig',newline='') as f: saved=list(csv.DictReader(f))
                            if len(saved)!=len(rows): raise AssertionError('Static reference row count changed.')
                            for actual,reference in zip(rows,saved):
                                if actual['scene_id']!=reference['scene_id'] or actual['channel_sha256']!=reference['channel_sha256']:
                                    raise AssertionError('Zero-speed scene or channel differs.')
                                for key in ('reward','coverage','payload'):
                                    if abs(float(actual[key])-float(reference[key]))>1e-7:
                                        raise AssertionError(f'Zero-speed {key} differs.')
                        for row in rows: all_rows.append(dict(method=model['method'],seed=model['seed'],speed_kmh=speed,**row))
                        per_seed.append(dict(method=model['method'],seed=model['seed'],speed_kmh=speed,
                                             layouts=len(rows),**experiment.runner.summary(rows)))
                    print(f'Finished paired mobility speed={speed} km/h.',flush=True)
            write_csv(OUTPUT/'per_layout.csv',all_rows); write_csv(OUTPUT/'per_seed.csv',per_seed)
            aggregate=[]
            for speed in SPEEDS:
                for method in ('mappo','ocpo'):
                    subset=[x for x in per_seed if x['speed_kmh']==speed and x['method']==method]
                    row=dict(method=method,speed_kmh=speed)
                    for key in ('reward','coverage','payload'):
                        row[key+'_mean']=statistics.mean(x[key] for x in subset)
                        row[key+'_sample_sd']=statistics.stdev(x[key] for x in subset)
                    aggregate.append(row)
            result=dict(status='completed',finished_at_utc=now(),zero_speed_matches_saved_test=True,
                        paired_layouts=32,seeds=[42,123,456],speeds_kmh=list(SPEEDS),
                        aggregate=aggregate,per_seed=per_seed,protocol_sha256=sha(OUTPUT/'protocol.json'),
                        geometry_sha256=sha(OUTPUT/'geometry.csv'),per_layout_sha256=sha(OUTPUT/'per_layout.csv'),
                        interpretation=protocol['split'])
            write_json(OUTPUT/'summary.json',result)
        write_json(OUTPUT/'status.json',dict(status='completed',finished_at_utc=now(),zero_speed_verified=True))
        print(json.dumps(result,indent=2),flush=True)
    except Exception as error:
        import traceback
        write_json(OUTPUT/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': run()
