"""Verify, plot, evaluate, or train the supplied review materials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np
import pandas as pd

from plot_figures import plotting

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'major/cross_scene/results/legacy_vehicle'
METRICS=('reward','coverage','payload')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def verify():
    for dataset in read(ROOT/'data/datasets.json'):
        path=ROOT/dataset['path'];bank=read(path)
        seen=set()
        for split in ('train','validation','test'):
            assert len(bank['splits'][split])==dataset[split]
            for record in bank['splits'][split]:
                assert sha(path.parent/record['path'])==record['file_sha256']
                assert record['scene_sha256'] not in seen
                seen.add(record['scene_sha256'])
    jobs=read(ROOT/'models.json')
    for job in jobs:
        assert sha(ROOT/job['checkpoint'])==job['checkpoint_sha256']
        record=read(ROOT/job['run_record'])
        assert record['completed_episodes']==record['config']['episodes']==1500
        folder=(ROOT/job['run_record']).parent
        training=pd.read_csv(folder/'training.csv')
        assert training.episode.tolist()==list(range(1,1501))
        validation=pd.read_csv(folder/'validation.csv')
        valid=validation[validation.episode>0]
        if job['family']=='maddpg':
            valid=valid[valid.episode*record['config']['episode_steps']>=record['config']['warmup_steps']]
        assert np.isclose(valid.reward.max(),record['best_validation_reward'],atol=1e-6)
        reference=pd.read_csv(ROOT/job['reference'])
        assert len(reference)==reference.scene_id.nunique()==32
        assert np.isfinite(reference[list(METRICS)]).all().all()
    per_seed=pd.read_csv(ROOT/'data/table4_per_seed.csv')
    table=pd.read_csv(ROOT/'data/table4.csv').set_index('method')
    for method,group in per_seed.groupby('method'):
        for key in METRICS:
            assert np.isclose(group[key].mean(),table.loc[method,key+'_mean'],atol=1e-12)
            assert np.isclose(group[key].std(ddof=1),table.loc[method,key+'_std'],atol=1e-12)
    expected={'mappo':4.8272,'ocpo':4.9198,'maddpg':3.5240,
              'team_vehicle':4.7907,'individual_pair':4.7182,'target_shuffle':4.8605}
    for method,value in expected.items():assert abs(table.loc[method,'reward_mean']-value)<.00005
    latency=pd.read_csv(ROOT/'data/table5_samples.csv')
    per=pd.read_csv(ROOT/'data/table5_per_seed.csv')
    keys=['density','penetration','kind','seed']
    means=latency.groupby(keys).decision_time_ms.mean()
    for row in per.itertuples():
        group=latency[(latency.density==row.density)&np.isclose(latency.penetration,row.penetration)
                      &(latency.kind==row.kind)&(latency.seed==row.seed)]
        assert len(group)==200
        assert np.isclose(means.loc[(row.density,row.penetration,row.kind,row.seed)],row.decision_time_ms,atol=1e-12)
    curves=pd.read_csv(ROOT/'data/fig5_training.csv')
    assert len(curves)==3*3*1500
    for job in jobs:
        reference=pd.read_csv(ROOT/job['reference'])
        if job['scope']=='main':
            selected=curves[(curves.method==job['method'])&(curves.seed==job['seed'])]
            original=pd.read_csv((ROOT/job['run_record']).parent/'training.csv')
            np.testing.assert_allclose(selected.reward,original.reward,atol=1e-12,rtol=0)
        if job['scope'] in ('main','ablation'):
            selected=per_seed[(per_seed.method==job['method'])&(per_seed.seed==job['seed'])]
            for key in METRICS:assert np.isclose(reference[key].mean(),selected[key].iloc[0],atol=1e-12)
    grid=pd.read_csv(ROOT/'data/fig6_7_summary.csv')
    assert len(grid)==72
    naive=pd.read_csv(BASE/'scenario_grid/single_seed/exploratory_test/naive_per_layout.csv')
    for row in grid.itertuples():
        if row.method=='naive':
            source=naive[(naive.density==row.density)&np.isclose(naive.penetration,row.penetration)&np.isclose(naive.target,row.target)]
        else:
            job=next(j for j in jobs if j['scope']=='grid' and j['method']==row.method and j['density']==row.density and j['penetration']==row.penetration and j['target']==row.target)
            source=pd.read_csv(ROOT/job['reference'])
        assert len(source)==32
        for key in METRICS:assert np.isclose(source[key].mean(),getattr(row,key+'_mean'),atol=1e-12)
    mobility=pd.read_csv(ROOT/'data/fig8_per_layout.csv')
    for row in read(ROOT/'data/fig8_summary.json')['aggregate']:
        group=mobility[(mobility.method==row['method'])&(mobility.speed_kmh==row['speed_kmh'])]
        assert len(group)==96
        for key in METRICS:
            means=group.groupby('seed')[key].mean()
            assert np.isclose(means.mean(),row[key+'_mean'],atol=1e-12)
            assert np.isclose(means.std(ddof=1),row[key+'_sample_sd'],atol=1e-12)
    target=pd.read_csv(ROOT/'data/fig9_per_cav.csv')
    for row in read(ROOT/'data/fig9_summary.json')['aggregate']:
        group=target[np.isclose(target.target,row['target'])]
        assert len(group)==195 and np.isclose(group.payload.mean(),row['payload_mean'],atol=1e-12)
    for target,value in ((.9,2.0513),(.95,3.0578)):
        selected=grid[np.isclose(grid.target,target)&(grid.method=='maddpg')]
        assert len(selected)==9 and abs(selected.reward_mean.mean()-value)<.00005
    print(f'Verified {len(jobs)} model entries, dataset splits, validation selection, tables and latency samples.')





def evaluate(scope='main',example=False):
    import torch
    from major.cross_scene.layout_only import FixedFadingEnv
    from major.model import JointPolicy
    from major.maddpg.discrete_grid import BinaryActors
    torch.set_num_threads(1)
    jobs=read(ROOT/'models.json')
    jobs=[job for job in jobs if scope=='all' or job['scope']==scope]
    if example:jobs=[next(job for job in jobs if job['method']=='ocpo' and job['seed']==123)]
    output=[]
    for job in jobs:
        data=ROOT/job['data'];bank=read(data/'manifest.json')
        records=bank['splits']['test'][:1] if example else bank['splits']['test']
        template=FixedFadingEnv(data/records[0]['path'],0,config=job['environment_config'])
        policy=(BinaryActors(template.state_dim,template.num_agents,template.max_targets_per_cav)
                if job['family']=='maddpg' else JointPolicy(template.state_dim,template.num_agents,template.max_targets_per_cav,[64,64],use_editor=False))
        saved=torch.load(ROOT/job['checkpoint'],map_location='cpu',weights_only=False)
        policy.load_state_dict(saved['policy'],strict=True);policy.eval()
        expected=pd.read_csv(ROOT/job['reference']).set_index('scene_id')
        metrics=[]
        with torch.inference_mode():
            for record in records:
                env=FixedFadingEnv(data/record['path'],0,config=job['environment_config'])
                obs=torch.from_numpy(env.reset(seed=0));mask=torch.from_numpy(env.mask)
                values=[]
                for _ in range(job['environment_config']['episode_steps']):
                    action=(policy.deterministic(obs[None],mask)[0] if job['family']=='maddpg'
                            else ((policy.probabilities(obs[None])[0]>.5)&mask))
                    nxt,reward,done,info=env.step(action.numpy())
                    values.append([float(np.mean(reward)),float(info['coverage_rate']),float(action.sum())/env.num_agents])
                    obs=torch.from_numpy(nxt)
                    if done:break
                mean=np.asarray(values).mean(0)
                reference=expected.loc[record['scene_id']]
                np.testing.assert_allclose(mean,reference[list(METRICS)].to_numpy(dtype=float),atol=1e-6,rtol=1e-6)
                for key,actual in [('channel_sha256',env.channel_hash),('unit_fading_sha256',env.unit_fading_sha256),('role_sha256',env.role_sha256)]:
                    if key in reference:assert actual==reference[key],key
                metrics.append(mean)
        result={key:float(value) for key,value in zip(METRICS,np.mean(metrics,axis=0))}
        result.update(scope=job['scope'],method=job['method'],seed=job['seed'],density=job['density'],penetration=job['penetration'],target=job['target'],layouts=len(records))
        output.append(result)
        print(json.dumps(result),flush=True)
    save_json(ROOT/'outputs'/('example.json' if example else f'evaluation_{scope}.json'),output)
    print('All evaluated metrics match the supplied raw records within 1e-6.')


def train(args):
    import torch
    from major.cross_scene import legacy_layout_experiment as experiment
    from major.cross_scene import legacy_scenario_grid as grid
    torch.set_num_threads(1)
    output=ROOT/f'major/cross_scene/results/new_training/{args.method}_seed_{args.seed}_{args.density}_{args.penetration}_{args.target}'
    if output.exists():raise FileExistsError(f'Choose a fresh copy or output folder: {output.relative_to(ROOT)}')
    if args.method=='maddpg':
        from major.maddpg import state_matched as baseline
        baseline.retained.SEED=args.seed
        baseline.run_case((args.density,args.penetration,args.target),output=output)
    else:
        source=grid.source_info(args.density,args.penetration)
        primary=(args.density,args.penetration)==(50,.5)
        data=experiment.DATA if primary else grid.DATA/grid.key(args.density,args.penetration)
        config=grid.config_for(experiment,source,args.target)
        context=experiment.adapted() if primary else grid.context(data,source)
        with context:experiment.runner.run_training(args.method,args.seed,data,output,config)


def smoke():
    from major.cross_scene import legacy_layout_experiment as experiment
    from major.maddpg import state_matched as baseline
    experiment.check()
    with tempfile.TemporaryDirectory(prefix='review_check_',dir=ROOT/'major/maddpg') as temp:
        baseline.run_case((50,.5,.95),output=Path(temp),smoke=True)
    print('Short runtime checks passed for MAPPO, OCPO and the adopted MADDPG. Temporary runs removed.')


def latency():
    from major.cross_scene import legacy_inference_latency as benchmark
    from major.cross_scene import legacy_layout_experiment as experiment
    output=ROOT/'outputs/latency'
    if (output/'protocol.json').exists():raise FileExistsError('outputs/latency already contains a benchmark.')
    old=experiment.OUTPUT
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='latency_inputs_',dir=ROOT/'outputs') as temp:
        folder=Path(temp);freeze=read(BASE/'training/final_test/freeze.json')
        jobs=read(ROOT/'models.json')
        for item in freeze['models']:
            job=next(j for j in jobs if j['scope']=='main' and j['method']==item['method'] and j['seed']==item['seed'])
            item['checkpoint']=(ROOT/job['checkpoint']).as_posix()
        save_json(folder/'final_test/freeze.json',freeze)
        try:
            experiment.OUTPUT=folder;benchmark.OUTPUT=output;benchmark.run(0)
        finally:experiment.OUTPUT=old


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','plot','example','evaluate','smoke','train','latency','all'])
    parser.add_argument('--scope',choices=['main','grid','ablation','target','all'],default='main')
    parser.add_argument('--method',choices=['mappo','ocpo','maddpg'],default='ocpo')
    parser.add_argument('--seed',type=int,choices=[42,123,456],default=123)
    parser.add_argument('--density',type=int,choices=[25,50,75],default=50)
    parser.add_argument('--penetration',type=float,choices=[.3,.5,.7],default=.5)
    parser.add_argument('--target',type=float,choices=[.9,.95],default=.95)
    args=parser.parse_args()
    if args.command=='plot':plotting();return
    if args.command=='train':train(args);return
    verify()
    if args.command=='all':plotting()
    if args.command in ('example','all'):evaluate(example=True)
    if args.command=='evaluate':evaluate(args.scope)
    if args.command=='smoke':smoke()
    if args.command=='latency':latency()
    import sys
    for name,module in sys.modules.items():
        if name.split('.')[0] in ('major','MAPPO') and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(ROOT), f'Imported outside the review folder: {name}'


if __name__=='__main__':main()
