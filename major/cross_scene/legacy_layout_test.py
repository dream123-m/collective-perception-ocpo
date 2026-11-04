"""Freeze the six trained policies and evaluate the held-out layout split once."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import torch
from . import legacy_layout_experiment as experiment
from .layout_only import FixedFadingEnv
from .run import ROOT, now, sha, state_hash, write_csv, write_json

OUTPUT = experiment.OUTPUT / 'final_test'


def prepare():
    data, train = experiment.DATA, experiment.OUTPUT
    bank = experiment.load_bank(data, verify=True)
    protocol = json.loads((train/'protocol.json').read_text(encoding='utf-8'))
    if protocol['data_manifest_sha256'] != sha(data/'manifest.json') or protocol['config'] != experiment.config():
        raise ValueError('Training data or settings changed.')
    for name, expected in protocol['code_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Training source changed: {name}')
    models=[]
    for seed in (42,123,456):
        initial=[]
        paths=[]
        for method in ('mappo','ocpo'):
            folder=train/f'{method}_seed_{seed}'
            record=json.loads((folder/'run.json').read_text(encoding='utf-8'))
            with (folder/'training.csv').open(encoding='utf-8-sig',newline='') as f:
                history=list(csv.DictReader(f))
            with (folder/'validation.csv').open(encoding='utf-8-sig',newline='') as f:
                validation=list(csv.DictReader(f))
            best=folder/'best.pt'
            checkpoint=torch.load(best,map_location='cpu',weights_only=False)
            if (record['status'] != 'completed' or record['completed_episodes'] != 1500
                or len(history) != 1500 or record['config'] != protocol['config']
                or record['data_manifest_sha256'] != protocol['data_manifest_sha256']
                or record['code_sha256'] != protocol['code_sha256']
                or sha(best) != record['selected_checkpoint_sha256']
                or checkpoint['episode'] != record['selected_episode']):
                raise ValueError(f'Incomplete or mismatched checkpoint: {method}/{seed}')
            selected=next(row for row in validation if int(row['episode']) == record['selected_episode'])
            if float(selected['reward']) != record['best_validation_reward']:
                raise ValueError('Selected checkpoint does not match validation record.')
            initial.append(record['initial_actor_sha256'])
            paths.append([(r['scene_id'],r['channel_sha256']) for r in history])
            models.append(dict(method=method,seed=seed,episode=record['selected_episode'],
                checkpoint=str(best.relative_to(train)),checkpoint_sha256=sha(best),
                initial_actor_sha256=record['initial_actor_sha256'],
                training_csv_sha256=sha(folder/'training.csv'),validation_csv_sha256=sha(folder/'validation.csv')))
        if initial[0] != initial[1] or paths[0] != paths[1]:
            raise ValueError(f'Paired seed {seed} had different initial actors or physical layout sequence.')
    freeze=dict(created_at_utc=now(),data_manifest_sha256=sha(data/'manifest.json'),
        protocol_sha256=sha(train/'protocol.json'),evaluation_code_sha256=sha(Path(__file__)),
        evaluation='32 disjoint held-out conditional layouts, deterministic actors, one fixed fading draw, 7 steps per layout.',
        checkpoint_selection='Highest validation reward per run; frozen before this test.',models=models)
    return bank, freeze


def main():
    torch.set_num_threads(1)
    bank, freeze=prepare()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'test_started.json').exists() or (OUTPUT/'summary.json').exists():
        raise RuntimeError('Held-out test was already started; refusing an adaptive rerun.')
    write_json(OUTPUT/'freeze.json',freeze)
    write_json(OUTPUT/'test_started.json',dict(started_at_utc=now(),freeze_sha256=sha(OUTPUT/'freeze.json')))
    config=experiment.config()
    per_run=[]
    hashes_by_seed={}
    with experiment.adapted():
        template=FixedFadingEnv(experiment.DATA/bank['splits']['train'][0]['path'],0,experiment.env_config(config))
        for model in freeze['models']:
            policy=experiment.runner.make_policy(template,config)
            checkpoint=torch.load(experiment.OUTPUT/model['checkpoint'],map_location='cpu',weights_only=False)
            policy.load_state_dict(checkpoint['policy'],strict=True)
            rows=experiment.runner.evaluate_frozen(policy,bank['splits']['test'],experiment.DATA,config,
                                      namespace='held_out_layouts',channels=1)
            if len(rows) != 32:
                raise AssertionError('Expected exactly 32 held-out layouts.')
            keys=[(r['scene_id'],r['scene_sha256'],r['channel_sha256'],r['unit_fading_sha256']) for r in rows]
            if model['seed'] in hashes_by_seed and hashes_by_seed[model['seed']] != keys:
                raise AssertionError('Paired test layouts or channel conditions differ.')
            hashes_by_seed[model['seed']]=keys
            name=f"{model['method']}_seed_{model['seed']}.csv"
            write_csv(OUTPUT/name,rows)
            per_run.append(dict(method=model['method'],seed=model['seed'],
                selected_episode=model['episode'],checkpoint_sha256=model['checkpoint_sha256'],
                policy_sha256=state_hash(policy),layouts=len(rows),
                **experiment.runner.summary(rows),evaluation_csv=name,evaluation_csv_sha256=sha(OUTPUT/name)))
    aggregate={}
    for method in ('mappo','ocpo'):
        subset=[r for r in per_run if r['method']==method]
        aggregate[method]={key:dict(mean=statistics.mean(r[key] for r in subset),
                                    sample_sd=statistics.stdev(r[key] for r in subset))
                           for key in ('reward','coverage','payload')}
    diffs=[next(r['reward'] for r in per_run if r['seed']==seed and r['method']=='ocpo')-
           next(r['reward'] for r in per_run if r['seed']==seed and r['method']=='mappo')
           for seed in (42,123,456)]
    result=dict(status='completed',finished_at_utc=now(),freeze_sha256=sha(OUTPUT/'freeze.json'),
        data_manifest_sha256=freeze['data_manifest_sha256'],per_run=per_run,aggregate=aggregate,
        paired_reward_differences=dict(seeds=[42,123,456],values=diffs,
                                       mean=statistics.mean(diffs),sample_sd=statistics.stdev(diffs)),
        limitation='Conditional original-count layouts, fixed CAV lane/rank roles and one fixed fading realization; not unconditional traffic sampling.')
    write_json(OUTPUT/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
