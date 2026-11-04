"""Layout generation and experiment configuration."""










from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import traceback

import numpy as np
import torch
from scipy.io import loadmat

from . import layout_only as runner
from . import legacy_update_layout as legacy
from .data import _geometric_los, _save_mat
from .multi_layout_update import optimize
from .run import HERE, ROOT, environment_config as base_environment_config, now, sha, write_json, acquire_batch_lock

SOURCE = Path(__file__).resolve()
OLD = ROOT / 'MAPPO/vehicles_density50_penetration0.5.mat'
BASE = HERE / 'results/legacy_vehicle'
DATA = BASE / 'data'
OUTPUT = BASE / 'training'
SEED = 20261029
COUNTS = {'train': 64, 'validation': 16, 'test': 32}


def source_roles():
    raw = loadmat(OLD)
    positions = np.asarray(raw['positions'], dtype=np.float64)
    labels = np.asarray(raw['is_cav']).reshape(-1).astype(np.uint8)
    lanes = np.unique(positions[:, 1])
    if len(positions) != 122 or labels.sum() != 65 or len(lanes) != 4:
        raise AssertionError('Original scene counts changed.')
    sizes, roles = [], []
    for y in lanes:
        ix = np.flatnonzero(positions[:, 1] == y)
        ix = ix[np.argsort(positions[ix, 0])]
        sizes.append(len(ix)); roles.extend(labels[ix].tolist())
    if sizes != [34, 24, 34, 30] or sum(roles) != 65:
        raise AssertionError('Original lane/rank roles changed.')
    return lanes, sizes, np.asarray(roles, dtype=np.uint8)


def conditional_lane(rng, size):
    
    
    for attempt in range(1, 10001):
        points = np.sort(rng.uniform(0., 1000., size=rng.poisson(50.)))
        kept = []
        for x in points:
            if not kept or x - kept[-1] >= 10.2:
                kept.append(float(x))
        if len(kept) == size:
            return np.asarray(kept), attempt
    raise RuntimeError(f'Failed to draw a conditional lane with {size} vehicles.')


def _record_hash(positions, labels):
    digest = hashlib.sha256()
    digest.update(np.asarray(positions, dtype='<f8').tobytes())
    digest.update(np.asarray(labels, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def load_bank(root=DATA, verify=True):
    root = Path(root).resolve()
    m = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    lanes, sizes, roles = source_roles()
    if m['original_mat_sha256'] != sha(OLD) or m['counts'] != COUNTS:
        raise ValueError('Source MAT or bank counts changed.')
    seen = set()
    for split, expected in COUNTS.items():
        if len(m['splits'][split]) != expected:
            raise ValueError('Incorrect split count.')
        for record in m['splits'][split]:
            path = (root/record['path']).resolve()
            if not path.is_relative_to(root) or sha(path) != record['file_sha256']:
                raise ValueError(f'Layout hash mismatch: {path}')
            if verify:
                raw = loadmat(path)
                pos = np.asarray(raw['positions'], dtype=np.float64)
                labels = np.asarray(raw['is_cav']).reshape(-1)
                if pos.shape != (122,2) or not np.array_equal(labels, roles):
                    raise ValueError('Layout vehicle roles changed.')
                offset = 0
                for lane, size in zip(lanes, sizes):
                    a = pos[offset:offset+size]
                    if (not np.all(a[:,1] == lane) or np.any(np.diff(a[:,0]) < 10.2 - 1e-9)
                            or np.any(a[:,0] < 0) or np.any(a[:,0] > 1000)):
                        raise ValueError('Invalid lane geometry.')
                    offset += size
                for key in ('los_matrix','comm_los_matrix'):
                    matrix = np.asarray(raw[key])
                    if matrix.shape != (122,122) or not np.array_equal(matrix,matrix.T) or np.diag(matrix).any():
                        raise ValueError('Invalid LOS matrix.')
                h = _record_hash(pos, labels)
                if h != record['scene_sha256'] or h in seen:
                    raise ValueError('Repeated or changed layout.')
                seen.add(h)
    return {'splits':m['splits'], 'split_counts':COUNTS, 'manifest':m}


def generate():
    if (DATA/'manifest.json').exists():
        load_bank(DATA)
        print('Existing compatible bank verified.', flush=True)
        return
    if DATA.exists() and any(DATA.iterdir()):
        raise FileExistsError('Refusing to overwrite an incomplete dataset.')
    DATA.mkdir(parents=True, exist_ok=True)
    lanes, sizes, roles = source_roles()
    manifest = dict(schema_version=1, original_mat_sha256=sha(OLD), generator_source_sha256=sha(SOURCE),
        original_mat='MAPPO/vehicles_density50_penetration0.5.mat', master_seed=SEED, counts=COUNTS,
        generator='Independent Poisson candidates of mean 50 per lane; uniform positions; sequential 10.2 m hard-core thinning.',
        conditioning='Retain only draws with per-lane counts [34,24,34,30]; keep original CAV lane/rank labels. Conditional, not unconditional, original process.',
        source_lane_counts=sizes, source_lane_cav_counts=[int(roles[sum(sizes[:i]):sum(sizes[:i+1])].sum()) for i in range(4)],
        geometry='Original road and vehicle dimensions; LOS recalculated with self excluded.',
        split_semantics='Independent per-layout RNG seeds; no complete scene reused across splits.',
        splits={name:[] for name in COUNTS})
    for split_index, (split, total) in enumerate(COUNTS.items(), 1):
        folder = DATA/split
        folder.mkdir()
        for index in range(total):
            positions=[]; attempts=[]
            for lane_index, (y, count) in enumerate(zip(lanes, sizes)):
                rng = np.random.default_rng(np.random.SeedSequence([SEED,split_index,index,lane_index]))
                x, tries = conditional_lane(rng, count)
                attempts.append(tries)
                positions.append(np.column_stack((x, np.full(count,y))))
            pos=np.concatenate(positions)
            los, comm=_geometric_los(pos)
            path=folder/f'layout_{index:03d}.mat'
            _save_mat(path, dict(positions=pos,is_cav=roles[:,None],vehicle_ids=np.arange(1,123)[:,None],
                                 los_matrix=los,comm_los_matrix=comm))
            manifest['splits'][split].append(dict(scene_id=f'{split}_{index:03d}',
                path=path.relative_to(DATA).as_posix(), file_sha256=sha(path),
                scene_sha256=_record_hash(pos,roles), rejection_attempts=attempts))
        print(f'Generated {split}: {total} independent conditional layouts.', flush=True)
    write_json(DATA/'manifest.json',manifest)
    load_bank(DATA)
    print('Dataset hashes and geometry verified.', flush=True)


def env_config(config):
    return {**base_environment_config(config), 'required_vehicle_count':122, 'required_cav_count':65}


def config(episodes=1500):
    c = legacy.config(episodes)
    
    c.update(episode_steps=7, minibatch_joint_steps=7)
    return c


def protocol(data, output, config):
    data, output = Path(data).resolve(), Path(output).resolve()
    if not data.is_relative_to(HERE) or not output.is_relative_to(HERE):
        raise ValueError('Keep data and results inside major/cross_scene.')
    legacy.check_config(config)
    manifest = load_bank(data)
    code = runner.source_hashes()
    for path in (SOURCE, Path(legacy.__file__).resolve()):
        code[path.relative_to(ROOT).as_posix()] = sha(path)
    info=dict(variant='conditional_original_layouts', config=config,
        data_manifest_sha256=sha(data/'manifest.json'), original_mat_sha256=sha(OLD), code_sha256=code,
        methods=['mappo','ocpo'], seeds=[42,123,456],
        layouts='Original candidate/thinning rule conditioned on old lane counts; fixed old CAV lane/rank roles.',
        training='One layout per update, original-style unscaled critic, window-local sample-SD advantages, 7 joint steps and 10 PPO epochs.',
        method_difference='MAPPO individual advantages and vehicle clipping; OCPO team advantage and overlap factor clipping.',
        actor='New layout-compatible observations, 65 independent actors. Not byte-identical to 2026 single-layout model.',
        fading='Corrected unit-mean draw, fixed across layouts and all splits; geometry remains layout dependent.',
        selection='Best 16-layout validation reward; initial model is recorded but not selected.',
        final_test='Not performed by the training runner.', created_at_utc=now())
    output.mkdir(parents=True, exist_ok=True)
    path=output/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text(encoding='utf-8'))
        for key in ('variant','config','data_manifest_sha256','original_mat_sha256','code_sha256'):
            if existing[key] != info[key]:
                raise ValueError(f'Incompatible protocol: {key}')
        return manifest,existing
    allowed={'batch.lock','status.json','stage.stdout.log','stage.stderr.log'}
    if any(p.name not in allowed for p in output.iterdir()):
        raise FileExistsError('Refusing to overwrite output.')
    write_json(path,info)
    for name in code:
        dest=output/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    return manifest,info


@contextmanager
def adapted():
    names=('ScaledValueCritic','pack_rollouts','optimize','prepare_protocol',
           'check_config','load_manifest','environment_config')
    saved={name:getattr(runner,name) for name in names}
    runner.ScaledValueCritic=legacy.LegacyCritic
    runner.pack_rollouts=legacy.pack
    runner.optimize=optimize
    runner.prepare_protocol=protocol
    runner.check_config=legacy.check_config
    runner.load_manifest=load_bank
    runner.environment_config=env_config
    try:
        yield
    finally:
        for name,value in saved.items(): setattr(runner,name,value)


def check():
    load_bank(DATA)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='legacy_bank_check_',dir=HERE) as temp:
        hashes=[]
        with adapted():
            for method in ('mappo','ocpo'):
                record=runner.run_training(method,42,DATA,Path(temp),config(2))
                if record['status'] != 'completed' or record['optimizer_steps'] != 20 or record['agents'] != 65:
                    raise AssertionError('Preflight training did not finish correctly.')
                hashes.append(record['initial_actor_sha256'])
        if hashes[0] != hashes[1]:
            raise AssertionError('Methods did not start with identical actors.')
    print('MAPPO and OCPO preflight passed; temporary results removed.',flush=True)


def run():
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=acquire_batch_lock(OUTPUT)
    try:
        with adapted():
            write_json(OUTPUT/'status.json',dict(status='running',method='mappo',seed=42,episodes=1500,
                       started_at_utc=now(),final_test=False))
            result=runner.run_training('mappo',42,DATA,OUTPUT,config())
        write_json(OUTPUT/'status.json',dict(status='completed',method='mappo',seed=42,episodes=1500,
                   finished_at_utc=now(),selected_episode=result['selected_episode'],final_test=False))
    except Exception as error:
        write_json(OUTPUT/'status.json',dict(status='failed',error=str(error),traceback=traceback.format_exc(),final_test=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('generate','check','run'))
    args=parser.parse_args()
    {'generate':generate,'check':check,'run':run}[args.mode]()
