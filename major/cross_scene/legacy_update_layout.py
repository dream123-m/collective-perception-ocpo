"""Compare the retained MAPPO update with OCPO on fixed-fading layouts.

The 2026 single-layout checkpoint is the reference, not the unsnapshotted 2025
run.  New layouts require new observation dimensions.  The shared training
loop uses one layout per update, an unscaled original critic and per-rollout
sample-SD advantage normalization.  OCPO keeps its team advantage and overlap
factors.  This is an adaptation, not a byte-identical historical reproduction.
"""
from __future__ import annotations

import argparse
import copy
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import traceback

import numpy as np
import torch
from MAPPO.mappo_network import CriticNetwork
from . import layout_only as runner
from . import multi_layout_update
from .data import load_manifest
from .run import HERE, ROOT, now, sha, state_hash, write_json, acquire_batch_lock
from .truncation_update import pack_rollouts as original_pack

DATA = HERE / 'data'
OUTPUT = HERE / 'results' / 'baseline_recovery' / 'legacy_update_layout'
SOURCE = Path(__file__).resolve()


class LegacyCritic(CriticNetwork):
    """Original 2T+1 critic; ignore extra geometry on the new state interface."""
    def __init__(self, num_targets, hidden_dims, unused_scale):
        super().__init__(num_targets, hidden_dims)
        self.num_targets = num_targets
        self.initial_unscaled_sha256 = state_hash(self)
        self.register_buffer('value_scale', torch.tensor(1., dtype=torch.float32))

    def forward(self, broadcast_state, coverage_rate):
        return super().forward(broadcast_state[..., :2 * self.num_targets], coverage_rate)


def pack(rollouts, method, config):
    if len(rollouts) != 1:
        raise ValueError('The legacy-adapted update requires exactly one rollout.')
    batch = original_pack(rollouts, method, config)
    adv = batch['raw_advantage']
    if len(adv) > 1:
        batch['advantage'] = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)
    return batch


def check_config(config):
    multi_layout_update.check_config(config)
    if (config['layouts_per_update'] != 1 or
        config['value_normalization'] != 'legacy_unscaled' or
        config['fading_mode'] != 'fixed_across_layouts' or
        config['fixed_fading_seed'] != 0 or
        config['validation_channels_per_layout'] != 1 or
        config['test_channels_per_layout'] != 1):
        raise ValueError('Invalid legacy-adapted fixed-layout protocol.')


def protocol(data, output, config):
    data, output = Path(data).resolve(), Path(output).resolve()
    if not data.is_relative_to(HERE) or not output.is_relative_to(HERE):
        raise ValueError('Data and results must be under major/cross_scene.')
    check_config(config)
    manifest = load_manifest(data, verify=True)
    code = runner.source_hashes()
    code[SOURCE.relative_to(ROOT).as_posix()] = sha(SOURCE)
    saved = dict(variant='legacy_update_layout', config=config, methods=['mappo','ocpo'],
        seeds=[42,123,456], data_manifest_sha256=sha(data / 'manifest.json'), code_sha256=code,
        reference='2026 MAPPO major/train.py snapshot; per-vehicle observations are correctly stored',
        adaptation='New layout observations and fixed-fading environment; single layout rollout/update, unscaled original critic and sample-SD rollout advantages.',
        common='Identical environment, layout bank, update budget, architecture and hyperparameters for both methods.',
        method_difference='MAPPO individual advantage and vehicle clipping; OCPO team advantage and overlap factor clipping.',
        selection='Highest reward on the 16 validation layouts, same for both methods.',
        final_test='Not performed by this training script.', created_at_utc=now())
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists():
        import json
        old = json.loads(path.read_text(encoding='utf-8'))
        for key in ('variant','config','data_manifest_sha256','code_sha256'):
            if old[key] != saved[key]:
                raise ValueError(f'Protocol mismatch: {key}')
        return manifest, old
    allowed = {'batch.lock','stage.stdout.log','stage.stderr.log','status.json'}
    if any(p.name not in allowed for p in output.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing experiment.')
    write_json(path, saved)
    for name in code:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return manifest, saved


@contextmanager
def adapted_runner():
    names = ('ScaledValueCritic','pack_rollouts','optimize','prepare_protocol','check_config')
    old = {name: getattr(runner, name) for name in names}
    runner.ScaledValueCritic = LegacyCritic
    runner.pack_rollouts = pack
    runner.optimize = multi_layout_update.optimize
    runner.prepare_protocol = protocol
    runner.check_config = check_config
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(runner, name, value)


def config(episodes=1500):
    result = copy.deepcopy(runner.DEFAULT_CONFIG)
    result.update(episodes=episodes, layouts_per_update=1,
                  value_normalization='legacy_unscaled')
    return result


def check():
    torch.set_num_threads(1)
    # The adapted critic must produce exactly the original critic output and
    # gradients when given the shared broadcast fields.
    torch.manual_seed(42)
    old = CriticNetwork(4, [8, 8])
    torch.manual_seed(42)
    new = LegacyCritic(4, [8, 8], .05)
    x = torch.randn(3, 24); c = torch.rand(3, 1)
    torch.testing.assert_close(old(x[:, :8], c), new(x, c), atol=0, rtol=0)
    with tempfile.TemporaryDirectory(prefix='legacy_check_', dir=HERE) as tmp:
        with adapted_runner():
            initial_hashes = []
            for method in ('mappo','ocpo'):
                result = runner.run_training(method, 42, DATA, Path(tmp), config(2))
                if result['status'] != 'completed' or result['optimizer_steps'] != 20:
                    raise AssertionError(f'Incomplete {method} preflight')
                initial_hashes.append(result['initial_actor_sha256'])
            if initial_hashes[0] != initial_hashes[1]:
                raise AssertionError('Methods did not start from identical actors.')
    print('Preflight passed for MAPPO and OCPO; temporary files removed.', flush=True)


def run():
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = acquire_batch_lock(OUTPUT)
    try:
        with adapted_runner():
            write_json(OUTPUT / 'status.json', dict(status='running', method='mappo', seed=42,
                       episodes=1500, started_at_utc=now(), final_test=False))
            result = runner.run_training('mappo', 42, DATA, OUTPUT, config())
        write_json(OUTPUT / 'status.json', dict(status='completed', method='mappo', seed=42,
                   episodes=1500, finished_at_utc=now(), final_test=False,
                   selected_episode=result['selected_episode']))
    except Exception as exc:
        write_json(OUTPUT / 'status.json', dict(status='failed', error=str(exc),
                   traceback=traceback.format_exc(), final_test=False))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check','run'))
    args = parser.parse_args()
    check() if args.mode == 'check' else run()
