"""Path, record and completed-run checks shared by ablation code."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .run import HERE, sha


def _inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError('Confirmation data and outputs must stay under major/cross_scene.')
    return path


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _completed_run(data, output, config, protocol, method, seed):
    folder = output / f'{method}_seed_{seed}'
    record = _read(folder / 'run.json')
    required = dict(method=method, seed=seed, status='completed', config=config,
                    data_manifest_sha256=protocol['data_manifest_sha256'],
                    code_sha256=protocol['code_sha256'], completed_episodes=config['episodes'],
                    completed_updates=(config['episodes'] + config['layouts_per_update'] - 1)
                                      // config['layouts_per_update'],
                    optimizer_steps=config['episodes'] * config['epochs'],
                    joint_steps=config['episodes'] * config['episode_steps'])
    if any(record.get(k) != value for k, value in required.items()):
        raise ValueError(f'Run differs from the fixed confirmation configuration: {folder.name}')
    histories = _rows(folder / 'training.csv')
    validation = _rows(folder / 'validation.csv')
    if (len(histories) != config['episodes']
            or [int(r['episode']) for r in histories] != list(range(1, config['episodes'] + 1))):
        raise ValueError('Incomplete training history.')
    best = max((r for r in validation if int(r['episode']) > 0), key=lambda r: float(r['reward']))
    if (float(best['reward']) != record['best_validation_reward']
            or int(best['episode']) != record['selected_episode']
            or sha(folder / 'best.pt') != record['selected_checkpoint_sha256']):
        raise ValueError('Validation selection or selected checkpoint changed.')
    train_conditions = [tuple(r[k] for k in ('episode', 'scene_id', 'channel_seed')) for r in histories]
    val_conditions = [tuple(r[k] for k in ('scene_id', 'replicate', 'channel_seed', 'channel_sha256', 'steps'))
                      for r in _rows(folder / 'validation_channels.csv')]
    return record, train_conditions, val_conditions
