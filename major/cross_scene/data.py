"""Vehicle layout and geometry utilities."""








from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'data'
DEFAULT_MASTER_SEED = 20261025
ARRAY_KEYS = ('positions', 'is_cav', 'vehicle_ids', 'los_matrix', 'comm_los_matrix')
SPLIT_NAMES = ('train', 'validation', 'test')
SPECIFICATION = {
    'road_length_m': 1000.0,
    'road_width_m': 15.0,
    'lanes': 4,
    'lane_width_m': 3.75,
    'vehicle_length_m': 5.2,
    'vehicle_width_m': 1.95,
    'minimum_center_gap_m': 10.2,
    'density_vehicles_per_lane_km': 50.0,
    'vehicles_per_lane': 50,
    'vehicle_count': 200,
    'cav_fraction': 0.5,
    'cavs_per_lane': 25,
    'cav_count': 100,
    'perception_range_m': 150.0,
    'communication_los_range_m': 300.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derived_seed(master_seed: int, namespace: int, index: int = 0) -> int:
    state = np.random.SeedSequence([master_seed, namespace, index]).generate_state(2)
    return int(state[0]) | (int(state[1]) << 32)


def _roles(master_seed: int) -> tuple[np.ndarray, int]:
    seed = _derived_seed(master_seed, 0x434156)
    rng = np.random.default_rng(seed)
    labels = np.zeros(SPECIFICATION['vehicle_count'], dtype=np.uint8)
    count = SPECIFICATION['vehicles_per_lane']
    for lane in range(SPECIFICATION['lanes']):
        selected = rng.choice(count, SPECIFICATION['cavs_per_lane'], replace=False)
        labels[lane * count + selected] = 1
    return labels, seed


def _positions(layout_seed: int) -> np.ndarray:
    rng = np.random.default_rng(layout_seed)
    count = SPECIFICATION['vehicles_per_lane']
    minimum = SPECIFICATION['minimum_center_gap_m']
    margin = SPECIFICATION['vehicle_length_m'] / 2
    free = SPECIFICATION['road_length_m'] - 2 * margin - (count - 1) * minimum
    positions = []
    for lane in range(SPECIFICATION['lanes']):
        gaps = rng.dirichlet(np.ones(count + 1)) * free
        x = margin + np.arange(count) * minimum + np.cumsum(gaps[:count])
        y = np.full(count, (lane + 0.5) * SPECIFICATION['lane_width_m'])
        positions.append(np.column_stack((x, y)))
    return np.concatenate(positions).astype(np.float64)


def _geometric_los(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:






    count = len(positions)
    delta = positions[:, None] - positions[None]
    distance = np.linalg.norm(delta, axis=-1)
    first, second = np.where(np.triu(distance <= SPECIFICATION['communication_los_range_m'], 1))
    comm = np.zeros((count, count), dtype=np.uint8)
    half = np.array([SPECIFICATION['vehicle_length_m'], SPECIFICATION['vehicle_width_m']]) / 2
    lower, upper = positions - half, positions + half
    obstacle_ids = np.arange(count)[None]
    for start in range(0, len(first), 256):
        a, b = first[start:start + 256], second[start:start + 256]
        source = positions[a, None]
        direction = (positions[b] - positions[a])[:, None]
        near = np.full((len(a), count, 2), -np.inf)
        far = np.full((len(a), count, 2), np.inf)
        moving = direction != 0
        t1 = np.zeros_like(near)
        t2 = np.zeros_like(far)
        np.divide(lower[None] - source, direction, out=t1, where=moving)
        np.divide(upper[None] - source, direction, out=t2, where=moving)
        near = np.where(moving, np.minimum(t1, t2), near)
        far = np.where(moving, np.maximum(t1, t2), far)
        outside_parallel = (~moving) & ((source < lower[None]) | (source > upper[None]))
        near = np.where(outside_parallel, np.inf, near)
        far = np.where(outside_parallel, -np.inf, far)
        entry = np.maximum(near.max(axis=-1), 0.0)
        exit = np.minimum(far.min(axis=-1), 1.0)
        intersects = entry <= exit
        intersects &= (obstacle_ids != a[:, None]) & (obstacle_ids != b[:, None])
        visible = ~intersects.any(axis=1)
        comm[a, b] = visible
        comm[b, a] = visible
    perception = comm * (distance <= SPECIFICATION['perception_range_m'])
    return perception.astype(np.uint8), comm


def _canonical_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        'positions': np.asarray(arrays['positions'], dtype='<f8'),
        'is_cav': np.asarray(arrays['is_cav'], dtype=np.uint8).reshape(-1, 1),
        'vehicle_ids': np.asarray(arrays['vehicle_ids'], dtype='<i8').reshape(-1, 1),
        'los_matrix': np.asarray(arrays['los_matrix'], dtype=np.uint8),
        'comm_los_matrix': np.asarray(arrays['comm_los_matrix'], dtype=np.uint8),
    }


def _array_hash(arrays: dict[str, np.ndarray], keys: tuple[str, ...] = ARRAY_KEYS) -> str:
    digest = hashlib.sha256()
    for key in keys:
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode('ascii'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(value.dtype.str.encode('ascii'))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _save_mat(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix('.pending.mat')
    savemat(temporary, arrays, do_compression=True, oned_as='column')
    
    
    header = b'MATLAB 5.0 MAT-file, cross_scene hard-core fixed-count bank, schema 1'
    with temporary.open('r+b') as handle:
        handle.write(header.ljust(116, b' '))
    temporary.replace(path)


def _validate_scene(arrays: dict[str, np.ndarray], roles: np.ndarray) -> None:
    spec = SPECIFICATION
    positions = arrays['positions']
    labels = arrays['is_cav'].reshape(-1)
    count = spec['vehicle_count']
    if positions.shape != (count, 2) or not np.isfinite(positions).all():
        raise ValueError('Invalid positions in scene.')
    if not np.array_equal(labels, roles):
        raise ValueError('CAV role assignments differ from the bank protocol.')
    if not np.array_equal(arrays['vehicle_ids'].reshape(-1), np.arange(1, count + 1)):
        raise ValueError('Global IDs must follow lane, then longitudinal position.')
    lane_count = spec['vehicles_per_lane']
    margin = spec['vehicle_length_m'] / 2
    if np.any(positions[:, 0] < margin) or np.any(positions[:, 0] > spec['road_length_m'] - margin):
        raise ValueError('Vehicle footprint crosses a road endpoint.')
    for lane in range(spec['lanes']):
        section = slice(lane * lane_count, (lane + 1) * lane_count)
        if not np.all(positions[section, 1] == (lane + 0.5) * spec['lane_width_m']):
            raise ValueError('Non-canonical lane assignment.')
        if np.any(np.diff(positions[section, 0]) < spec['minimum_center_gap_m'] - 1e-10):
            raise ValueError('Minimum lane spacing violated.')
        if int(labels[section].sum()) != spec['cavs_per_lane']:
            raise ValueError('Incorrect per-lane CAV count.')
    distance = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    for key, radius in (('los_matrix', spec['perception_range_m']),
                        ('comm_los_matrix', spec['communication_los_range_m'])):
        matrix = arrays[key]
        if matrix.shape != (count, count) or not np.isin(matrix, (0, 1)).all():
            raise ValueError(f'Invalid binary {key}.')
        if matrix.diagonal().any() or not np.array_equal(matrix, matrix.T):
            raise ValueError(f'{key} must be symmetric with zero diagonal.')
        if np.any(matrix[distance > radius]):
            raise ValueError(f'{key} exceeds its specified geometric range.')
    if np.any(arrays['los_matrix'] > arrays['comm_los_matrix']):
        raise ValueError('Perception LOS is not a subset of communication LOS.')


def load_manifest(root: str | Path = DEFAULT_OUTPUT, *, verify: bool = True) -> dict[str, Any]:

    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('specification') != SPECIFICATION:
        raise ValueError('Bank schema/specification does not match this data protocol.')
    if verify:
        _validate_manifest(root, manifest)
    return manifest


def _validate_manifest(root: Path, manifest: dict[str, Any]) -> None:
    roles, role_seed = _roles(int(manifest['master_seed']))
    if manifest.get('cav_role_seed') != role_seed:
        raise ValueError('Invalid CAV-role seed.')
    if manifest.get('cav_roles_sha256') != hashlib.sha256(roles.tobytes()).hexdigest():
        raise ValueError('Invalid shared CAV-role hash.')
    scene_hashes: set[str] = set()
    layout_seeds: set[int] = set()
    for split_index, split in enumerate(SPLIT_NAMES, start=1):
        records = manifest['splits'][split]
        if len(records) != manifest['split_counts'][split]:
            raise ValueError(f'Incorrect {split} scene count.')
        for index, record in enumerate(records):
            expected_seed = _derived_seed(int(manifest['master_seed']), split_index, index)
            if record['layout_seed'] != expected_seed or expected_seed in layout_seeds:
                raise ValueError('Invalid or repeated layout RNG seed.')
            path = (root / record['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('MAT path escapes the dataset root or does not exist.')
            if _sha256(path) != record['file_sha256']:
                raise ValueError(f'MAT byte hash mismatch: {record["path"]}')
            raw = loadmat(path)
            arrays = _canonical_arrays(raw)
            _validate_scene(arrays, roles)
            scene_hash = _array_hash(arrays, ('positions', 'is_cav'))
            if _array_hash(arrays) != record['content_sha256'] or scene_hash != record['scene_sha256']:
                raise ValueError(f'MAT content hash mismatch: {record["path"]}')
            if scene_hash in scene_hashes:
                raise ValueError('A complete layout is duplicated within/across splits.')
            scene_hashes.add(scene_hash)
            layout_seeds.add(expected_seed)


def validate_bank(root: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:

    return load_manifest(root, verify=True)


def generate_bank(root: str | Path = DEFAULT_OUTPUT, *, train_count: int = 64,
                  validation_count: int = 16, test_count: int = 32,
                  master_seed: int = DEFAULT_MASTER_SEED) -> dict[str, Any]:





    counts = dict(zip(SPLIT_NAMES, (train_count, validation_count, test_count)))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts.values()):
        raise ValueError('Each split must contain a positive integer scene count.')
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError('master_seed must be a nonnegative integer.')
    root = Path(root).resolve()
    if (root / 'manifest.json').exists():
        manifest = load_manifest(root, verify=True)
        if manifest['master_seed'] != master_seed or manifest['split_counts'] != counts:
            raise ValueError('Refusing to overwrite a bank with mismatched seeds/split counts.')
        return manifest
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty/incomplete dataset directory.')
    root.mkdir(parents=True, exist_ok=True)
    roles, role_seed = _roles(master_seed)
    manifest: dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'master_seed': master_seed,
        'specification': SPECIFICATION.copy(),
        'split_counts': counts,
        'generator': 'Fixed-count hard-core layouts; independent Dirichlet free spacings per lane.',
        'point_process': 'Constrained fixed-count process; not a Poisson point process.',
        'cav_role_semantics': 'Fixed lane/rank CAV labels, chosen once independently of layout positions.',
        'cav_role_seed': role_seed,
        'cav_roles_sha256': hashlib.sha256(roles.tobytes()).hexdigest(),
        'id_semantics': 'MAT vehicle_ids are one-based; array rows sort lane, then x.',
        'geometry_semantics': 'Closed axis-aligned vehicle rectangles; source and receiver do not occlude themselves; tangencies block LOS.',
        'split_semantics': 'Whole independent layouts; no scene reused between train, validation and test.',
        'mat_arrays': list(ARRAY_KEYS),
        'generator_source_sha256': _sha256(Path(__file__)),
        'splits': {name: [] for name in SPLIT_NAMES},
    }
    seen: set[str] = set()
    for split_index, split in enumerate(SPLIT_NAMES, start=1):
        folder = root / split
        folder.mkdir()
        for index in range(counts[split]):
            seed = _derived_seed(master_seed, split_index, index)
            positions = _positions(seed)
            perception, comm = _geometric_los(positions)
            arrays = _canonical_arrays(dict(
                positions=positions, is_cav=roles,
                vehicle_ids=np.arange(1, SPECIFICATION['vehicle_count'] + 1),
                los_matrix=perception, comm_los_matrix=comm))
            _validate_scene(arrays, roles)
            scene_hash = _array_hash(arrays, ('positions', 'is_cav'))
            if scene_hash in seen:
                raise RuntimeError('Independent generation produced a repeated layout.')
            seen.add(scene_hash)
            path = folder / f'layout_{index:03d}.mat'
            _save_mat(path, arrays)
            manifest['splits'][split].append({
                'scene_id': f'{split}_{index:03d}',
                'layout_seed': seed,
                'path': path.relative_to(root).as_posix(),
                'file_sha256': _sha256(path),
                'content_sha256': _array_hash(arrays),
                'scene_sha256': scene_hash,
            })
        print(f'Generated {split}: {counts[split]} independent layouts.', flush=True)
    _validate_manifest(root, manifest)
    temporary = root / 'manifest.pending.json'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(root / 'manifest.json')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--train-count', type=int, default=64)
    parser.add_argument('--validation-count', type=int, default=16)
    parser.add_argument('--test-count', type=int, default=32)
    parser.add_argument('--master-seed', type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        manifest = validate_bank(args.output)
    else:
        manifest = generate_bank(args.output, train_count=args.train_count,
                                 validation_count=args.validation_count,
                                 test_count=args.test_count, master_seed=args.master_seed)
    print(json.dumps({'output': str(args.output.resolve()),
                      'split_counts': manifest['split_counts'],
                      'master_seed': manifest['master_seed']}, indent=2))


if __name__ == '__main__':
    main()
