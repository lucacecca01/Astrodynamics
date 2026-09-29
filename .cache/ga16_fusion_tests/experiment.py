import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

ROOT = Path('/home/lucacecca/Astrodynamics')
HERE = Path(__file__).resolve().parent
OLD = ROOT / '.cache/ga16_monotonic_tests'
CACHE = ROOT / '.cache/ga16_sigma_recovery_tests'
spec = importlib.util.spec_from_file_location('previous_test', OLD / 'experiment.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
app = previous.app
LIMITS = np.array([.04, .05, 8.])
CAP = 1003


def groups_from_labels(labels):
    valid = np.flatnonzero(labels >= 0)
    order = valid[np.argsort(labels[valid], kind='stable')]
    cuts = np.flatnonzero(np.diff(labels[order])) + 1
    return np.split(order, cuts) if len(order) else []


class Pool:
    def __init__(self, p, f):
        self.p, self.f = p, f
        self.groups, self.stds, self.radial = [], [], []
        self.lookup = {}
        self.partitions = []

    def add_group(self, ids):
        ids = np.sort(np.asarray(ids, dtype=np.int32))
        if len(ids) < 25:
            return None
        key = ids.tobytes()
        if key in self.lookup:
            return self.lookup[key]
        assert len(np.unique(ids)) == len(ids)
        assert len(np.unique(self.p[ids, 6])) == 1
        std = app.physical_std(self.p[ids, :3])
        x = self.f[ids]
        distances = ((x - x.mean(axis=0))**2).sum(axis=1)
        variance = x.var(axis=0).sum()
        ratio = float(distances.max() / variance) if variance > 0 else 0.
        index = len(self.groups)
        self.lookup[key] = index
        self.groups.append(ids)
        self.stds.append(std)
        self.radial.append(ratio)
        return index

    def add_partition(self, name, labels):
        groups = [self.add_group(ids) for ids in groups_from_labels(labels)]
        self.partitions.append((name, [g for g in groups if g is not None]))

    def valid_indices(self, sigma, minimum, limits=LIMITS):
        sizes = np.array([len(ids) for ids in self.groups])
        return np.flatnonzero((sizes >= minimum) &
                              np.all(np.asarray(self.stds) < limits, axis=1) &
                              (np.asarray(self.radial) <= sigma**2))

    def save(self, path, source_ids):
        np.savez_compressed(path, members=np.concatenate(self.groups),
                            offsets=np.r_[0, np.cumsum([len(g) for g in self.groups])],
                            stds=self.stds, radial=self.radial, source_ids=source_ids,
                            partitions=json.dumps(self.partitions))

    @classmethod
    def load(cls, path, p, f, source_ids):
        with np.load(path) as data:
            np.testing.assert_array_equal(data['source_ids'], source_ids)
            pool = cls(p, f)
            pool.groups = list(np.split(data['members'], data['offsets'][1:-1]))
            pool.stds, pool.radial = list(data['stds']), list(data['radial'])
            pool.lookup = {ids.tobytes(): i for i, ids in enumerate(pool.groups)}
            pool.partitions = json.loads(str(data['partitions']))
        return pool


def incidence_matrix(groups, n):
    rows = np.concatenate(groups)
    cols = np.repeat(np.arange(len(groups), dtype=np.int32), [len(g) for g in groups])
    return coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(groups))).tocsr()


def pack(groups, required, cap, fallback, *, fixed=(), seconds=30):
    n = len(required)
    assert len(groups) > 0
    fallback = np.asarray(fallback, dtype=int)
    a = incidence_matrix(groups, n)
    fallback_cover = np.asarray(a[:, fallback].sum(axis=1)).ravel()
    assert np.all(fallback_cover <= 1) and np.all(fallback_cover[required] == 1)
    assert len(fallback) <= cap and set(fixed) <= set(fallback)
    sizes = np.array([len(g) for g in groups])
    gains = np.array([np.count_nonzero(~required[g]) for g in groups])
    # A one-point gain dominates every possible difference in cluster count.
    cost = 1. - (cap + 1.) * gains
    fallback_score = float(cost[fallback].sum())

    # Identical incidence rows express the same constraint; preserve the
    # strongest lower bound when a protected and an optional point coincide.
    patterns, unique_rows, lower = {}, [], []
    for i in range(n):
        ids = a.indices[a.indptr[i]:a.indptr[i + 1]]
        if len(ids) == 0:
            assert not required[i]
            continue
        key = ids.tobytes()
        if key not in patterns:
            patterns[key] = len(unique_rows)
            unique_rows.append(i)
            lower.append(float(required[i]))
        elif required[i]:
            lower[patterns[key]] = 1.
    reduced = a[unique_rows]
    matrix = vstack((reduced, csr_matrix(np.ones((1, len(groups)))))).tocsc()
    lb = np.r_[lower, 0.]
    ub = np.r_[np.ones(len(lower)), cap]
    variable_lb = np.zeros(len(groups))
    variable_lb[list(fixed)] = 1.
    start = time.perf_counter()
    result = milp(cost, integrality=np.ones(len(groups)), bounds=Bounds(variable_lb, 1.),
                  constraints=LinearConstraint(matrix, lb, ub),
                  options={'time_limit': seconds, 'mip_rel_gap': 0.})
    elapsed = time.perf_counter() - start
    chosen, used_fallback = fallback, True
    if result.x is not None:
        proposed = np.flatnonzero(result.x > .5)
        cover = np.asarray(a[:, proposed].sum(axis=1)).ravel()
        if (np.all(cover <= 1) and np.all(cover[required] == 1) and
                len(proposed) <= cap and set(fixed) <= set(proposed) and
                cost[proposed].sum() <= fallback_score):
            chosen, used_fallback = proposed, False
    cover = np.asarray(a[:, chosen].sum(axis=1)).ravel()
    assert np.all(cover <= 1) and np.all(cover[required] == 1)
    assert cost[chosen].sum() <= fallback_score
    labels = np.full(n, -1, dtype=np.int32)
    for c, j in enumerate(chosen):
        labels[groups[j]] = c
    return labels, dict(solver_seconds=elapsed, solver_status=int(result.status),
                        solver_message=str(result.message),
                        gap=float(result.mip_gap) if getattr(result, 'mip_gap', None) is not None else None,
                        used_fallback=used_fallback, candidates=len(groups),
                        constraint_patterns=len(unique_rows),
                        candidate_uncovered=int(np.sum(np.diff(a.indptr) == 0)),
                        selected_indices=chosen.tolist())


def audit(labels, pool, sigma, minimum, protected):
    sizes, stds, sigmas = [], [], []
    assert np.count_nonzero(protected & (labels < 0)) == 0
    groups = groups_from_labels(labels)
    assert len(groups) <= CAP
    for ids in groups:
        assert len(ids) >= minimum
        assert len(np.unique(pool.p[ids, 6])) == 1
        x = pool.p[ids, :3].copy()
        angle = np.deg2rad(x[:, 2])
        center = np.rad2deg(np.arctan2(np.sin(angle).sum(), np.cos(angle).sum()))
        x[:, 2] = (x[:, 2] - center + 180) % 360 - 180
        std = x.std(axis=0)
        assert np.all(std < LIMITS)
        f = pool.f[ids]
        distance = ((f - f.mean(axis=0))**2).sum(axis=1)
        variance = f.var(axis=0).sum()
        assert np.all(distance <= sigma**2 * variance)
        sigmas.append(float(np.sqrt(distance.max() / variance)) if variance > 0 else 0.)
        sizes.append(len(ids))
        stds.append(std)
    return dict(clusters=len(groups), unassigned=int(np.sum(labels < 0)),
                minimum_size=min(sizes, default=0), lost=int(np.sum(protected & (labels < 0))),
                recovered=int(np.sum(~protected & (labels >= 0))),
                mean_std=np.mean(stds, axis=0).tolist(),
                weighted_std=np.average(stds, axis=0, weights=sizes).tolist(),
                maximum_std=np.max(stds, axis=0).tolist(), largest_radial_sigma=max(sigmas))


def unit_tests():
    # Overlap is not permission to duplicate points; keep every protected point.
    groups = [np.arange(20), np.arange(10, 30), np.arange(20, 40)]
    required = np.arange(40) < 20
    labels, info = pack(groups, required, 2, [0], seconds=5)
    assert np.all(labels >= 0) and info['solver_status'] == 0
    labels, info = pack(groups[:2], required, 2, [0], seconds=5)
    assert np.all(labels[:20] >= 0) and np.all(labels[20:] < 0)
    # Fewer groups wins only when coverage is equal; cap does not drop incumbents.
    groups.append(np.arange(40))
    labels, info = pack(groups, required, 1, [0], seconds=5)
    assert np.all(labels == 0) and info['selected_indices'] == [3]
    labels, _ = pack(groups, required, 2, [0], fixed=[0], seconds=5)
    assert len(np.unique(labels)) == 2 and np.all(labels >= 0)
    # No solver incumbent: the feasible reference is retained.
    labels, info = pack(groups, required, 2, [0], seconds=0)
    assert np.all(labels[required] >= 0)
    print('UNIT TESTS PASS', flush=True)


def load_data():
    data = np.load(CACHE / 'features.npz')
    p, source_ids = data['features'], data['source_ids']
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100 * app.normalize_block(p[:, 6:7])))
    return p, f, source_ids


def prepare(p, f, source_ids):
    path = HERE / 'pool_cached.npz'
    if path.exists():
        return Pool.load(path, p, f, source_ids)
    pool = Pool(p, f)
    saved = np.load(OLD / 'all_labels.npz')
    np.testing.assert_array_equal(saved['source_ids'], source_ids)
    for i, labels in enumerate(saved['labels']):
        pool.add_partition(f'fixed_{i}', labels)
    print(f'CACHED fixed: {len(pool.groups)} distinct clusters', flush=True)
    for path in sorted(CACHE.glob('**/final_labels.npz')):
        data = np.load(path)
        np.testing.assert_array_equal(data['source_ids'], source_ids)
        for key in ('initial_labels', 'labels'):
            if key in data:
                pool.add_partition(f'{path.parent.name}_{key}', data[key])
    pool.save(HERE / 'pool_cached.npz', source_ids)
    print(f'CACHED complete: {len(pool.groups)} distinct clusters, '
          f'{len(pool.partitions)} partitions', flush=True)
    return pool


def baseline(sigma, minimum):
    with (OLD / 'cuts.csv').open() as stream:
        row = next(row for row in csv.DictReader(stream) if row['profile'] == 'current' and
                   float(row['max_sigma']) == sigma and int(row['minpoints']) == minimum)
    data = np.load(OLD / 'all_labels.npz')
    return data['labels'][int(row['case'])]


def run_case(pool, name, sigma, minimum, reference, *, freeze=False, seconds=30):
    start = time.perf_counter()
    valid = pool.valid_indices(sigma, minimum)
    groups = [pool.groups[i] for i in valid]
    local = {int(global_id): i for i, global_id in enumerate(valid)}
    fallback = [local[pool.lookup[np.asarray(ids, dtype=np.int32).tobytes()]]
                for ids in groups_from_labels(reference)]
    protected = reference >= 0
    labels, info = pack(groups, protected, CAP, fallback,
                        fixed=fallback if freeze else (), seconds=seconds)
    metrics = audit(labels, pool, sigma, minimum, protected)
    base_metrics = audit(reference, pool, sigma, minimum, protected)
    best = None
    valid_set = set(valid)
    for source, members in pool.partitions:
        selected = [g for g in members if g in valid_set]
        if len(selected) > CAP:
            continue
        score = (sum(len(pool.groups[g]) for g in selected), -len(selected))
        if best is None or score > best[0]:
            best = (score, source)
    info.pop('selected_indices')
    row = dict(name=name, sigma=sigma, minimum=minimum, freeze=freeze,
               baseline_clusters=base_metrics['clusters'], baseline_unassigned=base_metrics['unassigned'],
               best_single_unassigned=len(reference)-best[0][0], best_single_source=best[1],
               **metrics, **info, total_seconds=time.perf_counter()-start)
    np.savez_compressed(HERE / f'{name}.npz', labels=labels)
    (HERE / f'{name}.json').write_text(json.dumps(row, indent=2))
    print('RESULT', json.dumps(row), flush=True)
    return row, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()
    production_hash = hashlib.sha256(previous.SOURCE.read_bytes()).hexdigest()
    unit_tests()
    p, f, source_ids = load_data()
    pool = prepare(p, f, source_ids)
    cases = [(2., 50)] if not args.sweep else [(s, m) for s in (2., 3., 4.) for m in (25, 50, 101)]
    rows = []
    for sigma, minimum in cases:
        reference = baseline(sigma, minimum)
        row, _ = run_case(pool, f'cached_sigma_{sigma:g}_min_{minimum}', sigma, minimum,
                          reference, seconds=args.seconds)
        rows.append(row)
    if not args.sweep:
        row, _ = run_case(pool, 'cached_sigma_2_min_50_frozen', 2., 50, baseline(2., 50),
                          freeze=True, seconds=args.seconds)
        rows.append(row)
    with (HERE / ('comparison_sweep.csv' if args.sweep else 'comparison.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert production_hash == hashlib.sha256(previous.SOURCE.read_bytes()).hexdigest()
    print('DONE: production file unchanged', flush=True)


if __name__ == '__main__':
    main()
