import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT = Path('/home/lucacecca/Astrodynamics')
SOURCE = ROOT / 'CR3BP/Lunar Swingby/Clustering/CR3BP_prova_GA_16.py'
HERE = Path(__file__).resolve().parent
CACHE = ROOT / '.cache/ga16_sigma_recovery_tests'
sys.path.insert(0, str(ROOT))
tree = ast.parse(SOURCE.read_text())
cutoff = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'database' for t in node.targets))
app = types.ModuleType('ga16_monotonic_app')
app.__file__ = str(SOURCE)
exec(compile(ast.Module(body=tree.body[:cutoff], type_ignores=[]), str(SOURCE), 'exec'), app.__dict__)

FIXED_SCALE = np.array([.04, .05, 8.])
ANCHOR_SIGMA = 2.
SIGMAS = [2., 2.5, 3., 3.5, 4.]
MINIMA = [101, 50, 25, 12, 6, 3, 1]
PROFILES = [
    ('strict', [.03, .04, 6.]),
    ('strict_plus10', [.033, .044, 6.6]),
    ('strict_plus20', [.036, .048, 7.2]),
    ('current', [.04, .05, 8.]),
    ('current_plus10', [.044, .055, 8.8]),
    ('current_plus20', [.048, .06, 9.6]),
]


def function_signature():
    names = {'normalize_block', 'physical_std', 'within_sigma',
             'constrained_clustering', 'split_messy_clusters'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[])).encode()).hexdigest()


def canonical(labels):
    ids = np.unique(labels[labels >= 0])
    ids = sorted(ids, key=lambda c: int(np.flatnonzero(labels == c)[0]))
    result = np.full(len(labels), -1, dtype=np.int32)
    for new, old in enumerate(ids):
        result[labels == old] = new
    return result


class FixedHierarchy:
    def __init__(self, roots, physical, features, directions, scale=FIXED_SCALE):
        self.p = np.array(physical, dtype=float, copy=True)
        self.f = np.array(features, dtype=float, copy=True)
        self.directions = np.asarray(directions)
        roots = np.asarray(roots)
        scale = np.array(scale, dtype=float, copy=True)
        if (self.p.ndim != 2 or self.p.shape[1] != 3 or
                self.f.ndim != 2 or len(self.f) != len(self.p) or
                roots.shape != (len(self.p),) or not np.issubdtype(roots.dtype, np.integer) or
                np.any(roots < 0) or self.directions.shape != roots.shape or
                not np.all(np.isin(self.directions, [-1, 1])) or
                not np.all(np.isfinite(self.p)) or not np.all(np.isfinite(self.f)) or
                scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
            raise ValueError('Invalid fixed hierarchy input.')
        angles = np.deg2rad(self.p[:, 2])
        self.coordinates = np.column_stack((self.p[:, :2]/scale[:2],
                                           np.cos(angles)/np.deg2rad(scale[2]),
                                           np.sin(angles)/np.deg2rad(scale[2])))
        self.nodes = []
        self.independent_stats = {}
        self.roots = []
        self.fits = 0
        for c in np.unique(roots):
            ids = np.flatnonzero(roots == c)
            if len(np.unique(self.directions[ids])) != 1:
                raise ValueError('A root mixes orbital directions.')
            self.roots.append(self.make_node(ids))

    def make_node(self, ids):
        x = self.f[ids]
        squared = np.sum((x-x.mean(axis=0))**2, axis=1)
        node = dict(ids=ids, std=app.physical_std(self.p[ids]),
                    max_squared=float(squared.max()), variance=float(np.var(x, axis=0).sum()),
                    children=None)
        self.nodes.append(node)
        return len(self.nodes)-1

    def children(self, index):
        node = self.nodes[index]
        if node['children'] is None:
            ids = node['ids']
            model = app.KMeans(n_clusters=2, n_init=5, random_state=42,
                               max_iter=500, tol=1e-9, algorithm='lloyd')
            sides = model.fit_predict(self.coordinates[ids])
            if len(np.unique(sides)) != 2:
                raise RuntimeError('Invalid group cannot be subdivided in the fixed metric.')
            groups = sorted((ids[sides == side] for side in (0, 1)), key=lambda x: int(x[0]))
            node['children'] = tuple(self.make_node(group) for group in groups)
            self.fits += 1
        return node['children']

    def select(self, limits, sigma, minimum, maximum):
        limits = np.asarray(limits, dtype=float)
        if limits.shape != (3,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
            raise ValueError('Invalid STD limits.')
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Invalid sigma.')
        if not isinstance(minimum, (int, np.integer)) or minimum < 1:
            raise ValueError('Invalid minimum size.')
        if not isinstance(maximum, (int, np.integer)) or maximum < 0:
            raise ValueError('Invalid cluster cap.')
        selected = []
        pending = list(reversed(self.roots))
        with app.threadpool_limits(limits=2):
            while pending:
                index = pending.pop()
                node = self.nodes[index]
                if len(node['ids']) < minimum:
                    continue
                if np.all(node['std'] < limits) and node['max_squared'] <= sigma**2*node['variance']:
                    selected.append(index)
                else:
                    pending.extend(reversed(self.children(index)))
        if len(selected) > maximum:
            raise ValueError(f'The selected partition requires {len(selected)} clusters; '
                             f'cap is {maximum}. No groups were silently dropped.')
        selected.sort(key=lambda i: int(self.nodes[i]['ids'][0]))
        labels = np.full(len(self.p), -1, dtype=np.int32)
        for c, index in enumerate(selected):
            labels[self.nodes[index]['ids']] = c
        return labels, selected

    def audit(self, labels, selected, limits, sigma, minimum, maximum):
        assert len(selected) <= maximum
        counts, stds, representatives = [], [], []
        seen = np.zeros(len(labels), dtype=np.int8)
        for c, index in enumerate(selected):
            node = self.nodes[index]
            ids = node['ids']
            assert len(ids) >= minimum
            assert np.all(labels[ids] == c)
            seen[ids] += 1
            if index not in self.independent_stats:
                physical = self.p[ids].copy()
                angle = np.deg2rad(physical[:, 2])
                origin = np.rad2deg(np.arctan2(np.sin(angle).sum(), np.cos(angle).sum()))
                physical[:, 2] = (physical[:, 2]-origin+180) % 360 - 180
                std = physical.std(axis=0)
                np.testing.assert_allclose(std, node['std'], atol=1e-12, rtol=1e-12)
                x = self.f[ids]
                distance_squared = np.sum((x-x.mean(axis=0))**2, axis=1)
                variance = np.var(x, axis=0).sum()
                assert len(np.unique(self.directions[ids])) == 1
                self.independent_stats[index] = (std, distance_squared.max(), variance)
            std, farthest, variance = self.independent_stats[index]
            assert np.all(std < limits)
            assert farthest <= sigma**2*variance
            counts.append(len(ids))
            stds.append(std)
            representatives.append(int(ids[0]))
        assert np.all(seen <= 1)
        np.testing.assert_array_equal(seen > 0, labels >= 0)
        np.testing.assert_array_equal(np.unique(labels[labels >= 0]), np.arange(len(selected)))
        return dict(clusters=len(selected), unassigned=int(np.sum(labels < 0)),
                    assigned=int(np.sum(labels >= 0)), minimum_size=min(counts, default=0),
                    singletons=int(np.sum(np.asarray(counts) == 1)),
                    weighted_std=np.average(stds, axis=0, weights=counts).tolist() if counts else [],
                    maximum_std=np.max(stds, axis=0).tolist() if counts else []), np.array(representatives, dtype=np.intp)


def unit_tests():
    # Circular wrap, constant variables, singletons, small groups and mixed directions.
    p = np.array([[0., 0., 359.], [0., 0., 1.], [1., 1., 180.], [1., 1., 180.]])
    f = np.column_stack((p[:, :2], np.cos(np.deg2rad(p[:, 2])), np.sin(np.deg2rad(p[:, 2]))))
    model = FixedHierarchy(np.array([9, 9, 20, 20]), p, f, np.array([1, 1, -1, -1]))
    labels, nodes = model.select([.1, .1, 2.], 2., 2, 2)
    assert np.all(labels >= 0) and len(nodes) == 2
    model.audit(labels, nodes, [.1, .1, 2.], 2., 2, 2)
    noise, noise_nodes = model.select([.1, .1, 2.], 2., 3, 2)
    assert noise.tolist() == [-1]*4
    _, representatives = model.audit(noise, noise_nodes, [.1, .1, 2.], 2., 3, 2)
    assert representatives.dtype == np.dtype(np.intp)
    try:
        model.select([.1, .1, 2.], 2., 2, 1)
    except ValueError as exc:
        assert 'No groups were silently dropped' in str(exc)
    else:
        raise AssertionError('Cluster cap silently removed valid groups.')
    np.testing.assert_array_equal(model.select([.1, .1, 2.], 2., 2, 2)[0], labels)
    empty = FixedHierarchy(np.empty(0, dtype=int), np.empty((0, 3)), np.empty((0, 4)), np.empty(0))
    assert empty.select([1, 1, 1], 2., 1, 0)[0].size == 0
    empty.audit(np.empty(0, dtype=np.int32), [], [1, 1, 1], 2., 1, 0)
    one = FixedHierarchy(np.array([4]), p[:1], f[:1], np.array([1]))
    assert one.select([1e-6]*3, .01, 1, 1)[0].tolist() == [0]
    duplicate = FixedHierarchy(np.zeros(4, dtype=int), np.repeat(p[2:3], 4, axis=0),
                               np.repeat(f[2:3], 4, axis=0), np.ones(4))
    assert duplicate.select([1e-6]*3, 2., 1, 1)[0].tolist() == [0]*4
    boundary = FixedHierarchy(np.zeros(2, dtype=int), p[:2], f[:2], np.ones(2))
    angle_std = float(app.physical_std(p[:2])[2])
    assert len(boundary.select([.1, .1, angle_std], 2., 1, 2)[1]) == 2
    assert len(boundary.select([.1, .1, np.nextafter(angle_std, np.inf)], 2., 1, 2)[1]) == 1
    radial_p = np.array([[-1., .1, 359.], [1., .1, 1.]])
    radial = FixedHierarchy(np.zeros(2, dtype=int), radial_p,
                            np.array([[-1., 0.], [1., 0.]]), np.ones(2))
    assert np.all(radial.select([2., 1., 2.], .999, 2, 2)[0] == -1)
    assert np.all(radial.select([2., 1., 2.], 1., 2, 2)[0] == 0)
    try:
        FixedHierarchy(np.zeros(4, dtype=int), p, f, np.array([1, 1, -1, -1]))
    except ValueError:
        pass
    else:
        raise AssertionError('Mixed orbital directions accepted in root.')
    for limits, sigma, minimum, maximum in [([0, 1, 1], 2, 1, 1),
                                            ([1, 1, 1], 0, 1, 1),
                                            ([1, 1, 1], 2, 0, 1),
                                            ([1, 1, 1], 2, 1, -1)]:
        try:
            one.select(limits, sigma, minimum, maximum)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid parameters accepted.')
    print('UNIT TESTS PASS', flush=True)


def data_and_anchor():
    data = np.load(CACHE / 'features.npz')
    p = data['features']
    assert p.shape == (100313, 7)
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100*app.normalize_block(p[:, 6:7])))
    fingerprint = hashlib.sha256(p.tobytes()+data['source_ids'].tobytes()).hexdigest()
    labels = None
    for count in (99, 180):
        path = HERE / f'anchor_base_{count}.npz'
        if path.exists():
            saved = np.load(path)
            assert str(saved['fingerprint']) == fingerprint
            assert str(saved['function_signature']) == function_signature()
            labels = saved['labels']
        else:
            labels = app.constrained_clustering(f, p[:, 6], count, min_size=200,
                                                sigma_limit=ANCHOR_SIGMA, previous_labels=labels)
            np.savez_compressed(path, labels=labels, source_ids=data['source_ids'],
                                fingerprint=fingerprint, function_signature=function_signature())
    return p, f, labels, data['source_ids']


def write_csv(path, rows):
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(replay=False):
    initial_signature = function_signature()
    assert initial_signature == '7f979cff1dd4b94f39f25e75beb04ac4f71bfefea658472ad66f8a5271a9485a'
    unit_tests()
    p, f, roots, source_ids = data_and_anchor()
    model = FixedHierarchy(roots, p[:, :3], f, p[:, 6])
    cases = [(profile, limits, sigma, minimum)
             for profile, limits in PROFILES for sigma in SIGMAS for minimum in MINIMA]
    if replay:
        expected = np.load(HERE / 'all_labels.npz')
        np.testing.assert_array_equal(expected['source_ids'], source_ids)
        for index in reversed(range(len(cases))):
            _, limits, sigma, minimum = cases[index]
            labels, selected = model.select(limits, sigma, minimum, 1003)
            np.testing.assert_array_equal(labels, expected['labels'][index])
            model.audit(labels, selected, limits, sigma, minimum, 1003)
        (HERE / 'replay.json').write_text(json.dumps(dict(cases=len(cases), matches=True,
                                                        fits=model.fits, order='reverse')))
        print(f'REPLAY PASS: {len(cases)} exact partitions in reversed order in a fresh process.', flush=True)
        return

    started = time.perf_counter()
    reference_labels, reference_nodes = model.select(FIXED_SCALE, ANCHOR_SIGMA, 25, 1003)
    with contextlib.redirect_stdout(io.StringIO()):
        original = app.split_messy_clusters(roots, p[:, :3], FIXED_SCALE, 25, 1003,
                                            features=f, sigma_limit=ANCHOR_SIGMA)
    np.testing.assert_array_equal(reference_labels, canonical(original))
    print('REFERENCE PASS: identical current GA16 partition at sigma=2, STD=[.04,.05,8], min=25.', flush=True)

    rows, partitions, representatives = [], [], []
    for index, (profile, limits, sigma, minimum) in enumerate(cases):
        labels, selected = model.select(limits, sigma, minimum, 1003)
        metrics, reps = model.audit(labels, selected, limits, sigma, minimum, 1003)
        rows.append(dict(case=index, profile=profile, std_eps=limits[0], std_e=limits[1],
                         std_w=limits[2], max_sigma=sigma, minpoints=minimum, **metrics))
        partitions.append(labels)
        representatives.append(reps)
        if minimum == 25:
            print('CUT', json.dumps(rows[-1]), flush=True)
    arrays = np.stack(partitions)
    comparisons = []
    for i, (_, limits_i, sigma_i, min_i) in enumerate(cases):
        a = arrays[i]
        mask = a >= 0
        for j, (_, limits_j, sigma_j, min_j) in enumerate(cases):
            if (i == j or sigma_j < sigma_i or min_j > min_i or
                    not np.all(np.asarray(limits_j) >= limits_i)):
                continue
            b = arrays[j]
            lost = int(np.count_nonzero(mask & (b < 0)))
            assert lost == 0, (i, j, lost)
            mapping = b[representatives[i]]
            assert np.all(mapping >= 0)
            assert np.all(b[mask] == mapping[a[mask]]), (i, j, 'previous group fragmented')
            comparisons.append(dict(strict_case=i, relaxed_case=j, lost=lost,
                                    gained=int(np.count_nonzero((a < 0) & (b >= 0)))))

    comparison_old = []
    old_reference_mask = original >= 0
    for sigma in (2., 3., 3.5, 4.):
        if sigma == 2.:
            old = original
        else:
            saved = np.load(CACHE / f'sigma_{sigma:g}' / 'base_180.npz')
            np.testing.assert_array_equal(saved['source_ids'], source_ids)
            with contextlib.redirect_stdout(io.StringIO()):
                old = app.split_messy_clusters(saved['labels'], p[:, :3], FIXED_SCALE, 25, 1003,
                                                features=f, sigma_limit=sigma)
        new, _ = model.select(FIXED_SCALE, sigma, 25, 1003)
        comparison_old.append(dict(max_sigma=sigma,
            old_clusters=len(np.unique(old[old >= 0])), old_unassigned=int(np.sum(old < 0)),
            old_lost_from_sigma2=int(np.sum(old_reference_mask & (old < 0))),
            fixed_clusters=len(np.unique(new[new >= 0])), fixed_unassigned=int(np.sum(new < 0)),
            fixed_lost_from_sigma2=int(np.sum(old_reference_mask & (new < 0)))))
    strict_row = next(r for r in rows if r['profile'] == 'strict' and r['max_sigma'] == 2. and r['minpoints'] == 1)
    report = dict(trajectories=len(p), anchor_sigma=ANCHOR_SIGMA, split_scale=FIXED_SCALE.tolist(),
                  cases=len(cases), monotonic_comparisons=len(comparisons), lost_points=0,
                  reference_partition_matches=True, cap_policy='raise, never truncate',
                  old_vs_fixed=comparison_old, strictest_full_coverage=strict_row,
                  seconds=time.perf_counter()-started, fits=model.fits, nodes=len(model.nodes))
    write_csv(HERE / 'cuts.csv', rows)
    write_csv(HERE / 'monotonic_comparisons.csv', comparisons)
    write_csv(HERE / 'old_vs_fixed.csv', comparison_old)
    np.savez_compressed(HERE / 'all_labels.npz', labels=arrays, source_ids=source_ids)
    (HERE / 'report.json').write_text(json.dumps(report, indent=2))
    assert initial_signature == function_signature()
    print('RESULT', json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    main(parser.parse_args().replay)
