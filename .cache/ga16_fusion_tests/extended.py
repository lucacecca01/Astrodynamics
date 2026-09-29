import csv
import hashlib
import itertools
import json
import time

import numpy as np

import experiment as test


class SeededHierarchy(test.previous.FixedHierarchy):
    def __init__(self, *args, seed, **kwargs):
        self.seed = seed
        super().__init__(*args, **kwargs)

    def children(self, index):
        node = self.nodes[index]
        if node['children'] is None:
            ids = node['ids']
            model = test.app.KMeans(n_clusters=2, n_init=5, random_state=self.seed,
                                    max_iter=500, tol=1e-9, algorithm='lloyd')
            split = model.fit_predict(self.coordinates[ids])
            assert len(np.unique(split)) == 2
            children = sorted((ids[split == side] for side in (0, 1)), key=lambda x: int(x[0]))
            node['children'] = tuple(self.make_node(group) for group in children)
            self.fits += 1
        return node['children']


def brute_force_tests():
    rng = np.random.default_rng(731)
    for trial in range(60):
        n, k, cap = 16, 9, 4
        groups = [np.arange(4)] + [np.sort(rng.choice(n, size=rng.integers(2, 7), replace=False))
                                  for _ in range(k-1)]
        required = np.zeros(n, dtype=bool)
        required[:3] = True
        freeze = [0] if trial % 2 else []
        gains = np.array([np.sum(~required[g]) for g in groups])
        costs = 1 - (cap+1) * gains
        optimum = None
        for bits in itertools.product((False, True), repeat=k):
            chosen = np.flatnonzero(bits)
            if len(chosen) > cap or not set(freeze) <= set(chosen):
                continue
            count = np.zeros(n, dtype=int)
            for j in chosen:
                count[groups[j]] += 1
            if np.any(count > 1) or not np.all(count[required] == 1):
                continue
            score = int(costs[chosen].sum())
            optimum = score if optimum is None else min(optimum, score)
        labels, info = test.pack(groups, required, cap, [0], fixed=freeze, seconds=5)
        assert info['solver_status'] == 0
        assert int(costs[info['selected_indices']].sum()) == optimum
        assert np.all(labels[required] >= 0)
    print('BRUTE FORCE PASS: 60 randomized overlap problems, exact exhaustive optimum', flush=True)


def extend_pool(pool, p, f, source_ids):
    path = test.HERE / 'pool_extended.npz'
    if path.exists():
        return test.Pool.load(path, p, f, source_ids)
    started = time.perf_counter()
    base_paths = {
        'sigma2_99': test.OLD / 'anchor_base_99.npz',
        'sigma2_180': test.OLD / 'anchor_base_180.npz',
        'sigma3_180': test.CACHE / 'sigma_3/base_180.npz',
        'sigma3.5_180': test.CACHE / 'sigma_3.5/base_180.npz',
        'sigma4_180': test.CACHE / 'sigma_4/base_180.npz',
    }
    variants = [
        ('sigma2_99', 7, [.04, .05, 8.]),
        ('sigma2_180', 7, [.04, .05, 8.]),
        ('sigma2_180', 123, [.04, .05, 8.]),
        ('sigma3_180', 42, [.04, .05, 8.]),
        ('sigma3.5_180', 42, [.04, .05, 8.]),
        ('sigma4_180', 42, [.04, .05, 8.]),
        ('sigma2_180', 42, [.03, .04, 6.]),
        ('sigma2_180', 7, [.06, .035, 8.]),
        ('sigma2_180', 123, [.03, .08, 6.]),
        ('independent', 7, [.04, .05, 8.]),
        ('independent', 123, [.04, .05, 8.]),
    ]
    records = []
    for index, (base, seed, scale) in enumerate(variants):
        start = time.perf_counter()
        if base == 'independent':
            roots = np.full(len(p), -1, dtype=int)
            offset = 0
            with test.app.threadpool_limits(limits=2):
                for direction in np.unique(p[:, 6]):
                    ids = np.flatnonzero(p[:, 6] == direction)
                    k = max(1, round(180 * len(ids) / len(p)))
                    roots[ids] = test.app.KMeans(n_clusters=k, n_init=3, random_state=seed,
                                                 max_iter=500, tol=1e-9).fit_predict(f[ids, :4]) + offset
                    offset += k
        else:
            data = np.load(base_paths[base])
            if 'source_ids' in data:
                np.testing.assert_array_equal(data['source_ids'], source_ids)
            roots = data['labels']
        model = SeededHierarchy(roots, p[:, :3], f, p[:, 6], scale=np.array(scale), seed=seed)
        for sigma in (2., 3., 4.):
            labels, nodes = model.select(test.LIMITS, sigma, 25, test.CAP)
            model.audit(labels, nodes, test.LIMITS, sigma, 25, test.CAP)
            pool.add_partition(f'alternative_{index}_sigma_{sigma:g}', labels)
        record = dict(index=index, base=base, seed=seed, split_scale=scale,
                       fits=model.fits, seconds=time.perf_counter()-start,
                       total_candidates=len(pool.groups))
        records.append(record)
        print('VARIANT', json.dumps(record), flush=True)
    pool.save(path, source_ids)
    (test.HERE / 'generation.json').write_text(json.dumps(dict(
        seconds=time.perf_counter()-started, variants=records,
        physical_limits=test.LIMITS.tolist(), pool_candidates=len(pool.groups)), indent=2))
    return pool


def main():
    source_hash = hashlib.sha256(test.previous.SOURCE.read_bytes()).hexdigest()
    brute_force_tests()
    p, f, source_ids = test.load_data()
    pool = extend_pool(test.prepare(p, f, source_ids), p, f, source_ids)
    rows = []
    for sigma in (2., 3., 4.):
        for minimum in (25, 50, 101):
            cached = f'cached_sigma_{sigma:g}_min_{minimum}'
            prior = np.load(test.HERE / f'{cached}.npz')['labels']
            row, labels = test.run_case(pool, f'extended_sigma_{sigma:g}_min_{minimum}',
                                        sigma, minimum, prior, seconds=30)
            original = test.baseline(sigma, minimum)
            row['original_unassigned'] = int(np.sum(original < 0))
            row['original_clusters'] = int(len(np.unique(original[original >= 0])))
            row['recovered_from_original'] = int(np.sum(original < 0) - np.sum(labels < 0))
            rows.append(row)
    with (test.HERE / 'comparison_extended.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    row, _ = test.run_case(pool, 'extended_sigma_2_min_50_frozen', 2., 50, test.baseline(2., 50),
                           freeze=True, seconds=30)
    assert source_hash == hashlib.sha256(test.previous.SOURCE.read_bytes()).hexdigest()
    print('DONE: extended tests, production file unchanged', flush=True)


if __name__ == '__main__':
    main()
