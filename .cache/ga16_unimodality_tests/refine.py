import contextlib
import io
import json
import time

from experiment import HERE, ROOT, SOURCE, LIMITS, MINIMUM, SIGMA, ALPHA, ShapeCheck, metrics, projections
from experiment import np, app, hashlib


def targeted_split(labels, p, f, shape, mode, candidates):
    base_valid = app.make_cluster_validator(p[:, :3], f, LIMITS, MINIMUM, SIGMA, directions=p[:, 6])
    extra = shape.condition(mode)
    gamma = float(mode[4:])
    pending = app.cluster_groups(labels)
    accepted = []
    gaps, fallback, excluded = 0, 0, 0
    with app.threadpool_limits(limits=2):
        while pending:
            ids = pending.pop()
            candidates.append(ids)
            if len(ids) < MINIMUM:
                excluded += len(ids)
                continue
            valid_physical = base_valid(ids)
            if valid_physical and extra(ids):
                accepted.append(ids)
                continue
            if len(ids) <= MINIMUM:
                excluded += len(ids)
                continue
            cuts = []
            if valid_physical:
                values = projections(p[ids, :3])
                scores = shape.scores(ids)
                support = max(3, int(np.ceil(.1 * len(ids))))
                for axis in np.flatnonzero((scores[:, 1] < ALPHA) & (scores[:, 2] > gamma)):
                    order = np.argsort(values[:, axis], kind='stable')
                    sorted_values = values[order, axis]
                    spacing = np.diff(sorted_values) / np.ptp(sorted_values)
                    for cut in np.flatnonzero(spacing > gamma) + 1:
                        if support <= cut <= len(ids) - support:
                            cuts.append((spacing[cut - 1], axis, cut, order))
            if cuts:
                _, _, cut, order = max(cuts, key=lambda item: item[0])
                pending.extend([ids[order[:cut]], ids[order[cut:]]])
                gaps += 1
            else:
                w = np.deg2rad(p[ids, 2])
                values = np.column_stack((p[ids, :2] / LIMITS[:2], np.cos(w)/np.deg2rad(LIMITS[2]),
                                          np.sin(w)/np.deg2rad(LIMITS[2])))
                if np.all(values == values[0]):
                    excluded += len(ids)
                    continue
                split = app.KMeans(n_clusters=2, n_init=5, random_state=42,
                                   max_iter=500, tol=1e-9).fit_predict(values)
                assert len(np.unique(split)) == 2
                pending.extend([ids[split == 0], ids[split == 1]])
                fallback += 1
    accepted.sort(key=lambda ids: int(ids.min()))
    result = np.full(len(p), -1, dtype=int)
    for c, ids in enumerate(accepted):
        assert base_valid(ids) and extra(ids)
        result[ids] = c
    print('TARGETED', mode, dict(gap_splits=gaps, kmeans_splits=fallback, excluded=excluded), flush=True)
    return result


def main():
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [SOURCE, SOURCE.parent / 'Clusters/GA_16_fusion_candidates.npz']}
    data = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
    p, source_ids = data['features'], data['source_ids']
    n = len(p)
    local = np.load(HERE / 'local_labels.npz')
    np.testing.assert_array_equal(local['source_ids'], source_ids)
    baseline = local['baseline']
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100 * app.normalize_block(p[:, 6:7])))
    pool, _ = app.load_fusion_cache(ROOT / '.cache/ga16_fusion_implementation_tests/roundtrip.npz',
                                   app.fusion_fingerprint(source_ids, p, f), n)
    roots = []
    for k in (99, 180):
        root = np.load(ROOT / f'.cache/ga16_sigma_recovery_tests/sigma_3/base_{k}.npz')
        np.testing.assert_array_equal(root['source_ids'], source_ids)
        roots.append(root['labels'])
    for mode in ('dip', 'gap_0.1', 'gap_0.2', 'gap_0.3'):
        saved = np.load(HERE / f'pool_{mode}.npz')
        np.testing.assert_array_equal(saved['source_ids'], source_ids)
        pool.extend(np.split(saved['members'], saved['offsets'][1:-1]))
    shape = ShapeCheck(p)
    cases = {}
    report = dict(n=n, sigma=SIGMA, minimum=MINIMUM, limits=LIMITS.tolist(), cases=[])
    for mode in ('gap_0.1', 'gap_0.2', 'gap_0.3'):
        start = time.perf_counter()
        labels = targeted_split(baseline, p, f, shape, mode, pool)
        row = dict(mode=mode, stage='targeted_split', seconds=time.perf_counter()-start,
                   **metrics(labels, p, f, shape, baseline, shape.condition(mode)))
        cases['targeted_' + mode] = labels
        report['cases'].append(row)
        print('RESULT', json.dumps(row), flush=True)
    # Generate alternatives with the actual production pipeline and fixed initial roots.
    for mode in ('dip', 'gap_0.2'):
        start = time.perf_counter()
        labels, _, generated = app.refine_and_fuse_clusters(
            roots + [baseline], p[:, :3], f, p[:, 6], LIMITS, MINIMUM, n//100, SIGMA,
            extra_condition=shape.condition(mode), candidates=pool, time_limit=30.)
        pool = generated
        row = dict(mode=mode, stage='production_refine', seconds=time.perf_counter()-start,
                   **metrics(labels, p, f, shape, baseline, shape.condition(mode)))
        report['cases'].append(row)
        cases['refine_' + mode] = labels
        (HERE / 'refine_report.json').write_text(json.dumps(report, indent=2))
        print('RESULT', json.dumps(row), flush=True)
    np.savez_compressed(HERE / 'common_pool.npz', source_ids=source_ids,
                        members=np.concatenate(pool), offsets=np.r_[0, np.cumsum([len(ids) for ids in pool])])
    # Final controlled comparison: identical candidate universe in every case.
    for mode in ('baseline', 'dip', 'gap_0.1', 'gap_0.2', 'gap_0.3'):
        start = time.perf_counter()
        extra = shape.condition(mode)
        validator = app.make_cluster_validator(p[:, :3], f, LIMITS, MINIMUM, SIGMA,
                                              directions=p[:, 6], extra_condition=extra)
        protected_possible = True
        try:
            labels, _ = app.fuse_cluster_candidates(pool, n, validator, n//100,
                                                     previous_labels=baseline, time_limit=30.)
            protection_error = None
        except RuntimeError as exc:
            protected_possible = False
            protection_error = str(exc)
            labels, _ = app.fuse_cluster_candidates(pool, n, validator, n//100, time_limit=30.)
        row = dict(mode=mode, stage='common_pool', seconds=time.perf_counter()-start,
                   protected_possible=protected_possible, protection_error=protection_error,
                   **metrics(labels, p, f, shape, baseline, extra))
        report['cases'].append(row)
        cases['common_' + mode] = labels
        (HERE / 'refine_report.json').write_text(json.dumps(report, indent=2))
        print('RESULT', json.dumps(row), flush=True)
    np.savez_compressed(HERE / 'refine_labels.npz', source_ids=source_ids, baseline=baseline, **cases)
    for path, expected in hashes.items():
        assert hashlib.sha256(app.Path(path).read_bytes()).hexdigest() == expected
    print('DONE: production code and cache unchanged', flush=True)


if __name__ == '__main__':
    main()
