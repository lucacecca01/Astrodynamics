import ast
import csv
import json
from pathlib import Path
import time

import numpy as np

from test_fusion import app, ROOT, SOURCE

HERE = Path(__file__).resolve().parent


def audit(labels, p, f, limits, sigma, minimum, cap, protected=None, extra=None):
    if protected is not None:
        assert not np.any(protected & (labels < 0))
    groups = app.cluster_groups(labels)
    assert len(groups) <= cap
    stds, sizes = [], []
    for ids in groups:
        assert len(ids) >= minimum
        assert len(np.unique(p[ids, 6])) == 1
        physical = p[ids, :3].copy()
        angle = np.deg2rad(physical[:, 2])
        center = np.rad2deg(np.arctan2(np.sin(angle).sum(), np.cos(angle).sum()))
        physical[:, 2] = (physical[:, 2] - center + 180) % 360 - 180
        std = physical.std(axis=0)
        assert np.all(std < limits)
        x = f[ids]
        assert np.all(((x-x.mean(axis=0))**2).sum(axis=1) <= sigma**2*x.var(axis=0).sum())
        assert extra is None or extra(ids)
        stds.append(std)
        sizes.append(len(ids))
    return dict(clusters=len(groups), unassigned=int(np.sum(labels < 0)),
                smallest=min(sizes, default=0),
                weighted_std=np.average(stds, weights=sizes, axis=0).tolist() if sizes else [],
                maximum_std=np.max(stds, axis=0).tolist() if sizes else [],
                lost=int(np.sum(protected & (labels < 0))) if protected is not None else 0)


def main():
    saved = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
    p, source_ids = saved['features'], saved['source_ids']
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100*app.normalize_block(p[:, 6:7])))
    n = len(p)
    config = {'np': np, 'n_kept': n}
    names = {'STD_LIMITS', 'MAX_SIGMA', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS',
             'BASE_MIN_CLUSTER_SIZE', 'BASE_CLUSTER_COUNTS', 'FUSION_SEEDS', 'FUSION_TIME_LIMIT'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), config)
    limits, sigma, minimum, cap = [config[name] for name in
                                   ('STD_LIMITS', 'MAX_SIGMA', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS')]
    fingerprint = app.fusion_fingerprint(source_ids, p, f)

    # Regression: same candidate set and protected points as the prototype sweep.
    old = ROOT / '.cache/ga16_fusion_tests'
    data = np.load(old / 'pool_extended.npz')
    np.testing.assert_array_equal(data['source_ids'], source_ids)
    groups = list(np.split(data['members'], data['offsets'][1:-1]))
    regression = []
    for s in (2., 3., 4.):
        for m in (25, 50, 101):
            prior = np.load(old / f'cached_sigma_{s:g}_min_{m}.npz')['labels']
            validator = app.make_cluster_validator(p[:, :3], f, [.04, .05, 8.], m, s, directions=p[:, 6])
            result, _ = app.fuse_cluster_candidates(groups, n, validator, cap, previous_labels=prior)
            metrics = audit(result, p, f, [.04, .05, 8.], s, m, cap, prior >= 0)
            expected = json.loads((old / f'extended_sigma_{s:g}_min_{m}.json').read_text())
            assert metrics['clusters'] == expected['clusters']
            assert metrics['unassigned'] == expected['unassigned']
            regression.append(dict(sigma=s, minimum=m, **metrics))
    print('REGRESSION PASS: all 9 real-data prototype optima reproduced', flush=True)

    # Rebuild the actual initial GA16 clustering at its current parameter values.
    initial_cache = HERE / 'actual_initial.npz'
    key = f'{fingerprint}:{sigma}:{config["BASE_MIN_CLUSTER_SIZE"]}:{config["BASE_CLUSTER_COUNTS"]}'
    partitions = []
    if initial_cache.exists() and str(np.load(initial_cache)['key']) == key:
        partitions = list(np.load(initial_cache)['partitions'])
    else:
        labels = None
        for count in config['BASE_CLUSTER_COUNTS']:
            labels = app.constrained_clustering(f, p[:, 6], count,
                                                min_size=config['BASE_MIN_CLUSTER_SIZE'],
                                                sigma_limit=sigma, previous_labels=labels)
            partitions.append(labels.copy())
        np.savez_compressed(initial_cache, key=key, partitions=partitions)
    start = time.perf_counter()
    result, baseline, candidates = app.refine_and_fuse_clusters(
        partitions, p[:, :3], f, p[:, 6], limits, minimum, cap, sigma,
        seeds=config['FUSION_SEEDS'], base_count=config['BASE_CLUSTER_COUNTS'][-1])
    elapsed = time.perf_counter()-start
    initial = audit(baseline, p, f, limits, sigma, minimum, cap)
    final = audit(result, p, f, limits, sigma, minimum, cap, baseline >= 0)
    cache = HERE / 'roundtrip.npz'
    app.save_fusion_cache(cache, fingerprint, candidates, result)
    loaded, prior = app.load_fusion_cache(cache, fingerprint, n)
    replay, _, replay_pool = app.refine_and_fuse_clusters(
        partitions, p[:, :3], f, p[:, 6], limits, minimum, cap, sigma,
        seeds=config['FUSION_SEEDS'], base_count=config['BASE_CLUSTER_COUNTS'][-1],
        candidates=loaded, previous_labels=prior)
    replay_metrics = audit(replay, p, f, limits, sigma, minimum, cap, result >= 0)
    np.testing.assert_array_equal(replay, result)
    relaxed, _, _ = app.refine_and_fuse_clusters(
        partitions, p[:, :3], f, p[:, 6], limits * 1.1, minimum, cap, sigma + .5,
        seeds=config['FUSION_SEEDS'], base_count=config['BASE_CLUSTER_COUNTS'][-1],
        candidates=replay_pool, previous_labels=replay)
    relaxed_metrics = audit(relaxed, p, f, limits*1.1, sigma+.5, minimum, cap, replay >= 0)

    extra = lambda ids: len(ids) <= 400 and np.std(p[ids, 3]) < .1
    custom, _, _ = app.refine_and_fuse_clusters(
        partitions, p[:, :3], f, p[:, 6], limits, minimum, cap, sigma,
        seeds=(42, 7), base_count=config['BASE_CLUSTER_COUNTS'][-1], extra_condition=extra)
    custom_metrics = audit(custom, p, f, limits, sigma, minimum, cap, extra=extra)
    report = dict(n=n, limits=limits.tolist(), sigma=sigma, minimum=minimum, cap=cap,
                   baseline=initial, fused=final, fusion_seconds=elapsed,
                   cached_replay=replay_metrics, relaxed=relaxed_metrics,
                   custom_constraints=custom_metrics, prototype_regressions=regression)
    (HERE / 'report.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(HERE / 'actual_results.npz', source_ids=source_ids,
                        baseline=baseline, fused=result, relaxed=relaxed, custom=custom)
    print('REAL DATA PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
