import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ga16-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
import numpy as np
from test_fusion import app, SOURCE


def metrics(labels, reference, p, f, cfg, minimum, shape):
    groups = app.cluster_groups(labels)
    valid = app.make_cluster_validator(p[:, :3], f, cfg['STD_LIMITS'], minimum, cfg['MAX_SIGMA'],
                                      directions=p[:, 6], shape_condition=shape)
    assert len(groups) <= cfg['MAX_CLUSTERS']
    assert all(valid(ids) for ids in groups)
    sizes = np.array([len(ids) for ids in groups])
    stds = np.array([app.physical_std(p[ids, :3]) for ids in groups])
    return dict(clusters=len(groups), excluded=int(np.sum(labels < 0)),
                excluded_percent=float(100*np.mean(labels < 0)),
                minimum_size=int(sizes.min()), median_size=float(np.median(sizes)),
                five_point_clusters=int(np.sum(sizes == 5)),
                clusters_5_to_9=int(np.sum(sizes < 10)),
                points_in_clusters_5_to_9=int(sizes[sizes < 10].sum()),
                lost=int(np.sum((reference >= 0) & (labels < 0))),
                recovered=int(np.sum((reference < 0) & (labels >= 0))),
                maximum_std=stds.max(axis=0).tolist(),
                weighted_std=np.average(stds, axis=0, weights=sizes).tolist())


def main():
    watched = [SOURCE] + list((SOURCE.parent / 'Clusters').glob('*fusion*.npz'))
    watched += list((SOURCE.parent / 'Clusters').glob('*dip_gap*.npz'))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
    saved = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
    p, source_ids = saved['features'], saved['source_ids']
    n = len(p)
    config = dict(np=np, n_kept=n)
    names = {'STD_LIMITS', 'MAX_SIGMA', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS', 'DIP_ALPHA',
             'GAP_FRACTION', 'GAUSSIAN_QQ_MAX', 'FUSION_SEEDS', 'BASE_CLUSTER_COUNTS'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), config)
    assert config['MAX_SIGMA'] == 3
    assert np.all(np.isinf(config['GAUSSIAN_QQ_MAX']))
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100*app.normalize_block(p[:, 6:7])))
    roots = []
    for count in config['BASE_CLUSTER_COUNTS']:
        data = np.load(ROOT / f'.cache/ga16_sigma_recovery_tests/sigma_3/base_{count}.npz')
        np.testing.assert_array_equal(source_ids, data['source_ids'])
        roots.append(data['labels'])
    old = np.load(ROOT / '.cache/ga16_shape_implementation_tests/dip_only/real_inf.npz')
    np.testing.assert_array_equal(source_ids, old['source_ids'])
    reference = old['labels']
    pool, cached_reference = app.load_fusion_cache(
        ROOT / '.cache/ga16_shape_implementation_tests/dip_only/cache_inf.npz',
        app.fusion_fingerprint(source_ids, p, f), n)
    np.testing.assert_array_equal(reference, cached_reference)
    shape = app.ClusterShapeConstraint(p[:, :3], config['STD_LIMITS'],
                                       dip_alpha=config['DIP_ALPHA'], gap_fraction=config['GAP_FRACTION'],
                                       gaussian_qq_max=config['GAUSSIAN_QQ_MAX'])
    report = dict(n=n, sigma=config['MAX_SIGMA'], std_limits=config['STD_LIMITS'].tolist(),
                  dip_alpha=config['DIP_ALPHA'], gap_fraction=config['GAP_FRACTION'],
                  gaussian_constraint=False, cases=[])
    row = dict(mode='reference_min10', minimum=10, **metrics(reference, reference, p, f, config, 10, shape))
    report['cases'].append(row)
    print('RESULT', json.dumps(row), flush=True)
    results = dict(reference=reference)
    for mode, candidates, previous in [('fresh_min5', [], None), ('protected_min5', pool, reference)]:
        check = app.ClusterShapeConstraint(p[:, :3], config['STD_LIMITS'],
                                           dip_alpha=config['DIP_ALPHA'], gap_fraction=config['GAP_FRACTION'],
                                           gaussian_qq_max=config['GAUSSIAN_QQ_MAX'])
        start = time.perf_counter()
        labels, _, _ = app.refine_and_fuse_clusters(
            roots, p[:, :3], f, p[:, 6], config['STD_LIMITS'], 5, config['MAX_CLUSTERS'],
            config['MAX_SIGMA'], shape_condition=check, candidates=candidates, previous_labels=previous,
            seeds=config['FUSION_SEEDS'], base_count=config['BASE_CLUSTER_COUNTS'][-1],
        )
        row = dict(mode=mode, minimum=5, seconds=time.perf_counter()-start,
                   **metrics(labels, reference, p, f, config, 5, check))
        if previous is not None:
            assert row['lost'] == 0
        report['cases'].append(row)
        results[mode] = labels
        (HERE / 'report.json').write_text(json.dumps(report, indent=2))
        np.savez_compressed(HERE / 'labels.npz', source_ids=source_ids, **results)
        print('RESULT', json.dumps(row), flush=True)
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    report['production_unchanged'] = True
    (HERE / 'report.json').write_text(json.dumps(report, indent=2))
    print('PASS: constraints checked; production code and cache unchanged.', flush=True)


if __name__ == '__main__':
    main()
