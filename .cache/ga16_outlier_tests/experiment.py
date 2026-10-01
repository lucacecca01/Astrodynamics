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
from test_fusion import app


def trial_class():
    # The tested physical-MAD variant is now implemented in GA_16.
    return app.ClusterShapeConstraint


def main():
    saved = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
    p, source_ids = saved['features'], saved['source_ids']
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100 * app.normalize_block(p[:, 6:7])))
    reference = np.load(ROOT / '.cache/ga16_strong_gap_tests/labels.npz')
    np.testing.assert_array_equal(reference['source_ids'], source_ids)
    old = reference['labels']
    original = np.load(ROOT / '.cache/ga16_shape_implementation_tests/dip_only/real_inf.npz')['labels']
    example = np.flatnonzero(original == 9)
    assert len(example) == 36
    np.testing.assert_allclose(app.physical_std(p[example, :3]),
                               [.003624806812783081, .013106304316625284, 4.519943207329154])
    roots = []
    for count in (99, 180):
        root = np.load(ROOT / f'.cache/ga16_sigma_recovery_tests/sigma_3/base_{count}.npz')
        np.testing.assert_array_equal(root['source_ids'], source_ids)
        roots.append(root['labels'])
    klass = app.ClusterShapeConstraint if '--production' in sys.argv else trial_class()
    report = []
    for sigma in ([6.] if '--production' in sys.argv else [4., 6., 8.]):
        check = klass(p[:, :3], [.04, .05, 10])
        check.outlier_sigma = sigma
        valid = app.make_cluster_validator(p[:, :3], f, [.04, .05, 10], 5, 3,
                                          directions=p[:, 6], shape_condition=check)
        initial = np.full(len(p), -1, dtype=int)
        initial[example] = 0
        pool = []
        local = app.split_messy_clusters(initial, p[:, :3], [.04, .05, 10], 5, 1003,
                                        features=f, sigma_limit=3, group_validator=valid,
                                        gap_splitter=check.split, candidate_groups=pool)
        local, _ = app.fuse_cluster_candidates(pool, len(p), valid, 1003, previous_labels=local)
        start = time.perf_counter()
        labels, _, _ = app.refine_and_fuse_clusters(
            roots, p[:, :3], f, p[:, 6], [.04, .05, 10], 5, 1003, 3, shape_condition=check)
        elapsed = time.perf_counter() - start
        groups = app.cluster_groups(labels)
        assert all(valid(ids) for ids in groups)
        item = dict(sigma=str(sigma), clusters=len(groups), excluded=int(np.sum(labels < 0)),
                    seconds=elapsed, lost=int(np.sum((old >= 0) & (labels < 0))),
                    recovered=int(np.sum((old < 0) & (labels >= 0))),
                    local_sizes=[len(g) for g in app.cluster_groups(local)],
                    local_excluded=int(np.sum(local[example] < 0)))
        report.append(item)
        np.savez_compressed(HERE / f'labels_{sigma}.npz', labels=labels, local=local,
                            source_ids=source_ids, example=example)
        (HERE / 'report.json').write_text(json.dumps(report, indent=2))
        print('RESULT', json.dumps(item), flush=True)
    synthetic = []
    rng = np.random.default_rng(36)
    for count in (5, 10, 36, 100):
        for geometry in ('normal', 'line', 'arc'):
            flagged = np.zeros(3, dtype=int)
            for _ in range(100):
                noise = rng.normal(size=(count, 3))
                t = rng.uniform(-1, 1, count)
                values = noise
                if geometry == 'line':
                    values = np.column_stack((t, .5*t, -.2*t)) + .01*noise
                if geometry == 'arc':
                    values = np.column_stack((t, t*t, .4*t*t*t)) + .01*noise
                values = values * [.01, .025, 4.] + [-.6, .3, 180.]
                for j, sigma in enumerate((4., 6., 8.)):
                    check = klass(values, [.04, .05, 10])
                    check.outlier_sigma = sigma
                    flagged[j] += not check(np.arange(count))
            synthetic.append(dict(n=count, geometry=geometry, trials=100,
                                  robust4=int(flagged[0]), robust6=int(flagged[1]), robust8=int(flagged[2])))
    (HERE / 'synthetic.json').write_text(json.dumps(synthetic, indent=2))
    print('SYNTHETIC', json.dumps(synthetic), flush=True)


if __name__ == '__main__':
    main()
