import ast
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ga16-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if '--dip-only' in sys.argv:
    HERE = HERE / 'dip_only'
    HERE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
import numpy as np
from test_fusion import app, SOURCE


def main():
    data = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
    p, source_ids = data['features'], data['source_ids']
    n = len(p)
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100*app.normalize_block(p[:, 6:7])))
    roots = []
    for count in (99, 180):
        root = np.load(ROOT / f'.cache/ga16_sigma_recovery_tests/sigma_3/base_{count}.npz')
        np.testing.assert_array_equal(source_ids, root['source_ids'])
        roots.append(root['labels'])
    names = {'STD_LIMITS', 'MAX_SIGMA', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS', 'DIP_ALPHA',
             'GAP_FRACTION', 'GAUSSIAN_QQ_MAX'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    config = dict(np=np, n_kept=n)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), config)
    rows = []
    for qq_max in ([np.inf] if '--dip-only' in sys.argv else (.05, .03)):
        shape = app.ClusterShapeConstraint(p[:, :3], config['STD_LIMITS'], dip_alpha=config['DIP_ALPHA'],
                                            gap_fraction=config['GAP_FRACTION'], gaussian_qq_max=qq_max)
        start = time.perf_counter()
        labels, baseline, pool = app.refine_and_fuse_clusters(
            roots, p[:, :3], f, p[:, 6], config['STD_LIMITS'], config['MIN_CLUSTER_SIZE'],
            config['MAX_CLUSTERS'], config['MAX_SIGMA'], shape_condition=shape,
        )
        elapsed = time.perf_counter()-start
        groups = app.cluster_groups(labels)
        validator = app.make_cluster_validator(p[:, :3], f, config['STD_LIMITS'], config['MIN_CLUSTER_SIZE'],
                                              config['MAX_SIGMA'], directions=p[:, 6], shape_condition=shape)
        assert len(groups) <= config['MAX_CLUSTERS']
        assert all(validator(ids) for ids in groups)
        qq = np.array([app.gaussian_shape_penalty(p[ids, :3]) for ids in groups])
        assert np.all(qq <= qq_max)
        row = dict(qq_max='disabled' if np.isinf(qq_max) else qq_max, n=n, clusters=len(groups), excluded=int(np.sum(labels < 0)),
                   minimum_size=min(map(len, groups), default=0), seconds=elapsed,
                   qq_maximum=qq.max(axis=0).tolist() if len(qq) else [],
                   weighted_qq=np.average(qq, weights=list(map(len, groups)), axis=0).tolist() if len(qq) else [])
        rows.append(row)
        np.savez_compressed(HERE / f'real_{qq_max:g}.npz', source_ids=source_ids, labels=labels, baseline=baseline)
        app.save_fusion_cache(HERE / f'cache_{qq_max:g}.npz', app.fusion_fingerprint(source_ids, p, f),
                              pool, labels, constraints_key=str(qq_max))
        (HERE / 'real_results.json').write_text(json.dumps(rows, indent=2))
        print('RESULT', json.dumps(row), flush=True)
    print('PASS: actual production shape, split, fusion and independent Q-Q checks on all 100313 points', flush=True)


if __name__ == '__main__':
    main()
