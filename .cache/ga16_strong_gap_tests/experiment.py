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
from scipy.special import ndtri
from diptest import diptest
from test_fusion import app, SOURCE


def main():
    data = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
    p, source_ids = data['features'], data['source_ids']
    n = len(p)
    reference = np.load(ROOT / '.cache/ga16_min5_dip_tests/labels.npz')
    np.testing.assert_array_equal(reference['source_ids'], source_ids)
    old = reference['fresh_min5']
    original = np.load(ROOT / '.cache/ga16_shape_implementation_tests/dip_only/real_inf.npz')['labels']
    example = np.flatnonzero(original == 3)
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100*app.normalize_block(p[:, 6:7])))
    config = dict(np=np, n_kept=n)
    names = {'STD_LIMITS', 'MAX_SIGMA', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS', 'DIP_ALPHA',
             'GAP_FRACTION', 'STRONG_GAP_FRACTION', 'GAUSSIAN_QQ_MAX'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), config)
    assert config['MIN_CLUSTER_SIZE'] == 5
    assert np.isinf(config['GAUSSIAN_QQ_MAX'])
    new_shape = app.ClusterShapeConstraint(p[:, :3], config['STD_LIMITS'], dip_alpha=config['DIP_ALPHA'],
                                          gap_fraction=config['GAP_FRACTION'],
                                          strong_gap_fraction=config['STRONG_GAP_FRACTION'])
    legacy = app.ClusterShapeConstraint(p[:, :3], config['STD_LIMITS'], strong_gap_fraction=np.inf)
    assert legacy(example) and not new_shape(example)
    cut = new_shape.split(example)
    assert sorted(np.bincount(cut).tolist()) == [6, 28]
    validator = app.make_cluster_validator(p[:, :3], f, config['STD_LIMITS'], 5, config['MAX_SIGMA'],
                                          directions=p[:, 6], shape_condition=new_shape)
    assert all(validator(example[cut == c]) for c in (0, 1))
    # Regression of the actual split and fusion, including the rejected original parent.
    example_labels = np.full(n, -1, dtype=int)
    example_labels[example] = 0
    candidates = []
    split = app.split_messy_clusters(example_labels, p[:, :3], config['STD_LIMITS'], 5, 1003,
                                    features=f, sigma_limit=3, group_validator=validator,
                                    gap_splitter=new_shape.split, candidate_groups=candidates)
    fused, _ = app.fuse_cluster_candidates(candidates, n, validator, 1003, previous_labels=split)
    assert sorted(len(ids) for ids in app.cluster_groups(fused)) == [6, 28]
    assert np.all(fused[example] >= 0)
    roots = []
    for count in (99, 180):
        root = np.load(ROOT / f'.cache/ga16_sigma_recovery_tests/sigma_3/base_{count}.npz')
        np.testing.assert_array_equal(root['source_ids'], source_ids)
        roots.append(root['labels'])
    start = time.perf_counter()
    labels, _, _ = app.refine_and_fuse_clusters(roots, p[:, :3], f, p[:, 6], config['STD_LIMITS'],
                                               5, config['MAX_CLUSTERS'], config['MAX_SIGMA'],
                                               shape_condition=new_shape)
    seconds = time.perf_counter()-start
    groups = app.cluster_groups(labels)
    assert len(groups) <= config['MAX_CLUSTERS']
    assert all(validator(ids) for ids in groups)
    assert np.all(labels[example] >= 0)
    # The two populations must not be reunited, even as part of a larger final candidate.
    assert not set(labels[example[cut == 0]]).intersection(labels[example[cut == 1]])
    report = dict(n=n, minimum=5, sigma=3, strong_gap_fraction=config['STRONG_GAP_FRACTION'],
                  before=dict(clusters=len(app.cluster_groups(old)), excluded=int(np.sum(old < 0))),
                  after=dict(clusters=len(groups), excluded=int(np.sum(labels < 0)), seconds=seconds,
                             lost=int(np.sum((old >= 0) & (labels < 0))),
                             recovered=int(np.sum((old < 0) & (labels >= 0))),
                             minimum_size=min(map(len, groups))),
                  example=dict(n=34, sizes=np.bincount(cut).tolist(),
                               pvalue=float(diptest(p[example, 0])[1]),
                               std=app.physical_std(p[example, :3]).tolist(),
                               new_labels={str(int(k)): int(np.sum(labels[example] == k))
                                           for k in np.unique(labels[example])}))
    # Check how often the new fallback alone divides ordinary single clouds.
    rng = np.random.default_rng(34)
    synthetic = []
    for count in (10, 34, 100):
        for kind in ('normal', 'line', 'arc'):
            old_flags = new_flags = 0
            for repeat in range(100):
                noise = rng.normal(size=(count, 3))
                t = rng.uniform(-1, 1, count)
                x = noise if kind == 'normal' else np.column_stack((t, .5*t, -.2*t)) + .01*noise
                if kind == 'arc':
                    x = np.column_stack((t, t*t, .4*t*t*t)) + .01*noise
                values = x*np.array([.01, .025, 4.]) + [-.6, .3, 180.]
                before = app.ClusterShapeConstraint(values, [.04, .05, 10], strong_gap_fraction=np.inf)
                after = app.ClusterShapeConstraint(values, [.04, .05, 10])
                ids = np.arange(count)
                old_flags += int(not before(ids))
                new_flags += int(not after(ids))
            synthetic.append(dict(n=count, geometry=kind, trials=100, before=old_flags, after=new_flags))
    report['synthetic'] = synthetic
    (HERE / 'report.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(HERE / 'labels.npz', source_ids=source_ids, labels=labels, example=example, cut=cut)
    plot_example(p, example, cut)
    print('RESULT', json.dumps(report), flush=True)


def plot_example(p, ids, cut):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].scatter(p[ids, 0], p[ids, 1], s=18)
    axes[0].set_title('Before: 34 trajectories in one cluster')
    for side in (0, 1):
        chosen = ids[cut == side]
        axes[1].scatter(p[chosen, 0], p[chosen, 1], s=18, label=f'{len(chosen)} trajectories')
    axes[1].set_title('Gap split: 28 + 6, no points lost')
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel('eps [km^2/s^2]')
        ax.set_ylabel('e')
        ax.grid(alpha=.25)
    fig.savefig(HERE / 'cluster3_before_after.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
