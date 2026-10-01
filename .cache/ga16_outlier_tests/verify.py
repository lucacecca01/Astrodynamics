import ast
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ga16-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
import numpy as np
import matplotlib.pyplot as plt
from test_fusion import app, SOURCE


def plot_example(p, ids, labels, name):
    a = p[ids, :3].copy()
    origin = np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(a[:, 2])))))
    a[:, 2] = (a[:, 2] - origin + 180) % 360 - 180
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    names = ['eps [km^2/s^2]', 'e', 'delta omega [deg]']
    for col, (x, y) in enumerate(((0, 1), (0, 2), (1, 2))):
        axes[0, col].scatter(a[:, x], a[:, y], s=16)
        axes[0, col].set_title(f'Before: {len(ids)} trajectories')
        for group in np.unique(labels[ids]):
            mask = labels[ids] == group
            kw = dict(c='red', marker='x', label=f'Unassigned: {mask.sum()}') if group < 0 else {
                'label': f'Group: {mask.sum()}'}
            axes[1, col].scatter(a[mask, x], a[mask, y], s=20, **kw)
        axes[1, col].legend(fontsize=8)
        for row in range(2):
            axes[row, col].set_xlabel(names[x])
            axes[row, col].set_ylabel(names[y])
            axes[row, col].grid(alpha=.25)
    fig.savefig(HERE / f'{name}.png', dpi=150)
    plt.close(fig)


saved = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
p, source_ids = saved['features'], saved['source_ids']
w = np.deg2rad(p[:, 2])
f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                     app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                     100*app.normalize_block(p[:, 6:7])))
names = {'STD_LIMITS', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS', 'MAX_SIGMA', 'DIP_ALPHA',
         'GAP_FRACTION', 'STRONG_GAP_FRACTION', 'OUTLIER_SIGMA', 'GAUSSIAN_QQ_MAX'}
config = dict(np=np, n_kept=len(p))
nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), config)
minimum, cap, limits = config['MIN_CLUSTER_SIZE'], config['MAX_CLUSTERS'], config['STD_LIMITS']
shape = app.ClusterShapeConstraint(p[:, :3], limits, dip_alpha=config['DIP_ALPHA'],
                                   gap_fraction=config['GAP_FRACTION'],
                                   strong_gap_fraction=config['STRONG_GAP_FRACTION'],
                                   outlier_sigma=config['OUTLIER_SIGMA'],
                                   gaussian_qq_max=config['GAUSSIAN_QQ_MAX'])
valid = app.make_cluster_validator(p[:, :3], f, limits, minimum, config['MAX_SIGMA'],
                                  directions=p[:, 6], shape_condition=shape)
reference = np.load(ROOT / '.cache/ga16_shape_implementation_tests/dip_only/real_inf.npz')
np.testing.assert_array_equal(reference['source_ids'], source_ids)
examples = [np.flatnonzero(reference['labels'] == c) for c in (9, 24)]
assert list(map(len, examples)) == [36, 90]
report = dict(minimum=int(minimum), examples=[])
for name, ids in zip(('cluster9', 'cluster18'), examples):
    initial = np.full(len(p), -1, dtype=int)
    initial[ids] = 0
    pool = []
    local = app.split_messy_clusters(initial, p[:, :3], limits, minimum, cap,
                                    features=f, sigma_limit=config['MAX_SIGMA'], group_validator=valid,
                                    gap_splitter=shape.split, candidate_groups=pool)
    local, _ = app.fuse_cluster_candidates(pool, len(p), valid, cap, previous_labels=local)
    assert all(valid(g) for g in app.cluster_groups(local))
    report['examples'].append(dict(name=name, groups=[len(g) for g in app.cluster_groups(local)],
                                   excluded=int(np.sum(local[ids] < 0))))
    plot_example(p, ids, local, f'{name}_min{minimum}_local')

roots = []
for count in (99, 180):
    root = np.load(ROOT / f'.cache/ga16_sigma_recovery_tests/sigma_3/base_{count}.npz')
    np.testing.assert_array_equal(root['source_ids'], source_ids)
    roots.append(root['labels'])
database = np.loadtxt(ROOT / 'CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP.txt', skiprows=1)
fingerprint = app.fusion_fingerprint(source_ids, database[source_ids], p, f)
key = app.fusion_fingerprint(
    np.r_[limits, minimum, cap, config['MAX_SIGMA'], config['DIP_ALPHA'], config['GAP_FRACTION'],
          config['STRONG_GAP_FRACTION'], config['OUTLIER_SIGMA'], shape.qq_max],
    np.frombuffer(b'dip-gap-qq-v3-outliers', dtype=np.uint8))
cache_path = SOURCE.parent / 'Clusters/GA_16_dip_gap_candidates_sigma3.npz'
before_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
candidates, previous = app.load_fusion_cache(cache_path, fingerprint, len(p), constraints_key=key)
assert previous is None
for name, pool in (('fresh', []), ('cached_candidates', candidates)):
    labels, _, _ = app.refine_and_fuse_clusters(
        roots, p[:, :3], f, p[:, 6], limits, minimum, cap, config['MAX_SIGMA'],
        candidates=pool, shape_condition=shape)
    groups = app.cluster_groups(labels)
    assert all(valid(g) for g in groups)
    assert len(groups) <= cap
    record = dict(clusters=len(groups), excluded=int(np.sum(labels < 0)), minimum=min(map(len, groups)))
    record['examples'] = []
    for title, ids in zip(('cluster9', 'cluster18'), examples):
        keys, counts = np.unique(labels[ids], return_counts=True)
        record['examples'].append(dict(name=title, new_labels=keys.tolist(), points_per_label=counts.tolist()))
        plot_example(p, ids, labels, f'{title}_min{minimum}_{name}')
    report[name] = record
    np.savez_compressed(HERE / f'production_min{minimum}_{name}.npz', labels=labels, source_ids=source_ids)
assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == before_hash
(HERE / f'production_min{minimum}.json').write_text(json.dumps(report, indent=2))
print('VERIFIED', json.dumps(report), flush=True)
