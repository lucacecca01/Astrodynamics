import ast
import csv
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

source = SOURCE.read_bytes()
data = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
p, ids = data['features'], data['source_ids']
n = len(p)
w = np.deg2rad(p[:, 2])
f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                     app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                     100*app.normalize_block(p[:, 6:7])))
names = {'STD_LIMITS', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS', 'BASE_MIN_CLUSTER_SIZE', 'DIP_ALPHA',
         'GAP_FRACTION', 'STRONG_GAP_FRACTION', 'OUTLIER_SIGMA', 'GAUSSIAN_QQ_MAX',
         'BASE_CLUSTER_COUNTS', 'FUSION_SEEDS', 'FUSION_TIME_LIMIT', 'EXTRA_CLUSTER_CONDITION'}
c = dict(np=np, n_kept=n)
nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), c)
assert c['EXTRA_CLUSTER_CONDITION'] is None
settings = {key: (c[key].tolist() if isinstance(c[key], np.ndarray) else c[key]) for key in names}
settings.update(n_points=n, source_sha256=hashlib.sha256(source).hexdigest(), fusion_cache=False)
settings = json.loads(json.dumps(settings))
settings_file = HERE / 'settings.json'
if settings_file.exists():
    assert json.loads(settings_file.read_text()) == settings, 'Parameters changed: use a fresh test directory.'
else:
    settings_file.write_text(json.dumps(settings, indent=2))
print('SETTINGS', json.dumps(settings), flush=True)
data_key = app.fusion_fingerprint(ids, p, f)
original_split = app.split_messy_clusters


def checkpoint_split(*args, **kwargs):
    scale = kwargs.get('split_scale')
    key = app.fusion_fingerprint(
        np.frombuffer(data_key.encode(), dtype=np.uint8), args[0], np.asarray(args[2]),
        np.r_[args[3], args[4], kwargs['sigma_limit'], kwargs.get('random_state', 42)],
        np.asarray([] if scale is None else scale))
    path = directory / f'split_{key}.npz'
    if path.exists():
        added, result = app.load_fusion_cache(path, key, n, constraints_key=key)
        kwargs['candidate_groups'].extend(added)
        print('SPLIT CHECKPOINT RESTORED', path.name, flush=True)
        return result
    pool = kwargs['candidate_groups']
    start = len(pool)
    result = original_split(*args, **kwargs)
    app.save_fusion_cache(path, key, pool[start:], result, constraints_key=key)
    print('SPLIT CHECKPOINT SAVED', path.name, flush=True)
    return result


app.split_messy_clusters = checkpoint_split
report_file = HERE / 'comparison.json'
report = json.loads(report_file.read_text()) if report_file.exists() else []
for sigma in (2., 2.5, 1.5):
    if any(row['max_sigma'] == sigma for row in report):
        print('COMPLETED RESULT RESTORED', sigma, flush=True)
        continue
    print(f'\nTEST MAX_SIGMA={sigma:g}', flush=True)
    start = time.perf_counter()
    partitions, labels = [], None
    directory = HERE / f'maxsigma_{sigma:g}'
    directory.mkdir(exist_ok=True)
    for count in c['BASE_CLUSTER_COUNTS']:
        base_path = directory / f'base_{count}.npz'
        if base_path.exists():
            with np.load(base_path) as cache:
                np.testing.assert_array_equal(cache['source_ids'], ids)
                restored = cache['labels']
            partitions.append(restored)
            good = all(len(g) >= c['BASE_MIN_CLUSTER_SIZE'] and app.within_sigma(f[g], sigma)
                       for g in app.cluster_groups(restored))
            labels = restored.copy() if good else None
            print('BASE CHECKPOINT RESTORED', count, flush=True)
            continue
        try:
            labels = app.constrained_clustering(f, p[:, 6], count, min_size=c['BASE_MIN_CLUSTER_SIZE'],
                                                sigma_limit=sigma, previous_labels=labels)
            partitions.append(labels.copy())
        except app.NoFeasiblePartitionError as exc:
            print(f'Base fallback: {exc}', flush=True)
            partitions.append(app.kmeans_partition(f, p[:, 6], count))
            labels = None
        np.savez_compressed(base_path, labels=partitions[-1], source_ids=ids)
    shape = app.ClusterShapeConstraint(p[:, :3], c['STD_LIMITS'], dip_alpha=c['DIP_ALPHA'],
                                       gap_fraction=c['GAP_FRACTION'], outlier_sigma=c['OUTLIER_SIGMA'],
                                       strong_gap_fraction=c['STRONG_GAP_FRACTION'],
                                       gaussian_qq_max=c['GAUSSIAN_QQ_MAX'])
    labels, baseline, pool = app.refine_and_fuse_clusters(
        partitions, p[:, :3], f, p[:, 6], c['STD_LIMITS'], c['MIN_CLUSTER_SIZE'], c['MAX_CLUSTERS'], sigma,
        seeds=c['FUSION_SEEDS'], base_count=c['BASE_CLUSTER_COUNTS'][-1],
        time_limit=c['FUSION_TIME_LIMIT'], shape_condition=shape)
    valid = app.make_cluster_validator(p[:, :3], f, c['STD_LIMITS'], c['MIN_CLUSTER_SIZE'], sigma,
                                       directions=p[:, 6], shape_condition=shape)
    groups = app.cluster_groups(labels)
    assert len(groups) <= c['MAX_CLUSTERS'] and all(valid(g) for g in groups)
    excluded = int(np.sum(labels < 0))
    assert sum(map(len, groups)) + excluded == n
    row = dict(max_sigma=sigma, clusters=len(groups), excluded=excluded, percent=100*excluded/n,
               minimum_size=min(map(len, groups), default=0), seconds=time.perf_counter()-start)
    np.savez_compressed(directory / 'labels.npz', labels=labels, source_ids=ids)
    report.append(row)
    report.sort(key=lambda entry: entry['max_sigma'])
    (HERE / 'comparison.json').write_text(json.dumps(report, indent=2))
    with (HERE / 'comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(report)
    print('RESULT', json.dumps(row), flush=True)
print('SOURCE_UNCHANGED', SOURCE.read_bytes() == source, flush=True)
