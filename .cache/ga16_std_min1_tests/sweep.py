import argparse
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
parser = argparse.ArgumentParser()
parser.add_argument('--min-size', type=int, default=1)
parser.add_argument('--std-limits', type=float, nargs=3)
parser.add_argument('--percentages', type=int, nargs='+')
args = parser.parse_args()
assert args.min_size >= 1
percentages = (0, 10, 20, 50, 100) if args.percentages is None else tuple(args.percentages)
assert percentages and min(percentages) > -100
assert list(percentages) == sorted(set(percentages))
HERE = ROOT / f'.cache/ga16_std_min{args.min_size}_tests'
HERE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
import numpy as np
from test_fusion import app, SOURCE

source = SOURCE.read_bytes()
saved = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
p, source_ids = saved['features'], saved['source_ids']
n = len(p)
w = np.deg2rad(p[:, 2])
f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                     app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                     100*app.normalize_block(p[:, 6:7])))
names = {'STD_LIMITS', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS', 'MAX_SIGMA', 'BASE_MIN_CLUSTER_SIZE',
         'DIP_ALPHA', 'GAP_FRACTION', 'STRONG_GAP_FRACTION', 'OUTLIER_SIGMA', 'GAUSSIAN_QQ_MAX',
         'BASE_CLUSTER_COUNTS', 'FUSION_SEEDS', 'FUSION_TIME_LIMIT', 'EXTRA_CLUSTER_CONDITION'}
c = dict(np=np, n_kept=n)
nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), c)
c['MIN_CLUSTER_SIZE'] = args.min_size
if args.std_limits is not None:
    c['STD_LIMITS'] = np.asarray(args.std_limits)
assert c['EXTRA_CLUSTER_CONDITION'] is None
data_key = app.fusion_fingerprint(source_ids, p, f)
settings = {name: (c[name].tolist() if isinstance(c[name], np.ndarray) else c[name]) for name in names}
settings.update(source_sha256=hashlib.sha256(source).hexdigest(), data_key=data_key, n_points=n,
                cache_policy='Empty initial pool; reuse candidates as STD limits increase; no production cache writes')
if args.percentages is not None:
    settings['percentages'] = list(percentages)
settings = json.loads(json.dumps(settings))
settings_path = HERE / 'settings.json'
if settings_path.exists():
    assert json.loads(settings_path.read_text()) == settings, 'Test parameters changed.'
else:
    settings_path.write_text(json.dumps(settings, indent=2))
print('SETTINGS', json.dumps(settings), flush=True)

# Initial partitions depend on MAX_SIGMA and BASE_MIN_CLUSTER_SIZE, not physical STD limits.
partitions = []
for count in c['BASE_CLUSTER_COUNTS']:
    path = ROOT / f'.cache/ga16_maxsigma_outlier_tests/maxsigma_{c["MAX_SIGMA"]:g}/base_{count}.npz'
    with np.load(path) as base:
        np.testing.assert_array_equal(base['source_ids'], source_ids)
        labels = base['labels'].copy()
    assert np.all(labels >= 0)
    assert all(len(ids) >= c['BASE_MIN_CLUSTER_SIZE'] and app.within_sigma(f[ids], c['MAX_SIGMA'])
               and np.all(p[ids, 6] == p[ids[0], 6]) for ids in app.cluster_groups(labels))
    partitions.append(labels)

original_split = app.split_messy_clusters
original_milp = app.milp
solver_reports = []


def recorded_milp(*args, **kwargs):
    result = original_milp(*args, **kwargs)
    solver_reports.append(dict(status=int(result.status), message=result.message,
                               gap=getattr(result, 'mip_gap', None)))
    return result


def checkpoint_split(*args, **kwargs):
    scale = kwargs.get('split_scale')
    key = app.fusion_fingerprint(np.frombuffer(data_key.encode(), dtype=np.uint8), args[0], args[2],
                                 np.r_[args[3], args[4], kwargs['sigma_limit'], kwargs.get('random_state', 42)],
                                 np.asarray([] if scale is None else scale))
    path = directory / f'split_{key}.npz'
    if path.exists():
        groups, labels = app.load_fusion_cache(path, key, n, constraints_key=key)
        kwargs['candidate_groups'].extend(groups)
        return labels
    pool = kwargs['candidate_groups']
    start = len(pool)
    labels = original_split(*args, **kwargs)
    app.save_fusion_cache(path, key, pool[start:], labels, constraints_key=key)
    return labels


app.split_messy_clusters = checkpoint_split
app.milp = recorded_milp
pool, previous, report = [], None, []
for increase in percentages:
    factor = 1 + increase / 100
    limits = c['STD_LIMITS'] * factor
    directory = HERE / f'increase_{increase}'
    directory.mkdir(exist_ok=True)
    key = app.fusion_fingerprint(np.frombuffer(data_key.encode(), dtype=np.uint8), limits)
    result_path = directory / 'result.npz'
    row_path = directory / 'report.json'
    if result_path.exists() and row_path.exists():
        pool, previous = app.load_fusion_cache(result_path, key, n, constraints_key=key)
        report.append(json.loads(row_path.read_text()))
        print('RESULT RESTORED', report[-1], flush=True)
        continue
    print(f'\nSTD {increase:+d}%: {limits}', flush=True)
    shape = app.ClusterShapeConstraint(p[:, :3], limits, dip_alpha=c['DIP_ALPHA'],
                                       gap_fraction=c['GAP_FRACTION'],
                                       strong_gap_fraction=c['STRONG_GAP_FRACTION'],
                                       outlier_sigma=c['OUTLIER_SIGMA'], gaussian_qq_max=c['GAUSSIAN_QQ_MAX'])
    validator = app.make_cluster_validator(p[:, :3], f, limits, c['MIN_CLUSTER_SIZE'], c['MAX_SIGMA'],
                                         directions=p[:, 6], shape_condition=shape)
    if previous is not None:
        assert all(validator(ids) for ids in app.cluster_groups(previous))
    start = time.perf_counter()
    solver_reports.clear()
    labels, baseline, pool = app.refine_and_fuse_clusters(
        partitions, p[:, :3], f, p[:, 6], limits, c['MIN_CLUSTER_SIZE'], c['MAX_CLUSTERS'], c['MAX_SIGMA'],
        seeds=c['FUSION_SEEDS'], base_count=c['BASE_CLUSTER_COUNTS'][-1],
        candidates=pool, time_limit=c['FUSION_TIME_LIMIT'], shape_condition=shape)
    assert len(solver_reports) == 1
    assert solver_reports[0]['status'] == 0 and solver_reports[0]['gap'] == 0
    groups = app.cluster_groups(labels)
    assert len(groups) <= c['MAX_CLUSTERS'] and all(validator(ids) for ids in groups)
    sizes = np.array(list(map(len, groups)))
    excluded = int(np.sum(labels < 0))
    assert int(sizes.sum()) + excluded == n
    maximum_std = np.max([app.physical_std(p[ids, :3]) for ids in groups], axis=0)
    row = dict(increase_percent=increase, std_eps=float(limits[0]), std_e=float(limits[1]),
               std_omega=float(limits[2]), clusters=len(groups), excluded=excluded,
               excluded_percent=100*excluded/n, singletons=int(np.sum(sizes == 1)),
               clusters_under_10=int(np.sum(sizes < 10)), minimum_size=int(sizes.min()),
               seconds=time.perf_counter()-start,
               maximum_std=maximum_std.tolist(), solver=solver_reports[0])
    app.save_fusion_cache(result_path, key, pool, labels, constraints_key=key)
    np.savez_compressed(directory / 'labels.npz', labels=labels, source_ids=source_ids)
    row_path.write_text(json.dumps(row, indent=2))
    previous = labels
    report.append(row)
    (HERE / 'comparison.json').write_text(json.dumps(report, indent=2))
    with (HERE / 'comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(report)
    print('RESULT', json.dumps(row), flush=True)
print('SOURCE_UNCHANGED', SOURCE.read_bytes() == source, flush=True)
