import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ga16-mpl')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--min-size', type=int, choices=(1, 10), default=10)
args = parser.parse_args()
REFERENCE = ROOT / f'.cache/ga16_std_min{args.min_size}_tests'
if args.min_size != 10:
    HERE = ROOT / f'.cache/ga16_unprotected_min{args.min_size}_tests'
HERE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
import numpy as np
from test_fusion import app, SOURCE

source = SOURCE.read_bytes()
config = json.loads((REFERENCE / 'settings.json').read_text())
with np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz') as data:
    p = data['features'].copy()
    source_ids = data['source_ids'].copy()
n = len(p)
w = np.deg2rad(p[:, 2])
f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                     app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                     100 * app.normalize_block(p[:, 6:7])))
data_key = app.fusion_fingerprint(source_ids, p, f)
assert data_key == config['data_key']
assert config['MIN_CLUSTER_SIZE'] == args.min_size
settings = dict(reference=config, source_sha256=hashlib.sha256(source).hexdigest(),
                experiment='Same saved candidate pools and constraints; no protected assignments')
settings_path = HERE / 'settings.json'
if settings_path.exists():
    assert json.loads(settings_path.read_text()) == settings
else:
    settings_path.write_text(json.dumps(settings, indent=2))
with np.load(ROOT / '.cache/ga16_maxsigma_outlier_tests/maxsigma_2.5/base_180.npz') as data:
    np.testing.assert_array_equal(data['source_ids'], source_ids)
    base_labels = data['labels'].copy()

original_milp = app.milp
solver_reports = []


def recorded_milp(*args, **kwargs):
    result = original_milp(*args, **kwargs)
    solver_reports.append(dict(status=int(result.status), message=result.message,
                               gap=getattr(result, 'mip_gap', None),
                               objective=result.fun, dual_bound=getattr(result, 'mip_dual_bound', None)))
    return result


app.milp = recorded_milp
report = []
for increase in (0, 10, 20, 50, 100):
    directory = HERE / f'increase_{increase}'
    directory.mkdir(exist_ok=True)
    row_path = directory / 'report.json'
    if row_path.exists():
        report.append(json.loads(row_path.read_text()))
        print('RESTORED', json.dumps(report[-1]), flush=True)
        continue
    reference = REFERENCE / f'increase_{increase}'
    with np.load(reference / 'result.npz') as data:
        pool = [ids.copy() for ids in np.split(data['members'], data['offsets'][1:-1])]
        before = data['labels'].copy()
    limits = np.asarray(config['STD_LIMITS']) * (1 + increase / 100)
    shape = app.ClusterShapeConstraint(
        p[:, :3], limits, dip_alpha=config['DIP_ALPHA'], gap_fraction=config['GAP_FRACTION'],
        strong_gap_fraction=config['STRONG_GAP_FRACTION'], outlier_sigma=config['OUTLIER_SIGMA'],
        gaussian_qq_max=config['GAUSSIAN_QQ_MAX'])
    valid = app.make_cluster_validator(p[:, :3], f, limits, config['MIN_CLUSTER_SIZE'],
                                      config['MAX_SIGMA'], directions=p[:, 6], shape_condition=shape)
    before_groups = app.cluster_groups(before)
    assert all(valid(ids) for ids in before_groups)
    split_key = app.fusion_fingerprint(
        np.frombuffer(data_key.encode(), dtype=np.uint8), base_labels, limits,
        np.r_[config['MIN_CLUSTER_SIZE'], n, config['MAX_SIGMA'], 42], np.asarray([]))
    with np.load(reference / f'split_{split_key}.npz') as data:
        baseline = data['labels'].copy()
    protected = baseline >= 0
    assert np.all(before[protected] >= 0)

    print(f'\nSTD +{increase}%: {limits}; {len(pool)} same candidates', flush=True)
    start = time.perf_counter()
    solver_reports.clear()
    labels, returned_pool = app.fuse_cluster_candidates(
        pool, n, valid, config['MAX_CLUSTERS'], previous_labels=None,
        time_limit=config['FUSION_TIME_LIMIT'])
    elapsed = time.perf_counter() - start
    assert len(solver_reports) == 1
    assert solver_reports[0]['status'] == 0 and solver_reports[0]['gap'] == 0
    assert {ids.tobytes() for ids in pool} == {ids.tobytes() for ids in returned_pool}
    groups = app.cluster_groups(labels)
    assert len(groups) <= config['MAX_CLUSTERS'] and all(valid(ids) for ids in groups)
    sizes = np.array(list(map(len, groups)))
    before_excluded = int(np.sum(before < 0))
    excluded = int(np.sum(labels < 0))
    assert excluded <= before_excluded
    assert excluded < before_excluded or len(groups) <= len(before_groups)
    assert sizes.min() >= config['MIN_CLUSTER_SIZE'] and sizes.sum() + excluded == n
    with np.load(ROOT / f'.cache/ga16_std_min1_tests/increase_{increase}/labels.npz') as data:
        trimmed = data['labels'].copy()
    trimmed_groups = [ids for ids in app.cluster_groups(trimmed) if len(ids) >= config['MIN_CLUSTER_SIZE']]
    assert all(valid(ids) for ids in trimmed_groups)
    pool_keys = {ids.tobytes() for ids in pool}
    assert all(ids.tobytes() in pool_keys for ids in trimmed_groups)
    trimmed_excluded = n - sum(map(len, trimmed_groups))
    assert excluded <= trimmed_excluded
    assert excluded < trimmed_excluded or len(groups) <= len(trimmed_groups)
    stds = np.array([app.physical_std(p[ids, :3]) for ids in groups])
    row = dict(increase_percent=increase, std_limits=limits.tolist(),
               before_clusters=len(before_groups), clusters=len(groups),
               before_excluded=before_excluded, excluded=excluded, excluded_percent=100*excluded/n,
               newly_assigned=int(np.sum((before < 0) & (labels >= 0))),
               newly_excluded=int(np.sum((before >= 0) & (labels < 0))),
               previously_protected_lost=int(np.sum(protected & (labels < 0))),
               minimum_size=int(sizes.min()), singletons=int(np.sum(sizes == 1)),
               clusters_under_10=int(np.sum(sizes < 10)), mean_std=stds.mean(axis=0).tolist(),
               maximum_std=stds.max(axis=0).tolist(), seconds=elapsed, solver=solver_reports[0])
    np.savez_compressed(directory / 'labels.npz', labels=labels, source_ids=source_ids)
    row_path.write_text(json.dumps(row, indent=2))
    report.append(row)
    (HERE / 'comparison.json').write_text(json.dumps(report, indent=2))
    with (HERE / 'comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(report)
    print('RESULT', json.dumps(row), flush=True)
print('SOURCE_UNCHANGED', SOURCE.read_bytes() == source, flush=True)
