import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
from test_fusion import app, SOURCE
from real_data import audit

HERE = Path(__file__).resolve().parent
CACHE = ROOT / '.cache/ga16_sigma_recovery_tests'


def initial_partitions(sigma, source_ids):
    if sigma == 2.5:
        return list(np.load(ROOT / '.cache/ga16_fusion_implementation_tests/actual_initial.npz')['partitions'])
    partitions = []
    for count in (99, 180):
        data = np.load(CACHE / f'sigma_{sigma:g}/base_{count}.npz')
        np.testing.assert_array_equal(data['source_ids'], source_ids)
        partitions.append(data['labels'])
    return partitions


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data = np.load(CACHE / 'features.npz')
    p, source_ids = data['features'], data['source_ids']
    n = len(p)
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100 * app.normalize_block(p[:, 6:7])))
    config = dict(np=np, n_kept=n, Path=Path, __file__=str(SOURCE))
    names = {'STD_LIMITS', 'MAX_SIGMA', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTERS', 'FUSION_CACHE_FILE'}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), config)
    database = np.atleast_2d(np.loadtxt(app.DATA_FILE, skiprows=1))
    fingerprint = app.fusion_fingerprint(source_ids, database[source_ids], p, f)
    candidates, previous = app.load_fusion_cache(config['FUSION_CACHE_FILE'], fingerprint, n)
    if not candidates:
        candidates, previous = app.load_fusion_cache(
            ROOT / '.cache/ga16_fusion_implementation_tests/roundtrip.npz',
            app.fusion_fingerprint(source_ids, p, f), n)
    original_limits = config['STD_LIMITS']
    valid_one = app.make_cluster_validator(p[:, :3], f, original_limits, 10, 1., directions=p[:, 6])
    sigma_one = dict(sigma=1., candidates=len(candidates),
                     valid_candidates=int(sum(valid_one(ids) for ids in candidates)),
                     previous_valid_groups=int(sum(valid_one(ids) for ids in app.cluster_groups(previous))))
    print('SIGMA_ONE_POOL_CHECK', json.dumps(sigma_one), flush=True)
    (HERE / 'sigma_one.json').write_text(json.dumps(sigma_one, indent=2))

    rows, assignments = [], []
    cases = [(2.5, 1.), (3., 1.), (4., 1.), (4., 1.1), (4., 1.25), (4., 1.5), (4., 2.)]
    for sigma, scale in cases:
        limits = original_limits * scale
        validator = app.make_cluster_validator(p[:, :3], f, limits, 10, sigma, directions=p[:, 6])
        if previous is not None and not all(validator(ids) for ids in app.cluster_groups(previous)):
            raise RuntimeError('Prior partition would violate these test constraints.')
        start = time.perf_counter()
        labels, baseline, candidates = app.refine_and_fuse_clusters(
            initial_partitions(sigma, source_ids), p[:, :3], f, p[:, 6], limits, 10,
            config['MAX_CLUSTERS'], sigma, candidates=candidates, previous_labels=previous,
            seeds=(42, 7, 123), base_count=180)
        elapsed = time.perf_counter()-start
        protected = previous >= 0 if previous is not None else baseline >= 0
        metrics = audit(labels, p, f, limits, sigma, 10, config['MAX_CLUSTERS'], protected)
        coverage = np.zeros(n, dtype=bool)
        valid_candidates = 0
        for ids in candidates:
            if validator(ids):
                coverage[ids] = True
                valid_candidates += 1
        row = dict(sigma=sigma, std_scale=scale, std_eps=limits[0], std_e=limits[1], std_w=limits[2],
                    **metrics, unassigned_percent=100*metrics['unassigned']/n,
                    candidate_uncovered=int(np.sum(~coverage)),
                    valid_candidates=valid_candidates, seconds=elapsed)
        if metrics['unassigned'] and sigma == 2.5:
            try:
                forced, _ = app.fuse_cluster_candidates(
                    candidates, n, validator, config['MAX_CLUSTERS'],
                    previous_labels=np.zeros(n, dtype=int), time_limit=30.)
                assert np.all(forced >= 0)
                row['forced_full_coverage'] = 'feasible'
            except RuntimeError as error:
                row['forced_full_coverage'] = str(error)
        else:
            row['forced_full_coverage'] = 'feasible' if not metrics['unassigned'] else 'not_tested'
        rows.append(row)
        assignments.append(labels.copy())
        (HERE / 'report.json').write_text(json.dumps(dict(
            n=n, minimum=10, max_clusters=config['MAX_CLUSTERS'], sigma_one=sigma_one, cases=rows), indent=2))
        with (HERE / 'comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(HERE / 'labels.npz', source_ids=source_ids, labels=assignments)
        print('RESULT', json.dumps(row), flush=True)
        previous = labels
        if metrics['unassigned'] == 0:
            break
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    print('DONE: GA16 and its production cache unchanged', flush=True)


if __name__ == '__main__':
    main()
