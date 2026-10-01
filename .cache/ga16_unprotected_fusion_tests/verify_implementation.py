import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ga16-mpl')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
import numpy as np
from test_fusion import app, SOURCE

source = SOURCE.read_bytes()
with np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz') as saved:
    p, source_ids = saved['features'], saved['source_ids']
n = len(p)
w = np.deg2rad(p[:, 2])
f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                     app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                     100 * app.normalize_block(p[:, 6:7])))
data_key = app.fusion_fingerprint(source_ids, p, f)
with np.load(ROOT / '.cache/ga16_maxsigma_outlier_tests/maxsigma_2.5/base_180.npz') as saved:
    base = saved['labels']
    np.testing.assert_array_equal(source_ids, saved['source_ids'])

original_milp = app.milp


def optimal_milp(*args, **kwargs):
    result = original_milp(*args, **kwargs)
    assert result.status == 0 and result.mip_gap == 0
    return result


app.milp = optimal_milp
rows = []
for minimum in (1, 10):
    reference = ROOT / f'.cache/ga16_std_min{minimum}_tests'
    expected_root = HERE if minimum == 10 else ROOT / '.cache/ga16_unprotected_min1_tests'
    config = json.loads((reference / 'settings.json').read_text())
    assert config['data_key'] == data_key
    for increase in (0, 10, 20, 50, 100):
        limits = np.asarray(config['STD_LIMITS']) * (1 + increase / 100)
        directory = reference / f'increase_{increase}'
        with np.load(directory / 'result.npz') as saved:
            pool = [ids.copy() for ids in np.split(saved['members'], saved['offsets'][1:-1])]
        key = app.fusion_fingerprint(np.frombuffer(data_key.encode(), dtype=np.uint8), base, limits,
                                     np.r_[minimum, n, config['MAX_SIGMA'], 42], np.asarray([]))
        with np.load(directory / f'split_{key}.npz') as saved:
            baseline = saved['labels']
        shape = app.ClusterShapeConstraint(
            p[:, :3], limits, dip_alpha=config['DIP_ALPHA'], gap_fraction=config['GAP_FRACTION'],
            strong_gap_fraction=config['STRONG_GAP_FRACTION'], outlier_sigma=config['OUTLIER_SIGMA'],
            gaussian_qq_max=config['GAUSSIAN_QQ_MAX'])
        valid = app.make_cluster_validator(p[:, :3], f, limits, minimum, config['MAX_SIGMA'],
                                          directions=p[:, 6], shape_condition=shape)
        labels, _ = app.fuse_cluster_candidates(pool, n, valid, config['MAX_CLUSTERS'],
                                                previous_labels=baseline, time_limit=30.)
        groups = app.cluster_groups(labels)
        assert all(valid(ids) for ids in groups)
        expected = json.loads((expected_root / f'increase_{increase}/report.json').read_text())
        excluded = int(np.sum(labels < 0))
        assert (len(groups), excluded) == (expected['clusters'], expected['excluded'])
        row = dict(minimum=minimum, increase=increase, clusters=len(groups), excluded=excluded)
        rows.append(row)
        print('PASS', row, flush=True)
        (HERE / 'implementation_regression.json').write_text(json.dumps(rows, indent=2))
assert source == SOURCE.read_bytes()
print('PASS: all 10 full-data comparisons; production file unchanged during tests.', flush=True)
