import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ga16-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
import numpy as np
from test_fusion import app

p = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')['features']
w = np.deg2rad(p[:, 2])
f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                     app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                     100*app.normalize_block(p[:, 6:7])))
directory = HERE / 'maxsigma_2'
labels = np.load(directory / 'labels.npz')['labels']
unassigned = labels < 0
shape = app.ClusterShapeConstraint(p[:, :3], [.04, .05, 10], outlier_sigma=3)
names = ['size', 'std', 'max_sigma', 'shape']
coverage = {name: np.zeros(len(p), dtype=bool) for name in ['all'] + names}
seen = set()
overlapping_candidates = 0
for path in directory.glob('split_*.npz'):
    with np.load(path) as saved:
        groups = np.split(saved['members'], saved['offsets'][1:-1])
    for ids in groups:
        if not len(ids) or not np.any(unassigned[ids]):
            continue
        key = np.sort(ids).tobytes()
        if key in seen:
            continue
        seen.add(key)
        if not np.all(p[ids, 6] == p[ids[0], 6]):
            continue
        checks = [len(ids) >= 10, bool(np.all(app.physical_std(p[ids, :3]) < [.04, .05, 10])),
                  app.within_sigma(f[ids], 2), shape(ids)]
        if all(checks):
            coverage['all'][ids] = True
            assert np.any(labels[ids] >= 0), 'A disjoint valid group could be added.'
            overlapping_candidates += 1
        for j, name in enumerate(names):
            if all(checks[:j] + checks[j + 1:]):
                coverage[name][ids] = True
region = (p[:, 0] > -1.05) & (p[:, 0] < -.55) & (p[:, 2] % 360 > 0) & (p[:, 2] % 360 < 140)
report = dict(overlapping_candidates=overlapping_candidates,
              region_bounds=dict(eps=[-1.05, -.55], omega_deg=[0, 140]),
              region_total=int(region.sum()))
for name, mask in [('all', unassigned), ('central_window', unassigned & region)]:
    report[name] = dict(excluded=int(mask.sum()), with_valid_candidate=int(np.sum(mask & coverage['all'])),
                        recoverable_candidate_if_relaxing_only={
                            feature: int(np.sum(mask & coverage[feature] & ~coverage['all'])) for feature in names})
np.savez_compressed(directory / 'unassigned_diagnostics.npz',
                    excluded_ids=np.flatnonzero(unassigned),
                    excluded_with_valid_candidate=np.flatnonzero(unassigned & coverage['all']))
(directory / 'unassigned_diagnostics.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
