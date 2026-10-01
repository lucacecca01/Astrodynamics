import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'deps'))
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
import numpy as np
from diptest import diptest
from test_fusion import app, SOURCE
from real_data import audit

LIMITS = np.array([.04, .05, 10.])
MINIMUM = 10
SIGMA = 3.
ALPHA = .01


def projections(data):
    data = np.array(data, dtype=float, copy=True)
    origin = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(data[:, 2])))))
    data[:, 2] = (data[:, 2] - origin + 180) % 360 - 180
    x = (data - data.mean(axis=0)) / LIMITS
    _, _, axes = np.linalg.svd(x, full_matrices=False)
    return np.column_stack((x, x @ axes.T))


def projection_scores(values):
    scores = []
    n = len(values)
    support = max(3, int(np.ceil(.1 * n)))
    for col in values.T:
        ordered = np.sort(col)
        span = np.ptp(ordered)
        if span < 1e-12:
            scores.append([0., 1., 0.])
            continue
        statistic, pvalue = diptest(ordered, sort_x=False, boot_pval=False)
        gaps = np.diff(ordered)
        eligible = gaps[support - 1:n - support]
        gap = float(eligible.max() / span) if len(eligible) else 0.
        scores.append([float(statistic), min(1., float(pvalue) * values.shape[1]), gap])
    return np.array(scores)


class ShapeCheck:
    def __init__(self, data):
        self.data = data
        self.cache = {}

    def scores(self, ids):
        ids = np.sort(ids)
        key = hashlib.sha256(ids.tobytes()).digest()
        if key not in self.cache:
            self.cache[key] = projection_scores(projections(self.data[ids, :3]))
        return self.cache[key]

    def condition(self, mode, alpha=ALPHA):
        def valid(ids):
            if mode == 'baseline':
                return True
            scores = self.scores(ids)
            failing = scores[:, 1] < alpha
            if mode.startswith('gap_'):
                failing &= scores[:, 2] > float(mode[4:])
            return not np.any(failing)
        return valid


def metrics(labels, p, f, shape, baseline, extra=None):
    n = len(p)
    row = audit(labels, p, f, LIMITS, SIGMA, MINIMUM, n // 100, extra=extra)
    groups = app.cluster_groups(labels)
    sizes = np.array([len(ids) for ids in groups])
    stds = np.array([app.physical_std(p[ids, :3]) for ids in groups])
    qq = np.array([app.gaussian_shape_penalty(p[ids, :3]) for ids in groups])
    row.update(mean_std=stds.mean(axis=0).tolist(), mean_qq=qq.mean(axis=0).tolist(),
               weighted_qq=np.average(qq, weights=sizes, axis=0).tolist(),
               below_20=int(np.sum(sizes < 20)), median_size=float(np.median(sizes)),
               lost=int(np.sum((baseline >= 0) & (labels < 0))),
               recovered=int(np.sum((baseline < 0) & (labels >= 0))))
    for name, mode in [('dip_flagged', 'dip'), ('gap10_flagged', 'gap_0.1'),
                       ('gap20_flagged', 'gap_0.2'), ('gap30_flagged', 'gap_0.3')]:
        checks = np.array([not shape.condition(mode)(ids) for ids in groups])
        row[name] = int(checks.sum())
        row[name + '_points'] = int(sizes[checks].sum())
    return row


def synthetic():
    rng = np.random.default_rng(472122)
    results = []
    for n in (20, 55, 500):
        for name in ('normal', 'uniform_line', 'continuous_arc', 'two_blobs', 'two_branches'):
            flags = {'dip': 0, 'gap_0.1': 0, 'gap_0.2': 0, 'gap_0.3': 0}
            for _ in range(100):
                t = rng.uniform(-1, 1, n)
                noise = rng.normal(size=(n, 3))
                if name == 'normal':
                    x = noise
                elif name == 'uniform_line':
                    x = np.column_stack((t, .5*t, -.2*t)) + .01*noise
                elif name == 'continuous_arc':
                    x = np.column_stack((t, t*t, .4*t*t*t)) + .01*noise
                elif name == 'two_blobs':
                    x = .1*noise
                    x[:, 0] += np.where(np.arange(n) < n//2, -1., 1.)
                else:
                    branch = np.where(np.arange(n) < n//2, -1., 1.)
                    x = np.column_stack((branch + .02*t*t, .3*branch + .2*t, t)) + .01*noise
                data = x * np.array([.01, .025, 4.]) + [-.6, .3, 180.]
                scores = projection_scores(projections(data))
                bad = scores[:, 1] < ALPHA
                flags['dip'] += int(np.any(bad))
                for gap in (.1, .2, .3):
                    flags[f'gap_{gap}'] += int(np.any(bad & (scores[:, 2] > gap)))
            results.append(dict(n=n, geometry=name, repetitions=100, **flags))
    return results


def plot_case(p, original, cases, shape):
    import matplotlib.pyplot as plt
    groups = app.cluster_groups(original)
    rank = []
    for i, ids in enumerate(groups):
        scores = shape.scores(ids)
        if 20 <= len(ids) <= 150 and scores[0, 1] < ALPHA:
            rank.append((scores[0, 2], i))
    if not rank:
        return None
    _, cluster = max(rank)
    ids = groups[cluster]
    names = ['baseline', 'dip', 'gap_0.2']
    fig, axes = plt.subplots(2, len(names), figsize=(15, 8), constrained_layout=True)
    summary = dict(original_cluster=cluster, size=len(ids), feature_row_ids=ids.tolist(), outcomes={})
    for col, name in enumerate(names):
        labels = original if name == 'baseline' else cases[name]
        assigned = np.unique(labels[ids])
        for j, label in enumerate(assigned):
            members = ids[labels[ids] == label]
            color = 'gray' if label < 0 else plt.get_cmap('tab10')(j % 10)
            text = 'Unassigned' if label < 0 else f'Group {label}: {len(members)}'
            axes[0, col].scatter(p[members, 0], p[members, 1], s=14, color=color, label=text)
            axes[1, col].hist(p[members, 0], bins=np.linspace(p[ids, 0].min(), p[ids, 0].max(), 21),
                              color=color, alpha=.8)
        axes[0, col].set_title(name)
        axes[0, col].set_ylabel('e')
        axes[0, col].legend(fontsize=8)
        for ax in axes[:, col]:
            ax.set_xlabel('eps [km^2/s^2]')
            ax.grid(alpha=.2)
        summary['outcomes'][name] = {str(int(k)): int(np.sum(labels[ids] == k)) for k in assigned}
    fig.suptitle(f'Real-data example: original cluster {cluster}, {len(ids)} trajectories')
    fig.savefig(HERE / 'example.png', dpi=150)
    plt.close(fig)
    return summary


def main():
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [SOURCE, SOURCE.parent / 'Clusters/GA_16_fusion_candidates.npz']}
    data = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
    p, source_ids = data['features'], data['source_ids']
    labels_file = np.load(ROOT / '.cache/ga16_min10_tests/labels.npz')
    np.testing.assert_array_equal(source_ids, labels_file['source_ids'])
    baseline = labels_file['labels'][1].copy()
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100*app.normalize_block(p[:, 6:7])))
    shape = ShapeCheck(p)
    report = dict(n=len(p), sigma=SIGMA, limits=LIMITS.tolist(), minimum=MINIMUM,
                  alpha=ALPHA, tests='3 physical axes + 3 PCA axes; Bonferroni x6 heuristic',
                  source_hashes=hashes, synthetic=synthetic(), cases=[])
    row = dict(mode='baseline', stage='reference', seconds=0., **metrics(baseline, p, f, shape, baseline))
    report['cases'].append(row)
    print('RESULT', json.dumps(row), flush=True)
    cases = {}
    pools = {}
    for mode in ('dip', 'gap_0.1', 'gap_0.2', 'gap_0.3'):
        start = time.perf_counter()
        extra = shape.condition(mode)
        valid = app.make_cluster_validator(p[:, :3], f, LIMITS, MINIMUM, SIGMA,
                                          directions=p[:, 6], extra_condition=extra)
        pool = []
        labels = app.split_messy_clusters(baseline, p[:, :3], LIMITS, MINIMUM, len(p)//100,
                                         features=f, sigma_limit=SIGMA, group_validator=valid,
                                         candidate_groups=pool)
        seconds = time.perf_counter() - start
        row = dict(mode=mode, stage='local_split', seconds=seconds,
                   **metrics(labels, p, f, shape, baseline, extra))
        report['cases'].append(row)
        cases[mode] = labels
        pools[mode] = pool
        (HERE / 'report.json').write_text(json.dumps(report, indent=2))
        print('RESULT', json.dumps(row), flush=True)
    report['example'] = plot_case(p, baseline, cases, shape)
    np.savez_compressed(HERE / 'local_labels.npz', source_ids=source_ids, baseline=baseline, **cases)
    for mode, pool in pools.items():
        np.savez_compressed(HERE / f'pool_{mode}.npz', source_ids=source_ids,
                            members=np.concatenate(pool), offsets=np.r_[0, np.cumsum([len(x) for x in pool])])
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    (HERE / 'report.json').write_text(json.dumps(report, indent=2))
    flat = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in report['cases']]
    with (HERE / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print('DONE: production code and cache unchanged', flush=True)


if __name__ == '__main__':
    main()
