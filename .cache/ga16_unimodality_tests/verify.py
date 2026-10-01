import csv
import json
import time

from experiment import HERE, ROOT, SOURCE, LIMITS, MINIMUM, SIGMA, ALPHA, ShapeCheck, metrics, projections
from experiment import np, app, hashlib
from refine import targeted_split


def main():
    data = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')
    p, source_ids = data['features'], data['source_ids']
    n = len(p)
    w = np.deg2rad(p[:, 2])
    f = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                         app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                         100*app.normalize_block(p[:, 6:7])))
    saved = np.load(HERE / 'refine_labels.npz')
    np.testing.assert_array_equal(saved['source_ids'], source_ids)
    baseline = saved['baseline']
    candidate_file = np.load(HERE / 'common_pool.npz')
    np.testing.assert_array_equal(candidate_file['source_ids'], source_ids)
    pool = list(np.split(candidate_file['members'], candidate_file['offsets'][1:-1]))
    report = dict(cases=[], timings={}, validation={})
    shape = ShapeCheck(p)
    base_valid = app.make_cluster_validator(p[:, :3], f, LIMITS, MINIMUM, SIGMA, directions=p[:, 6])
    start = time.perf_counter()
    reference, _ = app.fuse_cluster_candidates(pool, n, base_valid, n//100, time_limit=30.)
    row = dict(mode='baseline', stage='common_pool_unprotected', seconds=time.perf_counter()-start,
               **metrics(reference, p, f, shape, baseline))
    report['cases'].append(row)
    print('RESULT', json.dumps(row), flush=True)
    final_labels = dict(reference=reference)
    for mode in ('gap_0.1', 'gap_0.2', 'gap_0.3'):
        target = saved['targeted_' + mode]
        groups, reverted = [], []
        for c, ids in enumerate(app.cluster_groups(baseline)):
            if np.any(target[ids] < 0):
                groups.append(ids)
                reverted.append(dict(original_cluster=c, size=len(ids),
                                     points_that_would_be_lost=int(np.sum(target[ids] < 0))))
            else:
                groups.extend(ids[target[ids] == label] for label in np.unique(target[ids]))
        groups.sort(key=lambda ids: int(ids.min()))
        labels = np.full(n, -1, dtype=int)
        for c, ids in enumerate(groups):
            labels[ids] = c
        np.testing.assert_array_equal(labels >= 0, baseline >= 0)
        row = dict(mode=mode, stage='conservative_no_loss', reverted_parents=reverted,
                   **metrics(labels, p, f, shape, baseline))
        report['cases'].append(row)
        final_labels[mode] = labels
        print('RESULT', json.dumps(row), flush=True)
    # Cold-cache timing of the extra refinement only, excluding integration and plotting.
    timings = []
    for _ in range(3):
        check = ShapeCheck(p)
        start = time.perf_counter()
        labels = targeted_split(baseline, p, f, check, 'gap_0.2', [])
        timings.append(time.perf_counter() - start)
        np.testing.assert_array_equal(labels, saved['targeted_gap_0.2'])
    report['timings']['targeted_gap20_seconds'] = timings
    # Order-independent outcome counts for the final MILP; memberships can have tied optima.
    check = shape.condition('gap_0.2')
    validator = app.make_cluster_validator(p[:, :3], f, LIMITS, MINIMUM, SIGMA,
                                          directions=p[:, 6], extra_condition=check)
    reversed_labels, _ = app.fuse_cluster_candidates(pool[::-1], n, validator, n//100, time_limit=30.)
    expected = saved['common_gap_0.2']
    assert np.sum(reversed_labels < 0) == np.sum(expected < 0)
    assert len(app.cluster_groups(reversed_labels)) == len(app.cluster_groups(expected))
    report['validation']['reversed_pool_same_coverage_and_K'] = True
    groups = app.cluster_groups(baseline)
    rotated = p.copy()
    rotated[:, 2] = (rotated[:, 2] + 237.) % 360.
    rotated_shape = ShapeCheck(rotated)
    for ids in groups:
        scores = shape.scores(ids)
        np.testing.assert_allclose(scores, rotated_shape.scores(ids), atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(scores, shape.scores(ids[::-1]), atol=1e-12)
    report['validation']['angular_rotation_and_row_order'] = True
    # Keep separate from the user's cluster IDs: these are the cached experiment labels.
    plot_gallery(p, baseline, final_labels['gap_0.2'], shape)
    np.savez_compressed(HERE / 'verified_labels.npz', source_ids=source_ids, baseline=baseline, **final_labels)
    original = json.loads((HERE / 'report.json').read_text())
    for path, expected_hash in original['source_hashes'].items():
        assert hashlib.sha256(app.Path(path).read_bytes()).hexdigest() == expected_hash
    report['validation']['production_code_and_cache_unchanged'] = True
    (HERE / 'verified_report.json').write_text(json.dumps(report, indent=2))
    rows = []
    previous = json.loads((HERE / 'refine_report.json').read_text())
    for r in original['cases'] + previous['cases'] + report['cases']:
        rows.append({k: r.get(k) for k in ['mode', 'stage', 'clusters', 'unassigned', 'lost',
                                          'recovered', 'smallest', 'median_size', 'dip_flagged',
                                          'gap10_flagged', 'gap20_flagged', 'gap30_flagged', 'seconds']})
    with (HERE / 'all_comparisons.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('VERIFIED', json.dumps(report['validation']), flush=True)
    print('TIMINGS', timings, flush=True)


def plot_gallery(p, baseline, refined, shape):
    import matplotlib.pyplot as plt
    groups = app.cluster_groups(baseline)
    candidates = []
    for c, ids in enumerate(groups):
        score = shape.scores(ids)
        if not shape.condition('gap_0.2')(ids) and len(np.unique(refined[ids])) > 1:
            candidates.append((score[0, 2], c))
    chosen = [c for _, c in sorted(candidates, reverse=True)[:6]]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 3.3*len(chosen)), constrained_layout=True)
    for row, c in enumerate(chosen):
        ids = groups[c]
        for col, labels in enumerate((baseline, refined)):
            for j, label in enumerate(np.unique(labels[ids])):
                members = ids[labels[ids] == label]
                axes[row, col].scatter(p[members, 0], p[members, 1], s=12,
                                       color=plt.get_cmap('tab10')(j % 10),
                                       label=f'{len(members)} trajectories')
            axes[row, col].legend(fontsize=8)
            axes[row, col].set_xlabel('eps [km^2/s^2]')
            axes[row, col].set_ylabel('e')
            axes[row, col].grid(alpha=.2)
        axes[row, 0].set_title(f'Before: cached cluster {c} ({len(ids)} trajectories)')
        axes[row, 1].set_title('After: gap-directed split, no previously assigned points lost')
    fig.savefig(HERE / 'before_after_gallery.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
