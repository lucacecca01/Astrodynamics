import contextlib
import hashlib
import io
import json

import matplotlib.pyplot as plt
import numpy as np

import experiment as test


def main():
    original_hash = hashlib.sha256(test.previous.SOURCE.read_bytes()).hexdigest()
    p, f, source_ids = test.load_data()
    pool = test.Pool.load(test.HERE / 'pool_extended.npz', p, f, source_ids)
    base = test.baseline(2., 50)
    fused = np.load(test.HERE / 'extended_sigma_2_min_50.npz')['labels']
    anchor = np.load(test.OLD / 'anchor_base_180.npz')['labels']
    with contextlib.redirect_stdout(io.StringIO()):
        production = test.app.split_messy_clusters(anchor, p[:, :3], test.LIMITS, 50,
                                                   test.CAP, features=f, sigma_limit=2.)
    np.testing.assert_array_equal(test.previous.canonical(production), base)
    test.audit(fused, pool, 2., 50, base >= 0)
    valid = pool.valid_indices(2., 50)
    groups = [pool.groups[i] for i in valid]
    lookup = {int(global_id): local for local, global_id in enumerate(valid)}

    def indices(labels):
        return [lookup[pool.lookup[np.asarray(ids, dtype=np.int32).tobytes()]]
                for ids in test.groups_from_labels(labels)]

    # Reordering candidate columns must not undo coverage already accepted.
    rng = np.random.default_rng(271)
    replay = []
    for iteration in range(3):
        order = rng.permutation(len(groups))
        inverse = np.argsort(order)
        labels, info = test.pack([groups[i] for i in order], fused >= 0, test.CAP,
                                 inverse[indices(fused)], seconds=30)
        metrics = test.audit(labels, pool, 2., 50, fused >= 0)
        assert info['solver_status'] == 0
        assert metrics['unassigned'] == int(np.sum(fused < 0))
        assert metrics['clusters'] == len(indices(fused))
        replay.append(dict(iteration=iteration, **metrics, solver_seconds=info['solver_seconds']))

    # Check the cost of keeping the original cluster count as an extra cap.
    labels, capped_info = test.pack(groups, base >= 0, len(indices(base)), indices(base), seconds=30)
    capped_metrics = test.audit(labels, pool, 2., 50, base >= 0)
    assert capped_metrics['clusters'] <= len(indices(base))
    np.savez_compressed(test.HERE / 'same_cluster_cap.npz', labels=labels, source_ids=source_ids)

    # Deliberately remove point protection to measure the identity-vs-count tradeoff.
    labels, free_info = test.pack(groups, np.zeros(len(p), dtype=bool), test.CAP,
                                  indices(base), seconds=30)
    free_metrics = test.audit(labels, pool, 2., 50, np.zeros(len(p), dtype=bool))
    free_metrics['lost_from_original'] = int(np.sum((base >= 0) & (labels < 0)))
    free_metrics['lost_from_fused'] = int(np.sum((fused >= 0) & (labels < 0)))

    unchanged = merged = fragmented = 0
    for ids in test.groups_from_labels(base):
        destination = np.unique(fused[ids])
        assert np.all(destination >= 0)
        if len(destination) > 1:
            fragmented += 1
        elif np.count_nonzero(fused == destination[0]) == len(ids):
            unchanged += 1
        else:
            merged += 1
    summary = dict(trajectories=len(p), source_fingerprint=hashlib.sha256(p.tobytes()).hexdigest(),
                    production_baseline_matches=True, limits=test.LIMITS.tolist(), sigma=2., minimum=50,
                    initial=test.audit(base, pool, 2., 50, base >= 0),
                    fused=test.audit(fused, pool, 2., 50, base >= 0),
                    initial_clusters_unchanged=unchanged,
                    initial_clusters_in_larger_group=merged,
                    initial_clusters_repartitioned=fragmented,
                    permutation_replays=replay,
                    same_cluster_cap=dict(**capped_metrics, solver_status=capped_info['solver_status']),
                    unprotected=dict(**free_metrics, solver_status=free_info['solver_status']))
    (test.HERE / 'verification.json').write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex='col', sharey='col', constrained_layout=True)
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = ['Energy [km^2/s^2]', 'Eccentricity', 'omega [deg]']
    physical = p[:, :3].copy()
    physical[:, 2] %= 360
    gained = (base < 0) & (fused >= 0)
    for row, labels in enumerate((base, fused)):
        for ax, (x, y) in zip(axes[row], pairs):
            accepted, excluded = labels >= 0, labels < 0
            ax.scatter(physical[accepted, x], physical[accepted, y], s=1, color='#b7bec8',
                       alpha=.3, rasterized=True, label='Accepted')
            if row == 1:
                ax.scatter(physical[gained, x], physical[gained, y], s=3, color='#00856a',
                           rasterized=True, label='Recovered')
            ax.scatter(physical[excluded, x], physical[excluded, y], s=3, color='#c74343',
                       rasterized=True, label='Unassigned')
            ax.set_xlabel(names[x])
            ax.set_ylabel(names[y])
            ax.grid(alpha=.2)
        axes[row, 1].set_title(f'{"Before" if row == 0 else "Fusion"}: '
                               f'{np.sum(excluded)} unassigned, {len(indices(labels))} clusters')
        axes[row, 2].legend(loc='upper right', markerscale=3)
    fig.suptitle('Same STD limits, MAX_SIGMA = 2, minimum 50: all previously accepted points retained')
    fig.savefig(test.HERE / 'coverage_comparison.png', dpi=150)
    plt.close(fig)
    assert original_hash == hashlib.sha256(test.previous.SOURCE.read_bytes()).hexdigest()
    print(json.dumps(summary, indent=2), flush=True)
    print('VERIFICATION PASS: GA16 unchanged', flush=True)


if __name__ == '__main__':
    main()
