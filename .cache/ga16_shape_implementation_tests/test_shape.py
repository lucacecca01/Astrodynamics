from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/ga16_fusion_implementation_tests'))
from test_fusion import app


def normal_group(n=100, omega=0.):
    q = ndtri((np.arange(n) + .5) / n)
    return np.column_stack((-.6 + .005*q, .3 + .01*q, (omega + 2*q) % 360))


class ShapeTests(unittest.TestCase):
    def test_normal_constant_singleton_and_uniform(self):
        data = normal_group()
        condition = app.ClusterShapeConstraint(data, [.04, .05, 10], gaussian_qq_max=.03)
        self.assertTrue(condition(np.arange(100)))
        self.assertIsNone(condition.split(np.arange(100)))
        self.assertFalse(condition(np.array([0])))
        const = app.ClusterShapeConstraint(np.zeros((20, 3)), [1, 1, 1], gaussian_qq_max=.03)
        self.assertFalse(const(np.arange(20)))
        line = np.column_stack([np.linspace(-1, 1, 100)] * 3)
        uniform = app.ClusterShapeConstraint(line, [1, 1, 1], gaussian_qq_max=.03)
        self.assertFalse(uniform(np.arange(100)))
        dip_only = app.ClusterShapeConstraint(line, [1, 1, 1])
        self.assertTrue(dip_only(np.arange(100)))

    def test_gap_cut_and_no_remerge(self):
        left = normal_group(40)
        right = normal_group(40)
        left[:, 0] -= .04
        right[:, 0] += .04
        data = np.vstack((left, right))
        ids = np.arange(len(data))
        condition = app.ClusterShapeConstraint(data, [.1, .1, 20], gaussian_qq_max=.03)
        self.assertFalse(condition(ids))
        cut = condition.split(ids)
        self.assertIsNotNone(cut)
        self.assertEqual(len(np.unique(cut[:40])), 1)
        self.assertEqual(len(np.unique(cut[40:])), 1)
        self.assertNotEqual(cut[0], cut[-1])
        features = np.column_stack((data[:, :2], np.cos(np.deg2rad(data[:, 2])),
                                    np.sin(np.deg2rad(data[:, 2]))))
        valid = app.make_cluster_validator(data, features, [.1, .1, 20], 10, np.inf,
                                          shape_condition=condition)
        pool = []
        labels = app.split_messy_clusters(np.zeros(len(data), dtype=int), data, [.1, .1, 20], 10, 10,
                                         features=features, sigma_limit=np.inf, group_validator=valid,
                                         candidate_groups=pool, gap_splitter=condition.split)
        self.assertEqual(len(app.cluster_groups(labels)), 2)
        self.assertTrue(np.all(labels >= 0))
        fused, _ = app.fuse_cluster_candidates(pool, len(data), valid, 10, previous_labels=labels)
        self.assertEqual(len(app.cluster_groups(fused)), 2)
        self.assertTrue(all(valid(g) for g in app.cluster_groups(fused)))

    def test_angle_wrap_order_and_cache(self):
        data = normal_group(55, 359.)
        data[25:, 0] += .06
        ids = np.arange(len(data))
        first = app.ClusterShapeConstraint(data, [.04, .05, 10])
        rotated = data.copy()
        rotated[:, 2] = (rotated[:, 2] + 237) % 360
        other = app.ClusterShapeConstraint(rotated, [.04, .05, 10])
        self.assertEqual(first(ids), other(ids))
        np.testing.assert_array_equal(first.split(ids), other.split(ids))
        np.testing.assert_array_equal(first.split(ids)[::-1], first.split(ids[::-1]))
        self.assertEqual(len(first.cache), 1)

    def test_per_feature_not_average_and_extra_condition(self):
        data = normal_group()
        data[:, 1] = .3
        condition = app.ClusterShapeConstraint(data, [.04, .05, 10], gaussian_qq_max=.03)
        self.assertFalse(condition(np.arange(len(data))))
        disabled = app.ClusterShapeConstraint(data, [.04, .05, 10], dip_alpha=0, gaussian_qq_max=np.inf)
        self.assertTrue(disabled(np.arange(len(data))))
        validator = app.make_cluster_validator(data, data, [1, 1, 10], 10, np.inf,
                                              shape_condition=disabled, extra_condition=lambda ids: False)
        self.assertFalse(validator(np.arange(len(data))))

    def test_invalid_parameters(self):
        for name, value in [('dip_alpha', -1), ('dip_alpha', 1), ('dip_alpha', np.nan),
                            ('gap_fraction', -1), ('gap_fraction', 1), ('gaussian_qq_max', 0),
                            ('gaussian_qq_max', np.nan), ('strong_gap_fraction', np.nan),
                            ('strong_gap_fraction', .1), ('outlier_sigma', 0),
                            ('outlier_sigma', -1), ('outlier_sigma', np.nan)]:
            with self.assertRaises(ValueError):
                app.ClusterShapeConstraint(normal_group(), [1, 1, 1], **{name: value})

    def test_exact_34_point_regression(self):
        data = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')['features'][:, :3]
        labels = np.load(ROOT / '.cache/ga16_shape_implementation_tests/dip_only/real_inf.npz')['labels']
        ids = np.flatnonzero(labels == 3)
        self.assertEqual(len(ids), 34)
        old = app.ClusterShapeConstraint(data, [.04, .05, 10], strong_gap_fraction=np.inf,
                                         outlier_sigma=np.inf)
        new = app.ClusterShapeConstraint(data, [.04, .05, 10])
        self.assertTrue(old(ids))
        self.assertFalse(new(ids))
        cut = new.split(ids)
        self.assertEqual(sorted(np.bincount(cut).tolist()), [6, 28])
        self.assertTrue(new(ids[cut == 0]))
        self.assertTrue(new(ids[cut == 1]))

    def test_one_two_and_multiple_outliers(self):
        main = normal_group(95)
        for count in (1, 2, 5):
            other = normal_group(count)
            other[:, 0] += 1.
            data = np.vstack((main, other))
            check = app.ClusterShapeConstraint(data, [.04, .05, 10], dip_alpha=0)
            ids = np.arange(len(data))
            self.assertFalse(check(ids))
            self.assertEqual(sorted(np.bincount(check.split(ids)).tolist()), [count, 95])
        disabled = app.ClusterShapeConstraint(data, [.04, .05, 10], dip_alpha=0,
                                              strong_gap_fraction=np.inf, outlier_sigma=np.inf)
        self.assertTrue(disabled(np.arange(len(data))))

    def test_opposite_tails_constant_core_and_minimum_size(self):
        for main in (normal_group(30), np.tile([-.6, .3, 359.], (30, 1))):
            other = main[:2].copy()
            other[:, 0] += [-1., 1.]
            data = np.vstack((main, other))
            shape = app.ClusterShapeConstraint(data, [.04, .05, 10], dip_alpha=0,
                                               strong_gap_fraction=np.inf)
            valid = app.make_cluster_validator(data, data, [.04, .05, 10], 5, np.inf,
                                              shape_condition=shape)
            pool = []
            labels = app.split_messy_clusters(np.zeros(32, dtype=int), data, [.04, .05, 10], 5, 10,
                                             features=data, sigma_limit=np.inf, group_validator=valid,
                                             gap_splitter=shape.split, candidate_groups=pool)
            self.assertEqual([len(g) for g in app.cluster_groups(labels)], [30])
            self.assertTrue(np.all(labels[:30] >= 0))
            self.assertTrue(np.all(labels[30:] == -1))
            fused, _ = app.fuse_cluster_candidates(pool, 32, valid, 10, previous_labels=labels)
            self.assertTrue(np.all(fused[30:] == -1))
            self.assertTrue(all(valid(g) for g in app.cluster_groups(fused)))

    def test_small_groups_and_outlier_angle_wrap(self):
        for count in (5, 6, 8):
            data = normal_group(count)
            data[-1, 0] += 1.
            shape = app.ClusterShapeConstraint(data, [.04, .05, 10])
            self.assertFalse(shape(np.arange(count)))
            self.assertEqual(sorted(np.bincount(shape.split(np.arange(count))).tolist()), [1, count-1])
        data = normal_group(36, omega=359.)
        data[-1, 2] += 80.
        rotated = data.copy()
        rotated[:, 2] = (rotated[:, 2] + 178) % 360
        first = app.ClusterShapeConstraint(data, [.04, .05, 10])
        second = app.ClusterShapeConstraint(rotated, [.04, .05, 10])
        ids = np.arange(36)
        self.assertFalse(first(ids))
        np.testing.assert_array_equal(first.split(ids), second.split(ids))

    def test_exact_36_point_outlier_regression(self):
        data = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')['features'][:, :3]
        labels = np.load(ROOT / '.cache/ga16_shape_implementation_tests/dip_only/real_inf.npz')['labels']
        ids = np.flatnonzero(labels == 9)
        self.assertEqual(len(ids), 36)
        np.testing.assert_allclose(app.physical_std(data[ids]),
                                   [.003624806812783081, .013106304316625284, 4.519943207329154])
        old = app.ClusterShapeConstraint(data, [.04, .05, 10], outlier_sigma=np.inf)
        new = app.ClusterShapeConstraint(data, [.04, .05, 10])
        self.assertTrue(old(ids))
        self.assertFalse(new(ids))
        self.assertEqual(sorted(np.bincount(new.split(ids)).tolist()), [1, 35])
        local_data = data[ids]
        check = app.ClusterShapeConstraint(local_data, [.04, .05, 10])
        valid = app.make_cluster_validator(local_data, local_data, [.04, .05, 10], 5, np.inf,
                                          shape_condition=check)
        pool = []
        result = app.split_messy_clusters(np.zeros(36, dtype=int), local_data, [.04, .05, 10], 5, 10,
                                         features=local_data, sigma_limit=np.inf, group_validator=valid,
                                         candidate_groups=pool, gap_splitter=check.split)
        self.assertEqual([len(g) for g in app.cluster_groups(result)], [33])
        self.assertEqual(np.count_nonzero(result < 0), 3)
        self.assertEqual(result[np.argmax(local_data[:, 0])], -1)
        fused, _ = app.fuse_cluster_candidates(pool, 36, valid, 10, previous_labels=result)
        self.assertEqual(np.count_nonzero(fused < 0), 3)

    def test_exact_90_point_regression_with_both_minima(self):
        p = np.load(ROOT / '.cache/ga16_sigma_recovery_tests/features.npz')['features']
        old = np.load(ROOT / '.cache/ga16_shape_implementation_tests/dip_only/real_inf.npz')['labels']
        ids = np.flatnonzero(old == 24)
        self.assertEqual(len(ids), 90)
        data = p[ids, :3]
        np.testing.assert_allclose(app.physical_std(data), [.00535554459, .0122159945, 8.92180965])
        w = np.deg2rad(p[:, 2])
        features = np.column_stack((app.normalize_block(p[:, :1]), app.normalize_block(p[:, 1:2]),
                                    app.normalize_block(np.column_stack((np.cos(w), np.sin(w)))),
                                    100*app.normalize_block(p[:, 6:7])))[ids]
        shape = app.ClusterShapeConstraint(data, [.04, .05, 10])
        self.assertFalse(shape(np.arange(90)))
        for minimum, expected, excluded in ((5, [6, 15, 69], 0), (10, [15, 69], 6)):
            valid = app.make_cluster_validator(data, features, [.04, .05, 10], minimum, 3,
                                              directions=p[ids, 6], shape_condition=shape)
            pool = []
            labels = app.split_messy_clusters(np.zeros(90, dtype=int), data, [.04, .05, 10], minimum, 90,
                                             features=features, sigma_limit=3, group_validator=valid,
                                             gap_splitter=shape.split, candidate_groups=pool)
            fused, _ = app.fuse_cluster_candidates(pool, 90, valid, 90, previous_labels=labels)
            self.assertEqual(sorted(map(len, app.cluster_groups(fused))), expected)
            self.assertEqual(np.count_nonzero(fused < 0), excluded)
            self.assertTrue(all(valid(g) for g in app.cluster_groups(fused)))

    def test_bad_splitter_rejected(self):
        data = normal_group()
        with self.assertRaises(RuntimeError):
            app.split_messy_clusters(np.zeros(100, dtype=int), data, [1, 1, 10], 10, 100,
                                     features=data, sigma_limit=np.inf, group_validator=lambda ids: False,
                                     gap_splitter=lambda ids: np.full(len(ids), 2))

    def test_cache_same_changed_and_legacy_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.npz'
            labels = np.zeros(20, dtype=int)
            groups = [np.arange(20)]
            app.save_fusion_cache(path, 'data', groups, labels, constraints_key='old')
            loaded, previous = app.load_fusion_cache(path, 'data', 20, constraints_key='old')
            np.testing.assert_array_equal(previous, labels)
            loaded, previous = app.load_fusion_cache(path, 'data', 20, constraints_key='new')
            self.assertIsNone(previous)
            np.testing.assert_array_equal(loaded[0], groups[0])
            app.save_fusion_cache(path, 'data', groups, labels)
            self.assertIsNone(app.load_fusion_cache(path, 'data', 20, constraints_key='new')[1])


if __name__ == '__main__':
    unittest.main()
