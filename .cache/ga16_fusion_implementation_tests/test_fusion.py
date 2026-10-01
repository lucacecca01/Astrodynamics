import ast
import contextlib
import io
import itertools
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'CR3BP/Lunar Swingby/Clustering/CR3BP_prova_GA_16.py'
sys.path.insert(0, str(ROOT))
tree = ast.parse(SOURCE.read_text())
cutoff = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'database' for t in node.targets))
app = types.ModuleType('ga16_implementation_test')
app.__file__ = str(SOURCE)
sys.modules[app.__name__] = app
exec(compile(ast.Module(body=tree.body[:cutoff], type_ignores=[]), str(SOURCE), 'exec'), app.__dict__)


class FusionTests(unittest.TestCase):
    def fuse(self, groups, n, validator=lambda ids: True, cap=100, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return app.fuse_cluster_candidates(groups, n, validator, cap, **kwargs)

    def test_overlap_and_recovery(self):
        groups = [np.arange(20), np.arange(10, 30), np.arange(20, 40)]
        base = np.r_[np.zeros(20, dtype=int), np.full(20, -1)]
        result, _ = self.fuse(groups, 40, cap=2, previous_labels=base)
        self.assertTrue(np.all(result >= 0))
        self.assertEqual(len(np.unique(result)), 2)
        result, _ = self.fuse(groups[:2], 40, previous_labels=base)
        np.testing.assert_array_equal(result, base)

    def test_cluster_count_is_secondary(self):
        result, _ = self.fuse([np.arange(20), np.arange(20, 40), np.arange(40)], 40)
        self.assertTrue(np.all(result == 0))

    def test_previous_points_can_be_replaced_for_better_coverage(self):
        base = np.r_[np.zeros(4, dtype=int), [-1]*3]
        result, _ = self.fuse([np.arange(1, 7)], 7, cap=1, previous_labels=base)
        self.assertEqual(result[0], -1)
        self.assertTrue(np.all(result[1:] == 0))

    def test_equal_coverage_can_use_fewer_clusters_and_different_points(self):
        base = np.array([0, 0, 1, 1, -1])
        result, _ = self.fuse([np.arange(1, 5)], 5, previous_labels=base)
        self.assertEqual(result[0], -1)
        self.assertTrue(np.all(result[1:] == 0))

    def test_refinement_baseline_is_only_a_fallback(self):
        base = np.r_[np.zeros(4, dtype=int), [-1]*3]

        def fake_split(*args, **kwargs):
            kwargs['candidate_groups'].extend([np.arange(4), np.arange(1, 7)])
            return base.copy()

        p = np.zeros((7, 3))
        with patch.object(app, 'split_messy_clusters', side_effect=fake_split):
            with contextlib.redirect_stdout(io.StringIO()):
                labels, _, _ = app.refine_and_fuse_clusters(
                    [base], p, p, np.ones(7), [1]*3, 1, 1, np.inf, seeds=(42,))
        self.assertEqual(labels[0], -1)
        self.assertEqual(np.sum(labels >= 0), 6)

    def test_incompatible_and_repairable_previous_constraints(self):
        base = np.zeros(8, dtype=int)
        valid = lambda ids: len(ids) <= 4
        result, _ = self.fuse([np.arange(4), np.arange(4, 8)], 8, valid,
                              cap=2, previous_labels=base)
        self.assertTrue(np.all(result >= 0))
        result, _ = self.fuse([np.arange(4), np.arange(4, 8)], 8, valid, cap=1, previous_labels=base)
        self.assertEqual(np.sum(result >= 0), 4)
        result, _ = self.fuse([np.arange(4)], 8, valid, previous_labels=base)
        self.assertEqual(np.sum(result >= 0), 4)

    def test_timeout_fallback_and_rejected_solver_output(self):
        base = np.r_[np.zeros(4, dtype=int), [-1]*4]
        for x in (None, np.array([1., 1., 1.]), np.array([np.nan]*3), np.zeros(3)):
            response = types.SimpleNamespace(x=x, message='forced timeout', mip_gap=None)
            with patch.object(app, 'milp', return_value=response):
                result, _ = self.fuse([np.arange(4), np.arange(2, 6), np.arange(4, 8)],
                                      8, previous_labels=base)
            np.testing.assert_array_equal(result, base)
        with patch.object(app, 'milp', return_value=types.SimpleNamespace(
                x=None, message='forced timeout', mip_gap=None)):
            result, _ = self.fuse([np.arange(4)], 8, lambda ids: len(ids) <= 4,
                                  previous_labels=np.zeros(8, dtype=int))
        self.assertTrue(np.all(result == -1))

    def test_empty_unassigned_and_zero_cap(self):
        self.assertEqual(self.fuse([], 0)[0].size, 0)
        self.assertTrue(np.all(self.fuse([np.arange(4)], 4, lambda ids: False)[0] == -1))
        self.assertTrue(np.all(self.fuse([np.arange(4)], 4, cap=0)[0] == -1))
        result, _ = self.fuse([np.arange(4)], 4, cap=0, previous_labels=np.zeros(4, dtype=int))
        self.assertTrue(np.all(result == -1))
        result, _ = self.fuse([], 4, lambda ids: False, previous_labels=np.zeros(4, dtype=int))
        self.assertTrue(np.all(result == -1))

    def test_input_validation(self):
        for ids in (np.array([0., 1.]), np.array([0, 0]), np.array([-1]), np.array([4]), np.zeros((2, 2), dtype=int)):
            with self.assertRaises(ValueError):
                self.fuse([ids], 4)
        for cap in (-1, .5, True):
            with self.assertRaises(ValueError):
                self.fuse([np.arange(4)], 4, cap=cap)
        for previous in (np.zeros(4), np.array([-2]*4), np.zeros(3, dtype=int)):
            with self.assertRaises(ValueError):
                self.fuse([], 4, previous_labels=previous)
        for limit in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.fuse([], 4, time_limit=limit)

    def test_deduplication_and_retained_rejected_candidates(self):
        groups = [np.arange(4), np.arange(4)[::-1], np.arange(5)]
        labels, pool = self.fuse(groups, 5, lambda ids: len(ids) == 4)
        self.assertEqual(len(pool), 2)
        result, _ = self.fuse(pool, 5, lambda ids: True, previous_labels=labels)
        self.assertTrue(np.all(result == 0))

    def test_validator_circular_constant_extra_and_disabled_constraints(self):
        p = np.array([[0., .1, 359.], [0., .1, 1.], [0., .1, 10.]])
        f = np.zeros((3, 4))
        dirs = np.array([1, 1, -1])
        valid = app.make_cluster_validator(p, f, [1., 1., 2.], 2, 2., directions=dirs)
        self.assertTrue(valid(np.array([0, 1])))
        self.assertFalse(valid(np.arange(3)))
        self.assertFalse(valid(np.array([0])))
        valid = app.make_cluster_validator(p, f, [np.inf]*3, 1, np.inf,
                                           extra_condition=lambda ids: np.all(ids != 2))
        self.assertTrue(valid(np.array([0])))
        self.assertFalse(valid(np.arange(3)))
        np.testing.assert_array_equal(app.normalize_block(f), f)
        self.assertTrue(app.within_sigma(f, np.inf))
        for limits, minimum, sigma in (([0, 1, 1], 1, 2), ([1, 1, np.nan], 1, 2),
                                        ([1]*3, 0, 2), ([1]*3, 1, 0), ([1]*3, 1, np.nan)):
            with self.assertRaises(ValueError):
                app.make_cluster_validator(p, f, limits, minimum, sigma)

    def test_split_can_keep_one_large_child_of_small_parent(self):
        p = np.column_stack((np.r_[np.zeros(6), [10.]*3], np.zeros((9, 2))))
        labels = app.split_messy_clusters(np.zeros(9, dtype=int), p, [.1, 1, 1], 5, 9,
                                          features=p, sigma_limit=np.inf, verbose=False)
        self.assertEqual(np.sum(labels >= 0), 6)

    def test_constant_and_unsplittable_data(self):
        p = np.zeros((8, 3))
        f = np.zeros((8, 5))
        with contextlib.redirect_stdout(io.StringIO()):
            labels, _, _ = app.refine_and_fuse_clusters([], p, f, np.ones(8), [1]*3, 2, 8, 2,
                                                        base_count=3, seeds=(1, 2))
            self.assertTrue(np.all(labels == 0))
            labels, _, _ = app.refine_and_fuse_clusters([], p, f, np.ones(8), [1]*3, 2, 8, 2,
                                                        extra_condition=lambda ids: False,
                                                        base_count=3, seeds=(1, 2))
            self.assertTrue(np.all(labels == -1))

    def test_cache_roundtrip_invalidation_empty_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.npz'
            groups, base = [np.arange(4)], np.zeros(4, dtype=int)
            fingerprint = app.fusion_fingerprint(base, np.arange(12).reshape(4, 3))
            app.save_fusion_cache(path, fingerprint, groups, base)
            restored, previous = app.load_fusion_cache(path, fingerprint, 4)
            np.testing.assert_array_equal(restored[0], groups[0])
            np.testing.assert_array_equal(previous, base)
            self.assertEqual(app.load_fusion_cache(path, 'changed', 4), ([], None))
            self.assertNotEqual(fingerprint, app.fusion_fingerprint(base, np.arange(12).reshape(4, 3)[::-1]))
            app.save_fusion_cache(path, fingerprint, [], np.full(4, -1))
            self.assertEqual(app.load_fusion_cache(path, fingerprint, 4)[0], [])
            np.savez(path, fingerprint=fingerprint, bad_key=np.arange(4))
            with self.assertRaises(RuntimeError):
                app.load_fusion_cache(path, fingerprint, 4)
            self.assertFalse(path.with_name(path.name + '.tmp').exists())

    def test_randomized_solver_against_exhaustive_search(self):
        rng = np.random.default_rng(631)
        for _ in range(30):
            n, cap = 14, 3
            groups = [np.arange(4)] + [np.sort(rng.choice(n, rng.integers(2, 7), replace=False)) for _ in range(7)]
            required = np.arange(n) < 4
            base = np.where(required, 0, -1)
            best = None
            for bits in itertools.product((False, True), repeat=len(groups)):
                selected = np.flatnonzero(bits)
                if len(selected) > cap:
                    continue
                count = np.zeros(n, dtype=int)
                for j in selected:
                    count[groups[j]] += 1
                if np.any(count > 1):
                    continue
                score = (np.sum(count), -len(selected))
                best = score if best is None else max(best, score)
            labels, _ = self.fuse(groups, n, cap=cap, previous_labels=base)
            self.assertEqual((np.sum(labels >= 0), -len(np.unique(labels[labels >= 0]))), best)


if __name__ == '__main__':
    unittest.main(verbosity=2)
