"""Small GA19 regressions: no full catalogue search and no image generation.

Run directly with the project's Astrodynamics Python environment.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pykep as pk
import spiceypy as spice

HELIOCENTRIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HELIOCENTRIC))
import ga19_transfer_search as search
import CR3BP_prova_GA_19 as app


def catalogue_subset(catalogue, radius=None):
    rows = np.array([0, 90, 250, 420, 563, 762])
    fields = {name: np.asarray(getattr(catalogue, name))[rows].copy()
              for name in ("cluster_ids", "cluster_sizes", "mean_r", "mean_v",
                           "tau_days", "radius_v")}
    if radius is not None:
        fields["radius_v"][:] = radius
    return replace(catalogue, **fields)


def key(transfer):
    return (transfer.tof_index, transfer.departure_cluster_row,
            transfer.arrival_cluster_row, transfer.branch_index)


def exhaustive(catalogue, epoch, tofs):
    """Independent all-pair/all-branch oracle without invariant filtering."""
    departure = search.build_endpoints(catalogue, epoch)
    records, branches = {}, 0
    for j, tof in enumerate(tofs):
        arrival = search.build_endpoints(catalogue, epoch + tof, incoming=True)
        for c, r0 in enumerate(departure.positions):
            for s, r1 in enumerate(arrival.positions):
                problem = pk.lambert_problem(r0=r0, r1=r1, tof=float(tof)*86400,
                                            mu=search.MU_SUN, cw=False, multi_revs=2)
                for branch, (v0, v1) in enumerate(zip(problem.v0, problem.v1)):
                    branches += 1
                    v0, v1 = np.asarray(v0), np.asarray(v1)
                    d0 = np.linalg.norm(v0-departure.velocities[c])
                    d1 = np.linalg.norm(v1-arrival.velocities[s])
                    outward = np.dot(r0-departure.earth_position,
                                     v0-departure.earth_velocity) > 0
                    inward = np.dot(r1-arrival.earth_position,
                                    v1-arrival.earth_velocity) < 0
                    if (d0 <= catalogue.radius_v[c]+search.VELOCITY_ATOL
                            and d1 <= catalogue.radius_v[s]+search.VELOCITY_ATOL
                            and outward and inward):
                        records[j, c, s, branch] = (v0, v1, d0, d1)
    return records, branches


class CaptureSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = pk.udpla.de440s.kernel_file()
        spice.furnsh(cls.kernel)
        data = HELIOCENTRIC.parent / (
            "Clustering/Clusters/CHAMPION_GA_16_plots_vinf_time_K780/"
            "cluster_velocity_statistics.txt"
        )
        full = search.load_catalogue(data, np.arange(763, 780), 2.5)
        cls.narrow = catalogue_subset(full)
        cls.broad = catalogue_subset(full, radius=200.)
        cls.epoch = 2451545.25
        cls.tofs = np.array([400., 800.])

    @classmethod
    def tearDownClass(cls):
        spice.unload(cls.kernel)

    def test_reflection_is_a_cr3bp_reversing_symmetry(self):
        mu = .0121505856
        symmetry = np.array([1., -1., 1., -1., 1., -1.])

        def rhs(state):
            x, y, z, vx, vy, vz = state
            r1 = ((x+mu)**2+y*y+z*z)**1.5
            r2 = ((x-1+mu)**2+y*y+z*z)**1.5
            return np.array([vx, vy, vz,
                             2*vy+x-(1-mu)*(x+mu)/r1-mu*(x-1+mu)/r2,
                             -2*vx+y-(1-mu)*y/r1-mu*y/r2,
                             -(1-mu)*z/r1-mu*z/r2])

        rng = np.random.default_rng(72)
        for _ in range(30):
            state = rng.normal(size=6)
            state[2] = state[5] = 0.
            np.testing.assert_allclose(rhs(symmetry*state), -symmetry*rhs(state),
                                       rtol=3e-15, atol=3e-15)

    def test_capture_reflection_and_future_reference_epoch(self):
        catalogue = search.Catalogue(
            np.array([5, 900]), np.array([10, 20]),
            np.array([[9e5, 1e5], [7e5, -5e5]]),
            np.array([[1.7, .3], [2., -.1]]), np.array([2.5, 8.]), np.array([.2, .3]),
        )
        earth_r, earth_v = np.array([1.4e8, -3e7, 0.]), np.array([-2., 29.7, 0.])
        angles = np.array([0., np.pi/2])

        def rotate(vectors):
            c, s = np.cos(angles), np.sin(angles)
            return np.column_stack((c*vectors[:, 0]-s*vectors[:, 1],
                                    s*vectors[:, 0]+c*vectors[:, 1]))

        departure = search.build_endpoints(catalogue, self.epoch, earth_state=(earth_r, earth_v),
                                           moon_angles=angles)
        arrival = search.build_endpoints(catalogue, self.epoch, incoming=True,
                                         earth_state=(earth_r, earth_v), moon_angles=angles)
        np.testing.assert_allclose(arrival.positions[:, :2]-earth_r[:2],
                                   rotate(catalogue.mean_r*[1, -1]), atol=2e-8)
        np.testing.assert_allclose(arrival.velocities[:, :2]-earth_v[:2],
                                   rotate(catalogue.mean_v*[-1, 1]), atol=5e-15)
        np.testing.assert_array_equal(departure.reference_epochs, self.epoch-catalogue.tau_days)
        np.testing.assert_array_equal(arrival.reference_epochs, self.epoch+catalogue.tau_days)
        r0, v0 = departure.positions-earth_r, departure.velocities-earth_v
        r1, v1 = arrival.positions-earth_r, arrival.velocities-earth_v
        np.testing.assert_allclose(np.linalg.norm(r0, axis=1), np.linalg.norm(r1, axis=1))
        np.testing.assert_allclose(np.linalg.norm(v0, axis=1), np.linalg.norm(v1, axis=1))
        np.testing.assert_allclose(np.sum(r0*v0, axis=1), -np.sum(r1*v1, axis=1), atol=1e-7)
        np.testing.assert_allclose(np.cross(r0, v0), np.cross(r1, v1), atol=1e-7)
        self.assertTrue(np.all(np.sum(r0*v0, axis=1) > 0))
        self.assertTrue(np.all(np.sum(r1*v1, axis=1) < 0))

    def test_every_branch_matches_exhaustive_and_propagates_to_capture(self):
        for catalogue in (self.narrow, self.broad):
            expected, branches = exhaustive(catalogue, self.epoch, self.tofs)
            unfiltered = search.solve_soi(catalogue, self.epoch, self.tofs, prefilter=False)
            filtered = search.solve_soi(catalogue, self.epoch, self.tofs)
            self.assertEqual(unfiltered.total_pairs, 72)
            self.assertEqual(unfiltered.attempted_pairs, 72)
            self.assertEqual(unfiltered.lambert_branches, branches)
            self.assertEqual(unfiltered.lambert_failures+filtered.lambert_failures, 0)
            self.assertLessEqual(filtered.attempted_pairs, unfiltered.attempted_pairs)
            for result in (unfiltered, filtered):
                actual = {key(t): t for t in result.transfers}
                self.assertEqual(len(actual), len(result.transfers))
                self.assertEqual(actual.keys(), expected.keys())
                for k, transfer in actual.items():
                    v0, v1, d0, d1 = expected[k]
                    np.testing.assert_allclose(transfer.v0, v0, atol=1e-13)
                    np.testing.assert_allclose(transfer.v1, v1, atol=1e-13)
                    np.testing.assert_allclose([transfer.departure_residual, transfer.arrival_residual],
                                               [d0, d1], atol=1e-12)
                    self.assertEqual(transfer.revolutions, (transfer.branch_index+1)//2)
                    self.assertEqual(transfer.tof_days, self.tofs[transfer.tof_index])
                    self.assertEqual(transfer.jd_arrival, self.epoch+transfer.tof_days)
                    cap = search.build_endpoints(catalogue, transfer.jd_arrival, incoming=True)
                    s = transfer.arrival_cluster_row
                    np.testing.assert_array_equal(transfer.r1, cap.positions[s])
                    self.assertEqual(transfer.arrival_reference_jd, cap.reference_epochs[s])
            if catalogue is self.broad:
                self.assertEqual({t.branch_index for t in filtered.transfers}, set(range(5)))
                representatives = {t.branch_index: t for t in filtered.transfers}
                for transfer in representatives.values():
                    r, v = pk.propagate_lagrangian([transfer.r0, transfer.v0],
                                                   transfer.tof_days*86400, search.MU_SUN)
                    np.testing.assert_allclose(r, transfer.r1, rtol=2e-11, atol=2e-3)
                    np.testing.assert_allclose(v, transfer.v1, rtol=2e-11, atol=2e-8)

    def test_cached_fractional_arrivals_equal_direct_ephemerides(self):
        epochs, tofs = np.array([2451545., 2451545.25]), np.array([400., 400.25])
        cache = search.prepare_arrival_cache(self.broad, epochs, tofs)
        self.assertEqual(cache.epochs.size, 3)
        for i, epoch in enumerate(epochs):
            direct = search.solve_soi(self.broad, epoch, tofs, soi_index=i)
            cached = search.solve_soi(self.broad, epoch, tofs, soi_index=i, arrival_cache=cache)
            self.assertEqual([key(t) for t in direct.transfers], [key(t) for t in cached.transfers])
            for t0, t1 in zip(direct.transfers, cached.transfers):
                np.testing.assert_array_equal(t0.r1, t1.r1)
                np.testing.assert_array_equal(t0.v1, t1.v1)

    def test_prefilter_retains_valid_boundary_and_zero_radius_solutions(self):
        rng = np.random.default_rng(312)
        checked = 0

        def endpoint(r, v):
            return search.EndpointStates(np.asarray(r)[None, :], np.asarray(v)[None, :],
                                         np.zeros(3), np.zeros(3), np.zeros(1), np.zeros(1))

        for _ in range(20):
            angles, lengths = rng.uniform(0, 2*np.pi, 2), rng.uniform(.6, 1.4, 2)
            r0, r1 = np.column_stack((lengths*np.cos(angles), lengths*np.sin(angles), np.zeros(2)))
            problem = pk.lambert_problem(r0=r0, r1=r1, tof=float(rng.uniform(4, 20)),
                                        mu=1., cw=False, multi_revs=2)
            for v0, v1 in zip(problem.v0, problem.v1):
                for radius in (0., 1e-12, 1e-6, .01, .5):
                    direction = rng.normal(size=(2, 2))
                    direction /= np.linalg.norm(direction, axis=1)[:, None]
                    delta = np.column_stack((radius*direction, np.zeros(2)))
                    b0 = search.invariant_bounds(endpoint(r0, np.asarray(v0)+delta[0]), np.array([radius]), 1.)
                    b1 = search.invariant_bounds(endpoint(r1, np.asarray(v1)+delta[1]), np.array([radius]), 1.)
                    c, s = search.candidate_pairs(b0, b1)
                    self.assertEqual(c.tolist(), [0])
                    self.assertEqual(s.tolist(), [0])
                    checked += 1
        self.assertGreater(checked, 300)

    def test_saved_rows_states_ids_and_empty_results(self):
        records = search.solve_soi(self.broad, self.epoch, self.tofs).transfers
        with tempfile.TemporaryDirectory(prefix="ga19-regression-") as temporary:
            for label, transfers in (("matches", records), ("empty", [])):
                with self.subTest(label=label):
                    folder = Path(temporary)/label
                    metadata = {"status": "test", "accepted_transfers": len(transfers)}
                    app.save_results(transfers, self.broad, folder, metadata)
                    with np.load(folder/"transfers_soi_to_soi.npz") as stored:
                        for array in stored.values():
                            self.assertEqual(array.shape[0], len(transfers))
                        for name in ("r0", "r1", "v0", "v1"):
                            self.assertEqual(stored[name].shape, (len(transfers), 3))
                        if transfers:
                            text = np.loadtxt(folder/"transfers_soi_to_soi.txt", ndmin=2)
                            self.assertEqual(text.shape, (len(transfers), 14))
                            for i, transfer in enumerate(transfers):
                                self.assertEqual(text[i, 1], self.broad.cluster_ids[transfer.departure_cluster_row])
                                self.assertEqual(text[i, 3], self.broad.cluster_ids[transfer.arrival_cluster_row])
                                self.assertEqual(text[i, 5], transfer.branch_index)
                                self.assertEqual(text[i, 4], transfer.tof_days)
                                for name in ("r0", "r1", "v0", "v1"):
                                    np.testing.assert_array_equal(stored[name][i], getattr(transfer, name))
                        else:
                            text = (folder/"transfers_soi_to_soi.txt").read_text()
                            self.assertTrue(all(not row or row.startswith("#") for row in text.splitlines()))
                    self.assertEqual(json.loads((folder/"run.json").read_text()), metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
