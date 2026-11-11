"""Small numerical regressions for the GA20 hybrid search and final SOI check.

No main program is imported and no full catalogue search is launched.  The
positive regression starts at fixed ten-day grid nodes, not saved fine epochs.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pykep as pk
import spiceypy as spice

HELIOCENTRIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HELIOCENTRIC))

import ga19_transfer_search as geometry
from ga20_soi_validation import validate_exterior_arc
from ga20_transfer_search import Asteroid, Search, Settings, unique_missions


class AsteroidSearchTests(unittest.TestCase):
    # These are coarse grid nodes.  Fine solution dates are deliberately absent
    # from this regression and are never read from a previous result file.
    SEED = (2461496.5, 2461846.5, 1470.0)

    @classmethod
    def setUpClass(cls):
        cls.kernel = pk.udpla.de440s.kernel_file()
        spice.furnsh(cls.kernel)
        source = HELIOCENTRIC.parent / (
            "Clustering/Clusters/CHAMPION_GA_16_plots_vinf_time_K780/"
            "cluster_velocity_statistics.txt"
        )
        catalogue = geometry.load_catalogue(source, np.arange(763, 780), 2.5)
        cls.soi_radius = (5.974e24 / 1.989e30)**0.4 * 1.496e8
        positions = catalogue.mean_r * (
            cls.soi_radius / np.linalg.norm(catalogue.mean_r, axis=1)
        )[:, None]
        cls.catalogue = replace(catalogue, mean_r=positions)

        raw = json.loads((HELIOCENTRIC / "asteroids_sbdb_all_i5.json").read_text())
        cls.record = dict(zip(raw["fields"], raw["data"][125]))
        cls.physical = json.loads(
            (HELIOCENTRIC / "asteroids_sbdb_physical.json").read_text()
        )["asteroids"][str(cls.record["spkid"])]
        r = cls.record
        elements = (
            float(r["a"]) * pk.AU / 1000,
            float(r["e"]), 0.0, 0.0,
            np.deg2rad((float(r["om"]) + float(r["w"])) % 360),
            np.deg2rad(float(r["ma"]) % 360),
        )
        cls.asteroid = Asteroid(
            elements, float(r["epoch"]),
            cls.physical["mass_kg"], cls.physical["radius_km"],
        )
        cls.settings = Settings(
            2461326.5, 2462056.5,
            tuple((365 * n - 100, 365 * n + 99) for n in range(1, 5)),
            half_width_days=10.0,
        )
        cls.departure_row = int(np.flatnonzero(cls.catalogue.cluster_ids == 68)[0])
        cls.capture_row = int(np.flatnonzero(cls.catalogue.cluster_ids == 114)[0])
        cls.refinement = Search(cls.catalogue, cls.asteroid, cls.settings)
        cls.mission = cls.refinement.refine(
            cls.SEED, cls.departure_row, cls.capture_row, 0, 6,
        )
        if cls.mission is None:
            raise AssertionError("The fixed ten-day FG3 seed did not refine to a valid mission.")

    @classmethod
    def tearDownClass(cls):
        spice.unload(cls.kernel)

    def test_coarse_seed_refines_all_times_and_respects_local_domains(self):
        m, cfg = self.mission, self.settings
        self.assertEqual(self.catalogue.cluster_ids.size, 763)
        self.assertEqual(str(self.record["spkid"]), "20175706")
        self.assertEqual(tuple(m.seed_times), self.SEED)
        self.assertEqual((m.outbound_branch, m.return_branch), (0, 6))
        for t in self.SEED[:2]:
            self.assertEqual((t - cfg.jd_start) % 10.0, 0.0)
        final = (m.jd_departure, m.jd_asteroid, m.tof_return_days)
        for seed, actual in zip(self.SEED, final):
            self.assertNotEqual(seed, actual)
            self.assertLessEqual(abs(actual - seed), cfg.half_width_days + 1e-8)
        self.assertLess(m.jd_departure, m.jd_asteroid)
        self.assertGreaterEqual(m.jd_departure, cfg.jd_start)
        self.assertLessEqual(m.jd_asteroid, cfg.jd_end)
        self.assertTrue(any(lo <= m.tof_return_days <= hi for lo, hi in cfg.return_windows))

    def test_refined_endpoint_disks_and_unpowered_flyby_are_physical(self):
        m = self.mission
        dep = geometry.build_endpoints(self.catalogue, m.jd_departure)
        cap = geometry.build_endpoints(self.catalogue, m.jd_capture, incoming=True)
        np.testing.assert_array_equal(m.r0, dep.positions[m.departure_row])
        np.testing.assert_array_equal(m.r_capture, cap.positions[m.capture_row])
        dep_residual = np.linalg.norm(m.v0 - dep.velocities[m.departure_row])
        cap_residual = np.linalg.norm(m.v_capture - cap.velocities[m.capture_row])
        self.assertLessEqual(dep_residual, self.catalogue.radius_v[m.departure_row])
        self.assertLessEqual(cap_residual, self.catalogue.radius_v[m.capture_row])
        self.assertAlmostEqual(dep_residual, m.departure_residual, delta=1e-12)
        self.assertAlmostEqual(cap_residual, m.capture_residual, delta=1e-12)

        _, asteroid_velocity = self.asteroid.state(m.jd_asteroid)
        minus, plus = m.v_arrival - asteroid_velocity, m.v_return - asteroid_velocity
        speed_error = abs(np.linalg.norm(minus) - np.linalg.norm(plus))
        turn = np.arctan2(np.linalg.norm(np.cross(minus, plus)), np.dot(minus, plus))
        mu_ast = 6.67430e-20 * self.physical["mass_kg"]
        minimum_pericentre = 4 * self.physical["radius_km"]
        maximum = 2 * np.arcsin(
            mu_ast / (mu_ast + minimum_pericentre * np.linalg.norm(minus)**2)
        )
        self.assertLessEqual(speed_error, self.settings.velocity_tolerance_km_s)
        self.assertAlmostEqual(speed_error, m.speed_error, delta=1e-12)
        self.assertGreater(turn, 0)
        self.assertLessEqual(turn, maximum)
        self.assertAlmostEqual(maximum, m.max_turn_angle, delta=maximum * 1e-12)
        self.assertGreater(np.dot(m.r0 - dep.earth_position,
                                  m.v0 - dep.earth_velocity), 0)
        self.assertLess(np.dot(m.r_capture - cap.earth_position,
                               m.v_capture - cap.earth_velocity), 0)

    def test_refined_legs_propagate_to_their_actual_endpoints(self):
        m = self.mission
        for initial_r, initial_v, tof, final_r, final_v in (
            (m.r0, m.v0, m.tof_out_days, m.r_asteroid, m.v_arrival),
            (m.r_asteroid, m.v_return, m.tof_return_days, m.r_capture, m.v_capture),
        ):
            with self.subTest(tof=tof):
                r, v = pk.propagate_lagrangian(
                    [initial_r, initial_v], tof * geometry.DAY, geometry.MU_SUN,
                )
                self.assertLess(np.linalg.norm(np.asarray(r) - final_r), 1e-3)
                self.assertLess(np.linalg.norm(np.asarray(v) - final_v), 1e-8)

    def test_both_refined_legs_remain_exterior_to_the_soi(self):
        m = self.mission
        for r0, v0, jd, tof in (
            (m.r0, m.v0, m.jd_departure, m.tof_out_days),
            (m.r_asteroid, m.v_return, m.jd_asteroid, m.tof_return_days),
        ):
            with self.subTest(jd=jd):
                self.assertTrue(validate_exterior_arc(
                    r0, v0, jd, tof, self.soi_radius, geometry.MU_SUN,
                ))

    def test_return_window_gap_is_not_refined(self):
        search = Search(self.catalogue, self.asteroid, self.settings)
        mission = search.refine(
            (self.SEED[0], self.SEED[1], 550.0),
            self.departure_row, self.capture_row, 0, 6,
        )
        self.assertIsNone(mission)
        self.assertEqual(search.lambert_calls, 0)

    def test_empty_departure_scan_returns_no_missions_or_lambert_calls(self):
        search = Search(self.catalogue, self.asteroid, self.settings)
        missions, stats = search.scan_encounter(self.SEED[1], [], [1470.0])
        self.assertEqual(missions, [])
        self.assertEqual(stats["outbound_seeds"], 0)
        self.assertEqual(stats["local_attempts"], 0)
        self.assertEqual(stats["accepted"], 0)
        self.assertEqual(stats["lambert_calls"], 0)
        self.assertEqual(search.lambert_calls, 0)

    def test_deduplication_preserves_exact_epochs_of_one_mission(self):
        duplicate = replace(self.mission)
        selected = unique_missions([self.mission, duplicate, self.mission])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].jd_departure, self.mission.jd_departure)
        self.assertEqual(selected[0].jd_asteroid, self.mission.jd_asteroid)
        self.assertEqual(selected[0].tof_return_days, self.mission.tof_return_days)

    def test_soi_validator_accepts_far_arc_and_rejects_inside_or_wrong_sign(self):
        jd = self.SEED[0]
        earth_r, earth_v = geometry.earth_state(jd)
        axis = earth_r / np.linalg.norm(earth_r)
        cases = (
            (earth_r + 10 * self.soi_radius * axis, earth_v + axis, 0.25, True),
            (earth_r, earth_v, 0.25, False),
            (earth_r + self.soi_radius * axis, earth_v - axis, 0.01, False),
        )
        for r0, v0, tof, expected in cases:
            with self.subTest(expected=expected, tof=tof):
                self.assertEqual(validate_exterior_arc(
                    r0, v0, jd, tof, self.soi_radius, geometry.MU_SUN,
                ), expected)

    def test_soi_validator_catches_passage_hidden_between_outside_samples(self):
        jd = self.SEED[0]
        earth_r, earth_v = geometry.earth_state(jd)
        axis = earth_r / np.linalg.norm(earth_r)
        r0 = earth_r - 2 * self.soi_radius * axis
        v0 = earth_v + 1000 * axis
        times = np.array([0.0, 0.25 * geometry.DAY])
        states = np.asarray(pk.propagate_lagrangian_grid(
            [r0, v0], times, geometry.MU_SUN,
        ))
        er, _ = geometry.earth_state(jd + times / geometry.DAY)
        self.assertTrue(np.all(np.linalg.norm(states[:, 0] - er, axis=1) > self.soi_radius))
        self.assertFalse(validate_exterior_arc(
            r0, v0, jd, 0.25, self.soi_radius, geometry.MU_SUN, step_days=1.0,
        ))


if __name__ == "__main__":
    unittest.main()
