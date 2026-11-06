"""Planar, SOI-to-SOI heliocentric transfer search for the GA16 catalogue.

SPICE kernels must be furnished by the caller, including in worker processes.
Importing this module neither loads a catalogue nor starts a search.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pykep as pk
import spiceypy as spice


DAY = 86400.0
MU_SUN = pk.MU_SUN / 1e9  # km^3/s^2
VELOCITY_ATOL = 1e-9  # km/s; also included in prefilter disk radii.
_EPHEMERIS_CHUNK_SIZE = 100_000


@dataclass(frozen=True)
class Catalogue:
    cluster_ids: np.ndarray
    cluster_sizes: np.ndarray
    mean_r: np.ndarray
    mean_v: np.ndarray
    tau_days: np.ndarray
    radius_v: np.ndarray
    source: str = ""

    def __post_init__(self):
        ids = np.asarray(self.cluster_ids)
        sizes = np.asarray(self.cluster_sizes)
        if ids.ndim != 1 or not ids.size:
            raise ValueError("The catalogue must contain at least one cluster.")
        if (not np.all(np.isfinite(ids)) or np.any(ids != np.floor(ids))
                or np.any(ids < 0) or np.any(np.diff(ids) <= 0)):
            raise ValueError("Cluster IDs must be nonnegative, sorted, unique integers.")
        if (sizes.shape != ids.shape or not np.all(np.isfinite(sizes))
                or np.any(sizes != np.floor(sizes)) or np.any(sizes <= 0)):
            raise ValueError("Cluster sizes must be positive integers, one per cluster.")
        object.__setattr__(self, "cluster_ids", ids.astype(int))
        object.__setattr__(self, "cluster_sizes", sizes.astype(int))
        for name, shape in (("mean_r", (ids.size, 2)), ("mean_v", (ids.size, 2)),
                            ("tau_days", ids.shape), ("radius_v", ids.shape)):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite and have shape {shape}.")
            object.__setattr__(self, name, values)
        if np.any(self.tau_days <= 0) or np.any(self.radius_v < 0):
            raise ValueError("ETD-to-SOI durations must be positive and radii nonnegative.")
        if np.any(np.einsum("ij,ij->i", self.mean_r, self.mean_v) <= 0):
            raise ValueError("Every catalogue mean state must point outwards at the SOI.")
        object.__setattr__(self, "source", str(self.source))


def load_catalogue(path, excluded_ids=(), n_sigma=2.5):
    """Read GA16 statistics; velocities are evaluated at a finite SOI radius."""
    if not np.isfinite(n_sigma) or n_sigma < 0:
        raise ValueError("n_sigma must be finite and nonnegative.")
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 14 or not np.all(np.isfinite(data)):
        raise ValueError("Catalogue statistics need at least 14 finite columns.")
    data = data[~np.isin(data[:, 0], np.asarray(excluded_ids))]
    if np.any(data[:, 4:6] < 0) or np.any(data[:, 8:10] < 0) or np.any(data[:, 12:14] < 0):
        raise ValueError("Catalogue standard deviations cannot be negative.")
    return Catalogue(
        cluster_ids=data[:, 0], cluster_sizes=data[:, 1],
        mean_r=data[:, 10:12], mean_v=data[:, 2:4], tau_days=data[:, 7],
        radius_v=float(n_sigma) * np.linalg.norm(data[:, 4:6], axis=1),
        source=str(Path(path)),
    )


def _states(body, jd, observer):
    """Planar ECLIPJ2000 geometric states, preserving the input epoch shape."""
    jd = np.asarray(jd, dtype=float)
    if not np.all(np.isfinite(jd)):
        raise ValueError("Ephemeris epochs must be finite JD TDB values.")
    epochs = (jd.ravel() - 2451545.0) * DAY
    states = np.empty((epochs.size, 6))
    # The installed SpiceyPy array API shares argument conversion and setup.
    # Calls remain bounded, including the wrapper's temporary list of states.
    for start in range(0, epochs.size, _EPHEMERIS_CHUNK_SIZE):
        stop = min(epochs.size, start + _EPHEMERIS_CHUNK_SIZE)
        states[start:stop] = spice.spkezr(
            body, epochs[start:stop], "ECLIPJ2000", "NONE", observer,
        )[0]
    states[:, (2, 5)] = 0.0
    return states.reshape(jd.shape + (6,))


def _earth_states(jd):
    return _states("EARTH", jd, "SUN")


def earth_state(jd):
    """Return planar heliocentric Earth position and velocity, in km and km/s."""
    states = _earth_states(jd)
    return states[..., :3], states[..., 3:]


def moon_phase(jd):
    states = _states("MOON", jd, "EARTH")
    return np.arctan2(states[..., 1], states[..., 0])


def earth_moon_geometry(jd):
    """Compatibility helper: Earth r/v, heliocentric Moon r, lunar angle."""
    earth = _earth_states(jd)
    moon = _states("MOON", jd, "EARTH")
    return (earth[..., :3], earth[..., 3:], earth[..., :3] + moon[..., :3],
            np.arctan2(moon[..., 1], moon[..., 0]))


def rotate_xy(vectors, angles):
    vectors = np.asarray(vectors, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if vectors.shape[-1] != 2:
        raise ValueError("rotate_xy needs two Cartesian components.")
    c, s = np.cos(angles), np.sin(angles)
    return np.stack((c * vectors[..., 0] - s * vectors[..., 1],
                     s * vectors[..., 0] + c * vectors[..., 1]), axis=-1)


@dataclass(frozen=True)
class EndpointStates:
    positions: np.ndarray
    velocities: np.ndarray
    earth_position: np.ndarray
    earth_velocity: np.ndarray
    reference_epochs: np.ndarray
    moon_angles: np.ndarray


def build_endpoints(catalogue, jd_soi, incoming=False, earth_state=None, moon_angles=None):
    """Orient catalogue means at an outgoing or reflected incoming SOI crossing.

    In the incoming family the reflected catalogue ETD is *after* SOI entry.
    Reflection and time reversal preserve the prograde motion of the primaries.
    The means define candidate families, not independently validated members.
    """
    jd_soi = float(jd_soi)
    if not np.isfinite(jd_soi):
        raise ValueError("The SOI epoch must be finite.")
    references = jd_soi + (1 if incoming else -1) * catalogue.tau_days
    if earth_state is None:
        earth = _earth_states(jd_soi)
    else:
        earth = np.asarray(earth_state, dtype=float)
        if earth.shape == (2, 3):
            earth = earth.reshape(6)
        if earth.shape != (6,) or not np.all(np.isfinite(earth)):
            raise ValueError("An Earth state must contain finite r/v vectors or six components.")
        earth = earth.copy()
        earth[[2, 5]] = 0.0
    if moon_angles is None:
        angles = moon_phase(references)
    else:
        angles = np.broadcast_to(np.asarray(moon_angles, dtype=float), references.shape)
        if not np.all(np.isfinite(angles)):
            raise ValueError("Lunar angles must be finite.")
    reflection = np.array([1.0, -1.0]) if incoming else np.ones(2)
    relative_positions = rotate_xy(catalogue.mean_r * reflection, angles)
    relative_velocities = rotate_xy(catalogue.mean_v * reflection, angles)
    if incoming:
        relative_velocities = -relative_velocities
    positions = np.broadcast_to(earth[:3], (len(catalogue.cluster_ids), 3)).copy()
    velocities = np.broadcast_to(earth[3:], positions.shape).copy()
    positions[:, :2] += relative_positions
    velocities[:, :2] += relative_velocities
    return EndpointStates(positions, velocities, earth[:3].copy(), earth[3:].copy(),
                          references, np.asarray(angles))


@dataclass(frozen=True)
class InvariantBounds:
    energy_min: np.ndarray
    energy_max: np.ndarray
    h_min: np.ndarray
    h_max: np.ndarray
    eccentricity: np.ndarray
    eccentricity_radius: np.ndarray


def invariant_bounds(endpoints, radius_v, mu=MU_SUN):
    """Conservative Kepler invariants for fixed positions and velocity disks.

    Energy and angular-momentum intervals are exact. The eccentricity-vector
    ball follows from its exact quadratic expansion about the disk centre.
    """
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("The gravitational parameter must be positive and finite.")
    r = np.asarray(endpoints.positions, dtype=float)[:, :2]
    u = np.asarray(endpoints.velocities, dtype=float)[:, :2]
    rho = np.asarray(radius_v, dtype=float)
    if r.shape != u.shape or rho.shape != (len(r),):
        raise ValueError("One planar state and one disk radius are needed per endpoint.")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(rho)):
        raise ValueError("Invariant-bound inputs must be finite.")
    if np.any(rho < 0):
        raise ValueError("Velocity disk radii cannot be negative.")
    R, U = np.linalg.norm(r, axis=1), np.linalg.norm(u, axis=1)
    if np.any(R <= 0):
        raise ValueError("Heliocentric endpoint positions cannot be zero.")
    h = r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]
    energy_min = 0.5 * np.maximum(0.0, U - rho) ** 2 - mu / R
    energy_max = 0.5 * (U + rho) ** 2 - mu / R
    ru = np.einsum("ij,ij->i", r, u)
    eccentricity = ((U ** 2)[:, None] * r - ru[:, None] * u) / mu - r / R[:, None]
    A = (2 * r[:, :, None] * u[:, None, :] - u[:, :, None] * r[:, None, :]
         - ru[:, None, None] * np.eye(2)[None, :, :])
    # The remainder is |delta|^2*r - (r.delta)*delta, with norm <= R*rho^2.
    eccentricity_radius = (np.linalg.norm(A, ord=2, axis=(1, 2)) * rho + R * rho ** 2) / mu
    return InvariantBounds(energy_min, energy_max, h - R * rho, h + R * rho,
                           eccentricity, eccentricity_radius)


def candidate_pairs(departure, arrival):
    """Return pairs passing necessary invariant tests; this is not a TOF test."""
    # Padding covers floating-point evaluations of nearly identical invariants.
    eps = np.finfo(float).eps
    e_scale = max(1.0, float(np.max(np.abs(departure.energy_min), initial=0)),
                  float(np.max(np.abs(departure.energy_max), initial=0)),
                  float(np.max(np.abs(arrival.energy_min), initial=0)),
                  float(np.max(np.abs(arrival.energy_max), initial=0)))
    h_scale = max(1.0, float(np.max(np.abs(departure.h_min), initial=0)),
                  float(np.max(np.abs(departure.h_max), initial=0)),
                  float(np.max(np.abs(arrival.h_min), initial=0)),
                  float(np.max(np.abs(arrival.h_max), initial=0)))
    e_tol = 1e-10 + 256 * eps * e_scale
    h_tol = 1e-8 + 256 * eps * h_scale
    possible = ((departure.energy_min[:, None] <= arrival.energy_max[None, :] + e_tol)
                & (arrival.energy_min[None, :] <= departure.energy_max[:, None] + e_tol)
                & (departure.h_min[:, None] <= arrival.h_max[None, :] + h_tol)
                & (arrival.h_min[None, :] <= departure.h_max[:, None] + h_tol))
    c, s = np.nonzero(possible)
    delta_e = departure.eccentricity[c] - arrival.eccentricity[s]
    bound = departure.eccentricity_radius[c] + arrival.eccentricity_radius[s]
    ecc_scale = 1 + np.maximum(np.linalg.norm(departure.eccentricity[c], axis=1),
                                np.linalg.norm(arrival.eccentricity[s], axis=1))
    keep = np.einsum("ij,ij->i", delta_e, delta_e) <= (bound + 1e-11 * ecc_scale) ** 2
    return c[keep], s[keep]


@dataclass(frozen=True)
class Transfer:
    soi_index: int
    tof_index: int
    jd_departure: float
    jd_arrival: float
    tof_days: float
    departure_cluster_row: int
    arrival_cluster_row: int
    branch_index: int
    revolutions: int
    r0: np.ndarray
    r1: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    departure_residual: float
    arrival_residual: float
    departure_relative_speed: float
    arrival_relative_speed: float
    departure_reference_jd: float
    arrival_reference_jd: float


@dataclass
class ScanResult:
    transfers: list
    total_pairs: int
    attempted_pairs: int
    lambert_branches: int
    lambert_failures: int


@dataclass(frozen=True)
class ArrivalCache:
    epochs: np.ndarray
    indices: np.ndarray
    earth_states: np.ndarray
    moon_angles: np.ndarray


def _epoch_grid(values, name, positive=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    return values


def prepare_arrival_cache(catalogue, jd_soi_grid, tof_grid):
    """Cache exact arrival epochs and future-ETD phases, with bounded work chunks."""
    jd_soi_grid = _epoch_grid(jd_soi_grid, "SOI epochs")
    tof_grid = _epoch_grid(tof_grid, "Times of flight", positive=True)
    epochs, inverse = np.unique(jd_soi_grid[:, None] + tof_grid[None, :], return_inverse=True)
    indices = inverse.reshape((jd_soi_grid.size, tof_grid.size))
    earth_states = np.empty((epochs.size, 6))
    angles = np.empty((epochs.size, len(catalogue.cluster_ids)))
    for start in range(0, epochs.size, _EPHEMERIS_CHUNK_SIZE):
        stop = min(epochs.size, start + _EPHEMERIS_CHUNK_SIZE)
        earth_states[start:stop] = _earth_states(epochs[start:stop])
    rows_per_chunk = max(1, _EPHEMERIS_CHUNK_SIZE // len(catalogue.cluster_ids))
    for start in range(0, epochs.size, rows_per_chunk):
        stop = min(epochs.size, start + rows_per_chunk)
        references = epochs[start:stop, None] + catalogue.tau_days[None, :]
        angles[start:stop] = moon_phase(references)
    return ArrivalCache(epochs, indices, earth_states, angles)


def solve_soi(catalogue, jd_soi, tof_days, *, soi_index=0, mu=MU_SUN,
              multi_revs=2, prefilter=True, arrival_cache: Optional[ArrivalCache] = None):
    """Solve every retained pair and every returned Lambert branch at one SOI.

    The prefilter consists solely of necessary two-body invariant conditions.
    Failed Lambert calls are counted; SPICE, input, and programming errors propagate.
    """
    jd_soi = float(jd_soi)
    if not np.isfinite(jd_soi) or not np.isfinite(mu) or mu <= 0:
        raise ValueError("The departure epoch and positive gravitational parameter must be finite.")
    if not isinstance(multi_revs, (int, np.integer)) or multi_revs < 0:
        raise ValueError("multi_revs must be a nonnegative integer.")
    if not isinstance(soi_index, (int, np.integer)) or soi_index < 0:
        raise ValueError("soi_index must be a nonnegative integer.")
    tof_days = _epoch_grid(tof_days, "Times of flight", positive=True)
    n = len(catalogue.cluster_ids)
    result = ScanResult([], n * n * tof_days.size, 0, 0, 0)
    if not tof_days.size:
        return result
    departure = build_endpoints(catalogue, jd_soi)
    # The identical roundoff allowance is used in the final residual test.
    radii = catalogue.radius_v + VELOCITY_ATOL
    departure_bounds = invariant_bounds(departure, radii, mu) if prefilter else None
    if not prefilter:
        all_c, all_s = np.repeat(np.arange(n), n), np.tile(np.arange(n), n)
    for j, tof in enumerate(tof_days):
        jd_arrival = jd_soi + float(tof)
        if arrival_cache is None:
            arrival = build_endpoints(catalogue, jd_arrival, incoming=True)
        else:
            k = arrival_cache.indices[soi_index, j]
            if arrival_cache.epochs[k] != jd_arrival:
                raise ValueError("Arrival cache does not match the SOI/TOF grid.")
            arrival = build_endpoints(catalogue, jd_arrival, incoming=True,
                                      earth_state=arrival_cache.earth_states[k],
                                      moon_angles=arrival_cache.moon_angles[k])
        if prefilter:
            c_rows, s_rows = candidate_pairs(departure_bounds, invariant_bounds(arrival, radii, mu))
        else:
            c_rows, s_rows = all_c, all_s
        for c, s in zip(c_rows, s_rows):
            result.attempted_pairs += 1
            try:
                lambert = pk.lambert_problem(r0=departure.positions[c], r1=arrival.positions[s],
                                             tof=float(tof) * DAY, mu=mu, cw=False,
                                             multi_revs=int(multi_revs))
            except (ValueError, RuntimeError):
                result.lambert_failures += 1
                continue
            v0s, v1s = lambert.v0, lambert.v1
            if len(v0s) != len(v1s):
                raise RuntimeError("Lambert returned inconsistent departure/arrival branches.")
            result.lambert_branches += len(v0s)
            for branch_index, (v0, v1) in enumerate(zip(v0s, v1s)):
                v0, v1 = np.asarray(v0, dtype=float), np.asarray(v1, dtype=float)
                if (v0.shape != (3,) or v1.shape != (3,)
                        or not np.all(np.isfinite(v0)) or not np.all(np.isfinite(v1))):
                    raise RuntimeError("Lambert returned a malformed or nonfinite velocity branch.")
                residual0 = float(np.linalg.norm(v0 - departure.velocities[c]))
                residual1 = float(np.linalg.norm(v1 - arrival.velocities[s]))
                if residual0 > radii[c] or residual1 > radii[s]:
                    continue
                geo_v0, geo_v1 = v0 - departure.earth_velocity, v1 - arrival.earth_velocity
                geo_r0 = departure.positions[c] - departure.earth_position
                geo_r1 = arrival.positions[s] - arrival.earth_position
                if np.dot(geo_r0, geo_v0) <= 0 or np.dot(geo_r1, geo_v1) >= 0:
                    continue
                result.transfers.append(Transfer(
                    soi_index=int(soi_index), tof_index=j, jd_departure=jd_soi,
                    jd_arrival=jd_arrival, tof_days=float(tof),
                    departure_cluster_row=int(c), arrival_cluster_row=int(s),
                    branch_index=branch_index, revolutions=(branch_index + 1) // 2,
                    r0=departure.positions[c].copy(), r1=arrival.positions[s].copy(),
                    v0=v0.copy(), v1=v1.copy(), departure_residual=residual0,
                    arrival_residual=residual1, departure_relative_speed=float(np.linalg.norm(geo_v0)),
                    arrival_relative_speed=float(np.linalg.norm(geo_v1)),
                    departure_reference_jd=float(departure.reference_epochs[c]),
                    arrival_reference_jd=float(arrival.reference_epochs[s]),
                ))
    return result
