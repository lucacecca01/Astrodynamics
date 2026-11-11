"""Conservative exterior-SOI checks for an accepted heliocentric Kepler arc.

The Earth ephemeris and planar convention are shared with GA19.  SPICE kernels
must already be loaded by the caller.  Sampling and minimisation identify close
approaches; adaptive chord bounds prevent accepting an unresolved passage between
samples.  These bounds assume heliocentric Earth acceleration <=1e-4 km/s² and
speed <=50 km/s over the requested ephemeris interval.  They are intentionally
conservative for the terrestrial ephemerides used by GA20, rather than a proof
for arbitrary observers, ephemerides, or force models.
"""

import math

import numpy as np
import pykep as pk
from scipy.optimize import brentq, minimize_scalar

from ga19_transfer_search import earth_state


_DAY = 86400.0
_EARTH_ACCELERATION_BOUND = 1e-4
_EARTH_SPEED_BOUND = 50.0
_MIN_ADAPTIVE_INTERVAL_SECONDS = 0.25


def validate_exterior_arc(
    r0, v0, jd0, tof_days, soi_radius, mu_sun, *,
    step_days=0.25, distance_tolerance_km=0.01,
):
    """Return whether this Kepler arc remains outside the terrestrial SOI.

    Distances may fall below the radius only by ``distance_tolerance_km``.  If
    an endpoint is on that surface within tolerance, its relative radial speed
    must be outgoing at the start and incoming at the end.  An asteroid endpoint
    far from the SOI has no such sign requirement.  An unresolved interval at
    0.25-second resolution is rejected conservatively.

    This validates a specified arc; it does not exclude other flybys/epochs or
    replace an N-body propagation.  Invalid input raises ``ValueError``; an arc
    that cannot be propagated or certified returns ``False``.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    scalars = np.asarray(
        [jd0, tof_days, soi_radius, mu_sun, step_days, distance_tolerance_km],
        dtype=float,
    )
    if (r0.shape != (3,) or v0.shape != (3,)
            or not np.isfinite(r0).all() or not np.isfinite(v0).all()
            or not np.isfinite(scalars).all()):
        raise ValueError("Arc states and parameters must be finite Cartesian values.")
    jd0, tof_days, soi_radius, mu_sun, step_days, tolerance = scalars
    if min(tof_days, soi_radius, mu_sun, step_days) <= 0 or tolerance < 0:
        raise ValueError("TOF, radii, mu and step must be positive; tolerance nonnegative.")
    if np.linalg.norm(r0) == 0:
        return False

    duration = float(tof_days * _DAY)
    count = max(1, int(math.ceil(tof_days / step_days)))
    times = np.linspace(0.0, duration, count + 1)
    try:
        states = np.asarray(pk.propagate_lagrangian_grid(
            rv=[r0, v0], tofs=times, mu=float(mu_sun),
        ))
    except (RuntimeError, ValueError, FloatingPointError):
        return False
    if states.shape != (count + 1, 2, 3) or not np.isfinite(states).all():
        return False
    er, ev = earth_state(jd0 + times / _DAY)
    positions, velocities = states[:, 0] - er, states[:, 1] - ev
    distances = np.linalg.norm(positions, axis=1)
    radial_products = np.einsum("ij,ij->i", positions, velocities)
    threshold = float(soi_radius - tolerance)
    if np.any(distances < threshold):
        return False
    if abs(distances[0] - soi_radius) <= tolerance and radial_products[0] <= 0:
        return False
    if abs(distances[-1] - soi_radius) <= tolerance and radial_products[-1] >= 0:
        return False

    cache = {
        float(t): (p, v) for t, p, v in zip(times, positions, velocities)
    }

    def relative(t):
        t = float(t)
        if t not in cache:
            r, v = pk.propagate_lagrangian([r0, v0], t, float(mu_sun))
            earth_r, earth_v = earth_state(float(jd0 + t / _DAY))
            cache[t] = np.asarray(r) - earth_r, np.asarray(v) - earth_v
        return cache[t]

    def distance(t):
        return float(np.linalg.norm(relative(t)[0]))

    def radial_product(t):
        p, v = relative(t)
        return float(np.dot(p, v))

    # Check both sampled minima and radial-velocity stationarity brackets.
    # The latter catches a short fly-through even when both sample endpoints
    # happen to be well outside the SOI.
    brackets = set()
    local = np.flatnonzero(
        (distances[1:-1] <= distances[:-2])
        & (distances[1:-1] <= distances[2:])
    ) + 1
    brackets.update((float(times[i - 1]), float(times[i + 1])) for i in local)
    stationary = np.flatnonzero(
        (radial_products[:-1] <= 0) & (radial_products[1:] >= 0)
    )
    try:
        for i in stationary:
            a, b = float(times[i]), float(times[i + 1])
            root = brentq(radial_product, a, b, xtol=1e-5)
            if distance(root) < threshold:
                return False
            brackets.add((a, b))
        for a, b in brackets:
            result = minimize_scalar(
                lambda offset: distance(a + offset),
                bounds=(0.0, b - a), method="bounded", options={"xatol": 0.01},
            )
            if not result.success or result.fun < threshold:
                return False

        # On each segment the position deviates from its chord by at most
        # A*dt²/8.  The Kepler perihelion gives a global spacecraft-acceleration
        # bound; the ephemeris assumptions supply the Earth contribution.
        h = np.cross(r0, v0)
        eccentricity = np.linalg.norm(np.cross(v0, h) / mu_sun - r0 / np.linalg.norm(r0))
        perihelion = np.dot(h, h) / (mu_sun * (1.0 + eccentricity))
        if not np.isfinite(perihelion) or perihelion <= 0:
            return False
        acceleration = mu_sun / perihelion**2 + _EARTH_ACCELERATION_BOUND
        # Absolute JD addition quantises the Earth epoch; cover one JD ulp
        # conservatively using the stated Earth-speed bound.
        roundoff = _EARTH_SPEED_BOUND * _DAY * np.spacing(abs(jd0) + tof_days)
        stack = [
            (float(a), float(b), pa, pb)
            for a, b, pa, pb in zip(times[:-1], times[1:], positions[:-1], positions[1:])
        ]
        while stack:
            a, b, pa, pb = stack.pop()
            delta = pb - pa
            length2 = float(np.dot(delta, delta))
            fraction = np.clip(-np.dot(pa, delta) / length2, 0, 1) if length2 else 0.0
            chord_minimum = float(np.linalg.norm(pa + fraction * delta))
            error_bound = acceleration * (b - a)**2 / 8.0 + roundoff
            if chord_minimum - error_bound >= threshold:
                continue
            midpoint = (a + b) / 2.0
            pm = relative(midpoint)[0]
            if not np.isfinite(pm).all() or np.linalg.norm(pm) < threshold:
                return False
            if b - a <= _MIN_ADAPTIVE_INTERVAL_SECONDS:
                return False
            stack.extend(((a, midpoint, pa, pm), (midpoint, b, pm, pb)))
    except (RuntimeError, ValueError, FloatingPointError):
        return False
    return True
