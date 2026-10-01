from Celestial_Mechanics import Integration, Transformations
from pathlib import Path
from multiprocessing import get_context
import hashlib
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.integrate import cumulative_trapezoid
from scipy.stats import skew

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from diptest import diptest
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

from scipy.optimize import brentq
import csv


SAVE = True
PLOT_CLUSTERS = False
PLOT_MEDOIDS = False



if not PLOT_CLUSTERS:
    plt.ioff()



# Define constants
G = 6.67430e-20              # Gravitational constant in km^3 kg^-1 s^-2
d = 384400                   # Average distance between Earth and Moon in km
m1 = 5.974e24                # Mass of Earth in kg
m2 = 7.348e22                # Mass of Moon in kg
ms = 1.989e30                # Mass of Sun in kg
d_E = 1.496e8                # Average distance between Earth and Sun in km
R_E = 6378                   # Earth radius in km
R_L = 1737                   # Moon radius in km
day = 86400                  # Day in s
M = m1 + m2                  # Total mass of the Earth-Moon system in kg
mu = m2/M                    # Moon-to-total mass ratio
n = np.sqrt(G * M / d**3)    # Moon’s mean motion
TU = 1 / n                   # Moon’s mean revolution period
E_SOI = (m1 / ms)**(2/5) * d_E
M_SOI = (m2 / m1)**(2/5) * d
mu_earth = G * m1



# Define masses and integration parameters
masses = [m1, m2, 0]
T_min = 0

T_max_f = 4 * np.pi
dt_f = 0.0001

T_max_b = -2 * np.pi
dt_b = -0.0001

N_branch = 500




# Define function to build initial conditions database from file
def build_x0_database(filename, mu, collisions = "all"):

    fields = ["x20", "y20", "sigma0", "NRev", "NRevALL", "TimeEn", "CasoEn", "Coll", "min_r2", "min_r2_1stRev", "a", "e", "aBW", "eBW", "GAMMA"]

    with open(filename, encoding = "utf-8-sig") as file:
        rows = [np.fromstring(line, sep = " ") for line in file if line.strip()]

    data = dict(zip(fields, rows))
    data["GAMMA"] = np.broadcast_to(data["GAMMA"], data["x20"].shape)
    indices = np.arange(len(data["x20"]))

    collision_tau = 2 * np.pi * data["Coll"]

    if collisions == "exclude":
        indices = indices[(data["Coll"] < 0) | (collision_tau > T_max_f)]

    elif collisions == "only":
        indices = indices[(data["Coll"] > 0) & (collision_tau <= T_max_f)]

    elif collisions != "all":
        raise ValueError("collisions: 'all', 'exclude' oppure 'only'.")

    data = {name: values[indices] for name, values in data.items()}
    x20, y20 = data["x20"], data["y20"]

    def equilibrium(x):
        return x - (1 - mu)/(x + mu)**2 + mu/(1 - mu - x)**2

    L1 = brentq(equilibrium, -mu + 1e-10, 1 - mu - 1e-10, xtol = 1e-14)
    CJ_L1 = L1**2 + 2*(1 - mu)/(L1 + mu) + 2*mu/(1 - mu - L1)
    CJ_L4 = 3 - mu + mu**2
    CJ = CJ_L1 + data["GAMMA"]*(CJ_L4 - CJ_L1)

    r1 = np.sqrt((1 + x20)**2 + y20**2)
    r2 = np.sqrt(x20**2 + y20**2)
    v2 = np.sqrt(2*mu/r2)

    sin_sigma = ((1 - mu)**2 + 2*(1 - mu)*x20 + 2*(1 - mu)/r1 - CJ)/(2*r2*v2)

    if np.any(~np.isfinite(sin_sigma) | (np.abs(sin_sigma) > 1 + 1e-12)):
        raise ValueError("Condizioni iniziali non ammissibili: controlla mu e GAMMA.")

    sigma = np.arcsin(np.clip(sin_sigma, -1, 1))
    alpha = sigma - np.arctan2(y20, -x20)

    X0 = np.zeros((len(x20), 6))
    X0[:, 0] = 1 - mu + x20
    X0[:, 1] = y20
    X0[:, 3] = v2*np.cos(alpha) + y20
    X0[:, 4] = v2*np.sin(alpha) - x20

    data["column_indices"] = indices
    data["tau_ref"] = 2*np.pi*data["TimeEn"]
    data["mu"] = mu

    return X0, data



#Define stop condition for exiting Earth's sphere of influence
def earth_SOI_exit(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)

    return E_SOI / d - r_earth



# Define perigee events
def earth_perigee_b(t, x, mu):

    return (x[0] + mu)*x[3] + x[1]*x[4] + x[2]*x[5]



# Define perigee events
def earth_perigee_f(t, x, mu):

    return (x[0] + mu)*x[3] + x[1]*x[4] + x[2]*x[5]




# Define lunar distance event
def moon_SOI_crossing(t, x, mu):

    r_moon = np.linalg.norm(x[:3] - np.array([1 - mu, 0, 0]))

    return r_moon - 2.5 * M_SOI / d




# Define collision events
def collision(t, x, mu):

    r_earth = np.linalg.norm(x[:3] - np.array([-mu, 0, 0]))
    r_moon = np.linalg.norm(x[:3] - np.array([1 - mu, 0, 0]))

    return min(r_earth - R_E / d, r_moon - R_L / d)



# Compute geocentric osculating parameters
def orbital_parameters(t, x):

    x_earth = np.asarray(x, dtype=float).copy()
    x_earth[0] += mu
    x_inertial = Transformations.CR3BP_to_inertial(x_earth, 1, t)[:, 0]

    r = x_inertial[:3] * d
    v = x_inertial[3:] * d / TU

    with np.errstate(divide="ignore", invalid="ignore"):
        _, a, e, _, _, w, _ = Transformations.coe_from_sv(r, v, mu_earth)

    eps = 0.5 * np.dot(v, v) - mu_earth / np.linalg.norm(r)
    parameters = np.array([eps, e, w])
    parameters[~np.isfinite(parameters)] = np.nan

    return parameters



# Select the first eligible perigee
def first_perigee_parameters(sol):

    collision_time = abs(sol.t_events[2][0] - T_min) if len(sol.t_events[2]) else np.inf

    if collision(T_min, sol.y[:, 0], mu) <= 0:
        return np.full(3, np.nan), np.nan

    for t, x in zip(sol.t_events[1], sol.y_events[1]):

        if abs(t - T_min) >= collision_time:
            break

        r_earth = np.linalg.norm(x[:3] - np.array([-mu, 0, 0]))
        r_moon = np.linalg.norm(x[:3] - np.array([1 - mu, 0, 0]))

        if abs(t - T_min) > 1e-10 and r_earth > R_E / d and r_moon > R_L / d:
            return orbital_parameters(t, x), t

    return np.full(3, np.nan), np.nan



# Select the first lunar distance crossing before a collision
def first_section_parameters(sol):

    collision_time = abs(sol.t_events[2][0] - T_min) if len(sol.t_events[2]) else np.inf

    if collision(T_min, sol.y[:, 0], mu) <= 0:
        return np.full(3, np.nan), np.nan

    for t, x in zip(sol.t_events[1], sol.y_events[1]):

        if abs(t - T_min) >= collision_time:
            break

        if abs(t - T_min) > 1e-10:
            return orbital_parameters(t, x), t

    return np.full(3, np.nan), np.nan




# Define integration function
def integrate_one(x0, T_min, T_max, dt, events=None, rtol=1e-9):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='CR3BP',
        integrator='scipy',
        method='DOP853',
        rtol=rtol,
        events=events,
        dense_output=True
    )

    return sol



# Define integration and sampling function
def integrate_and_sample_one(x0):

    solution_b = integrate_one(x0, T_min, T_max_b, dt_b, (earth_SOI_exit, moon_SOI_crossing, collision))

    sampled_b = sample_branch(solution_b)
    oe_b, t_b = first_section_parameters(solution_b)

    del solution_b


    solution_f = integrate_one(x0, T_min, T_max_f, dt_f, (earth_SOI_exit, moon_SOI_crossing, collision))

    sampled_f = sample_branch(solution_f)
    oe_f, t_f = first_section_parameters(solution_f)

    del solution_f


    assert np.allclose(sampled_b[:, 0], sampled_f[:, 0])

    trajectory = np.concatenate((sampled_b[:, ::-1], sampled_f[:, 1:]), axis=1)
    elements = np.array([oe_b, oe_f])

    return trajectory, elements, np.array([t_b, t_f])



# Define sample function based on curvature
def sample_branch(sol):

    t = sol.t
    trajectory = sol.y

    derivative = Integrator.CR3BP_ODE(0, trajectory, mu)

    velocity = derivative[:3]
    acceleration = derivative[3:]

    v_squared = np.sum(velocity**2, axis=0)

    cross_norm = np.linalg.norm(np.cross(velocity, acceleration, axis=0), axis=0)

    curvature_rate = np.divide(cross_norm, v_squared, out=np.zeros_like(v_squared), where=v_squared > 1e-14)

    progress = np.abs(t - t[0])

    cumulative_curvature = cumulative_trapezoid(curvature_rate, x=progress, initial=0)

    cumulative_curvature = np.maximum.accumulate(cumulative_curvature)

    cumulative_curvature, unique_ids = np.unique(cumulative_curvature, return_index=True)

    if cumulative_curvature[-1] <= 1e-14:

        target_times = np.linspace(t[0], t[-1], N_branch)

    else:

        targets = np.linspace(0, cumulative_curvature[-1], N_branch)
        target_times = np.interp(targets, cumulative_curvature, t[unique_ids])

        target_times[0] = t[0]
        target_times[-1] = t[-1]

    return sol.sol(target_times)



# Define normalization function for features
def normalize_block(features):

    centered = features - np.mean(features, axis=0)
    scale = np.linalg.norm(centered) / np.sqrt(len(centered))

    if scale <= 1e-14:
        raise ValueError("Feature block has zero variance.")

    centered /= scale

    return centered



# Full-space dispersion and maximum XY-pair deviation, in bounded batches.
def geometric_statistics(features, ids, batch_size=512):
    if features.ndim != 2 or not features.shape[1] or features.shape[1] % 2:
        raise ValueError("Servono coppie di componenti geometriche (x, y).")
    ids = np.asarray(ids, dtype=int)
    if not len(ids):
        raise ValueError("Servono almeno una traiettoria e una feature.")
    center = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, len(ids), batch_size):
        center += np.sum(features[ids[start:start + batch_size]], axis=0, dtype=np.float64)
    center /= len(ids)
    squared_distances = np.empty(len(ids), dtype=np.float64)
    max_pointwise_squared = 0.
    for start in range(0, len(ids), batch_size):
        residual = features[ids[start:start + batch_size]].astype(np.float64) - center
        squared_distances[start:start + len(residual)] = np.einsum("ij,ij->i", residual, residual)
        # Each block interleaves x/y (or tangent x/y) at successive samples.
        pairs = residual.reshape(len(residual), -1, 2)
        max_pointwise_squared = max(max_pointwise_squared,
                                    float(np.einsum("ijk,ijk->ij", pairs, pairs).max()))
    variance_sum = squared_distances.mean()
    mean_std = np.sqrt(variance_sum / features.shape[1])
    radial_sigma = np.sqrt(squared_distances.max() / variance_sum) if variance_sum > 0 else 0.
    return float(mean_std), float(radial_sigma), center, squared_distances, np.sqrt(max_pointwise_squared)


class GeometricClusterConstraint:
    """Bound every normalized XY pair's distance from its cluster mean."""

    def __init__(self, features, std_limit, min_size, sigma_limit, *,
                 dip_alpha=0.01, gap_fraction=0.2, strong_gap_fraction=0.5,
                 outlier_sigma=3.):
        self.features = np.asarray(features)
        if (self.features.ndim != 2 or not self.features.shape[1] or self.features.shape[1] % 2
                or not np.issubdtype(self.features.dtype, np.floating)):
            raise ValueError("Servono feature floating point (N, D), con D pari per le coppie (x, y).")
        for start in range(0, len(features), 512):
            if not np.all(np.isfinite(features[start:start + 512])):
                raise ValueError("Feature geometriche non finite.")
        if np.isnan(std_limit) or std_limit <= 0:
            raise ValueError("STD_MEAN_LIMIT deve essere positivo.")
        if not isinstance(min_size, (int, np.integer)) or isinstance(min_size, bool) or min_size < 1:
            raise ValueError("MIN_CLUSTER_SIZE deve essere un intero positivo.")
        if np.isnan(sigma_limit) or sigma_limit <= 0:
            raise ValueError("MAX_SIGMA deve essere positivo.")
        if not np.isfinite(dip_alpha) or not 0 <= dip_alpha < 1:
            raise ValueError("DIP_ALPHA deve essere in [0, 1).")
        if not np.isfinite(gap_fraction) or not 0 <= gap_fraction < 1:
            raise ValueError("GAP_FRACTION deve essere in [0, 1).")
        if np.isnan(strong_gap_fraction) or strong_gap_fraction <= 0 or strong_gap_fraction < gap_fraction:
            raise ValueError("STRONG_GAP_FRACTION deve essere positivo e >= GAP_FRACTION.")
        if np.isnan(outlier_sigma) or outlier_sigma <= 0:
            raise ValueError("OUTLIER_SIGMA deve essere positivo.")
        self.std_limit, self.min_size, self.sigma_limit = std_limit, min_size, sigma_limit
        self.alpha, self.gap_fraction = dip_alpha, gap_fraction
        self.strong_gap_fraction, self.outlier_sigma = strong_gap_fraction, outlier_sigma
        self.stats_cache, self.shape_cache = {}, {}

    @staticmethod
    def key(ids):
        return hashlib.sha256(np.sort(np.asarray(ids, dtype=np.int64)).tobytes()).digest()

    def statistics(self, ids):
        key = self.key(ids)
        if key not in self.stats_cache:
            std, sigma, _, _, max_deviation = geometric_statistics(self.features, ids)
            self.stats_cache[key] = (std, sigma, max_deviation)
        return self.stats_cache[key]

    def split(self, ids):
        ids = np.sort(np.asarray(ids, dtype=int))
        key = self.key(ids)
        if key not in self.shape_cache:
            self.shape_cache[key] = self._gap_split(ids)
        return self.shape_cache[key]

    def _gap_split(self, ids):
        if len(ids) < 3 or (self.alpha == 0 and np.isinf(self.strong_gap_fraction)
                            and np.isinf(self.outlier_sigma)):
            return None
        _, _, center, distance_squared, _ = geometric_statistics(self.features, ids)
        if distance_squared.max() <= 1e-24:
            return None
        # PCA is only a shape diagnostic. KMeans, STD and sigma use all original features.
        sample = ids[np.linspace(0, len(ids) - 1, min(len(ids), 1024), dtype=int)]
        values = self.features[sample].astype(np.float64) - center
        rank = min(3, values.shape[1], len(values) - 1)
        with threadpool_limits(limits=2):
            _, _, axes = randomized_svd(values, n_components=rank, n_iter=3, random_state=42)
            projections = np.empty((len(ids), rank + 1))
            for start in range(0, len(ids), 512):
                residual = self.features[ids[start:start + 512]].astype(np.float64) - center
                projections[start:start + len(residual), :rank] = residual @ axes.T
        projections[:, -1] = np.sqrt(distance_squared)
        support = max(3, int(np.ceil(0.1 * len(ids))))
        best_gap, result = self.gap_fraction, None
        for column in range(projections.shape[1]):
            order = np.argsort(projections[:, column], kind="stable")
            ordered = projections[order, column]
            span = np.ptp(ordered)
            if span <= 1e-12:
                continue
            gaps = np.diff(ordered)
            choices = []
            if len(ids) >= 6:
                at = 2 + int(np.argmax(gaps[2:len(ids) - 3]))
                if gaps[at] / span > self.strong_gap_fraction:
                    choices.append(at)
                if self.alpha > 0:
                    _, pvalue = diptest(ordered, sort_x=False, boot_pval=False)
                    eligible = gaps[support - 1:len(ids) - support]
                    if pvalue < self.alpha / projections.shape[1] and len(eligible):
                        choices.append(support - 1 + int(np.argmax(eligible)))
            # A small detached radial tail can escape the dip test.
            if column == rank and np.isfinite(self.outlier_sigma):
                median = np.median(ordered)
                scale = max(1.4826 * np.median(np.abs(ordered - median)), 1e-12)
                isolated = ordered[1:] > median + self.outlier_sigma * scale
                eligible = np.flatnonzero(isolated & (gaps / span > self.gap_fraction))
                if len(eligible):
                    choices.append(int(eligible[np.argmax(gaps[eligible])]))
            for at in choices:
                if gaps[at] / span > best_gap:
                    best_gap = gaps[at] / span
                    result = np.zeros(len(ids), dtype=np.int8)
                    result[order[at + 1:]] = 1
        return result

    def __call__(self, ids):
        if len(ids) < self.min_size:
            return False
        _, sigma, max_deviation = self.statistics(ids)
        return (max_deviation < self.std_limit and sigma <= self.sigma_limit
                and self.split(ids) is None)


def cluster_groups(labels):
    ids = np.flatnonzero(labels >= 0)
    ids = ids[np.argsort(labels[ids], kind="stable")]
    return np.split(ids, np.flatnonzero(np.diff(labels[ids])) + 1) if len(ids) else []


def split_geometric_clusters(labels, features, validator, candidates, seed=42):
    pending = [np.sort(ids) for ids in cluster_groups(labels)]
    accepted = []
    splits = gap_splits = 0
    with threadpool_limits(limits=2):
        while pending:
            ids = pending.pop()
            candidates.append(ids)
            if len(ids) < validator.min_size:
                continue
            if validator(ids):
                accepted.append(ids)
                continue
            if len(ids) <= max(validator.min_size, 1):
                continue
            split = validator.split(ids)
            if split is None:
                if validator.statistics(ids)[0] == 0:
                    continue
                split = KMeans(n_clusters=2, n_init=5, random_state=seed,
                               max_iter=300, tol=1e-4, algorithm="lloyd").fit_predict(features[ids])
                if len(np.unique(split)) != 2:
                    continue
            else:
                gap_splits += 1
            splits += 1
            pending.extend(ids[split == side] for side in (0, 1))
    result = np.full(len(features), -1, dtype=int)
    for c, ids in enumerate(sorted(accepted, key=lambda ids: int(ids[0]))):
        result[ids] = c
    print(f"Split: {splits} divisioni ({gap_splits} nei vuoti), "
          f"{len(accepted)} cluster validi, {np.count_nonzero(result < 0)} non assegnate.", flush=True)
    return result


def add_neighbor_candidates(labels, features, validator, candidates):
    groups = cluster_groups(labels)
    if len(groups) < 2:
        return
    stats = [geometric_statistics(features, ids)[:3] for ids in groups]
    centers = np.array([item[2] for item in stats])
    with threadpool_limits(limits=2):
        neighbors = NearestNeighbors(n_neighbors=min(5, len(groups)), algorithm="brute").fit(centers)
        nearest = neighbors.kneighbors(centers, return_distance=False)
    pairs = {tuple(sorted((i, int(j)))) for i, row in enumerate(nearest) for j in row if i != j}
    for i, j in sorted(pairs):
        ni, nj = len(groups[i]), len(groups[j])
        # RMS XY deviation is a lower bound on the maximum pointwise deviation.
        delta_squared = np.sum((centers[i] - centers[j])**2)
        pooled = ((ni * stats[i][0]**2 + nj * stats[j][0]**2) / (ni + nj)
                  + ni * nj / (ni + nj)**2 * delta_squared / features.shape[1])
        if np.sqrt(2 * pooled) < validator.std_limit:
            candidates.append(np.sort(np.r_[groups[i], groups[j]]))


def fuse_cluster_candidates(candidates, n_points, valid_group, max_clusters,
                            *, previous_labels=None, time_limit=30.):
    """Maximize coverage, then minimize K; previous assignments are not mandatory.

    valid_group(ids) defines all local constraints and must be deterministic.
    Previous groups are candidates and, when feasible, a solver-timeout fallback.
    Only disjointness and the global cluster cap live here.
    Returned candidates include rejected groups, for revalidation on later runs.
    """
    for name, value in (("n_points", n_points), ("max_clusters", max_clusters)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
            raise ValueError(f"{name} deve essere un intero non negativo.")
    if not callable(valid_group):
        raise TypeError("valid_group deve essere una funzione degli indici del cluster.")
    if not np.isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit deve essere finito e non negativo.")
    if previous_labels is None:
        previous_labels = np.full(n_points, -1, dtype=int)
    previous_labels = np.asarray(previous_labels)
    if (previous_labels.shape != (n_points,) or not np.issubdtype(previous_labels.dtype, np.integer)
            or np.any(previous_labels < -1)):
        raise ValueError("Etichette precedenti non valide.")
    previous_groups = cluster_groups(previous_labels)
    pool, seen = [], set()
    for members in list(candidates) + previous_groups:
        ids = np.asarray(members)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Ogni candidato deve contenere indici interi in un vettore.")
        if len(ids) == 0:
            continue
        ids = np.sort(ids.astype(np.int64, copy=True))
        if ids[0] < 0 or ids[-1] >= n_points or np.any(np.diff(ids) == 0):
            raise ValueError("Candidato con indici fuori intervallo o ripetuti.")
        key = ids.tobytes()
        if key not in seen:
            seen.add(key)
            pool.append(ids)
    groups = [ids for ids in pool if valid_group(ids)]
    result_labels = np.full(n_points, -1, dtype=int)
    if not groups or max_clusters == 0:
        print("Fusione: nessun cluster ammissibile.", flush=True)
        return result_labels, pool

    lookup = {ids.tobytes(): j for j, ids in enumerate(groups)}
    fallback = [lookup.get(np.asarray(ids, dtype=np.int64).tobytes()) for ids in previous_groups]
    fallback_valid = all(j is not None for j in fallback) and len(fallback) <= max_clusters
    fallback = np.asarray(fallback, dtype=int) if fallback_valid else np.empty(0, dtype=int)
    rows = np.concatenate(groups)
    cols = np.repeat(np.arange(len(groups)), [len(ids) for ids in groups])
    incidence = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_points, len(groups))).tocsr()
    # Compress identical incidence rows.
    patterns, representatives = set(), []
    for i in range(n_points):
        pattern = incidence.indices[incidence.indptr[i]:incidence.indptr[i + 1]]
        if not len(pattern):
            continue
        key = pattern.tobytes()
        if key not in patterns:
            patterns.add(key)
            representatives.append(i)
    matrix = vstack((incidence[representatives], csr_matrix(np.ones((1, len(groups)))))).tocsc()
    gains = np.array([len(ids) for ids in groups])
    # One additional assigned point outweighs any possible cluster-count change.
    cap = min(max_clusters, n_points, len(groups))
    cost = 1. - (cap + 1.) * gains
    result = milp(
        cost, integrality=np.ones(len(groups)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, 0., np.r_[np.ones(len(representatives)), cap]),
        options={"time_limit": time_limit, "mip_rel_gap": 0.},
    )
    chosen = fallback
    if result.x is not None and np.all(np.isfinite(result.x)):
        proposal = np.flatnonzero(result.x > .5)
        coverage = np.asarray(incidence[:, proposal].sum(axis=1)).ravel()
        feasible = len(proposal) <= cap and np.all(coverage <= 1)
        if feasible and cost[proposal].sum() < cost[chosen].sum():
            chosen = proposal

    selected = sorted((groups[j] for j in chosen), key=lambda ids: int(ids[0]))
    for c, ids in enumerate(selected):
        if not valid_group(ids) or np.any(result_labels[ids] >= 0):
            raise RuntimeError("La verifica finale della fusione ha rilevato un cluster non valido.")
        result_labels[ids] = c
    if len(selected) > max_clusters:
        raise RuntimeError("La fusione supera il massimo di cluster.")
    assigned = np.count_nonzero(result_labels >= 0)
    print(f"Fusione: {len(pool)} candidati, {len(groups)} validi; {len(selected)} cluster, "
          f"{assigned} punti assegnati, {np.count_nonzero(result_labels < 0)} non assegnati. "
          f"MILP: {result.message}; gap={getattr(result, 'mip_gap', None)}", flush=True)
    return result_labels, pool



def fusion_fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        array = np.atleast_1d(array)
        for start in range(0, len(array), 512):
            digest.update(np.ascontiguousarray(array[start:start + 512]).tobytes())
    return digest.hexdigest()


def load_fusion_cache(path, fingerprint, n_points, *, constraints_key=None):
    if path is None or not Path(path).exists():
        return [], None
    try:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["fingerprint"]) != fingerprint:
                print("Cache fusione ignorata: dati, ordine delle righe o feature diversi.", flush=True)
                return [], None
            members, offsets, labels = cache["members"], cache["offsets"], cache["labels"]
            if (members.ndim != 1 or offsets.ndim != 1 or not len(offsets)
                    or not np.issubdtype(members.dtype, np.integer)
                    or not np.issubdtype(offsets.dtype, np.integer)
                    or offsets[0] != 0 or offsets[-1] != len(members) or np.any(np.diff(offsets) <= 0)
                    or labels.shape != (n_points,) or not np.issubdtype(labels.dtype, np.integer)
                    or np.any(labels < -1) or np.any(members < 0) or np.any(members >= n_points)):
                raise ValueError("Struttura della cache non valida.")
            groups = list(np.split(members, offsets[1:-1])) if len(members) else []
            if constraints_key is not None and ("constraints_key" not in cache
                                                or str(cache["constraints_key"]) != constraints_key):
                print(f"Cache: vincoli diversi; riuso {len(groups)} candidati da rivalidare.", flush=True)
                return groups, None
        print(f"Cache fusione: {len(groups)} candidati e {np.count_nonzero(labels >= 0)} "
              "assegnazioni precedenti non vincolanti.", flush=True)
        return groups, labels
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Cache fusione non leggibile: {path}. Nessuna assegnazione precedente ignorata.") from exc


def save_fusion_cache(path, fingerprint, candidates, labels, *, constraints_key=None):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, fingerprint=fingerprint, labels=labels,
                                constraints_key="" if constraints_key is None else constraints_key,
                                members=np.concatenate(candidates) if candidates else np.empty(0, dtype=int),
                                offsets=np.r_[0, np.cumsum([len(ids) for ids in candidates], dtype=int)])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)



def refine_geometric_clustering(features, validator, base_counts, seeds, max_clusters, *,
                                candidates=(), previous_labels=None, time_limit=30.):
    if not isinstance(max_clusters, (int, np.integer)) or isinstance(max_clusters, bool) or max_clusters < 1:
        raise ValueError("MAX_CLUSTERS deve essere un intero positivo.")
    if (not base_counts or not seeds or any(not isinstance(k, (int, np.integer)) or k < 1 for k in base_counts)
            or any(not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds)):
        raise ValueError("Servono basi positive e seed interi non negativi.")
    pool = list(candidates)
    specs = [(k, seeds[0]) for k in base_counts]
    specs.extend((base_counts[-1], seed) for seed in seeds[1:])
    best = np.full(len(features), -1, dtype=int)
    for n_base, seed in specs:
        k = min(n_base, max(1, len(features)))
        if not len(features):
            break
        with threadpool_limits(limits=2):
            partition = KMeans(n_clusters=k, n_init=1, random_state=seed,
                               max_iter=300, tol=1e-4, algorithm="lloyd").fit_predict(features)
        print(f"Base KMeans geometrica: K={len(np.unique(partition))}, seed={seed}", flush=True)
        refined = split_geometric_clusters(partition, features, validator, pool, seed)
        add_neighbor_candidates(refined, features, validator, pool)
        groups = sorted(cluster_groups(refined), key=lambda ids: (-len(ids), int(ids[0])))[:max_clusters]
        fallback = np.full(len(features), -1, dtype=int)
        for c, ids in enumerate(groups):
            fallback[ids] = c
        score = (np.count_nonzero(fallback >= 0), -len(groups))
        if score > (np.count_nonzero(best >= 0), -len(cluster_groups(best))):
            best = fallback
    if previous_labels is not None:
        # Revalidate and split old groups; no old point is protected.
        refined = split_geometric_clusters(previous_labels, features, validator, pool, seeds[0])
        add_neighbor_candidates(refined, features, validator, pool)
        groups = cluster_groups(refined)
        if len(groups) <= max_clusters and (
                np.count_nonzero(refined >= 0), -len(groups)) > (
                np.count_nonzero(best >= 0), -len(cluster_groups(best))):
            best = refined
    labels, pool = fuse_cluster_candidates(pool, len(features), validator, max_clusters,
                                          previous_labels=best, time_limit=time_limit)
    if any(not validator(ids) for ids in cluster_groups(labels)):
        raise RuntimeError("Verifica finale dei vincoli geometrici fallita.")
    return labels, pool


def geometric_representatives(features, labels):
    representatives = []
    distances = np.full(len(features), np.nan)
    for ids in cluster_groups(labels):
        _, _, _, squared, _ = geometric_statistics(features, ids)
        representative = ids[np.argmin(squared)]
        representatives.append(representative)
        for start in range(0, len(ids), 512):
            rows = ids[start:start + 512]
            delta = features[rows].astype(np.float64) - features[representative]
            distances[rows] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return np.asarray(representatives, dtype=int), distances



# Compute statistics
def parameter_statistics(values, representative, circular=False):

    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    
    result = np.full(15, np.nan)
    result[:2] = len(valid), len(values) - len(valid)

    if len(valid) == 0:
        return result

    if np.isfinite(representative):
        errors = valid - representative

        if circular:
            errors = (errors + 180) % 360 - 180

        result[11:13] = np.percentile(np.abs(errors), 95), np.max(np.abs(errors))

    if circular:

        center = np.mean(np.exp(1j * np.radians(valid)))
        result[14] = abs(center)

        if abs(center) < 1e-8:
            return result
        
        result[13] = np.degrees(np.angle(center)) % 360
        valid = (valid - result[13] + 180) % 360 - 180

    p05, median, p95 = np.percentile(valid, [5, 50, 95])
    std = np.std(valid)
    asymmetry = np.nan

    if len(valid) >= 3 and std > 1e-12 * max(1, abs(np.mean(valid))):
        asymmetry = skew(valid, bias=False)

    result[2:11] = (median, std, p05, p95, p95 - p05, np.min(valid), np.max(valid), np.max(valid) - np.min(valid), asymmetry)

    return result



# Save figure function
def save_figure(fig, name):

    if SAVE:
        fig.savefig(output_directory / name, dpi=200)
    if not PLOT_CLUSTERS:
        plt.close(fig)




# Save per-cluster statistics
def save_orbital_report(k, labels, representative_ids):

    physical_stds = np.full((len(phase_names), len(parameter_names)), np.nan)

    if not SAVE or not np.any(labels >= 0):
        return physical_stds

    cluster_ids = np.unique(labels[labels >= 0])
    groups = [np.flatnonzero(labels == c) for c in cluster_ids]
    cluster_stds = np.full((len(phase_names), len(parameter_names), len(cluster_ids)), np.nan)

    with (output_directory / "orbital_statistics.csv").open("a", newline="", encoding="utf-8") as table, \
         (output_directory / "analysis.log").open("a", encoding="utf-8") as log, \
         PdfPages(output_directory / f"orbital_parameters_K{k:04d}.pdf") as pdf:

        writer = csv.writer(table)

        for phase_id, phase in enumerate(phase_names):
            for parameter_id, (parameter, unit) in enumerate(zip(parameter_names, parameter_units)):

                statistics = np.array([
                    parameter_statistics(
                        orbital_elements[ids, phase_id, parameter_id],
                        orbital_elements[representative_ids[c], phase_id, parameter_id],
                        circular=(parameter == "w"))             
                    for c, ids in zip(cluster_ids, groups)])

                cluster_stds[phase_id, parameter_id] = statistics[:, 3]
                valid = np.isfinite(statistics[:, 3])
                if np.any(valid):
                    physical_stds[phase_id, parameter_id] = np.sqrt(np.average(statistics[valid, 3]**2, weights=statistics[valid, 0]))

                log.write(f"\nK={k}, phase={phase}, parameter={parameter} [{unit}]\n")
                log.write("cluster representative_source n_cluster " + " ".join(statistic_names) + "\n")

                for c, ids, values in zip(cluster_ids, groups, statistics):
                    source_id = source_ids[representative_ids[c]]
                    writer.writerow([k, c, source_id, phase, parameter, unit, len(ids), *values])
                    log.write(f"{c} {source_id} {len(ids)} " + " ".join(f"{value:.10g}" for value in values) + "\n")

                fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)

                axes[0].vlines(cluster_ids, statistics[:, 7], statistics[:, 8], color="lightgray", label="Min-max")
                axes[0].vlines(cluster_ids, statistics[:, 4], statistics[:, 5], color="blue", label="P5-P95")
                axes[0].plot(cluster_ids, statistics[:, 2], ".", color="black", label="Median")
                axes[0].set_ylabel(f"{parameter} [{unit}]")

                axes[1].plot(cluster_ids, statistics[:, 3], ".", label="STD")
                axes[1].plot(cluster_ids, statistics[:, 11], ".", label="P95 absolute error from representative")
                axes[1].set_ylabel(f"Dispersion [{unit}]")

                axes[2].plot(cluster_ids, statistics[:, 10], ".", color="darkred", label="Skewness")
                axes[2].axhline(0, color="black", linewidth=0.7)
                axes[2].set_ylabel("Skewness [-]")
                axes[2].set_xlabel("Cluster ID")

                for ax in axes:
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                title = (f"K={k} | {phase} | {parameter} | valid={int(np.sum(statistics[:, 0]))}/{np.count_nonzero(labels >= 0)} assigned")

                if parameter == "w":
                    title += " | offsets from each cluster circular mean"

                fig.suptitle(title)
                pdf.savefig(fig)
                plt.close(fig)

    # Save histograms of the within-cluster standard deviations
    fig, axes = plt.subplots(1, len(parameter_names), figsize=(16, 5), constrained_layout=True)

    for j, (ax, parameter, unit) in enumerate(zip(np.atleast_1d(axes), parameter_names, parameter_units)):

        values = cluster_stds[:, j, :]
        finite = values[np.isfinite(values)]

        if len(finite) == 0:
            ax.set_visible(False)
            continue

        bins = np.histogram_bin_edges(finite, bins=12)

        if parameter == "eps" and np.all(finite > 0) and np.ptp(finite) > 0:
            bins = np.geomspace(finite.min() * 0.99, finite.max() * 1.01, 13)
            ax.set_xscale("log")

        for phase_id, phase in enumerate(phase_names):

            stds = values[phase_id]
            stds = stds[np.isfinite(stds)]

            if len(stds) == 0:
                continue

            color = f"C{phase_id}"
            ax.hist(stds, bins=bins, histtype="step", linewidth=2, color=color,
                    label=f"{phase}: {len(stds)} clusters")
            ax.axvline(np.median(stds), color=color, linestyle=":",
                       label=f"Median {phase}: {np.median(stds):.3g}")

        ax.set_xlabel(f"STD {parameter} [{unit}]")
        ax.set_ylabel("Number of clusters")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"K={k} | Within-cluster STD distributions | Each cluster counts once")
    fig.savefig(output_directory / f"STD_histograms_K{k:04d}.png", dpi=200)
    fig.savefig(output_directory / f"STD_histograms_K{k:04d}.pdf")
    plt.close(fig)


    return physical_stds






# Input
DATA_FILE = "/home/lucacecca/Astrodynamics/CR3BP/Ballistic Captures/Clustering/Database_Lorenzo/strsysT8Gamma20V16.dat"

database, data = build_x0_database(DATA_FILE, mu=mu, collisions="exclude")


x0_selection = database[::1]
source_ids = data["column_indices"][::1]

with np.load(Path(__file__).resolve().parent / "Clusters/strsysT8Gamma20V16_20260922_004745/orbital_parameters.npz") as previous:
    excluded_ids = previous["source_ids"][~np.isfinite(previous["perigee_times_tau"]).all(axis=1)]

keep = ~np.isin(source_ids, excluded_ids)
x0_selection, source_ids = x0_selection[keep], source_ids[keep]


print(f"\nDatabase trajectories: {len(database)}")
print(f"Selected trajectories: {len(x0_selection)}")


# Initialize the packages
start_time = time.perf_counter()
Integrator = Integration()




# Define orbital report fields and output directory
phase_names = ("backward", "forward")

parameter_names = ("eps", "e", "w")

parameter_units = ("km^2/s^2", "-", "deg")

statistic_names = ("n_valid", "n_missing", "median", "std", "p05", "p95", "width90", "minimum", "maximum", "range", "skewness", "error95", "error_max", "angle_origin_deg", "angle_resultant")

output_directory = (Path(__file__).resolve().parent / "Clusters" / f"{Path(DATA_FILE).stem}_{time.strftime('%Y%m%d_%H%M%S')}")

if SAVE:
    output_directory.mkdir(parents=True, exist_ok=True)

    with (output_directory / "analysis.log").open("w", encoding="utf-8") as log:
        log.write(f"N={len(x0_selection)}, N_branch={N_branch}, dt_b={dt_b}, dt_f={dt_f}, rtol=1e-9\n")
        log.write(f"T_min={T_min}, T_max_b={T_max_b}, T_max_f={T_max_f}\n")
        log.write("Cluster views: full trajectories and lunar zoom (+/-0.25 LU).\n")
        log.write("Features: unchanged normalized position and unit tangent blocks.\n")
        log.write("OE event: first outward crossing of 2.5 lunar SOI along each integration direction, before any detected collision.\n")
        log.write("Frame: geocentric inertial axes coincident with synodic axes at t=0.\n")
        log.write("STD: ddof=0; w statistics use offsets from the cluster circular mean (angle_origin_deg).\n")
        log.write("angle_resultant: 1=concentrated directions; 0=no defined mean direction. Broad angular clouds require caution with linearized moments.\n")
        log.write("NaN: missing event, undefined parameter/statistic, or missing representative parameter.\n")

    with (output_directory / "orbital_statistics.csv").open("w", newline="", encoding="utf-8") as table:
        csv.writer(table).writerow(["K", "cluster", "representative_source", "phase", "parameter", "unit", "n_cluster", *statistic_names])





# Define events
earth_SOI_exit.terminal = True
earth_SOI_exit.direction = -1

earth_perigee_b.terminal = False
earth_perigee_b.direction = -1

earth_perigee_f.terminal = False
earth_perigee_f.direction = 1

collision.terminal = False
collision.direction = -1

moon_SOI_crossing.terminal = False
moon_SOI_crossing.direction = 1


# Parallel integration backward and forward
X_curvature = np.empty((len(x0_selection), 6, 2 * N_branch - 1), dtype=np.float32)
orbital_elements = np.full((len(x0_selection), len(phase_names), len(parameter_names)), np.nan)
perigee_times = np.full((len(x0_selection), 2), np.nan)

with get_context("fork").Pool() as pool:
    for i, (trajectory, elements, event_times) in enumerate(
        pool.imap(
            integrate_and_sample_one,
            x0_selection,
            chunksize=10,
        )
    ):
        X_curvature[i] = trajectory
        orbital_elements[i] = elements
        perigee_times[i] = event_times

del trajectory, elements, event_times
assert np.allclose(X_curvature[:, :, N_branch - 1], x0_selection)




# Save orbital parameters to file
if SAVE:
    np.savez(
        output_directory / "orbital_parameters.npz",
        source_ids=source_ids,
        parameters=orbital_elements,
        perigee_times_tau=perigee_times,
        phase_names=phase_names,
        parameter_names=parameter_names,
        parameter_units=parameter_units
    )




# Compute clustering features
sampled_position = X_curvature[:, :2]
sampled_velocity = X_curvature[:, 3:5]

sampled_speed = np.linalg.norm(sampled_velocity, axis=1, keepdims=True)
unit_tangent = np.divide(sampled_velocity, sampled_speed, out=np.zeros_like(sampled_velocity), where=sampled_speed > 1e-14)

del sampled_speed

backward_position = sampled_position[:, :, :N_branch]
forward_position = sampled_position[:, :, N_branch - 1:]

backward_tangent = unit_tangent[:, :, :N_branch]
forward_tangent = unit_tangent[:, :, N_branch - 1:]

backward_position = backward_position.transpose(0, 2, 1).reshape(len(X_curvature), -1)
forward_position = forward_position.transpose(0, 2, 1).reshape(len(X_curvature), -1)

backward_tangent = backward_tangent.transpose(0, 2, 1).reshape(len(X_curvature), -1)
forward_tangent = forward_tangent.transpose(0, 2, 1).reshape(len(X_curvature), -1)

del sampled_position, sampled_velocity, unit_tangent

backward_position = normalize_block(backward_position)
forward_position = normalize_block(forward_position)

backward_tangent = normalize_block(backward_tangent)
forward_tangent = normalize_block(forward_tangent)

clustering_features = np.hstack((backward_position, backward_tangent, forward_position, forward_tangent))

del backward_position, forward_position
del backward_tangent, forward_tangent





# GA_16-style candidate splitting and fusion on the unchanged geometric features.
STD_MEAN_LIMIT = 0.1                  # Max distance from mean for EVERY normalized XY/tangent pair
MIN_CLUSTER_SIZE = 100
MAX_CLUSTERS = max(1, len(clustering_features) // 100)
MAX_SIGMA = 2.5                        # Distance / RMS radius in the full geometric space
DIP_ALPHA = 0.01
GAP_FRACTION = 0.20
STRONG_GAP_FRACTION = 0.50
OUTLIER_SIGMA = 3.0
BASE_CLUSTER_COUNTS = (32, 64)
FUSION_SEEDS = (42, 7, 123)
FUSION_TIME_LIMIT = 30.
FUSION_CACHE_FILE = Path(__file__).resolve().parent / "Clusters/BC_2_geometric_fusion_candidates.npz"

validator = GeometricClusterConstraint(
    clustering_features, STD_MEAN_LIMIT, MIN_CLUSTER_SIZE, MAX_SIGMA,
    dip_alpha=DIP_ALPHA, gap_fraction=GAP_FRACTION,
    strong_gap_fraction=STRONG_GAP_FRACTION, outlier_sigma=OUTLIER_SIGMA,
)
fingerprint = fusion_fingerprint(source_ids, clustering_features)
constraints_key = fusion_fingerprint(
    np.r_[STD_MEAN_LIMIT, MIN_CLUSTER_SIZE, MAX_CLUSTERS, MAX_SIGMA,
          DIP_ALPHA, GAP_FRACTION, STRONG_GAP_FRACTION, OUTLIER_SIGMA],
    np.frombuffer(b"bc-geometric-pointwise-xy-max-pca-gap-v2", dtype=np.uint8),
)
candidates, previous_labels = load_fusion_cache(
    FUSION_CACHE_FILE, fingerprint, len(clustering_features), constraints_key=constraints_key,
)
description = (f"Scarto massimo puntuale normalizzato dalla media < {STD_MEAN_LIMIT:g}, minimo {MIN_CLUSTER_SIZE}, "
               f"massimo {MAX_CLUSTERS} cluster, MAX_SIGMA={MAX_SIGMA:g}")
print(description, flush=True)
if SAVE:
    with (output_directory / "analysis.log").open("a", encoding="utf-8") as log:
        log.write(description + "\n")
        log.write(f"Forma geometrica: DIP_ALPHA={DIP_ALPHA}, GAP_FRACTION={GAP_FRACTION}, "
                  f"STRONG_GAP_FRACTION={STRONG_GAP_FRACTION}, OUTLIER_SIGMA={OUTLIER_SIGMA}\n")
        log.write("Limite su ogni coppia XY di posizioni e tangenti normalizzate, in entrambi i rami.\n")
        log.write("Confronto allo stesso indice del campionamento in curvatura, non allo stesso tempo.\n")
        log.write("STD RMS solo diagnostica; sigma e KMeans usano tutte le feature; PCA individua vuoti.\n")

labels, fusion_candidates = refine_geometric_clustering(
    clustering_features, validator, BASE_CLUSTER_COUNTS, FUSION_SEEDS, MAX_CLUSTERS,
    candidates=candidates, previous_labels=previous_labels, time_limit=FUSION_TIME_LIMIT,
)
if SAVE:
    save_fusion_cache(FUSION_CACHE_FILE, fingerprint, fusion_candidates, labels,
                      constraints_key=constraints_key)

# Representatives describe the validated groups; do not reassign points to medoids.
representative_ids, distances = geometric_representatives(clustering_features, labels)
minimum_distances = distances[labels >= 0]
representative_source_ids = source_ids[representative_ids]
unassigned_count = int(np.count_nonzero(labels < 0))
unassigned_percent = 100 * unassigned_count / len(labels)
print(f"Non assegnate: {unassigned_count}/{len(labels)} ({unassigned_percent:.2f}%)", flush=True)

cluster_counts = []
mean_stds = []
small_cluster_counts = []
small_trajectory_fractions = []
orbital_std_history = []

cluster_ids, sizes = np.unique(labels[labels >= 0], return_counts=True)

cluster_stds = np.array([validator.statistics(ids)[0] for ids in cluster_groups(labels)])
cluster_max_deviations = np.array([validator.statistics(ids)[2] for ids in cluster_groups(labels)])
max_pointwise_deviation = cluster_max_deviations.max() if len(cluster_max_deviations) else np.nan
variances = cluster_stds**2

mean_std = np.sqrt(np.average(variances, weights=sizes)) if len(sizes) else np.nan
k = len(cluster_ids)

cluster_counts.append(len(cluster_ids))
mean_stds.append(mean_std)

print(f"K = {len(cluster_ids)}, STD = {mean_std:.6f}", flush=True)
if len(cluster_stds):
    print(f"STD media geometrica per cluster (diagnostica): media={cluster_stds.mean():.6f}, "
          f"massimo={cluster_stds.max():.6f}", flush=True)
    print(f"Scarto massimo puntuale normalizzato: {max_pointwise_deviation:.6f}; "
          f"limite={STD_MEAN_LIMIT:g}", flush=True)



masks = (sizes == 1, sizes < 10, sizes < 100)

small_cluster_counts.append([np.count_nonzero(mask) for mask in masks])
small_trajectory_fractions.append([np.sum(sizes[mask]) / len(labels) for mask in masks])


if SAVE:
    np.savez(
        output_directory / f"labels_K{k:04d}.npz",
        labels=labels,
        representative_ids=representative_ids,
        source_ids=source_ids,
        cluster_stds=cluster_stds,
        cluster_max_deviations=cluster_max_deviations,
        constraint_metric="max_pointwise_xy_distance_from_mean",
        constraint_units="normalized",
        std_mean_limit=STD_MEAN_LIMIT,
        min_cluster_size=MIN_CLUSTER_SIZE,
        max_clusters=MAX_CLUSTERS,
        max_sigma=MAX_SIGMA,
        dip_alpha=DIP_ALPHA,
        gap_fraction=GAP_FRACTION,
        strong_gap_fraction=STRONG_GAP_FRACTION,
        outlier_sigma=OUTLIER_SIGMA,
        n_unassigned=np.count_nonzero(labels < 0)
    )

    summary = np.column_stack((cluster_counts, mean_stds, small_cluster_counts, small_trajectory_fractions,
                               [unassigned_count], [unassigned_percent], [max_pointwise_deviation]))

    np.savetxt(
        output_directory / "clustering_summary.csv",
        summary,
        delimiter=",",
        comments="",
        header="K,mean_std,n_clusters_eq1,n_clusters_lt10,n_clusters_lt100,traj_fraction_eq1,traj_fraction_lt10,traj_fraction_lt100,n_unassigned,unassigned_percent,max_pointwise_deviation"
    )

    with (output_directory / "analysis.log").open("a", encoding="utf-8") as log:
        log.write(f"\nK={len(cluster_ids)}, STD={mean_std:.10g}\n")
        log.write(f"Scarto massimo puntuale normalizzato={max_pointwise_deviation:.10g}, limite={STD_MEAN_LIMIT:g}\n")
        log.write(f"Non assegnate: {unassigned_count}/{len(labels)} ({unassigned_percent:.2f}%)\n")
        log.write("n = trajectories per cluster; conditions overlap.\n")

        for name, mask in zip(("n = 1", "n < 10", "n < 100"), masks):
            n_clusters_small = np.count_nonzero(mask)
            n_trajectories_small = np.sum(sizes[mask])

            log.write(
                f"{name}: clusters={n_clusters_small}/{len(cluster_ids)} "
                f"({100 * n_clusters_small / max(1, len(cluster_ids)):.2f}%); "
                f"trajectories={n_trajectories_small}/{len(labels)} "
                f"({100 * n_trajectories_small / len(labels):.2f}%)\n"
            )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(cluster_counts, mean_stds, "o-", color="blue")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Within-cluster STD")
    ax.set_title("Cluster dispersion vs number of clusters")

    fig.savefig(output_directory / "STD_vs_K.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.scatter(cluster_ids, cluster_stds, s=15)
    ax.set(xlabel="Cluster ID", ylabel="Geometric RMS STD",
           title=f"Within-cluster geometric dispersion (diagnostic) - K={k}")
    save_figure(fig, "geometric_STD_by_cluster.png")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.scatter(cluster_ids, cluster_max_deviations, s=15)
    ax.axhline(STD_MEAN_LIMIT, color="darkred", linestyle="--", label="Pointwise limit")
    ax.set(xlabel="Cluster ID", ylabel="Maximum pointwise deviation (normalized)",
           title=f"Maximum XY-pair distance from cluster mean - K={k}")
    ax.legend()
    save_figure(fig, "pointwise_deviation_by_cluster.png")

orbital_std_history.append(save_orbital_report(int(k), labels, representative_ids))

if SAVE:
    physical_summary = np.column_stack((cluster_counts, np.asarray(orbital_std_history).reshape(len(cluster_counts), -1)))

    physical_header = [f"{phase}_{parameter}" for phase in phase_names for parameter in parameter_names]

    np.savetxt(
        output_directory / "orbital_STD_vs_K.csv",
        physical_summary,
        delimiter=",",
        comments="",
        header="K," + ",".join(physical_header)
    )



# Save representative trajectories to file
if SAVE and len(representative_ids):

    output_file = (output_directory / f"{Path(DATA_FILE).stem}_medoids.txt")

    wanted_ids = set(representative_source_ids)

    with open(DATA_FILE, encoding="utf-8-sig") as src, \
        output_file.open("w", encoding="utf-8") as dst:

        for line in src:
            values = line.split()

            if not values:
                continue

            selected = (values if len(values) == 1
                        
                else [values[i] for i in representative_source_ids])
            
            dst.write(" ".join(selected) + "\n")
    
    print(f"Medoidi salvati in: {output_file}")




print("\n=== GEOMETRIC SPLIT + FUSION ===")
print(f"Representatives: {len(representative_ids)}")
if len(minimum_distances):
    print(f"Final mean distance: {np.mean(minimum_distances):.6f}")
    print(f"Final 95th percentile: {np.percentile(minimum_distances, 95):.6f}")
    print(f"Final 99th percentile: {np.percentile(minimum_distances, 99):.6f}")
    print(f"Final maximum distance: {np.max(minimum_distances):.6f}")





# Plot cluster STD versus number of clusters
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(cluster_counts, mean_stds, "o-", color="blue")

ax.set_xlabel("Number of clusters")
ax.set_ylabel("Within-cluster STD (normalized geometric features)")
ax.set_title("Cluster dispersion vs number of clusters")
save_figure(fig, "STD_vs_K.png")



# Plot small clusters and their population
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)

for j, name in enumerate(("n = 1", "n < 10", "n < 100")):
    counts = np.asarray(small_cluster_counts)[:, j]
    cluster_percent = 100 * counts / np.maximum(1, np.asarray(cluster_counts))
    trajectory_percent = 100 * np.asarray(small_trajectory_fractions)[:, j]

    axes[0].plot(cluster_counts, counts, "o-", label=name)
    axes[1].plot(cluster_counts, cluster_percent, "o-", label=name)
    axes[2].plot(cluster_counts, trajectory_percent, "o-", label=name)

axes[0].set_ylabel("Number of clusters")
axes[1].set_ylabel("Clusters [% of K]")
axes[2].set_ylabel("Trajectories [% of total N]")
axes[2].set_xlabel("Total number of clusters K")

axes[1].set_title("100 * selected clusters / all clusters")
axes[2].set_title(f"100 * trajectories in selected clusters / {len(x0_selection)}")
fig.suptitle("n = trajectories per cluster; conditions overlap")

for ax in axes:
    ax.grid(True)
    ax.legend()

save_figure(fig, "small_clusters_vs_K.png")



# Plot pooled physical STD versus K, separately for each parameter
if SAVE:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    history = np.asarray(orbital_std_history)
    fig.suptitle("Geocentric elements at 2.5 lunar SOI; each phase uses its valid trajectories")
    for j, ax in enumerate(axes.ravel()):
        if j >= len(parameter_names):
            ax.axis("off")
            continue

        for phase_id, phase in enumerate(phase_names):
            values = orbital_elements[labels >= 0, phase_id, j]
            n_valid = np.count_nonzero(np.isfinite(values))
            label = f"{phase}: {n_valid}/{len(values)} valid ({100 * n_valid / max(1, len(values)):.1f}%)"
            ax.plot(cluster_counts, history[:, phase_id, j], "o-", label=label)

        ax.set_xlabel("Number of clusters K")
        ax.set_ylabel(f"STD {parameter_names[j]} [{parameter_units[j]}]")
        ax.grid(True)
        ax.legend()

    save_figure(fig, "orbital_STD_vs_K.png")




# Compute cluster statistics
unique_clusters, cluster_sizes = np.unique(labels[labels >= 0], return_counts=True)
n_clusters = len(unique_clusters)
mean_variance = mean_std**2

del clustering_features, validator, candidates, fusion_candidates, previous_labels


print(f"\nNumber of clusters: {n_clusters}")
print(f"Mean within-cluster standard deviation: {np.sqrt(mean_variance):.6f}\n")




# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")



# Plot clusters and representative trajectories
if PLOT_CLUSTERS or SAVE:

    if unassigned_count:
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        for trajectory_id in np.flatnonzero(labels < 0):
            trajectory = X_curvature[trajectory_id]
            ax.plot(trajectory[0], trajectory[1], color="gray", alpha=0.12, linewidth=0.7)
        ax.scatter(-mu, 0, color="blue", s=30, label="Earth")
        ax.scatter(1 - mu, 0, color="darkred", s=30, label="Moon")
        ax.set(xlabel="X [LU]", ylabel="Y [LU]",
               title=f"Unassigned trajectories: {unassigned_count} ({unassigned_percent:.2f}%)")
        ax.set_aspect("equal")
        ax.legend()
        save_figure(fig, "unassigned_trajectories.png")

    cmap = plt.get_cmap("turbo", max(n_clusters, 1),)

    cluster_colors = {cluster_id: cmap(color_id) for color_id, cluster_id in enumerate(unique_clusters)}

    groups = []

    for cluster_id in unique_clusters:
        groups.append((cluster_id, f"Cluster {cluster_id}"))


    max_panels = 24
    max_columns = 6


    for view in ("full", "zoom"):
        for start in range(0, len(groups), max_panels):

            plot_groups = groups[start:start + max_panels]

            n_panels = len(plot_groups)

            ncols = min(max_columns, n_panels)

            nrows = int(np.ceil(n_panels / ncols))


            fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9), constrained_layout=True, squeeze=False)

            axes = axes.ravel()


            for ax, (group_id, title) in zip(axes, plot_groups):

                group_mask = labels == group_id
                color = cluster_colors[group_id]

                for trajectory_id in np.flatnonzero(group_mask):

                    trajectory = X_curvature[trajectory_id]

                    ax.plot(trajectory[0], trajectory[1], color=color, alpha=0.12, linewidth=0.7)

                ax.scatter(-mu, 0, color="blue", s=30, label="Earth", zorder=3,)
                ax.scatter(1 - mu, 0, color="darkred", s=30, label="Moon", zorder=3,)

                ax.set_title(f"{title} || {np.count_nonzero(group_mask)} trajectories")

                ax.set_xlabel("X [LU]")
                ax.set_ylabel("Y [LU]")
                ax.set_aspect("equal")

                if view == "zoom":
                    ax.set_xlim(-0.25 + (1 - mu), 0.25 + (1 - mu))
                    ax.set_ylim(-0.25, 0.25)

                ax.set_box_aspect(1)


            for ax in axes[n_panels:]:
                ax.axis("off")


            figure_number = (start // max_panels + 1)

            fig.suptitle(f"Cluster groups - {view} view - Figure {figure_number}")
            save_figure(fig, f"clusters_{view}_{figure_number:03d}.png")



# Plot representative trajectories
if (PLOT_MEDOIDS or SAVE) and len(representative_ids):

    fig_representatives, ax_representatives = plt.subplots(figsize=(10, 8), constrained_layout=True)

    for cluster_id, representative_id in enumerate(representative_ids):

        trajectory = X_curvature[representative_id]

        ax_representatives.plot(trajectory[0], trajectory[1], color="navy", alpha=0.045, linewidth=0.35, rasterized=True)

    ax_representatives.scatter(-mu, 0, color="blue", s=50, label="Earth", zorder=3)
    ax_representatives.scatter(1 - mu, 0, color="darkred", s=50, label="Moon", zorder=3)

    ax_representatives.set_xlabel("X [LU]")
    ax_representatives.set_ylabel("Y [LU]")
    ax_representatives.set_title(f"Final representative trajectories — {len(representative_ids)} medoids")
    ax_representatives.set_aspect("equal")
    ax_representatives.set_box_aspect(1)
    ax_representatives.grid(True, alpha=0.25)
    ax_representatives.legend()
    save_figure(fig_representatives, "representatives.png")



# Show or close figures
if PLOT_CLUSTERS:
    plt.show()
else:
    plt.close("all")
