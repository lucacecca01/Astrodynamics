from Celestial_Mechanics import Integration, Transformations
from pathlib import Path
from multiprocessing import get_context
from itertools import combinations
from contextlib import nullcontext
import hashlib
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.special import ndtri
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from diptest import diptest



SAVE = True
PLOT_CLUSTERS = False
PLOT_MEDOIDS = False



if not PLOT_CLUSTERS:
    plt.ioff()



# Save figure function
def save_figure(fig, name):

    if SAVE:
        fig.savefig(output_directory / name, dpi=200)
    if not PLOT_CLUSTERS:
        plt.close(fig)




def plot_dispersion_rankings(cluster_ids, sizes, values, metric_names, title, stem):
    """Rank all final clusters per metric, with bounded-size pages."""
    if not (SAVE or PLOT_CLUSTERS) or len(cluster_ids) == 0:
        return
    cluster_ids = np.asarray(cluster_ids)
    sizes = np.asarray(sizes)
    values = np.asarray(values, dtype=float)
    colors = plt.get_cmap("turbo", len(cluster_ids))(np.arange(len(cluster_ids)))
    # Undefined dispersions go last, visibly marked rather than treated as zero.
    orders = np.argsort(-np.where(np.isfinite(values), values, -np.inf), axis=0, kind="stable")
    rows_per_page = 40
    page_count = (len(cluster_ids) + rows_per_page - 1) // rows_per_page
    document = PdfPages(output_directory / f"{stem}.pdf") if SAVE else nullcontext(None)
    with document as pdf:
        for page, start in enumerate(range(0, len(cluster_ids), rows_per_page), start=1):
            stop = min(start + rows_per_page, len(cluster_ids))
            positions = np.arange(stop - start)
            fig, axes = plt.subplots(
                1, len(metric_names), figsize=(6 * len(metric_names), max(4, .24 * len(positions) + 1.8)),
                squeeze=False, constrained_layout=True,
            )
            for j, ax in enumerate(axes[0]):
                order = orders[start:stop, j]
                ranked = values[order, j]
                finite = np.isfinite(ranked)
                bars = ax.barh(positions[finite], ranked[finite], color=colors[order[finite]], height=.7)
                ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=7)
                ax.set_yticks(positions, [f"{cluster_ids[i]} | N={sizes[i]}" for i in order], fontsize=8)
                for position in positions[~finite]:
                    ax.text(.01, position, "N/A", transform=ax.get_yaxis_transform(), va="center",
                            fontsize=8, color="dimgray")
                maximum = np.max(values[np.isfinite(values[:, j]), j], initial=0.)
                ax.set_xlim(0, maximum * 1.2 if maximum > 0 else 1.)
                ax.set_ylim(len(positions) - .5, -.5)
                ax.set_title(metric_names[j], fontsize=11)
                ax.set_xlabel(metric_names[j])
                ax.set_ylabel("Cluster ID | Number of trajectories")
                ax.set_axisbelow(True)
                ax.grid(axis="x", alpha=.25)
            fig.suptitle(f"{title} | K={len(cluster_ids)} | "
                         f"Ranks {start + 1}-{stop} | Page {page}/{page_count}", fontsize=13)
            if pdf is not None:
                pdf.savefig(fig)
            save_figure(fig, f"{stem}_{page:03d}.png")



#Define stop condition for exiting Earth's sphere of influence
def earth_SOI_exit(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)

    return E_SOI / d - r_earth



# Define collision events
def collision(t, x, mu):

    r_earth = np.linalg.norm(x[:3] - np.array([-mu, 0, 0]))
    r_moon = np.linalg.norm(x[:3] - np.array([1 - mu, 0, 0]))

    return min(r_earth - R_E / d, r_moon - R_L / d)



# Define perigee events
def earth_perigee(t, x, mu):

    return (x[0] + mu)*x[3] + x[1]*x[4] + x[2]*x[5]



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

    collision_time = (abs(sol.t_events[1][0] - T_min) if len(sol.t_events[1]) else np.inf)

    if collision(T_min, sol.y[:, 0], mu) <= 0:
        return np.full(3, np.nan), np.nan

    for t, x in zip(sol.t_events[0], sol.y_events[0]):

        if abs(t - T_min) >= collision_time:
            break

        r_earth = np.linalg.norm(x[:3] - np.array([-mu, 0, 0]))
        r_moon = np.linalg.norm(x[:3] - np.array([1 - mu, 0, 0]))

        if abs(t - T_min) > 1e-10 and r_moon > 2 * M_SOI / d:
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

    solution_b = integrate_one(x0, T_min, T_max_b, dt_b, (earth_perigee, collision))

    parameters_b, t_perigee = first_perigee_parameters(solution_b)

    if not np.isfinite(t_perigee):
        return None

    if not np.all(np.isfinite(parameters_b)):
        raise RuntimeError("Parametri al perigeo non finiti.")

    x_perigee = solution_b.sol(t_perigee)
    rx = x_perigee[0] + mu
    ry = x_perigee[1]

    hz = rx * (x_perigee[4] + rx) - ry * (x_perigee[3] - ry)
    direction = np.sign(hz)

    x_b = solution_b.sol(np.linspace(t_perigee, T_min, N_PLOT))[:2].copy()

    del solution_b


    solution_f = integrate_one(x0, T_min, T_max_f, dt_f, (earth_SOI_exit, collision))

    if not solution_f.success or len(solution_f.t_events[0]) == 0:
        raise RuntimeError("Integrazione forward fallita o SOI non raggiunta.")

    t_soi = solution_f.t_events[0][0]
    x_soi = solution_f.y_events[0][0].copy()

    if len(solution_f.t_events[1]) and solution_f.t_events[1][0] <= t_soi:
        raise RuntimeError("Collisione precedente all'uscita dalla SOI.")

    x_f = solution_f.sol(np.linspace(T_min, t_soi, N_PLOT)[1:])[:2].copy()

    x_soi[0] += mu
    x_inertial = Transformations.CR3BP_to_inertial(x_soi, 1, t_soi)[:, 0]
    velocity_f = x_inertial[3:] * d / TU
        
    del solution_f, 


    features = np.concatenate((parameters_b, velocity_f, [direction]))

    return features, np.hstack((x_b, x_f)), (t_perigee, t_soi)


def geocentric_inertial_trajectory(trajectory, time_bounds):

    # Reconstruct the original sampling times, with T_min stored only once.
    times = np.concatenate((
        np.linspace(time_bounds[0], T_min, N_PLOT),
        np.linspace(T_min, time_bounds[1], N_PLOT)[1:],
    ))
    state = np.zeros((6, trajectory.shape[1]))
    state[:2] = trajectory
    state[0] += mu
    return Transformations.CR3BP_to_inertial(state, 1, times)[:2]


def plot_frame_bodies(ax, inertial=False, size=30):

    if inertial:
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "--", color="gray", alpha=0.4, linewidth=0.7)
        earth, moon = (0, 0), (np.cos(T_min), np.sin(T_min))
        moon_label = f"Moon (t = {T_min:g} TU)"
    else:
        earth, moon = (-mu, 0), (1-mu, 0)
        moon_label = "Moon"
    ax.scatter(*earth, color="blue", s=size, label="Earth", zorder=3)
    ax.scatter(*moon, color="darkred", s=size, label=moon_label, zorder=3)



# Standardized Q-Q error for eps, e and circularly centered omega (degrees).
def gaussian_shape_penalty(parameters):

    data = np.array(parameters, dtype=float, copy=True)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("Servono almeno due righe finite [eps, e, omega_deg].")

    angular_mean = np.mean(np.exp(1j * np.deg2rad(data[:, 2])))
    origin = np.rad2deg(np.angle(angular_mean))
    data[:, 2] = (data[:, 2] - origin + 180) % 360 - 180

    std = np.std(data, axis=0)
    variable = (np.ptp(data, axis=0) > 0) & (std > 0)
    standardized = np.divide(data - data.mean(axis=0), std,
                             out=np.zeros_like(data), where=variable)
    normal = ndtri((np.arange(len(data)) + 0.5) / len(data))
    penalty = np.mean((np.sort(standardized, axis=0) - normal[:, None])**2, axis=0)

    # A constant variable or an undefined mean angle is not a Gaussian fit.
    penalty[~variable] = 1.0
    if abs(angular_mean) < 1e-8:
        penalty[2] = 1.0
    return penalty


class ClusterShapeConstraint:
    """Reject separated modes and isolated tails, optionally bounding Gaussian Q-Q error."""

    def __init__(self, parameters, scale, *, dip_alpha=0.01, gap_fraction=0.2,
                 gaussian_qq_max=np.inf, strong_gap_fraction=0.5, outlier_sigma=6.):
        self.data = np.asarray(parameters, dtype=float)
        scale = np.asarray(scale, dtype=float)
        self.qq_max = np.broadcast_to(np.asarray(gaussian_qq_max, dtype=float), (3,)).copy()
        if (self.data.ndim != 2 or self.data.shape[1] != 3 or not np.all(np.isfinite(self.data))
                or scale.shape != (3,) or np.any(np.isnan(scale)) or np.any(scale <= 0)):
            raise ValueError("Servono feature finite [eps, e, omega_deg] e tre scale positive.")
        if not np.isfinite(dip_alpha) or not 0 <= dip_alpha < 1:
            raise ValueError("DIP_ALPHA deve essere in [0, 1); 0 disabilita il dip test.")
        if not np.isfinite(gap_fraction) or not 0 <= gap_fraction < 1:
            raise ValueError("GAP_FRACTION deve essere in [0, 1).")
        if np.isnan(strong_gap_fraction) or strong_gap_fraction < gap_fraction or strong_gap_fraction <= 0:
            raise ValueError("STRONG_GAP_FRACTION deve essere positivo e >= GAP_FRACTION; np.inf lo disabilita.")
        if np.any(np.isnan(self.qq_max)) or np.any(self.qq_max <= 0):
            raise ValueError("GAUSSIAN_QQ_MAX deve essere positivo; np.inf disabilita il limite.")
        if np.isnan(outlier_sigma) or outlier_sigma <= 0:
            raise ValueError("OUTLIER_SIGMA deve essere positivo; np.inf disabilita il controllo.")
        self.scale = np.where(np.isfinite(scale), scale, 1.)
        self.alpha = dip_alpha
        self.gap_fraction = gap_fraction
        self.strong_gap_fraction = strong_gap_fraction
        self.outlier_sigma = outlier_sigma
        self.cache = {}

    def _measure(self, ids):
        ids = np.sort(np.asarray(ids, dtype=np.int64))
        key = hashlib.sha256(ids.tobytes()).digest()
        if key in self.cache:
            return self.cache[key]
        if len(ids) < 2:
            result = (np.full(3, np.inf), None)
            self.cache[key] = result
            return result

        data = self.data[ids].copy()
        qq = gaussian_shape_penalty(data) if np.any(np.isfinite(self.qq_max)) else np.zeros(3)
        origin = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(data[:, 2])))))
        data[:, 2] = (data[:, 2] - origin + 180) % 360 - 180
        scaled = data / self.scale
        center = scaled.mean(axis=0)
        x = scaled - center
        best_gap, cut = self.gap_fraction, None
        if (self.alpha > 0 or np.isfinite(self.strong_gap_fraction)
                or np.isfinite(self.outlier_sigma)) and len(ids) >= 3:
            _, _, axes = np.linalg.svd(x, full_matrices=False)
            directions = np.vstack((np.eye(3), axes))
            # The dip test can miss a small detached population, even across a huge gap.
            support = max(3, int(np.ceil(0.1 * len(ids))))
            for j, direction in enumerate(directions):
                values = np.sort(x @ direction)
                span = np.ptp(values)
                if span <= 1e-12:
                    continue
                gaps = np.diff(values)
                choices = []
                if len(ids) >= 6:
                    strong_at = 2 + int(np.argmax(gaps[2:len(ids) - 3]))
                    if gaps[strong_at] / span > self.strong_gap_fraction:
                        choices.append(strong_at)
                if self.alpha > 0 and len(ids) >= 6:
                    _, pvalue = diptest(values, sort_x=False, boot_pval=False)
                    eligible = gaps[support - 1:len(ids) - support]
                    if pvalue < self.alpha / len(directions) and len(eligible):
                        choices.append(support - 1 + int(np.argmax(eligible)))
                # Check physical marginals: PCA tails can be artifacts of curved families.
                # A detached tail may contain only one point; final min_size is checked later.
                if j < 3 and np.isfinite(self.outlier_sigma):
                    median = np.median(values)
                    robust_std = 1.4826 * np.median(np.abs(values - median))
                    bound = self.outlier_sigma * max(robust_std, 1e-12)
                    isolated = ((values[:-1] < median - bound)
                                | (values[1:] > median + bound))
                    eligible = np.flatnonzero(isolated & (gaps / span > self.gap_fraction))
                    if len(eligible):
                        choices.append(int(eligible[np.argmax(gaps[eligible])]))
                for at in choices:
                    relative_gap = gaps[at] / span
                    if relative_gap > best_gap:
                        best_gap = relative_gap
                        threshold = (values[at] + values[at + 1]) / 2 + center @ direction
                        cut = (direction.copy(), float(threshold), float(origin))
        result = (qq, cut)
        self.cache[key] = result
        return result

    def __call__(self, ids):
        qq, cut = self._measure(ids)
        return bool(np.all(qq <= self.qq_max) and cut is None)

    def split(self, ids):
        """Return a binary cut at the detected gap, or None for the usual KMeans split."""
        _, cut = self._measure(ids)
        if cut is None:
            return None
        direction, threshold, origin = cut
        data = self.data[ids].copy()
        data[:, 2] = (data[:, 2] - origin + 180) % 360 - 180
        return ((data / self.scale) @ direction > threshold).astype(int)



# Physical STD with omega centered on its circular mean.
def physical_std(parameters):

    data = np.array(parameters, dtype=float, copy=True)
    origin = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(data[:, 2])))))
    data[:, 2] = (data[:, 2] - origin + 180) % 360 - 180
    return data.std(axis=0)



# Define normalization function for features
def normalize_block(features):

    centered = features - np.mean(features, axis=0)
    scale = np.linalg.norm(centered) / np.sqrt(len(centered))

    if scale <= 1e-14:
        return np.zeros_like(centered)

    return centered / scale



# Use the same normalized radial sigma constraint in both clustering stages.
def within_sigma(features, sigma_limit):

    if sigma_limit == np.inf:
        return True
    distance_squared = np.sum((features - features.mean(axis=0))**2, axis=1)
    return np.all(distance_squared <= sigma_limit**2 * np.var(features, axis=0).sum())


def make_cluster_validator(physical_features, features, std_limits, min_size, sigma_limit,
                           *, directions=None, extra_condition=None, shape_condition=None):
    """Return a deterministic predicate on row indices; extra_condition adds local constraints."""
    data = np.asarray(physical_features, dtype=float)
    features = np.asarray(features, dtype=float)
    limits = np.asarray(std_limits, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3 or not np.all(np.isfinite(data)):
        raise ValueError("Servono feature fisiche finite con forma (N, 3).")
    if (features.ndim != 2 or len(features) != len(data) or features.shape[1] == 0
            or not np.all(np.isfinite(features))):
        raise ValueError("Servono feature normalizzate finite con N righe.")
    if limits.shape != (3,) or np.any(np.isnan(limits)) or np.any(limits <= 0):
        raise ValueError("I limiti STD devono essere positivi; np.inf disabilita un limite.")
    if not isinstance(min_size, (int, np.integer)) or isinstance(min_size, (bool, np.bool_)) or min_size < 1:
        raise ValueError("min_size deve essere un intero positivo.")
    if np.isnan(sigma_limit) or sigma_limit <= 0:
        raise ValueError("MAX_SIGMA deve essere positivo; np.inf disabilita il limite.")
    if directions is not None:
        directions = np.asarray(directions)
        if directions.shape != (len(data),) or not np.all(np.isin(directions, [-1, 1])):
            raise ValueError("Verso orbitale non definito per ogni traiettoria.")
    if extra_condition is not None and not callable(extra_condition):
        raise TypeError("extra_condition deve essere una funzione degli indici del cluster.")
    if shape_condition is not None and not callable(shape_condition):
        raise TypeError("shape_condition deve essere una funzione degli indici del cluster.")

    def valid_group(ids):
        return (len(ids) >= min_size
                and (directions is None or np.all(directions[ids] == directions[ids[0]]))
                and np.all(physical_std(data[ids]) < limits)
                and (np.isinf(sigma_limit) or within_sigma(features[ids], sigma_limit))
                and (extra_condition is None or bool(extra_condition(ids)))
                and (shape_condition is None or bool(shape_condition(ids))))

    return valid_group


def cluster_groups(labels):
    ids = np.flatnonzero(labels >= 0)
    ids = ids[np.argsort(labels[ids], kind="stable")]
    return np.split(ids, np.flatnonzero(np.diff(labels[ids])) + 1) if len(ids) else []


def kmeans_partition(features, directions, n_base, random_state=42):
    if not isinstance(n_base, (int, np.integer)) or n_base < 1:
        raise ValueError("Il numero di cluster base deve essere un intero positivo.")
    labels = np.full(len(features), -1, dtype=int)
    offset = 0
    with threadpool_limits(limits=2):
        for sign in np.unique(directions):
            ids = np.flatnonzero(directions == sign)
            k = min(len(ids), max(1, round(n_base * len(ids) / len(features))))
            k = min(k, len(np.unique(features[ids], axis=0)))
            model = KMeans(n_clusters=k, n_init=5, random_state=random_state,
                           max_iter=500, tol=1e-9, algorithm="lloyd")
            labels[ids] = model.fit_predict(features[ids]) + offset
            offset += k
    return labels


class NoFeasiblePartitionError(RuntimeError):
    """The initial full-coverage clustering is unavailable for these constraints."""


# Select valid unions of standard KMeans groups, minimizing normalized SSE.
def constrained_clustering(features, directions, n_base, min_size=1000,
                           sigma_limit=2.0, previous_labels=None):

    if not np.all(np.isfinite(features)) or not np.all(np.isin(directions, [-1, 1])):
        raise ValueError("Feature non finite o verso del perigeo non definito.")

    # Within a single direction, its feature is constant: fit only the OE.
    base_labels = np.empty(len(features), dtype=int)
    offset = 0
    with threadpool_limits(limits=2):
        for sign in np.unique(directions):
            ids = np.flatnonzero(directions == sign)
            if len(ids) < min_size:
                raise NoFeasiblePartitionError(f"Verso {sign:+g}: meno di {min_size} traiettorie.")
            k = min(len(ids), max(1, round(n_base * len(ids) / len(features))))
            model = KMeans(n_clusters=k, n_init=5, random_state=42,
                           max_iter=500, tol=1e-9, algorithm="lloyd")
            base_labels[ids] = model.fit_predict(features[ids, :4]) + offset
            offset += k

    # Intersections let us refine the preceding solution without losing it.
    if previous_labels is not None:
        _, base_labels = np.unique(
            np.column_stack((base_labels, previous_labels)), axis=0, return_inverse=True,
        )

    blocks = [np.flatnonzero(base_labels == c) for c in np.unique(base_labels)]
    counts = np.array([len(ids) for ids in blocks])
    means = np.array([features[ids, :4].mean(axis=0) for ids in blocks])
    residuals = [np.sum((features[ids, :4] - center)**2, axis=1)
                 for ids, center in zip(blocks, means)]
    sse = np.array([values.sum() for values in residuals])
    radii = np.array([np.sqrt(values.max()) for values in residuals])
    block_signs = np.array([directions[ids[0]] for ids in blocks])

    def valid_group(ids):
        return (len(ids) >= min_size
                and np.all(directions[ids] == directions[ids[0]])
                and within_sigma(features[ids], sigma_limit))

    # Local unions: at most 5 blocks from each anchor's 12 nearest blocks.
    specs = set()
    for sign in np.unique(directions):
        ids = np.flatnonzero(block_signs == sign)
        for anchor in ids:
            distance_squared = np.sum((means[ids] - means[anchor])**2, axis=1)
            neighbors = ids[np.argsort(distance_squared)[:12]]
            others = [int(j) for j in neighbors if j != anchor]
            for size in range(1, min(5, len(neighbors)) + 1):
                for rest in combinations(others, size - 1):
                    specs.add(tuple(sorted((int(anchor),) + rest)))

    print(f"Base KMeans: {offset}; blocchi: {len(blocks)}; unioni da verificare: {len(specs)}", flush=True)
    feasible = []
    for spec in sorted(specs, key=lambda item: (len(item), item)):
        ids = np.asarray(spec)
        n = counts[ids].sum()
        if n < min_size:
            continue
        center = np.average(means[ids], axis=0, weights=counts[ids])
        shifts = np.linalg.norm(means[ids] - center, axis=1)
        cost = np.sum(sse[ids] + counts[ids] * shifts**2)
        # Cheap lower bound; every surviving union is checked point by point.
        if np.any(np.maximum(radii[ids] - shifts, 0)**2 > sigma_limit**2 * cost / n * (1 + 1e-10)):
            continue
        members = np.concatenate([blocks[j] for j in spec])
        if valid_group(members):
            feasible.append((spec, float(cost)))

    previous_cost = 0.0
    if previous_labels is not None:
        existing = {spec for spec, _ in feasible}
        for c in np.unique(previous_labels):
            members = np.flatnonzero(previous_labels == c)
            if not valid_group(members):
                raise RuntimeError("La partizione precedente non rispetta i vincoli.")
            spec = tuple(j for j, ids in enumerate(blocks) if previous_labels[ids[0]] == c)
            data = features[members, :4]
            cost = float(np.sum((data - data.mean(axis=0))**2))
            previous_cost += cost
            if spec not in existing:
                feasible.append((spec, cost))

    covered = {j for spec, _ in feasible for j in spec}
    if len(covered) != len(blocks):
        raise NoFeasiblePartitionError("Unioni ammissibili insufficienti: provare un'altra base KMeans.")

    rows, cols = [], []
    for j, (spec, _) in enumerate(feasible):
        rows.extend(spec)
        cols.extend([j] * len(spec))
    incidence = coo_matrix(
        (np.ones(len(rows)), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(blocks), len(feasible)),
    ).tocsc()
    print(f"Unioni ammissibili: {len(feasible)}; ottimizzazione...", flush=True)
    result = milp(
        c=np.array([cost for _, cost in feasible]),
        integrality=np.ones(len(feasible), dtype=np.int32), bounds=Bounds(0, 1),
        constraints=LinearConstraint(incidence, 1, 1),
        options={"time_limit": 60, "mip_rel_gap": 1e-6},
    )
    print(f"MILP: {result.message}; gap={getattr(result, 'mip_gap', None)}", flush=True)
    if result.x is None:
        if previous_labels is not None:
            print("Mantengo la precedente partizione ammissibile.", flush=True)
            return previous_labels.copy()
        raise NoFeasiblePartitionError("Nessuna partizione ammissibile trovata entro il limite MILP.")

    chosen = np.flatnonzero(result.x > 0.5)
    if not np.all(np.asarray(incidence[:, chosen].sum(axis=1)).ravel() == 1):
        raise RuntimeError("La soluzione MILP non copre ogni blocco esattamente una volta.")

    labels = np.full(len(features), -1, dtype=int)
    total_cost = 0.0
    for c, j in enumerate(chosen):
        members = np.concatenate([blocks[b] for b in feasible[j][0]])
        if not valid_group(members):
            raise RuntimeError("Cluster finale non ammissibile.")
        labels[members] = c
        data = features[members, :4]
        total_cost += np.sum((data - data.mean(axis=0))**2)

    if previous_labels is not None and total_cost >= previous_cost:
        print("Mantengo la precedente partizione: SSE non migliorata.", flush=True)
        return previous_labels.copy()
    return labels


# Split invalid parents, preferring detected gaps; never move points between existing clusters.
def split_messy_clusters(labels, physical_features, std_limits, min_size, max_clusters,
                         *, features, sigma_limit, group_validator=None, random_state=42,
                         split_scale=None, candidate_groups=None, verbose=True, gap_splitter=None):

    labels = np.asarray(labels)
    data = np.asarray(physical_features, dtype=float)
    limits = np.asarray(std_limits, dtype=float)
    features = np.asarray(features, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3 or not np.all(np.isfinite(data)):
        raise ValueError("Servono feature fisiche finite con forma (N, 3).")
    if labels.shape != (len(data),) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < -1):
        raise ValueError("Le etichette devono essere intere, con -1 per le non assegnate.")
    if (features.ndim != 2 or features.shape[0] != len(data) or features.shape[1] == 0
            or not np.all(np.isfinite(features))):
        raise ValueError("Servono le feature normalizzate della prima fase, con N righe finite.")
    if limits.shape != (3,) or np.any(np.isnan(limits)) or np.any(limits <= 0):
        raise ValueError("I tre limiti STD devono essere positivi.")
    if not isinstance(min_size, (int, np.integer)) or min_size < 1:
        raise ValueError("min_size deve essere un intero positivo.")
    if not isinstance(max_clusters, (int, np.integer)) or max_clusters < 0:
        raise ValueError("max_clusters deve essere un intero non negativo.")
    if np.isnan(sigma_limit) or sigma_limit <= 0:
        raise ValueError("sigma_limit deve essere positivo.")

    valid_group = group_validator or make_cluster_validator(data, features, limits, min_size, sigma_limit)
    if not callable(valid_group):
        raise TypeError("group_validator deve essere una funzione degli indici del cluster.")
    if gap_splitter is not None and not callable(gap_splitter):
        raise TypeError("gap_splitter deve essere una funzione degli indici del cluster.")

    preserved, pending, children = [], [], []
    small_points = 0
    for ids in cluster_groups(labels):
        if candidate_groups is not None:
            candidate_groups.append(ids)
        if len(ids) < min_size:
            small_points += len(ids)
        elif valid_group(ids):
            preserved.append(ids)
        else:
            pending.append(ids)

    if len(preserved) > max_clusters:
        raise ValueError("Il massimo di cluster e' inferiore ai gruppi gia' validi: "
                         "impossibile conservarli tutti.")

    messy_parents = len(pending)
    n_splits = 0
    n_gap_splits = 0
    angles = np.deg2rad(data[:, 2])
    scale = np.where(np.isfinite(limits), limits, 1.) if split_scale is None else np.asarray(split_scale, dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("La scala di suddivisione deve contenere tre valori finiti e positivi.")
    coordinates = np.column_stack((
        data[:, :2] / scale[:2],
        np.cos(angles) / np.deg2rad(scale[2]),
        np.sin(angles) / np.deg2rad(scale[2]),
    ))

    with threadpool_limits(limits=2):
        while pending:
            ids = pending.pop()
            if len(ids) <= min_size or len(ids) < 2:
                small_points += len(ids)
                continue
            split = None if gap_splitter is None else gap_splitter(ids)
            if split is None:
                # These coordinates avoid an artificial cut at omega = 0/360.
                values = coordinates[ids]
                if np.all(values == values[0]):
                    values = features[ids]
                if np.all(values == values[0]):
                    small_points += len(ids)
                    continue
                model = KMeans(n_clusters=2, n_init=5, random_state=random_state,
                               max_iter=500, tol=1e-9, algorithm="lloyd")
                split = model.fit_predict(values)
            else:
                n_gap_splits += 1
            split = np.asarray(split)
            if split.shape != (len(ids),) or not np.array_equal(np.unique(split), [0, 1]):
                raise RuntimeError("Cluster fuori soglia non divisibile in due gruppi distinti.")
            n_splits += 1
            for side in (0, 1):
                members = ids[split == side]
                if candidate_groups is not None:
                    candidate_groups.append(members)
                if len(members) < min_size:
                    small_points += len(members)
                elif valid_group(members):
                    children.append(members)
                else:
                    pending.append(members)

    # If the cap binds, retain the largest children, leaving good parents intact.
    children.sort(key=lambda ids: (-len(ids), int(ids.min())))
    available = max_clusters - len(preserved)
    capped_points = sum(len(ids) for ids in children[available:])
    groups = preserved + children[:available]
    groups.sort(key=lambda ids: (int(labels[ids[0]]), int(ids.min())))
    result = np.full(len(labels), -1, dtype=int)
    final_stds = []
    for c, ids in enumerate(groups):
        std = physical_std(data[ids])
        if len(ids) < min_size or not valid_group(ids):
            raise RuntimeError("Un cluster finale non rispetta i vincoli STD/MAX_SIGMA/dimensione/forma.")
        final_stds.append(std)
        result[ids] = c

    if verbose:
        print(f"Split vincoli: {len(preserved)} cluster conservati, {messy_parents} da dividere, "
              f"{n_splits} divisioni ({n_gap_splits} nei vuoti), {len(groups)} cluster finali.", flush=True)
        print(f"Nuove non assegnate: {small_points} in frammenti non ammissibili, "
              f"{capped_points} per il massimo di cluster.", flush=True)
        if final_stds:
            print(f"STD fisiche [eps, e, omega_deg]: media {np.mean(final_stds, axis=0)}, "
                  f"massimo {np.max(final_stds, axis=0)}", flush=True)
    return result


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


def refine_and_fuse_clusters(partitions, physical_features, features, directions,
                             std_limits, min_size, max_clusters, sigma_limit, *,
                             seeds=(42, 7, 123), base_count=180, extra_condition=None,
                             candidates=(), previous_labels=None, time_limit=30., shape_condition=None):
    valid_group = make_cluster_validator(physical_features, features, std_limits, min_size,
                                         sigma_limit, directions=directions, extra_condition=extra_condition,
                                         shape_condition=shape_condition)
    gap_splitter = None if shape_condition is None else shape_condition.split
    seeds = tuple(seeds)
    if not seeds or any(not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds):
        raise ValueError("Servono seed interi non negativi per i candidati alternativi.")
    if not partitions:
        partitions = [kmeans_partition(features, directions, base_count, seeds[0])]
    pool = list(candidates)
    n_points = len(features)
    baseline = split_messy_clusters(
        partitions[-1], physical_features, std_limits, min_size, n_points,
        features=features, sigma_limit=sigma_limit, group_validator=valid_group,
        candidate_groups=pool, gap_splitter=gap_splitter,
    )
    # Supply an optional fallback, without fixing any point's assignment.
    if previous_labels is None and len(cluster_groups(baseline)) <= max_clusters:
        previous_labels = baseline

    scale = np.where(np.isfinite(std_limits), std_limits, 1.)
    variants = [(partition, seeds[0], scale) for partition in partitions[:-1]]
    variants.extend((partitions[-1], seed, scale) for seed in seeds[1:])
    # Relative metric variations generate alternatives; validity never changes.
    variants.extend((partitions[-1], seeds[0], scale * weights)
                    for weights in ([1.5, .7, 1.], [.75, 1.6, .75]))
    variants.extend((kmeans_partition(features, directions, base_count, seed), seed, scale)
                    for seed in seeds[1:])
    if previous_labels is not None:
        variants.append((previous_labels, seeds[0], scale))
    for i, (partition, seed, split_scale) in enumerate(variants):
        split_messy_clusters(
            partition, physical_features, std_limits, min_size, n_points,
            features=features, sigma_limit=sigma_limit, group_validator=valid_group,
            random_state=seed, split_scale=split_scale, candidate_groups=pool, verbose=False,
            gap_splitter=gap_splitter,
        )
        print(f"Candidati alternativi: {i + 1}/{len(variants)}", flush=True)
    labels, pool = fuse_cluster_candidates(pool, n_points, valid_group, max_clusters,
                                           previous_labels=previous_labels, time_limit=time_limit)
    return labels, baseline, pool


def fusion_fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(array.tobytes())
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


def plot_unassigned_trajectories(trajectories, labels, mu):

    ids = np.flatnonzero(labels < 0)
    percentage = 100 * len(ids) / len(labels) if len(labels) else 0.0
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(ids):
        ax.add_collection(LineCollection(
            [trajectories[i].T for i in ids], colors="dimgray", alpha=0.2, linewidths=0.7,
        ))
    ax.scatter(-mu, 0, color="blue", s=50, label="Earth", zorder=3)
    ax.scatter(1 - mu, 0, color="darkred", s=50, label="Moon", zorder=3)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("X [LU]")
    ax.set_ylabel("Y [LU]")
    ax.set_title(f"Unassigned trajectories: {len(ids)} ({percentage:.2f}%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    save_figure(fig, "unassigned_trajectories.png")



def print_clustering_summary(labels):

    unassigned = np.count_nonzero(labels < 0)
    percentage = 100 * unassigned / len(labels) if len(labels) else 0.0
    assigned_percentage = 100 - percentage if len(labels) else 0.0
    print("\n=== RISULTATO FINALE CLUSTERING ===")
    print(f"Traiettorie con caratteristiche valide: {len(labels)}")
    print(f"Cluster: {len(np.unique(labels[labels >= 0]))}")
    print(f"Traiettorie assegnate: {len(labels) - unassigned} ({assigned_percentage:.2f}%)")
    print(f"Traiettorie non assegnate: {unassigned} ({percentage:.2f}%)", flush=True)




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




# Input
DATA_FILE = "/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP.txt"

database = np.atleast_2d(np.loadtxt(DATA_FILE, skiprows=1))

columns = {
    "Cj": 6,
    "E_SOI": 7,
    "h_min_moon": 8,
    "h_perigee": 9,
}

valid_mask = (database[:, columns["h_min_moon"]] >= 0) & (database[:, columns["h_perigee"]] >= 0)
valid_ids = np.where(valid_mask)[0]

source_ids = valid_ids[::1]
selection = database[source_ids, :]
x0_selection = selection[:, :6]


print(f"\nDatabase trajectories: {len(database)}")
print(f"Valid trajectories:    {len(valid_ids)}")
print(f"Selected trajectories: {len(x0_selection)}")


# Initialize the packages
start_time = time.perf_counter()
Integrator = Integration()



# Define masses and integration parameters
masses = [m1, m2, 0]
T_min = 0

T_max_f = 4 * np.pi
dt_f = 0.001

T_max_b = -4 * np.pi
dt_b = -0.001

event_features = []
N_PLOT = 500



# Define events
earth_SOI_exit.terminal = True
earth_SOI_exit.direction = -1

earth_perigee.terminal = False
earth_perigee.direction = -1

collision.terminal = False
collision.direction = 0




# Parallel integration backward and forward
event_features = np.empty((len(x0_selection), 7), dtype=np.float64)
X_curvature = np.empty((len(x0_selection), 2, 2 * N_PLOT - 1), dtype=np.float64)
trajectory_time_bounds = np.empty((len(x0_selection), 2), dtype=np.float64)
keep = np.zeros(len(x0_selection), dtype=bool)
n_kept = 0

with get_context("fork").Pool() as pool:

    for i, result in enumerate(
        pool.imap(
            integrate_and_sample_one,
            x0_selection,
            chunksize=10,
        )
    ):
        if result is None:
            continue

        event_features[n_kept], X_curvature[n_kept], trajectory_time_bounds[n_kept] = result
        keep[i] = True
        n_kept += 1


if n_kept == 0:
    raise RuntimeError("Nessuna traiettoria con perigeo ammissibile.")

event_features = event_features[:n_kept]
X_curvature = X_curvature[:n_kept]
trajectory_time_bounds = trajectory_time_bounds[:n_kept]
source_ids = source_ids[keep]
selection = database[source_ids, :]
x0_selection = selection[:, :6]

print(f"Traiettorie mantenute: {n_kept}; escluse: {len(keep) - n_kept}")



# Clustering limits are based on all retained trajectories, including unassigned points.
STD_LIMITS = np.array([0.04, 0.05, 12])
MIN_CLUSTER_SIZE = 50
MAX_CLUSTERS = max(1, n_kept // 100)    # floor(1% of the total), at least one
BASE_MIN_CLUSTER_SIZE = 200
MAX_SIGMA = 2.5
DIP_ALPHA = 0.01                        # Family threshold for 3 physical + 3 PCA projections
GAP_FRACTION = 0.20                     # Internal empty interval / projected range
STRONG_GAP_FRACTION = 0.50              # Cut also without dip significance; at least 3 points per side
OUTLIER_SIGMA = 3.0                     # Median/MAD tail separation, also requiring GAP_FRACTION
GAUSSIAN_QQ_MAX = np.inf                # Disabled: Q-Q is diagnostic only
BASE_CLUSTER_COUNTS = (99, 180)
FUSION_SEEDS = (42, 7, 123)
FUSION_TIME_LIMIT = 30.
FUSION_CACHE_FILE = Path(__file__).resolve().parent / "Clusters/GA_16_dip_gap_candidates_sigma2_5.npz"
EXTRA_CLUSTER_CONDITION = None          # Optional callable(ids), using event_features/selection

# Same initial clustering and normalization as GA_15.
w = np.deg2rad(event_features[:, 2])
angular_features = np.column_stack((np.cos(w), np.sin(w)))

clustering_features = np.column_stack((
    normalize_block(event_features[:, 0:1]),                          # eps
    normalize_block(event_features[:, 1:2]),                          # e
    normalize_block(angular_features),                                # w
    100 *normalize_block(event_features[:, 6:7]),                     # verso al perigeo
   # normalize_block(event_features[:, 3:5]),                          # v_SOI
))

print(f"Vincoli: STD < {STD_LIMITS}, minimo {MIN_CLUSTER_SIZE} punti, "
      f"massimo {MAX_CLUSTERS} cluster, MAX_SIGMA = {MAX_SIGMA:g}", flush=True)
shape_condition = ClusterShapeConstraint(event_features[:, :3], STD_LIMITS, dip_alpha=DIP_ALPHA,
                                         gap_fraction=GAP_FRACTION, gaussian_qq_max=GAUSSIAN_QQ_MAX,
                                         strong_gap_fraction=STRONG_GAP_FRACTION,
                                         outlier_sigma=OUTLIER_SIGMA)
print(f"Forma: DIP_ALPHA = {DIP_ALPHA:g}, GAP_FRACTION = {GAP_FRACTION:g}, "
      f"STRONG_GAP_FRACTION = {STRONG_GAP_FRACTION:g}, OUTLIER_SIGMA = {OUTLIER_SIGMA:g}", flush=True)
if np.any(np.isfinite(shape_condition.qq_max)):
    print(f"Gaussian Q-Q <= {shape_condition.qq_max} per feature (approssimazione, non normalita' esatta).",
          flush=True)
else:
    print("Vincolo gaussiano disattivato: Q-Q solo diagnostico.", flush=True)
final_validator = make_cluster_validator(
    event_features[:, :3], clustering_features, STD_LIMITS, MIN_CLUSTER_SIZE, MAX_SIGMA,
    directions=event_features[:, 6], extra_condition=EXTRA_CLUSTER_CONDITION, shape_condition=shape_condition,
)
fingerprint = fusion_fingerprint(source_ids, selection, event_features, clustering_features)
constraints_key = fusion_fingerprint(
    np.r_[STD_LIMITS, MIN_CLUSTER_SIZE, MAX_CLUSTERS, MAX_SIGMA, DIP_ALPHA, GAP_FRACTION,
          STRONG_GAP_FRACTION, OUTLIER_SIGMA, shape_condition.qq_max],
    np.frombuffer(b"dip-gap-qq-v3-outliers", dtype=np.uint8),
)
fusion_candidates, previous_fused_labels = load_fusion_cache(
    FUSION_CACHE_FILE, fingerprint, n_kept, constraints_key=constraints_key,
)
partitions = []
labels = None
for n_base in BASE_CLUSTER_COUNTS:
    try:
        labels = constrained_clustering(
            clustering_features, event_features[:, 6], n_base,
            min_size=BASE_MIN_CLUSTER_SIZE, sigma_limit=MAX_SIGMA, previous_labels=labels,
        )
        partitions.append(labels.copy())
    except NoFeasiblePartitionError as exc:
        print(f"Base vincolata non disponibile ({exc}); genero candidati KMeans.", flush=True)
        partitions.append(kmeans_partition(clustering_features, event_features[:, 6], n_base))
        labels = None

labels, split_labels, fusion_candidates = refine_and_fuse_clusters(
    partitions, event_features[:, :3], clustering_features, event_features[:, 6],
    STD_LIMITS, MIN_CLUSTER_SIZE, MAX_CLUSTERS, MAX_SIGMA,
    seeds=FUSION_SEEDS, base_count=BASE_CLUSTER_COUNTS[-1], extra_condition=EXTRA_CLUSTER_CONDITION,
    candidates=fusion_candidates, previous_labels=previous_fused_labels, time_limit=FUSION_TIME_LIMIT,
    shape_condition=shape_condition,
)
final_groups = cluster_groups(labels)
if any(not final_validator(ids) for ids in final_groups):
    raise RuntimeError("Verifica finale fallita: un cluster viola i vincoli fisici o di forma.")
if final_groups:
    qq_values = [gaussian_shape_penalty(event_features[ids, :3]) for ids in final_groups if len(ids) >= 2]
    qq_maximum = np.max(qq_values, axis=0) if qq_values else np.full(3, np.nan)
    print(f"Gaussian Q-Q massimo finale [eps, e, omega]: {qq_maximum}", flush=True)
partitions.extend((split_labels, labels))
save_fusion_cache(FUSION_CACHE_FILE, fingerprint, fusion_candidates, labels, constraints_key=constraints_key)

cluster_counts = []
mean_stds = []
small_cluster_counts = []
small_trajectory_fractions = []
escape_stats = []

output_directory = (Path(__file__).resolve().parent / f"Clusters/GA_16_plots_K{len(np.unique(labels[labels >= 0]))}")
output_directory.mkdir(parents=True, exist_ok=True)

if SAVE or PLOT_CLUSTERS:
    plot_unassigned_trajectories(X_curvature, labels, mu)

if not np.any(labels >= 0):
    print_clustering_summary(labels)
    if PLOT_CLUSTERS:
        plt.show()
    raise SystemExit(0)

partitions = [partition for partition in partitions if np.any(partition >= 0)]
physical_std_history = []
gaussian_shape_history = []
outlier_cluster_percent = []
outlier_cluster_percent_2sigma = []


# Keep constrained assignments when selecting representatives.
for stage, labels in enumerate(partitions):

    assigned = labels >= 0
    cluster_ids, sizes = np.unique(labels[assigned], return_counts=True)
    representative_ids = np.empty(len(cluster_ids), dtype=int)

    for cluster_id in cluster_ids:

        members = np.flatnonzero(labels == cluster_id)
        cluster_center = clustering_features[members].mean(axis=0)

        distances_squared = np.sum((clustering_features[members] - cluster_center)**2, axis=1)

        representative_ids[cluster_id] = members[np.argmin(distances_squared)]


    minimum_distances = np.full(len(labels), np.nan)
    minimum_distances[assigned] = np.linalg.norm(
        clustering_features[assigned] - clustering_features[representative_ids[labels[assigned]]], axis=1,
    )

    representative_source_ids = source_ids[representative_ids]

    variances = [np.mean(np.var(clustering_features[labels == c], axis=0)) for c in cluster_ids]

    mean_std = np.sqrt(np.average(variances, weights=sizes))

    variances = []
    physical_stds = []
    shape_penalties = []
    outlier_clusters = 0
    outlier_clusters_2sigma = 0
    outlier_clusters_max_sigma = 0

    for c in cluster_ids:
        mask = labels == c
        features_c = clustering_features[mask]

        variance_c = np.var(features_c, axis=0)
        variances.append(np.mean(variance_c))

        delta = features_c - np.mean(features_c, axis=0)
        distance_squared = np.sum(delta**2, axis=1)
        sigma_total_squared = np.sum(variance_c)

        outlier_clusters += int(np.any(distance_squared > 9 * sigma_total_squared))
        outlier_clusters_2sigma += int(np.any(distance_squared > 4 * sigma_total_squared))
        outlier_clusters_max_sigma += int(not within_sigma(features_c, MAX_SIGMA))

        physical_c = event_features[mask, :5]
        shape_penalties.append(gaussian_shape_penalty(physical_c[:, :3]) if len(physical_c) >= 2 else np.ones(3))

        angle_mean = np.mean(np.exp(1j * np.deg2rad(physical_c[:, 2])))

        if abs(angle_mean) < 1e-8:
            physical_c[:, 2] = np.nan
        else:
            origin = np.rad2deg(np.angle(angle_mean))
            physical_c[:, 2] = (physical_c[:, 2] - origin + 180) % 360 - 180

        physical_stds.append(np.std(physical_c, axis=0))

        if stage == len(partitions) - 1:
            v_c = event_features[mask, 3:6]
            speed_c = np.linalg.norm(v_c, axis=1)
            valid = speed_c > 0

            direction_spread = np.nan
            if np.any(valid):
                unit_v = v_c[valid] / speed_c[valid, None]
                direction_spread = np.clip(1.0 - np.linalg.norm(np.mean(unit_v, axis=0)), 0, 1)

            escape_stats.append([np.std(speed_c), direction_spread])


    physical_stds = np.asarray(physical_stds)
    gaussian_shape_history.append(np.average(shape_penalties, axis=0, weights=sizes))
    valid_counts = np.sum(np.isfinite(physical_stds), axis=0)

    physical_std_history.append(np.divide(
        np.nansum(physical_stds, axis=0),
        valid_counts,
        out=np.full(5, np.nan),
        where=valid_counts > 0,
    ))

    outlier_cluster_percent.append(100 * outlier_clusters / len(cluster_ids))
    outlier_cluster_percent_2sigma.append(100 * outlier_clusters_2sigma / len(cluster_ids))

    cluster_counts.append(len(cluster_ids))
    mean_stds.append(mean_std)

    print(f"K = {len(cluster_ids)}, STD = {mean_std:.6f}, min size = {sizes.min()}, "
          f"clusters > 2sigma (radiale) = {outlier_clusters_2sigma}, "
          f"clusters > MAX_SIGMA ({MAX_SIGMA:g}) = {outlier_clusters_max_sigma}, "
          f"Gaussian Q-Q [eps, e, w] = {gaussian_shape_history[-1]}", flush=True)


    masks = (sizes == 1, sizes < 10, sizes < 100)

    small_cluster_counts.append([np.count_nonzero(mask) for mask in masks])
    small_trajectory_fractions.append([np.sum(sizes[mask]) / len(labels) for mask in masks])



# Plot mean within-cluster physical STD vs number of clusters
if SAVE or PLOT_CLUSTERS:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for j, name in enumerate(("eps", "e", "omega")):
        ax.plot(np.arange(1, len(partitions) + 1), np.asarray(gaussian_shape_history)[:, j], "o-", label=name)
    ax.set_xlabel("Refinement stage")
    ax.set_ylabel("Weighted mean Gaussian Q-Q error (lower is better)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "gaussian_shape_vs_stage.png")

    history = np.asarray(physical_std_history)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()

    names = ["eps [km^2/s^2]", "e [-]", "w [deg]"]

    for j, name in enumerate(names):
        axes[j].plot(cluster_counts, history[:, j], "o-")
        axes[j].set_xlabel("Number of clusters")
        axes[j].set_ylabel(f"Mean STD — {name}")
        axes[j].grid(alpha=0.3)

        k_plot = np.asarray(cluster_counts)
    small_numbers = np.asarray(small_cluster_counts)

    axes[3].plot(k_plot, history[:, 3], "o-", label="vx")
    axes[3].plot(k_plot, history[:, 4], "o-", label="vy")
    axes[3].set_ylabel("Mean STD — v SOI [km/s]")

    axes[4].plot(k_plot, np.rint(np.asarray(outlier_cluster_percent_2sigma) * k_plot / 100), "o-", color="darkorange", label="At least one outlier > 2σ")
    axes[4].plot(k_plot, np.rint(np.asarray(outlier_cluster_percent) * k_plot / 100), "o-", color="darkred", label="At least one outlier > 3σ")
    axes[4].plot(k_plot, small_numbers[:, 1], "s--", color="blue", label="< 10 trajectories")
    axes[4].plot(k_plot, small_numbers[:, 2], "s--", color="green", label="< 100 trajectories")
    axes[4].set_ylabel("Number of clusters")
    axes[4].set_title("Initial-state outliers and cluster sizes")
    axes[4].set_ylim(bottom=0)

    for ax in axes[3:5]:
        ax.set_xlabel("Number of clusters")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    small_percent = (100 * np.asarray(small_cluster_counts) / np.asarray(cluster_counts)[:, None])

    axes[5].plot(cluster_counts, outlier_cluster_percent_2sigma, "o-", color="darkorange", label="At least one outlier > 2σ")
    axes[5].plot(cluster_counts, outlier_cluster_percent, "o-", color="darkred", label="At least one outlier > 3σ")
    axes[5].plot(cluster_counts, small_percent[:, 1], "s--", color="blue", label="< 10 trajectories")
    axes[5].plot(cluster_counts, small_percent[:, 2], "s--", color="green", label="< 100 trajectories")

    axes[5].set_xlabel("Number of clusters")
    axes[5].set_ylabel("Clusters [%]")
    axes[5].set_title("Initial-state outliers and cluster sizes")
    axes[5].set_ylim(0, 100)
    axes[5].grid(alpha=0.3)
    axes[5].legend(fontsize=8)

    save_figure(fig, "physical_STD_and_3sigma_vs_K.png")



    diagnostics = np.column_stack((physical_stds[:, :3], np.asarray(escape_stats), sizes))

    names = [
        "STD eps [km^2/s^2]",
        "STD e [-]",
        "STD w [deg]",
        "STD |v SOI| [km/s]",
        "Direction dispersion [0, 1]",
        "Number of trajectories",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    for j, ax in enumerate(axes.ravel()):
        ax.scatter(cluster_ids, diagnostics[:, j], s=15)
        if j < 3:
            ax.axhline(STD_LIMITS[j], color="darkred", linestyle="--", linewidth=1)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel(names[j])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    axes.ravel()[4].set_ylim(0, 1)

    fig.suptitle(f"Initial compactness and escape diversity — K={len(cluster_ids)}")

    save_figure(fig, f"cluster_diagnostics_K{len(cluster_ids)}.png")    

    plot_dispersion_rankings(
        cluster_ids, sizes, escape_stats,
        ("STD |v SOI| [km/s]", "Direction dispersion [0, 1]"),
        "Escape diversity by orbital cluster", f"escape_dispersion_ranking_K{len(cluster_ids)}",
    )






# Plot cluster dispersion vs number of clusters
if SAVE:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(cluster_counts, mean_stds, "o-", color="blue")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Within-cluster STD")
    ax.set_title("Cluster dispersion vs number of clusters")

    fig.savefig(output_directory / "STD_vs_K.png", dpi=200)
    plt.close(fig)


print("\n=== GA_15 + SPLIT + FUSIONE + RAPPRESENTANTI ===")
print(f"Representatives: {len(representative_ids)}")
print(f"Final mean distance: {np.mean(minimum_distances[assigned]):.6f}")
print(f"Final 95th percentile: {np.percentile(minimum_distances[assigned], 95):.6f}")
print(f"Final 99th percentile: {np.percentile(minimum_distances[assigned], 99):.6f}")
print(f"Final maximum distance: {np.max(minimum_distances[assigned]):.6f}")










# Compute cluster statistics
unique_clusters, cluster_sizes = np.unique(labels[labels >= 0], return_counts=True)

cluster_standard_deviations = []
cluster_maximum_distances = []

n_clusters = len(unique_clusters)

cluster_variances = [np.mean(np.var(clustering_features[labels == cluster_id], axis=0))
    for cluster_id in unique_clusters]

mean_variance = np.average(cluster_variances, weights=cluster_sizes)


for cluster_id in unique_clusters:

    cluster_mask = labels == cluster_id
    cluster_features = clustering_features[cluster_mask]
    cluster_standard_deviations.append(np.sqrt(np.sum(np.var(cluster_features, axis=0))))

    cluster_maximum_distances.append(np.max(minimum_distances[cluster_mask]))


mean_cluster_standard_deviation = np.mean(cluster_standard_deviations)

mean_cluster_maximum_distance = np.mean(cluster_maximum_distances)



print(f"\nNumber of clusters: {n_clusters}")
print(f"Mean within-cluster standard deviation: {np.sqrt(mean_variance):.6f}")
print(f"\nMean cluster STD: {mean_cluster_standard_deviation:.6f}")
print(f"Mean cluster maximum distance: {mean_cluster_maximum_distance:.6f}\n")





# Mean within-cluster STD of physical quantities
velocity_unit = d / TU  

physical_quantities = {
    "Initial speed [km/s]"          :(np.linalg.norm(x0_selection[:, 3:6], axis=1) * velocity_unit),
    "Cj [-]"                        :selection[:, columns["Cj"]],
    "E_SOI [km^2/s^2]"              :selection[:, columns["E_SOI"]],
    "Minimum lunar altitude [km]"   :selection[:, columns["h_min_moon"]],
    "Perigee altitude [km]"         :selection[:, columns["h_perigee"]],
}

mean_physical_stds = {}

for quantity_name, values in physical_quantities.items():

    cluster_stds = [np.std(values[labels == cluster_id], ddof=1) for cluster_id in unique_clusters
        if np.sum(labels == cluster_id) > 1]

    mean_physical_stds[quantity_name] = np.mean(cluster_stds) if cluster_stds else np.nan


print("\n=== MEAN WITHIN-CLUSTER PHYSICAL STD ===\n" + "\n".join(f"{quantity_name}: {mean_std:.6f}" for quantity_name, mean_std in mean_physical_stds.items()) + "\n")







# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")




# Plot clusters and representative trajectories in both frames.
plot_frames = (
    (False, "Rotating frame", ""),
    (True, f"Geocentric inertial frame (Moon at t = {T_min:g} TU)", "Geocentric_inertial_"),
)

if PLOT_CLUSTERS or SAVE:

    cmap = plt.get_cmap("turbo", max(n_clusters, 1),)
    cluster_colors = {cluster_id: cmap(color_id) for color_id, cluster_id in enumerate(unique_clusters)}
    groups = [(cluster_id, f"Cluster {cluster_id}") for cluster_id in unique_clusters]
    max_panels = 24
    max_columns = 6

    for inertial, frame_name, suffix in plot_frames:
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
                    if inertial:
                        trajectory = geocentric_inertial_trajectory(
                            trajectory, trajectory_time_bounds[trajectory_id],
                        )
                    ax.plot(trajectory[0], trajectory[1], color=color, alpha=0.12, linewidth=0.7)

                plot_frame_bodies(ax, inertial)
                ax.set_title(f"{title} || {np.count_nonzero(group_mask)} trajectories")
                ax.set_xlabel("X [LU]")
                ax.set_ylabel("Y [LU]")
                ax.set_aspect("equal")
                ax.set_box_aspect(1)

            for ax in axes[n_panels:]:
                ax.axis("off")

            figure_number = start // max_panels + 1
            fig.suptitle(f"{frame_name} - Cluster groups - Figure {figure_number}")
            save_figure(fig, f"{suffix}clusters_{figure_number:03d}.png")


# Plot representative trajectories.
if PLOT_MEDOIDS or SAVE:
    for inertial, frame_name, suffix in plot_frames:
        fig_representatives, ax_representatives = plt.subplots(figsize=(10, 8), constrained_layout=True)

        for representative_id in representative_ids:
            trajectory = X_curvature[representative_id]
            if inertial:
                trajectory = geocentric_inertial_trajectory(
                    trajectory, trajectory_time_bounds[representative_id],
                )
            ax_representatives.plot(
                trajectory[0], trajectory[1], color="navy", alpha=0.2, linewidth=1, rasterized=True,
            )

        plot_frame_bodies(ax_representatives, inertial, size=50)
        ax_representatives.set_xlabel("X [LU]")
        ax_representatives.set_ylabel("Y [LU]")
        ax_representatives.set_title(f"{frame_name} - {len(representative_ids)} medoids")
        ax_representatives.set_aspect("equal")
        ax_representatives.set_box_aspect(1)
        ax_representatives.grid(True, alpha=0.25)
        ax_representatives.legend()
        save_figure(fig_representatives, f"representatives{suffix}.png")


# Plot clustering in physical feature space
if SAVE or PLOT_CLUSTERS:

    features_plot = event_features[:, :3].copy()
    features_plot[:, 2] %= 360

    names = [r"$\varepsilon$ [km$^2$/s$^2$]", "e [-]", r"$\omega$ [deg]"]
    pairs = [(0, 1), (0, 2), (1, 2)]

    id_style = dict(
        fontsize=8,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )

    fig2, axes2 = plt.subplots(
        1, 3, figsize=(18, 5), constrained_layout=True
    )

    fig3 = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax3 = fig3.add_subplot(111, projection="3d")

    for cid in unique_clusters:

        values = features_plot[labels == cid]
        color = cluster_colors[cid]
        representative = features_plot[representative_ids[cid]]

        for ax, (i, j) in zip(axes2, pairs):
            ax.scatter(
                values[:, i], values[:, j],
                s=3, alpha=0.3, color=color, rasterized=True,
            )
            ax.annotate(
                str(cid),
                representative[[i, j]],
                xytext=(4, 4),
                textcoords="offset points",
                **id_style,
            )

        ax3.scatter(
            values[:, 0], values[:, 1], values[:, 2],
            s=3, alpha=0.3, color=color, depthshade=False,
        )
        ax3.text(*representative, str(cid), **id_style)

    for ax, (i, j) in zip(axes2, pairs):
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.grid(alpha=0.3)

    ax3.set_xlabel(names[0])
    ax3.set_ylabel(names[1])
    ax3.set_zlabel(names[2])

    fig2.suptitle(f"Feature projections — K={n_clusters}")
    ax3.set_title(f"Feature space — K={n_clusters}")

    save_figure(fig2, f"features_2D_K{n_clusters}.png")
    save_figure(fig3, f"features_3D_K{n_clusters}.png")

    # One additional feature-dispersion figure per cluster
    for cid in unique_clusters:

        values = features_plot[labels == cid].copy()
        shape_penalty = gaussian_shape_penalty(values) if len(values) >= 2 else np.ones(3)
        color = cluster_colors[cid]
        local_names = names.copy()

        # Recenter omega to avoid artificial spreading across 0/360 degrees.
        angular_mean = np.mean(np.exp(1j * np.deg2rad(values[:, 2])))
        angle_defined = abs(angular_mean) > 1e-8

        if angle_defined:
            origin = np.rad2deg(np.angle(angular_mean))
            values[:, 2] = (values[:, 2] - origin + 180) % 360 - 180
            local_names[2] = r"$\Delta\omega$ [deg]"

        stds = np.std(values, axis=0)

        if not angle_defined:
            stds[2] = np.nan

        fig, axes = plt.subplots(
            2, 3, figsize=(15, 8), constrained_layout=True
        )

        for ax, (i, j) in zip(axes[0], pairs):
            ax.scatter(
                values[:, i], values[:, j],
                s=5, alpha=0.35, color=color, rasterized=True,
            )
            ax.set_xlabel(local_names[i])
            ax.set_ylabel(local_names[j])
            ax.grid(alpha=0.3)

        for j, ax in enumerate(axes[1]):
            ax.hist(values[:, j], bins=30, color=color, alpha=0.8)
            ax.set_xlabel(local_names[j])
            ax.set_ylabel("Number of trajectories")
            ax.set_title(f"STD = {stds[j]:.5g} | Gaussian Q-Q = {shape_penalty[j]:.4g}")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Cluster {cid} — {len(values)} trajectories")
        save_figure(fig, f"cluster_{cid:03d}_features.png")



print_clustering_summary(labels)

# Show or close figures
if PLOT_CLUSTERS:
    plt.show()
else:
    plt.close("all")


if PLOT_CLUSTERS or PLOT_MEDOIDS:
    plt.show()
