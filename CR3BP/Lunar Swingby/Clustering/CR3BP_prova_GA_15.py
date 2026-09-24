from Celestial_Mechanics import Integration, Transformations
from pathlib import Path
from multiprocessing import get_context
import time
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, diags, hstack, vstack
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


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

    return features, np.hstack((x_b, x_f))



# Define normalization function for features
def normalize_block(features):

    centered = features - np.mean(features, axis=0)
    scale = np.linalg.norm(centered) / np.sqrt(len(centered))

    if scale <= 1e-14:
        raise ValueError("Feature block has zero variance.")

    return centered / scale



def physical_cluster_stds(physical, labels):
    """Population STD of eps, e and omega, with omega wrapped per cluster."""
    rows = []
    for cluster_id in np.unique(labels):
        values = physical[labels == cluster_id, :3].copy()
        angular_mean = np.mean(np.exp(1j * np.deg2rad(values[:, 2])))
        if abs(angular_mean) < 1e-8:
            raise RuntimeError("Orientamento medio del cluster non definito.")
        origin = np.rad2deg(np.angle(angular_mean))
        values[:, 2] = (values[:, 2] - origin + 180) % 360 - 180
        rows.append(np.std(values, axis=0, ddof=0))
    return np.asarray(rows)


def constrained_clustering(physical, directions, reference_std, final_std_limits,
                           clusters_by_direction, min_size=100, large_size=1000,
                           max_small=16, max_outlier_fraction=0.01):
    """Standard KMeans centres followed by constrained, integer assignments.

    The density filter trains centres only. Every input row is assigned.
    Moment bounds use physical units; 2-sigma diagnostics retain the original
    globally normalized eps/e/(cos omega, sin omega) metric.
    """
    physical = np.asarray(physical, dtype=np.float64)
    directions = np.asarray(directions)
    reference_std = np.asarray(reference_std, dtype=np.float64)
    final_std_limits = np.asarray(final_std_limits, dtype=np.float64)
    if (physical.ndim != 2 or physical.shape[1] != 3 or
            directions.shape != (len(physical),) or
            not np.all(np.isfinite(physical)) or
            not np.all(np.isin(directions, [-1, 1]))):
        raise ValueError("Feature non finite, forma errata o verso non definito.")
    if (reference_std.shape != (3,) or final_std_limits.shape != (3,) or
            not np.all(np.isfinite(reference_std)) or
            not np.all(np.isfinite(final_std_limits)) or
            np.any(reference_std <= 0) or np.any(final_std_limits <= 0) or
            not 0 < min_size < large_size):
        raise ValueError("Limiti di dispersione o numerosita non validi.")
    if set(np.unique(directions)) != set(clusters_by_direction):
        raise ValueError("I versi presenti non corrispondono ai conteggi richiesti.")

    angle = np.deg2rad(physical[:, 2])
    angular = np.column_stack((np.cos(angle), np.sin(angle)))
    metric = np.column_stack((physical[:, :2] / reference_std[:2],
                              angular / np.deg2rad(reference_std[2])))

    def solve_lp(cost, eq, ub, rhs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected")
            return linprog(
                cost, A_eq=eq, b_eq=np.ones(eq.shape[0]), A_ub=ub, b_ub=rhs,
                bounds=(0, 1), method="highs-ds",
                options={"time_limit": 60, "threads": 2},
            )

    def initial_assignment(z, centres, lower):
        distances = cdist(z, centres, metric="sqeuclidean")
        ordered = np.argsort(distances, axis=1)
        n, k = distances.shape
        for neighbours in sorted({min(k, value) for value in (4, 8, 16, k)}):
            nearest = ordered[:, :neighbours]
            rows = np.repeat(np.arange(n), neighbours)
            edges = np.arange(len(rows))
            eq = coo_matrix((np.ones(len(rows)), (rows, edges)),
                            shape=(n, len(rows))).tocsr()
            ub = coo_matrix((-np.ones(len(rows)), (nearest.ravel(), edges)),
                            shape=(k, len(rows))).tocsr()
            result = solve_lp(distances[rows, nearest.ravel()], eq, ub, -lower)
            if not result.success:
                continue
            # Pure transportation is integral; never silently round a relaxation.
            if np.max(np.abs(result.x - np.rint(result.x))) > 1e-6:
                raise RuntimeError("Assegnazione iniziale non intera.")
            chosen = np.flatnonzero(result.x > 0.5)
            if len(chosen) != n or np.unique(rows[chosen]).size != n:
                raise RuntimeError("Copertura iniziale non valida.")
            labels = np.empty(n, dtype=int)
            labels[rows[chosen]] = nearest.ravel()[chosen]
            if np.all(np.bincount(labels, minlength=k) >= lower):
                return labels
        raise RuntimeError("Assegnazione iniziale non trovata: " + result.message)

    def moment_problem(p, z, assigned, lower, factor, neighbours, angle_limit):
        n, k = len(p), len(lower)
        centres = np.empty((k, 3))
        metric_centres = np.empty((k, 4))
        for j in range(k):
            members = assigned == j
            centres[j, :2] = p[members, :2].mean(axis=0)
            centres[j, 2] = np.rad2deg(np.angle(
                np.mean(np.exp(1j * np.deg2rad(p[members, 2])))))
            metric_centres[j] = z[members].mean(axis=0)
        distances = cdist(z, metric_centres, metric="sqeuclidean")
        angle_delta = (p[:, None, 2] - centres[None, :, 2] + 180) % 360 - 180
        distances[np.abs(angle_delta) >= angle_limit] = np.inf
        take = min(neighbours, k)
        nearest = np.argsort(distances, axis=1)[:, :take]
        rows = np.repeat(np.arange(n), take)
        cols = nearest.ravel()
        costs = distances[rows, cols]
        finite = np.isfinite(costs)
        rows, cols, costs = rows[finite], cols[finite], costs[finite]
        if np.any(np.bincount(rows, minlength=n) == 0):
            raise RuntimeError("Una traiettoria non ha destinazioni ammissibili.")
        deviations = np.vstack((
            ((p[rows, 0] - centres[cols, 0]) / (reference_std[0] * factor))**2 - 1,
            ((p[rows, 1] - centres[cols, 1]) / (reference_std[1] * factor))**2 - 1,
            (angle_delta[rows, cols] / (reference_std[2] * factor))**2 - 1,
        ))
        edges = np.arange(len(rows))
        eq = coo_matrix((np.ones(len(rows)), (rows, edges)),
                        shape=(n, len(rows))).tocsr()
        # Sum [(deviation / cap)^2 - 1] <= 0 bounds the second moment.
        ub = coo_matrix(
            (np.r_[-np.ones(len(rows)), deviations.ravel()],
             (np.r_[cols, k + cols, 2*k + cols, 3*k + cols], np.tile(edges, 4))),
            shape=(4*k, len(rows)),
        ).tocsr()
        return dict(eq=eq, ub=ub, rows=rows, cols=cols, cost=costs,
                    lower=lower.copy(), n=n, k=k, factor=factor)

    def select_small_clusters(problem, budget, number):
        n, k, m = problem["n"], problem["k"], len(problem["rows"])
        eq = hstack((problem["eq"], csr_matrix((n, k))), format="csr")
        ypart = vstack((-(large_size - min_size) * diags(np.ones(k)),
                       csr_matrix((3*k, k))), format="csr")
        ub = hstack((problem["ub"], ypart), format="csr")
        budget_row = hstack((csr_matrix((1, m)), csr_matrix(np.ones((1, k)))),
                            format="csr")
        ub = vstack((ub, budget_row), format="csr")
        rhs = np.r_[np.full(k, -large_size), np.zeros(3*k), budget]
        result = solve_lp(np.r_[problem["cost"], np.zeros(k)], eq, ub, rhs)
        if not result.success:
            raise RuntimeError("Selezione dei cluster piccoli fallita: " + result.message)
        # The continuous y values rank candidates; they are NOT final labels.
        lower = np.full(k, large_size)
        lower[np.argsort(-result.x[m:])[:number]] = min_size
        problem["lower"] = lower

    def repair_assignment(problem, x):
        """Repair fractional rows with a small MILP, then check every row."""
        rows, cols = problem["rows"], problem["cols"]
        n, k = problem["n"], problem["k"]
        fractional = np.unique(rows[(x > 1e-7) & (x < 1 - 1e-7)])
        integral = x > 1 - 1e-7
        # Margins only assist integer repair; actual final STD is checked below.
        attempts = ((1., 0), (1., 2), (1., 8),
                    (1.0001, 0), (1.001, 0), (1.005, 0),
                    (1.001, 2), (1.005, 8), (1.01, 8))
        for margin, release in attempts:
            ub = problem["ub"].copy()
            start = ub.indptr[k]
            ub.data[start:] = (ub.data[start:] + 1) / margin**2 - 1
            rhs = np.r_[-problem["lower"], np.zeros(3*k)]
            free = np.zeros(n, dtype=bool)
            free[fractional] = True
            if release:
                alternative = np.full(n, np.inf)
                fixed_cost = np.zeros(n)
                fixed_cost[rows[integral]] = problem["cost"][integral]
                np.minimum.at(alternative, rows[~integral], problem["cost"][~integral])
                for j in range(k):
                    points = rows[integral & (cols == j)]
                    points = points[~free[points]]
                    chosen = np.argsort(alternative[points] - fixed_cost[points])[:release]
                    free[points[chosen]] = True
            fixed = integral & ~free[rows]
            active = np.flatnonzero(free[rows])
            free_ids = np.flatnonzero(free)
            chosen = np.flatnonzero(fixed)
            if len(active):
                remap = np.full(n, -1, dtype=int)
                remap[free_ids] = np.arange(len(free_ids))
                eq = coo_matrix((np.ones(len(active)),
                                 (remap[rows[active]], np.arange(len(active)))),
                                shape=(len(free_ids), len(active))).tocsr()
                remaining = rhs - np.asarray(ub[:, fixed].sum(axis=1)).ravel()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Unrecognized options detected")
                    result = milp(
                        problem["cost"][active], integrality=np.ones(len(active), dtype=int),
                        bounds=Bounds(0, 1),
                        constraints=[
                            LinearConstraint(eq, np.ones(len(free_ids)), np.ones(len(free_ids))),
                            LinearConstraint(ub[:, active], np.full(4*k, -np.inf), remaining),
                        ],
                        options={"time_limit": 60, "mip_rel_gap": 1e-7, "threads": 2},
                    )
                # A time-limited incumbent is usable only after these checks.
                if (result.x is None or not np.all(np.isfinite(result.x)) or
                        np.max(np.abs(result.x - np.rint(result.x))) > 1e-6):
                    continue
                chosen = np.r_[chosen, active[result.x > 0.5]]
            if len(chosen) != n or np.unique(rows[chosen]).size != n:
                continue
            assigned = np.empty(n, dtype=int)
            assigned[rows[chosen]] = cols[chosen]
            violation = np.asarray(ub[:, chosen].sum(axis=1)).ravel() - rhs
            if (np.all(np.bincount(assigned, minlength=k) >= problem["lower"]) and
                    np.max(violation) <= 1e-5):
                return assigned
        raise RuntimeError("Nessuna assegnazione intera rispetta i vincoli dello stadio.")

    # (STD factor, relaxed small-cluster budget, final small slots,
    #  candidate neighbours, angular arc). Recompute anchors after each stage.
    stages = {
        -1: ((2., 3, 7, 12, 90), (1.75, 6, 8, 12, 90),
             (1.65, None, 8, 16, 181)),
         1: ((2., 6, 6, 12, 90), (1.5, 7, 8, 12, 90)),
    }
    initial_small = {-1: 3, 1: 6}
    labels = np.full(len(physical), -1, dtype=int)
    offset = 0
    with threadpool_limits(limits=2):
        for sign in sorted(clusters_by_direction):
            ids = np.flatnonzero(directions == sign)
            p, z = physical[ids], metric[ids]
            k = clusters_by_direction[sign]
            if k < 8 or len(ids) < (k - initial_small[sign]) * large_size + initial_small[sign] * min_size:
                raise ValueError(f"Numerosita insufficiente per il verso {sign:+g} e K={k}.")
            density_radius = cKDTree(z).query(z, k=30, workers=2)[0][:, -1]
            core = density_radius <= np.quantile(density_radius, 0.99)
            model = KMeans(n_clusters=k, n_init=10, random_state=42,
                           max_iter=500, tol=1e-9, algorithm="lloyd")
            model.fit(z[core])
            counts = np.bincount(model.predict(z), minlength=k)
            lower = np.full(k, large_size)
            lower[np.argsort(counts)[:initial_small[sign]]] = min_size
            assigned = initial_assignment(z, model.cluster_centers_, lower)
            print(f"Verso {sign:+g}: K={k}, assegnazione iniziale completa.", flush=True)
            for factor, budget, slots, neighbours, arc in stages[sign]:
                problem = moment_problem(p, z, assigned, lower, factor, neighbours, arc)
                if budget is not None:
                    select_small_clusters(problem, budget, slots)
                lower = problem["lower"]
                rhs = np.r_[-lower, np.zeros(3*k)]
                result = solve_lp(problem["cost"], problem["eq"], problem["ub"], rhs)
                if not result.success:
                    raise RuntimeError(f"Vincoli STD non risolti per verso {sign:+g}: " + result.message)
                assigned = repair_assignment(problem, result.x)
                print(f"Verso {sign:+g}: stadio STD x{factor:g} completato.", flush=True)
            labels[ids] = assigned + offset
            offset += k

    # Validate the actual members, including wrapped angles and every outlier.
    cluster_ids, sizes = np.unique(labels, return_counts=True)
    if (not np.array_equal(cluster_ids, np.arange(sum(clusters_by_direction.values()))) or
            sizes.sum() != len(physical) or sizes.min() < min_size or
            np.count_nonzero(sizes < large_size) > max_small):
        raise RuntimeError("Partizione finale non conforme a copertura/numerosita.")
    stds = physical_cluster_stds(physical, labels)
    if np.any(stds > final_std_limits + 1e-8):
        raise RuntimeError(f"STD finali oltre i limiti: {stds.max(axis=0)}.")
    normalized = np.column_stack((normalize_block(physical[:, 0:1]),
                                  normalize_block(physical[:, 1:2]),
                                  normalize_block(angular)))
    outlier_points = 0
    for cluster_id in cluster_ids:
        members = labels == cluster_id
        if np.unique(directions[members]).size != 1:
            raise RuntimeError("Cluster con versi misti.")
        values = normalized[members]
        distance_squared = np.sum((values - values.mean(axis=0))**2, axis=1)
        outlier_points += np.count_nonzero(distance_squared > 4 * np.var(values, axis=0).sum())
    if outlier_points / len(labels) > max_outlier_fraction:
        raise RuntimeError(f"Punti oltre 2sigma: {100 * outlier_points / len(labels):.3f}% "
                           f"(limite {100 * max_outlier_fraction:g}%).")
    print(f"Copertura: {len(labels)}/{len(physical)}; K={len(sizes)}; "
          f"min={sizes.min()}; cluster <{large_size}: {np.count_nonzero(sizes < large_size)}; "
          f"punti >2sigma: {outlier_points} ({100 * outlier_points / len(labels):.3f}%); "
          f"STD massime eps/e/w: {stds.max(axis=0)}", flush=True)
    return labels



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

        event_features[n_kept], X_curvature[n_kept] = result
        keep[i] = True
        n_kept += 1


if n_kept == 0:
    raise RuntimeError("Nessuna traiettoria con perigeo ammissibile.")

event_features = event_features[:n_kept]
X_curvature = X_curvature[:n_kept]
source_ids = source_ids[keep]
selection = database[source_ids, :]
x0_selection = selection[:, :6]

print(f"Traiettorie mantenute: {n_kept}; escluse: {len(keep) - n_kept}")



# Prepare clustering features
w = np.deg2rad(event_features[:, 2])
angular_features = np.column_stack((np.cos(w), np.sin(w)))

clustering_features = np.column_stack((
    normalize_block(event_features[:, 0:1]),   # eps
    normalize_block(event_features[:, 1:2]),   # e
    normalize_block(angular_features),         # w
    100 *normalize_block(event_features[:, 6:7]),                      # verso al perigeo
 #  normalize_block(event_features[:, 3:6]),   # v_SOI
))

#clustering_features[:, :4] /= np.sqrt(3)



# Build a feasible partition, then refine it with a finer KMeans base.
MIN_CLUSTER_SIZE = 1000
MAX_SIGMA = 2.0
BASE_CLUSTER_COUNTS = (99,)  # Initial groups, not the final number of clusters.

partitions = []
labels = None
for n_base in BASE_CLUSTER_COUNTS:
    labels = constrained_clustering(
        clustering_features, event_features[:, 6], n_base,
        min_size=MIN_CLUSTER_SIZE, sigma_limit=MAX_SIGMA, previous_labels=labels,
    )
    partitions.append(labels)

cluster_counts = []
mean_stds = []
small_cluster_counts = []
small_trajectory_fractions = []
escape_stats = []

output_directory = (Path(__file__).resolve().parent / f"Clusters/GA_15_plots_K{len(np.unique(labels))}")
output_directory.mkdir(parents=True, exist_ok=True)

physical_std_history = []
outlier_cluster_percent = []
outlier_cluster_percent_2sigma = []


# Keep constrained assignments when selecting representatives.
for stage, labels in enumerate(partitions):

    representative_ids = np.empty(len(np.unique(labels)), dtype=int)

    for cluster_id in np.unique(labels):

        cluster_ids = np.flatnonzero(labels == cluster_id)
        cluster_center = clustering_features[cluster_ids].mean(axis=0)

        distances_squared = np.sum((clustering_features[cluster_ids] - cluster_center)**2, axis=1)

        representative_ids[cluster_id] = cluster_ids[np.argmin(distances_squared)]


    minimum_distances = np.linalg.norm(
        clustering_features - clustering_features[representative_ids[labels]], axis=1,
    )

    representative_source_ids = source_ids[representative_ids]

    cluster_ids, sizes = np.unique(labels, return_counts=True)

    variances = [np.mean(np.var(clustering_features[labels == c], axis=0)) for c in cluster_ids]

    mean_std = np.sqrt(np.average(variances, weights=sizes))

    variances = []
    physical_stds = []
    outlier_clusters = 0
    outlier_clusters_2sigma = 0
    outlier_points_2sigma = 0
    initial_stds = physical_cluster_stds(event_features[:, :3], labels)

    for cluster_position, c in enumerate(cluster_ids):
        mask = labels == c
        features_c = clustering_features[mask]

        variance_c = np.var(features_c, axis=0)
        variances.append(np.mean(variance_c))

        delta = features_c - np.mean(features_c, axis=0)
        distance_squared = np.sum(delta**2, axis=1)
        sigma_total_squared = np.sum(variance_c)

        outlier_clusters += int(np.any(distance_squared > 9 * sigma_total_squared))
        outlier_clusters_2sigma += int(np.any(distance_squared > 4 * sigma_total_squared))
        outlier_points_2sigma += np.count_nonzero(distance_squared > 4 * sigma_total_squared)

        physical_stds.append(np.r_[initial_stds[cluster_position],
                                   np.std(event_features[mask, 3:5], axis=0)])

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
          f"clusters > 2sigma = {outlier_clusters_2sigma}/{len(cluster_ids)}, "
          f"points > 2sigma = {outlier_points_2sigma}/{len(labels)} "
          f"({100 * outlier_points_2sigma / len(labels):.3f}%), "
          f"max STD eps/e/w = {initial_stds.max(axis=0)}", flush=True)


    masks = (sizes == 1, sizes < MIN_CLUSTER_SIZE, sizes < LARGE_CLUSTER_SIZE)

    small_cluster_counts.append([np.count_nonzero(mask) for mask in masks])
    small_trajectory_fractions.append([np.sum(sizes[mask]) / len(labels) for mask in masks])



# Plot mean within-cluster physical STD vs number of clusters
if SAVE or PLOT_CLUSTERS:
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
    axes[4].plot(k_plot, small_numbers[:, 1], "s--", color="blue", label=f"< {MIN_CLUSTER_SIZE} trajectories")
    axes[4].plot(k_plot, small_numbers[:, 2], "s--", color="green", label=f"< {LARGE_CLUSTER_SIZE} trajectories")
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
    axes[5].plot(cluster_counts, small_percent[:, 1], "s--", color="blue", label=f"< {MIN_CLUSTER_SIZE} trajectories")
    axes[5].plot(cluster_counts, small_percent[:, 2], "s--", color="green", label=f"< {LARGE_CLUSTER_SIZE} trajectories")

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
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel(names[j])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    axes.ravel()[4].set_ylim(0, 1)

    fig.suptitle(f"Initial compactness and escape diversity — K={len(cluster_ids)}")

    save_figure(fig, f"cluster_diagnostics_K{len(cluster_ids)}.png")    





# Plot cluster dispersion vs number of clusters
if SAVE:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(cluster_counts, mean_stds, "o-", color="blue")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Within-cluster STD")
    ax.set_title("Cluster dispersion vs number of clusters")

    fig.savefig(output_directory / "STD_vs_K.png", dpi=200)
    plt.close(fig)


print("\n=== KMEANS + ASSEGNAZIONE VINCOLATA + RAPPRESENTANTI ===")
print(f"Representatives: {len(representative_ids)}")
print(f"Final mean distance: {np.mean(minimum_distances):.6f}")
print(f"Final 95th percentile: {np.percentile(minimum_distances, 95):.6f}")
print(f"Final 99th percentile: {np.percentile(minimum_distances, 99):.6f}")
print(f"Final maximum distance: {np.max(minimum_distances):.6f}")










# Compute cluster statistics
unique_clusters, cluster_sizes = np.unique(labels, return_counts=True)

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

    mean_physical_stds[quantity_name] = np.mean(cluster_stds)


print("\n=== MEAN WITHIN-CLUSTER PHYSICAL STD ===\n" + "\n".join(f"{quantity_name}: {mean_std:.6f}" for quantity_name, mean_std in mean_physical_stds.items()) + "\n")







# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")




# Plot clusters and representative trajectories
if PLOT_CLUSTERS or SAVE:

    cmap = plt.get_cmap("turbo", max(n_clusters, 1),)

    cluster_colors = {cluster_id: cmap(color_id) for color_id, cluster_id in enumerate(unique_clusters)}

    groups = []

    for cluster_id in unique_clusters:
        groups.append((cluster_id, f"Cluster {cluster_id}"))


    max_panels = 24
    max_columns = 6


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

            ax.set_box_aspect(1)


        for ax in axes[n_panels:]:
            ax.axis("off")


        figure_number = (start // max_panels + 1)

        fig.suptitle(f"Cluster groups — Figure {figure_number}")
        save_figure(fig, f"clusters_{figure_number:03d}.png")



# Plot representative trajectories
if PLOT_MEDOIDS or SAVE:

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


if PLOT_CLUSTERS or PLOT_MEDOIDS:
    plt.show()
