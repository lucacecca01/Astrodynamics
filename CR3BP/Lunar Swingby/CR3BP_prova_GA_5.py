from Celestial_Mechanics import Integration, Transformations
import numpy as np
from multiprocessing import get_context
import time
from matplotlib import pyplot as plt
import pykep as pk
from scipy.integrate import solve_ivp
from scipy.optimize import root, differential_evolution


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



PLOT = True
MASK = False
SAVE = True



# Define integration function
def integrate_one(x0, T_min, T_max, dt, events=None):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='CR3BP',
        integrator='scipy',
        method='DOP853',
        rtol=1e-9,
        events=events,
    )

    return sol



# Define stop condition for exiting Earth's sphere of influence
def earth_SOI_exit(t, x, mu):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)

    return E_SOI / d - r_earth



# Define function to compute the corrected departure state based on delta_v
def corrected_departure_state(delta_v):

    v_trial = v_lambert_departure + delta_v

    x_trial_geocentric = np.hstack((r_departure, v_trial))

    x_trial_barycentric = (x_trial_geocentric + earth_departure_inertial)

    x_trial_rotating = Transformations.inertial_to_CR3BP(x_trial_barycentric, n, departure_time).ravel()

    x_trial_normalized = (Transformations.SV_to_CR3BP_normalized_units(x_trial_rotating, d, G, M))

    return x_trial_normalized



# Define the shooting residual function for root-finding
def shooting_residual(delta_v):

    x0_trial = corrected_departure_state(delta_v)

    trial_solution = solve_ivp(
        Integrator.CR3BP_ODE,
        [0, transfer_duration],
        x0_trial,
        args=(mu,),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        t_eval=[transfer_duration]
    )

    if not trial_solution.success:
        return np.array([1e3, 1e3, 1e3]) 

    final_position = trial_solution.y[:3, -1]

    return (final_position - x_target_normalized[:3])



# Define the objective function for Lambert's problem optimization
def lambert_objective(design):

    target_id_candidate = int(design[0])
    theta_candidate = design[1] % (2 * np.pi)
    tof_fraction = design[2]
    direction = bool(int(design[3]))

    target_time_candidate = (far_times[trajectory_id][target_id_candidate])

    direction_candidate = (-1.0 if direction else 1.0)

    tof_max_candidate = (target_time_candidate - T_max * TU)

    if tof_max_candidate <= tof_min:
        return 1e12

    tof_candidate = (tof_min + tof_fraction * (tof_max_candidate - tof_min))

    target_state = (far_points_geocentric[trajectory_id][:, target_id_candidate])

    r_target_candidate = target_state[:3]
    v_target_candidate = target_state[3:]

    r_candidate = r0 * np.array([np.cos(theta_candidate), np.sin(theta_candidate), 0])

    v_circular_candidate = (direction_candidate * v_circular_magnitude * np.array([-np.sin(theta_candidate), np.cos(theta_candidate), 0]))

    try:
        lambert_candidate = pk.lambert_problem(
            r0=r_candidate,
            r1=r_target_candidate,
            tof=tof_candidate,
            mu=mu_earth,
            cw=direction,
            multi_revs=0,
        )

    except (RuntimeError, ValueError):
        return 1e12

    v0_candidate = np.asarray(lambert_candidate.v0[0])

    v1_candidate = np.asarray(lambert_candidate.v1[0])

    return (np.linalg.norm(v0_candidate - v_circular_candidate) + np.linalg.norm(v1_candidate - v_target_candidate))



# Define the homotopy ODE for gradually introducing the Moon's influence
def homotopy_ode(t, x, mu, strength):

    dxdt = np.zeros(6)
    dxdt[:3] = x[3:]

    x_E = x[0] + mu
    x_M = x[0] - (1 - mu)

    r_E = np.sqrt(x_E**2 + x[1]**2 + x[2]**2)
    r_M = np.sqrt(x_M**2 + x[1]**2 + x[2]**2)

    # Earth-only dynamics in the rotating frame
    dxdt[3] = 2*x[4] + x_E - (1-mu)*x_E/r_E**3
    dxdt[4] = -2*x[3] + x[1] - (1-mu)*x[1]/r_E**3
    dxdt[5] = -(1-mu)*x[2]/r_E**3

    # Gradually add the Moon
    dxdt[3] += strength*mu*(-x_M/r_M**3 - 1)
    dxdt[4] += strength*mu*(-x[1]/r_M**3)
    dxdt[5] += strength*mu*(-x[2]/r_M**3)

    return dxdt



# Define the homotopy residual function for root-finding
def homotopy_residual(delta_v, strength):

    x0 = corrected_departure_state(delta_v)

    sol = solve_ivp(
        homotopy_ode,
        [0, transfer_duration],
        x0,
        args=(mu, strength),
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
        t_eval=[transfer_duration]
    )

    if not sol.success:
        return np.full(3, 1e3)

    return sol.y[:3, -1] - x_target_normalized[:3]




earth_SOI_exit.terminal = True
earth_SOI_exit.direction = -1



# Input
DATA_FILE = "/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Escape_initial_conditions_CR3BP.txt"



# Initialize the integrator
start_time = time.perf_counter()
Integrator = Integration()



# Initialize lists to store results
T = []
X = []
far_times = [] 
far_points_normalized = []
far_points_rotating = []
far_points_inertial = [] 
far_points_geocentric = []



# Define masses and integration parameters
masses = [m1, m2, 0]
T_min = 0
T_max = -4 * np.pi
dt = -0.001



r_p = (R_E + 200) / d
r_a_max = E_SOI / d
mu_E = 1 - mu

v_circ = np.sqrt(mu_E / r_p)

a_max = 0.5 * (r_p + r_a_max)
v_max = np.sqrt(mu_E * (2 / r_p - 1 / a_max))



# Load initial conditions from data files
database = np.atleast_2d(np.loadtxt(DATA_FILE, skiprows=1))

SORT_BY = "E_SOI"
DESCENDING = True

columns = {"Cj": 6, "E_SOI": 7, "h_min_moon": 8, "h_last_earth": 9}


order = np.argsort(database[:, columns[SORT_BY]], kind="stable")
selection = database[order[::-1] if DESCENDING else order]

source_ids = np.array([4508, 54212, 35554, 99374, 98436])
#x0_selection = database[source_ids[1:2], :6]
x0_selection = selection[2:3, :6]


pool = get_context("fork").Pool()

sol = pool.starmap(integrate_one, [(x0, T_min, T_max, dt, earth_SOI_exit) for x0 in x0_selection])

T = [res[0] for res in sol]
X = [res[1] for res in sol]

X_bar = [Transformations.CR3BP_to_inertial(Transformations.CR3BP_normalized_units_to_SV(traj, d, G, M), n, time * TU) for time, traj in zip(T, X)]

pool.close()
pool.join()




# Compute far points for each trajectory
for t, traj in zip(T, X):

    r_moon = traj[:3].T - np.array([1 - mu, 0, 0])
    r_earth = traj[:3].T - np.array([-mu, 0, 0])

    r_m = np.linalg.norm(r_moon, axis=1)
    r_e = np.linalg.norm(r_earth, axis=1)

    a_earth = -(1 - mu) * r_earth / r_e[:, None]**3
    a_moon = -mu * r_moon / r_m[:, None]**3

    a_moon_earth = np.array([mu, 0, 0])
    a_moon_perturbation = a_moon - a_moon_earth

    threshold = 0.01
    perturbation = eta = (np.linalg.norm(a_moon_perturbation, axis=1)/ np.linalg.norm(a_earth, axis=1))

    #mask1 = r_m >= 2 * M_SOI / d
    #mask2 = r_e <= 1

    if MASK:
        mask = np.logical_and(eta <= threshold, r_e < 1)
    else:
        mask = np.ones_like(r_e, dtype=bool)

    far_points_rotating_i = Transformations.CR3BP_normalized_units_to_SV(traj[:, mask], d, G, M)

    far_times.append(t[mask] * TU)
    far_points_normalized.append(traj[:, mask])
    far_points_rotating.append(far_points_rotating_i)
    far_points_inertial.append(Transformations.CR3BP_to_inertial(far_points_rotating_i, n, t[mask] * TU))


for t, x_baricentric in zip(far_times, far_points_inertial):

    earth = Transformations.CR3BP_to_inertial([-mu*d, 0, 0, 0, 0, 0], n, t)

    x_geocentric = x_baricentric - earth

    far_points_geocentric.append(x_geocentric)

    



# Lambert's problem and optimization to find the best transfer trajectory
trajectory_id = 0

number_of_targets = len(far_times[trajectory_id])

r0 = R_E + 200
tof_min = 0.5 * day

v_circular_magnitude = np.sqrt(mu_earth / r0)

optimization_result = differential_evolution(
    lambert_objective,
    bounds=[(0, number_of_targets - 1), (0, 2 * np.pi), (0, 1), (0, 1),],
    integrality=[True, False, False, True,],
    strategy="rand1bin",
    popsize=30,
    maxiter=150,
    tol=1e-6,
    polish=True,
    updating="immediate",
    workers=1,
    rng=np.random.default_rng(0),
)


target_id = int(optimization_result.x[0])
theta_0 = (optimization_result.x[1] % (2 * np.pi))
tof_fraction = optimization_result.x[2]
retrograde = bool(int(optimization_result.x[3]))

target_time = (far_times[trajectory_id][target_id])
tof_max = (target_time - T_max * TU)
tof = (tof_min + tof_fraction * (tof_max - tof_min))
direction = (-1.0 if retrograde else 1.0)

x_target_normalized = (far_points_normalized[trajectory_id][:, target_id])
x_target_geocentric = (far_points_geocentric[trajectory_id][:, target_id])

r_target = x_target_geocentric[:3]
v_target = x_target_geocentric[3:]

r_departure = r0 * np.array([np.cos(theta_0), np.sin(theta_0), 0])
v_circular = (direction * v_circular_magnitude * np.array([-np.sin(theta_0), np.cos(theta_0), 0]))


lambert_champion = pk.lambert_problem(
    r0=r_departure,
    r1=r_target,
    tof=tof,
    mu=mu_earth,
    cw=retrograde,
    multi_revs=0,
)


v_lambert_departure = np.asarray(lambert_champion.v0[0])
v_lambert_arrival = np.asarray(lambert_champion.v1[0])

departure_time = target_time - tof
earth_departure_inertial = Transformations.CR3BP_to_inertial([-mu*d, 0, 0, 0, 0, 0], n, departure_time).ravel()

x_departure_geocentric = np.hstack((r_departure, v_lambert_departure))
x_departure_barycentric = (x_departure_geocentric + earth_departure_inertial)
x_departure_rotating = Transformations.inertial_to_CR3BP(x_departure_barycentric, n, departure_time).ravel()
x_departure_normalized = (Transformations.SV_to_CR3BP_normalized_units(x_departure_rotating, d, G, M))


print("\n=== Lambert optimizer ===")
print(f"Success:     {optimization_result.success}")
print(f"Direction:   {'Retrograde' if retrograde else 'Prograde'}")
print(f"Evaluations: {optimization_result.nfev}")
print(f"Target time: {target_time / day:.1f} days")
print(f"Theta:       {np.degrees(theta_0):.1f} deg")
print(f"ToF:         {tof / day:.1f} days")
print(f"DV:          {optimization_result.fun:.3f} km/s")




# Numerically integrate the transfer in the CR3BP
transfer_duration = tof / TU
dt_transfer = 0.001

transfer_result = integrate_one(x_departure_normalized, 0, transfer_duration, dt_transfer, events=None)

T_transfer = transfer_result[0]
X_transfer = transfer_result[1]

X_transfer_rotating = (Transformations.CR3BP_normalized_units_to_SV(X_transfer, d, G, M))
transfer_times = (departure_time + T_transfer * TU)

X_transfer_inertial = Transformations.CR3BP_to_inertial(X_transfer_rotating, n, transfer_times)
X_target_barycentric = (far_points_inertial[trajectory_id][:, target_id])

position_error = np.linalg.norm(X_transfer_inertial[:3, -1] - X_target_barycentric[:3])

print("\n=== LAMBERT TRANSFER ===")
print(f"Position error: {position_error:.3f} km")




# Refine the departure velocity using CR3BP integration and a root-finding method
initial_correction = np.array([0, 0, 0])

shooting_result = root(
    shooting_residual,
    initial_correction,
    method="hybr",
    tol=1e-12
    )

if not shooting_result.success:
    print("\nCR3BP root refinement failed: Proceeding with homotopy method.")

converged = (shooting_result.success and np.linalg.norm(shooting_result.fun) * d < 0.1)

if not converged:

    delta_v_refinement = initial_correction

    for strength in [0.1, 0.25, 0.45, 0.65, 0.85, 1.0]:

        shooting_result = root(
            homotopy_residual,
            delta_v_refinement,
            args=(strength,),
            method="hybr",
            tol=1e-10,
            options={"maxfev": 120}
        )

        if not shooting_result.success:
            raise RuntimeError(
                f"Refinement failed at strength={strength}"
            )

        delta_v_refinement = shooting_result.x

else:
    delta_v_refinement = shooting_result.x


v_refined_departure = v_lambert_departure + delta_v_refinement
x_departure_normalized_refined = corrected_departure_state(delta_v_refinement)





# Numerically integrate the refined transfer in the CR3BP
transfer_duration = tof / TU
dt_transfer = 0.001

transfer_result = integrate_one(x_departure_normalized_refined, 0, transfer_duration, dt_transfer, events=None)

T_transfer_refined = transfer_result[0]
X_transfer_refined = transfer_result[1]

X_transfer_rotating_refined = (Transformations.CR3BP_normalized_units_to_SV(X_transfer_refined, d, G, M))
transfer_times = (departure_time + T_transfer_refined * TU)

X_transfer_inertial_refined = Transformations.CR3BP_to_inertial(X_transfer_rotating_refined, n, transfer_times)

position_error_refined = np.linalg.norm(X_transfer_inertial_refined[:3, -1] - X_target_barycentric[:3])
velocity_error_refined = np.linalg.norm(X_transfer_inertial_refined[3:, -1] - X_target_barycentric[3:])

print("\n=== REFINED CR3BP TRANSFER ===")
print(f"Position error: {position_error_refined:.3f} km")




# Compute delta-v for departure and arrival
delta_v_departure_refined = np.linalg.norm(v_refined_departure - v_circular)
delta_v_arrival_refined = velocity_error_refined
delta_v_total = delta_v_departure_refined + delta_v_arrival_refined

x_departure = X_transfer_inertial_refined[:, 0]
dx_manovra = np.hstack((np.zeros(3), X_target_barycentric[3:] - X_transfer_inertial_refined[3:, -1]))

print(f"\nTime of flight:     {tof / day:.1f} days")
print(f"DV departure:       {delta_v_departure_refined:.3f} km/s")
print(f"DV transfer:        {delta_v_arrival_refined:.3f} km/s")
print(f"DV total:           {delta_v_total:.3f} km/s")




# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")




# Save the results to a text file
if SAVE:
    data = np.r_[
        departure_time,
        x_departure,
        target_time,
        dx_manovra,
        tof,
        delta_v_departure_refined,
        delta_v_arrival_refined,
        delta_v_total,
        selection[trajectory_id, columns["E_SOI"]],
        selection[trajectory_id, columns["h_min_moon"]]]

    np.savetxt(
        "Escape_transfers.txt",
        data[None, :],
        header=(
            "t_departure x_departure y_departure z_departure vx_departure vy_departure vz_departure t_manovra dx_manovra dy_manovra dz_manovra tof DV1 DV2 DVtotal ESOI h_moon"))




# Plotting the results
if PLOT:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Rotating frame
    for traj, far_point in zip(X, far_points_rotating):
        ax1.plot(traj[0] * d, traj[1] * d)
        ax1.scatter(traj[0, 0] * d, traj[1, 0] * d)


    ax1.plot(X_transfer[0] * d, X_transfer[1] * d, color="blue", linewidth=2, label="Transfer")

    ax1.scatter(X_transfer[0, 0] * d, X_transfer[1, 0] * d, color="green", s=40)

    ax1.scatter(X_transfer[0, -1] * d, X_transfer[1, -1] * d, color="blue", s=40)


    ax1.plot(X_transfer_refined[0] * d, X_transfer_refined[1] * d, color="red", linewidth=2, label="Transfer refined")
    
    ax1.scatter(X_transfer_refined[0, 0] * d, X_transfer_refined[1, 0] * d, color="green", s=40, label="Transfer departure")
    
    ax1.scatter(X_transfer_refined[0, -1] * d, X_transfer_refined[1, -1] * d, color="red", s=40, label="Transfer arrival")


    ax1.scatter(-mu*d, 0, color="blue", label="Earth")
    ax1.scatter((1-mu)*d, 0, color="darkred", label="Moon")
    ax1.scatter(far_point[0], far_point[1], color="orange", s=2, label="Target points") if MASK else None

    ax1.set_title("Rotating Frame")
    ax1.set_xlabel("X [km]")
    ax1.set_ylabel("Y [km]")
    ax1.set_aspect("equal")
    ax1.set_box_aspect(1)
    ax1.legend()


    # Barycentric inertial frame
    for time, traj, far_point in zip(T, X_bar, far_points_inertial):

        earth_I = Transformations.CR3BP_to_inertial(
            [-mu*d, 0, 0, 0, 0, 0],
            n,
            time * TU
        )

        moon_I = Transformations.CR3BP_to_inertial(
            [(1-mu)*d, 0, 0, 0, 0, 0],
            n,
            time * TU
        )

        ax2.plot(traj[0], traj[1])
        ax2.scatter(far_point[0], far_point[1], color="orange", s=2, label="Target points") if MASK else None
        ax2.scatter(traj[0, 0], traj[1, 0])

        ax2.plot(earth_I[0], earth_I[1], "--", color="blue")
        ax2.plot(moon_I[0], moon_I[1], "--", color="darkred")

    ax2.plot(X_transfer_inertial[0], X_transfer_inertial[1], color="blue", linewidth=2, label="Transfer")

    ax2.scatter(X_transfer_inertial[0, 0], X_transfer_inertial[1, 0], color="green", s=40)

    ax2.scatter(X_transfer_inertial[0, -1], X_transfer_inertial[1, -1], color="blue", s=40)

    ax2.plot(X_transfer_inertial_refined[0], X_transfer_inertial_refined[1], color="red", linewidth=2, label="Transfer refined")
    
    ax2.scatter(X_transfer_inertial_refined[0, 0], X_transfer_inertial_refined[1, 0], color="green", s=40, label="Transfer departure")
    
    ax2.scatter(X_transfer_inertial_refined[0, -1], X_transfer_inertial_refined[1, -1], color="red", s=40, label="Transfer arrival")

    ax2.scatter(-mu*d, 0, color="blue", label="Earth at $t=0$")
    ax2.scatter((1-mu)*d, 0, color="darkred", label="Moon at $t=0$")

    ax2.set_title("Barycentric Inertial Frame")
    ax2.set_xlabel("X [km]")
    ax2.set_ylabel("Y [km]")
    ax2.set_aspect("equal")
    ax2.set_box_aspect(1)
    ax2.legend()

    plt.tight_layout()
    plt.show()