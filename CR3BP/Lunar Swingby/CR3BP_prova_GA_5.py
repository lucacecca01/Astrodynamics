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
SAVE = False
PRINT = True



# Define integration function
def integrate_one(x0, T_min, T_max, dt, events=None, rtol=1e-9):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='CR3BP',
        integrator='scipy',
        method='DOP853',
        rtol=rtol,
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

    target_time_candidate = (far_times[target_id_candidate])

    direction_candidate = (-1.0 if direction else 1.0)

    tof_max_candidate = (target_time_candidate - T_max * TU)

    if tof_max_candidate <= tof_min:
        return 1e12

    tof_candidate = (tof_min + tof_fraction * (tof_max_candidate - tof_min))

    target_state = (far_points_geocentric[:, target_id_candidate])

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
        rtol=1e-10,
        atol=1e-12,
        t_eval=[transfer_duration]
    )

    if not sol.success:
        return np.full(3, 1e3)

    return sol.y[:3, -1] - x_target_normalized[:3]



# Define the main function to process each trajectory
def process_trajectory(current_row):

    global far_times
    global far_points_geocentric
    global tof_min
    global r0
    global v_circular_magnitude
    global v_lambert_departure
    global r_departure
    global earth_departure_inertial
    global departure_time
    global transfer_duration
    global x_target_normalized
    global T, X
    global far_points_rotating, far_points_inertial
    global X_transfer, X_transfer_inertial
    global X_transfer_refined, X_transfer_inertial_refined


    if current_row[columns["h_min_moon"]] < 0 or current_row[columns["h_last_earth"]] < 0:
        return None


    result = integrate_one(
        current_row[:6],
        T_min,
        T_max,
        dt,
        earth_SOI_exit,
        rtol=1e-9
    )

    t = result[0]
    traj = result[1]


    if MASK:
    
        r_moon = traj[:3].T - np.array([1 - mu, 0, 0])
        r_earth = traj[:3].T - np.array([-mu, 0, 0])

        r_m = np.linalg.norm(r_moon, axis=1)
        r_e = np.linalg.norm(r_earth, axis=1)

        a_earth = -(1 - mu) * r_earth / r_e[:, None]**3
        a_moon = -mu * r_moon / r_m[:, None]**3

        a_moon_earth = np.array([mu, 0, 0])
        a_moon_perturbation = a_moon - a_moon_earth

        perturbation = eta = (np.linalg.norm(a_moon_perturbation, axis=1)/ np.linalg.norm(a_earth, axis=1))
    
        mask = np.logical_and(eta <= 0.01, r_e < 1)

    else:
        mask = np.ones(traj.shape[1], dtype=bool)
        

    far_times = t[mask] * TU

    earth_inertial = Transformations.CR3BP_to_inertial([-mu*d, 0, 0, 0, 0, 0], n, far_times)

    far_points_normalized = traj[:, mask]
    far_points_rotating = Transformations.CR3BP_normalized_units_to_SV(traj[:, mask], d, G, M)
    far_points_inertial = Transformations.CR3BP_to_inertial(far_points_rotating, n, t[mask] * TU)
    far_points_geocentric = far_points_inertial - earth_inertial

    


    # Lambert's problem and optimization to find the best transfer trajectory
    number_of_targets = len(far_times)

    if number_of_targets == 0:
        print("No target points available. Skipping trajectory.")
        return None

    r0 = R_E + 200
    tof_min = 0.5 * day

    v_circular_magnitude = np.sqrt(mu_earth / r0)

    optimization_result = differential_evolution(
        lambert_objective,
        bounds=[(0, number_of_targets - 1), (0, 2 * np.pi), (0, 1), (0, 1),],
        integrality=[True, False, False, True,],
        strategy="rand1bin",
        popsize=20,
        maxiter=100,
        tol=1e-4,
        polish=True,
        updating="immediate",
        workers=1,
        rng=np.random.default_rng(0),
    )


    if (not np.isfinite(optimization_result.fun) or optimization_result.fun >= 1e12):
        print("No feasible Lambert transfer. Skipping trajectory.")
        return None


    target_id = int(optimization_result.x[0])
    theta_0 = (optimization_result.x[1] % (2 * np.pi))
    tof_fraction = optimization_result.x[2]
    retrograde = bool(int(optimization_result.x[3]))

    target_time = (far_times[target_id])
    tof_max = (target_time - T_max * TU)
    tof = (tof_min + tof_fraction * (tof_max - tof_min))
    direction = (-1.0 if retrograde else 1.0)

    x_target_normalized = (far_points_normalized[:, target_id])
    X_target_barycentric = far_points_inertial[:, target_id]
    x_target_geocentric = (far_points_geocentric[:, target_id])

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


    if PRINT:
        print("\n=== Lambert optimizer ===")
        print(f"Success:     {optimization_result.success}")
        print(f"Direction:   {'Retrograde' if retrograde else 'Prograde'}")
        print(f"Evaluations: {optimization_result.nfev}")
        print(f"Target time: {target_time / day:.1f} days")
        print(f"Theta:       {np.degrees(theta_0):.1f} deg")
        print(f"ToF:         {tof / day:.1f} days")
        print(f"DV:          {optimization_result.fun:.3f} km/s")




    # Numerically integrate the transfer in the CR3BP
    if PRINT or PLOT:
        transfer_duration = tof / TU
        dt_transfer = 0.001

        transfer_result = integrate_one(x_departure_normalized, 0, transfer_duration, dt_transfer, events=None)

        T_transfer = transfer_result[0]
        X_transfer = transfer_result[1]

        transfer_times = (departure_time + T_transfer * TU)

        X_transfer_rotating = (Transformations.CR3BP_normalized_units_to_SV(X_transfer, d, G, M))
        X_transfer_inertial = Transformations.CR3BP_to_inertial(X_transfer_rotating, n, transfer_times)

        position_error = np.linalg.norm(X_transfer_inertial[:3, -1] - X_target_barycentric[:3])


        if PRINT:
            print("\n=== LAMBERT TRANSFER ===")
            print(f"Position error: {position_error:.3f} km")




    # Refine the departure velocity using CR3BP integration and a root-finding method
    initial_correction = np.array([0, 0, 0])
    transfer_duration = tof / TU

    shooting_result = root(
        shooting_residual,
        initial_correction,
        method="hybr",
        tol=1e-10
        )

    residual_error = np.linalg.norm(shooting_result.fun) * d

    converged = (np.isfinite(residual_error) and residual_error < 0.1)

    if not converged:

        if PRINT:
            print("\nCR3BP root refinement failed: Proceeding with homotopy method.")

        delta_v_refinement = initial_correction
        refinement_failed = False

        for strength in [0.02, 0.05, 0.1, 0.25, 0.45, 0.65, 0.85, 1.0]:

            shooting_result = root(
                homotopy_residual,
                delta_v_refinement,
                args=(strength,),
                method="hybr",
                tol=1e-10,
                options={"maxfev": 300}
            )

            residual_error = np.linalg.norm(shooting_result.fun) * d

            if not residual_error < 0.1:
                print(f"Refinement failed at strength={strength}: "f"residual={residual_error:.3f} km")
                refinement_failed = True
                break

            delta_v_refinement = shooting_result.x

        if refinement_failed:
            return None

    else:
        delta_v_refinement = shooting_result.x


    v_refined_departure = v_lambert_departure + delta_v_refinement
    x_departure_normalized_refined = corrected_departure_state(delta_v_refinement)





    # Numerically integrate the refined transfer in the CR3BP
    transfer_duration = tof / TU
    dt_transfer = 0.001

    transfer_result = integrate_one(x_departure_normalized_refined, 0, transfer_duration, dt_transfer, events=None, rtol=1e-10)

    T_transfer_refined = transfer_result[0]
    X_transfer_refined = transfer_result[1]

    X_transfer_rotating_refined = (Transformations.CR3BP_normalized_units_to_SV(X_transfer_refined, d, G, M))
    transfer_times = (departure_time + T_transfer_refined * TU)

    X_transfer_inertial_refined = Transformations.CR3BP_to_inertial(X_transfer_rotating_refined, n, transfer_times)

    position_error_refined = np.linalg.norm(X_transfer_inertial_refined[:3, -1] - X_target_barycentric[:3])
    velocity_error_refined = np.linalg.norm(X_transfer_inertial_refined[3:, -1] - X_target_barycentric[3:])

    if not np.isfinite(position_error_refined) or position_error_refined >= 0.1:
        print(f"Final refinement error too large: {position_error_refined:.3f} km")
        return None


    if PRINT:
        print("\n=== REFINED CR3BP TRANSFER ===")
        print(f"Position error: {position_error_refined:.3f} km")




    # Compute delta-v for departure and arrival
    delta_v_departure_refined = np.linalg.norm(v_refined_departure - v_circular)
    delta_v_arrival_refined = velocity_error_refined
    delta_v_total = delta_v_departure_refined + delta_v_arrival_refined

    x_departure = X_transfer_inertial_refined[:, 0]
    dx_manovra = np.hstack((np.zeros(3), X_target_barycentric[3:] - X_transfer_inertial_refined[3:, -1]))



    # Avoiding Collisions
    r_earth_transfer = np.linalg.norm(X_transfer_refined[:3].T - np.array([-mu, 0, 0]), axis=1)
    r_moon_transfer = np.linalg.norm(X_transfer_refined[:3].T - np.array([1 - mu, 0, 0]), axis=1)

    h_earth_transfer = r_earth_transfer.min() * d - R_E
    h_moon_transfer = r_moon_transfer.min() * d - R_L

    if h_earth_transfer < 0 or h_moon_transfer < 0: 
        print("Transfer collision. "f"h_E={h_earth_transfer:.3f} km, "f"h_M={h_moon_transfer:.3f} km")
        return None


    if PLOT:
        T.append(t)
        X.append(traj)


    if PRINT:
            print(f"\nTime of flight:     {tof / day:.1f} days")
            print(f"DV departure:       {delta_v_departure_refined:.3f} km/s")
            print(f"DV transfer:        {delta_v_arrival_refined:.3f} km/s")
            print(f"DV total:           {delta_v_total:.3f} km/s")


    return np.r_[
        departure_time,
        x_departure,
        target_time,
        dx_manovra,
        tof,
        delta_v_departure_refined,
        delta_v_arrival_refined,
        delta_v_total,
        current_row[columns["E_SOI"]],
        current_row[columns["h_min_moon"]]
    ]



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
output_rows = []


# Define masses and integration parameters
masses = [m1, m2, 0]
T_min = 0
T_max = -4 * np.pi
dt = -0.001



# Define distances and velocities for circular orbit and Lambert
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
print("\n")


order = np.argsort(database[:, columns[SORT_BY]], kind="stable")
selection = database[order[::-1] if DESCENDING else order]

source_ids = np.array([4508, 54212, 35554, 99374, 98436])

selected = selection[10000:10001]
selected = selection[:1]
selected = database[source_ids[4:5]]




# Use multiprocessing to integrate multiple trajectories in parallel
if PLOT:

    results = [process_trajectory(current_row) for current_row in selected]

else:

    with get_context("fork").Pool() as pool:

        results = pool.map(
            process_trajectory,
            selected,
            chunksize=1
        )


output_rows = [result for result in results if result is not None]

valid_initial_conditions = np.count_nonzero((selected[:, columns["h_min_moon"]] >= 0) & (selected[:, columns["h_last_earth"]] >= 0))
success_percentage = (100 * len(output_rows) / valid_initial_conditions if valid_initial_conditions > 0 else 0)
print(f"\nSuccessful transfers: {len(output_rows)}/"f"{valid_initial_conditions} ({success_percentage:.1f}%)")



# Print execution time
elapsed_time = time.perf_counter() - start_time
print(f"\nExecution time: {elapsed_time:.2f} s\n")




# Save the results to a text file
if SAVE:

    if output_rows:
    
        data = np.vstack(output_rows)

        np.savetxt(
            "/home/lucacecca/Astrodynamics/CR3BP/Lunar Swingby/Escape_transfers_database.txt",
            data,
            header=(
                "t_departure_s "
                "x_departure_km y_departure_km z_departure_km "
                "vx_departure_km_s vy_departure_km_s vz_departure_km_s "
                "t_maneuver_s "
                "dx_maneuver_km dy_maneuver_km dz_maneuver_km "
                "dvx_maneuver_km_s dvy_maneuver_km_s dvz_maneuver_km_s "
                "tof_s DV1_km_s DV2_km_s DVtotal_km_s "
                "E_SOI_km2_s2 h_moon_km"
            )
        )

    else:
        print("No valid transfers to save.")


# Plotting the results
if PLOT:

    X_bar = [Transformations.CR3BP_to_inertial(Transformations.CR3BP_normalized_units_to_SV(traj, d, G, M), n, time * TU) for time, traj in zip(T, X)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Rotating frame
    for traj in X:
        ax1.plot(traj[0] * d, traj[1] * d)
        ax1.scatter(traj[0, 0] * d, traj[1, 0] * d)


    ax1.plot(X_transfer[0] * d, X_transfer[1] * d, color="blue", linewidth=2, label="Transfer")

    ax1.scatter(X_transfer[0, 0] * d, X_transfer[1, 0] * d, color="green", s=40)

    ax1.scatter(X_transfer[0, -1] * d, X_transfer[1, -1] * d, color="blue", s=40)

    if MASK:
        ax1.scatter(far_points_rotating[0], far_points_rotating[1], color="orange", s=2, label="Target points")

    ax1.plot(X_transfer_refined[0] * d, X_transfer_refined[1] * d, color="red", linewidth=2, label="Transfer refined")
    
    ax1.scatter(X_transfer_refined[0, 0] * d, X_transfer_refined[1, 0] * d, color="green", s=40, label="Transfer departure")
    
    ax1.scatter(X_transfer_refined[0, -1] * d, X_transfer_refined[1, -1] * d, color="red", s=40, label="Transfer arrival")


    ax1.scatter(-mu*d, 0, color="blue", label="Earth")
    ax1.scatter((1-mu)*d, 0, color="darkred", label="Moon")

    ax1.set_title("Rotating Frame")
    ax1.set_xlabel("X [km]")
    ax1.set_ylabel("Y [km]")
    ax1.set_aspect("equal")
    ax1.set_box_aspect(1)
    ax1.legend()


    # Barycentric inertial frame
    for time, traj in zip(T, X_bar):

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

    if MASK:
        ax2.scatter(far_points_inertial[0], far_points_inertial[1], color="orange", s=2, label="Target points")

    ax2.set_title("Barycentric Inertial Frame")
    ax2.set_xlabel("X [km]")
    ax2.set_ylabel("Y [km]")
    ax2.set_aspect("equal")
    ax2.set_box_aspect(1)
    ax2.legend()

    plt.tight_layout()
    plt.show()