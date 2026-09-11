from multiprocessing import get_context

from numpy.linalg import norm
from Celestial_Mechanics import Transformations
from Celestial_Mechanics import Integration
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
import time

start_time = time.perf_counter()

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



mu_s = ms / M
r_s = d_E / d
n_s = np.sqrt(G * (ms + M) / d_E**3)
omega_s = n_s / n - 1


# Define Spheres of Influence
M_SOI = (m2/m1)**(2/5) * d
E_SOI = (m1/ms)**(2/5) * d_E 
SOI = ((m1 + m2)/ms)**(2/5) * (d_E + mu * d)



# Define Dimension Space
dim = '2D'
BYPASS_I = False
BYPASS_F = False
RETROGRADE = True
SAVE = True
PLOT = True




# Sample from sphere surface
def sample_on_ipersphere(Rmin, Rmax, dim, n_samples_r, n_samples_theta, offset=None):

    if offset is None:
        offset = [0, 0, 0]

    theta = np.linspace(0, 2*np.pi, n_samples_theta, endpoint=False)
    r = np.linspace(Rmin, Rmax, n_samples_r)

    if dim == '3D':

        if Rmin == Rmax:
            r = Rmin

        cos_phi = np.random.uniform(-1, 1, n_samples_theta)

        r, theta, cos_phi = np.meshgrid(r, theta, cos_phi)

        sin_phi = np.sqrt(1 - cos_phi**2)

        x = r * np.cos(theta) * sin_phi + offset[0]
        y = r * np.sin(theta) * sin_phi + offset[1]
        z = r * cos_phi + offset[2]

        return np.column_stack((x.flatten(), y.flatten(), z))

    elif dim == '2D':

        if Rmin == Rmax:
            r = Rmin

        r, theta = np.meshgrid(r, theta)

        x = r * np.cos(theta) + offset[0]
        y = r * np.sin(theta) + offset[1]
        z = np.full(x.size, offset[2])

        return np.column_stack((x.flatten(), y.flatten(), z))



# Define the EDT velocity function
def ETD_velocity(pos, Cj, v_norm):

    r = np.linalg.norm(pos)

    sin_delta = (Cj / 2 - r**2) / (v_norm * r)

    delta_1 = np.arcsin(sin_delta)
    delta_2 = np.pi - np.arcsin(sin_delta)

    r_hat = pos / r 

    theta = np.arctan2(r_hat[1], r_hat[0])

    vx_1 = v_norm * np.cos(delta_1 + theta)
    vy_1 = v_norm * np.sin(delta_1 + theta)

    vx_2 = v_norm * np.cos(delta_2 + theta)
    vy_2 = v_norm * np.sin(delta_2 + theta)

    return np.array([vx_1, vy_1, 0]), np.array([vx_2, vy_2, 0])



# Define the perilune velocity function
def perilune_velocity(pos, v_norm, mu):

    r_rel = pos - np.array([1 - mu, 0, 0])

    theta = np.arctan2(r_rel[1], r_rel[0])
    alpha_1 = theta + np.pi / 2
    alpha_2 = theta - np.pi / 2

    vx_1 = v_norm * np.cos(alpha_1)
    vy_1 = v_norm * np.sin(alpha_1)

    vx_2 = v_norm * np.cos(alpha_2)
    vy_2 = v_norm * np.sin(alpha_2)

    return np.array([vx_1, vy_1, 0]), np.array([vx_2, vy_2, 0])



# Define collision function
def min_distance(t, x, mu, *args):

    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
    r_moon = np.sqrt((x[0] - (1 - mu))**2 + x[1]**2 + x[2]**2)

    return min(r_earth - R_E / d, r_moon - R_L / d)



# Define moon SOI exit function
def moon_SOI(t, x, mu, *args): 
    r_moon = np.sqrt((x[0] - (1 - mu))**2 + x[1]**2 + x[2]**2)
    return M_SOI / d - r_moon



# Define moon encouter function
def moon_encounter(t, x, mu, *args): 
    r_moon = np.sqrt((x[0] - (1 - mu))**2 + x[1]**2 + x[2]**2)
    return M_SOI / d - r_moon



# Define earth SOI exit function
def earth_SOI(t, x, mu, *args): 
    r_earth = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
    return E_SOI / d - r_earth



# Define 2 Body Energy
def two_body_energy(r, v, mu, body=2, frame='rotating'):

    r = np.asarray(r)
    v = np.asarray(v)

    off1 = np.array([-mu, 0, 0])
    off2 = np.array([1 - mu, 0, 0])

    offsets = [off1, off2]
    mu_bodies = [1 - mu, mu]

    offset = offsets[body - 1][:, None] if r.ndim == 2 else offsets[body - 1]
    r_rel = r - offset

    if frame == 'rotating':
        omega = np.array([0, 0, 1])
        v_rel = v + np.cross(omega, r_rel, axis=0)
    else:
        v_rel = v

    return 0.5 * np.sum(v_rel**2, axis=0) - mu_bodies[body - 1] / np.linalg.norm(r_rel, axis=0)



# Define 2 Body Energy Derivative
def two_body_energy_derivative(r, v, mu, body=2, frame='rotating'):

    r = np.asarray(r) 
    v = np.asarray(v)

    off1 = np.array([-mu, 0, 0])
    off2 = np.array([1 - mu, 0, 0])

    offsets = [off1, off2]
    mu_bodies = [1 - mu, mu]

    r_rel = r - offsets[body - 1]
    mu_body = mu_bodies[body - 1]

    r1 = np.linalg.norm(r - off1)
    r2 = np.linalg.norm(r - off2)

    ax = 2 * v[1]  +  r[0]  -  (1 - mu) * (r[0] + mu) / r1**3  -  mu * (r[0] - (1 - mu)) / r2**3
    ay = -2 * v[0]  +  r[1]  -  (1 - mu) * r[1] / r1**3 - mu * r[1] / r2**3
    az = - (1 - mu) * r[2] / r1**3 - mu * r[2] / r2**3

    a_rot = np.array([ax, ay, az])

    if frame == 'rotating':

        omega = np.array([0, 0, 1])
        v_rel = v + np.cross(omega, r_rel)

        a_rel = (a_rot + 2 * np.cross(omega, v) + np.cross(omega, np.cross(omega, r)))
        a_body = np.cross(omega, np.cross(omega, offsets[body - 1]))

        a_rel = a_rel - a_body

    else:
        v_rel = v
        a_rel = a_rot


    return np.dot(v_rel, a_rel) + mu_body / np.linalg.norm(r_rel)**3 * np.dot(r_rel, v_rel)



# Define 3 Body Energy
def baricentric_energy(r, v, mu, frame='rotating'):

    r = np.asarray(r)
    v = np.asarray(v)

    off1 = np.array([-mu, 0, 0]).reshape(3, *([1] * (r.ndim - 1)))
    off2 = np.array([1 - mu, 0, 0]).reshape(3, *([1] * (r.ndim - 1)))

    r1 = r - off1
    r2 = r - off2

    if frame == 'rotating':
        omega = np.array([0, 0, 1])
        v_rel = v + np.cross(omega, r, axis=0)
    else:
        v_rel = v

    return 0.5 * np.sum(v_rel**2, axis=0) - (1 - mu) / np.linalg.norm(r1, axis=0) - mu / np.linalg.norm(r2, axis=0)



# Define Baricentric Energy Derivative
def baricentric_energy_derivative(r, v, mu, frame='rotating', t=0, theta_s_0=0):

    r = np.asarray(r) 

    off1 = np.array([-mu, 0, 0]).reshape(3, *([1] * (r.ndim - 1)))
    off2 = np.array([1 - mu, 0, 0]).reshape(3, *([1] * (r.ndim - 1)))

    r1 = np.linalg.norm(r - off1)
    r2 = np.linalg.norm(r - off2)

    eps = mu * (1 - mu) * r[1] * (1/r1**3 - 1/r2**3)

    theta_s = omega_s * t + theta_s_0
    sun_pos = r_s * np.array([np.cos(theta_s), np.sin(theta_s), 0])

    delta = sun_pos - r
    a_sun = mu_s * (delta / np.linalg.norm(delta)**3 - sun_pos / r_s**3)

    v_nonrot = v + np.cross([0, 0, 1], r)

    return eps + np.dot(v_nonrot, a_sun)



# Define Lunar Radial Velocity
def lunar_radial_velocity(t, x, mu, *args):

    r_dot = (x[0] - (1 - mu))*x[3] + x[1]*x[4] + x[2]*x[5]

    return r_dot



# Define Earth Radial Velocity
def earth_radial_velocity(t, x, mu, *args):

    r_dot = (x[0] + mu)*x[3] + x[1]*x[4] + x[2]*x[5]

    return r_dot



# Define Earth Perigee Function
def earth_perigee(t, x, mu, *args):

    r_earth = np.linalg.norm(x[:3] - np.array([-mu, 0, 0]))
    radial_velocity = earth_radial_velocity(t, x, mu)
    eps_earth = two_body_energy(x[:3], x[3:], mu, body=1, frame='rotating')

    h_perigee_filter = r_earth - (R_E + 800) / d

    if filters["escape"]:      
        return max(radial_velocity, h_perigee_filter)
        
    elif filters["capture"]:                      
        return min(radial_velocity, -h_perigee_filter)



# Define Lunar Revolution Angle
def lunar_revolution_angle(t, x, mu, *args):

    x_inertial = Transformations.CR3BP_to_inertial(x, 1, t)
    x_moon = Transformations.CR3BP_to_inertial([(1 - mu), 0, 0, 0, 0, 0], 1, t)

    x_moon_rel = x_inertial - x_moon

    rev_angle = np.unwrap(np.arctan2(x_moon_rel[1], x_moon_rel[0]))

    return rev_angle



# Define Earth Revolution Angle
def earth_revolution_angle(t, x, mu, *args):

    x_inertial = Transformations.CR3BP_to_inertial(x, 1, t)
    x_earth = Transformations.CR3BP_to_inertial([-mu, 0, 0, 0, 0, 0], 1, t)

    x_earth_rel = x_inertial - x_earth

    rev_angle = np.unwrap(np.arctan2(x_earth_rel[1], x_earth_rel[0]))

    return rev_angle



# Define Moon Capture Function
def moon_capture(t, x, mu, *args):

    return two_body_energy(x[:3], x[3:], mu, body=2, frame='rotating')



# Define Jacobi Constant Range Function
def Jacobi_range(rp_min, rp_max, mu, rot=None):

    if rot is None:
        return None

    if rot == "prograde":
        Cj_min_1 = np.sqrt(8*rp_min * (1 - mu)) + 2 * mu
        Cj_min_2 = np.sqrt(8*rp_max * (1 - mu)) + 2 * mu

        Cj_max_1 = 2 * (1 - mu) / (rp_min + 1) + np.sqrt(8 * (1 - mu) * rp_min / (rp_min + 1)) + 2 * mu
        Cj_max_2 = 2 * (1 - mu) / (rp_max + 1) + np.sqrt(8 * (1 - mu) * rp_max / (rp_max + 1)) + 2 * mu

        return min(Cj_min_1, Cj_min_2), max(Cj_max_1, Cj_max_2)
    

    elif rot == "retrograde":
        Cj_min_1 = - np.sqrt(8*rp_min * (1 - mu)) + 2 * mu
        Cj_min_2 = - np.sqrt(8*rp_max * (1 - mu)) + 2 * mu
    
        Cj_max_1 = 2 * (1 - mu) / (rp_min + 1) - np.sqrt(8 * (1 - mu) * rp_min / (rp_min + 1)) + 2 * mu
        Cj_max_2 = 2 * (1 - mu) / (rp_max + 1) - np.sqrt(8 * (1 - mu) * rp_max / (rp_max + 1)) + 2 * mu
    
    return min(Cj_min_1, Cj_min_2), max(Cj_max_1, Cj_max_2)
    


# Define the initial conditions filters
def accept_initial_condition(pos, v, mu, filters, Cj):

    conditions = []

    if filters["lunar_flyby"]:
        conditions.append(two_body_energy(pos, v, mu, body=2, frame='rotating') > 0)

    if filters["ETD"]:
        conditions.append(np.abs(baricentric_energy(pos, v, mu, frame='rotating')) < 1e-12)

        if filters["escape"]:
            conditions.append(baricentric_energy_derivative(pos, v, mu, frame='rotating') > 0)
        
        if filters["capture"]:
            conditions.append(baricentric_energy_derivative(pos, v, mu, frame='rotating') < 0)

    if filters["perilune"]:
        x = np.concatenate((pos, v))
        conditions.append(abs(lunar_radial_velocity(0, x, mu)) < 10e-12)

    return all(conditions)



# Define the final conditions filters
def accept_final_condition(pos, v, mu, filters, sol=None, SOI_exit_index=None, rev_n=1):

    conditions = []

    if filters["escape"]:
        if len(sol[3][2]) == 0:
            return False

        x_soi = sol[3][2][-1]

        conditions.append(baricentric_energy(x_soi[:3], x_soi[3:], mu) > 0)
        conditions.append(earth_radial_velocity(0, x_soi, mu) > 0)


    if filters["capture"]:
        if len(sol[3][2]) == 0:
            return False
        
        x_perigee = sol[3][2][-1]
        earth_distance = np.linalg.norm(x_perigee[:3] - np.array([-mu, 0, 0]))
        r_rel = x_perigee[:3] - np.array([-mu, 0, 0])
        v_rel = x_perigee[3:] + np.cross([0, 0, 1], r_rel)

        conditions.append(two_body_energy(x_perigee[:3], x_perigee[3:], mu, body=1, frame='rotating') < 0)
        conditions.append(earth_distance > (200 + R_E) / d and earth_distance < (800 + R_E) / d)

        if filters["prograde"]:
            conditions.append(np.cross(r_rel, v_rel)[2] > 0)
        
        if filters["retrograde"]:
            conditions.append(np.cross(r_rel, v_rel)[2] < 0)


    if filters["lunar_flyby"]:
        if SOI_exit_index is None:
            return False

        x_moon_exit = sol[3][1][-1]
        x_end = sol[3][2][-1]
        
        moon_energy_SOI_exit = two_body_energy(x_moon_exit[:3], x_moon_exit[3:], mu, body=2, frame='rotating')
        moon_final_distance = np.linalg.norm(x_end[:3] - np.array([(1 - mu), 0, 0]))

        conditions.append(moon_final_distance > M_SOI / d)
        conditions.append(moon_energy_SOI_exit > 0)


    if filters["significant_first_flyby"]:
        if len(sol[3][1]) == 0:
            return False
        
        x_in  = sol[1][:, 0]
        x_out = sol[3][1][0]

        E_in  = two_body_energy(x_in[:3],  x_in[3:],  mu, body=1, frame="rotating")
        E_out = two_body_energy(x_out[:3], x_out[3:], mu, body=1, frame="rotating")

        relative_kick = abs(E_out - E_in) / abs(E_in)
        conditions.append(relative_kick > flyby_tol)


    if filters["maximum_flybys"]:

        conditions.append(len(sol[3][1]) == flyby_n)


    if filters["maximum_revolutions"]:
        if SOI_exit_index is None:
            return False

        theta_earth_SOI_exit = earth_revolution_angle(sol[0][:SOI_exit_index + 1], sol[1][:, :SOI_exit_index + 1], mu)
        conditions.append(max(abs(theta_earth_SOI_exit - theta_earth_SOI_exit[0])) < rev_n * 2*np.pi)


    if filters["no_collision"]:
        conditions.append(len(sol[2][0]) == 0)


    return all(conditions)



# Define the retrograde conditions filters
def accept_retro_condition(pos, v, mu, filters, sol=None, SOI_exit_index=None, rev_n=1):

    conditions = []

    if filters["escape"]:
        if len(sol[3][2]) == 0:
            return False

        x_perigee = sol[3][2][-1]
        r_rel = x_perigee[:3] - np.array([-mu, 0, 0])
        v_rel = x_perigee[3:] + np.cross([0, 0, 1], r_rel)

        earth_distance = np.linalg.norm(x_perigee[:3] - np.array([-mu, 0, 0]))
        theta_earth = earth_revolution_angle(sol[0], sol[1], mu)

        eps_earth = two_body_energy(x_perigee[:3], x_perigee[3:],mu, body=1, frame="rotating")
        a_earth = -(1 - mu) / (2 * eps_earth)
        r_apogee = 2 * a_earth - earth_distance


        conditions.append(two_body_energy(x_perigee[:3], x_perigee[3:], mu, body=1, frame='rotating') < 0)
        #conditions.append(earth_distance > (200 + R_E) / d and earth_distance < (800 + R_E) / d)
        conditions.append(r_apogee < E_SOI / d)
        conditions.append(len(sol[2][1]) == 0)

        if filters["maximum_revolutions"]:
            conditions.append(max(abs(theta_earth - theta_earth[0])) < 2*np.pi)

        if filters["prograde"]:
            conditions.append(np.cross(r_rel, v_rel)[2] > 0)

        if filters["retrograde"]:
            conditions.append(np.cross(r_rel, v_rel)[2] < 0)
    

    if filters["capture"]:
        if len(sol[3][2]) == 0:
            return False

        x_soi = sol[3][2][-1]
        
        conditions.append(baricentric_energy(x_soi[:3], x_soi[3:], mu, frame='rotating') > 0)
        conditions.append(earth_radial_velocity(0, x_soi, mu) < 0)


    if filters["no_collision"]:
        conditions.append(len(sol[2][0]) == 0)

    return all(conditions)



# Integration function
def integrate_one(x0, T_min, T_max, dt, events=(min_distance, moon_SOI), filter_function=accept_final_condition):

    sol = Integrator.Integrator(
        masses, G, x0, T_min, T_max, dt,
        model='BCR4BP',
        integrator='scipy',
        method='DOP853',
        rtol=1e-9,
        atol=1e-12,
        events=events,
        r_s=r_s,
        theta_s_0=0
    )

    SOI_exit_index = None

    if len(sol[2][1]) > 0:
        t_exits = sol[2][1]
        last_exit = t_exits[-1]
        SOI_exit_index = np.argmin(np.abs(sol[0] - last_exit))

    collided = len(sol[2][0]) > 0


    if (filter_function(sol[1][:3, :], sol[1][3:, :], mu, filters, sol=sol, SOI_exit_index=SOI_exit_index, rev_n=rev_n) or BYPASS_F):

        trajectory = Transformations.CR3BP_normalized_units_to_SV(sol[1], d, G, M)

        moon_trajectory = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], n, sol[0]* TU)

        earth_trajectory = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], n, sol[0] * TU)

        bar_inertial = Transformations.CR3BP_to_inertial(trajectory, n, sol[0] * TU)

        e_energy = two_body_energy(sol[1][:3], sol[1][3:], mu, body=1, frame='rotating')

        m_energy = two_body_energy(sol[1][:3], sol[1][3:], mu, body=2, frame='rotating')

        b_energy = baricentric_energy(sol[1][:3], sol[1][3:], mu, frame='rotating')

        Jacobi_constant = Integrator.CR3BP_Cj_from_v(sol[1], mu)

        return (sol, trajectory, bar_inertial, moon_trajectory, earth_trajectory, e_energy, m_energy, b_energy, Jacobi_constant), collided

    else:
        return None, collided



# Conditions to apply
filters = {
    "escape": True,
    "capture": False,
    "ETD": True,
    "perilune": False,
    "lunar_flyby": True,
    "maximum_flybys": False,
    "no_collision": True,
    "significant_first_flyby": False,
    "maximum_revolutions": False,
    "prograde": False,
    "retrograde": False
}




flyby_colors = [
    "#6A00A8",  # 1: viola
    "#00B4D8",  # 2: ciano
    "#FF8C00",  # 3: arancione
    "#2DC653",  # 4: verde
    "#FF006E",  # 5: magenta
    "#FFD60A",  # 6: giallo
    "#FF1FFF",  # 7: fucsia
    "#000DFF",  # 8: blu
    "#5F2106",  # 9: marrone
    "#000000",  # 10: nero
]




# Define integrator and event functions
Integrator = Integration()

min_distance.terminal = True
min_distance.direction = -1

moon_SOI.terminal = False
moon_SOI.direction = -1

moon_encounter.terminal = False
moon_encounter.direction = -1

earth_SOI.terminal = True
earth_SOI.direction = -1

lunar_radial_velocity.terminal = False
lunar_radial_velocity.direction = 0


earth_radial_velocity.terminal = False

if filters["escape"]:
    earth_radial_velocity.direction = -1
else:
    earth_radial_velocity.direction = 1


earth_perigee.terminal = True

if filters["escape"]:
    earth_perigee.direction = -1
else:
    earth_perigee.direction = 1



# Define Lists
all_initial_conditions = []
times = []
SV = []
SV_inertial = []
moon_inertial = []
earth_inertial = []
moon_energy = []
earth_energy = []
bar_energy = []
Cj = []
x0_retro = []
forward_results = []
retro_results = []
transition_indices = []
rp = []
delta_E = []
Cj_cicle = []
flyby_counts = []
perigee = []
final_energy = []
encounters = []
delta_v = []
accepted_initial_conditions = []
rp_m = []


# Define integration initial conditions
Cj_max = 3.2
Cj_min = -3
Cj_samples = 10
cmap_1 = plt.cm.viridis
cmap_2 = plt.cm.Spectral_r
cmap_4 = plt.cm.plasma_r


rp_min = (R_E + 200) / d
rp_max = (R_E + 800) / d




if False:
    if filters["prograde"]:
        Cj_min, Cj_max = Jacobi_range(rp_min, rp_max, mu, rot="prograde")

    elif filters["retrograde"]:
        Cj_min, Cj_max = Jacobi_range(rp_min, rp_max, mu, rot="retrograde")




# Define integration quantities and counters
masses = [m1, m2, ms]
T_min = 0
total = 0
forward_collisions = 0
retro_collisions = 0
n_samples_r = 100
n_samples_theta = 360
rev_n = 1
flyby_n = 3
flyby_tol = 0.1


# Define integration time and step size
if filters["escape"]:
    T_max = 2 * np.pi
    T_retro = -4 * np.pi
    dt = 0.1
    dt_retro = -0.1
elif filters["capture"]:
    T_max = 4 * np.pi
    T_retro = -2 * np.pi
    dt = 0.001
    dt_retro = -0.01



#Define a sphere of radius M_SOI as initial position                        
initial_positions = sample_on_ipersphere((1 + 1e-12) * R_L / d, (1 - 1e-12) * M_SOI / d, dim, n_samples_r, n_samples_theta, [(1-mu), 0, 0])



# Create a multiprocessing pool using the "fork" context
pool = get_context("fork").Pool()



# Iterate over Jacobi constants
for Cj_norm in np.arange(Cj_min, Cj_max, 0.1):


    initial_conditions = []


    # Define initial velocities
    for pos in initial_positions:
        try:
            v_norm = Integrator.CR3BP_v_from_Cj(Cj_norm, pos, mu)
        except ValueError: 
            continue

        r = np.linalg.norm(pos)

        if filters["ETD"]:
            if np.abs((Cj_norm / 2 - r**2) / (v_norm * r)) > 1:
                continue
            v_set = ETD_velocity(pos, Cj_norm, v_norm)

        elif filters["perilune"]:
            v_set = perilune_velocity(pos, v_norm, mu)

        for v in v_set:
            if accept_initial_condition(pos, v, mu, filters, Cj_norm) or BYPASS_I:
                initial_conditions.append(np.concatenate((pos, v)))



    # Initialize initial conditions
    if len(initial_conditions) == 0:
        continue

    all_initial_conditions.extend(initial_conditions)
    total += len(initial_conditions)
    initial_conditions = np.array(initial_conditions)



    # Integration
    if filters["escape"]:

        # Integrate retrograde trajectories
        if RETROGRADE:
          
            retro_res = pool.starmap(integrate_one, ((x0, T_min, T_retro, dt_retro, (min_distance, earth_SOI, earth_radial_velocity, moon_SOI, moon_encounter, lunar_radial_velocity), accept_retro_condition) for x0 in initial_conditions), chunksize=1) 
            retro_results = [result[0] for result in retro_res if result[0] is not None]
            x0_forward = [result[0][1][:, 0] for result in retro_results]

        else:
            retro_res = []
            retro_results = [None] * len(initial_conditions)
            x0_forward = initial_conditions

        # Integrate prograde trajectories
        total_sol = pool.starmap(integrate_one, ((x0, T_min, T_max, dt, (min_distance, moon_SOI, earth_SOI, moon_encounter, lunar_radial_velocity), accept_final_condition) for x0 in x0_forward), chunksize=1)

        # Store the results
        retro_collisions += sum(result[1] for result in retro_res)
        forward_collisions += sum(result[1] for result in total_sol)
        forward_results = [result[0] for result in total_sol]


    # Integration
    elif filters["capture"]:

        # Integrate prograde trajectories
        total_sol = pool.starmap(integrate_one, ((x0, T_min, T_max, dt, (min_distance, moon_SOI, earth_perigee, moon_encounter, lunar_radial_velocity), accept_final_condition) for x0 in initial_conditions), chunksize=1)

        # Fill the lists with the accepted trajectories
        forward_results = [result[0] for result in total_sol   if result[0] is not None]

        # Integrate retrograde trajectories
        if RETROGRADE:

            x0_retro = [result[0][1][:, 0] for result in forward_results]
            retro_res = pool.starmap(integrate_one, ((x0, T_min, T_retro, dt_retro, (min_distance, moon_SOI, earth_SOI, earth_perigee, moon_encounter, lunar_radial_velocity), accept_retro_condition) for x0 in x0_retro), chunksize=1)
            retro_results = [result[0] for result in retro_res]

        else:
            retro_res = []
            retro_results = [None] * len(forward_results)

        retro_collisions += sum(result[1] for result in retro_res)
        forward_collisions += sum(result[1] for result in total_sol)



    # Fill the lists with the accepted trajectories
    for forward, retro in zip(forward_results, retro_results):

        if forward is None:
            continue

        sol_f, traj_f, bar_f, moon_f, earth_f, e_f, m_f, b_f, Cj_f = forward
        accepted_initial_conditions.append(sol_f[1][:, 0].copy())

        if retro is None:

            if RETROGRADE:
                continue

            times.append(sol_f[0])
            SV.append(traj_f)
            SV_inertial.append(bar_f)
            moon_inertial.append(bar_f - moon_f)
            earth_inertial.append(bar_f - earth_f)
            earth_energy.append(e_f)
            moon_energy.append(m_f)
            bar_energy.append(b_f)
            
            Cj.append(Cj_f)
            Cj_cicle.append(Cj_norm)
            transition_indices.append(0)
            flyby_counts.append(len(sol_f[2][3]))
            encounters.append(sol_f[2][3])
            final_energy.append(e_f[-1] if filters["escape"] else 0)

            if filters["escape"]:
                perigee.append(0)
                delta_v.append(1)
                delta_E.append(e_f[-1] - e_f[0])

            elif filters["capture"]:
                x_perigee = sol_f[3][2][-1]
                e_perigee = two_body_energy(x_perigee[:3], x_perigee[3:], mu, body=1, frame='rotating')

                r_rel = x_perigee[:3] - np.array([-mu, 0, 0])
                v_rel = x_perigee[3:] + np.cross([0, 0, 1], r_rel)
                v_circ = np.sqrt((G * m1) / (np.linalg.norm(r_rel) * d))

                perigee.append(np.linalg.norm(r_rel))
                delta_v.append(np.linalg.norm(v_rel) * d / TU - v_circ)
                delta_E.append(e_f[-1] - e_f[0])

            continue

        
        sol_r, traj_r, bar_r, moon_r, earth_r, e_r, m_r, b_r, Cj_r = retro

        moon_apsides = np.vstack((*sol_r[3][5],*sol_f[3][4]))

        rp_m.append(np.min(np.linalg.norm(moon_apsides[:, :3] - np.array([1 - mu, 0, 0]), axis=1)) * d / R_L)

        transition_indices.append(len(sol_r[0]) - 1)

        times.append(np.concatenate((sol_r[0][::-1], sol_f[0][1:])))
        SV.append(np.concatenate((traj_r[:, ::-1], traj_f[:, 1:]), axis=1))
        SV_inertial.append(np.concatenate((bar_r[:, ::-1], bar_f[:, 1:]), axis=1))
        moon_inertial.append(np.concatenate((bar_r[:, ::-1] - moon_r[:, ::-1], bar_f[:, 1:] - moon_f[:, 1:]), axis=1))
        earth_inertial.append(np.concatenate((bar_r[:, ::-1] - earth_r[:, ::-1], bar_f[:, 1:] - earth_f[:, 1:]), axis=1))
        moon_energy.append(np.concatenate((m_r[::-1], m_f[1:]), axis=0))
        earth_energy.append(np.concatenate((e_r[::-1], e_f[1:]), axis=0))
        bar_energy.append(np.concatenate((b_r[::-1], b_f[1:]), axis=0))
        Cj.append(np.concatenate((Cj_r[::-1], Cj_f[1:]), axis=0))
        Cj_cicle.append(Cj_norm)
        flyby_counts.append(len(sol_f[2][3]) + len(sol_r[2][4]) - 1)
        encounters.append(np.concatenate((sol_r[2][4][1:], sol_f[2][3][1:])))

        sol_perigee = sol_r if filters["escape"] else sol_f
        sol_SOI = sol_f if filters["escape"] else sol_r
        
        x_perigee = sol_perigee[3][2][-1]
        x_SOI = sol_SOI[3][2][-1]

        e_perigee = two_body_energy(x_perigee[:3], x_perigee[3:], mu, body=1, frame='rotating')
        e_SOI = two_body_energy(x_SOI[:3], x_SOI[3:], mu, body=1, frame='rotating')

        r_rel = x_perigee[:3] - np.array([-mu, 0, 0])
        v_rel = x_perigee[3:] + np.cross([0, 0, 1], r_rel)
        v_circ = np.sqrt((G * m1) / (np.linalg.norm(r_rel) * d))

        perigee.append(np.linalg.norm(r_rel))
        delta_v.append(np.linalg.norm(v_rel) * d / TU - v_circ)
        delta_E.append(e_SOI - e_perigee)
        final_energy.append(e_SOI)




# Close the multiprocessing pool
pool.close()
pool.join()



# Print results and exit if no trajectories were accepted
if len(SV) == 0:
    elapsed_time = time.perf_counter() - start_time

    print("\n=== Simulation Results ===")
    print(f"Total computed trajectories: {total}")
    print("Total accepted trajectories: 0")
    print(f"Execution time: {elapsed_time:.2f} s")

    raise SystemExit



# Create the color normalization based on the accepted Jacobi constants
if Cj_cicle:
    Cj_accepted = np.asarray(Cj_cicle)
    norm_1 = Normalize(Cj_accepted.min(), Cj_accepted.max())
else:
    norm_1 = Normalize(Cj_min, Cj_max)




# Create the color normalization based on the accepted final energies
x0 = np.asarray(accepted_initial_conditions)
delta_E  = np.asarray(delta_E)
flyby_counts = np.asarray(flyby_counts)
perigee = np.asarray(perigee)
delta_v = np.asarray(delta_v)
rp_m = np.asarray(rp_m)
final_energy = np.asarray(final_energy)


try:
    norm_2 = Normalize(delta_v.min(), delta_v.max())
    bounds_3 = np.arange(flyby_counts.min() - 0.5, flyby_counts.max() + 1.5)
    n_flyby_levels = len(bounds_3) - 1
    cmap_3 = ListedColormap(flyby_colors[:n_flyby_levels])
    norm_3 = BoundaryNorm(bounds_3, cmap_3.N)
    norm_4 = Normalize(final_energy.min(), final_energy.max())
except ValueError:
    print("No accepted trajectories.")




# Save the results to a text file
trajectory_type = "Escape" if filters["escape"] else "Capture"

if False:
    i = 0
    r0, v0 = np.array([x[:3, i] for x in SV_inertial]), np.array([x[3:, i] for x in SV_inertial])
    t0 = np.array([t[i] * TU for t in times])
    data = np.column_stack((r0, v0, t0, Cj_cicle, delta_E*G*M/d, delta_v, final_energy*G*M/d, perigee*d-R_E, (rp_m-1)*R_L))
    np.savetxt(f"{trajectory_type}_trajectories_2.txt", data, header="r0x r0y r0z v0x v0y v0z t0_days Cj DE DV Emax h_perigeo h_min_periluneo")
elif SAVE:
    data = np.column_stack((x0, Cj_cicle, final_energy*G*M/d, (rp_m-1)*R_L, perigee*d-R_E))
    np.savetxt(f"{trajectory_type}_initial_conditions_CR3BP_solar.txt", data, header="x0 y0 z0 v0x v0y v0z Cj E_SOI h_min_periluneo h_perigeo")




# Compute Inertial Trajectories
T = np.linspace(0, 2 * np.pi, 10000)
Earth = Transformations.CR3BP_to_inertial([-mu * d, 0, 0, 0, 0, 0], 1, T)
Moon = Transformations.CR3BP_to_inertial([(1 - mu) * d, 0, 0, 0, 0, 0], 1, T)



# Print results
initial_conditions = np.array(all_initial_conditions)
if total > 0:
    print("\n=== Simulation Results ===")
    print(f"Total computed trajectories:      {total}")
    print(f"Total accepted trajectories:      {len(SV)} ({100 * len(SV) / total:.1f} %)")
    print(f"Forward collisions:               {forward_collisions} ({100 * forward_collisions / total:.1f} %)")
    print(f"Retrograde collisions:            {retro_collisions} ({100 * retro_collisions / total:.1f} %)\n")
else:
    print("\n=== Simulation Results ===")
    print("No initial conditions were accepted. Please check the parameters and try again.\n")

elapsed_time = time.perf_counter() - start_time
print(f"Execution time: {elapsed_time:.2f} s\n")




# Plot the results
if dim=='3D' and total > 0 and PLOT:

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter3D(initial_positions[:,0] * d, initial_positions[:,1] * d, initial_positions[:,2] * d, s=0.5, alpha=0.8, color='grey')
    ax.scatter3D(initial_conditions[:, 0] * d, initial_conditions[:, 1] * d, initial_conditions[:, 2] * d, s=0.5, color='black', marker="o")

    for trajectory in SV:

        X = trajectory[0, :]
        Y = trajectory[1, :]
        Z = trajectory[2, :]

        ax.plot3D(X, Y, Z)

    ax.scatter3D(-mu*d, 0, 0, s=50, zorder=-1, label='Earth', color='b')
    ax.scatter3D((1-mu)*d, 0, 0, s=50, zorder=-1, label='Moon', color='r')

    ax.scatter3D(0.8369151258 * d, 0.0, 0.0, s=45, marker='x', color='black')
    ax.scatter3D(1.1556821654 * d, 0.0, 0.0, s=45, marker='x', color='black')
    #ax.scatter3D(-1.0050626458 * d, 0.0, 0.0, s=45, marker='x', color='grey')
    #ax.scatter3D((0.5 - mu) * d, np.sqrt(3) / 2 * d, 0.0, s=45, marker='x', color='grey')
    #ax.scatter3D((0.5 - mu) * d, -np.sqrt(3) / 2 * d, 0.0, s=45, marker='x', color='grey')


    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    ax.legend()
    plt.show()



elif dim == '2D' and total > 0 and PLOT:

    fig = plt.figure(figsize=(19, 10))

    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[2, 1], hspace=0.2, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect(1)


    ax1.set_title("Rotating Frame")  
    ax1.scatter(initial_positions[:,0] * d, initial_positions[:,1] * d, s=0.5, alpha=0.8, color='grey')

    for trajectory, index, Cj_value in zip(SV, transition_indices, Cj_cicle):

        X = trajectory[0, :]
        Y = trajectory[1, :]

        ax1.plot(X, Y, color=cmap_1(norm_1(Cj_value)))
        ax1.scatter(X[index], Y[index], color='r', s=10, zorder=10)

    x = np.linspace(-1.5, 1.5, 600) 
    X, Y = np.meshgrid(x, x)

    r1 = np.sqrt((X + mu)**2 + Y**2)
    r2 = np.sqrt((X - (1 - mu))**2 + Y**2)
    C = X**2 + Y**2 + 2*((1 - mu)/r1 + mu/r2)

    #ax1.contour(X*d, Y*d, C, levels=[Cj_norm], colors="k")

    ax1.scatter((1-mu)*d, 0, s=50, zorder=10, label='Moon', color='darkred')
    ax1.scatter(-mu*d, 0, s=50, zorder=10, label='Earth', color='b')
    ax1.scatter(0.8369151258 * d, 0, s=45, marker='x', color='black', label='Lagrange points', zorder=10)
    ax1.scatter(1.1556821654 * d, 0, s=45, marker='x', color='black', zorder=10)



    ax2.set_title("Baricentric Inertial Frame")  

    ax2.plot(Earth[0, :], Earth[1, :], '--', color='b', alpha=1, linewidth=2, zorder=10)
    ax2.plot(Moon[0, :], Moon[1, :], '--', color='k', alpha=1, linewidth=2, zorder=10)
    
    for traj, index, delta_v_value in zip(SV_inertial, transition_indices, delta_v):

        color = cmap_2(norm_2(delta_v_value))

        ax2.plot(traj[0, :], traj[1, :], alpha=1, color=color)
        ax2.scatter(traj[0, index], traj[1, index], s=5, zorder=10, color=color)

    ax2.scatter(Earth[0, 0], Earth[1, 0], s=50, color='b', label='Earth', zorder=0)
    ax2.scatter(Moon[0, 0], Moon[1, 0], s=50, color='darkred', label='Moon', zorder=0) 



    ax3.plot(Earth[0, :], Earth[1, :], '--', color='b', alpha=1, linewidth=2, zorder=10)
    ax3.plot(Moon[0, :], Moon[1, :], '--', color='k', alpha=1, linewidth=2, zorder=10)
            
    for traj, index, flyby_value in zip(SV_inertial, transition_indices, flyby_counts):

        color = cmap_3(norm_3(flyby_value))
        
        ax3.plot(traj[0, :], traj[1, :], alpha=1, color=color)
        ax3.scatter(traj[0, index], traj[1, index], s=5, zorder=10, color=color)
        
    ax3.scatter(Earth[0, 0], Earth[1, 0], s=50, color='b', label='Earth', zorder=0)
    ax3.scatter(Moon[0, 0], Moon[1, 0], s=50, color='darkred', label='Moon', zorder=0) 


    
    for time, e_e, delta_v_value, Cj_value, energy_value, t_encounters in zip(times, earth_energy, delta_v, Cj_cicle, final_energy, encounters):

        color = cmap_2(norm_2(delta_v_value))

        ax4.plot(time, e_e, alpha=1, color=color)

        for t_moon in t_encounters:
            ax4.axvline(t_moon, color=color, linestyle=":", alpha=1, linewidth=1)

    ax5.scatter(rp_m, delta_E, c=delta_v, alpha=1, s=5, cmap=cmap_2, norm=norm_2)

    ax6.scatter(perigee*d - R_E, (delta_E * G * M / d) / delta_v**2, c=final_energy, alpha=1, s=5, cmap=cmap_4, norm=norm_4)


    fig.colorbar(plt.cm.ScalarMappable(norm=norm_1, cmap=cmap_1), ax=ax1, label=r"$C_J$")

    fig.colorbar(plt.cm.ScalarMappable(norm=norm_2, cmap=cmap_2), ax=ax2, label=r"$\Delta V$ [km/s]")

    fig.colorbar(plt.cm.ScalarMappable(norm=norm_3, cmap=cmap_3), ax=ax3, label="Number of encounters", ticks=np.arange(flyby_counts.min(), flyby_counts.max() + 1))

    fig.colorbar(plt.cm.ScalarMappable(norm=norm_4, cmap=cmap_4), ax=ax6, label=r"$E_{SOI}$")


    moon_earth_energy = two_body_energy([1 - mu, 0, 0], [0, 0, 0], mu, body=1, frame='rotating')

    ax4.axhline(0, color='k', linestyle='--', alpha=0.8)
    ax4.axvline(0, color='k', linestyle='--', alpha=0.8)
    ax4.axhline(moon_earth_energy, color='darkred', linestyle='--', alpha=0.8, label='Moon Energy')

    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True) 
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax1.set_xlabel("X [km]")
    ax1.set_ylabel("Y [km]")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True) 
    ax2.set_xlabel("X [km]") 
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.25) 
    ax2.legend()

    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax3.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True) 
    ax3.set_xlabel("X [km]") 
    ax3.set_aspect("equal")
    ax3.grid(alpha=0.25) 
    ax3.legend()

    ax4.set_title("Earth Specific Energy")
    ax4.set_xlabel("Time [TU]")
    ax4.set_ylabel(r"$\varepsilon_E$")
    ax4.legend()

    ax5.set_title("Energy Change vs Perilune Distance")
    ax5.set_xlabel(r"$r_{p,M}/R_L$")
    ax5.set_ylabel(r"$\Delta E$")

    ax6.set_title("Trajectory Efficiency vs Perigee Distance")
    ax6.set_xlabel(r"$h_{p,E}$")
    ax6.set_ylabel(r"$\Delta E/\Delta V^2$")

    plt.show()