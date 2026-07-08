import numpy as np
import scipy
import matplotlib.pyplot as plt


mu_s = 1.3271244e11    # sun gravitational parameter (km^3/s^2)
mu_e = 398600          # earth gravitational parameter (km^3/s^2)
DAY = 86400        


#Kepler Equation Solver
def mean_motion(a, mu):
    return np.sqrt(mu) / a**1.5 

def mean_anomaly(M0, n, t, t0):
    return M0 + n * (t - t0) 

def keplers_equation(E, e, M):
    return E - e * np.sin(E) - M

def true_anomaly(E, e):
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))

def Kepler(M, e, tol):
    
    if M > np.pi:
        E0 = M - e/2
    else:
        E0 = M + e/2

    E = scipy.optimize.newton(keplers_equation, E0, args=(e, M), tol=tol)

    theta = true_anomaly(E, e)

    if theta<0:

        theta+= 2*np.pi

    return theta


#Trasformation Function
def coe_2_rv(a, e, RAAN, i, w, mu, TA):  

    i = np.radians(i)
    w = np.radians(w)
    RAAN = np.radians(RAAN)
    TA = np.radians(TA)
       
    r_perifocal = a * (1 - e**2) / (1 + e * np.cos(TA)) * np.array([np.cos(TA), np.sin(TA), 0])
    v_perifocal = mu / np.sqrt(mu * a * (1 - e**2)) * np.array([-np.sin(TA), e + np.cos(TA), 0])
    
    R_z_RA = np.array([[np.cos(RAAN), -np.sin(RAAN), 0],
                       [np.sin(RAAN), np.cos(RAAN), 0],
                       [0, 0, 1]])
    
    R_x_incl = np.array([[1, 0, 0],
                         [0, np.cos(i), -np.sin(i)],
                         [0, np.sin(i), np.cos(i)]])
    
    R_z_w = np.array([[np.cos(w), -np.sin(w), 0],
                      [np.sin(w), np.cos(w), 0],
                      [0, 0, 1]])

    Q = np.dot(R_z_RA, np.dot(R_x_incl, R_z_w))

    r = np.dot(Q, r_perifocal)
    v = np.dot(Q, v_perifocal)

    return r, v



#Ephemeris
def Ephemeris(a, e, RAAN, i, w, mu, t, t0, M0):

    n = mean_motion(a, mu)
    M = mean_anomaly(np.deg2rad(M0), n, t, t0)
    TA = np.rad2deg(Kepler(M, e, 1e-12))

    return coe_2_rv(a, e, RAAN, i, w, mu, TA)



# Asteroid Initial State
Y_a = 3.764419202360106e8     
Y_e = 6.616071771587672e-1
Y_i = 3.408286057191753
Y_RAAN = 2.713674649188756e2
Y_w = 1.343644678984687e2
Y_M0 = 1.693237490356061e1


# Earth Initial State
E_a = 1.495988209443421e8
E_e = 1.669829008180246e-2
E_i = 3.248050135173038e-3
E_RAAN = 1.744712892867145e2
E_w = 2.884490093009512e2
E_M0 = 2.621190445180298e1


# Mean Motions
Y_n = mean_motion(Y_a, mu_s)
E_n = mean_motion(E_a, mu_s)


# Time Definition
t0 = 2460705.5 * DAY
t = t0
t1 = t0
t2 = t0 + 100*DAY


# Mean Anomalies
M1 = mean_anomaly(np.deg2rad(Y_M0), Y_n, t0, t0)
M2 = mean_anomaly(np.deg2rad(Y_M0), Y_n, t2, t0)


# Lists Creation
R_E = []
R_Y = []

V_E = []
V_Y = []

dist = []
time = []



# Exercise 1.2
Y_TA0 = np.rad2deg(Kepler(M1, Y_e, 1e-12))
Y_TA1 = np.rad2deg(Kepler(M2, Y_e, 1e-12))

print(f"\nt0         | True Anomaly = {Y_TA0:.3f} deg")
print(f"t0+100 d   | True Anomaly = {Y_TA1:.3f} deg\n")



# Exercise 1.3
SV_0 = coe_2_rv(Y_a, Y_e, Y_RAAN, Y_i, Y_w, mu_s, Y_TA0)
SV = coe_2_rv(Y_a, Y_e, Y_RAAN, Y_i, Y_w, mu_s, Y_TA1)

r0, v0 = SV_0
r, v = SV

print(f"t0         | "
      f"x={r0[0]:.8e}  y={r0[1]:.8e}  z={r0[2]:.8e}  | "
      f"vx={v0[0]:.8f}  vy={v0[1]:.8f}  vz={v0[2]:.8f}")

print(f"t0+100 d   | "
      f"x={r[0]:.8e}  y={r[1]:.8e}  z={r[2]:.8e}  | "
      f"vx={v[0]:.8f}  vy={v[1]:.8f}  vz={v[2]:.8f}\n")



# Exercise 1.4
while t < (t0 + 365.25*DAY):

    SV_Y = Ephemeris(Y_a, Y_e, Y_RAAN, Y_i, Y_w, mu_s, t, t0, Y_M0)
    SV_E = Ephemeris(E_a, E_e, E_RAAN, E_i, E_w, mu_s, t, t0, E_M0)

    R_E.append(np.linalg.norm(SV_E[0]))
    V_E.append(np.linalg.norm(SV_E[1]))

    R_Y.append(np.linalg.norm(SV_Y[0]))
    V_Y.append(np.linalg.norm(SV_Y[1]))

    dist.append(np.linalg.norm(SV_E[0] - SV_Y[0]))

    time.append(t)

    t += DAY


plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.limits'] = (-3, 4)  # evita sci tranne casi estremi

plt.figure(figsize=(12,8))

plt.plot((np.array(time) - t0)/DAY/365.25, np.array(R_E)/149597871, label='Earth R')
plt.plot((np.array(time) - t0)/DAY/365.25, np.array(R_Y)/149597871, label='Asteroid R')
plt.xlabel('Earth Orbital Phase')
plt.ylabel('R [AU]')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))

plt.plot((np.array(time) - t0)/DAY/365.25, np.array(V_E), label='Earth V')
plt.plot((np.array(time) - t0)/DAY/365.25, np.array(V_Y), label='Asteroid V')
plt.xlabel('Earth Orbital Phase')
plt.ylabel('V [km/s]')
plt.legend()
plt.show()



# Exercise 1.5
while t < (t0 + 365.25*DAY*10):

    SV_Y = Ephemeris(Y_a, Y_e, Y_RAAN, Y_i, Y_w, mu_s, t, t0, Y_M0)
    SV_E = Ephemeris(E_a, E_e, E_RAAN, E_i, E_w, mu_s, t, t0, E_M0)

    R_E.append(np.linalg.norm(SV_E[0]))
    V_E.append(np.linalg.norm(SV_E[1]))

    R_Y.append(np.linalg.norm(SV_Y[0]))
    V_Y.append(np.linalg.norm(SV_Y[1]))

    dist.append(np.linalg.norm(SV_E[0] - SV_Y[0]))

    time.append(t)

    t += DAY


plt.figure(figsize=(12,8))

plt.plot((np.array(time) - t0)/DAY/365.25, np.array(R_E)/149597871, label='Earth R')
plt.plot((np.array(time) - t0)/DAY/365.25, np.array(R_Y)/149597871, label='Asteroid R')
plt.xlabel('Earth Orbital Phase')
plt.ylabel('R [AU]')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))

plt.plot((np.array(time) - t0)/DAY/365.25, np.array(V_E), label='Earth V')
plt.plot((np.array(time) - t0)/DAY/365.25, np.array(V_Y), label='Asteroid V')
plt.xlabel('Earth Orbital Phase')
plt.ylabel('V [km/s]')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))

plt.plot((np.array(time) - t0)/DAY/365.25, np.array(dist), label='Earth-Asteroid Distance')
plt.xlabel('Earth Orbital Phase')
plt.ylabel('Distance [km]')
plt.legend()
plt.show()

print(f"Minimum Earth–Asteroid distance = {np.min(dist):.3e} km\n")