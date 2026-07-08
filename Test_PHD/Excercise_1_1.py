import numpy as np
import scipy

mu_s = 1.3271244e11    # sun gravitational parameter (km^3/s^2)
mu_e = 398600          # sun gravitational parameter (km^3/s^2)

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


# Asteroid Initial State
Y_a = 3.764419202360106e8     
Y_e = 6.616071771587672e-1
Y_i = 3.408286057191753
Y_RAAN = 2.713674649188756e2
Y_w = 1.343644678984687e2
Y_M0 = 1.693237490356061e1


# Time Definition
t0 = 2460705.5 * 86400
t1 = t0
t2 = t0 + 100*86400


# Results
n = mean_motion(Y_a, mu_s)

M1 = mean_anomaly(np.deg2rad(Y_M0), n, t1, t0)
M2 = mean_anomaly(np.deg2rad(Y_M0), n, t2, t0)

theta1 = Kepler(M1, Y_e, 1e-12)
theta2 = Kepler(M2, Y_e, 1e-12)

print(f"True Anomaly at t0 = {np.rad2deg(theta1):.8f} deg")
print(f"True Anomaly at t0 + 100 days = {np.rad2deg(theta2):.8f} deg")