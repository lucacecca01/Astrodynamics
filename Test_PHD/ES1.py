import numpy as np
import scipy

mu = 398600.4415  # gravitational parameter (km^3/s^2)

#KEPLER EQUATION SOLVER
def mean_anomaly(M0, n, t, t0):
    return M0 + n * (t - t0)

def keplers_equation(E, e, M):
    return E - e * np.sin(E) - M

def eccentric_anomaly(M, e):

    M = np.radians(M)
    
    if M > np.pi:
        E0 = M - e/2
    else:
        E0 = M + e/2

    return scipy.optimize.newton(keplers_equation, E0, args=(e, M))

def true_anomaly(E, e):
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))



#TRASFORMATION FUNCTION
def sv_from_coe(a, e, RA, i, w, mu, Me):  

    E = eccentric_anomaly(Me, e)
    TA = true_anomaly(E, e)
               
    r_perifocal = a * (1 - e**2) / (1 + e * np.cos(TA)) * np.array([np.cos(TA), np.sin(TA), 0])
    v_perifocal = mu / np.sqrt(mu * a * (1 - e**2)) * np.array([-np.sin(TA), e + np.cos(TA), 0])
    
    R_z_RA = np.array([[np.cos(RA), -np.sin(RA), 0],
                       [np.sin(RA), np.cos(RA), 0],
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