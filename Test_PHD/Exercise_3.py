import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


mu_s = 1.3271244e11    # sun gravitational parameter (km^3/s^2)
mu_e = 398600.4415     # earth gravitational parameter (km^3/s^2)


# COE from SV Function
def rv_2_coe(R, V, mu):

    eps = 1e-12

    R = np.array(R)
    V = np.array(V)

    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    vr = np.dot(R, V) / r

    H = np.cross(R, V)
    h = np.linalg.norm(H)

    incl = np.arccos(H[2] / h)

    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)

    if n != 0:
        RA = np.arccos(N[0] / n)
        if N[1] < 0:
            RA = 2 * np.pi - RA
    else:
        RA = 0

    E = 1 / mu * ((v**2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)

    if n != 0 and e > eps:
        w = np.arccos(np.dot(N, E) / n / e)
        if E[2] < 0:
            w = 2 * np.pi - w
    else:
        w = 0

    if e > eps:
        if np.dot(E, R) / e / r > 1 and np.dot(E, R) / e / r < 1.00000001:
            TA = 0.0
        else:
            TA = np.arccos(np.dot(E, R) / e / r)
        if vr < 0:
            TA = 2 * np.pi - TA
    else:
        cp = np.cross(N, R)
        if cp[2] >= 0:
            TA = np.arccos(np.dot(N, R) / n / r)
        else:
            TA = 2 * np.pi - np.arccos(np.dot(N, R) / n / r)

    a = h**2 / mu / (1 - e**2)


    return a, h, e, np.rad2deg(incl), np.rad2deg(RA), np.rad2deg(w), np.rad2deg(TA)



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



#Trasformation Function
def rtn_to_eci(vec_rtn, r_eci, v_eci):

    n = np.cross(r_eci, v_eci)

    R = r_eci / np.linalg.norm(r_eci)
    N = n / np.linalg.norm(n)

    T = np.cross(N, R)

    Q = np.column_stack((R, T, N))

    return np.dot(Q, vec_rtn)



#Trasformation Function
def rtn_to_eci_2(vec_rtn, RAAN, i, w, TA):
    
    TA   = np.radians(TA)
    i    = np.radians(i)
    w    = np.radians(w)
    RAAN = np.radians(RAAN)

    u = w + TA


    R_z_RA = np.array([[np.cos(RAAN), -np.sin(RAAN), 0],
                       [np.sin(RAAN),  np.cos(RAAN), 0],
                       [0, 0, 1]])
    R_x_incl = np.array([[1, 0, 0],
                         [0, np.cos(i), -np.sin(i)],
                         [0, np.sin(i),  np.cos(i)]])
    R_z_u  = np.array([[np.cos(u), -np.sin(u), 0],
                       [np.sin(u),  np.cos(u), 0],
                       [0, 0, 1]])


    Q = np.dot(R_z_RA, np.dot(R_x_incl, R_z_u))

    return np.dot(Q, vec_rtn)



R = [4604.49276873138, 1150.81472538679, 4694.55079634563]
V = [-5.10903235110107, -2.48824074138143, 5.62098648967432]



# Exercise 3.1
coe = rv_2_coe(R,V,mu_e)

a = coe[0]
h = coe[1]
e = coe[2]
i = coe[3]
RAAN = coe[4]
w = coe[5]
TA = coe[6]

print("\nORBITAL ELEMENTS:")
print("Semi-major axis (km):", round(a,3))
print("Eccentricity:", round(e,3))
print("Inclination (deg):", round(i,3))
print("Right ascension of ascending node (deg):", round(RAAN,3))
print("Argument of perigee (deg):", round(w,3))
print("True anomaly (deg):", round(TA,3))



DV_r_RTN = np.array([0.01, 0.0, 0.0])
DV_v_RTN = np.array([0.0, 0.01, 0.0])
DV_n_RTN = np.array([0.0, 0.0, 0.01])



# Exercise 3.3
DV_ECI = rtn_to_eci(DV_r_RTN, R, V)

V_new = V + DV_ECI

coe_new = rv_2_coe(R, V_new, mu_e)

a_new = coe_new[0]
h_new = coe_new[1]
e_new = coe_new[2]
i_new = coe_new[3]
RAAN_new = coe_new[4]
w_new = coe_new[5]
TA_new = coe_new[6]

print("\nNEW ORBITAL ELEMENTS AFTER DV:")
print("Semi-major axis (km):", round(a_new,3))
print("Eccentricity:", round(e_new,5))
print("Inclination (deg):", round(i_new,3))
print("Right ascension of ascending node (deg):", round(RAAN_new,3))
print("Argument of perigee (deg):", round(w_new,3))
print("True anomaly (deg):", round(TA_new,3))



a_list, e_list, i_list, RAAN_list, w_list, TA_list = [], [], [], [], [], []



# Exercise 3.4
for TA in range(0,361):

    r, v = coe_2_rv(a, e, RAAN, i, w, mu_e, TA)

    DV_r_ECI = rtn_to_eci(DV_r_RTN, r, v)
    DV_v_ECI = rtn_to_eci(DV_v_RTN, r, v)
    DV_n_ECI = rtn_to_eci(DV_n_RTN, r, v)

    new_v = v + DV_n_ECI

    coe = rv_2_coe(r, new_v, mu_e)

    a_list.append(coe[0] - a)
    e_list.append(coe[2] - e)
    i_list.append(coe[3] - i)
    RAAN_list.append(coe[4] - RAAN)
    w_list.append(coe[5] - w)
    TA_list.append(TA)

plt.rcParams['axes.formatter.useoffset'] = False
    
plt.figure(figsize=(12, 8))
plt.plot(TA_list, a_list, linewidth=1)
plt.xlabel("TA [degrees]")
plt.ylabel("a - a0 [km]")
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(TA_list, e_list, linewidth=1)
plt.xlabel("TA [degrees]")
plt.ylabel("e - e0 [-]")
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(TA_list, i_list, linewidth=1)
plt.xlabel("TA [degrees]")
plt.ylabel("i - i0 [degrees]")

plt.show()

plt.figure(figsize=(12, 8))
plt.plot(TA_list, RAAN_list, linewidth=1)
plt.xlabel("TA [degrees]")
plt.ylabel("RAAN - RAAN0 [degrees]")

plt.show()

plt.figure(figsize=(12, 8))
plt.plot(TA_list, w_list, linewidth=1)
plt.xlabel("TA [degrees]")
plt.ylabel("w - w0 [degrees]")
plt.show()