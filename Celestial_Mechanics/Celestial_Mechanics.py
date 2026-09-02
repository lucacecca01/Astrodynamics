import numpy as np
import scipy
import rebound


class Transformations:

    def M_from_t(self, M0, n, t, t0):
        return M0 + n * (t - t0)

    def keplers_equation(self, E, e, M):
        return E - e * np.sin(E) - M
    
    def TA_from_E(self, E, e, deg_or_rad='rad'):
        if deg_or_rad == 'deg':
            E = np.radians(E)
        return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    def E_from_TA(self, TA, e, deg_or_rad='rad'):
        if deg_or_rad == 'deg':
            TA = np.radians(TA)
        return 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(TA / 2))

    def TA_from_M(self, M, e, deg_or_rad='rad'):

        if deg_or_rad == 'deg':
            M = np.radians(M)

        if M > np.pi:
            E0 = M - e/2
        else:
            E0 = M + e/2

        E = scipy.optimize.newton(self.keplers_equation, E0, args=(e, M))

        return self.TA_from_E(E, e)
    
    def M_from_TA(self, TA, e, deg_or_rad='rad'):
        E = self.E_from_TA(TA, e, deg_or_rad)
        M = E - e * np.sin(E)
        return M



    @staticmethod
    def sv_from_coe(mu, a, e, i, RA, w, anomaly, r_p=None, anomaly_type='true', deg_or_rad='deg'):  

        if deg_or_rad not in ("deg", "rad"):
            raise ValueError("deg_or_rad has to be 'deg' or 'rad'")
        
        if anomaly_type not in ("mean", "eccentric", "true"):
            raise ValueError("anomaly_type has to be 'mean', 'eccentric' or 'true'")

        if anomaly_type == 'mean':
            TA = self.TA_from_M(anomaly, e, deg_or_rad)

        elif anomaly_type == 'eccentric':
            TA = self.TA_from_E(anomaly, e, deg_or_rad)

        elif anomaly_type == 'true':
            if deg_or_rad == 'deg':
                TA = np.radians(anomaly)
            else:
                TA = anomaly

        if deg_or_rad == 'deg':
            i = np.radians(i)
            RA = np.radians(RA)
            w = np.radians(w)


        if e < 1:
            h = np.sqrt(mu * a * (1 - e**2))
        elif np.isclose(e, 1.0):
            if r_p is None:
                raise ValueError("For a parabolic orbit, you must specify r_p.")
            h = np.sqrt(2 * mu * r_p)
        elif e > 1:
            h = np.sqrt(mu * np.abs(a * (1 - e**2)))
     
      
        r_perifocal = h**2 / mu / (1 + e * np.cos(TA)) * np.array([np.cos(TA), np.sin(TA), 0])
        v_perifocal = mu / h * np.array([-np.sin(TA), e + np.cos(TA), 0])
        
    
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
    


    @staticmethod
    def coe_from_sv(R, V, mu, deg_or_rad='deg'):

        if deg_or_rad not in ("deg", "rad"):
            raise ValueError("deg_or_rad has to be 'deg' or 'rad'")
        
        R = np.array(R)
        V = np.array(V)

        eps = 1e-12

        r = np.linalg.norm(R)
        v = np.linalg.norm(V)

        if r < eps or v < eps:
            raise ValueError("The position and velocity vectors cannot be zero")

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
            if n != 0:
                cp = np.cross(N, R)
                if cp[2] >= 0:
                    TA = np.arccos(np.dot(N, R) / n / r)
                else:
                    TA = 2 * np.pi - np.arccos(np.dot(N, R) / n / r)
            else:
                TA = np.arccos(R[0] / r)

        a = h**2 / mu / (1 - e**2)

        if deg_or_rad == 'deg':
            return h, a, e, np.rad2deg(incl), np.rad2deg(RA), np.rad2deg(w), np.rad2deg(TA)
        
        elif deg_or_rad == 'rad':
            return h, a, e, incl, RA, w, TA



    @staticmethod
    def rtn_to_inertial(vec_rtn, r_eci, v_eci):

        n = np.cross(r_eci, v_eci)

        if np.linalg.norm(n) == 0:
            raise ValueError("RTN frame is undefined when position and velocity are collinear or equal to zero.")

        R = r_eci / np.linalg.norm(r_eci)
        N = n / np.linalg.norm(n)

        T = np.cross(N, R)

        Q = np.column_stack((R, T, N))

        return np.dot(Q, vec_rtn)



    @staticmethod
    def SV_to_CR3BP_normalized_units(x, d, G, M):

        index = len(x) // 2
        x = np.array(x)

        LU = d
        TU = np.sqrt(d**3 / G / M)

        x[:index] /= LU
        x[index:] *= TU/LU    

        return x



    @staticmethod
    def CR3BP_normalized_units_to_SV(x, d, G, M):

        index = len(x) // 2
        x = np.array(x)

        LU = d
        TU = np.sqrt(d**3 / G / M)

        x[:index] *= LU
        x[index:] /= TU/LU    

        return x



    @staticmethod
    def CR3BP_to_inertial(x, n, t):

        x = np.asarray(x)
        t = np.atleast_1d(t)

        if x.ndim == 1:
            x = np.tile(x[:, None], (1, len(t)))

        omega = np.array([0, 0, n])

        r = x[:3]
        v_rot = x[3:] + np.cross(omega, r, axis=0)

        R = np.array([[np.cos(n*t), -np.sin(n*t)],
                      [np.sin(n*t),  np.cos(n*t)]])

        r_I = np.vstack((R[0,0]*r[0] + R[0,1]*r[1],
                         R[1,0]*r[0] + R[1,1]*r[1],
                         r[2]))

        v_I = np.vstack((R[0,0]*v_rot[0] + R[0,1]*v_rot[1],
                         R[1,0]*v_rot[0] + R[1,1]*v_rot[1],
                         v_rot[2]))

        return np.vstack((r_I, v_I))
    

    
    @staticmethod
    def inertial_to_CR3BP(x, n, t):

        x = np.asarray(x, dtype=float)
        t = np.atleast_1d(t)

        if x.ndim == 1:
            x = np.tile(x[:, None], (1, len(t)))

        omega = np.array([0, 0, n])

        r = x[:3]
        v_rot = x[3:] - np.cross(omega, r, axis=0)

        R = np.array([[np.cos(n*t), -np.sin(n*t)],
                      [np.sin(n*t),  np.cos(n*t)]])
        
        R_T = R.swapaxes(0, 1)

        r_R = np.vstack((R_T[0,0]*r[0] + R_T[0,1]*r[1],
                         R_T[1,0]*r[0] + R_T[1,1]*r[1],
                         r[2]))

        v_R = np.vstack((R_T[0,0]*v_rot[0] + R_T[0,1]*v_rot[1],
                         R_T[1,0]*v_rot[0] + R_T[1,1]*v_rot[1],
                         v_rot[2]))

        return np.vstack((r_R, v_R))
    





class Integration:

    def RK4(self, F, x, dt):   
        k1 = dt * F(x)
        k2 = dt * F(x + 0.5*k1)
        k3 = dt * F(x + 0.5*k2)
        k4 = dt * F(x + k3)
        x += (k1 + 2*k2 + 2*k3 + k4) / 6
        return x

    

    def NBP_ODE(self, t, x, mu):

        F = np.zeros(len(x))
    
        for i in range(0, len(x), 6):
            
            F[i] = x[i+3]
            F[i+1] = x[i+4]
            F[i+2]= x[i+5]
    
        for i in range(0, len(x), 6):
            for j in range(len(mu)):
                if j * 6 != i:
                    
                    dx = x[i] - x[j*6]
                    dy = x[i+1] - x[j*6+1]
                    dz = x[i+2] - x[j*6+2]

                    dist_cubed = (dx**2 + dy**2 + dz**2) ** 1.5
                    
                    F[i+3] += -mu[j] * dx / dist_cubed
                    F[i+4] += -mu[j] * dy / dist_cubed
                    F[i+5] += -mu[j] * dz / dist_cubed
        return F



    def CR3BP_ODE(self, t, x, mu):

        F = np.zeros_like(x, dtype=float)
            
        F[0] = x[3]
        F[1] = x[4]
        F[2] = x[5]

        r1_cubed = ((x[0] + mu)**2 + x[1]**2 + x[2]**2) ** 1.5
        r2_cubed = (((x[0] - (1 - mu)))**2 + x[1]**2 + x[2]**2) ** 1.5

        F[3] = 2 * x[4]  +  x[0]  -  (1 - mu) * (x[0] + mu) / r1_cubed  -  mu * (x[0] - (1 - mu)) / r2_cubed
        F[4] = -2 * x[3]  +  x[1]  -  (1 - mu) * x[1] / r1_cubed - mu * x[1] / r2_cubed
        F[5] = - (1 - mu) * x[2] / r1_cubed - mu * x[2] / r2_cubed

        return F



    def CR3BP_Cj_from_v(self, x, mu):

        v_2 = x[3]**2 + x[4]**2 + x[5]**2

        r1 = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
        r2 = np.sqrt(((x[0] - (1 - mu)))**2 + x[1]**2 + x[2]**2)

        return -v_2 + (x[0]**2 + x[1]**2) + 2 * ((1 - mu) / r1 + mu / r2)



    def CR3BP_v_from_Cj(self, Cj, x, mu):

        r1 = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
        r2 = np.sqrt(((x[0] - (1 - mu)))**2 + x[1]**2 + x[2]**2)

        v_2 = (x[0]**2 + x[1]**2) + 2 * ((1 - mu) / r1 + mu / r2) - Cj

        if v_2 < 0:
            raise ValueError("Computed v_2 is negative.")

        return np.sqrt(v_2)



    def Integrator(self, m, G, x0_list, t0, tf, dt, model='NBP', integrator='scipy', method='RK45', rtol=1e-9, atol=1e-12, events=None, stop_condition=None):
         
        if model == 'NBP':
            m = np.atleast_1d(m)
            mu = G * m

        elif model == 'CR3BP':
            M = np.sum(m)
            mu = m[1] / M

        else:
            raise ValueError(f"Model not found: {model}")


        if integrator == 'scipy':

            N = int(round((tf - t0) / dt)) + 1
            T = np.linspace(t0, tf, N)

            if isinstance(x0_list[0], (list, tuple, np.ndarray)):
                x0 = np.concatenate(x0_list)
            else:
                x0 = x0_list

            sol = scipy.integrate.solve_ivp(getattr(self, f"{model}_ODE"), [t0, tf], x0, args=(mu, ), t_eval=T, method=method, rtol=rtol, atol=atol, events=events)

            return sol.t, sol.y, sol.t_events, sol.y_events
        

        if integrator == 'rebound':

            sim = rebound.Simulation()
            sim.integrator = method
            sim.integrator.epsilon = rtol
            sim.G = G

            if isinstance(x0_list[0], (list, tuple, np.ndarray)):
                n_body = len(x0_list)
                for mass, x0 in zip(m, x0_list):
                    sim.add(m=mass, x=x0[0], y=x0[1], z=x0[2], vx=x0[3], vy=x0[4], vz=x0[5])
            else:
                n_body = len(x0_list) // 6
                for i in range(n_body):
                    sim.add(m=m[i], x=x0_list[6*i], y=x0_list[6*i+1], z=x0_list[6*i+2], vx=x0_list[6*i+3], vy=x0_list[6*i+4], vz=x0_list[6*i+5])

            N = int((tf - t0)/dt) + 1
            T = np.linspace(t0, tf, N)
            sol = np.zeros((n_body, N, 6))

            for i, t in enumerate(T):
    
                sim.integrate(t)
                    
                for j, p in enumerate(sim.particles):
                    sol[j, i] = [p.x, p.y, p.z, p.vx, p.vy, p.vz]

                if stop_condition is not None and stop_condition(t, sol[:, i]):
                    return T[:i+1], sol[:, :i+1]

            return T, sol



    def Total_energy(self, x_list, m):

        if isinstance(x_list[0], (list, tuple, np.ndarray)):
          x = np.concatenate(x_list)
        else:
          x = x_list

        G = 6.67430e-11
        m = np.atleast_1d(m)
        mu = G * m

        kinetic_energy, potential_energy = 0, 0
        
        for i in range(0, len(x), 6):
            vx, vy, vz = x[i+3], x[i+4],  x[i+5]
            kinetic_energy += 0.5 * m[i//6] * (vx**2 + vy**2 + vz**2)
        
        for i in range(0, len(x), 6):
            for j in range(i + 6, len(x), 6):
                dx = x[i] - x[j]
                dy = x[i+1] - x[j+1]
                dz = x[i+2] - x[j+2]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                potential_energy += - mu[j//6] * m[i//6] / r
                
        return kinetic_energy, potential_energy, kinetic_energy + potential_energy
    
 

    def Total_angular_momentum(self, x_list, m):  

        G = 6.67430e-11
        L = 0

        if isinstance(x_list[0], (list, tuple, np.ndarray)):
            x = np.concatenate(x_list)
        else:
            x = x_list

        for i in range(0, len(x), 6):

            r = np.array(x[i:i+3])
            v = np.array(x[i+3:i+6])

            L += np.cross(r, m[i//6] * v)

        return np.linalg.norm(L)