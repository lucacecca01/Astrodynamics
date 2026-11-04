import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import pykep as pk
import spiceypy as spice
from Celestial_Mechanics import Integration, Transformations
import time
from multiprocessing import get_context
from matplotlib.colors import PowerNorm



start_time = time.perf_counter()


def element_periodicity(values, times, indices, reference=None, chunk_size=256):
    """Ripetitivita delle oscillazioni di un OE dopo la sola rimozione lineare.

    Le differenze normalizzate come in YIN individuano tre possibili ritorni:
    https://www.ee.columbia.edu/~dpwe/papers/deChevK02-yin.pdf
    Il punteggio finale e la frazione di varianza predetta dagli altri cicli,
    con validazione lasciando fuori il ciclo da predire (anche l'ultimo parziale).
    Restituisce punteggio, tempo di ritorno, variazione lineare stimata nelle
    unita originali e frazione di varianza spiegata dalla retta. La retta puo
    approssimare parte di una lunga oscillazione: non prova una deriva secolare.
    Il tempo di ritorno (4 campioni..T/3) puo essere multiplo del periodo.
    reference=a0 normalizza il semiasse prima del fit; e resta adimensionale.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.ndim != 2 or times.ndim != 1 or values.shape[0] != times.size:
        raise ValueError("OE e times devono avere dimensioni temporali compatibili.")
    scores = np.full(values.shape[1], np.nan)
    return_times = np.full_like(scores, np.nan)
    linear_change = np.full_like(scores, np.nan)
    linear_fraction = np.full_like(scores, np.nan)
    reference = np.ones_like(scores) if reference is None else np.asarray(reference, dtype=float)
    if reference.shape != scores.shape:
        raise ValueError("reference deve contenere un valore per particella.")
    if times.size < 13:
        return scores, return_times, linear_change, linear_fraction
    steps = np.diff(times)
    if not np.all(np.isfinite(times)) or not np.all(steps > 0) or not np.allclose(steps, steps[0], rtol=1e-8, atol=0):
        raise ValueError("La periodicita richiede tempi finiti, crescenti ed equispaziati.")

    n = times.size
    fit_time = (times - times.mean()) / (times[-1] - times[0])
    fit_energy = np.sum(fit_time**2)
    lags = np.arange(1, int(np.ceil((n - 1) / 3)) + 2)
    fft_size = 1 << (2 * n - 1).bit_length()
    for start in range(0, len(indices), chunk_size):
        columns = np.asarray(indices[start:start + chunk_size], dtype=int)
        valid = (np.all(np.isfinite(values[:, columns]), axis=0)
                 & np.isfinite(reference[columns]) & (reference[columns] > 0))
        columns = columns[valid]
        if not columns.size:
            continue
        x = (values[:, columns] - values[0, columns]) / reference[columns]
        x -= x.mean(axis=0, keepdims=True)
        total_variation = np.sum(x*x, axis=0)
        fitted_change = np.sum(fit_time[:, None] * x, axis=0) / fit_energy
        linear_change[columns] = fitted_change * reference[columns]
        linear_fraction[columns] = np.clip(np.divide(
            fitted_change**2 * fit_energy, total_variation,
            out=np.full(columns.size, np.nan), where=total_variation > n * 1e-24,
        ), 0, 1)
        x -= fit_time[:, None] * fitted_change
        x = x[:, :, None]
        amplitude = np.max(np.abs(x), axis=(0, 2))
        varying = amplitude > 1e-12  # Un segnale costante non identifica un periodo.
        columns, x = columns[varying], x[:, varying]
        if not columns.size:
            continue
        x /= amplitude[varying][None, :, None]
        scores[columns] = 0.0

        # Differenze quadratiche medie per ogni lag, senza autocorrelazione circolare.
        spectrum = np.fft.rfft(x, n=fft_size, axis=0)
        correlation = np.fft.irfft(np.sum(np.abs(spectrum)**2, axis=2), n=fft_size, axis=0)
        energy = np.vstack((np.zeros(columns.size), np.cumsum(np.sum(x*x, axis=2), axis=0)))
        differences = np.maximum(
            energy[n - lags] + energy[-1] - energy[lags] - 2 * correlation[lags], 0
        ) / (n - lags[:, None])
        cumulative = np.cumsum(differences, axis=0)
        normalized = np.divide(lags[:, None] * differences, cumulative,
                               out=np.ones_like(differences), where=cumulative > 0)

        for j, column in enumerate(columns):
            curve = normalized[:, j]
            candidates = np.flatnonzero(
                (curve[1:-1] < curve[:-2]) & (curve[1:-1] <= curve[2:])
                & (lags[1:-1] >= 4)
            ) + 1
            candidates = candidates[np.argsort(curve[candidates], kind="stable")[:3]]
            for candidate in candidates:
                left, center, right = differences[candidate - 1:candidate + 2, j]
                curvature = left - 2 * center + right
                shift = 0.5 * (left - right) / curvature if curvature > 0 else 0.0
                period = np.clip(lags[candidate] + np.clip(shift, -0.5, 0.5), 4, (n - 1) / 3)
                cycles = int(np.floor((n - 1) / period + 1e-12))
                phase_count = max(4, int(np.ceil(period)))
                sample_at = period * (np.arange(cycles + 1)[:, None]
                                      + np.arange(phase_count)[None, :] / phase_count)
                valid_phase = sample_at <= n - 1
                sample_at = np.minimum(sample_at, n - 1)
                lower = np.minimum(sample_at.astype(int), n - 2)
                fraction = (sample_at - lower)[..., None]
                folded = ((1 - fraction) * x[lower, j] + fraction * x[lower + 1, j])
                folded = np.where(valid_phase[..., None], folded, 0.0)
                counts = valid_phase.sum(axis=0)
                phase_mean = folded.sum(axis=0) / counts[:, None]
                mean = folded.sum(axis=(0, 1)) / valid_phase.sum()
                variance = np.sum((folded - mean)**2 * valid_phase[..., None])
                if variance <= 0:
                    continue
                residual = (folded - phase_mean) * valid_phase[..., None]
                prediction_error = np.sum(residual**2 * (counts / (counts - 1))[None, :, None]**2)
                score = float(np.clip(1 - prediction_error / variance, 0, 1))
                if score > scores[column]:
                    scores[column] = score
                    return_times[column] = period * steps[0]

    return scores, return_times, linear_change, linear_fraction



# Define earth-moon geometry function
def earth_moon_geometry(jd):

    jd = np.asarray(jd, dtype=float)
    day = 86400.0
    et = (jd - 2451545.0) * day
   
    def state(body, observer):
        values = np.array([spice.spkezr(body, float(epoch), "ECLIPJ2000", "NONE", observer)[0] for epoch in et.ravel()]).reshape(jd.shape + (6,))
        values[..., 2] = 0
        values[..., 5] = 0
        return values

    earth = state("EARTH", "SUN")
    moon_geo = state("MOON", "EARTH")

    theta_moon = np.arctan2(moon_geo[..., 1], moon_geo[..., 0])

    return (earth[..., :3], earth[..., 3:], earth[..., :3] + moon_geo[..., :3], earth[..., 3:] + moon_geo[..., 3:], theta_moon)



# Define integration function
def integration(index, times):

    sim_local = sim.copy()

    for k in range(sim_local.N - 1, 2, -1):
        if k not in index:
            sim_local.remove(k)

    sim_local.collision_resolve = "merge"

    for j, p in enumerate(sim_local.particles):
        p.name = str(j)

    n_particles = sim_local.N
    positions = np.full((N, n_particles, 3), np.nan)
    velocities = np.full_like(positions, np.nan)

    for i, t in enumerate(times):
    
        sim_local.integrate(t)
            
        for p in sim_local.particles:
            j = int(p.name)
            positions[i, j] = [p.x, p.y, p.z]
            velocities[i, j] = [p.vx, p.vy, p.vz]

    r_geo = positions[:, 3:, :] - positions[:, 1:2, :]
    v_geo = velocities[:, 3:, :] - velocities[:, 1:2, :]

    a = np.full((N, n_particles - 3), np.nan)
    e = np.full_like(a, np.nan)
    w = np.full_like(a, np.nan)
    stability = np.full_like(a, -1.0)

    for i in range(N):
        for j in range(n_particles - 3):

            if not np.all(np.isfinite(r_geo[i, j])):
                continue

            _, a[i, j], e[i, j], _, _, w[i, j], _ = (transformations.coe_from_sv(r_geo[i, j], v_geo[i, j], mu_earth))

            if e[i, j] >= 0.5 or a[i, j] >= 400000:
                stability [i, j] = -1
            else:
                a0 = a_initial[index[j] - 3]
                stability[i, j] = np.sqrt(e[i, j]**2 + ((a[i, j] - a0) / a0)**2)


    w_plot = (w + 180) % 360 - 180
    stability_max = np.where(np.any(stability == -1, axis=0), -1, np.max(stability, axis=0))

    return index, r_geo, v_geo, a, e, w_plot, stability_max



#CREATE SIMULATION
sim = rebound.Simulation()
spice.furnsh(pk.udpla.de440s.kernel_file())
transformations = Transformations()


#DEFINE THE LISTS
E = []
L = []
a_initial = []


#DYNAMICAL PARAMETERS
sim.G = 6.6743e-20       # km^3 / kg s^2
AU = 149597870.7         # km
YR = 31557600            # s
DAY = 86400              # s
M_SUN = 1.989e30         # kg
M_EARTH = 5.9722e24      # kg
M_MOON = 7.342e22        # kg
R_SUN = 695700.0         # km
R_EARTH = 6378.0         # km
R_MOON = 1737.4          # km
mu_earth = sim.G * M_EARTH
d = 384400   
M_SOI = (M_MOON / M_EARTH)**(2/5) * d
np.random.seed(546765)

JD = 2460293.5
distance_range = np.linspace(140000, 360000, 200)


earth_x, earth_v, moon_x, moon_v, theta_moon = earth_moon_geometry(JD)




#ADD BODIES TO THE SYSTEM
sim.add(m=1.989e30, x=0, y=0, vx=0, vy=0, r=R_SUN)  
sim.add(m=5.9722e24,  x=earth_x[0], y=earth_x[1], vx=earth_v[0], vy=earth_v[1], r=R_EARTH)
sim.add(m=7.342e22,  x=moon_x[0], y=moon_x[1], vx=moon_v[0], vy=moon_v[1], r=M_SOI/2)

for distance in distance_range:
    for TA in np.linspace(theta_moon, 2*np.pi + theta_moon, 500, endpoint=False):
        sim.add(m=0, primary=sim.particles[1], a=distance, e=0, inc=np.pi, f=-TA, r=0)
        a_initial.append(distance)


sim.N_active = 3
sim.testparticle_type = 0




#DEFINE COLLISION FUNCTION
def collisions(sim_pointer, collision):
    sim = sim_pointer.contents 
    p1 = sim.particles[collision.p1]
    p2 = sim.particles[collision.p2]
      
    print("Time of collision:", sim.t/YR)  
    print("Colliding particles removed:", collision.p1, collision.p2)
    
    p1.m = 0
    p1.x = 1e4*AU
    p2.m = 0
    p2.x = 1e4*AU
                      
    return 0                                


#INTEGRATOR CHOICE
sim.integrator = "ias15"


#SETTING INTEGRATOR ACCURACY
sim.integrator.epsilon = 1e-12


#MOVE TO CENTER OF MASS AND INITIAL ENERGY
sim.move_to_com()
E0 = sim.energy()


#COLLISION FUNCTION
sim.collision = "line"
#sim.collision_resolve = "merge"


#DEFINE TIME ARRAY FOR VISUALIZATION
Tmax = 10*YR
Tmin = 0
dt = DAY
N = int((Tmax - Tmin)/dt)
times = np.linspace(Tmin, Tmax, N)


#DEFINE POSITION ARRAY
n_particles = len(sim.particles) 
positions = np.empty((N, n_particles, 3))
a_initial = np.asarray(a_initial)


#INITIAL E AND L
E0 = sim.energy()
Lx0, Ly0, Lz0 = sim.angular_momentum()
L0 = np.sqrt(Lx0**2 + Ly0**2 + Lz0**2)


#INTEGRATION AND DATA STORAGE
n_processi = min(14, n_particles - 3)
gruppi = np.array_split(np.arange(3, n_particles), n_processi * 10)

with get_context("fork").Pool(n_processi) as pool:
    risultati = pool.starmap(integration, [(indici, times) for indici in gruppi], chunksize=1)        

risultati = [(r[0], r[1], None, r[3], r[4], None, r[6]) for r in risultati]

r_geo = np.concatenate([r[1] for r in risultati], axis=1)
a = np.concatenate([r[3] for r in risultati], axis=1)
e = np.concatenate([r[4] for r in risultati], axis=1)
stability_max = np.concatenate([r[6] for r in risultati])
positions[0, :3] = [[p.x, p.y, p.z] for p in sim.particles[:3]]

del risultati

print("Calcolo separato della ripetitivita delle oscillazioni di a ed e...", flush=True)
stable_indices = np.flatnonzero(stability_max >= 0)
periodicity_a, recurrence_time_a, linear_change_a, linear_fraction_a = element_periodicity(
    a, times, stable_indices, reference=a_initial
)
periodicity_e, recurrence_time_e, linear_change_e, linear_fraction_e = element_periodicity(
    e, times, stable_indices
)




#ORBIT PLOT

# plt.figure(figsize=(9,8))

# plt.plot(positions[:, 0, 0]/AU, positions[:, 0, 1]/AU, color='yellow', linewidth=2)
# plt.plot(positions[:, 1, 0]/AU, positions[:, 1, 1]/AU, color='blue', linewidth=2)
# plt.plot(positions[:, 2, 0]/AU, positions[:, 2, 1]/AU, color='gray', linewidth=1)

# plt.xlabel("X [AU]")
# plt.ylabel("Y [AU]")
# plt.title("Planetary System")
# plt.show()


# Scala comune per i valori di stability delle particelle stabili.
valid_scores = stability_max[np.isfinite(stability_max) & (stability_max >= 0)]
vmax = float(valid_scores.max()) if valid_scores.size else 1.0
#norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax if vmax > 0 else 1)
cmap = plt.get_cmap("RdYlGn_r").copy()
cmap.set_bad("gray")
scores = np.ma.masked_invalid(stability_max)
scores = np.ma.masked_less(scores, 0)
colors = cmap(norm(scores))

# Cause di scarto ricavate dagli array esistenti. Le rimozioni hanno priorita
# sulle soglie OE se una particella soddisfa entrambi i criteri.
# "merge" agisce anche su Sole e Terra: senza un registro degli eventi,
# una rimozione non identifica da sola un ingresso nel filtro lunare.
discarded_removed = np.any(~np.isfinite(r_geo), axis=(0, 2))
discarded_oe = np.any((e >= 0.5) | (a >= 400000), axis=0) & ~discarded_removed

# Raggi kepleriani nominali delle risonanze retrograde p:-q, p e q <= 5.
# Con e iniziale nullo, il semiasse iniziale coincide con il raggio disegnato.
from math import gcd

r_moon_initial = moon_x - earth_x
v_moon_initial = moon_v - earth_v
mu_earth_moon = sim.G * (M_EARTH + M_MOON)
a_moon_initial = 1 / (
    2 / np.linalg.norm(r_moon_initial)
    - np.dot(v_moon_initial, v_moon_initial) / mu_earth_moon
)
resonances = []
for p in range(1, 6):
    for q in range(1, 6):
        if gcd(p, q) != 1:
            continue
        radius = (
            a_moon_initial
            * (M_EARTH / (M_EARTH + M_MOON))**(1 / 3)
            * (q / p)**(2 / 3)
        )
        if distance_range.min() <= radius <= distance_range.max():
            resonances.append((radius, p, q))
resonances.sort()

t_years = times / YR
groups = (("stabili", np.flatnonzero(stability_max >= 0)), ("instabili", np.flatnonzero(stability_max == -1)))

output_dir = Path(__file__).resolve().parent

for name, indices in groups:
    title = f"Particelle {name} ({indices.size})"

    # TRAIETTORIE GEOCENTRICHE
    if name == "stabili":
        fig_orbits, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        ax.scatter(
            r_geo[0, discarded_oe, 0], r_geo[0, discarded_oe, 1],
            color="lightgray", s=1, alpha=1, zorder=1,
            label=f"Scarti soglie OE ({np.count_nonzero(discarded_oe)})",
        )
        ax.scatter(
            r_geo[0, discarded_removed, 0], r_geo[0, discarded_removed, 1],
            color="dimgray", s=1, alpha=1, zorder=1,
            label=f"Rimosse: filtro lunare/collisioni ({np.count_nonzero(discarded_removed)})",
        )
        stable_points = ax.scatter(r_geo[0, indices, 0], r_geo[0, indices, 1], c=colors[indices], s=1, alpha=1, zorder=2)

        circle_angle = np.linspace(0, 2 * np.pi, 720)
        label_angles = np.linspace(25, 305, len(resonances))
        for k, (radius, p, q) in enumerate(resonances):
            ax.plot(
                radius * np.cos(circle_angle), radius * np.sin(circle_angle),
                "k--", lw=0.7, alpha=0.8, zorder=3,
                label="Risonanze retrograde nominali" if k == 0 else "_nolegend_",
            )
            angle = np.radians(label_angles[k])
            ax.text(
                radius * np.cos(angle), radius * np.sin(angle),
                rf"${p}:-{q}$", fontsize=8, ha="center", va="center", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
            )

        ax.scatter(positions[0, 2, 0] - positions[0, 1, 0], positions[0, 2, 1] - positions[0, 1, 1], color="grey", s=20, label="Luna iniziale", zorder=5)
        colorbar = fig_orbits.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"$\max_t\sqrt{e(t)^2+\left(\frac{a(t)-a_0}{a_0}\right)^2}$")
        ax.scatter(0, 0, color="blue", s=20, label="Terra", zorder=5)
        ax.legend(fontsize=7, loc="upper right")
        ax.set(
            xlabel="X geocentrica [km]", ylabel="Y geocentrica [km]",
            title=f"Posizioni iniziali — {title}, con scarti\n"
                  f"Filtro lunare: {sim.particles[2].r:,.0f} km; risonanze kepleriane nominali",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        fig_orbits.savefig(output_dir / f"traiettorie_{name}_2.png", dpi=300, bbox_inches="tight")

        # Mappe distinte delle oscillazioni residue dopo il fit lineare.
        periodicity_norm = Normalize(vmin=0, vmax=1)
        periodicity_cmap = plt.get_cmap("RdYlGn").copy()
        periodicity_cmap.set_bad("black")
        missing_handle = None
        for element, values in (("a", periodicity_a), ("e", periodicity_e)):
            selected_periodicity = np.ma.masked_invalid(values[indices])
            stable_points.set_color(periodicity_cmap(periodicity_norm(selected_periodicity)))
            colorbar.update_normal(plt.cm.ScalarMappable(norm=periodicity_norm, cmap=periodicity_cmap))
            colorbar.set_label(rf"Ripetitività $\mathcal{{P}}_{element}$: 0 = bassa, 1 = alta")
            if missing_handle is not None:
                missing_handle.remove()
                missing_handle = None
            missing = np.count_nonzero(~np.isfinite(values[indices]))
            if missing:
                missing_handle = ax.scatter([], [], color="black", s=4,
                                           label=f"Oscillazioni assenti/non valutabili ({missing})")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(
                f"Posizioni iniziali — Ripetitività delle oscillazioni di {element}\n"
                f"Tendenza lineare rimossa; almeno 3 cicli; T = {(times[-1] - times[0]) / YR:g} anni\n"
                f"Ritorni: {4 * (times[1] - times[0]) / DAY:.1f}–{(times[-1] - times[0]) / (3 * DAY):.1f} giorni"
            )
            fig_orbits.savefig(output_dir / f"traiettorie_periodicita_{element}_2.png", dpi=300, bbox_inches="tight")

        # La tendenza resta visibile a parte, nelle unita fisiche di ogni OE.
        fig_trend, trend_axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        trend_cmap = plt.get_cmap("RdBu_r").copy()
        trend_cmap.set_bad("black")
        for trend_ax, element, values, unit in zip(
            trend_axes, ("a", "e"), (linear_change_a, linear_change_e), (" [km]", "")
        ):
            selected = np.ma.masked_invalid(values[indices])
            finite = selected.compressed()
            limit = float(np.max(np.abs(finite))) if finite.size else 1.0
            trend_norm = Normalize(vmin=-limit if limit > 0 else -1, vmax=limit if limit > 0 else 1)
            trend_ax.scatter(r_geo[0, discarded_oe, 0], r_geo[0, discarded_oe, 1],
                             color="lightgray", s=1, label="Scarti soglie OE")
            trend_ax.scatter(r_geo[0, discarded_removed, 0], r_geo[0, discarded_removed, 1],
                             color="dimgray", s=1, label="Rimosse: filtro lunare/collisioni")
            trend_ax.scatter(r_geo[0, indices, 0], r_geo[0, indices, 1],
                             c=trend_cmap(trend_norm(selected)), s=1, zorder=2)
            for k, (radius, p, q) in enumerate(resonances):
                trend_ax.plot(radius * np.cos(circle_angle), radius * np.sin(circle_angle),
                              "k--", lw=0.7, alpha=0.8, zorder=3,
                              label="Risonanze retrograde nominali" if k == 0 else "_nolegend_")
                angle = np.radians(label_angles[k])
                trend_ax.text(radius * np.cos(angle), radius * np.sin(angle), rf"${p}:-{q}$",
                              fontsize=7, ha="center", va="center", zorder=4,
                              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
            trend_ax.scatter(positions[0, 2, 0] - positions[0, 1, 0],
                             positions[0, 2, 1] - positions[0, 1, 1], color="grey", s=20, label="Luna iniziale", zorder=5)
            trend_ax.scatter(0, 0, color="blue", s=20, label="Terra", zorder=5)
            trend_ax.set(xlabel="X geocentrica [km]", ylabel="Y geocentrica [km]",
                         title=rf"Variazione lineare stimata: $\Delta {element}_{{\rm lin}}${unit}")
            trend_ax.set_aspect("equal", adjustable="box")
            trend_ax.grid(alpha=0.3)
            trend_ax.legend(fontsize=6, loc="upper right")
            fig_trend.colorbar(plt.cm.ScalarMappable(norm=trend_norm, cmap=trend_cmap), ax=trend_ax,
                              label=rf"$\Delta {element}_{{\rm lin}} = b_{element} T${unit}")
        fig_trend.suptitle(f"Tendenze lineari stimate nella finestra di {(times[-1] - times[0]) / YR:g} anni")
        fig_trend.savefig(output_dir / "traiettorie_tendenze_lineari_2.png", dpi=300, bbox_inches="tight")

    if False:
        fig_orbits, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        for j in indices:
            ax.plot(r_geo[:, j, 0], r_geo[:, j, 1], color=colors[j], lw=0.7, alpha=0.5)

        ax.scatter(0, 0, color="blue", s=20, label="Terra")
        ax.legend()
        ax.set(xlabel="X geocentrica [km]", ylabel="Y geocentrica [km]", title=f"Traiettorie — {title}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    if False:
        # ELEMENTI ORBITALI GEOCENTRICI
        fig_oe, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        for j in indices:
            axes[0].plot(t_years, a[:, j], color=colors[j], lw=0.7, alpha=0.5)
            axes[1].plot(t_years, e[:, j], color=colors[j], lw=0.7, alpha=0.5)
            axes[2].scatter(t_years, w_plot[:, j], color=colors[j], s=1, alpha=0.5)

        axes[0].set_ylabel("Semiasse maggiore [km]")
        axes[1].set_ylabel("Eccentricità")
        axes[2].set_ylabel(r"$\omega$ [deg]")
        axes[2].set_ylim(-180, 180)
        axes[2].set_xlabel("Tempo [anni]")
        fig_oe.suptitle(f"Elementi orbitali — {title}")

        for axis in axes:
            axis.grid(True, alpha=0.3)

        if not indices.size:
            ax.text(0.5, 0.5, "Nessuna particella", transform=ax.transAxes, ha="center", va="center")
            axes[0].text(0.5, 0.5, "Nessuna particella", transform=axes[0].transAxes, ha="center", va="center")
        elif name == "stabili":
            for figure, targets in ((fig_orbits, ax), (fig_oe, axes)):
                figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=targets, label=r"$\max_{t}\sqrt{e(t)^2 + \left(\frac{a(t)-a_0}{a_0}\right)^2}$")
        else:
            ax.plot([], [], color="gray", label="Stability = -1")
            ax.legend()
            axes[0].plot([], [], color="gray", label="Stability = -1")
            axes[0].legend()

    #fig_orbits.savefig(output_dir / f"traiettorie_{name}_2.png", dpi=300, bbox_inches="tight")
    #fig_oe.savefig(output_dir / f"elementi_orbitali_{name}_2.png", dpi=300, bbox_inches="tight")

np.savez_compressed(
    output_dir / "stability_periodicity_values.npz",
    r0=r_geo[0], stability_max=stability_max,
    periodicity_a=periodicity_a, periodicity_e=periodicity_e,
    recurrence_days_a=recurrence_time_a / DAY, recurrence_days_e=recurrence_time_e / DAY,
    linear_change_a_km=linear_change_a, linear_change_e=linear_change_e,
    linear_variance_fraction_a=linear_fraction_a, linear_variance_fraction_e=linear_fraction_e,
    discarded_oe=discarded_oe,
    discarded_removed=discarded_removed, duration=times[-1] - times[0],
    sample_interval=times[1] - times[0], min_return_samples=4, min_cycles=3,
    metric="separate_a_e_linear_detrend_cycle_prediction_v2",
)

print(f"Grafici salvati in: {output_dir}")



elapsed_time = time.perf_counter() - start_time
print(f"Execution time: {elapsed_time:.2f} s\n")
