import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import pykep as pk
import spiceypy as spice
from Celestial_Mechanics import Integration, Transformations
import time
import hashlib
import inspect
from multiprocessing import get_context
from matplotlib.colors import PowerNorm



start_time = time.perf_counter()



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



# MEGNO della singola particella, mantenendo fisse le variazioni dei primari.
def integration_megno(j):
    j = int(j)
    sim_megno = megno_template.copy()
    sim_megno.add(sim.particles[j + 3])
    sim_megno.collision_resolve = "halt"
    # Seed diverso per orbita e riproducibile, indipendente dal processo worker.
    sim_megno.init_megno(seed=j + 1)

    for k in range(3):
        variation = sim_megno.particles_var[k]
        variation.x = variation.y = variation.z = 0.0
        variation.vx = variation.vy = variation.vz = 0.0
        variation.m = 0.0

    try:
        sim_megno.integrate(times[-1])
    except rebound.Collision:
        return j, np.nan

    value = sim_megno.megno()
    return j, value if np.isfinite(value) else np.nan


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
Tmax = 2*YR
Tmin = 0
dt = DAY
N = int((Tmax - Tmin)/dt)
times = np.linspace(Tmin, Tmax, N)


# Riusa la prima passata salvata; False forza un nuovo test di stabilita.
reuse_stability = True
output_dir = Path(__file__).resolve().parent
cache_path = output_dir / "stability_megno_values.npz"

n_particles = len(sim.particles)
positions = np.array([[[p.x, p.y, p.z] for p in sim.particles[:3]]])
a_initial = np.asarray(a_initial)

# Identifica modello, stati iniziali, campionamento e criterio della prima passata.
# Il seed MEGNO e le impostazioni dei grafici non invalidano questa cache.
initial_states = np.array([
    [p.m, p.r, p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles
])
signature = hashlib.sha256()
for values in (initial_states, times, [JD, sim.G, sim.N_active,
                                      sim.testparticle_type, sim.integrator.epsilon]):
    signature.update(np.asarray(values, dtype=np.float64).tobytes())
signature.update(f"{rebound.__version__}|{sim.integrator}|{sim.gravity}|{sim.collision}".encode())
signature.update(inspect.getsource(integration).encode())
signature.update(inspect.getsource(transformations.coe_from_sv).encode())
stability_signature = signature.hexdigest()
stability_metadata = {"stability_signature": stability_signature}


#INITIAL E AND L
E0 = sim.energy()
Lx0, Ly0, Lz0 = sim.angular_momentum()
L0 = np.sqrt(Lx0**2 + Ly0**2 + Lz0**2)


#INTEGRATION AND DATA STORAGE
n_processi = min(14, n_particles - 3)
if reuse_stability and cache_path.exists():
    with np.load(cache_path, allow_pickle=False) as data:
        r0 = initial_states[3:, 2:5] - initial_states[1, 2:5]
        expected_shape = (n_particles - 3,)
        compatible = (
            data["r0"].shape == r0.shape
            and np.allclose(data["r0"], r0, rtol=0, atol=1e-6)
            and np.isclose(data["duration"].item(), times[-1] - sim.t, rtol=0, atol=1e-6)
            and all(data[key].shape == expected_shape for key in
                    ("stability_max", "discarded_oe", "discarded_removed"))
        )
        if "stability_signature" in data:
            compatible = compatible and data["stability_signature"].item() == stability_signature
        else:
            stability_metadata = {}  # Non certificare retroattivamente il modello della cache precedente.
            print("Cache precedente: verificati posizioni e durata; modello e filtri non registrati.", flush=True)
        if not compatible:
            raise ValueError("Cache di stabilita incompatibile: imposta reuse_stability = False per ricalcolarla.")
        # La cache contiene le sole posizioni iniziali, senza le serie temporali OE.
        r_geo = data["r0"][None, :, :]
        stability_max = data["stability_max"]
        discarded_oe = data["discarded_oe"].astype(bool)
        discarded_removed = data["discarded_removed"].astype(bool)
        if (not np.all(np.isfinite(stability_max))
                or np.any((stability_max < 0) & (stability_max != -1))
                or np.any(discarded_oe & discarded_removed)
                or not np.array_equal(discarded_oe | discarded_removed, stability_max == -1)):
            raise ValueError("Classificazione nella cache non valida: imposta reuse_stability = False.")
    print(f"Stabilita caricata da {cache_path}: prima integrazione saltata.", flush=True)
else:
    gruppi = np.array_split(np.arange(3, n_particles), n_processi * 10)
    with get_context("fork").Pool(n_processi) as pool:
        risultati = pool.starmap(integration, [(indici, times) for indici in gruppi], chunksize=1)

    r_geo = np.concatenate([r[1] for r in risultati], axis=1)
    v_geo = np.concatenate([r[2] for r in risultati], axis=1)
    a = np.concatenate([r[3] for r in risultati], axis=1)
    e = np.concatenate([r[4] for r in risultati], axis=1)
    w_plot = np.concatenate([r[5] for r in risultati], axis=1)
    stability_max = np.concatenate([r[6] for r in risultati])
    del risultati

    # Le rimozioni hanno priorita sulle soglie OE. "merge" agisce anche su
    # Sole e Terra: una rimozione non identifica da sola il filtro lunare.
    discarded_removed = np.any(~np.isfinite(r_geo), axis=(0, 2))
    discarded_oe = np.any((e >= 0.5) | (a >= 400000), axis=0) & ~discarded_removed

# Seconda passata: una simulazione indipendente per ogni orbita sopravvissuta.
# La copia del template grande viene effettuata una sola volta nel padre.
megno_template = sim.copy()
for k in range(megno_template.N - 1, 2, -1):
    megno_template.remove(k)

megno_values = np.full(stability_max.shape, np.nan)
megno_indices = np.flatnonzero(stability_max >= 0)
if megno_indices.size:
    print(f"MEGNO: {megno_indices.size} orbite da integrare", flush=True)
    with get_context("fork").Pool(min(n_processi, megno_indices.size)) as pool:
        for completed, (j, value) in enumerate(
            pool.imap_unordered(integration_megno, megno_indices, chunksize=1), 1
        ):
            megno_values[j] = value
            if completed % 1000 == 0 or completed == megno_indices.size:
                print(f"MEGNO: {completed}/{megno_indices.size}", flush=True)








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

        # Stessa figura e stessi punti: cambia soltanto la misura rappresentata.
        megno_selected = np.ma.masked_invalid(megno_values[indices])
        finite_megno = megno_selected.compressed()
        megno_vmin = min(0.0, float(finite_megno.min())) if finite_megno.size else 0.0
        megno_vmax = max(4.0, float(finite_megno.max())) if finite_megno.size else 4.0
        megno_norm = Normalize(vmin=megno_vmin, vmax=megno_vmax)
        megno_cmap = plt.get_cmap("viridis").copy()
        megno_cmap.set_bad("black")
        stable_points.set_color(megno_cmap(megno_norm(megno_selected)))
        colorbar.update_normal(plt.cm.ScalarMappable(norm=megno_norm, cmap=megno_cmap))
        colorbar.set_label(r"MEGNO $\langle Y\rangle(T)$")
        missing_megno = np.count_nonzero(~np.isfinite(megno_values[indices]))
        if missing_megno:
            ax.scatter([], [], color="black", s=4,
                       label=f"MEGNO non disponibile ({missing_megno})")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(
            f"Posizioni iniziali — MEGNO, {indices.size} orbite, con scarti\n"
            f"T = {(times[-1] - sim.t) / YR:g} anni; "
            f"filtro lunare: {sim.particles[2].r:,.0f} km"
        )
        fig_orbits.savefig(output_dir / "traiettorie_megno_2.png", dpi=300, bbox_inches="tight")


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

np.savez_compressed(cache_path,
                    r0=r_geo[0], stability_max=stability_max, megno=megno_values,
                    discarded_oe=discarded_oe, discarded_removed=discarded_removed,
                    megno_seed=np.arange(1, megno_values.size + 1, dtype=np.uint32),
                    duration=times[-1] - sim.t, **stability_metadata)

print(f"Grafici salvati in: {output_dir}")



elapsed_time = time.perf_counter() - start_time
print(f"Execution time: {elapsed_time:.2f} s\n")
