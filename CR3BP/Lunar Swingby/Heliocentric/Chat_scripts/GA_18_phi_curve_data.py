"""Recover verified CR3BP curves for the epoch-dependent perigee plots.

Each original trajectory is integrated at most once. Its rotations at different
winning ETDs are applied by the renderer, not saved as duplicate curves here.
"""

import hashlib
import inspect
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

import GA_18_plots as plots
import GA_18_high_occurrence_trajectories as high
from Celestial_Mechanics import Integration


WORKSPACE = Path(__file__).resolve().parents[4]
CHUNK = 2048


def _source_file(catalog_path):
    catalog_path = Path(catalog_path)
    member_path = (catalog_path if catalog_path.name == "trajectory_exit_catalog.txt"
                   else catalog_path.with_name("trajectory_exit_catalog.txt"))
    with member_path.open() as stream:
        for line in stream:
            if line.startswith("# Source:"):
                source = Path(line.split(":", 1)[1].strip())
                return source if source.is_absolute() else member_path.parent / source
            if line.strip() and not line.lstrip().startswith("#"):
                break
    raise ValueError(f"Manca Source nell'header di {member_path}.")


def _read_states(source, ids):
    """Read selected zero-based numeric rows directly from the original text."""
    states = np.empty((len(ids), 6), dtype=float)
    if not len(ids):
        return states
    wanted = 0
    numeric_row = -1
    last_report = time.monotonic()
    with source.open() as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            numeric_row += 1
            if numeric_row == ids[wanted]:
                values = stripped.split(maxsplit=6)
                if len(values) < 6:
                    raise ValueError(f"Stato incompleto nella riga {numeric_row} di {source}.")
                states[wanted] = [float(value) for value in values[:6]]
                wanted += 1
                if wanted == len(ids):
                    break
            now = time.monotonic()
            if now - last_report > 25:
                print(f"Stati originali: {wanted}/{len(ids)} letti, riga {numeric_row}.", flush=True)
                last_report = now
    if wanted != len(ids):
        raise ValueError(f"source_id {ids[wanted]} assente dal database originale {source}.")
    if not np.isfinite(states).all():
        raise ValueError("Stati originali non finiti.")
    return states


def _valid_curves(curves, features, members, states, n_plot):
    """Validate geometry and saved osculating elements, independently per row."""
    mu = plots.MOON_MASS_KG / (5.974e24 + plots.MOON_MASS_KG)
    bounds = members[:, 5:7] * 86400 / plots.TIME_UNIT_S
    expected_etd = states[:, :2].copy()
    expected_etd[:, 0] += mu
    expected_etd *= plots.EARTH_MOON_DISTANCE_KM
    valid = np.isfinite(curves).all(axis=(1, 2)) & np.isfinite(features).all(axis=1)
    valid &= np.linalg.norm(curves[:, :, n_plot-1] - expected_etd, axis=1) <= 1e-7
    for sample, time_index in ((0, 0), (-1, 1)):
        c, s = np.cos(bounds[:, time_index]), np.sin(bounds[:, time_index])
        x = c * curves[:, 0, sample] - s * curves[:, 1, sample]
        y = s * curves[:, 0, sample] + c * curves[:, 1, sample]
        if sample == -1:
            valid &= np.hypot(x-members[:, 7], y-members[:, 8]) <= 1.0
        else:
            eps, ecc = members[:, 9], members[:, 10]
            delta = (np.degrees(np.arctan2(y, x)) - features[:, 3]*members[:, 11] + 180) % 360 - 180
            # A minimum of Earth distance in the CR3BP can be the APOCENTER
            # of the instantaneous Earth-only osculating ellipse. Validate
            # its conic radius and apsidal axis, not r == a*(1-e).
            semilatus = np.full(len(members), np.nan)
            np.divide(-plots.MU_EARTH * (1-ecc**2), 2*eps,
                      out=semilatus, where=eps != 0)
            denominator = 1 + ecc*np.cos(np.radians(delta))
            expected_radius = np.full(len(members), np.nan)
            np.divide(semilatus, denominator, out=expected_radius, where=denominator > 0)
            valid &= np.isfinite(expected_radius) & (expected_radius > 0)
            valid &= np.abs(np.hypot(x, y)-expected_radius) <= 1.0
            apsidal_axis_error = (delta+90) % 180 - 90
            valid &= (ecc <= 1e-8) | (np.abs(apsidal_axis_error) <= 0.02)
    valid &= np.isclose(features[:, 0], members[:, 9], rtol=1e-5, atol=1e-6)
    valid &= np.isclose(features[:, 1], members[:, 10], rtol=1e-5, atol=1e-6)
    valid &= np.abs((features[:, 2]-members[:, 11]+180) % 360-180) <= 0.02
    valid &= np.isin(features[:, 3], (-1, 1))
    return valid


def prepare_curves(members, catalog_path, output, workers=4, n_plot=101, reusable_products=()):
    """Return (curves, provenance) with geocentric synodic positions in km.

    ``members`` must be unique and sorted by source_id. Completed old dense
    integrations are reused only after checking their original source states,
    full member rows, numerical model, sampling and endpoint/element values.
    Newly recovered data are explicit products in ``output``.
    """
    members = np.asarray(members, dtype=float)
    if members.ndim != 2 or members.shape[1] < 12 or not np.isfinite(members).all():
        raise ValueError("Membri attesi: matrice finita con almeno 12 colonne.")
    ids = members[:, 0].astype(np.int64)
    if np.any(ids < 0) or np.any(ids != members[:, 0]) or np.any(np.diff(ids) <= 0):
        raise ValueError("source_id devono essere interi, unici, non negativi e ordinati.")
    if int(n_plot) != n_plot or n_plot < 2 or int(workers) != workers or workers < 1:
        raise ValueError("n_plot >= 2 e workers >= 1 devono essere interi.")
    n_plot, workers = int(n_plot), int(workers)
    bounds = members[:, 5:7] * 86400 / plots.TIME_UNIT_S
    if np.any(bounds[:, 0] >= 0) or np.any(bounds[:, 1] <= 0) or np.any(np.abs(bounds) > 4*np.pi):
        raise ValueError("Eventi attesi: perigeo prima di ETD e SOI dopo ETD, entro il dominio GA16 di ±4π TU.")
    source = _source_file(catalog_path).resolve()
    source_stat = source.stat()
    rhs = hashlib.sha256((inspect.getsource(high.cr3bp_rhs)
                         + inspect.getsource(Integration.CR3BP_ODE)
                         + inspect.getsource(high._integrate_dense)).encode()).hexdigest()
    signature = dict(n_plot=n_plot, rtol=1e-9, atol=1e-12, rhs=rhs,
                     source=str(source), source_size=source_stat.st_size,
                     source_mtime_ns=source_stat.st_mtime_ns,
                     members_sha256=hashlib.sha256(np.ascontiguousarray(members).tobytes()).hexdigest())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "reconstructed_curves.npy"
    done_path = output / "reconstructed_done.npy"
    features_path = output / "reconstructed_features.npy"
    report_path = output / "curve_provenance.json"
    shape = (len(ids), 2, 2*n_plot-1)
    if path.exists():
        if not report_path.exists() or json.loads(report_path.read_text()).get("signature") != signature:
            raise ValueError(f"{path}: dati di una selezione o sorgente diversa; usa un'altra cartella di output.")
        curves = np.load(path, mmap_mode="r+")
        done = np.load(done_path, mmap_mode="r+")
        features = np.load(features_path, mmap_mode="r+")
        if curves.shape != shape or done.shape != (len(ids),) or features.shape != (len(ids), 4):
            raise ValueError("Forma dei prodotti di ricostruzione incoerente.")
    else:
        curves = np.lib.format.open_memmap(path, mode="w+", dtype=float, shape=shape)
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=bool, shape=(len(ids),))
        features = np.lib.format.open_memmap(features_path, mode="w+", dtype=float, shape=(len(ids), 4))
        done[:] = False
        done.flush()
    report = dict(signature=signature, curves_path=str(path.resolve()),
                  coordinates="Earth-centered synodic, km", samples=2*n_plot-1,
                  trajectories=len(ids), verified_existing=0, reused=[], integrated=0,
                  validation="Original source states; full member rows; ETD <=1e-7 km; SOI and radial-event conic radius <=1 km; omega and apsidal axis <=0.02 deg; eps/e rtol=1e-5, atol=1e-6",
                  radial_event_note="A CR3BP minimum Earth distance may coincide with the osculating apocenter; phi is the osculating pericenter direction, not necessarily the radial-event position angle.")

    def checkpoint():
        curves.flush()
        features.flush()
        done.flush()
        report["completed"] = int(done.sum())
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    checkpoint()
    print(f"Lettura degli stati originali per {len(ids)} traiettorie...", flush=True)
    states = _read_states(source, ids)
    current_stat = source.stat()
    if (current_stat.st_size, current_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise ValueError("Il database originale è cambiato durante la lettura.")
    for start in range(0, len(ids), CHUNK):
        indices = np.flatnonzero(done[start:start+CHUNK]) + start
        if len(indices):
            done[indices] = _valid_curves(curves[indices], features[indices], members[indices], states[indices], n_plot)
    report["verified_existing"] = int(done.sum())
    checkpoint()

    # Other explicit plot products can contain curves missing from the older
    # dense integrations. Match their provenance and validate against the source.
    for directory in map(Path, reusable_products):
        if done.all():
            break
        provenance_path = directory / "curve_provenance.json"
        selection_path = directory / "selezione_istanze.npz"
        if not selection_path.is_file():
            selection_path = directory / "membri_selezionati.npz"
        if not provenance_path.is_file() or not selection_path.is_file():
            continue
        previous_report = json.loads(provenance_path.read_text())
        previous_signature = previous_report.get("signature", {})
        matching_keys = ("n_plot", "rtol", "atol", "rhs", "source", "source_size", "source_mtime_ns")
        if any(previous_signature.get(key) != signature[key] for key in matching_keys):
            continue
        with np.load(selection_path) as data:
            old_members = data["members"]
        digest = hashlib.sha256(np.ascontiguousarray(old_members).tobytes()).hexdigest()
        if digest != previous_signature.get("members_sha256"):
            raise ValueError(f"{directory}: membri e provenienza delle curve non corrispondono.")
        if old_members.ndim != 2 or old_members.shape[1] != members.shape[1] or not len(old_members):
            continue
        old_ids = old_members[:, 0]
        if np.any(np.diff(old_ids) <= 0):
            raise ValueError(f"{directory}: source_id non unici e ordinati.")
        old_curves = np.load(directory / "reconstructed_curves.npy", mmap_mode="r")
        old_features = np.load(directory / "reconstructed_features.npy", mmap_mode="r")
        old_done = np.load(directory / "reconstructed_done.npy", mmap_mode="r")
        if (old_curves.shape != (len(old_members), 2, 2*n_plot-1)
                or old_features.shape != (len(old_members), 4) or old_done.shape != (len(old_members),)):
            raise ValueError(f"{directory}: forma dei prodotti delle curve incoerente.")
        reused = 0
        for start in range(0, len(ids), CHUNK):
            current = np.flatnonzero(~done[start:start+CHUNK]) + start
            old = np.searchsorted(old_ids, ids[current])
            present = old < len(old_ids)
            current, old = current[present], old[present]
            matching = (old_ids[old] == ids[current]) & old_done[old]
            current, old = current[matching], old[matching]
            matching = np.all(old_members[old] == members[current], axis=1)
            current, old = current[matching], old[matching]
            if not len(current):
                continue
            part, perigee_features = old_curves[old], old_features[old]
            valid = _valid_curves(part, perigee_features, members[current], states[current], n_plot)
            current = current[valid]
            curves[current], features[current] = part[valid], perigee_features[valid]
            done[current] = True
            reused += len(current)
        if reused:
            report["reused"].append(dict(directory=str(directory.resolve()), count=reused,
                                         original_states_checked=True))
            print(f"Curve dei grafici precedenti verificate e riutilizzate: {reused} da {directory.name}.", flush=True)
            checkpoint()

    prior_root = WORKSPACE / ".cache/ga18_high_occurrence_dense"
    previous = sorted(prior_root.glob("*/metadata.json"),
                      key=lambda p: (p.parent / "curves.npy").stat().st_size
                      if (p.parent / "curves.npy").is_file() else 0, reverse=True)
    for metadata_path in previous:
        if done.all():
            break
        directory = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != signature[key] for key in ("n_plot", "rtol", "atol", "rhs")):
            continue
        old_source_path = Path(metadata.get("source", ""))
        required = ("members.npy", "curves.npy", "features.npy", "done.npy")
        if not all((directory / name).is_file() for name in required) or not old_source_path.is_file():
            continue
        if old_source_path.stat().st_mtime_ns != metadata.get("mtime_ns"):
            continue
        old_members = np.load(directory / "members.npy", mmap_mode="r")
        if old_members.ndim != 2 or old_members.shape[1] != members.shape[1] or not len(old_members):
            continue
        old_ids = old_members[:, 0]
        if np.any(np.diff(old_ids) <= 0):
            continue
        old_curves = np.load(directory / "curves.npy", mmap_mode="r")
        old_features = np.load(directory / "features.npy", mmap_mode="r")
        old_done = np.load(directory / "done.npy", mmap_mode="r")
        old_states = np.load(old_source_path, mmap_mode="r")
        if (old_curves.shape != (len(old_members), 2, 2*n_plot-1)
                or old_features.shape != (len(old_members), 4) or old_done.shape != (len(old_members),)
                or old_states.ndim != 2 or old_states.shape[1] < 6):
            continue
        reused = 0
        for start in range(0, len(ids), CHUNK):
            current = np.flatnonzero(~done[start:start+CHUNK]) + start
            if not len(current):
                continue
            old = np.searchsorted(old_ids, ids[current])
            present = (old < len(old_ids)) & (ids[current] < len(old_states))
            current, old = current[present], old[present]
            matching = (old_ids[old] == ids[current]) & old_done[old]
            current, old = current[matching], old[matching]
            matching = (np.all(old_members[old] == members[current], axis=1)
                        & np.all(old_states[ids[current], :6] == states[current], axis=1))
            current, old = current[matching], old[matching]
            if not len(current):
                continue
            part, perigee_features = old_curves[old], old_features[old]
            valid = _valid_curves(part, perigee_features, members[current], states[current], n_plot)
            current = current[valid]
            curves[current], features[current] = part[valid], perigee_features[valid]
            done[current] = True
            reused += len(current)
        if reused:
            report["reused"].append(dict(directory=str(directory.resolve()), count=reused,
                                         original_states_checked=True))
            print(f"Curve dense verificate e riutilizzate: {reused} da {directory.name}.", flush=True)
            checkpoint()

    pending = np.flatnonzero(~done)
    print(f"Curve pronte: {int(done.sum())}/{len(ids)}; da integrare: {len(pending)}.", flush=True)
    tasks = ((int(i), states[i], *bounds[i], n_plot) for i in pending)
    pool = get_context("fork").Pool(workers) if workers > 1 and len(pending) else None
    results = pool.imap_unordered(high._integrate_dense, tasks, chunksize=8) if pool else map(high._integrate_dense, tasks)
    last_report = time.monotonic()
    try:
        for i, curve, perigee_features in results:
            curve[0] += plots.MOON_MASS_KG / (5.974e24 + plots.MOON_MASS_KG)
            curve *= plots.EARTH_MOON_DISTANCE_KM
            if not _valid_curves(curve[None], perigee_features[None], members[i:i+1], states[i:i+1], n_plot)[0]:
                raise ValueError(f"source_id {ids[i]}: reintegrazione incoerente con eventi/elementi salvati.")
            curves[i], features[i], done[i] = curve, perigee_features, True
            report["integrated"] += 1
            if time.monotonic() - last_report > 25:
                checkpoint()
                print(f"Curve ricostruite: {int(done.sum())}/{len(ids)}.", flush=True)
                last_report = time.monotonic()
    except BaseException:
        if pool:
            pool.terminate()
        raise
    finally:
        if pool:
            pool.close()
            pool.join()
        checkpoint()
    if not done.all():
        raise ValueError("Ricostruzione delle traiettorie incompleta.")
    return curves, report
