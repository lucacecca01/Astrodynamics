"""Seed a GA16 plot recovery from individually validated compatible orbits."""

import json
from pathlib import Path

import numpy as np

from ga16_recover_orbit_plots import _array_digest, _open_cache, load_original_context


def seed_compatible_orbits(source_script, target_directory, previous_directory,
                           source_ids, states, all_states, catalog, *, batch_size=65536):
    """Copy matching source IDs only after validating states and saved events."""
    previous_directory = Path(previous_directory)
    source_ids = np.asarray(source_ids)
    states = np.asarray(states)
    catalog = np.asarray(catalog)
    if (source_ids.ndim != 1 or not len(source_ids)
            or not np.issubdtype(source_ids.dtype, np.integer)
            or source_ids[0] < 0 or np.any(np.diff(source_ids) <= 0)
            or source_ids[-1] >= len(all_states)
            or states.shape != (len(source_ids), 6)
            or catalog.shape != (len(source_ids), 12)
            or not np.array_equal(catalog[:, 0], source_ids)):
        raise ValueError("Identificativi, stati o righe del catalogo di destinazione non validi.")
    for start in range(0, len(source_ids), batch_size):
        stop = start+batch_size
        if (not np.isfinite(states[start:stop]).all()
                or not np.array_equal(states[start:stop], all_states[source_ids[start:stop]])):
            raise ValueError("Gli stati di destinazione non corrispondono al database originale.")
    context = load_original_context(source_script)
    metadata = json.loads((previous_directory / "metadata.json").read_text())
    if metadata["source_fingerprint"] != context["recovery_fingerprint"]:
        raise ValueError("L'archivio precedente usa un'integrazione diversa.")
    arrays = {key: np.load(previous_directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
              for key in ("source_ids", "curves", "bounds", "features", "done")}
    previous_ids = arrays["source_ids"]
    if (not len(previous_ids) or not np.issubdtype(previous_ids.dtype, np.integer)
            or previous_ids[0] < 0 or np.any(np.diff(previous_ids) <= 0)
            or previous_ids[-1] >= len(all_states)
            or _array_digest(previous_ids) != metadata["source_ids_sha256"]):
        raise ValueError("Identificativi dell'archivio precedente non validi.")
    if _array_digest(all_states[previous_ids]) != metadata["states_sha256"]:
        raise ValueError("Gli stati originali sono cambiati: impossibile riusare le orbite.")
    target = _open_cache(target_directory, source_ids, states, context)
    reused = 0
    for start in range(0, len(source_ids), batch_size):
        stop = min(start + batch_size, len(source_ids))
        old = np.searchsorted(previous_ids, source_ids[start:stop])
        valid = old < len(previous_ids)
        valid[valid] &= previous_ids[old[valid]] == source_ids[start:stop][valid]
        valid[valid] &= arrays["done"][old[valid]]
        valid &= ~target.done[start:stop]
        rows = np.flatnonzero(valid) + start
        old = old[valid]
        if not len(rows):
            continue
        bounds = arrays["bounds"][old]
        features = arrays["features"][old]
        expected = catalog[rows]
        days = (bounds-context["T_min"]) * context["TU"]/context["day"]
        if not np.allclose(days, expected[:, 5:7], rtol=1e-9, atol=1e-8):
            raise ValueError("Tempi degli eventi incompatibili nell'archivio precedente.")
        if (not np.allclose(features[:, :2], expected[:, 9:11], rtol=1e-9, atol=1e-8)
                or not np.allclose(features[:, 3:6], expected[:, 2:5], rtol=1e-9, atol=1e-8)
                or not np.allclose((features[:, 2]-expected[:, 11]+180)%360-180, 0, atol=1e-7)):
            raise ValueError("Elementi o velocita SOI incompatibili nell'archivio precedente.")
        curves = arrays["curves"][old]
        x = curves[:, 0, -1]+context["mu"]
        y = curves[:, 1, -1]
        c, s = np.cos(bounds[:, 1]), np.sin(bounds[:, 1])
        positions = context["d"]*np.column_stack((c*x-s*y, s*x+c*y))
        if not np.isfinite(curves).all() or not np.allclose(positions, expected[:, 7:9], rtol=1e-9, atol=1e-4):
            raise ValueError("Posizioni SOI incompatibili nell'archivio precedente.")
        target.curves[rows] = curves
        target.bounds[rows] = bounds
        target.features[rows] = features
        # Commit samples before declaring these rows reusable on the next run.
        for name in ("curves", "bounds", "features"):
            getattr(target, name).flush()
        target.done[rows] = True
        target.done.flush()
        reused += len(rows)
        if start == 0 or stop == len(source_ids) or (start//batch_size) % 8 == 0:
            print(f"Orbite precedenti validate e riusate: {np.count_nonzero(target.done):,}/{len(source_ids):,}", flush=True)
    report = dict(previous=str(previous_directory.resolve()), newly_reused=reused,
                  ready=int(np.count_nonzero(target.done)), total=len(source_ids),
                  remaining=int(np.count_nonzero(~target.done)))
    (Path(target_directory) / "reuse_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Riutilizzo concluso: {report['ready']:,} pronte, {report['remaining']:,} da ricostruire.", flush=True)
    return report
