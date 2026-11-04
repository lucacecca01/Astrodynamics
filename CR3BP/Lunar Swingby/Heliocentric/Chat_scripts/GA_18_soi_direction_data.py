"""Direzioni della velocita SOI nei tre riferimenti delle mappe 05–12.

Ogni traiettoria pesa una volta per cella. Le diverse ETD dello stesso membro
sono prima mediate come vettori complessi, senza rinormalizzare il risultato.
"""

import gc
import hashlib
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np

import GA_18_plots as plots
from GA_18_catalog_azimuth_trajectories import cells_in_grid, geometry_from_cells, read_eligible_members
from GA_18_soi_speed_maps import CATALOG_TABLES, PHI_TABLES, MAPS, read_reference


FRAMES = ('synodic', 'j2000', 'catalog')
ANGLE_TOL = 1e-12


def direction_vectors(members):
    """Return (v_synodic_xy, v_catalog_xy), in km/s, at each SOI epoch.

    The synodic velocity is the derivative in rotating axes, including the
    subtraction of Omega cross r; it is not just rotated inertial components.
    """
    members = np.asarray(members)
    catalog = members[:, 2:4].copy()
    x, y = members[:, 7:9].T
    tau = members[:, 6]*86400.0/plots.TIME_UNIT_S
    ux = catalog[:, 0]+y/plots.TIME_UNIT_S
    uy = catalog[:, 1]-x/plots.TIME_UNIT_S
    c, s = np.cos(tau), np.sin(tau)
    synodic = np.column_stack((c*ux+s*uy, -s*ux+c*uy))
    return synodic, catalog


def canonical_angle(z):
    result = np.mod(np.degrees(np.angle(z)), 360.0)
    # Floating point can turn a tiny negative angle into exactly 360 degrees.
    return np.where(result >= 360.0-ANGLE_TOL, 0.0, result)


def complex_sum(cells, values, size):
    return (np.bincount(cells, weights=values.real, minlength=size)
            + 1j*np.bincount(cells, weights=values.imag, minlength=size))


def circular_statistics(cells, units, shape):
    """Average complex values per distinct source/cell; units may have norm<1."""
    size = int(np.prod(shape))
    count = np.bincount(cells, minlength=size)
    means = np.zeros((3, size), dtype=np.complex128)
    for frame, values in enumerate(units):
        np.divide(complex_sum(cells, values, size), count,
                  out=means[frame], where=count > 0)
    coherence = np.clip(np.abs(means), 0.0, 1.0)
    angles = canonical_angle(means)
    angles[(coherence < ANGLE_TOL) | (count[None, :] == 0)] = np.nan
    coherence[:, count == 0] = np.nan
    return count.reshape(shape), angles.reshape((3, *shape)), coherence.reshape((3, *shape))


def maximum_source(cells, member_indices, speeds, size):
    """Max speed, then smallest source index; members are sorted by source_id."""
    maximum = np.full(size, -np.inf)
    np.maximum.at(maximum, cells, speeds)
    best = speeds == maximum[cells]
    chosen = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(chosen, cells[best], member_indices[best])
    empty = chosen == np.iinfo(np.int64).max
    chosen[empty] = -1
    maximum[empty] = np.nan
    return chosen, maximum


def representative_fields(chosen, maximum, chosen_winner, members, syn_unit, cat_unit, phase_unit, shape):
    present = chosen >= 0
    angles = np.full((3, len(chosen)), np.nan)
    ids = chosen[present]
    jj = chosen_winner[present]
    angles[0, present] = canonical_angle(syn_unit[ids])
    angles[1, present] = canonical_angle(cat_unit[ids]*phase_unit[jj])
    angles[2, present] = canonical_angle(cat_unit[ids])
    source = np.full(len(chosen), -1, dtype=np.int64)
    source[present] = members[ids, 0].astype(np.int64)
    return dict(maximum_angle_deg=angles.reshape((3, *shape)), max_speed_km_s=maximum.reshape(shape),
                max_source_id=source.reshape(shape), max_winner_index=chosen_winner.reshape(shape))


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((CATALOG_TABLES/'riepilogo.json').read_text())
    phi_summary = json.loads((PHI_TABLES/'riepilogo.json').read_text())
    winners = summary['winners']
    if len(winners) != len(phi_summary['winners']):
        raise ValueError('Insiemi di asteroidi diversi fra le griglie di riferimento.')
    for a, b in zip(winners, phi_summary['winners']):
        if any(a[k] != b[k] for k in ('spkid', 'cluster_id', 'cluster_row')):
            raise ValueError('Asteroidi o cluster diversi fra le griglie di riferimento.')
        for key in ('jd_etd', 'jd_arrival', 'dv_km_s', 'tof_days', 'phase_etd_deg'):
            if not np.isclose(a[key], b[key], rtol=0, atol=1e-10):
                raise ValueError(f'Vincitori non coerenti: {key}.')
    catalog = Path(summary['catalog'])
    stats = plots.load_table(catalog, 15, allow_extra=True)
    stats = stats[np.argsort(stats[:, 0])]
    winner_rows = np.array([w['cluster_row'] for w in winners])
    np.testing.assert_array_equal(stats[winner_rows, 0], [w['cluster_id'] for w in winners])
    members = read_eligible_members(catalog.with_name('trajectory_exit_catalog.txt'),
                                    np.unique(stats[winner_rows, 0]))
    if len(members) != summary['population_count']:
        raise ValueError('Popolazione diversa dalle mappe gia verificate.')
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    syn, cat = direction_vectors(members)
    syn_norm, cat_norm = np.linalg.norm(syn, axis=1), np.linalg.norm(cat, axis=1)
    if np.any(syn_norm < ANGLE_TOL) or np.any(cat_norm < ANGLE_TOL):
        raise ValueError('Velocita planare nulla: direzione non definita.')
    syn_unit = (syn[:, 0]+1j*syn[:, 1])/syn_norm
    cat_unit = (cat[:, 0]+1j*cat[:, 1])/cat_norm
    speeds = np.linalg.norm(members[:, 2:5], axis=1)
    del syn, cat, syn_norm, cat_norm
    phases = np.radians([w['phase_etd_deg'] for w in winners])
    phase_unit = np.exp(1j*phases)
    frequency = np.bincount(winner_rows, minlength=len(stats))
    phase_mean = np.zeros(len(stats), dtype=np.complex128)
    phase_total = (np.bincount(winner_rows, weights=phase_unit.real, minlength=len(stats))
                   + 1j*np.bincount(winner_rows, weights=phase_unit.imag, minlength=len(stats)))
    np.divide(phase_total, frequency, out=phase_mean, where=frequency > 0)
    epoch_order = np.lexsort((np.arange(len(winners)), [w['jd_etd'] for w in winners]))
    epoch_rank = np.empty(len(winners), dtype=np.int32)
    epoch_rank[epoch_order] = np.arange(len(winners), dtype=np.int32)
    earliest = np.full(len(stats), -1, dtype=np.int64)
    for j in epoch_order:
        if earliest[winner_rows[j]] < 0:
            earliest[winner_rows[j]] = j
    eps, ecc = members[:, 9:11].T
    hp = -plots.MU_EARTH*(1-ecc)/(2*eps)-plots.R_EARTH_KM
    omega = plots.catalog_perigee_azimuth(members, stats)
    theta = (omega-plots.MOON_RATE_DEG_DAY*members[:, 5]) % 360
    coordinates = ((eps, ecc), (eps, omega), (hp, ecc), (hp, omega), (hp, theta), (eps, theta))
    records = []

    def save_map(spec, grid, count, angles, coherence, maxima):
        key, _, xlabel, ylabel = spec
        with np.load(plots.OUTPUT_DIR/'mappe_velocita_soi_100x100'/f'{key}_velocita_soi.npz') as z:
            np.testing.assert_array_equal(count, z['trajectory_count'])
            np.testing.assert_allclose(maxima['max_speed_km_s'], z['speed_maximum'],
                                       rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(count > 0, grid['occurrences'] > 0)
        np.savez_compressed(output/f'{key}_direzioni_soi.npz', **grid, count=count,
                            trajectory_count=count, mean_angle_deg=angles, resultant_length=coherence,
                            **maxima, x_label=xlabel, y_label=ylabel, frame_order=np.array(FRAMES), units='deg')
        records.append(dict(key=key, xlabel=xlabel, ylabel=ylabel, assignments=int(count.sum())))
        print(f'{key}: direzioni nei tre riferimenti, medie e massimi verificati.', flush=True)

    member_indices = np.arange(len(members), dtype=np.int64)
    for spec, (x, y) in zip(MAPS[:6], coordinates):
        grid = read_reference(CATALOG_TABLES, spec[0])
        ex, ey, shape = grid['edges_x'], grid['edges_y'], grid['occurrences'].shape
        cells = cells_in_grid(x, y, ex, ey)
        geometry = geometry_from_cells(cells, rows, ex, ey, len(stats))
        actual = plots.best_transfer_occurrences(geometry, winner_rows, len(stats))['occurrences']
        np.testing.assert_array_equal(actual, grid['occurrences'])
        count, angles, coherence = circular_statistics(
            cells, (syn_unit, cat_unit*phase_mean[rows], cat_unit), shape)
        chosen, maximum = maximum_source(cells, member_indices, speeds, grid['occurrences'].size)
        chosen_winner = np.full(len(chosen), -1, dtype=np.int64)
        present = chosen >= 0
        chosen_winner[present] = earliest[rows[chosen[present]]]
        maxima = representative_fields(chosen, maximum, chosen_winner, members,
                                       syn_unit, cat_unit, phase_unit, shape)
        save_map(spec, grid, count, angles, coherence, maxima)

    # Each assignment is a member plus one physical winning ETD. The native
    # directions are ETD-independent; the J2000 direction uses that ETD phase.
    groups = {c: np.flatnonzero(rows == c) for c in np.unique(rows)}
    grids = [read_reference(PHI_TABLES, m[0]) for m in MAPS[6:]]
    reconstructed = [np.zeros_like(g['occurrences']) for g in grids]
    key_parts, winner_parts = [[], []], []
    for j, winner in enumerate(winners):
        ids = groups.get(winner['cluster_row'])
        if ids is None:
            continue
        phi = (omega[ids]+winner['phase_etd_deg']) % 360
        winner_parts.append(np.full(len(ids), j, dtype=np.int32))
        for column, (x, grid) in enumerate(zip((eps, hp), grids)):
            cells = cells_in_grid(x[ids], phi, grid['edges_x'], grid['edges_y'])
            reconstructed[column].flat[np.unique(cells)] += 1
            key_parts[column].append(ids*grid['occurrences'].size+cells)
    assignment_winners = np.concatenate(winner_parts)
    winner_parts.clear()
    for column, (spec, grid) in enumerate(zip(MAPS[6:], grids)):
        np.testing.assert_array_equal(reconstructed[column], grid['occurrences'])
        all_keys = np.concatenate(key_parts[column])
        key_parts[column].clear()
        keys, inverse, repeated = np.unique(all_keys, return_inverse=True, return_counts=True)
        del all_keys
        phase_real = np.bincount(inverse, weights=phase_unit.real[assignment_winners], minlength=len(keys))
        phase_imag = np.bincount(inverse, weights=phase_unit.imag[assignment_winners], minlength=len(keys))
        pair_phase = (phase_real+1j*phase_imag)/repeated
        del phase_real, phase_imag, repeated
        pair_rank = np.full(len(keys), np.iinfo(np.int32).max, dtype=np.int32)
        np.minimum.at(pair_rank, inverse, epoch_rank[assignment_winners])
        del inverse
        ids, cells = keys//grid['occurrences'].size, keys % grid['occurrences'].size
        shape = grid['occurrences'].shape
        count, angles, coherence = circular_statistics(
            cells, (syn_unit[ids], cat_unit[ids]*pair_phase, cat_unit[ids]), shape)
        chosen, maximum = maximum_source(cells, ids, speeds[ids], grid['occurrences'].size)
        chosen_winner = np.full(len(chosen), -1, dtype=np.int64)
        present = chosen >= 0
        pair_indices = np.searchsorted(keys, chosen[present]*grid['occurrences'].size+np.flatnonzero(present))
        np.testing.assert_array_equal(keys[pair_indices], chosen[present]*grid['occurrences'].size+np.flatnonzero(present))
        chosen_winner[present] = epoch_order[pair_rank[pair_indices]]
        maxima = representative_fields(chosen, maximum, chosen_winner, members,
                                       syn_unit, cat_unit, phase_unit, shape)
        save_map(spec, grid, count, angles, coherence, maxima)
        del keys, ids, cells, pair_phase, pair_rank, pair_indices
        gc.collect()
    represented = np.unique(rows)
    metadata = dict(catalog=str(catalog), maps=records, bins=100, frame_order=list(FRAMES),
                    frame_labels=['Sinodico Terra–Luna', 'Geocentrico ECLIPJ2000',
                                  'Geocentrico del catalogo: Luna su +x a ETD'],
                    selected_asteroids=len(winners), represented_asteroids=int(np.isin(winner_rows, represented).sum()),
                    represented_clusters=len(represented), distinct_members=len(members), winners=winners,
                    total_instances=int(frequency[rows].sum()),
                    direction_definition='Angolo antiorario da +x al vettore velocita geocentrico alla SOI, in gradi [0,360).',
                    synodic_definition='v_syn = R(-n*t_soi) @ (v_catalogo - Omega cross r_catalogo), n = 1/TU.',
                    j2000_definition='v_J2000 = R(fase_lunare_ETD) @ v_catalogo; origine geocentrica.',
                    mean_definition='Ogni source_id pesa una volta per cella; media complessa sulle ETD valide di quel membro, poi media complessa sui membri; nessuna rinormalizzazione intermedia.',
                    maximum_definition='Direzione della traiettoria con massimo modulo geocentrico della velocita SOI; parita: source_id minimo, poi prima ETD valida.',
                    coherence_definition='Modulo della media complessa gerarchica; tra 0 e 1. La direzione media non e definita sotto la tolleranza numerica.',
                    undefined_direction_tolerance=ANGLE_TOL, units='deg',
                    report_sha256=summary['report_sha256'],
                    source_metadata_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                        CATALOG_TABLES/'riepilogo.json', PHI_TABLES/'riepilogo.json')})
    (output/'dati.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
