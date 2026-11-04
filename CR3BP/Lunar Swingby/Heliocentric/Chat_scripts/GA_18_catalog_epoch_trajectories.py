"""Mappe 05–10: sinodico, geocentrico ECLIPJ2000 e assi del catalogo.

Le celle usano gli elementi del catalogo; la fase lunare reale ruota soltanto
la vista J2000. Ogni membro e associato alle ETD vincenti del proprio cluster.
"""

import argparse
import gc
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import re
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np

import GA_18_plots as plots
from GA_18_catalog_azimuth_trajectories import (
    ANGLE_CONVENTION, cells_in_grid, geometry_from_cells, read_eligible_members,
)
from GA_18_phi_trajectories import read_winners, render_maps
from GA_18_percentile_trajectories import package_pdf


MAPS = (
    ('05_eps_e', '05_migliori_transfer_occorrenze_eps_e', plots.EPS_LABEL, 'e al perigeo [-]'),
    ('06_eps_omega', '06_migliori_transfer_occorrenze_eps_omega', plots.EPS_LABEL, plots.PHI_CATALOG_LABEL),
    ('07_hp_e', '07_migliori_transfer_occorrenze_rp_e', plots.HP_LABEL, 'e al perigeo [-]'),
    ('08_hp_omega', '08_migliori_transfer_occorrenze_rp_omega', plots.HP_LABEL, plots.PHI_CATALOG_LABEL),
    ('09_hp_theta', '09_migliori_transfer_occorrenze_rp_theta', plots.HP_LABEL, plots.THETA_LABEL),
    ('10_eps_theta', '10_migliori_transfer_occorrenze_eps_theta', plots.EPS_LABEL, plots.THETA_LABEL),
)
INDIVIDUAL_PERCENTILES = (99.9, 99.95, 99.99)


def prepare(input_dir, output, n_bins):
    report_path = input_dir/'riepilogo.txt'
    report = report_path.read_text()
    catalog = Path(re.search(r'^Catalogo: (.+)$', report, re.M)[1])
    stats = plots.load_table(catalog, 15, allow_extra=True)
    stats = stats[np.argsort(stats[:, 0])]
    winners = read_winners(report_path, stats)
    winner_rows = np.array([w['cluster_row'] for w in winners])
    frequency = np.bincount(winner_rows, minlength=len(stats))
    members = read_eligible_members(catalog.with_name('trajectory_exit_catalog.txt'), stats[frequency > 0, 0])
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    scores = frequency[rows]
    individual_thresholds = np.percentile(scores, INDIVIDUAL_PERCENTILES)
    eps, ecc = members[:, 9:11].T
    hp = np.empty(len(members))
    np.divide(-plots.MU_EARTH*(1-ecc), 2*eps, out=hp)
    hp -= plots.R_EARTH_KM
    omega = plots.catalog_perigee_azimuth(members, stats)
    theta = (omega-plots.MOON_RATE_DEG_DAY*members[:, 5]) % 360
    coordinates = ((eps, ecc), (eps, omega), (hp, ecc), (hp, omega), (hp, theta), (eps, theta))
    grids, count_columns = [], []
    union = scores >= individual_thresholds.min()
    input_hashes = {'riepilogo.txt': hashlib.sha256(report_path.read_bytes()).hexdigest()}
    for (key, stem, xlabel, ylabel), (x, y) in zip(MAPS, coordinates):
        path = input_dir/'mappe_percentili'/f'{stem}_griglie.npz'
        input_hashes[str(path.relative_to(input_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            previous, ex, ey = z['occurrences'], z['edges_x'], z['edges_y']
        # Verifica indipendente delle griglie congelate prima di cambiare bin.
        prior_cells = cells_in_grid(x, y, ex, ey)
        geometry = geometry_from_cells(prior_cells, rows, ex, ey, len(stats))
        prior = plots.best_transfer_occurrences(geometry, winner_rows, len(stats))['occurrences']
        if not np.array_equal(previous, prior):
            raise ValueError(f'{key}: griglia di input diversa da membri, verso e vincitori correnti.')
        ex, ey = np.linspace(ex[0], ex[-1], n_bins+1), np.linspace(ey[0], ey[-1], n_bins+1)
        cells = cells_in_grid(x, y, ex, ey)
        geometry = geometry_from_cells(cells, rows, ex, ey, len(stats))
        full = plots.best_transfer_occurrences(geometry, winner_rows, len(stats))['occurrences']
        thresholds = np.percentile(full[full > 0], plots.OCCURRENCE_PERCENTILES)
        member_counts = full.ravel()[cells]
        count_columns.append(member_counts.astype(np.int32))
        union |= member_counts >= thresholds.min()
        coverage = []
        for cutoff in individual_thresholds:
            take = frequency[geometry['pair_rows']] >= cutoff
            value = np.bincount(geometry['pair_cells'][take],
                                weights=frequency[geometry['pair_rows'][take]], minlength=full.size)
            coverage.append(value.reshape(full.shape).astype(np.int64))
        grid = dict(key=key, xlabel=xlabel, ylabel=ylabel, occurrences=full,
                    edges_x=ex, edges_y=ey, thresholds=thresholds, individual_counts=np.array(coverage))
        if np.any(grid['individual_counts'] > full):
            raise ValueError(f'{key}: copertura selezionata superiore al totale.')
        grids.append(grid)
        print(f'{key}: input verificato; griglia {n_bins}x{n_bins}, soglie {thresholds.tolist()}.', flush=True)
    selected_members = members[union]
    selected_rows = rows[union]
    selected_counts = np.column_stack(count_columns)[union]
    selected_scores = scores[union]
    total_instances = int(scores.sum())
    population_count = len(members)
    del members, eps, ecc, hp, omega, theta, coordinates, rows, scores, count_columns, union
    gc.collect()
    groups = {c: np.flatnonzero(selected_rows == c).astype(np.int32) for c in np.unique(selected_rows)}
    curve_parts, winner_parts = [], []
    for j, winner in enumerate(winners):
        ids = groups.get(winner['cluster_row'])
        if ids is not None:
            curve_parts.append(ids)
            winner_parts.append(np.full(len(ids), j, dtype=np.int32))
    indices, winner_indices = np.concatenate(curve_parts), np.concatenate(winner_parts)
    counts, individual_scores = selected_counts[indices], selected_scores[indices]
    summary = dict(catalog=str(catalog), report=str(report_path), report_sha256=input_hashes['riepilogo.txt'],
                   input_dir=str(input_dir.resolve()), input_sha256=input_hashes,
                   catalog_angle_convention=ANGLE_CONVENTION, bins=n_bins,
                   selected_asteroids=len(winners), population_count=population_count,
                   total_instances=total_instances, reconstructed_trajectories=len(selected_members),
                   selected_instances=len(indices), cell_percentiles=list(plots.OCCURRENCE_PERCENTILES),
                   individual_percentiles=list(INDIVIDUAL_PERCENTILES),
                   individual_thresholds=individual_thresholds.tolist(), maximum_individual_score=int(frequency.max()),
                   winners=winners, maps=[list(m) for m in MAPS],
                   convention='Celle del catalogo; una istanza per membro e asteroide che sceglie il suo cluster. Fase lunare alla sua ETD.',
                   model='CR3BP planare; prestazioni del modello medio attribuite ai membri.',
                   grid_verification='Tutte le sei griglie di input riprodotte esattamente prima del ricalcolo dei bin.')
    np.savez_compressed(output/'membri_selezionati.npz', members=selected_members,
                        per_map_counts=selected_counts, individual_scores=selected_scores)
    np.savez_compressed(output/'selezione_istanze.npz', members=selected_members, curve_indices=indices,
                        winner_indices=winner_indices, cell_counts=counts, individual_scores=individual_scores)
    for grid in grids:
        np.savez_compressed(output/f'{grid["key"]}_griglie.npz', **grid, angle_convention=ANGLE_CONVENTION)
    (output/'preparazione.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Selezione: {len(selected_members):,} curve uniche / {len(indices):,} istanze ETD.', flush=True)
    return selected_members, indices, winner_indices, counts, individual_scores, grids, summary


def load_render_inputs(output):
    path = output/'riepilogo.json'
    summary = json.loads((path if path.exists() else output/'preparazione.json').read_text())
    if summary.get('catalog_angle_convention') != ANGLE_CONVENTION or summary['maps'] != [list(m) for m in MAPS]:
        raise ValueError('Convenzioni o mappe salvate diverse.')
    with np.load(output/'selezione_istanze.npz') as z:
        members, indices, winners, counts, scores = (z[k] for k in (
            'members', 'curve_indices', 'winner_indices', 'cell_counts', 'individual_scores'))
    provenance = json.loads((output/'curve_provenance.json').read_text())
    digest = hashlib.sha256(np.ascontiguousarray(members).tobytes()).hexdigest()
    curves = np.load(output/'reconstructed_curves.npy', mmap_mode='r')
    done = np.load(output/'reconstructed_done.npy', mmap_mode='r')
    if (digest != provenance['signature']['members_sha256'] or not done.all()
            or curves.shape != (len(members), 2, provenance['samples']) or done.shape != (len(members),)):
        raise ValueError('Curve incomplete o non associate ai membri salvati.')
    if (counts.shape != (len(indices), len(MAPS)) or scores.shape != indices.shape
            or winners.shape != indices.shape or np.any(indices < 0) or np.any(indices >= len(members))
            or np.any(winners < 0) or np.any(winners >= len(summary['winners']))):
        raise ValueError('Associazioni delle istanze non valide.')
    grids = []
    for key, _, xlabel, ylabel in MAPS:
        with np.load(output/f'{key}_griglie.npz') as z:
            if str(z['angle_convention']) != ANGLE_CONVENTION:
                raise ValueError('Griglia con una diversa convenzione angolare.')
            grid = {k: z[k] for k in ('occurrences', 'edges_x', 'edges_y', 'thresholds', 'individual_counts')}
        grid.update(key=key, xlabel=xlabel, ylabel=ylabel)
        grids.append(grid)
    summary['curve_provenance'] = provenance
    return curves, members, indices, winners, counts, scores, grids, summary


_RENDER_ARGUMENTS = None


def render_column(arguments, column):
    summary, output = arguments[-2:]
    return render_maps(*arguments, columns=[column],
                       trajectory_layer_dir=output/'livelli_traiettorie',
                       trajectory_layer_token=json.dumps(summary['curve_provenance']['signature'], sort_keys=True),
                       resume=True)


def render_one_map(column):
    return render_column(_RENDER_ARGUMENTS, column)


def render(curves, members, indices, winners, counts, scores, grids, summary, output, workers):
    arguments = (curves, members, indices, winners, counts, scores, grids, summary, output)
    if workers > 1:
        global _RENDER_ARGUMENTS
        _RENDER_ARGUMENTS = arguments
        try:
            with get_context('fork').Pool(min(workers, len(MAPS))) as pool:
                results = pool.map(render_one_map, range(len(MAPS)))
        finally:
            _RENDER_ARGUMENTS = None
    else:
        results = [render_column(arguments, c) for c in range(len(MAPS))]
    pages = [p for group, _ in results for p in group]
    summary['figures'] = [r for _, group in results for r in group]
    summary['trajectory_panels'] = ['sinodico Terra-Luna', 'geocentrico ECLIPJ2000',
                                    'geocentrico del catalogo, Luna su +x a ETD']
    package_pdf(pages, output/'tavole_05_10_J2000_catalogo_tutti_percentili.pdf')
    (output/'riepilogo.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Completate {len(pages)} tavole in {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=plots.OUTPUT_DIR/'traiettorie_azimut_catalogo/mappe_corrette')
    parser.add_argument('--output', type=Path, default=plots.OUTPUT_DIR/'traiettorie_05_10_J2000_catalogo_100x100')
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--reuse-curves', type=Path, nargs='*', default=[
        plots.OUTPUT_DIR/'traiettorie_omega_catalogo_100x100', plots.OUTPUT_DIR/'traiettorie_phi'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--render-workers', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if min(args.bins, args.workers, args.render_workers) < 1 or (args.prepare_only and args.render_only):
        parser.error('Bin e processi devono essere positivi; prepare-only e render-only sono alternativi.')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        data = load_render_inputs(args.output)
        if data[-1]['bins'] != args.bins:
            parser.error('I bin salvati sono diversi: ometti --render-only per ricalcolare le selezioni.')
    else:
        members, indices, winners, counts, scores, grids, summary = prepare(args.input, args.output, args.bins)
        if args.prepare_only:
            return
        from GA_18_phi_curve_data import prepare_curves
        curves, provenance = prepare_curves(members, Path(summary['catalog']).with_name('trajectory_exit_catalog.txt'),
                                            args.output, workers=args.workers, reusable_products=args.reuse_curves)
        summary['curve_provenance'] = provenance
        data = (curves, members, indices, winners, counts, scores, grids, summary)
    render(*data, args.output, workers=args.render_workers)


if __name__ == '__main__':
    main()
