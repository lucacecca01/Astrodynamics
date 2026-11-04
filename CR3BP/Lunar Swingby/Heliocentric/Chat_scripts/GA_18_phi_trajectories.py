"""Tavole delle traiettorie per le mappe 11–12, con la fase ETD di ogni asteroide.

Riusa i vincitori del riepilogo GA18 e verifica le griglie originali prima di
selezionare le traiettorie. Nessun nuovo problema di Lambert o clustering.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np
import pykep as pk
import spiceypy as spice
from matplotlib.colors import Normalize

import GA_18_plots as plots
from GA_18_combined_trajectories import draw_combined
from GA_18_percentile_trajectories import package_pdf


MAPS = (
    ("11_eps_phi", "11_migliori_transfer_occorrenze_eps_phi", plots.EPS_LABEL),
    ("12_hp_phi", "12_migliori_transfer_occorrenze_hp_phi", plots.HP_LABEL),
)


def read_winners(report_path, stats):
    """Legge i trasferimenti gia scelti, conservando identita ed epoca fisica."""
    winners = []
    identity = None
    pattern = (r"Miglior trasferimento: cluster (\d+); DV ([\d.eE+\-]+) km/s; "
               r"partenza (SOI|ETD) JD ([\d.eE+\-]+); arrivo JD ([\d.eE+\-]+); "
               r"TOF ([\d.eE+\-]+) giorni")
    for line in Path(report_path).read_text().splitlines():
        name = re.match(r"(.+) \(SPK-ID: (\d+); epoche (SOI|ETD)\):", line)
        if name:
            identity = dict(name=name[1], spkid=name[2])
        match = re.search(pattern, line)
        if match:
            if identity is None:
                raise ValueError("Vincitore senza identita dell'asteroide nel riepilogo.")
            cluster = int(match[1])
            row = int(np.searchsorted(stats[:, 0], cluster))
            if row >= len(stats) or stats[row, 0] != cluster:
                raise ValueError(f"Cluster {cluster} assente dal catalogo.")
            jd = float(match[4])
            jd_etd = jd - stats[row, 7] if match[3] == "SOI" else jd
            winners.append(dict(identity, cluster_id=cluster, cluster_row=row,
                                dv_km_s=float(match[2]), jd_etd=jd_etd,
                                jd_arrival=float(match[5]), tof_days=float(match[6])))
    if not winners or len({w['spkid'] for w in winners}) != len(winners):
        raise ValueError("Riepilogo vuoto o asteroidi vincenti duplicati.")
    spice.furnsh(pk.udpla.de440s.kernel_file())
    for winner in winners:
        moon = spice.spkezr("MOON", (winner['jd_etd'] - 2451545.) * 86400.,
                             "ECLIPJ2000", "NONE", "EARTH")[0]
        winner['phase_etd_deg'] = float(np.degrees(np.arctan2(moon[1], moon[0])))
    return winners


def grid_cells(grid, x, phi):
    """Stessa convenzione di histogram2d, incluso l'ultimo bordo."""
    ex, ey = grid['edges_x'], grid['edges_y']
    if np.any((x < ex[0]) | (x > ex[-1]) | (phi < ey[0]) | (phi > ey[-1])):
        raise ValueError("Membro fuori dai bordi della griglia salvata.")
    ix = np.minimum(np.searchsorted(ex, x, side='right') - 1, len(ex) - 2)
    iy = np.minimum(np.searchsorted(ey, phi, side='right') - 1, len(ey) - 2)
    return ix * (len(ey) - 1) + iy


def prepare(input_dir, output, cell_percentiles, individual_percentiles):
    report_path = input_dir / 'riepilogo.txt'
    report = report_path.read_text()
    catalog = Path(re.search(r'^Catalogo: (.+)$', report, re.M)[1])
    stats = plots.load_table(catalog, 15, allow_extra=True)
    stats = stats[np.argsort(stats[:, 0])]
    if not np.isin(stats[:, 14], (-1, 1)).all():
        raise ValueError('Versi al perigeo non validi nel catalogo.')
    winners = read_winners(report_path, stats)
    winner_rows = np.array([w['cluster_row'] for w in winners])
    frequency = np.bincount(winner_rows, minlength=len(stats))
    print('Lettura dei membri e selezione delle orbite legate...', flush=True)
    all_members = plots.load_table(catalog.with_name('trajectory_exit_catalog.txt'), 12, allow_extra=True)
    all_members = all_members[:, :12]
    eps, ecc = all_members[:, 9:11].T
    rp = np.full(len(all_members), np.nan)
    np.divide(-plots.MU_EARTH * (1 - ecc), 2 * eps, out=rp, where=eps != 0)
    keep = ((eps < 0) & (ecc >= 0) & (ecc < 1) & (rp > plots.R_EARTH_KM)
            & np.isfinite(rp) & np.isin(all_members[:, 1], stats[frequency > 0, 0]))
    members = all_members[keep]
    del all_members, rp, eps, ecc, keep
    if not len(members):
        raise ValueError('Nessun membro ammissibile.')
    members = members[np.argsort(members[:, 0])]
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    groups = {c: np.flatnonzero(rows == c) for c in np.unique(rows)}
    scores = frequency[rows]
    individual_thresholds = np.percentile(scores, individual_percentiles)
    eps, ecc = members[:, 9:11].T
    hp = np.empty(len(members))
    np.divide(-plots.MU_EARTH * (1 - ecc), 2 * eps, out=hp)
    hp -= plots.R_EARTH_KM
    phi_catalog = (stats[rows, 14] * members[:, 11]) % 360
    coordinates = (eps, hp)
    grids = []
    for key, stem, xlabel in MAPS:
        with np.load(input_dir / 'mappe_percentili' / f'{stem}_griglie.npz') as z:
            grid = {k: z[k] for k in ('occurrences', 'edges_x', 'edges_y')}
        occupied = grid['occurrences'] > 0
        if not occupied.any():
            raise ValueError(f'{key}: griglia vuota.')
        grid.update(key=key, xlabel=xlabel, ylabel=r'$\phi_p$ ECLIPJ2000 [deg]',
                    thresholds=np.percentile(grid['occurrences'][occupied], cell_percentiles),
                    reconstructed=np.zeros_like(grid['occurrences']),
                    individual_counts=np.zeros((len(individual_percentiles), *occupied.shape), dtype=np.int64))
        grids.append(grid)
    # Non si deduplicano source_id di asteroidi diversi: hanno fasi diverse.
    member_parts, winner_parts, count_parts = [], [], []
    total_instances = 0
    for j, winner in enumerate(winners):
        c = winner['cluster_row']
        indices = groups.get(c)
        if indices is None:
            continue
        total_instances += len(indices)
        phi = (phi_catalog[indices] + winner['phase_etd_deg']) % 360
        row_counts = []
        selected = np.full(len(indices), frequency[c] >= individual_thresholds.min())
        for grid, x in zip(grids, coordinates):
            cells = grid_cells(grid, x[indices], phi)
            unique_cells = np.unique(cells)
            grid['reconstructed'].flat[unique_cells] += 1
            counts = grid['occurrences'].ravel()[cells]
            row_counts.append(counts)
            selected |= counts >= grid['thresholds'].min()
            for level, cutoff in enumerate(individual_thresholds):
                if frequency[c] >= cutoff:
                    grid['individual_counts'][level].flat[unique_cells] += 1
        if selected.any():
            member_parts.append(indices[selected].astype(np.int32))
            winner_parts.append(np.full(selected.sum(), j, dtype=np.int32))
            count_parts.append(np.column_stack(row_counts)[selected].astype(np.int32))
    for grid in grids:
        if not np.array_equal(grid.pop('reconstructed'), grid['occurrences']):
            raise ValueError(f"{grid['key']}: griglia non riprodotta da catalogo, fasi e vincitori correnti.")
        if np.any(grid['individual_counts'] > grid['occurrences']):
            raise ValueError('Copertura selezionata maggiore della griglia completa.')
    pair_members = np.concatenate(member_parts)
    pair_winners = np.concatenate(winner_parts)
    pair_counts = np.concatenate(count_parts)
    unique = np.unique(pair_members)
    curve_indices = np.searchsorted(unique, pair_members).astype(np.int32)
    selected_members = members[unique]
    pair_scores = frequency[winner_rows[pair_winners]]
    summary = dict(catalog=str(catalog), report=str(report_path),
                   report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                   selected_asteroids=len(winners), population_count=len(members),
                   total_instances=total_instances, reconstructed_trajectories=len(unique),
                   selected_instances=len(pair_members), cell_percentiles=list(cell_percentiles),
                   individual_percentiles=list(individual_percentiles),
                   individual_thresholds=individual_thresholds.tolist(),
                   maximum_individual_score=int(scores.max()), winners=winners,
                   convention='Una istanza = membro del catalogo + asteroide e ETD vincente; fase al tempo ETD medio del cluster. Integrazione unica per source_id.',
                   model='CR3BP planare ruotato alla fase ETD ECLIPJ2000; prestazioni del modello medio, non Lambert individuali.',
                   grid_verification='Entrambe le griglie ricostruite esattamente, un asteroide al massimo una volta per cella.')
    np.savez_compressed(output / 'selezione_istanze.npz', members=selected_members,
                        curve_indices=curve_indices, winner_indices=pair_winners,
                        cell_counts=pair_counts, individual_scores=pair_scores)
    (output / 'preparazione.json').write_text(json.dumps(summary, indent=2) + '\n')
    for grid in grids:
        np.savez_compressed(output / f"{grid['key']}_griglie.npz", **{
            k: grid[k] for k in ('occurrences', 'edges_x', 'edges_y', 'thresholds', 'individual_counts')})
    print(f"Griglie verificate: {len(members):,} membri, {total_instances:,} istanze; "
          f"da rappresentare {len(unique):,} curve uniche / {len(pair_members):,} istanze.", flush=True)
    return selected_members, curve_indices, pair_winners, pair_counts, pair_scores, grids, summary


def render_maps(curves, members, curve_indices, winner_indices, cell_counts, individual_scores,
                grids, summary, output, columns, trajectory_layer_dir=None,
                trajectory_layer_token=None, resume=False):
    radius = 0.
    for start in range(0, len(curves), 2048):
        radius = max(radius, float(np.linalg.norm(curves[start:start+2048], axis=1).max()))
    limits = np.array([[-radius, -radius], [radius, radius]])
    phases = np.array([w['phase_etd_deg'] for w in summary['winners']])
    pages, records = [], []
    for column in columns:
        grid = grids[column]
        key = grid['key']
        maximum = int(grid['occurrences'].max())
        grid['normalization'] = Normalize(0, maximum)
        for kind, levels, thresholds in (
            ('celle', summary['cell_percentiles'], grid['thresholds']),
            ('individuali', summary['individual_percentiles'], summary['individual_thresholds']),
        ):
            folder = output / kind / key
            folder.mkdir(parents=True, exist_ok=True)
            group_pages = []
            for level, (percentile, threshold) in enumerate(zip(levels, thresholds)):
                scores = cell_counts[:, column] if kind == 'celle' else individual_scores
                take = np.flatnonzero(scores >= threshold)
                ids = curve_indices[take]
                winners = winner_indices[take]
                selected_members = members[ids]
                tag = f'P{percentile:g}'.replace('.', '_')
                figure_path = folder / f'{tag}_tre_riferimenti.png'
                count_unique = len(np.unique(ids))
                n_asteroids = len(np.unique(winners))
                print(f"Disegno {key} {kind} {tag}: {len(take):,} istanze / {count_unique:,} traiettorie...", flush=True)
                if kind == 'celle':
                    grid.pop('selected_occurrences', None)
                    grid['threshold'] = threshold
                    grid['selection_title'] = 'Celle sopra soglia'
                    grid['selection_caption'] = 'Colore: celle selezionate · grigio: altre celle occupate'
                    label = f'Asteroidi per cella — mappa {key[:2]}'
                    criterion = f'P{percentile:g} delle celle — almeno {int(np.ceil(threshold))} asteroidi/cella'
                    normalization = Normalize(0, maximum)
                else:
                    grid['selected_occurrences'] = grid['individual_counts'][level]
                    grid['selection_title'] = 'Celle con traiettorie selezionate'
                    grid['selection_caption'] = 'Colore: copertura dei membri selezionati nelle rispettive epoche'
                    label = 'Asteroidi il cui minimo DV sceglie il cluster della traiettoria'
                    criterion = f'P{percentile:g} delle singole traiettorie — almeno {int(np.ceil(threshold))} asteroidi per cluster'
                    normalization = Normalize(0, summary['maximum_individual_score'])
                title = (f'Mappa {key[:2]}: {criterion}\n'
                         f'{count_unique:,} traiettorie distinte · {len(take):,} istanze a ETD diverse · '
                         f'{n_asteroids} asteroidi · {curves.shape[-1]} punti per curva')
                selection_path = folder / f'{tag}_selezione.npz'
                reusable = False
                if resume and figure_path.is_file() and selection_path.is_file():
                    from PIL import Image
                    try:
                        with np.load(selection_path) as z:
                            reusable = (np.array_equal(z['source_ids'], selected_members[:, 0])
                                        and np.array_equal(z['cluster_ids'], selected_members[:, 1])
                                        and np.array_equal(z['winner_indices'], winners)
                                        and np.array_equal(z['phase_etd_deg'], phases[winners])
                                        and np.array_equal(z['scores'], scores[take])
                                        and float(z['threshold']) == float(threshold)
                                        and float(z['percentile']) == float(percentile))
                        if reusable:
                            with Image.open(figure_path) as image:
                                reusable = image.size == (3600, 2340)
                                image.verify()
                    except (OSError, ValueError, KeyError):
                        reusable = False
                if reusable:
                    print(f'Tavola gia verificata: {key} {kind} {tag}.', flush=True)
                else:
                    draw_combined(curves, selected_members, scores[take], label, figure_path,
                                  summary['population_count'], title=title, normalization=normalization,
                                  fixed_alpha=0.03, global_limits=limits, raster_chunks=True,
                                  perigee_markers=True, selection_grid=grid,
                                  phase_etd_deg=phases[winners], curve_indices=ids,
                                  third_frame='catalog',
                                  trajectory_layer_dir=(trajectory_layer_dir if kind == 'individuali' else None),
                                  trajectory_layer_token=trajectory_layer_token)
                np.savez_compressed(selection_path,
                                    source_ids=selected_members[:, 0].astype(np.int64),
                                    cluster_ids=selected_members[:, 1].astype(np.int64),
                                    winner_indices=winners, phase_etd_deg=phases[winners],
                                    scores=scores[take], threshold=threshold, percentile=percentile)
                records.append(dict(map=key, criterion=kind, percentile=percentile,
                                    threshold=float(threshold), trajectories=count_unique,
                                    instances=len(take), asteroids=n_asteroids, file=str(figure_path)))
                group_pages.append(figure_path)
                gc.collect()
            package_pdf(group_pages, folder / 'percentili_tre_riferimenti.pdf')
            pages.extend(group_pages)
    return pages, records


_RENDER_ARGUMENTS = None


def render_one_map(column):
    return render_maps(*_RENDER_ARGUMENTS, columns=[column])


def render(curves, members, curve_indices, winner_indices, cell_counts, individual_scores,
           grids, summary, output, workers=1):
    arguments = (curves, members, curve_indices, winner_indices, cell_counts,
                 individual_scores, grids, summary, output)
    if workers > 1:
        from multiprocessing import get_context
        global _RENDER_ARGUMENTS
        _RENDER_ARGUMENTS = arguments
        try:
            with get_context('fork').Pool(min(workers, len(MAPS))) as pool:
                results = pool.map(render_one_map, range(len(MAPS)))
        finally:
            _RENDER_ARGUMENTS = None
        pages = [page for group, _ in results for page in group]
        records = [record for _, group in results for record in group]
    else:
        pages, records = render_maps(*arguments, columns=range(len(MAPS)))
    package_pdf(pages, output / 'tavole_phi_tutti_percentili.pdf')
    summary['figures'] = records
    summary['trajectory_panels'] = ['sinodico Terra-Luna', 'geocentrico ECLIPJ2000',
                                    'geocentrico del catalogo, Luna su +x a ETD']
    (output / 'riepilogo.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Completate {len(pages)} tavole in {output}', flush=True)


def load_render_inputs(output):
    """Riapre i prodotti verificati: una modifica grafica non richiede integrazioni."""
    summary = json.loads((output / 'riepilogo.json').read_text())
    with np.load(output / 'selezione_istanze.npz') as z:
        members, indices, winners, counts, scores = (
            z[k] for k in ('members', 'curve_indices', 'winner_indices',
                          'cell_counts', 'individual_scores'))
    provenance = json.loads((output / 'curve_provenance.json').read_text())
    digest = hashlib.sha256(np.ascontiguousarray(members).tobytes()).hexdigest()
    if digest != provenance['signature']['members_sha256']:
        raise ValueError('Curve e selezione dei membri non corrispondono.')
    curves = np.load(output / 'reconstructed_curves.npy', mmap_mode='r')
    done = np.load(output / 'reconstructed_done.npy', mmap_mode='r')
    if (curves.shape != (len(members), 2, provenance['samples'])
            or done.shape != (len(members),) or not done.all()):
        raise ValueError('Curve salvate incomplete o di forma incoerente.')
    if (len(indices) != len(winners) or counts.shape != (len(indices), len(MAPS))
            or scores.shape != indices.shape or np.any(indices < 0)
            or np.any(indices >= len(members)) or np.any(winners < 0)
            or np.any(winners >= len(summary['winners']))):
        raise ValueError('Associazioni tra curve e asteroidi non valide.')
    grids = []
    for key, _, xlabel in MAPS:
        with np.load(output / f'{key}_griglie.npz') as z:
            grid = {k: z[k] for k in ('occurrences', 'edges_x', 'edges_y',
                                     'thresholds', 'individual_counts')}
        grid.update(key=key, xlabel=xlabel, ylabel=r'$\phi_p$ ECLIPJ2000 [deg]')
        grids.append(grid)
    return curves, members, indices, winners, counts, scores, grids, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=plots.OUTPUT_DIR)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cell-percentiles', type=float, nargs='+', default=plots.OCCURRENCE_PERCENTILES)
    parser.add_argument('--individual-percentiles', type=float, nargs='+', default=(99.9, 99.95, 99.99))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--render-workers', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true', help='Ridisegna le selezioni e le curve gia salvate.')
    args = parser.parse_args()
    for levels in (args.cell_percentiles, args.individual_percentiles):
        if any(not np.isfinite(p) or not 0 < p < 100 for p in levels):
            parser.error('Percentili finiti, strettamente compresi fra 0 e 100.')
    output = args.output or args.input / 'traiettorie_phi'
    output.mkdir(parents=True, exist_ok=True)
    if args.workers < 1 or args.render_workers < 1:
        parser.error('Il numero di processi deve essere positivo.')
    if args.prepare_only and args.render_only:
        parser.error('--prepare-only e --render-only sono alternativi.')
    if args.render_only:
        render(*load_render_inputs(output), output, workers=args.render_workers)
        return
    members, indices, winners, counts, scores, grids, summary = prepare(
        args.input, output, sorted(set(args.cell_percentiles)), sorted(set(args.individual_percentiles)))
    if args.prepare_only:
        return
    from GA_18_phi_curve_data import prepare_curves
    curves, provenance = prepare_curves(members, Path(summary['catalog']).with_name('trajectory_exit_catalog.txt'),
                                        output, workers=args.workers)
    summary['curve_provenance'] = provenance
    render(curves, members, indices, winners, counts, scores, grids, summary, output,
           workers=args.render_workers)


if __name__ == '__main__':
    main()
