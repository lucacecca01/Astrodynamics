"""Rigenera le mappe 06/08 e le loro traiettorie usando l'azimut del catalogo."""

import argparse
import gc
import hashlib
from itertools import islice
import json
from pathlib import Path
import re
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import GA_18_plots as plots
from GA_18_combined_trajectories import draw_combined
from GA_18_percentile_trajectories import package_pdf


MAPS = (
    ('06_eps_omega', '06_migliori_transfer_occorrenze_eps_omega', plots.EPS_LABEL),
    ('08_hp_omega', '08_migliori_transfer_occorrenze_rp_omega', plots.HP_LABEL),
)
ANGLE_CONVENTION = 'phi_catalog = direction_perigee * omega mod 360'
INDIVIDUAL_PERCENTILES = (99.9, 99.95, 99.99)


def read_eligible_members(path, winning_ids):
    """Filtra il catalogo a blocchi senza caricare tutti i membri in RAM."""
    chunks = []
    total = 0
    with Path(path).open() as stream:
        while lines := list(islice(stream, 50000)):
            values = np.loadtxt(lines, usecols=range(12), ndmin=2)
            if not len(values):
                continue
            if not np.isfinite(values).all():
                raise ValueError('Valori non finiti nel catalogo dei membri.')
            eps, ecc = values[:, 9:11].T
            rp = np.full(len(values), np.nan)
            np.divide(-plots.MU_EARTH*(1-ecc), 2*eps, out=rp, where=eps != 0)
            keep = ((eps < 0) & (ecc >= 0) & (ecc < 1) & np.isfinite(rp) & (rp > plots.R_EARTH_KM)
                    & np.isin(values[:, 1], winning_ids))
            if keep.any():
                chunks.append(values[keep])
            total += len(values)
            if total % 1000000 < 50000:
                print(f'Membri letti: {total:,}.', flush=True)
    if not chunks:
        raise ValueError('Nessun membro fisicamente ammissibile dei cluster vincenti.')
    members = np.concatenate(chunks)
    chunks.clear()
    if np.any(np.diff(members[:, 0]) < 0):
        members = members[np.argsort(members[:, 0])]
    if np.any(np.diff(members[:, 0]) <= 0):
        raise ValueError('source_id duplicati nel catalogo.')
    return members


def cells_in_grid(x, y, ex, ey):
    if np.any((x < ex[0]) | (x > ex[-1]) | (y < ey[0]) | (y > ey[-1])):
        raise ValueError('Membri fuori dai bordi della griglia originale.')
    nx, ny = len(ex)-1, len(ey)-1
    ix = np.minimum(np.searchsorted(ex, x, side='right')-1, nx-1)
    iy = np.minimum(np.searchsorted(ey, y, side='right')-1, ny-1)
    return ix*ny+iy


def geometry_from_cells(cells, rows, ex, ey, n_clusters):
    pairs, multiplicities = np.unique(cells*n_clusters+rows, return_counts=True)
    return dict(edges_x=ex, edges_y=ey,
                population=np.bincount(cells, minlength=(len(ex)-1)*(len(ey)-1)).reshape(len(ex)-1, len(ey)-1),
                pair_cells=pairs//n_clusters, pair_rows=pairs % n_clusters,
                pair_counts=multiplicities, frame_note=plots.PHI_CATALOG_NOTE)


def save_canonical_maps(input_dir, grids):
    """Sostituisce solo 06/08; i PDF includono anche le altre PNG gia salvate."""
    grid_dir = input_dir/'mappe_percentili'
    report_path = grid_dir/'riepilogo.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for grid, (_, stem, xlabel) in zip(grids, MAPS):
        data = grid['map_data']
        fig = plots.plot_best_transfer_occurrences(
            data, plots.PHI_CATALOG_LABEL, int(data['occurrences'].max()), True, xlabel)
        fig.savefig(input_dir/f'{stem}.png', dpi=180)
        plt.close(fig)
        fig, records, masks = plots.plot_occurrence_percentiles(data, xlabel, plots.PHI_CATALOG_LABEL)
        fig.savefig(grid_dir/f'{stem}_percentili.png', dpi=180)
        plt.close(fig)
        np.savez_compressed(grid_dir/f'{stem}_griglie.npz',
                            occurrences=data['occurrences'], edges_x=data['edges_x'], edges_y=data['edges_y'],
                            percentiles=plots.OCCURRENCE_PERCENTILES,
                            thresholds=[r['threshold'] for r in records], masks=masks,
                            x_label=xlabel, y_label=plots.PHI_CATALOG_LABEL,
                            frame_note=plots.PHI_CATALOG_NOTE, angle_convention=ANGLE_CONVENTION)
        report[stem] = records
        print(f'Aggiornata mappa {stem[:2]}: massimo {int(data["occurrences"].max())} asteroidi/cella.', flush=True)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    paths = sorted(grid_dir.glob('*_migliori_transfer_occorrenze_*_percentili.png'))
    if paths:
        package_pdf(paths, grid_dir/'mappe_percentili.pdf')
        first_six = [p for p in paths if 5 <= int(p.name[:2]) <= 10]
        if first_six:
            package_pdf(first_six, grid_dir/'mappe_percentili_05_10.pdf')


def prepare(input_dir, output, n_bins=None):
    if n_bins is not None and (n_bins < 1 or int(n_bins) != n_bins):
        raise ValueError('Il numero di bin per asse deve essere un intero positivo.')
    report_path = input_dir/'riepilogo.txt'
    report = report_path.read_text()
    match = re.search(r'^Catalogo: (.+)$', report, re.M)
    if match is None:
        raise ValueError('Catalogo non indicato nel riepilogo.')
    catalog = Path(match[1])
    stats = plots.load_table(catalog, 15, allow_extra=True)
    stats = stats[np.argsort(stats[:, 0])]
    winner_ids = np.array([int(c) for c in re.findall(r'Miglior trasferimento: cluster (\d+)', report)])
    rows_winner = np.searchsorted(stats[:, 0], winner_ids)
    if not len(winner_ids) or np.any(rows_winner >= len(stats)) or not np.array_equal(stats[rows_winner, 0], winner_ids):
        raise ValueError('Cluster vincenti del riepilogo incoerenti col catalogo.')
    frequency = np.bincount(rows_winner, minlength=len(stats))
    print('Lettura dei membri del catalogo...', flush=True)
    members = read_eligible_members(catalog.with_name('trajectory_exit_catalog.txt'), stats[frequency > 0, 0])
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    score = frequency[rows]
    individual_thresholds = np.percentile(score, INDIVIDUAL_PERCENTILES)
    phi = plots.catalog_perigee_azimuth(members, stats)
    eps, ecc = members[:, 9:11].T
    hp = np.empty(len(members))
    np.divide(-plots.MU_EARTH*(1-ecc), 2*eps, out=hp)
    hp -= plots.R_EARTH_KM
    grids, counts_columns = [], []
    union = score >= individual_thresholds.min()
    for (key, stem, xlabel), x in zip(MAPS, (eps, hp)):
        with np.load(input_dir/'mappe_percentili'/f'{stem}_griglie.npz') as z:
            previous = z['occurrences']
            ex, ey = z['edges_x'], z['edges_y']
            previous_is_corrected = (('angle_convention' in z and str(z['angle_convention']) == ANGLE_CONVENTION)
                                     or ('frame_note' in z and str(z['frame_note']) == plots.PHI_CATALOG_NOTE)
                                     or ('y_label' in z and str(z['y_label']) in (
                                         plots.PHI_CATALOG_LABEL, r"$\phi_p$ al perigeo, catalogo [deg]")))
        # Verifica dei dati di partenza, nella convenzione effettivamente salvata.
        previous_y = phi if previous_is_corrected else members[:, 11] % 360
        prior_geometry = geometry_from_cells(cells_in_grid(x, previous_y, ex, ey), rows, ex, ey, len(stats))
        reconstructed = plots.best_transfer_occurrences(prior_geometry, rows_winner, len(stats))
        if not np.array_equal(previous, reconstructed['occurrences']):
            raise ValueError(f'{key}: la griglia salvata non corrisponde ai vincitori e ai membri correnti.')
        if n_bins is not None:
            # Mantiene il dominio e ricalcola le celle dai membri originali.
            ex = np.linspace(ex[0], ex[-1], int(n_bins)+1)
            ey = np.linspace(ey[0], ey[-1], int(n_bins)+1)
        cells = cells_in_grid(x, phi, ex, ey)
        geometry = geometry_from_cells(cells, rows, ex, ey, len(stats))
        data = plots.best_transfer_occurrences(geometry, rows_winner, len(stats))
        counts = data['occurrences']
        thresholds = np.percentile(counts[counts > 0], plots.OCCURRENCE_PERCENTILES)
        per_member = counts.ravel()[cells]
        union |= per_member >= thresholds.min()
        counts_columns.append(per_member)
        selected_counts = []
        for threshold in individual_thresholds:
            pair_take = frequency[geometry['pair_rows']] >= threshold
            flat = np.bincount(geometry['pair_cells'][pair_take],
                               weights=frequency[geometry['pair_rows'][pair_take]], minlength=counts.size)
            selected_counts.append(flat.reshape(counts.shape).astype(np.int64))
        grid = dict(key=key, occurrences=counts, edges_x=ex, edges_y=ey, thresholds=thresholds,
                    individual_counts=np.array(selected_counts), xlabel=xlabel,
                    ylabel=plots.PHI_CATALOG_LABEL, map_data=data,
                    old_convention=('phi_catalog' if previous_is_corrected else 'omega_classico'))
        if np.any(grid['individual_counts'] > counts):
            raise ValueError('Copertura selezionata maggiore di quella completa.')
        grids.append(grid)
        print(f'{key}: conteggi originali verificati; nuove soglie {thresholds.tolist()}.', flush=True)
    selected_members = members[union]
    per_map_counts = np.column_stack(counts_columns)[union].astype(np.int64)
    selected_scores = score[union]
    summary = dict(catalog=str(catalog), report=str(report_path),
                   report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                   catalog_angle_convention=ANGLE_CONVENTION, population_count=len(members),
                   grid_shapes={grid['key']: list(grid['occurrences'].shape) for grid in grids},
                   selected_asteroids=len(winner_ids), winning_cluster_ids=winner_ids.tolist(),
                   reconstructed_trajectories=len(selected_members), cell_percentiles=list(plots.OCCURRENCE_PERCENTILES),
                   individual_percentiles=list(INDIVIDUAL_PERCENTILES),
                   individual_thresholds=individual_thresholds.tolist(), maximum_individual_score=int(score.max()),
                   model='CR3BP nel riferimento del catalogo; ricorrenza del cluster attribuita ai membri; nessuna rotazione ETD ECLIPJ2000.',
                   grid_verification='Griglie originali ricostruite esattamente; griglie corrette contano un asteroide al massimo una volta per cella.')
    del members, eps, ecc, hp, phi, rows, score, counts_columns, union, cells, per_member
    gc.collect()
    np.savez_compressed(output/'membri_selezionati.npz', members=selected_members,
                        per_map_counts=per_map_counts, individual_scores=selected_scores)
    for grid in grids:
        np.savez_compressed(output/f'{grid["key"]}_griglie.npz', **{k:grid[k] for k in (
            'occurrences', 'edges_x', 'edges_y', 'thresholds', 'individual_counts')}, angle_convention=ANGLE_CONVENTION)
    (output/'preparazione.json').write_text(json.dumps(summary, indent=2)+'\n')
    save_canonical_maps(input_dir, grids)
    print(f'Selezione pronta: {len(selected_members):,} curve uniche su {summary["population_count"]:,} membri.', flush=True)
    return selected_members, per_map_counts, selected_scores, grids, summary


def render_maps(curves, members, per_map_counts, individual_scores, grids, summary, output, columns):
    radius = 0.
    for start in range(0, len(curves), 2048):
        radius = max(radius, float(np.linalg.norm(curves[start:start+2048], axis=1).max()))
    limits = np.array([[-radius, -radius], [radius, radius]])
    pages, records = [], []
    for column in columns:
        key = MAPS[column][0]
        grid = grids[column]
        maximum = int(grid['occurrences'].max())
        grid['normalization'] = Normalize(0, maximum)
        for kind, levels, thresholds in (
            ('celle', summary['cell_percentiles'], grid['thresholds']),
            ('individuali', summary['individual_percentiles'], summary['individual_thresholds']),
        ):
            folder = output/kind/key
            folder.mkdir(parents=True, exist_ok=True)
            group_pages = []
            for level, (percentile, threshold) in enumerate(zip(levels, thresholds)):
                scores = per_map_counts[:, column] if kind == 'celle' else individual_scores
                take = np.flatnonzero(scores >= threshold)
                tag = f'P{percentile:g}'.replace('.', '_')
                path = folder/f'{tag}_tre_riferimenti.png'
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
                    grid['selection_title'] = 'Celle dei cluster selezionati'
                    grid['selection_caption'] = 'Colore: copertura dei membri dei cluster selezionati'
                    label = 'Asteroidi il cui minimo DV sceglie il cluster della traiettoria'
                    criterion = (f'Ricorrenza dei cluster: P{percentile:g} sui membri — '
                                 f'almeno {int(np.ceil(threshold))} asteroidi per cluster')
                    normalization = Normalize(0, summary['maximum_individual_score'])
                cluster_count = len(np.unique(members[take, 1]))
                title = (f'Mappa {key[:2]}: {criterion}\n'
                         f'{len(take):,} traiettorie distinte · {cluster_count} cluster · '
                         f'griglia {grid["occurrences"].shape[0]} × {grid["occurrences"].shape[1]} · '
                         f'{curves.shape[-1]} punti per curva · ω del catalogo corretto per il verso')
                print(f'Disegno {key} {kind} {tag}: {len(take):,} traiettorie / {cluster_count} cluster...', flush=True)
                draw_combined(curves, members[take], scores[take], label, path,
                              summary['population_count'], title=title, normalization=normalization,
                              fixed_alpha=0.03, global_limits=limits, raster_chunks=True,
                              perigee_markers=True, selection_grid=grid, curve_indices=take, chunk_size=512)
                np.savez_compressed(folder/f'{tag}_selezione.npz', source_ids=members[take, 0].astype(np.int64),
                                    cluster_ids=members[take, 1].astype(np.int64), scores=scores[take],
                                    threshold=threshold, percentile=percentile, angle_convention=ANGLE_CONVENTION)
                records.append(dict(map=key, criterion=kind, percentile=percentile, threshold=float(threshold),
                                    trajectories=len(take), clusters=cluster_count, file=str(path)))
                group_pages.append(path)
                gc.collect()
            package_pdf(group_pages, folder/'percentili_tre_riferimenti.pdf')
            pages.extend(group_pages)
    return pages, records


_RENDER_ARGUMENTS = None


def render_one_map(column):
    return render_maps(*_RENDER_ARGUMENTS, columns=[column])


def render(curves, members, counts, scores, grids, summary, output, workers=1):
    arguments = (curves, members, counts, scores, grids, summary, output)
    if workers > 1:
        from multiprocessing import get_context
        global _RENDER_ARGUMENTS
        _RENDER_ARGUMENTS = arguments
        try:
            with get_context('fork').Pool(min(workers, len(MAPS))) as pool:
                result = pool.map(render_one_map, range(len(MAPS)))
        finally:
            _RENDER_ARGUMENTS = None
        pages = [page for group, _ in result for page in group]
        records = [record for _, group in result for record in group]
    else:
        pages, records = render_maps(*arguments, columns=range(len(MAPS)))
    package_pdf(pages, output/'tavole_azimut_catalogo_tutti_percentili.pdf')
    summary['figures'] = records
    summary['trajectory_panels'] = ['sinodico Terra-Luna', 'geocentrico del catalogo', 'lunocentrico, assi del catalogo']
    (output/'riepilogo.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Completate {len(pages)} tavole in {output}', flush=True)


def load_render_inputs(output):
    path = output/'riepilogo.json'
    summary = json.loads((path if path.exists() else output/'preparazione.json').read_text())
    if summary.get('catalog_angle_convention') != ANGLE_CONVENTION:
        raise ValueError('Selezioni salvate con una diversa convenzione angolare.')
    with np.load(output/'membri_selezionati.npz') as z:
        members, counts, scores = (z[k] for k in ('members', 'per_map_counts', 'individual_scores'))
    provenance = json.loads((output/'curve_provenance.json').read_text())
    digest = hashlib.sha256(np.ascontiguousarray(members).tobytes()).hexdigest()
    if digest != provenance['signature']['members_sha256']:
        raise ValueError('Curve e membri selezionati non corrispondono.')
    curves = np.load(output/'reconstructed_curves.npy', mmap_mode='r')
    done = np.load(output/'reconstructed_done.npy', mmap_mode='r')
    if (curves.shape != (len(members), 2, provenance['samples'])
            or done.shape != (len(members),) or not done.all()
            or counts.shape != (len(members), len(MAPS)) or scores.shape != (len(members),)):
        raise ValueError('Curve o conteggi salvati incompleti/incoerenti.')
    grids = []
    for key, _, xlabel in MAPS:
        with np.load(output/f'{key}_griglie.npz') as z:
            if str(z['angle_convention']) != ANGLE_CONVENTION:
                raise ValueError('Griglia con convenzione angolare diversa.')
            grid = {k:z[k] for k in ('occurrences', 'edges_x', 'edges_y', 'thresholds', 'individual_counts')}
        grid.update(key=key, xlabel=xlabel, ylabel=plots.PHI_CATALOG_LABEL)
        grids.append(grid)
    summary['curve_provenance'] = provenance
    return curves, members, counts, scores, grids, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=plots.OUTPUT_DIR)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bins', type=int, help='Bin per asse; se omesso usa la griglia di input.')
    parser.add_argument('--reuse-curves', type=Path, help='Prodotti verificati delle curve da riutilizzare.')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--render-workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.render_workers < 1:
        parser.error('Il numero di processi deve essere positivo.')
    if args.prepare_only and args.render_only:
        parser.error('--prepare-only e --render-only sono alternativi.')
    if args.bins is not None and args.bins < 1:
        parser.error('--bins deve essere positivo.')
    if args.render_only and args.bins is not None:
        parser.error('Cambiare i bin richiede il ricalcolo delle selezioni: ometti --render-only.')
    output = args.output or args.input/'traiettorie_azimut_catalogo'
    output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        render(*load_render_inputs(output), output, workers=args.render_workers)
        return
    members, counts, scores, grids, summary = prepare(args.input, output, n_bins=args.bins)
    if args.prepare_only:
        return
    from GA_18_phi_curve_data import prepare_curves
    curves, provenance = prepare_curves(members, Path(summary['catalog']).with_name('trajectory_exit_catalog.txt'),
                                        output, workers=args.workers, n_plot=101,
                                        reusable_products=[args.reuse_curves or args.input/'traiettorie_phi'])
    summary['curve_provenance'] = provenance
    render(curves, members, counts, scores, grids, summary, output, workers=args.render_workers)


if __name__ == '__main__':
    main()
