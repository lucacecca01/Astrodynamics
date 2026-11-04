"""Mappe 05–12 delle occorrenze medie per traiettoria nella cella.

La popolazione resta quella delle mappe dei cluster vincenti. Nelle mappe
J2000 ogni membro contribuisce soltanto alle celle raggiunte dalle sue ETD.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import GA_18_plots as plots
from GA_18_catalog_azimuth_trajectories import cells_in_grid, geometry_from_cells, read_eligible_members
from GA_18_percentile_trajectories import package_pdf
from GA_18_soi_speed_maps import CATALOG_TABLES, PHI_TABLES, MAPS, read_reference


OUTPUT = plots.OUTPUT_DIR/'mappe_occorrenze_medie_100x100'


def save_map(output, spec, grid, individual, trajectory_count, total_instances):
    key, _, xlabel, ylabel = spec
    occupied = trajectory_count > 0
    np.testing.assert_array_equal(occupied, grid['occurrences'] > 0)
    if individual.sum() != total_instances or np.any(individual[occupied] < trajectory_count[occupied]):
        raise ValueError(f'{key}: conteggio delle associazioni individuali incoerente.')
    mean, ratio = np.full(individual.shape, np.nan), np.full(individual.shape, np.nan)
    np.divide(individual, trajectory_count, out=mean, where=occupied)
    np.divide(grid['occurrences'], trajectory_count, out=ratio, where=occupied)
    np.savez_compressed(output/f'{key}_occorrenze_medie.npz', **grid,
                        individual_occurrences=individual, trajectory_count=trajectory_count,
                        mean_occurrences_per_trajectory=mean,
                        distinct_asteroids_per_trajectory=ratio, x_label=xlabel, y_label=ylabel)
    print(f'{key}: media per traiettoria {np.nanmin(mean):.6g}–{np.nanmax(mean):.6g}; '
          f'{int(trajectory_count.sum()):,} contributi distinti al denominatore.', flush=True)
    return dict(key=key, xlabel=xlabel, ylabel=ylabel,
                individual_occurrences=int(individual.sum()),
                denominator_total=int(trajectory_count.sum()),
                minimum=float(np.nanmin(mean)), maximum=float(np.nanmax(mean)))


def prepare(output):
    summary_path = CATALOG_TABLES/'riepilogo.json'
    summary = json.loads(summary_path.read_text())
    phi_summary = json.loads((PHI_TABLES/'riepilogo.json').read_text())
    winners = summary['winners']
    if len(winners) != len(phi_summary['winners']):
        raise ValueError('Vincitori diversi nelle mappe del catalogo e J2000.')
    for a, b in zip(winners, phi_summary['winners']):
        if any(a[k] != b[k] for k in ('spkid', 'cluster_id', 'jd_etd', 'phase_etd_deg')):
            raise ValueError('Identita, epoche o fasi diverse tra le mappe.')
    catalog = Path(summary['catalog'])
    stats = plots.load_table(catalog, 15, allow_extra=True)
    stats = stats[np.argsort(stats[:, 0])]
    winner_rows = np.array([w['cluster_row'] for w in winners])
    np.testing.assert_array_equal(stats[winner_rows, 0], [w['cluster_id'] for w in winners])
    frequency = np.bincount(winner_rows, minlength=len(stats))
    members = read_eligible_members(catalog.with_name('trajectory_exit_catalog.txt'),
                                    stats[frequency > 0, 0])
    if len(members) != summary['population_count']:
        raise ValueError('Popolazione diversa dalle mappe originali.')
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    total_instances = int(frequency[rows].sum())
    eps, ecc = members[:, 9:11].T
    hp = -plots.MU_EARTH*(1-ecc)/(2*eps)-plots.R_EARTH_KM
    omega = plots.catalog_perigee_azimuth(members, stats)
    theta = (omega-plots.MOON_RATE_DEG_DAY*members[:, 5]) % 360
    coordinates = ((eps, ecc), (eps, omega), (hp, ecc), (hp, omega), (hp, theta), (eps, theta))
    records = []
    denominator_dir = plots.OUTPUT_DIR/'mappe_velocita_soi_100x100'
    for spec, (x, y) in zip(MAPS[:6], coordinates):
        grid = read_reference(CATALOG_TABLES, spec[0])
        cells = cells_in_grid(x, y, grid['edges_x'], grid['edges_y'])
        geometry = geometry_from_cells(cells, rows, grid['edges_x'], grid['edges_y'], len(stats))
        actual = plots.best_transfer_occurrences(geometry, winner_rows, len(stats))['occurrences']
        np.testing.assert_array_equal(actual, grid['occurrences'])
        shape = grid['occurrences'].shape
        count = np.bincount(cells, minlength=int(np.prod(shape))).reshape(shape)
        individual = np.bincount(cells, weights=frequency[rows], minlength=count.size).astype(np.int64).reshape(shape)
        with np.load(denominator_dir/f'{spec[0]}_velocita_soi.npz') as z:
            np.testing.assert_array_equal(count, z['trajectory_count'])
        records.append(save_map(output, spec, grid, individual, count, total_instances))
    groups = {c: np.flatnonzero(rows == c) for c in np.unique(rows)}
    grids = [read_reference(PHI_TABLES, spec[0]) for spec in MAPS[6:]]
    individuals = [np.zeros_like(g['occurrences']) for g in grids]
    reconstructed = [np.zeros_like(g['occurrences']) for g in grids]
    for winner in winners:
        ids = groups.get(winner['cluster_row'])
        if ids is None:
            continue
        phi = (omega[ids]+winner['phase_etd_deg']) % 360
        for column, (x, grid) in enumerate(zip((eps, hp), grids)):
            cells = cells_in_grid(x[ids], phi, grid['edges_x'], grid['edges_y'])
            individuals[column] += np.bincount(cells, minlength=grid['occurrences'].size).reshape(grid['occurrences'].shape)
            reconstructed[column].flat[np.unique(cells)] += 1
    for spec, grid, individual, actual in zip(MAPS[6:], grids, individuals, reconstructed):
        np.testing.assert_array_equal(actual, grid['occurrences'])
        with np.load(denominator_dir/f'{spec[0]}_velocita_soi.npz') as z:
            count = z['trajectory_count']
        records.append(save_map(output, spec, grid, individual, count, total_instances))
    represented = np.unique(rows)
    metadata = dict(catalog=str(catalog), maps=records, bins=100,
                    population='Traiettorie valide dei cluster vincenti gia rappresentate nelle mappe 05–12.',
                    selected_asteroids=len(winners),
                    represented_asteroids=int(np.isin(winner_rows, represented).sum()),
                    represented_clusters=len(represented), distinct_members=len(members),
                    total_individual_occurrences=total_instances,
                    formula='Media = somma delle occorrenze individuali / numero di source_id distinti nella cella.',
                    epoch_rule='Nelle mappe 11/12 si contano soltanto le associazioni asteroide–ETD che collocano il membro nella cella.',
                    report_sha256=summary['report_sha256'],
                    source_metadata_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                        summary_path, PHI_TABLES/'riepilogo.json')})
    (output/'dati.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def render(output, metadata, metric, scale):
    field, label, suffix = (
        ('mean_occurrences_per_trajectory', 'Occorrenze medie per traiettoria', 'media_per_traiettoria')
        if metric == 'mean' else
        ('distinct_asteroids_per_trajectory', 'Asteroidi distinti / traiettorie', 'asteroidi_diviso_traiettorie')
    )
    values_all = []
    for record in metadata['maps']:
        with np.load(output/f'{record["key"]}_occorrenze_medie.npz') as z:
            a = z[field]
            values_all.append(a[np.isfinite(a)])
    lower, upper = min(a.min() for a in values_all), max(a.max() for a in values_all)
    if lower == upper:
        lower, upper = lower/2, upper*2
    norm = LogNorm(lower, upper) if scale == 'log' else Normalize(lower, upper)
    pages = []
    for record in metadata['maps']:
        key = record['key']
        with np.load(output/f'{key}_occorrenze_medie.npz') as z:
            ex, ey, a = z['edges_x'], z['edges_y'], z[field]
        fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
        mesh = ax.pcolormesh(ex, ey, np.ma.masked_invalid(a.T), shading='flat', cmap='viridis', norm=norm)
        ax.set(xlabel=record['xlabel'], ylabel=record['ylabel'],
               title=(f'Mappa {key[:2]}: {label.lower()}\n'
                      f"{metadata['represented_asteroids']}/{metadata['selected_asteroids']} asteroidi — "
                      f"{metadata['represented_clusters']} cluster — {metadata['distinct_members']:,} traiettorie distinte"))
        ax.grid(alpha=.15)
        colorbar = fig.colorbar(mesh, ax=ax, label=label)
        if scale == 'log':
            from matplotlib.ticker import FuncFormatter
            ticks = np.array([lower, 1, 2, 5, 10, 20, 50, 100, 200, 500, upper])
            ticks = np.unique(ticks[(ticks >= lower) & (ticks <= upper)])
            if len(ticks) > 11:
                ticks = norm.inverse(np.linspace(0, 1, 8))
            colorbar.set_ticks(ticks)
            colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
        formula = ('Somma delle occorrenze delle singole traiettorie / numero di traiettorie distinte nella cella.'
                   if metric == 'mean' else
                   'Numero di asteroidi distinti della cella / numero di traiettorie distinte nella cella.')
        fig.supxlabel(
            f'{formula}\n'
            'Popolazione: traiettorie valide dei cluster vincenti, come nelle mappe originali; modello medio. '
            'Celle vuote bianche.\n'
            f"Nelle mappe 11/12 si usano soltanto le ETD che collocano il membro nella cella. "
            f"Scala colore {'logaritmica' if scale == 'log' else 'lineare'}, comune alle otto mappe.", fontsize=9)
        path = output/f'{key}_{suffix}_{scale}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pages.append(path)
    pdf = output/f'mappe_05_12_occorrenze_{suffix}_{scale}.pdf'
    package_pdf(pages, pdf)
    metadata.update(metric=metric, color_scale=scale, color_limits=[float(lower), float(upper)],
                    figures=[str(p) for p in pages], pdf=str(pdf))
    text = json.dumps(metadata, indent=2)+'\n'
    (output/'riepilogo.json').write_text(text)
    (output/f'riepilogo_{suffix}_{scale}.json').write_text(text)
    print(f'Completate otto mappe: {pdf}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--metric', choices=('mean', 'ratio'), default='mean')
    parser.add_argument('--scale', choices=('log', 'linear'), default='log')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only and args.render_only:
        parser.error('prepare-only e render-only sono alternativi.')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.output/'dati.json').read_text()) if args.render_only else prepare(args.output)
    if not args.prepare_only:
        render(args.output, metadata, args.metric, args.scale)


if __name__ == '__main__':
    main()
