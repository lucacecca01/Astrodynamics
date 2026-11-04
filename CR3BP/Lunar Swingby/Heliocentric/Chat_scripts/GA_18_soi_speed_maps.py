"""Mappe 05–12 colorate con la velocita geocentrica individuale alla SOI.

Ogni traiettoria contribuisce una volta per cella, anche quando piu asteroidi
ne selezionano il cluster. La media e calcolata sui moduli, non sui vettori.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, PowerNorm

import GA_18_plots as plots
from GA_18_catalog_azimuth_trajectories import cells_in_grid, geometry_from_cells, read_eligible_members
from GA_18_catalog_epoch_trajectories import MAPS as CATALOG_MAPS
from GA_18_percentile_trajectories import package_pdf


BASE = plots.OUTPUT_DIR
CATALOG_TABLES = BASE/'traiettorie_05_10_J2000_catalogo_100x100'
PHI_TABLES = BASE/'traiettorie_phi'
PHI_MAPS = (
    ('11_eps_phi', '11_migliori_transfer_occorrenze_eps_phi', plots.EPS_LABEL, r'$\phi_p$ ECLIPJ2000 [deg]'),
    ('12_hp_phi', '12_migliori_transfer_occorrenze_hp_phi', plots.HP_LABEL, r'$\phi_p$ ECLIPJ2000 [deg]'),
)
MAPS = (*CATALOG_MAPS, *PHI_MAPS)


def speed_statistics(cells, speeds, shape):
    """Una riga per contributo gia deduplicato; celle vuote salvate come NaN."""
    size = int(np.prod(shape))
    count = np.bincount(cells, minlength=size)
    total = np.bincount(cells, weights=speeds, minlength=size)
    squared = np.bincount(cells, weights=speeds**2, minlength=size)
    mean = np.full(size, np.nan)
    np.divide(total, count, out=mean, where=count > 0)
    minimum, maximum = np.full(size, np.inf), np.full(size, -np.inf)
    np.minimum.at(minimum, cells, speeds)
    np.maximum.at(maximum, cells, speeds)
    minimum[count == 0] = maximum[count == 0] = np.nan
    second = np.full(size, np.nan)
    np.divide(squared, count, out=second, where=count > 0)
    std = np.sqrt(np.maximum(second-mean**2, 0))
    return {k: value.reshape(shape) for k, value in dict(
        trajectory_count=count, speed_sum=total, speed_squared_sum=squared,
        speed_mean=mean, speed_minimum=minimum, speed_maximum=maximum, speed_std=std).items()}


def read_reference(folder, key):
    path = folder/f'{key}_griglie.npz'
    with np.load(path) as z:
        grid = {k: z[k] for k in ('occurrences', 'edges_x', 'edges_y')}
    if grid['occurrences'].shape != (100, 100):
        raise ValueError(f'{key}: attesa la griglia 100x100 gia verificata.')
    return grid


def prepare(output):
    summary = json.loads((CATALOG_TABLES/'riepilogo.json').read_text())
    phi_summary = json.loads((PHI_TABLES/'riepilogo.json').read_text())
    winners = summary['winners']
    if len(winners) != len(phi_summary['winners']):
        raise ValueError('Asteroidi differenti tra le mappe del catalogo e J2000.')
    for a, b in zip(winners, phi_summary['winners']):
        if any(a[k] != b[k] for k in ('spkid', 'cluster_id', 'cluster_row')):
            raise ValueError('Identita o cluster vincenti differenti.')
        for k in ('jd_etd', 'jd_arrival', 'dv_km_s', 'tof_days', 'phase_etd_deg'):
            if not np.isclose(a[k], b[k], rtol=0, atol=1e-10):
                raise ValueError(f'Epoche o prestazioni differenti: {k}.')
    catalog = Path(summary['catalog'])
    stats = plots.load_table(catalog, 15, allow_extra=True)
    stats = stats[np.argsort(stats[:, 0])]
    winner_rows = np.array([w['cluster_row'] for w in winners])
    np.testing.assert_array_equal(stats[winner_rows, 0], [w['cluster_id'] for w in winners])
    members = read_eligible_members(catalog.with_name('trajectory_exit_catalog.txt'),
                                    np.unique(stats[winner_rows, 0]))
    if len(members) != summary['population_count']:
        raise ValueError('Popolazione del catalogo diversa dalle mappe originali.')
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    speeds = np.linalg.norm(members[:, 2:5], axis=1)
    if not np.isfinite(speeds).all() or np.any(speeds < 0):
        raise ValueError('Velocita SOI non valide.')
    eps, ecc = members[:, 9:11].T
    hp = np.empty(len(members))
    np.divide(-plots.MU_EARTH*(1-ecc), 2*eps, out=hp)
    hp -= plots.R_EARTH_KM
    omega = plots.catalog_perigee_azimuth(members, stats)
    theta = (omega-plots.MOON_RATE_DEG_DAY*members[:, 5]) % 360
    coordinates = ((eps, ecc), (eps, omega), (hp, ecc), (hp, omega), (hp, theta), (eps, theta))
    records = []
    for (key, _, xlabel, ylabel), (x, y) in zip(CATALOG_MAPS, coordinates):
        grid = read_reference(CATALOG_TABLES, key)
        cells = cells_in_grid(x, y, grid['edges_x'], grid['edges_y'])
        geometry = geometry_from_cells(cells, rows, grid['edges_x'], grid['edges_y'], len(stats))
        actual = plots.best_transfer_occurrences(geometry, winner_rows, len(stats))['occurrences']
        np.testing.assert_array_equal(actual, grid['occurrences'])
        values = speed_statistics(cells, speeds, grid['occurrences'].shape)
        np.testing.assert_array_equal(values['trajectory_count'] > 0, grid['occurrences'] > 0)
        np.savez_compressed(output/f'{key}_velocita_soi.npz', **grid, **values,
                            x_label=xlabel, y_label=ylabel,
                            counting='Una traiettoria distinta per cella.', units='km/s')
        records.append(dict(key=key, xlabel=xlabel, ylabel=ylabel, assignments=len(cells)))
        print(f'{key}: conteggi originali verificati; velocita aggregate.', flush=True)
    groups = {c: np.flatnonzero(rows == c) for c in np.unique(rows)}
    grids = [read_reference(PHI_TABLES, m[0]) for m in PHI_MAPS]
    reconstructed = [np.zeros_like(g['occurrences']) for g in grids]
    key_parts = [[], []]
    for winner in winners:
        ids = groups.get(winner['cluster_row'])
        if ids is None:
            continue
        phi = (omega[ids]+winner['phase_etd_deg']) % 360
        for column, (x, grid) in enumerate(zip((eps, hp), grids)):
            cells = cells_in_grid(x[ids], phi, grid['edges_x'], grid['edges_y'])
            reconstructed[column].flat[np.unique(cells)] += 1
            key_parts[column].append(ids*grid['occurrences'].size+cells)
    for column, ((key, _, xlabel, ylabel), grid) in enumerate(zip(PHI_MAPS, grids)):
        np.testing.assert_array_equal(reconstructed[column], grid['occurrences'])
        # Piu ETD possono riportare lo stesso membro nella stessa cella.
        keys = np.unique(np.concatenate(key_parts[column]))
        key_parts[column].clear()
        ids, cells = keys//grid['occurrences'].size, keys % grid['occurrences'].size
        values = speed_statistics(cells, speeds[ids], grid['occurrences'].shape)
        np.testing.assert_array_equal(values['trajectory_count'] > 0, grid['occurrences'] > 0)
        np.savez_compressed(output/f'{key}_velocita_soi.npz', **grid, **values,
                            x_label=xlabel, y_label=ylabel,
                            counting='Tutte le ETD; una traiettoria distinta per cella, senza ripetizioni.', units='km/s')
        records.append(dict(key=key, xlabel=xlabel, ylabel=ylabel, assignments=len(keys)))
        print(f'{key}: fasi e conteggi originali verificati; membri deduplicati per cella.', flush=True)
        del keys, ids, cells, values
        gc.collect()
    represented = np.unique(rows)
    metadata = dict(catalog=str(catalog), bins=100, maps=records, selected_asteroids=len(winners),
                    represented_asteroids=int(np.isin(winner_rows, represented).sum()),
                    represented_clusters=len(represented), distinct_members=len(members),
                    speed_definition='Norma euclidea delle colonne vx_soi, vy_soi, vz_soi individuali; geocentrica, km/s.',
                    counting='Ogni source_id contribuisce una volta per cella; nelle mappe 11/12 si deduplicano anche le ETD.',
                    individual_speed_min=float(speeds.min()), individual_speed_max=float(speeds.max()),
                    report_sha256=summary['report_sha256'],
                    source_metadata_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                        CATALOG_TABLES/'riepilogo.json', PHI_TABLES/'riepilogo.json')})
    (output/'dati.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def render(output, metadata, statistic, high_peaks=False):
    field, name = {
        'mean': ('speed_mean', 'media'),
        'minimum': ('speed_minimum', 'minima'),
        'maximum': ('speed_maximum', 'massima'),
    }[statistic]
    lower, upper = np.inf, -np.inf
    for record in metadata['maps']:
        with np.load(output/f'{record["key"]}_velocita_soi.npz') as z:
            values = z[field]
            lower, upper = min(lower, np.nanmin(values)), max(upper, np.nanmax(values))
    if lower == upper:
        lower, upper = lower-.001, upper+.001
    normalization = (PowerNorm(gamma=3, vmin=float(lower), vmax=float(upper))
                     if high_peaks else Normalize(float(lower), float(upper)))
    cmap = 'inferno' if high_peaks else 'viridis'
    suffix = '_picchi' if high_peaks else ''
    scale_note = ('Scala a potenza γ = 3, comune alle otto mappe.' if high_peaks
                  else 'Scala colore comune alle otto mappe.')
    pages = []
    for record in metadata['maps']:
        key = record['key']
        with np.load(output/f'{key}_velocita_soi.npz') as z:
            ex, ey, values = z['edges_x'], z['edges_y'], z[field]
        fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
        mesh = ax.pcolormesh(ex, ey, np.ma.masked_invalid(values.T), shading='flat',
                             cmap=cmap, norm=normalization)
        ax.set(xlabel=record['xlabel'], ylabel=record['ylabel'],
               title=(f'Mappa {key[:2]}: velocita {name} alla SOI — migliori trasferimenti\n'
                      f"{metadata['represented_asteroids']}/{metadata['selected_asteroids']} asteroidi — "
                      f"{metadata['represented_clusters']} cluster — "
                      f"{metadata['distinct_members']:,} traiettorie distinte"))
        ax.grid(alpha=.15)
        colorbar = fig.colorbar(mesh, ax=ax, label=f'Velocita geocentrica {name} alla SOI [km/s]')
        if high_peaks:
            from matplotlib.ticker import FormatStrFormatter
            colorbar.set_ticks(normalization.inverse(np.linspace(0, 1, 7)))
            colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        frame = ('ω corretto per il verso; assi del catalogo, Luna su +x a ETD.'
                 if key.startswith(('06', '08')) else
                 'θ: Luna → periapside alla sua epoca, con verso gia corretto.'
                 if key.startswith(('09', '10')) else
                 'Azimut ECLIPJ2000 con la fase lunare all’ETD del miglior trasferimento.'
                 if key.startswith(('11', '12')) else
                 'Elementi osculatori geocentrici al perigeo nel riferimento del catalogo.')
        fig.supxlabel(
            'Velocita individuali dei membri dei cluster vincenti; orbite legate con perigeo sopra la superficie.\n'
            'Ogni traiettoria conta una volta per cella, anche se associata a piu asteroidi. Celle vuote bianche.\n'
            f'{frame} {scale_note} Modello medio dei cluster.', fontsize=9)
        path = output/f'{key}_velocita_soi_{name}{suffix}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pages.append(path)
    pdf = output/f'mappe_05_12_velocita_soi_{name}{suffix}.pdf'
    package_pdf(pages, pdf)
    metadata.update(statistic=statistic, color_limits_km_s=[float(lower), float(upper)],
                    normalization='power' if high_peaks else 'linear', gamma=3 if high_peaks else 1,
                    cmap=cmap, figures=[str(p) for p in pages], pdf=str(pdf))
    (output/'riepilogo.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (output/f'riepilogo_{name}{suffix}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Salvate otto mappe e PDF in {output}.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'mappe_velocita_soi_100x100')
    parser.add_argument('--statistic', choices=('mean', 'minimum', 'maximum'), default='mean')
    parser.add_argument('--high-peaks', action='store_true', help='Evidenzia i massimi con scala a potenza gamma=3 e palette inferno.')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only and args.render_only:
        parser.error('prepare-only e render-only sono alternativi.')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = (json.loads((args.output/'dati.json').read_text()) if args.render_only else prepare(args.output))
    if not args.prepare_only:
        render(args.output, metadata, args.statistic, high_peaks=args.high_peaks)


if __name__ == '__main__':
    main()
