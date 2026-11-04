"""Confronti visivi media/massimo SOI per le sette risoluzioni gia studiate.

Ricostruisce solo gli aggregati dalle traiettorie salvate. Nessun nuovo Lambert
o clustering. --render-only riutilizza gli aggregati di questo confronto.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))

import numpy as np

from GA_18_soi_speed_maps import BASE, MAPS, CATALOG_TABLES
from GA_18_catalog_azimuth_trajectories import read_eligible_members, cells_in_grid
import GA_18_plots as plots


BINS = (100, 200, 500, 1000, 2000, 5000, 10000)
OUTPUT = BASE/'confronto_immagini_binning_velocita_soi'
SPEED_DATA = BASE/'mappe_velocita_soi_100x100'
STUDY = BASE/'studio_binning_velocita_soi'/'confronto_completo.json'


def save_sparse(output, key, n, ex, ey, cells, values, reference):
    """Una riga per cella occupata: evita array float 10000 x 10000."""
    occupied, inverse, count = np.unique(cells, return_inverse=True, return_counts=True)
    mean = np.bincount(inverse, weights=values)/count
    maximum = np.full(len(count), -np.inf)
    np.maximum.at(maximum, inverse, values)
    gap = np.maximum(0, (maximum-mean)/maximum)
    observed = dict(occupied_cells=len(count), contributions=int(count.sum()),
                    singleton_pct_cells=float(100*np.mean(count == 1)),
                    median_relative_gap_pct=float(100*np.median(gap)),
                    within_5pct_cells_pct=float(100*np.mean(gap <= .05)),
                    within_5pct_contributions_pct=float(100*count[gap <= .05].sum()/count.sum()))
    for name, value in observed.items():
        np.testing.assert_allclose(value, reference[name], rtol=0, atol=1e-10,
                                   err_msg=f'{key}, {n}: {name} diverso dallo studio.')
    if n == 100:
        with np.load(SPEED_DATA/f'{key}_velocita_soi.npz') as z:
            np.testing.assert_array_equal(count, z['trajectory_count'].ravel()[occupied])
            np.testing.assert_allclose(mean, z['speed_mean'].ravel()[occupied], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(maximum, z['speed_maximum'].ravel()[occupied])
    path = output/'dati'/f'{key}_{n}.npz'
    np.savez_compressed(path, occupied=occupied.astype(np.int32), count=count.astype(np.int32),
                        speed_mean=mean, speed_maximum=maximum, edges_x=ex, edges_y=ey)
    print(f'{key} {n} x {n}: aggregati esatti salvati e verificati.', flush=True)
    return path


def prepare(output, render=True):
    (output/'dati').mkdir(parents=True, exist_ok=True)
    reference = {(r['map'], r['bins']): r for r in json.loads(STUDY.read_text())}
    summary = json.loads((CATALOG_TABLES/'riepilogo.json').read_text())
    catalog = Path(summary['catalog'])
    stats = plots.load_table(catalog, 15, allow_extra=True)
    stats = stats[np.argsort(stats[:, 0])]
    winners = summary['winners']
    winner_rows = np.array([w['cluster_row'] for w in winners])
    np.testing.assert_array_equal(stats[winner_rows, 0], [w['cluster_id'] for w in winners])
    members = read_eligible_members(catalog.with_name('trajectory_exit_catalog.txt'),
                                    np.unique(stats[winner_rows, 0]))
    if len(members) != summary['population_count']:
        raise ValueError('Popolazione diversa dalle mappe e dallo studio del binning.')
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    speeds = np.linalg.norm(members[:, 2:5], axis=1)
    context = dict(bins=list(BINS), distinct_members=len(members), selected_asteroids=len(winners),
                   represented_asteroids=4443, represented_clusters=len(np.unique(rows)),
                   speed_limits_km_s=[float(speeds.min()), float(speeds.max())],
                   catalog=str(catalog), report_sha256=summary['report_sha256'],
                   study_sha256=hashlib.sha256(STUDY.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                   maps=[dict(key=k, xlabel=x, ylabel=y) for k, _, x, y in MAPS])
    # Queste consistenze impediscono di associare immagini a vincitori differenti.
    metadata = json.loads((SPEED_DATA/'dati.json').read_text())
    for name in ('distinct_members', 'selected_asteroids', 'represented_clusters', 'report_sha256'):
        if context[name] != metadata[name]:
            raise ValueError(f'Popolazioni differenti: {name}.')
    context['represented_asteroids'] = metadata['represented_asteroids']
    (output/'dati.json').write_text(json.dumps(context, indent=2)+'\n')
    eps, ecc = members[:, 9:11].T
    hp = np.empty(len(members))
    np.divide(-plots.MU_EARTH*(1-ecc), 2*eps, out=hp)
    hp -= plots.R_EARTH_KM
    omega = plots.catalog_perigee_azimuth(members, stats)
    theta = (omega-plots.MOON_RATE_DEG_DAY*members[:, 5]) % 360
    coordinates = ((eps, ecc), (eps, omega), (hp, ecc), (hp, omega), (hp, theta), (eps, theta))
    if render:
        from GA_18_speed_binning_render import render_map
    products = []
    for (key, *_), (x, y) in zip(MAPS[:6], coordinates):
        with np.load(SPEED_DATA/f'{key}_velocita_soi.npz') as z:
            original_x, original_y = z['edges_x'], z['edges_y']
        cases = []
        for n in BINS:
            ex = original_x if n == 100 else np.linspace(original_x[0], original_x[-1], n+1)
            ey = original_y if n == 100 else np.linspace(original_y[0], original_y[-1], n+1)
            cells = cells_in_grid(x, y, ex, ey)
            cases.append(save_sparse(output, key, n, ex, ey, cells, speeds, reference[(key, n)]))
            del cells
        if render:
            products.append(render_map(key, cases, context, output))
            (output/'prodotti.json').write_text(json.dumps(products, indent=2)+'\n')
        gc.collect()
    groups = {row: np.flatnonzero(rows == row) for row in np.unique(rows)}
    for (key, *_), x in zip(MAPS[6:], (eps, hp)):
        with np.load(SPEED_DATA/f'{key}_velocita_soi.npz') as z:
            original_x, original_y = z['edges_x'], z['edges_y']
        cases = []
        for n in BINS:
            ex = original_x if n == 100 else np.linspace(original_x[0], original_x[-1], n+1)
            ey = original_y if n == 100 else np.linspace(original_y[0], original_y[-1], n+1)
            parts = []
            for winner in winners:
                ids = groups.get(winner['cluster_row'])
                if ids is None:
                    continue
                phi = (omega[ids]+winner['phase_etd_deg']) % 360
                cells = cells_in_grid(x[ids], phi, ex, ey)
                parts.append(ids*(n*n)+cells)
            combined = np.concatenate(parts)
            parts.clear()
            unique = np.unique(combined)
            del combined
            ids, cells = unique//(n*n), unique % (n*n)
            cases.append(save_sparse(output, key, n, ex, ey, cells, speeds[ids], reference[(key, n)]))
            del unique, ids, cells
            gc.collect()
        if render:
            products.append(render_map(key, cases, context, output))
            (output/'prodotti.json').write_text(json.dumps(products, indent=2)+'\n')
        gc.collect()
    return context, products


def package(output, context, products):
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    path = output/'confronto_05_12_tutti_binning.pdf'
    # Una PNG alla volta, senza nuove dipendenze o ricompressione JPEG.
    with PdfPages(path) as full:
        for record in products:
            for image_path in record['figures']:
                with Image.open(image_path) as source:
                    pixels = np.array(source.convert('RGB'))
                height, width = pixels.shape[:2]
                fig = plt.figure(figsize=(width/150, height/150))
                ax = fig.add_axes((0, 0, 1, 1))
                ax.imshow(pixels, interpolation='nearest')
                ax.set_axis_off()
                full.savefig(fig, dpi=150)
                plt.close(fig)
                del pixels
    result = dict(**context, products=products, pdf=str(path),
                  native_definition='Un pixel per cella originale; palette con colore del valore, bianco per celle vuote.',
                  overview_definition='Riduzione BOX della superficie colorata, non un nuovo binning delle velocita.',
                  zoom_definition='Stesso intervallo fisico in tutte le sette risoluzioni di ciascuna mappa.')
    (output/'riepilogo.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Confronto completo: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only and args.render_only:
        parser.error('prepare-only e render-only sono alternativi.')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        from GA_18_speed_binning_render import render_map
        context = json.loads((args.output/'dati.json').read_text())
        products = [render_map(key, [args.output/'dati'/f'{key}_{n}.npz' for n in BINS],
                               context, args.output) for key, *_ in MAPS]
    else:
        context, products = prepare(args.output, render=not args.prepare_only)
    if not args.prepare_only:
        package(args.output, context, products)


if __name__ == '__main__':
    main()
