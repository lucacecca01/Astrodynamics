"""Tavole 05–12: occorrenze, velocita SOI massima e sua direzione nel catalogo.

Usa le griglie gia calcolate: non ripete clustering, Lambert o propagazioni.
Le occorrenze restano il numero di asteroidi distinti, come nelle mappe originali.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from GA_18_soi_speed_maps import BASE, MAPS
from GA_18_percentile_trajectories import package_pdf


def load_grids(speed_dir, direction_dir):
    """Controlla che le tre metriche usino popolazione e celle identiche."""
    speed_metadata = json.loads((speed_dir/'dati.json').read_text())
    direction_metadata = json.loads((direction_dir/'dati.json').read_text())
    for field in ('catalog', 'bins', 'selected_asteroids', 'represented_asteroids',
                  'represented_clusters', 'distinct_members', 'report_sha256',
                  'source_metadata_sha256', 'maps'):
        if speed_metadata[field] != direction_metadata[field]:
            raise ValueError(f'Popolazioni velocita/direzioni differenti: {field}.')
    if speed_metadata['bins'] != 100:
        raise ValueError('Sono richieste le griglie 100 x 100 gia verificate.')

    records, source_hashes = [], {}
    for key, _, xlabel, ylabel in MAPS:
        speed_path = speed_dir/f'{key}_velocita_soi.npz'
        direction_path = direction_dir/f'{key}_direzioni_soi.npz'
        with np.load(speed_path) as speed, np.load(direction_path) as direction:
            for field in ('edges_x', 'edges_y', 'occurrences', 'trajectory_count'):
                np.testing.assert_array_equal(speed[field], direction[field],
                                              err_msg=f'{key}: {field} differente.')
            np.testing.assert_array_equal(speed['speed_maximum'], direction['max_speed_km_s'],
                                          err_msg=f'{key}: massimi SOI differenti.')
            frames = direction['frame_order'].tolist()
            catalog_frame = frames.index('catalog')
            record = dict(key=key, xlabel=xlabel, ylabel=ylabel,
                          edges_x=speed['edges_x'], edges_y=speed['edges_y'],
                          occurrences=speed['occurrences'], speed=speed['speed_maximum'],
                          angle=direction['maximum_angle_deg'][catalog_frame])
            occupied = record['occurrences'] > 0
            for field in ('occurrences', 'speed', 'angle'):
                if record[field].shape != (100, 100):
                    raise ValueError(f'{key}: dimensioni errate per {field}.')
            for field in ('speed', 'angle'):
                np.testing.assert_array_equal(np.isfinite(record[field]), occupied)
            if np.any((record['angle'][occupied] < 0) | (record['angle'][occupied] >= 360)):
                raise ValueError(f'{key}: direzioni fuori da [0, 360).')
            if np.any(record['speed'][occupied] <= 0):
                raise ValueError(f'{key}: moduli delle velocita non positivi.')
            np.testing.assert_array_equal(direction['max_source_id'] >= 0, occupied)
            records.append(record)
        for path in (speed_path, direction_path):
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records, speed_metadata, source_hashes


def render(speed_dir, direction_dir, output):
    records, metadata, source_hashes = load_grids(speed_dir, direction_dir)
    speed_min = min(float(np.nanmin(r['speed'])) for r in records)
    speed_max = max(float(np.nanmax(r['speed'])) for r in records)
    # Come nelle mappe dei picchi: gamma > 1 amplia il contrasto alle alte velocita.
    # Nessun taglio dei valori; stessi estremi per tutte le otto tavole.
    speed_norm = PowerNorm(gamma=3, vmin=speed_min, vmax=speed_max)
    angle_norm = Normalize(0, 360)
    occurrence_limits, pages = {}, []
    for record in records:
        key = record['key']
        occurrences = record['occurrences']
        occurrence_max = int(occurrences.max())
        occurrence_norm = Normalize(0, occurrence_max)
        occurrence_limits[key] = [0, occurrence_max]
        fig, axes = plt.subplots(1, 3, figsize=(19.2, 7.2), sharex=True, sharey=True,
                                 layout='constrained')
        panels = (
            (np.ma.masked_where(occurrences == 0, occurrences), 'viridis', occurrence_norm,
             'Occorrenze dei migliori trasferimenti', 'Asteroidi distinti nella cella'),
            (np.ma.masked_invalid(record['speed']), 'inferno', speed_norm,
             'Velocita massima alla SOI', r'$|\mathbf{v}_{\mathrm{SOI}}|_{\max}$ geocentrica [km/s]'),
            (np.ma.masked_invalid(record['angle']), 'twilight', angle_norm,
             'Direzione SOI — assi del catalogo', 'Direzione della traiettoria piu veloce [deg]'),
        )
        for index, (ax, (values, cmap, norm, title, label)) in enumerate(zip(axes, panels)):
            ax.set_facecolor('white')
            mesh = ax.pcolormesh(record['edges_x'], record['edges_y'], values.T,
                                 cmap=cmap, norm=norm, shading='flat')
            ax.set(xlabel=record['xlabel'], title=title,
                   xlim=(record['edges_x'][0], record['edges_x'][-1]),
                   ylim=(record['edges_y'][0], record['edges_y'][-1]))
            ax.xaxis.label.set_size(9)
            ax.tick_params(labelsize=9)
            ax.grid(alpha=.12)
            colorbar = fig.colorbar(mesh, ax=ax, shrink=.88, pad=.025, fraction=.045)
            colorbar.set_label(label, fontsize=9)
            colorbar.ax.tick_params(labelsize=8)
            if index == 0:
                colorbar.locator = MaxNLocator(nbins=7, integer=True)
                colorbar.update_ticks()
            elif index == 1:
                colorbar.set_ticks(speed_norm.inverse(np.linspace(0, 1, 7)))
                colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            else:
                colorbar.set_ticks(np.arange(0, 361, 60))
        axes[0].set_ylabel(record['ylabel'], fontsize=10)
        fig.suptitle(
            f'Mappa {key[:2]}: orbite al perigeo dei migliori trasferimenti\n'
            f"{metadata['represented_asteroids']}/{metadata['selected_asteroids']} asteroidi "
            f"· {metadata['represented_clusters']} cluster "
            f"· {metadata['distinct_members']:,} traiettorie distinte · griglia 100 × 100",
            fontsize=15)
        fig.supxlabel(
            'Sinistra: ogni asteroide conta al massimo una volta per cella; un solo trasferimento a minimo ΔV per asteroide.\n'
            'Centro e destra: massimo modulo SOI e direzione della stessa traiettoria. '
            'Angolo antiorario da +x; assi inerziali del catalogo, Luna su +x a ETD (t = 0).\n'
            'Velocita: scala a potenza γ = 3 comune alle otto mappe. '
            'Direzione: scala ciclica 0–360°. Celle vuote bianche. Modello medio dei cluster.', fontsize=9)
        path = output/f'{key}_occorrenze_velocita_direzione.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pages.append(path)
        print(f'{key}: tre pannelli salvati.', flush=True)
    pdf = output/'mappe_05_12_occorrenze_velocita_direzione.pdf'
    package_pdf(pages, pdf)
    summary = dict(
        bins=100, selected_asteroids=metadata['selected_asteroids'],
        represented_asteroids=metadata['represented_asteroids'],
        represented_clusters=metadata['represented_clusters'], distinct_members=metadata['distinct_members'],
        catalog=metadata['catalog'], report_sha256=metadata['report_sha256'],
        source_files_sha256=source_hashes,
        maps=[dict(key=r['key'], xlabel=r['xlabel'], ylabel=r['ylabel']) for r in records],
        occurrences=dict(field='occurrences', definition='Asteroidi distinti per cella, senza normalizzazione per il numero di traiettorie.',
                         normalization='linear', cmap='viridis', color_limits_by_map=occurrence_limits),
        speed=dict(field='speed_maximum', definition=metadata['speed_definition'],
                   normalization='power', gamma=3, cmap='inferno', color_limits_km_s=[speed_min, speed_max]),
        direction=dict(field='maximum_angle_deg[catalog]', definition='Direzione della stessa traiettoria che fornisce speed_maximum.',
                       frame='Geocentrico inerziale del catalogo, Luna su +x a ETD.',
                       zero='+x', positive='antiorario', normalization='linear', cmap='twilight', color_limits_deg=[0, 360],
                       tie_break='source_id minimo, poi prima ETD valida.'),
        figures=[str(path) for path in pages], pdf=str(pdf))
    (output/'riepilogo.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'PDF con otto tavole: {pdf}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed-data', type=Path, default=BASE/'mappe_velocita_soi_100x100')
    parser.add_argument('--direction-data', type=Path, default=BASE/'mappe_direzioni_soi_100x100')
    parser.add_argument('--output', type=Path, default=BASE/'pannelli_occorrenze_velocita_direzione_100x100')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render(args.speed_data, args.direction_data, args.output)


if __name__ == '__main__':
    main()
