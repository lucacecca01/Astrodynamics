"""Mappe 05–12 della direzione della velocita alla SOI in tre riferimenti."""

import argparse
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

import GA_18_plots as plots
from GA_18_percentile_trajectories import package_pdf


FRAME_TITLES = (
    'Sinodico Terra–Luna alla SOI',
    'Geocentrico, assi ECLIPJ2000',
    'Inerziale del catalogo — Luna su +x a t = 0',
)


def render(output, metadata, aggregation):
    mean = aggregation == 'circular'
    field = 'mean_angle_deg' if mean else 'maximum_angle_deg'
    suffix = 'media_circolare' if mean else 'traiettoria_piu_veloce'
    description = 'media circolare' if mean else 'traiettoria a velocita SOI massima'
    norm = Normalize(0, 360)
    cmap = plt.get_cmap('twilight')
    pages = []
    for record in metadata['maps']:
        key = record['key']
        with np.load(output/f'{key}_direzioni_soi.npz') as z:
            ex, ey, count = z['edges_x'], z['edges_y'], z['count']
            angles, concentration = z[field], z['resultant_length']
        if angles.shape != (3, 100, 100) or concentration.shape != angles.shape:
            raise ValueError(f'{key}: dimensioni delle direzioni incoerenti.')
        occupied = count > 0
        fig, axes = plt.subplots(1, 3, figsize=(19.2, 7.2), layout='constrained')
        for frame, (ax, title) in enumerate(zip(axes, FRAME_TITLES)):
            ax.set_facecolor('white')
            ax.pcolormesh(ex, ey, np.ma.masked_where(~occupied.T, np.ones(count.T.shape)),
                          cmap=ListedColormap(['#9e9e9e']), vmin=0, vmax=1, shading='flat')
            alpha = np.clip(np.nan_to_num(concentration[frame].T), 0, 1) if mean else 1
            ax.pcolormesh(ex, ey, np.ma.masked_invalid(angles[frame].T),
                          cmap=cmap, norm=norm, alpha=alpha, shading='flat')
            ax.set(xlabel=record['xlabel'], ylabel=record['ylabel'], title=title)
            ax.grid(alpha=.12)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(),
                     ticks=np.arange(0, 361, 60), shrink=.82,
                     label='Direzione della velocita alla SOI [deg]')
        if mean:
            axes[0].legend(handles=[Patch(facecolor='#9e9e9e', label='Direzioni poco concordi (R vicino a 0)')],
                           loc='lower left', fontsize='small')
        fig.suptitle(
            f'Mappa {key[:2]}: direzione di uscita alla SOI — {description}\n'
            f"Griglia 100 × 100 · {metadata['selected_asteroids']} asteroidi · "
            f"{metadata['distinct_members']:,} traiettorie distinte", fontsize=15)
        counting = (
            'Ogni traiettoria pesa una volta per cella; si mediano prima le sue ETD. '
            'Opacita del colore = concentrazione circolare R (0: distribuite, 1: allineate).'
            if mean else
            'Per cella si sceglie il massimo modulo della velocita geocentrica alla SOI; '
            'a parita: source_id minimo, poi ETD valida piu precoce.'
        )
        fig.supxlabel(
            'Angolo antiorario dalla +x di ciascun riferimento. Scala ciclica: 0° e 360° coincidono. '
            'Celle vuote bianche.\n'
            f'{counting}\n'
            'Nel sinodico e incluso il termine di rotazione Ω×r; J2000 usa la fase lunare all’ETD. '
            'Membri dei cluster vincenti del modello medio.', fontsize=9)
        path = output/f'{key}_direzioni_soi_{suffix}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pages.append(path)
    pdf = output/f'mappe_05_12_direzioni_soi_{suffix}.pdf'
    package_pdf(pages, pdf)
    metadata.update(aggregation=aggregation, frame_titles=list(FRAME_TITLES), cmap='twilight',
                    color_limits_deg=[0, 360], opacity='resultant_length' if mean else 'opaque',
                    figures=[str(path) for path in pages], pdf=str(pdf))
    text = json.dumps(metadata, indent=2)+'\n'
    (output/'riepilogo.json').write_text(text)
    (output/f'riepilogo_{suffix}.json').write_text(text)
    print(f'Completate otto tavole con tre riferimenti: {pdf}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=plots.OUTPUT_DIR/'mappe_direzioni_soi_100x100')
    parser.add_argument('--aggregation', choices=('circular', 'max-speed'), default='circular')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only and args.render_only:
        parser.error('prepare-only e render-only sono alternativi.')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        metadata = json.loads((args.output/'dati.json').read_text())
    else:
        from GA_18_soi_direction_data import prepare
        metadata = prepare(args.output)
    if not args.prepare_only:
        render(args.output, metadata, args.aggregation)


if __name__ == '__main__':
    main()
