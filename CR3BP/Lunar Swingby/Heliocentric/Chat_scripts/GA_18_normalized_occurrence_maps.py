"""Asteroidi distinti per cella / tutte le traiettorie valide dei cluster.

Le mappe J2000 usano tutte le date SOI della scansione GA18, ricavando l'ETD
di ogni cluster dal suo tempo medio alla SOI. Ogni traiettoria conta una sola
volta nella cella, anche quando e presente a piu epoche.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(SCRIPT.parent), str(SCRIPT.parents[1]), str(SCRIPT.parents[4])]

import numpy as np
import pykep as pk
import spiceypy as spice
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import GA_18_plots as plots
from GA_18_catalog_azimuth_trajectories import read_eligible_members
from GA_18_percentile_trajectories import package_pdf
from GA_18_soi_speed_maps import CATALOG_TABLES, PHI_TABLES, MAPS, read_reference


OUTPUT = plots.OUTPUT_DIR/'mappe_occorrenze_normalizzate_totali_100x100'


def scan_configuration(path):
    """Read numeric scan assignments without importing or executing GA18."""
    names = ('JD_START', 'JD_END', 'STEP_SOI_DAYS')
    expressions = {name: [] for name in names}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in expressions:
                    expressions[target.id].append(node.value)
    if any(len(values) != 1 for values in expressions.values()):
        raise ValueError('Configurazione temporale GA18 assente o ambigua.')

    def number(node, values):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp):
            a, b = number(node.left, values), number(node.right, values)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div): return a/b
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -number(node.operand, values)
        raise ValueError('La configurazione temporale non e una semplice espressione numerica.')

    values = {}
    for name in names:
        values[name] = number(expressions[name][0], values)
    duration = values['JD_END']-values['JD_START']
    if duration <= 0 or values['STEP_SOI_DAYS'] <= 0:
        raise ValueError('Intervallo o passo temporale non valido.')
    grid = values['JD_START']+np.append(np.arange(0, duration, values['STEP_SOI_DAYS']), duration)
    return grid, values


def lunar_phase_grid(stats, soi_grid):
    """All candidate ETDs, with exactly the mean time offset used in GA18."""
    spice.furnsh(pk.udpla.de440s.kernel_file())
    phases = np.empty((len(stats), len(soi_grid)))
    for row in range(len(stats)):
        jd_etd = soi_grid-stats[row, 7]
        positions = np.asarray(spice.spkpos(
            'MOON', (jd_etd-2451545.)*86400., 'ECLIPJ2000', 'NONE', 'EARTH')[0])
        phases[row] = np.mod(np.degrees(np.arctan2(positions[:, 1], positions[:, 0])), 360)
        if (row+1) % 100 == 0:
            print(f'Fasi lunari della scansione: {row+1}/{len(stats)} cluster.', flush=True)
    ordered = np.sort(phases, axis=1)
    gaps = np.diff(np.concatenate((ordered, ordered[:, :1]+360), axis=1), axis=1)
    return phases, gaps.max(axis=1)


def phi_denominator(x, omega, rows, ex, ey, phases, maximum_gaps):
    """Unique source/cell occupancy over the full scan, without epoch weighting."""
    nx, ny = len(ex)-1, len(ey)-1
    in_domain = (x >= ex[0]) & (x <= ex[-1])
    ix = np.minimum(np.searchsorted(ex, x, side='right')-1, nx-1)
    # If the largest circular gap is smaller than a cell width, every angular
    # cell contains a sampled phase for ANY rotation by the source's omega.
    complete = maximum_gaps < np.diff(ey).min()-1e-9
    dense = in_domain & complete[rows]
    result = np.repeat(np.bincount(ix[dense], minlength=nx)[:, None], ny, axis=1)
    # Exact interval queries for a shorter/sparser future scan. Each source is
    # tested once per angular cell; repeated epochs never increase its weight.
    for row in np.unique(rows[in_domain & ~complete[rows]]):
        ids = np.flatnonzero(in_domain & (rows == row))
        ordered = np.sort(phases[row])
        extended = np.concatenate((ordered, ordered+360))
        for angular in range(ny):
            start = np.mod(ey[angular]-omega[ids], 360)
            stop = start+ey[angular+1]-ey[angular]
            left = np.searchsorted(extended, start, side='left')
            right = np.searchsorted(extended, stop, side='left')
            hit = right > left
            # Near a bin edge, subtracting omega can round differently from
            # the original (omega + phase) expression. Check those rare cases
            # directly with the exact histogram convention used by GA18.
            ambiguous = np.zeros(len(ids), dtype=bool)
            for boundary, index in ((start, left), (stop, right)):
                after = extended[np.minimum(index, len(extended)-1)]
                before = extended[np.maximum(index-1, 0)]
                ambiguous |= np.minimum(abs(after-boundary), abs(before-boundary)) < 1e-9
            near = np.flatnonzero(ambiguous)
            for begin in range(0, len(near), 4096):
                part = near[begin:begin+4096]
                actual = np.mod(omega[ids[part], None]+phases[row][None, :], 360)
                bins = np.minimum(np.searchsorted(ey, actual, side='right')-1, ny-1)
                hit[part] = (bins == angular).any(axis=1)
            result[:, angular] += np.bincount(ix[ids[hit]], minlength=nx)
    return result, complete, int(in_domain.sum())


def save_grid(output, spec, reference, denominator, valid_in_domain):
    key, _, xlabel, ylabel = spec
    numerator = reference['occurrences']
    if np.any((numerator > 0) & (denominator == 0)):
        raise ValueError(f'{key}: asteroidi presenti in una cella senza traiettorie valide.')
    ratio = np.full(denominator.shape, np.nan)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    with np.load(plots.OUTPUT_DIR/'mappe_velocita_soi_100x100'/f'{key}_velocita_soi.npz') as z:
        if np.any(z['trajectory_count'] > denominator):
            raise ValueError(f'{key}: il denominatore totale non comprende quello dei vincitori.')
    np.savez_compressed(output/f'{key}_occorrenze_normalizzate.npz', **reference,
                        asteroid_count=numerator, trajectory_count=denominator,
                        asteroids_per_trajectory=ratio, x_label=xlabel, y_label=ylabel)
    print(f'{key}: {np.count_nonzero(denominator):,} celle con traiettorie, '
          f'{np.count_nonzero((denominator > 0) & (numerator == 0)):,} con zero asteroidi.', flush=True)
    return dict(key=key, xlabel=xlabel, ylabel=ylabel,
                trajectories_in_display_domain=valid_in_domain,
                denominator_sum=int(denominator.sum()), occupied_cells=int(np.count_nonzero(denominator)),
                zero_asteroid_cells=int(np.count_nonzero((denominator > 0) & (numerator == 0))))


def prepare(output):
    source = plots.BASE_DIR/'CR3BP_prova_GA_18.py'
    soi_grid, scan = scan_configuration(source)
    summary = json.loads((CATALOG_TABLES/'riepilogo.json').read_text())
    catalog = Path(summary['catalog'])
    stats = plots.load_table(catalog, 15, allow_extra=True)
    stats = stats[np.argsort(stats[:, 0])]
    winners = summary['winners']
    winner_rows = np.array([w['cluster_row'] for w in winners])
    np.testing.assert_array_equal(stats[winner_rows, 0], [w['cluster_id'] for w in winners])
    # Verify that the actual saved winning departure dates belong to this scan.
    winning_soi = np.array([w['jd_etd'] for w in winners])+stats[winner_rows, 7]
    distance = np.abs(winning_soi[:, None]-soi_grid).min(axis=1)
    if distance.max() > 1e-8:
        raise ValueError('Le date SOI dei vincitori non appartengono alla scansione GA18 corrente.')
    if max(w['jd_arrival'] for w in winners) > scan['JD_END']+1e-8:
        raise ValueError('La scansione corrente non copre gli arrivi salvati.')
    # Include every assigned cluster, even if no asteroid ever chose it.
    members = read_eligible_members(catalog.with_name('trajectory_exit_catalog.txt'), stats[:, 0])
    rows = np.searchsorted(stats[:, 0], members[:, 1])
    np.testing.assert_array_equal(stats[rows, 0], members[:, 1])
    frequency = np.bincount(winner_rows, minlength=len(stats))
    winning_mask = frequency[rows] > 0
    if int(winning_mask.sum()) != summary['population_count']:
        raise ValueError('Il sottoinsieme dei vincitori non corrisponde alle mappe originali.')
    eps, ecc = members[:, 9:11].T
    hp = -plots.MU_EARTH*(1-ecc)/(2*eps)-plots.R_EARTH_KM
    omega = plots.catalog_perigee_azimuth(members, stats)
    theta = (omega-plots.MOON_RATE_DEG_DAY*members[:, 5]) % 360
    coordinates = ((eps, ecc), (eps, omega), (hp, ecc), (hp, omega), (hp, theta), (eps, theta))
    records = []
    for spec, (x, y) in zip(MAPS[:6], coordinates):
        reference = read_reference(CATALOG_TABLES, spec[0])
        denominator = np.histogram2d(x, y, bins=(reference['edges_x'], reference['edges_y']))[0].astype(np.int64)
        records.append(save_grid(output, spec, reference, denominator, int(denominator.sum())))
    phases, maximum_gaps = lunar_phase_grid(stats, soi_grid)
    for spec, x in zip(MAPS[6:], (eps, hp)):
        reference = read_reference(PHI_TABLES, spec[0])
        denominator, complete, n_domain = phi_denominator(
            x, omega, rows, reference['edges_x'], reference['edges_y'], phases, maximum_gaps)
        records.append(save_grid(output, spec, reference, denominator, n_domain))
    np.savez_compressed(output/'fasi_scansione.npz', cluster_ids=stats[:, 0].astype(np.int64),
                        jd_soi_grid=soi_grid, mean_t_soi_days=stats[:, 7],
                        phase_etd_deg=phases, maximum_circular_gap_deg=maximum_gaps,
                        full_angular_coverage=complete)
    metadata = dict(catalog=str(catalog), maps=records, bins=100,
                    valid_trajectories=len(members), assigned_clusters=len(stats),
                    winning_trajectories=int(winning_mask.sum()),
                    trajectories_in_never_chosen_clusters=int((~winning_mask).sum()),
                    selected_asteroids=len(winners),
                    formula='Asteroidi distinti nella cella / tutte le traiettorie valide dei cluster nella cella.',
                    population='Tutti i cluster assegnati del catalogo; eps<0, 0<=e<1, periapside sopra la superficie. Label -1 esclusa.',
                    display_domain='Bordi originali delle mappe 100x100; i membri esterni al dominio non vengono accumulati ai bordi.',
                    scan_configuration=scan, scan_soi_dates=len(soi_grid),
                    etd_rule='JD_ETD(cluster) = JD_SOI - mean_t_soi(cluster), come nella sincronizzazione GA18 alla SOI.',
                    phase_sampling='Tutte le date SOI della scansione, per ogni cluster; una sola occorrenza di ciascun source_id per cella.',
                    complete_phase_clusters=int(complete.sum()),
                    maximum_circular_phase_gap_deg=float(maximum_gaps.max()),
                    phase_coverage_proof='Gap circolare massimo < ampiezza minima della cella implica copertura di tutti i bin per ogni omega del catalogo.',
                    report_sha256=summary['report_sha256'],
                    core_script_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (output/'dati.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def render(output, metadata, color_scope='shared', cmap='viridis'):
    ranges = {}
    for record in metadata['maps']:
        with np.load(output/f'{record["key"]}_occorrenze_normalizzate.npz') as z:
            a = z['asteroids_per_trajectory']
            positive = a[np.isfinite(a) & (a > 0)]
            lower, upper = float(positive.min()), float(positive.max())
            if lower == upper:
                lower, upper = lower/2, upper*2
            ranges[record['key']] = [lower, upper]
    global_range = [min(r[0] for r in ranges.values()), max(r[1] for r in ranges.values())]
    suffix = '_scala_locale' if color_scope == 'per-map' else ('' if cmap == 'viridis' else f'_{cmap}')
    scale_note = ('Scala logaritmica ricalibrata per questa mappa, senza taglio degli estremi.'
                  if color_scope == 'per-map' else 'Scala logaritmica comune per i valori positivi.')
    effective_ranges = {}
    pages = []
    for record in metadata['maps']:
        key = record['key']
        lower, upper = ranges[key] if color_scope == 'per-map' else global_range
        effective_ranges[key] = [lower, upper]
        norm = LogNorm(lower, upper)
        with np.load(output/f'{key}_occorrenze_normalizzate.npz') as z:
            ex, ey, a, count = z['edges_x'], z['edges_y'], z['asteroids_per_trajectory'], z['trajectory_count']
        fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
        ax.pcolormesh(ex, ey, np.ma.masked_where(count.T == 0, np.ones(count.T.shape)),
                      cmap=ListedColormap(['#cecece']), vmin=0, vmax=1, shading='flat')
        mesh = ax.pcolormesh(ex, ey, np.ma.masked_where(~np.isfinite(a.T) | (a.T <= 0), a.T),
                             cmap=cmap, norm=norm, shading='flat')
        ax.set(xlabel=record['xlabel'], ylabel=record['ylabel'],
               title=(f'Mappa {key[:2]}: asteroidi distinti / traiettorie valide nella cella\n'
                      f"{metadata['valid_trajectories']:,} traiettorie valide — tutti i {metadata['assigned_clusters']} cluster"))
        ax.grid(alpha=.15)
        colorbar = fig.colorbar(mesh, ax=ax, label='Asteroidi distinti / traiettorie valide')
        colorbar.set_ticks(np.geomspace(lower, upper, 7))
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2g}'))
        ax.legend(handles=[Patch(facecolor='#cecece', label='Traiettorie presenti, zero asteroidi'),
                           Patch(facecolor='white', edgecolor='0.6', label='Nessuna traiettoria nella cella')],
                  loc='upper left', fontsize='small')
        fig.supxlabel(
            'Numeratore: asteroidi distinti dei migliori trasferimenti. Denominatore: tutte le traiettorie valide assegnate, incluse quelle dei cluster mai scelti.\n'
            f"Mappe 11/12: tutte le {metadata['scan_soi_dates']} date SOI della scansione GA18; "
            'ETD ricavata per ciascun cluster. Ogni traiettoria conta una volta per cella.\n'
            f'{scale_note} Zero in grigio. Stessi bin e domini delle mappe originali. Modello medio dei cluster.', fontsize=9)
        path = output/f'{key}_asteroidi_diviso_tutte_traiettorie{suffix}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pages.append(path)
    pdf = output/f'mappe_05_12_asteroidi_diviso_tutte_traiettorie{suffix}.pdf'
    package_pdf(pages, pdf)
    metadata.update(metric='distinct_asteroids_divided_by_all_valid_trajectories', color_scale='log_positive_zero_gray',
                    color_scope=color_scope, cmap=cmap, color_limits=global_range if color_scope == 'shared' else None,
                    color_limits_global=global_range, color_limits_by_map=effective_ranges,
                    figures=[str(p) for p in pages], pdf=str(pdf))
    text = json.dumps(metadata, indent=2)+'\n'
    (output/'riepilogo.json').write_text(text)
    (output/f'riepilogo{suffix}.json').write_text(text)
    print(f'Completate otto mappe normalizzate: {pdf}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--color-scope', choices=('shared', 'per-map'), default='shared')
    parser.add_argument('--cmap', default='viridis')
    args = parser.parse_args()
    if args.prepare_only and args.render_only:
        parser.error('prepare-only e render-only sono alternativi.')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.output/'dati.json').read_text()) if args.render_only else prepare(args.output)
    if not args.prepare_only:
        render(args.output, metadata, color_scope=args.color_scope, cmap=args.cmap)


if __name__ == '__main__':
    main()
