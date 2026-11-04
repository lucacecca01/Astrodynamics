"""Render sparse SOI-speed grids without hiding small fast cells.

Native indexed PNGs preserve one cell per pixel. Overview RGB pixels are BOX
area averages on white, while a fixed three-cell window of the 100-bin grid
is shown at native resolution in every zoom panel.
"""

import gc
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, PowerNorm


METRICS = (
    ('media', 'Velocita media alla SOI', 'speed_mean'),
    ('massima', 'Velocita massima alla SOI', 'speed_maximum'),
    ('scarto', 'Scarto relativo massimo–media', 'relative_gap'),
)
OVERVIEW_PIXELS = 1000


def _labels(context, key):
    record = next(item for item in context['maps'] if item['key'] == key)
    return record['xlabel'], record['ylabel']


def _palette(name):
    """Index zero is empty/white; 1..255 sample the complete colormap."""
    result = np.empty((256, 3), dtype=np.uint8)
    result[0] = 255
    result[1:] = np.rint(plt.get_cmap(name)(np.linspace(0, 1, 255))[:, :3]*255).astype(np.uint8)
    return result


def _indices(values, norm):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite values in occupied cells.')
    mapped = np.asarray(norm(values))
    if np.any(mapped < -1e-10) or np.any(mapped > 1+1e-10):
        raise ValueError('A physical value is outside the declared color scale.')
    return (np.rint(np.clip(mapped, 0, 1)*254).astype(np.uint16)+1).astype(np.uint8)


def _overview_rgb(occupied, colors, n, palette, target=OVERVIEW_PIXELS):
    """Return a y-up RGB display, averaging ALL native RGB pixels including white."""
    ix, iy = occupied//n, occupied % n
    if n < target:
        if target % n:
            raise ValueError('The overview size must be a multiple of the coarse grid.')
        small = np.zeros((n, n), dtype=np.uint8)
        small[n-1-iy, ix] = colors
        rgb = palette[small]
        factor = target//n
        return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)
    if n % target:
        raise ValueError('The fine grid must be a multiple of the overview size.')
    factor = n//target
    # Work in x-major logical indices, matching the sparse producer.
    block = (ix//factor)*target+iy//factor
    output = np.empty((target*target, 3), dtype=np.uint8)
    rgb = palette[colors]
    area = factor*factor
    for channel in range(3):
        delta = np.bincount(block, weights=rgb[:, channel].astype(np.int16)-255,
                            minlength=target*target)
        output[:, channel] = np.rint((255*area+delta)/area).astype(np.uint8)
    return np.ascontiguousarray(output.reshape(target, target, 3).transpose(1, 0, 2)[::-1])


def _select_zoom(path):
    """Pick a supported high-speed/gap cell and three complete neighboring bins."""
    with np.load(path) as data:
        ex, ey = data['edges_x'], data['edges_y']
        occupied, count = data['occupied'], data['count']
        mean, maximum = data['speed_mean'], data['speed_maximum']
        if len(ex) != 101 or len(ey) != 101:
            raise ValueError('Zoom selection needs the 100x100 reference.')
        threshold = float(np.percentile(maximum, 90))
        candidates = np.flatnonzero((maximum >= threshold) & (count >= 20))
        selection = 'maximum >= P90 and at least 20 distinct trajectories'
        if not len(candidates):
            candidates = np.flatnonzero(maximum >= threshold)
            selection = 'maximum >= P90; no candidate has 20 trajectories'
        gap = (maximum-mean)/maximum
        chosen = int(candidates[np.lexsort((occupied[candidates], -gap[candidates]))[0]])
        cell = int(occupied[chosen])
        ix, iy = divmod(cell, 100)
        x0, y0 = min(max(ix-1, 0), 97), min(max(iy-1, 0), 97)
        limits = [float(ex[x0]), float(ex[x0+3]), float(ey[y0]), float(ey[y0+3])]
        return dict(limits=limits, base_indices=[x0, x0+3, y0, y0+3], cell_flat=cell,
                    cell_indices=[ix, iy], trajectory_count=int(count[chosen]),
                    speed_mean=float(mean[chosen]), speed_maximum=float(maximum[chosen]),
                    relative_gap_percent=float(100*gap[chosen]), high_speed_p90=threshold,
                    selection=selection, domain_fraction_per_axis=.03,
                    angular_wrap=False)


def _zoom_rgb(occupied, colors, n, palette, ex, ey, zoom):
    if n % 100:
        raise ValueError('The grid must be a multiple of 100 for aligned native zooms.')
    factor = n//100
    x0, x1, y0, y1 = [index*factor for index in zoom['base_indices']]
    ix, iy = occupied//n, occupied % n
    take = (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
    native = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    native[y1-1-iy[take], ix[take]-x0] = colors[take]
    extent = [float(ex[x0]), float(ex[x1]), float(ey[y0]), float(ey[y1])]
    # Floating point construction can shift matching edges by an ulp only.
    np.testing.assert_allclose(extent, zoom['limits'], rtol=1e-12, atol=1e-9)
    return palette[native], extent


def _save_native(path, occupied, colors, n, palette):
    """Allocate just one full-sized uint8 grid; never a full RGB native grid."""
    native = np.zeros((n, n), dtype=np.uint8)
    native[n-1-occupied % n, occupied//n] = colors
    image = Image.fromarray(native, mode='P')
    image.putpalette(palette.ravel().tolist())
    image.save(path, compress_level=4, optimize=False)
    image.close()


def _save_pdf(images, output):
    """Lossless, one contact-sheet PNG per page, loaded serially."""
    try:
        import fitz
    except ImportError:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(output) as pdf:
            for path in images:
                with Image.open(path) as source:
                    rgb = np.array(source.convert('RGB'))
                height, width = rgb.shape[:2]
                fig = plt.figure(figsize=(width/150, height/150))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(rgb, interpolation='none', aspect='auto')
                ax.set_axis_off()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
                del rgb
        return 'matplotlib PdfPages; one lossless RGB page at a time'
    document = fitz.open()
    try:
        for path in images:
            with Image.open(path) as source:
                width, height = source.size
            page = document.new_page(width=width*72/150, height=height*72/150)
            page.insert_image(page.rect, filename=str(path))
        document.save(output, deflate=True)
    finally:
        document.close()
    return 'PyMuPDF insert_image; lossless PNG streams'


def _contact_sheet(key, rows, previews, extents, labels, context, speed_norm, gap_norm,
                   zoom, output, is_zoom):
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 3, figsize=(19.2, 4.4*nrows), layout='constrained', squeeze=False)
    domain = extents[0][0]
    for row, n in enumerate(rows):
        for column, (_, title, _) in enumerate(METRICS):
            ax = axes[row, column]
            ax.imshow(previews[row][column], origin='upper', extent=extents[row][column],
                      interpolation='nearest', aspect='auto')
            limits = zoom['limits'] if is_zoom else domain
            ax.set_xlim(limits[:2])
            ax.set_ylim(limits[2:])
            ax.set(xlabel=labels[0], ylabel=labels[1], title=f'{n:,} × {n:,} · {title}')
            ax.title.set_fontsize(11)
            ax.xaxis.label.set_fontsize(9)
            ax.yaxis.label.set_fontsize(9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.12)
    for column in range(3):
        speed = column < 2
        norm, cmap = (speed_norm, 'inferno') if speed else (gap_norm, 'cividis')
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                ax=axes[:, column].tolist(), shrink=.85, aspect=55, pad=.015)
        colorbar.set_label('Velocita geocentrica alla SOI [km/s]' if speed else
                           '100 × (massima − media) / massima [%]', fontsize=10)
        if speed:
            ticks = norm.inverse(np.linspace(0, 1, 7))
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels([f'{value:.3f}' for value in ticks])
        else:
            colorbar.set_ticks(np.arange(0, 61, 10))
        colorbar.ax.tick_params(labelsize=9)
    title = 'zoom fisso, celle native' if is_zoom else 'dominio completo, confronto delle griglie'
    fig.suptitle(
        f'Mappa {key[:2]}: {title}\n'
        f"{context['selected_asteroids']} asteroidi · {context['distinct_members']:,} traiettorie distinte · "
        'scale comuni tra tutte le risoluzioni', fontsize=17)
    note = (
        'Stessa finestra fisica in ogni riga: tre celle consecutive della griglia 100 × 100, pari al 3% del dominio per asse.\n'
        'I colori dello zoom sono quelli nativi; i bordi delle celle vengono rispettati, senza interpolazione o avvolgimento angolare.'
        if is_zoom else
        'Panoramica: riduzione BOX dei colori RGB su sfondo bianco; ogni pixel comprende tutta la sua area, non il massimo locale.\n'
        'La legenda descrive i colori nativi; i PNG in nativi conservano una cella per pixel. Il bianco indica celle vuote.'
    )
    fig.supxlabel(note+'\nVelocita: PowerNorm γ = 3, inferno. Scarto relativo: scala lineare 0–60%, cividis.', fontsize=11)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_map(key, cases, context, output):
    """Render seven sparse resolutions of a single map, returning provenance paths."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = _labels(context, key)
    lower, upper = map(float, context['speed_limits_km_s'])
    if not 0 < lower < upper:
        raise ValueError('Invalid common speed scale.')
    speed_norm, gap_norm = PowerNorm(3, vmin=lower, vmax=upper), Normalize(0, 60)
    speed_palette, gap_palette = _palette('inferno'), _palette('cividis')
    indexed_cases = []
    for path in map(Path, cases):
        with np.load(path) as data:
            n = len(data['edges_x'])-1
            if len(data['edges_y']) != n+1:
                raise ValueError('The grid must be square.')
        indexed_cases.append((n, path))
    indexed_cases.sort()
    if [n for n, _ in indexed_cases] != [100, 200, 500, 1000, 2000, 5000, 10000]:
        raise ValueError('Expected exactly the seven requested grid resolutions.')
    zoom = _select_zoom(indexed_cases[0][1])
    natives = output/'nativi'/key
    natives.mkdir(parents=True, exist_ok=True)
    rows, full_previews, zoom_previews, full_extents, zoom_extents = [], [], [], [], []
    native_records = []
    domain = None
    for n, path in indexed_cases:
        with np.load(path) as data:
            occupied = data['occupied'].astype(np.int64)
            count = data['count']
            mean, maximum = data['speed_mean'], data['speed_maximum']
            ex, ey = data['edges_x'], data['edges_y']
        if (occupied.ndim != 1 or not len(occupied) or count.shape != occupied.shape
                or mean.shape != occupied.shape or maximum.shape != occupied.shape
                or np.any(occupied < 0) or np.any(occupied >= n*n) or np.any(count <= 0)
                or np.any(np.diff(occupied) <= 0)):
            raise ValueError(f'{path}: sparse occupied cells must be sorted, unique and valid.')
        extent = [float(ex[0]), float(ex[-1]), float(ey[0]), float(ey[-1])]
        if domain is None:
            domain = extent
        else:
            np.testing.assert_array_equal(extent, domain)
        np.testing.assert_allclose(ex, np.linspace(ex[0], ex[-1], n+1), rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(ey, np.linspace(ey[0], ey[-1], n+1), rtol=1e-12, atol=1e-9)
        gap = 100*(maximum-mean)/maximum
        if np.any(gap < -1e-9) or np.any(gap > 60+1e-9):
            raise ValueError('Relative gaps fall outside the declared 0–60% physical range.')
        values = (mean, maximum, np.maximum(gap, 0))
        overviews, zooms, zoom_bounds = [], [], []
        for column, ((name, _, field), value) in enumerate(zip(METRICS, values)):
            norm, palette = (speed_norm, speed_palette) if column < 2 else (gap_norm, gap_palette)
            colors = _indices(value, norm)
            overviews.append(_overview_rgb(occupied, colors, n, palette))
            crop, bounds = _zoom_rgb(occupied, colors, n, palette, ex, ey, zoom)
            zooms.append(crop)
            zoom_bounds.append(bounds)
            native = natives/f'{n}_{name}.png'
            _save_native(native, occupied, colors, n, palette)
            native_records.append(dict(bins=n, metric=field, file=str(native), pixels=[n, n],
                                       mode='P', empty_palette_index=0))
            del colors, crop
        rows.append(n)
        full_previews.append(overviews)
        zoom_previews.append(zooms)
        full_extents.append([extent]*3)
        zoom_extents.append(zoom_bounds)
        print(f'{key}: {n}x{n}, tre PNG nativi e anteprime pronti.', flush=True)
        del occupied, count, mean, maximum, gap, values
        gc.collect()
    overview_path, zoom_path = output/f'overview_{key}.png', output/f'zoom_{key}.png'
    _contact_sheet(key, rows, full_previews, full_extents, labels, context, speed_norm, gap_norm,
                   zoom, overview_path, False)
    _contact_sheet(key, rows, zoom_previews, zoom_extents, labels, context, speed_norm, gap_norm,
                   zoom, zoom_path, True)
    pdf = output/f'confronto_{key}.pdf'
    pdf_engine = _save_pdf([overview_path, zoom_path], pdf)
    return dict(key=key, bins=rows, figures=[str(overview_path), str(zoom_path)], pdf=str(pdf),
                native_count=len(native_records), native_images=native_records,
                zoom_limits=zoom['limits'], zoom_selection=zoom,
                scales=dict(velocity=dict(limits_km_s=[lower, upper], normalization='power', gamma=3, cmap='inferno'),
                            gap=dict(limits_percent=[0, 60], normalization='linear', cmap='cividis')),
                overview=dict(pixels=[OVERVIEW_PIXELS, OVERVIEW_PIXELS],
                              reduction='exact BOX area-average of quantized native RGB colors on white; nearest rounding'),
                native_orientation='x increases right; y increases up; native PNG row 0 is the highest y cell',
                pdf_engine=pdf_engine)
