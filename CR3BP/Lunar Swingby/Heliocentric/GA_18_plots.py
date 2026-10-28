"""Mappe delle orbite terrestri, senza ricalcolare Lambert.

Gli asteroidi reali sono identificati da nome e SPK-ID nell'header dei risultati.
Le orbite degli asteroidi sono approssimate nel piano, come nel calcolo GA18.
I risultati del modello medio di un cluster sono attribuiti ai suoi membri.
"""

from pathlib import Path
from collections.abc import Sequence
import gc
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator


# Stessa cartella di salvataggio usata da GA18; selezione dai metadati dei file.
BASE_DIR = Path(__file__).resolve().parent
TRANSFER_DIR = BASE_DIR / "GA18_real_asteroids"  # Cartella con i risultati di GA18, per tutti gli asteroidi reali.
TRANSFER_FILES = None          # None = tutti gli asteroidi reali; oppure lista di file.
OUTPUT_DIR = BASE_DIR / "GA18_parking_orbits"
MAX_DV_KM_S = None             # Solo rendez-vous all'arrivo; None = nessun limite.
MAX_DURATION_DAYS = None       # Perigeo MEDIO -> arrivo; None = nessun limite.
ONLY_BOUND_PARKING = True      # eps < 0, 0 <= e < 1, perigeo sopra la superficie.
N_BINS = 100                   # Celle per asse nelle mappe eps-e ed eps-omega.
PANELS_PER_PAGE = 144          # Pagine 16:9: griglia completa di 16 x 9 asteroidi.
PLOT_COLUMNS = 16              # Massimo; la griglia si adatta nelle pagine parziali.
PORKCHOP_CELL_DAYS = 1.0        # Larghezza massima delle celle grafiche, non interpolazione.
BEST_OCCURRENCE_MODE = "asteroids"  # "asteroids": uno per cella; "trajectories": tutti i membri.
MU_EARTH = 6.67430e-20 * 5.974e24
R_EARTH_KM = 6378.0
EARTH_MOON_DISTANCE_KM = 384400.0
MOON_MASS_KG = 7.348e22
TIME_UNIT_S = np.sqrt(EARTH_MOON_DISTANCE_KM**3 / (MU_EARTH + 6.67430e-20 * MOON_MASS_KG))
MOON_RATE_DEG_DAY = np.rad2deg(86400.0 / TIME_UNIT_S)
EPS_LABEL = r"$\varepsilon$ al perigeo [km$^2$/s$^2$]"
HP_LABEL = r"$h_p = r_p - R_\oplus$ sopra la superficie [km]"
THETA_LABEL = r"$\theta$: Luna → perigeo al perigeo [deg]"
OCCURRENCE_PERCENTILES = (97, 99, 99.5, 99.75)


class ScenarioMaps(Sequence):
    """Calcola le mappe richieste senza conservare quelle di tutti gli asteroidi."""

    def __init__(self, geometry, minimum, counts):
        self.geometry = geometry
        self.minimum = minimum
        self.counts = counts

    def __len__(self):
        return len(self.minimum)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]
        index = range(len(self))[index]
        return maps_from_geometry(self.geometry, self.minimum[index:index+1], self.counts[index:index+1])


def load_table(path, n_columns):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
        values = np.loadtxt(path, ndmin=2)
    if values.size == 0:
        return np.empty((0, n_columns))
    if values.shape[1] != n_columns or not np.isfinite(values).all():
        raise ValueError(f"{path}: richieste {n_columns} colonne finite.")
    return values


def integer_ids(values, name):
    if np.any(values != np.floor(values)):
        raise ValueError(f"{name}: identificativi non interi.")
    return values.astype(int)


def read_transfer_header(path):
    """Legge identita dell'asteroide, catalogo e significato della prima epoca."""
    path = Path(path)
    metadata = dict(name=path.parent.name, spkid=None, epoch_reference=None, catalog=None)
    with path.open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            first = text.split(maxsplit=1)[0] if text else ""
            if first in ("jd_soi_tdb", "jd_etd_tdb"):
                metadata["epoch_reference"] = "SOI" if first == "jd_soi_tdb" else "ETD"
            elif text.startswith("Catalog:"):
                catalog = Path(text.partition(":")[2].strip())
                metadata["catalog"] = (catalog if catalog.is_absolute() else path.parent / catalog).resolve()
            elif text.startswith("Asteroid:"):
                name, separator, spkid = text[len("Asteroid:"):].partition("; SPK-ID:")
                if not separator or not name.strip() or not spkid.strip().isdigit():
                    raise ValueError(f"{path}: header Asteroid/SPK-ID non valido.")
                metadata.update(name=name.strip(), spkid=spkid.strip())
    return metadata


def discover_transfer_files(directory):
    """Esclude i vecchi asteroidi casuali, anche se salvati nella stessa cartella."""
    found = []
    for path in Path(directory).glob("*/transfers_rendezvous.txt"):
        if path.is_file():
            metadata = read_transfer_header(path)
            if metadata["spkid"] is not None:
                found.append((int(metadata["spkid"]), path))
    if not found:
        raise ValueError(f"Nessun transfers_rendezvous.txt di asteroidi reali in {directory}.")
    return [path for _, path in sorted(found)]


def load_inputs(transfer_files):
    paths = [Path(p) if Path(p).is_absolute() else BASE_DIR / p for p in transfer_files]
    paths = [p.resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Selezionare almeno un file, senza duplicare lo stesso file.")
    catalog_paths, tables, metadata = [], [], []
    seen_spkids = set()
    for path in paths:
        info = read_transfer_header(path)
        catalog = info["catalog"]
        if catalog is None:
            raise ValueError(f"{path}: manca l'header '# Catalog:'.")
        if info["epoch_reference"] is None:
            raise ValueError(f"{path}: la prima colonna deve essere jd_soi_tdb oppure jd_etd_tdb.")
        if info["spkid"] is not None:
            if info["spkid"] in seen_spkids:
                raise ValueError(f"SPK-ID {info['spkid']} duplicato: selezionare una sola esecuzione per asteroide.")
            seen_spkids.add(info["spkid"])
        metadata.append(info)
        catalog_paths.append(catalog)
        tables.append(load_table(path, 5))
    if len(set(catalog_paths)) != 1:
        raise ValueError("Cataloghi diversi: non si possono unire i soli cluster_id.")
    catalog_path = catalog_paths[0]
    stats = load_table(catalog_path, 14)
    members = load_table(catalog_path.with_name("trajectory_exit_catalog.txt"), 12)
    if not len(stats) or not len(members):
        raise ValueError("Il catalogo o le statistiche sono vuoti.")
    stats = stats[np.argsort(stats[:, 0])]
    ids = integer_ids(stats[:, 0], "cluster_id delle statistiche")
    labels = integer_ids(members[:, 1], "cluster_id dei membri")
    source_ids = integer_ids(members[:, 0], "source_id")
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError("cluster_id duplicati o negativi nelle statistiche.")
    if len(np.unique(source_ids)) != len(source_ids) or np.any(source_ids < 0):
        raise ValueError("source_id duplicati o negativi nel catalogo.")
    if np.any(labels < -1) or not np.isin(labels[labels >= 0], ids).all():
        raise ValueError("Assegnazioni dei membri incoerenti con le statistiche.")
    for row in stats:
        group = members[labels == int(row[0])]
        if len(group) != row[1] or not len(group):
            raise ValueError(f"Cluster {int(row[0])}: conteggio dei membri incoerente.")
        if not np.allclose(group[:, 5:7].mean(axis=0), row[6:8], rtol=0, atol=1e-8):
            raise ValueError(f"Cluster {int(row[0])}: tempi medi del catalogo incoerenti.")
    for path, table, info in zip(paths, tables, metadata):
        labels_t = integer_ids(table[:, 1], f"cluster_id in {path}")
        if not np.isin(labels_t, ids).all():
            raise ValueError(f"{path}: cluster non presente nel catalogo.")
        rows = np.searchsorted(ids, labels_t)
        if np.any(table[:, 3] <= 0) or np.any(table[:, 4] < 0):
            raise ValueError(f"{path}: TOF non positivo o delta-v negativo.")
        offset = 0.0 if info["epoch_reference"] == "SOI" else stats[rows, 7]
        if not np.allclose(table[:, 2] - table[:, 0] - table[:, 3], offset, rtol=0, atol=1e-7):
            raise ValueError(f"{path}: epoche/TOF incoerenti con il riferimento {info['epoch_reference']}.")
    return paths, catalog_path, stats, members, tables, metadata


def transfer_mask(table, stats, max_dv=None, max_duration_days=None):
    """Stessi filtri sui trasferimenti per mappe delle orbite e porkchop."""
    rows = np.searchsorted(stats[:, 0], table[:, 1])
    # TOF dalla SOI + tempo perigeo->SOI; valido per file sia SOI sia ETD.
    duration = table[:, 3] + stats[rows, 7] - stats[rows, 6]
    keep = np.ones(len(table), dtype=bool)
    if max_dv is not None:
        keep &= table[:, 4] <= max_dv
    if max_duration_days is not None:
        keep &= duration <= max_duration_days
    return keep


def cluster_metrics(table, stats, max_dv=None, max_duration_days=None):
    """Minimo DV e numero di epoche distinte; arrivi multipli non duplicano un'epoca."""
    rows = np.searchsorted(stats[:, 0], table[:, 1])
    keep = transfer_mask(table, stats, max_dv, max_duration_days)
    minimum = np.full(len(stats), np.inf)
    np.minimum.at(minimum, rows[keep], table[keep, 4])
    pairs = np.unique(np.column_stack((rows[keep], table[keep, 0])), axis=0)
    counts = np.bincount(pairs[:, 0].astype(int), minlength=len(stats))
    return minimum, counts, int(keep.sum())


def porkchop_values(table, stats, max_dv=None, max_duration_days=None):
    """Un solo minimo DV per coppia esatta (epoca di partenza, epoca di arrivo)."""
    selected = table[transfer_mask(table, stats, max_dv, max_duration_days)]
    pairs, inverse = np.unique(selected[:, [0, 2]], axis=0, return_inverse=True)
    minimum = np.full(len(pairs), np.inf)
    np.minimum.at(minimum, inverse, selected[:, 4])
    return pairs, minimum


def best_transfer_index(table, stats, max_dv=None, max_duration_days=None):
    """Un vincitore per asteroide; parita risolta per partenza, arrivo e cluster ID."""
    candidates = np.flatnonzero(transfer_mask(table, stats, max_dv, max_duration_days))
    if not len(candidates):
        return -1
    best = candidates[table[candidates, 4] == table[candidates, 4].min()]
    order = np.lexsort((table[best, 1], table[best, 2], table[best, 0]))
    return int(best[order[0]])


def best_transfer_occurrences(geometry, winner_rows, n_clusters, mode="asteroids"):
    """Elementi dei membri dei cluster vincenti, aggregati senza duplicare le epoche."""
    if mode not in ("asteroids", "trajectories"):
        raise ValueError("La modalita delle occorrenze deve essere 'asteroids' o 'trajectories'.")
    winners = np.asarray(winner_rows, dtype=int)
    winners = winners[winners >= 0]
    frequency = np.bincount(winners, minlength=n_clusters)
    cells, rows = geometry["pair_cells"], geometry["pair_rows"]
    # Le coppie (cella, cluster) sono uniche: un asteroide conta una volta per cella.
    weights = frequency[rows]
    if mode == "trajectories":
        weights = weights * geometry["pair_counts"]
    occurrences = np.zeros(geometry["population"].size, dtype=np.int64)
    np.add.at(occurrences, cells, weights)
    represented = frequency[np.unique(rows)]
    return dict(
        occurrences=occurrences.reshape(geometry["population"].shape),
        edges_x=geometry["edges_x"], edges_y=geometry["edges_y"], mode=mode,
        selected_asteroids=len(winners), represented_asteroids=int(represented.sum()),
        winning_clusters=int(np.count_nonzero(frequency)),
        represented_clusters=int(np.count_nonzero(represented)),
        distinct_members=int(geometry["pair_counts"][frequency[rows] > 0].sum()),
    )


def plot_best_transfer_occurrences(data, y_label, vmax, only_bound=True, x_label=EPS_LABEL):
    """Mappa unica di tutti i vincitori, colorata con conteggi interi."""
    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout="constrained")
    counts = data["occurrences"]
    ax.set_facecolor("white")
    mesh = ax.pcolormesh(data["edges_x"], data["edges_y"], np.ma.masked_equal(counts.T, 0),
                        shading="flat", cmap="viridis", vmin=0, vmax=max(1, vmax))
    colourbar = fig.colorbar(mesh, ax=ax)
    colourbar.set_label("Numero di asteroidi nella cella" if data["mode"] == "asteroids"
                       else "Occorrenze delle traiettorie nella cella")
    colourbar.locator = MaxNLocator(nbins=8, integer=True)
    colourbar.update_ticks()
    ax.set(
        xlabel=x_label, ylabel=y_label,
        title=("Orbite al perigeo dei migliori trasferimenti\n"
               f"{data['represented_asteroids']}/{data['selected_asteroids']} asteroidi rappresentati — "
               f"{data['represented_clusters']} cluster — {data['distinct_members']} traiettorie distinte"),
    )
    ax.grid(alpha=0.15)
    if not np.any(counts):
        ax.text(0.5, 0.5, "Nessun membro dei cluster vincenti ammesso dai filtri",
                transform=ax.transAxes, ha="center", va="center")
    counting = ("Ogni asteroide conta al massimo una volta per cella."
                if data["mode"] == "asteroids" else
                "Ogni membro conta una volta per asteroide che seleziona il suo cluster.")
    parking = "Orbite legate con perigeo sopra la superficie." if only_bound else "Tutti i membri dei cluster vincenti."
    fig.supxlabel(
        "Un solo trasferimento a minimo ΔV di rendez-vous per asteroide, tra quelli ammessi dalle soglie.\n"
        f"{counting} {parking}\n"
        "Elementi dei membri del cluster vincente nel riferimento del catalogo; prestazioni del modello medio.",
        fontsize=9,
    )
    return fig


def plot_occurrence_percentiles(data, x_label, y_label):
    """Percentili sulle celle non vuote; maschere esatte, senza interpolazione."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    counts = np.asarray(data["occurrences"])
    occupied = counts > 0
    levels = OCCURRENCE_PERCENTILES
    thresholds = (np.percentile(counts[occupied], levels) if occupied.any()
                  else np.full(len(levels), np.nan))
    colors = ("#56b4e9", "#e69f00", "#cc0077", "#4c0099")
    masks = [occupied & (counts >= threshold) for threshold in thresholds]
    categories = np.zeros(counts.shape, dtype=int)
    for index, mask in enumerate(masks, 1):
        categories[mask] = index
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), layout="constrained")
    axes[-1, -1].axis("off")
    records = []
    for percentile, threshold, mask in zip(levels, thresholds, masks):
        number = int(mask.sum())
        records.append(dict(percentile=percentile, threshold=float(threshold),
                            minimum_integer_count=int(np.ceil(threshold)) if occupied.any() else None,
                            selected_cells=number, occupied_cells=int(occupied.sum()),
                            total_cells=int(counts.size),
                            occupied_fraction=number / max(1, int(occupied.sum())),
                            grid_fraction=number / counts.size))
    for ax in list(axes.flat)[:5]:
        ax.pcolormesh(data["edges_x"], data["edges_y"],
                      np.ma.masked_where(~occupied.T, np.ones(counts.T.shape)),
                      cmap=ListedColormap(["#e3e5e8"]), vmin=0, vmax=1, shading="flat")
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
    axes[0, 0].pcolormesh(
        data["edges_x"], data["edges_y"], np.ma.masked_equal(categories.T, 0),
        cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(0.5, 5), 4), shading="flat")
    axes[0, 0].set_title("Fasce sovrapposte: dal P97 ai nuclei P99,75")
    legend = []
    for ax, color, mask, record in zip(list(axes.flat)[1:], colors, masks, records):
        ax.pcolormesh(data["edges_x"], data["edges_y"],
                      np.ma.masked_where(~mask.T, np.ones(mask.T.shape)),
                      cmap=ListedColormap([color]), vmin=0, vmax=1, shading="flat")
        label = f"P{record['percentile']:g}".replace(".", ",")
        minimum = record["minimum_integer_count"]
        ax.set_title(
            f"{label}: almeno {minimum} asteroidi/cella\n"
            f"{record['selected_cells']} / {record['occupied_cells']} celle occupate "
            f"({100*record['occupied_fraction']:.2f}%) · "
            f"{100*record['grid_fraction']:.2f}% della griglia totale", fontsize=11)
        legend.append(Patch(facecolor=color, label=f"{label}: ≥ {minimum}"))
    axes[0, 0].legend(handles=legend, title="Asteroidi per cella", loc="best", fontsize=9)
    fig.suptitle(
        "Zone più popolate dei perigei dei migliori trasferimenti\n"
        f"{data['represented_asteroids']}/{data['selected_asteroids']} asteroidi — "
        f"{data['represented_clusters']} cluster — griglia {counts.shape[0]} × {counts.shape[1]}",
        fontsize=16)
    fig.supxlabel(
        "Percentili separati per mappa, sulle sole celle con conteggio > 0; selezione conteggio ≥ percentile.\n"
        "Grigio: celle occupate sotto soglia; bianco: celle vuote. Parità incluse: le percentuali effettive possono variare.\n"
        "Le percentuali descrivono celle, non asteroidi o traiettorie. Un asteroide conta al massimo una volta per cella; prestazioni del modello medio.",
        fontsize=10)
    return fig, records, np.asarray(masks)


def save_occurrence_percentiles(map_items, output):
    """Salva sei figure, PDF multipagina, griglie e statistiche verificabili."""
    import json
    from matplotlib.backends.backend_pdf import PdfPages

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    with PdfPages(output / "mappe_percentili_05_10.pdf") as pdf:
        for data, filename, xlabel, ylabel in map_items:
            stem = Path(filename).stem
            fig, records, masks = plot_occurrence_percentiles(data, xlabel, ylabel)
            try:
                fig.savefig(output / f"{stem}_percentili.png", dpi=180)
                pdf.savefig(fig)
            finally:
                plt.close(fig)
            np.savez_compressed(output / f"{stem}_griglie.npz",
                                occurrences=data["occurrences"], edges_x=data["edges_x"],
                                edges_y=data["edges_y"], percentiles=OCCURRENCE_PERCENTILES,
                                thresholds=[r["threshold"] for r in records], masks=masks)
            summaries[stem] = records
            print(f"Percentili salvati: {stem}", flush=True)
    (output / "riepilogo.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def epoch_cell_bounds(epochs, width_days):
    """Celle centrate sulle epoche, limitate dai vicini senza riempire le lacune."""
    centers = np.unique(epochs)
    lower = centers - width_days / 2
    upper = centers + width_days / 2
    midpoints = (centers[:-1] + centers[1:]) / 2
    lower[1:] = np.maximum(lower[1:], midpoints)
    upper[:-1] = np.minimum(upper[:-1], midpoints)
    return centers, lower, upper


def plot_porkchop(table, stats, info, max_dv=None, max_duration_days=None,
                  cell_days=PORKCHOP_CELL_DAYS):
    """Porkchop sparso: nessuna interpolazione delle coppie di date non salvate."""
    pairs, minimum = porkchop_values(table, stats, max_dv, max_duration_days)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout="constrained")
    reference = info["epoch_reference"]
    ax.set(xlabel=f"Data di partenza {reference} (TDB)", ylabel="Data di arrivo (TDB)",
           title=f"{info['name']} — porkchop\nMinimo ΔV di rendez-vous tra i cluster compatibili")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)

    # Conversione del calendario JD TDB per Matplotlib; nessuna conversione a UTC.
    j2000_plot = mdates.date2num(np.datetime64("2000-01-01T12:00:00"))

    def calendar_dates(jd):
        return np.asarray(jd) - 2451545.0 + j2000_plot

    if len(table):
        departures, left, right = epoch_cell_bounds(table[:, 0], cell_days)
        arrivals, bottom, top = epoch_cell_bounds(table[:, 2], cell_days)
        ax.set_xlim(calendar_dates(left[0]), calendar_dates(right[-1]))
        ax.set_ylim(calendar_dates(bottom[0]), calendar_dates(top[-1]))
        for axis in (ax.xaxis, ax.yaxis):
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            axis.set_major_locator(locator)
            axis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if len(pairs):
        ix = np.searchsorted(departures, pairs[:, 0])
        iy = np.searchsorted(arrivals, pairs[:, 1])
        vertices = np.array([
            np.column_stack((left[ix], bottom[iy])),
            np.column_stack((right[ix], bottom[iy])),
            np.column_stack((right[ix], top[iy])),
            np.column_stack((left[ix], top[iy])),
        ]).transpose(1, 0, 2)
        low, high = float(minimum.min()), float(minimum.max())
        if high == low:
            high = low + max(1e-6, 0.01 * abs(low))
        cells = PolyCollection(calendar_dates(vertices), array=minimum, cmap="viridis_r",
                               norm=plt.Normalize(low, high), edgecolors="none", antialiased=False)
        ax.add_collection(cells)
        fig.colorbar(cells, ax=ax, label="Minimo ΔV di rendez-vous [km/s]")
        best = int(np.argmin(minimum))
        ax.plot(*calendar_dates(pairs[best]), marker="*", linestyle="none", markersize=11,
                color="white", markeredgecolor="black", zorder=3,
                label=f"Minimo: {minimum[best]:.3f} km/s")
        ax.legend(loc="best", fontsize="small")
    else:
        ax.text(0.5, 0.5, "Nessun trasferimento sotto soglia" if len(table)
                else "Nessun trasferimento salvato", transform=ax.transAxes, ha="center", va="center")

    dv_text = "nessun limite" if max_dv is None else f"≤ {max_dv:g} km/s"
    duration_text = "nessun limite" if max_duration_days is None else f"≤ {max_duration_days:g} giorni"
    fig.supxlabel(
        f"ΔV: {dv_text}; durata perigeo medio → arrivo: {duration_text}\n"
        "Bianco: nessun trasferimento salvato ammesso dai filtri. Nessuna interpolazione.\n"
        "Modello medio dei cluster; solo impulso all'arrivo.", fontsize=9,
    )
    return fig


def prepare_parking_geometry(members, stats, only_bound=True, n_bins=30, y_variable="e"):
    """Selezione e molteplicita cella-cluster, riutilizzabili per tutti gli scenari."""
    if y_variable not in ("e", "omega"):
        raise ValueError("y_variable deve essere 'e' oppure 'omega'.")
    assigned = members[:, 1] >= 0
    eps, ecc = members[:, 9], members[:, 10]
    keep = assigned.copy()
    if only_bound:
        bound = (eps < 0) & (ecc >= 0) & (ecc < 1)
        rp = np.full(len(members), np.nan)
        rp[bound] = -MU_EARTH / (2 * eps[bound]) * (1 - ecc[bound])
        keep &= bound & (rp > R_EARTH_KM)
    if not keep.any():
        raise ValueError("Nessuna orbita di parcheggio supera la selezione del catalogo.")
    rows = np.searchsorted(stats[:, 0], members[keep, 1])
    x = eps[keep]
    edges_x = np.histogram_bin_edges(x, n_bins)
    if y_variable == "omega":
        y = np.mod(members[keep, 11], 360.0)
        edges_y = np.linspace(0.0, 360.0, n_bins + 1)
    else:
        y = ecc[keep]
        edges_y = np.histogram_bin_edges(y, n_bins)
    ix = np.minimum(np.searchsorted(edges_x, x, side="right") - 1, n_bins - 1)
    iy = np.minimum(np.searchsorted(edges_y, y, side="right") - 1, n_bins - 1)
    cell = ix * n_bins + iy
    population = np.bincount(cell, minlength=n_bins**2)
    pairs, pair_counts = np.unique(cell * len(stats) + rows, return_counts=True)
    return dict(edges_x=edges_x, edges_y=edges_y, population=population.reshape(n_bins, n_bins),
                pair_cells=pairs // len(stats), pair_rows=pairs % len(stats), pair_counts=pair_counts,
                selected=int(keep.sum()), unassigned=int((~assigned).sum()),
                excluded=int((assigned & ~keep).sum()))


def winning_cluster_directions(members, catalog_path):
    """Recover perigee sense for GA16 clusters, which are homogeneous in sense.

    The legacy 12-column catalog omits this flag. Integrate one representative
    per cluster to its SAVED perigee, and verify its elements against the catalog.
    SOI angular momentum is not used: lunar encounters can change its sign.
    """
    from scipy.integrate import solve_ivp

    catalog_path = Path(catalog_path)
    source = None
    with catalog_path.open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            if line.startswith("# Source:"):
                source = Path(line.partition(":")[2].strip())
                if not source.is_absolute():
                    source = catalog_path.parent / source
                break
    if source is None:
        raise ValueError(f"{catalog_path}: manca Source per recuperare il verso al perigeo.")
    clusters, representatives = np.unique(members[:, 1].astype(int), return_index=True)
    wanted = {int(members[i, 0]): i for i in representatives}
    states = {}
    with source.open() as stream:
        row_id = 0
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            if row_id in wanted:
                states[wanted[row_id]] = np.fromstring(line, sep=" ")[:6]
            row_id += 1
            if len(states) == len(wanted):
                break
    if len(states) != len(wanted):
        raise ValueError("Source non contiene tutti i source_id richiesti.")
    mass_ratio = MOON_MASS_KG / (5.974e24 + MOON_MASS_KG)

    def rhs(t, state):
        x, y, z, vx, vy, vz = state
        r1 = np.linalg.norm([x + mass_ratio, y, z])
        r2 = np.linalg.norm([x - 1 + mass_ratio, y, z])
        force = (1 - mass_ratio) / r1**3 + mass_ratio / r2**3
        return (vx, vy, vz,
                2 * vy + x - (1 - mass_ratio) * (x + mass_ratio) / r1**3
                - mass_ratio * (x - 1 + mass_ratio) / r2**3,
                -2 * vx + y - force * y, -force * z)

    directions = np.empty(len(clusters))
    theta_errors = []
    for j, i in enumerate(representatives):
        target = members[i, 5] * 86400.0 / TIME_UNIT_S
        sol = solve_ivp(rhs, (0, target), states[i], method="DOP853",
                        rtol=1e-10, atol=1e-12, t_eval=[target])
        if not sol.success:
            raise ValueError(f"Integrazione del rappresentante del cluster {clusters[j]} fallita.")
        state = sol.y[:, -1]
        r = state[:3] + [mass_ratio, 0, 0]
        v = state[3:] + np.cross([0, 0, 1], r)
        h = np.cross(r, v)
        direction = np.sign(h[2])
        energy = (v @ v / 2 - (1 - mass_ratio) / np.linalg.norm(r)) * (
            MU_EARTH + 6.67430e-20 * MOON_MASS_KG) / EARTH_MOON_DISTANCE_KM
        eccentricity_vector = np.cross(v, h) / (1 - mass_ratio) - r / np.linalg.norm(r)
        ecc = np.linalg.norm(eccentricity_vector)
        azimuth = np.rad2deg(np.arctan2(eccentricity_vector[1], eccentricity_vector[0]) + target)
        omega_error = (direction * azimuth - members[i, 11] + 180) % 360 - 180
        # Roberto's definition: perigee position angle in the synodic frame,
        # where the Earth-Moon direction is +x at that SAME epoch.
        theta_direct = np.rad2deg(np.arctan2(r[1], r[0]))
        theta_catalog = direction * members[i, 11] - np.rad2deg(target)
        theta_error = (theta_catalog - theta_direct + 180) % 360 - 180
        theta_errors.append(abs(theta_error))
        if (direction == 0 or abs(state[2]) > 1e-10
                or not np.isclose(energy, members[i, 9], rtol=1e-5, atol=1e-6)
                or not np.isclose(ecc, members[i, 10], rtol=1e-5, atol=1e-6)
                or abs(omega_error) > 0.02 or abs(theta_error) > 0.02):
            raise ValueError(f"Cluster {clusters[j]}: reintegrazione incoerente con il perigeo salvato: "
                             f"eps={energy} (catalogo {members[i, 9]}), "
                             f"e={ecc} (catalogo {members[i, 10]}), errore omega={omega_error} deg, "
                             f"errore theta sinodico={theta_error} deg.")
        directions[j] = direction
    print(f"Theta verificato dalla posizione sinodica al perigeo su {len(clusters)} cluster; "
          f"errore angolare massimo {max(theta_errors, default=0):.6g} deg.", flush=True)
    return directions[np.searchsorted(clusters, members[:, 1].astype(int))]


def additional_best_transfer_maps(members, stats, winner_rows, catalog_path,
                                  only_bound=True, n_bins=100, mode="asteroids", occurrence_threshold=None, occurrence_fraction=None, return_member_occurrences=False):
    """Four additional projections of the same winning-cluster members as 05/06."""
    winner_rows = np.asarray(winner_rows, dtype=int)
    winning = np.unique(winner_rows[winner_rows >= 0])
    subset = members[np.isin(members[:, 1], stats[winning, 0])]
    eps, ecc = subset[:, 9], subset[:, 10]
    # r_p = h^2 / (mu * (1+e)) = -mu/(2 eps) * (1-e), also for hyperbolas.
    rp = np.full(len(subset), np.nan)
    np.divide(-MU_EARTH * (1 - ecc), 2 * eps, out=rp, where=eps != 0)
    valid = np.isfinite(rp) & (rp > 0)
    if only_bound:
        valid &= (eps < 0) & (ecc >= 0) & (ecc < 1) & (rp > R_EARTH_KM)
    subset, rp = subset[valid], rp[valid]
    print("Recupero verso al perigeo dei cluster vincenti...", flush=True)
    directions = winning_cluster_directions(subset, catalog_path) if len(subset) else np.empty(0)
    omega = subset[:, 11] % 360
    theta = (directions * omega - MOON_RATE_DEG_DAY * subset[:, 5]) % 360
    altitude = rp - R_EARTH_KM
    projections = [
        ("07_migliori_transfer_occorrenze_rp_e.png", altitude, subset[:, 10], HP_LABEL, "e al perigeo [-]", False),
        ("08_migliori_transfer_occorrenze_rp_omega.png", altitude, omega, HP_LABEL, r"$\omega$ al perigeo, catalogo [deg]", True),
        ("09_migliori_transfer_occorrenze_rp_theta.png", altitude, theta, HP_LABEL, THETA_LABEL, True),
        ("10_migliori_transfer_occorrenze_eps_theta.png", subset[:, 9], theta, EPS_LABEL, THETA_LABEL, True),
    ]
    result = []
    rows = np.searchsorted(stats[:, 0], subset[:, 1])
    for filename, x, y, xlabel, ylabel, angular in projections:
        edges_x = np.histogram_bin_edges(x, n_bins)
        edges_y = np.linspace(0, 360, n_bins + 1) if angular else np.histogram_bin_edges(y, n_bins)
        ix = np.clip(np.searchsorted(edges_x, x, side="right") - 1, 0, n_bins - 1)
        iy = np.clip(np.searchsorted(edges_y, y, side="right") - 1, 0, n_bins - 1)
        cell = ix * n_bins + iy
        pairs, pair_counts = np.unique(cell * len(stats) + rows, return_counts=True)
        geometry = dict(edges_x=edges_x, edges_y=edges_y,
                        population=np.bincount(cell, minlength=n_bins**2).reshape(n_bins, n_bins),
                        pair_cells=pairs // len(stats), pair_rows=pairs % len(stats), pair_counts=pair_counts)
        data = best_transfer_occurrences(geometry, winner_rows, len(stats), mode)
        if occurrence_threshold is not None or occurrence_fraction is not None:
            threshold = (max(1, int(np.ceil(occurrence_fraction * data["occurrences"].max())))
                         if occurrence_fraction is not None else occurrence_threshold)
            data["selection_threshold"] = threshold
            data["selected_source_ids"] = subset[
                data["occurrences"].ravel()[cell] >= threshold, 0].astype(np.int64)
        if return_member_occurrences:
            data["member_source_ids"] = subset[:, 0].astype(np.int64)
            data["member_occurrences"] = data["occurrences"].ravel()[cell]
        result.append((data, filename, xlabel, ylabel))
    return result


def maps_from_geometry(geometry, minimum, counts):
    """Denominatore di ogni cella: tutti i membri assegnati ammessi, anche senza hit."""
    population = geometry["population"].ravel()
    cell, rows, weights = geometry["pair_cells"], geometry["pair_rows"], geometry["pair_counts"]
    best_dv = np.full(population.size, np.inf)
    np.minimum.at(best_dv, cell, minimum.min(axis=0)[rows])
    mean_epochs = np.bincount(cell, weights=counts.mean(axis=0)[rows] * weights,
                           minlength=population.size)
    coverage = np.bincount(cell, weights=100 * (counts > 0).mean(axis=0)[rows] * weights,
                          minlength=population.size)
    occupied = population > 0
    mean_epochs[occupied] /= population[occupied]
    coverage[occupied] /= population[occupied]
    best_dv[~np.isfinite(best_dv)] = np.nan
    mean_epochs[~occupied] = np.nan
    coverage[~occupied] = np.nan
    shape = geometry["population"].shape
    return dict(edges_x=geometry["edges_x"], edges_y=geometry["edges_y"], population=geometry["population"],
                best_dv=best_dv.reshape(shape), mean_epochs=mean_epochs.reshape(shape),
                coverage=coverage.reshape(shape), selected=geometry["selected"],
                unassigned=geometry["unassigned"], excluded=geometry["excluded"])


def parking_maps(members, stats, minimum, counts, only_bound=True, n_bins=30, y_variable="e"):
    """Costruisce le mappe preparando la geometria del catalogo una sola volta."""
    geometry = prepare_parking_geometry(members, stats, only_bound, n_bins, y_variable)
    return maps_from_geometry(geometry, minimum, counts)


def main(transfer_files=None, output_dir=None, max_dv=MAX_DV_KM_S,
         max_duration_days=MAX_DURATION_DAYS, only_bound=ONLY_BOUND_PARKING, n_bins=N_BINS,
         panels_per_page=PANELS_PER_PAGE, plot_columns=PLOT_COLUMNS,
         porkchop_cell_days=PORKCHOP_CELL_DAYS, best_occurrence_mode=BEST_OCCURRENCE_MODE):
    for limit in (max_dv, max_duration_days):
        if limit is not None and (not np.isfinite(limit) or limit < 0):
            raise ValueError("Le soglie devono essere None oppure numeri finiti non negativi.")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins deve essere un intero positivo.")
    for value in (panels_per_page, plot_columns):
        if not isinstance(value, int) or value < 1:
            raise ValueError("panels_per_page e plot_columns devono essere interi positivi.")
    if not np.isfinite(porkchop_cell_days) or porkchop_cell_days <= 0:
        raise ValueError("porkchop_cell_days deve essere finito e positivo.")
    if best_occurrence_mode not in ("asteroids", "trajectories"):
        raise ValueError("best_occurrence_mode deve essere 'asteroids' oppure 'trajectories'.")
    print("Caricamento e verifica dei cataloghi...", flush=True)
    selected_files = TRANSFER_FILES if transfer_files is None else transfer_files
    if selected_files is None:
        selected_files = discover_transfer_files(TRANSFER_DIR)
    paths, catalog, stats, members, tables, metadata = load_inputs(selected_files)
    references = {info["epoch_reference"] for info in metadata}
    epoch_label = next(iter(references)) if len(references) == 1 else "partenza (ETD/SOI)"
    epoch_suffix = next(iter(references)).lower() if len(references) == 1 else "partenza"
    print(f"Calcolo delle statistiche per {len(paths)} asteroidi...", flush=True)
    transfer_counts = [len(table) for table in tables]
    metrics = [cluster_metrics(t, stats, max_dv, max_duration_days) for t in tables]
    minimum = np.array([m[0] for m in metrics])
    counts = np.array([m[1] for m in metrics])
    best_indices = [best_transfer_index(t, stats, max_dv, max_duration_days) for t in tables]
    winner_rows = np.array([
        np.searchsorted(stats[:, 0], table[index, 1]) if index >= 0 else -1
        for table, index in zip(tables, best_indices)
    ], dtype=int)
    print("Preparazione delle celle eps-e ed eps-omega (una volta per proiezione)...", flush=True)
    geometry = prepare_parking_geometry(members, stats, only_bound, n_bins)
    omega_geometry = prepare_parking_geometry(members, stats, only_bound, n_bins, y_variable="omega")
    additional_maps = additional_best_transfer_maps(
        members, stats, winner_rows, catalog.with_name("trajectory_exit_catalog.txt"),
        only_bound, n_bins, best_occurrence_mode)
    del members
    best_maps = best_transfer_occurrences(geometry, winner_rows, len(stats), best_occurrence_mode)
    best_omega_maps = best_transfer_occurrences(omega_geometry, winner_rows, len(stats), best_occurrence_mode)
    maps = maps_from_geometry(geometry, minimum, counts)
    scenario_maps = ScenarioMaps(geometry, minimum, counts)
    omega_maps = maps_from_geometry(omega_geometry, minimum, counts)
    omega_scenario_maps = ScenarioMaps(omega_geometry, minimum, counts)
    output = Path(OUTPUT_DIR if output_dir is None else output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dv_label = "nessun limite" if max_dv is None else f"<= {max_dv:g} km/s"
    duration_label = "nessun limite" if max_duration_days is None else f"<= {max_duration_days:g} giorni"
    report = [f"Catalogo: {catalog}", f"Scenari caricati: {len(paths)}",
              f"Mappe individuali in formato 16:9, massimo {panels_per_page} pannelli e {plot_columns} colonne per pagina.",
              "Mappe eps-e ed eps-omega; omega al perigeo negli assi del catalogo, modulo 360 gradi.",
              "Mappe aggiuntive 07-10: hp-e, hp-omega, hp-theta, eps-theta; hp = rp - 6378 km dalla superficie terrestre.",
              "Theta = (senso_perigeo * omega - n_moon * t_perigee) mod 360; antiorario dalla Luna al perigeo.",
              "Verso recuperato al perigeo salvato da un rappresentante per cluster GA16, omogenei per verso.",
              "Theta verificato con atan2(y_perigeo, x_perigeo) nel sinodico, sulla posizione dei rappresentanti.",
              "Un porkchop per asteroide: partenza e arrivo TDB, minimo DV tra cluster per ogni coppia esatta di date.",
              f"Celle porkchop larghe al massimo {porkchop_cell_days:g} giorni, senza interpolazione o riempimento delle lacune.",
              "Porkchop: tutti i trasferimenti salvati che rispettano le soglie DV/durata, senza filtro sugli elementi delle orbite di parcheggio.",
              "Mappe dei migliori trasferimenti: un solo minimo DV per asteroide, dopo i filtri DV/durata; parita per partenza, arrivo, cluster ID.",
              f"Occorrenze migliori trasferimenti: {best_occurrence_mode}; "
              f"{best_maps['represented_asteroids']}/{best_maps['selected_asteroids']} asteroidi rappresentati, "
              f"{best_maps['represented_clusters']}/{best_maps['winning_clusters']} cluster vincenti rappresentati.",
              "Un cluster vincente senza membri ammessi dal filtro delle orbite non viene sostituito da un secondo classificato.",
              f"Orbita di parcheggio: {'legata, perigeo sopra la superficie' if only_bound else 'nessun filtro fisico'}",
              f"Membri assegnati analizzati: {maps['selected']}; non assegnati esclusi: {maps['unassigned']}; altri esclusi: {maps['excluded']}",
              f"DV rendez-vous: {dv_label}; durata perigeo MEDIO -> arrivo: {duration_label}",
              "Ogni membro eredita le prestazioni del modello medio del proprio cluster.",
              "Min DV per cella, per scenario: migliore trasferimento tra i cluster rappresentati nella cella.",
              f"Epoche {epoch_label} per cella, per scenario: media per membro del numero di epoche distinte compatibili.",
              "Copertura per cella: media per membro della percentuale di scenari raggiungibili.",
              "Il denominatore include anche i membri senza trasferimenti sotto soglia.",
              "La copertura riguarda solo gli asteroidi caricati: GA18 non salva file per gli asteroidi senza soluzioni.",
              "La griglia completa delle epoche non e salvata: i conteggi non sono probabilita o frazioni delle partenze provate.",
              "Finestre e passi temporali diversi fra scenari possono influenzare i conteggi delle epoche.",
              "Le prestazioni si riferiscono al modello planare di GA18, non a effemeridi complete 3D.",
              "Il DV include solo il rendez-vous, non il costo di partenza dall'orbita terrestre."]
    for path, info, n_transfers, metric, table, best_index in zip(paths, metadata, transfer_counts, metrics, tables, best_indices):
        report.append(f"{info['name']} (SPK-ID: {info['spkid']}; epoche {info['epoch_reference']}): "
                      f"{n_transfers} trasferimenti caricati; {metric[2]} sotto soglia; "
                      f"{np.count_nonzero(metric[1])} cluster raggiungibili; file: {path}")
        if best_index >= 0:
            best = table[best_index]
            report.append(f"  Miglior trasferimento: cluster {int(best[1])}; DV {best[4]:.12g} km/s; "
                          f"partenza {info['epoch_reference']} JD {best[0]:.8f}; arrivo JD {best[2]:.8f}; TOF {best[3]:.8f} giorni.")
    (output / "riepilogo.txt").write_text("\n".join(report) + "\n")
    print(f"Analizzate {maps['selected']} orbite; dettagli in {output / 'riepilogo.txt'}", flush=True)

    figures = [
        ("best_dv", "01_minimo_dv.png", "Miglior rendez-vous disponibile", "Minimo DV di rendez-vous [km/s]", "viridis_r"),
        ("mean_epochs", f"02_opportunita_{epoch_suffix}.png", "Opportunita temporali delle orbite di parcheggio", f"Epoche {epoch_label} distinte compatibili per orbita (media nella cella)", "viridis"),
        ("coverage", "03_copertura_scenari.png", "Accesso agli scenari selezionati", "Scenari raggiungibili per orbita [% medio]", "viridis"),
    ]
    projections = [
        (maps, scenario_maps, "", "e al perigeo [-]"),
        (omega_maps, omega_scenario_maps, "_eps_omega", r"$\omega$ al perigeo [deg]"),
    ]
    generated_files = []
    best_occurrence_files = []
    best_vmax = max(data["occurrences"].max() for data in [best_maps, best_omega_maps] + [item[0] for item in additional_maps])
    occurrence_items = [
        (best_maps, "05_migliori_transfer_occorrenze_eps_e.png", EPS_LABEL, "e al perigeo [-]"),
        (best_omega_maps, "06_migliori_transfer_occorrenze_eps_omega.png", EPS_LABEL, r"$\omega$ al perigeo, catalogo [deg]"),
    ] + additional_maps
    save_occurrence_percentiles(occurrence_items, output / "mappe_percentili")
    for data, filename, x_label, y_label in occurrence_items:
        fig = plot_best_transfer_occurrences(data, y_label, best_vmax, only_bound, x_label)
        try:
            image_path = output / filename
            fig.savefig(image_path, dpi=180)
            generated_files.append(image_path)
            best_occurrence_files.append(image_path)
            print(f"Salvata {image_path.name}", flush=True)
        finally:
            plt.close(fig)
            fig.clear()
            gc.collect()
    porkchop_files = []
    for index, (table, info) in enumerate(zip(tables, metadata)):
        fig = plot_porkchop(table, stats, info, max_dv, max_duration_days, porkchop_cell_days)
        try:
            identifier = info["spkid"] or f"scenario_{index:04d}"
            image_path = output / f"04_porkchop_{identifier}.png"
            fig.savefig(image_path, dpi=180)
            generated_files.append(image_path)
            porkchop_files.append(image_path)
        finally:
            plt.close(fig)
            fig.clear()
            gc.collect()
        if (index + 1) % 20 == 0 or index + 1 == len(tables):
            print(f"Porkchop salvati: {index + 1}/{len(tables)}", flush=True)
    del tables
    for aggregate, scenarios, suffix, y_label in projections:
        for key, filename, title, color_label, cmap in figures:
            panels = [aggregate] if key == "coverage" else scenarios
            n_pages = (len(panels) + panels_per_page - 1) // panels_per_page
            for page, start in enumerate(range(0, len(panels), panels_per_page), 1):
                stop = min(start + panels_per_page, len(panels))
                page_panels = stop - start
                ncols = min(plot_columns, page_panels, max(1, round(np.sqrt(page_panels * 16 / 9))))
                nrows = (page_panels + ncols - 1) // ncols
                compact = page_panels > 3
                # Rapporto 16:9 anche nelle pagine parziali; piena: 7680 x 4320 px a 180 dpi.
                page_width = max(8, 8 * ncols / 3)
                fig = plt.figure(figsize=(page_width, page_width * 9 / 16),
                                 layout="constrained")
                try:
                    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03)
                    grid = fig.add_gridspec(nrows, 1)
                    axes = []
                    for row in range(nrows):
                        # Anche l'ultima riga occupa tutta la larghezza disponibile.
                        row_columns = min(ncols, stop - start - row * ncols)
                        row_grid = grid[row].subgridspec(1, row_columns)
                        for column in range(row_columns):
                            ax = fig.add_subplot(row_grid[0, column],
                                                 sharex=axes[0] if axes else None,
                                                 sharey=axes[0] if axes else None)
                            ax.tick_params(labelsize=6 if compact else 9,
                                           labelbottom=row == nrows - 1, labelleft=column == 0)
                            axes.append(ax)
                    for i, scenario_index in enumerate(range(start, stop)):
                        panel = panels[scenario_index]
                        # Scala indipendente per asteroide; copertura aggregata sempre 0-100%.
                        finite = panel[key][np.isfinite(panel[key])]
                        low = float(finite.min()) if key != "coverage" and finite.size else 0.0
                        high = 100.0 if key == "coverage" else float(finite.max()) if finite.size else 1.0
                        if high <= low:
                            high = low + max(1e-6, 0.01 * abs(low))
                        ax = axes[i]
                        ax.set_facecolor("0.92")
                        mesh = ax.pcolormesh(panel["edges_x"], panel["edges_y"], np.ma.masked_invalid(panel[key].T),
                                             shading="flat", cmap=cmap, vmin=low, vmax=high)
                        ax.set_title(f"{len(paths)} scenari caricati" if key == "coverage" else metadata[scenario_index]["name"],
                                     fontsize=8 if compact else 10, pad=3)
                        if key == "best_dv" and not np.isfinite(panel[key]).any():
                            ax.text(0.5, 0.5, "Nessun trasferimento\nsotto soglia", transform=ax.transAxes,
                                    ha="center", va="center", fontsize=7 if compact else 10)
                        colorbar = fig.colorbar(mesh, ax=ax, shrink=0.96, fraction=0.055, pad=0.02)
                        colorbar.ax.tick_params(labelsize=6 if compact else 9)
                        colorbar.ax.yaxis.get_offset_text().set_fontsize(6 if compact else 9)
                    page_label = f" — pagina {page}/{n_pages}" if n_pages > 1 else ""
                    scale_label = "scala per asteroide" if key != "coverage" else "scala 0-100%"
                    fig.suptitle(f"{title}{page_label}\n{aggregate['selected']} orbite — prestazioni del modello medio dei cluster\n"
                                 f"Colore: {color_label} — {scale_label}", fontsize=12)
                    fig.supylabel(y_label, fontsize=12)
                    fig.supxlabel(r"$\varepsilon$ al perigeo [km$^2$/s$^2$]" + "\n"
                                  f"DV: {dv_label}; durata perigeo medio -> arrivo: {duration_label}", fontsize=10)
                    page_suffix = f"_pagina_{page:03d}" if n_pages > 1 else ""
                    image_path = output / f"{Path(filename).stem}{suffix}{page_suffix}.png"
                    fig.savefig(image_path, dpi=180)
                    generated_files.append(image_path)
                    print(f"Salvata {image_path.name} ({page}/{n_pages})", flush=True)
                finally:
                    plt.close(fig)
                    fig.clear()
                    gc.collect()
    (output / "immagini_generate.txt").write_text("\n".join(p.name for p in generated_files) + "\n")
    print(f"Salvate {len(generated_files)} immagini (mappe orbitali e porkchop) in: {output}")
    return dict(maps=maps, scenario_maps=scenario_maps, omega_maps=omega_maps, metadata=metadata,
                omega_scenario_maps=omega_scenario_maps, minimum=minimum, counts=counts,
                metrics=metrics, output_dir=output, generated_files=generated_files, porkchop_files=porkchop_files,
                best_maps=best_maps, best_omega_maps=best_omega_maps, best_indices=best_indices,
                best_occurrence_files=best_occurrence_files, additional_best_maps=additional_maps)


if __name__ == "__main__":
    main()
