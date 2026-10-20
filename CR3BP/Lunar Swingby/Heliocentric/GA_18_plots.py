"""Mappe delle orbite terrestri, senza ricalcolare Lambert.

Ogni file caricato rappresenta uno scenario: il nome numerico dell'asteroide
non identifica necessariamente la stessa orbita in esecuzioni diverse.
I risultati del modello medio di un cluster sono attribuiti ai suoi membri.
"""

from pathlib import Path
from collections.abc import Sequence
import gc
import warnings

import numpy as np
from matplotlib import pyplot as plt


# Scansione delle cartelle asteroide_<numero>, ordinate numericamente.
BASE_DIR = Path(__file__).resolve().parent
TRANSFER_FILES = sorted(
    (path for path in (BASE_DIR / "GA18_plots").glob("asteroide_*/transfers_rendezvous.txt")
     if path.is_file() and path.parent.name.removeprefix("asteroide_").isdigit()),
    key=lambda path: int(path.parent.name.removeprefix("asteroide_")),
)
OUTPUT_DIR = BASE_DIR / "GA18_parking_plots"
MAX_DV_KM_S = None             # Solo rendez-vous all'arrivo; None = nessun limite.
MAX_DURATION_DAYS = None       # Perigeo MEDIO -> arrivo; None = nessun limite.
ONLY_BOUND_PARKING = True      # eps < 0, 0 <= e < 1, perigeo sopra la superficie.
N_BINS = 100                   # Celle per asse nelle mappe eps-e ed eps-omega.
PANELS_PER_PAGE = 144          # Pagine 16:9: griglia completa di 16 x 9 asteroidi.
PLOT_COLUMNS = 16              # Massimo; la griglia si adatta nelle pagine parziali.
MU_EARTH = 6.67430e-20 * 5.974e24
R_EARTH_KM = 6378.0


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


def load_inputs(transfer_files):
    paths = [Path(p) if Path(p).is_absolute() else BASE_DIR / p for p in transfer_files]
    paths = [p.resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Selezionare almeno un file, senza duplicare lo stesso file.")
    catalog_paths, tables = [], []
    for path in paths:
        catalog = None
        with path.open() as stream:
            for line in stream:
                if not line.startswith("#"):
                    break
                if line.startswith("# Catalog:"):
                    catalog = Path(line.partition(":")[2].strip())
        if catalog is None:
            raise ValueError(f"{path}: manca l'header '# Catalog:'.")
        catalog_paths.append((catalog if catalog.is_absolute() else path.parent / catalog).resolve())
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
    for path, table in zip(paths, tables):
        labels_t = integer_ids(table[:, 1], f"cluster_id in {path}")
        if not np.isin(labels_t, ids).all():
            raise ValueError(f"{path}: cluster non presente nel catalogo.")
        rows = np.searchsorted(ids, labels_t)
        if np.any(table[:, 3] <= 0) or np.any(table[:, 4] < 0):
            raise ValueError(f"{path}: TOF non positivo o delta-v negativo.")
        if not np.allclose(table[:, 2] - table[:, 0] - table[:, 3], stats[rows, 7], rtol=0, atol=1e-7):
            raise ValueError(f"{path}: epoche/TOF incoerenti con il tempo SOI del catalogo.")
    return paths, catalog_path, stats, members, tables


def cluster_metrics(table, stats, max_dv=None, max_duration_days=None):
    """Minimo DV e numero di ETD distinte; arrivi multipli non duplicano un'ETD."""
    rows = np.searchsorted(stats[:, 0], table[:, 1])
    duration = table[:, 2] - table[:, 0] - stats[rows, 6]
    keep = np.ones(len(table), dtype=bool)
    if max_dv is not None:
        keep &= table[:, 4] <= max_dv
    if max_duration_days is not None:
        keep &= duration <= max_duration_days
    minimum = np.full(len(stats), np.inf)
    np.minimum.at(minimum, rows[keep], table[keep, 4])
    pairs = np.unique(np.column_stack((rows[keep], table[keep, 0])), axis=0)
    counts = np.bincount(pairs[:, 0].astype(int), minlength=len(stats))
    return minimum, counts, int(keep.sum())


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


def maps_from_geometry(geometry, minimum, counts):
    """Denominatore di ogni cella: tutti i membri assegnati ammessi, anche senza hit."""
    population = geometry["population"].ravel()
    cell, rows, weights = geometry["pair_cells"], geometry["pair_rows"], geometry["pair_counts"]
    best_dv = np.full(population.size, np.inf)
    np.minimum.at(best_dv, cell, minimum.min(axis=0)[rows])
    mean_etds = np.bincount(cell, weights=counts.mean(axis=0)[rows] * weights,
                           minlength=population.size)
    coverage = np.bincount(cell, weights=100 * (counts > 0).mean(axis=0)[rows] * weights,
                          minlength=population.size)
    occupied = population > 0
    mean_etds[occupied] /= population[occupied]
    coverage[occupied] /= population[occupied]
    best_dv[~np.isfinite(best_dv)] = np.nan
    mean_etds[~occupied] = np.nan
    coverage[~occupied] = np.nan
    shape = geometry["population"].shape
    return dict(edges_x=geometry["edges_x"], edges_y=geometry["edges_y"], population=geometry["population"],
                best_dv=best_dv.reshape(shape), mean_etds=mean_etds.reshape(shape),
                coverage=coverage.reshape(shape), selected=geometry["selected"],
                unassigned=geometry["unassigned"], excluded=geometry["excluded"])


def parking_maps(members, stats, minimum, counts, only_bound=True, n_bins=30, y_variable="e"):
    """Costruisce le mappe preparando la geometria del catalogo una sola volta."""
    geometry = prepare_parking_geometry(members, stats, only_bound, n_bins, y_variable)
    return maps_from_geometry(geometry, minimum, counts)


def main(transfer_files=None, output_dir=None, max_dv=MAX_DV_KM_S,
         max_duration_days=MAX_DURATION_DAYS, only_bound=ONLY_BOUND_PARKING, n_bins=N_BINS,
         panels_per_page=PANELS_PER_PAGE, plot_columns=PLOT_COLUMNS):
    for limit in (max_dv, max_duration_days):
        if limit is not None and (not np.isfinite(limit) or limit < 0):
            raise ValueError("Le soglie devono essere None oppure numeri finiti non negativi.")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins deve essere un intero positivo.")
    for value in (panels_per_page, plot_columns):
        if not isinstance(value, int) or value < 1:
            raise ValueError("panels_per_page e plot_columns devono essere interi positivi.")
    print("Caricamento e verifica dei cataloghi...", flush=True)
    paths, catalog, stats, members, tables = load_inputs(TRANSFER_FILES if transfer_files is None else transfer_files)
    print(f"Calcolo delle statistiche per {len(paths)} asteroidi...", flush=True)
    transfer_counts = [len(table) for table in tables]
    metrics = [cluster_metrics(t, stats, max_dv, max_duration_days) for t in tables]
    del tables
    minimum = np.array([m[0] for m in metrics])
    counts = np.array([m[1] for m in metrics])
    print("Preparazione delle celle eps-e ed eps-omega (una volta per proiezione)...", flush=True)
    geometry = prepare_parking_geometry(members, stats, only_bound, n_bins)
    omega_geometry = prepare_parking_geometry(members, stats, only_bound, n_bins, y_variable="omega")
    del members
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
              f"Orbita di parcheggio: {'legata, perigeo sopra la superficie' if only_bound else 'nessun filtro fisico'}",
              f"Membri assegnati analizzati: {maps['selected']}; non assegnati esclusi: {maps['unassigned']}; altri esclusi: {maps['excluded']}",
              f"DV rendez-vous: {dv_label}; durata perigeo MEDIO -> arrivo: {duration_label}",
              "Ogni membro eredita le prestazioni del modello medio del proprio cluster.",
              "Min DV per cella, per scenario: migliore trasferimento tra i cluster rappresentati nella cella.",
              "ETD per cella, per scenario: media per membro del numero di ETD distinte compatibili.",
              "Copertura per cella: media per membro della percentuale di scenari raggiungibili.",
              "Il denominatore include anche i membri senza trasferimenti sotto soglia.",
              "La griglia completa delle ETD non e salvata: i conteggi non sono probabilita o frazioni delle partenze provate.",
              "Finestre e passi temporali diversi fra scenari possono influenzare i conteggi di ETD.",
              "Il DV include solo il rendez-vous, non il costo di partenza dall'orbita terrestre."]
    for path, n_transfers, metric in zip(paths, transfer_counts, metrics):
        report.append(f"{path.parent.name}: {n_transfers} trasferimenti caricati; {metric[2]} sotto soglia; {np.count_nonzero(metric[1])} cluster raggiungibili")
    (output / "riepilogo.txt").write_text("\n".join(report) + "\n")
    print(f"Analizzate {maps['selected']} orbite; dettagli in {output / 'riepilogo.txt'}", flush=True)

    figures = [
        ("best_dv", "01_minimo_dv.png", "Miglior rendez-vous disponibile", "Minimo DV di rendez-vous [km/s]", "viridis_r"),
        ("mean_etds", "02_opportunita_etd.png", "Opportunita temporali delle orbite di parcheggio", "ETD distinte compatibili per orbita (media nella cella)", "viridis"),
        ("coverage", "03_copertura_scenari.png", "Accesso agli scenari selezionati", "Scenari raggiungibili per orbita [% medio]", "viridis"),
    ]
    projections = [
        (maps, scenario_maps, "", "e al perigeo [-]"),
        (omega_maps, omega_scenario_maps, "_eps_omega", r"$\omega$ al perigeo [deg]"),
    ]
    generated_files = []
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
                        ax.set_title(f"{len(paths)} scenari caricati" if key == "coverage" else paths[scenario_index].parent.name,
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
    print(f"Salvate {len(generated_files)} immagini (eps-e ed eps-omega) in: {output}")
    return dict(maps=maps, scenario_maps=scenario_maps, omega_maps=omega_maps,
                omega_scenario_maps=omega_scenario_maps, minimum=minimum, counts=counts,
                metrics=metrics, output_dir=output, generated_files=generated_files)


if __name__ == "__main__":
    main()
