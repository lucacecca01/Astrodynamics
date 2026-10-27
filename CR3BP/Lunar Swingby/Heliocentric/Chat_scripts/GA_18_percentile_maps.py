"""Rigenera solo le mappe dei percentili, dai risultati GA18 correnti."""

import numpy as np

import GA_18_plots as plots


def main():
    files = (plots.discover_transfer_files(plots.TRANSFER_DIR)
             if plots.TRANSFER_FILES is None else plots.TRANSFER_FILES)
    print("Caricamento dei risultati correnti e verifica del catalogo...", flush=True)
    paths, catalog, stats, members, tables, metadata = plots.load_inputs(files)
    winners = []
    for table in tables:
        index = plots.best_transfer_index(table, stats, plots.MAX_DV_KM_S, plots.MAX_DURATION_DAYS)
        winners.append(int(np.searchsorted(stats[:, 0], table[index, 1])) if index >= 0 else -1)
    items = []
    for variable, filename, ylabel in (
        ("e", "05_migliori_transfer_occorrenze_eps_e.png", "e al perigeo [-]"),
        ("omega", "06_migliori_transfer_occorrenze_eps_omega.png",
         r"$\omega$ al perigeo, catalogo [deg]"),
    ):
        geometry = plots.prepare_parking_geometry(members, stats, plots.ONLY_BOUND_PARKING,
                                                 plots.N_BINS, variable)
        data = plots.best_transfer_occurrences(geometry, winners, len(stats), plots.BEST_OCCURRENCE_MODE)
        items.append((data, filename, plots.EPS_LABEL, ylabel))
    items += plots.additional_best_transfer_maps(
        members, stats, winners, catalog.with_name("trajectory_exit_catalog.txt"),
        plots.ONLY_BOUND_PARKING, plots.N_BINS, plots.BEST_OCCURRENCE_MODE)
    output = plots.OUTPUT_DIR / "mappe_percentili"
    plots.save_occurrence_percentiles(items, output)
    print(f"Salvate sei mappe e PDF in {output}", flush=True)


if __name__ == "__main__":
    main()
