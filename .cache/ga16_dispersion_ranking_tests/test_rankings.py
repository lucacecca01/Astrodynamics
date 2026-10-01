import ast
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "CR3BP/Lunar Swingby/Clustering"
OUT = Path(__file__).resolve().parent


def load_plotter(filename):
    path = SOURCE / filename
    tree = ast.parse(path.read_text())
    compile(tree, str(path), "exec")
    keep = {"save_figure", "plot_dispersion_rankings", "normalize_block", "fusion_fingerprint"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in keep]
    import hashlib
    namespace = dict(np=np, plt=plt, PdfPages=PdfPages, nullcontext=nullcontext, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


SOURCES = ("CR3BP_prova_GA_16.py", "CR3BP_prova_GA_16_vinf.py")


class RankingTests(unittest.TestCase):
    def run_plot(self, filename, ids, sizes, values):
        namespace = load_plotter(filename)
        pages = []

        def capture(fig, name):
            pages.append(dict(name=name, title=fig._suptitle.get_text(), axes=[dict(
                ticks=[label.get_text() for label in ax.get_yticklabels()],
                widths=[bar.get_width() for bar in ax.patches],
                colors=[bar.get_facecolor() for bar in ax.patches],
                texts=[text.get_text() for text in ax.texts],
                xlim=ax.get_xlim(), ylim=ax.get_ylim(),
            ) for ax in fig.axes]))
            plt.close(fig)

        namespace.update(SAVE=False, PLOT_CLUSTERS=True, save_figure=capture)
        namespace["plot_dispersion_rankings"](
            ids, sizes, values, [f"Metric {i}" for i in range(values.shape[1])], "Test", "test")
        return pages

    def test_all_clusters_once_per_metric(self):
        for filename, metrics in zip(SOURCES, (2, 3)):
            n = 83
            ids, sizes = np.arange(n) * 3 + 7, np.arange(n) + 100
            values = np.random.default_rng(41).random((n, metrics))
            original = values.copy()
            pages = self.run_plot(filename, ids, sizes, values)
            self.assertEqual(len(pages), 3)
            for metric in range(metrics):
                observed = [tick for page in pages for tick in page["axes"][metric]["ticks"]]
                order = np.argsort(-values[:, metric], kind="stable")
                expected = [f"{ids[i]} | N={sizes[i]}" for i in order]
                self.assertEqual(observed, expected)
                limits = [page["axes"][metric]["xlim"] for page in pages]
                self.assertTrue(all(limit == limits[0] for limit in limits))
                self.assertTrue(all(page["axes"][metric]["ylim"][0] > page["axes"][metric]["ylim"][1]
                                    for page in pages))
            np.testing.assert_equal(values, original)

    def test_ties_undefined_and_zero(self):
        for filename in SOURCES:
            ids, sizes = np.array([2, 7, 19, 23, 99]), np.array([10, 20, 30, 40, 50])
            values = np.column_stack(([.1, np.nan, .4, np.inf, .4], np.zeros(5)))
            axes = self.run_plot(filename, ids, sizes, values)[0]["axes"]
            self.assertEqual(axes[0]["ticks"], ["19 | N=30", "99 | N=50", "2 | N=10", "7 | N=20", "23 | N=40"])
            self.assertEqual(axes[0]["texts"].count("N/A"), 2)
            self.assertEqual(axes[1]["widths"], [0] * 5)
            self.assertEqual(axes[1]["xlim"], (0, 1))
            palette = plt.get_cmap("turbo", 5)(np.arange(5))
            np.testing.assert_allclose(axes[0]["colors"], palette[[2, 4, 0]])
            np.testing.assert_allclose(axes[1]["colors"], palette)

    def test_empty_and_singleton(self):
        for filename in SOURCES:
            self.assertEqual(self.run_plot(filename, np.array([], int), np.array([], int), np.empty((0, 2))), [])
            pages = self.run_plot(filename, np.array([35]), np.array([1]), np.array([[0., np.nan]]))
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0]["axes"][0]["ticks"], ["35 | N=1"])
            self.assertEqual(pages[0]["axes"][1]["texts"], ["N/A"])

    def test_no_plot_when_disabled(self):
        for filename in SOURCES:
            namespace = load_plotter(filename)
            namespace.update(SAVE=False, PLOT_CLUSTERS=False)
            namespace["plot_dispersion_rankings"]([1], [10], [[.1]], ["Metric"], "Test", "test")
            self.assertEqual(plt.get_fignums(), [])

    def test_boundary_and_many_clusters(self):
        for n in (40, 41, 1003):
            pages = self.run_plot(SOURCES[0], np.arange(n), np.ones(n, int), np.zeros((n, 2)))
            self.assertEqual(len(pages), (n+39)//40)
            self.assertEqual(sum(len(p["axes"][0]["ticks"]) for p in pages), n)
            self.assertTrue(all(len(p["axes"][0]["ticks"]) <= 40 for p in pages))

    def test_actual_pdf_and_png_saving(self):
        for filename in SOURCES:
            namespace = load_plotter(filename)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                namespace.update(SAVE=True, PLOT_CLUSTERS=False, output_directory=directory)
                namespace["plot_dispersion_rankings"](
                    np.array([4, 17]), np.array([27, 85]), np.array([[0., np.nan], [.1, .5]]),
                    ("STD |v SOI| [km/s]", "Direction dispersion [0, 1]"), "Test", "ranks")
                self.assertTrue((directory/"ranks.pdf").read_bytes().startswith(b"%PDF"))
                self.assertEqual(len(list(directory.glob("ranks_*.png"))), 1)
                self.assertEqual(plt.get_fignums(), [])


def real_data_previews():
    with np.load(ROOT / ".cache/ga16_sigma_recovery_tests/features.npz") as data:
        events, source_ids = data["features"], data["source_ids"]
    database = np.loadtxt(ROOT / "CR3BP/Lunar Swingby/Generation/Escape_initial_conditions_CR3BP.txt", skiprows=1)
    for filename, cache_name, stem in (
        (SOURCES[0], "GA_16_dip_gap_candidates_sigma2_5.npz", "escape_dispersion_ranking"),
        (SOURCES[1], "GA_16_vxvy_candidates.npz", "orbital_dispersion_ranking"),
    ):
        namespace = load_plotter(filename)
        norm = namespace["normalize_block"]
        if filename == SOURCES[0]:
            # Historical OE partition with explicit row IDs; do not guess cache alignment.
            with np.load(ROOT / ".cache/ga16_std_min50_tests/increase_0/labels.npz") as data:
                np.testing.assert_array_equal(data["source_ids"], source_ids)
                labels = data["labels"]
        else:
            features = norm(events[:, 3:6])
            fingerprint = namespace["fusion_fingerprint"](source_ids, database[source_ids], events, features)
            with np.load(SOURCE / "Clusters" / cache_name) as data:
                assert str(data["fingerprint"]) == fingerprint, "Labels/features mismatch"
                labels = data["labels"]
        ids, sizes = np.unique(labels[labels >= 0], return_counts=True)
        values = []
        for cid in ids:
            data = events[labels == cid]
            if filename == SOURCES[0]:
                v = data[:, 3:6]
                speed = np.linalg.norm(v, axis=1)
                unit = v[speed > 0] / speed[speed > 0, None]
                spread = np.clip(1-np.linalg.norm(unit.mean(axis=0)), 0, 1) if len(unit) else np.nan
                values.append([speed.std(), spread])
            else:
                physical = data[:, :3].copy()
                angle_mean = np.mean(np.exp(1j * np.deg2rad(physical[:, 2])))
                physical[:, 2] = ((physical[:, 2]-np.rad2deg(np.angle(angle_mean))+180)%360-180
                                  if abs(angle_mean) >= 1e-8 else np.nan)
                values.append(physical.std(axis=0))
        metric_names = (("STD |v SOI| [km/s]", "Direction dispersion [0, 1]") if filename == SOURCES[0]
                        else ("STD eps [km^2/s^2]", "STD e [-]", "STD omega [deg]"))
        namespace.update(SAVE=True, PLOT_CLUSTERS=False, output_directory=OUT)
        namespace["plot_dispersion_rankings"](ids, sizes, values, metric_names,
                                             stem.replace("_", " "), stem)
        print(f"Real preview: {filename}, K={len(ids)}, pages={(len(ids)+39)//40}", flush=True)


if __name__ == "__main__":
    result = unittest.main(exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
    real_data_previews()
