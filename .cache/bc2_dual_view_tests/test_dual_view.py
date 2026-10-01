import ast
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "CR3BP/Ballistic Captures/Clustering/BC_clustering_2.py"
OUT = Path(__file__).resolve().parent
TREE = ast.parse(SOURCE.read_text())
compile(TREE, str(SOURCE), "exec")
BLOCK = next(node for node in TREE.body if isinstance(node, ast.If)
             and ast.unparse(node.test) == "PLOT_CLUSTERS or SAVE")
SAVE_FIGURE = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                   and node.name == "save_figure")


def plot_fixture(n_clusters, *, save=True, show=False, unassigned=False, render=False):
    ids = np.arange(n_clusters) * 3 + 7
    labels = np.repeat(ids, 2)
    if unassigned:
        labels = np.r_[labels, -1]
    t = np.linspace(0, 2*np.pi, 100)
    trajectories = np.zeros((len(labels), 6, len(t)))
    for i in range(len(labels)):
        trajectories[i, 0] = -.2 + (1.2+.002*i)*np.cos(t)
        trajectories[i, 1] = (1.2+.002*i)*np.sin(t)
    namespace = dict(np=np, plt=plt, SAVE=save, PLOT_CLUSTERS=show, labels=labels,
                     unique_clusters=ids, n_clusters=n_clusters, X_curvature=trajectories,
                     mu=.01215, unassigned_count=int(unassigned),
                     unassigned_percent=100/max(1, len(labels)) if unassigned else 0,
                     output_directory=OUT)
    exec(compile(ast.Module(body=[SAVE_FIGURE], type_ignores=[]), str(SOURCE), "exec"), namespace)
    save_figure = namespace["save_figure"]
    snapshots = {}

    def capture(fig, name):
        fig.canvas.draw()
        snapshots[name] = [dict(title=ax.get_title(), xlim=ax.get_xlim(), ylim=ax.get_ylim(),
                               lines=[line.get_xydata().copy() for line in ax.lines],
                               colors=[line.get_color() for line in ax.lines], active=ax.axison)
                           for ax in fig.axes]
        if render:
            save_figure(fig, name)
        elif not show:
            plt.close(fig)

    namespace["save_figure"] = capture
    exec(compile(ast.Module(body=[BLOCK], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return snapshots, labels, trajectories, ids


class DualViewTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_both_views_same_trajectories_colors_and_ids(self):
        snapshots, _, _, ids = plot_fixture(3)
        self.assertEqual(set(snapshots), {"clusters_full_001.png", "clusters_zoom_001.png"})
        for cid, full, zoom in zip(ids, snapshots["clusters_full_001.png"], snapshots["clusters_zoom_001.png"]):
            self.assertEqual(full["title"], f"Cluster {cid} || 2 trajectories")
            self.assertEqual(full["title"], zoom["title"])
            self.assertEqual(full["colors"], zoom["colors"])
            np.testing.assert_equal(full["lines"], zoom["lines"])

    def test_full_contains_all_points_and_zoom_uses_lunar_bounds(self):
        snapshots, _, _, _ = plot_fixture(1)
        full = snapshots["clusters_full_001.png"][0]
        zoom = snapshots["clusters_zoom_001.png"][0]
        points = np.vstack(full["lines"])
        for j, limits in enumerate((full["xlim"], full["ylim"])):
            self.assertLessEqual(limits[0], points[:, j].min())
            self.assertGreaterEqual(limits[1], points[:, j].max())
        np.testing.assert_allclose(zoom["xlim"], [1-.01215-.25, 1-.01215+.25])
        np.testing.assert_allclose(zoom["ylim"], [-.25, .25])
        self.assertGreater(full["xlim"][1]-full["xlim"][0], .5)

    def test_multiple_pages_and_unassigned(self):
        snapshots, _, _, _ = plot_fixture(25, unassigned=True)
        self.assertEqual(set(snapshots), {"unassigned_trajectories.png", "clusters_full_001.png",
                                         "clusters_full_002.png", "clusters_zoom_001.png",
                                         "clusters_zoom_002.png"})
        for view in ("full", "zoom"):
            self.assertEqual(len(snapshots[f"clusters_{view}_001.png"]), 24)
            self.assertEqual(len(snapshots[f"clusters_{view}_002.png"]), 1)

    def test_no_clusters(self):
        snapshots, _, _, _ = plot_fixture(0, unassigned=True)
        self.assertEqual(set(snapshots), {"unassigned_trajectories.png"})
        snapshots, _, _, _ = plot_fixture(0)
        self.assertEqual(snapshots, {})

    def test_plot_flags(self):
        snapshots, _, _, _ = plot_fixture(1, save=False, show=False)
        self.assertEqual(snapshots, {})
        snapshots, _, _, _ = plot_fixture(1, save=False, show=True)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(len(plt.get_fignums()), 2)

    def test_actual_png_outputs_and_closed_figures(self):
        snapshots, _, _, _ = plot_fixture(2, render=True)
        for name in snapshots:
            self.assertTrue((OUT/name).read_bytes().startswith(b"\x89PNG"))
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
