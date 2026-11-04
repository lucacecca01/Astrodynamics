"""Exact 2-D radius-graph components without enumerating dense neighborhoods.

Cells have side r/2, hence their members form cliques. Only links between
occupied cells require a search. A KD-tree nearest-neighbor query proposes
cross-cell links, which are confirmed with the same radius predicate used by
the original GA16 BFS. Binning only accelerates edge discovery; it does not
replace distances or change the connectivity radius.
"""

import hashlib

import numpy as np
from scipy.spatial import cKDTree


_FORWARD_OFFSETS = ((0, 1), (0, 2), (0, 3)) + tuple(
    (dx, dy) for dx in (1, 2, 3) for dy in range(-3, 4)
)
_ROUNDING_MARGIN = 32 * np.finfo(float).eps


def _canonical_labels(roots):
    """Number components in order of their first original (sorted-ID) member."""
    _, first, inverse = np.unique(roots, return_index=True, return_inverse=True)
    if len(first) == 1:
        return None
    rank = np.empty(len(first), dtype=np.int32)
    rank[np.argsort(first)] = np.arange(len(first), dtype=np.int32)
    return rank[inverse]


def _reference_components(values, radius):
    """Conservative fallback for floating-point scales unsafe for grid indices."""
    tree = cKDTree(values)
    labels = np.full(len(values), -1, dtype=np.int32)
    component = 0
    for seed in range(len(values)):
        if labels[seed] >= 0:
            continue
        labels[seed] = component
        pending = [seed]
        while pending:
            node = pending.pop()
            for neighbor in tree.query_ball_point(tree.data[node], radius, eps=0.):
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    pending.append(neighbor)
        component += 1
    return None if component == 1 else labels


def _grid_components(values, radius):
    """Return exact components of a finite (N,2) sample in its original order."""
    if len(values) < 2:
        return None
    side = radius / 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = (values - values.min(axis=0)) / side
    # The guard leaves ample floating-point resolution for r/2 cells and the
    # +/-3 candidate stencil. Exotic scales retain the original exact BFS.
    if (not side or not np.isfinite(scaled).all()
            or np.max(scaled) >= 2**45 or not np.isfinite(radius * (1 + _ROUNDING_MARGIN))):
        return _reference_components(values, radius)
    cells = np.floor(scaled).astype(np.int64)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_cells, axis=0), axis=1)) + 1]
    count = len(starts)
    if count == 1:
        return None
    stops = np.r_[starts[1:], len(values)]
    coordinates = sorted_cells[starts]
    sorted_values = np.ascontiguousarray(values[order])
    minima = np.minimum.reduceat(sorted_values, starts, axis=0)
    maxima = np.maximum.reduceat(sorted_values, starts, axis=0)
    cell_sizes = stops - starts
    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(coordinates)}
    parents = np.arange(count, dtype=np.int64)
    sizes = cell_sizes.copy()
    trees = {}
    number_components = count
    radius_out = radius * (1 + _ROUNDING_MARGIN)
    radius_in = radius * (1 - _ROUNDING_MARGIN)

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(a, b):
        nonlocal number_components
        a, b = root(a), root(b)
        if a == b:
            return
        if sizes[a] < sizes[b]:
            a, b = b, a
        parents[b] = a
        sizes[a] += sizes[b]
        number_components -= 1

    def linked(a, b):
        # Actual point bounding boxes, rather than cell boundaries, tighten the
        # tests. Conservative margins defer boundary cases to the KD-tree.
        gaps = np.maximum(np.maximum(minima[a] - maxima[b], minima[b] - maxima[a]), 0)
        if np.hypot(gaps[0], gaps[1]) > radius_out:
            return False
        far = np.maximum(np.abs(maxima[a] - minima[b]), np.abs(maxima[b] - minima[a]))
        if np.hypot(far[0], far[1]) < radius_in:
            return True
        if cell_sizes[a] > cell_sizes[b]:
            a, b = b, a
        if b not in trees:
            trees[b] = cKDTree(sorted_values[starts[b]:stops[b]])
        tree = trees[b]
        queries = sorted_values[starts[a]:stops[a]]
        for begin in range(0, len(queries), 128):
            block = queries[begin:begin + 128]
            distances, _ = tree.query(block, k=1, distance_upper_bound=radius_out)
            candidates = np.flatnonzero(np.isfinite(distances))
            if not len(candidates):
                continue
            closest = candidates[np.argmin(distances[candidates])]
            if tree.query_ball_point(block[closest], radius, eps=0., return_length=True):
                return True
            # A proposed nearest distance can be a rounding-near-boundary
            # false positive; inspect all such candidates before rejecting.
            if np.any(tree.query_ball_point(block[candidates], radius, eps=0., return_length=True)):
                return True
        return False

    for a, (x, y) in enumerate(coordinates):
        for dx, dy in _FORWARD_OFFSETS:
            b = lookup.get((int(x) + dx, int(y) + dy))
            if b is None or root(a) == root(b):
                continue
            if linked(a, b):
                join(a, b)
                if number_components == 1:
                    return None
    cell_roots = np.array([root(i) for i in range(count)], dtype=np.int64)
    roots_sorted = np.repeat(cell_roots, cell_sizes)
    roots = np.empty(len(values), dtype=np.int64)
    roots[order] = roots_sorted
    return _canonical_labels(roots)


class ConnectivityShapeConstraint:
    """The original eps/e connectivity interface, with exact grid acceleration."""

    def __init__(self, features, radius):
        self.features = np.asarray(features, dtype=float)
        if (self.features.ndim != 2 or self.features.shape[1] != 2
                or not np.isfinite(self.features).all()):
            raise ValueError("Expected finite normalized epsilon/eccentricity features, shape (N, 2).")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("The connectivity radius must be finite and positive.")
        self.radius = float(radius)
        self.cache = {}

    def _components(self, ids):
        ids = np.asarray(ids)
        if ids.ndim == 1 and not len(ids):
            return None
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Expected a one-dimensional array of integer member indices.")
        ids = ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= len(self.features)):
            raise ValueError("Member index outside the feature catalog.")
        if len(ids) < 2:
            return None
        ordered = np.sort(ids)
        key = hashlib.sha256(ordered.tobytes()).digest()
        if key not in self.cache:
            labels = _grid_components(self.features[ordered], self.radius)
            if len(self.cache) >= 4096:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = labels
        labels = self.cache[key]
        return None if labels is None else labels[np.searchsorted(ordered, ids)]

    def __call__(self, ids):
        return bool(len(ids) > 0 and self._components(ids) is None)

    def split(self, ids):
        return self._components(ids)
