"""Read-only orbit view of every saved catalog row, reusing existing archives.

K83 supplies its assigned trajectories; the nineteen additional catalog rows
already exist in K798. Mapping small integer indices avoids copying the
multi-gigabyte orbit buffers and never launches an integration.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVES = (
    ROOT / ".cache/ga16_plot_recovery/GA_16_plots_eps_e_K83/orbits",
    ROOT / ".cache/ga16_plot_recovery/GA_16_plots_vinf_time_K798/orbits",
)


class IndexedReadOnlyArray:
    """Gather requested leading rows from existing memory maps on demand.

    Supports scalar, slice and advanced leading-row selections, followed by
    ordinary component/sample slices or integers. No full buffer is made when
    constructing the view. Returned arrays are read-only, too.
    """

    def __init__(self, arrays, owners, rows):
        self._arrays = tuple(arrays)
        self._owners = owners
        self._rows = rows
        self.shape = (len(rows), *arrays[0].shape[1:])
        self.dtype = arrays[0].dtype
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if not key:
            key = (slice(None),)
        if key[0] is Ellipsis:
            key = (slice(None),) * (self.ndim-len(key)+1) + key[1:]
        leading, tail = key[0], key[1:]
        if any(not isinstance(item, (int, np.integer, slice)) for item in tail):
            raise IndexError("Only ordinary component/sample slices are supported after the row index.")
        mapped = np.asarray(self._rows[leading])
        owners = np.asarray(self._owners[leading])
        flat_rows = mapped.reshape(-1)
        flat_owners = owners.reshape(-1)
        result = np.empty((len(flat_rows), *self.shape[1:]), dtype=self.dtype)
        for number, archive in enumerate(self._arrays):
            positions = np.flatnonzero(flat_owners == number)
            if len(positions):
                result[positions] = archive[flat_rows[positions]]
        result = result.reshape((*mapped.shape, *self.shape[1:]))
        if tail:
            result = result[(slice(None),)*mapped.ndim + tail]
        if isinstance(result, np.ndarray):
            result.setflags(write=False)
        return result

    def __setitem__(self, key, value):
        raise TypeError("The catalog orbit view is read-only.")

    def __array__(self, dtype=None, copy=None):
        raise TypeError("Select orbit rows explicitly instead of materializing the complete archive.")


@dataclass
class CatalogOrbitView:
    directory: Path
    source_ids: np.ndarray
    curves: IndexedReadOnlyArray
    bounds: IndexedReadOnlyArray
    features: IndexedReadOnlyArray
    done: np.ndarray
    context: dict
    reuse_report: dict

    def indices(self, source_ids):
        requested = np.asarray(source_ids, dtype=np.int64)
        indices = np.searchsorted(self.source_ids, requested)
        if (np.any(indices >= len(self.source_ids))
                or not np.array_equal(self.source_ids[indices], requested)):
            raise ValueError("Requested source IDs are absent from the catalog orbit view.")
        return indices

    def flush(self):
        raise TypeError("The catalog orbit view is read-only; there is nothing to flush.")


def _open_archive(directory, context):
    from ga16_recover_orbit_plots import _array_digest

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["source_fingerprint"] != context["recovery_fingerprint"]:
        raise ValueError(f"Archive integration dynamics differ: {directory}")
    arrays = {key: np.load(directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
              for key in ("source_ids", "curves", "bounds", "features", "done")}
    ids = arrays["source_ids"]
    count = len(ids)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or np.any(np.diff(ids) <= 0) or metadata["count"] != count
            or _array_digest(ids) != metadata["source_ids_sha256"]
            or arrays["curves"].shape != (count, 2, 2*context["N_PLOT"]-1)
            or arrays["bounds"].shape != (count, 2)
            or arrays["features"].shape != (count, 7)
            or arrays["done"].shape != (count,) or arrays["done"].dtype != np.bool_):
        raise ValueError(f"Invalid orbit archive structure or source IDs: {directory}")
    for key in ("curves", "bounds", "features"):
        if arrays[key].dtype != np.float64:
            raise ValueError(f"Expected float64 saved {key}: {directory}")
    return directory, arrays


def _validate_view(catalog, selected_rows, cache):
    context = cache.context
    for first in range(0, len(selected_rows), 32768):
        last = min(first+32768, len(selected_rows))
        expected = catalog[selected_rows[first:last]]
        block = slice(first, last)
        features, bounds, curves = cache.features[block], cache.bounds[block], cache.curves[block]
        days = (bounds-context["T_min"])*context["TU"]/context["day"]
        x, y = curves[:, 0, -1]+context["mu"], curves[:, 1, -1]
        cosine, sine = np.cos(bounds[:, 1]), np.sin(bounds[:, 1])
        positions = context["d"]*np.column_stack((cosine*x-sine*y, sine*x+cosine*y))
        if (not np.isfinite(curves).all() or not np.isfinite(features).all()
                or not np.isfinite(bounds).all() or not np.isfinite(expected[:, 2:]).all()
                or not np.allclose(days, expected[:, 5:7], rtol=1e-9, atol=1e-8)
                or not np.allclose(features[:, :2], expected[:, 9:11], rtol=1e-9, atol=1e-8)
                or not np.allclose(features[:, 3:6], expected[:, 2:5], rtol=1e-9, atol=1e-8)
                or not np.allclose((features[:, 2]-expected[:, 11]+180)%360-180, 0, atol=1e-7)
                or not np.allclose(positions, expected[:, 7:9], rtol=1e-9, atol=1e-4)):
            raise ValueError(f"Archived orbit events disagree with catalog rows {first}:{last}.")


def open_catalog_orbit_view(catalog, source_script, *, archives=None, selected_rows=None):
    """Open all catalog orbits (or a selected subset) without reintegration.

    Archives are tried in order, by exact source ID, and every reused curve is
    validated against catalog event times, SOI position/velocity and perigee
    elements. Missing or unready rows raise an error; no integration fallback
    exists. Leading indices of the returned view follow ``selected_rows``.
    """
    from ga16_recover_orbit_plots import load_original_context

    catalog = np.asarray(catalog)
    if catalog.ndim != 2 or catalog.shape[1] != 12 or not len(catalog):
        raise ValueError("Expected the nonempty 12-column GA16 exit catalog.")
    if selected_rows is None:
        selected_rows = np.arange(len(catalog), dtype=np.int64)
    else:
        selected_rows = np.asarray(selected_rows)
        if (selected_rows.ndim != 1 or not len(selected_rows)
                or not np.issubdtype(selected_rows.dtype, np.integer)
                or np.any(selected_rows < 0) or np.any(selected_rows >= len(catalog))
                or np.any(np.diff(selected_rows) <= 0)):
            raise ValueError("Selected catalog rows must be sorted, unique, nonempty and in bounds.")
    raw_ids = catalog[selected_rows, 0]
    if (not np.isfinite(raw_ids).all() or np.any(raw_ids < 0) or np.any(raw_ids >= 2**53)
            or np.any(raw_ids != np.floor(raw_ids)) or np.any(np.diff(raw_ids) <= 0)):
        raise ValueError("Catalog source IDs must be nonnegative, sorted, unique exact integers.")
    source_ids = raw_ids.astype(np.int64)
    context = load_original_context(source_script)
    opened = [_open_archive(path, context) for path in (DEFAULT_ARCHIVES if archives is None else archives)]
    if not opened:
        raise ValueError("At least one existing orbit archive is required.")
    owners = np.full(len(source_ids), -1, dtype=np.int16)
    mapped_rows = np.full(len(source_ids), -1, dtype=np.int64)
    for number, (_, arrays) in enumerate(opened):
        pending = np.flatnonzero(owners < 0)
        requested = source_ids[pending]
        archived_ids = arrays["source_ids"]
        matches = np.searchsorted(archived_ids, requested)
        present = matches < len(archived_ids)
        present[present] &= archived_ids[matches[present]] == requested[present]
        present[present] &= arrays["done"][matches[present]]
        found = pending[present]
        owners[found] = number
        mapped_rows[found] = matches[present]
    if np.any(owners < 0):
        missing = source_ids[owners < 0]
        raise ValueError(f"Missing ready archived curves for {len(missing)} catalog rows; "
                         f"first IDs: {missing[:20].tolist()}. No integration was launched.")
    source_ids.setflags(write=False)
    owners.setflags(write=False)
    mapped_rows.setflags(write=False)
    done = np.ones(len(source_ids), dtype=bool)
    done.setflags(write=False)
    buffers = {key: IndexedReadOnlyArray([arrays[key] for _, arrays in opened], owners, mapped_rows)
               for key in ("curves", "bounds", "features")}
    reuse = dict(archived_orbits=len(source_ids), integration_calls=0,
                 archives=[dict(path=str(path), reused=int(np.count_nonzero(owners == i)))
                           for i, (path, _) in enumerate(opened)])
    cache = CatalogOrbitView(opened[0][0], source_ids, context=context, done=done,
                            reuse_report=reuse, **buffers)
    _validate_view(catalog, selected_rows, cache)
    print(f"Orbite archiviate validate: {len(source_ids):,}; integrazioni: 0; "
          + ", ".join(f"{item['reused']:,} da {Path(item['path']).parent.name}" for item in reuse["archives"]),
          flush=True)
    return cache
