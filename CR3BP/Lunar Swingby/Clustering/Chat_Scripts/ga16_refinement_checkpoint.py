"""Save reusable candidate/fusion stages for long isolated GA16 trials."""

import json
from pathlib import Path

import numpy as np


def _save(directory, fingerprint, pool, baseline, *, labels=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    offsets = np.r_[0, np.cumsum([len(ids) for ids in pool], dtype=np.int64)]
    pending = directory / "members.tmp.npy"
    members = np.lib.format.open_memmap(pending, mode="w+", dtype=np.int64, shape=(int(offsets[-1]),))
    for ids, first, last in zip(pool, offsets[:-1], offsets[1:]):
        members[first:last] = ids
    members.flush()
    del members
    pending.replace(directory / "members.npy")
    values = dict(offsets=offsets, baseline=baseline)
    if labels is not None:
        values["labels"] = labels
    for name, array in values.items():
        pending = directory / f"{name}.tmp.npy"
        np.save(pending, array, allow_pickle=False)
        pending.replace(directory / f"{name}.npy")
    pending = directory / "metadata.tmp.json"
    pending.write_text(json.dumps(dict(fingerprint=fingerprint, candidates=len(pool),
                                        n_points=len(baseline), fused=labels is not None))+"\n")
    pending.replace(directory / "metadata.json")


def _load(directory, fingerprint):
    directory = Path(directory)
    marker = directory / "metadata.json"
    if not marker.exists():
        return None
    meta = json.loads(marker.read_text())
    if meta["fingerprint"] != fingerprint:
        raise ValueError(f"Il checkpoint {directory} usa dati o codice diversi.")
    members = np.load(directory / "members.npy", mmap_mode="r")
    offsets = np.load(directory / "offsets.npy")
    baseline = np.load(directory / "baseline.npy")
    if (len(offsets) != meta["candidates"]+1 or offsets[0] != 0 or offsets[-1] != len(members)
            or np.any(np.diff(offsets) <= 0) or baseline.shape != (meta["n_points"],)):
        raise ValueError("Checkpoint dei candidati non valido.")
    pool = list(np.split(members, offsets[1:-1]))
    labels = np.load(directory / "labels.npy") if meta["fused"] else None
    if labels is not None and labels.shape != baseline.shape:
        raise ValueError("Etichette del checkpoint non valide.")
    return pool, baseline, labels


def refine_with_checkpoint(helpers, directory, fingerprint, *args, **kwargs):
    """Run the original refinement, checkpoint immediately before/after fusion.

    Resume skips completed proposal generation and initial fusion, then runs
    exactly the original recovery/fusion closure with the current validators.
    No prior input or output is changed.
    """
    directory = Path(directory)
    restored = _load(directory / "fused", fingerprint)
    if restored is None:
        restored = _load(directory / "candidates", fingerprint)
    original_fuse = helpers["fuse_cluster_candidates"]

    def checkpointed_fuse(pool, n_points, valid_group, max_clusters, **options):
        baseline = options.get("previous_labels")
        if baseline is None:
            baseline = np.full(n_points, -1, dtype=np.int64)
        _save(directory / "candidates", fingerprint, pool, baseline)
        print(f"Checkpoint candidati: {len(pool):,} gruppi salvati.", flush=True)
        labels, updated = original_fuse(pool, n_points, valid_group, max_clusters, **options)
        _save(directory / "fused", fingerprint, updated, baseline, labels=labels)
        print("Checkpoint fusione salvato prima del recupero individuale.", flush=True)
        return labels, updated

    if restored is None:
        helpers["fuse_cluster_candidates"] = checkpointed_fuse
        try:
            return helpers["refine_and_fuse_clusters"](*args, **kwargs)
        finally:
            helpers["fuse_cluster_candidates"] = original_fuse

    _, physical, features, directions, limits, min_size, max_clusters, sigma_limit = args
    pool, baseline, labels = restored
    valid = helpers["make_cluster_validator"](
        physical, features, limits, min_size, sigma_limit, directions=directions,
        extra_condition=kwargs.get("extra_condition"), shape_condition=kwargs.get("shape_condition"),
    )
    if labels is None:
        labels, pool = checkpointed_fuse(
            pool, len(features), valid, max_clusters,
            previous_labels=baseline, time_limit=kwargs.get("time_limit", 30.),
        )
    else:
        if any(not valid(ids) for ids in helpers["cluster_groups"](labels)):
            raise ValueError("Un cluster del checkpoint non rispetta i vincoli.")
        print("Checkpoint della fusione riutilizzato.", flush=True)
    # This is the recovery closure of the production refinement, unchanged.
    seen = {ids.tobytes() for ids in pool}
    while True:
        labels, expanded = helpers["recover_unassigned_members"](
            labels, features, valid, directions=directions,
        )
        if not expanded:
            break
        for ids in expanded:
            if ids.tobytes() not in seen:
                seen.add(ids.tobytes())
                pool.append(ids)
        labels, unions = helpers["merge_valid_clusters"](labels, pool, valid)
        for ids in unions:
            if ids.tobytes() not in seen:
                seen.add(ids.tobytes())
                pool.append(ids)
        if not unions:
            break
    return labels, baseline, pool
