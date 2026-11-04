"""Exact compressed bridge footprints for GA16's unchanged merge priorities.

Each initially assigned group is an indivisible supernode during a merge
call; each initially unassigned point has its own node. Candidate footprints
are computed once. Their current owners then require reading only those
supernodes, instead of all candidate point IDs on every union iteration.
"""

import ast
import copy
import hashlib
import inspect
import linecache
import textwrap

import numpy as np


class _BridgeFootprints:
    def __init__(self, initial_owner, candidates):
        owner = np.asarray(initial_owner)
        if owner.ndim != 1 or not np.issubdtype(owner.dtype, np.integer) or np.any(owner < -1):
            raise ValueError("Expected the original integer owner array.")
        assigned = owner >= 0
        group_count = int(owner[assigned].max())+1 if np.any(assigned) else 0
        if not np.array_equal(np.unique(owner[assigned]), np.arange(group_count)):
            raise ValueError("Initial merge owners must be contiguous group IDs.")
        unassigned = np.flatnonzero(~assigned)
        self.point_nodes = owner.astype(np.int64, copy=True)
        self.point_nodes[unassigned] = group_count+np.arange(len(unassigned), dtype=np.int64)
        self.roots = np.r_[np.arange(group_count, dtype=np.int64),
                           -1-np.arange(len(unassigned), dtype=np.int64)]
        self.footprints = []
        self.multiplicities = []
        raw_references = 0
        for candidate in candidates:
            candidate = np.asarray(candidate)
            if (candidate.ndim != 1 or not np.issubdtype(candidate.dtype, np.integer)
                    or np.any(candidate < 0) or np.any(candidate >= len(owner))):
                raise ValueError("Candidate member indices must be one-dimensional integers in bounds.")
            nodes, multiplicity = np.unique(self.point_nodes[candidate], return_counts=True)
            self.footprints.append(nodes)
            self.multiplicities.append(multiplicity)
            raw_references += len(candidate)
        self.statistics = dict(candidates=len(candidates), initial_groups=group_count,
                               initial_unassigned=len(unassigned), point_references=raw_references,
                               node_references=sum(len(nodes) for nodes in self.footprints),
                               signatures=0, updates=0)

    def signature(self, candidate_index):
        roots = self.roots[self.footprints[candidate_index]]
        inactive = roots < 0
        groups = tuple(np.unique(roots[~inactive]))
        # Normally every inactive node has multiplicity one. Keeping counts
        # also preserves the original score for any repeated candidate IDs.
        recovered = int(self.multiplicities[candidate_index][inactive].sum())
        self.statistics["signatures"] += 1
        return groups, recovered

    def advance(self, best_groups, best_candidate, next_id):
        self.roots[np.isin(self.roots, best_groups)] = next_id
        if best_candidate is not None:
            # Every active candidate owner is in best_groups by construction;
            # this also activates precisely its previously unassigned nodes.
            self.roots[self.footprints[best_candidate]] = next_id
        self.statistics["updates"] += 1


def make_bridge_compressed_merge(original_merge):
    """Clone the original merge function with exact compressed bridge lookup.

    Candidate membership arrays, full validity checks, cached signatures,
    iteration order and tie-breaking remain untouched. The original function
    and its globals are never changed. ``bridge_runs`` contains call counters.
    """
    parsed = ast.parse(textwrap.dedent(inspect.getsource(original_merge)))
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None or function.decorator_list:
        raise ValueError("Expected the original undecorated merge function.")
    function = copy.deepcopy(function)
    bridge_loops = [node for node in ast.walk(function)
                    if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                    and [getattr(item, "id", None) for item in node.target.elts] == ["j", "candidate"]]
    expected_iter = ast.parse("enumerate(candidates)", mode="eval").body
    expected_prefix = ast.parse(
        "memberships = owner[candidate]\n"
        "group_ids = tuple(np.unique(memberships[memberships >= 0]))\n"
        "recovered = int(np.count_nonzero(memberships < 0))\n"
    ).body
    if (len(bridge_loops) != 1
            or ast.dump(bridge_loops[0].iter) != ast.dump(expected_iter)
            or [ast.dump(node) for node in bridge_loops[0].body[:3]]
            != [ast.dump(node) for node in expected_prefix]):
        raise ValueError("Original bridge lookup changed; no accelerated clone produced.")
    bridge_loops[0].body[:3] = ast.parse(
        "group_ids, recovered = _ga16_bridge_index.signature(j)\n"
    ).body

    expected_init = ast.parse("next_id = initial_count\n").body[0]
    expected_update = ast.parse("owner[merged] = next_id\n").body[0]
    init = ast.parse("_ga16_bridge_index = _ga16_bridge_index_factory(owner, candidates)\n").body[0]
    update = ast.parse("_ga16_bridge_index.advance(best_groups, best_candidate, next_id)\n").body[0]

    class InsertIndex(ast.NodeTransformer):
        initializations = 0
        updates = 0

        def visit_Assign(self, node):
            if ast.dump(node) == ast.dump(expected_init):
                self.initializations += 1
                return [copy.deepcopy(init), node]
            if ast.dump(node) == ast.dump(expected_update):
                self.updates += 1
                return [node, copy.deepcopy(update)]
            return node

    injector = InsertIndex()
    function = injector.visit(function)
    if injector.initializations != 1 or injector.updates != 1:
        raise ValueError("Original owner initialization/update changed; clone refused.")
    runs = []

    def factory(owner, candidates):
        index = _BridgeFootprints(owner, candidates)
        runs.append(index.statistics)
        return index

    scope = dict(original_merge.__globals__)
    scope["_ga16_bridge_index_factory"] = factory
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    # Register readable source so the existing moment-prefilter factory can
    # inspect and compose this clone without modifying its frozen module.
    source = ast.unparse(module)+"\n"
    filename = "<ga16-bridge-merge-"+hashlib.sha256(source.encode()).hexdigest()[:16]+">"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), scope)
    accelerated = scope[function.name]
    accelerated.bridge_runs = runs
    return accelerated


def make_bridge_moment_prefiltered_merge(original_merge, observables, directions,
                                        std_limits, *, min_size=10,
                                        stop_after_first_valid_pair=True):
    """Compose compressed bridges with the existing frozen pair accelerator."""
    from ga16_size_merge_acceleration import make_moment_prefiltered_merge

    bridge = make_bridge_compressed_merge(original_merge)
    result = make_moment_prefiltered_merge(
        bridge, observables, directions, std_limits, min_size=min_size,
        stop_after_first_valid_pair=stop_after_first_valid_pair,
    )
    result.bridge_runs = bridge.bridge_runs
    return result
