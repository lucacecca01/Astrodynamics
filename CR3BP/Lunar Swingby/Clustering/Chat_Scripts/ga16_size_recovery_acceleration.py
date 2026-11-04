"""Optional conservative STD screening for the unchanged GA16 recovery loop.

The factory clones the original function; importing this module changes no
production function or running experiment. Every candidate surviving the
screen still uses the original complete validator, including radial and graph
constraints. Distances, retries, insertion order and returned members are
preserved.
"""

import ast
import copy
import inspect
import textwrap

import numpy as np

from ga16_size_merge_acceleration import _FLOAT_EPS, _LONG_EPS, _PairMoments, _gamma


class _RecoveryMoments:
    """Keep versioned group summaries and screen all groups in one operation."""

    def __init__(self, observables, directions, limits, min_size):
        self.data = observables
        self.directions = directions
        self.limits = limits
        self.min_size = min_size
        # Relative float error bounds do not cover subnormal square underflow.
        # Tiny limits therefore always defer that coordinate to the validator.
        self.std_screen_dimensions = limits >= np.sqrt(np.longdouble(np.finfo(float).tiny))
        self.summaries = _PairMoments(observables, directions, limits, min_size)
        self.seen_versions = None
        self.statistics = dict(masks=0, groups_screened=0, moments=0,
                               rejected_direction=0, rejected_count=0,
                               rejected_std=0)
        # Extended arithmetic must accommodate squares of every finite float64
        # value. Otherwise omit the optional STD screen rather than risk an
        # overflow changing an admission decision.
        self.wide_arithmetic = (
            np.finfo(np.longdouble).maxexp >= 2*np.finfo(float).maxexp+8
        )

    def _refresh(self, groups, versions):
        if self.seen_versions is None:
            size = len(groups)
            dimensions = self.data.shape[1]
            self.seen_versions = np.full(size, -1, dtype=versions.dtype)
            self.counts = np.empty(size, dtype=np.longdouble)
            self.means = np.zeros((size, dimensions), dtype=np.longdouble)
            self.mean_errors = np.zeros_like(self.means)
            self.lower_m2 = np.zeros_like(self.means)
            self.signs = np.empty(size, dtype=self.directions.dtype)
        for j in np.flatnonzero(self.seen_versions != versions):
            if self.wide_arithmetic:
                self.summaries.cache.pop(j, None)
                n, mean, mean_error, lower_m2, sign = self.summaries._summary(j, groups)
                self.means[j] = mean
                self.mean_errors[j] = mean_error
                self.lower_m2[j] = lower_m2
            else:
                ids = groups[j]
                n = len(ids)
                signs = self.directions[ids]
                sign = signs[0] if np.all(signs == signs[0]) else 0
            self.counts[j] = n
            self.signs[j] = sign
            self.seen_versions[j] = versions[j]
            self.statistics["moments"] += 1

    def __call__(self, point, groups, versions):
        # A preceding point can have enlarged a group. Refresh before *each*
        # point, including retries on a later pass through the missing rows.
        self._refresh(groups, versions)
        totals = self.counts+1
        direction_ok = (self.signs == self.directions[point]) & (self.signs != 0)
        count_ok = totals >= self.min_size
        if self.wide_arithmetic:
            guard = _gamma(8, _LONG_EPS)
            # The point is an exactly represented float64 in longdouble. Its
            # mean interval therefore has zero width. Group intervals and
            # downward guards are shared with the merge accelerator.
            delta = np.maximum(
                (np.abs(self.means-self.data[point])*(1-guard)
                 - self.mean_errors)*(1-guard), 0,
            )
            between = delta**2*(self.counts/totals)[:, None]*(1-guard)
            lower_variance = ((self.lower_m2+between)/totals[:, None])*(1-guard)
            product = (totals+8)*_FLOAT_EPS
            # If the standard floating error bound is unavailable, the zero
            # lower bound sends that group to the original full validator.
            numpy_error = np.ones_like(product)
            usable = product < 1
            numpy_error[usable] = product[usable]/(1-product[usable])
            lower_std = np.sqrt(np.maximum(
                lower_variance*(1-numpy_error[:, None]), 0,
            ))
            lower_std *= 1-_gamma(8, _FLOAT_EPS)
            std_ok = ~np.any(
                (lower_std > self.limits) & self.std_screen_dimensions, axis=1,
            )
        else:
            std_ok = np.ones(len(groups), dtype=bool)
        self.statistics["masks"] += 1
        self.statistics["groups_screened"] += len(groups)
        self.statistics["rejected_direction"] += int(np.count_nonzero(~direction_ok))
        self.statistics["rejected_count"] += int(np.count_nonzero(~count_ok))
        self.statistics["rejected_std"] += int(np.count_nonzero(~std_ok))
        return direction_ok & count_ok & std_ok


def make_moment_prefiltered_recovery(original_recovery, observables, directions,
                                    std_limits, *, min_size=10):
    """Clone ``recover_unassigned_members`` with vectorized safe STD rejections.

    Supply the same finite float64 scalar observables and necessary population
    STD limits used by the validator, normally [eps, e, a, rp]. The first two
    limits can be strict and the last two inclusive: equality always proceeds
    to the full validator. Orbit directions and minimum size must also agree
    with that validator. ``prefilter_runs`` contains per-invocation counters;
    group-mask counters include groups after the eventual successful attempt.
    The supplied function and its globals are never modified.
    """
    observables = np.asarray(observables)
    directions = np.asarray(directions)
    limits = np.asarray(std_limits, dtype=np.longdouble)
    if (observables.dtype != np.dtype(float)
            or observables.ndim != 2 or not observables.shape[1]
            or not np.isfinite(observables).all()
            or limits.shape != (observables.shape[1],)
            or np.any(np.isnan(limits)) or np.any(limits <= 0)
            or directions.shape != (len(observables),)
            or not np.isin(directions, (-1, 1)).all()
            or not isinstance(min_size, (int, np.integer))
            or isinstance(min_size, (bool, np.bool_)) or min_size < 1):
        raise ValueError("Expected finite scalar observables, STD limits and signed orbit directions.")
    parsed = ast.parse(textwrap.dedent(inspect.getsource(original_recovery)))
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None or function.decorator_list:
        raise ValueError("Expected the original undecorated recovery function.")
    function = copy.deepcopy(function)
    loops = [node for node in ast.walk(function)
             if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
             and node.target.id == "j" and isinstance(node.iter, ast.Name)
             and node.iter.id == "order"]
    if len(loops) != 1:
        raise ValueError("Original ordered recovery loop not found; no accelerated clone produced.")
    loop = loops[0]
    expected_members = ast.parse("np.sort(np.r_[groups[j], point])", mode="eval").body
    members = [node for node in loop.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "members"
                       for target in node.targets)
               and ast.dump(node.value) == ast.dump(expected_members)]
    expected_skip = ast.parse(
        "if (failed_versions[row, j] == versions[j]\n"
        "        or (directions is not None and directions[point] != directions[groups[j][0]])):\n"
        "    continue\n"
    ).body[0]
    if (len(members) != 1 or len(loop.body) < 2
            or ast.dump(loop.body[0]) != ast.dump(expected_skip)
            or loop.body[1] is not members[0]):
        raise ValueError("Original member materialization and retry guard could not be verified.")
    guard = ast.parse(
        "if not _ga16_recovery_mask[j]:\n"
        "    failed_versions[row, j] = versions[j]\n"
        "    continue\n"
    ).body[0]
    loop.body.insert(1, guard)
    parents = [node for node in ast.walk(function)
               if any(isinstance(value, list) and loop in value
                      for _, value in ast.iter_fields(node))]
    if len(parents) != 1:
        raise ValueError("Original recovery loop parent could not be verified.")
    parent = parents[0]
    containing_body = next(value for _, value in ast.iter_fields(parent)
                           if isinstance(value, list) and loop in value)
    mask = ast.parse(
        "_ga16_recovery_mask = _ga16_recovery_filter(point, groups, versions)\n"
    ).body[0]
    containing_body.insert(containing_body.index(loop), mask)
    create = ast.parse("_ga16_recovery_filter = _ga16_recovery_filter_factory()\n").body[0]
    insertion = 1 if (function.body and isinstance(function.body[0], ast.Expr)
                      and isinstance(function.body[0].value, ast.Constant)
                      and isinstance(function.body[0].value.value, str)) else 0
    function.body.insert(insertion, create)
    runs = []

    def factory():
        result = _RecoveryMoments(observables, directions, limits, int(min_size))
        runs.append(result.statistics)
        return result

    scope = dict(original_recovery.__globals__)
    scope["_ga16_recovery_filter_factory"] = factory
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "<ga16-moment-prefiltered-recovery>", "exec"), scope)
    accelerated = scope[function.name]
    accelerated.prefilter_runs = runs
    return accelerated
