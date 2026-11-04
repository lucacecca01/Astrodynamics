"""Optional moment prefilter for the unchanged GA16 all-pairs merge algorithm.

This module is not imported by production or by a running experiment. The
factory clones the original merge function and inserts a conservative pair
test before materializing union member IDs. All surviving pairs still use
the original full validator; bridge candidates, ordering, scores and coverage
rules are unchanged.
"""

import ast
import copy
import inspect
import textwrap

import numpy as np


_LONG_EPS = np.finfo(np.longdouble).eps
_FLOAT_EPS = np.finfo(float).eps


def _gamma(n, epsilon):
    product = np.longdouble(n) * epsilon
    return np.inf if product >= 1 else product / (1-product)


class _PairMoments:
    def __init__(self, observables, directions, limits, min_size):
        self.data = observables
        self.directions = directions
        self.limits = limits
        self.min_size = min_size
        self.cache = {}
        self.statistics = dict(pairs=0, rejected_direction=0, rejected_count=0,
                               rejected_std=0, full_validator_pairs=0, moments=0)

    def _summary(self, group_id, groups):
        # The original algorithm assigns a fresh immutable ID to every union.
        if group_id not in self.cache:
            ids = groups[group_id]
            values = self.data[ids].astype(np.longdouble)
            n = len(values)
            mean = values.mean(axis=0)
            maximum = np.max(np.abs(values), axis=0)
            error = _gamma(n+8, _LONG_EPS)
            guard = _gamma(8, _LONG_EPS)
            mean_error = error * maximum * (1+guard)
            m2 = np.sum((values-mean)**2, axis=0)
            # Sum around an approximate center exceeds true central M2 by
            # n*(center-error)^2. Subtract its conservative bound, and allow
            # rounding in subtraction, square, reduction and division.
            lower_m2 = np.maximum(
                (m2/(1+error)-n*mean_error**2*(1+guard))*(1-guard), 0,
            )
            signs = self.directions[ids]
            direction = signs[0] if np.all(signs == signs[0]) else 0
            self.cache[group_id] = (n, mean, mean_error, lower_m2, direction)
            self.statistics["moments"] += 1
        return self.cache[group_id]

    def __call__(self, group_ids, groups):
        self.statistics["pairs"] += 1
        a = self._summary(group_ids[0], groups)
        b = self._summary(group_ids[1], groups)
        na, ma, ea, m2a, sign_a = a
        nb, mb, eb, m2b, sign_b = b
        if not sign_a or sign_a != sign_b:
            self.statistics["rejected_direction"] += 1
            return False
        n = na+nb
        if n < self.min_size:
            self.statistics["rejected_count"] += 1
            return False
        guard = _gamma(8, _LONG_EPS)
        numpy_error = _gamma(n+8, _FLOAT_EPS)
        if not np.isfinite(numpy_error):
            self.statistics["full_validator_pairs"] += 1
            return True
        # Parallel central moments: M2 = M2a + M2b + (ma-mb)^2*na*nb/n.
        # Mean intervals and downward guards make this a lower bound, not an
        # approximate rejection test. Near-boundary pairs go to the original
        # validator, retaining its strict eps/e and inclusive a/rp semantics.
        delta = np.maximum((np.abs(ma-mb)*(1-guard)-ea-eb)*(1-guard), 0)
        between = delta**2 * (np.longdouble(na)*nb/n) * (1-guard)
        lower_variance = ((m2a+m2b+between)/n)*(1-guard)
        # A finite NumPy population STD can underestimate exact variance only
        # through its floating arithmetic. This factor conservatively allows
        # its subtract/square/reduce/divide/sqrt roundoff as well.
        lower_numpy_std = np.sqrt(np.maximum(lower_variance*(1-numpy_error), 0))
        lower_numpy_std *= 1-_gamma(8, _FLOAT_EPS)
        if np.any(lower_numpy_std > self.limits):
            self.statistics["rejected_std"] += 1
            return False
        self.statistics["full_validator_pairs"] += 1
        return True


def make_moment_prefiltered_merge(original_merge, observables, directions,
                                 std_limits, *, min_size=10,
                                 stop_after_first_valid_pair=True):
    """Clone ``merge_valid_clusters`` with safe scalar-STD pair rejections.

    ``observables`` must contain float64 scalar columns [eps, e, a, rp] (or any
    finite float64 coordinates whose STD limits are part of the full validator).
    The original validator remains the decision authority for every candidate
    surviving this necessary-condition test. The supplied function and its
    globals are never modified. ``prefilter_runs`` exposes per-call counters.
    By default, stop the sorted pair scan at its first valid pair: all pair
    scores are (-1, 0, pair_ids, -1), so later pairs cannot improve that score.
    Every bridge candidate is still examined with the original priorities.
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
            or not isinstance(min_size, (int, np.integer)) or min_size < 1):
        raise ValueError("Expected finite scalar observables, STD limits and signed orbit directions.")
    parsed = ast.parse(textwrap.dedent(inspect.getsource(original_merge)))
    function = next((node for node in parsed.body if isinstance(node, ast.FunctionDef)), None)
    if function is None or function.decorator_list:
        raise ValueError("Expected the original undecorated merge function.")
    function = copy.deepcopy(function)
    target = None
    for node in ast.walk(function):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "group_ids"
                and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.NotIn)
                and isinstance(node.test.comparators[0], ast.Name)
                and node.test.comparators[0].id == "pair_validity"):
            if target is not None:
                raise ValueError("Ambiguous pair-validity branch in merge function.")
            target = node
    if target is None:
        raise ValueError("Original all-pairs validity branch not found; no accelerated clone produced.")
    if stop_after_first_valid_pair:
        pair_loops = [node for node in ast.walk(function)
                      if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                      and node.target.id == "group_ids"]
        expected_loop = ast.parse("combinations(sorted(groups), 2)", mode="eval").body
        expected_score = ast.parse("(-1, 0, group_ids, -1)", mode="eval").body
        valid_branches = [node for node in ast.walk(function)
                          if isinstance(node, ast.If) and isinstance(node.test, ast.Subscript)
                          and isinstance(node.test.value, ast.Name)
                          and node.test.value.id == "pair_validity"
                          and isinstance(node.test.slice, ast.Name)
                          and node.test.slice.id == "group_ids"]
        if (len(pair_loops) != 1 or len(valid_branches) != 1
                or ast.dump(pair_loops[0].iter) != ast.dump(expected_loop)):
            raise ValueError("Sorted GA16 pair order could not be verified for the early-stop optimization.")
        branch = valid_branches[0]
        score = next((node.value for node in branch.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "score" for t in node.targets)), None)
        if score is None or ast.dump(score) != ast.dump(expected_score):
            raise ValueError("GA16 pair-score priority changed; early-stop optimization refused.")
        branch.body.append(ast.Break())
    guard = ast.parse(
        "if _ga16_pair_filter(group_ids, groups):\n"
        "    pass\n"
        "else:\n"
        "    pair_validity[group_ids] = False\n"
    ).body[0]
    guard.body = target.body
    target.body = [guard]
    create = ast.parse("_ga16_pair_filter = _ga16_pair_filter_factory()\n").body[0]
    insertion = 1 if (function.body and isinstance(function.body[0], ast.Expr)
                      and isinstance(function.body[0].value, ast.Constant)
                      and isinstance(function.body[0].value.value, str)) else 0
    function.body.insert(insertion, create)
    runs = []

    def factory():
        result = _PairMoments(observables, directions, limits, int(min_size))
        runs.append(result.statistics)
        return result

    scope = dict(original_merge.__globals__)
    scope["_ga16_pair_filter_factory"] = factory
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "<ga16-moment-prefiltered-merge>", "exec"), scope)
    accelerated = scope[function.name]
    accelerated.prefilter_runs = runs
    accelerated.stop_after_first_valid_pair = bool(stop_after_first_valid_pair)
    return accelerated
