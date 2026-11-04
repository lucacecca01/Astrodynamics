"""Additional population-STD bounds on individual osculating a and rp."""

import numpy as np


class OrbitSizeConstraint:
    def __init__(self, sizes, std_a_km=10000., std_rp_km=10000.):
        self.sizes = np.asarray(sizes, dtype=float)
        self.limits = np.array([std_a_km, std_rp_km], dtype=float)
        if (self.sizes.ndim != 2 or self.sizes.shape[1] != 2
                or not np.isfinite(self.sizes).all()
                or not np.isfinite(self.limits).all() or np.any(self.limits <= 0)):
            raise ValueError("Servono a/rp finiti e due limiti STD positivi in km.")

    def __call__(self, ids):
        return bool(len(ids) and np.all(self.sizes[ids].std(axis=0) <= self.limits))

    def split(self, ids):
        """Exact two-means split along the most violated size coordinate.

        Sorting and prefix moments evaluate every distinct 1-D cut. Values
        are centered and scaled before accumulation. This is only a proposal:
        all physical, radial, direction and connectivity checks still apply.
        """
        values = self.sizes[ids]
        ratios = values.std(axis=0)/self.limits
        dimension = int(np.argmax(ratios))
        if ratios[dimension] <= 1:
            return None
        values = values[:, dimension]
        order = np.argsort(values, kind="stable")
        ordered = values[order]
        cuts = np.flatnonzero(np.diff(ordered) > 0)+1
        if not len(cuts):
            return None
        # Scaling by range bounds all squared moments even near epsilon=0.
        ordered = (ordered-ordered[0])/(ordered[-1]-ordered[0])
        sums = np.cumsum(ordered)
        squares = np.cumsum(ordered*ordered)
        left = squares[cuts-1]-sums[cuts-1]**2/cuts
        right_n = len(values)-cuts
        right = (squares[-1]-squares[cuts-1]
                 -(sums[-1]-sums[cuts-1])**2/right_n)
        cut = int(cuts[np.argmin(left+right)])
        labels = np.ones(len(ids), dtype=np.int32)
        labels[order[:cut]] = 0
        return labels


class OrbitSizeGuidedConnectivity:
    """Retain the original eps/e connectivity, guide invalid size splits."""

    def __init__(self, connectivity, sizes):
        self.connectivity = connectivity
        self.sizes = sizes

    def __call__(self, ids):
        return self.connectivity(ids)

    def split(self, ids):
        labels = self.connectivity.split(ids)
        return self.sizes.split(ids) if labels is None else labels


def augmented_validator_factory(original_factory, size_constraint):
    """Add size checks before the unchanged original validity predicate.

    Early rejection saves time for the many candidate unions whose a/rp
    dispersion already exceeds a limit. Every accepted group is checked by
    the original validator as well.
    """
    def factory(*args, **kwargs):
        original = original_factory(*args, **kwargs)

        def valid(ids):
            return size_constraint(ids) and original(ids)

        return valid

    return factory
