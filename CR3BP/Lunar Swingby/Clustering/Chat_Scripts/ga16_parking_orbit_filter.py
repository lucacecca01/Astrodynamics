"""Select bound parking orbits whose semimajor axis fits Earth's model SOI.

These helpers read saved energies or constant definitions only. Importing the
module never runs the production script or integrates a trajectory.
"""

import ast
import operator
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().with_name("CR3BP_prova_GA_16_eps_e.py")
_CONSTANT_NAMES = {"G", "m1", "ms", "d_E", "E_SOI", "mu_earth"}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _constant_expression(node, values):
    """Evaluate numeric arithmetic and previously read names, never calls."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_constant_expression(node.left, values),
                                      _constant_expression(node.right, values))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_constant_expression(node.operand, values))
    raise ValueError("Model constants must use numeric arithmetic and preceding constants only.")


def _model_constants(source_script):
    values = {}
    for node in ast.parse(Path(source_script).read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in _CONSTANT_NAMES):
            try:
                values[node.targets[0].id] = _constant_expression(node.value, values)
            except (ArithmeticError, TypeError, ValueError) as error:
                raise ValueError("Cannot safely read the model's Earth constants.") from error
    return values


def _positive_scalar(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite positive real scalar.")
    value = np.asarray(value, dtype=float)
    if value.ndim != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar.")
    return float(value)


def model_earth_soi(source_script=SOURCE):
    """Return ``E_SOI`` in km from source constants, without executing source."""
    values = _model_constants(source_script)
    if "E_SOI" not in values:
        raise ValueError("Earth sphere of influence E_SOI is missing from the model.")
    return _positive_scalar(values["E_SOI"], "E_SOI")


def model_mu_earth(source_script=SOURCE):
    """Return ``mu_earth`` in km^3/s^2 without executing production code."""
    values = _model_constants(source_script)
    if "mu_earth" not in values:
        raise ValueError("Earth gravitational parameter mu_earth is missing from the model.")
    return _positive_scalar(values["mu_earth"], "mu_earth")


def parking_orbit_mask(eps, mu_earth, earth_soi):
    """Keep finite negative energies with finite ``0 < -mu/(2*eps) <= SOI``.

    Energies are in km^2/s^2; mu is in km^3/s^2 and the SOI radius is in km.
    A scalar returns bool; an array returns a boolean array of the same shape.
    Nonfinite, parabolic and hyperbolic energies return False. The semimajor
    axis formula is the floating-point decision authority, including equality.
    The mathematically equivalent threshold ``eps <= -mu/(2*SOI)`` must not
    replace it, since the two formulas can round differently at the boundary.
    """
    mu = _positive_scalar(mu_earth, "mu_earth")
    soi = _positive_scalar(earth_soi, "earth_soi")
    if np.iscomplexobj(eps):
        raise ValueError("Energies must be real scalars or arrays.")
    energy = np.asarray(eps, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        a = -mu/(2*energy)
    result = np.isfinite(energy) & (energy < 0) & np.isfinite(a) & (a > 0) & (a <= soi)
    return bool(result) if energy.ndim == 0 else result
