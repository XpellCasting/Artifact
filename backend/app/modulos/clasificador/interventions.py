"""Value operations without label access or name-derived physical assumptions."""
import math


def altered_value(kind, before, *, amount=1., progress=1., initial=None,
                  categories=(), rng=None, limits=(None, None), shape="linear"):
    if kind == "unavailable":
        return "unavailable"
    if kind == "stuck":
        return initial
    if kind == "state_replace":
        return next((v for v in sorted(categories) if v != before), before)
    if kind == "noise":
        delta = rng.uniform(-abs(amount)*1.732, abs(amount)*1.732) if shape == "uniform" else rng.gauss(0, abs(amount))
    elif kind == "drift":
        delta = amount * (progress**2 if shape == "quadratic" else progress)
    elif kind == "environmental_anomaly":
        delta = amount*(.25+.75*math.sin(math.pi*progress))
    else:
        delta = amount
    value = float(before)+delta
    lo, hi = limits
    value = max(lo, value) if lo is not None else value
    value = min(hi, value) if hi is not None else value
    if not math.isfinite(value):
        raise ValueError("Intervención no finita")
    return repr(value) if value != float(before) else before
