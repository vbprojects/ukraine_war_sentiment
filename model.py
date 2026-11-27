"""Profile the discrete onset and half-life of a piecewise exponential burst."""

import numpy as np
from scipy.optimize import nnls

HALF_LIVES = np.geomspace(0.5, 72, 16)


def fit(counts):
    y = np.asarray(counts, dtype=float)
    t = np.arange(len(y))
    best_error = np.inf
    # Three nonnegative coefficients enforce baseline_after >= baseline_before.
    for onset in range(6, len(y) - 6):
        post = t >= onset
        age = np.maximum(t - onset, 0)
        for half_life in HALF_LIVES:
            pulse = post * np.exp2(-age / half_life)
            design = np.column_stack((np.ones(len(t)), post, pulse))
            coef, error = nnls(design, y)
            if error < best_error:
                best_error = error
                baseline, increase, amplitude = coef
                prediction = design @ coef
                result = {
                    "onset_hour": onset,
                    "pre_baseline": baseline,
                    "post_baseline": baseline + increase,
                    "amplitude": amplitude,
                    "half_life_hours": half_life,
                    "decay": np.log(2) / half_life,
                    "residual_sd": error / np.sqrt(len(t)),
                }
    spread = np.sum((y - y.mean()) ** 2)
    result["r2"] = 1 - best_error**2 / spread if spread else 0
    return result, prediction
