"""Canonical daily log-return units; jump sizes do not annualize."""
import numpy as np

NAMES = ["mu_log", "sigma", "lambda", "mu_j", "sigma_j"]


def validate_parameters(parameters):
    p = np.asarray(parameters, dtype=float)
    if p.shape[-1] != 5 or not np.isfinite(p).all():
        raise ValueError("Parameters must be finite [..., 5] arrays.")
    if np.any(p[..., 1] <= 0) or np.any(p[..., [2, 4]] < 0):
        raise ValueError("Require sigma > 0, lambda >= 0, sigma_j >= 0.")
    return p


def total_variance(parameters):
    p = validate_parameters(parameters)
    return p[..., 1] ** 2 + p[..., 2] * (p[..., 3] ** 2 + p[..., 4] ** 2)


def total_volatility(parameters, annualization=252):
    if annualization <= 0:
        raise ValueError("Annualization must be positive.")
    return np.sqrt(annualization * total_variance(parameters))


def annual_parameters(parameters, annualization=252):
    p = validate_parameters(parameters).copy()
    p[..., 0] *= annualization
    p[..., 1] *= np.sqrt(annualization)
    p[..., 2] *= annualization
    return p


def realized_volatility(returns, annualization=252, axis=-1):
    """Demeaned sample SD (ddof=1), annualized; not root mean square."""
    r = np.asarray(returns, dtype=float)
    if r.shape[axis] < 2 or not np.isfinite(r).all():
        raise ValueError("Realized volatility needs at least two finite returns.")
    return np.std(r, ddof=1, axis=axis) * np.sqrt(annualization)


class ParameterTransform:
    """Log diffusion sigma; square-root jump rate/SD; signed means.

    Squaring the jump coordinates is an explicit nonnegative output link,
    including zero, not clipping. Negative latent predictions are folded.
    The transform round-trips valid labels; the link is many-to-one.
    """
    def transform(self, parameters):
        z = validate_parameters(parameters).copy()
        z[..., 1] = np.log(z[..., 1])
        z[..., [2, 4]] = np.sqrt(z[..., [2, 4]])
        return z

    def inverse_transform(self, latent):
        z = np.asarray(latent, dtype=float).copy()
        with np.errstate(over="raise", invalid="raise"):
            z[..., 1] = np.exp(z[..., 1])
            z[..., [2, 4]] = z[..., [2, 4]] ** 2
        return validate_parameters(z)
