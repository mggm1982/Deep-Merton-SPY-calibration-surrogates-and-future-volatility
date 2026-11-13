"""Stable compound-Poisson Gaussian mixture; exact nested Gaussian candidate."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy, gammaln
from scipy.stats import poisson
from .units import validate_parameters


def mixture_terms(max_lambda, tolerance=1e-12):
    if max_lambda < 0 or not 0 < tolerance < 1:
        raise ValueError("Invalid Poisson truncation settings.")
    count = max(1, int(poisson.isf(tolerance, max_lambda)) + 1)
    while poisson.sf(count - 1, max_lambda) > tolerance:
        count += 1
    return count


def log_weights(lam, terms):
    if lam < 0 or terms < 1:
        raise ValueError("Invalid mixture.")
    k = np.arange(terms)
    return -lam + xlogy(k, lam) - gammaln(k + 1)


def logpdf(returns, parameters, tolerance=1e-12, terms=None):
    mu, sigma, lam, mj, sj = validate_parameters(parameters)
    r = np.asarray(returns, dtype=float)
    if not np.isfinite(r).all():
        raise ValueError("Nonfinite returns.")
    count = mixture_terms(lam, tolerance) if terms is None else terms
    if poisson.sf(count - 1, lam) > tolerance * 1.01:
        raise ValueError("Poisson truncation is too short.")
    k = np.arange(count)
    variance = sigma ** 2 + k * sj ** 2
    delta = r[..., None] - mu - k * mj
    return logsumexp(log_weights(lam, count) - 0.5 * (np.log(2 * np.pi * variance) + delta ** 2 / variance), axis=-1)


def nll(parameters, returns, tolerance=1e-12):
    return float(-np.sum(logpdf(returns, parameters, tolerance)))


def objective_gradient(q, returns, scale, terms):
    """Analytic gradient; lambda > 0 here, zero handled by nested model."""
    mu, sigma, lam, mj, sj = q * scale
    k = np.arange(terms)
    v = sigma * sigma + k * sj * sj
    residual = returns[:, None] - mu - k * mj
    log_component = log_weights(lam, terms) - .5 * (np.log(2 * np.pi * v) + residual ** 2 / v)
    density = logsumexp(log_component, axis=1)
    responsibility = np.exp(log_component - density[:, None])
    mean_grad = residual / v
    var_grad = residual ** 2 / v ** 2 - 1 / v
    gradients = [mean_grad, sigma * var_grad, np.broadcast_to(k / lam - 1, residual.shape),
                 k * mean_grad, k * sj * var_grad]
    gradient = -np.array([np.sum(responsibility * g) for g in gradients]) * scale
    return -float(density.sum()), gradient


def calibrate(returns, config, warm_start=None):
    begin = time.perf_counter()
    r = np.asarray(returns, dtype=float)
    if len(r) < 20 or not np.isfinite(r).all() or r.std() <= 0:
        raise ValueError("MLE requires at least 20 finite, nonconstant returns.")
    bounds = np.asarray(config["bounds"], dtype=float)
    scale = np.array([.01, .01, .1, .02, .02])
    terms = mixture_terms(bounds[2, 1], config["poisson_tail_tolerance"])
    # Bounded Gaussian MLE is exact and includes lambda = sigma_j = mu_j = 0.
    gaussian = np.array([np.clip(r.mean(), *bounds[0]), np.clip(r.std(), *bounds[1]), 0, 0, 0])
    gaussian_nll = nll(gaussian, r)
    starts = [np.array([r.mean(), max(r.std() * factor, .001), lam, sign * r.std(), max(r.std(), .002)])
              for factor, lam, sign in [(0.8, .03, -1), (0.8, .03, 1), (0.6, .2, 0)]]
    if warm_start is not None and warm_start[2] > 0:
        starts.append(np.asarray(warm_start).copy())
    opt_bounds = bounds.copy()
    opt_bounds[2, 0] = 1e-8  # Numerical interior for analytic k/lambda gradient only.
    attempts, fits = [], []
    for i, start in enumerate(starts):
        start = np.clip(start, opt_bounds[:, 0], opt_bounds[:, 1])
        result = minimize(objective_gradient, start / scale, args=(r, scale, terms), jac=True,
                          bounds=opt_bounds / scale[:, None], method="L-BFGS-B",
                          options={"maxiter": config["mle_maxiter"], "ftol": 1e-11, "gtol": 1e-6, "maxls": 50})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise FloatingPointError("Nonfinite Merton optimization.")
        fits.append(result)
        attempts.append({"start": i, "success": bool(result.success), "message": str(result.message),
                         "nll": float(result.fun), "iterations": int(result.nit)})
    successful = [result for result in fits if result.success]
    if not successful:
        raise RuntimeError(f"All Merton starts failed: {attempts}")
    best = min(successful, key=lambda z: z.fun)
    mjd = best.x * scale
    # The full model contains the exact Gaussian; never report worse full MLE.
    full = mjd if best.fun < gaussian_nll else gaussian.copy()
    full_nll = min(float(best.fun), gaussian_nll)
    gaussian_bic = 2 * gaussian_nll + 2 * np.log(len(r))
    jump_bic = 2 * full_nll + 5 * np.log(len(r))
    use_jump = jump_bic < gaussian_bic
    selected = full.copy() if use_jump else gaussian.copy()
    near = np.isclose(full, bounds[:, 0], atol=1e-5) | np.isclose(full, bounds[:, 1], atol=1e-5)
    diag = {"nll_mjd": full_nll, "nll_gaussian": gaussian_nll,
            "bic_mjd": float(jump_bic), "bic_gaussian": float(gaussian_bic),
            "selected_family": "merton" if use_jump else "gaussian",
            "full_boundary": bool(near.any()), "boundary_parameters": ",".join(np.array(["mu_log", "sigma", "lambda", "mu_j", "sigma_j"])[near]),
            "successful_starts": len(successful), "failed_starts": len(fits) - len(successful),
            "start_nll_range": float(np.ptp([x.fun for x in fits])), "poisson_terms": terms,
            "omitted_mass": float(poisson.sf(terms - 1, full[2])),
            "seconds": time.perf_counter() - begin}
    return selected, full, diag, attempts


def simulate(parameters, n, seed=42):
    mu, sigma, lam, mj, sj = validate_parameters(parameters)
    rng = np.random.default_rng(seed)
    count = rng.poisson(lam, n)
    return mu + sigma * rng.normal(size=n) + count * mj + np.sqrt(count) * sj * rng.normal(size=n)
