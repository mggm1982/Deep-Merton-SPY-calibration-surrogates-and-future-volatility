import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import norm, poisson
from finance_project.merton import logpdf, nll, mixture_terms, log_weights, objective_gradient, calibrate, simulate
from finance_project.units import total_variance, total_volatility, annual_parameters, ParameterTransform, realized_volatility


@pytest.mark.parametrize("lam", [0., 1e-10, .1, 1., 5.])
def test_poisson_normalization_and_adaptive_tail(lam):
    count = mixture_terms(lam)
    assert poisson.sf(count - 1, lam) <= 1e-12
    assert abs(np.exp(log_weights(lam, count)).sum() - 1) < 1.01e-12
    assert mixture_terms(5) > mixture_terms(.1)


def test_density_integrates_to_one_and_is_finite_in_tails():
    p = np.array([.0003, .01, .15, -.025, .02])
    area = quad(lambda r: float(np.exp(logpdf(r, p))), -2, 2, epsabs=1e-8)[0]
    assert np.isclose(area, 1, atol=1e-8)
    assert np.isfinite(logpdf(np.array([-100., -1., 0., 1., 100.]), p)).all()
    with pytest.raises(ValueError):
        logpdf([0], p, terms=1)


def test_zero_jump_case_equals_gaussian():
    x = np.linspace(-.1, .1, 30)
    np.testing.assert_allclose(logpdf(x, [.001, .02, 0, .2, .1]), norm.logpdf(x, .001, .02), atol=1e-12)


def test_gradient_matches_central_difference():
    r = np.random.default_rng(9).normal(0, .02, 90)
    scale = np.array([.01, .01, .1, .02, .02])
    q = np.array([.001, .012, .1, -.02, .025]) / scale
    count = mixture_terms(1.)
    value, analytical = objective_gradient(q, r, scale, count)
    numerical = approx_derivative(lambda v: objective_gradient(v, r, scale, count)[0], q).ravel()
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-6)
    assert np.isclose(value, nll(q * scale, r), atol=1e-9)


def test_units_and_horizon_scaling():
    p = np.array([.001, .01, .05, -.02, .03])
    variance = .01 ** 2 + .05 * (.02 ** 2 + .03 ** 2)
    assert total_variance(p) == pytest.approx(variance)
    assert total_volatility(p) == pytest.approx(np.sqrt(252 * variance))
    annual = annual_parameters(p)
    assert total_volatility(annual, annualization=1) == pytest.approx(total_volatility(p))
    np.testing.assert_array_equal(annual[3:], p[3:])
    assert realized_volatility([-.01, 0, .01]) == pytest.approx(.01 * np.sqrt(252))
    assert realized_volatility([.02, .03, .04]) == pytest.approx(realized_volatility([-.01, 0, .01]))


def test_parameter_transforms_round_trip_and_constraints():
    p = np.array([[.001, .01, 0, 0, 0], [-.005, .02, .2, -.05, .06]])
    transform = ParameterTransform()
    np.testing.assert_allclose(transform.inverse_transform(transform.transform(p)), p, atol=1e-14)
    predicted = transform.inverse_transform([[0, -8, -2, -.2, -1]])
    assert predicted[0, 1] > 0 and predicted[0, 2] == 4 and predicted[0, 4] == 1
    with pytest.raises(ValueError):
        total_variance([0, 0, .1, 0, .01])


def test_nested_model_accepts_zero_jumps(config):
    returns = norm.ppf((np.arange(400) + .5) / 400) * .01 + .0003
    selected, raw, diag, attempts = calibrate(returns, config)
    assert diag["selected_family"] == "gaussian"
    assert selected[2] == selected[3] == selected[4] == 0
    assert selected[0] == pytest.approx(returns.mean())
    assert selected[1] == pytest.approx(returns.std())
    assert diag["nll_mjd"] <= diag["nll_gaussian"] + 1e-10


def test_approximate_recovery_on_identifiable_simulated_data(config):
    # A large, clearly separated simulated jump component tests implementation.
    # This is not a claim that 126 market observations identify all parameters.
    true = np.array([.0005, .005, .12, -.045, .015])
    r = simulate(true, 20000, seed=1976)
    selected, raw, diag, attempts = calibrate(r, config)
    assert diag["selected_family"] == "merton"
    for estimate, reference, tolerance in zip(raw, true, [.0006, .001, .025, .006, .006]):
        assert abs(estimate - reference) <= tolerance, (raw, true)
    assert abs(total_variance(raw) / total_variance(true) - 1) < .15
