import numpy as np
import pandas as pd
import pytest
from finance_project.data import log_returns, validate_dates, load_data, make_partitions, window_at, future_at
from finance_project.features import make_inputs, make_labels, sample_metadata
from finance_project.evaluation import aligned_join


def test_log_returns_and_price_validation():
    p = np.array([100., 110., 99.])
    np.testing.assert_allclose(log_returns(p), [np.log(1.1), np.log(.9)])
    assert not np.allclose(log_returns(p), [.1, -.1])
    for bad in [[100, 0], [100, -1], [100, np.nan]]:
        with pytest.raises(ValueError):
            log_returns(bad)


@pytest.mark.parametrize("dates", [["2020-01-02", "2020-01-01"], ["2020-01-01", "2020-01-01"], ["2020-01-01", None]])
def test_date_errors_are_not_silently_sorted_or_deduplicated(dates):
    with pytest.raises(ValueError):
        validate_dates(dates)


def test_rolling_and_future_boundaries():
    r = np.arange(12, dtype=float)
    np.testing.assert_array_equal(window_at(r, 5, 3), [3, 4, 5])
    np.testing.assert_array_equal(future_at(r, 5, 3), [6, 7, 8])
    with pytest.raises(ValueError):
        window_at(r, 1, 3)
    with pytest.raises(ValueError):
        future_at(r, 10, 3)


def test_raw_data_and_purged_partitions(root, config):
    obs, audit = load_data(root / "data/raw")
    assert len(obs) == 1237
    assert np.isfinite(obs.log_return).all()
    assert audit["max_supplied_return_error"] < audit["supplied_return_tolerance"]
    parts = make_partitions(obs, config)
    assert parts["fit"][-1] + config["horizon"] < parts["validation"][0]
    assert parts["validation"][-1] + config["horizon"] < parts["test"][0]
    assert parts["test"][-1] + config["horizon"] < len(obs)


def test_future_perturbation_cannot_change_features(root, config):
    obs, _ = load_data(root / "data/raw")
    end = 300
    original = make_inputs(obs, np.array([end]), config)
    changed = obs.copy()
    changed.loc[end + 1:, "log_return"] *= 1000
    changed_inputs = make_inputs(changed, np.array([end]), config)
    for before, after in zip(original, changed_inputs):
        np.testing.assert_array_equal(before, after)
    assert not np.allclose(make_labels(obs, [end], config), make_labels(changed, [end], config))


def test_sequence_and_targets_use_the_same_origin(root, config):
    obs, _ = load_data(root / "data/raw")
    ends = np.array([300, 305])
    x, stats = make_inputs(obs, ends, config)
    assert x.shape == (2, config["sequence_length"], 2)
    np.testing.assert_array_equal(x[:, -1, 0], obs.log_return.iloc[ends])
    meta = sample_metadata(obs, ends, config)
    np.testing.assert_array_equal(meta.target_start, obs.date.iloc[ends + 1])
    np.testing.assert_array_equal(meta.target_end, obs.date.iloc[ends + config["horizon"]])
    y = make_labels(obs, ends, config)
    for i, end in enumerate(ends):
        assert np.isclose(y[i], np.std(obs.log_return.iloc[end + 1:end + 22], ddof=1) * np.sqrt(252))


def test_join_uses_dates_not_positions():
    reference = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "actual": [1, 2]})
    predictions = pd.DataFrame({"date": ["2023-01-02", "2023-01-01"], "pred": [20, 10]})
    assert aligned_join(reference, predictions).pred.tolist() == [10, 20]
    with pytest.raises(ValueError):
        aligned_join(reference, predictions.iloc[:1])
    with pytest.raises(ValueError):
        aligned_join(reference, pd.concat([predictions, predictions]))
