"""Strict source validation and explicit chronological forecast partitions."""
from pathlib import Path
import numpy as np
import pandas as pd


def log_returns(prices):
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Prices must be a finite positive vector with at least two rows.")
    return np.log(p[1:]) - np.log(p[:-1])


def validate_dates(dates):
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be nonmissing, unique, and strictly increasing.")


def load_data(raw_dir):
    frames, audit = [], {}
    for name in ["train", "test"]:
        raw = pd.read_csv(Path(raw_dir) / f"spy_{name}_data.csv")
        date_col = "Date" if "Date" in raw else "Price"
        if date_col not in raw or "Close" not in raw:
            raise ValueError(f"{name}: missing date/Close column.")
        dates = pd.to_datetime(raw[date_col], format="%d/%m/%y", errors="raise")
        validate_dates(dates)
        numeric = raw.select_dtypes(include=[np.number])
        if raw.isna().any().any() or not np.isfinite(numeric).all().all():
            raise ValueError(f"{name}: missing or nonfinite source values; no silent imputation.")
        if np.any(pd.to_numeric(raw.Close, errors="raise") <= 0):
            raise ValueError(f"{name}: nonpositive close.")
        audit[name] = {"rows": len(raw), "columns": raw.dtypes.astype(str).to_dict(),
                       "start": str(dates.iloc[0].date()), "end": str(dates.iloc[-1].date()),
                       "missing": raw.isna().sum().to_dict(), "duplicate_dates": 0,
                       "calendar_gap_counts": dates.diff().dt.days.value_counts().to_dict()}
        frame = raw.rename(columns={date_col: "date"}).copy()
        frame["date"], frame["source"] = dates, name
        frame["source_row"] = np.arange(len(frame))
        frames.append(frame)
    all_prices = pd.concat(frames, ignore_index=True)
    validate_dates(all_prices.date)
    returns = log_returns(all_prices.Close)
    # Only the initial row lacks a previous close. Keep its price in raw data,
    # but exclude it explicitly from the processed return table.
    observations = all_prices.iloc[1:].copy().reset_index(drop=True)
    observations["log_return"] = returns
    audit["initial_price_only_row"] = str(all_prices.date.iloc[0].date())
    audit["max_supplied_return_error"] = float(np.max(np.abs(returns - observations.returns)))
    # Supplied text exports differ at ~1e-8 for a few tiny returns. A 1e-6
    # absolute-return tolerance (0.01 basis point) admits serialization noise;
    # it does not change the recomputed modelling returns.
    audit["supplied_return_tolerance"] = 1e-6
    if audit["max_supplied_return_error"] > audit["supplied_return_tolerance"]:
        raise ValueError("Supplied returns disagree with log(Close[t]/Close[t-1]).")
    audit["price_policy"] = "Use supplied Close unchanged; adjustment/dividend provenance unknown."
    audit["unused_fields"] = [c for c in all_prices if c not in ["date", "Close", "source", "source_row"]]
    audit["missing_sessions_policy"] = "Observed sessions only; no forward-fill. Exchange-calendar completeness not established."
    return observations, audit


def make_partitions(observations, config):
    """Purge all forward labels crossing validation/test forecast origins."""
    w, h, step = config["mle_window"], config["horizon"], config["stride"]
    first_test = int(np.flatnonzero(observations.source.to_numpy() == "test")[0])
    dev = np.arange(w - 1, first_test, step, dtype=int)
    cut = int(len(dev) * (1 - config["validation_fraction"]))
    if cut <= 0 or cut >= len(dev):
        raise ValueError("Insufficient development observations for chronological splitting.")
    first_validation = dev[cut]
    fit = dev[(dev < first_validation) & (dev + h < first_validation)]
    val = dev[(dev >= first_validation) & (dev + h < first_test)]
    test = np.arange(first_test, len(observations) - h, step, dtype=int)
    parts = {"fit": fit, "validation": val, "test": test}
    if min(map(len, parts.values())) < 3:
        raise ValueError("Each split needs at least three complete forecast horizons.")
    if not (fit[-1] + h < val[0] and val[-1] + h < test[0]):
        raise AssertionError("Forward labels cross split boundaries.")
    return parts


def window_at(returns, end, length):
    if end - length + 1 < 0 or end >= len(returns):
        raise ValueError("Incomplete trailing window.")
    r = np.asarray(returns, dtype=float)[end - length + 1:end + 1]
    if len(r) != length or not np.isfinite(r).all():
        raise ValueError("Invalid window.")
    return r


def future_at(returns, end, horizon):
    if horizon < 2 or end + horizon >= len(returns):
        raise ValueError("Incomplete future horizon.")
    return np.asarray(returns, dtype=float)[end + 1:end + horizon + 1]
