"""Date-aligned descriptive metrics; no IID confidence-interval claims."""
import numpy as np
import pandas as pd


def metric_row(actual, prediction, benchmark):
    y, p, b = map(lambda x: np.asarray(x, dtype=float), [actual, prediction, benchmark])
    if y.shape != p.shape or y.shape != b.shape or not np.isfinite([y, p, b]).all():
        raise ValueError("Metrics require aligned, finite arrays with identical shape.")
    error, mse = p - y, np.mean((p - y) ** 2)
    denominator, benchmark_mse = np.mean((y - y.mean()) ** 2), np.mean((b - y) ** 2)
    correlation_ok = y.std() > 1e-12 and p.std() > 1e-12
    return {"n": len(y), "MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(mse)),
            "bias": float(error.mean()), "R2": float(1 - mse / denominator) if denominator > 1e-24 else None,
            "R2_oos": float(1 - mse / benchmark_mse) if benchmark_mse > 1e-24 else None,
            "correlation": float(np.corrcoef(y, p)[0, 1]) if correlation_ok else None,
            "error_q05": float(np.quantile(error, .05)), "error_q50": float(np.quantile(error, .5)),
            "error_q95": float(np.quantile(error, .95)),
            "undefined_reason": "; ".join([name for name, valid in [("correlation: constant series", correlation_ok),
              ("R2: constant reference", denominator > 1e-24), ("R2_oos: zero benchmark error", benchmark_mse > 1e-24)] if not valid])}


def aligned_join(reference, predictions):
    if reference.date.duplicated().any() or predictions.date.duplicated().any():
        raise ValueError("Duplicate prediction/reference dates.")
    if set(reference.date) != set(predictions.date):
        raise ValueError("Prediction/reference dates do not match.")
    return reference.merge(predictions, on="date", validate="one_to_one").sort_values("date").reset_index(drop=True)


def parameter_metrics(frame, models, names, split):
    return pd.DataFrame([{ "split": split, "model": model, "target": target,
                           **metric_row(frame[f"actual_{target}"], frame[f"{model}_{target}"], frame[f"mean_{target}"])}
                         for model in models for target in names])


def volatility_metrics(frame, models, split):
    return pd.DataFrame([{"split": split, "model": model,
                          **metric_row(frame.actual_future_vol, frame[model], frame.historical_21d)} for model in models])
