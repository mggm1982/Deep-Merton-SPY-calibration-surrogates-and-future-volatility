"""Independent recomputation from raw observations, saved predictions and models."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .data import load_data, make_partitions, window_at, future_at
from .features import make_inputs, make_labels, sample_metadata
from .models import load_models, predict
from .units import NAMES, total_volatility, realized_volatility
from .evaluation import metric_row
from .pipeline import save_json, ROOT, example_predict


def verify(root=ROOT):
    root = Path(root)
    m = json.loads((root / "run_manifest.json").read_text())
    c = json.loads((root / "config.json").read_text())
    if c != m["config"]:
        raise AssertionError("Config differs from the run manifest; rerun after editing.")
    checks = []
    for name, digest in m["raw_sha256"].items():
        assert hashlib.sha256((root / "data/raw" / name).read_bytes()).hexdigest() == digest
    checks.append("Raw-file SHA256 hashes verified")
    obs, _ = load_data(root / "data/raw")
    parts = make_partitions(obs, c)
    assert parts["fit"][-1] + c["horizon"] < parts["validation"][0]
    assert parts["validation"][-1] + c["horizon"] < parts["test"][0]
    checks.append("Forward targets strictly precede the next split's first forecast origin")
    fit_inputs = make_inputs(obs, parts["fit"], c)
    fit_params = pd.read_csv(root / "data/processed/fit_targets.csv")[NAMES].to_numpy()
    for task in ["parameters", "volatility"]:
        bundle, _ = load_models(root / "models", task)
        prep = bundle["preprocessing"]
        assert prep.fit_sample_count == len(parts["fit"])
        np.testing.assert_allclose(prep.stats_scaler.mean_, fit_inputs[1].mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(prep.sequence_scaler.mean_, fit_inputs[0].reshape(-1, 2).mean(axis=0), atol=1e-14)
        target = fit_params if task == "parameters" else make_labels(obs, parts["fit"], c)
        np.testing.assert_allclose(prep.target_scaler.mean_, prep.encode(target).mean(axis=0), atol=1e-12)
    checks.append("All saved scaler means recomputed from fit data only")
    pm = pd.read_csv(root / "results/metrics/parameter_metrics.csv")
    vm = pd.read_csv(root / "results/metrics/future_volatility.csv")
    for split in ["validation", "test"]:
        ends = parts[split]
        meta = sample_metadata(obs, ends, c)
        p = pd.read_csv(root / f"results/predictions/{split}_parameters.csv")
        v = pd.read_csv(root / f"results/predictions/{split}_volatility.csv")
        for table in [p, v]:
            for key in ["date", "window_start", "sequence_start", "target_start", "target_end"]:
                np.testing.assert_array_equal(pd.to_datetime(table[key]), meta[key])
            np.testing.assert_array_equal(table.origin_index, ends)
            np.testing.assert_array_equal(table.source_row, meta.source_row)
            numeric = table.select_dtypes(include=[np.number])
            assert np.isfinite(numeric).all().all()
        np.testing.assert_allclose(v.actual_future_vol, make_labels(obs, ends, c), rtol=1e-12, atol=1e-14)
        for i, end in enumerate(ends):
            expected = np.std(obs.log_return.iloc[end + 1:end + 1 + c["horizon"]], ddof=1) * np.sqrt(c["annualization"])
            assert np.isclose(v.actual_future_vol.iloc[i], expected, atol=1e-14)
        inputs = make_inputs(obs, ends, c)
        for task in ["parameters", "volatility"]:
            bundle, net = load_models(root / "models", task)
            pred = predict(bundle, net, *inputs)
            again = predict(bundle, net, *inputs)
            for model in pred:
                np.testing.assert_array_equal(pred[model], again[model])
                if task == "parameters":
                    np.testing.assert_allclose(p[[f"{model}_{name}" for name in NAMES]], pred[model], atol=1e-12)
                    np.testing.assert_allclose(v[f"{model}_total_vol"], total_volatility(pred[model], c["annualization"]), atol=1e-12)
                else:
                    col = "training_mean" if model == "mean" else f"direct_{model}"
                    np.testing.assert_allclose(v[col], pred[model], atol=1e-12)
        for _, row in pm[pm.split == split].iterrows():
            calculated = metric_row(p[f"actual_{row.target}"], p[f"{row.model}_{row.target}"], p[f"mean_{row.target}"])
            for key, value in calculated.items():
                if key in ["undefined_reason"]:
                    continue
                assert (pd.isna(row[key]) if value is None else np.isclose(row[key], value, atol=1e-10)), (split, key)
        for _, row in vm[vm.split == split].iterrows():
            calculated = metric_row(v.actual_future_vol, v[row.model], v.historical_21d)
            for key, value in calculated.items():
                if key == "undefined_reason":
                    continue
                assert (pd.isna(row[key]) if value is None else np.isclose(row[key], value, atol=1e-10)), (split, key)
    checks += ["Prediction/reference dates and source rows match raw observations", "Future labels independently recomputed from t+1 through t+H",
               "Saved Ridge/LSTM/mean inference matches all prediction CSVs", "Inference is deterministic", "Parameter and future-volatility metrics independently recompute"]
    diag = pd.read_csv(root / "results/data_audit/calibration_diagnostics.csv")
    assert np.isfinite(diag.select_dtypes(include=[np.number])).all().all()
    assert (diag.omitted_mass <= c["poisson_tail_tolerance"] * 1.01).all()
    assert (diag.successful_starts >= 1).all()
    checks.append("Every calibration has a successful start and an adequate Poisson truncation")
    actual_example = example_predict(root)
    saved_example = json.loads((root / "results/predictions/example_prediction.json").read_text())
    assert actual_example == saved_example
    checks.append("Saved example prediction reproduces exactly")
    warnings = json.loads((root / "results/logs/warnings.json").read_text())
    assert not warnings, f"Review runtime warnings: {warnings}"
    checks.append("No runtime warnings captured")
    save_json(root / "results/data_audit/verification.json", {"passed": checks,
              "limitations": m["limitations"], "not_executed": ["Hosted Colab session", "Optional live option comparison"]})
    return checks
