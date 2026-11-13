"""Single shared pipeline used by CLI, notebook, smoke test, and verification."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import shutil
import sys
import time
import warnings
import numpy as np
import pandas as pd
from .data import load_data, make_partitions, window_at
from .features import make_inputs, make_labels, sample_metadata, FEATURE_NAMES
from .merton import calibrate
from .models import seed_everything, fit_models, predict, save_models, load_models
from .units import NAMES, total_volatility
from .evaluation import aligned_join, parameter_metrics, volatility_metrics

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(c):
    for key in ["mle_window", "sequence_length", "horizon", "stride", "epochs", "patience", "mle_maxiter", "annualization"]:
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if c["mle_window"] < 21 or not 2 <= c["sequence_length"] <= c["mle_window"] or c["horizon"] < 2:
        raise ValueError("Require MLE window >= 21, 2 <= sequence <= window, horizon >= 2.")
    if not 0 < c["validation_fraction"] < .5 or not 0 < c["ewma_decay"] < 1:
        raise ValueError("Invalid fraction/decay.")
    b = np.asarray(c["bounds"])
    if b.shape != (5, 2) or not np.isfinite(b).all() or np.any(b[:, 0] >= b[:, 1]):
        raise ValueError("Invalid optimization bounds.")
    if b[1, 0] <= 0 or b[2, 0] != 0 or b[4, 0] != 0 or not b[3, 0] <= 0 <= b[3, 1]:
        raise ValueError("Bounds must support positive sigma and the exact zero-jump model.")
    if not 1 <= len(c["lstm_hidden_sizes"]) <= 2 or not c["ridge_alphas"]:
        raise ValueError("Configure one or two LSTM sizes and at least one Ridge alpha.")


def run_pipeline(project_root=ROOT, output_root=None, config=None):
    started = time.perf_counter()
    project_root = Path(project_root).resolve()
    root = Path(output_root).resolve() if output_root else project_root
    c = dict(config) if config else json.loads((project_root / "config.json").read_text())
    validate_config(c)
    for sub in ["data/raw", "data/processed", "models", "results/data_audit", "results/metrics",
                "results/predictions", "results/figures", "results/logs", "report"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    if root != project_root:
        for path in (project_root / "data/raw").glob("*"):
            if path.is_file():
                shutil.copyfile(path, root / "data/raw" / path.name)
    save_json(root / "config.json", c)
    logger = logging.getLogger(f"finance.{id(c)}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(root / "results/logs/run.log", mode="w")
    stream = logging.StreamHandler(sys.stdout)
    for handler in [file_handler, stream]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            seed_everything(c["seed"])
            logger.info("Starting fresh calibration and model training; no cached fits or weights reused.")
            obs, audit = load_data(root / "data/raw")
            obs.to_csv(root / "data/processed/observations.csv", index=False)
            save_json(root / "results/data_audit/data_audit.json", audit)
            parts = make_partitions(obs, c)
            partition_info = {name: {"n": len(ends), "first_origin": str(obs.date.iloc[ends[0]].date()),
                              "last_origin": str(obs.date.iloc[ends[-1]].date()),
                              "last_target_end": str(obs.date.iloc[ends[-1] + c["horizon"]].date())}
                              for name, ends in parts.items()}
            logger.info("Chronological partitions: %s", partition_info)
            returns = obs.log_return.to_numpy()
            diagnostics, attempts, cache = [], [], {}

            def estimate_split(name):
                ends = parts[name]
                needed = list(ends)
                if name != "fit":
                    needed = [int(ends[0] - c["stride"])] + needed
                warm = None
                for index, end in enumerate(needed):
                    if end not in cache:
                        selected, full, diag, runs = calibrate(window_at(returns, end, c["mle_window"]), c, warm)
                        cache[end] = (selected, full)
                        purpose = "current" if end in ends else "persistence_context"
                        context = {"split": name, "purpose": purpose, "date": str(obs.date.iloc[end].date()),
                                   "origin_index": int(end), "window_start_index": int(end - c["mle_window"] + 1)}
                        diagnostics.append({**context, **diag})
                        attempts.extend([{**context, **run} for run in runs])
                        warm = full
                    if index % 30 == 0:
                        logger.info("%s MLE %d/%d", name, index + 1, len(needed))
                selected = np.array([cache[e][0] for e in ends])
                raw = np.array([cache[e][1] for e in ends])
                previous = np.array([cache[e - c["stride"]][0] for e in ends]) if name != "fit" else None
                meta = sample_metadata(obs, ends, c)
                for j, target in enumerate(NAMES):
                    meta[target], meta[f"raw_mjd_{target}"] = selected[:, j], raw[:, j]
                meta.to_csv(root / f"data/processed/{name}_targets.csv", index=False)
                return selected, raw, previous

            # Test response values are not constructed until validation locks every choice.
            estimates = {name: estimate_split(name) for name in ["fit", "validation"]}
            inputs = {name: make_inputs(obs, parts[name], c) for name in ["fit", "validation"]}
            labels = {name: make_labels(obs, parts[name], c) for name in ["fit", "validation"]}
            fitted, histories, trials = {}, [], []
            for task in ["parameters", "volatility"]:
                target = estimates if task == "parameters" else labels
                yfit = target["fit"][0] if task == "parameters" else target["fit"]
                yval = target["validation"][0] if task == "parameters" else target["validation"]
                bundle, net, history, candidates = fit_models((*inputs["fit"], yfit), (*inputs["validation"], yval), task, c)
                fitted[task] = bundle, net
                histories.append(history)
                trials.append(candidates)
                save_models(root / "models", task, bundle, net)
                logger.info("Fitted %s: Ridge alpha=%s; LSTM units=%s", task, bundle["ridge_alpha"], bundle["architecture"]["hidden"])

            predictions, parameter_tables, vol_tables, param_metrics, vol_metrics = {}, {}, {}, [], []
            vol_models = ["training_mean", "historical_21d", "ewma", "mle_total_vol", "raw_mjd_total_vol",
                          "mean_total_vol", "ridge_total_vol", "lstm_total_vol", "direct_ridge", "direct_lstm"]

            def evaluate_split(name):
                x, s = inputs[name]
                pp = predict(*fitted["parameters"], x, s)
                pp["persistence"] = estimates[name][2]
                dp = predict(*fitted["volatility"], x, s)
                predictions[name] = {"parameters": pp, "volatility": dp}
                reference = sample_metadata(obs, parts[name], c)
                predictions_by_date = pd.DataFrame({"date": reference.date})
                for j, target in enumerate(NAMES):
                    reference[f"actual_{target}"] = estimates[name][0][:, j]
                    reference[f"raw_mjd_{target}"] = estimates[name][1][:, j]
                    for model, p in pp.items():
                        predictions_by_date[f"{model}_{target}"] = p[:, j]
                table = aligned_join(reference, predictions_by_date)
                parameter_tables[name] = table
                table.to_csv(root / f"results/predictions/{name}_parameters.csv", index=False)
                param_metrics.append(parameter_metrics(table, pp, NAMES, name))
                fv = sample_metadata(obs, parts[name], c)
                fv["actual_future_vol"] = labels[name]
                forecast = pd.DataFrame({"date": fv.date, "training_mean": dp["mean"],
                                         "historical_21d": s[:, FEATURE_NAMES.index("vol_21")],
                                         "ewma": s[:, FEATURE_NAMES.index("ewma_vol")],
                                         "mle_total_vol": total_volatility(estimates[name][0], c["annualization"]),
                                         "raw_mjd_total_vol": total_volatility(estimates[name][1], c["annualization"]),
                                         "direct_ridge": dp["ridge"], "direct_lstm": dp["lstm"]})
                for model in ["mean", "ridge", "lstm"]:
                    forecast[f"{model}_total_vol"] = total_volatility(pp[model], c["annualization"])
                fv = aligned_join(fv, forecast)
                vol_tables[name] = fv
                fv.to_csv(root / f"results/predictions/{name}_volatility.csv", index=False)
                vol_metrics.append(volatility_metrics(fv, vol_models, name))

            evaluate_split("validation")
            prep = fitted["parameters"][0]["preprocessing"]
            py = prep.targets(estimates["validation"][0])
            parameter_scores = {name: float(np.mean((prep.targets(p) - py) ** 2))
                                for name, p in predictions["validation"]["parameters"].items()}
            validation_scores = dict(zip(vol_metrics[0].model, vol_metrics[0].RMSE))
            selection = {"parameter_model": min(parameter_scores, key=parameter_scores.get),
                         "parameter_scores": parameter_scores, "parameter_criterion": "standardized transformed-parameter MSE",
                         "forecast_model": min(validation_scores, key=validation_scores.get),
                         "forecast_scores": validation_scores, "forecast_criterion": "next-H-session annualized volatility RMSE",
                         "models_frozen_before_test": True, "test_used_for_tuning": False,
                         "prior_test_exposure": "Supplied 2023 test results were already reported in the original project; this is a fixed rerun holdout, not a newly unseen dataset."}
            save_json(root / "results/metrics/selection.json", selection)
            logger.info("Choices locked before test: %s", selection)
            estimates["test"] = estimate_split("test")
            inputs["test"] = make_inputs(obs, parts["test"], c)
            labels["test"] = make_labels(obs, parts["test"], c)
            evaluate_split("test")

            history = pd.concat(histories, ignore_index=True)
            history.to_csv(root / "results/metrics/training_history.csv", index=False)
            pd.concat(trials, ignore_index=True).to_csv(root / "results/metrics/model_candidates.csv", index=False)
            pd.concat(param_metrics, ignore_index=True).to_csv(root / "results/metrics/parameter_metrics.csv", index=False)
            pd.concat(vol_metrics, ignore_index=True).to_csv(root / "results/metrics/future_volatility.csv", index=False)
            # Also assess reconstructed total volatility against the historical calibrated reference.
            reconstruction = []
            from .evaluation import metric_row
            for name in ["validation", "test"]:
                reference = total_volatility(estimates[name][0], c["annualization"])
                benchmark = vol_tables[name].mean_total_vol
                for model in ["mean", "ridge", "lstm"]:
                    reconstruction.append({"split": name, "model": model, **metric_row(reference, vol_tables[name][f"{model}_total_vol"], benchmark)})
            pd.DataFrame(reconstruction).to_csv(root / "results/metrics/reconstructed_volatility.csv", index=False)
            # Nonoverlapping subsequence gives a descriptive sensitivity check, not a CI.
            gap = int(np.ceil(c["horizon"] / c["stride"]))
            volatility_metrics(vol_tables["test"].iloc[::gap], vol_models, "test_nonoverlapping").to_csv(
                root / "results/metrics/nonoverlapping_volatility.csv", index=False)
            diag = pd.DataFrame(diagnostics)
            diag.to_csv(root / "results/data_audit/calibration_diagnostics.csv", index=False)
            pd.DataFrame(attempts).to_csv(root / "results/data_audit/optimizer_attempts.csv", index=False)
            timing = {"median_test_mle_seconds": float(diag.loc[(diag.split == "test") & (diag.purpose == "current"), "seconds"].median())}
            for task, fitted_model in fitted.items():
                samples = []
                for _ in range(20):
                    tick = time.perf_counter()
                    predict(*fitted_model, inputs["test"][0][:1], inputs["test"][1][:1])
                    samples.append(time.perf_counter() - tick)
                timing[f"{task}_all_three_models_median_seconds"] = float(np.median(samples))
            timing["scope"] = "Warm CPU wall clock. MLE includes multistart calibration; inference includes scaling and mean/Ridge/LSTM, excludes feature construction, disk IO and training."
            save_json(root / "results/metrics/timing.json", timing)
            scaler_audit = {task: {"fit_sample_count": bundle["preprocessing"].fit_sample_count,
                            "last_fit_origin": partition_info["fit"]["last_origin"],
                            "last_fit_target_end": partition_info["fit"]["last_target_end"],
                            "feature_condition_number": bundle["preprocessing"].feature_condition_number,
                            "ridge_alpha": bundle["ridge_alpha"], "architecture": bundle["architecture"]}
                            for task, (bundle, _) in fitted.items()}
            save_json(root / "results/data_audit/scaling.json", scaler_audit)
            for task, (bundle, net) in fitted.items():
                loaded = load_models(root / "models", task)
                expected = predict(bundle, net, *inputs["test"])
                actual = predict(*loaded, *inputs["test"])
                for model in expected:
                    if not np.array_equal(expected[model], actual[model]):
                        raise AssertionError(f"{task}/{model} reload changed predictions.")
            # Save an example with all required historical context, including EWMA state inputs.
            obs.iloc[:parts["test"][0] + 1][["date", "log_return"]].to_csv(root / "data/processed/example_history.csv", index=False)
            # Exercise the public, single-example reload path. Batch versus
            # single-example float32 matrix kernels may differ in final bits.
            example = example_predict(root)
            np.testing.assert_allclose(list(example["lstm_parameters"].values()),
                                       predictions["test"]["parameters"]["lstm"][0], rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(example["direct_lstm_future_volatility"],
                                       predictions["test"]["volatility"]["lstm"][0], rtol=1e-6, atol=1e-8)
            save_json(root / "results/predictions/example_prediction.json", example)
            from .plotting import plots
            plots(root, parameter_tables["test"], vol_tables["test"], history, diag, selection, c["horizon"])
            raw_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "data/raw").iterdir() if p.is_file()}
            warning_records = [{"category": w.category.__name__, "message": str(w.message),
                                "file": Path(w.filename).name, "line": w.lineno} for w in captured]
            save_json(root / "results/logs/warnings.json", warning_records)
            manifest = {"config": c, "versions": {pkg: importlib.metadata.version(pkg) for pkg in
                        ["numpy", "pandas", "scipy", "scikit-learn", "torch", "joblib", "matplotlib"]},
                        "python": platform.python_version(), "platform": platform.platform(), "raw_sha256": raw_hashes,
                        "partitions": partition_info, "feature_names": FEATURE_NAMES,
                        "sequence_channels": ["log_return", "absolute_log_return"],
                        "selection": selection, "elapsed_seconds": time.perf_counter() - started,
                        "warning_count": len(warning_records), "reload_exact": True,
                        "single_vs_batch_tolerance": {"rtol": 1e-6, "atol": 1e-8},
                        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src").rglob("*.py"))},
                        "generated_outputs": sorted(str(p.relative_to(root)) for sub in ["models", "data/processed", "results", "report"]
                          for p in (root / sub).rglob("*") if p.is_file()),
                        "omitted_final_incomplete_origins": int(sum(e + c["horizon"] >= len(obs) for e in
                          np.arange(parts["test"][0], len(obs), c["stride"]))),
                        "limitations": ["Prior test exposure; no claim of a fresh unseen holdout.",
                          "Physical-measure calibration; not option-market pricing.",
                          "Raw price adjustment provenance unknown.", "Overlapping windows and outcomes.",
                          "BIC is a conservative heuristic in a nonregular mixture, not a jump significance test."]}
            save_json(root / "run_manifest.json", manifest)
            from .reporting import write_summary
            write_summary(root, manifest, vol_metrics[-1], diag)
            logger.info("Run complete in %.2f seconds; captured warnings=%d", time.perf_counter() - started, len(warning_records))
            return manifest
    finally:
        for handler in [file_handler, stream]:
            handler.close()
            logger.removeHandler(handler)


def example_predict(root=ROOT):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    obs = pd.read_csv(root / "data/processed/example_history.csv", parse_dates=["date"])
    inputs = make_inputs(obs, np.array([len(obs) - 1]), config)
    pp = predict(*load_models(root / "models", "parameters"), *inputs)
    vp = predict(*load_models(root / "models", "volatility"), *inputs)
    return {"date": str(obs.date.iloc[-1].date()), "lstm_parameters": dict(zip(NAMES, pp["lstm"][0].tolist())),
            "direct_lstm_future_volatility": float(vp["lstm"][0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional separate output directory")
    parser.add_argument("--config", type=Path, help="Optional configuration JSON")
    parser.add_argument("--example", action="store_true")
    args = parser.parse_args()
    if args.example:
        print(json.dumps(example_predict(args.output or ROOT), indent=2))
    else:
        config = json.loads(args.config.read_text()) if args.config else None
        run_pipeline(ROOT, args.output, config)
