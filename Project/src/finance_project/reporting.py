"""Reports are generated from actual saved run results, never static scores."""
from pathlib import Path
import html


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = [f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(root, manifest, test_metrics, diagnostics):
    root = Path(root)
    c = manifest["config"]
    current = diagnostics[diagnostics.purpose == "current"]
    selected = manifest["selection"]["forecast_model"]
    table = markdown_table(test_metrics.sort_values("RMSE"), ["model", "n", "MAE", "RMSE", "bias", "R2_oos"])
    selected_row = test_metrics.set_index("model").loc[selected]
    baseline = test_metrics.set_index("model").loc["historical_21d"]
    verdict = "improved" if selected_row.RMSE < baseline.RMSE else "did not improve"
    text = f"""# SPY Merton calibration and future-volatility evaluation

This project estimates a historical physical-measure return distribution and learns an approximation to that calibration. A separate supervised task predicts future realized volatility. Neither task predicts prices or establishes trading profitability.

## Method

Use the supplied chronological CSVs unchanged. Recompute decimal log returns from Close. Exclude only the first price-only row, whose preceding close is unavailable. The raw adjustment/dividend policy cannot be independently established. Features are known after the forecast-origin close.

Calibrate the exact compound-Poisson normal mixture on {c['mle_window']} past observed returns at stride {c['stride']}. Use scaled multistart constrained MLE, analytic derivatives, log-sum-exp and a Poisson tail tolerance of {c['poisson_tail_tolerance']}. Preserve all raw fits and optimizer attempts. Compare the full five-parameter fit to an exact two-parameter Gaussian with BIC. The latter admits lambda=0 and records unused jump parameters as canonical zeros. BIC in this boundary mixture is a model-selection heuristic, not a formal jump test. Canonical zeros do not prove a zero-sized jump distribution.

Train mean, tuned Ridge and small LSTM models on the selected historical reference parameters. The LSTM uses {c['sequence_length']} returns and absolute returns plus historical summary statistics. The reference parameters are estimated labels, not ground truth. Direct Ridge/LSTM volatility models use the same historical inputs and the actual following {c['horizon']} returns as targets. Positive predictions use documented output links, with no post-prediction clipping.

Each future label is a demeaned sample SD with ddof=1, multiplied by sqrt({c['annualization']}). Model total volatility is sqrt(A*(sigma² + lambda*(mu_j² + sigma_j²))). These are compatible variance targets; expected sample SD need not equal the square root of expected variance. Parameter constancy across a future horizon is only a proxy assumption.

Fit/validation/test sizes: {manifest['partitions']['fit']['n']}/{manifest['partitions']['validation']['n']}/{manifest['partitions']['test']['n']}. Purge forward labels that cross the next partition's first forecast origin. Historical feature windows may overlap across partitions because that history was already known. Preprocessing uses fit samples only. Checkpoints, regularization and model family are selected on validation; models then freeze.

## Actual test results

Forecast model selected on validation: **{selected}**. It {verdict} the historical-volatility baseline on this fixed test rerun. Selection was not changed after observing these results. R2_oos uses historical_21d as the declared benchmark; ordinary R2 uses the evaluation sample mean and is saved separately.

{table}

Errors are prediction minus observation. Decimal 0.01 is one annualized volatility percentage point. Full per-parameter results and reconstructed historical total-volatility results are in results/metrics. Correlation is undefined for a constant series; blank metric cells have explicit reasons, not unhandled missing values.

## Numerical and statistical limitations

Raw five-parameter fits at bounds: {int(current.full_boundary.sum())}/{len(current)}. BIC selected a jump component in {int((current.selected_family == 'merton').sum())}/{len(current)} windows. Failed optimizer starts (including nonselected starts) total {int(diagnostics.failed_starts.sum())}; every attempt is retained. No window is dropped for poor fit or poor forecast performance.

Longer windows reduce small-sample instability but mix changing volatility regimes. Adjacent windows and {c['horizon']}-session future targets overlap. These metrics are descriptive; no IID confidence interval or significance claim is made. A nonoverlapping subsequence sensitivity table is included, but its smaller sample still does not establish deployment reliability. The dataset is too small for strong deep-learning conclusions.

The supplied 2023 test results had already been examined in the original project. This repair keeps them out of fitting, scaling and tuning, but cannot turn them into genuinely unseen data. A new future dataset is needed for a fresh scientific holdout. No raw data were replaced and no scores were manufactured.

## Reproduction

Run `python run_project.py`, `python -m pytest -q`, and `python src/verify_project.py`. Saved weights and preprocessing reproduce inference exactly in the tested environment. The manifest records versions, partitions and raw hashes. The CLI and notebook share source modules. A hosted Google Colab service session and the guide's optional live option comparison are not executed here. The source runs offline after dependency installation.
"""
    (root / "report/project_summary.md").write_text(text, encoding="utf-8")
    sections = [
        ("Problem", "Estimate the historical return distribution, learn a calibration shortcut, and evaluate a separate future-volatility forecast."),
        ("Data and timing", f"{manifest['partitions']['fit']['n']} fit, {manifest['partitions']['validation']['n']} validation, {manifest['partitions']['test']['n']} test origins. All inputs are available after close; future labels are purged at boundaries."),
        ("Model", f"{c['mle_window']}-session compound-Poisson MLE, with an exact no-jump alternative. Raw fits and boundary diagnostics remain visible."),
        ("Learning", "Mean, Ridge and small LSTM models; fit-only scaling, validation-only selection, identical inference-mode loss reporting."),
        ("Results", f"Validation selected {selected}. Test RMSE: {selected_row.RMSE:.5f}; historical-volatility RMSE: {baseline.RMSE:.5f}. It {verdict} that benchmark."),
        ("Limits", "Estimated labels are not true parameters. Overlapping outcomes, weak jump identification, limited data and previous test exposure prevent strong generalization claims.")]
    slides = "".join(f"<section><small>{i+1} / 6</small><h1>{html.escape(title)}</h1><p>{html.escape(body)}</p></section>" for i, (title, body) in enumerate(sections))
    (root / "report/presentation.html").write_text("""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SPY project results</title><style>body{margin:0;background:#122331;color:#f4f7fa;font:24px system-ui}section{padding:10vh 9vw;min-height:75vh;border-bottom:1px solid #436272}h1{font-size:48px}p{max-width:1050px;line-height:1.6}small{color:#78dbc5}@media print{section{break-after:page}}</style></head><body>""" + slides + "</body></html>", encoding="utf-8")
