"""All plots use saved date-aligned data and canonical annualized units."""
from pathlib import Path
import os
import tempfile

# A writable cache avoids user-home permission warnings in sandboxed runs.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finance-project-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from .units import NAMES


def plots(root, parameter_frame, vol_frame, histories, diagnostics, selection, horizon=21):
    output = Path(root) / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    dates = pd.to_datetime(parameter_frame.date)
    labels = ["Log drift / session", "Diffusion SD / sqrt(session)", "Jumps / session", "Mean log jump", "Log jump SD"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    for j, name in enumerate(NAMES):
        axes[j].plot(dates, parameter_frame[f"raw_mjd_{name}"], color="0.6", ls=":", label="Raw 5-parameter MLE")
        axes[j].plot(dates, parameter_frame[f"actual_{name}"], color="black", lw=1.5, label="BIC reference")
        for model in ["mean", "ridge", "lstm"]:
            axes[j].plot(dates, parameter_frame[f"{model}_{name}"], lw=1, alpha=.85, label=model)
        axes[j].set_ylabel(labels[j])
        axes[j].grid(alpha=.2)
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_title("Held-out calibration estimates and surrogate predictions (not true latent parameters)")
    axes[-1].set_xlabel("Information date, after close")
    fig.tight_layout()
    fig.savefig(output / "parameters.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    d = pd.to_datetime(vol_frame.date)
    ax.plot(d, vol_frame.actual_future_vol, color="black", lw=2, label=f"Next {horizon} sessions realized")
    models = list(dict.fromkeys(["historical_21d", "mle_total_vol", "direct_ridge", "direct_lstm", selection["forecast_model"]]))
    for model in models:
        ax.plot(d, vol_frame[model], lw=1.3, label=model)
    ax.set(title="Genuine future-volatility evaluation: every curve shares the same forecast origin",
           ylabel="Annualized volatility (decimal)", xlabel="Forecast origin after close")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "future_volatility.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for row, task in enumerate(["parameters", "volatility"]):
        sizes = sorted(histories.loc[histories.task == task, "hidden"].unique())
        for col, hidden in enumerate(sizes):
            ax = axes[row, col]
            hist = histories[(histories.task == task) & (histories.hidden == hidden)]
            ax.plot(hist.epoch, hist.train_eval_mse, label="Train, dropout off")
            ax.plot(hist.epoch, hist.validation_eval_mse, label="Validation, dropout off")
            ax.set(title=f"{task}: {hidden} LSTM units", xlabel="Epoch", ylabel="Standardized latent-target MSE")
            ax.legend(fontsize=8)
        for col in range(len(sizes), 2):
            axes[row, col].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model in ["historical_21d", "direct_ridge", "direct_lstm", "lstm_total_vol"]:
        ax.hist(vol_frame[model] - vol_frame.actual_future_vol, bins=14, alpha=.45, label=model)
    ax.axvline(0, color="black", lw=1)
    ax.set(title="Future-volatility errors; overlapping horizons, descriptive distribution",
           xlabel="Prediction minus realized annualized volatility", ylabel="Forecast count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "errors.png", dpi=150)
    plt.close(fig)

    diag = diagnostics[diagnostics.purpose == "current"].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    d = pd.to_datetime(diag.date)
    axes[0].plot(d, diag.bic_gaussian - diag.bic_mjd, lw=1)
    axes[0].axhline(0, color="black", ls="--")
    axes[0].set_ylabel("BIC Gaussian - BIC Merton")
    axes[0].set_title("Model identification and optimizer diagnostics (positive BIC difference favors jumps)")
    axes[1].scatter(d, diag.full_boundary.astype(int), s=8, label="Raw fit at a bound")
    axes[1].plot(d, diag.failed_starts, lw=1, label="Failed optimization starts")
    axes[1].set(ylabel="Flag / count", xlabel="Calibration end date")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "calibration_diagnostics.png", dpi=150)
    plt.close(fig)
