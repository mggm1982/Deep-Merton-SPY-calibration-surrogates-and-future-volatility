# SPY Merton calibration and future-volatility evaluation

This project estimates a historical physical-measure return distribution and learns an approximation to that calibration. A separate supervised task predicts future realized volatility. Neither task predicts prices or establishes trading profitability.

## Method

Use the supplied chronological CSVs unchanged. Recompute decimal log returns from Close. Exclude only the first price-only row, whose preceding close is unavailable. The raw adjustment/dividend policy cannot be independently established. Features are known after the forecast-origin close.

Calibrate the exact compound-Poisson normal mixture on 126 past observed returns at stride 5. Use scaled multistart constrained MLE, analytic derivatives, log-sum-exp and a Poisson tail tolerance of 1e-12. Preserve all raw fits and optimizer attempts. Compare the full five-parameter fit to an exact two-parameter Gaussian with BIC. The latter admits lambda=0 and records unused jump parameters as canonical zeros. BIC in this boundary mixture is a model-selection heuristic, not a formal jump test. Canonical zeros do not prove a zero-sized jump distribution.

Train mean, tuned Ridge and small LSTM models on the selected historical reference parameters. The LSTM uses 21 returns and absolute returns plus historical summary statistics. The reference parameters are estimated labels, not ground truth. Direct Ridge/LSTM volatility models use the same historical inputs and the actual following 21 returns as targets. Positive predictions use documented output links, with no post-prediction clipping.

Each future label is a demeaned sample SD with ddof=1, multiplied by sqrt(252). Model total volatility is sqrt(A*(sigma² + lambda*(mu_j² + sigma_j²))). These are compatible variance targets; expected sample SD need not equal the square root of expected variance. Parameter constancy across a future horizon is only a proxy assumption.

Fit/validation/test sizes: 134/31/46. Purge forward labels that cross the next partition's first forecast origin. Historical feature windows may overlap across partitions because that history was already known. Preprocessing uses fit samples only. Checkpoints, regularization and model family are selected on validation; models then freeze.

## Actual test results

Forecast model selected on validation: **mle_total_vol**. It did not improve the historical-volatility baseline on this fixed test rerun. Selection was not changed after observing these results. R2_oos uses historical_21d as the declared benchmark; ordinary R2 uses the evaluation sample mean and is saved separately.

| model | n | MAE | RMSE | bias | R2_oos |
| --- | --- | --- | --- | --- | --- |
| direct_lstm | 46 | 0.023191 | 0.026553 | 0.010763 | 0.169487 |
| direct_ridge | 46 | 0.024167 | 0.028088 | 0.009254 | 0.070707 |
| ewma | 46 | 0.024097 | 0.028179 | 0.009395 | 0.064625 |
| historical_21d | 46 | 0.023669 | 0.029137 | 0.006252 | 0.000000 |
| lstm_total_vol | 46 | 0.025338 | 0.029806 | 0.009660 | -0.046460 |
| mean_total_vol | 46 | 0.027362 | 0.031435 | 0.014552 | -0.163988 |
| ridge_total_vol | 46 | 0.030913 | 0.034349 | 0.010317 | -0.389816 |
| mle_total_vol | 46 | 0.044278 | 0.050488 | 0.036432 | -2.002556 |
| raw_mjd_total_vol | 46 | 0.044577 | 0.050878 | 0.036738 | -2.049225 |
| training_mean | 46 | 0.056934 | 0.063378 | 0.056924 | -3.731434 |

Errors are prediction minus observation. Decimal 0.01 is one annualized volatility percentage point. Full per-parameter results and reconstructed historical total-volatility results are in results/metrics. Correlation is undefined for a constant series; blank metric cells have explicit reasons, not unhandled missing values.

## Numerical and statistical limitations

Raw five-parameter fits at bounds: 165/211. BIC selected a jump component in 69/211 windows. Failed optimizer starts (including nonselected starts) total 0; every attempt is retained. No window is dropped for poor fit or poor forecast performance.

Longer windows reduce small-sample instability but mix changing volatility regimes. Adjacent windows and 21-session future targets overlap. These metrics are descriptive; no IID confidence interval or significance claim is made. A nonoverlapping subsequence sensitivity table is included, but its smaller sample still does not establish deployment reliability. The dataset is too small for strong deep-learning conclusions.

The supplied 2023 test results had already been examined in the original project. This repair keeps them out of fitting, scaling and tuning, but cannot turn them into genuinely unseen data. A new future dataset is needed for a fresh scientific holdout. No raw data were replaced and no scores were manufactured.

## Reproduction

Run `python run_project.py`, `python -m pytest -q`, and `python src/verify_project.py`. Saved weights and preprocessing reproduce inference exactly in the tested environment. The manifest records versions, partitions and raw hashes. The CLI and notebook share source modules. A hosted Google Colab service session and the guide's optional live option comparison are not executed here. The source runs offline after dependency installation.
