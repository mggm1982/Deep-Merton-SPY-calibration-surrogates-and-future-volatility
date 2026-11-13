# Debug and completion report

## Repository audit and reconstructed pipeline

The supplied repository contained `src/complete_pipeline.py` (one monolithic implementation), `src/verify_project.py`, one Colab notebook, config/requirements, two raw SPY CSVs, the supplied guide, processed data, model artifacts, four figures and three reports. `audit/original_tree.txt` records every original file. Original source and configuration snapshots are retained as audit evidence; they are not used by the repaired entry point.

The original actual pipeline was: load supplied prices → recompute log returns → take 21-return windows every five observations → two-start bounded MLE on each same input window → fit-only per-lag/parameter scaling → fixed-alpha Ridge and 16-unit LSTM → clip predictions to MLE bounds → select on validation parameter imitation → evaluate 2023 parameter imitation → secondarily compare frozen parameter total-volatility proxies to the following 21 returns. The notebook invoked that script; it did not contain competing mathematical formulas, but it depended unconditionally on Colab and a fixed extraction folder.

The repaired pipeline retains calibration approximation as one task, adds a separately supervised future-volatility task, and makes both use the same explicit chronological sample metadata and shared mathematical modules.

## Reproduction before repair

1. Running the unchanged script using the machine's default Python 3.14.3 failed with `ModuleNotFoundError: No module named 'torch'`. This is an environment error, not evidence against the model. See `audit/initial_failure.log`.
2. The original notebook contained a saved “Failed to connect to the remote Jupyter Server” error. It happened before its first upload cell executed. Source changes cannot restore an expired hosted runtime; the new notebook and README describe fresh local/Colab setup.
3. A new Python 3.12.14 environment installed the original pinned versions successfully. These pins were **not** fictitious or inherently unavailable. The original code was run without edits in an isolated project copy. It completed. The exact command, output and recalculated diagnostics are in `audit/original_pipeline.log`, `audit/original_reproduction_summary.json` and associated original CSVs.
4. All 240 original calibrations reported optimizer success; **228/240 hit bounds**. The original test historical-volatility RMSE was about 0.02914; the local rerun's LSTM total-volatility RMSE was about 0.04348. Local optimization/model outputs differed from the old Linux artifacts despite the same seed and direct package versions. Cross-platform bitwise reproduction was never guaranteed.

The pre-repair ranking was: (1) weak identification from 21 observations plus restrictive bounds; (2) mismatch between parameter-imitation selection and forecast usefulness; (3) clipped outputs and incomparable loss curves; (4) weak benchmark/tuning design; (5) missing executable mathematical/leakage tests and notebook/environment fragility. The source's existing compound-Poisson formula, total-variance formula and basic future-target slicing were already substantially correct; they were verified and centralized, not labeled as defects without evidence.

## Findings, corrections and verification

| Observed issue | Root cause and correction | Evidence / affected modules |
| --- | --- | --- |
| Import error / notebook fails before Python | Missing PyTorch in default Python; saved remote kernel connection and unconditional Colab APIs. Use a clean pinned Python 3.12 environment and a portable notebook with local/Colab branches and automatic root discovery. | requirements files; README; notebooks/Finance_Project.ipynb; original failure log |
| MLE boundary spikes and discontinuous jump labels | Five parameters from 21 observations are weakly identified. Use a predeclared 126-session window, exact Gaussian submodel, conservative BIC selection, multistart optimization and retained raw-fit diagnostics. This is a modelling correction, not a promise to eliminate boundaries. | merton.calibrate; config.json; calibration_diagnostics.csv; optimizer_attempts.csv |
| Zero jumps unavailable | Original lambda lower bound 0.0001 excluded zero, and log(lambda) was undefined at zero. Stable xlogy weights plus an exact Gaussian candidate permit lambda=0. Unidentified Gaussian jump parameters have documented canonical zeros. | merton.log_weights/logpdf/calibrate; zero-jump test |
| Likelihood reliability untested | Fixed 12 terms were adequate for the old lambda cap, but not adaptive to changed bounds. Derive sufficient support from the new cap and tail tolerance; use log-sum-exp and analytic gradients. Never floor tiny densities. | normalization, integration, gradient, tail and simulated-recovery tests |
| Train loss declines while validation stays high | Original train loss used dropout and unclipped standardized outputs before the update; validation used dropout off, inverse scaling and clipping after the update. Record both evaluation losses in the same latent space, after each update, dropout off; keep stochastic objective separate. | models.fit_models; training_history.csv; checkpoint test |
| Ridge noise / impossible predictions | Raw standardized linear outputs were silently clipped; alpha=10 was fixed. Use fit-only scaling, validation-tuned alpha, exp/square output links, signed means and visible extrapolation. | units.ParameterTransform; models.Preprocessor/fit_models; constraint and scaler tests |
| Excessively smooth LSTM predictions | May arise from conditional averaging, weak labels or overfitting, not a proven software bug. Compare 8/16 units, weight decay, clipping gradients, early stopping and best-checkpoint restore. Keep any remaining poor performance in the results. | model_candidates.csv; comparable loss plots; native parameter and volatility errors |
| Parameter imitation does not imply forecasting skill | Old selection optimized historical labels. Preserve that question and train separate direct future-volatility Ridge/LSTM models. Compare with fit mean, historical-volatility persistence, EWMA and MLE proxies. | pipeline selection saved before test; future_volatility.csv |
| Risk of new forward-label leakage | Adding future supervised labels requires an H-session boundary purge. Exclude fit/validation origins whose forward targets enter the next split; use historical context only at each origin. | data.make_partitions; features.make_labels; perturb-future and boundary tests |
| Variance/volatility and daily/annual ambiguity | Centralize the existing correct compound-Poisson variance identity. State diffusion log drift, jumps/session, invariant jump sizes, annualization and ddof=1 future SD. Do not confuse instantaneous SD with expected finite-window sample SD. | units.py; MODELING.md; formula and unit conversion tests |
| Positional joins can silently misalign series | Carry actual origin, window start, sequence start, target start/end and source row. Join on unique dates; reject missing/duplicate keys. | evaluation.aligned_join; verification.py; date-order and join tests |
| Return audit precision | A few supplied serialized returns differ from recomputed returns by up to 1.17e-8. Set an explicitly documented 1e-6 audit tolerance (0.01 basis point) and always use recomputed returns. Do not modify raw prices or returns. | data.load_data; raw hashes and return tests |
| Missing independent verification | The old verifier largely rechecked outputs/assertions and had no numerical/leakage test suite. Add tests and independent recomputation of labels, scalers, model outputs, dates, metrics and constraints. | tests/; src/verify_project.py; verification.json |
| Stale hand-written metrics / missing bias metrics | Generate summary and HTML presentation from each run's actual results. Save bias, error quantiles and benchmark-relative OOS R², with explicit reasons for undefined statistics. | reporting.py; evaluation.py; report/project_summary.md |

## Repair-stage issues caught by tests

The first stricter return audit used a 1e-8 threshold and rejected harmless serialized precision differences; the corrected tolerance and rationale are documented above. The new single-output volatility Ridge path exposed scikit-learn's one-dimensional prediction shape; inverse scaling now restores `(n,1)` explicitly. A simulated-recovery assertion initially used an unsupported array tolerance formatter; it now checks each parameter against the same predefined tolerance individually. These were repair-stage issues, not alleged defects in the original single-task code.

During local execution, reading a newly installed environment from Documents caused long dependency import delays. A clean temporary environment was installed from the already-downloaded wheels; this changed runtime location, not dependencies, source, datasets or model settings. No temporary runtime path is required by the returned project.

## Execution evidence

- `results/logs/run.log`: full repaired raw-data-to-results execution.
- `results/logs/tests.log`: automated test results, including the end-to-end smoke test.
- `results/logs/warnings.json`: recorded runtime warnings, including library warnings rather than suppressing them.
- `results/data_audit/verification.json`: independent artifact checks.
- `run_manifest.json`: actual versions, platform, configuration, date partitions, hashes and selection decisions.
- `results/logs/delivery_verification.json`: final notebook, extracted-package and reproducibility checks performed during delivery.

The original MLE code, supplied raw files and guide remain available as evidence. The runnable corrected source is in `src/finance_project`; the notebook imports it. No external local dataset, cached model, hidden notebook state or machine-specific path is needed.

## Remaining scientific and operational limits

Boundary hits, optimizer start disagreements and poor ML forecasts may remain. They are recorded, not concealed. A longer MLE window trades identification variance for regime mixing. BIC is nonregular at the mixture boundary; it is a parsimony heuristic. No short market sample can certify the “true” five parameters, and Gaussian canonical jump-size targets should not be mistaken for observed zero jump sizes.

The supplied test set was already inspected in the original project. This implementation freezes choices before using its targets, but cannot make prior exposure disappear. Results require confirmation on genuinely new data. Overlapping forecasts are dependent; no uncorrected confidence intervals or significance claims are made. No option quotes were supplied, so no live option-market or risk-neutral pricing comparison is claimed. The hosted Colab service itself remains unexecuted locally; the notebook's shared-source workflow is checked locally.

## Final delivery result

All 26 tests and all 11 independent artifact checks pass. Every notebook cell executed from the extracted ZIP after removing generated artifacts. All five prediction files reproduced byte-for-byte. See EXECUTION_SUMMARY.md and results/logs/delivery_verification.json. The final pipeline captured no Python warnings; a post-execution Jupyter parent-shutdown notice in the packaging log is a harness cleanup message, not a project failure.
