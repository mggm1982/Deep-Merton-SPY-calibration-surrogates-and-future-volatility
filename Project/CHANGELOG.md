# Changelog

## 2.0.0 — 2026-10-28

- Audited and reproduced the original project before repair; preserved its source/configuration evidence and all raw data.
- Split the monolithic implementation into shared data, units, MLE, features, learning, evaluation, plotting and reporting modules.
- Added longer rolling MLE, an exact zero-jump Gaussian alternative, adaptive Poisson support, analytic gradients and complete optimizer diagnostics.
- Replaced silent prediction clipping with explicit valid output links.
- Added Ridge tuning, smaller LSTM comparison, early stopping, weight decay, gradient clipping and comparable inference-mode losses.
- Added separately supervised future-volatility forecasts with chronological label purges and strong simple baselines.
- Added date-keyed predictions, richer error metrics, independent verification and automated numerical/leakage/end-to-end tests.
- Rebuilt the portable notebook, documented exact installation/run commands, and generated reports from actual execution.
- Explicitly documented earlier test exposure, parameter identifiability, raw-data provenance and overlap limitations.
