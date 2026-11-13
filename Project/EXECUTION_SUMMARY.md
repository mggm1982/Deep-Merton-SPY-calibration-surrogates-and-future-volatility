# Submission-ready execution summary

Verified on 28 October 2026, using Python 3.12.14 and the pinned CPU environment.

- Original project reproduced before repairs; raw source files preserved unchanged.
- Full repaired pipeline completed from raw data.
- 26 automated tests passed, including numerical, alignment, leakage, reload and end-to-end checks.
- 11 independent saved-artifact checks passed.
- Every notebook code cell executed in a fresh local Jupyter kernel after removing generated models and results from the extracted package.
- All five validation/test/example prediction files reproduced byte-for-byte.
- All five diagnostic figures visually inspected.
- No Python warnings were captured by the final pipeline. The packaging harness logged a kernel-parent shutdown notice after successful completion; this was kernel cleanup, not a failed notebook cell. First-run Matplotlib font-cache creation was also informational.

The direct volatility LSTM test RMSE was 0.026553, versus 0.029137 for historical volatility. However, validation selected the MLE volatility proxy, whose test RMSE was 0.050488. Selection was preserved rather than changed after seeing test performance. Thus the model-selection result does not establish consistent forecasting improvement.

Raw Merton fits still touched bounds in 165 of 211 evaluated calibration windows, and the simpler BIC rule selected jumps in 69. These results remain visible. The returned software is tested; weak identification, regime changes and limited data are scientific limitations, not exceptions to conceal.

The supplied 2023 test set had been examined before this repair. It is excluded from fitting and tuning here, but is not a new unseen holdout. A hosted Google Colab service session and optional live-option comparison were not executed.

See README.md for installation and run commands, DEBUG_REPORT.md for the audit, MODELING.md for conventions, report/project_summary.md for generated results, and results/logs/delivery_verification.json for delivery evidence. Finance_Project.ipynb is the portable notebook; Finance_Project.executed.ipynb records the successful local execution.
