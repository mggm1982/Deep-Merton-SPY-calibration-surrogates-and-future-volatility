# Completed SPY Deep Merton project

See `EXECUTION_SUMMARY.md` for the final verification record. `notebooks/Finance_Project.executed.ipynb` also preserves the successful local notebook run.

Start here. This folder contains the complete source, unchanged supplied data, trained models, generated results, tests, notebook, reports and execution evidence.

Two questions are evaluated separately:

1. Can mean, Ridge and LSTM models approximate rolling historical Merton calibration?
2. Can direct volatility models or calibrated total-volatility proxies predict the next 21 observed sessions better than simple baselines?

Read `report/project_summary.md` for the **actual results of the included run**, and `DEBUG_REPORT.md` for defects, corrections and verification. Complex models are not assumed to win.

## Main workflow: Google Colab

1. Keep the complete `finance_project_completed.zip` together. Extract it on your computer.
2. Open [Google Colab](https://colab.research.google.com/). Choose **File → Upload notebook** and select `finance_project_completed/notebooks/Finance_Project.ipynb`.
3. Use **Connect → Connect to a hosted runtime**. A connection error occurs before project Python executes: reconnect or start a fresh runtime. Do not reconnect to the old remote-server URL saved in the original notebook.
4. Run the notebook cells in order. The first cell requests the complete project ZIP and finds the extracted project automatically. It does not require a particular `/content/finance_project` directory.
5. The dependency cell installs pinned requirements. On a fresh Linux Colab runtime, it first installs the CPU PyTorch wheel to avoid unnecessary GPU packages. This project deliberately runs on CPU.
6. The run cell executes the same `run_project.py` used locally. It recalibrates, trains, validates, freezes model choices, evaluates test dates, and regenerates tables and plots. Allow several minutes on a fresh runtime.
7. The verification cell runs automated tests and independently recomputes metrics and date alignment. Subsequent cells display reports, figures and a saved-model prediction.
8. The final cell offers a ZIP backup of the completed run. Download it before closing the Colab session.

The notebook has been executed locally using the shared Python source. A hosted Colab service session is not part of the local verification; its availability and networking are outside the project.

## Local workflow (Python 3.12 recommended)

Open a terminal in this folder. Use Python 3.12, which was tested. The operating system's default Python is not necessarily your project environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_project.py
python -m pytest -q
python src/verify_project.py
python run_project.py --example
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\activate`. Use the remaining `python` commands unchanged. Do not copy an existing virtual environment across computers.

For Linux CPU-only installation, before installing the requirements:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

The requirements pin direct dependencies. `requirements-lock-macos-arm64.txt` additionally records the exact full local environment; it is platform-specific, not the recommended Linux/Windows installation file. No data download, credentials, GPU, external files or network access are required after installing dependencies.

The script resolves all paths from its own project folder, so an absolute script invocation also works from another directory. `python src/complete_pipeline.py` remains a compatibility entry point. Use `python run_project.py --output /path/to/new_run` for separate generated outputs; source data are copied into that run. `--config path/to/config.json` supports an alternate configuration. Do not tune settings using final test results.

## What is in the folder?

```text
finance_project_completed/
  README.md, DEBUG_REPORT.md, CHANGELOG.md, MODELING.md
  config.json, requirements.txt, requirements-lock-macos-arm64.txt
  run_project.py, run_manifest.json, pytest.ini
  data/raw/              unchanged CSVs and supplied Guide.pdf
  data/processed/        validated returns, aligned targets, example history
  src/finance_project/   data, units, MLE, features, models, evaluation, plots
  src/complete_pipeline.py, src/verify_project.py
  tests/                 numerical, timing, model and end-to-end tests
  notebooks/Finance_Project.ipynb
  models/                both tasks' preprocessing, Ridge and LSTM weights
  results/data_audit/     data, optimizer attempts, boundaries, verification
  results/metrics/        parameter/volatility metrics and candidate selection
  results/predictions/    validation/test predictions with true dates
  results/figures/        regenerated diagnostics
  results/logs/           execution, warnings, test and delivery logs
  report/                generated summary, presentation, viva answers
  audit/                 original prompt, source snapshots, failures and rerun
```

All model `.joblib` and `.pt` files are required for saved inference. Only load serialized model files from a trusted source. The notebook contains no separate implementation of the mathematics.

## Reading the results

- `future_volatility.csv`: MAE, RMSE, bias, correlation, ordinary R², and out-of-sample R² relative to **historical_21d**. All volatility values are annualized decimals; 0.01 is one percentage point.
- `parameter_metrics.csv`: errors against **estimated historical references**, not observed true latent parameters. Parameter OOS R² uses the fit-parameter-mean benchmark.
- `reconstructed_volatility.csv`: calibration-surrogate total volatility against historical reference total volatility.
- `nonoverlapping_volatility.csv`: sensitivity on every fifth origin (default configuration); only descriptive because the sample is small.
- `model_candidates.csv` and `selection.json`: validation-only alpha, network/checkpoint and model-family choices.
- `training_history.csv`: comparable train and validation MSE with dropout **disabled for both**, evaluated after the same update. The stochastic training objective is a separate column.
- `calibration_diagnostics.csv` and `optimizer_attempts.csv`: full MLE likelihood, Gaussian/Merton BIC, boundary flags, omitted mixture mass, each start's convergence and messages.

Only mathematically undefined statistics (for example correlation with a constant mean baseline) are blank, with an `undefined_reason`. Prediction tables must contain no NaN/inf. No rows are removed for bad MLE or model performance.

## Important limitations

The original 2023 test outputs were already exposed before this repair. This run excludes test outcomes from fitting, scaling and selection, but it is a **fixed rerun holdout**, not a newly unseen dataset. A new independent dataset is needed to validate generalization.

The small dataset and overlapping windows limit deep-learning evidence. Jump parameters can remain weakly identified even with 126 returns. BIC's no-jump choice is a conservative heuristic, not proof that jumps never occur. The raw Close adjustment/dividend policy is unknown. No trading strategy, profitability, risk-neutral option calibration or market-pricing claim is made. The optional live-option comparison is not executed.
