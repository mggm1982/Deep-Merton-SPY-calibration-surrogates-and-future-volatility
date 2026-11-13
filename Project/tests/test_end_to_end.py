import pandas as pd
from finance_project.pipeline import run_pipeline
from finance_project.verification import verify


def test_small_end_to_end_from_raw_data(root, config, tmp_path):
    source = tmp_path / "source"
    (source / "data/raw").mkdir(parents=True)
    raw = pd.read_csv(root / "data/raw/spy_train_data.csv")
    raw.iloc[:240].to_csv(source / "data/raw/spy_train_data.csv", index=False)
    raw.iloc[240:340].rename(columns={"Date": "Price"}).to_csv(source / "data/raw/spy_test_data.csv", index=False)
    c = {**config, "mle_window": 63, "horizon": 5, "stride": 10, "epochs": 3,
         "patience": 2, "lstm_hidden_sizes": [4], "ridge_alphas": [1.]}
    out = tmp_path / "output"
    manifest = run_pipeline(source, out, c)
    assert manifest["warning_count"] == 0
    assert manifest["selection"]["test_used_for_tuning"] is False
    assert len(verify(out)) >= 10
    assert (out / "results/figures/future_volatility.png").is_file()
    assert (out / "models/parameters_lstm.pt").is_file()
