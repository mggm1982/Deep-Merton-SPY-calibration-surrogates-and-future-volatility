import numpy as np
import pandas as pd
import torch
from finance_project.models import Preprocessor, Net, seed_everything, inference, fit_models, predict, save_models, load_models
from finance_project.evaluation import metric_row


def test_scalers_only_learn_fit_data():
    x = np.arange(48, dtype=float).reshape(4, 6, 2)
    s = np.arange(12, dtype=float).reshape(4, 3)
    y = np.array([.1, .2, .3, .4])
    prep = Preprocessor().fit(x, s, y, "volatility")
    prep.inputs(x + 1e6, s + 1e6)
    prep.targets(y * 1e6)
    np.testing.assert_array_equal(prep.sequence_scaler.mean_, x.reshape(-1, 2).mean(0))
    np.testing.assert_array_equal(prep.stats_scaler.mean_, s.mean(0))
    np.testing.assert_allclose(prep.target_scaler.mean_, np.log(y).mean())
    np.testing.assert_allclose(prep.decode(prep.targets(y)), y)


def test_lstm_shapes_determinism_and_dropout_mode():
    seed_everything(42)
    net = Net(8, 3, 5, .5)
    x, s = np.ones((4, 21, 2)), np.ones((4, 3))
    p = inference(net, x, s)
    assert p.shape == (4, 5)
    np.testing.assert_array_equal(p, inference(net, x, s))
    assert not net.training
    net.train()
    with torch.no_grad():
        a = net(torch.tensor(x, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)).numpy()
        b = net(torch.tensor(x, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)).numpy()
    assert not np.array_equal(a, b)


def test_training_history_and_model_reload(config, tmp_path):
    c = {**config, "epochs": 5, "patience": 3, "lstm_hidden_sizes": [4], "ridge_alphas": [1, 10]}
    rng = np.random.default_rng(1)
    x, s = rng.normal(size=(20, 6, 2)), rng.normal(size=(20, 3))
    y = np.exp(.1 * s[:, 0] - 2)
    bundle, net, history, trials = fit_models((x[:14], s[:14], y[:14]), (x[14:], s[14:], y[14:]), "volatility", c)
    assert np.isfinite(history.select_dtypes(include=[np.number])).all().all()
    save_models(tmp_path, "volatility", bundle, net)
    loaded = load_models(tmp_path, "volatility")
    pred = predict(bundle, net, x[14:], s[14:])
    again = predict(*loaded, x[14:], s[14:])
    for name in pred:
        np.testing.assert_array_equal(pred[name], again[name])
        assert (pred[name] > 0).all()
    # Restored checkpoint's inference validation loss must match the recorded epoch.
    prep = bundle["preprocessing"]
    vx, vs = prep.inputs(x[14:], s[14:])
    score = np.mean((inference(net, vx, vs) - prep.targets(y[14:])) ** 2)
    best_epoch = int(trials[trials.model == "lstm"].best_epoch.iloc[0])
    assert np.isclose(score, history.loc[history.epoch == best_epoch, "validation_eval_mse"].iloc[0])


def test_metrics_use_declared_benchmark_and_bias_direction():
    y, p, b = np.array([1., 2, 3]), np.array([2., 3, 4]), np.array([0., 0, 0])
    m = metric_row(y, p, b)
    assert m["bias"] == 1 and m["MAE"] == 1 and m["RMSE"] == 1
    assert np.isclose(m["R2_oos"], 1 - 3 / 14)
    constant = metric_row(y, np.ones(3), b)
    assert constant["correlation"] is None
    assert "constant series" in constant["undefined_reason"]
