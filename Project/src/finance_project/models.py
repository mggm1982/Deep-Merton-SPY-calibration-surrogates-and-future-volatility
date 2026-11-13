"""Fit-only scaling, chronological validation, constrained output links."""
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .units import ParameterTransform


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class Preprocessor:
    def fit(self, sequence, stats, targets, task):
        self.task = task
        self.sequence_scaler = StandardScaler().fit(sequence.reshape(-1, sequence.shape[-1]))
        self.stats_scaler = StandardScaler().fit(stats)
        self.target_scaler = StandardScaler().fit(self.encode(targets))
        self.fit_sample_count = len(sequence)
        self.feature_condition_number = float(np.linalg.cond(self.stats_scaler.transform(stats)))
        return self

    def encode(self, y):
        if self.task == "parameters":
            return ParameterTransform().transform(y)
        if np.any(np.asarray(y) <= 0):
            raise ValueError("Future volatility targets must be positive.")
        return np.log(np.asarray(y)).reshape(-1, 1)

    def decode(self, y):
        # sklearn may return (n,) for a single output even when fit received
        # (n, 1). Preserve the target axis expected by StandardScaler.
        y = np.asarray(y)
        if self.task == "volatility":
            y = y.reshape(-1, 1)
        z = self.target_scaler.inverse_transform(y)
        if self.task == "parameters":
            return ParameterTransform().inverse_transform(z)
        with np.errstate(over="raise", invalid="raise"):
            result = np.exp(z).reshape(-1)
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise FloatingPointError("Invalid positive volatility output.")
        return result

    def inputs(self, sequence, stats):
        x = self.sequence_scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
        return x, self.stats_scaler.transform(stats)

    def targets(self, y):
        return self.target_scaler.transform(self.encode(y))


class Net(nn.Module):
    """Small LSTM with historical summary features at the output head."""
    def __init__(self, hidden, n_stats, n_outputs, dropout):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + n_stats, n_outputs)

    def forward(self, sequence, stats):
        states, _ = self.lstm(sequence)
        return self.head(self.dropout(torch.cat([states[:, -1], stats], dim=1)))


def tensor(array):
    return torch.tensor(array, dtype=torch.float32)


def inference(net, sequence, stats):
    net.eval()
    with torch.no_grad():
        return net(tensor(sequence), tensor(stats)).numpy()


def fit_models(train, validation, task, config):
    x, s, y = train
    vx, vs, vy = validation
    prep = Preprocessor().fit(x, s, y, task)
    tx, ts = prep.inputs(x, s)
    vx, vs = prep.inputs(vx, vs)
    ty, vty = prep.targets(y), prep.targets(vy)
    ridge_x = np.column_stack([tx[:, :, 0], ts])
    ridge_vx = np.column_stack([vx[:, :, 0], vs])
    trials, histories, ridge_best, score_best = [], [], None, np.inf

    def selection_score(pred):
        if task == "parameters":
            # Same transformed, standardized space used for fitting and selection.
            return float(np.mean((pred - vty) ** 2))
        return float(np.mean((prep.decode(pred) - vy) ** 2))

    for alpha in config["ridge_alphas"]:
        model = Ridge(alpha=alpha).fit(ridge_x, ty)
        score = selection_score(model.predict(ridge_vx))
        trials.append({"task": task, "model": "ridge", "alpha": alpha, "hidden": 0,
                       "best_epoch": 0, "validation_score": score})
        if score < score_best:
            ridge_best, score_best, best_alpha = model, score, alpha
    net_best, score_best = None, np.inf
    for hidden in config["lstm_hidden_sizes"]:
        seed_everything(config["seed"])
        net = Net(hidden, ts.shape[1], ty.shape[1], config["dropout"])
        opt = torch.optim.AdamW(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        lossfn = nn.MSELoss()
        best_loss, checkpoint, wait, best_epoch = np.inf, None, 0, 0
        # Independent windows in chronological order, full-batch updates; no carried state.
        for epoch in range(1, config["epochs"] + 1):
            net.train()
            opt.zero_grad()
            objective = lossfn(net(tensor(tx), tensor(ts)), tensor(ty))
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite training objective.")
            objective.backward()
            nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            opt.step()
            train_pred, val_pred = inference(net, tx, ts), inference(net, vx, vs)
            train_mse, val_mse = float(np.mean((train_pred - ty) ** 2)), float(np.mean((val_pred - vty) ** 2))
            score = selection_score(val_pred)
            histories.append({"task": task, "hidden": hidden, "epoch": epoch,
                              "train_eval_mse": train_mse, "validation_eval_mse": val_mse,
                              "dropout_training_objective": float(objective.detach()),
                              "validation_selection_score": score})
            if score < best_loss - 1e-10:
                best_loss, best_epoch, checkpoint, wait = score, epoch, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
            if wait >= config["patience"]:
                break
        net.load_state_dict(checkpoint)
        net.eval()
        trials.append({"task": task, "model": "lstm", "alpha": 0, "hidden": hidden,
                       "best_epoch": best_epoch, "validation_score": best_loss})
        if best_loss < score_best:
            net_best, score_best, architecture = net, best_loss, {"hidden": hidden, "n_stats": ts.shape[1],
                         "n_outputs": ty.shape[1], "dropout": config["dropout"]}
    bundle = {"preprocessing": prep, "ridge": ridge_best, "mean": np.mean(y, axis=0),
              "architecture": architecture, "ridge_alpha": best_alpha, "task": task}
    return bundle, net_best, pd.DataFrame(histories), pd.DataFrame(trials)


def predict(bundle, net, sequence, stats):
    prep = bundle["preprocessing"]
    x, s = prep.inputs(sequence, stats)
    result = {"ridge": prep.decode(bundle["ridge"].predict(np.column_stack([x[:, :, 0], s]))),
              "lstm": prep.decode(inference(net, x, s))}
    mean = np.asarray(bundle["mean"])
    result["mean"] = np.repeat(mean[None], len(x), axis=0) if mean.ndim else np.full(len(x), mean)
    return result


def save_models(directory, task, bundle, net):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / f"{task}.joblib")
    torch.save(net.state_dict(), directory / f"{task}_lstm.pt")


def load_models(directory, task):
    directory = Path(directory)
    bundle = joblib.load(directory / f"{task}.joblib")
    net = Net(**bundle["architecture"])
    net.load_state_dict(torch.load(directory / f"{task}_lstm.pt", map_location="cpu", weights_only=True))
    net.eval()
    return bundle, net
