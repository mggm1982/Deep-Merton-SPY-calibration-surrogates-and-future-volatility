"""Features available after close t; labels use exactly t+1 through t+H."""
import numpy as np
import pandas as pd
from .data import window_at, future_at
from .units import realized_volatility

FEATURE_NAMES = ["mean_5", "vol_5", "mean_21", "vol_21", "vol_63", "vol_window",
                 "skew_window", "excess_kurtosis_window", "q05_window", "q95_window",
                 "last_return", "ewma_vol", "downside_rms_window"]


def ewma_volatility(returns, decay=.94, annualization=252):
    # Zero-mean second-moment EWMA, initialized from the first known squared return.
    r = np.asarray(returns)
    variance = np.empty(len(r))
    variance[0] = r[0] ** 2
    for i in range(1, len(r)):
        variance[i] = decay * variance[i - 1] + (1 - decay) * r[i] ** 2
    return np.sqrt(annualization * variance)


def make_inputs(observations, ends, config):
    returns = observations.log_return.to_numpy()
    ewma = ewma_volatility(returns, config["ewma_decay"], config["annualization"])
    sequences, stats = [], []
    for end in ends:
        w = window_at(returns, end, config["mle_window"])
        seq = window_at(returns, end, config["sequence_length"])
        sequences.append(np.column_stack([seq, np.abs(seq)]))
        centered, sd = w - w.mean(), w.std()
        if sd <= 0:
            raise ValueError("Constant return window cannot identify a distribution.")
        vol = lambda x: realized_volatility(x, config["annualization"])
        stats.append([w[-5:].mean(), vol(w[-5:]), w[-21:].mean(), vol(w[-21:]),
                      vol(w[-min(63, len(w)):]), vol(w), np.mean((centered / sd) ** 3),
                      np.mean((centered / sd) ** 4) - 3, *np.quantile(w, [.05, .95]),
                      w[-1], ewma[end], np.sqrt(np.mean(np.minimum(w, 0) ** 2))])
    x, s = np.asarray(sequences), np.asarray(stats)
    if not np.isfinite(x).all() or not np.isfinite(s).all():
        raise FloatingPointError("Nonfinite input features.")
    return x, s


def make_labels(observations, ends, config):
    returns = observations.log_return.to_numpy()
    return np.array([realized_volatility(future_at(returns, int(e), config["horizon"]),
                                        config["annualization"]) for e in ends])


def sample_metadata(observations, ends, config):
    w, h, s = config["mle_window"], config["horizon"], config["sequence_length"]
    d = observations.date
    return pd.DataFrame({"date": d.iloc[ends].to_numpy(), "origin_index": ends,
                         "source_row": observations.source_row.iloc[ends].to_numpy(),
                         "window_start": d.iloc[ends - w + 1].to_numpy(),
                         "sequence_start": d.iloc[ends - s + 1].to_numpy(),
                         "target_start": d.iloc[ends + 1].to_numpy(),
                         "target_end": d.iloc[ends + h].to_numpy()})
