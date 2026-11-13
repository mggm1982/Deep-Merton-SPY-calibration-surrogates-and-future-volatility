# Model, units and information timing

## Physical-measure convention

For one observed trading session:

`r_t = mu_log + sigma * epsilon_t + sum_{i=1}^{N_t} Y_i`

`epsilon_t ~ N(0,1)`, `N_t ~ Poisson(lambda)`, `Y_i ~ N(mu_j, sigma_j^2)`, independently.

The general interval form replaces drift and intensity with `mu_log * dt`, `lambda * dt`, and diffusion SD with `sigma * sqrt(dt)`. Here **dt=1 observed session**, including Monday returns spanning a weekend. There is no separate calendar-day adjustment or missing-session imputation.

`mu_log` is uncompensated diffusion LOG drift, not the full mean log return and not expected arithmetic asset return. `E[r] = mu_log + lambda*mu_j`. Under the compensated arithmetic-drift convention `dS/S = mu*dt + sigma*dW + (exp(Y)-1)*dN - lambda*kappa*dt`, it corresponds to `mu_log = mu - sigma^2/2 - lambda*kappa`, where `kappa=exp(mu_j+sigma_j^2/2)-1`. Estimating `mu_log` freely is an equivalent reparameterization for this physical return likelihood. No risk-neutral identification is assumed.

| Parameter | Native unit | Annual conversion (A=252) |
| --- | --- | --- |
| mu_log | log drift / session | multiply by A |
| sigma | diffusion SD / sqrt(session) | multiply by sqrt(A) |
| lambda | expected jumps / session | multiply by A |
| mu_j | mean log jump size | unchanged |
| sigma_j | log jump size SD | unchanged |

Lambda is an intensity, not the probability of a jump. The probability of at least one jump in a session is `1-exp(-lambda)`.

## Likelihood and calibration

Conditional on k jumps, `r ~ Normal(mu_log + k*mu_j, sigma^2 + k*sigma_j^2)`. The unconditional density is the Poisson-weighted sum. Use `xlogy` and `gammaln` for stable log weights including lambda=0 and log-sum-exp for the mixture. No PDF flooring changes the objective.

Choose the number of terms from the configured maximum lambda so the discarded Poisson tail is at most 1e-12. Use this same sufficient support during optimization to avoid discontinuous truncation. The public density adapts to its own supplied lambda. With the default maximum intensity 1/session, 15 terms suffice. The omitted mass is recorded; weights are not secretly renormalized.

Optimize on scaled coordinates with analytic derivatives and three fixed starts, plus the immediately preceding available fit within a split as a warm start. Bounds in config are explicit numerical/statistical guardrails: sigma >= 0.0001 prevents degenerate mixture spikes; lambda <= 1 defines an occasional-jump model; signed jump means and jump SD have broad bounds. Bound hits remain diagnostics, not successful parameter identification. The exact two-parameter Gaussian candidate allows lambda=0. The interior numerical lambda floor of 1e-8 is only for the derivative `k/lambda` in jump fits and cannot prevent the exact zero-jump candidate.

Return both the best raw full MLE (including the Gaussian submodel) and a BIC-selected reference. BIC is `2*NLL + k*log(n)` with k=5 or 2. Because the mixture is nonregular at zero intensity, interpret this as a conservative parsimony heuristic, not a chi-square test or posterior probability. Gaussian jump parameters are stored as canonical zeros; their true values are unidentifiable when lambda=0. A selected Gaussian's jump sizes should not be interpreted economically.

All starts, convergence messages, objective values and boundary flags are retained. If every jump fit fails, execution stops instead of silently substituting a successful result. A failed nonselected start is exposed in the attempt table. Optimizer success is not proof of a global optimum. Simulated recovery is tested in a deliberately identifiable large sample; recovery from short SPY windows is not guaranteed.

## Volatility definitions

The law of total variance gives `Var(sum Y) = E[N]*Var(Y) + Var(N)*E[Y]^2 = lambda*(sigma_j^2+mu_j^2)`. Thus:

`V_daily = sigma^2 + lambda*(sigma_j^2 + mu_j^2)`

`sigma_total_annual = sqrt(252 * V_daily)`

This is log-return total variance including jumps. Do not add the squared diffusion drift. If parameters were annualized first, do not multiply the resulting variance by 252 again.

The future reference at information date t is the sample SD of exactly `r[t+1], ..., r[t+H]`, demeaned within that window with ddof=1, annualized by sqrt(252). Default H=21. No additional sqrt(H) factor belongs in an annualized daily volatility comparison. Future sample variance is unbiased under constant IID increments; taking its square root introduces finite-sample bias, so instantaneous model SD is a proxy rather than exactly the expected sample SD.

Historical_21d is the last 21-session sample SD, and also the persistence forecast of the latest available realized-volatility measure. EWMA uses zero-mean squared returns with decay 0.94 and only observations through t. Parameter persistence uses the preceding scheduled calibration (t-stride); it is not a machine-learning model and still incurs calibration cost.

## Features, splits and learning

The 126-session MLE window is a predeclared compromise between identification and changing regimes; it is not selected using test scores. The LSTM reads `(batch, 21, 2)` sequences of returns and absolute returns; its output head also receives trailing summaries computed over up to 126 sessions. Ridge uses the same 21 signed return lags and historical summary statistics. No calibrated target is supplied as an input feature. Flattened lag positions do preserve ordering; LSTMs are not the only model capable of using ordered information.

Drop the initial price-only row from processed returns because its previous price is unavailable, keeping the original raw row unchanged. Supplied returns have export differences up to ~1.2e-8; their audit tolerance is 1e-6 absolute return (0.01 basis point). This tolerance never alters recomputed modelling returns. Reject duplicate, unordered, missing or nonpositive prices rather than silently sorting/deduplicating/imputing them. Do not use supplied `is_jump` or precomputed summaries as predictors.

Last 20% of development origins define chronological validation. Retain fit labels only if their last future return precedes the first validation origin. Similarly, validation labels must finish before the first supplied test origin. Historical feature overlap is permitted because those prices were already known; no validation/test future outcome is used to build an earlier feature. Final origins without all H future observations are excluded explicitly from both tasks so comparisons share identical dates. This exclusion is based on availability, not performance.

Fit sequence-channel, summary-feature and target scalers on fit examples only. Sequence observations may repeat across overlapping fit windows; this weighting is documented and contains no validation data. Ridge alpha and LSTM size/checkpoint use validation. No refit uses validation or test targets. This is frozen-model rolling-origin evaluation, not fixed-origin forecasting.

Parameter labels are transformed: log(sigma), sqrt(lambda), sqrt(sigma_j), with signed means unchanged. Prediction uses exp for sigma and squares for jump intensity and jump SD. The square link allows exact zero; it folds negative latent predictions and is many-to-one. It is not post-hoc clipping and can exceed MLE guardrails; all predictions remain visible. Native parameter metrics and reconstructed total-volatility metrics expose any resulting extrapolation error. Gaussian canonical labels can make jump-size metrics misleading, so prefer total-volatility evidence when judging risk forecasts.

Direct-volatility targets use log volatility and an exp output link. Training minimizes squared standardized log-target error; this targets a log-scale conditional center, not necessarily an arithmetic conditional mean. Validation model selection uses native volatility RMSE. Mean baselines are arithmetic fit-label means. Parameter candidate selection uses standardized latent MSE; final parameter-family comparison applies the same transform to decoded valid predictions, accounting for the square-link folding.

LSTMs use 8/16 hidden units, AdamW, dropout at the output head, gradient clipping, patience-based stopping, and restoration of the best validation checkpoint. Training loss shown beside validation loss is recomputed after each update with dropout disabled for both. A separate stochastic objective column records dropout-on optimization. This resolves loss comparability without disguising overfitting or distribution shift.

## Evaluation and limits

Join predictions to references by unique dates and validate source row identifiers. Report MAE, RMSE, prediction-minus-actual bias, error quantiles, correlation, ordinary R² and OOS R². OOS R² benchmarks are historical_21d for future volatility and fit-parameter means for parameter errors. Correlation of constant predictions and R² for a constant target are undefined, with an explicit reason.

Horizon overlap creates dependent forecast errors. No IID significance tests or confidence intervals are reported; a nonoverlapping subsequence table is a sensitivity check only. Different regimes and limited sample size remain scientific limitations. The prior test set has already been exposed; it is not scientifically fresh even though this implementation keeps it out of training and tuning.

## Sources and implementation references

- [Merton (1976), original jump-diffusion paper](https://doi.org/10.1016/0304-405X(76)90022-2).
- [SciPy Poisson distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html) for log probabilities and survival tails.
- [SciPy logsumexp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html) for stable mixture evaluation.
- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) for batch-first tensor orientation.
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) for deterministic controls and cross-platform limits.

The supplied guide is retained unchanged in data/raw. Its one-jump approximation, treatment of MLE labels as truth, diffusion-only volatility comparison and claims that flattened features cannot encode order are not adopted.
