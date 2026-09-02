"""Phase G: Logistic-Regression fitting, ROC-AUC/PR-AUC scoring, and the
fixed-seed bootstrap PR-AUC-delta confidence interval used for the B0-F2
graduation decision.

No hyperparameter tuning anywhere in this module -- `LogisticRegression` is
fit with library defaults (only `max_iter` is raised from sklearn's default
100, a SOLVER-CONVERGENCE budget, not a model-capacity/regularization
choice -- lbfgs is deterministic given fixed data, so this changes nothing
about what is being modeled, only whether it finishes optimizing). Every
constant here (`BOOTSTRAP_N_RESAMPLES`, `BOOTSTRAP_SEED`, `LOGREG_MAX_ITER`)
is fixed before any evaluation runs, matching this project's universal
fixed-before-evaluation discipline.

Scikit-learn is a new project dependency, introduced for this phase only
(every prior phase's own AUC computation used a hand-rolled rank-based
formula specifically to avoid this dependency for a simple summary
statistic -- fitting an actual model is different enough to justify it).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

LOGREG_MAX_ITER = 1000
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260101  # arbitrary, fixed before any evaluation runs -- not tuned to any result
BOOTSTRAP_CI_LEVEL = 0.95


def score_metrics(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """(roc_auc, pr_auc) for one set of predicted probabilities."""
    roc_auc = float(roc_auc_score(y_true, proba))
    pr_auc = float(average_precision_score(y_true, proba))
    return roc_auc, pr_auc


def constant_prevalence_scores(y_train: np.ndarray, n_validation: int) -> np.ndarray:
    """B0: predict the TRAIN-partition empirical prevalence for every
    validation row -- a constant score, no fitting, no scaling."""
    prevalence = float(np.mean(y_train))
    return np.full(n_validation, prevalence, dtype="float64")


def fit_and_score(
    X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray
) -> dict:
    """Fit `StandardScaler` on TRAIN ONLY, then a default-hyperparameter
    `LogisticRegression` (only `max_iter` raised -- see module docstring) on
    the scaled train matrix; score on validation. `X_validation` is
    transformed with the scaler already fit on `X_train` -- never re-fit,
    never fit on combined train+validation.

    Returns `{"roc_auc", "pr_auc", "proba_validation", "n_features",
    "converged"}`.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    model = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_validation_scaled)[:, 1]

    roc_auc, pr_auc = score_metrics(y_validation, proba)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "proba_validation": proba,
        "n_features": int(X_train.shape[1]),
        "n_iter": int(np.max(model.n_iter_)),
        "converged": bool(np.max(model.n_iter_) < LOGREG_MAX_ITER),
    }


def bootstrap_pr_auc_delta_ci(
    y_validation: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> dict:
    """Fixed-seed, PAIRED resample-with-replacement bootstrap 95% CI for
    PR-AUC(proba_b) - PR-AUC(proba_a) over `validation` rows. PAIRED: the
    SAME resampled row indices are used for both `proba_a` and `proba_b` in
    each resample, since both are predictions for the identical validation
    set -- this is the appropriate bootstrap design for comparing two
    models on the same evaluation set (not two independent bootstraps).

    A resample with only one class present makes PR-AUC undefined for that
    resample; such resamples are skipped (not treated as a delta of 0 or
    excluded from `n_resamples` by adjusting the loop -- `n_resamples_used`
    reports how many of the `n_resamples` draws were actually usable, so a
    caller can see if this ever matters; at validation's real ~2.2% fraud
    prevalence and n~57,806 this is not expected to occur in practice).
    """
    y = np.asarray(y_validation)
    a = np.asarray(proba_a)
    b = np.asarray(proba_b)
    n = len(y)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_s = y[idx]
        if y_s.min() == y_s.max():
            continue
        pr_a = average_precision_score(y_s, a[idx])
        pr_b = average_precision_score(y_s, b[idx])
        deltas.append(pr_b - pr_a)

    deltas_arr = np.array(deltas, dtype="float64")
    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.quantile(deltas_arr, alpha)) if len(deltas_arr) else float("nan")
    ci_upper = float(np.quantile(deltas_arr, 1.0 - alpha)) if len(deltas_arr) else float("nan")

    return {
        "n_resamples_requested": n_resamples,
        "n_resamples_used": int(len(deltas_arr)),
        "seed": seed,
        "ci_level": ci_level,
        "mean_delta": float(deltas_arr.mean()) if len(deltas_arr) else float("nan"),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
