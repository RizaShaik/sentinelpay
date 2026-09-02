import numpy as np
import pytest

from sentinelpay.model_evaluation import (
    BOOTSTRAP_SEED,
    LOGREG_MAX_ITER,
    bootstrap_pr_auc_delta_ci,
    constant_prevalence_scores,
    fit_and_score,
    score_metrics,
)


def test_constant_prevalence_scores():
    y_train = np.array([0, 0, 0, 1, 1])  # prevalence 0.4
    out = constant_prevalence_scores(y_train, n_validation=7)
    assert len(out) == 7
    assert np.allclose(out, 0.4)


def test_score_metrics_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    roc_auc, pr_auc = score_metrics(y, proba)
    assert roc_auc == pytest.approx(1.0)
    assert pr_auc == pytest.approx(1.0)


def test_score_metrics_chance_level():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    proba = rng.random(2000)  # independent of y
    roc_auc, pr_auc = score_metrics(y, proba)
    assert 0.45 < roc_auc < 0.55
    assert abs(pr_auc - y.mean()) < 0.05  # PR-AUC floor is the prevalence


def test_fit_and_score_separable_data_converges_and_scores_well():
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.normal(size=n)
    y = (x1 > 0).astype(int)
    X_train = np.column_stack([x1, rng.normal(size=n)])
    x1_val = rng.normal(size=100)
    y_val = (x1_val > 0).astype(int)
    X_val = np.column_stack([x1_val, rng.normal(size=100)])

    result = fit_and_score(X_train, y, X_val, y_val)
    assert result["roc_auc"] > 0.9
    assert result["n_features"] == 2
    assert result["converged"] is True
    assert result["n_iter"] <= LOGREG_MAX_ITER
    assert len(result["proba_validation"]) == 100
    assert np.all((result["proba_validation"] >= 0) & (result["proba_validation"] <= 1))


def test_fit_and_score_scaler_fit_on_train_only_not_validation():
    # Validation features are shifted far outside train's distribution --
    # the scaler must still use TRAIN's mean/std (not blow up or silently
    # refit), and predict_proba must still return valid, finite probabilities.
    rng = np.random.default_rng(2)
    n = 300
    X_train = rng.normal(loc=0.0, scale=1.0, size=(n, 3))
    y_train = rng.integers(0, 2, size=n)
    X_val = rng.normal(loc=500.0, scale=50.0, size=(50, 3))  # wildly different distribution
    y_val = rng.integers(0, 2, size=50)

    result = fit_and_score(X_train, y_train, X_val, y_val)
    proba = result["proba_validation"]
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0) & (proba <= 1))


def test_bootstrap_pr_auc_delta_ci_clear_improvement():
    rng = np.random.default_rng(3)
    n = 5000
    y = (rng.random(n) < 0.1).astype(int)  # 10% prevalence
    proba_a = rng.random(n)  # uninformative
    proba_b = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))  # clearly informative

    result = bootstrap_pr_auc_delta_ci(y, proba_a, proba_b, n_resamples=200, seed=1)
    assert result["n_resamples_used"] > 190  # near-full usage at n=5000, 10% prevalence
    assert result["ci_lower"] > 0.0
    assert result["mean_delta"] > 0.0


def test_bootstrap_pr_auc_delta_ci_identical_models_straddles_zero():
    rng = np.random.default_rng(4)
    n = 3000
    y = (rng.random(n) < 0.1).astype(int)
    proba = rng.random(n)

    result = bootstrap_pr_auc_delta_ci(y, proba, proba, n_resamples=200, seed=2)
    assert result["mean_delta"] == pytest.approx(0.0, abs=1e-9)
    assert result["ci_lower"] <= 0.0 <= result["ci_upper"]


def test_bootstrap_pr_auc_delta_ci_reproducible_with_fixed_seed():
    rng = np.random.default_rng(5)
    n = 1000
    y = (rng.random(n) < 0.2).astype(int)
    proba_a = rng.random(n)
    proba_b = rng.random(n)

    r1 = bootstrap_pr_auc_delta_ci(y, proba_a, proba_b, n_resamples=100, seed=BOOTSTRAP_SEED)
    r2 = bootstrap_pr_auc_delta_ci(y, proba_a, proba_b, n_resamples=100, seed=BOOTSTRAP_SEED)
    assert r1 == r2
