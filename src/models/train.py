"""
Model training with temporal cross-validation.

Uses season-boundary splits (not random k-fold) to avoid look-ahead bias:
    Fold 1: train=2019,       val=2020
    Fold 2: train=2019-2020,  val=2021
    Fold 3: train=2019-2021,  val=2022
    Fold 4: train=2019-2022,  val=2023
    Fold 5: train=2019-2023,  val=2024-2025

Models saved to models_saved/:
    win_classifier.joblib
    runs_regressor.joblib
    cv_metrics.json
"""
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score,
    mean_squared_error, mean_absolute_error,
)

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    MODELS_DIR, FEATURE_COLUMNS, TARGET_WIN, SPORT,
    TARGET_HOME_RUNS, TARGET_AWAY_RUNS,
    TARGET_HOME_GOALS, TARGET_AWAY_GOALS,
    XGB_CLASSIFIER_PARAMS, XGB_REGRESSOR_PARAMS, TRAIN_SEASONS,
    TOP_FEATURES,
)


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

CLASSIFIER_PARAM_GRID = {
    'n_estimators': [200, 400, 600],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
}

REGRESSOR_PARAM_GRID = {
    'n_estimators': [200, 400, 600],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
}

_CV_FOLDS = [
    (list(range(2019, 2020)), [2020]),
    (list(range(2019, 2021)), [2021]),
    (list(range(2019, 2022)), [2022]),
    (list(range(2019, 2023)), [2023]),
    (list(range(2019, 2024)), list(range(2024, 2026))),
]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _split(df: pd.DataFrame, train_seasons: list[int], val_seasons: list[int]):
    train = df[df["season"].isin(train_seasons)]
    val   = df[df["season"].isin(val_seasons)]

    if SPORT == "mlb":
        return (
            train[FEATURE_COLUMNS], train[TARGET_WIN], train[TARGET_RUNS],
            train[TARGET_HOME_RUNS], train[TARGET_AWAY_RUNS],
            val[FEATURE_COLUMNS],   val[TARGET_WIN],   val[TARGET_RUNS],
            val[TARGET_HOME_RUNS], val[TARGET_AWAY_RUNS],
        )
    elif SPORT == "soccer":
        return (
            train[FEATURE_COLUMNS], train[TARGET_WIN], train[TARGET_HOME_GOALS] + train[TARGET_AWAY_GOALS],  # total goals
            train[TARGET_HOME_GOALS], train[TARGET_AWAY_GOALS],
            val[FEATURE_COLUMNS],   val[TARGET_WIN],   val[TARGET_HOME_GOALS] + val[TARGET_AWAY_GOALS],
            val[TARGET_HOME_GOALS], val[TARGET_AWAY_GOALS],
        )
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")


def _fill_features(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with column medians (computed on X itself — safe within each fold)."""
    return X.fillna(X.median(numeric_only=True))


def _tune_xgb(model, param_grid, X: pd.DataFrame, y: pd.Series, n_iter: int = 20):
    """Randomized hyperparameter search for XGBoost models."""
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='roc_auc' if isinstance(model, xgb.XGBClassifier) else 'neg_mean_squared_error',
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Win probability classifier
# ---------------------------------------------------------------------------

def train_win_classifier(
    features_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[CalibratedClassifierCV, dict]:
    """
    Train XGBoost win classifier with temporal CV.

    Returns (calibrated_classifier, cv_metrics_dict)
    where cv_metrics has per-fold and aggregate accuracy/logloss/AUC/brier.
    """
    params = XGB_CLASSIFIER_PARAMS.copy()
    # Home teams win ~54% → slight class adjustment for calibration
    n_home_wins  = (features_df[TARGET_WIN] == 1).sum()
    n_home_losses= (features_df[TARGET_WIN] == 0).sum()
    if n_home_losses > 0:
        params["scale_pos_weight"] = float(n_home_losses) / float(n_home_wins)

    # Hyperparameter tuning on all training data (small search to avoid huge runtime)
    if len(features_df) > 1000:
        X_full = _fill_features(features_df[FEATURE_COLUMNS])
        y_full = features_df[TARGET_WIN]
        tuned_clf, tuned_params = _tune_xgb(xgb.XGBClassifier(**params), CLASSIFIER_PARAM_GRID, X_full, y_full, n_iter=15)
        params.update(tuned_params)
        if verbose:
            print(f"  [train] Tuned classifier params: {tuned_params}")

    fold_metrics = []

    if verbose:
        print("  [train] Win classifier — temporal CV:")

    for i, (train_seasons, val_seasons) in enumerate(_CV_FOLDS, 1):
        X_tr, y_tr, _, _, _, X_val, y_val, _, _, _ = _split(features_df, train_seasons, val_seasons)
        if len(X_val) == 0 or len(X_tr) == 0:
            continue

        X_tr  = _fill_features(X_tr)
        X_val = _fill_features(X_val)

        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        metrics = {
            "fold":       i,
            "train_size": len(X_tr),
            "val_size":   len(X_val),
            "accuracy":   float((preds == y_val).mean()),
            "log_loss":   float(log_loss(y_val, probs)),
            "brier":      float(brier_score_loss(y_val, probs)),
            "auc_roc":    float(roc_auc_score(y_val, probs)),
        }
        fold_metrics.append(metrics)

        if verbose:
            print(
                f"    Fold {i}: acc={metrics['accuracy']:.3f} "
                f"logloss={metrics['log_loss']:.3f} "
                f"AUC={metrics['auc_roc']:.3f} "
                f"brier={metrics['brier']:.3f}"
            )

    # Aggregate
    cv_metrics = {
        "folds":             fold_metrics,
        "mean_accuracy":     float(np.mean([m["accuracy"] for m in fold_metrics])),
        "mean_log_loss":     float(np.mean([m["log_loss"]  for m in fold_metrics])),
        "mean_brier":        float(np.mean([m["brier"]     for m in fold_metrics])),
        "mean_auc_roc":      float(np.mean([m["auc_roc"]   for m in fold_metrics])),
    }

    if verbose:
        print(
            f"  [train] CV summary: acc={cv_metrics['mean_accuracy']:.3f} "
            f"AUC={cv_metrics['mean_auc_roc']:.3f}"
        )

    # Train final model on ALL data with Platt calibration
    X_all = _fill_features(features_df[FEATURE_COLUMNS])
    y_all = features_df[TARGET_WIN]

    base_clf = xgb.XGBClassifier(**params)
    base_clf.fit(X_all, y_all, verbose=False)

    # Isotonic calibration on full data (improves probability quality)
    cal_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    cal_clf.fit(X_all, y_all)

    return cal_clf, cv_metrics


# ---------------------------------------------------------------------------
# Home runs regressor
# ---------------------------------------------------------------------------

def train_home_runs_regressor(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    verbose: bool = True,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost regressor for home team runs/goals.
    Returns (regressor, cv_metrics_dict).
    """
    if SPORT == "mlb":
        target = TARGET_HOME_RUNS
        model_name = "home runs"
    elif SPORT == "soccer":
        target = TARGET_HOME_GOALS
        model_name = "home goals"
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    fold_metrics = []

    if verbose:
        print(f"  [train] {model_name} regressor — temporal CV:")

    # Hyperparameter tuning for the regressor
    if len(features_df) > 1000:
        X_full = _fill_features(features_df[feature_cols])
        y_full = features_df[target]
        tuned_reg, tuned_params = _tune_xgb(xgb.XGBRegressor(**XGB_REGRESSOR_PARAMS), REGRESSOR_PARAM_GRID, X_full, y_full, n_iter=15)
        reg_params = tuned_params
        if verbose:
            print(f"  [train] Tuned {model_name} regressor params: {tuned_params}")
    else:
        reg_params = XGB_REGRESSOR_PARAMS.copy()

    for i, (train_seasons, val_seasons) in enumerate(_CV_FOLDS, 1):
        X_tr, _, _, y_tr_home, _, X_val, _, _, y_val_home, _ = _split(features_df, train_seasons, val_seasons)
        if len(X_val) == 0 or len(X_tr) == 0:
            continue

        X_tr  = _fill_features(X_tr[feature_cols])
        X_val = _fill_features(X_val[feature_cols])

        reg = xgb.XGBRegressor(**reg_params)
        reg.fit(
            X_tr, y_tr_home,
            eval_set=[(X_val, y_val_home)],
            verbose=False,
        )

        preds = reg.predict(X_val)
        rmse  = float(np.sqrt(mean_squared_error(y_val_home, preds)))
        mae   = float(mean_absolute_error(y_val_home, preds))

        metrics = {
            "fold":       i,
            "train_size": len(X_tr),
            "val_size":   len(X_val),
            "rmse":       rmse,
            "mae":        mae,
        }
        fold_metrics.append(metrics)

        if verbose:
            print(f"    Fold {i}: RMSE={rmse:.3f}  MAE={mae:.3f}")

    cv_metrics = {
        "folds":     fold_metrics,
        "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mean_mae":  float(np.mean([m["mae"]  for m in fold_metrics])),
    }

    if verbose:
        print(f"  [train] CV summary: RMSE={cv_metrics['mean_rmse']:.3f} "
              f"MAE={cv_metrics['mean_mae']:.3f}")

    # Train final model on ALL data
    X_all = _fill_features(features_df[feature_cols])
    y_all = features_df[target]

    final_reg = xgb.XGBRegressor(**reg_params)
    final_reg.fit(X_all, y_all, verbose=False)

    return final_reg, cv_metrics


# ---------------------------------------------------------------------------
# Away runs regressor
# ---------------------------------------------------------------------------

def train_away_runs_regressor(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    verbose: bool = True,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost regressor for away team runs/goals.
    Returns (regressor, cv_metrics_dict).
    """
    if SPORT == "mlb":
        target = TARGET_AWAY_RUNS
        model_name = "away runs"
    elif SPORT == "soccer":
        target = TARGET_AWAY_GOALS
        model_name = "away goals"
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    fold_metrics = []

    if verbose:
        print(f"  [train] {model_name} regressor — temporal CV:")

    # Hyperparameter tuning for the regressor
    if len(features_df) > 1000:
        X_full = _fill_features(features_df[feature_cols])
        y_full = features_df[target]
        tuned_reg, tuned_params = _tune_xgb(xgb.XGBRegressor(**XGB_REGRESSOR_PARAMS), REGRESSOR_PARAM_GRID, X_full, y_full, n_iter=15)
        reg_params = tuned_params
        if verbose:
            print(f"  [train] Tuned {model_name} regressor params: {tuned_params}")
    else:
        reg_params = XGB_REGRESSOR_PARAMS.copy()

    for i, (train_seasons, val_seasons) in enumerate(_CV_FOLDS, 1):
        X_tr, _, _, _, y_tr_away, X_val, _, _, _, y_val_away = _split(features_df, train_seasons, val_seasons)
        if len(X_val) == 0 or len(X_tr) == 0:
            continue

        X_tr  = _fill_features(X_tr[feature_cols])
        X_val = _fill_features(X_val[feature_cols])

        reg = xgb.XGBRegressor(**reg_params)
        reg.fit(
            X_tr, y_tr_away,
            eval_set=[(X_val, y_val_away)],
            verbose=False,
        )

        preds = reg.predict(X_val)
        rmse  = float(np.sqrt(mean_squared_error(y_val_away, preds)))
        mae   = float(mean_absolute_error(y_val_away, preds))

        metrics = {
            "fold":       i,
            "train_size": len(X_tr),
            "val_size":   len(X_val),
            "rmse":       rmse,
            "mae":        mae,
        }
        fold_metrics.append(metrics)

        if verbose:
            print(f"    Fold {i}: RMSE={rmse:.3f}  MAE={mae:.3f}")

    cv_metrics = {
        "folds":     fold_metrics,
        "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mean_mae":  float(np.mean([m["mae"]  for m in fold_metrics])),
    }

    if verbose:
        print(f"  [train] CV summary: RMSE={cv_metrics['mean_rmse']:.3f} "
              f"MAE={cv_metrics['mean_mae']:.3f}")

    # Train final model on ALL data
    X_all = _fill_features(features_df[feature_cols])
    y_all = features_df[target]

    final_reg = xgb.XGBRegressor(**reg_params)
    final_reg.fit(X_all, y_all, verbose=False)

    return final_reg, cv_metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(classifier: CalibratedClassifierCV) -> pd.DataFrame:
    """Extract feature importance from the calibrated classifier's base estimator."""
    try:
        base = classifier.calibrated_classifiers_[0].estimator
        scores = base.feature_importances_
        return (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_models(classifier, home_regressor, away_regressor, cv_win: dict, cv_home: dict, cv_away: dict) -> None:
    """Save trained models and CV metrics to MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, MODELS_DIR / "win_classifier.joblib")

    if SPORT == "mlb":
        joblib.dump(home_regressor,  MODELS_DIR / "home_runs_regressor.joblib")
        joblib.dump(away_regressor,  MODELS_DIR / "away_runs_regressor.joblib")
        all_metrics = {
            "win_classifier":  cv_win,
            "home_runs_regressor": cv_home,
            "away_runs_regressor": cv_away,
            "feature_columns": FEATURE_COLUMNS,
        }
    elif SPORT == "soccer":
        joblib.dump(home_regressor,  MODELS_DIR / "home_goals_regressor.joblib")
        joblib.dump(away_regressor,  MODELS_DIR / "away_goals_regressor.joblib")
        all_metrics = {
            "win_classifier":  cv_win,
            "home_goals_regressor": cv_home,
            "away_goals_regressor": cv_away,
            "feature_columns": FEATURE_COLUMNS,
        }
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    with open(MODELS_DIR / "cv_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"  [train] Models saved to {MODELS_DIR}")


def load_models() -> tuple:
    """
    Load classifier, home_regressor, away_regressor, and cv_metrics.
    Raises FileNotFoundError if models have not been trained yet.
    """
    clf_path = MODELS_DIR / "win_classifier.joblib"

    if SPORT == "mlb":
        home_reg_path = MODELS_DIR / "home_runs_regressor.joblib"
        away_reg_path = MODELS_DIR / "away_runs_regressor.joblib"
    elif SPORT == "soccer":
        home_reg_path = MODELS_DIR / "home_goals_regressor.joblib"
        away_reg_path = MODELS_DIR / "away_goals_regressor.joblib"
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    met_path = MODELS_DIR / "cv_metrics.json"

    if not clf_path.exists():
        raise FileNotFoundError(
            f"Model not found at {clf_path}. Run: python main.py train"
        )

    classifier = joblib.load(clf_path)
    home_regressor = joblib.load(home_reg_path)
    away_regressor = joblib.load(away_reg_path)

    with open(met_path) as f:
        metrics = json.load(f)

    return classifier, home_regressor, away_regressor, metrics


# ---------------------------------------------------------------------------
# Full training entry point
# ---------------------------------------------------------------------------

def run_training(features_df: pd.DataFrame) -> dict:
    """
    Train both models and save to disk.
    Returns cv_metrics dict.
    """
    # Sanity checks
    missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print(f"  [train] Training on {len(features_df):,} games "
          f"({features_df['season'].min()}-{features_df['season'].max()})")

    classifier, cv_win  = train_win_classifier(features_df)

    # Feature-selection guided regressors (uses top classifiers features)
    imp = get_feature_importance(classifier)
    selected_features = imp.head(TOP_FEATURES)["feature"].tolist()
    selected_features = [f for f in selected_features if f in FEATURE_COLUMNS]
    if not selected_features:
        selected_features = FEATURE_COLUMNS

    home_regressor, cv_home = train_home_runs_regressor(features_df, selected_features)
    away_regressor, cv_away = train_away_runs_regressor(features_df, selected_features)

    save_models(classifier, home_regressor, away_regressor, cv_win, cv_home, cv_away)

    if SPORT == "mlb":
        return {"win_classifier": cv_win, "home_runs_regressor": cv_home, "away_runs_regressor": cv_away}
    elif SPORT == "soccer":
        return {"win_classifier": cv_win, "home_goals_regressor": cv_home, "away_goals_regressor": cv_away}
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")
