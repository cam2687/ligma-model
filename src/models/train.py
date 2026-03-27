"""
Model training with temporal cross-validation.

Uses season-boundary splits (not random k-fold) to avoid look-ahead bias:
    Fold 1: train=2019,       val=2020
    Fold 2: train=2019-2020,  val=2021
    Fold 3: train=2019-2021,  val=2022
    Fold 4: train=2019-2022,  val=2023
    Fold 5: train=2019-2023,  val=2024-2025
"""
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score, f1_score,
    mean_squared_error, mean_absolute_error,
)

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    MODELS_DIR, FEATURE_COLUMNS, TARGET_WIN, TARGET_RESULT, SPORT,
    TARGET_HOME_RUNS, TARGET_AWAY_RUNS,
    TARGET_HOME_GOALS, TARGET_AWAY_GOALS,
    XGB_CLASSIFIER_PARAMS, XGB_REGRESSOR_PARAMS,
    TOP_FEATURES,
)

def _artifact_path(name: str, suffix: str) -> Path:
    return MODELS_DIR / f"{SPORT}_{name}{suffix}"


_CV_FOLDS = [
    (list(range(2019, 2020)), [2020]),
    (list(range(2019, 2021)), [2021]),
    (list(range(2019, 2022)), [2022]),
    (list(range(2019, 2023)), [2023]),
    (list(range(2019, 2024)), list(range(2024, 2026))),
]


def _split(df: pd.DataFrame, train_seasons: list[int], val_seasons: list[int]):
    train = df[df["season"].isin(train_seasons)]
    val = df[df["season"].isin(val_seasons)]

    if SPORT == "mlb":
        return (
            train[FEATURE_COLUMNS], train[TARGET_WIN],
            train[TARGET_HOME_RUNS], train[TARGET_AWAY_RUNS],
            val[FEATURE_COLUMNS], val[TARGET_WIN],
            val[TARGET_HOME_RUNS], val[TARGET_AWAY_RUNS],
        )
    if SPORT == "soccer":
        return (
            train[FEATURE_COLUMNS], train[TARGET_RESULT],
            train[TARGET_HOME_GOALS], train[TARGET_AWAY_GOALS],
            val[FEATURE_COLUMNS], val[TARGET_RESULT],
            val[TARGET_HOME_GOALS], val[TARGET_AWAY_GOALS],
        )
    raise ValueError(f"Unsupported sport: {SPORT}")


def _fit_feature_fill(X: pd.DataFrame) -> pd.Series:
    return X.median(numeric_only=True)


def _apply_feature_fill(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return X.fillna(medians)


def _multiclass_brier_score(y_true: pd.Series, probs: np.ndarray) -> float:
    y_arr = np.asarray(y_true, dtype=int)
    one_hot = np.eye(probs.shape[1])[y_arr]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _classifier_params() -> dict:
    params = XGB_CLASSIFIER_PARAMS.copy()
    if SPORT == "soccer":
        params.update({
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
        })
        params.pop("scale_pos_weight", None)
    return params


def train_win_classifier(
    features_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[CalibratedClassifierCV, dict]:
    """
    Train the main outcome classifier with temporal CV.

    MLB: binary home-win classifier.
    Soccer: 3-way outcome classifier (away/draw/home).
    """
    params = _classifier_params()
    target_col = TARGET_WIN if SPORT == "mlb" else TARGET_RESULT

    if SPORT == "mlb":
        n_home_wins = int((features_df[TARGET_WIN] == 1).sum())
        n_home_losses = int((features_df[TARGET_WIN] == 0).sum())
        if n_home_losses > 0 and n_home_wins > 0:
            params["scale_pos_weight"] = float(n_home_losses) / float(n_home_wins)

    fold_metrics = []

    if verbose:
        print("  [train] Outcome classifier - temporal CV:")

    for i, (train_seasons, val_seasons) in enumerate(_CV_FOLDS, 1):
        X_tr, y_tr, _, _, X_val, y_val, _, _ = _split(features_df, train_seasons, val_seasons)
        if len(X_val) == 0 or len(X_tr) == 0:
            continue

        medians = _fit_feature_fill(X_tr)
        X_tr = _apply_feature_fill(X_tr, medians)
        X_val = _apply_feature_fill(X_val, medians)

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        if SPORT == "soccer":
            probs = clf.predict_proba(X_val)
            preds = probs.argmax(axis=1)
            try:
                auc = float(roc_auc_score(y_val, probs, multi_class="ovr"))
            except ValueError:
                auc = float("nan")
            metrics = {
                "fold": i,
                "train_size": len(X_tr),
                "val_size": len(X_val),
                "accuracy": float(accuracy_score(y_val, preds)),
                "log_loss": float(log_loss(y_val, probs, labels=[0, 1, 2])),
                "brier": _multiclass_brier_score(y_val, probs),
                "auc_roc": auc,
                "macro_f1": float(f1_score(y_val, preds, average="macro")),
            }
        else:
            probs = clf.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            metrics = {
                "fold": i,
                "train_size": len(X_tr),
                "val_size": len(X_val),
                "accuracy": float(accuracy_score(y_val, preds)),
                "log_loss": float(log_loss(y_val, probs)),
                "brier": float(brier_score_loss(y_val, probs)),
                "auc_roc": float(roc_auc_score(y_val, probs)),
            }
        fold_metrics.append(metrics)

        if verbose:
            summary = (
                f"    Fold {i}: acc={metrics['accuracy']:.3f} "
                f"logloss={metrics['log_loss']:.3f} "
                f"AUC={metrics['auc_roc']:.3f} "
                f"brier={metrics['brier']:.3f}"
            )
            if SPORT == "soccer":
                summary += f" macro_f1={metrics['macro_f1']:.3f}"
            print(summary)

    cv_metrics = {
        "folds": fold_metrics,
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "mean_log_loss": float(np.mean([m["log_loss"] for m in fold_metrics])),
        "mean_brier": float(np.mean([m["brier"] for m in fold_metrics])),
        "mean_auc_roc": float(np.nanmean([m["auc_roc"] for m in fold_metrics])),
    }
    if SPORT == "soccer":
        cv_metrics["mean_macro_f1"] = float(np.mean([m["macro_f1"] for m in fold_metrics]))

    if verbose:
        print(
            f"  [train] CV summary: acc={cv_metrics['mean_accuracy']:.3f} "
            f"AUC={cv_metrics['mean_auc_roc']:.3f}"
        )

    medians = _fit_feature_fill(features_df[FEATURE_COLUMNS])
    X_all = _apply_feature_fill(features_df[FEATURE_COLUMNS], medians)
    y_all = features_df[target_col]

    base_clf = xgb.XGBClassifier(**params)
    cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    cal_clf.fit(X_all, y_all)

    return cal_clf, cv_metrics


def _regressor_target_and_name(side: str) -> tuple[str, str]:
    if SPORT == "mlb":
        return (
            TARGET_HOME_RUNS if side == "home" else TARGET_AWAY_RUNS,
            f"{side} runs",
        )
    if SPORT == "soccer":
        return (
            TARGET_HOME_GOALS if side == "home" else TARGET_AWAY_GOALS,
            f"{side} goals",
        )
    raise ValueError(f"Unsupported sport: {SPORT}")


def _train_regressor(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    side: str,
    verbose: bool = True,
) -> tuple[xgb.XGBRegressor, dict]:
    target, model_name = _regressor_target_and_name(side)
    fold_metrics = []

    if verbose:
        print(f"  [train] {model_name} regressor - temporal CV:")

    for i, (train_seasons, val_seasons) in enumerate(_CV_FOLDS, 1):
        X_tr, _, y_tr_home, y_tr_away, X_val, _, y_val_home, y_val_away = _split(
            features_df, train_seasons, val_seasons
        )
        if len(X_val) == 0 or len(X_tr) == 0:
            continue

        y_tr = y_tr_home if side == "home" else y_tr_away
        y_val = y_val_home if side == "home" else y_val_away

        medians = _fit_feature_fill(X_tr[feature_cols])
        X_tr_fold = _apply_feature_fill(X_tr[feature_cols], medians)
        X_val_fold = _apply_feature_fill(X_val[feature_cols], medians)

        reg = xgb.XGBRegressor(**XGB_REGRESSOR_PARAMS)
        reg.fit(X_tr_fold, y_tr, eval_set=[(X_val_fold, y_val)], verbose=False)

        preds = reg.predict(X_val_fold)
        metrics = {
            "fold": i,
            "train_size": len(X_tr_fold),
            "val_size": len(X_val_fold),
            "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
            "mae": float(mean_absolute_error(y_val, preds)),
        }
        fold_metrics.append(metrics)

        if verbose:
            print(f"    Fold {i}: RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}")

    cv_metrics = {
        "folds": fold_metrics,
        "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
    }

    if verbose:
        print(
            f"  [train] CV summary: RMSE={cv_metrics['mean_rmse']:.3f} "
            f"MAE={cv_metrics['mean_mae']:.3f}"
        )

    medians = _fit_feature_fill(features_df[feature_cols])
    X_all = _apply_feature_fill(features_df[feature_cols], medians)
    y_all = features_df[target]

    final_reg = xgb.XGBRegressor(**XGB_REGRESSOR_PARAMS)
    final_reg.fit(X_all, y_all, verbose=False)
    return final_reg, cv_metrics


def train_home_runs_regressor(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    verbose: bool = True,
) -> tuple[xgb.XGBRegressor, dict]:
    return _train_regressor(features_df, feature_cols, side="home", verbose=verbose)


def train_away_runs_regressor(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    verbose: bool = True,
) -> tuple[xgb.XGBRegressor, dict]:
    return _train_regressor(features_df, feature_cols, side="away", verbose=verbose)


def get_feature_importance(classifier: CalibratedClassifierCV) -> pd.DataFrame:
    """Extract feature importance from the calibrated classifier's base estimator."""
    try:
        calibrated = classifier.calibrated_classifiers_[0]
        base = getattr(calibrated, "estimator", getattr(calibrated, "base_estimator", None))
        if base is None:
            return pd.DataFrame()
        scores = base.feature_importances_
        return (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


def save_models(classifier, home_regressor, away_regressor, cv_win: dict, cv_home: dict, cv_away: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, _artifact_path("win_classifier", ".joblib"))

    if SPORT == "mlb":
        joblib.dump(home_regressor, _artifact_path("home_runs_regressor", ".joblib"))
        joblib.dump(away_regressor, _artifact_path("away_runs_regressor", ".joblib"))
        all_metrics = {
            "win_classifier": cv_win,
            "home_runs_regressor": cv_home,
            "away_runs_regressor": cv_away,
            "feature_columns": FEATURE_COLUMNS,
        }
    elif SPORT == "soccer":
        joblib.dump(home_regressor, _artifact_path("home_goals_regressor", ".joblib"))
        joblib.dump(away_regressor, _artifact_path("away_goals_regressor", ".joblib"))
        all_metrics = {
            "win_classifier": cv_win,
            "home_goals_regressor": cv_home,
            "away_goals_regressor": cv_away,
            "feature_columns": FEATURE_COLUMNS,
        }
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    with open(_artifact_path("cv_metrics", ".json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"  [train] Models saved to {MODELS_DIR}")


def load_models() -> tuple:
    clf_path = _artifact_path("win_classifier", ".joblib")

    if SPORT == "mlb":
        home_reg_path = _artifact_path("home_runs_regressor", ".joblib")
        away_reg_path = _artifact_path("away_runs_regressor", ".joblib")
    elif SPORT == "soccer":
        home_reg_path = _artifact_path("home_goals_regressor", ".joblib")
        away_reg_path = _artifact_path("away_goals_regressor", ".joblib")
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    met_path = _artifact_path("cv_metrics", ".json")

    if SPORT == "mlb" and not clf_path.exists():
        legacy_clf = MODELS_DIR / "win_classifier.joblib"
        if legacy_clf.exists():
            clf_path = legacy_clf

    if SPORT == "mlb" and not home_reg_path.exists():
        legacy_home = MODELS_DIR / (
            "home_runs_regressor.joblib" if SPORT == "mlb" else "home_goals_regressor.joblib"
        )
        if legacy_home.exists():
            home_reg_path = legacy_home

    if SPORT == "mlb" and not away_reg_path.exists():
        legacy_away = MODELS_DIR / (
            "away_runs_regressor.joblib" if SPORT == "mlb" else "away_goals_regressor.joblib"
        )
        if legacy_away.exists():
            away_reg_path = legacy_away

    if SPORT == "mlb" and not met_path.exists():
        legacy_metrics = MODELS_DIR / "cv_metrics.json"
        if legacy_metrics.exists():
            met_path = legacy_metrics

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


def run_training(features_df: pd.DataFrame) -> dict:
    missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print(
        f"  [train] Training on {len(features_df):,} games "
        f"({features_df['season'].min()}-{features_df['season'].max()})"
    )

    classifier, cv_win = train_win_classifier(features_df)

    imp = get_feature_importance(classifier)
    selected_features = imp.head(TOP_FEATURES)["feature"].tolist()
    selected_features = [f for f in selected_features if f in FEATURE_COLUMNS]
    if not selected_features:
        selected_features = FEATURE_COLUMNS

    home_regressor, cv_home = train_home_runs_regressor(features_df, selected_features)
    away_regressor, cv_away = train_away_runs_regressor(features_df, selected_features)

    save_models(classifier, home_regressor, away_regressor, cv_win, cv_home, cv_away)

    if SPORT == "mlb":
        return {
            "win_classifier": cv_win,
            "home_runs_regressor": cv_home,
            "away_runs_regressor": cv_away,
        }
    if SPORT == "soccer":
        return {
            "win_classifier": cv_win,
            "home_goals_regressor": cv_home,
            "away_goals_regressor": cv_away,
        }
    raise ValueError(f"Unsupported sport: {SPORT}")
