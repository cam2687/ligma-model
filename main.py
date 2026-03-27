"""
MLB AI Prediction System -- CLI entry point.

Commands:
    python main.py fetch      Fetch all historical game data (2019-2025)
    python main.py engineer   Build feature matrix from fetched data
    python main.py train      Train XGBoost models with temporal CV
    python main.py predict    Generate predictions for today's games
    python main.py serve      Launch Streamlit dashboard
    python main.py pipeline   Run fetch -> engineer -> train -> predict in sequence

Flags:
    --force     Re-fetch / re-engineer / re-train even if cached data exists
    --seasons   Comma-separated seasons to fetch (default: 2019,2020,...,2025)
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import CACHE_DIR, MODELS_DIR, TRAIN_SEASONS


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_fetch(args) -> None:
    from config import SPORT
    from src.data.fetch import fetch_all_game_logs, fetch_all_fangraphs, fetch_team_batting_stats

    seasons = _parse_seasons(args)
    print(f"[fetch] Fetching game logs for seasons: {seasons}")
    game_logs = fetch_all_game_logs(seasons=seasons, force=args.force)
    print(f"[fetch] Game logs: {len(game_logs):,} rows")

    if SPORT == "mlb":
        print("[fetch] Fetching FanGraphs batting + pitching stats...")
        batting, pitching = fetch_all_fangraphs(seasons=seasons, force=args.force)
        print(f"[fetch] Batting seasons: {sorted(batting)}, Pitching seasons: {sorted(pitching)}")

        print("[fetch] Fetching bullpen ERA (FanGraphs individual pitching)...")
        from src.data.fetch import fetch_all_bullpen_stats
        bullpen = fetch_all_bullpen_stats(seasons=seasons, force=args.force)
        print(f"[fetch] Bullpen seasons: {sorted(bullpen)}")

        print("[fetch] Fetching historical weather for all stadiums (Open-Meteo)...")
        from src.data.fetch import fetch_weather_all_teams
        fetch_weather_all_teams(seasons=seasons, force=args.force)
        print("[fetch] Weather data cached.")

    elif SPORT == "soccer":
        print("[fetch] Computing soccer team stats...")
        team_stats = {}
        for season in seasons:
            stats = fetch_team_batting_stats(season, force=args.force)
            if not stats.empty:
                team_stats[season] = stats
        print(f"[fetch] Team stats seasons: {sorted(team_stats)}")

    print("[fetch] Done.")


def cmd_engineer(args) -> None:
    from config import SPORT
    from src.data.fetch import fetch_all_game_logs, fetch_all_fangraphs, fetch_team_batting_stats
    from src.data.features import build_training_dataset

    seasons = _parse_seasons(args)
    features_path = CACHE_DIR / "features.parquet"

    if features_path.exists() and not args.force:
        print(f"[engineer] Features already built at {features_path}. Use --force to rebuild.")
        return

    print("[engineer] Loading game logs...")
    game_logs = fetch_all_game_logs(seasons=seasons, force=args.force)

    bullpen_by_season = {}
    weather_df = None

    if SPORT == "mlb":
        print("[engineer] Loading FanGraphs stats...")
        batting, pitching = fetch_all_fangraphs(seasons=seasons)

        print("[engineer] Loading bullpen stats...")
        from src.data.fetch import fetch_all_bullpen_stats
        bullpen_by_season = fetch_all_bullpen_stats(seasons=seasons)

        print("[engineer] Loading historical weather...")
        from src.data.fetch import fetch_weather_all_teams
        weather_df = fetch_weather_all_teams(seasons=seasons)

    elif SPORT == "soccer":
        print("[engineer] Loading soccer team stats...")
        team_stats = {}
        for season in seasons:
            stats = fetch_team_batting_stats(season, force=args.force)
            if not stats.empty:
                team_stats[season] = stats
        batting, pitching = team_stats, {}

    print("[engineer] Building feature matrix...")
    df = build_training_dataset(
        game_logs, batting, pitching,
        bullpen_by_season=bullpen_by_season,
        weather_df=weather_df,
        save_path=features_path,
    )
    print(f"[engineer] Feature matrix: {df.shape}  ->  {features_path}")


def cmd_train(args) -> None:
    from config import SPORT
    from src.models.train import run_training, get_feature_importance, load_models
    from src.data.features import build_training_dataset
    from src.data.fetch import fetch_all_game_logs, fetch_all_fangraphs

    features_path = CACHE_DIR / "features.parquet"

    if features_path.exists() and not args.force:
        print(f"[train] Loading features from {features_path}")
        import pandas as pd
        df = pd.read_parquet(features_path)
    else:
        print("[train] Feature file not found -- running engineer step first...")
        cmd_engineer(args)
        import pandas as pd
        df = pd.read_parquet(features_path)

    print("[train] Starting training...")
    metrics = run_training(df)

    print("\n[train] === Results ===")
    wc = metrics["win_classifier"]
    if SPORT == "mlb":
        hr = metrics["home_runs_regressor"]
        ar = metrics["away_runs_regressor"]
        print(f"  Win Classifier  -- Accuracy: {wc['mean_accuracy']:.3f}  AUC: {wc['mean_auc_roc']:.3f}  Brier: {wc['mean_brier']:.4f}")
        print(f"  Home Runs Reg   -- RMSE:     {hr['mean_rmse']:.3f}  MAE:  {hr['mean_mae']:.3f}")
        print(f"  Away Runs Reg   -- RMSE:     {ar['mean_rmse']:.3f}  MAE:  {ar['mean_mae']:.3f}")
    elif SPORT == "soccer":
        hr = metrics["home_goals_regressor"]
        ar = metrics["away_goals_regressor"]
        print(f"  Win Classifier  -- Accuracy: {wc['mean_accuracy']:.3f}  AUC: {wc['mean_auc_roc']:.3f}  Brier: {wc['mean_brier']:.4f}")
        print(f"  Home Goals Reg  -- RMSE:     {hr['mean_rmse']:.3f}  MAE:  {hr['mean_mae']:.3f}")
        print(f"  Away Goals Reg  -- RMSE:     {ar['mean_rmse']:.3f}  MAE:  {ar['mean_mae']:.3f}")

    # Save feature importance
    try:
        clf, _, _, _ = load_models()
        from src.models.train import get_feature_importance
        imp = get_feature_importance(clf)
        if not imp.empty:
            imp.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
            print("[train] Feature importance saved.")
    except Exception as e:
        print(f"[train] Could not save feature importance: {e}")


def cmd_predict(args) -> None:
    from config import SPORT
    from src.data.fetch import fetch_all_game_logs
    from src.models.predict import predict_today
    import pandas as pd

    seasons = _parse_seasons(args)

    print("[predict] Loading game logs for rolling features...")
    game_logs = fetch_all_game_logs(seasons=seasons)

    bullpen_by_season = {}

    if SPORT == "mlb":
        from src.data.fetch import fetch_all_fangraphs, fetch_all_bullpen_stats
        print("[predict] Loading FanGraphs stats...")
        batting, pitching = fetch_all_fangraphs(seasons=seasons)
        print("[predict] Loading bullpen stats...")
        bullpen_by_season = fetch_all_bullpen_stats(seasons=seasons)
    elif SPORT == "soccer":
        from src.data.fetch import fetch_team_batting_stats
        print("[predict] Loading soccer team stats...")
        batting = {}
        for season in seasons:
            stats = fetch_team_batting_stats(season)
            if not stats.empty:
                batting[season] = stats
        pitching = {}

    print("[predict] Generating predictions...")
    results = predict_today(game_logs, batting, pitching,
                            bullpen_by_season=bullpen_by_season)

    if not results:
        print("[predict] No games found for today. Season may not have started yet.")
        print("[predict] Tip: Try running on a game day or set target_date in fetch_today_schedule().")
        return

    # Pretty-print to console
    from datetime import datetime, timezone, timedelta

    def _fmt_et(dt_str: str) -> str:
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            dt_et = dt.astimezone(timezone(timedelta(hours=-4)))
            h = dt_et.hour % 12 or 12
            return f"{h}:{dt_et.strftime('%M %p')} ET"
        except Exception:
            return ""

    print(f"\n[predict] === Today's Games ({len(results)}) ===\n")
    for g in results:
        rest_info = (
            f" | Rest: {g.get('home_team_br','?')} {g.get('home_rest_days',4):.0f}d / "
            f"{g.get('away_team_br','?')} {g.get('away_rest_days',4):.0f}d"
        ) if SPORT == "mlb" else ""
        weather_info = (
            f" | {g.get('temp_f', 70):.0f}°F {g.get('wind_mph', 7):.0f}mph"
        ) if SPORT == "mlb" and not g.get("is_dome") else ""

        time_et = _fmt_et(g.get("game_datetime", ""))
        print(
            f"  {g['away_team_br']} @ {g['home_team_br']}  |  "
            f"{time_et}  |  "
            f"Win: {g['home_team_br']} {g['home_win_prob']:.1%} / "
            f"{g['away_team_br']} {g['away_win_prob']:.1%}  |  "
            f"Total: {g['predicted_total']}  |  "
            f"ML: {g['home_team_br']} {g['home_moneyline_str']} / "
            f"{g['away_team_br']} {g['away_moneyline_str']}  |  "
            f"Conf: {g['confidence']}"
            f"{rest_info}{weather_info}"
        )

    # Save for dashboard
    from config import SPORT as CURRENT_SPORT
    pred_path = CACHE_DIR / f"{CURRENT_SPORT}_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[predict] Saved to {pred_path}")


def cmd_serve(args) -> None:
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    print(f"[serve] Launching Streamlit dashboard: {dashboard_path}")
    print("[serve] Press Ctrl+C to stop.")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )


def cmd_pipeline(args) -> None:
    print("=" * 60)
    print("MLB AI Pipeline: fetch -> engineer -> train -> predict")
    print("=" * 60)
    cmd_fetch(args)
    print()
    cmd_engineer(args)
    print()
    cmd_train(args)
    print()
    cmd_predict(args)
    print()
    print("[pipeline] All steps complete. Run 'python main.py serve' for the dashboard.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_seasons(args) -> list[int]:
    if hasattr(args, "seasons") and args.seasons:
        return [int(s) for s in args.seasons.split(",")]
    return TRAIN_SEASONS


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLB AI Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(p):
        p.add_argument("--force", action="store_true",
                       help="Re-fetch/re-build even if cached data exists")
        p.add_argument("--seasons", type=str, default=None,
                       help="Comma-separated seasons, e.g. 2022,2023,2024")

    fetch_p = sub.add_parser("fetch",    help="Fetch all historical game data")
    eng_p   = sub.add_parser("engineer", help="Build feature matrix")
    train_p = sub.add_parser("train",    help="Train prediction models")
    pred_p  = sub.add_parser("predict",  help="Predict today's games")
    serve_p = sub.add_parser("serve",    help="Launch Streamlit dashboard")
    pipe_p  = sub.add_parser("pipeline", help="Run full pipeline end-to-end")

    for p in [fetch_p, eng_p, train_p, pred_p, pipe_p]:
        _add_common(p)

    return parser


COMMAND_MAP = {
    "fetch":    cmd_fetch,
    "engineer": cmd_engineer,
    "train":    cmd_train,
    "predict":  cmd_predict,
    "serve":    cmd_serve,
    "pipeline": cmd_pipeline,
}


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    COMMAND_MAP[args.command](args)
