"""
Prediction module.

Loads trained models and generates win probabilities, total runs,
and moneyline odds for today's scheduled games.
"""
import sys
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import statsapi

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    FEATURE_COLUMNS, CACHE_DIR, MODELS_DIR, BR_TO_FULL_NAME,
    TRAIN_SEASONS, PREDICT_SEASON, SPORT,
)
from src.models.train import load_models


# ---------------------------------------------------------------------------
# Moneyline conversion
# ---------------------------------------------------------------------------

def prob_to_moneyline(p: float) -> int:
    """
    Convert a win probability to American odds (fair line, no vig).
    p=0.60 → -150 (favorite)
    p=0.40 → +150 (underdog)
    """
    p = float(np.clip(p, 0.001, 0.999))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    else:
        return int(round(100 * (1 - p) / p))


def format_moneyline(ml: int) -> str:
    return f"+{ml}" if ml > 0 else str(ml)


def _get_pitcher_era(pitcher_id: int, season: int) -> float:
    """Fetch pitcher season ERA from statsapi if available."""
    try:
        if pitcher_id is None:
            return 4.5
        # Statsapi API may use player_stat with group/year; if unavailable fallback.
        data = statsapi.player_stats(player_id=pitcher_id, group="pitching", type="yearByYear")
        # data may be a dict with 'stats' rows containing year and era
        for stat in reversed(data.get("stats", [])):
            if str(stat.get("season")) == str(season):
                era = stat.get("era") or 4.5
                return float(era)
    except Exception:
        pass
    return 4.5


# ---------------------------------------------------------------------------
# Single-game prediction
# ---------------------------------------------------------------------------

def predict_game(
    game_features: pd.DataFrame,
    classifier,
    home_regressor,
    away_regressor,
) -> dict:
    """
    Predict a single game.
    game_features: single-row DataFrame with exactly FEATURE_COLUMNS columns.
    """
    # Guarantee column alignment for classifier
    X = game_features.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

    home_win_prob = float(classifier.predict_proba(X)[0, 1])
    away_win_prob = 1.0 - home_win_prob

    # Use each regressor's training feature signature to avoid mismatch
    home_features = X.reindex(columns=getattr(home_regressor, 'feature_names_in_', FEATURE_COLUMNS), fill_value=0.0)
    away_features = X.reindex(columns=getattr(away_regressor, 'feature_names_in_', FEATURE_COLUMNS), fill_value=0.0)

    pred_home_runs = float(max(0.0, home_regressor.predict(home_features)[0]))
    pred_away_runs = float(max(0.0, away_regressor.predict(away_features)[0]))
    predicted_total = pred_home_runs + pred_away_runs

    home_ml = prob_to_moneyline(home_win_prob)
    away_ml = prob_to_moneyline(away_win_prob)

    # Confidence based on distance from 50%
    confidence_pct = abs(home_win_prob - 0.5) * 200  # 0-100 scale
    if confidence_pct < 10:
        confidence = "Low"
    elif confidence_pct < 25:
        confidence = "Moderate"
    else:
        confidence = "High"

    if SPORT == "mlb":
        return {
            "home_win_prob":    round(home_win_prob, 3),
            "away_win_prob":    round(away_win_prob, 3),
            "predicted_total":  round(predicted_total, 1),
            "pred_home_runs":   round(pred_home_runs, 1),
            "pred_away_runs":   round(pred_away_runs, 1),
            "home_moneyline":   home_ml,
            "away_moneyline":   away_ml,
            "home_moneyline_str": format_moneyline(home_ml),
            "away_moneyline_str": format_moneyline(away_ml),
            "predicted_winner": "home" if home_win_prob >= 0.5 else "away",
            "confidence":       confidence,
            "confidence_pct":   round(confidence_pct, 1),
        }
    elif SPORT == "soccer":
        return {
            "home_win_prob":    round(home_win_prob, 3),
            "away_win_prob":    round(away_win_prob, 3),
            "predicted_total":  round(predicted_total, 1),
            "pred_home_goals":  round(pred_home_runs, 1),  # reusing variable names
            "pred_away_goals":  round(pred_away_runs, 1),
            "home_moneyline":   home_ml,
            "away_moneyline":   away_ml,
            "home_moneyline_str": format_moneyline(home_ml),
            "away_moneyline_str": format_moneyline(away_ml),
            "predicted_winner": "home" if home_win_prob >= 0.5 else "away",
            "confidence":       confidence,
            "confidence_pct":   round(confidence_pct, 1),
        }
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")


# ---------------------------------------------------------------------------
# Full pipeline: today's games
# ---------------------------------------------------------------------------

def predict_today(
    game_logs: pd.DataFrame,
    batting_by_season: dict[int, pd.DataFrame],
    pitching_by_season: dict[int, pd.DataFrame],
) -> list[dict]:
    """
    Orchestrate predictions for all of today's scheduled games.

    For MLB: fetches today's schedule from statsapi.
    For soccer: reads from cache/soccer_fixtures.csv or falls back to recent demo games.
    """
    from src.data.features import get_end_of_season_rolling, build_prediction_row

    classifier, home_regressor, away_regressor, _ = load_models()
    last_season = max(TRAIN_SEASONS)
    today_str = date.today().strftime("%Y-%m-%d")

    end_rolling = get_end_of_season_rolling(game_logs, last_season)
    rolling_lookup = end_rolling.set_index("team_br").to_dict("index")

    results = []

    if SPORT == "mlb":
        from src.data.fetch import fetch_today_schedule

        batting_df  = batting_by_season.get(last_season, pd.DataFrame())
        pitching_df = pitching_by_season.get(last_season, pd.DataFrame())

        bat_lookup = (
            batting_df.set_index("team_br").to_dict("index")
            if not batting_df.empty and "team_br" in batting_df.columns
            else {}
        )
        pit_lookup = (
            pitching_df.set_index("team_br").to_dict("index")
            if not pitching_df.empty and "team_br" in pitching_df.columns
            else {}
        )

        games = fetch_today_schedule(today_str)
        if not games:
            print("  [predict] No regular season games found for today.")
            return []

        for game in games:
            h = game["home_team_br"]
            a = game["away_team_br"]

            h_roll = pd.Series(rolling_lookup.get(h, {}))
            a_roll = pd.Series(rolling_lookup.get(a, {}))
            h_bat  = pd.Series(bat_lookup.get(h, {}))
            h_pit  = pd.Series(pit_lookup.get(h, {}))
            a_bat  = pd.Series(bat_lookup.get(a, {}))
            a_pit  = pd.Series(pit_lookup.get(a, {}))

            h_pitcher_era = _get_pitcher_era(game.get("home_pitcher_id"), last_season)
            a_pitcher_era = _get_pitcher_era(game.get("away_pitcher_id"), last_season)

            features = build_prediction_row(
                home_team_br=h, away_team_br=a,
                home_rolling=h_roll, away_rolling=a_roll,
                home_batting=h_bat, home_pitching=h_pit,
                away_batting=a_bat, away_pitching=a_pit,
                game_date=today_str,
                home_pitcher_era=h_pitcher_era,
                away_pitcher_era=a_pitcher_era,
            )

            pred = predict_game(features, classifier, home_regressor, away_regressor)
            pred.update({
                "home_team_br":    h,
                "away_team_br":    a,
                "home_team_name":  game.get("home_team_name") or BR_TO_FULL_NAME.get(h, h),
                "away_team_name":  game.get("away_team_name") or BR_TO_FULL_NAME.get(a, a),
                "home_pitcher":    game.get("home_pitcher_name", "TBD"),
                "away_pitcher":    game.get("away_pitcher_name", "TBD"),
                "venue":           game.get("venue", ""),
                "game_datetime":   game.get("game_datetime", ""),
                "recent_form_home": _get_recent_form(game_logs, h, last_season),
                "recent_form_away": _get_recent_form(game_logs, a, last_season),
            })
            results.append(pred)

    elif SPORT == "soccer":
        from src.data.fetch import fetch_soccer_fixtures

        team_stats_df = batting_by_season.get(last_season, pd.DataFrame())
        if not team_stats_df.empty and "team_br" in team_stats_df.columns:
            # If a team appears in multiple leagues, keep the row with most games played
            gp_col = "games_played" if "games_played" in team_stats_df.columns else None
            if gp_col:
                deduped = team_stats_df.sort_values(gp_col, ascending=False).drop_duplicates("team_br")
            else:
                deduped = team_stats_df.drop_duplicates("team_br")
            stats_lookup = deduped.set_index("team_br").to_dict("index")
        else:
            stats_lookup = {}

        games = fetch_soccer_fixtures(today_str)
        if not games:
            print("  [predict] No soccer fixtures found.")
            print("  [predict] To add fixtures, create cache/soccer_fixtures.csv with columns:")
            print("  [predict]   home_team,away_team,game_datetime,league")
            return []

        for game in games:
            h = game["home_team_br"]
            a = game["away_team_br"]

            h_roll  = pd.Series(rolling_lookup.get(h, {}))
            a_roll  = pd.Series(rolling_lookup.get(a, {}))
            h_stats = pd.Series(stats_lookup.get(h, {}))
            a_stats = pd.Series(stats_lookup.get(a, {}))

            features = build_prediction_row(
                home_team_br=h, away_team_br=a,
                home_rolling=h_roll, away_rolling=a_roll,
                home_batting=h_stats, home_pitching=pd.Series(),
                away_batting=a_stats, away_pitching=pd.Series(),
                game_date=game.get("game_datetime", today_str)[:10],
            )

            pred = predict_game(features, classifier, home_regressor, away_regressor)
            pred.update({
                "home_team_br":    h,
                "away_team_br":    a,
                "home_team_name":  game.get("home_team_name", h),
                "away_team_name":  game.get("away_team_name", a),
                "venue":           game.get("venue", ""),
                "game_datetime":   game.get("game_datetime", today_str),
                "recent_form_home": _get_recent_form(game_logs, h, last_season),
                "recent_form_away": _get_recent_form(game_logs, a, last_season),
            })
            results.append(pred)

    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    results.sort(key=lambda g: g["game_datetime"])
    return results


def _get_recent_form(game_logs: pd.DataFrame, team_br: str,
                     season: int, n: int = 10) -> str:
    """
    Return the last n W/L results for a team as a string like 'W L W W L W W W L W'.
    """
    try:
        team_games = (
            game_logs[
                (game_logs["team_br"] == team_br) &
                (game_logs["season"] == season)
            ]
            .sort_values("date")
            .tail(n)
        )
        return " ".join("W" if win else "L" for win in team_games["win"])
    except Exception:
        return "N/A"
