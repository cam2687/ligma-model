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
    p=0.60 -> -150 (favorite)
    p=0.40 -> +150 (underdog)
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
        data = statsapi.player_stats(player_id=pitcher_id, group="pitching", type="yearByYear")
        for stat in reversed(data.get("stats", [])):
            if str(stat.get("season")) == str(season):
                era = stat.get("era") or 4.5
                return float(era)
    except Exception:
        pass
    return 4.5


def _get_rest_days(game_logs: pd.DataFrame, team_br: str, before_date: str) -> float:
    """
    Calculate days of rest for a team before a given date.
    Returns days since last game, capped at 14. Defaults to 4 (avg MLB spacing).
    """
    try:
        target = pd.to_datetime(before_date)
        team_games = game_logs[
            (game_logs["team_br"] == team_br) &
            (game_logs["date"] < target)
        ].sort_values("date")
        if team_games.empty:
            return 4.0
        last_game_date = pd.to_datetime(team_games.iloc[-1]["date"])
        days = (target - last_game_date).days
        return float(min(max(days, 0), 14))
    except Exception:
        return 4.0


def _get_h2h_win_rate(game_logs: pd.DataFrame,
                      home_br: str, away_br: str,
                      before_date: str) -> float:
    """
    Historical home-team win rate for this specific (home, away) matchup,
    using all games before before_date.
    """
    try:
        target = pd.to_datetime(before_date)
        h2h = game_logs[
            (game_logs["team_br"]     == home_br) &
            (game_logs["opponent_br"] == away_br) &
            (game_logs["home_flag"]   == True) &
            (game_logs["date"]        <  target)
        ]
        if h2h.empty:
            return 0.5
        return float(h2h["win"].mean())
    except Exception:
        return 0.5


def _confidence_from_probs(probs: list[float]) -> tuple[str, float]:
    top_prob = max(probs)
    confidence_pct = round(top_prob * 100, 1)
    if top_prob < 0.45:
        return "Low", confidence_pct
    if top_prob < 0.6:
        return "Moderate", confidence_pct
    return "High", confidence_pct


def _soccer_probs(classifier, X: pd.DataFrame) -> tuple[float, float, float]:
    classes = list(getattr(classifier, "classes_", [0, 1, 2]))
    raw_probs = classifier.predict_proba(X)[0]
    prob_map = {int(cls): float(prob) for cls, prob in zip(classes, raw_probs)}
    away = prob_map.get(0, 0.0)
    draw = prob_map.get(1, 0.0)
    home = prob_map.get(2, 0.0)
    return home, draw, away


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
    X = game_features.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

    home_features = X.reindex(
        columns=getattr(home_regressor, "feature_names_in_", FEATURE_COLUMNS),
        fill_value=0.0,
    )
    away_features = X.reindex(
        columns=getattr(away_regressor, "feature_names_in_", FEATURE_COLUMNS),
        fill_value=0.0,
    )

    pred_home_runs = float(max(0.0, home_regressor.predict(home_features)[0]))
    pred_away_runs = float(max(0.0, away_regressor.predict(away_features)[0]))
    predicted_total = pred_home_runs + pred_away_runs

    if SPORT == "mlb":
        home_win_prob = float(classifier.predict_proba(X)[0, 1])
        away_win_prob = 1.0 - home_win_prob
        home_ml = prob_to_moneyline(home_win_prob)
        away_ml = prob_to_moneyline(away_win_prob)
        confidence, confidence_pct = _confidence_from_probs([home_win_prob, away_win_prob])
        return {
            "home_win_prob":      round(home_win_prob, 3),
            "away_win_prob":      round(away_win_prob, 3),
            "predicted_total":    round(predicted_total, 1),
            "pred_home_runs":     round(pred_home_runs, 1),
            "pred_away_runs":     round(pred_away_runs, 1),
            "home_moneyline":     home_ml,
            "away_moneyline":     away_ml,
            "home_moneyline_str": format_moneyline(home_ml),
            "away_moneyline_str": format_moneyline(away_ml),
            "predicted_winner":   "home" if home_win_prob >= away_win_prob else "away",
            "confidence":         confidence,
            "confidence_pct":     round(confidence_pct, 1),
        }
    elif SPORT == "soccer":
        home_win_prob, draw_prob, away_win_prob = _soccer_probs(classifier, X)
        home_ml = prob_to_moneyline(home_win_prob)
        draw_ml = prob_to_moneyline(draw_prob)
        away_ml = prob_to_moneyline(away_win_prob)
        confidence, confidence_pct = _confidence_from_probs([home_win_prob, draw_prob, away_win_prob])
        predicted_winner = max(
            {"home": home_win_prob, "draw": draw_prob, "away": away_win_prob},
            key=lambda k: {"home": home_win_prob, "draw": draw_prob, "away": away_win_prob}[k],
        )
        return {
            "home_win_prob":      round(home_win_prob, 3),
            "draw_prob":          round(draw_prob, 3),
            "away_win_prob":      round(away_win_prob, 3),
            "predicted_total":    round(predicted_total, 1),
            "pred_home_goals":    round(pred_home_runs, 1),
            "pred_away_goals":    round(pred_away_runs, 1),
            "home_moneyline":     home_ml,
            "draw_moneyline":     draw_ml,
            "away_moneyline":     away_ml,
            "home_moneyline_str": format_moneyline(home_ml),
            "draw_moneyline_str": format_moneyline(draw_ml),
            "away_moneyline_str": format_moneyline(away_ml),
            "predicted_winner":   predicted_winner,
            "confidence":         confidence,
            "confidence_pct":     round(confidence_pct, 1),
        }
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")


# ---------------------------------------------------------------------------
# Full pipeline: today's games
# ---------------------------------------------------------------------------

def predict_today(
    game_logs: pd.DataFrame,
    batting_by_season: dict,
    pitching_by_season: dict,
    bullpen_by_season: dict = None,
) -> list[dict]:
    """
    Orchestrate predictions for all of today's scheduled games.

    New parameter:
      bullpen_by_season : {season: DataFrame} with bullpen ERA, or None.
    """
    from src.data.features import get_end_of_season_rolling, build_prediction_row

    bullpen_by_season = bullpen_by_season or {}

    classifier, home_regressor, away_regressor, _ = load_models()
    last_season = max(TRAIN_SEASONS)
    today_str   = date.today().strftime("%Y-%m-%d")

    end_rolling = get_end_of_season_rolling(game_logs, last_season)
    if SPORT == "soccer" and "league" in end_rolling.columns:
        rolling_lookup = {
            (row["league"], row["team_br"]): row.to_dict()
            for _, row in end_rolling.iterrows()
        }
    else:
        rolling_lookup = end_rolling.set_index("team_br").to_dict("index")

    results = []

    if SPORT == "mlb":
        from src.data.fetch import fetch_today_schedule, fetch_weather_forecast

        batting_df  = batting_by_season.get(last_season, pd.DataFrame())
        pitching_df = pitching_by_season.get(last_season, pd.DataFrame())
        bullpen_df  = bullpen_by_season.get(last_season, pd.DataFrame())

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
        bull_lookup = (
            bullpen_df.set_index("team_br").to_dict("index")
            if not bullpen_df.empty and "team_br" in bullpen_df.columns
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

            # Starting pitcher ERA
            h_pitcher_era = _get_pitcher_era(game.get("home_pitcher_id"), last_season)
            a_pitcher_era = _get_pitcher_era(game.get("away_pitcher_id"), last_season)

            # Bullpen ERA
            h_bullpen_era = float(bull_lookup.get(h, {}).get("bullpen_era", 4.50))
            a_bullpen_era = float(bull_lookup.get(a, {}).get("bullpen_era", 4.50))

            # Rest days
            h_rest = _get_rest_days(game_logs, h, today_str)
            a_rest = _get_rest_days(game_logs, a, today_str)

            # Head-to-head historical win rate
            h2h = _get_h2h_win_rate(game_logs, h, a, today_str)

            # Weather for the home stadium
            weather = fetch_weather_forecast(h, today_str)

            features = build_prediction_row(
                home_team_br=h, away_team_br=a,
                home_rolling=h_roll, away_rolling=a_roll,
                home_batting=h_bat, home_pitching=h_pit,
                away_batting=a_bat, away_pitching=a_pit,
                game_date=today_str,
                home_pitcher_era=h_pitcher_era,
                away_pitcher_era=a_pitcher_era,
                home_bullpen_era=h_bullpen_era,
                away_bullpen_era=a_bullpen_era,
                h2h_win_rate=h2h,
                home_rest_days=h_rest,
                away_rest_days=a_rest,
                temp_f=weather["temp_f"],
                wind_mph=weather["wind_mph"],
                is_dome=weather["is_dome"],
            )

            pred = predict_game(features, classifier, home_regressor, away_regressor)
            pred.update({
                "home_team_br":     h,
                "away_team_br":     a,
                "home_team_name":   game.get("home_team_name") or BR_TO_FULL_NAME.get(h, h),
                "away_team_name":   game.get("away_team_name") or BR_TO_FULL_NAME.get(a, a),
                "home_pitcher":     game.get("home_pitcher_name", "TBD"),
                "away_pitcher":     game.get("away_pitcher_name", "TBD"),
                "venue":            game.get("venue", ""),
                "game_datetime":    game.get("game_datetime", ""),
                "home_rest_days":   h_rest,
                "away_rest_days":   a_rest,
                "h2h_win_rate":     round(h2h, 3),
                "temp_f":           weather["temp_f"],
                "wind_mph":         weather["wind_mph"],
                "recent_form_home": _get_recent_form(game_logs, h, last_season),
                "recent_form_away": _get_recent_form(game_logs, a, last_season),
            })
            results.append(pred)

    elif SPORT == "soccer":
        from src.data.fetch import fetch_soccer_fixtures

        team_stats_df = batting_by_season.get(last_season, pd.DataFrame())
        if not team_stats_df.empty and {"team_br", "league"}.issubset(team_stats_df.columns):
            stats_lookup = {
                (row["league"], row["team_br"]): row.to_dict()
                for _, row in team_stats_df.iterrows()
            }
            fallback_stats = (
                team_stats_df.sort_values("games_played", ascending=False)
                .drop_duplicates("team_br")
                .set_index("team_br")
                .to_dict("index")
            )
        else:
            stats_lookup = {}
            fallback_stats = {}

        games = fetch_soccer_fixtures(today_str)
        if not games:
            print("  [predict] No soccer fixtures found.")
            return []

        for game in games:
            h = game["home_team_br"]
            a = game["away_team_br"]
            league = game.get("league", "")

            h_roll = pd.Series(rolling_lookup.get((league, h), rolling_lookup.get(h, {})))
            a_roll = pd.Series(rolling_lookup.get((league, a), rolling_lookup.get(a, {})))
            h_stats = pd.Series(stats_lookup.get((league, h), fallback_stats.get(h, {})))
            a_stats = pd.Series(stats_lookup.get((league, a), fallback_stats.get(a, {})))

            features = build_prediction_row(
                home_team_br=h, away_team_br=a,
                home_rolling=h_roll, away_rolling=a_roll,
                home_batting=h_stats, home_pitching=pd.Series(dtype=float),
                away_batting=a_stats, away_pitching=pd.Series(dtype=float),
                game_date=game.get("game_datetime", today_str)[:10],
            )

            pred = predict_game(features, classifier, home_regressor, away_regressor)
            pred.update({
                "home_team_br":     h,
                "away_team_br":     a,
                "home_team_name":   game.get("home_team_name", h),
                "away_team_name":   game.get("away_team_name", a),
                "venue":            game.get("venue", ""),
                "league":           league,
                "game_datetime":    game.get("game_datetime", today_str),
                "recent_form_home": _get_recent_form(game_logs, h, last_season, league=league),
                "recent_form_away": _get_recent_form(game_logs, a, last_season, league=league),
            })
            results.append(pred)

    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    results.sort(key=lambda g: g.get("game_datetime", ""))
    return results


def _get_recent_form(game_logs: pd.DataFrame, team_br: str,
                     season: int, n: int = 10,
                     league: str | None = None) -> str:
    """Return the last n results as a compact form string."""
    try:
        mask = (
            (game_logs["team_br"] == team_br) &
            (game_logs["season"] == season)
        )
        if league and "league" in game_logs.columns:
            mask &= game_logs["league"] == league
        team_games = game_logs[mask].sort_values("date").tail(n)
        if SPORT == "soccer" and {"goals_scored", "goals_allowed"}.issubset(team_games.columns):
            outcomes = []
            for _, row in team_games.iterrows():
                if row["goals_scored"] > row["goals_allowed"]:
                    outcomes.append("W")
                elif row["goals_scored"] == row["goals_allowed"]:
                    outcomes.append("D")
                else:
                    outcomes.append("L")
            return " ".join(outcomes)
        return " ".join("W" if win else "L" for win in team_games["win"])
    except Exception:
        return "N/A"
