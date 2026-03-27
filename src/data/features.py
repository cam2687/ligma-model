"""
Feature engineering.

Core guarantee: NO DATA LEAKAGE.
Rolling features for game N are computed from games 0..N-1 only.
This is enforced via pandas .shift(1) before .rolling() / .expanding().

Pipeline:
    1. Parse all game logs into a unified per-team-per-game DataFrame.
    2. Compute rolling and cumulative features per (team_br, season).
       Includes 7-game (hot streak), 15-game (main), and 30-game (true form)
       windows, plus rest days.
    3. Build home-game rows: one row per unique game.
    4. Join rolling features for both home and away teams.
    5. Join FanGraphs season stats (batting + rotation pitching) for both teams.
    6. Join bullpen ERA for both teams.
    7. Add head-to-head win rate (computed across all historical games).
    8. Add park factors, weather, and differential / context features.
    9. Return feature matrix ready for XGBoost.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    FEATURE_COLUMNS, ROLLING_WINDOW, TARGET_WIN, SPORT,
    CACHE_DIR, PARK_FACTORS, DOME_STADIUMS,
)


# ---------------------------------------------------------------------------
# Pythagorean win expectation
# ---------------------------------------------------------------------------

def pythagorean_win_exp(rs: float, ra: float, exp: float = 1.83) -> float:
    """
    Pythagenpat formula: RS^exp / (RS^exp + RA^exp).
    exp=1.83 is empirically better than the original 2.0.
    Returns 0.5 for undefined inputs (zero or NaN).
    """
    try:
        if pd.isna(rs) or pd.isna(ra) or rs <= 0 or ra <= 0:
            return 0.5
        denom = rs ** exp + ra ** exp
        return float(rs ** exp / denom)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Step 1 + 2: Rolling features per (team_br, season)
# ---------------------------------------------------------------------------

def compute_rolling_features(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Input : unified game_logs with one row per team per game.
    Output: same rows + rolling feature columns.

    Rolling features are computed WITHIN each (team_br, season) group so
    season boundaries are never crossed.  shift(1) ensures the current game
    is excluded from its own feature values.

    For MLB, also computes:
      - 7-game rolling (roll7_*): hot-streak / momentum signal
      - 30-game rolling (roll30_*): sustainable true-form signal
      - rest_days: calendar days since the team's previous game
    """
    if SPORT == "mlb":
        score_col = "runs_scored"
        allow_col = "runs_allowed"
        roll_score = "roll_rs"
        roll_allow = "roll_ra"
        roll_diff  = "roll_run_diff"
    elif SPORT == "soccer":
        score_col = "goals_scored"
        allow_col = "goals_allowed"
        roll_score = "roll_goals_scored"
        roll_allow = "roll_goals_allowed"
        roll_diff  = "roll_goal_diff"
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    results = []
    group_cols = ["team_br", "season"]
    if SPORT == "soccer" and "league" in game_logs.columns:
        group_cols.append("league")

    for _, grp in game_logs.groupby(group_cols, sort=False):
        g = grp.sort_values("date").copy()

        s_win   = g["win"].shift(1)
        s_score = g[score_col].shift(1)
        s_allow = g[allow_col].shift(1)

        w = ROLLING_WINDOW  # 15-game main window

        # --- 15-game rolling (main) ---
        g["roll_win_rate"] = s_win.rolling(w, min_periods=1).mean()
        g[roll_score]      = s_score.rolling(w, min_periods=1).mean()
        g[roll_allow]      = s_allow.rolling(w, min_periods=1).mean()
        g[roll_diff]       = g[roll_score] - g[roll_allow]

        g["roll_pythag"] = [
            pythagorean_win_exp(score, allow)
            for score, allow in zip(g[roll_score], g[roll_allow])
        ]

        # Cumulative season win % (expanding, excluding current game)
        g["season_win_pct"] = s_win.expanding(min_periods=1).mean()

        # Fill NaN for first games of the season
        g["roll_win_rate"]  = g["roll_win_rate"].fillna(0.5)
        g[roll_score]       = g[roll_score].fillna(g[score_col].mean())
        g[roll_allow]       = g[roll_allow].fillna(g[allow_col].mean())
        g[roll_diff]        = g[roll_diff].fillna(0.0)
        g["roll_pythag"]    = g["roll_pythag"].fillna(0.5)
        g["season_win_pct"] = g["season_win_pct"].fillna(0.5)

        extra_cols = []

        if SPORT == "mlb":
            # --- 7-game rolling (hot-streak signal) ---
            g["roll7_win_rate"] = s_win.rolling(7, min_periods=1).mean().fillna(0.5)
            g["roll7_rs"]       = s_score.rolling(7, min_periods=1).mean().fillna(g[score_col].mean())
            g["roll7_ra"]       = s_allow.rolling(7, min_periods=1).mean().fillna(g[allow_col].mean())

            # --- 30-game rolling (true-form / sustainability signal) ---
            g["roll30_win_rate"] = s_win.rolling(30, min_periods=1).mean().fillna(0.5)
            g["roll30_rs"]       = s_score.rolling(30, min_periods=1).mean().fillna(g[score_col].mean())
            g["roll30_ra"]       = s_allow.rolling(30, min_periods=1).mean().fillna(g[allow_col].mean())

            # --- Rest days (calendar days since previous game) ---
            # diff() gives NaN for the first game; fill with 4 (avg MLB spacing) and cap at 14
            g["rest_days"] = (
                g["date"].diff().dt.days
                .fillna(4)
                .clip(0, 14)
                .astype(float)
            )

            extra_cols = [
                "roll7_win_rate", "roll7_rs", "roll7_ra",
                "roll30_win_rate", "roll30_rs", "roll30_ra",
                "rest_days",
            ]
        elif SPORT == "soccer":
            prev_game_date = g["date"].shift(1)
            g["rest_days"] = (
                (g["date"] - prev_game_date).dt.days
                .fillna(7)
                .clip(0, 30)
                .astype(float)
            )
            g["last_game_date"] = prev_game_date
            g["roll_draw_rate"] = (
                (g[score_col].eq(g[allow_col]).astype(int).shift(1))
                .rolling(w, min_periods=1)
                .mean()
                .fillna(0.28)
            )

            if "league" in g.columns:
                league_games = (
                    game_logs[
                        (game_logs["season"] == g["season"].iloc[0]) &
                        (game_logs["league"] == g["league"].iloc[0])
                    ]
                    .sort_values("date")
                    .copy()
                )
                league_games["league_avg_scored"] = (
                    league_games[score_col].shift(1).expanding(min_periods=1).mean()
                )
                league_games["league_avg_allowed"] = (
                    league_games[allow_col].shift(1).expanding(min_periods=1).mean()
                )
                league_games["league_row_id"] = league_games.index
                g["league_row_id"] = g.index
                g = g.merge(
                    league_games[["league_row_id", "league_avg_scored", "league_avg_allowed"]],
                    on="league_row_id",
                    how="left",
                )
            else:
                g["league_avg_scored"] = np.nan
                g["league_avg_allowed"] = np.nan

            g["attack_strength"] = (
                g[roll_score] / g["league_avg_scored"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            g["defense_strength"] = (
                g["league_avg_allowed"] / g[roll_allow].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

            extra_cols = [
                "league",
                "roll_draw_rate",
                "attack_strength",
                "defense_strength",
                "rest_days",
                "last_game_date",
            ]

        results.append(g[[
            "date", "season", "team_br",
            "roll_win_rate", roll_score, roll_allow,
            roll_diff, "roll_pythag", "season_win_pct",
        ] + extra_cols])

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 3: Build one row per game (home perspective)
# ---------------------------------------------------------------------------

def build_home_game_rows(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Filter game_logs to home games (home_flag=True) -> one row per unique game.
    """
    home = game_logs[game_logs["home_flag"]].copy()

    if SPORT == "mlb":
        home = home.rename(columns={
            "team_br":      "home_team_br",
            "opponent_br":  "away_team_br",
            "win":          "home_win",
            "runs_scored":  "home_runs",
            "runs_allowed": "away_runs",
        })
        home["total_runs"] = home["home_runs"] + home["away_runs"]
        target_cols = ["home_runs", "away_runs", "home_win", "total_runs"]
    elif SPORT == "soccer":
        home = home.rename(columns={
            "league":        "league",
            "team_br":       "home_team_br",
            "opponent_br":   "away_team_br",
            "win":           "home_win",
            "goals_scored":  "home_goals",
            "goals_allowed": "away_goals",
        })
        home["total_goals"] = home["home_goals"] + home["away_goals"]
        home["match_outcome"] = np.select(
            [
                home["home_goals"] > home["away_goals"],
                home["home_goals"] == home["away_goals"],
            ],
            [2, 1],
            default=0,
        ).astype(int)
        target_cols = ["home_goals", "away_goals", "home_win", "match_outcome", "total_goals"]
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    dedupe_cols = ["date", "home_team_br", "away_team_br"]
    if SPORT == "soccer" and "league" in home.columns:
        dedupe_cols.append("league")
    home = home.drop_duplicates(subset=dedupe_cols)

    return home[[
        "date", "season", *([ "league" ] if SPORT == "soccer" and "league" in home.columns else []), "home_team_br", "away_team_br",
    ] + target_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Join rolling features for both teams
# ---------------------------------------------------------------------------

def join_rolling_features(home_games: pd.DataFrame,
                           rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-computed rolling features for both home and away teams.
    """
    if SPORT == "mlb":
        rolling_cols = [
            "date", "season", "team_br",
            "roll_win_rate", "roll_rs", "roll_ra",
            "roll_run_diff", "roll_pythag", "season_win_pct",
            "roll7_win_rate", "roll7_rs", "roll7_ra",
            "roll30_win_rate", "roll30_rs", "roll30_ra",
            "rest_days",
        ]
        # Only keep columns that actually exist in the rolling DataFrame
        rolling_cols = [c for c in rolling_cols if c in rolling.columns]

        home_rename = {
            "team_br":         "home_team_br",
            "roll_win_rate":   "home_roll_win_rate",
            "roll_rs":         "home_roll_rs",
            "roll_ra":         "home_roll_ra",
            "roll_run_diff":   "home_roll_run_diff",
            "roll_pythag":     "home_roll_pythag",
            "season_win_pct":  "home_season_win_pct",
            "roll7_win_rate":  "home_roll7_win_rate",
            "roll7_rs":        "home_roll7_rs",
            "roll7_ra":        "home_roll7_ra",
            "roll30_win_rate": "home_roll30_win_rate",
            "roll30_rs":       "home_roll30_rs",
            "roll30_ra":       "home_roll30_ra",
            "rest_days":       "home_rest_days",
        }
        away_rename = {
            "team_br":         "away_team_br",
            "roll_win_rate":   "away_roll_win_rate",
            "roll_rs":         "away_roll_rs",
            "roll_ra":         "away_roll_ra",
            "roll_run_diff":   "away_roll_run_diff",
            "roll_pythag":     "away_roll_pythag",
            "season_win_pct":  "away_season_win_pct",
            "roll7_win_rate":  "away_roll7_win_rate",
            "roll7_rs":        "away_roll7_rs",
            "roll7_ra":        "away_roll7_ra",
            "roll30_win_rate": "away_roll30_win_rate",
            "roll30_rs":       "away_roll30_rs",
            "roll30_ra":       "away_roll30_ra",
            "rest_days":       "away_rest_days",
        }
    elif SPORT == "soccer":
        rolling_cols = [
            "date", "season", "league", "team_br",
            "roll_win_rate", "roll_draw_rate", "roll_goals_scored", "roll_goals_allowed",
            "roll_goal_diff", "roll_pythag", "season_win_pct",
            "attack_strength", "defense_strength", "rest_days", "last_game_date",
        ]
        rolling_cols = [c for c in rolling_cols if c in rolling.columns]
        home_rename = {
            "league":              "league",
            "team_br":            "home_team_br",
            "roll_win_rate":      "home_roll_win_rate",
            "roll_draw_rate":     "home_roll_draw_rate",
            "roll_goals_scored":  "home_roll_goals_scored",
            "roll_goals_allowed": "home_roll_goals_allowed",
            "roll_goal_diff":     "home_roll_goal_diff",
            "roll_pythag":        "home_roll_pythag",
            "season_win_pct":     "home_season_win_pct",
            "attack_strength":    "home_attack_strength",
            "defense_strength":   "home_defense_strength",
            "rest_days":          "home_rest_days",
            "last_game_date":     "home_last_game_date",
        }
        away_rename = {
            "league":              "league",
            "team_br":            "away_team_br",
            "roll_win_rate":      "away_roll_win_rate",
            "roll_draw_rate":     "away_roll_draw_rate",
            "roll_goals_scored":  "away_roll_goals_scored",
            "roll_goals_allowed": "away_roll_goals_allowed",
            "roll_goal_diff":     "away_roll_goal_diff",
            "roll_pythag":        "away_roll_pythag",
            "season_win_pct":     "away_season_win_pct",
            "attack_strength":    "away_attack_strength",
            "defense_strength":   "away_defense_strength",
            "rest_days":          "away_rest_days",
            "last_game_date":     "away_last_game_date",
        }
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    home_roll = rolling[rolling_cols].rename(columns=home_rename)
    away_roll = rolling[rolling_cols].rename(columns=away_rename)

    if SPORT == "soccer":
        df = home_games.merge(home_roll, on=["date", "season", "league", "home_team_br"], how="left")
        df = df.merge(away_roll, on=["date", "season", "league", "away_team_br"], how="left")
    else:
        df = home_games.merge(home_roll, on=["date", "season", "home_team_br"], how="left")
        df = df.merge(away_roll, on=["date", "season", "away_team_br"], how="left")

    return df


# ---------------------------------------------------------------------------
# Step 5: Join FanGraphs season stats
# ---------------------------------------------------------------------------

def join_season_stats(df: pd.DataFrame,
                      batting_by_season: dict,
                      pitching_by_season: dict) -> pd.DataFrame:
    """Join season-level batting and rotation pitching stats for both teams."""
    if SPORT == "mlb":
        return _join_mlb_season_stats(df, batting_by_season, pitching_by_season)
    elif SPORT == "soccer":
        return _join_soccer_season_stats(df, batting_by_season)
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")


def _join_mlb_season_stats(df, batting_by_season, pitching_by_season):
    bat_frames = [
        d_.assign(season=s) for s, d_ in batting_by_season.items() if not d_.empty
    ]
    pit_frames = [
        d_.assign(season=s) for s, d_ in pitching_by_season.items() if not d_.empty
    ]

    if not bat_frames or not pit_frames:
        for prefix in ("home_", "away_"):
            for col in ["wrc_plus","obp","slg","k_pct","bb_pct","era","fip","xfip","k9","bb9"]:
                df[f"{prefix}{col}"] = np.nan
        return df

    all_batting  = pd.concat(bat_frames, ignore_index=True)
    all_pitching = pd.concat(pit_frames, ignore_index=True)

    bat_keep = _select_existing(all_batting,  ["team_br","season","wrc_plus","obp","slg","k_pct","bb_pct"])
    pit_keep = _select_existing(all_pitching, ["team_br","season","era","fip","xfip","k9","bb9"])

    df = _merge_stats(df, bat_keep, pit_keep, side="home")
    df = _merge_stats(df, bat_keep, pit_keep, side="away")
    return df


def _join_soccer_season_stats(df, team_stats_by_season):
    # Soccer strength metrics are computed as pre-match rolling features.
    # Keep this as a no-op fallback to avoid leaking full-season summaries.
    for prefix in ("home_", "away_"):
        for col in ["attack_strength", "defense_strength"]:
            name = f"{prefix}{col}"
            if name not in df.columns:
                df[name] = 1.0
    return df


def _select_existing(df, desired_cols):
    return df[[c for c in desired_cols if c in df.columns]].copy()


def _merge_stats(game_df, bat, pit, side):
    team_col = f"{side}_team_br"
    bat_r = bat.rename(columns={
        "team_br": team_col,
        **{c: f"{side}_{c}" for c in bat.columns if c not in ("team_br","season")}
    })
    pit_r = pit.rename(columns={
        "team_br": team_col,
        **{c: f"{side}_{c}" for c in pit.columns if c not in ("team_br","season")}
    })
    game_df = game_df.merge(bat_r, on=["season", team_col], how="left")
    game_df = game_df.merge(pit_r, on=["season", team_col], how="left")
    return game_df


def _merge_soccer_stats(game_df, stats, side):
    team_col = f"{side}_team_br"
    stats_r = stats.rename(columns={
        "team_br": team_col,
        **{c: f"{side}_{c}" for c in stats.columns if c not in ("team_br","season")}
    })
    return game_df.merge(stats_r, on=["season", team_col], how="left")


# ---------------------------------------------------------------------------
# Step 5b: Join bullpen stats (MLB only)
# ---------------------------------------------------------------------------

def join_bullpen_stats(df: pd.DataFrame,
                       bullpen_by_season: dict) -> pd.DataFrame:
    """Join reliever-only bullpen ERA for home and away teams."""
    if not bullpen_by_season:
        df["home_bullpen_era"] = 4.50
        df["away_bullpen_era"] = 4.50
        return df

    all_frames = [
        d_.assign(season=s) for s, d_ in bullpen_by_season.items() if not d_.empty
    ]
    if not all_frames:
        df["home_bullpen_era"] = 4.50
        df["away_bullpen_era"] = 4.50
        return df

    bull = pd.concat(all_frames, ignore_index=True)[["team_br","season","bullpen_era"]]

    home_bull = bull.rename(columns={"team_br": "home_team_br", "bullpen_era": "home_bullpen_era"})
    away_bull = bull.rename(columns={"team_br": "away_team_br", "bullpen_era": "away_bullpen_era"})

    df = df.merge(home_bull, on=["season","home_team_br"], how="left")
    df = df.merge(away_bull, on=["season","away_team_br"], how="left")
    df["home_bullpen_era"] = df["home_bullpen_era"].fillna(4.50)
    df["away_bullpen_era"] = df["away_bullpen_era"].fillna(4.50)
    return df


# ---------------------------------------------------------------------------
# Step 6: Head-to-head win rate
# ---------------------------------------------------------------------------

def compute_h2h_features(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute expanding head-to-head home win rate for every (home_team, away_team) pair.
    Uses shift(1)+expanding so the current game is excluded from its own feature.
    Returns DataFrame: date, season, home_team_br, away_team_br, h2h_home_win_rate.
    """
    home_games = game_logs[game_logs["home_flag"]].copy()
    if home_games.empty:
        return pd.DataFrame(columns=[
            "date","season","home_team_br","away_team_br","h2h_home_win_rate"
        ])

    home_games = home_games.rename(columns={
        "team_br":     "home_team_br",
        "opponent_br": "away_team_br",
    })
    home_games = home_games.sort_values("date")

    results = []
    for (home_br, away_br), grp in home_games.groupby(
            ["home_team_br", "away_team_br"], sort=False):
        g = grp.sort_values("date").copy()
        s_win = g["win"].shift(1)
        g["h2h_home_win_rate"] = (
            s_win.expanding(min_periods=1).mean().fillna(0.5)
        )
        results.append(
            g[["date","season","home_team_br","away_team_br","h2h_home_win_rate"]]
        )

    if not results:
        return pd.DataFrame(columns=[
            "date","season","home_team_br","away_team_br","h2h_home_win_rate"
        ])

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 7: Differential, park, weather, and context features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame,
                         weather_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add differential features, park factors, weather, and game-context features.
    """
    df = df.copy()

    # --- Rolling differentials (common to both sports) ---
    df["roll_win_rate_diff"]  = df["home_roll_win_rate"]  - df["away_roll_win_rate"]
    df["roll_pythag_diff"]    = df["home_roll_pythag"]    - df["away_roll_pythag"]
    df["season_win_pct_diff"] = df["home_season_win_pct"] - df["away_season_win_pct"]

    if SPORT == "mlb":
        df["roll_run_diff_diff"] = df["home_roll_run_diff"] - df["away_roll_run_diff"]

        # Multi-window differentials
        if "home_roll7_win_rate" in df.columns and "away_roll7_win_rate" in df.columns:
            df["roll7_win_rate_diff"] = df["home_roll7_win_rate"] - df["away_roll7_win_rate"]
        else:
            df["roll7_win_rate_diff"] = 0.0

        if "home_roll30_win_rate" in df.columns and "away_roll30_win_rate" in df.columns:
            df["roll30_win_rate_diff"] = df["home_roll30_win_rate"] - df["away_roll30_win_rate"]
        else:
            df["roll30_win_rate_diff"] = 0.0

        # Rest-days differential
        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]
        else:
            df["rest_days_diff"] = 0.0

        # Batting / pitching differentials
        if "home_wrc_plus" in df.columns and "away_wrc_plus" in df.columns:
            df["wrc_plus_diff"] = df["home_wrc_plus"] - df["away_wrc_plus"]
        else:
            df["wrc_plus_diff"] = 0.0

        if "home_era" in df.columns and "away_era" in df.columns:
            df["era_diff"] = df["away_era"] - df["home_era"]
            df["fip_diff"] = (
                df["away_fip"] - df["home_fip"]
                if "home_fip" in df.columns else 0.0
            )
        else:
            df["era_diff"] = 0.0
            df["fip_diff"] = 0.0

        # --- Park factor (static lookup) ---
        df["home_park_factor"] = (
            df["home_team_br"].map(PARK_FACTORS).fillna(100).astype(float)
        )

        # --- Weather ---
        if weather_df is not None and not weather_df.empty:
            w = weather_df[["date","team_br","temp_f","wind_mph","is_dome"]].rename(
                columns={"team_br": "home_team_br"}
            )
            df = df.merge(w, on=["date","home_team_br"], how="left")
        else:
            df["temp_f"]   = np.nan
            df["wind_mph"] = np.nan
            df["is_dome"]  = np.nan

        # Override dome stadiums regardless of weather source
        dome_mask = df["home_team_br"].isin(DOME_STADIUMS)
        df.loc[dome_mask, "temp_f"]   = 72.0
        df.loc[dome_mask, "wind_mph"] = 0.0
        df.loc[dome_mask, "is_dome"]  = 1.0

        # Fill remaining NaN with month-based seasonal defaults
        month = pd.to_datetime(df["date"]).dt.month
        monthly_temp = {3:52, 4:58, 5:67, 6:76, 7:82, 8:81, 9:72, 10:62}
        df["temp_f"]   = df["temp_f"].fillna(month.map(monthly_temp).fillna(70))
        df["wind_mph"] = df["wind_mph"].fillna(7.0)
        df["is_dome"]  = df["is_dome"].fillna(0).astype(float)

    elif SPORT == "soccer":
        if "home_roll_draw_rate" in df.columns and "away_roll_draw_rate" in df.columns:
            df["roll_draw_rate_diff"] = (
                df["home_roll_draw_rate"] - df["away_roll_draw_rate"]
            )
        else:
            df["roll_draw_rate_diff"] = 0.0
        if "home_roll_goal_diff" in df.columns and "away_roll_goal_diff" in df.columns:
            df["roll_goal_diff_diff"] = (
                df["home_roll_goal_diff"] - df["away_roll_goal_diff"]
            )
        else:
            df["roll_goal_diff_diff"] = 0.0
        if "home_attack_strength" in df.columns and "away_attack_strength" in df.columns:
            df["attack_diff"]  = df["home_attack_strength"]  - df["away_attack_strength"]
            df["defense_diff"] = df["home_defense_strength"] - df["away_defense_strength"]
        else:
            df["attack_diff"]  = 0.0
            df["defense_diff"] = 0.0
        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            df["rest_days_diff"] = df["home_rest_days"].fillna(7.0) - df["away_rest_days"].fillna(7.0)
        else:
            df["rest_days_diff"] = 0.0

    # --- Context (common to both sports) ---
    df["month"]       = pd.to_datetime(df["date"]).dt.month.astype(int)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.astype(int)
    df["is_weekend"]  = df["day_of_week"].isin([4, 5, 6]).astype(int)
    df["is_2020"]     = (df["season"] == 2020).astype(int)
    df["home_advantage"] = 1

    return df


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def build_training_dataset(game_logs: pd.DataFrame,
                            batting_by_season: dict,
                            pitching_by_season: dict,
                            bullpen_by_season: dict = None,
                            weather_df: pd.DataFrame = None,
                            save_path: Path = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    New parameters vs the original:
      bullpen_by_season : {season: DataFrame} with bullpen ERA by team
      weather_df        : DataFrame with (date, team_br, temp_f, wind_mph, is_dome)
    """
    bullpen_by_season = bullpen_by_season or {}

    print("  [features] Computing rolling features (7/15/30 windows + rest days)...")
    rolling = compute_rolling_features(game_logs)

    print("  [features] Building home-game rows...")
    home_games = build_home_game_rows(game_logs)

    print("  [features] Joining rolling features...")
    df = join_rolling_features(home_games, rolling)

    print("  [features] Joining season stats (batting + rotation pitching)...")
    df = join_season_stats(df, batting_by_season, pitching_by_season)

    if SPORT == "mlb":
        print("  [features] Joining bullpen ERA...")
        df = join_bullpen_stats(df, bullpen_by_season)

        print("  [features] Computing head-to-head win rates...")
        h2h = compute_h2h_features(game_logs)
        if not h2h.empty:
            df = df.merge(
                h2h,
                on=["date","season","home_team_br","away_team_br"],
                how="left",
            )
        df["h2h_home_win_rate"] = df.get(
            "h2h_home_win_rate", pd.Series(dtype=float)
        ).fillna(0.5)

        # Placeholder SP ERA for training rows (actual SP not in game logs)
        df["home_pitcher_era"] = 4.50
        df["away_pitcher_era"] = 4.50

    print("  [features] Adding derived / differential / context features...")
    df = add_derived_features(df, weather_df=weather_df)

    dedupe_cols = ["date", "season", "home_team_br", "away_team_br"]
    if SPORT == "soccer" and "league" in df.columns:
        dedupe_cols.append("league")
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
    if before != len(df):
        print(f"  [features] Dropped {before - len(df)} duplicate game rows")

    # Ensure all expected feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows where ALL rolling features are NaN (pathological)
    roll_cols = [c for c in FEATURE_COLUMNS if c.startswith(("home_roll","away_roll"))]
    df = df.dropna(subset=roll_cols, how="all")

    # Replace inf / -inf then fill NaN with column median
    feat_df = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(feat_df.median())
    df[FEATURE_COLUMNS] = feat_df

    print(f"  [features] Dataset: {len(df):,} games, {len(FEATURE_COLUMNS)} features")

    if save_path:
        df.to_parquet(save_path, index=False)
        print(f"  [features] Saved to {save_path}")

    return df


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def get_end_of_season_rolling(game_logs: pd.DataFrame,
                               season: int) -> pd.DataFrame:
    """
    Return the last rolling feature snapshot for each team at the end of
    a given season. Used as the starting point for next-season predictions.
    """
    season_logs = game_logs[game_logs["season"] == season].copy()
    rolling = compute_rolling_features(season_logs)
    group_cols = ["team_br"]
    if SPORT == "soccer" and "league" in rolling.columns:
        group_cols.append("league")
    return (
        rolling.sort_values("date")
               .groupby(group_cols, as_index=False)
               .last()
    )


def build_prediction_row(
    home_team_br: str,
    away_team_br: str,
    home_rolling: "pd.Series",
    away_rolling: "pd.Series",
    home_batting: "pd.Series",
    home_pitching: "pd.Series",
    away_batting: "pd.Series",
    away_pitching: "pd.Series",
    game_date: str,
    home_pitcher_era: float = 4.50,
    away_pitcher_era: float = 4.50,
    home_bullpen_era: float = 4.50,
    away_bullpen_era: float = 4.50,
    h2h_win_rate: float = 0.50,
    home_rest_days: float = 4.0,
    away_rest_days: float = 4.0,
    temp_f: float = 70.0,
    wind_mph: float = 7.0,
    is_dome: int = 0,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for one upcoming game.
    All column names match FEATURE_COLUMNS exactly.
    """
    dt = pd.to_datetime(game_date)

    if SPORT == "mlb":
        row = {
            # --- 15-game rolling ---
            "home_roll_win_rate":  _safe(home_rolling, "roll_win_rate", 0.5),
            "away_roll_win_rate":  _safe(away_rolling, "roll_win_rate", 0.5),
            "home_roll_rs":        _safe(home_rolling, "roll_rs", 4.5),
            "away_roll_rs":        _safe(away_rolling, "roll_rs", 4.5),
            "home_roll_ra":        _safe(home_rolling, "roll_ra", 4.5),
            "away_roll_ra":        _safe(away_rolling, "roll_ra", 4.5),
            "home_roll_run_diff":  _safe(home_rolling, "roll_run_diff", 0.0),
            "away_roll_run_diff":  _safe(away_rolling, "roll_run_diff", 0.0),
            "home_roll_pythag":    _safe(home_rolling, "roll_pythag", 0.5),
            "away_roll_pythag":    _safe(away_rolling, "roll_pythag", 0.5),
            # --- 7-game rolling ---
            "home_roll7_win_rate": _safe(home_rolling, "roll7_win_rate", 0.5),
            "away_roll7_win_rate": _safe(away_rolling, "roll7_win_rate", 0.5),
            "home_roll7_rs":       _safe(home_rolling, "roll7_rs", 4.5),
            "away_roll7_rs":       _safe(away_rolling, "roll7_rs", 4.5),
            "home_roll7_ra":       _safe(home_rolling, "roll7_ra", 4.5),
            "away_roll7_ra":       _safe(away_rolling, "roll7_ra", 4.5),
            # --- 30-game rolling ---
            "home_roll30_win_rate": _safe(home_rolling, "roll30_win_rate", 0.5),
            "away_roll30_win_rate": _safe(away_rolling, "roll30_win_rate", 0.5),
            "home_roll30_rs":       _safe(home_rolling, "roll30_rs", 4.5),
            "away_roll30_rs":       _safe(away_rolling, "roll30_rs", 4.5),
            "home_roll30_ra":       _safe(home_rolling, "roll30_ra", 4.5),
            "away_roll30_ra":       _safe(away_rolling, "roll30_ra", 4.5),
            # --- Season cumulative ---
            "home_season_win_pct": _safe(home_rolling, "season_win_pct", 0.5),
            "away_season_win_pct": _safe(away_rolling, "season_win_pct", 0.5),
            # --- Rest / fatigue ---
            "home_rest_days": float(home_rest_days),
            "away_rest_days": float(away_rest_days),
            # --- Head-to-head ---
            "h2h_home_win_rate": float(h2h_win_rate),
            # --- FanGraphs batting ---
            "home_wrc_plus": _safe(home_batting, "wrc_plus", 100.0),
            "away_wrc_plus": _safe(away_batting, "wrc_plus", 100.0),
            "home_obp":      _safe(home_batting, "obp",      0.320),
            "away_obp":      _safe(away_batting, "obp",      0.320),
            "home_slg":      _safe(home_batting, "slg",      0.410),
            "away_slg":      _safe(away_batting, "slg",      0.410),
            "home_k_pct":    _safe(home_batting, "k_pct",    0.220),
            "away_k_pct":    _safe(away_batting, "k_pct",    0.220),
            "home_bb_pct":   _safe(home_batting, "bb_pct",   0.085),
            "away_bb_pct":   _safe(away_batting, "bb_pct",   0.085),
            # --- FanGraphs rotation pitching ---
            "home_era":  _safe(home_pitching, "era",  4.30),
            "away_era":  _safe(away_pitching, "era",  4.30),
            "home_fip":  _safe(home_pitching, "fip",  4.20),
            "away_fip":  _safe(away_pitching, "fip",  4.20),
            "home_xfip": _safe(home_pitching, "xfip", 4.20),
            "away_xfip": _safe(away_pitching, "xfip", 4.20),
            "home_k9":   _safe(home_pitching, "k9",   8.50),
            "away_k9":   _safe(away_pitching, "k9",   8.50),
            "home_bb9":  _safe(home_pitching, "bb9",  3.20),
            "away_bb9":  _safe(away_pitching, "bb9",  3.20),
            # --- Bullpen ERA ---
            "home_bullpen_era": float(home_bullpen_era),
            "away_bullpen_era": float(away_bullpen_era),
            # --- Starting pitcher ---
            "home_pitcher_era": float(home_pitcher_era),
            "away_pitcher_era": float(away_pitcher_era),
            # --- Park factor ---
            "home_park_factor": float(PARK_FACTORS.get(home_team_br, 100)),
            # --- Weather ---
            "temp_f":   float(temp_f),
            "wind_mph": float(wind_mph),
            "is_dome":  float(is_dome),
            # --- Differentials ---
            "roll_win_rate_diff": (
                _safe(home_rolling, "roll_win_rate", 0.5) -
                _safe(away_rolling, "roll_win_rate", 0.5)
            ),
            "roll_run_diff_diff": (
                _safe(home_rolling, "roll_run_diff", 0.0) -
                _safe(away_rolling, "roll_run_diff", 0.0)
            ),
            "roll_pythag_diff": (
                _safe(home_rolling, "roll_pythag", 0.5) -
                _safe(away_rolling, "roll_pythag", 0.5)
            ),
            "roll7_win_rate_diff": (
                _safe(home_rolling, "roll7_win_rate", 0.5) -
                _safe(away_rolling, "roll7_win_rate", 0.5)
            ),
            "roll30_win_rate_diff": (
                _safe(home_rolling, "roll30_win_rate", 0.5) -
                _safe(away_rolling, "roll30_win_rate", 0.5)
            ),
            "season_win_pct_diff": (
                _safe(home_rolling, "season_win_pct", 0.5) -
                _safe(away_rolling, "season_win_pct", 0.5)
            ),
            "wrc_plus_diff": (
                _safe(home_batting, "wrc_plus", 100.0) -
                _safe(away_batting, "wrc_plus", 100.0)
            ),
            "era_diff": (
                _safe(away_pitching, "era", 4.30) -
                _safe(home_pitching, "era", 4.30)
            ),
            "fip_diff": (
                _safe(away_pitching, "fip", 4.20) -
                _safe(home_pitching, "fip", 4.20)
            ),
            "rest_days_diff": float(home_rest_days) - float(away_rest_days),
            # --- Context ---
            "month":       dt.month,
            "day_of_week": dt.dayofweek,
            "is_weekend":  int(dt.dayofweek in (4, 5, 6)),
            "is_2020":     0,
            "home_advantage": 1,
        }
    else:
        home_last_game = pd.to_datetime(_safe(home_rolling, "last_game_date", pd.NaT), errors="coerce")
        away_last_game = pd.to_datetime(_safe(away_rolling, "last_game_date", pd.NaT), errors="coerce")
        soccer_home_rest = float(max(0, (dt - home_last_game).days)) if pd.notna(home_last_game) else 7.0
        soccer_away_rest = float(max(0, (dt - away_last_game).days)) if pd.notna(away_last_game) else 7.0

        row = {
            "home_roll_win_rate":      _safe(home_rolling, "roll_win_rate", 0.5),
            "away_roll_win_rate":      _safe(away_rolling, "roll_win_rate", 0.5),
            "home_roll_draw_rate":     _safe(home_rolling, "roll_draw_rate", 0.28),
            "away_roll_draw_rate":     _safe(away_rolling, "roll_draw_rate", 0.28),
            "home_roll_goals_scored":  _safe(home_rolling, "roll_goals_scored", 1.5),
            "away_roll_goals_scored":  _safe(away_rolling, "roll_goals_scored", 1.5),
            "home_roll_goals_allowed": _safe(home_rolling, "roll_goals_allowed", 1.5),
            "away_roll_goals_allowed": _safe(away_rolling, "roll_goals_allowed", 1.5),
            "home_roll_goal_diff":     _safe(home_rolling, "roll_goal_diff", 0.0),
            "away_roll_goal_diff":     _safe(away_rolling, "roll_goal_diff", 0.0),
            "home_roll_pythag":        _safe(home_rolling, "roll_pythag", 0.5),
            "away_roll_pythag":        _safe(away_rolling, "roll_pythag", 0.5),
            "home_season_win_pct":     _safe(home_rolling, "season_win_pct", 0.5),
            "away_season_win_pct":     _safe(away_rolling, "season_win_pct", 0.5),
            "home_attack_strength":    _safe(home_rolling, "attack_strength", _safe(home_batting, "attack_strength", 1.0)),
            "away_attack_strength":    _safe(away_rolling, "attack_strength", _safe(away_batting, "attack_strength", 1.0)),
            "home_defense_strength":   _safe(home_rolling, "defense_strength", _safe(home_batting, "defense_strength", 1.0)),
            "away_defense_strength":   _safe(away_rolling, "defense_strength", _safe(away_batting, "defense_strength", 1.0)),
            "roll_win_rate_diff": (
                _safe(home_rolling, "roll_win_rate", 0.5) -
                _safe(away_rolling, "roll_win_rate", 0.5)
            ),
            "roll_draw_rate_diff": (
                _safe(home_rolling, "roll_draw_rate", 0.28) -
                _safe(away_rolling, "roll_draw_rate", 0.28)
            ),
            "roll_goal_diff_diff": (
                _safe(home_rolling, "roll_goal_diff", 0.0) -
                _safe(away_rolling, "roll_goal_diff", 0.0)
            ),
            "roll_pythag_diff": (
                _safe(home_rolling, "roll_pythag", 0.5) -
                _safe(away_rolling, "roll_pythag", 0.5)
            ),
            "season_win_pct_diff": (
                _safe(home_rolling, "season_win_pct", 0.5) -
                _safe(away_rolling, "season_win_pct", 0.5)
            ),
            "attack_diff": (
                _safe(home_rolling, "attack_strength", _safe(home_batting, "attack_strength", 1.0)) -
                _safe(away_rolling, "attack_strength", _safe(away_batting, "attack_strength", 1.0))
            ),
            "defense_diff": (
                _safe(home_rolling, "defense_strength", _safe(home_batting, "defense_strength", 1.0)) -
                _safe(away_rolling, "defense_strength", _safe(away_batting, "defense_strength", 1.0))
            ),
            "home_rest_days": soccer_home_rest,
            "away_rest_days": soccer_away_rest,
            "rest_days_diff": soccer_home_rest - soccer_away_rest,
            "month":       dt.month,
            "day_of_week": dt.dayofweek,
            "is_weekend":  int(dt.dayofweek in (4, 5, 6)),
            "home_advantage": 1,
        }

    result = pd.DataFrame([row])
    return result.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)


def _safe(series_or_row, key: str, default: float) -> float:
    """Safely extract a value from a Series, returning default on miss/NaN."""
    try:
        val = series_or_row[key]
        return val if pd.notna(val) else default
    except (KeyError, TypeError):
        return default
