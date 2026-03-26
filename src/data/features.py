"""
Feature engineering.

Core guarantee: NO DATA LEAKAGE.
Rolling features for game N are computed from games 0..N-1 only.
This is enforced via pandas .shift(1) before .rolling() / .expanding().

Pipeline:
    1. Parse all game logs into a unified per-team-per-game DataFrame.
    2. Compute rolling and cumulative features per (team_br, season).
    3. Build home-game rows: one row per unique game.
    4. Join rolling features for both home and away teams.
    5. Join FanGraphs season stats for both teams.
    6. Add differential and context features.
    7. Return feature matrix ready for XGBoost.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    FEATURE_COLUMNS, ROLLING_WINDOW, TARGET_WIN, SPORT,
    CACHE_DIR,
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
    Input : unified game_logs with one row per team per game
            (columns: date, season, team_br, win, score_col, allow_col, game_number)
    Output: same rows + rolling feature columns.

    Rolling features are computed WITHIN each (team_br, season) group so that
    season boundaries are never crossed. shift(1) ensures the current game is
    excluded from its own feature values.
    """
    # Determine column names based on sport
    if SPORT == "mlb":
        score_col = "runs_scored"
        allow_col = "runs_allowed"
        roll_score = "roll_rs"
        roll_allow = "roll_ra"
        roll_diff = "roll_run_diff"
    elif SPORT == "soccer":
        score_col = "goals_scored"
        allow_col = "goals_allowed"
        roll_score = "roll_goals_scored"
        roll_allow = "roll_goals_allowed"
        roll_diff = "roll_goal_diff"
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    results = []

    for (team_br, season), grp in game_logs.groupby(["team_br", "season"], sort=False):
        g = grp.sort_values("date").copy()

        # Shift by 1 so each row's rolling window is games BEFORE this game
        s_win = g["win"].shift(1)
        s_score = g[score_col].shift(1)
        s_allow = g[allow_col].shift(1)

        w = ROLLING_WINDOW

        g["roll_win_rate"] = s_win.rolling(w, min_periods=1).mean()
        g[roll_score] = s_score.rolling(w, min_periods=1).mean()
        g[roll_allow] = s_allow.rolling(w, min_periods=1).mean()
        g[roll_diff] = g[roll_score] - g[roll_allow]

        # Pythagorean expectation from rolling averages
        g["roll_pythag"] = [
            pythagorean_win_exp(score, allow)
            for score, allow in zip(g[roll_score], g[roll_allow])
        ]

        # Cumulative season win % (expanding, excluding current game)
        g["season_win_pct"] = s_win.expanding(min_periods=1).mean()

        # Fill NaN in first games of each season with neutral defaults
        g["roll_win_rate"] = g["roll_win_rate"].fillna(0.5)
        g[roll_score] = g[roll_score].fillna(g[score_col].mean())
        g[roll_allow] = g[roll_allow].fillna(g[allow_col].mean())
        g[roll_diff] = g[roll_diff].fillna(0.0)
        g["roll_pythag"] = g["roll_pythag"].fillna(0.5)
        g["season_win_pct"] = g["season_win_pct"].fillna(0.5)

        results.append(g[[
            "date", "season", "team_br",
            "roll_win_rate", roll_score, roll_allow,
            roll_diff, "roll_pythag", "season_win_pct",
        ]])

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 3: Build one row per game (home perspective)
# ---------------------------------------------------------------------------

def build_home_game_rows(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Filter game_logs to home games (home_flag=True) → one row per unique game.
    Renames columns to home_* perspective and adds targets based on sport.
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
            "team_br":      "home_team_br",
            "opponent_br":  "away_team_br",
            "win":          "home_win",
            "goals_scored":  "home_goals",
            "goals_allowed": "away_goals",
        })
        home["total_goals"] = home["home_goals"] + home["away_goals"]
        target_cols = ["home_goals", "away_goals", "home_win", "total_goals"]
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    # Drop duplicates that can appear if both teams' raw data overlap
    home = home.drop_duplicates(subset=["date", "home_team_br", "away_team_br"])

    return home[[
        "date", "season", "home_team_br", "away_team_br",
    ] + target_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Join rolling features for both teams
# ---------------------------------------------------------------------------

def join_rolling_features(home_games: pd.DataFrame,
                           rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-computed rolling features for both home and away teams
    onto the home_games DataFrame.

    Merge key: (date, season, team_br) → since rolling is indexed per team,
    we merge twice — once for home, once for away.
    """
    if SPORT == "mlb":
        rolling_cols = [
            "date", "season", "team_br",
            "roll_win_rate", "roll_rs", "roll_ra",
            "roll_run_diff", "roll_pythag", "season_win_pct",
        ]
        home_rename = {
            "team_br":       "home_team_br",
            "roll_win_rate": "home_roll_win_rate",
            "roll_rs":       "home_roll_rs",
            "roll_ra":       "home_roll_ra",
            "roll_run_diff": "home_roll_run_diff",
            "roll_pythag":   "home_roll_pythag",
            "season_win_pct":"home_season_win_pct",
        }
        away_rename = {
            "team_br":       "away_team_br",
            "roll_win_rate": "away_roll_win_rate",
            "roll_rs":       "away_roll_rs",
            "roll_ra":       "away_roll_ra",
            "roll_run_diff": "away_roll_run_diff",
            "roll_pythag":   "away_roll_pythag",
            "season_win_pct":"away_season_win_pct",
        }
    elif SPORT == "soccer":
        rolling_cols = [
            "date", "season", "team_br",
            "roll_win_rate", "roll_goals_scored", "roll_goals_allowed",
            "roll_goal_diff", "roll_pythag", "season_win_pct",
        ]
        home_rename = {
            "team_br":             "home_team_br",
            "roll_win_rate":       "home_roll_win_rate",
            "roll_goals_scored":   "home_roll_goals_scored",
            "roll_goals_allowed":  "home_roll_goals_allowed",
            "roll_goal_diff":      "home_roll_goal_diff",
            "roll_pythag":         "home_roll_pythag",
            "season_win_pct":      "home_season_win_pct",
        }
        away_rename = {
            "team_br":             "away_team_br",
            "roll_win_rate":       "away_roll_win_rate",
            "roll_goals_scored":   "away_roll_goals_scored",
            "roll_goals_allowed":  "away_roll_goals_allowed",
            "roll_goal_diff":      "away_roll_goal_diff",
            "roll_pythag":         "away_roll_pythag",
            "season_win_pct":      "away_season_win_pct",
        }
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    # Home team rolling stats
    home_roll = rolling[rolling_cols].rename(columns=home_rename)

    # Away team rolling stats
    away_roll = rolling[rolling_cols].rename(columns=away_rename)

    df = home_games.merge(home_roll, on=["date", "season", "home_team_br"], how="left")
    df = df.merge(away_roll, on=["date", "season", "away_team_br"], how="left")

    return df


# ---------------------------------------------------------------------------
# Step 5: Join FanGraphs season stats
# ---------------------------------------------------------------------------

def join_season_stats(df: pd.DataFrame,
                      batting_by_season: dict[int, pd.DataFrame],
                      pitching_by_season: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    For each game row, join season-level stats for both home and away teams.
    For MLB: FanGraphs batting/pitching stats.
    For soccer: computed attack/defense strength.
    """
    if SPORT == "mlb":
        return _join_mlb_season_stats(df, batting_by_season, pitching_by_season)
    elif SPORT == "soccer":
        return _join_soccer_season_stats(df, batting_by_season)
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")


def _join_mlb_season_stats(df: pd.DataFrame,
                          batting_by_season: dict[int, pd.DataFrame],
                          pitching_by_season: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Join FanGraphs season-level batting and pitching stats for MLB.
    """
    # Build unified batting and pitching lookup DataFrames
    bat_frames = [
        df_.assign(season=s) for s, df_ in batting_by_season.items()
        if not df_.empty
    ]
    pit_frames = [
        df_.assign(season=s) for s, df_ in pitching_by_season.items()
        if not df_.empty
    ]
    if not bat_frames or not pit_frames:
        # Add placeholder columns if FanGraphs data is unavailable
        for prefix in ("home_", "away_"):
            for col in ["wrc_plus","obp","slg","k_pct","bb_pct","era","fip","xfip","k9","bb9"]:
                df[f"{prefix}{col}"] = np.nan
        return df

    all_batting  = pd.concat(bat_frames, ignore_index=True)
    all_pitching = pd.concat(pit_frames, ignore_index=True)

    # Select relevant columns (keep only what we need)
    bat_keep = _select_existing(
        all_batting, ["team_br", "season", "wrc_plus", "obp", "slg", "k_pct", "bb_pct"]
    )
    pit_keep = _select_existing(
        all_pitching, ["team_br", "season", "era", "fip", "xfip", "k9", "bb9"]
    )

    # Merge for home and away teams
    df = _merge_stats(df, bat_keep, pit_keep, side="home")
    df = _merge_stats(df, bat_keep, pit_keep, side="away")

    return df


def _join_soccer_season_stats(df: pd.DataFrame,
                             team_stats_by_season: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Join computed attack/defense strength for soccer teams.
    """
    if not team_stats_by_season:
        # Add placeholder columns
        for prefix in ("home_", "away_"):
            for col in ["attack_strength", "defense_strength"]:
                df[f"{prefix}{col}"] = 1.0  # Neutral values
        return df

    # Build unified team stats DataFrame
    stats_frames = [
        df_.assign(season=s) for s, df_ in team_stats_by_season.items()
        if not df_.empty
    ]
    if not stats_frames:
        for prefix in ("home_", "away_"):
            for col in ["attack_strength", "defense_strength"]:
                df[f"{prefix}{col}"] = 1.0
        return df

    all_stats = pd.concat(stats_frames, ignore_index=True)

    # Select relevant columns
    stats_keep = _select_existing(
        all_stats, ["team_br", "season", "attack_strength", "defense_strength"]
    )

    # Merge for home and away teams
    df = _merge_soccer_stats(df, stats_keep, side="home")
    df = _merge_soccer_stats(df, stats_keep, side="away")

    return df


def _select_existing(df: pd.DataFrame, desired_cols: list[str]) -> pd.DataFrame:
    """Return only columns that actually exist in df."""
    return df[[c for c in desired_cols if c in df.columns]].copy()


def _merge_stats(game_df: pd.DataFrame,
                 bat: pd.DataFrame,
                 pit: pd.DataFrame,
                 side: str) -> pd.DataFrame:
    """Merge batting + pitching stats for home or away team."""
    team_col = f"{side}_team_br"

    # Rename team_br and prefix the stat columns
    bat_renamed = bat.rename(columns={
        "team_br": team_col,
        **{c: f"{side}_{c}" for c in bat.columns if c not in ("team_br", "season")}
    })
    pit_renamed = pit.rename(columns={
        "team_br": team_col,
        **{c: f"{side}_{c}" for c in pit.columns if c not in ("team_br", "season")}
    })

    game_df = game_df.merge(bat_renamed, on=["season", team_col], how="left")
    game_df = game_df.merge(pit_renamed, on=["season", team_col], how="left")

    return game_df


def _merge_soccer_stats(game_df: pd.DataFrame,
                       stats: pd.DataFrame,
                       side: str) -> pd.DataFrame:
    """Merge attack/defense strength stats for home or away team."""
    team_col = f"{side}_team_br"

    # Rename team_br and prefix the stat columns
    stats_renamed = stats.rename(columns={
        "team_br": team_col,
        **{c: f"{side}_{c}" for c in stats.columns if c not in ("team_br", "season")}
    })

    game_df = game_df.merge(stats_renamed, on=["season", team_col], how="left")

    return game_df


# ---------------------------------------------------------------------------
# Step 6: Differential and context features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add differential features (home minus away) and game-context features.
    """
    df = df.copy()

    # Rolling differentials (common to both sports)
    df["roll_win_rate_diff"] = df["home_roll_win_rate"] - df["away_roll_win_rate"]
    df["roll_pythag_diff"]   = df["home_roll_pythag"]   - df["away_roll_pythag"]
    df["season_win_pct_diff"]= df["home_season_win_pct"]- df["away_season_win_pct"]

    if SPORT == "mlb":
        # MLB-specific differentials
        df["roll_run_diff_diff"] = df["home_roll_run_diff"] - df["away_roll_run_diff"]
        if "home_wrc_plus" in df.columns and "away_wrc_plus" in df.columns:
            df["wrc_plus_diff"] = df["home_wrc_plus"] - df["away_wrc_plus"]
        else:
            df["wrc_plus_diff"] = 0.0
        if "home_era" in df.columns and "away_era" in df.columns:
            df["era_diff"] = df["away_era"] - df["home_era"]
            df["fip_diff"] = df["away_fip"] - df["home_fip"] if "home_fip" in df.columns else 0.0
        else:
            df["era_diff"] = 0.0
            df["fip_diff"] = 0.0
    elif SPORT == "soccer":
        # Soccer-specific differentials
        df["roll_goal_diff_diff"] = df["home_roll_goal_diff"] - df["away_roll_goal_diff"]
        if "home_attack_strength" in df.columns and "away_attack_strength" in df.columns:
            df["attack_diff"] = df["home_attack_strength"] - df["away_attack_strength"]
            df["defense_diff"] = df["home_defense_strength"] - df["away_defense_strength"]
        else:
            df["attack_diff"] = 0.0
            df["defense_diff"] = 0.0

    # Context (common to both sports)
    df["month"]       = pd.to_datetime(df["date"]).dt.month.astype(int)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek.astype(int)
    df["is_weekend"]  = df["day_of_week"].isin([4, 5, 6]).astype(int)
    df["home_advantage"] = 1  # constant home-field advantage indicator

    return df


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def build_training_dataset(game_logs: pd.DataFrame,
                            batting_by_season: dict[int, pd.DataFrame],
                            pitching_by_season: dict[int, pd.DataFrame],
                            save_path: Path = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    For MLB: uses FanGraphs batting/pitching stats
    For soccer: uses computed attack/defense strength (batting_by_season contains team stats)
    """
    print("  [features] Computing rolling features...")
    rolling = compute_rolling_features(game_logs)

    print("  [features] Building home-game rows...")
    home_games = build_home_game_rows(game_logs)

    print("  [features] Joining rolling features...")
    df = join_rolling_features(home_games, rolling)

    print("  [features] Joining season stats...")
    df = join_season_stats(df, batting_by_season, pitching_by_season)

    print("  [features] Adding derived/differential features...")
    df = add_derived_features(df)

    # Ensure all expected feature columns exist (fill missing with NaN)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows where all rolling features are NaN (pathological cases)
    roll_cols = [c for c in FEATURE_COLUMNS if c.startswith(("home_roll", "away_roll"))]
    df = df.dropna(subset=roll_cols, how="all")

    # Replace inf / -inf with NaN then fill with column median (safety for edge cases)
    feat_df = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.fillna(feat_df.median())
    df[FEATURE_COLUMNS] = feat_df

    print(f"  [features] Dataset: {len(df):,} games, {len(FEATURE_COLUMNS)} features")

    if save_path:
        df.to_parquet(save_path, index=False)
        print(f"  [features] Saved to {save_path}")

    return df


def get_end_of_season_rolling(game_logs: pd.DataFrame,
                               season: int) -> pd.DataFrame:
    """
    Return the last rolling feature snapshot for each team at the end of
    a given season. Used as the starting point for predictions in the
    following season (e.g., 2025 end → 2026 season-start features).
    """
    season_logs = game_logs[game_logs["season"] == season].copy()
    rolling = compute_rolling_features(season_logs)

    # Take the last row per team (latest game of the season)
    return (
        rolling.sort_values("date")
               .groupby("team_br", as_index=False)
               .last()
    )


def build_prediction_row(
    home_team_br: str,
    away_team_br: str,
    home_rolling: pd.Series,  # from get_end_of_season_rolling
    away_rolling: pd.Series,
    home_batting: pd.Series,  # FanGraphs stats (MLB) or attack/defense stats (soccer)
    home_pitching: pd.Series,
    away_batting: pd.Series,
    away_pitching: pd.Series,
    game_date: str,
    home_pitcher_era: float = 4.5,
    away_pitcher_era: float = 4.5,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for one upcoming game.
    All column names match FEATURE_COLUMNS exactly.
    """
    dt = pd.to_datetime(game_date)

    if SPORT == "soccer":
        row = {
            # Rolling form (soccer column names)
            "home_roll_win_rate":     _safe(home_rolling, "roll_win_rate", 0.5),
            "away_roll_win_rate":     _safe(away_rolling, "roll_win_rate", 0.5),
            "home_roll_goals_scored": _safe(home_rolling, "roll_goals_scored", 1.5),
            "away_roll_goals_scored": _safe(away_rolling, "roll_goals_scored", 1.5),
            "home_roll_goals_allowed":_safe(home_rolling, "roll_goals_allowed", 1.5),
            "away_roll_goals_allowed":_safe(away_rolling, "roll_goals_allowed", 1.5),
            "home_roll_goal_diff":    _safe(home_rolling, "roll_goal_diff", 0.0),
            "away_roll_goal_diff":    _safe(away_rolling, "roll_goal_diff", 0.0),
            "home_roll_pythag":       _safe(home_rolling, "roll_pythag", 0.5),
            "away_roll_pythag":       _safe(away_rolling, "roll_pythag", 0.5),
            "home_season_win_pct":    _safe(home_rolling, "season_win_pct", 0.5),
            "away_season_win_pct":    _safe(away_rolling, "season_win_pct", 0.5),
            # Attack/defense strength (from _compute_soccer_team_stats)
            "home_attack_strength":   _safe(home_batting, "attack_strength", 1.0),
            "away_attack_strength":   _safe(away_batting, "attack_strength", 1.0),
            "home_defense_strength":  _safe(home_batting, "defense_strength", 1.0),
            "away_defense_strength":  _safe(away_batting, "defense_strength", 1.0),
            # Differentials
            "roll_win_rate_diff":  _safe(home_rolling, "roll_win_rate", 0.5) - _safe(away_rolling, "roll_win_rate", 0.5),
            "roll_goal_diff_diff": _safe(home_rolling, "roll_goal_diff", 0.0) - _safe(away_rolling, "roll_goal_diff", 0.0),
            "roll_pythag_diff":    _safe(home_rolling, "roll_pythag", 0.5) - _safe(away_rolling, "roll_pythag", 0.5),
            "season_win_pct_diff": _safe(home_rolling, "season_win_pct", 0.5) - _safe(away_rolling, "season_win_pct", 0.5),
            "attack_diff":   _safe(home_batting, "attack_strength", 1.0) - _safe(away_batting, "attack_strength", 1.0),
            "defense_diff":  _safe(home_batting, "defense_strength", 1.0) - _safe(away_batting, "defense_strength", 1.0),
            # Context
            "month":       dt.month,
            "day_of_week": dt.dayofweek,
            "is_weekend":  int(dt.dayofweek in (4, 5, 6)),
            "home_advantage": 1,
        }
    else:
        row = {
            # Rolling form (MLB column names)
            "home_roll_win_rate":   _safe(home_rolling, "roll_win_rate", 0.5),
            "away_roll_win_rate":   _safe(away_rolling, "roll_win_rate", 0.5),
            "home_roll_rs":         _safe(home_rolling, "roll_rs", 4.5),
            "away_roll_rs":         _safe(away_rolling, "roll_rs", 4.5),
            "home_roll_ra":         _safe(home_rolling, "roll_ra", 4.5),
            "away_roll_ra":         _safe(away_rolling, "roll_ra", 4.5),
            "home_roll_run_diff":   _safe(home_rolling, "roll_run_diff", 0.0),
            "away_roll_run_diff":   _safe(away_rolling, "roll_run_diff", 0.0),
            "home_roll_pythag":     _safe(home_rolling, "roll_pythag", 0.5),
            "away_roll_pythag":     _safe(away_rolling, "roll_pythag", 0.5),
            "home_season_win_pct":  _safe(home_rolling, "season_win_pct", 0.5),
            "away_season_win_pct":  _safe(away_rolling, "season_win_pct", 0.5),
            # FanGraphs batting
            "home_wrc_plus":  _safe(home_batting, "wrc_plus", 100.0),
            "away_wrc_plus":  _safe(away_batting, "wrc_plus", 100.0),
            "home_obp":       _safe(home_batting, "obp", 0.320),
            "away_obp":       _safe(away_batting, "obp", 0.320),
            "home_slg":       _safe(home_batting, "slg", 0.410),
            "away_slg":       _safe(away_batting, "slg", 0.410),
            "home_k_pct":     _safe(home_batting, "k_pct", 0.22),
            "away_k_pct":     _safe(away_batting, "k_pct", 0.22),
            "home_bb_pct":    _safe(home_batting, "bb_pct", 0.085),
            "away_bb_pct":    _safe(away_batting, "bb_pct", 0.085),
            # FanGraphs pitching
            "home_era":   _safe(home_pitching, "era", 4.30),
            "away_era":   _safe(away_pitching, "era", 4.30),
            "home_pitcher_era": float(home_pitcher_era),
            "away_pitcher_era": float(away_pitcher_era),
            "home_fip":   _safe(home_pitching, "fip", 4.20),
            "away_fip":   _safe(away_pitching, "fip", 4.20),
            "home_xfip":  _safe(home_pitching, "xfip", 4.20),
            "away_xfip":  _safe(away_pitching, "xfip", 4.20),
            "home_k9":    _safe(home_pitching, "k9", 8.5),
            "away_k9":    _safe(away_pitching, "k9", 8.5),
            "home_bb9":   _safe(home_pitching, "bb9", 3.2),
            "away_bb9":   _safe(away_pitching, "bb9", 3.2),
            # Differentials
            "roll_win_rate_diff":  _safe(home_rolling, "roll_win_rate", 0.5) - _safe(away_rolling, "roll_win_rate", 0.5),
            "roll_run_diff_diff":  _safe(home_rolling, "roll_run_diff", 0.0) - _safe(away_rolling, "roll_run_diff", 0.0),
            "roll_pythag_diff":    _safe(home_rolling, "roll_pythag", 0.5)   - _safe(away_rolling, "roll_pythag", 0.5),
            "season_win_pct_diff": _safe(home_rolling, "season_win_pct", 0.5)- _safe(away_rolling, "season_win_pct", 0.5),
            "wrc_plus_diff":  _safe(home_batting, "wrc_plus", 100.0) - _safe(away_batting, "wrc_plus", 100.0),
            "era_diff":       _safe(away_pitching, "era", 4.30) - _safe(home_pitching, "era", 4.30),
            "fip_diff":       _safe(away_pitching, "fip", 4.20) - _safe(home_pitching, "fip", 4.20),
            # Context
            "month":       dt.month,
            "day_of_week": dt.dayofweek,
            "is_weekend":  int(dt.dayofweek in (4, 5, 6)),
            "is_2020":     0,
            "home_advantage": 1,
        }

    df = pd.DataFrame([row])
    # Guarantee column order matches FEATURE_COLUMNS exactly
    return df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)


def _safe(series_or_row, key: str, default: float) -> float:
    """Safely extract a value from a Series, returning default on miss/NaN."""
    try:
        val = series_or_row[key]
        return float(val) if pd.notna(val) else default
    except (KeyError, TypeError):
        return default
