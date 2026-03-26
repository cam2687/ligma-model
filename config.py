"""Central configuration for the MLB AI prediction system."""
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Sport configuration
# ---------------------------------------------------------------------------
SPORT = "soccer"  # Change to "mlb" for baseball, "soccer" for football
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models_saved"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
TRAIN_SEASONS = list(range(2019, 2026))   # 2019-2025
PREDICT_SEASON = 2026
ROLLING_WINDOW = 15                        # games used for rolling form features

# ---------------------------------------------------------------------------
# Team abbreviations
# Canonical internal key = Baseball Reference (BR) abbreviations.
# FanGraphs (FG) uses different abbreviations for 6 teams.
# ---------------------------------------------------------------------------
FG_TO_BR: dict[str, str] = {
    "CWS": "CHW",
    "KC":  "KCR",
    "SD":  "SDP",
    "SF":  "SFG",
    "TB":  "TBR",
    "WSH": "WSN",
}
BR_TO_FG: dict[str, str] = {v: k for k, v in FG_TO_BR.items()}

# All 30 teams using BR abbreviations (canonical)
MLB_TEAMS_BR: list[str] = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SFG", "SEA", "STL", "TBR", "TEX", "TOR", "WSN",
]

# statsapi returns full city names → BR abbreviations
STATSAPI_NAME_TO_BR: dict[str, str] = {
    "Arizona Diamondbacks":  "ARI",
    "Atlanta Braves":        "ATL",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHC",
    "Chicago White Sox":     "CHW",
    "Cincinnati Reds":       "CIN",
    "Cleveland Guardians":   "CLE",
    "Cleveland Indians":     "CLE",  # legacy name
    "Colorado Rockies":      "COL",
    "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU",
    "Kansas City Royals":    "KCR",
    "Los Angeles Angels":    "LAA",
    "Los Angeles Dodgers":   "LAD",
    "Miami Marlins":         "MIA",
    "Milwaukee Brewers":     "MIL",
    "Minnesota Twins":       "MIN",
    "New York Mets":         "NYM",
    "New York Yankees":      "NYY",
    "Oakland Athletics":     "OAK",
    "Athletics":             "OAK",  # franchise moved 2025+
    "Sacramento Athletics":  "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SDP",
    "San Francisco Giants":  "SFG",
    "Seattle Mariners":      "SEA",
    "St. Louis Cardinals":   "STL",
    "Tampa Bay Rays":        "TBR",
    "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR",
    "Washington Nationals":  "WSN",
}

BR_TO_FULL_NAME: dict[str, str] = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
XGB_CLASSIFIER_PARAMS: dict = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}

XGB_REGRESSOR_PARAMS: dict = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}

# ---------------------------------------------------------------------------
# Feature column names (must match between training and prediction)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    # Rolling form features (last ROLLING_WINDOW games, prior to current game)
    "home_roll_win_rate",
    "away_roll_win_rate",
    "home_roll_goals_scored",
    "away_roll_goals_scored",
    "home_roll_goals_allowed",
    "away_roll_goals_allowed",
    "home_roll_goal_diff",
    "away_roll_goal_diff",
    "home_roll_pythag",
    "away_roll_pythag",
    # Cumulative season win percentage
    "home_season_win_pct",
    "away_season_win_pct",
    # Team stats (season-level attack/defense)
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    # Differentials
    "roll_win_rate_diff",
    "roll_goal_diff_diff",
    "roll_pythag_diff",
    "season_win_pct_diff",
    "attack_diff",
    "defense_diff",
    # Game context
    "month",
    "day_of_week",
    "is_weekend",
    "home_advantage",
]

# Targets (conditional based on sport)
if SPORT == "mlb":
    TARGET_WIN = "home_win"
    TARGET_RUNS = "total_runs"
    TARGET_HOME_RUNS = "home_runs"
    TARGET_AWAY_RUNS = "away_runs"
    TARGET_HOME_GOALS = "home_goals"  # Keep for compatibility
    TARGET_AWAY_GOALS = "away_goals"  # Keep for compatibility
    TARGET_TOTAL_GOALS = "total_goals"  # Keep for compatibility
elif SPORT == "soccer":
    TARGET_WIN = "home_win"
    TARGET_HOME_RUNS = "home_runs"  # Keep for compatibility
    TARGET_AWAY_RUNS = "away_runs"  # Keep for compatibility
    TARGET_RUNS = "total_runs"  # Keep for compatibility
    TARGET_HOME_GOALS = "home_goals"
    TARGET_AWAY_GOALS = "away_goals"
    TARGET_TOTAL_GOALS = "total_goals"
else:
    raise ValueError(f"Unsupported sport: {SPORT}")

# Feature selection
TOP_FEATURES = 24


def normalize_col(name: str) -> str:
    """Convert a raw column name from FanGraphs to a safe Python identifier."""
    name = (
        name
        .replace("wRC+", "wrc_plus")
        .replace("K/9",  "k9")
        .replace("BB/9", "bb9")
        .replace("HR/9", "hr9")
        .replace("H/9",  "h9")
        .replace("HR/FB","hr_fb")
        .replace("xFIP", "xfip")
        .replace("FIP",  "fip")
        .replace("ERA",  "era")
        .replace("WHIP", "whip")
        .replace("OBP",  "obp")
        .replace("SLG",  "slg")
        .replace("ISO",  "iso")
        .replace("wOBA", "woba")
        .replace("xwOBA","xwoba")
        .replace("BABIP","babip")
        .replace("BB%",  "bb_pct")
        .replace("K%",   "k_pct")
        .replace("LOB%", "lob_pct")
        .replace("GB%",  "gb_pct")
        .replace("WAR",  "war")
    )
    # Replace remaining special chars with underscore, collapse, lowercase
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    name = re.sub(r"_+", "_", name).strip("_")
    return name
