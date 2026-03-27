"""Central configuration for the MLB AI prediction system."""
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Sport configuration
# ---------------------------------------------------------------------------
SPORT = "mlb"
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
ROLLING_WINDOW = 15                        # games used for main rolling form features

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
# Park factors (FanGraphs 5-year park run factor, normalized to 100 = neutral)
# ---------------------------------------------------------------------------
PARK_FACTORS: dict[str, int] = {
    "ARI": 105,  # Chase Field  (roof/heat)
    "ATL": 100,  # Truist Park
    "BAL": 104,  # Camden Yards
    "BOS": 105,  # Fenway Park (Green Monster)
    "CHC": 103,  # Wrigley Field
    "CHW": 97,   # Guaranteed Rate Field
    "CIN": 103,  # Great American Ball Park
    "CLE": 98,   # Progressive Field
    "COL": 115,  # Coors Field (altitude — biggest outlier in MLB)
    "DET": 100,  # Comerica Park
    "HOU": 97,   # Minute Maid Park (dome)
    "KCR": 97,   # Kauffman Stadium
    "LAA": 97,   # Angel Stadium
    "LAD": 98,   # Dodger Stadium
    "MIA": 95,   # loanDepot park (dome, suppresses)
    "MIL": 98,   # American Family Field (dome)
    "MIN": 99,   # Target Field
    "NYM": 100,  # Citi Field
    "NYY": 104,  # Yankee Stadium (short right-field porch)
    "OAK": 96,   # Oakland Coliseum (large foul territory)
    "PHI": 100,  # Citizens Bank Park
    "PIT": 99,   # PNC Park
    "SDP": 93,   # Petco Park (biggest suppressor)
    "SFG": 94,   # Oracle Park (marine layer)
    "SEA": 97,   # T-Mobile Park
    "STL": 99,   # Busch Stadium
    "TBR": 98,   # Tropicana Field (dome)
    "TEX": 107,  # Globe Life Field (hot, hitter-friendly)
    "TOR": 100,  # Rogers Centre (dome)
    "WSN": 100,  # Nationals Park
}

# ---------------------------------------------------------------------------
# Stadium coordinates (latitude, longitude) for weather lookup
# ---------------------------------------------------------------------------
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.445, -112.066),
    "ATL": (33.890, -84.468),
    "BAL": (39.284, -76.622),
    "BOS": (42.347, -71.097),
    "CHC": (41.948, -87.656),
    "CHW": (41.830, -87.634),
    "CIN": (39.098, -84.507),
    "CLE": (41.496, -81.685),
    "COL": (39.756, -104.994),
    "DET": (42.339, -83.049),
    "HOU": (29.757, -95.355),
    "KCR": (39.051, -94.480),
    "LAA": (33.800, -117.883),
    "LAD": (34.074, -118.240),
    "MIA": (25.778, -80.220),
    "MIL": (43.028, -87.971),
    "MIN": (44.982, -93.278),
    "NYM": (40.757, -73.846),
    "NYY": (40.829, -73.926),
    "OAK": (37.751, -122.200),
    "PHI": (39.906, -75.166),
    "PIT": (40.447, -80.006),
    "SDP": (32.708, -117.157),
    "SFG": (37.779, -122.389),
    "SEA": (47.591, -122.332),
    "STL": (38.623, -90.193),
    "TBR": (27.768, -82.653),
    "TEX": (32.748, -97.083),
    "TOR": (43.641, -79.389),
    "WSN": (38.873, -77.008),
}

# Teams that play in a dome (weather irrelevant; use controlled-environment defaults)
DOME_STADIUMS: set[str] = {"ARI", "HOU", "MIA", "MIL", "TBR", "TOR"}

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
# Feature column names (must match exactly between training and prediction)
# ---------------------------------------------------------------------------
MLB_FEATURE_COLUMNS: list[str] = [
    # ---- Rolling form — 15-game (main window) ----
    "home_roll_win_rate",
    "away_roll_win_rate",
    "home_roll_rs",
    "away_roll_rs",
    "home_roll_ra",
    "away_roll_ra",
    "home_roll_run_diff",
    "away_roll_run_diff",
    "home_roll_pythag",
    "away_roll_pythag",
    # ---- Rolling form — 7-game (hot streak signal) ----
    "home_roll7_win_rate",
    "away_roll7_win_rate",
    "home_roll7_rs",
    "away_roll7_rs",
    "home_roll7_ra",
    "away_roll7_ra",
    # ---- Rolling form — 30-game (true form / sustainability) ----
    "home_roll30_win_rate",
    "away_roll30_win_rate",
    "home_roll30_rs",
    "away_roll30_rs",
    "home_roll30_ra",
    "away_roll30_ra",
    # ---- Cumulative season win % ----
    "home_season_win_pct",
    "away_season_win_pct",
    # ---- Rest / fatigue ----
    "home_rest_days",
    "away_rest_days",
    # ---- Head-to-head ----
    "h2h_home_win_rate",
    # ---- FanGraphs batting ----
    "home_wrc_plus",
    "away_wrc_plus",
    "home_obp",
    "away_obp",
    "home_slg",
    "away_slg",
    "home_k_pct",
    "away_k_pct",
    "home_bb_pct",
    "away_bb_pct",
    # ---- FanGraphs rotation pitching ----
    "home_era",
    "away_era",
    "home_fip",
    "away_fip",
    "home_xfip",
    "away_xfip",
    "home_k9",
    "away_k9",
    "home_bb9",
    "away_bb9",
    # ---- Bullpen ERA (relievers only, FanGraphs individual stats) ----
    "home_bullpen_era",
    "away_bullpen_era",
    # ---- Starting pitcher (game-day specific) ----
    "home_pitcher_era",
    "away_pitcher_era",
    # ---- Park factor ----
    "home_park_factor",
    # ---- Weather ----
    "temp_f",
    "wind_mph",
    "is_dome",
    # ---- Differentials (home minus away; away minus home for ERA) ----
    "roll_win_rate_diff",
    "roll_run_diff_diff",
    "roll_pythag_diff",
    "roll7_win_rate_diff",
    "roll30_win_rate_diff",
    "season_win_pct_diff",
    "wrc_plus_diff",
    "era_diff",
    "fip_diff",
    "rest_days_diff",
    # ---- Game context ----
    "month",
    "day_of_week",
    "is_weekend",
    "is_2020",
    "home_advantage",
]

SOCCER_FEATURE_COLUMNS: list[str] = [
    "home_roll_win_rate",
    "away_roll_win_rate",
    "home_roll_draw_rate",
    "away_roll_draw_rate",
    "home_roll_goals_scored",
    "away_roll_goals_scored",
    "home_roll_goals_allowed",
    "away_roll_goals_allowed",
    "home_roll_goal_diff",
    "away_roll_goal_diff",
    "home_roll_pythag",
    "away_roll_pythag",
    "home_season_win_pct",
    "away_season_win_pct",
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "roll_win_rate_diff",
    "roll_draw_rate_diff",
    "roll_goal_diff_diff",
    "roll_pythag_diff",
    "season_win_pct_diff",
    "attack_diff",
    "defense_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "month",
    "day_of_week",
    "is_weekend",
    "home_advantage",
]

FEATURE_COLUMNS: list[str] = MLB_FEATURE_COLUMNS if SPORT == "mlb" else SOCCER_FEATURE_COLUMNS

# Targets
TARGET_WIN = "home_win"
TARGET_RESULT = "home_win" if SPORT == "mlb" else "match_outcome"
TARGET_RUNS = "total_runs"
TARGET_HOME_RUNS = "home_runs"
TARGET_AWAY_RUNS = "away_runs"
TARGET_HOME_GOALS = "home_goals"
TARGET_AWAY_GOALS = "away_goals"
TARGET_TOTAL_GOALS = "total_goals"

# Feature selection — top N features used for regressor training
TOP_FEATURES = 30


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
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    name = re.sub(r"_+", "_", name).strip("_")
    return name
