"""
Data fetching layer.
All API calls are cached to disk so we never re-hit sources.

MLB:
- Game logs  : MLB Stats API (statsapi) - official, no scraping blocks
- Team stats : FanGraphs via pybaseball.team_batting/pitching - works fine
- Today's games: MLB Stats API (statsapi)

Soccer:
- Game logs  : football-data.co.uk CSV files
- Team stats : Computed from game logs (no external stats API)
- Today's games: Not implemented (predictions only for historical)
"""
import sys
import time
from pathlib import Path
from datetime import date, timedelta, datetime
from typing import Optional
import requests

import pandas as pd
import statsapi

sys.path.insert(0, str(Path(__file__).parents[2]))

if __import__('config').SPORT == "mlb":
    import pybaseball
    pybaseball.cache.enable()

from config import (
    CACHE_DIR, TRAIN_SEASONS, SPORT,
    MLB_TEAMS_BR, FG_TO_BR, STATSAPI_NAME_TO_BR, normalize_col,
)

# ---------------------------------------------------------------------------
# Generic cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def _load(key: str) -> Optional[pd.DataFrame]:
    p = _cache_path(key)
    if p.exists():
        return pd.read_pickle(p)
    return None


def _save(key: str, df: pd.DataFrame) -> None:
    df.to_pickle(_cache_path(key))


# ---------------------------------------------------------------------------
# Game logs via MLB Stats API
# statsapi.schedule() is the official MLB API -- no scraping, no blocking
# ---------------------------------------------------------------------------

# Season date ranges (start/end of regular season, MM/DD/YYYY format)
_SEASON_DATES = {
    2019: ("03/20/2019", "10/01/2019"),
    2020: ("07/23/2020", "09/27/2020"),  # COVID-shortened 60-game season
    2021: ("04/01/2021", "10/03/2021"),
    2022: ("04/07/2022", "10/05/2022"),
    2023: ("03/30/2023", "10/01/2023"),
    2024: ("03/20/2024", "09/29/2024"),
    2025: ("03/27/2025", "09/28/2025"),
}

# Soccer leagues and their football-data.co.uk URLs
_SOCCER_LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League 1",
    "E3": "League 2",
    "SC0": "Scottish Premiership",
    "D1": "Bundesliga",
    "D2": "Bundesliga 2",
    "I1": "Serie A",
    "I2": "Serie B",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Jupiler League",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
}


def fetch_season_game_logs(season: int, force: bool = False) -> pd.DataFrame:
    """
    Fetch all completed regular-season games for a season.
    For MLB: via statsapi. For soccer: via football-data.co.uk.
    Returns one row per game with sport-specific columns.
    """
    if SPORT == "mlb":
        return _fetch_mlb_season_game_logs(season, force)
    elif SPORT == "soccer":
        return _fetch_soccer_season_game_logs(season, force)
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")


def _fetch_mlb_season_game_logs(season: int, force: bool = False) -> pd.DataFrame:
    """
    Fetch all completed regular-season games for a season via statsapi.
    Returns one row per game with: date, home_team_br, away_team_br,
    home_runs, away_runs, home_win, total_runs.
    """
    key = f"season_games_{season}"
    if not force:
        cached = _load(key)
        if cached is not None:
            return cached

    start, end = _SEASON_DATES.get(season, (f"04/01/{season}", f"10/01/{season}"))
    print(f"  [fetch] Season {season}: {start} -> {end}")

    # Chunk by ~30 days to avoid any response size limits
    all_raw = []
    chunk_start = datetime.strptime(start, "%m/%d/%Y")
    chunk_end_dt = datetime.strptime(end, "%m/%d/%Y")

    while chunk_start <= chunk_end_dt:
        c_end = min(chunk_start + timedelta(days=30), chunk_end_dt)
        try:
            raw = statsapi.schedule(
                start_date=chunk_start.strftime("%m/%d/%Y"),
                end_date=c_end.strftime("%m/%d/%Y"),
                sportId=1,
            )
            if raw:
                all_raw.extend(raw)
        except Exception as exc:
            print(f"    [fetch] WARN {chunk_start.strftime('%m/%d')}: {exc}")
        chunk_start = c_end + timedelta(days=1)
        time.sleep(0.3)

    if not all_raw:
        print(f"  [fetch] WARN: No data for season {season}")
        return pd.DataFrame()

    # Filter to completed regular-season games only
    completed = [
        g for g in all_raw
        if g.get("game_type") == "R"
        and g.get("status", "").lower() == "final"
        and g.get("away_score") is not None
        and g.get("home_score") is not None
    ]

    if not completed:
        return pd.DataFrame()

    rows = []
    for g in completed:
        home_name = g.get("home_name", "")
        away_name = g.get("away_name", "")
        home_br = STATSAPI_NAME_TO_BR.get(home_name, home_name[:3].upper())
        away_br = STATSAPI_NAME_TO_BR.get(away_name, away_name[:3].upper())
        home_runs = int(g.get("home_score", 0))
        away_runs = int(g.get("away_score", 0))

        # Parse date (statsapi returns 'YYYY-MM-DD' in game_date)
        game_date_str = g.get("game_date") or g.get("game_datetime", "")[:10]
        try:
            gdate = pd.to_datetime(game_date_str)
        except Exception:
            continue

        rows.append({
            "date":         gdate,
            "season":       season,
            "home_team_br": home_br,
            "away_team_br": away_br,
            "home_runs":    home_runs,
            "away_runs":    away_runs,
            "home_win":     int(home_runs > away_runs),
            "total_runs":   home_runs + away_runs,
            "game_id":      g.get("game_id"),
        })

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["date", "home_team_br", "away_team_br"]
    ).sort_values("date").reset_index(drop=True)

    print(f"  [fetch] Season {season}: {len(df)} games")
    _save(key, df)
    return df


def _fetch_soccer_season_game_logs(season: int, force: bool = False) -> pd.DataFrame:
    """
    Fetch soccer games from football-data.co.uk for major European leagues.
    Returns one row per game with: date, home_team, away_team,
    home_goals, away_goals, home_win, total_goals.
    """
    key = f"soccer_season_games_{season}"
    if not force:
        cached = _load(key)
        if cached is not None:
            return cached

    all_games = []
    season_str = str(season)[-2:]        # e.g., 2019 -> "19"
    season_str_next = str(season + 1)[-2:]  # e.g., 2019 -> "20"

    for league_code, league_name in _SOCCER_LEAGUES.items():
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{season_str}{season_str_next}/{league_code}.csv"
            print(f"  [fetch] Downloading {league_name} ({season}/{season+1})...")

            df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
            if df.empty:
                continue

            # Standardize column names (football-data.co.uk uses different formats)
            col_map = {
                'Date': 'date',
                'HomeTeam': 'home_team',
                'AwayTeam': 'away_team',
                'FTHG': 'home_goals',  # Full Time Home Goals
                'FTAG': 'away_goals',  # Full Time Away Goals
                'FTR': 'result',       # Full Time Result (H=Home win, D=Draw, A=Away win)
            }

            # Check which columns exist
            available_cols = {k: v for k, v in col_map.items() if k in df.columns}
            if not all(k in df.columns for k in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']):
                print(f"    [fetch] Missing required columns in {league_code}, skipping")
                continue

            games = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
            games.columns = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']

            # Parse date
            games['date'] = pd.to_datetime(games['date'], dayfirst=True, errors='coerce')
            games = games.dropna(subset=['date'])

            # Convert to standard format
            games['season'] = season
            games['home_win'] = (games['result'] == 'H').astype(int)
            games['total_goals'] = games['home_goals'] + games['away_goals']
            games['league'] = league_code

            # Use team names as-is (no BR mapping for soccer)
            games['home_team_br'] = games['home_team']
            games['away_team_br'] = games['away_team']

            all_games.append(games)
            print(f"    [fetch] {league_name}: {len(games)} games")

        except Exception as exc:
            print(f"    [fetch] WARN {league_code}: {exc}")
            continue

    if not all_games:
        print(f"  [fetch] WARN: No soccer data for season {season}")
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"]
    ).sort_values("date").reset_index(drop=True)

    print(f"  [fetch] Soccer season {season}: {len(combined)} total games")
    _save(key, combined)
    return combined


def fetch_all_game_logs(seasons: list[int] = TRAIN_SEASONS,
                        force: bool = False) -> pd.DataFrame:
    """
    Fetch game logs for all seasons.
    For soccer: combines club league data + international match history.
    Returns a unified per-team-per-game DataFrame for rolling feature computation.
    """
    key = f"{SPORT}_all_game_logs_{'_'.join(map(str, seasons))}"
    if not force:
        cached = _load(key)
        if cached is not None:
            return cached

    season_dfs = []
    for season in seasons:
        df = fetch_season_game_logs(season, force=force)
        if not df.empty:
            season_dfs.append(df)

    # For soccer: also pull international match history (gives national teams rolling stats)
    if SPORT == "soccer":
        intl_df = _fetch_international_game_logs(seasons, force=force)
        if not intl_df.empty:
            season_dfs.append(intl_df)

    if not season_dfs:
        raise RuntimeError("No game data retrieved. Check network/statsapi.")

    games = pd.concat(season_dfs, ignore_index=True)

    # Build per-team-per-game rows (home AND away perspective) for rolling features
    team_rows = _games_to_team_rows(games)

    _save(key, team_rows)
    print(f"  [fetch] Total: {len(games)} games, {len(team_rows)} team-game rows")
    return team_rows


def _fetch_international_game_logs(seasons: list[int], force: bool = False) -> pd.DataFrame:
    """
    Fetch international football match history from martj42/international_results (GitHub).
    Covers all FIFA-affiliated international matches 1872–present.
    Uses calendar year to assign season (games in year 2023 → season 2023).
    """
    key = f"international_games_{'_'.join(map(str, seasons))}"
    if not force:
        cached = _load(key)
        if cached is not None:
            return cached

    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        print(f"  [fetch] WARN: international data unavailable: {exc}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])

    # Filter to calendar years that overlap with training seasons
    df = df[df["date"].dt.year.isin(seasons)].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    df["home_goals"]  = df["home_goals"].astype(int)
    df["away_goals"]  = df["away_goals"].astype(int)
    df["home_win"]    = (df["home_goals"] > df["away_goals"]).astype(int)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["home_team_br"] = df["home_team"]
    df["away_team_br"] = df["away_team"]
    df["league"]       = "international"
    df["season"]       = df["date"].dt.year  # calendar-year season for international

    df = df[["date", "season", "home_team_br", "away_team_br",
             "home_goals", "away_goals", "home_win", "total_goals", "league"]].copy()
    df = df.drop_duplicates(subset=["date", "home_team_br", "away_team_br"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  [fetch] International: {len(df)} matches ({min(seasons)}–{max(seasons)})")
    _save(key, df)
    return df


def _games_to_team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-game DataFrame into per-team-per-game rows.
    Each game produces 2 rows (one for each team), with sport-specific columns.
    This format is required by features.py for rolling computation.
    """
    if SPORT == "mlb":
        score_col = "runs_scored"
        allow_col = "runs_allowed"
    elif SPORT == "soccer":
        score_col = "goals_scored"
        allow_col = "goals_allowed"
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")

    home_rows = games.rename(columns={
        "home_team_br": "team_br",
        "away_team_br": "opponent_br",
        f"home_{score_col.split('_')[0]}": score_col,  # home_runs -> runs_scored or home_goals -> goals_scored
        f"away_{allow_col.split('_')[0]}": allow_col,  # away_runs -> runs_allowed or away_goals -> goals_allowed
        "home_win":     "win",
    }).assign(home_flag=True)

    away_rows = games.rename(columns={
        "away_team_br": "team_br",
        "home_team_br": "opponent_br",
        f"away_{score_col.split('_')[0]}": score_col,
        f"home_{allow_col.split('_')[0]}": allow_col,
    }).assign(
        home_flag=False,
        win=lambda df: (df[score_col] > df[allow_col]).astype(int),
    )

    combined = pd.concat([home_rows, away_rows], ignore_index=True)
    combined = combined.assign(
        game_number=combined.groupby(["team_br", "season"]).cumcount(),
        is_doubleheader=False,
    )

    return combined[[
        "date", "season", "team_br", "opponent_br", "home_flag",
        "win", score_col, allow_col, "game_number", "is_doubleheader",
    ]].sort_values(["team_br", "season", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team stats (MLB: FanGraphs, Soccer: computed from games)
# ---------------------------------------------------------------------------

def fetch_team_batting_stats(season: int, force: bool = False) -> pd.DataFrame:
    """Fetch team batting stats (MLB) or computed attack/defense (Soccer)."""
    if SPORT == "mlb":
        return fetch_fangraphs_batting(season, force)
    elif SPORT == "soccer":
        return _compute_soccer_team_stats(season, force)
    else:
        raise ValueError(f"Unsupported sport: {SPORT}")


def fetch_team_pitching_stats(season: int, force: bool = False) -> pd.DataFrame:
    """Fetch team pitching stats (MLB only). For soccer, returns empty."""
    if SPORT == "mlb":
        return fetch_fangraphs_pitching(season, force)
    else:
        # Soccer doesn't have pitching stats
        return pd.DataFrame()


def fetch_fangraphs_batting(season: int, force: bool = False) -> pd.DataFrame:
    key = f"fg_batting_{season}"
    if not force:
        cached = _load(key)
        if cached is not None:
            return cached

    try:
        df = pybaseball.team_batting(season, season, ind=1)
    except Exception as exc:
        print(f"  [fetch] WARN: FG batting {season}: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Explicitly select only the columns we need (avoids duplicate-name issues
    # that arise when normalizing 300+ FanGraphs columns)
    col_map = {
        "Team":  "team",
        "wRC+":  "wrc_plus",
        "OBP":   "obp",
        "SLG":   "slg",
        "K%":    "k_pct",
        "BB%":   "bb_pct",
        "ISO":   "iso",
        "wOBA":  "woba",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    if "Team" not in available:
        return pd.DataFrame()

    result = df[list(available.keys())].copy()
    result.columns = [available[c] for c in result.columns]
    result["team_br"] = result["team"].map(lambda t: FG_TO_BR.get(str(t).upper(), str(t).upper()))
    result["season"] = season
    result = result.drop(columns=["team"])

    _save(key, result)
    return result


def fetch_fangraphs_pitching(season: int, force: bool = False) -> pd.DataFrame:
    key = f"fg_pitching_{season}"
    if not force:
        cached = _load(key)
        if cached is not None:
            return cached

    try:
        df = pybaseball.team_pitching(season, season, ind=1)
    except Exception as exc:
        print(f"  [fetch] WARN: FG pitching {season}: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {
        "Team":  "team",
        "ERA":   "era",
        "FIP":   "fip",
        "xFIP":  "xfip",
        "K/9":   "k9",
        "BB/9":  "bb9",
        "WHIP":  "whip",
        "SIERA": "siera",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    if "Team" not in available:
        return pd.DataFrame()

    result = df[list(available.keys())].copy()
    result.columns = [available[c] for c in result.columns]
    result["team_br"] = result["team"].map(lambda t: FG_TO_BR.get(str(t).upper(), str(t).upper()))
    result["season"] = season
    result = result.drop(columns=["team"])

    _save(key, result)
    return result


def fetch_all_fangraphs(seasons: list[int] = TRAIN_SEASONS,
                        force: bool = False) -> tuple[dict, dict]:
    batting, pitching = {}, {}
    for season in seasons:
        bat = fetch_fangraphs_batting(season, force=force)
        pit = fetch_fangraphs_pitching(season, force=force)
        if not bat.empty:
            batting[season] = bat
        if not pit.empty:
            pitching[season] = pit
    return batting, pitching


def _compute_soccer_team_stats(season: int, force: bool = False) -> pd.DataFrame:
    """
    Compute attack and defense strength for all soccer teams (club + international).
    Attack strength  = goals scored per game relative to group average.
    Defense strength = goals allowed per game relative to group average (inverted).
    Club teams are grouped by league; international teams form their own group.
    """
    key = f"soccer_team_stats_{season}"
    if not force:
        cached = _load(key)
        if cached is not None:
            return cached

    # Club league games
    club_games = _fetch_soccer_season_game_logs(season, force)

    # International games (calendar year == season)
    all_intl = _load(f"international_games_{'_'.join(map(str, TRAIN_SEASONS))}")
    if all_intl is not None and not all_intl.empty:
        intl_games = all_intl[all_intl["season"] == season].copy()
    else:
        # Fetch just for this year
        intl_year = _fetch_international_game_logs_year(season, force)
        intl_games = intl_year

    all_games = []
    if not club_games.empty:
        all_games.append(club_games)
    if not intl_games.empty:
        all_games.append(intl_games)

    if not all_games:
        return pd.DataFrame()

    games = pd.concat(all_games, ignore_index=True)

    league_stats = []
    for league in games["league"].unique():
        league_games = games[games["league"] == league].copy()
        stats = _compute_league_team_stats(league_games, season, league)
        if stats is not None:
            league_stats.append(stats)

    if not league_stats:
        return pd.DataFrame()

    result = pd.concat(league_stats, ignore_index=True)
    _save(key, result)
    return result


def _fetch_international_game_logs_year(year: int, force: bool = False) -> pd.DataFrame:
    """Fetch international games for a single calendar year (used in team stats computation)."""
    df = _fetch_international_game_logs([year], force=force)
    return df


def _compute_league_team_stats(league_games: pd.DataFrame,
                                season: int, league: str):
    """Compute attack/defense strength for all teams in a single league/group."""
    home_stats = league_games.groupby("home_team_br").agg({
        "home_goals": "sum",
        "away_goals": "sum",
    }).rename(columns={"home_goals": "goals_scored_home", "away_goals": "goals_allowed_home"})

    away_stats = league_games.groupby("away_team_br").agg({
        "away_goals": "sum",
        "home_goals": "sum",
    }).rename(columns={"away_goals": "goals_scored_away", "home_goals": "goals_allowed_away"})

    team_stats = pd.concat([home_stats, away_stats], axis=0, keys=["home", "away"])
    team_stats = team_stats.groupby(level=1).sum()

    home_cnt = league_games.groupby("home_team_br").size()
    away_cnt = league_games.groupby("away_team_br").size()
    team_stats["games_played"] = home_cnt.add(away_cnt, fill_value=0)
    team_stats = team_stats[team_stats["games_played"] >= 2]  # skip teams with too few games

    if team_stats.empty:
        return None

    team_stats["avg_goals_scored"]  = (team_stats["goals_scored_home"]  + team_stats["goals_scored_away"])  / team_stats["games_played"]
    team_stats["avg_goals_allowed"] = (team_stats["goals_allowed_home"] + team_stats["goals_allowed_away"]) / team_stats["games_played"]

    league_avg_scored  = team_stats["avg_goals_scored"].mean()
    league_avg_allowed = team_stats["avg_goals_allowed"].mean()

    if league_avg_scored == 0 or league_avg_allowed == 0:
        return None

    team_stats["attack_strength"]  = team_stats["avg_goals_scored"]  / league_avg_scored
    # Avoid division by zero: teams that allowed 0 goals get a high but finite defense strength
    team_stats["defense_strength"] = league_avg_allowed / team_stats["avg_goals_allowed"].replace(0, 0.1)

    team_stats["season"] = season
    team_stats["league"] = league
    team_stats.index.name = "team_br"
    team_stats = team_stats.reset_index()

    return team_stats[["team_br", "season", "league", "attack_strength", "defense_strength", "games_played"]]


# ---------------------------------------------------------------------------
# Soccer fixtures — ESPN public API + CSV override + demo fallback
# ---------------------------------------------------------------------------

# ESPN league slugs to check for today's fixtures (covers major leagues + WC qualifiers)
_ESPN_SOCCER_LEAGUES = [
    # International friendlies (covers South America, Africa, Asia, CONCACAF)
    "fifa.friendly",
    # World Cup qualifiers
    "fifa.worldq.uefa",
    "fifa.worldq.conmebol",
    "fifa.worldq.concacaf",
    "fifa.worldq.caf",
    "fifa.worldq.afc",
    "fifa.worldq.ofc",
    # Continental competitions
    "uefa.champions",
    "uefa.europa",
    "uefa.europa.conf",
    "uefa.nations",
    "concacaf.gold",
    "concacaf.nations.league",
    "conmebol.copa",
    "conmebol.libertadores",
    "caf.champions",
    # Top club leagues
    "eng.1",
    "esp.1",
    "ger.1",
    "ita.1",
    "fra.1",
    "ned.1",
    "por.1",
    "eng.2",
    "mls",
]


def fetch_soccer_fixtures(target_date: Optional[str] = None) -> list[dict]:
    """
    Load soccer fixtures for the target date.

    Priority order:
      1. cache/soccer_fixtures.csv  (manual override)
      2. ESPN public API            (today's live schedule, no key required)
      3. Demo fallback              (recent historical games from training data)
    """
    game_date = target_date or date.today().isoformat()

    # 1. Manual CSV override
    fixtures_path = CACHE_DIR / "soccer_fixtures.csv"
    if fixtures_path.exists():
        try:
            df = pd.read_csv(fixtures_path)
            games = []
            for _, row in df.iterrows():
                home = str(row.get("home_team", "")).strip()
                away = str(row.get("away_team", "")).strip()
                if not home or not away:
                    continue
                games.append({
                    "home_team_br":   home,
                    "away_team_br":   away,
                    "home_team_name": home,
                    "away_team_name": away,
                    "game_datetime":  str(row.get("game_datetime", game_date)),
                    "venue":          str(row.get("venue", "")),
                    "league":         str(row.get("league", "")),
                })
            if games:
                print(f"  [fetch] Loaded {len(games)} fixtures from soccer_fixtures.csv")
                return games
        except Exception as exc:
            print(f"  [fetch] WARN: Could not read soccer_fixtures.csv: {exc}")

    # 2. ESPN public API
    espn_games = _fetch_espn_soccer_today(game_date)
    if espn_games:
        return espn_games

    # 3. Demo fallback
    return _get_demo_soccer_fixtures(game_date)


def _fetch_espn_soccer_today(game_date: str) -> list[dict]:
    """
    Fetch soccer fixtures from ESPN's public scoreboard API.
    Checks today and up to 3 days ahead if today has fewer than 3 games.
    No API key required. Returns [] on any failure.
    """
    base_dt = datetime.strptime(game_date, "%Y-%m-%d")
    all_games: list[dict] = []
    seen: set[str] = set()

    # Check today first, then upcoming days until we have enough games
    for day_offset in range(14):
        check_dt = base_dt + timedelta(days=day_offset)
        date_str = check_dt.strftime("%Y%m%d")
        day_games: list[dict] = []

        for league_slug in _ESPN_SOCCER_LEAGUES:
            url = (
                f"https://site.api.espn.com/apis/site/v2/sports/soccer"
                f"/{league_slug}/scoreboard?dates={date_str}&limit=50"
            )
            try:
                resp = requests.get(url, timeout=8)
                if resp.status_code != 200:
                    continue
                for event in resp.json().get("events", []):
                    comps = event.get("competitions", [{}])[0]
                    competitors = comps.get("competitors", [])
                    if len(competitors) < 2:
                        continue

                    home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                    away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

                    home_name = home_c.get("team", {}).get("displayName", "")
                    away_name = away_c.get("team", {}).get("displayName", "")
                    if not home_name or not away_name:
                        continue

                    key = f"{home_name}|{away_name}"
                    if key in seen:
                        continue
                    seen.add(key)

                    venue = (comps.get("venue") or {}).get("fullName", "")
                    game_time = event.get("date", game_date)[:16]

                    home_br = _ESPN_TO_FDUK.get(home_name, home_name)
                    away_br = _ESPN_TO_FDUK.get(away_name, away_name)

                    day_games.append({
                        "home_team_br":   home_br,
                        "away_team_br":   away_br,
                        "home_team_name": home_name,
                        "away_team_name": away_name,
                        "game_datetime":  game_time,
                        "venue":          venue,
                        "league":         league_slug,
                    })
            except Exception:
                continue

        all_games.extend(day_games)

        # Stop looking ahead once we have enough games
        if len(all_games) >= 20:
            break

    if all_games:
        dates = sorted({g["game_datetime"][:10] for g in all_games})
        print(f"  [fetch] ESPN: {len(all_games)} fixture(s) across {dates}")
    return all_games


# Mapping from ESPN display names → football-data.co.uk team names
# (for clubs that appear in both; national teams stay as-is)
_ESPN_TO_FDUK: dict[str, str] = {
    "Manchester City":     "Man City",
    "Manchester United":   "Man United",
    "Tottenham Hotspur":   "Tottenham",
    "Newcastle United":    "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest":   "Nott'm Forest",
    "Leicester City":      "Leicester",
    "Leeds United":        "Leeds",
    "West Ham United":     "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth":     "Bournemouth",
    "Atletico de Madrid":  "Ath Madrid",
    "Athletic Club":       "Ath Bilbao",
    "Real Betis":          "Betis",
    "Real Sociedad":       "Sociedad",
    "Bayer Leverkusen":    "Leverkusen",
    "Borussia Dortmund":   "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "Paris Saint-Germain": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais":  "Lyon",
    "AS Monaco":           "Monaco",
    "Internazionale":      "Inter",
    "AC Milan":            "Milan",
    "AS Roma":             "Roma",
    "SSC Napoli":          "Napoli",
    "Juventus":            "Juventus",
    "SS Lazio":            "Lazio",
    "Atalanta BC":         "Atalanta",
    "ACF Fiorentina":      "Fiorentina",
    "SL Benfica":          "Benfica",
    "FC Porto":            "Porto",
    "Sporting CP":         "Sp Lisbon",
    "Ajax":                "Ajax",
    "PSV Eindhoven":       "PSV",
}


def _get_demo_soccer_fixtures(game_date: str) -> list[dict]:
    """Fallback: recent matches from the last 7 days of available training data."""
    last_season = max(TRAIN_SEASONS)
    cached = _load(f"soccer_season_games_{last_season}")
    if cached is None or cached.empty:
        print("  [fetch] No cached soccer data found for demo fixtures.")
        return []

    max_date = cached["date"].max()
    cutoff = max_date - pd.Timedelta(days=7)
    recent = cached[cached["date"] >= cutoff].copy()

    if recent.empty:
        recent = cached.sort_values("date").tail(10)

    top_leagues = ["E0", "D1", "SP1", "I1", "F1"]
    top = recent[recent["league"].isin(top_leagues)]
    if len(top) >= 5:
        recent = top

    recent = recent.head(10)

    games = []
    for _, row in recent.iterrows():
        games.append({
            "home_team_br":   row["home_team_br"],
            "away_team_br":   row["away_team_br"],
            "home_team_name": row["home_team_br"],
            "away_team_name": row["away_team_br"],
            "game_datetime":  str(row["date"].date()),
            "venue":          "",
            "league":         str(row.get("league", "")),
        })

    print(f"  [fetch] Demo fallback: {len(games)} fixtures from {cutoff.date()} – {max_date.date()}")
    return games


# ---------------------------------------------------------------------------
# Today's schedule (statsapi)
# ---------------------------------------------------------------------------

def fetch_today_schedule(target_date: Optional[str] = None) -> list[dict]:
    """
    Fetch today's MLB regular season schedule with probable starters.
    target_date: 'YYYY-MM-DD'. Defaults to today.
    If no regular-season games today, checks next 2 days.
    """
    check_date = target_date or date.today().strftime("%Y-%m-%d")

    for offset in range(3):
        d = date.fromisoformat(check_date) + timedelta(days=offset)
        date_str = d.strftime("%m/%d/%Y")
        try:
            raw = statsapi.schedule(date=date_str, sportId=1)
            regular = [g for g in (raw or []) if g.get("game_type") == "R"]
            if regular:
                return [_parse_statsapi_game(g) for g in regular]
        except Exception as exc:
            print(f"  [fetch] WARN: statsapi {date_str}: {exc}")

    return []


def _parse_statsapi_game(g: dict) -> dict:
    home_name = g.get("home_name", "")
    away_name = g.get("away_name", "")
    return {
        "game_id":           g.get("game_id"),
        "game_datetime":     g.get("game_datetime", ""),
        "home_team_br":      STATSAPI_NAME_TO_BR.get(home_name, home_name[:3].upper()),
        "away_team_br":      STATSAPI_NAME_TO_BR.get(away_name, away_name[:3].upper()),
        "home_team_name":    home_name,
        "away_team_name":    away_name,
        "home_pitcher_name": g.get("home_probable_pitcher", "TBD") or "TBD",
        "away_pitcher_name": g.get("away_probable_pitcher", "TBD") or "TBD",
        "home_pitcher_id":   g.get("home_pitcher_id"),
        "away_pitcher_id":   g.get("away_pitcher_id"),
        "venue":             g.get("venue_name", ""),
        "status":            g.get("status", ""),
    }
