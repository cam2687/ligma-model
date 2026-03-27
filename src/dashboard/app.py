"""
Streamlit dashboard for MLB AI predictions.

Run with:
    streamlit run src/dashboard/app.py

Or via:
    python main.py serve
"""
import sys
import json
import os
import subprocess
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import MODELS_DIR, CACHE_DIR, BR_TO_FULL_NAME


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MLB AI Predictions",
    page_icon="⚾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS overrides for cleaner card style
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Mobile-first base styles ── */
.game-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 14px 12px;
    margin-bottom: 16px;
    background: #fafafa;
}
.winner-highlight { color: #2ecc71; font-weight: bold; }
.loser-dim        { color: #aaaaaa; }
.moneyline-fav    { color: #e74c3c; font-weight: bold; }
.moneyline-dog    { color: #27ae60; font-weight: bold; }
.confidence-high  { background: #2ecc71; color: white; border-radius: 4px; padding: 3px 10px; font-size: 0.85rem; }
.confidence-mod   { background: #f39c12; color: white; border-radius: 4px; padding: 3px 10px; font-size: 0.85rem; }
.confidence-low   { background: #95a5a6; color: white; border-radius: 4px; padding: 3px 10px; font-size: 0.85rem; }

/* Matchup header: big team names */
.matchup-header {
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.5;
    margin-bottom: 4px;
}

/* Meta row: badge + time inline */
.game-meta {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

/* Stat label */
.stat-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #888;
    margin-bottom: 2px;
}

/* Score block */
.score-block {
    font-size: 1rem;
    line-height: 1.7;
}

/* Moneyline block */
.ml-block {
    font-size: 1rem;
    line-height: 1.7;
}

/* Form block */
.form-block {
    font-size: 0.88rem;
    line-height: 1.7;
}

/* Make Streamlit columns stack on very small screens */
@media (max-width: 480px) {
    .matchup-header { font-size: 0.95rem; }
    .score-block, .ml-block { font-size: 0.92rem; }
    [data-testid="column"] { min-width: 100% !important; }
}

/* Remove excess padding from the main block */
.block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 800px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def load_predictions(sport: str) -> list[dict]:
    """Load pre-computed predictions for the given sport."""
    pred_path = CACHE_DIR / f"{sport}_predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            predictions = json.load(f)
        today = date.today()
        filtered = []
        for game in predictions:
            raw = str(game.get("game_datetime", "")).strip()
            parsed = pd.to_datetime(raw, errors="coerce")
            if pd.isna(parsed) and raw.isdigit() and len(raw) == 8:
                parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
            if pd.isna(parsed) or parsed.date() >= today:
                filtered.append(game)
        return filtered
    return []


def _run_sport_command(sport: str, command: str, timeout_s: int) -> tuple[bool, str]:
    """Run a CLI subcommand for one sport using SPORT_OVERRIDE."""
    project_root = Path(__file__).parents[2]
    env = dict(os.environ)
    env["SPORT_OVERRIDE"] = sport
    try:
        result = subprocess.run(
            [sys.executable, "main.py", command],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"`python main.py {command}` timed out for {sport}."
    except Exception as exc:
        return False, str(exc)

    if result.returncode == 0:
        return True, (result.stdout or "").strip()

    combined = "\n".join(
        part.strip() for part in [result.stdout or "", result.stderr or ""] if part.strip()
    )
    return False, combined or f"`python main.py {command}` failed for {sport}."


def refresh_prediction_file(sport: str) -> tuple[bool, str]:
    """Regenerate one sport's prediction file via the CLI."""
    try:
        ok, output = _run_sport_command(sport, "predict", timeout_s=120)
        if ok:
            return True, f"Predictions refreshed for {sport}."

        if "Model not found at" in output:
            ok, train_output = _run_sport_command(sport, "train", timeout_s=900)
            if not ok:
                last_line = train_output.strip().splitlines()[-1] if train_output.strip() else "Training failed."
                return False, f"{sport.title()} model training failed: {last_line}"

            ok, predict_output = _run_sport_command(sport, "predict", timeout_s=120)
            if ok:
                return True, f"{sport.title()} model trained and predictions refreshed."
            last_line = predict_output.strip().splitlines()[-1] if predict_output.strip() else "Prediction failed."
            return False, f"{sport.title()} prediction failed after training: {last_line}"

        last_line = output.strip().splitlines()[-1] if output.strip() else "Prediction refresh failed."
        return False, last_line
    except Exception as exc:
        return False, str(exc)


@st.cache_data(ttl=3600, show_spinner=False)
def load_cv_metrics(sport: str) -> dict:
    metrics_path = MODELS_DIR / f"{sport}_cv_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_feature_importance(sport: str) -> pd.DataFrame:
    imp_path = MODELS_DIR / f"{sport}_feature_importance.csv"
    if imp_path.exists():
        return pd.read_csv(imp_path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def win_prob_bar(home_team: str, away_team: str,
                 home_prob: float, away_prob: float,
                 predicted_winner: str,
                 draw_prob: float | None = None) -> go.Figure:
    """Horizontal bar chart showing outcome probabilities."""
    home_pct = home_prob * 100
    away_pct = away_prob * 100

    home_color = "#2ecc71" if predicted_winner == "home" else "#95a5a6"
    away_color = "#2ecc71" if predicted_winner == "away" else "#95a5a6"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[home_pct], y=[home_team],
        orientation="h",
        marker_color=home_color,
        text=[f"{home_pct:.1f}%"],
        textposition="outside",
        name=home_team,
    ))
    fig.add_trace(go.Bar(
        x=[away_pct], y=[away_team],
        orientation="h",
        marker_color=away_color,
        text=[f"{away_pct:.1f}%"],
        textposition="outside",
        name=away_team,
    ))
    if draw_prob is not None:
        draw_pct = draw_prob * 100
        draw_color = "#2ecc71" if predicted_winner == "draw" else "#95a5a6"
        fig.add_trace(go.Bar(
            x=[draw_pct], y=["Draw"],
            orientation="h",
            marker_color=draw_color,
            text=[f"{draw_pct:.1f}%"],
            textposition="outside",
            name="Draw",
        ))
    fig.update_layout(
        xaxis=dict(range=[0, 115], showticklabels=False, showgrid=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        showlegend=False,
        height=100,
        margin=dict(l=0, r=40, t=4, b=4),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_confidence_badge(confidence: str) -> str:
    cls_map = {"High": "confidence-high", "Moderate": "confidence-mod", "Low": "confidence-low"}
    cls = cls_map.get(confidence, "confidence-low")
    return f'<span class="{cls}">{confidence}</span>'


def render_form_string(form: str) -> str:
    """Color-code W/L form string."""
    parts = []
    for ch in form.split():
        if ch == "W":
            parts.append(f'<span style="color:#2ecc71;font-weight:bold">W</span>')
        elif ch == "L":
            parts.append(f'<span style="color:#e74c3c">L</span>')
        elif ch == "D":
            parts.append(f'<span style="color:#f39c12;font-weight:bold">D</span>')
        else:
            parts.append(ch)
    return " ".join(parts)


def _utc_to_et(dt_str: str) -> str:
    """Convert a UTC ISO datetime string to 12-hour ET time string, e.g. '4:35 PM ET'."""
    if not dt_str:
        return ""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        # EDT = UTC-4 covers the entire MLB regular season (April-October)
        dt_et = dt.astimezone(timezone(timedelta(hours=-4)))
        hour = dt_et.hour % 12 or 12
        return f"{hour}:{dt_et.strftime('%M %p')} ET"
    except Exception:
        return dt_str[:16].replace("T", " ")


def render_game_card(game: dict) -> None:
    """Render a single game prediction card (mobile-first layout)."""
    h = game["home_team_name"]
    a = game["away_team_name"]
    h_br = game.get("home_team_br", "")
    a_br = game.get("away_team_br", "")
    winner = game["predicted_winner"]

    if winner == "draw":
        h_class = "winner-highlight"
        a_class = "winner-highlight"
    else:
        h_class = "winner-highlight" if winner == "home" else "loser-dim"
        a_class = "winner-highlight" if winner == "away" else "loser-dim"

    h_ml = game["home_moneyline_str"]
    a_ml = game["away_moneyline_str"]
    h_ml_class = "moneyline-fav" if game["home_moneyline"] < 0 else "moneyline-dog"
    a_ml_class = "moneyline-fav" if game["away_moneyline"] < 0 else "moneyline-dog"

    venue = game.get("venue", "")
    dt    = _utc_to_et(game.get("game_datetime", ""))
    conf_badge = render_confidence_badge(game["confidence"])

    with st.container():
        # ── Row 1: Matchup + meta (badge + time) ──
        st.markdown(
            f"<div class='matchup-header'>"
            f"<span class='{a_class}'>{a} ({a_br})</span>"
            f" <span style='color:#bbb'>@</span> "
            f"<span class='{h_class}'>{h} ({h_br})</span>"
            f"</div>"
            f"<div class='game-meta'>{conf_badge} &nbsp; {dt}</div>",
            unsafe_allow_html=True,
        )

        # ── Row 2: Win probability bar (full width) ──
        st.markdown("<div class='stat-label'>Win Probability</div>", unsafe_allow_html=True)
        fig = win_prob_bar(
            f"{h_br} (Home)", f"{a_br} (Away)",
            game["home_win_prob"], game["away_win_prob"], winner,
            draw_prob=game.get("draw_prob"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Row 3: Score | Moneyline (2 equal columns) ──
        col_score, col_ml = st.columns(2)

        with col_score:
            if "pred_home_goals" in game:
                pred_home = game.get("pred_home_goals", 0)
                pred_away = game.get("pred_away_goals", 0)
                unit = "goals"
            elif "pred_home_runs" in game:
                pred_home = game.get("pred_home_runs", 0)
                pred_away = game.get("pred_away_runs", 0)
                unit = "runs"
            else:
                pred_home, pred_away, unit = 0, 0, "pts"

            st.markdown(
                f"<div class='stat-label'>Predicted Score</div>"
                f"<div class='score-block'>"
                f"<span class='{h_class}'>{h_br} <b>{pred_home}</b></span><br>"
                f"<span class='{a_class}'>{a_br} <b>{pred_away}</b></span><br>"
                f"<small style='color:#888'>Total: {game['predicted_total']} {unit}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_ml:
            lines = [f"<span class='{h_ml_class}'>{h_br}: {h_ml}</span>"]
            if "draw_moneyline_str" in game:
                lines.append(f"<span class='moneyline-dog'>Draw: {game['draw_moneyline_str']}</span>")
            lines.append(f"<span class='{a_ml_class}'>{a_br}: {a_ml}</span>")
            st.markdown(
                f"<div class='stat-label'>Fair Moneyline</div>"
                f"<div class='ml-block'>"
                + "<br>".join(lines) +
                f"<br><small style='color:#888'>no-vig implied</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Row 4: Recent form ──
        h_form = render_form_string(game.get("recent_form_home", "N/A"))
        a_form = render_form_string(game.get("recent_form_away", "N/A"))
        st.markdown(
            f"<div class='stat-label' style='margin-top:6px'>Recent Form (last 10)</div>"
            f"<div class='form-block'>"
            f"{h_br}: {h_form}<br>{a_br}: {a_form}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Footer captions ──
        captions = []
        if game.get("home_pitcher") or game.get("away_pitcher"):
            captions.append(
                f"SP: {a_br} — {game.get('away_pitcher','TBD')} | "
                f"{h_br} — {game.get('home_pitcher','TBD')}"
            )
        if venue:
            captions.append(f"Venue: {venue}")
        for cap in captions:
            st.caption(cap)

        st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, dict]:
    """Render sidebar and return the selected sport and its metrics."""
    with st.sidebar:
        st.title("AI Sports Predictions")
        st.caption(f"Today: {date.today().strftime('%B %d, %Y')}")

        st.divider()

        sport = st.radio(
            "Sport",
            options=["mlb", "soccer"],
            format_func=lambda s: "⚾ MLB Baseball" if s == "mlb" else "⚽ Soccer",
            horizontal=True,
        )
        cv_metrics = load_cv_metrics(sport)

        st.divider()

        if st.button("🔄 Refresh Predictions", use_container_width=True):
            with st.spinner(f"Refreshing {sport} predictions..."):
                ok, msg = refresh_prediction_file(sport)
            st.cache_data.clear()
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            st.rerun()

        st.divider()

        # Model performance
        st.subheader("Model Performance (CV)")
        if "win_classifier" in cv_metrics:
            wc = cv_metrics["win_classifier"]
            total_val_games = int(sum(f.get("val_size", 0) for f in wc.get("folds", [])))
            if sport == "soccer":
                st.metric("1X2 Accuracy", f"{wc.get('mean_accuracy', 0):.1%}")
                st.metric("ROC-AUC (OvR)", f"{wc.get('mean_auc_roc', 0):.3f}")
                st.metric("Brier Score", f"{wc.get('mean_brier', 0):.4f}")
                if "mean_macro_f1" in wc:
                    st.metric("Macro-F1", f"{wc.get('mean_macro_f1', 0):.3f}")
                st.caption(
                    "Soccer is a 3-way outcome model (home/draw/away). "
                    "Accuracy and Macro-F1 matter more than the old binary baseline."
                )
                if total_val_games:
                    st.caption(f"Validation sample: {total_val_games:,} held-out matches across 5 temporal folds.")
            else:
                st.metric("Win Accuracy", f"{wc.get('mean_accuracy', 0):.1%}")
                st.metric("ROC-AUC", f"{wc.get('mean_auc_roc', 0):.3f}")
                st.metric("Brier Score", f"{wc.get('mean_brier', 0):.4f}")
                st.caption("Lower Brier = better calibration. Naive guess = 0.25")
                if total_val_games:
                    st.caption(f"Validation sample: {total_val_games:,} held-out games across 5 temporal folds.")

        if "home_runs_regressor" in cv_metrics and "away_runs_regressor" in cv_metrics:
            hr = cv_metrics["home_runs_regressor"]
            ar = cv_metrics["away_runs_regressor"]
            st.metric("Home Runs RMSE", f"{hr.get('mean_rmse', 0):.2f}")
            st.metric("Away Runs RMSE", f"{ar.get('mean_rmse', 0):.2f}")
        elif "home_goals_regressor" in cv_metrics and "away_goals_regressor" in cv_metrics:
            hr = cv_metrics["home_goals_regressor"]
            ar = cv_metrics["away_goals_regressor"]
            st.metric("Home Goals RMSE", f"{hr.get('mean_rmse', 0):.2f}")
            st.metric("Away Goals RMSE", f"{ar.get('mean_rmse', 0):.2f}")

        st.divider()

        st.markdown("""
        **How to read this:**
        - Soccer uses 1X2 probabilities: home, draw, away
        - Win % = model's probability for each outcome
        - Soccer CV is stricter now: no same-season leakage and draws are modeled explicitly
        - Fair Moneyline = no-vig implied odds
        - Confidence: based on the top predicted outcome probability
        - Rolling form based on last 15 games
        """)

    return sport, cv_metrics


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    sport, cv_metrics = render_sidebar()

    if sport == "soccer":
        st.title("⚽ Soccer Game Predictions")
        st.caption(f"Powered by XGBoost | Data: football-data.co.uk | {date.today()}")
    else:
        st.title("⚾ MLB Game Predictions")
        st.caption(f"Powered by XGBoost | Data: pybaseball + MLB Stats API | {date.today()}")

    predictions = load_predictions(sport)
    feat_imp = load_feature_importance(sport)

    # ---- Games tab ----
    tab_games, tab_model, tab_importance = st.tabs([
        "Today's Games", "Model Performance", "Feature Importance"
    ])

    with tab_games:
        if not predictions:
            if sport == "mlb":
                st.info("No MLB predictions yet. Run:")
                st.code("python main.py predict")
            else:
                st.info("No soccer predictions yet. Run:")
                st.code("python main.py predict")
        else:
            st.subheader(f"{len(predictions)} Games Today")
            for game in sorted(predictions, key=lambda g: g.get("game_datetime", "")):
                render_game_card(game)

    with tab_model:
        cv = cv_metrics
        if not cv:
            st.info("No model metrics available. Train the model first.")
        else:
            wc = cv.get("win_classifier", {})
            if "home_goals_regressor" in cv:
                hr = cv.get("home_goals_regressor", {})
                ar = cv.get("away_goals_regressor", {})
                reg_title_home = "Home Goals Regressor (Temporal CV)"
                reg_title_away = "Away Goals Regressor (Temporal CV)"
            elif "home_runs_regressor" in cv:
                hr = cv.get("home_runs_regressor", {})
                ar = cv.get("away_runs_regressor", {})
                reg_title_home = "Home Runs Regressor (Temporal CV)"
                reg_title_away = "Away Runs Regressor (Temporal CV)"
            else:
                hr = {}
                ar = {}
                reg_title_home = "Home Regressor (Temporal CV)"
                reg_title_away = "Away Regressor (Temporal CV)"

            with st.expander("Win Classifier (Temporal CV)", expanded=True):
                if wc.get("folds"):
                    fold_df = pd.DataFrame(wc["folds"])
                    st.dataframe(fold_df.round(4), use_container_width=True)
                    fig = px.bar(fold_df, x="fold", y="accuracy",
                                 title="Accuracy by CV Fold",
                                 color="accuracy",
                                 color_continuous_scale="RdYlGn",
                                 range_color=[0.52, 0.62])
                    fig.add_hline(y=0.54, line_dash="dash",
                                  annotation_text="Home-team baseline (~54%)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No fold data available.")

            with st.expander(reg_title_home, expanded=False):
                if hr.get("folds"):
                    fold_df = pd.DataFrame(hr["folds"])
                    st.dataframe(fold_df.round(4), use_container_width=True)
                    fig = px.bar(fold_df, x="fold", y="rmse",
                                 title="RMSE by CV Fold",
                                 color="rmse",
                                 color_continuous_scale="RdYlGn_r")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No fold data available.")

            with st.expander(reg_title_away, expanded=False):
                if ar.get("folds"):
                    fold_df = pd.DataFrame(ar["folds"])
                    st.dataframe(fold_df.round(4), use_container_width=True)
                    fig = px.bar(fold_df, x="fold", y="rmse",
                                 title="RMSE by CV Fold",
                                 color="rmse",
                                 color_continuous_scale="RdYlGn_r")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No fold data available.")

    with tab_importance:
        if feat_imp.empty:
            st.info("Feature importance not available.")
        else:
            st.subheader("Top Features Driving Win Prediction")
            top_n = feat_imp.head(20)
            fig = px.bar(
                top_n, x="importance", y="feature", orientation="h",
                title="XGBoost Feature Importance (top 20)",
                color="importance", color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
