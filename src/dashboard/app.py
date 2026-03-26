"""
Streamlit dashboard for MLB AI predictions.

Run with:
    streamlit run src/dashboard/app.py

Or via:
    python main.py serve
"""
import sys
import json
from pathlib import Path
from datetime import date

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
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS overrides for cleaner card style
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.game-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    background: #fafafa;
}
.winner-highlight { color: #2ecc71; font-weight: bold; }
.loser-dim        { color: #aaaaaa; }
.moneyline-fav    { color: #e74c3c; font-weight: bold; }
.moneyline-dog    { color: #27ae60; font-weight: bold; }
.confidence-high  { background: #2ecc71; color: white; border-radius: 4px; padding: 2px 8px; }
.confidence-mod   { background: #f39c12; color: white; border-radius: 4px; padding: 2px 8px; }
.confidence-low   { background: #95a5a6; color: white; border-radius: 4px; padding: 2px 8px; }
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
            return json.load(f)
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def load_cv_metrics() -> dict:
    metrics_path = MODELS_DIR / "cv_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    imp_path = MODELS_DIR / "feature_importance.csv"
    if imp_path.exists():
        return pd.read_csv(imp_path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def win_prob_bar(home_team: str, away_team: str,
                 home_prob: float, away_prob: float,
                 predicted_winner: str) -> go.Figure:
    """Horizontal bar chart showing win probabilities."""
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
    fig.update_layout(
        xaxis=dict(range=[0, 115], showticklabels=False, showgrid=False),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=120,
        margin=dict(l=10, r=10, t=5, b=5),
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
        else:
            parts.append(ch)
    return " ".join(parts)


def render_game_card(game: dict) -> None:
    """Render a single game prediction card."""
    h = game["home_team_name"]
    a = game["away_team_name"]
    h_br = game.get("home_team_br", "")
    a_br = game.get("away_team_br", "")
    winner = game["predicted_winner"]

    h_class = "winner-highlight" if winner == "home" else "loser-dim"
    a_class = "winner-highlight" if winner == "away" else "loser-dim"

    h_ml = game["home_moneyline_str"]
    a_ml = game["away_moneyline_str"]
    h_ml_class = "moneyline-fav" if game["home_moneyline"] < 0 else "moneyline-dog"
    a_ml_class = "moneyline-fav" if game["away_moneyline"] < 0 else "moneyline-dog"

    venue = game.get("venue", "")
    dt    = game.get("game_datetime", "")[:16].replace("T", " ") if game.get("game_datetime") else ""

    with st.container():
        # Header row
        col_teams, col_meta = st.columns([3, 1])
        with col_teams:
            st.markdown(
                f"**<span class='{a_class}'>{a} ({a_br})</span>** "
                f"@ **<span class='{h_class}'>{h} ({h_br})</span>**",
                unsafe_allow_html=True,
            )
        with col_meta:
            conf_badge = render_confidence_badge(game["confidence"])
            st.markdown(f"{conf_badge} &nbsp; {dt}", unsafe_allow_html=True)

        # Main prediction columns
        col_bar, col_score, col_ml, col_form = st.columns([3, 2, 2, 3])

        with col_bar:
            st.markdown("**Win Probability**")
            fig = win_prob_bar(
                f"{h_br} (Home)", f"{a_br} (Away)",
                game["home_win_prob"], game["away_win_prob"], winner,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_score:
            st.markdown("**Predicted Score**")
            if "pred_home_goals" in game:
                pred_home = game.get('pred_home_goals', 0)
                pred_away = game.get('pred_away_goals', 0)
                unit = "goals"
            elif "pred_home_runs" in game:
                pred_home = game.get('pred_home_runs', 0)
                pred_away = game.get('pred_away_runs', 0)
                unit = "runs"
            else:
                pred_home = 0
                pred_away = 0
                unit = "pts"

            st.markdown(
                f"<span class='{h_class}'>{h_br} **{pred_home}**</span><br>"
                f"<span class='{a_class}'>{a_br} **{pred_away}**</span><br>"
                f"<small>Total: {game['predicted_total']} {unit}</small>",
                unsafe_allow_html=True,
            )

        with col_ml:
            st.markdown("**Fair Moneyline**")
            st.markdown(
                f"{h_br}: <span class='{h_ml_class}'>{h_ml}</span><br>"
                f"{a_br}: <span class='{a_ml_class}'>{a_ml}</span><br>"
                f"<small style='color:#888'>Model implied (no vig)</small>",
                unsafe_allow_html=True,
            )

        with col_form:
            st.markdown("**Recent Form (last 10)**")
            h_form = render_form_string(game.get("recent_form_home", "N/A"))
            a_form = render_form_string(game.get("recent_form_away", "N/A"))
            st.markdown(
                f"{h_br}: {h_form}<br>{a_br}: {a_form}",
                unsafe_allow_html=True,
            )

        # Starting pitchers
        if game.get("home_pitcher") or game.get("away_pitcher"):
            st.caption(
                f"SP: {a_br} — {game.get('away_pitcher','TBD')} | "
                f"{h_br} — {game.get('home_pitcher','TBD')}"
            )
        if venue:
            st.caption(f"Venue: {venue}")

        st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(cv_metrics: dict) -> str:
    """Render sidebar and return the selected sport ('mlb' or 'soccer')."""
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

        st.divider()

        if st.button("🔄 Refresh Predictions", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Model performance
        st.subheader("Model Performance (CV)")
        if "win_classifier" in cv_metrics:
            wc = cv_metrics["win_classifier"]
            st.metric("Win Accuracy",  f"{wc.get('mean_accuracy', 0):.1%}")
            st.metric("ROC-AUC",       f"{wc.get('mean_auc_roc', 0):.3f}")
            st.metric("Brier Score",   f"{wc.get('mean_brier', 0):.4f}")
            st.caption("Lower Brier = better calibration. Naive guess = 0.25")

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
        - Win % = model's probability for each team
        - Fair Moneyline = no-vig implied odds
        - Confidence: how far prob is from 50%
        - Rolling form based on last 15 games
        """)

    return sport


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    sport = render_sidebar(load_cv_metrics())

    if sport == "soccer":
        st.title("⚽ Soccer Game Predictions")
        st.caption(f"Powered by XGBoost | Data: football-data.co.uk | {date.today()}")
    else:
        st.title("⚾ MLB Game Predictions")
        st.caption(f"Powered by XGBoost | Data: pybaseball + MLB Stats API | {date.today()}")

    predictions = load_predictions(sport)
    feat_imp    = load_feature_importance()

    # ---- Games tab ----
    tab_games, tab_model, tab_importance = st.tabs([
        "Today's Games", "Model Performance", "Feature Importance"
    ])

    with tab_games:
        if not predictions:
            if sport == "mlb":
                st.info("No MLB predictions yet. Set `SPORT = 'mlb'` in config.py then run:")
                st.code("python main.py predict")
            else:
                st.info("No soccer predictions yet. Run:")
                st.code("python main.py predict")
        else:
            st.subheader(f"{len(predictions)} Games Today")
            for game in predictions:
                render_game_card(game)

    with tab_model:
        cv = load_cv_metrics()
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

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Win Classifier (Temporal CV)")
                if wc.get("folds"):
                    fold_df = pd.DataFrame(wc["folds"])
                    st.dataframe(fold_df.round(4), use_container_width=True)
                    # Accuracy by fold chart
                    fig = px.bar(fold_df, x="fold", y="accuracy",
                                 title="Accuracy by CV Fold",
                                 color="accuracy",
                                 color_continuous_scale="RdYlGn",
                                 range_color=[0.52, 0.62])
                    fig.add_hline(y=0.54, line_dash="dash",
                                  annotation_text="Home-team baseline (~54%)")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader(reg_title_home)
                if hr.get("folds"):
                    fold_df = pd.DataFrame(hr["folds"])
                    st.dataframe(fold_df.round(4), use_container_width=True)
                    fig = px.bar(fold_df, x="fold", y="rmse",
                                 title="RMSE by CV Fold",
                                 color="rmse",
                                 color_continuous_scale="RdYlGn_r")
                    st.plotly_chart(fig, use_container_width=True)

            with col3:
                st.subheader(reg_title_away)
                if ar.get("folds"):
                    fold_df = pd.DataFrame(ar["folds"])
                    st.dataframe(fold_df.round(4), use_container_width=True)
                    fig = px.bar(fold_df, x="fold", y="rmse",
                                 title="RMSE by CV Fold",
                                 color="rmse",
                                 color_continuous_scale="RdYlGn_r")
                    st.plotly_chart(fig, use_container_width=True)

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
