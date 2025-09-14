# props_app.py
# NFL Player Prop Projections (QB/WR/RB) — MVP using last season usage + opponent defensive adjustment
# Requires: Python 3.10+, streamlit, pandas, numpy, nfl_data_py

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from nfl_data_py import import_weekly_data, import_schedules

st.set_page_config(page_title="NFL Player Prop Projections (MVP)", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def _col(df, name, fallback=None):
    """Safe column accessor with fallback if column doesn't exist."""
    return df[name] if name in df.columns else fallback

def to_utc(ts):
    if pd.isna(ts):
        return ts
    if isinstance(ts, str):
        try:
            return pd.to_datetime(ts, utc=True)
        except Exception:
            return pd.NaT
    dt = pd.to_datetime(ts)
    if dt.tzinfo is None:
        return dt.tz_localize("UTC")
    return dt.tz_convert("UTC")

@st.cache_data(show_spinner=False)
def load_last_season_weekly(last_season:int):
    # Pull weekly player-level stats for prior season
    wk = import_weekly_data([last_season])
    # Normalize common columns used below
    rename_map = {
        "recent_team": "team",
        "opponent": "opponent_team",
        "player": "player_name",
        "player_display_name": "player_name",
        "player_name": "player_name",
        "pass_attempts": "attempts",
        "pass_completions": "completions",
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "rushing_attempts": "carries",
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_tds": "receiving_tds",
        "position": "position",
        "pos": "position",
        "team_abbr": "team",
        "opponent_abbr": "opponent_team",
        "nfl_player_id": "player_id",
        "gsis_id": "player_id",
        "pfr_player_id": "player_id",
    }
    for k,v in list(rename_map.items()):
        if k in wk.columns and v not in wk.columns:
            wk = wk.rename(columns={k:v})

    keep = [c for c in [
        "season","week","team","opponent_team","position","player_id","player_name",
        "attempts","completions","passing_yards","passing_tds",
        "carries","rushing_yards","rushing_tds",
        "targets","receptions","receiving_yards","receiving_tds",
        "snaps"
    ] if c in wk.columns]
    wk = wk[keep].copy()

    num_cols = [c for c in keep if c not in ["season","week","team","opponent_team","position","player_id","player_name"]]
    for c in num_cols:
        wk[c] = pd.to_numeric(wk[c], errors="coerce").fillna(0.0)

    wk = wk[~wk["team"].isna()]
    return wk

@st.cache_data(show_spinner=False)
def load_schedule(season:int):
    sch = import_schedules([season])
    sch = sch[sch["game_type"].isin(["REG","POST"])].copy()
    if "start_time" in sch.columns:
        sch["start_time_utc"] = pd.to_datetime(sch["start_time"], utc=True, errors="coerce")
    elif "gameday" in sch.columns:
        sch["start_time_utc"] = pd.to_datetime(sch["gameday"], utc=True, errors="coerce")
    else:
        sch["start_time_utc"] = pd.NaT
    return sch

def per_game(df, group_cols, sum_cols):
    agg = df.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    games = df.groupby(group_cols, dropna=False).size().reset_index(name="games")
    out = agg.merge(games, on=group_cols, how="left")
    for c in sum_cols:
        out[c+"_pg"] = out[c] / out["games"].replace(0, np.nan)
    return out

def select_likely_starters(baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Pick QB1 (by attempts_pg), RB1 (by carries_pg), WR1/WR2 (by targets_pg) per team.
    Returns a single DataFrame; avoids mixing Series/namedtuples.
    """
    rows = []

    # Ensure the rank columns exist even if your data source is missing one
    for col in ["attempts_pg", "carries_pg", "targets_pg"]:
        if col not in baselines.columns:
            baselines[col] = 0.0

    # Build per-team selections as DataFrames, then concat
    for team, g in baselines.groupby("team", dropna=False):
        qb = g.loc[g["position"] == "QB"].sort_values("attempts_pg", ascending=False).head(1)
        rb = g.loc[g["position"].isin(["RB", "FB"])].sort_values("carries_pg", ascending=False).head(1)
        wr = g.loc[g["position"] == "WR"].sort_values("targets_pg", ascending=False).head(2)

        picks = pd.concat([qb, rb, wr], axis=0)
        if not picks.empty:
            rows.append(picks)

    if not rows:
        # Return empty with expected columns if nothing is found
        return baselines.head(0)

    out = pd.concat(rows, ignore_index=True)

    # Drop dupes (player traded mid-season, etc.)
    if "player_id" in out.columns:
        out = out.drop_duplicates(subset=["team", "player_id"], keep="first")
    else:
        out = out.drop_duplicates(subset=["team", "player_name", "position"], keep="first")

    return out
def build_defense_allowance_tables(wk):
    pass_allow = per_game(wk, ["opponent_team"], ["passing_yards","completions","passing_tds"]).rename(columns={"opponent_team":"defteam"})
    rush_allow = per_game(wk, ["opponent_team"], ["rushing_yards","carries","rushing_tds"]).rename(columns={"opponent_team":"defteam"})
    recv_allow = per_game(wk, ["opponent_team"], ["receiving_yards","receptions","receiving_tds"]).rename(columns={"opponent_team":"defteam"})
    league = {
        "pass_yards_pg": pass_allow["passing_yards_pg"].mean(),
        "completions_pg": pass_allow["completions_pg"].mean(),
        "pass_tds_pg": pass_allow["passing_tds_pg"].mean(),
        "rush_yards_pg": rush_allow["rushing_yards_pg"].mean(),
        "carries_pg": rush_allow["carries_pg"].mean(),
        "rush_tds_pg": rush_allow["rushing_tds_pg"].mean(),
        "rec_yards_pg": recv_allow["receiving_yards_pg"].mean(),
        "receptions_pg": recv_allow["receptions_pg"].mean(),
        "rec_tds_pg": recv_allow["receiving_tds_pg"].mean(),
    }
    return pass_allow, rush_allow, recv_allow, league

def adjust_player_row(row, opp, pass_allow, rush_allow, recv_allow, league, weight=0.5):
    position = row.get("position","")
    proj = {}
    pa = pass_allow[pass_allow["defteam"] == opp]
    ra = rush_allow[rush_allow["defteam"] == opp]
    rca = recv_allow[recv_allow["defteam"] == opp]

    def scale(base, opp_val, league_avg, w=weight):
        if pd.isna(base):
            return np.nan
        if league_avg is None or league_avg == 0 or pd.isna(opp_val):
            return base
        factor = 1.0 + w * ((opp_val - league_avg) / league_avg)
        return max(0.0, base * factor)

    if position == "QB":
        proj["proj_pass_yards"] = scale(row.get("passing_yards_pg",0), pa["passing_yards_pg"].mean(), league["pass_yards_pg"])
        proj["proj_completions"] = scale(row.get("completions_pg",0), pa["completions_pg"].mean(), league["completions_pg"])
        proj["proj_pass_tds"] = scale(row.get("passing_tds_pg",0), pa["passing_tds_pg"].mean(), league["pass_tds_pg"])
    elif position in ["RB","FB"]:
        proj["proj_rush_yards"] = scale(row.get("rushing_yards_pg",0), ra["rushing_yards_pg"].mean(), league["rush_yards_pg"])
        proj["proj_attempts"]   = scale(row.get("carries_pg",0),        ra["carries_pg"].mean(),        league["carries_pg"])
        proj["proj_rush_tds"]   = scale(row.get("rushing_tds_pg",0),    ra["rushing_tds_pg"].mean(),    league["rush_tds_pg"])
        proj["proj_rec_yards"]  = scale(row.get("receiving_yards_pg",0), rca["receiving_yards_pg"].mean(), league["rec_yards_pg"])
        proj["proj_receptions"] = scale(row.get("receptions_pg",0),      rca["receptions_pg"].mean(),    league["receptions_pg"])
        proj["proj_rec_tds"]    = scale(row.get("receiving_tds_pg",0),   rca["receiving_tds_pg"].mean(), league["rec_tds_pg"])
    elif position == "WR":
        proj["proj_rec_yards"]  = scale(row.get("receiving_yards_pg",0), rca["receiving_yards_pg"].mean(), league["rec_yards_pg"])
        proj["proj_receptions"] = scale(row.get("receptions_pg",0),      rca["receptions_pg"].mean(),    league["receptions_pg"])
        proj["proj_rec_tds"]    = scale(row.get("receiving_tds_pg",0),   rca["receiving_tds_pg"].mean(), league["rec_tds_pg"])
    return proj

# ----------------------------
# UI
# ----------------------------
st.title("🏈 Player Prop Projections (QB/WR/RB) — MVP")
st.caption(
    "Last season player usage + opponent defensive allowances → current games projections. "
    "Data: nflverse/nflfastR via nfl_data_py. This is a simple MVP—tune weights and add injury/role context."
)

with st.sidebar:
    tz = "America/New_York"
    today_local = datetime.now(ZoneInfo(tz))
    this_year = today_local.year

    st.header("Settings")
    season = st.number_input("Current season", min_value=2015, max_value=2030, value=this_year)
    last_season = st.number_input("Baseline season (last year)", min_value=2015, max_value=2030, value=this_year-1)
    min_games = st.slider("Min games for baseline", 4, 17, 6, 1)
    weight = st.slider("Opponent weight (0=off, 1=full)", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.write("Positions included: **QB, WR (top 2), RB**. Edit table to tweak starters/teams if needed, then re-run projections.")

with st.spinner("Loading last season player-week data..."):
    wk = load_last_season_weekly(last_season)

sum_cols = [c for c in [
    "attempts","completions","passing_yards","passing_tds",
    "carries","rushing_yards","rushing_tds",
    "targets","receptions","receiving_yards","receiving_tds"
] if c in wk.columns]

player_pg = per_game(wk, ["team","player_id","player_name","position"], sum_cols)
player_pg = player_pg[player_pg["games"] >= min_games].copy()

likely = select_likely_starters(player_pg)
if likely.empty:
    st.warning("Could not identify likely starters from last season. Try lowering 'Min games for baseline'.")
else:
    st.subheader("Likely starters from last season (editable)")
    editable_cols = [c for c in likely.columns if c in ["team","player_name","position"] or c.endswith("_pg") or c=="games"]
    edited = st.data_editor(likely[editable_cols].reset_index(drop=True), use_container_width=True, num_rows="dynamic")

    with st.spinner("Loading current season schedule..."):
        sched = load_schedule(season)

    now_utc = datetime.now(timezone.utc)
    sched["start_time_utc"] = pd.to_datetime(sched["start_time_utc"], utc=True, errors="coerce")
    upcoming = sched[sched["start_time_utc"] >= now_utc].copy().sort_values("start_time_utc")

    st.subheader(f"Upcoming games ({len(upcoming)})")
    show_cols = [c for c in ["week","game_type","away_team","home_team","start_time_utc"] if c in upcoming.columns]
    st.dataframe(upcoming[show_cols], use_container_width=True, hide_index=True)

    pass_allow, rush_allow, recv_allow, league = build_defense_allowance_tables(wk)

    teams_in_upcoming = set(upcoming["home_team"].unique()).union(set(upcoming["away_team"].unique()))
    edited["team"] = edited["team"].astype(str)
    pool = edited[edited["team"].isin(teams_in_upcoming)].copy()

    game_rows = []
    for _, g in upcoming.iterrows():
        h, a = g["home_team"], g["away_team"]
        if pd.isna(h) or pd.isna(a): 
            continue
        for tm, opp in [(h, a), (a, h)]:
            subset = pool[pool["team"] == tm].copy()
            if subset.empty:
                continue
            subset["opponent"] = opp
            subset["week"] = g.get("week", np.nan)
            subset["start_time_utc"] = g.get("start_time_utc", pd.NaT)
            subset["game_type"] = g.get("game_type", "")
            game_rows.append(subset)

    if len(game_rows) == 0:
        st.info("No matching players for upcoming games. You can adjust teams/players in the table above.")
    else:
        candidates = pd.concat(game_rows, ignore_index=True)

        proj_rows = []
        for r in candidates.to_dict(orient="records"):
            proj = adjust_player_row(
                r, r.get("opponent"), pass_allow, rush_allow, recv_allow, league, weight=weight
            )
            if proj:
                out = {
                    "team": r.get("team"),
                    "opponent": r.get("opponent"),
                    "player_name": r.get("player_name"),
                    "position": r.get("position"),
                    "week": r.get("week"),
                    "start_time_utc": r.get("start_time_utc"),
                    "game_type": r.get("game_type"),
                    "attempts_pg": r.get("attempts_pg"),
                    "completions_pg": r.get("completions_pg"),
                    "passing_yards_pg": r.get("passing_yards_pg"),
                    "passing_tds_pg": r.get("passing_tds_pg"),
                    "carries_pg": r.get("carries_pg"),
                    "rushing_yards_pg": r.get("rushing_yards_pg"),
                    "rushing_tds_pg": r.get("rushing_tds_pg"),
                    "targets_pg": r.get("targets_pg"),
                    "receptions_pg": r.get("receptions_pg"),
                    "receiving_yards_pg": r.get("receiving_yards_pg"),
                    "receiving_tds_pg": r.get("receiving_tds_pg"),
                }
                out.update(proj)
                proj_rows.append(out)

        if len(proj_rows) == 0:
            st.info("No projections produced. Try increasing the opponent weight or lowering min games.")
        else:
            proj_df = pd.DataFrame(proj_rows)
            order = [
                "start_time_utc","week","game_type","team","opponent","player_name","position",
                "proj_pass_yards","proj_completions","proj_pass_tds",
                "proj_rush_yards","proj_attempts","proj_rush_tds",
                "proj_rec_yards","proj_receptions","proj_rec_tds",
                "passing_yards_pg","completions_pg","passing_tds_pg",
                "rushing_yards_pg","carries_pg","rushing_tds_pg",
                "receiving_yards_pg","receptions_pg","receiving_tds_pg",
                "targets_pg","attempts_pg"
            ]
            order = [c for c in order if c in proj_df.columns]
            proj_df = proj_df[order].sort_values(["start_time_utc","team","position"], na_position="last")

            st.subheader("Projected props")
            st.dataframe(proj_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download projections (CSV)",
                proj_df.to_csv(index=False).encode("utf-8"),
                file_name=f"player_props_{season}_upcoming.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption(
    "Sources: nfl_data_py (nflverse/nflfastR weekly stats & schedules). "
    "This MVP uses last year's baselines + opponent allowed per game. "
    "Enhance with role/injury news, snaps, air yards, pressure, pace, and market lines for calibration."
)
