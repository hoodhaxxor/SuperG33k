# props_app_sim.py
# NFL Player Prop Projections (QB/WR/RB) — MVP + Simulation & EV vs Sportsbook
# Adds: per-player variance from last season, normal simulations for Over prob, EV given odds

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from nfl_data_py import import_weekly_data, import_schedules

st.set_page_config(page_title="NFL Player Props — Sim & EV", layout="wide")

# ----------------------------
# Data loaders
# ----------------------------
@st.cache_data(show_spinner=False)
def load_last_season_weekly(last_season:int):
    wk = import_weekly_data([last_season])

    # Normalize columns
    ren = {
        "recent_team":"team",
        "team_abbr":"team",
        "opponent":"opponent_team",
        "opponent_abbr":"opponent_team",
        "player_display_name":"player_name",
        "player":"player_name",
        "pass_attempts":"attempts",
        "pass_completions":"completions",
        "rushing_attempts":"carries",
        "pos":"position",
        "nfl_player_id":"player_id",
        "gsis_id":"player_id",
        "pfr_player_id":"player_id",
    }
    for s,t in ren.items():
        if s in wk.columns and t not in wk.columns:
            wk = wk.rename(columns={s:t})

    needed = ["season","week","team","opponent_team","position","player_id","player_name",
              "attempts","completions","passing_yards","passing_tds",
              "carries","rushing_yards","rushing_tds",
              "targets","receptions","receiving_yards","receiving_tds"]
    for c in needed:
        if c not in wk.columns:
            wk[c] = 0.0 if c not in ["season","week","team","opponent_team","position","player_id","player_name"] else wk.get(c, None)

    # keep and coerce
    keep = [c for c in needed if c in wk.columns]
    wk = wk[keep].copy()
    num_cols = [c for c in keep if c not in ["season","week","team","opponent_team","position","player_id","player_name"]]
    for c in num_cols:
        wk[c] = pd.to_numeric(wk[c], errors="coerce").fillna(0.0)

    wk = wk[~wk["team"].isna()]
    return wk

@st.cache_data(show_spinner=False)
def load_schedule(season:int):
    sch = import_schedules([season])
    if "game_type" in sch.columns:
        sch = sch[sch["game_type"].isin(["REG","POST"])].copy()
    if "start_time" in sch.columns:
        sch["start_time_utc"] = pd.to_datetime(sch["start_time"], utc=True, errors="coerce")
    elif "gameday" in sch.columns:
        sch["start_time_utc"] = pd.to_datetime(sch["gameday"], utc=True, errors="coerce")
    else:
        sch["start_time_utc"] = pd.NaT
    if "home_team" not in sch.columns and "home_team_abbr" in sch.columns:
        sch = sch.rename(columns={"home_team_abbr":"home_team"})
    if "away_team" not in sch.columns and "away_team_abbr" in sch.columns:
        sch = sch.rename(columns={"away_team_abbr":"away_team"})
    return sch

# ----------------------------
# Aggregations
# ----------------------------
def per_game(df, group_cols, sum_cols):
    agg = df.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    games = df.groupby(group_cols, dropna=False).size().reset_index(name="games")
    out = agg.merge(games, on=group_cols, how="left")
    for c in sum_cols:
        out[c+"_pg"] = out[c] / out["games"].replace(0, np.nan)
    return out

def per_player_std(df, group_cols, stat_cols):
    stds = df.groupby(group_cols, dropna=False)[stat_cols].std(ddof=1).reset_index()
    # rename *_std
    stds = stds.rename(columns={c: f"{c}_std" for c in stat_cols})
    return stds

def select_likely_starters(baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["attempts_pg", "carries_pg", "targets_pg"]:
        if col not in baselines.columns:
            baselines[col] = 0.0

    for team, g in baselines.groupby("team", dropna=False):
        qb = g.loc[g["position"] == "QB"].sort_values("attempts_pg", ascending=False).head(1)
        rb = g.loc[g["position"].isin(["RB","FB"])].sort_values("carries_pg", ascending=False).head(1)
        wr = g.loc[g["position"] == "WR"].sort_values("targets_pg", ascending=False).head(2)
        picks = pd.concat([qb, rb, wr], axis=0)
        if not picks.empty:
            rows.append(picks)

    if not rows:
        return baselines.head(0)
    out = pd.concat(rows, ignore_index=True)
    if "player_id" in out.columns:
        out = out.drop_duplicates(subset=["team","player_id"], keep="first")
    else:
        out = out.drop_duplicates(subset=["team","player_name","position"], keep="first")
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
        if pd.isna(base): return np.nan
        if league_avg is None or league_avg == 0 or pd.isna(opp_val): return base
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
# Simulation helpers
# ----------------------------
def american_to_prob(odds:int) -> float:
    """Convert American odds to implied probability (no vig removed)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def american_payout(odds:int, stake:float=100.0) -> float:
    """Gross profit (not including returned stake) for a winning bet with American odds."""
    if odds > 0:
        return stake * (odds / 100.0)
    else:
        return stake * (100.0 / -odds)

def simulate_over_prob(mean: float, std: float, line: float, draws:int=20000, clip_zero:bool=True) -> float:
    """Monte Carlo with normal approximation; clip at zero for yards/attempts."""
    if std is None or np.isnan(std) or std <= 0:
        # fallback: small variance to avoid 50/50
        std = max(1e-6, 0.25 * max(1.0, mean))
    samples = np.random.normal(loc=mean, scale=std, size=draws)
    if clip_zero:
        samples = np.clip(samples, 0, None)
    return float(np.mean(samples > line))

# ----------------------------
# UI
# ----------------------------
st.title("🏈 Player Props — Projections + Simulated Edge")
st.caption("Projections from last-season usage + opponent adjustment, PLUS simulation for Over/Under & EV vs sportsbook.")

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
    st.subheader("Simulation defaults")
    draws = st.number_input("Monte Carlo draws", min_value=2000, max_value=100000, value=20000, step=1000)

with st.spinner("Loading last season player-week data..."):
    wk = load_last_season_weekly(last_season)

sum_cols = [c for c in [
    "attempts","completions","passing_yards","passing_tds",
    "carries","rushing_yards","rushing_tds",
    "targets","receptions","receiving_yards","receiving_tds"
] if c in wk.columns]

# Per-game means (baselines)
player_pg = per_game(wk, ["team","player_id","player_name","position"], sum_cols)
player_pg = player_pg[player_pg["games"] >= min_games].copy()

# Per-player STD from weekly samples (for simulation)
std_cols = [c for c in ["passing_yards","completions","rushing_yards","carries","receiving_yards","receptions"] if c in wk.columns]
player_std = per_player_std(wk, ["team","player_id","player_name","position"], std_cols)

# Merge mean + std
baselines = player_pg.merge(player_std, on=["team","player_id","player_name","position"], how="left")

# Likely starters
likely = select_likely_starters(baselines)
if likely.empty:
    st.warning("Could not identify likely starters from last season. Try lowering 'Min games for baseline'.")
    st.stop()

st.subheader("Likely starters from last season (editable)")
editable_cols = [c for c in likely.columns if c in ["team","player_name","position"] or c.endswith("_pg") or c.endswith("_std") or c=="games"]
edited = st.data_editor(likely[editable_cols].reset_index(drop=True), use_container_width=True, num_rows="dynamic")

with st.spinner("Loading current season schedule..."):
    sched = load_schedule(season)

now_utc = datetime.now(timezone.utc)
sched["start_time_utc"] = pd.to_datetime(sched["start_time_utc"], utc=True, errors="coerce")
upcoming = sched[sched["start_time_utc"] >= now_utc].copy().sort_values("start_time_utc")

st.subheader(f"Upcoming games ({len(upcoming)})")
show_cols = [c for c in ["week","game_type","away_team","home_team","start_time_utc"] if c in upcoming.columns]
st.dataframe(upcoming[show_cols], use_container_width=True, hide_index=True)

# Opponent allowances
pass_allow, rush_allow, recv_allow, league = build_defense_allowance_tables(wk)

teams_in_upcoming = set(upcoming["home_team"].unique()).union(set(upcoming["away_team"].unique()))
edited["team"] = edited["team"].astype(str)
pool = edited[edited["team"].isin(teams_in_upcoming)].copy()

# Build per-game projections
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
    st.info("No matching players for upcoming games. Adjust starters if needed.")
    st.stop()

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
            # baselines
            "passing_yards_pg": r.get("passing_yards_pg"),
            "completions_pg": r.get("completions_pg"),
            "rushing_yards_pg": r.get("rushing_yards_pg"),
            "carries_pg": r.get("carries_pg"),
            "receiving_yards_pg": r.get("receiving_yards_pg"),
            "receptions_pg": r.get("receptions_pg"),
            # stds
            "passing_yards_std": r.get("passing_yards_std"),
            "completions_std": r.get("completions_std"),
            "rushing_yards_std": r.get("rushing_yards_std"),
            "carries_std": r.get("carries_std"),
            "receiving_yards_std": r.get("receiving_yards_std"),
            "receptions_std": r.get("receptions_std"),
        }
        out.update(proj)
        proj_rows.append(out)

proj_df = pd.DataFrame(proj_rows)
order = [
    "start_time_utc","week","game_type","team","opponent","player_name","position",
    "proj_pass_yards","proj_completions","proj_pass_tds",
    "proj_rush_yards","proj_attempts","proj_rush_tds",
    "proj_rec_yards","proj_receptions","proj_rec_tds",
    "passing_yards_pg","completions_pg","rushing_yards_pg","carries_pg","receiving_yards_pg","receptions_pg",
    "passing_yards_std","completions_std","rushing_yards_std","carries_std","receiving_yards_std","receptions_std",
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

# ----------------------------
# Betting Helper: pick a player & stat, enter sportsbook line/odds → get Over % and EV
# ----------------------------
st.markdown("### 📈 Betting Helper — Over/Under probability & EV")
if proj_df.empty:
    st.info("No projections available.")
    st.stop()

# Player picker
players = proj_df["player_name"].dropna().unique().tolist()
player_pick = st.selectbox("Player", players, index=0)

row = proj_df[proj_df["player_name"] == player_pick].iloc[0]

# Stat options based on position & available projections
options = []
if not pd.isna(row.get("proj_pass_yards")): options.append(("Passing Yards","proj_pass_yards","passing_yards_std"))
if not pd.isna(row.get("proj_completions")): options.append(("Completions","proj_completions","completions_std"))
if not pd.isna(row.get("proj_rush_yards")): options.append(("Rushing Yards","proj_rush_yards","rushing_yards_std"))
if not pd.isna(row.get("proj_attempts")): options.append(("Rushing Attempts","proj_attempts","carries_std"))
if not pd.isna(row.get("proj_rec_yards")): options.append(("Receiving Yards","proj_rec_yards","receiving_yards_std"))
if not pd.isna(row.get("proj_receptions")): options.append(("Receptions","proj_receptions","receptions_std"))

label_to_cols = {lab:(mean_col,std_col) for (lab,mean_col,std_col) in options}
stat_label = st.selectbox("Stat", list(label_to_cols.keys()), index=0)
mean_col, std_baseline_col = label_to_cols[stat_label]

# Sportsbook inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    line = st.number_input("Sportsbook line", min_value=0.0, value=float(row[mean_col]) if not pd.isna(row[mean_col]) else 50.0, step=0.5)
with col2:
    odds_over = st.number_input("Over odds (American)", value=-110, step=5, format="%d")
with col3:
    odds_under = st.number_input("Under odds (American)", value=-110, step=5, format="%d")
with col4:
    stake = st.number_input("Stake ($)", min_value=1.0, value=100.0, step=1.0)

# Determine std to use: scale last-season std by ratio of projected mean to last-season mean
baseline_mean_col = mean_col.replace("proj_", "").replace("attempts","carries_pg")  # rough mapping
if baseline_mean_col in row.index:
    baseline_mean = row[baseline_mean_col]
else:
    # attempt smarter mapping
    baseline_map = {
        "proj_pass_yards":"passing_yards_pg",
        "proj_completions":"completions_pg",
        "proj_rush_yards":"rushing_yards_pg",
        "proj_attempts":"carries_pg",
        "proj_rec_yards":"receiving_yards_pg",
        "proj_receptions":"receptions_pg",
    }
    baseline_mean = row.get(baseline_map.get(mean_col, ""), np.nan)

std_baseline = row.get(std_baseline_col, np.nan)
mean_proj = float(row[mean_col]) if not pd.isna(row[mean_col]) else np.nan

if pd.isna(mean_proj):
    st.warning("No projection available for this stat/player.")
else:
    # scale std: if baseline mean is 0 or nan, just use baseline std
    if pd.isna(std_baseline) or std_baseline <= 0:
        std_use = max(1e-6, 0.25 * max(1.0, mean_proj))
    else:
        scale = 1.0 if (pd.isna(baseline_mean) or baseline_mean <= 0) else float(mean_proj) / float(baseline_mean)
        std_use = float(std_baseline) * scale

    p_over = simulate_over_prob(mean_proj, std_use, line, draws=int(draws), clip_zero=True)
    p_under = 1.0 - p_over

    # EV
    brk_over = american_to_prob(int(odds_over))
    brk_under = american_to_prob(int(odds_under))
    win_pay_over = american_payout(int(odds_over), stake)
    win_pay_under = american_payout(int(odds_under), stake)
    ev_over = p_over * win_pay_over - (1 - p_over) * stake
    ev_under = p_under * win_pay_under - (1 - p_under) * stake

    st.write(f"**Projected mean:** {mean_proj:.2f} | **Std used:** {std_use:.2f}")
    st.write(f"**P(Over {line}):** {p_over:.3f} | **P(Under):** {p_under:.3f}")

    colA, colB = st.columns(2)
    with colA:
        st.metric(label=f"EV Over ({odds_over})", value=f"${ev_over:.2f}")
    with colB:
        st.metric(label=f"EV Under ({odds_under})", value=f"${ev_under:.2f}")

    st.caption("Normal approximation with zero floor. For more precision, add current-season rolling variance or bootstrap from weekly samples.")
