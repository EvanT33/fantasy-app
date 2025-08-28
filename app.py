import io
import os
import re
from typing import List, Optional, Tuple, Dict

import pandas as pd
import pulp
import streamlit as st
from rapidfuzz import process, fuzz  # (not required for core flow; kept for future fuzzy features)

# ================== Helpers ==================

SUFFIXES = (" jr", " sr", " ii", " iii", " iv", " v")

def to_money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def ceil_dollars(x: Optional[float]) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    return int(-(-float(x) // 1))

def normalize_pos(pos: str) -> str:
    pos = (pos or "").upper().strip()
    pos = pos.replace("D/ST", "DST").replace("DEF", "DST").replace("PK", "K")
    if "/" in pos: pos = pos.split("/")[0]
    if "-" in pos: pos = pos.split("-")[0]
    pos = re.sub(r"\d+$", "", pos).strip()
    return pos

def require_columns(df: pd.DataFrame, cols: list, context: str = "input CSV"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {context}: {missing}\nAvailable columns: {list(df.columns)}")

def build_working_table(df: pd.DataFrame, w_consensus: float, w_ds: float, w_market: float, w_dsa: float) -> pd.DataFrame:
    require_columns(df, ["Player", "Fantasy Position"], "values CSV")

    work = pd.DataFrame({
        "Name": df["Player"].astype(str),
        "Team": df["Team"].astype(str) if "Team" in df.columns else "",
        "Pos": df["Fantasy Position"].astype(str).apply(normalize_pos),
    })

    has_consensus = "Consensus Proj" in df.columns
    has_ds = "DS Proj" in df.columns
    if not (has_consensus or has_ds):
        raise ValueError("CSV must include 'Consensus Proj' and/or 'DS Proj'")

    work["Consensus_Pts"] = pd.to_numeric(df["Consensus Proj"], errors="coerce") if has_consensus else pd.NA
    work["DS_Pts"] = pd.to_numeric(df["DS Proj"], errors="coerce") if has_ds else pd.NA

    def blend_points(row):
        vals, wts = [], []
        if pd.notna(row["Consensus_Pts"]): vals.append(row["Consensus_Pts"]); wts.append(w_consensus)
        if pd.notna(row["DS_Pts"]):       vals.append(row["DS_Pts"]);       wts.append(w_ds)
        if not vals: return float("nan")
        wsum = sum(wts) if sum(wts) > 0 else len(vals)
        if wsum == 0: wts = [1.0]*len(vals); wsum = len(vals)
        return float(sum(v * (wts[i]/wsum) for i, v in enumerate(vals)))
    work["Points"] = work.apply(blend_points, axis=1)

    has_mkt = "AuctionMarketValue" in df.columns
    has_dsa = "DS AuctionValue" in df.columns
    if not (has_mkt or has_dsa):
        raise ValueError("CSV must include 'AuctionMarketValue' and/or 'DS AuctionValue'")

    work["MarketValue"] = to_money(df["AuctionMarketValue"]) if has_mkt else pd.NA
    work["DSValue"] = to_money(df["DS AuctionValue"]) if has_dsa else pd.NA

    def blend_cost(row):
        vals, wts = [], []
        if pd.notna(row["MarketValue"]): vals.append(row["MarketValue"]); wts.append(w_market)
        if pd.notna(row["DSValue"]):     vals.append(row["DSValue"]);     wts.append(w_dsa)
        if not vals: return float("nan")
        wsum = sum(wts) if sum(wts) > 0 else len(vals)
        if wsum == 0: wts = [1.0]*len(vals); wsum = len(vals)
        return float(sum(v * (wts[i]/wsum) for i, v in enumerate(vals)))
    work["Cost"] = work.apply(blend_cost, axis=1).apply(ceil_dollars)
    return work

# ================== Optimization (Starters only, no DST) ==================

def build_and_solve_lp(players: pd.DataFrame,
                       budget: float,
                       locked_starters: pd.DataFrame,
                       slots: Dict[str, int]) -> Tuple[pd.DataFrame, float, float]:
    """
    Optimize starters under 'budget' given some players are pre-locked as starters.
    slots: {'QB','RB','WR','TE','K','FLEX'} counts (no DST here).
    """
    qb_slots = int(slots.get("QB", 1))
    rb_slots = int(slots.get("RB", 2))
    wr_slots = int(slots.get("WR", 2))
    te_slots = int(slots.get("TE", 1))
    k_slots  = int(slots.get("K", 1))
    flex_slots = int(slots.get("FLEX", 2))

    total_slots = qb_slots + rb_slots + wr_slots + te_slots + k_slots + flex_slots

    # Validate locks don't exceed slots
    lc = locked_starters["Pos"].value_counts().to_dict() if not locked_starters.empty else {}
    if lc.get("QB", 0) > qb_slots: raise ValueError(f"Locked QB ({lc.get('QB')}) exceeds slots ({qb_slots}).")
    if lc.get("K", 0)  > k_slots:  raise ValueError(f"Locked K ({lc.get('K')}) exceeds slots ({k_slots}).")

    total_needed = total_slots - len(locked_starters)
    if total_needed < 0:
        raise ValueError(f"Too many locked starters (> {total_slots}).")

    rb_l = lc.get("RB", 0)
    wr_l = lc.get("WR", 0)
    te_l = lc.get("TE", 0)

    need_qb = max(0, qb_slots - lc.get("QB", 0))
    need_k  = max(0, k_slots  - lc.get("K",  0))
    need_te_min = max(0, te_slots - te_l)  # TE can also fill FLEX
    min_rb  = max(0, rb_slots - rb_l)
    min_wr  = max(0, wr_slots - wr_l)
    rwt_total = rb_slots + wr_slots + te_slots + flex_slots
    need_rwt_total = max(0, rwt_total - (rb_l + wr_l + te_l))

    cost_locked = float(locked_starters["Cost"].sum()) if not locked_starters.empty else 0.0
    points_locked = float(locked_starters["Points"].sum()) if not locked_starters.empty else 0.0
    budget_left = budget - cost_locked
    if budget_left < 0:
        raise ValueError(f"Locked starters cost ${cost_locked:.0f} exceeds working starters budget ${budget:.0f}.")

    df = players.copy()
    df["Pos"] = df["Pos"].astype(str).str.upper().replace({"D/ST": "DST", "DEF": "DST", "PK": "K"})
    df = df[df["Pos"].isin(["QB","RB","WR","TE","K"])].reset_index(drop=True)

    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce").apply(ceil_dollars)
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
    df = df.dropna(subset=["Cost","Points"])
    df = df[df["Cost"] >= 0].reset_index(drop=True)

    if df.empty and total_needed > 0:
        raise ValueError("No candidates left for starters after locks/exclusions.")

    idx = list(df.index)
    x = pulp.LpVariable.dicts("pick", idx, lowBound=0, upBound=1, cat=pulp.LpBinary)

    prob = pulp.LpProblem("AuctionRoster_Starters_NoDST", pulp.LpMaximize)
    prob += (pulp.lpSum(df.loc[i,"Points"] * x[i] for i in idx) + points_locked)

    prob += pulp.lpSum(df.loc[i,"Cost"] * x[i] for i in idx) <= budget_left
    prob += pulp.lpSum(x[i] for i in idx) == total_needed

    # Exact counts for QB & K
    prob += pulp.lpSum(x[i] for i in idx if df.loc[i,"Pos"]=="QB") == need_qb
    prob += pulp.lpSum(x[i] for i in idx if df.loc[i,"Pos"]=="K")  == need_k

    # Minimum base counts; FLEX filled by R/W/T
    if need_te_min > 0: prob += pulp.lpSum(x[i] for i in idx if df.loc[i,"Pos"]=="TE") >= need_te_min
    if min_rb > 0:      prob += pulp.lpSum(x[i] for i in idx if df.loc[i,"Pos"]=="RB") >= min_rb
    if min_wr > 0:      prob += pulp.lpSum(x[i] for i in idx if df.loc[i,"Pos"]=="WR") >= min_wr

    # Total R/W/T needed (includes base + flex)
    prob += pulp.lpSum(x[i] for i in idx if df.loc[i,"Pos"] in ("RB","WR","TE")) == need_rwt_total

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] not in ("Optimal","Feasible"):
        raise ValueError(f"Solver status: {pulp.LpStatus[status]}")

    chosen_idx = [i for i in idx if x[i].value() == 1]
    chosen = df.loc[chosen_idx].copy()
    starters = pd.concat([locked_starters, chosen], ignore_index=True)

    # Validate final counts
    counts = starters["Pos"].value_counts().to_dict()
    def c(p): return int(counts.get(p, 0))
    rwt_count = c("RB") + c("WR") + c("TE")
    if not (c("QB")==qb_slots and c("K")==k_slots and rwt_count==rwt_total and len(starters)==total_slots):
        raise ValueError("Starter counts violated. Adjust slots or overrides and try again.")

    total_points = float(starters["Points"].sum())
    total_cost = float(starters["Cost"].sum())
    return starters.sort_values(["Pos","Points"], ascending=[True,False]), total_points, total_cost

# ================== DST + Bench ==================

def pick_dst_and_bench(work_no_dst: pd.DataFrame,
                       dst_pool: pd.DataFrame,
                       starters: pd.DataFrame,
                       bench_locked_df: pd.DataFrame,
                       reserve: int,
                       bench_slots: int) -> Tuple[Optional[Dict], List[Dict], int]:
    remaining_reserve = reserve
    dst_choice = None

    if not bench_locked_df.empty and (bench_locked_df["Pos"]=="DST").any():
        dst_locked_df = bench_locked_df[bench_locked_df["Pos"]=="DST"].sort_values("Cost")
        dst_choice = dst_locked_df.iloc[0][["Name","Team","Pos","Points","Cost"]].copy()
        remaining_reserve -= int(ceil_dollars(dst_choice["Cost"]))
        bench_locked_df = bench_locked_df[bench_locked_df["Pos"]!="DST"]
    else:
        if not dst_pool.empty and remaining_reserve > 0:
            cheapest_dst = dst_pool.sort_values("Cost").iloc[0][["Name","Team","Pos","Points","Cost"]].copy()
            cheapest_dst["Cost"] = ceil_dollars(cheapest_dst["Cost"])
            if remaining_reserve - int(cheapest_dst["Cost"]) >= 0:
                dst_choice = cheapest_dst
                remaining_reserve -= int(cheapest_dst["Cost"])

    bench_used = set(n.lower() for n in starters["Name"])
    if not bench_locked_df.empty:
        bench_used.update(n.lower() for n in bench_locked_df["Name"])

    bench_list = bench_locked_df[["Name","Team","Pos","Points","Cost"]].to_dict("records") if not bench_locked_df.empty else []

    bench_pool = work_no_dst.copy()
    bench_pool = bench_pool[~bench_pool["Name"].str.lower().isin(bench_used)]
    bench_pool = bench_pool[bench_pool["Pos"].isin(["QB","RB","WR","TE","K"])]
    bench_pool["Cost"] = bench_pool["Cost"].apply(ceil_dollars)

    # prefer filling RB/WR/TE first, then QB, then K (no QB cap now)
    pos_bucket = {"RB":0,"WR":0,"TE":0,"QB":1,"K":2}
    bench_pool = bench_pool.sort_values(by=["Pos","Cost"],
                                        key=lambda s: s.map(pos_bucket) if s.name=="Pos" else s)

    bench_spots_left = max(0, int(bench_slots) - len(bench_list))

    for _, r in bench_pool.iterrows():
        if bench_spots_left <= 0 or remaining_reserve <= 0: break
        cost_i = int(r["Cost"])
        if remaining_reserve - cost_i < 0: continue
        bench_list.append({"Name": r["Name"], "Team": r["Team"], "Pos": r["Pos"], "Points": r["Points"], "Cost": cost_i})
        remaining_reserve -= cost_i
        bench_spots_left -= 1

    return (None if dst_choice is None else dict(dst_choice)), bench_list, remaining_reserve

# ================== Streamlit UI ==================

st.set_page_config(page_title="Auction Optimizer (Live Draft)", layout="wide")
st.title("🏈 Live Auction Optimizer")
st.caption("Projections are treated as **Half-PPR**, based on the CSV columns (Consensus/DS).")

# --- Auto-load CSV from your app folder ---
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "auction-values-half-ppr.csv")
uploaded = DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else None

# session state for work table + overrides + taken (leaguemates' picks)
if "work" not in st.session_state:
    st.session_state.work = None
if "overrides_df" not in st.session_state:
    st.session_state.overrides_df = pd.DataFrame(columns=["Name","Cost","Starter"])
if "taken" not in st.session_state:
    st.session_state.taken = []  # list of player names marked as drafted by others

with st.sidebar:
    st.header("Settings")
    budget = st.number_input("Total budget", min_value=1, value=200, step=1)
    reserve = st.number_input("Reserve for DST + Bench", min_value=0, value=5, step=1)

    st.subheader("Starter slots (no DST)")
    qb_slots = st.number_input("QB", min_value=0, value=1, step=1)
    rb_slots = st.number_input("RB", min_value=0, value=2, step=1)
    wr_slots = st.number_input("WR", min_value=0, value=2, step=1)
    te_slots = st.number_input("TE", min_value=0, value=1, step=1)
    k_slots  = st.number_input("K",  min_value=0, value=1, step=1)
    flex_slots = st.number_input("FLEX (RB/WR/TE)", min_value=0, value=2, step=1)

    st.subheader("Bench")
    bench_slots = st.number_input("Bench spots (non-DST)", min_value=0, value=4, step=1)

    st.subheader("Projection & Cost Blends")
    w_consensus = st.number_input("Weight: Consensus Proj", min_value=0.0, value=1.0, step=0.1)
    w_ds = st.number_input("Weight: DS Proj", min_value=0.0, value=1.0, step=0.1)
    w_market = st.number_input("Weight: Auction Market Value", min_value=0.0, value=0.5, step=0.1)
    w_dsa = st.number_input("Weight: DS Auction Value", min_value=0.0, value=0.5, step=0.1)

    st.subheader("Alternatives")
    show_alts = st.checkbox("Show alternatives", value=True)
    alts_min_points = st.number_input("Min projected points", min_value=0.0, value=170.0, step=5.0)
    alts_per_pos = st.number_input("Alternatives per position", min_value=1, value=12, step=1)

def ensure_overrides_df(names_list: List[str]):
    df = st.session_state.overrides_df.copy()
    df = df[df["Name"].isin(names_list)]
    if "Starter" in df.columns:
        df["Starter"] = df["Starter"].astype(bool)
    st.session_state.overrides_df = df

def add_override_row(name: str, cost: int, starter: bool):
    df = st.session_state.overrides_df.copy()
    if name in set(df["Name"]):
        df.loc[df["Name"] == name, ["Cost","Starter"]] = [int(ceil_dollars(cost)), bool(starter)]
    else:
        df = pd.concat([df, pd.DataFrame([{"Name": name, "Cost": int(ceil_dollars(cost)), "Starter": bool(starter)}])], ignore_index=True)
    st.session_state.overrides_df = df

# UI once CSV is available
if uploaded is not None:
    try:
        df_values = pd.read_csv(uploaded) if isinstance(uploaded, str) else pd.read_csv(uploaded)
        work = build_working_table(df_values, w_consensus, w_ds, w_market, w_dsa)
        st.session_state.work = work

        # Build options for search dropdowns
        options_df = work.copy()
        options_df["Label"] = options_df.apply(lambda r: f"{r['Name']} ({r['Pos']}, {r['Team']})".strip(), axis=1)
        name_by_label = dict(zip(options_df["Label"], options_df["Name"]))
        labels_sorted = sorted(options_df["Label"].tolist())

        # ===== Top row: right box shows taken players =====
        top_left, top_right = st.columns([3,1])
        with top_left:
            st.subheader("Add a purchase (live override)")
            colp1, colp2, colp3, colp4 = st.columns([3,1,1,1])
            with colp1:
                pick_label = st.selectbox("Search player", options=[""] + labels_sorted, index=0, placeholder="Type to search…")
            with colp2:
                paid = st.number_input("Price paid ($)", min_value=0, value=0, step=1, key="price_paid")
            with colp3:
                starter_flag = st.toggle("Starter?", value=True, key="starter_toggle")
            with colp4:
                if st.button("Add / Update", key="add_update_purchase"):
                    if pick_label and pick_label in name_by_label and paid > 0:
                        add_override_row(name_by_label[pick_label], int(paid), bool(starter_flag))
                    else:
                        st.warning("Pick a player and enter a positive price.")

            # Editable overrides table
            st.markdown("#### Current purchases")
            ensure_overrides_df(work["Name"].tolist())

            overrides_editor = st.data_editor(
                st.session_state.overrides_df,
                num_rows="dynamic",
                use_container_width=True,
                key="overrides_editor",
                column_config={
                    "Name": st.column_config.SelectboxColumn(
                        "Name",
                        options=sorted(work["Name"].unique().tolist()),
                        required=True,
                        width="medium"
                    ),
                    "Cost": st.column_config.NumberColumn("Cost ($)", min_value=0, step=1, required=True),
                    "Starter": st.column_config.CheckboxColumn("Starter?", default=True)
                }
            )
            st.session_state.overrides_df = overrides_editor

        with top_right:
            st.subheader("Taken")
            if st.session_state.taken:
                taken_df = pd.DataFrame({"Player": st.session_state.taken})
                st.dataframe(taken_df, hide_index=True, use_container_width=True, height=240)
            else:
                st.caption("No opponents’ picks logged yet.")

        # ===== Log leaguemates' picks (exclusions) =====
        st.markdown("---")
        st.subheader("Leaguemates’ Picks (exclude from pool)")
        colx1, colx2, colx3 = st.columns([3,1,1])
        with colx1:
            taken_label = st.selectbox("Search player to mark as TAKEN", options=[""] + labels_sorted, index=0, placeholder="Type to search…", key="taken_select")
        with colx2:
            if st.button("Mark TAKEN", key="mark_taken"):
                if taken_label and taken_label in name_by_label:
                    pname = name_by_label[taken_label]
                    if pname not in st.session_state.taken:
                        st.session_state.taken.append(pname)
                else:
                    st.warning("Pick a player to mark as taken.")
        with colx3:
            if st.button("Clear TAKEN list", type="secondary", key="clear_taken"):
                st.session_state.taken = []

        # ===== Action buttons =====
        col_run, col_clear = st.columns([1,1])
        with col_run:
            run_btn = st.button("⚡ Optimize", type="primary")
        with col_clear:
            if st.button("Clear purchases", type="secondary"):
                st.session_state.overrides_df = pd.DataFrame(columns=["Name","Cost","Starter"])

        # ======== RUN OPTIMIZER ========
        if run_btn:
            work = st.session_state.work.copy()

            # Determine taken players that are NOT your purchases (yours override exclusions)
            my_purchases = set(st.session_state.overrides_df["Name"].tolist())
            taken_not_mine = [n for n in st.session_state.taken if n not in my_purchases]

            # Separate DST pool, excluding taken_not_mine
            dst_pool = work[(work["Pos"] == "DST") & (~work["Name"].isin(taken_not_mine))].copy()
            work_no_dst = work[(work["Pos"] != "DST") & (~work["Name"].isin(taken_not_mine))].copy()

            # Convert overrides into locks
            ov = st.session_state.overrides_df.copy()
            ov["Cost"] = ov["Cost"].apply(ceil_dollars).fillna(0).astype(int)
            ov["Starter"] = ov["Starter"].fillna(True).astype(bool)

            locked_starters = (
                ov[ov["Starter"]]
                .merge(work_no_dst, on="Name", how="inner", suffixes=("", "_w"))
                [["Name", "Team", "Pos", "Points", "Cost"]]
            )

            bench_locked = (
                ov[~ov["Starter"]]
                .merge(pd.concat([work_no_dst, dst_pool], ignore_index=True), on="Name", how="inner", suffixes=("", "_w"))
                [["Name", "Team", "Pos", "Points", "Cost"]]
            )
            bench_cost_locked = int(bench_locked["Cost"].sum()) if not bench_locked.empty else 0

            # Adjust starters budget if bench spend > reserve
            working_budget_base = int(budget) - int(reserve)
            overage = max(0, bench_cost_locked - int(reserve))
            effective_working_budget = working_budget_base - overage
            if overage > 0:
                st.warning(f"Bench purchases ${bench_cost_locked} > reserve ${int(reserve)} by ${overage}. "
                           f"Reducing starters’ working budget to ${effective_working_budget} to stay under ${int(budget)} total.")
            if effective_working_budget <= 0:
                st.error("No budget left for starters after bench purchases. Lower bench spend or increase reserve.")
            else:
                # Solve starters with configurable slots
                slots = {"QB": qb_slots, "RB": rb_slots, "WR": wr_slots, "TE": te_slots, "K": k_slots, "FLEX": flex_slots}

                starters_pool = work_no_dst[~work_no_dst["Name"].isin(locked_starters["Name"])]
                starters, starters_pts, starters_cost = build_and_solve_lp(
                    starters_pool[["Name","Team","Pos","Points","Cost"]],
                    effective_working_budget,
                    locked_starters[["Name","Team","Pos","Points","Cost"]],
                    slots
                )

                # DST + Bench from remaining reserve
                remaining_for_bench = max(0, int(reserve) - bench_cost_locked)
                dst_choice, bench_list, reserve_left = pick_dst_and_bench(
                    work_no_dst.copy(), dst_pool.copy(), starters, bench_locked.copy(),
                    remaining_for_bench, int(bench_slots)
                )

                # Order starters nicely
                order = {"QB":0,"RB":1,"WR":2,"TE":3,"K":4}
                starters_disp = starters.sort_values(by=["Pos","Points"],
                                                     ascending=[True,False],
                                                     key=lambda s: s.map(order).fillna(99) if s.name=="Pos" else s).reset_index(drop=True)

                total_slots = int(qb_slots + rb_slots + wr_slots + te_slots + k_slots + flex_slots)

                col1, col2 = st.columns([2,1])
                with col1:
                    st.subheader(f"Optimized Starters ({total_slots} slots • Pts: {starters_pts:.1f} | Cost: ${int(starters_cost)})")
                    st.dataframe(starters_disp[["Pos","Name","Team","Points","Cost"]], hide_index=True, use_container_width=True)
                with col2:
                    st.metric("Starters Working Budget", f"${int(effective_working_budget)}")
                    st.metric("Bench/DST spent", f"${int(int(reserve) - reserve_left)}")
                    st.metric("Reserve remaining", f"${int(reserve_left)}")

                st.subheader("DST (from Reserve)")
                if dst_choice is None:
                    st.write("No DST chosen (no data or insufficient reserve).")
                else:
                    st.write(f"**{dst_choice['Name']}** — ${int(ceil_dollars(dst_choice['Cost']))}")

                st.subheader(f"Bench ({int(bench_slots)} spots; not points-optimized)")
                bench_df = pd.DataFrame(bench_list)
                if not bench_df.empty:
                    st.dataframe(bench_df[["Name","Team","Pos","Cost"]], hide_index=True, use_container_width=True)
                else:
                    st.write("No bench players selected.")

                # Alternatives
                if show_alts:
                    chosen_names = set(n.lower() for n in starters_disp["Name"])
                    if dst_choice is not None:
                        chosen_names.add(str(dst_choice["Name"]).lower())
                    if not bench_df.empty:
                        chosen_names.update(n.lower() for n in bench_df["Name"])

                    remaining_pool = work_no_dst.copy()
                    remaining_pool = remaining_pool[~remaining_pool["Name"].str.lower().isin(chosen_names)]
                    remaining_pool = remaining_pool[remaining_pool["Pos"].isin(["QB","RB","WR","TE","K"])]
                    remaining_pool["Cost"] = remaining_pool["Cost"].apply(ceil_dollars)
                    remaining_pool = remaining_pool.dropna(subset=["Points","Cost"])
                    remaining_pool = remaining_pool[(remaining_pool["Cost"] > 0) & (remaining_pool["Points"] >= float(alts_min_points))]

                    st.subheader(f"Value Alternatives (Proj ≥ {int(alts_min_points)}; ranked by Points per $)")
                    if not remaining_pool.empty:
                        remaining_pool["ValueScore"] = remaining_pool["Points"] / remaining_pool["Cost"]
                        for pos in ["QB","RB","WR","TE","K"]:
                            pos_df = remaining_pool[remaining_pool["Pos"]==pos].copy()
                            if pos_df.empty: continue
                            pos_df = pos_df.sort_values(["ValueScore","Points"], ascending=[False,False]).head(int(alts_per_pos))
                            st.markdown(f"**{pos}**")
                            st.dataframe(pos_df[["Name","Team","Points","Cost","ValueScore"]], hide_index=True, use_container_width=True)
                    else:
                        st.write("No candidates met the filter.")

                # Download CSV
                out_rows = []
                for _, r in starters_disp.iterrows():
                    out_rows.append({"Section":"STARTER", **r[["Name","Team","Pos","Points","Cost"]].to_dict()})
                if dst_choice is not None:
                    out_rows.append({"Section":"DST", "Name": dst_choice["Name"], "Team": dst_choice.get("Team",""),
                                     "Pos":"DST","Points":"", "Cost": int(ceil_dollars(dst_choice["Cost"]))})
                for rec in bench_list:
                    out_rows.append({"Section":"BENCH", **rec})
                out_df = pd.DataFrame(out_rows)
                buf = io.BytesIO(); out_df.to_csv(buf, index=False)
                st.download_button("Download roster CSV", data=buf.getvalue(),
                                   file_name="optimized_roster.csv", mime="text/csv")
    except Exception as e:
        st.exception(e)
else:
    st.error("Couldn't find 'auction-values-half-ppr.csv' next to app.py. Place the file there and refresh.")
