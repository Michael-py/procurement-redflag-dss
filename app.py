"""Decision-support dashboard for the Procurement Red-Flag DSS.

    streamlit run app.py

Reads the SQLite warehouse produced by `python -m redflag run` in read-only
mode. Nothing here writes to the warehouse; weight changes re-rank in memory.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing

import pandas as pd
import streamlit as st

from redflag import DISCLAIMER, __version__
from redflag.config import INDICATORS, load_config
from redflag.indicators import INDICATOR_LABELS, METHOD_LABELS
from redflag.prioritise import score_contracts

st.set_page_config(page_title="Procurement Red-Flag DSS", page_icon=None, layout="wide")

CFG = load_config()
DB = CFG.resolve("warehouse")
SOURCE_LABELS = {s["id"]: s["label"] for s in CFG.sources}
STATUS_LABELS = {"flagged": "Flagged", "clear": "Clear", "not_applicable": "Not applicable", "non_computable": "Not computable"}
DQ_LABELS = {
    "buyer_identifier": "Buyer identifier", "procurement_method": "Procurement method", "number_of_tenderers": "Number of tenderers",
    "tender_period": "Tender period", "tender_estimate": "Tender estimate", "award_date": "Award date", "award_value": "Award value",
    "supplier_identifier": "Supplier identifier", "supplier_name": "Supplier name", "tender_status": "Tender status",
    "award_status": "Award status", "tender_period_duration": "Tender-period duration matches dates",
    "method_distinguishes_competition": "Method separates competitive from non-competitive",
    "buyer_identifier_usable": "Buyer identity usable", "supplier_identity_usable": "Supplier identity usable",
    "estimate_independent_of_award": "Estimate independent of award", "tenderers_not_defaulted": "Tenderer count not a default value",
}


USE_POSTGRES = os.environ.get("REDFLAG_DB", "sqlite").lower() == "postgres"


@st.cache_data(show_spinner=False)
def table(sql: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache when the warehouse is rebuilt
    if USE_POSTGRES:  # read-only login; see docker-compose.yml and redflag/pgload.py
        import psycopg

        from redflag.pgload import conninfo

        with psycopg.connect(conninfo(CFG.root, reader=True)) as con, con.cursor() as cur:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    with closing(sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)) as con:
        return pd.read_sql_query(sql, con)


@st.cache_data(show_spinner="Re-ranking contracts...")
def rescore(weights: tuple, mtime: float) -> pd.DataFrame:
    contracts = table("SELECT * FROM contract_mart", mtime)
    results = table("SELECT * FROM indicator_result", mtime)
    anomaly = table("SELECT ocid, anomaly_score FROM contract_score", mtime).set_index("ocid")["anomaly_score"].reindex(contracts["ocid"])
    w = dict(weights)
    return score_contracts(contracts, results, anomaly if w.get("anomaly", 0) > 0 else None, CFG, w)


def pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def naira(x) -> str:
    if pd.isna(x):
        return "n/a"
    for unit, size in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
        if abs(x) >= size:
            return f"NGN {x / size:,.1f}{unit}"
    return f"NGN {x:,.0f}"


if not DB.exists():
    st.error(f"Warehouse not found at {DB}. Run `python -m redflag run` first.")
    st.stop()
MT = DB.stat().st_mtime

st.sidebar.title("Procurement Red-Flag DSS")
PAGES = ["Overview", "Data quality", "Indicators", "Watchlist", "Buyers", "About"]
requested = st.query_params.get("view", "Overview")  # allows a direct link such as ?view=Watchlist
page = st.sidebar.radio("View", PAGES, index=PAGES.index(requested) if requested in PAGES else 0, label_visibility="collapsed")
manifest = table("SELECT key, value FROM run_manifest", MT).set_index("key")["value"]
st.sidebar.caption(f"Database: {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
st.sidebar.caption(f"Version {__version__} | data retrieved {manifest['retrieval_date']} | config {manifest['config_sha256'][:10]}")
st.warning(DISCLAIMER)

contracts = table("SELECT * FROM contract_mart", MT)
scores = table("SELECT * FROM contract_score", MT)
sources = table("SELECT * FROM source_file", MT)

# ------------------------------------------------------------------ overview
if page == "Overview":
    st.header("Overview")
    n_proc = table("SELECT COUNT(*) AS n FROM contracting_process", MT)["n"].iat[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contracting processes", f"{n_proc:,}")
    c2.metric("Contracts scored", f"{len(contracts):,}", help="Processes that reached the tender or award stage")
    c3.metric("With at least one flag", pct((scores["n_flags"] > 0).mean()))
    c4.metric("With two or more flags", f"{int((scores['n_flags'] >= 2).sum()):,}")

    st.subheader("Sources")
    per = contracts.merge(scores[["ocid", "n_flags"]], on="ocid").groupby("source").agg(
        scored=("ocid", "count"), flagged=("n_flags", lambda s: int((s > 0).sum())), value=("award_amount", "sum"))
    src = sources.set_index("source").join(per).reset_index()
    src["Source"] = src["source"].map(SOURCE_LABELS)
    src["Awarded value"] = src["value"].map(naira)
    src["Share flagged"] = (src["flagged"] / src["scored"]).map(pct)
    st.dataframe(src.rename(columns={"lines": "Records read", "accepted": "Accepted", "rejected": "Rejected", "scored": "Contracts scored"})[
        ["Source", "Records read", "Accepted", "Rejected", "Contracts scored", "Share flagged", "Awarded value"]], hide_index=True, use_container_width=True)

    st.subheader("Contracts by number of flags")
    dist = scores["n_flags"].value_counts().sort_index().rename_axis("Flags").rename("Contracts")
    st.bar_chart(dist)

# -------------------------------------------------------------- data quality
elif page == "Data quality":
    st.header("Data quality: published data against the warehouse")
    st.caption("Each figure is the share of records passing the check. 'Published' is the data as downloaded; "
               "'Warehouse' is the same check after harmonisation. Gaps that cannot be repaired stay visible.")
    dq = table("SELECT * FROM dq_metric", MT)
    source = st.selectbox("Source", list(SOURCE_LABELS), format_func=SOURCE_LABELS.get)
    d = dq[dq["source"] == source]
    wide = d.pivot_table(index=["dimension", "metric"], columns="stage", values="value", aggfunc="first").reset_index()
    den = d[d["stage"] == "raw"].set_index(["dimension", "metric"])["denominator"]
    notes = d[d["note"].fillna("") != ""].groupby(["dimension", "metric"])["note"].first()
    wide["Records checked"] = [den.get((a, b), 0) for a, b in zip(wide["dimension"], wide["metric"])]
    wide["Note"] = [notes.get((a, b), "") for a, b in zip(wide["dimension"], wide["metric"])]
    wide["Check"] = wide["metric"].map(DQ_LABELS)
    wide["Change"] = ((wide["warehouse"] - wide["raw"]) * 100).round(1)
    for dim, title in (("fitness", "Fitness for indicators"), ("validity", "Validity"), ("consistency", "Consistency"), ("completeness", "Completeness")):
        st.subheader(title)
        part = wide[wide["dimension"] == dim].copy()
        part["Published"] = part["raw"].map(pct)
        part["Warehouse"] = part["warehouse"].map(pct)
        st.dataframe(part[["Check", "Published", "Warehouse", "Change", "Records checked", "Note"]].rename(columns={"Change": "Change (points)"}),
                     hide_index=True, use_container_width=True)

    st.subheader("Validation log")
    log = table("SELECT source, severity, code, COUNT(*) AS records FROM validation_log GROUP BY 1, 2, 3 ORDER BY 4 DESC", MT)
    st.dataframe(log[log["source"] == source].drop(columns="source").rename(columns={"severity": "Severity", "code": "Issue", "records": "Records"}),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- indicators
elif page == "Indicators":
    st.header("Red-flag indicators")
    res = table("SELECT r.*, c.source FROM indicator_result r JOIN contract_mart c USING (ocid)", MT)
    source = st.selectbox("Source", ["all"] + list(SOURCE_LABELS), format_func=lambda s: "All sources" if s == "all" else SOURCE_LABELS[s])
    r = res if source == "all" else res[res["source"] == source]
    summary = r.pivot_table(index="indicator", columns="status", values="ocid", aggfunc="count", fill_value=0).reindex(INDICATORS).fillna(0).astype(int)
    for col in STATUS_LABELS:
        if col not in summary:
            summary[col] = 0
    evaluated = summary["flagged"] + summary["clear"]
    out = summary[list(STATUS_LABELS)].rename(columns=STATUS_LABELS)
    out.insert(0, "Indicator", [INDICATOR_LABELS[i] for i in out.index])
    out["Flag rate where assessed"] = (summary["flagged"] / evaluated.where(evaluated > 0)).map(pct)
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.caption("'Not applicable' means the rule does not apply to that kind of contract (for example, a single tenderer under "
               "Direct Procurement). 'Not computable' means the data needed is missing or failed its fitness check.")

    st.subheader("Why an indicator could not be computed")
    ind = st.selectbox("Indicator", INDICATORS, format_func=INDICATOR_LABELS.get)
    reasons = r[(r["indicator"] == ind) & (r["status"] == "non_computable")]["reason"]
    reasons = reasons.str.replace(r"in [\d.]+% of pairs", "in most pairs", regex=True).value_counts().rename_axis("Reason").rename("Contracts").reset_index()
    st.dataframe(reasons, hide_index=True, use_container_width=True)

    st.subheader("Single-bidder rate by procurement method")
    sb = r[(r["indicator"] == "single_bidder") & r["status"].isin(["flagged", "clear"])].merge(contracts[["ocid", "method"]], on="ocid")
    rate = sb.groupby("method").agg(Tenders=("ocid", "count"), rate=("status", lambda s: (s == "flagged").mean())).reset_index()
    rate["Method"] = rate["method"].map(METHOD_LABELS)
    rate["Single-bidder rate"] = rate["rate"].map(pct)
    st.dataframe(rate[["Method", "Tenders", "Single-bidder rate"]], hide_index=True, use_container_width=True)

# ----------------------------------------------------------------- watchlist
elif page == "Watchlist":
    st.header("Ranked watchlist")
    defaults = CFG.weights
    with st.sidebar.expander("Indicator weights", expanded=False):
        st.caption("Defaults come from config.toml. Changes here re-rank this view only.")
        weights = {}
        for name in [k for k in defaults if k != "bid_level_screens"]:
            weights[name] = st.slider(INDICATOR_LABELS[name], 0.0, 1.0, float(defaults[name]), 0.05, key=f"w_{name}")
        weights["bid_level_screens"] = 0.0
    if sum(weights.values()) <= 0:
        st.error("At least one weight must be above zero.")
        st.stop()
    custom = any(abs(weights[k] - defaults[k]) > 1e-9 for k in defaults)
    ranked = rescore(tuple(sorted(weights.items())), MT) if custom else scores
    if custom:
        st.info("Showing a ranking with adjusted weights. The stored watchlist uses the configured defaults.")

    wl = ranked.merge(contracts, on="ocid")
    f1, f2, f3, f4 = st.columns([2, 3, 2, 2])
    src_pick = f1.multiselect("Source", list(SOURCE_LABELS), format_func=SOURCE_LABELS.get)
    buyer_pick = f2.multiselect("Buyer", sorted(wl["buyer_name"].dropna().unique()))
    min_flags = f3.selectbox("Minimum flags", [0, 1, 2, 3], index=1)
    top_n = f4.selectbox("Show", [50, 100, 250, 500, 1000], index=1)
    if src_pick:
        wl = wl[wl["source"].isin(src_pick)]
    if buyer_pick:
        wl = wl[wl["buyer_name"].isin(buyer_pick)]
    wl = wl[wl["n_flags"] >= min_flags].sort_values("rank").head(top_n)

    view = pd.DataFrame({
        "Rank": wl["rank"], "Risk score": wl["risk_score"].round(3), "Flags": wl["n_flags"],
        "Why it is on the list": wl["explanation"], "Buyer": wl["buyer_name"], "Contract": wl["title"],
        "Value": wl["award_amount"].map(naira), "Source": wl["source"].map(SOURCE_LABELS),
        "Assessed": wl["coverage"].map(pct), "ocid": wl["ocid"],
    })
    st.dataframe(view, hide_index=True, use_container_width=True, height=420,
                 column_config={"Assessed": st.column_config.TextColumn(help="Share of the indicator weight that could be assessed for this contract")})
    st.download_button("Download this list (CSV)", view.to_csv(index=False).encode("utf-8-sig"), "watchlist_export.csv", "text/csv")

    st.subheader("Inspect a contract")
    if len(wl):
        pick = st.selectbox("Contract", wl["ocid"].tolist(), format_func=lambda o: f"#{int(wl.set_index('ocid').at[o, 'rank'])}  {o}")
        row = wl.set_index("ocid").loc[pick]
        a, b, c, d = st.columns(4)
        a.metric("Risk score", f"{row['risk_score']:.3f}")
        b.metric("Value", naira(row["award_amount"]))
        c.metric("Tenderers", "n/a" if pd.isna(row["num_tenderers"]) else int(row["num_tenderers"]))
        d.metric("Submission period", "n/a" if pd.isna(row["period_days"]) else f"{int(row['period_days'])} days")
        st.write(f"**{row['title'] or 'Untitled'}**")
        st.write(f"Buyer: {row['buyer_name']}  |  Supplier: {row['supplier_name'] or 'not resolved'}  |  "
                 f"Method: {METHOD_LABELS.get(row['method'], 'not mapped')}  |  Award date: {row['award_date'] or 'n/a'}")
        all_results = table("SELECT ocid, indicator, status, reason FROM indicator_result", MT)
        detail = all_results[all_results["ocid"] == pick].copy()
        detail["Indicator"] = detail["indicator"].map(INDICATOR_LABELS)
        detail["Result"] = detail["status"].map(STATUS_LABELS)
        detail["Weight"] = detail["indicator"].map(weights)
        st.dataframe(detail.rename(columns={"reason": "Evidence"})[["Indicator", "Result", "Evidence", "Weight"]], hide_index=True, use_container_width=True)
        if pd.notna(row["anomaly_score"]):
            st.caption(f"Anomaly percentile within its source: {row['anomaly_score']:.2f} (1.00 is the most unusual).")
    else:
        st.write("No contracts match these filters.")

# -------------------------------------------------------------------- buyers
elif page == "Buyers":
    st.header("Buyers")
    st.caption("Buyer-level view of the same flags. Rates are shown only for buyers with enough contracts to be meaningful.")
    res = table("SELECT ocid, indicator, status FROM indicator_result", MT)
    wide = res.pivot(index="ocid", columns="indicator", values="status")
    c = contracts.merge(scores[["ocid", "risk_score", "n_flags"]], on="ocid").join(wide, on="ocid")
    min_n = st.slider("Minimum contracts per buyer", 5, 200, 20, 5)
    g = c.groupby(["source", "buyer_name"]).agg(
        Contracts=("ocid", "count"), value=("award_amount", "sum"), flagged=("n_flags", lambda s: (s > 0).mean()),
        sb_flag=("single_bidder", lambda s: (s == "flagged").sum()), sb_eval=("single_bidder", lambda s: s.isin(["flagged", "clear"]).sum()),
        nc_flag=("non_competitive", lambda s: (s == "flagged").sum()), nc_eval=("non_competitive", lambda s: s.isin(["flagged", "clear"]).sum()),
        risk=("risk_score", "mean")).reset_index()
    g = g[g["Contracts"] >= min_n].sort_values("risk", ascending=False)
    out = pd.DataFrame({
        "Buyer": g["buyer_name"], "Source": g["source"].map(SOURCE_LABELS), "Contracts": g["Contracts"], "Awarded value": g["value"].map(naira),
        "Share flagged": g["flagged"].map(pct), "Single-bidder rate": (g["sb_flag"] / g["sb_eval"].where(g["sb_eval"] > 0)).map(pct),
        "Non-competitive share": (g["nc_flag"] / g["nc_eval"].where(g["nc_eval"] > 0)).map(pct), "Mean risk score": g["risk"].round(3),
    })
    st.dataframe(out, hide_index=True, use_container_width=True, height=560)
    st.download_button("Download buyer table (CSV)", out.to_csv(index=False).encode("utf-8-sig"), "buyers_export.csv", "text/csv")

# --------------------------------------------------------------------- about
else:
    st.header("About this system")
    st.markdown(
        """
This system consolidates Nigerian federal (NOCOPO / BPP) and state open-contracting data, measures how fit each
field is for analysis, computes the red-flag indicators the data can support, and ranks contracts for review.

**How to read the output**

- A flag is a *risk signal*. Many flags have legitimate explanations. The watchlist prioritises contracts for a
  person to review; it does not establish wrongdoing by any buyer or supplier.
- An indicator marked *not computable* was not assessed because the data is missing or unreliable. It is never
  treated as "no risk".
- The risk score is a weighted sum of triggered flags divided by the total weight. Identical inputs and
  configuration always produce the same ranking.

**Indicators**
"""
    )
    w = CFG.weights
    info = pd.DataFrame({"Indicator": [INDICATOR_LABELS[k] for k in w], "Configured weight": list(w.values())})
    st.dataframe(info, hide_index=True)
    st.markdown("**Inputs to this run**")
    st.dataframe(sources.assign(Source=sources["source"].map(SOURCE_LABELS))[["Source", "file", "sha256", "lines"]]
                 .rename(columns={"file": "File", "sha256": "SHA-256", "lines": "Records"}), hide_index=True, use_container_width=True)
