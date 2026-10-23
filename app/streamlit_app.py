"""RecRankingLab dashboard. Reads only committed artefacts: docs/02-results.md and results/."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from recsys.abtest import mde_for_n, power_for_n, sample_size_per_arm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "02-results.md"
RESULTS_DIR = ROOT / "results"


def results_tables(md: str) -> list[tuple[str, str]]:
    """(heading, markdown table) for each markdown table in the results doc, verbatim."""
    out, heading, rows = [], "", []
    for line in md.splitlines() + [""]:
        if line.startswith("|"):
            rows.append(line)
            continue
        if rows:
            out.append((heading, "\n".join(rows)))
            rows = []
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
    return out


st.set_page_config(page_title="RecRankingLab", layout="wide")
st.title("RecRankingLab")
st.caption("Two-stage recommender on MovieLens 25M. The A/B tabs are SIMULATED, not a real experiment.")
tab_metrics, tab_recs, tab_power, tab_cuped = st.tabs(
    ["Metrics", "Recommendations", "A/B power (simulated)", "CUPED (simulated)"]
)

with tab_metrics:
    st.write("Test split, 3,833 users. Tables are copied from `docs/02-results.md` "
             "(mean [95% bootstrap CI over users]).")
    for heading, table in results_tables(RESULTS.read_text()):
        if "epoch" in table.splitlines()[0]:
            continue
        st.subheader(heading)
        st.markdown(table)

with tab_recs:
    recs = pl.read_parquet(RESULTS_DIR / "recs_sample.parquet").to_pandas()
    st.write("Top-10 for a seeded sample of test users: the popularity control and the M4 "
             "LightGBM ranker (single stage). `liked_later` = the user rated it >= 4 in the "
             "test period. The two-tower stage is not in this export.")
    user = st.selectbox("User", sorted(recs["user_id"].unique()))
    one = recs[recs["user_id"] == user]
    cols = st.columns(2)
    for col, (arm, d) in zip(cols, one.groupby("arm", sort=False), strict=False):
        col.subheader(arm)
        col.caption(f"{int(d['liked_later'].sum())} of 10 liked later")
        col.dataframe(d[["rank", "title", "liked_later"]], hide_index=True)

with tab_power:
    study = json.loads((RESULTS_DIR / "ab_study.json").read_text())
    plan, s = study["plan"], study["study"]
    st.warning("SIMULATED. Engagement is generated from held-out ratings under assumed click "
               "probabilities (docs/03-ab-simulation.md). No real users were randomised.")
    alpha = st.select_slider("alpha", [0.01, 0.05, 0.10], value=0.05)
    power = st.select_slider("power", [0.7, 0.8, 0.9], value=0.8)
    ns = np.arange(500, 40_001, 500)
    curve = pd.DataFrame({"users per arm": ns,
                          "MDE": [mde_for_n(int(n), plan["sd"], alpha, power) for n in ns]})
    st.line_chart(curve, x="users per arm", y="MDE")
    st.write(f"Modelled effect {plan['delta']:.4f}; needs "
             f"**{sample_size_per_arm(plan['delta'], plan['sd'], alpha, power):,}** users per arm "
             f"at alpha={alpha}, power={power}.")
    n_arm = st.slider("users per arm", 200, 5000, s["n_users"] // 2, 100)
    st.write(f"Planned power at {n_arm:,} per arm: "
             f"{power_for_n(plan['delta'], n_arm, plan['sd'], alpha):.3f}")
    st.table(pd.DataFrame({
        "": ["unadjusted", "CUPED"],
        f"A/A rejection (target {s['alpha']})": [s["aa"]["reject_naive"], s["aa"]["reject_cuped"]],
        f"A/B power ({s['n_sims']:,} sims, {s['n_users']:,} users)":
            [s["ab"]["reject_naive"], s["ab"]["reject_cuped"]],
    }).set_index(""))

with tab_cuped:
    st.warning("SIMULATED.")
    ab = s["ab"]
    st.write(f"On this population the covariate (log prior ratings) has rho = {plan['rho']:.3f} "
             "with the outcome, so CUPED barely helps.")
    st.table(pd.DataFrame({"mean SE": [ab["mean_se_naive"], ab["mean_se_cuped"]]},
                          index=["before (unadjusted)", "after (CUPED)"]))
    st.subheader("What if the covariate were better?")
    st.caption("Analytic: CUPED scales variance by 1 - rho^2. Not a simulation result.")
    rho = st.slider("assumed rho", 0.0, 0.9, 0.5, 0.05)
    base = sample_size_per_arm(plan["delta"], plan["sd"])
    adj = sample_size_per_arm(plan["delta"], plan["sd"], variance_ratio=1 - rho**2)
    st.write(f"Users per arm: {base:,} unadjusted vs {adj:,} with CUPED ({1 - adj / base:.0%} fewer).")
