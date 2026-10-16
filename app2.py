"""
Home Energy Manager — Savings Analysis (v4)
===========================================
Rebuilt around one question: can every rupee on this screen survive being
challenged?

What changed from v3, and why
-----------------------------
* Savings are now derived from MEASURED telemetry flows, not from a simulated
  controller policy. A simulation answers "what would a smart inverter do";
  it cannot answer "what did this site actually save".
* The five benefit streams are reported SEPARATELY and never summed into one
  headline before the split is shown: billing savings, islanding savings
  (solar-hour outage only), night battery discharge, off-solar outage backup
  value, export earning — less grid-charging cost.
* The local clock is fitted against solar noon instead of trusted.
* Tariffs are built up from published components (slab + wheeling + duty)
  rather than hardcoded all-in numbers, and Maharashtra and Uttar Pradesh are
  both selectable.
* Bills are shown as full itemised bills — energy, wheeling, fixed, meter
  rent, duty — per real billing cycle, with daily and monthly views.
* The lead-acid coulomb-counting panel, Peukert calibration panel, backup-time
  estimator and product-sizing tool are removed from this app. They are useful
  tools but they belong in a battery/sizing app, not in a savings analysis
  that a customer-facing number comes out of.

Run:  streamlit run app4.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from streamlit.engine import (
    COL_ACTIVE, COL_BAT_SOC, COL_GRID_V, COL_TS_ALT, COL_TS_PRIMARY,
    account, bill_table, derive_flows, fit_local_time_offset, full_cycle_bills,
    estimate_array_size, integrity_report, measured_round_trip, outage_episodes,
    outage_model, outage_slab_working, preprocess,
)
from streamlit.dess_loader import load_export
from streamlit.tariffs import DEFAULT_STATE, STATE_TARIFFS, build_tariff

st.set_page_config(page_title="Solar Savings Analysis", layout="wide")

RS = "₹"


def rs(x: float, dp: int = 0) -> str:
    return f"{RS}{x:,.{dp}f}"


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Inverter export (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
    sheet = st.text_input("Sheet name (single-sheet xlsx only)", value="Sheet1",
                          help="Ignored for one-sheet-per-day exports — every daily sheet is "
                               "read and concatenated automatically.")

    st.header("2 · State & tariff")
    _names = list(STATE_TARIFFS.keys())

    def _label(n: str) -> str:
        sp = STATE_TARIFFS[n]
        tag = "" if sp.get("verified", True) else "  ⚠"
        reg = sp.get("region", "")
        return f"{n}{tag}" + (f"   ·  {reg}" if reg else "")

    state = st.selectbox("Billing state / discom", _names,
                         index=_names.index(DEFAULT_STATE), format_func=_label)
    spec = STATE_TARIFFS[state]
    if not spec.get("verified", True):
        st.error(
            "**This schedule is not verified.** Some part of it — a slab boundary, a duty "
            "rate, or a published figure that contradicts itself — could not be confirmed "
            "against the regulator's own order. Use it to size an opportunity, never to "
            "bill a customer or quote a saving. The note at the foot of this section says "
            "exactly what is uncertain.")
    fixed_opt = None
    if spec["fixed_options"]:
        fixed_opt = st.selectbox("Connection type", list(spec["fixed_options"].keys()))
    sanctioned_kw = st.number_input(
        "Sanctioned load (kW)", 0.5, 50.0, 5.0, 0.5,
        help="Drives the per-kW fixed charge in UP. Not used for Maharashtra, "
             "whose LT-I(B) fixed charge is a flat monthly figure.")
    with st.expander("Override tariff components"):
        fac = st.number_input("FAC (₹/kWh)", 0.0, 5.0, float(spec["fac_per_kwh"]), 0.01,
                              help="Fuel adjustment charge is notified monthly. Take it "
                                   "from the customer's actual bill.")
        duty = st.number_input("Electricity duty (%)", 0.0, 30.0, float(spec["duty_pct"]), 0.5)
        duty_fixed = st.checkbox(
            "Duty applies to fixed charge", value=bool(spec["duty_on_fixed"]),
            help="Maharashtra Electricity Duty Act 2016, s.2(1)(e) defines consumption "
                 "charges as including fixed charges such as demand charges, so by statute "
                 "this is on. Untick only if the customer's actual bill shows otherwise.")
        duty_meter = st.checkbox(
            "Duty applies to meter rent", value=bool(spec["duty_on_meter"]),
            help="The Act names meter rent neither way. It is a charge for equipment rather "
                 "than for energy consumed, so it is left out by default. This is the one "
                 "genuinely unsettled component — check a real bill.")
        wheel = st.number_input("Wheeling (₹/kWh)", 0.0, 5.0,
                                float(spec["wheeling_per_kwh"]), 0.05)
        meter_rent = st.number_input("Meter rent (₹/month)", 0.0, 500.0,
                                     float(spec["meter_rent_month"]), 1.0,
                                     help="Not part of the tariff schedule — a separate "
                                          "miscellaneous charge. Take it from the bill.")
        levy = st.number_input(
            "Flat per-unit levy (₹/kWh)", 0.0, 5.0,
            float(spec.get("levy_per_kwh", 0.0)), 0.01,
            help="Levies charged as a flat amount per unit rather than a percentage: "
                 "Andhra's ₹0.06 duty, Chhattisgarh's ₹0.10 cess, Goa's ₹0.20 duty plus "
                 "₹0.05 public-lighting duty, Himachal's ₹0.10 milk cess. They scale with "
                 "units, so they belong in the marginal rate — but outside the duty "
                 "gross-up, because they are not a percentage of anything.")
        st.caption(spec["notes"])

    with st.expander("Compare all states in the catalogue"):
        _rows = []
        for _n, _sp in STATE_TARIFFS.items():
            _t = build_tariff(_n, sanctioned_kw=sanctioned_kw)
            _first = _t.slabs[0][1]
            _rows.append({
                "State / discom": _n,
                "Group": _sp.get("region", "—"),
                "First slab ₹": _first,
                "Top slab ₹": _t.slabs[-1][1],
                "All-in marginal ₹/kWh": _t.marginal_rate(9e9),
                "Duty %": _t.duty_pct,
                "Flat levy ₹/kWh": _t.levy_per_kwh,
                "Fixed ₹/mo": _t.fixed_month + _t.fixed_per_kw_month * sanctioned_kw,
                "Meter ₹/mo": _t.meter_rent_month,
                "Bill @ 250 u": _t.build_bill(250.0)["total"],
                "Verified": "yes" if _t.verified else "NO",
            })
        _cmp = pd.DataFrame(_rows).sort_values("All-in marginal ₹/kWh", ascending=False)
        st.dataframe(_cmp.style.format({
            "First slab ₹": "{:.2f}", "Top slab ₹": "{:.2f}",
            "All-in marginal ₹/kWh": "{:.2f}", "Duty %": "{:.0f}%",
            "Flat levy ₹/kWh": "{:.2f}", "Fixed ₹/mo": "{:,.0f}",
            "Meter ₹/mo": "{:,.0f}", "Bill @ 250 u": "₹{:,.0f}"}),
            hide_index=True, use_container_width=True)
        st.caption(
            f"All-in marginal rate = (top energy slab + wheeling + FAC) × (1 + duty%) + flat "
            f"levy. Fixed charge and meter rent are deliberately outside it: they do not "
            f"change with units, so an avoided unit does not avoid them. Fixed charge is "
            f"shown at the sanctioned load set above ({sanctioned_kw:g} kW), which is why "
            f"the per-kW states move when you change it. Rows marked **NO** under Verified "
            f"have something unconfirmed in the schedule — see the warning when you select "
            f"them.")
        st.download_button("Download tariff comparison (CSV)",
                           _cmp.to_csv(index=False).encode(),
                           "state_tariff_comparison.csv", "text/csv")

    st.header("3 · Export settlement")
    settle = st.radio("Mode", ["net_metering", "net_billing", "no_export"],
                      format_func=lambda s: {
                          "net_metering": "Net metering — banked, carried forward (MERC Reg 11.4)",
                          "net_billing": "Net billing (all export at generic rate)",
                          "no_export": "No export / export not credited",
                      }[s])
    export_rate = st.number_input("Surplus export rate (₹/kWh)", 0.0, 15.0,
                                  float(spec["export_rate_default"]), 0.01,
                                  help="Generic tariff used only for the year-end cash-out of "
                                       "credit still unadjusted, not for every month's export.")
    sett_month = st.selectbox(
        "Settlement year starts", list(range(1, 13)), index=3,
        format_func=lambda m: pd.Timestamp(2026, m, 1).strftime("%B"),
        help="MERC settlement period runs 1 April to 31 March. The banked kWh credit "
             "resets to zero at the start of each period.")

    st.header("3b · Outage pricing")
    out_price = st.radio(
        "Outage energy valued at", ["incremental", "chronological"],
        format_func=lambda s: "Top of the month's stack (incremental)" if s == "incremental"
        else "Slab position at that moment (chronological)",
        help="Outage energy never appears on a bill, so it needs a counterfactual. Had the grid "
             "been up, that energy would have been ADDITIONAL consumption on top of the month's "
             "units — and a telescopic tariff puts additional units at the top of the stack.")

    st.header("4 · Billing cycle")
    cyc_mode = st.radio("Cycle basis", ["monthly", "fixed_days"],
                        format_func=lambda s: "Monthly (meter reading day)" if s == "monthly"
                        else "Fixed number of days")
    anchor = st.number_input("Meter reading day of month", 1, 28, 1, 1) if cyc_mode == "monthly" else 1
    cyc_days = st.number_input("Days per cycle", 7, 90, 30, 1) if cyc_mode == "fixed_days" else 30

    st.header("5 · Clock & solar hours")
    ts_choice = st.selectbox("Timestamp column", [COL_TS_PRIMARY, COL_TS_ALT])
    auto_off = st.checkbox("Fit local-time offset from solar noon", value=True)
    man_off = st.number_input("Manual offset (hours)", -12.0, 12.0, 0.0, 0.25,
                              disabled=auto_off)
    lon = st.number_input("Site longitude (°E)", 68.0, 98.0, 75.0, 0.5,
                          help="Used to compute astronomical solar noon. Maharashtra "
                               "spans roughly 72.5-80.5°E.")
    sol_s, sol_e = st.slider("Solar-hour window (local)", 0.0, 24.0, (7.0, 18.0), 0.5)

    st.header("6 · Outage definition")
    out_v = st.number_input("Grid considered down at/below (V)", 0.0, 150.0, 1.0, 1.0)
    min_out = st.number_input("Minimum outage duration (minutes)", 0.0, 120.0, 5.0, 1.0,
                              help="Zero-voltage runs shorter than this are treated as "
                                   "sampling artefacts rather than supply interruptions.")

    st.header("7 · Battery")
    bat_ah = st.number_input("Rated capacity (Ah)", 10.0, 2000.0, 200.0, 10.0)
    bat_v = st.number_input("Nominal bank voltage (V)", 12.0, 240.0, 48.0, 12.0)
    cap = bat_ah * bat_v / 1000.0
    st.caption(f"Nameplate = **{cap:.2f} kWh**  ({bat_ah:.0f} Ah × {bat_v:.0f} V)")
    dod = st.slider("Usable depth of discharge", 0.2, 1.0, 0.50, 0.05,
                    help="0.50 for lead-acid, 0.80–0.90 for lithium.")
    st.caption(f"Usable = **{cap * dod:.2f} kWh**")
    crate = st.number_input("Charge C-rate limit", 0.02, 1.0, 0.15, 0.01,
                            help="Applied to NAMEPLATE capacity, which is how C-rate is "
                                 "defined. 0.15C on 200 Ah = 30 A.")
    st.caption(f"Charge cap = **{crate * cap:.2f} kW**  ({crate * bat_ah:.0f} A at {bat_v:.0f} V)")
    rte_mode = st.radio("Round-trip efficiency", ["measured", "manual"],
                        format_func=lambda s: "Use the measured value" if s == "measured"
                        else "Enter a value",
                        help="Both charge and discharge are metered, so the efficiency can be "
                             "observed rather than assumed. It is used only to value stored "
                             "energy, never to adjust a kWh figure.")
    rte_manual = st.number_input("Round-trip efficiency", 0.3, 1.0, 0.75, 0.01,
                                 disabled=(rte_mode == "measured"))
    keep_float = st.checkbox("Count float current at 100% SOC as stored energy",
                            value=False,
                            help="Leave off. Float current offsets self-discharge and is "
                                 "lost as heat, not stored.")

    st.header("8 · Dispatch order")
    dispatch = st.radio("Who serves the load first", ["pv_first", "battery_first"],
                        format_func=lambda s: "PV first, then battery/grid (how the inverter works)"
                        if s == "pv_first" else "Battery first (sensitivity check)",
                        help="PV-first is physically correct for a hybrid inverter: solar carries "
                             "the load whenever available, outage or not. The alternative is kept "
                             "only to show how little the choice changes the result.")

    st.header("9 · Load channel")
    load_src = st.radio("Load from", ["active", "apparent_pf"],
                        format_func=lambda s: "AC active power (W) — measured"
                        if s == "active" else "Apparent power (VA) x assumed PF")
    pf = st.number_input("Assumed power factor", 0.5, 1.0, 0.92, 0.01,
                         disabled=(load_src == "active"))


# =====================================================================
# LOAD
# =====================================================================
st.title("Solar savings analysis — measured, separated, auditable")

if up is None:
    st.info(
        "Upload the inverter export to begin.\n\n"
        "**What this app will not do:** produce a single headline savings number without "
        "showing the five benefit streams separately, value a kWh using fixed charges, or "
        "trust the logger's clock. Those three shortcuts are what make a savings analysis "
        "fall apart under questioning."
    )
    st.stop()


@st.cache_data(show_spinner="Reading file…")
def _read(buf: bytes, name: str, sheet_name: str):
    bio = io.BytesIO(buf)
    bio.name = name
    return load_export(bio, sheet_hint=sheet_name or None)


raw, finfo = _read(up.getvalue(), up.name, sheet)
st.caption(
    f"Loaded **{len(raw):,}** rows × {raw.shape[1]} columns from `{up.name}` — "
    f"{finfo.get('format')}, {finfo.get('sheets')} sheet(s)"
    + (f", serial {', '.join(finfo.get('serial', []))}" if finfo.get("serial") else ""))
if finfo.get("missing_days"):
    md = finfo["missing_days"]
    st.caption(
        f"{len(md)} day(s) have no sheet at all ({', '.join(md[:5])}"
        f"{'…' if len(md) > 5 else ''}). Those are days with no telemetry, not days with no "
        f"consumption — the gap rules exclude them from energy integration rather than "
        f"interpolating across them, and annualised figures scale by metered time.")
if finfo.get("ist_offset_vs_utc_h") is not None:
    st.caption(
        f"Clock chain: `recordedAt` is UTC and `Timestamp` sits "
        f"{finfo['ist_offset_vs_utc_h']:+.2f} h from it, so `Timestamp` is server time in IST. "
        f"`Date Time` is the device's own RTC. The solar-noon fit below is the independent check "
        f"on whichever one is selected.")

# ---------------------------------------------------------------------
# PIPELINE — cached end to end
# ---------------------------------------------------------------------
# Streamlit re-runs this whole script on every widget change. Without a cache
# that means re-reading the workbook and re-walking a year of minute data each
# time somebody nudges a slider, which on a 103k-row file is tens of seconds
# for a result that has not changed.
#
# The cache key below is every input the pipeline actually depends on: the
# file bytes, and each sidebar setting that feeds a calculation. Settings that
# only change how a number is DISPLAYED are deliberately absent, so changing
# one of those costs nothing. The file read is cached separately inside
# `_read`, so two different tariff settings on the same upload share one read.
#
# Nothing here is an approximation — it is the same calls in the same order,
# answered from memory when the inputs repeat.
@st.cache_data(show_spinner="Crunching the series…", max_entries=8)
def _pipeline(buf: bytes, fname: str, sheet_name: str, ts_col: str, lon_deg: float,
              auto_offset: bool, manual_offset: float, load_source: str, pf_: float,
              out_v_: float, min_out_: float, cap_: float, dod_: float, keep_float_: bool,
              cyc_mode_: str, anchor_: int, cyc_days_: int, sol_s_: float, sol_e_: float,
              dispatch_: str, rte_mode_: str, rte_manual_: float, crate_: float,
              state_: str, sanctioned_kw_: float, fixed_opt_: str | None, settle_: str,
              export_rate_: float, fac_: float, duty_: float, wheel_: float,
              duty_fixed_: bool, duty_meter_: bool, levy_: float, meter_rent_: float,
              sett_month_: int, out_price_: str):
    raw_, finfo_ = _read(buf, fname, sheet_name)

    _r = raw_.copy()
    _r[ts_col] = pd.to_datetime(_r[ts_col], errors="coerce")
    fit_ = fit_local_time_offset(_r.dropna(subset=[ts_col]), ts_col, lon_deg)
    offset_ = fit_["offset_h"] if (auto_offset and fit_["ok"]) else manual_offset

    tariff_ = build_tariff(
        state_, sanctioned_kw=sanctioned_kw_, fixed_option=fixed_opt_, settlement=settle_,
        export_rate=export_rate_, fac_per_kwh=fac_, duty_pct=duty_, wheeling_override=wheel_,
        duty_on_fixed=duty_fixed_, duty_on_meter=duty_meter_, levy_override=levy_,
    )
    tariff_.meter_rent_month = float(meter_rent_)

    df_ = preprocess(
        raw_, ts_col=ts_col, local_offset_h=offset_, load_source=load_source, power_factor=pf_,
        outage_voltage_max=out_v_, min_outage_minutes=min_out_, battery_capacity_kwh=cap_,
        battery_dod=dod_, count_float_current_as_charge=keep_float_,
        cycle_mode=cyc_mode_, cycle_anchor_day=int(anchor_), cycle_days=int(cyc_days_),
    )
    df_ = derive_flows(df_, sol_s_, sol_e_, dispatch_order=dispatch_)
    rt_ = measured_round_trip(df_)
    rte_ = rt_["eff"] if (rte_mode_ == "measured" and np.isfinite(rt_["eff"])) else rte_manual_
    rte_ = float(min(max(rte_, 0.05), 1.0))
    res_ = account(df_, tariff_, round_trip_eff=rte_,
                   settlement_start_month=int(sett_month_), outage_pricing=out_price_)
    # `raw` is deliberately NOT returned: st.cache_data deep-copies whatever a
    # cached function hands back, and the caller already holds the same frame
    # from `_read`, which is cached in its own right. Returning it again would
    # copy a second 100k-row frame on every rerun for nothing.
    return dict(
        finfo=finfo_, fit=fit_, offset=offset_, tariff=tariff_, df=df_,
        rt=rt_, rte=rte_, res=res_,
        om=outage_model(df_, cap_, crate_, usable_kwh=cap_ * dod_),
        eps=outage_episodes(df_),
        findings=integrity_report(raw_, df_, ts_col),
    )


_P = _pipeline(
    up.getvalue(), up.name, sheet, ts_choice, lon, auto_off, man_off, load_src, pf,
    out_v, min_out, cap, dod, keep_float, cyc_mode, int(anchor), int(cyc_days),
    sol_s, sol_e, dispatch, rte_mode, rte_manual, crate,
    state, sanctioned_kw, fixed_opt, settle, export_rate, fac, duty, wheel,
    duty_fixed, duty_meter, levy, meter_rent, int(sett_month), out_price,
)
fit, offset, tariff, df = _P["fit"], _P["offset"], _P["tariff"], _P["df"]
rt, rte, res = _P["rt"], _P["rte"], _P["res"]
T = res["totals"]
om, eps, findings = _P["om"], _P["eps"], _P["findings"]

n_crit = sum(1 for f in findings if f["severity"] == "critical")
if n_crit:
    st.error(
        f"**{n_crit} critical data-integrity finding(s).** These affect how the numbers below "
        f"should be read — open *Data integrity* before quoting anything.", icon="⚠️")

if auto_off and fit["ok"] and abs(fit["offset_h"]) > 0.5:
    st.warning(
        f"**Clock corrected by {fit['offset_h']:+.2f} h.** PV in `{ts_choice}` peaks at "
        f"{fit['observed_noon']:.2f} h, but solar noon at {lon:.1f}°E is "
        f"{fit['astronomical_noon']:.2f} h (fitted on {fit['n_days']} clear days, "
        f"±{fit['scatter_min']:.0f} min). Every day/night split below uses the corrected clock. "
        f"Confirm the real site coordinates and the platform's timezone setting before "
        f"presenting the solar-hour vs night outage split as final.", icon="🕐")

metered_days = float(df["dt_h"].sum()) / 24
scale = 365.0 / metered_days if metered_days > 0 else 1.0

tabs = st.tabs([
    "Savings — separated", "Bills", "Outage model", "Load pattern",
    "Charge / discharge logic", "Energy balance", "Data integrity", "Audit trail",
])

# =====================================================================
# TAB 1 — SAVINGS
# =====================================================================
with tabs[0]:
    st.subheader("Five separate streams")
    st.caption(
        f"Measured over **{metered_days:.1f} days** of valid telemetry. Annualised figures "
        f"scale by metered time (×{scale:.2f}), never by calendar days.")

    c = st.columns(3)
    c[0].metric("1 · Billing savings (self-consumption)", rs(T["billing_savings"]),
                help="PV served the load while the grid was up. Reduces metered import "
                     "directly. Valued at the all-in marginal rate (slab + wheeling + duty) "
                     "that applied at that moment, given the cycle-cumulative slab position.")
    c[1].metric("2 · Slab displacement", rs(T["slab_displacement"]),
                help="The units still imported become cheaper because solar pushed the "
                     "monthly total down the telescopic slab ladder. A real saving, but not "
                     "an avoided kWh — so it is named separately rather than folded into "
                     "self-consumption.")
    c[2].metric("3 · Islanding savings (solar-hour outage)", rs(T["islanding_savings"]),
                help="Grid down during solar hours. Load carried by PV directly and by the "
                     "bank during that window, plus later discharge of energy that was "
                     "charged by PV during such a window — island energy keeps its identity "
                     "through storage.")
    c = st.columns(3)
    c[0].metric("4 · Night battery discharge", rs(T["night_discharge_savings"]),
                help="Battery served the load off-solar-hours with the grid up. Net of the "
                     "cost of grid energy used to charge it.")
    c[1].metric("5 · Export earning", rs(T["export_earning"]),
                help="Net metering: exported units offset import 1:1 within the month, surplus "
                     "banks as kWh credit and carries forward, and only credit still unadjusted "
                     "at the end of the settlement year is cashed at the generic tariff.")
    c[2].metric("less: grid charging cost", "−" + rs(T["grid_charging_cost"]),
                help="Energy bought from the grid to fill the battery. A real cash cost.")
    st.metric("**Net savings over the metered period**", rs(T["net_savings"]),
              delta=f"{rs(T['net_savings'] * scale)} / yr annualised")
    share = T["slab_displacement"] / max(T["net_savings"], 1e-9)
    if share > 0.08:
        st.info(
            f"**{share:.0%} of the net saving is slab displacement, not avoided energy.** "
            f"On a steep telescopic tariff that is expected — {state.split('—')[0].strip()}'s "
            f"second slab is several times the first — but it is worth saying out loud, "
            f"because it means the saving is highly sensitive to the customer's consumption "
            f"level. The same array on a lower-consuming household saves materially less.",
            icon="📐")

    st.divider()
    st.markdown("##### Reported outside net savings")
    c = st.columns(2)
    c[0].metric("Off-solar outage backup value", rs(T["outage_backup_value"]),
                help="Load carried during a NON-solar-hour outage from battery energy that was "
                     "not island-charged. Keeping the lights on at night has real value to the "
                     "customer, but it is resilience, not a reduction in a tariff bill — a "
                     "household with no system is simply dark and is billed nothing. Including "
                     "it in 'savings' is the most common way an outage-heavy site's business "
                     "case gets inflated.")
    c[1].metric("Net savings incl. backup value", rs(T["net_savings_incl_backup"]))

    st.divider()
    st.markdown("##### Does the billing bucket reconcile with the bill?")
    recon = pd.DataFrame([
        dict(Item="Baseline energy cost (no system, grid-up load)",
             Value=T["baseline_bill_energy"]),
        dict(Item="Actual energy cost (metered import)", Value=T["actual_bill_energy"]),
        dict(Item="Difference — must be explained", Value=T["bill_delta"]),
        dict(Item="  bucket 1 — billing savings", Value=T["billing_savings"]),
        dict(Item="  bucket 2 — slab displacement", Value=T["slab_displacement"]),
        dict(Item="  bucket 4 — night discharge", Value=T["night_discharge_savings"]),
        dict(Item="  less grid charging cost", Value=-T["grid_charging_cost"]),
        dict(Item="Unexplained residual",
             Value=T["bill_delta"] - T["billing_savings"] - T["slab_displacement"]
             - T["night_discharge_savings"] + T["grid_charging_cost"]),
    ])
    st.dataframe(recon.style.format({"Value": lambda v: rs(v, 1)}),
                 hide_index=True, use_container_width=True)
    st.caption(
        "The residual is the test. Islanding and backup value are deliberately absent from "
        "both sides of this reconciliation — the baseline bill counts no outage-time load, "
        "because an unpowered house is billed nothing. A residual near zero means the buckets "
        "partition the bill delta without double counting.")

    st.divider()
    st.markdown("##### Per-kWh reality check")
    served = float(df["load_kwh"].sum())
    pv_tot = float(df["pv_kwh"].sum())
    k = st.columns(4)
    k[0].metric("PV generated", f"{pv_tot:,.0f} kWh")
    k[1].metric("Load served", f"{served:,.0f} kWh")
    k[2].metric("Self-consumption", f"{100 * T['self_consumed_kwh'] / max(pv_tot, 1e-9):.1f}%")
    k[3].metric("Realised ₹/kWh of PV",
                rs(T["net_savings"] / max(pv_tot, 1e-9), 2),
                help="Net savings divided by every kWh the array produced. Compare against the "
                     "first-slab marginal rate: a figure far above it means something is being "
                     "double counted; far below it means PV is being exported or curtailed "
                     "rather than displacing expensive units.")

    st.divider()
    gran = st.radio("Bifurcation", ["month", "day", "cycle"], horizontal=True,
                    format_func=lambda s: s.capitalize())
    led = res["ledger"].copy()
    if gran == "month":
        led["_k"] = pd.to_datetime(led["date"]).dt.to_period("M").astype(str)
    elif gran == "day":
        led["_k"] = pd.to_datetime(led["date"]).dt.date.astype(str)
    else:
        led["_k"] = led["cycle_id"]
    agg = led.groupby("_k").agg(
        billing=("billing_savings", "sum"), slab_displacement=("slab_displacement", "sum"),
        islanding=("islanding_savings", "sum"), night=("night_discharge_savings", "sum"),
        backup=("outage_backup_value", "sum"), grid_chg_cost=("grid_charging_cost", "sum"),
    ).reset_index().rename(columns={"_k": gran})
    agg["net_ex_backup_ex_export"] = (agg["billing"] + agg["slab_displacement"]
                                      + agg["islanding"] + agg["night"] - agg["grid_chg_cost"])
    st.dataframe(agg.style.format({c: "{:,.1f}" for c in agg.columns if c != gran}),
                 hide_index=True, use_container_width=True)
    st.bar_chart(agg.set_index(gran)[
        ["billing", "slab_displacement", "islanding", "night", "backup"]])
    st.download_button(f"Download {gran} bifurcation (CSV)",
                       agg.to_csv(index=False).encode(), f"savings_by_{gran}.csv")

# =====================================================================
# TAB 2 — BILLS
# =====================================================================
with tabs[1]:
    st.subheader(f"Itemised bill — {state}")
    st.caption(tariff.source)

    with st.expander("How the marginal savings rate is built — and why fixed charges are not in it",
                     expanded=False):
        rows = []
        cum = 0.0
        for capu, rate in tariff.slabs:
            lbl = f"{cum:.0f}–{capu:.0f}" if np.isfinite(capu) else f"above {cum:.0f}"
            allin = (rate + tariff.wheeling_per_kwh + tariff.fac_per_kwh) * (1 + tariff.duty_pct / 100)
            rows.append(dict(Slab=f"{lbl} units", **{
                "Energy ₹/kWh": rate,
                "Wheeling ₹/kWh": tariff.wheeling_per_kwh,
                "FAC ₹/kWh": tariff.fac_per_kwh,
                f"Duty {tariff.duty_pct:.0f}%": (rate + tariff.wheeling_per_kwh
                                                 + tariff.fac_per_kwh) * tariff.duty_pct / 100,
                "All-in marginal ₹/kWh": allin}))
            cum = capu if np.isfinite(capu) else cum
        st.dataframe(pd.DataFrame(rows).style.format(
            {c: "{:.3f}" for c in rows[0] if c != "Slab"}),
            hide_index=True, use_container_width=True)
        st.markdown(
            f"Fixed charge **{rs(tariff.fixed_month + tariff.fixed_per_kw_month * sanctioned_kw, 0)}/month** "
            f"and meter rent **{rs(tariff.meter_rent_month, 0)}/month** are paid whether or not the "
            f"system exists. They appear in both the baseline and the actual bill below, so they "
            f"cancel in the reduction — which is correct. Rolling them into a ₹/kWh savings rate "
            f"would manufacture savings from a charge that never went away.\n\n"
        f"Electricity duty at {tariff.duty_pct:.0f}% applies to "
        f"{'the fixed charge' if tariff.duty_on_fixed else 'neither fixed charge'}"
        f"{' and meter rent' if tariff.duty_on_meter else ' nor meter rent' if not tariff.duty_on_fixed else ' but not meter rent'}, "
        f"so a zero-consumption month still bills "
        f"{rs(tariff.build_bill(0.0)['total'], 2)}. Both switches are in the sidebar — a real "
        f"bill is what should settle them."
        )

    cycles = full_cycle_bills(df, tariff, int(sett_month))
    st.markdown("##### Per billing cycle — baseline vs actual")
    show = cycles[[
        "cycle", "days", "baseline_units", "actual_units", "export_units",
        "net_billed_units", "bank_kwh",
        "base_energy_charge", "base_wheeling_charge", "base_fixed_charge",
        "base_meter_rent", "base_electricity_duty", "base_total",
        "act_energy_charge", "act_wheeling_charge", "act_fixed_charge",
        "act_meter_rent", "act_electricity_duty", "act_total",
        "export_credit", "act_total_after_credit", "bill_reduction_after_credit",
    ]].rename(columns={
        "baseline_units": "Base units", "actual_units": "Import units",
        "export_units": "Export units", "net_billed_units": "Billed after credit",
        "bank_kwh": "Bank c/f (kWh)",
        "base_energy_charge": "Base energy", "base_wheeling_charge": "Base wheeling",
        "base_fixed_charge": "Base fixed", "base_meter_rent": "Base meter",
        "base_electricity_duty": "Base duty", "base_total": "BASE TOTAL",
        "act_energy_charge": "Act energy", "act_wheeling_charge": "Act wheeling",
        "act_fixed_charge": "Act fixed", "act_meter_rent": "Act meter",
        "act_electricity_duty": "Act duty", "act_total": "ACT TOTAL (gross import)",
        "export_credit": "Export credit", "act_total_after_credit": "ACT NET PAYABLE",
        "bill_reduction_after_credit": "REDUCTION",
    })
    unit_cols = ("Base units", "Import units", "Export units",
                 "Billed after credit", "Bank c/f (kWh)")
    money = [c for c in show.columns if c not in ("cycle", "days") + unit_cols]
    st.dataframe(show.style.format({**{c: "{:,.0f}" for c in money},
                                    **{c: "{:,.1f}" for c in unit_cols}}),
                 hide_index=True, use_container_width=True)
    st.download_button("Download cycle bills (CSV)", cycles.to_csv(index=False).encode(),
                       "cycle_bills.csv")

    k = st.columns(4)
    k[0].metric("Baseline bill (no system)", rs(cycles["base_total"].sum()))
    k[1].metric("Actual net payable", rs(cycles["act_total_after_credit"].sum()))
    k[2].metric("Bill reduction", rs(cycles["bill_reduction_after_credit"].sum()))
    k[3].metric("Annualised", rs(cycles["bill_reduction_after_credit"].sum() * scale))
    st.caption(
        "The reduction here includes the export credit, so it is the number the customer sees "
        "on paper. The savings tab keeps export separate from bill reduction, which is why the "
        "two tabs split the same total differently — deliberately, so the export assumption can "
        "be challenged on its own without disturbing the self-consumption figure.")

    st.divider()
    bgran = st.radio("Finer view", ["month", "day"], horizontal=True, key="bg",
                     format_func=str.capitalize)
    bt = bill_table(df, tariff, bgran)
    st.caption(
        "Slabs are a billing-cycle concept, so a sub-period view apportions that cycle's "
        "charges by its share of units. It never resets the slab counter at midnight, which "
        "would price every day as if it were the customer's first 100 units.")
    st.dataframe(bt.style.format({c: "{:,.1f}" for c in bt.columns
                                  if bt[c].dtype.kind in "fc"}),
                 hide_index=True, use_container_width=True)
    st.download_button(f"Download {bgran} bills (CSV)", bt.to_csv(index=False).encode(),
                       f"bills_by_{bgran}.csv")

    if settle == "net_metering" and not res["net_metering_detail"].empty:
        st.divider()
        st.markdown("##### Net-metering bank — carried forward, settled yearly")
        nmd = res["net_metering_detail"]
        st.caption(
            "MERC Grid-Interactive Rooftop RE Regulations, Reg. 11.4: exported units are set off "
            "against import in the same billing period; surplus carries forward to the next period "
            "**as credited units**, not as money; the carried-forward quantum resets at the start of "
            "each settlement year (1 April – 31 March); and only credit still unadjusted at year end "
            "is bought by the licensee at the generic tariff. A banked unit that offsets a later "
            "month's import is worth that month's retail marginal rate — several times the generic "
            "tariff — so settling every month in cash would understate the benefit badly.")
        st.dataframe(nmd.style.format({
            **{c: "{:,.1f}" for c in ["import_kwh", "export_kwh", "opening_bank_kwh",
                                      "offset_kwh", "billed_kwh", "closing_bank_kwh",
                                      "cashed_out_kwh"]},
            **{c: "{:,.0f}" for c in ["offset_value", "cashout_value", "total_value"]}}),
            hide_index=True, use_container_width=True)
        k = st.columns(4)
        k[0].metric("Value of banked offsets", rs(nmd["offset_value"].sum()))
        k[1].metric("Year-end cash-out", rs(nmd["cashout_value"].sum()),
                    help=f"{nmd['cashed_out_kwh'].sum():,.0f} kWh at {rs(export_rate, 2)}/kWh")
        k[2].metric("Total export value", rs(nmd["total_value"].sum()))
        k[3].metric("Still banked", f"{T.get('unsettled_bank_kwh', 0):,.0f} kWh",
                    help="Settlement year has not closed inside this data, so this credit is "
                         "left unsettled rather than cashed out early.")
        cashed = nmd["cashed_out_kwh"].sum()
        if cashed > 0:
            lo, hi = cashed * 14.38, cashed * 19.29
            st.warning(
                f"**The 31 March reset is expensive on this site.** {cashed:,.0f} kWh of credit "
                f"built up over the low-load winter months and was cashed out at "
                f"{rs(export_rate, 2)}/kWh = {rs(nmd['cashout_value'].sum())}. Had those units "
                f"survived into the heavy June–September months, they would have offset import at "
                f"₹14.38–19.29 — worth roughly {rs(lo)}–{rs(hi)}. The surplus season ends exactly "
                f"at the settlement boundary and the deficit season starts just after it, which is "
                f"the worst possible alignment. Worth raising with the customer: it is an argument "
                f"about system sizing and load timing, not about the hardware.", icon="📅")

# =====================================================================
# TAB 3 — OUTAGE MODEL
# =====================================================================
with tabs[2]:
    st.subheader("Outage model — the variable everything else hinges on")
    s = om["stats"]
    st.markdown(
        f"A **{crate:.2f}C** limit on a **{cap:.2f} kWh** nameplate bank ({bat_ah:.0f} Ah) means "
        f"the charger can absorb at most **{s['charge_kw_cap']:.2f} kW** = {crate * bat_ah:.0f} A. "
        f"During a solar-hour outage the system is islanded, so surplus PV has nowhere to go but "
        f"the battery. That makes the ceiling on recoverable headroom:")
    st.latex(r"\text{headroom}_{\text{day}} = \min\Big(C_{\text{rate}} \times "
             r"E_{\text{nameplate}} \times t_{\text{solar-hour outage}},\; E_{\text{usable}}\Big)")
    st.caption(
        f"C-rate applies to nameplate capacity — that is its definition — while the usable bank "
        f"({cap * dod:.2f} kWh at {dod:.0%} DoD) caps how much headroom there is to refill at all. "
        f"Both constraints bind, so both are in the formula.")

    k = st.columns(5)
    k[0].metric("Mean", f"{s['mean_headroom']:.2f} kWh/day")
    k[1].metric("Median", f"{s['median_headroom']:.2f} kWh/day")
    k[2].metric("P10", f"{s['p10_headroom']:.2f} kWh/day")
    k[3].metric("P90", f"{s['p90_headroom']:.2f} kWh/day")
    k[4].metric("Days with zero daytime outage", f"{s['zero_days_pct']:.0f}%")
    st.caption(
        f"Median headroom is **{s['headroom_as_pct_of_usable']:.0f}% of the usable bank** "
        f"({cap * dod:.2f} kWh). That is the fraction of the bank a pre-drain policy could "
        f"free up on a typical night and still expect to recover the next day.")

    st.error(
        f"**The mean is the wrong number to design a pre-drain policy on.** Mean headroom is "
        f"{s['mean_headroom']:.2f} kWh/day but the median is only {s['median_headroom']:.2f} "
        f"kWh/day, and on {s['zero_days_pct']:.0f}% of days there is no daytime outage at all. "
        f"A controller that pre-drains to the mean every night over-drains on most nights: the "
        f"outage it was holding capacity for never arrives, the bank is refilled from PV that "
        f"would otherwise have been exported, and the site loses export revenue for nothing. "
        f"Size the reserve from a low percentile (P10 here is "
        f"{s['p10_headroom']:.2f} kWh), and let it adapt monthly.", icon="📉")

    st.markdown("##### The third constraint: does the surplus physically exist?")
    k = st.columns(4)
    k[0].metric("Hardware ceiling", f"{s['total_headroom']:,.0f} kWh",
                help="What the charger and the bank could absorb — C-rate and usable capacity only.")
    k[1].metric("PV surplus in daytime outages", f"{s['total_pv_surplus']:,.0f} kWh",
                help="PV minus load during those windows. The energy that actually exists to absorb.")
    k[2].metric("Recoverable (the real figure)", f"{s['total_recoverable']:,.0f} kWh")
    k[3].metric("Actually charged", f"{s['realised_total']:,.0f} kWh")
    binds = s.get("binding_counts", {})
    top = max(binds, key=binds.get) if binds else "—"
    st.error(
        f"**The binding constraint is PV surplus, on {binds.get(top, 0)} of {s['days']} days.** "
        f"The C-rate and the usable bank describe what the hardware *could* take; neither asks "
        f"whether any PV is left over to take. During an outage the load is being carried by PV "
        f"as well, so the energy available to the bank is PV minus load — and only in the "
        f"intervals where that is positive. Here that is {s['total_pv_surplus']:,.0f} kWh a year "
        f"against a {s['total_headroom']:,.0f} kWh hardware ceiling, "
        f"{s['total_headroom'] / max(s['total_pv_surplus'], 1e-9):.1f}× smaller. Quoting the "
        f"hardware ceiling as the opportunity would be the single easiest number in this analysis "
        f"to pull apart.", icon="🧱")
    st.caption(
        "Which is why the pre-drain case has to be argued on the recoverable figure, not the "
        "ceiling. Charge-rate and bank-size limits only start to bind once the surplus is large "
        "enough to reach them — on a site with a bigger array relative to its load, or far "
        "longer daytime outages, that ordering flips.")

    st.divider()
    st.markdown("##### Monthly — how the hours in each month account for themselves")
    mon = om["monthly"]

    st.info(
        "**How to read the hours.** Every hour of the month is in exactly one of three "
        "buckets, and they add up:\n\n"
        "`Grid ON` **+** `Grid OFF (outage)` **=** `Hours logged`  \n"
        "`Hours logged` **+** `Not logged` **=** `Hours in month`\n\n"
        "Outage hours are **inside** the logged hours, not on top of them — the inverter "
        "keeps running on battery during an outage and keeps sending data, so those hours "
        "are recorded like any other. Adding outage hours to logged hours double-counts "
        "them.\n\n"
        "`Not logged` is the opposite case: no data arrived at all, because the logger or "
        "its connection was down. That is **not** an outage.")

    hours = mon[["month", "days", "calendar_h", "metered_h", "grid_up_h", "outage_h",
                 "unlogged_h", "data_completeness_pct"]].rename(columns={
        "month": "Month", "days": "Days",
        "calendar_h": "Hours in month (24 × days)",
        "metered_h": "Hours logged",
        "grid_up_h": "… of which grid ON",
        "outage_h": "… of which grid OFF (outage)",
        "unlogged_h": "Not logged (no data)",
        "data_completeness_pct": "Data completeness %"})
    st.dataframe(hours.style.format({
        "Hours in month (24 × days)": "{:,.0f}", "Hours logged": "{:,.1f}",
        "… of which grid ON": "{:,.1f}", "… of which grid OFF (outage)": "{:,.1f}",
        "Not logged (no data)": "{:,.1f}", "Data completeness %": "{:.1f}%"}),
        hide_index=True, use_container_width=True)

    _gap = float(mon["unlogged_h"].sum())
    _cal = float(mon["calendar_h"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Data completeness", f"{100 * (1 - _gap / _cal):.1f}%" if _cal else "—")
    c2.metric("Hours never logged", f"{_gap:,.0f} h")
    c3.metric("Worst month",
              str(mon.loc[mon["data_completeness_pct"].idxmin(), "month"])
              if len(mon) else "—",
              f"{mon['data_completeness_pct'].min():.0f}% logged" if len(mon) else "")
    if mon["data_completeness_pct"].min() < 90:
        st.warning(
            f"At least one month is below 90% logged. Totals for that month are "
            f"under-reported — read its per-day figures, not its totals.")

    st.markdown("###### Outage split and the refill opportunity")
    opp = mon[["month", "solar_outage_h", "offsolar_outage_h", "solar_outage_h_per_day",
               "outage_load_kwh", "headroom_ceiling_kwh", "pv_surplus_kwh",
               "recoverable_kwh", "realised_pv_to_batt_kwh"]].rename(columns={
        "month": "Month",
        "solar_outage_h": "Outage in solar hours",
        "offsolar_outage_h": "Outage at night",
        "solar_outage_h_per_day": "Solar-hour outage per day",
        "outage_load_kwh": "Energy used during outage (kWh)",
        "headroom_ceiling_kwh": "Bank could absorb (kWh)",
        "pv_surplus_kwh": "Spare PV actually existed (kWh)",
        "recoverable_kwh": "Realistically recoverable (kWh)",
        "realised_pv_to_batt_kwh": "Captured today (kWh)"})
    st.dataframe(opp.style.format({
        "Outage in solar hours": "{:,.1f}", "Outage at night": "{:,.1f}",
        "Solar-hour outage per day": "{:.2f}",
        "Energy used during outage (kWh)": "{:,.1f}",
        "Bank could absorb (kWh)": "{:,.1f}",
        "Spare PV actually existed (kWh)": "{:,.2f}",
        "Realistically recoverable (kWh)": "{:,.2f}",
        "Captured today (kWh)": "{:,.2f}"}), hide_index=True, use_container_width=True)
    st.caption(
        "Read the last four columns together. The bank can absorb far more than the spare "
        "PV that actually exists during an outage, so **PV surplus is the binding "
        "constraint, not battery size or charge rate**. A bigger bank would not move the "
        "recoverable figure.")
    st.bar_chart(mon.set_index("month")[["solar_outage_h", "offsolar_outage_h"]])
    st.caption(
        "A single annual outage probability cannot drive this controller. If daytime outage "
        "hours per day move by an order of magnitude between the dry season and the monsoon, "
        "a fixed reserve is wrong eleven months of the year. The reserve has to be a function "
        "of month — and ideally of a trailing window, so it tracks the actual feeder.")

    st.divider()
    st.markdown("##### Outage probability by hour of local day")
    hh = df.groupby(df["hour"].astype(int)).apply(
        lambda x: pd.Series({"outage_h": x.loc[x["outage"], "dt_h"].sum(),
                             "total_h": x["dt_h"].sum()}), include_groups=False)
    hh["outage_pct"] = 100 * hh["outage_h"] / hh["total_h"]
    st.bar_chart(hh["outage_pct"])

    st.divider()
    st.markdown("##### Islanding savings — where the money comes from")
    ig = st.radio("Breakdown by", ["month", "day", "cycle"], horizontal=True, key="islg",
                  format_func=str.capitalize)
    il = res["ledger"].copy()
    if ig == "month":
        il["_k"] = pd.to_datetime(il["date"]).dt.to_period("M").astype(str)
    elif ig == "day":
        il["_k"] = pd.to_datetime(il["date"]).dt.date.astype(str)
    else:
        il["_k"] = il["cycle_id"]
    ia = il.groupby("_k").agg(
        outage_h=("outage_h", "sum"),
        pv_kwh=("isl_kwh_pv", "sum"), batt_kwh=("isl_kwh_batt", "sum"),
        pv_direct=("isl_pv_direct", "sum"), battery=("isl_battery", "sum"),
        island_charged=("isl_island_charged", "sum"),
        islanding_total=("islanding_savings", "sum"),
        outage_backup=("outage_backup_value", "sum"),
    ).reset_index().rename(columns={"_k": ig})
    ia = ia[ia["islanding_total"].abs() + ia["outage_backup"].abs() > 0.005]
    ia["avg_rs_per_kwh"] = ia["islanding_total"] / (ia["pv_kwh"] + ia["batt_kwh"]).replace(0, np.nan)
    st.dataframe(ia.style.format({
        "outage_h": "{:,.1f}", "pv_kwh": "{:,.1f}", "batt_kwh": "{:,.1f}",
        "pv_direct": "{:,.0f}", "battery": "{:,.0f}", "island_charged": "{:,.0f}",
        "islanding_total": "{:,.0f}", "outage_backup": "{:,.0f}",
        "avg_rs_per_kwh": "{:,.2f}"}), hide_index=True, use_container_width=True)
    if ig != "day":
        st.bar_chart(ia.set_index(ig)[["pv_direct", "battery", "island_charged"]])
    st.download_button(f"Download islanding breakdown by {ig} (CSV)",
                       ia.to_csv(index=False).encode(), f"islanding_by_{ig}.csv")
    st.caption(
        "**pv_direct** is PV carrying the load during a daytime outage. **battery** is the bank "
        "carrying it — PV-charged energy earns the full avoided rate, but grid-charged energy "
        "earns only the SPREAD between the rate it displaces and what it cost, because that unit "
        "was already billed once. **island_charged** is energy PV put into the bank during an "
        "islanding window and the bank gave back later; it keeps its islanding identity whenever "
        "it leaves. Pricing all outage kWh at full retail regardless of origin is the single "
        "easiest way to overstate this bucket.")

    st.divider()
    st.markdown("##### How the outage units were billed — slab by slab")
    st.caption(
        "The arithmetic behind the figures above, written out rather than summarised. Outage "
        "energy never appears on a bill, so it is valued against what it WOULD have cost: had "
        "the grid been up, those units would have been additional consumption on top of the "
        "month's without-solar total, and a telescopic tariff puts additional units at the top "
        "of the stack. One row per slab the block passes through.")
    W = outage_slab_working(df, tariff)
    if not W.empty:
        Wv = W.rename(columns={
            "period": "Cycle", "base_units_no_solar": "Month units (no solar)",
            "slab": "Outage units land here",
            "energy_rate": "Energy ₹", "wheeling_rate": "+ Wheeling ₹",
            "fac_rate": "+ FAC ₹",
            "duty_per_kwh": f"+ Duty {tariff.duty_pct:.0f}% ₹",
            "all_in_rate": "= All-in ₹/kWh",
            "outage_units": "Outage kWh", "islanding_units": "of which islanding",
            "night_units": "of which night", "value": "Value ₹",
            "islanding_value": "Islanding share ₹"})
        st.dataframe(Wv.style.format({
            "Month units (no solar)": "{:,.1f}",
            "Energy ₹": "{:,.2f}", "+ Wheeling ₹": "{:,.2f}", "+ FAC ₹": "{:,.2f}",
            f"+ Duty {tariff.duty_pct:.0f}% ₹": "{:,.2f}", "= All-in ₹/kWh": "{:,.2f}",
            "Outage kWh": "{:,.2f}", "of which islanding": "{:,.2f}",
            "of which night": "{:,.2f}", "Value ₹": "{:,.0f}",
            "Islanding share ₹": "{:,.0f}"}), hide_index=True, use_container_width=True)
        st.caption(
            "The four rate columns add up to the all-in rate, so the duty is on the page rather "
            "than buried in a constant. Fixed charge and meter rent are deliberately absent — "
            "they do not vary with units, so avoiding a unit does not avoid them.")
        k = st.columns(3)
        k[0].metric("Total outage units", f"{W['outage_units'].sum():,.1f} kWh")
        k[1].metric("Valued at", rs(W["value"].sum()))
        k[2].metric("Islanding share", rs(W["islanding_value"].sum()))
        st.download_button("Download slab working (CSV)",
                           W.to_csv(index=False).encode(), "outage_slab_working.csv")
        st.info(
            f"**This table is the gross counterfactual — the savings bucket is lower, on "
            f"purpose.** It values every outage kWh at the full avoided rate. But "
            f"{T['outage_backup_kwh'] + T['islanding_kwh']:,.0f} kWh of it came out of the "
            f"battery, and some of that was charged from the grid at a real cost. Grid-charged "
            f"energy earns only the SPREAD between the rate it displaces and what it cost, "
            f"because that unit was already billed once. That is why islanding reports "
            f"{rs(T['islanding_savings'])} rather than "
            f"{rs(W['islanding_value'].sum())}.", icon="⚖️")

    st.divider()
    st.markdown("##### Episodes & threshold sensitivity")
    if not eps.empty:
        sens = []
        for thr in [0, 5, 10, 15, 30, 60]:
            kp = eps[eps["duration_h"] * 60 >= thr]
            sens.append(dict(min_minutes=thr, episodes=len(kp),
                             outage_hours=kp["duration_h"].sum(),
                             pct_of_hours=100 * kp["duration_h"].sum()
                             / max(eps["duration_h"].sum(), 1e-9)))
        st.dataframe(pd.DataFrame(sens).style.format(
            {"outage_hours": "{:,.1f}", "pct_of_hours": "{:.0f}"}),
            hide_index=True, use_container_width=True)
        st.caption(
            "Where you set the minimum duration changes the outage total materially, so it is a "
            "stated assumption rather than a hidden one. Very short zero-voltage runs are "
            "usually a sampling artefact; long ones with a data gap inside them are the device "
            "going offline, not the grid, and are split apart before counting.")
        st.markdown(f"**Longest episodes** (of {len(eps)})")
        st.dataframe(eps.nlargest(15, "duration_h")[
            ["start", "end", "duration_h", "type", "load_kwh", "pv_kwh", "bat_kwh"]
        ].style.format({"duration_h": "{:.2f}", "load_kwh": "{:.2f}",
                        "pv_kwh": "{:.2f}", "bat_kwh": "{:.2f}"}),
            hide_index=True, use_container_width=True)
        st.download_button("Download all episodes (CSV)", eps.to_csv(index=False).encode(),
                           "outage_episodes.csv")

# =====================================================================
# TAB 4 — CONTROL LOGIC
# =====================================================================
# =====================================================================
# TAB 4 — LOAD PATTERN (day vs night, by month)
# =====================================================================
with tabs[3]:
    st.subheader("When this household actually uses its electricity")
    st.caption(
        f"Split at the solar window ({sol_s:.0f}:00–{sol_e:.0f}:00 local). "
        "Everything outside it is counted as off-solar. Mean kW is energy divided by "
        "metered time, so logger gaps cannot inflate it.")

    _lp = df.copy()
    _lp["month"] = pd.PeriodIndex(pd.to_datetime(_lp["date"]), freq="M").astype(str)

    def _split(x):
        sol, off = x[x["solar_hr"]], x[~x["solar_hr"]]
        nd = x["date"].nunique()
        return pd.Series({
            "days": nd,
            "solar_kW": sol["load_kwh"].sum() / sol["dt_h"].sum() if sol["dt_h"].sum() else 0.0,
            "solar_kWh_day": sol["load_kwh"].sum() / nd if nd else 0.0,
            "offsolar_kW": off["load_kwh"].sum() / off["dt_h"].sum() if off["dt_h"].sum() else 0.0,
            "offsolar_kWh_day": off["load_kwh"].sum() / nd if nd else 0.0,
            "total_kWh_day": x["load_kwh"].sum() / nd if nd else 0.0,
        })

    LP = _lp.groupby("month").apply(_split, include_groups=False).reset_index()
    LP["solar_share_pct"] = np.where(
        LP["total_kWh_day"] > 0, 100 * LP["solar_kWh_day"] / LP["total_kWh_day"], np.nan)

    _s, _o = df[df["solar_hr"]], df[~df["solar_hr"]]
    _nd = df["date"].nunique()
    y_sol_kw = _s["load_kwh"].sum() / _s["dt_h"].sum() if _s["dt_h"].sum() else 0.0
    y_off_kw = _o["load_kwh"].sum() / _o["dt_h"].sum() if _o["dt_h"].sum() else 0.0
    y_share = 100 * _s["load_kwh"].sum() / df["load_kwh"].sum() if df["load_kwh"].sum() else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean load, solar hours", f"{y_sol_kw:.2f} kW",
              f"{_s['load_kwh'].sum() / _nd:.1f} kWh/day" if _nd else "")
    c2.metric("Mean load, off-solar", f"{y_off_kw:.2f} kW",
              f"{_o['load_kwh'].sum() / _nd:.1f} kWh/day" if _nd else "")
    c3.metric("Daytime share of consumption", f"{y_share:.0f}%")
    c4.metric("Total consumption", f"{df['load_kwh'].sum() / _nd:.1f} kWh/day" if _nd else "—",
              f"{df['load_kwh'].sum() * scale:,.0f} kWh/yr")

    st.markdown("##### Month by month")
    _disp = LP.rename(columns={
        "month": "Month", "days": "Days",
        "solar_kW": "Solar hours — mean kW",
        "solar_kWh_day": "Solar hours — kWh/day",
        "offsolar_kW": "Off-solar — mean kW",
        "offsolar_kWh_day": "Off-solar — kWh/day",
        "total_kWh_day": "Total kWh/day",
        "solar_share_pct": "Daytime share %"})
    st.dataframe(_disp.style.format({
        "Solar hours — mean kW": "{:.2f}", "Solar hours — kWh/day": "{:.1f}",
        "Off-solar — mean kW": "{:.2f}", "Off-solar — kWh/day": "{:.1f}",
        "Total kWh/day": "{:.1f}", "Daytime share %": "{:.0f}%"}),
        hide_index=True, use_container_width=True)
    st.download_button("Download load pattern (CSV)", LP.to_csv(index=False).encode(),
                       "load_pattern_by_month.csv", "text/csv")

    st.markdown("##### Energy per day — daytime vs night")
    st.bar_chart(LP.set_index("month")[["solar_kWh_day", "offsolar_kWh_day"]])

    st.markdown("##### Mean load by hour of local day")
    _hp = df.groupby(df["hour"].astype(int)).apply(
        lambda x: x["load_kwh"].sum() / x["dt_h"].sum() if x["dt_h"].sum() else 0.0,
        include_groups=False).rename("mean_kW")
    st.line_chart(_hp)

    _pk = LP.loc[LP["solar_kW"].idxmax()] if len(LP) else None
    _tr = LP.loc[LP["solar_kW"].idxmin()] if len(LP) else None
    if _pk is not None and _tr is not None and _tr["solar_kW"] > 0:
        st.info(
            f"**Why one reserve setting cannot work all year.** Daytime load runs at "
            f"{_tr['solar_kW']:.2f} kW in {_tr['month']} and {_pk['solar_kW']:.2f} kW in "
            f"{_pk['month']} — a factor of {_pk['solar_kW'] / _tr['solar_kW']:.1f}. The same "
            f"reserve that carries the house for hours in the lean month lasts minutes in "
            f"the peak one, so the reserve has to be a function of the month.")

    st.caption(
        "The daytime share matters more than the total. Where it is high, PV can serve load "
        "directly and the battery is needed less; where it is low, the same solar system "
        "has to route its output through the bank and pay the round-trip loss to be useful.")

with tabs[4]:
    st.subheader("What the controller would need to decide, and what it would need to know")
    st.caption(
        "This site is a simple hybrid: grid-priority, battery as pure backup. The decision "
        "logic below is what a reserve-planning controller would have to run, and the data "
        "requirement list is what it would have to be fed. Both are stated here so the "
        "savings model and the future controller share one definition of each quantity.")

    night = T["night_discharge_kwh"]
    st.metric("Battery discharged off-solar-hours with the grid up, over the whole period",
              f"{night:,.2f} kWh", help="The 'night discharge' item, measured.")
    if night < 1.0:
        st.info(
            f"**Measured night discharge is {night:.2f} kWh across {metered_days:.0f} days — "
            f"effectively zero.** On this site the battery only ever discharges when the grid "
            f"fails. So the night-discharge savings line is genuinely ₹0 today, and the entire "
            f"arbitrage opportunity the controller would unlock is still unrealised. That is "
            f"the honest framing: it is an opportunity size, not a saving already being "
            f"delivered.", icon="🔋")

    st.markdown("##### Decision 1 — how much to hold in reserve tonight")
    st.code(
        "reserve_kwh = max(\n"
        "    expected_outage_load_tonight(month, hour_profile) * confidence_factor,\n"
        "    hard_floor_kwh                      # protects battery health, never breached\n"
        ")\n"
        "drainable_kwh = max(0, soc_kwh - reserve_kwh)\n"
        "\n"
        "# Only drain what tomorrow can actually refill, or you wake up with an\n"
        "# empty bank and no daytime outage to refill it from:\n"
        "recoverable = min(\n"
        "    c_rate * usable_kwh * p10_solar_outage_hours(month),\n"
        "    forecast_surplus_pv_tomorrow\n"
        ")\n"
        "drain_tonight = min(drainable_kwh, recoverable)",
        language="python")

    st.markdown("##### Decision 2 — charge or export the surplus right now")
    st.code(
        "if grid_down:\n"
        "    # Islanded: export is physically impossible. Surplus PV charges the\n"
        "    # bank up to the C-rate limit; anything beyond that is curtailed.\n"
        "    charge(min(pv_surplus, c_rate * usable_kwh))\n"
        "elif soc_kwh < reserve_kwh:\n"
        "    charge(pv_surplus)                 # rebuilding reserve outranks export\n"
        "elif marginal_tariff_now > export_rate:\n"
        "    # A unit stored and used tonight is worth the tariff it displaces;\n"
        "    # a unit exported is worth only the surplus rate. In Maharashtra the\n"
        "    # second slab is ~5x the export rate, so storing wins decisively.\n"
        "    charge(pv_surplus)\n"
        "else:\n"
        "    export(pv_surplus)",
        language="python")

    st.markdown("##### Decision 3 — discharge or import right now")
    st.code(
        "if grid_down:\n"
        "    discharge_to_load()                # no choice; this is the backup job\n"
        "elif soc_kwh <= reserve_kwh:\n"
        "    import_from_grid()                 # reserve is untouchable\n"
        "elif marginal_tariff_now > stored_cost_basis + cycle_cost_per_kwh:\n"
        "    discharge_to_load()                # arbitrage clears the round-trip cost\n"
        "else:\n"
        "    import_from_grid()",
        language="python")
    st.caption(
        "`cycle_cost_per_kwh` is the throughput cost of the cycle itself — battery capital "
        "divided by warranted lifetime throughput. Omitting it makes every arbitrage look "
        "profitable and quietly spends battery life to save a few rupees. On a lead-acid bank "
        "it is often large enough to kill the arbitrage case entirely.")

    st.divider()
    st.markdown("##### Data this logic needs, and whether this export has it")
    have = pd.DataFrame([
        ("Per-row timestamp in confirmed local time", "PARTIAL",
         "Two clock columns disagree and drift; offset fitted from solar noon here, not confirmed from the platform."),
        ("PV DC input power", "YES", "PV1 populated; PV2 constant zero — confirm the second string is genuinely absent, not unwired."),
        ("Load real power", "YES", "AC output active power(W) is populated and is used directly."),
        ("Grid present/absent", "YES", "Inferred from grid voltage = 0; needs a minimum-duration rule."),
        ("Battery SOC", "NOISY", "Jumps implausibly between consecutive rows; unusable for differencing energy."),
        ("Battery charge / discharge current", "YES", "Used for battery energy; must exclude float current at 100% SOC."),
        ("Grid import / export (bidirectional meter)", "NO",
         "Not in this export. Export and import are inferred from the energy balance — the single largest unverified quantity in the model."),
        ("Sanctioned load, phase, tariff category", "NO", "Entered manually; must come from the customer's bill."),
        ("Real meter reading dates (billing cycle)", "NO", "Entered manually; slab position is wrong if the cycle boundary is wrong."),
        ("FAC per month as billed", "NO", "Entered manually; varies month to month."),
        ("Ambient / battery temperature", "NO", "Needed for honest capacity derating and cycle-life accounting."),
        ("Irradiance or a PV forecast", "NO", "Needed for Decision 1 — without it, tomorrow's refill is a guess."),
        ("Feeder outage history beyond this meter", "NO",
         "One meter-year gives a thin outage prior. Fleet data on the same feeder would sharpen it far more than any modelling."),
    ], columns=["Input", "Available here", "Note"])
    st.dataframe(have, hide_index=True, use_container_width=True)
    st.caption(
        "The three NOs that matter most are the bidirectional meter, the PV forecast and the "
        "feeder outage history. Without the meter, export revenue is modelled rather than "
        "measured. Without the other two, the reserve policy cannot be anything better than a "
        "seasonal lookup table — which, given the seasonality in the outage tab, would still "
        "beat a fixed reserve by a wide margin.")

# =====================================================================
# TAB 5 — ENERGY BALANCE
# =====================================================================
with tabs[5]:
    st.subheader("Measured energy balance")
    flows = dict(
        pv_generated=df["pv_kwh"].sum(), load_served=df["load_kwh"].sum(),
        pv_to_load=df["pv_to_load"].sum(), pv_to_battery=df["pv_to_batt"].sum(),
        pv_export=df["pv_export"].sum(), pv_curtailed=df["pv_curtailed"].sum(),
        battery_to_load=df["bat_to_load"].sum(), grid_to_load=df["grid_to_load"].sum(),
        grid_to_battery=df["grid_to_batt"].sum(), grid_import=df["grid_import"].sum(),
        float_current_excluded=df["float_kwh"].sum(),
    )
    fb = pd.DataFrame([dict(Flow=k.replace("_", " "), kWh=v) for k, v in flows.items()])
    st.dataframe(fb.style.format({"kWh": "{:,.1f}"}), hide_index=True, use_container_width=True)

    resid_load = flows["load_served"] - (flows["pv_to_load"] + flows["battery_to_load"]
                                         + flows["grid_to_load"] + df["unserved"].sum())
    resid_pv = flows["pv_generated"] - (flows["pv_to_load"] + flows["pv_to_battery"]
                                        + flows["pv_export"] + flows["pv_curtailed"])
    c = st.columns(2)
    c[0].metric("Load identity residual", f"{resid_load:,.3f} kWh")
    c[1].metric("PV identity residual", f"{resid_pv:,.3f} kWh")
    st.caption("Both residuals must be ~0. They are the proof that no kWh is counted twice "
               "or lost between the buckets.")

    st.divider()
    st.markdown("##### PV array size — estimated, because no export states it")
    A = estimate_array_size(df)
    inv = None
    for c in ("AC output rating apparent power(VA)",):
        if c in raw.columns:
            inv = pd.to_numeric(raw[c], errors="coerce").dropna()
            inv = float(inv.iloc[0]) / 1000.0 if len(inv) else None
    k = st.columns(4)
    k[0].metric("Inverter rating (stated)", f"{inv:.1f} kVA" if inv else "—",
                help="This is the only rating the file carries, and it says nothing about how "
                     "much array is wired to it. Do not quote it as the system size.")
    k[1].metric("From best peak", f"{A['est_from_peak_kwp']:.2f} kWp",
                help=f"Best observed DC power {A['peak_observed_kw']:.2f} kW ÷ 0.80 derate.")
    k[2].metric("From clear-day peak", f"{A['est_from_clear_day_kwp']:.2f} kWp",
                help=f"Median daily peak across the top 10% of days "
                     f"({A['clear_day_peak_kw']:.2f} kW) ÷ 0.80. The most reliable of the three.")
    k[3].metric("From annual yield",
                f"{A['est_from_yield_kwp'][0]:.2f}–{A['est_from_yield_kwp'][1]:.2f} kWp",
                help=f"{A['annual_kwh']:,.0f} kWh/yr ÷ 1,400–1,600 kWh/kWp.")
    if A["consistent"] and A["shortfall_pct"] <= 0:
        st.success(
            f"**The three methods agree, and the array is performing normally.** Best estimate "
            f"**{A['est_from_clear_day_kwp']:.1f} kWp**, returning "
            f"{A['implied_yield_at_clear_estimate']:,.0f} kWh/kWp a year — inside the healthy "
            f"band for an Indian rooftop. Confirm against the commissioning document before "
            f"quoting it, but nothing here suggests a problem.", icon="✅")
    elif A["shortfall_pct"] > 0:
        st.warning(
            f"**Best estimate {A['est_from_clear_day_kwp']:.1f} kWp, but it is returning only "
            f"{A['implied_yield_at_clear_estimate']:,.0f} kWh/kWp — about "
            f"{A['shortfall_pct']:.0f}% below what a healthy rooftop of that size returns.** "
            f"Worth checking for shading, soiling or a weak string. A shortfall of that size is "
            f"usually worth more to the customer than anything in the savings table, and it is "
            f"invisible unless someone compares yield against size.", icon="🔆")
    if A["second_string_dead"]:
        st.caption(
            "`PV2 Input Power(W)` reads exactly zero on every row of the file. That is normal if "
            "only one string was ever wired, and a real fault if a second string exists but its "
            "MPPT or wiring has failed — in which case roughly half the array is producing "
            "nothing. The telemetry cannot tell these apart; a site photo or the commissioning "
            "BOM can, in a minute.")
    st.caption(
        "Why three methods: peak-based estimates say what the array CAN do on its best interval, "
        "while the yield estimate says what it DOES over a year. When they agree the size is "
        "settled; when the peak estimates sit well above the yield estimate, the array is larger "
        "than its output justifies and the gap is the finding.")

    st.divider()
    st.markdown("##### Battery round trip")
    c = st.columns(4)
    c[0].metric("Energy into bank", f"{rt['charged_kwh']:,.1f} kWh")
    c[1].metric("Energy out of bank", f"{rt['discharged_kwh']:,.1f} kWh")
    c[2].metric("Measured round trip", f"{100 * rt['eff']:.0f}%")
    c[3].metric("Applied to valuation", f"{100 * rte:.0f}%",
                help="Used only to value stored energy, never to adjust a measured kWh.")
    st.caption(
        f"Both legs are metered, so this is observed, not assumed. {rt['float_excluded_kwh']:,.0f} "
        f"kWh of float/absorption current at 100% SOC is excluded from 'into bank' — including it "
        f"would push the figure far below anything physically possible, which is how you can tell "
        f"the uncorrected number is wrong rather than merely pessimistic. A plausible lead-acid "
        f"result is roughly 70–85%; well outside that means the charge or discharge channel needs "
        f"checking at the device, not patching in the model.")
    st.info(
        f"**Where the efficiency is actually used.** Nowhere on the energy side — applying an "
        f"assumed efficiency to measured kWh would double-count a loss the meter already saw. It "
        f"is used on the money side: a grid unit bought at the marginal rate only delivers "
        f"{100 * rte:.0f}% of itself to the load, so its cost basis is divided by {rte:.2f} before "
        f"any discharge is credited. At this efficiency a unit bought at ₹6.45 costs "
        f"₹{6.45 / rte:.2f} by the time it reaches the load — which is what decides whether night "
        f"arbitrage is worth doing at all.", icon="🔁")
    ud = float(df["unexplained_discharge"].sum())
    if ud > 0.01:
        st.warning(
            f"**{ud:,.1f} kWh of measured battery discharge does not reconcile under PV-first "
            f"dispatch** — PV alone already covered the load in those intervals. This is "
            f"5-minute averaging hiding sub-interval crossings, not missing energy. It is left "
            f"unattributed rather than pushed into a bucket. Switching to battery-first dispatch "
            f"in the sidebar makes it vanish and moves net savings by about 0.1%.")
    if T["unattributed_discharge_kwh"] > 0.5:
        st.warning(
            f"{T['unattributed_discharge_kwh']:.1f} kWh of measured discharge could not be traced "
            f"to charging seen in this file — the bank started the period with energy already in "
            f"it. That energy is reported here and left unvalued rather than being credited to a "
            f"bucket on an assumption.")

    st.divider()
    st.markdown("##### Monthly energy")
    _d = df.copy()
    _d["month"] = _d["local"].dt.to_period("M").astype(str)
    _d["outage_dt"] = _d["dt_h"].where(_d["outage"], 0.0)
    me = _d.groupby("month").agg(
        days=("date", "nunique"), pv=("pv_kwh", "sum"), load=("load_kwh", "sum"),
        pv_to_load=("pv_to_load", "sum"), export=("pv_export", "sum"),
        curtailed=("pv_curtailed", "sum"), grid_import=("grid_import", "sum"),
        bat_out=("bat_to_load", "sum"), outage_h=("outage_dt", "sum"),
    ).reset_index()
    st.dataframe(me.style.format({c: "{:,.1f}" for c in me.columns
                                  if me[c].dtype.kind in "fc"}),
                 hide_index=True, use_container_width=True)
    st.line_chart(me.set_index("month")[["pv", "load", "grid_import"]])

# =====================================================================
# TAB 6 — DATA INTEGRITY
# =====================================================================
with tabs[6]:
    st.subheader("Data integrity")
    icon = {"critical": "🔴", "warning": "🟠", "info": "🟢"}
    for f in findings:
        with st.expander(f"{icon[f['severity']]} {f['area']} — {f['severity'].upper()}",
                         expanded=(f["severity"] == "critical")):
            st.markdown(f"**Finding.** {f['finding']}")
            st.markdown(f"**Why it matters.** {f['impact']}")

# =====================================================================
# TAB 7 — AUDIT TRAIL
# =====================================================================
with tabs[7]:
    st.subheader("Audit trail")
    st.markdown("##### Assumptions in force")
    assume = pd.DataFrame([
        ("Tariff", state), ("Tariff source", tariff.source),
        ("Timestamp column", ts_choice),
        ("Local-time offset applied", f"{offset:+.3f} h ({'fitted from solar noon' if auto_off else 'manual'})"),
        ("Site longitude for solar noon", f"{lon:.1f}°E"),
        ("Solar-hour window", f"{sol_s:.1f}–{sol_e:.1f} local"),
        ("Load channel", df.attrs.get("load_source", "")),
        ("Dispatch order", "PV first, then battery/grid" if dispatch == "pv_first"
         else "Battery first (sensitivity)"),
        ("Grid down at/below", f"{out_v:.1f} V"),
        ("Minimum outage duration", f"{min_out:.0f} min"),
        ("Device-offline gap threshold", "30 min (energy integration suspended)"),
        ("Battery", f"{bat_ah:.0f} Ah × {bat_v:.0f} V = {cap:.2f} kWh nameplate; "
                    f"× {dod:.0%} DoD = {cap * dod:.2f} kWh usable"),
        ("Charge rate limit", f"{crate:.2f}C of NAMEPLATE = {crate * cap:.2f} kW "
                              f"({crate * bat_ah:.0f} A)"),
        ("Round-trip efficiency", f"{rte:.3f} ({rte_mode}) — applied to stored-energy "
                                  f"valuation only, never to a measured kWh"),
        ("Float current at 100% SOC", "counted" if keep_float else "excluded from charge energy"),
        ("Export settlement", settle), ("Surplus export rate", f"{rs(export_rate, 2)}/kWh"),
        ("Outage energy priced at", "top of the month's stack (incremental)"
         if out_price == "incremental" else "slab position at that moment (chronological)"),
        ("Billing cycle", f"{cyc_mode} (anchor day {anchor})" if cyc_mode == "monthly"
         else f"{cyc_days}-day blocks"),
        ("Metered time", f"{metered_days:.1f} days"),
        ("Annualisation factor", f"×{scale:.3f} (365 / metered days)"),
    ], columns=["Assumption", "Value"])
    st.dataframe(assume, hide_index=True, use_container_width=True)

    st.markdown("##### Accounting rules")
    st.markdown(
        "- **Baseline counterfactual** counts household demand only while the grid is up. A "
        "house with no solar and no battery is dark during an outage and is billed nothing, so "
        "outage-time load must not inflate the baseline bill. Its value lives in the islanding "
        "and backup buckets instead — never in both.\n"
        "- **Marginal rate** = (slab + wheeling + FAC) × (1 + duty). Fixed charges and meter "
        "rent are excluded from any per-kWh valuation and appear only in the displayed bills.\n"
        "- **Slab position** is tracked on cycle-cumulative units — separately for the baseline "
        "stream and the actual import stream, because the two sit in different slabs once solar "
        "cuts import. Slabs never reset at midnight.\n"
        "- **Battery provenance** is tracked in three buckets (island-charged, PV-charged, "
        "grid-charged with a running ₹/kWh cost basis) and discharge draws proportionally. "
        "Grid-charged energy is credited only the spread between the rate it displaces and the "
        "rate it cost, since it was already billed once.\n"
        "- **Export during an outage is zero.** An islanded inverter cannot export; surplus PV "
        "is curtailed. Crediting it would invent revenue that was physically unearnable.\n"
        "- **Off-solar outage backup** is reported outside net savings by default.")

    st.markdown("##### Row-level ledger")
    st.caption("Every row's contribution to every bucket, at the rate that applied at that "
               "moment. This is the file to hand to anyone who wants to re-derive the totals.")
    st.download_button("Download full row-level ledger (CSV)",
                       res["ledger"].to_csv(index=False).encode(), "savings_ledger.csv")
    st.dataframe(res["ledger"].head(500), hide_index=True, use_container_width=True)