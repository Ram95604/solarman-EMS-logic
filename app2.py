import streamlit as st
import pandas as pd
import numpy as np
from datetime import time as dtime

st.set_page_config(page_title="Solar Inverter Time-Series Simulator", layout="wide")

DT_HOURS = 5.0 / 60.0                       # fixed 5-minute interval, per strict spec
SLAB_TIERS = [(100, 7.5), (300, 11.5), (500, 15.5), (float("inf"), 17.5)]  # (upper bound kWh, Rs/kWh)

BATTERY_PRESETS = {
    "Lithium (48V / 314Ah)": dict(name="Lithium", capacity_kwh=15.07, dod=0.80, efficiency=0.95),
    "Lead-Acid (48V / 200Ah)": dict(name="Lead-Acid", capacity_kwh=9.6, dod=0.50, efficiency=0.80),
}
METRIC_ORDER = [
    "pv_available", "total_demand", "load_served", "pv_to_load", "pv_export", "batt_to_load",
    "grid_to_battery", "grid_import", "unserved",
    "solar_savings", "tod_optimization", "outage_savings", "solar_earning", "net_savings",
]
METRIC_LABELS = {
    "pv_available": "PV Available (kWh)",
    "total_demand": "Total Household Demand (kWh)",
    "load_served": "Total Load Served (kWh)",
    "pv_to_load": "PV directly to Load (kWh)",
    "pv_export": "PV Export (kWh)",
    "batt_to_load": "Battery to Load (kWh)",
    "grid_to_battery": "Grid → Battery Charging (kWh)",
    "grid_import": "Grid Import (kWh)",
    "unserved": "Unserved Load (kWh)",
    "solar_savings": "Solar Savings (Rs)",
    "tod_optimization": "TOD Optimization Savings (Rs)",
    "outage_savings": "Outage Savings (Rs)",
    "solar_earning": "Solar Earning — PV Export (Rs)",
    "net_savings": "Net Savings (Rs)",
}
INVERTER_ORDER = ["grid_tie", "dumb", "smart"]
INVERTER_LABELS = {"grid_tie": "Grid-Tie", "dumb": "Dumb Hybrid", "smart": "Smart Hybrid"}


# =====================================================================
# LEAD-ACID LIVE SOC — COULOMB COUNTING (NEW — additive only, does not
# touch any existing formula above/below). This is for the LIVE battery
# telemetry payload from the field devices:
#     status / chemistry / voltage / soc / chargingCurrent /
#     dischargingCurrent / backupTime
# — a different pipeline from the historical-CSV PV/load simulator in this
# file. Lithium's `soc` field already comes fused from its own BMS
# (coulomb counting + OCV recalibration + balancing done onboard), so it
# can be trusted directly and needs no extra formula here. Lead-acid
# telemetry only gives voltage + charge/discharge current, and lead-acid
# voltage sags/rises too much under load to trust a voltage-only SOC
# mid-cycle, so its SOC must be tracked via coulomb (Ah) counting instead —
# exactly as specified by the manager:
#
#     SOC2 = SOC1 - (Current * Time / Capacity) * 100
#
# with the anchoring rule: "wait for full charge of the battery, then take
# SOC as 100%; after that, keep incrementing/decrementing SOC based on Ah
# in/out." Coulomb counting alone has no absolute reference and drifts over
# time (current-sensor offset, integration error, self-discharge,
# temperature), so it must be periodically re-anchored to a known point —
# a detected full charge is that point.
# =====================================================================
LEAD_ACID_STATUS_NONE = 0
LEAD_ACID_STATUS_DISCHARGING = 1
LEAD_ACID_STATUS_CHARGING = 2

LEAD_ACID_FULL_CHARGE_VOLTAGE_DEFAULT = 54.0     # float/absorption voltage for a 48V bank — tune to the charger's real set-point
LEAD_ACID_FULL_CHARGE_TAPER_FRACTION = 0.02      # "full" once charging current tapers below 2% of capacity_Ah

# --- Coulombic charge efficiency & Peukert correction (both LEAD-ACID SPECIFIC) ---
# These are NOT applied by default inside the function below (defaults = no-op,
# so the previously verified manager's worked example, 80% -> 40%, is unchanged
# unless these are explicitly turned on). The UI panel below sets realistic
# defaults for actual field use. See the accompanying explanation for why each
# one matters for lead-acid but not lithium.
LEAD_ACID_CHARGE_EFFICIENCY_DEFAULT = 0.85   # ~80-90% typical for flooded/AGM lead-acid (see notes below)

# Peukert exponent + reference discharge rate — CALIBRATED from the actual vendor
# spec sheet for this fleet's 200Ah lead-acid battery (both the 24V and 48V wired
# variants, Small/Medium/Heavy backup points — 6 data points total). Fitting
# ln(t) = ln(Cp) - k*ln(I) by least squares gives k=1.454, Cp=791.5 (A^k * h),
# and predicts all 6 vendor-listed "Backup time from Peukart equation" values to
# within rounding (e.g. 24V/Small: predicted 5.37h vs vendor's 5.4h). This is a
# MUCH stronger derating than a generic guess — this specific battery line loses
# usable capacity fast at higher discharge currents. The nameplate 200Ah figure
# corresponds to roughly a 9.7-hour discharge rate (~20.7A), close to a C10
# rating rather than the more commonly assumed C20 — that's the reference rate
# used below. Re-fit these two numbers from the datasheet if a different lead-
# acid model/vendor is used.
LEAD_ACID_PEUKERT_EXPONENT_DEFAULT = 1.45
LEAD_ACID_PEUKERT_REFERENCE_RATE_HOURS_DEFAULT = 9.7   # capacity_ah / this = reference discharge current

# Lead-acid sags under load; the same vendor sheet shows the pack's average
# discharge voltage as ~95% of nominal for BOTH the 24V (22.8V) and 48V (45.6V)
# wiring of this battery. Lithium holds voltage much flatter under load, so its
# default stays at 1.0 (no sag) unless you have the pack's actual figure.
LEAD_ACID_DISCHARGE_VOLTAGE_SAG_DEFAULT = 0.95
LITHIUM_DISCHARGE_VOLTAGE_SAG_DEFAULT = 1.0

# Extra system/inverter derate applied on top of the DoD-limited backup time.
# The vendor sheet's last row ("Backup considering efficiency/losses") is
# consistently ~87-94% of the DoD row across all 6 rows shown — ~0.90-0.92
# is a reasonable single representative value; tune per actual inverter spec.
SYSTEM_EFFICIENCY_LOSS_DEFAULT = 0.92


def estimate_backup_time_hours(
    load_watts: float,
    nominal_voltage: float,
    capacity_ah: float,
    dod: float,
    discharge_voltage_sag: float = 1.0,
    peukert_exponent: float = 1.0,
    peukert_reference_rate_hours: float = None,
    system_efficiency: float = 1.0,
) -> dict:
    """
    Watts -> backup-time-in-hours estimator, built to mirror (and CORRECT) the
    methodology in the vendor spec sheet:

      1. avg_discharge_voltage = nominal_voltage * discharge_voltage_sag
         (accounts for lead-acid's voltage sag under load; use sag=1.0 for
         lithium, which stays much flatter).
      2. current_a = load_watts / avg_discharge_voltage   <-- Amps = Watts / Volts.
         The vendor sheet's LITHIUM table appears to instead compute this as
         load_watts / capacity_ah (Watts / Amp-hours, a unit mismatch): e.g.
         its "Maximum Current" of 5A for a 1500W/24V/314Ah row is exactly
         1500/314 = 4.78 ≈ 5, NOT 1500/24 = 62.5A (the physically correct
         current). Every downstream Lithium backup-time figure in that sheet
         inherits this error and comes out ~12-13x too optimistic. This
         function always divides by VOLTAGE, so it does not reproduce that
         bug — use it to see the corrected numbers.
      3. Peukert derating (lead-acid only — leave peukert_exponent=1.0, i.e.
         no-op, for lithium):
             reference_current_a = capacity_ah / peukert_reference_rate_hours
             effective_capacity_ah = capacity_ah / (current_a / reference_current_a) ** (peukert_exponent - 1)
         applied only when current_a exceeds the reference current (matches
         the coulomb-counting module above, and is exactly the calibration
         fitted from the vendor's own lead-acid data — see the constants above).
      4. backup_time_plain_h = effective_capacity_ah / current_a
      5. backup_time_with_dod_h = backup_time_plain_h * dod
      6. backup_time_with_losses_h = backup_time_with_dod_h * system_efficiency

    Returns every intermediate value so the arithmetic is fully auditable.
    """
    avg_v = nominal_voltage * discharge_voltage_sag
    if avg_v <= 0 or load_watts <= 0:
        return dict(avg_discharge_voltage=avg_v, current_a=0.0, reference_current_a=None,
                    effective_capacity_ah=capacity_ah, backup_time_plain_h=float("inf"),
                    backup_time_with_dod_h=float("inf"), backup_time_with_losses_h=float("inf"))

    current_a = load_watts / avg_v

    reference_current_a = None
    effective_capacity_ah = capacity_ah
    if peukert_reference_rate_hours and peukert_exponent != 1.0 and capacity_ah > 0:
        reference_current_a = capacity_ah / peukert_reference_rate_hours
        if current_a > reference_current_a:
            effective_capacity_ah = capacity_ah / (current_a / reference_current_a) ** (peukert_exponent - 1)

    backup_time_plain_h = effective_capacity_ah / current_a
    backup_time_with_dod_h = backup_time_plain_h * dod
    backup_time_with_losses_h = backup_time_with_dod_h * system_efficiency

    return dict(
        avg_discharge_voltage=avg_v,
        current_a=current_a,
        reference_current_a=reference_current_a,
        effective_capacity_ah=effective_capacity_ah,
        backup_time_plain_h=backup_time_plain_h,
        backup_time_with_dod_h=backup_time_with_dod_h,
        backup_time_with_losses_h=backup_time_with_losses_h,
    )


def estimate_soc_leadacid_coulomb_counting(
    prev_soc_pct: float,
    status: int,
    charging_current_a: float,
    discharging_current_a: float,
    dt_hours: float,
    capacity_ah: float,
    charge_efficiency: float = 1.0,
    peukert_exponent: float = 1.0,
    nominal_discharge_rate_a: float = None,
) -> float:
    """
    Live SOC update for a LEAD-ACID battery via coulomb (Ah) counting.
    Once SOC has been anchored to 100% at a detected full charge (see
    `is_leadacid_fully_charged` below), every subsequent reading only
    increments (charging) or decrements (discharging) SOC from there based
    on Ah in/out — never recomputed from voltage mid-cycle. Base formula:

        SOC2 = SOC1 - (Current * Time / Capacity) * 100

    `status` (from the telemetry payload) selects which current column is
    the live one and which sign to apply:
      - status == 2 (Charging)    -> SOC increases using `charging_current_a`
      - status == 1 (Discharging) -> SOC decreases using `discharging_current_a`
      - status == 0 (None)        -> SOC unchanged (no current flowing)

    Two OPTIONAL lead-acid-specific corrections, both off by default:

      1. `charge_efficiency` (coulombic / Ah round-trip efficiency, applied
         ONLY to the charging leg): not every Amp-hour pushed into a
         lead-acid battery becomes usable capacity — some is lost to
         gassing/heat, especially near full charge. Lead-acid is typically
         ~80-90% here; lithium is close to 98-99% and its `soc` telemetry
         field is BMS-fused already, so this correction is lead-acid-only.
         There is no separate "discharge efficiency" term: the coulombic
         loss on the way OUT is close to 100% for lead-acid (the Ah that
         leave the battery are the Ah that leave, full stop) — the round
         trip loss lives almost entirely on the charging side. This is
         different from ENERGY (kWh) round-trip efficiency, which also
         reflects the voltage gap between charge and discharge; coulomb
         counting only needs the Ah (coulombic) efficiency, not the kWh one.

      2. Peukert derating (`peukert_exponent` + `nominal_discharge_rate_a`):
         a lead-acid battery's USABLE capacity is not a fixed number — it
         shrinks at higher discharge currents (an effect lithium barely
         shows, Peukert exponent ~1.0). Pass `nominal_discharge_rate_a`
         (the rate `capacity_ah` is rated at, e.g. the C/20 rate =
         capacity_ah/20) to enable this; leaving it as None disables the
         correction entirely (matches the plain formula).

    This function does NOT perform the 100% anchor itself — check
    `is_leadacid_fully_charged(...)` on the same telemetry row first; when
    it returns True, set SOC to 100.0 directly for that row instead of
    calling this function (see the worked example in the UI below).
    """
    if capacity_ah <= 0:
        return prev_soc_pct

    if status == LEAD_ACID_STATUS_CHARGING:
        delta_soc_pct = (charging_current_a * charge_efficiency * dt_hours / capacity_ah) * 100.0
    elif status == LEAD_ACID_STATUS_DISCHARGING:
        effective_capacity_ah = capacity_ah
        if nominal_discharge_rate_a and discharging_current_a > 0 and peukert_exponent != 1.0:
            rate_ratio = discharging_current_a / nominal_discharge_rate_a
            if rate_ratio > 1:
                effective_capacity_ah = capacity_ah / (rate_ratio ** (peukert_exponent - 1))
        delta_soc_pct = -(discharging_current_a * dt_hours / effective_capacity_ah) * 100.0
    else:
        delta_soc_pct = 0.0

    new_soc_pct = prev_soc_pct + delta_soc_pct
    return float(np.clip(new_soc_pct, 0.0, 100.0))


def is_leadacid_fully_charged(
    status: int,
    voltage: float,
    charging_current_a: float,
    capacity_ah: float,
    float_voltage_threshold: float = LEAD_ACID_FULL_CHARGE_VOLTAGE_DEFAULT,
    taper_current_frac: float = LEAD_ACID_FULL_CHARGE_TAPER_FRACTION,
) -> bool:
    """
    Returns True when telemetry shows the standard lead-acid "battery is
    full" signal: actively charging, sitting at/above the float/absorption
    voltage, AND the charging current has tapered down near zero. This is
    the manager-specified anchor point ("wait for full charge... then take
    SOC as 100%") — call this every row; when it returns True, snap SOC to
    100.0 and resume coulomb counting (the function above) from there.
    """
    if capacity_ah <= 0 or status != LEAD_ACID_STATUS_CHARGING:
        return False
    taper_threshold_a = taper_current_frac * capacity_ah
    return (voltage >= float_voltage_threshold) and (0 <= charging_current_a <= taper_threshold_a)


def compute_leadacid_soc_series(
    telemetry_df: pd.DataFrame,
    capacity_ah: float,
    float_voltage_threshold: float = LEAD_ACID_FULL_CHARGE_VOLTAGE_DEFAULT,
    taper_current_frac: float = LEAD_ACID_FULL_CHARGE_TAPER_FRACTION,
    initial_soc_pct: float = None,
    charge_efficiency: float = 1.0,
    peukert_exponent: float = 1.0,
    nominal_discharge_rate_a: float = None,
) -> pd.DataFrame:
    """
    Runs `estimate_soc_leadacid_coulomb_counting()` + `is_leadacid_fully_charged()`
    row-by-row over a live lead-acid telemetry log and returns a copy of
    `telemetry_df` with an added `SOC_CoulombCounting` column.

    Required columns in `telemetry_df` (matching the field device payload):
      'Date Time' (any parseable timestamp column of that name), 'status',
      'voltage', 'chargingCurrent', 'dischargingCurrent'. An optional 'soc'
      column (the device's own onboard estimate, if it reports one) is only
      used to seed the very first row when `initial_soc_pct` isn't given.

    Anchoring, per the manager's approach: start at `initial_soc_pct` if
    given, else at the first row's own reported `soc` if present, else a
    neutral 50% guess — until the first full-charge event anchors SOC to
    100%, after which it is purely incremented/decremented from Ah in/out.
    """
    df = telemetry_df.copy()
    df.columns = [c.strip() for c in df.columns]
    required = ["Date Time", "status", "voltage", "chargingCurrent", "dischargingCurrent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found columns: {list(df.columns)}")

    df["Date Time"] = pd.to_datetime(df["Date Time"], errors="coerce")
    df = df.dropna(subset=["Date Time"]).sort_values("Date Time").reset_index(drop=True)

    if initial_soc_pct is None:
        if "soc" in df.columns and len(df) > 0 and pd.notna(df["soc"].iloc[0]):
            initial_soc_pct = float(df["soc"].iloc[0])
        else:
            initial_soc_pct = 50.0

    dt_hours_series = df["Date Time"].diff().dt.total_seconds() / 3600.0
    if len(dt_hours_series) > 0:
        dt_hours_series.iloc[0] = 0.0  # no elapsed time before the first sample

    soc_values = []
    prev_soc = initial_soc_pct
    for i in range(len(df)):
        status = int(df["status"].iloc[i])
        voltage = float(df["voltage"].iloc[i])
        i_chg = float(df["chargingCurrent"].iloc[i])
        i_dis = float(df["dischargingCurrent"].iloc[i])
        dt_h = float(dt_hours_series.iloc[i])

        if is_leadacid_fully_charged(status, voltage, i_chg, capacity_ah, float_voltage_threshold, taper_current_frac):
            prev_soc = 100.0
        else:
            prev_soc = estimate_soc_leadacid_coulomb_counting(
                prev_soc, status, i_chg, i_dis, dt_h, capacity_ah,
                charge_efficiency=charge_efficiency, peukert_exponent=peukert_exponent,
                nominal_discharge_rate_a=nominal_discharge_rate_a,
            )
        soc_values.append(prev_soc)

    df["SOC_CoulombCounting"] = soc_values
    return df


# =====================================================================
# RAW-FILE DATA SANITY REPORT (NEW — additive only, read-only diagnostics)
# Runs on the RAW uploaded CSV, before preprocess() touches anything. Does
# NOT change preprocess()'s existing behaviour — this only surfaces findings
# for a human (developer) to review and decide on, exactly as requested:
# every finding here was derived from actually analysing a real 7-day
# Maharashtra field export (Aug 1-7, 2026, a Grid-Tie inverter), so these are
# checks proven to catch real issues, not hypothetical ones.
# =====================================================================
import re as _re


def run_data_sanity_report(raw_df: pd.DataFrame, grid_v_min: float = None, grid_v_max: float = None) -> list:
    """
    Returns a list of finding dicts: {severity, area, finding, decision}.
    severity in {"critical", "warning", "info"}. Pure read-only inspection —
    never raises, never modifies `raw_df`.
    """
    findings = []
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]

    # --- 1. Duplicate column headers in the source file ---
    dupe_cols = [c for c in df.columns if _re.search(r"\.\d+$", c)]
    if dupe_cols:
        findings.append(dict(
            severity="warning", area="Column headers",
            finding=f"Source file has a duplicate header name — pandas auto-renamed the repeat as "
                    f"{dupe_cols}. The original column name appears more than once in the raw CSV.",
            decision="Not currently used by any of our required columns, so harmless today — but if a "
                     "future required column name ever collides with a duplicated header, this rename "
                     "would silently shift which column gets picked up. Worth a defensive check upstream.",
        ))

    # --- 2. Timestamp column grain check (THE critical one — found in real data) ---
    # A usable per-row timestamp column needs BOTH date variation (across a multi-day
    # file) AND time-of-day variation. Checking only one of the two is exactly how the
    # real field export's 'Time' column (HH:MM:SS with no date) could get mistaken for
    # a full timestamp — pandas silently defaults missing date info to "today", which
    # would corrupt sorting/interval math across a multi-day file without erroring.
    candidate_ts_cols = [c for c in df.columns if ("time" in c.lower() or "date" in c.lower())]
    ts_analysis = {}
    for c in candidate_ts_cols:
        parsed = pd.to_datetime(df[c], errors="coerce")
        n_valid = int(parsed.notna().sum())
        if n_valid < 2:
            continue
        dropped = parsed.dropna()
        n_distinct_times = int(dropped.dt.time.nunique())
        n_distinct_dates = int(dropped.dt.date.nunique())
        ts_analysis[c] = dict(parsed=parsed, n_valid=n_valid,
                               n_distinct_times=n_distinct_times, n_distinct_dates=n_distinct_dates)

        if n_distinct_times <= 1:
            findings.append(dict(
                severity="critical", area="Timestamp columns",
                finding=f"Column '{c}' parses as a valid date, but shows almost NO time-of-day variation "
                        f"({n_distinct_times} distinct time-of-day value(s) across {n_valid} rows) — it "
                        f"looks DATE-ONLY (e.g. '01-08-2026'), not a true per-row timestamp.",
                decision=f"DO NOT use '{c}' as the row timestamp for interval/energy integration — every "
                         f"row on the same day would collapse to the same instant, destroying dt_hours and "
                         f"every downstream kWh figure. This is exactly what happens with the real field "
                         f"export: it has THREE time-related columns — 'Timestamp' (date-only), 'Time' "
                         f"(time-of-day only), and 'Date Time' (the real combined per-row timestamp). Use "
                         f"the column that actually varies at BOTH date and time-of-day as the authoritative "
                         f"timestamp — confirm which column name your pipeline points at before trusting "
                         f"any results.",
            ))
        elif n_distinct_dates <= 1 and n_valid > 20:
            findings.append(dict(
                severity="critical" if n_valid > 100 else "warning", area="Timestamp columns",
                finding=f"Column '{c}' varies by time-of-day ({n_distinct_times} distinct values) but its "
                        f"DATE component never changes ({n_distinct_dates} unique date) across {n_valid} "
                        f"rows — it looks like a time-only field (e.g. '00:02:20') that pandas defaulted to "
                        f"today's date when parsing, not a real per-row calendar date.",
                decision=f"DO NOT sort/diff by '{c}' alone across a multi-day file — rows from different "
                         f"days sharing the same clock time would collide or sort incorrectly, corrupting "
                         f"every interval and energy figure just as badly as the date-only case above. Only "
                         f"safe when combined with a genuine date column, or when the whole file covers a "
                         f"single day.",
            ))
        else:
            findings.append(dict(
                severity="info", area="Timestamp columns",
                finding=f"Column '{c}' varies across both date ({n_distinct_dates} distinct dates) and "
                        f"time-of-day ({n_distinct_times} distinct values, {n_valid} rows) — looks like a "
                        f"genuine per-row timestamp.",
                decision=f"Safe to use '{c}' as the authoritative timestamp column.",
            ))

    # pick the best timestamp column for the remaining checks: must vary across BOTH
    # date and time to be trustworthy; break ties by finest overall granularity.
    best_ts_col, best_parsed, best_n_distinct = None, None, -1
    for c, info in ts_analysis.items():
        if info["n_distinct_dates"] <= 1 or info["n_distinct_times"] <= 1:
            continue  # not a safe full-timestamp candidate, skip for interval/outage analysis
        n_distinct_full = int(info["parsed"].dropna().nunique())
        if n_distinct_full > best_n_distinct:
            best_ts_col, best_parsed, best_n_distinct = c, info["parsed"], n_distinct_full

    if best_ts_col is not None and best_n_distinct > 1:
        s = best_parsed.dropna().sort_values()
        interval_min = s.diff().dt.total_seconds() / 60.0
        interval_min = interval_min.dropna()
        if len(interval_min) > 0:
            median_iv = float(interval_min.median())
            n_gt6 = int((interval_min > 6).sum())
            n_gt30 = int((interval_min > 30).sum())
            max_iv = float(interval_min.max())
            findings.append(dict(
                severity="info", area="Sampling interval",
                finding=f"Using '{best_ts_col}': median interval ≈{median_iv:.2f} min (nominal cadence is "
                        f"~5 min but real logger jitter is normal). {n_gt6} interval(s) exceed 6 min, "
                        f"{n_gt30} exceed 30 min, longest gap ≈{max_iv:.1f} min.",
                decision="Confirms the gap-tolerance (6 min) / max-gap (30 min) defaults in preprocess() "
                         "are reasonable: normal jitter stays under 6 min, and genuine device-offline "
                         "windows (tens of minutes) get correctly excluded from energy integration rather "
                         "than extrapolated.",
            ))

    # --- 3. Grid voltage range vs configured Grid_Status band ---
    gv_col = next((c for c in df.columns if c.lower().startswith("grid voltage")), None)
    if gv_col is not None:
        gv = pd.to_numeric(df[gv_col], errors="coerce").dropna()
        if len(gv) > 0:
            gv_min_obs, gv_max_obs = float(gv.min()), float(gv.max())
            n_zero = int((gv == 0).sum())
            msg = (f"'{gv_col}' observed range: {gv_min_obs:.1f}V to {gv_max_obs:.1f}V across {len(gv):,} rows. "
                   f"{n_zero} row(s) read exactly 0V (candidate outage intervals).")
            sev = "info"
            decision = "Observed range fits comfortably inside the configured Grid_Status band."
            if grid_v_max is not None and gv_max_obs > grid_v_max:
                sev = "warning"
                decision = (f"Observed max ({gv_max_obs:.1f}V) EXCEEDS the configured upper threshold "
                            f"({grid_v_max}V) — these rows would be misclassified as a grid outage even "
                            f"though the grid was actually up, just running high. Consider raising the "
                            f"upper threshold for this feeder, or confirm this is expected voltage headroom.")
            elif grid_v_min is not None and gv_min_obs < grid_v_min and n_zero == 0:
                sev = "warning"
                decision = (f"Observed min ({gv_min_obs:.1f}V) is below the configured lower threshold "
                            f"({grid_v_min}V) without ever reading exactly 0V — could be genuine brownout "
                            f"conditions rather than a full outage; confirm the lower threshold matches "
                            f"what should count as 'grid down' for this site.")
            findings.append(dict(severity=sev, area="Grid voltage / outage threshold", finding=msg, decision=decision))

            # outage episode detection (contiguous zero-voltage runs)
            if best_ts_col is not None and n_zero > 0:
                tmp = df.copy()
                tmp["_ts"] = pd.to_datetime(tmp[best_ts_col], errors="coerce")
                tmp = tmp.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
                tmp["_gv"] = pd.to_numeric(tmp[gv_col], errors="coerce")
                is_zero = (tmp["_gv"] == 0).astype(int).values
                episodes = []
                i, n = 0, len(is_zero)
                while i < n:
                    if is_zero[i] == 1:
                        j = i
                        while j < n and is_zero[j] == 1:
                            j += 1
                        episodes.append((tmp["_ts"].iloc[i], tmp["_ts"].iloc[j - 1], j - i))
                        i = j
                    else:
                        i += 1
                if episodes:
                    ep_desc = "; ".join(
                        f"{s.strftime('%Y-%m-%d %H:%M')}→{e.strftime('%H:%M')} (~{n_rows} row(s))"
                        for s, e, n_rows in episodes[:8]
                    )
                    findings.append(dict(
                        severity="info", area="Outage episodes",
                        finding=f"{len(episodes)} distinct outage episode(s) detected (contiguous 0V runs): {ep_desc}"
                                f"{' ...' if len(episodes) > 8 else ''}",
                        decision="Useful for manually cross-checking Outage Savings / Unserved Load figures "
                                 "against known real outage windows for this site.",
                    ))

    # --- 4. Constant / all-zero numeric columns (possible unused hardware or wiring fault) ---
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols_all:
        col = df[c].dropna()
        if len(col) > 5 and col.nunique() == 1:
            findings.append(dict(
                severity="info", area="Constant columns",
                finding=f"Column '{c}' is constant ({col.iloc[0]}) across all {len(col):,} rows.",
                decision="If this is a power/current column (e.g. a second PV string), confirm whether "
                         "that hardware is genuinely unused/not installed, or whether it's a wiring/logging "
                         "fault worth checking on-site — a constantly-zero PV2 string, for instance, quietly "
                         "halves expected generation without ever throwing an error.",
            ))

    # --- 5. Missing values & negative values on numeric columns ---
    na_counts = df[numeric_cols_all].isna().sum()
    na_counts = na_counts[na_counts > 0]
    if len(na_counts) > 0:
        findings.append(dict(
            severity="warning", area="Missing values",
            finding="Missing values found in: " + ", ".join(f"{c} ({n})" for c, n in na_counts.items()),
            decision="preprocess() already drops rows with missing values in its required columns; "
                     "columns outside that required set are simply not checked — confirm none of them "
                     "matter for future formulas before ignoring these gaps.",
        ))

    power_like_cols = [c for c in numeric_cols_all if any(
        k in c.lower() for k in ["power", "current", "voltage", "capacity"]
    )]
    neg_report = {}
    for c in power_like_cols:
        n_neg = int((df[c] < 0).sum())
        if n_neg > 0:
            neg_report[c] = n_neg
    if neg_report:
        findings.append(dict(
            severity="warning", area="Negative values",
            finding="Negative values found in: " + ", ".join(f"{c} ({n})" for c, n in neg_report.items()),
            decision="preprocess() clips PV_kWh/Load_kWh at zero after computing them, but negative raw "
                     "sensor readings elsewhere are worth a source-side check (sensor wiring/polarity fault).",
        ))

    # --- 6. Cumulative-counter monotonicity (generation/total-style columns) ---
    cum_like_cols = [c for c in numeric_cols_all if any(
        k in c.lower() for k in ["generation", "total generation", "cumulative"]
    )]
    for c in cum_like_cols:
        series = pd.to_numeric(df[c], errors="coerce")
        n_decrease = int((series.diff() < 0).sum())
        if n_decrease > 0:
            findings.append(dict(
                severity="info", area="Cumulative counters (not used by our formulas)",
                finding=f"'{c}' decreases {n_decrease} time(s) across the file (daily/monthly/yearly "
                        f"counter resets or logging glitches).",
                decision="Confirms the app's existing design choice to use ONLY instantaneous power "
                         "columns (PV1/PV2 Input Power, AC output power) for all energy integration, "
                         "never a cumulative 'generation' counter — this column is never read by "
                         "preprocess() or simulate() for exactly this reason.",
            ))

    return findings


# =====================================================================
# TARIFF / RATE HELPERS
# =====================================================================
def marginal_slab_rate(cum_units: float) -> float:
    """Rate applicable to the NEXT unit given `cum_units` already consumed this billing cycle."""
    if cum_units < 0:
        cum_units = 0.0
    for cap, rate in SLAB_TIERS:
        if cum_units < cap - 1e-9:
            return rate
    return SLAB_TIERS[-1][1]


def in_window(t: dtime, start: dtime, end: dtime) -> bool:
    """True if t falls in [start, end), handling windows that wrap past midnight."""
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def effective_rate(t: dtime, cum_units: float, tariff: dict) -> float:
    """Per-unit ₹ price for the next unit at time `t`, given cumulative units so far this cycle.
    Slab rate → optional % TOD-surcharge markup (peak window) → optional flat solar-hour rebate.
    """
    rate = marginal_slab_rate(cum_units)
    if tariff.get("surcharge_on") and in_window(t, tariff["tod_start"], tariff["tod_end"]):
        rate *= (1.0 + tariff.get("tod_pct", 0.0) / 100.0)
    if tariff.get("rebate_on") and in_window(t, tariff["solar_start"], tariff["solar_end"]):
        rate -= tariff.get("rebate_rate", 0.0)
    return max(rate, 0.0)


# =====================================================================
# STRICT DATA PROCESSING (Section 1 of spec)
# =====================================================================
def preprocess(raw: pd.DataFrame, grid_v_min: float, grid_v_max: float, power_factor: float,
               cycle_days: int) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    required = [
        "Timestamp", "PV1 Input Power(W)", "PV2 Input Power(W)",
        "AC output apparent power(VA)", "Grid voltage(V)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found columns: {list(df.columns)}")

    # --- basic cleaning ---
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    numeric_cols = ["PV1 Input Power(W)", "PV2 Input Power(W)", "AC output apparent power(VA)", "Grid voltage(V)"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    df["PV_kWh"] = (df["PV1 Input Power(W)"] + df["PV2 Input Power(W)"]) / 1000.0 * DT_HOURS
    # Load is billed/real energy, not the inverter's apparent-power rating draw, so it's
    # derived from AC output apparent power(VA) x power factor rather than read directly
    # off an "active power" column (which isn't populated on every inverter model).
    df["Load_kWh"] = df["AC output apparent power(VA)"] * power_factor / 1000.0 * DT_HOURS
    df["PV_kWh"] = df["PV_kWh"].clip(lower=0)
    df["Load_kWh"] = df["Load_kWh"].clip(lower=0)

    df["Grid_Status"] = ((df["Grid voltage(V)"] >= grid_v_min) & (df["Grid voltage(V)"] <= grid_v_max)).astype(int)
    df["t_of_day"] = df["Timestamp"].dt.time

    # Slab/billing cycle resets every `cycle_days` from the first timestamp in the dataset
    # (electricity boards typically run bi-monthly cycles; kept as a UI-configurable variable).
    first_date = df["Timestamp"].min().normalize()
    df["cycle_id"] = ((df["Timestamp"].dt.normalize() - first_date).dt.days // cycle_days).astype(int)

    return df


# =====================================================================
# ROW-BY-ROW SIMULATION ENGINE
# =====================================================================
def simulate(df: pd.DataFrame, mode: str, tariff: dict, battery: dict = None, reserve_pct: float = 0.20,
             max_charge_pct: float = 1.0, pv_zero_threshold_kwh: float = 0.0) -> dict:
    """
    mode: 'grid_tie' | 'dumb' | 'smart'

    Financial accounting is done LIVE, row-by-row, alongside the energy simulation:
      - `baseline_bill`   : what the full household demand would cost, at the time-and-slab
                            aware rate, cycle by cycle (the "no solar at all" counterfactual).
      - `actual_bill_gross`: what was REALLY billed for every real grid import (serving load
                            OR charging the battery), at the rate that applied at that moment.
      - `outage_bill_notional`: the notional cost of energy served by PV/battery during a grid
                            outage, valued at the rate it would have carried in the baseline
                            stream (replaces a flat VoLL with the actual applicable slab/TOD price).
      - `tod_optimization`: Smart-Hybrid-only. Battery energy discharged to load, while grid is
                            up, during the TOD-surcharge (peak) window. PV-origin energy (free)
                            is credited in full at the current avoided rate; grid-origin energy
                            is credited only the DIFFERENCE between the avoided rate now and
                            the (weighted-average) rate it actually cost when it was charged —
                            since that unit was already billed once, in `actual_bill_gross`.

    Battery SOC is split into two provenance buckets so grid-origin kWh can be told apart from
    free PV-origin kWh when valuing a later discharge (`soc_pv`, `soc_grid` + a running
    weighted-average `grid_cost_basis` ₹/kWh for the grid-origin bucket). This is what lets
    Solar Savings / TOD Optimization / Outage Savings partition (Base − Actual) without any
    kWh being counted as a saving twice (see solar_savings' residual definition below).
    """
    usable_kwh = battery["capacity_kwh"] * battery["dod"] if battery else 0.0
    eff = battery["efficiency"] if battery else 1.0
    reserve_soc = usable_kwh * reserve_pct if battery else 0.0
    max_soc = usable_kwh * max_charge_pct if battery else 0.0

    # Both hybrids start fully charged (max-charge ceiling). Initial charge is treated as
    # PV-origin (cost-free) by convention — it's a starting assumption, not a real purchase.
    soc_pv = max_soc
    soc_grid = 0.0
    grid_cost_basis = 0.0
    soc = soc_pv + soc_grid

    def discharge_from_buckets(discharge_gross: float):
        nonlocal soc_pv, soc_grid, soc
        if discharge_gross <= 0:
            return 0.0, 0.0
        from_pv = min(soc_pv, discharge_gross)
        from_grid = discharge_gross - from_pv
        soc_pv -= from_pv
        soc_grid -= from_grid
        soc -= discharge_gross
        return from_pv, from_grid

    pv_to_load = export = batt_to_load = grid_import = unserved = 0.0
    opportunity_loss = notional_loss = pv_to_battery = 0.0
    outage_served = 0.0
    grid_to_battery = 0.0
    total_pv = float(df["PV_kWh"].sum())
    total_demand = float(df["Load_kWh"].sum())

    baseline_bill = 0.0
    actual_bill_gross = 0.0
    outage_bill_notional = 0.0
    tod_optimization = 0.0

    baseline_cum_cycle = 0.0
    actual_cum_cycle = 0.0
    cur_cycle = None

    row_cycle_ids, row_grid_import, row_pv_export, row_actual_bill = [], [], [], []

    for row in df.itertuples(index=False):
        pv, load, grid_up, t, cyc = row.PV_kWh, row.Load_kWh, bool(row.Grid_Status), row.t_of_day, row.cycle_id
        solar_hr = in_window(t, tariff["solar_start"], tariff["solar_end"])

        if cyc != cur_cycle:
            baseline_cum_cycle = 0.0
            actual_cum_cycle = 0.0
            cur_cycle = cyc

        row_grid_import_amt = 0.0
        row_export_amt = 0.0
        row_bill_amt = 0.0

        # --- baseline (no-solar-at-all) bill contribution, every row, every mode ---
        rate_baseline = effective_rate(t, baseline_cum_cycle, tariff)
        baseline_bill += rate_baseline * load
        baseline_cum_cycle += load

        if mode == "grid_tie":
            if grid_up:
                u = min(pv, load)
                pv_to_load += u
                exp = pv - u
                export += exp
                row_export_amt += exp
                imp = load - u
                if imp > 0:
                    rate_now = effective_rate(t, actual_cum_cycle, tariff)
                    bill_amt = rate_now * imp
                    actual_bill_gross += bill_amt
                    row_bill_amt += bill_amt
                    actual_cum_cycle += imp
                    row_grid_import_amt += imp
                    grid_import += imp
            else:
                # System dead: PV_to_load = Export = Import = 0, Unserved = Load, PV wasted.
                unserved += load
                opportunity_loss += pv

        elif grid_up:
            if mode == "dumb":
                # Battery never discharges to load while grid is up (emergency-backup only).
                # It CAN charge from the grid, but only once PV has dropped near 0 — while PV
                # is generating above that, charging is PV-surplus-only (Generation - Load).
                u = min(pv, load)
                pv_to_load += u
                pv_rem = pv - u
                charge = min(pv_rem, max(max_soc - soc, 0.0))
                soc_pv += charge
                soc += charge
                pv_to_battery += charge
                pv_rem -= charge
                export += pv_rem
                row_export_amt += pv_rem
                imp = load - u
                if imp > 0:
                    rate_now = effective_rate(t, actual_cum_cycle, tariff)
                    bill_amt = rate_now * imp
                    actual_bill_gross += bill_amt
                    row_bill_amt += bill_amt
                    actual_cum_cycle += imp
                    row_grid_import_amt += imp
                    grid_import += imp

                if pv <= pv_zero_threshold_kwh and soc < max_soc:
                    charge_needed = max_soc - soc
                    grid_pull = charge_needed / eff if eff > 0 else 0.0
                    rate_now = effective_rate(t, actual_cum_cycle, tariff)
                    bill_amt = rate_now * grid_pull
                    actual_bill_gross += bill_amt
                    row_bill_amt += bill_amt
                    actual_cum_cycle += grid_pull
                    row_grid_import_amt += grid_pull
                    grid_to_battery += grid_pull
                    grid_import += grid_pull
                    new_soc_grid = soc_grid + charge_needed
                    grid_cost_basis = (
                        (soc_grid * grid_cost_basis + charge_needed * rate_now) / new_soc_grid
                        if new_soc_grid > 1e-12 else 0.0
                    )
                    soc_grid = new_soc_grid
                    soc += charge_needed

            else:  # mode == "smart"
                if pv >= load:
                    # Surplus PV: serve load fully, charge battery with the rest, export any excess.
                    u = load
                    pv_to_load += u
                    pv_rem = pv - u
                    charge = min(pv_rem, max(max_soc - soc, 0.0))
                    soc_pv += charge
                    soc += charge
                    pv_to_battery += charge
                    pv_rem -= charge
                    export += pv_rem
                    row_export_amt += pv_rem
                else:
                    # PV deficit: battery discharges to cover it, but stops at Reserve_SOC.
                    u = pv
                    pv_to_load += u
                    deficit = load - pv
                    avail_above_reserve = max(soc - reserve_soc, 0.0)
                    discharge_gross = min(avail_above_reserve, deficit / eff)
                    from_pv_g, from_grid_g = discharge_from_buckets(discharge_gross)
                    delivered = discharge_gross * eff
                    delivered_pv = from_pv_g * eff
                    delivered_grid = from_grid_g * eff
                    batt_to_load += delivered
                    rem_deficit = deficit - delivered
                    if rem_deficit > 0:
                        rate_now = effective_rate(t, actual_cum_cycle, tariff)
                        bill_amt = rate_now * rem_deficit
                        actual_bill_gross += bill_amt
                        row_bill_amt += bill_amt
                        actual_cum_cycle += rem_deficit
                        row_grid_import_amt += rem_deficit
                        grid_import += rem_deficit

                    if delivered > 0:
                        if tariff.get("surcharge_on") and in_window(t, tariff["tod_start"], tariff["tod_end"]):
                            r2 = effective_rate(t, actual_cum_cycle, tariff)
                            tod_optimization += delivered_pv * r2 + delivered_grid * (r2 - grid_cost_basis)
                        # else: falls under Solar Savings automatically via the residual below —
                        # no separate accumulation needed (see compute_financials()).

                if not solar_hr and soc < reserve_soc:
                    charge_needed = reserve_soc - soc
                    grid_pull = charge_needed / eff if eff > 0 else 0.0
                    rate_now = effective_rate(t, actual_cum_cycle, tariff)
                    bill_amt = rate_now * grid_pull
                    actual_bill_gross += bill_amt
                    row_bill_amt += bill_amt
                    actual_cum_cycle += grid_pull
                    row_grid_import_amt += grid_pull
                    grid_to_battery += grid_pull
                    grid_import += grid_pull
                    new_soc_grid = soc_grid + charge_needed
                    grid_cost_basis = (
                        (soc_grid * grid_cost_basis + charge_needed * rate_now) / new_soc_grid
                        if new_soc_grid > 1e-12 else 0.0
                    )
                    soc_grid = new_soc_grid
                    soc += charge_needed

        else:
            # -------------------------------------- OUTAGE (identical for Dumb & Smart)
            u = min(pv, load)
            pv_to_load += u
            load_rem = load - u
            discharge_gross = min(soc, load_rem / eff)
            discharge_from_buckets(discharge_gross)
            delivered = discharge_gross * eff
            batt_to_load += delivered
            load_rem -= delivered

            unserved += load_rem
            row_outage_amt = u + delivered
            outage_served += row_outage_amt
            outage_bill_notional += row_outage_amt * rate_baseline

            pv_rem = pv - u
            charge = min(pv_rem, max(max_soc - soc, 0.0))
            soc_pv += charge
            soc += charge
            pv_to_battery += charge
            pv_rem -= charge
            notional_loss += pv_rem

        row_cycle_ids.append(cyc)
        row_grid_import.append(row_grid_import_amt)
        row_pv_export.append(row_export_amt)
        row_actual_bill.append(row_bill_amt)

    pv_available = total_pv - (opportunity_loss if mode == "grid_tie" else notional_loss)
    load_served = total_demand - unserved

    return dict(
        mode=mode,
        total_pv=total_pv,
        opportunity_loss=opportunity_loss,
        notional_loss=notional_loss,
        pv_available=pv_available,
        total_demand=total_demand,
        load_served=load_served,
        pv_to_load=pv_to_load,
        pv_export=export,
        pv_to_battery=pv_to_battery,
        batt_to_load=batt_to_load,
        grid_import=grid_import,
        grid_to_battery=grid_to_battery,
        unserved=unserved,
        outage_served=outage_served,
        final_soc=soc,
        usable_kwh=usable_kwh,
        reserve_soc=reserve_soc,
        max_soc=max_soc,
        baseline_bill=baseline_bill,
        actual_bill_gross=actual_bill_gross,
        outage_bill_notional=outage_bill_notional,
        tod_optimization=tod_optimization,
        row_cycle_ids=row_cycle_ids,
        row_grid_import=row_grid_import,
        row_pv_export=row_pv_export,
        row_actual_bill=row_actual_bill,
    )


# =====================================================================
# FINANCIALS — bill-delta partition, net metering / solar earning, no double counting
# =====================================================================
def compute_financials(res: dict, net_metering: bool, export_rate: float) -> dict:
    """
      Base Bill    = time+slab+cycle aware bill on the FULL household demand (no solar at all).
      Actual Bill  = time+slab+cycle aware bill on what was REALLY imported (load AND any
                     battery charging), net-metering-settled if applicable.
      Outage Savings = notional cost, at the rate that would have applied, of energy served by
                     PV/battery during a grid outage (replaces a flat VoLL).
      TOD Optimization = Smart-Hybrid-only arbitrage credit for battery energy discharged to
                     load during the TOD-surcharge window (see simulate()'s docstring).
      Solar Savings = (Base Bill - Actual Bill) - Outage Savings - TOD Optimization
                     — the RESIDUAL of the true bill delta after removing the other two, so the
                     three savings buckets are a strict, non-overlapping partition of the real
                     bill difference: no kWh's value is ever counted under more than one bucket.
      Solar Earning = min(cycle PV export, cycle grid import) x export_rate, ONLY for
                     Non-Net-Metering (kept as its own column — never folded into Solar Savings).
      Net Savings   = Solar Savings + Outage Savings + TOD Optimization (Solar Earning is real
                     export income, reported separately, per spec).
    """
    baseline_bill = res["baseline_bill"]
    outage_savings = res["outage_bill_notional"]
    tod_optimization = res["tod_optimization"]

    rows = pd.DataFrame({
        "cycle_id": res["row_cycle_ids"],
        "grid_import": res["row_grid_import"],
        "pv_export": res["row_pv_export"],
        "bill": res["row_actual_bill"],
    })
    cyc = rows.groupby("cycle_id").agg(
        import_sum=("grid_import", "sum"), export_sum=("pv_export", "sum")
    ).reset_index()
    cyc["settled"] = np.minimum(cyc["import_sum"], cyc["export_sum"])

    if net_metering:
        # 1:1 settlement nets export against import WITHIN each cycle. Each row's import is
        # scaled down by that cycle's netting fraction and re-billed at the SAME per-row rate
        # it was actually charged at, so the time-of-day rate structure is preserved while
        # exactly the settled kWh worth of billing is removed, cycle by cycle.
        cyc["scale"] = np.where(
            cyc["import_sum"] > 1e-12,
            np.maximum(cyc["import_sum"] - cyc["settled"], 0.0) / cyc["import_sum"],
            1.0,
        )
        rows = rows.merge(cyc[["cycle_id", "scale"]], on="cycle_id", how="left")
        actual_bill = float((rows["bill"] * rows["scale"]).sum())
        solar_earning = 0.0
    else:
        actual_bill = res["actual_bill_gross"]
        solar_earning = float(cyc["settled"].sum()) * export_rate

    solar_savings = (baseline_bill - actual_bill) - outage_savings - tod_optimization
    net_savings = solar_savings + outage_savings + tod_optimization

    res["baseline_bill"] = baseline_bill
    res["actual_bill"] = actual_bill
    res["settlement_kwh"] = float(cyc["settled"].sum())
    res["solar_savings"] = solar_savings
    res["outage_savings"] = outage_savings
    res["tod_optimization"] = tod_optimization
    res["solar_earning"] = solar_earning
    res["net_savings"] = net_savings
    res["net_metering"] = net_metering
    return res


# =====================================================================
# UI — SIDEBAR
# =====================================================================
st.title("☀️ Solar Inverter Time-Series Simulator")
st.caption("Grid-Tie vs Dumb Hybrid vs Smart Hybrid — row-by-row 5-minute simulation, instantaneous-power based, fully auditable.")

with st.expander("🔎 Where Efficiency, Charge Rate & Discharge Rate Are Used (both batteries) — click to read"):
    st.markdown(f"""
**Efficiency — a single round-trip number, not separate charge/discharge/coulombic values.**

`BATTERY_PRESETS` sets one `efficiency` scalar per chemistry — **Lithium = {BATTERY_PRESETS['Lithium (48V / 314Ah)']['efficiency']*100:.0f}%**,
**Lead-Acid = {BATTERY_PRESETS['Lead-Acid (48V / 200Ah)']['efficiency']*100:.0f}%** — and `simulate()` uses the exact same formulas for
both chemistries, just with that different number plugged in as `eff`. There is no separate "charging
efficiency" vs "discharging efficiency" vs "round-trip efficiency" concept in this part of the app — it's
one number, and where it gets applied is asymmetric:

- **Grid → Battery charging** (`grid_pull = charge_needed / eff`): the battery's SOC goes up by exactly
  `charge_needed`, but the grid import (and its cost) is inflated to `charge_needed / eff` — i.e. you pay
  for more than lands in the battery. `eff` is applied HERE.
- **Battery → Load discharging** (`discharge_gross = deficit / eff`, then `delivered = discharge_gross * eff`):
  SOC is drawn down by more than what actually reaches the load — `eff` is applied HERE too.
- **PV → Battery charging** (`charge = min(pv_rem, room)`): PV surplus lands in the battery **1:1, with NO
  efficiency loss applied at all** — this path does not use `eff`.

**Net effect (same for both chemistries, just different `eff` value):** a kWh that came from PV and is
later discharged to load loses efficiency ONCE (at discharge). A kWh that came from the GRID, got stored,
and is later discharged loses efficiency TWICE (once charging in, once discharging out) — effectively
`eff²` round-trip for grid-sourced battery energy, vs `eff` for PV-sourced battery energy. For Lead-Acid
(eff=0.80) that's 80% vs 64%; for Lithium (eff=0.95) that's 95% vs 90.25%. This asymmetry exists in the
current formulas for BOTH chemistries — flagging it here rather than changing it silently, since it's a
pre-existing formula.

**Charge rate / discharge rate (C-rate) — NOT modeled anywhere in `simulate()`.** There is currently no
cap on how much power the battery can accept or deliver in a single 5-minute interval based on a hardware
current/C-rate limit, for either chemistry. The only limits enforced are energy-based: available PV
surplus, load deficit, and remaining SOC headroom (`max_soc - soc` for charging, `soc - reserve_soc` for
discharging) — never a "battery can't physically charge/discharge faster than X kW" constraint. This
matters more for Lead-Acid in practice (typically limited to something like 0.2C-0.3C in solar/backup use)
than Lithium (commonly rated for higher continuous C-rates), but neither chemistry has this cap today. If
a large PV surplus or a big load deficit occurs inside one 5-minute window, the simulator will happily
move the entire corresponding kWh in or out of the battery in that single interval, which a real battery
+ inverter combo may not physically be able to do. This is a real gap worth a decision, not yet fixed.

**Where the Peukert / Ah-based coulomb-counting corrections above are separate.** The charge-efficiency
and Peukert-derating controls in the 🔋 Lead-Acid Live SOC panel below are part of the NEW live-telemetry
SOC module (added on top, using `chargingCurrent`/`dischargingCurrent` from the device payload) — they do
**not** feed back into this historical-CSV `simulate()` engine's `eff` value above. The two modules are
intentionally separate: one estimates real-time SOC from live device telemetry, the other simulates
hypothetical inverter behaviour on historical PV/load data.
""")

with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload 5-min interval CSV", type=["csv"])
    st.caption(
        "Required columns: `Timestamp`, `PV1 Input Power(W)`, `PV2 Input Power(W)`, "
        "`AC output apparent power(VA)`, `Grid voltage(V)`"
    )
    power_factor = st.slider(
        "Load Power Factor (PF)", 0.50, 1.00, 0.90, 0.01,
        help="Load is billed as real/active power, but the field data reports AC output "
             "Apparent Power (VA). Load_kWh = VA × PF / 1000 × Δt. PF varies by inverter "
             "and connected load mix — adjust to match the site under test."
    )

    st.header("2. Battery Configuration")
    battery_choice = st.selectbox("Battery Type", list(BATTERY_PRESETS.keys()))
    battery = BATTERY_PRESETS[battery_choice]
    st.caption(
        f"Capacity: {battery['capacity_kwh']} kWh | Max DoD: {int(battery['dod']*100)}% "
        f"→ Usable: {battery['capacity_kwh']*battery['dod']:.2f} kWh | Efficiency: {int(battery['efficiency']*100)}%"
    )
    max_charge_pct = st.slider(
        "Battery Max Charge SOC (%)", 50, 100, 98,
        help="Ceiling on charging (PV or grid) as a % of usable capacity. Some batteries "
             "stabilise at e.g. 98% rather than reaching a literal 100% full charge."
    ) / 100.0
    st.caption(
        f"Both hybrids start the simulation fully charged, i.e. at the {int(max_charge_pct*100)}% "
        f"max-charge ceiling ({battery['capacity_kwh']*battery['dod']*max_charge_pct:.2f} kWh)."
    )
    reserve_pct = st.slider(
        "Smart Hybrid — Reserve SOC (%, kept for outages during grid-up discharge)", 0, 100, 20,
        help="Smart Hybrid stops discharging to load (while grid is up) once SOC drops to this level."
    ) / 100.0
    pv_zero_threshold_w = st.slider(
        "Dumb Hybrid — Grid charges battery when PV drops below (W)", 0, 200, 20,
        help="Dumb Hybrid can only charge its battery from the grid once PV generation is "
             "effectively zero (below this threshold). While PV is above this, the battery "
             "can only be charged from PV surplus (Generation − Load), never from the grid."
    )
    pv_zero_threshold_kwh = pv_zero_threshold_w / 1000.0 * DT_HOURS

    st.header("3. Grid Status Threshold")
    grid_v_min, grid_v_max = st.slider(
        "Valid Grid Voltage Band (V) — Grid_Status = 1 inside this range", 0, 300, (180, 270),
        help="Default upper bound raised from 260V to 270V based on real field data: two separate real "
             "exports from this fleet both showed genuine grid-up readings above 260V (up to 268.2V) — "
             "260V was flagging real grid-up intervals as outages. Still adjustable per site."
    )

    st.header("4. Tariff & Billing Cycle")
    st.caption("Slab reference (fixed): 0–100u @ ₹7.5 · 101–300u @ ₹11.5 · 300–500u @ ₹15.5 · >500u @ ₹17.5")
    cycle_days = st.slider(
        "Slab/billing cycle length (days)", 1, 90, 60,
        help="Cumulative kWh used to pick the slab rate resets to 0 at the start of every "
             "cycle (e.g. MSEDCL residential bills bi-monthly ≈ 60 days). Board-specific."
    )
    col_sh1, col_sh2 = st.columns(2)
    solar_start = col_sh1.time_input("Solar hours start", dtime(9, 0))
    solar_end = col_sh2.time_input("Solar hours end", dtime(18, 0))
    st.caption("Solar hours are set by each city/state electricity board — adjust to match the local rule.")
    rebate_on = st.checkbox("Apply solar-hour rebate", value=True)
    rebate_rate = st.number_input(
        "Solar-hour rebate (₹/unit)", min_value=0.0, value=0.80, step=0.05,
        help="Subtracted from the slab rate for grid import during solar hours. This is a "
             "government/board discount, not an inverter-provided saving — it reduces the "
             "actual bill directly rather than being added as a separate savings line."
    )
    surcharge_on = st.checkbox("Apply TOD surcharge (peak-hour price bump)", value=False)
    col_tod1, col_tod2 = st.columns(2)
    tod_start = col_tod1.time_input("TOD surcharge start", dtime(18, 0))
    tod_end = col_tod2.time_input("TOD surcharge end", dtime(22, 0))
    tod_pct = st.number_input(
        "TOD surcharge (% increase over the base slab rate)", min_value=0.0, value=10.0, step=1.0,
        help="e.g. 10% turns a ₹5/unit slab into ₹5.50/unit during the surcharge window. "
             "Applied on TOP of whatever slab tier the cumulative usage is in at that moment."
    )

    tariff = dict(
        solar_start=solar_start, solar_end=solar_end,
        tod_start=tod_start, tod_end=tod_end, tod_pct=tod_pct, surcharge_on=surcharge_on,
        rebate_rate=rebate_rate, rebate_on=rebate_on,
    )

    st.header("5. Net Metering")
    net_metering_choice = st.radio(
        "Settlement mode",
        ["Net Metering — 1:1 settlement", "Non-Net Metering — Solar Earning"],
        index=0,
        help="Net metering: net grid import billed = Grid Import − min(Grid Import, PV Export), "
             "settled by the board. Non-net-metering: full Grid Import is billed, and PV export "
             "is paid separately at a configured ₹/unit earning rate."
    )
    net_metering_enabled = net_metering_choice.startswith("Net Metering")
    export_rate = st.number_input(
        "Non-Net-Metering — PV export earning rate (₹/unit)", min_value=0.0, value=3.0, step=0.10,
        help="Only used in Non-Net-Metering mode. Paid on min(cycle PV export, cycle grid import)."
    )


# =====================================================================
# UI — LEAD-ACID LIVE SOC PANEL (NEW — additive only)
# Independent of the PV/load CSV simulator above; driven by the separate
# live battery-telemetry payload. Shown only when the selected preset is
# Lead-Acid, since Lithium's own `soc` field is already BMS-fused and
# doesn't need this.
# =====================================================================
if "Lead-Acid" in battery_choice:
    st.header("🔋 Lead-Acid Live SOC — Coulomb Counting")
    st.caption(
        "Lithium's `soc` field already comes fused from its BMS (coulomb counting + OCV "
        "recalibration done onboard) — trust it directly. Lead-acid telemetry only gives "
        "voltage + charge/discharge current, and lead-acid voltage sags/rises too much under "
        "load to trust a voltage-only SOC mid-cycle, so SOC is tracked here via coulomb "
        "counting instead, anchored to 100% once a full charge is detected."
    )
    st.latex(r"SOC_2 = SOC_1 - \left(\frac{Current \times Time}{Capacity}\right) \times 100")
    st.markdown(
        "- **status = 2 (Charging)** → SOC increases using `chargingCurrent`.\n"
        "- **status = 1 (Discharging)** → SOC decreases using `dischargingCurrent`.\n"
        "- **status = 0 (None)** → SOC unchanged.\n"
        "- **Anchor**: once the battery is detected fully charged (actively charging, at/above "
        "float voltage, and charging current tapered near zero), SOC is snapped to 100% and "
        "coulomb counting resumes from there. This anchor step is required because coulomb "
        "counting alone drifts over time (current-sensor offset, self-discharge, temperature) "
        "with no way to self-correct otherwise — exactly the approach specified: *\"wait for "
        "full charge of the battery, then take SOC as 100%; after that, keep incrementing or "
        "decrementing SOC based on Ah in/out.\"*"
    )

    lc1, lc2, lc3 = st.columns(3)
    la_capacity_ah = lc1.number_input(
        "Battery capacity (Ah)", min_value=1.0, value=200.0, step=10.0,
        help="Rated Ah capacity of the lead-acid bank (e.g. 200 Ah for the 48V/200Ah preset)."
    )
    la_float_voltage = lc2.number_input(
        "Full-charge / float voltage (V)", min_value=0.0, value=LEAD_ACID_FULL_CHARGE_VOLTAGE_DEFAULT, step=0.5,
        help="Set to the charger's actual absorption/float set-point for this bank."
    )
    la_taper_frac = lc3.number_input(
        "Full-charge taper current (fraction of capacity_Ah)", min_value=0.0, max_value=1.0,
        value=LEAD_ACID_FULL_CHARGE_TAPER_FRACTION, step=0.01,
        help="Charging current below this fraction of capacity_Ah, at/above float voltage, is treated as 'full'."
    )

    with st.expander("⚙️ Charge efficiency & Peukert correction (lead-acid-specific accuracy)"):
        st.markdown(
            "Both corrections below are **lead-acid-specific** — lithium's `soc` telemetry is already "
            "BMS-fused (coulombic efficiency ~98-99%, Peukert exponent ~1.0), so neither applies there. "
            "Both default OFF (charge efficiency = 100%, Peukert = disabled) so the plain formula matches "
            "the manager's worked example exactly unless you turn these on."
        )
        ec1, ec2 = st.columns(2)
        la_charge_eff_on = ec1.checkbox(
            "Apply charging (coulombic) efficiency", value=True,
            help="Not every Ah pushed into a lead-acid battery becomes usable capacity — some is lost to "
                 "gassing/heat, especially near full charge. Applied ONLY to the charging leg — there is no "
                 "separate 'discharging efficiency' because the coulombic loss on the way OUT is close to "
                 "100% for lead-acid (the Ah that leave the battery are the Ah that leave). This is the "
                 "Ah-based (coulombic) round-trip loss, distinct from ENERGY (kWh) round-trip efficiency, "
                 "which also reflects the voltage gap between charging and discharging — coulomb counting "
                 "only needs the coulombic number."
        )
        la_charge_efficiency = ec1.number_input(
            "Charge efficiency", min_value=0.5, max_value=1.0,
            value=LEAD_ACID_CHARGE_EFFICIENCY_DEFAULT, step=0.01, disabled=not la_charge_eff_on,
            help="Typical flooded/AGM lead-acid: 0.80-0.90. Check the battery datasheet if available."
        ) if la_charge_eff_on else 1.0

        la_peukert_on = ec2.checkbox(
            "Apply Peukert derating on discharge", value=True,
            help="Lead-acid's usable capacity shrinks at higher discharge currents (unlike lithium, which is "
                 "close to Peukert exponent 1.0 and doesn't need this). Defaults below (k=1.45, reference "
                 "≈ capacity_Ah/9.7) are CALIBRATED from this fleet's actual 200Ah lead-acid vendor spec sheet "
                 "(fitted against 6 real backup-time data points, matching to within rounding) — re-fit these "
                 "two numbers if a different lead-acid model/vendor is used."
        )
        pc1, pc2 = st.columns(2)
        la_nominal_rate_a = pc1.number_input(
            "Rated discharge reference current (A)", min_value=0.1,
            value=max(la_capacity_ah / LEAD_ACID_PEUKERT_REFERENCE_RATE_HOURS_DEFAULT, 0.1),
            step=0.5, disabled=not la_peukert_on,
            help="The current capacity_Ah is rated at. For this fleet's 200Ah battery, fitting the vendor's "
                 "own spec-sheet data implies a ~9.7-hour reference rate (≈20.7A on 200Ah) — closer to a C10 "
                 "rating than the more commonly assumed C20. Recompute for other battery models."
        ) if la_peukert_on else None
        la_peukert_exponent = pc2.number_input(
            "Peukert exponent", min_value=1.0, max_value=1.6,
            value=LEAD_ACID_PEUKERT_EXPONENT_DEFAULT, step=0.01, disabled=not la_peukert_on,
            help="1.1-1.3 is the commonly cited generic range, but this fleet's actual 200Ah battery fits "
                 "k≈1.45 — noticeably steeper — from its own vendor spec sheet (both 24V and 48V wired "
                 "variants, 6 data points, predictions match to within rounding). Use the generic range only "
                 "if you don't have a real spec sheet to fit against."
        ) if la_peukert_on else 1.0

    with st.expander("Try it — worked example (matches the manager's spec: 80% start, 20A discharge, 2h → 40%)"):
        st.caption(
            "This demo uses the PLAIN formula (no charge efficiency / Peukert) so it always reproduces the "
            "manager's exact numbers, regardless of the settings above. Those corrections apply when you "
            "process a real telemetry file below."
        )
        wc1, wc2, wc3, wc4 = st.columns(4)
        w_prev_soc = wc1.number_input("Starting SOC (%)", 0.0, 100.0, 80.0, key="la_demo_soc")
        w_status = wc2.selectbox(
            "Status", [0, 1, 2], index=1,
            format_func=lambda s: {0: "0 - None", 1: "1 - Discharging", 2: "2 - Charging"}[s],
            key="la_demo_status",
        )
        w_current = wc3.number_input("Current (A)", 0.0, 1000.0, 20.0, key="la_demo_current")
        w_hours = wc4.number_input("Time (hours)", 0.0, 100.0, 2.0, key="la_demo_hours")
        w_new_soc = estimate_soc_leadacid_coulomb_counting(
            prev_soc_pct=w_prev_soc,
            status=w_status,
            charging_current_a=(w_current if w_status == LEAD_ACID_STATUS_CHARGING else 0.0),
            discharging_current_a=(w_current if w_status == LEAD_ACID_STATUS_DISCHARGING else 0.0),
            dt_hours=w_hours,
            capacity_ah=la_capacity_ah,
        )
        st.markdown(f"**New SOC = {w_new_soc:.2f}%**")

    la_telemetry_file = st.file_uploader(
        "Optional: upload a lead-acid telemetry log to compute the full SOC series "
        "(columns: `Date Time`, `status`, `voltage`, `chargingCurrent`, `dischargingCurrent`, "
        "optionally `soc`)",
        type=["csv"], key="la_telemetry_upload",
    )
    if la_telemetry_file is not None:
        try:
            la_raw = pd.read_csv(la_telemetry_file)
            la_result = compute_leadacid_soc_series(
                la_raw, capacity_ah=la_capacity_ah,
                float_voltage_threshold=la_float_voltage, taper_current_frac=la_taper_frac,
                charge_efficiency=la_charge_efficiency,
                peukert_exponent=la_peukert_exponent,
                nominal_discharge_rate_a=la_nominal_rate_a,
            )
            show_cols = [c for c in ["Date Time", "status", "voltage", "chargingCurrent",
                                      "dischargingCurrent", "soc", "SOC_CoulombCounting"] if c in la_result.columns]
            st.dataframe(la_result[show_cols], use_container_width=True)
            chart_cols = [c for c in ["soc", "SOC_CoulombCounting"] if c in la_result.columns]
            if chart_cols:
                st.line_chart(la_result.set_index("Date Time")[chart_cols])
        except Exception as e:
            st.error(f"Could not process the lead-acid telemetry file: {e}")

    st.divider()


# =====================================================================
# UI — BACKUP TIME ESTIMATOR (NEW — additive only)
# Watts -> hours, for whichever battery is currently selected. Built from the
# vendor spec-sheet method, with the Peukert calibration above for lead-acid
# and a corrected current formula throughout (Amps = Watts / Volts).
# =====================================================================
st.header("⏱️ Backup Time Estimator (Watts → Hours)")
is_lead_acid = "Lead-Acid" in battery_choice
st.caption(
    "Reproduces the vendor spec-sheet's own methodology (Load → Current → capacity-derating → ÷ current "
    "→ × DoD → × system losses), corrected so current is always **Watts ÷ Volts** (see the flagged issue "
    "below for what the original Lithium sheet did instead)."
)
st.info(
    f"**Formula currently ACTIVE for this panel: {battery['name']}** (selected in the sidebar under "
    f"'2. Battery Configuration'). The two chemistries use different formulas below — this badge tells "
    f"you which one the numbers underneath are actually using."
)

fc1, fc2 = st.columns(2)
with fc1:
    st.markdown(f"**{'✅ ' if is_lead_acid else ''}Lead-Acid formula**")
    st.latex(r"I = \frac{P_{load}}{V_{nominal} \times V_{sag}}")
    st.latex(r"C_{eff} = \frac{C_{ah}}{\left(\dfrac{I}{I_{ref}}\right)^{k-1}} \ \text{if } I > I_{ref}")
    st.latex(r"t = \frac{C_{eff}}{I} \times DoD \times \eta_{sys}")
    st.caption(
        f"Peukert-derated (k={LEAD_ACID_PEUKERT_EXPONENT_DEFAULT:.2f}, reference ≈"
        f"{LEAD_ACID_PEUKERT_REFERENCE_RATE_HOURS_DEFAULT:.1f}h — calibrated from this fleet's own 200Ah "
        f"spec sheet), voltage sag ≈{LEAD_ACID_DISCHARGE_VOLTAGE_SAG_DEFAULT*100:.0f}% of nominal."
    )
with fc2:
    st.markdown(f"**{'✅ ' if not is_lead_acid else ''}Lithium formula**")
    st.latex(r"I = \frac{P_{load}}{V_{nominal}}")
    st.latex(r"t = \frac{C_{ah}}{I} \times DoD \times \eta_{sys}")
    st.caption(
        "No Peukert correction (k=1, i.e. no-op) and no voltage sag by default — lithium's usable "
        "capacity and voltage stay much flatter across discharge rates than lead-acid's."
    )

bc1, bc2, bc3, bc4 = st.columns(4)
bt_load_w = bc1.number_input("Backup load (W)", min_value=1.0, value=1500.0, step=50.0, key="bt_load_w")
bt_voltage = bc2.number_input(
    "Nominal system voltage (V)", min_value=1.0, value=48.0, step=1.0, key="bt_voltage",
    help="The pack's nominal voltage — e.g. 24 or 48 for these presets."
)
bt_capacity_ah = bc3.number_input(
    "Battery capacity (Ah)", min_value=1.0,
    value=200.0 if is_lead_acid else 314.0, step=1.0, key="bt_capacity_ah",
)
bt_dod = bc4.number_input(
    "DoD (%)", min_value=1.0, max_value=100.0,
    value=battery["dod"] * 100.0, step=1.0, key="bt_dod",
) / 100.0

bt_sag = st.number_input(
    "Average discharge voltage as % of nominal (voltage sag under load)",
    min_value=50.0, max_value=100.0,
    value=(LEAD_ACID_DISCHARGE_VOLTAGE_SAG_DEFAULT if is_lead_acid else LITHIUM_DISCHARGE_VOLTAGE_SAG_DEFAULT) * 100.0,
    step=1.0, key="bt_sag",
    help="Lead-acid sags under load — this fleet's own 200Ah spec sheet shows ~95% of nominal "
         "(22.8/24V, 45.6/48V) for BOTH voltage variants. Lithium holds voltage much flatter; 100% is a "
         "reasonable default unless you have the pack's real sag figure."
) / 100.0
bt_sys_eff = st.slider(
    "Additional system/inverter efficiency (%)", 50, 100,
    int(SYSTEM_EFFICIENCY_LOSS_DEFAULT * 100), key="bt_sys_eff",
    help="Losses beyond the battery itself (inverter conversion, wiring, etc.) — the vendor sheet's own "
         "last row implies roughly 87-94% for this product; tune to your actual inverter's rated efficiency."
) / 100.0

if is_lead_acid:
    bt_peukert_k = LEAD_ACID_PEUKERT_EXPONENT_DEFAULT
    bt_ref_hours = LEAD_ACID_PEUKERT_REFERENCE_RATE_HOURS_DEFAULT
    st.caption(
        f"Using the calibrated lead-acid Peukert model: exponent k={bt_peukert_k:.2f}, "
        f"reference rate ≈{bt_ref_hours:.1f}h (≈{bt_capacity_ah/bt_ref_hours:.1f}A on this capacity)."
    )
else:
    bt_peukert_k = 1.0
    bt_ref_hours = None
    st.caption("Lithium: no Peukert correction applied (its usable capacity barely changes with discharge rate).")

bt_result = estimate_backup_time_hours(
    load_watts=bt_load_w, nominal_voltage=bt_voltage, capacity_ah=bt_capacity_ah, dod=bt_dod,
    discharge_voltage_sag=bt_sag, peukert_exponent=bt_peukert_k,
    peukert_reference_rate_hours=bt_ref_hours, system_efficiency=bt_sys_eff,
)

rc1, rc2, rc3 = st.columns(3)
rc1.metric("Current drawn", f"{bt_result['current_a']:.2f} A")
rc2.metric("Backup time (after DoD)", f"{bt_result['backup_time_with_dod_h']:.2f} h")
rc3.metric("Backup time (after DoD + system losses)", f"{bt_result['backup_time_with_losses_h']:.2f} h")
st.caption(
    f"Current = {bt_load_w:.0f}W ÷ {bt_result['avg_discharge_voltage']:.2f}V (avg discharge voltage) = "
    f"{bt_result['current_a']:.2f}A. Effective capacity after Peukert = {bt_result['effective_capacity_ah']:.1f} Ah "
    f"(nameplate {bt_capacity_ah:.0f} Ah). Plain backup = {bt_result['backup_time_plain_h']:.2f}h → "
    f"×DoD({bt_dod*100:.0f}%) = {bt_result['backup_time_with_dod_h']:.2f}h → "
    f"×system-eff({bt_sys_eff*100:.0f}%) = {bt_result['backup_time_with_losses_h']:.2f}h."
)

if not is_lead_acid:
    with st.expander("⚠️ Flagged issue in the vendor Lithium spec sheet — worth fixing at the source"):
        wrong_current = bt_load_w / bt_capacity_ah
        st.markdown(f"""
The Lithium spec sheet you shared computes **"Maximum Current" as Load(W) ÷ Ah rating**, not
Load(W) ÷ Voltage — a units mismatch (Watts ÷ Amp-hours is not Amps). Checked against the exact
numbers in that sheet:

- 1500W / 24V pack, 314Ah → sheet shows Maximum Current = **5A**. That's `1500 ÷ 314 = {1500/314:.2f} ≈ 5`,
  **not** `1500 ÷ 24 = {1500/24:.1f}A` (the physically correct current).
- Same pattern at 3000W (`3000÷314={3000/314:.2f}≈10` vs correct `3000÷24={3000/24:.1f}A`) and
  4500W (`4500÷314={4500/314:.2f}≈14` vs correct `4500÷24={4500/24:.1f}A`).

Every backup-time figure downstream of that column (52.6h, 26.3h, 17.5h, and the DoD/loss-adjusted
rows) inherits this error and comes out **roughly 12-13x too optimistic**. For your current inputs here
(Load={bt_load_w:.0f}W, Voltage={bt_voltage:.0f}V, Capacity={bt_capacity_ah:.0f}Ah):

- What the sheet's formula implies: current = {wrong_current:.2f}A → backup ≈ {(bt_capacity_ah*bt_dod*bt_sys_eff/wrong_current) if wrong_current>0 else 0:.1f}h
- Physically correct: current = {bt_result['current_a']:.2f}A → backup ≈ {bt_result['backup_time_with_losses_h']:.2f}h

Worth getting this corrected in the source spec sheet before it's used for real sizing/customer-facing numbers.
""")


# =====================================================================
# UI — MAIN
# =====================================================================
if not uploaded:
    st.info("⬅️ Upload your 5-minute interval CSV in the sidebar to run the simulation.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

# =====================================================================
# DATA SANITY — FINDINGS & DECISIONS (NEW — for the developer)
# Runs on the RAW file, before any cleaning. Purely diagnostic — does not
# alter preprocess()'s behaviour. Every check here was derived from issues
# actually found reviewing a real 7-day Maharashtra field export.
# =====================================================================
st.header("🧪 Data Sanity — Findings & Decisions (For the Developer)")
st.caption(
    "Read-only diagnostics on the RAW uploaded file, run before any cleaning happens. This section "
    "does not change how the file gets processed below — it surfaces findings for a human to review "
    "and decide on, so the logic behind every data-quality decision is visible on this screen, not "
    "buried in code."
)
_sanity_findings = run_data_sanity_report(raw_df, grid_v_min=grid_v_min, grid_v_max=grid_v_max)
_sev_icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
_sev_order = {"critical": 0, "warning": 1, "info": 2}
_sanity_sorted = sorted(_sanity_findings, key=lambda f: _sev_order.get(f["severity"], 9))
_n_critical = sum(1 for f in _sanity_findings if f["severity"] == "critical")
_n_warning = sum(1 for f in _sanity_findings if f["severity"] == "warning")

if _n_critical:
    st.error(
        f"🔴 {_n_critical} CRITICAL finding(s) below — these can silently corrupt every downstream "
        f"number (wrong timestamp column, etc.) if not addressed before trusting results."
    )
elif _n_warning:
    st.warning(f"🟠 {_n_warning} warning(s) found — review before fully trusting the results below.")
else:
    st.success("No critical or warning-level issues found in this file's raw data.")

with st.expander(f"📋 View all {len(_sanity_sorted)} live finding(s) for THIS uploaded file", expanded=bool(_n_critical)):
    for f in _sanity_sorted:
        st.markdown(f"**{_sev_icon.get(f['severity'], '⚪')} [{f['area']}]** {f['finding']}")
        st.caption(f"➡️ Decision / recommendation: {f['decision']}")
        st.markdown("---")

with st.expander("📌 Deep-dive findings from the full plant export (plant_august.xlsx, 31,473 rows) — reference"):
    st.markdown("""
An earlier 7-day sample turned out to have its `Timestamp` column recorded as date-only (no time-of-day),
which would have silently broken the simulation — that finding is now superseded by this larger, corrected
export, which was confirmed to use `Timestamp` as a genuine full per-row date+time value. These are
additional findings from manually reviewing THIS actual file that go beyond what the automatic checks
above can catch on their own — kept here as a permanent reference for the developer:

**1. The file is not one month — it spans 112 days (2026-05-17 to 2026-09-05), and only 29% of its rows
fall in August.** Despite the filename, only 9,242 of the 31,473 rows (Aug 1 00:02 - Aug 31 23:57) are
actually August; the rest cover mid-May through early September. **Decision needed from the team: should
the app analyze the FULL 112-day range as-is, or should August specifically be isolated (e.g. a date-range
filter in the UI) before computing monthly savings/comparison figures?** Whichever is intended, the
person reading the output needs to know which period the numbers actually cover.

**2. `Timestamp` and `Date Time` are two DIFFERENT, both-valid full timestamps, offset by ~40-45 seconds**
(e.g. row 1: Timestamp=05:34:43, Date Time=05:35:26) — most likely device-send-time vs cloud-received-time,
not a data error. Per instruction, the app should use **`Timestamp`** as authoritative (it already does —
`preprocess()`'s required-column check looks for `Timestamp` by name, so this file works with no column
mapping change, unlike the earlier 7-day sample).

**3. Grid voltage swings much wider than the 7-day sample suggested: 0V to 268.2V.** 1,214 rows (3.9%)
read exactly 0V (candidate outages), and 467 rows (1.5%) exceed the default 260V upper threshold —
roughly 90x more over-260V rows than the smaller sample showed, because a longer window catches more of
the tail. **Decision: the default Grid Voltage band (180-260V) needs to be widened for this feeder before
trusting Grid_Status — 260V is clearly too tight; something like 265-270V looks safer given this data.**

**4. Data gaps are much more severe over a full month than the 7-day sample implied.** 34 gaps exceed 60
minutes; the worst is a ~24.7-hour gap starting 2026-07-20 09:03 (the logger was offline for essentially
a full day), with several other multi-hour gaps (10.5h on Jul 18, 7.3h on Jul 16, 7.0h on Jul 17, 6.7h on
Jul 26, and more). Under the current gap-tolerance/max-gap defaults (6 min / 30 min), **5.4% of the total
elapsed time in this file (≈6.0 days out of 111.8) falls into intervals long enough to be excluded from
energy integration entirely** — meaning close to a week's worth of real generation/consumption is
deliberately left uncounted rather than guessed at. **Decision: this is a large enough share that it
should be reported alongside any monthly total (e.g. "X kWh across Y% of the month's elapsed time, Z days
excluded due to data gaps") rather than presented as if it covers the full period.**

**5. `Machine type` is NOT constant in this file — it alternates between "Grid Tie" (29,325 rows) and
"Off Grid Tie" (2,148 rows), interleaved 15 separate times across the dataset** (not a clean before/after
split — e.g. it flips within the same afternoon on 2026-05-18). This does not feed into our
`simulate()` engine at all (the app always computes all three hypothetical inverter behaviours from the
PV/load/grid readings, regardless of what this field says), but it's worth understanding what's actually
happening on-site — repeated mode-switching, a firmware event, or possibly two devices' exports merged
together. **Flagged for the hardware/product team to confirm, not resolved here.**

**6. `Battery Voltage(V)` reads implausibly low (often 1.6-2V, min 0.2V) on 4,412 rows (14% of the file)**
for a nominal 48V bank — physically impossible as a real battery voltage, almost certainly a sensor
dropout or "no reading available" placeholder. These low readings occur across BOTH `Machine type`
values (2,301 rows under "Grid Tie", 2,111 under "Off Grid Tie"), so they don't cleanly correlate with
the mode flag above. **Decision: any battery-voltage-based logic (e.g. the Lead-Acid full-charge detector
using a float-voltage threshold) should treat sub-10V readings as invalid/missing rather than real data,
or these dropouts will falsely fail every full-charge check during those windows.**

**7. PV2 is entirely unused (constant 0W) across all 31,473 rows** — consistent with the 7-day sample;
this is very likely a genuine single-PV-string installation rather than a fault, given it holds across
the whole 112-day span.

**8. Original row order is NOT fully chronological** — 60 rows go backwards in time relative to the row
before them (all by less than an hour, likely minor export/merge interleaving, not large batch
misordering). `preprocess()` already sorts by timestamp so this self-heals, but it's a sign the source
export process isn't perfectly sequential.

**9. Cumulative generation counters still reset/glitch** across the full file, consistent with the 7-day
sample — continued confirmation that using only instantaneous PV/load power (never a cumulative counter)
is the right call.

**10. Otherwise clean**: no duplicate timestamps, no fully duplicate rows, no missing values, no negative
sensor readings, Battery Capacity(%) stays within a valid 0-100% range, and every calendar day in the
112-day span has at least some data (no fully-missing days).
""")

try:
    df = preprocess(raw_df, grid_v_min=grid_v_min, grid_v_max=grid_v_max, power_factor=power_factor,
                     cycle_days=cycle_days)
except Exception as e:
    st.error(f"Could not process the uploaded file: {e}")
    st.stop()

if df.empty:
    st.error("No valid rows found after cleaning the uploaded file.")
    st.stop()

# ---------------------------------------------------------------
# 1. RAW DATA SANITY HEADER
# ---------------------------------------------------------------
st.header("🧪 Raw Data Sanity Check")
n_days = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 86400
n_outage_rows = int((df["Grid_Status"] == 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Raw Load (kWh)", f"{df['Load_kWh'].sum():,.2f}")
c2.metric("Total Raw PV (kWh)", f"{df['PV_kWh'].sum():,.2f}")
c3.metric("Intervals / Span", f"{len(df):,} rows / {n_days:.1f} days")
c4.metric("Outage Intervals", f"{n_outage_rows:,} ({n_outage_rows*DT_HOURS:.1f} h)")
st.caption(
    f"Date range: {df['Timestamp'].min()} → {df['Timestamp'].max()}. "
    f"Formulas used: `PV_kWh = (PV1+PV2)/1000 × {DT_HOURS:.4f}h`, "
    f"`Load_kWh = AC_apparent_power(VA) × PF({power_factor:.2f})/1000 × {DT_HOURS:.4f}h`, "
    f"`Grid_Status = 1` if `{grid_v_min} ≤ V ≤ {grid_v_max}` else `0`, "
    f"billing cycle resets every {cycle_days} days. Verify these totals look sane before trusting the table below."
)

with st.expander("Preview cleaned data (first 20 rows)"):
    st.dataframe(
        df[["Timestamp", "PV_kWh", "Load_kWh", "Grid voltage(V)", "Grid_Status", "cycle_id"]].head(20),
        use_container_width=True,
    )

# ---------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------
res_gt = simulate(df, "grid_tie", tariff)
res_dumb = simulate(df, "dumb", tariff, battery, reserve_pct, max_charge_pct, pv_zero_threshold_kwh)
res_smart = simulate(df, "smart", tariff, battery, reserve_pct, max_charge_pct, pv_zero_threshold_kwh)

for res in (res_gt, res_dumb, res_smart):
    compute_financials(res, net_metering_enabled, export_rate)

results = {"grid_tie": res_gt, "dumb": res_dumb, "smart": res_smart}

# ---------------------------------------------------------------
# 2. MAIN COMPARISON TABLE
# ---------------------------------------------------------------
st.header("📊 Inverter Comparison")
summary_df = pd.DataFrame(
    {INVERTER_LABELS[inv]: [round(results[inv][k], 2) for k in METRIC_ORDER] for inv in INVERTER_ORDER},
    index=[METRIC_LABELS[k] for k in METRIC_ORDER],
)
st.dataframe(summary_df, use_container_width=True)
st.caption(
    "Note: **Total Household Demand** is identical across all three (same house, same appliances). "
    "**Total Load Served** is where Grid-Tie diverges — it shuts down completely during an outage, "
    "so anything the house wanted to consume during that window is simply not served. See the "
    "'Total Load Served (kWh)' expander below for the exact arithmetic."
)

# ---------------------------------------------------------------
# 3. DATA TRANSPARENCY EXPANDERS — one per metric
# ---------------------------------------------------------------
st.header("🔍 Data Transparency — Formula & Arithmetic Audit")
st.caption("Every metric: how it differs by inverter, the formula, and the real numbers plugged into it.")


def f2(x):
    return f"{x:,.2f}"


def render_metric(key: str, explanation_md: str, formula_latex: str, arithmetic_fn):
    with st.expander(f"{METRIC_LABELS[key]}"):
        st.markdown(explanation_md)
        st.latex(formula_latex)
        for inv in INVERTER_ORDER:
            st.markdown(f"**{INVERTER_LABELS[inv]}**")
            st.markdown(arithmetic_fn(inv, results[inv]))


# --- PV Available ---
render_metric(
    "pv_available",
    """
- **Grid-Tie**: dead during an outage, so any PV that *could* have been generated in that window is a pure
  **Opportunity Loss** — it never left the panels' potential.
- **Dumb / Smart Hybrid**: stay alive during an outage. PV is only wasted (**Notional Loss**) if the battery is
  already full *and* the load is already met — i.e. there is nowhere for the surplus to go (can't export to a dead grid).
    """,
    r"PV_{available} = \begin{cases}\text{Total PV} - \text{Opportunity Loss} & \text{Grid-Tie}\\ \text{Total PV} - \text{Notional Loss} & \text{Hybrid}\end{cases}",
    lambda inv, r: (
        f"Total PV ({f2(r['total_pv'])}) − Opportunity Loss ({f2(r['opportunity_loss'])}) = **{f2(r['pv_available'])} kWh**"
        if inv == "grid_tie" else
        f"Total PV ({f2(r['total_pv'])}) − Notional Loss ({f2(r['notional_loss'])}) = **{f2(r['pv_available'])} kWh**"
    ),
)

# --- Total Household Demand ---
render_metric(
    "total_demand",
    "Identical across all three inverters — this is the raw household demand (what the house *wanted* to "
    "consume). It doesn't change with the inverter type because it's measured upstream of the inverter's "
    "decision-making — the appliances draw what they draw regardless of what's powering them. Derived from "
    "**Apparent Power (VA)** × **Power Factor**, not read directly off an active-power column, since PF varies "
    "by site and isn't reliably reported by every inverter model.",
    r"Demand_{total} = \sum_i \frac{S_{AC,i} \times PF}{1000} \times \Delta t,\quad \Delta t = \frac{5}{60}\text{h}",
    lambda inv, r: f"Sum of all 5-min `AC output apparent power(VA) × PF({power_factor:.2f})/1000 × {DT_HOURS:.4f}h` readings = **{f2(r['total_demand'])} kWh**",
)

# --- Total Load Served ---
render_metric(
    "load_served",
    """
Because a Grid-Tie system shuts down during a grid outage, the house goes dark. Therefore, the actual load
served to the house is lower than the Hybrid systems by exactly the amount of the Unserved Load.

- **Grid-Tie**: loses the *entire* load for every outage interval (100% unserved during a cut) — the biggest gap
  vs. demand.
- **Dumb / Smart Hybrid**: stay alive through the outage via PV + battery, so they only lose the load that PV and
  battery together couldn't cover (typically much smaller, and zero if the battery never runs dry).
    """,
    r"Load_{served} = Demand_{total} - Load_{unserved}",
    lambda inv, r: (
        f"{f2(r['total_demand'])} kWh (Demand) − {f2(r['unserved'])} kWh (Unserved) = **{f2(r['load_served'])} kWh (Served)**"
    ),
)

# --- PV to Load ---
render_metric(
    "pv_to_load",
    """
- **Grid-Tie**: only during grid-up intervals (0 during outage — system is dead).
- **Dumb / Smart Hybrid**: in every interval, up or down, since the inverter stays energised — PV always serves
  load first before charging the battery or exporting.
    """,
    r"PV_{to\ load} = \sum_i \min(PV_i,\ Load_i)",
    lambda inv, r: f"Σ min(PV, Load) across all qualifying intervals = **{f2(r['pv_to_load'])} kWh**",
)

# --- PV Export ---
render_metric(
    "pv_export",
    """
Only possible while the grid is up (can never export to a dead grid).
- **Grid-Tie**: any PV surplus over load exports immediately.
- **Dumb Hybrid**: PV surplus charges the battery first, *then* exports.
- **Smart Hybrid**: same as Dumb Hybrid on grid-up surplus intervals (battery already gets priority charge).
    """,
    r"PV_{export} = \sum_{i \in Grid\ Up} \max(PV_i - Load_i - Charge_i,\ 0)",
    lambda inv, r: f"Σ leftover PV after load + battery charge, grid-up only = **{f2(r['pv_export'])} kWh**",
)

# --- Battery to Load ---
render_metric(
    "batt_to_load",
    """
- **Grid-Tie**: N/A — no battery.
- **Dumb Hybrid**: strictly **0 while grid is up** (emergency-backup only, never discharges to load); discharges
  only during an outage, all the way down to Min SOC (0% of usable). While the grid is up it instead *charges*
  from the grid like a UPS float-charge whenever PV drops near zero (see Grid → Battery Charging below).
- **Smart Hybrid**: discharges in **both** states — during grid-up whenever `PV < Load` (stopping at the Reserve
  SOC to protect outage readiness), and during an outage it drains fully to Min SOC just like Dumb Hybrid.
    """,
    r"Batt_{to\ load} = \underbrace{\sum_{Grid\ Up,\ PV<Load} \min\!\Big(SOC_i-Reserve,\ \tfrac{Load_i-PV_i}{\eta}\Big)\eta}_{\text{Smart only}} + \underbrace{\sum_{Grid\ Down} \min\!\Big(SOC_i,\ \tfrac{Load_{rem,i}}{\eta}\Big)\eta}_{\text{Dumb \& Smart}}",
    lambda inv, r: (
        "N/A (no battery) = **0.00 kWh**" if inv == "grid_tie" else
        f"Battery discharged to load = **{f2(r['batt_to_load'])} kWh** (final SOC at end of dataset: {f2(r['final_soc'])} / {f2(r['usable_kwh'])} kWh usable)"
    ),
)

# --- Grid to Battery Charging ---
render_metric(
    "grid_to_battery",
    f"""
- **Grid-Tie**: no battery → always 0.
- **Dumb Hybrid**: battery is still backup-only (never discharges to load while grid is up), but it now *charges*
  from the grid like a UPS float-charge whenever PV generation drops below {pv_zero_threshold_w:.0f} W — while PV
  is generating above that, charging can only come from PV surplus (Generation − Load), never the grid. Either
  way it charges up to the {int(max_charge_pct*100)}% max-charge ceiling, not literal 100%.
- **Smart Hybrid**: proactively pulls from the grid at night if SOC drops below the safety Reserve SOC, so it
  isn't caught short if the grid drops overnight. Unlike Dumb Hybrid this only applies outside solar hours
  and only tops up to Reserve, not the full max-charge ceiling.
    """,
    r"Grid_{to\ battery}^{Dumb} = \sum_{i \in Grid\ Up,\ PV_i \le PV_{thresh}} \frac{MaxSOC - SOC_i}{\eta}\qquad Grid_{to\ battery}^{Smart} = \sum_{i \in Grid\ Up,\ \neg Solar\ Hr,\ SOC_i<Reserve} \frac{Reserve - SOC_i}{\eta}",
    lambda inv, r: (
        "N/A (no battery) = **0.00 kWh**" if inv == "grid_tie" else
        f"Σ (MaxSOC − SOC) / η across grid-up intervals with PV ≤ {pv_zero_threshold_w:.0f} W = **{f2(r['grid_to_battery'])} kWh**"
        if inv == "dumb" else
        f"Σ (Reserve − SOC) / η across qualifying night-time intervals = **{f2(r['grid_to_battery'])} kWh**"
    ),
)

# --- Grid Import ---
render_metric(
    "grid_import",
    """
Always 0 during an outage (system dead for Grid-Tie, and Dumb/Smart Hybrid never import from a dead grid — deficit
falls to battery/unserved instead). During grid-up:
- **Grid-Tie**: `Load − PV_to_load`.
- **Dumb Hybrid**: `Load − PV_to_load` (battery never assists the load), **plus** any UPS-style `Grid_to_battery`
  top-up pulls whenever PV is near zero.
- **Smart Hybrid**: `Load − PV_to_load − Battery_to_load`, **plus** any night-time `Grid_to_battery` top-up pulls.
    """,
    r"Grid_{import} = \sum_{i \in Grid\ Up} \max(Load_i - PV_{to\ load,i} - Batt_{to\ load,i},\ 0)\ +\ Grid_{to\ battery}",
    lambda inv, r: (
        f"Σ (Load − PV_to_load − Battery_to_load), grid-up only = **{f2(r['grid_import'])} kWh**"
        if inv == "grid_tie" else
        f"Σ (Load − PV_to_load − Battery_to_load) + Grid_to_battery ({f2(r['grid_to_battery'])} kWh) = **{f2(r['grid_import'])} kWh**"
    ),
)

# --- Unserved Load ---
render_metric(
    "unserved",
    """
- **Grid-Tie**: 100% of load during every outage interval (system is fully dead).
- **Dumb / Smart Hybrid**: only the load remaining after PV *and* battery are exhausted during an outage
  (i.e. battery hits Min SOC and load still isn't met).
    """,
    r"Unserved = \sum_{i \in Grid\ Down} \max(Load_i - PV_{to\ load,i} - Batt_{to\ load,i},\ 0)",
    lambda inv, r: (
        f"Σ Load during outage (system dead) = **{f2(r['unserved'])} kWh**" if inv == "grid_tie" else
        f"Σ (Load − PV_to_load − Battery_to_load) during outage = **{f2(r['unserved'])} kWh**"
    ),
)

# --- Solar Savings ---
render_metric(
    "solar_savings",
    """
The RESIDUAL of the true bill delta after removing Outage Savings and TOD Optimization from it — by
construction this guarantees no kWh's value is ever counted in more than one savings bucket:

`Solar Savings = (Base Bill − Actual Bill) − Outage Savings − TOD Optimization`

- **Base Bill** = time+slab+cycle-aware bill on the FULL household demand (the "no solar at all" counterfactual —
  same rate rules: slab tier, TOD surcharge %, solar-hour rebate).
- **Actual Bill** = time+slab+cycle-aware bill on what was REALLY imported (serving load *and* any battery
  charging), net-metering-settled if that mode is selected.

The solar-hour rebate is a board discount, not an inverter saving — it's baked directly into both bills' per-unit
rate (see Tariff & Billing Cycle in the sidebar), never added again here.
    """,
    r"Solar_{savings} = (Base\ Bill - Actual\ Bill) - Outage_{savings} - TOD_{optimization}",
    lambda inv, r: (
        f"(Base Bill ₹{f2(r['baseline_bill'])} − Actual Bill ₹{f2(r['actual_bill'])}) "
        f"− Outage Savings ₹{f2(r['outage_savings'])} − TOD Optimization ₹{f2(r['tod_optimization'])} "
        f"= **₹{f2(r['solar_savings'])}**"
    ),
)

# --- TOD Optimization Savings ---
render_metric(
    "tod_optimization",
    """
**Smart Hybrid only** (Dumb Hybrid never discharges to load while grid is up; Grid-Tie has no battery — both
always ₹0 here). Credits battery energy discharged to load, while the grid is up, during the configured
TOD-surcharge (peak-price) window:

- **PV-origin** battery kWh (free to begin with) is credited in **full** at the current avoided rate.
- **Grid-origin** battery kWh (charged earlier from the grid) is credited only the **difference** between the
  avoided rate now and the rate it actually cost when it was charged — because that unit was already billed once,
  inside Actual Bill. Crediting its full value again here would double-count it.
    """,
    r"TOD_{opt} = \sum_{TOD\ window,\ Grid\ Up} \Big[\ \Delta_{PV}\cdot r_{now}\ +\ \Delta_{grid}\cdot(r_{now}-r_{charge})\ \Big]",
    lambda inv, r: (
        "N/A — always ₹0 (no grid-up battery discharge in this mode)" if inv != "smart" else
        f"**₹{f2(r['tod_optimization'])}**"
    ),
)

# --- Outage Savings ---
render_metric(
    "outage_savings",
    """
- **Grid-Tie**: always ₹0 — the system is dead during an outage, so it never serves any load then.
- **Dumb / Smart Hybrid**: every kWh served by PV or battery *while the grid is down* is valued at the
  **actual grid price** that would have applied at that exact slab-cycle position and time of day (including the
  solar-hour rebate if the outage falls in solar hours) — not a flat VoLL.
    """,
    r"Outage_{savings} = \sum_{i \in Grid\ Down} \big(PV_{to\ load,i} + Batt_{to\ load,i}\big) \times rate_{baseline}(t_i,\ cum_i)",
    lambda inv, r: f"{f2(r['outage_served'])} kWh valued row-by-row at the applicable slab/TOD rate = **₹{f2(r['outage_savings'])}**",
)

# --- Solar Earning ---
render_metric(
    "solar_earning",
    """
Only used for **Non-Net-Metering**. Kept in its own column — never added into Solar Savings.
Settled quantity is `min(cycle PV export, cycle grid import)`, paid at the configured export rate.
For **Net Metering**, this is always ₹0 because export is settled 1:1 against import inside Actual Bill instead.
    """,
    r"Solar\ Earning = \sum_{cycles} \min(Export_{cycle},\ Import_{cycle}) \times ExportRate",
    lambda inv, r: (
        f"{f2(r['settlement_kwh'])} kWh settled × ₹{export_rate:.2f} = **₹{f2(r['solar_earning'])}**"
        if not r["net_metering"] else
        f"Net metering selected → Solar Earning = **₹0.00** (settlement handled inside Actual Bill; "
        f"{f2(r['settlement_kwh'])} kWh settled 1:1)"
    ),
)

# --- Net Savings ---
render_metric(
    "net_savings",
    "The sum of exactly three disjoint financial levers, per spec — Solar Savings, Outage Savings, and TOD "
    "Optimization (₹0 when not applicable). Solar Earning is real export income, reported separately and NOT "
    "added here, so it never inflates this figure.",
    r"Net_{savings} = Solar_{savings} + Outage_{savings} + TOD_{optimization}",
    lambda inv, r: (
        f"₹{f2(r['solar_savings'])} + ₹{f2(r['outage_savings'])} + ₹{f2(r['tod_optimization'])} "
        f"= **₹{f2(r['net_savings'])}**"
    ),
)

st.divider()
st.caption(
    "All figures above are computed live, row-by-row, from the uploaded dataset — nothing is hard-coded. "
    "Battery SOC is tracked statefully (split into PV-origin and grid-origin buckets) across the full time "
    "series for each inverter type independently."
)