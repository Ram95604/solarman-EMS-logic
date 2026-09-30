"""
Solar savings simulator — single-device IoT workbook (sheets `Device` + `Battery`).

Pipeline:  read workbook -> adapt_site_data() -> prepare_rows() -> engines -> compute_financials()

  * Grid-Tie       : no-battery counterfactual on the same PV / load / grid timeline.
  * Hybrid (live)  : battery energy read DIRECTLY from batterySoc (per-row ΔSOC × rated capacity) —
                     nothing about the battery is simulated.

Every energy figure is power × each row's REAL elapsed time (dt_hours); there is no fixed sampling interval.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import time as dtime

st.set_page_config(page_title="Solar Savings Simulator", layout="wide")

# =====================================================================
# CONSTANTS
# =====================================================================
# Gap handling for the per-row elapsed time (dt_hours):
#   <= 6 min  normal logger jitter — integrate with the real interval
#   6–30 min  short gap — still integrate with the real interval, flagged for audit
#   > 30 min  device offline — row contributes NO energy (dt_hours = 0)
GAP_NORMAL_MAX_MIN = 6.0
GAP_SHORT_MAX_MIN = 30.0

TELEMETRY_MSGTYPE = "DeviceStatus"      # BatteryInfo rows carry no telemetry on either sheet
JOIN_KEYS = ["receivedAt", "messageType"]

REQUIRED_DEVICE_FIELDS = [
    "receivedAt", "messageType", "gridVoltage", "activePowerOutput",
    "pvConnected", "pvVoltageMeasured", "pvCurrentMeasured", "pvPower",
]
REQUIRED_BATTERY_FIELDS = ["receivedAt", "messageType", "batterySoc", "batteryChemistry", "batteryVoltage"]

CHEMISTRY_LABELS = {0: "Lead-Acid", 1: "Lithium"}   # batteryChemistry: 0 = lead-acid, 1 = lithium

# Residential tariff, FY 2026-27. Per-unit price of a grid kWh =
#   (slab energy charge + wheeling + FPPA/FAC  − solar-hour ToD rebate) × (1 + peak surcharge on energy)
#   × (1 + electricity duty)
# Fixed charges are identical with and without solar, so they are shown separately and never move a saving.
TARIFF_PRESETS = {
    "Uttar Pradesh (UPPCL LMV-1 urban)": dict(
        # UPERC tariff order FY 2026-27 (FY 2025-26 rates retained). ED 5%. No ToD for domestic.
        # Net metering (UPERC RSPV Regulations 2019): surplus carried forward cycle to cycle; unadjusted
        # credits at the end of the settlement period paid at ₹2/kWh (confirm the current rate with UPPCL).
        slabs=[(150, 5.50), (300, 6.00), (float("inf"), 6.50)],
        wheeling=0.0, fppa=0.0, duty_pct=5.0, fixed_per_kw=110.0, fixed_flat=0.0,
        cycle_days=30, solar_start=dtime(9, 0), solar_end=dtime(18, 0),
        rebate_default=False, rebate_rate=0.0, surcharge_default=False, surcharge_pct=0.0,
        tod_start=dtime(18, 0), tod_end=dtime(22, 0),      # UP domestic has no ToD: evening peak used for TOD Optimization only
        yearend_buyback=2.00,
        note="Energy 0–150u ₹5.50 · 151–300u ₹6.00 · >300u ₹6.50; electricity duty 5%; fixed ₹110/kW/month; "
             "no domestic ToD.",
    ),
    "Maharashtra (MSEDCL LT-I residential)": dict(
        # MERC MYT order (Case 217 of 2024), FY 2026-27: energy 4.32 / 9.40 / 12.51 / 13.97, wheeling ₹1.20,
        # fixed ₹130/month (single phase), ED 16%. ToD for LT-domestic with a ToD/smart meter: rebate ₹0.80/unit
        # 09:00–17:00, no peak surcharge. Net metering (MERC Rooftop RE Regulations 2019, reg. 11.4): surplus
        # carried forward; unadjusted units bought at the generic tariff at FY end (₹2.90 was the last figure
        # found, FY 2021-22 — confirm the current year's).
        slabs=[(100, 4.32), (300, 9.40), (500, 12.51), (float("inf"), 13.97)],
        wheeling=1.20, fppa=0.0, duty_pct=16.0, fixed_per_kw=0.0, fixed_flat=130.0,
        cycle_days=30, solar_start=dtime(9, 0), solar_end=dtime(17, 0),
        rebate_default=True, rebate_rate=0.80, surcharge_default=False, surcharge_pct=0.0,
        tod_start=dtime(17, 0), tod_end=dtime(0, 0),       # MERC peak zone 17:00–24:00 (0% for residential)
        yearend_buyback=2.90,
        note="Energy 0–100u ₹4.32 · 101–300u ₹9.40 · 301–500u ₹12.51 · >500u ₹13.97 + wheeling ₹1.20/unit; "
             "electricity duty 16%; fixed ₹130/month; ToD rebate ₹0.80/unit 09:00–17:00 (ToD/smart meter).",
    ),
}

# capacity_kwh = rated (nameplate) energy. batterySoc % is a % of this.
BATTERY_PRESETS = {
    "Lead-Acid (24V / 200Ah)": dict(name="Lead-Acid", capacity_kwh=4.8, dod=0.50, efficiency=0.80, chemistry=0, nominal_v=24),
    "Lead-Acid (48V / 200Ah)": dict(name="Lead-Acid", capacity_kwh=9.6, dod=0.50, efficiency=0.80, chemistry=0, nominal_v=48),
    "Lithium (24V / 100Ah)": dict(name="Lithium", capacity_kwh=2.4, dod=0.80, efficiency=0.95, chemistry=1, nominal_v=24),
    "Lithium (48V / 280Ah)": dict(name="Lithium", capacity_kwh=13.44, dod=0.80, efficiency=0.95, chemistry=1, nominal_v=48),
    "Lithium (48V / 314Ah)": dict(name="Lithium", capacity_kwh=15.07, dod=0.80, efficiency=0.95, chemistry=1, nominal_v=48),
}
PRESET_BY_PACK = {   # (batteryChemistry, pack voltage) -> preset
    (0, 24): "Lead-Acid (24V / 200Ah)",
    (0, 48): "Lead-Acid (48V / 200Ah)",
    (1, 24): "Lithium (24V / 100Ah)",
    (1, 48): "Lithium (48V / 280Ah)",
}

MODES = ["grid_tie", "hybrid"]
MODE_LABELS = {"grid_tie": "Grid-Tie", "hybrid": "Hybrid (live batterySoc)"}

METRIC_ORDER = [
    "pv_available", "total_demand", "load_served", "pv_to_load", "pv_export", "batt_to_load",
    "grid_to_battery", "grid_import", "unserved",
    "baseline_bill", "actual_bill",
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
    "baseline_bill": "Expected Bill — no solar/battery (Rs)",
    "actual_bill": "Actual Bill — really paid (Rs)",
    "solar_savings": "Solar Savings (Rs)",
    "tod_optimization": "TOD Optimization Savings (Rs)",
    "outage_savings": "Outage Savings (Rs)",
    "solar_earning": "Solar Earning — PV Export (Rs)",
    "net_savings": "Net Savings (Rs)",
}

# Backup-time estimator (vendor spec-sheet method, calibrated on the fleet's 200Ah lead-acid sheet)
LEAD_ACID_PEUKERT_EXPONENT = 1.45
LEAD_ACID_PEUKERT_REFERENCE_HOURS = 9.7
LEAD_ACID_VOLTAGE_SAG = 0.95
LITHIUM_VOLTAGE_SAG = 1.0
SYSTEM_EFFICIENCY = 0.92


# =====================================================================
# TARIFF HELPERS  (slabs travel inside the tariff dict)
# =====================================================================
def marginal_slab_rate(cum_units: float, slabs: list) -> float:
    """Rate for the NEXT unit given `cum_units` already consumed this billing cycle."""
    cum_units = max(cum_units, 0.0)
    for cap, rate in slabs:
        if cum_units < cap - 1e-9:
            return rate
    return slabs[-1][1]


def slab_bill_for_lump_kwh(kwh: float, tariff: dict) -> float:
    """Bill for one lump quantity starting from 0 units (net-metering settlement): progressive energy slabs
    + per-unit wheeling/FPPA, then electricity duty. No ToD — a net meter reports one net number."""
    if kwh <= 0:
        return 0.0
    remaining, bill, prev_cap = kwh, 0.0, 0.0
    for cap, rate in tariff["slabs"]:
        used = min(remaining, cap - prev_cap)
        if used > 0:
            bill += used * rate
            remaining -= used
        prev_cap = cap
        if remaining <= 1e-9:
            break
    bill += kwh * (tariff.get("wheeling", 0.0) + tariff.get("fppa", 0.0))
    return bill * (1.0 + tariff.get("duty_pct", 0.0) / 100.0)


def in_window(t: dtime, start: dtime, end: dtime) -> bool:
    """True if t is in [start, end), handling windows that wrap past midnight."""
    if start <= end:
        return start <= t < end
    return t >= start or t < end


def effective_rate(t: dtime, cum_units: float, tariff: dict) -> float:
    """All-in ₹ for the next grid unit at time t:
    (slab energy [× peak surcharge in the TOD window] + wheeling + FPPA − solar-hour rebate) × (1 + duty)."""
    energy = marginal_slab_rate(cum_units, tariff["slabs"])
    if tariff.get("surcharge_on") and in_window(t, tariff["tod_start"], tariff["tod_end"]):
        energy *= 1.0 + tariff.get("tod_pct", 0.0) / 100.0
    rate = energy + tariff.get("wheeling", 0.0) + tariff.get("fppa", 0.0)
    if tariff.get("rebate_on") and in_window(t, tariff["solar_start"], tariff["solar_end"]):
        rate -= tariff.get("rebate_rate", 0.0)
    return max(rate, 0.0) * (1.0 + tariff.get("duty_pct", 0.0) / 100.0)


# =====================================================================
# READ + ADAPT THE DEVICE / BATTERY WORKBOOK
# =====================================================================
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_site_workbook(file) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Return the (Device, Battery) sheets. Sheet names are matched case-insensitively."""
    xl = pd.ExcelFile(file)
    names = {str(s).strip().lower(): s for s in xl.sheet_names}
    if "device" not in names or "battery" not in names:
        raise ValueError(f"Expected sheets 'Device' and 'Battery'; found {list(xl.sheet_names)}.")
    dev, bat = xl.parse(names["device"]), xl.parse(names["battery"])
    dev.columns = [str(c).strip() for c in dev.columns]
    bat.columns = [str(c).strip() for c in bat.columns]
    missing = [f"Device.{c}" for c in REQUIRED_DEVICE_FIELDS if c not in dev.columns] + \
              [f"Battery.{c}" for c in REQUIRED_BATTERY_FIELDS if c not in bat.columns]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}.")
    return dev, bat


def _finding(key, severity, title, decision, evidence, question=None, **extra) -> dict:
    return dict(key=key, severity=severity, title=title, decision=decision,
                evidence=evidence, question=question, **extra)


def adapt_site_data(dev: pd.DataFrame, bat: pd.DataFrame, *, tz_shift_hours: float = 5.5,
                    trim_commissioning: bool = True) -> "tuple[pd.DataFrame, list[dict]]":
    """
    Turn one device's Device + Battery sheets into clean per-row signals:
        Timestamp (local), PV_W, Load_W, Grid_V, SOC_pct
    and list every judgement call made on the way (`findings`, rendered in the UI).
    """
    F: "list[dict]" = []

    # ---- 1. Join the two sheets on (receivedAt, messageType) ------------------------------------
    # Not by row position: the Device sheet can carry extra blank trailing rows. A per-key occurrence
    # counter keeps the join exact even if two messages share a key.
    dev = dev.dropna(subset=JOIN_KEYS, how="all").copy()
    bat = bat.dropna(subset=JOIN_KEYS, how="all").copy()
    for f in (dev, bat):
        f["receivedAt"] = pd.to_datetime(f["receivedAt"], errors="coerce")
        f["_k"] = f.groupby(JOIN_KEYS, dropna=False).cumcount()
    bat_only = [c for c in bat.columns if c not in dev.columns]
    merged = dev.merge(bat[JOIN_KEYS + ["_k"] + bat_only], on=JOIN_KEYS + ["_k"], how="left") \
                .drop(columns="_k").reset_index(drop=True)
    tele = merged["messageType"].astype(str).eq(TELEMETRY_MSGTYPE)
    probe = [c for c in bat_only if c not in ("imei", "topic")]
    matched = int(merged.loc[tele, probe].notna().any(axis=1).sum()) if probe else 0
    F.append(_finding(
        "sheet_join", "info", "Device + Battery sheets joined on receivedAt + messageType",
        "Key join (exact per-message match), not row position.",
        f"Device sheet {len(dev)} rows, Battery sheet {len(bat)} rows after dropping fully blank rows; "
        f"{matched} of {int(tele.sum())} {TELEMETRY_MSGTYPE} rows carry battery values after the join."))

    # ---- 2. Keep telemetry rows only ------------------------------------------------------------
    n_all = len(merged)
    counts = merged["messageType"].value_counts(dropna=False).to_dict()
    merged = merged[tele]
    if n_all != len(merged):
        F.append(_finding(
            "msgtype_filter", "info", f"Dropped {n_all - len(merged)} of {n_all} rows that carry no telemetry",
            f"Kept only messageType == '{TELEMETRY_MSGTYPE}' rows.",
            f"messageType counts: {counts}. Every telemetry field is blank on the other message type(s)."))
    before = len(merged)
    merged = merged[merged["gridVoltage"].notna()]
    if before != len(merged):
        F.append(_finding(
            "blank_telemetry", "warning", f"Dropped a further {before - len(merged)} telemetry rows that were entirely blank",
            "Rows with a null gridVoltage were removed.",
            f"{before - len(merged)} of {before} {TELEMETRY_MSGTYPE} rows had no telemetry values at all.",
            "Why do some DeviceStatus messages arrive with an empty payload?"))
    merged = merged.sort_values("receivedAt")
    pvc_whole_file = int(_num(merged["pvConnected"]).eq(1).sum())

    # ---- 3. Commissioning / install stretch ----------------------------------------------------
    # A new install shows hours of ~zero load on the hybrid output before the house is connected.
    # "Load appears" = first run of 5 consecutive rows above 50 W. Only a LEADING stretch under a
    # quarter of the file is trimmed; anything longer is reported, not removed.
    pvc_trimmed = 0
    if len(merged) > 10:
        on = (_num(merged["activePowerOutput"]).fillna(0.0) > 50.0).astype(int).values
        run = pd.Series(on).rolling(5).sum().values
        first = int(np.argmax(run >= 5)) - 4 if (run >= 5).any() else None
        if first is not None and first > 0:
            t0, t1 = merged["receivedAt"].iloc[0], merged["receivedAt"].iloc[first]
            frac = first / len(merged)
            if trim_commissioning and frac < 0.25:
                pvc_trimmed = int(_num(merged["pvConnected"].iloc[:first]).eq(1).sum())
                merged = merged.iloc[first:]
                F.append(_finding(
                    "commissioning", "warning",
                    f"Trimmed {first} leading rows with no load on the hybrid output (install stretch)",
                    "Rows before load first appears are excluded from the calculation.",
                    f"Load stayed at or below 50 W from the first telemetry row until "
                    f"{t1 + pd.Timedelta(hours=tz_shift_hours):%d %b %H:%M} local time "
                    f"({(t1 - t0).total_seconds() / 3600:.1f} h, {frac:.0%} of rows).",
                    "Confirm the install date/time so the commissioning window comes from the job record."))
            else:
                F.append(_finding(
                    "commissioning", "warning" if frac >= 0.25 else "info",
                    f"{first} leading rows show no load on the hybrid output",
                    "Kept (trimming is switched off)." if not trim_commissioning else
                    "Kept — the stretch is too long to be treated as commissioning.",
                    f"Load stayed at or below 50 W for the first {first} rows ({frac:.0%} of the file).",
                    "Were the house loads connected to the hybrid output during this period?"))

    merged = merged.reset_index(drop=True)
    if len(merged) == 0:
        F.append(_finding("empty", "critical", "No usable telemetry rows in this file", "Nothing to calculate.",
                          f"All {n_all} rows were dropped as non-telemetry or blank.",
                          "Is this device reporting at all?"))
        return pd.DataFrame(columns=["Timestamp", "PV_W", "Load_W", "Grid_V", "SOC_pct"]), F

    raw_ts = merged["receivedAt"]
    out = pd.DataFrame(index=merged.index)
    ts = raw_ts
    try:
        ts = ts.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    out["Timestamp"] = ts + pd.to_timedelta(tz_shift_hours, unit="h")

    # ---- 4. PV, chosen by pvConnected ------------------------------------------------------------
    #   pvConnected = 0 -> PV on the grid-tie inverter, seen by the IoT string sensor: voltage × current.
    #                      (pvPowerMeasured is NOT used — it rounds current down to whole amps, ~10% low.)
    #   pvConnected = 1 -> PV switched to the hybrid during an outage: pvPower (+ pv2Power if present).
    zero = pd.Series(0.0, index=merged.index)
    pv1 = _num(merged["pvPower"]).fillna(0.0)
    pv2 = _num(merged.get("pv2Power", zero)).fillna(0.0)
    vm = _num(merged["pvVoltageMeasured"]).fillna(0.0)
    im = _num(merged["pvCurrentMeasured"]).fillna(0.0)
    pvc = _num(merged["pvConnected"]).fillna(0.0).eq(1.0)
    out["PV_W"] = np.where(pvc, pv1 + pv2, vm * im)
    F.append(_finding(
        "pv_rule", "info", "PV read by pvConnected: grid-tie string (voltage × current) or hybrid PV1 during outages",
        "pvConnected = 0 → pvVoltageMeasured × pvCurrentMeasured; pvConnected = 1 → pvPower (PV1)"
        + (" + pv2Power" if float(pv2.abs().sum()) > 0 else "") + ".",
        f"pvConnected = 1 on {int(pvc.sum())} of {len(pvc)} rows used in the calculation "
        f"({pvc_whole_file} in the whole file; {pvc_trimmed} of those fell in the trimmed install stretch). "
        f"pvPower is non-zero on {int(((pv1 > 0) & pvc).sum())} of the {int(pvc.sum())}. pv2Power is "
        + ("zero on every row." if float(pv2.abs().sum()) == 0 else "non-zero on some rows."),
        "Confirm that PV1 (pvPower) is the hybrid's PV input."))

    if "pvPowerMeasured" in merged.columns:
        pvm = _num(merged["pvPowerMeasured"]).fillna(0.0)
        pos = (vm * im) > 0
        if int(pos.sum()) >= 10:
            trunc = (np.abs(pvm[pos] - vm[pos] * np.floor(im[pos])) <= 0.02 * vm[pos] + 1.0).mean()
            dt_h = raw_ts.diff().dt.total_seconds().div(3600)
            dt_h = dt_h.where(dt_h <= GAP_SHORT_MAX_MIN / 60.0, 0.0).fillna(0.0)
            e_vi = float((vm * im * dt_h).sum()) / 1000.0
            e_pm = float((pvm * dt_h).sum()) / 1000.0
            if trunc > 0.5 and e_vi > 0:
                F.append(_finding(
                    "pv_truncation", "warning",
                    f"pvPowerMeasured under-reports PV by {(1 - e_pm / e_vi):.1%} (current rounded down to whole amps)",
                    "Not used — voltage × current is used instead.",
                    f"On {trunc:.0%} of rows with PV, pvPowerMeasured = pvVoltageMeasured × floor(pvCurrentMeasured). "
                    f"Energy: {e_pm:.2f} kWh as reported vs {e_vi:.2f} kWh from voltage × current.",
                    "Can the firmware compute pvPowerMeasured from the full-resolution current?"))

    # ---- 5. Timezone check against daylight ------------------------------------------------------
    # After the shift almost no PV energy should fall between 19:00 and 05:00 local.
    w = pd.Series(np.asarray(out["PV_W"], dtype=float), index=merged.index).clip(lower=0)

    def night_share(shift_h):
        h = (raw_ts + pd.to_timedelta(shift_h, unit="h")).dt.hour
        return float(w[(h >= 19) | (h < 5)].sum() / w.sum()) if float(w.sum()) > 0 else float("nan")

    if float(w.sum()) > 0 and int((w > 0).sum()) >= 20:
        alt = 0.0 if abs(tz_shift_hours) > 1e-9 else 5.5
        ns, ns_alt = night_share(tz_shift_hours), night_share(alt)
        ok = ns <= 0.05
        hrs = out["Timestamp"].dt.hour + out["Timestamp"].dt.minute / 60.0
        first_pv, last_pv = float(hrs[w > 0].min()), float(hrs[w > 0].max())
        F.append(_finding(
            "timezone", "info" if ok else "critical",
            f"Timestamps shifted by {tz_shift_hours:+.1f} h — "
            + ("no PV at night, consistent with local time" if ok else f"{ns:.0%} of PV energy falls at night"),
            f"`receivedAt` shifted {tz_shift_hours:+.1f} h (5.5 = UTC to IST).",
            f"After the shift, PV appears between {int(first_pv):02d}:{int(first_pv % 1 * 60):02d} and "
            f"{int(last_pv):02d}:{int(last_pv % 1 * 60):02d} local time; {ns:.1%} of PV energy falls between "
            f"19:00 and 05:00. With a {alt:+.1f} h shift instead, that share would be {ns_alt:.1%}.",
            None if ok else (f"The data fits a {alt:+.1f} h shift better — check the device's time zone."
                             if ns_alt < ns else "PV appears at night under either shift — check the device clock.")))
    else:
        F.append(_finding(
            "timezone", "warning", f"Timestamps shifted by {tz_shift_hours:+.1f} h (not enough PV to verify)",
            f"`receivedAt` shifted {tz_shift_hours:+.1f} h.",
            "Too little PV in this file to check the shift against daylight hours.", "Confirm receivedAt is UTC."))

    # ---- 6. Load and grid voltage ----------------------------------------------------------------
    out["Load_W"] = _num(merged["activePowerOutput"]).fillna(0.0)     # real active power, no power factor
    gv = _num(merged["gridVoltage"]).fillna(0.0)
    out["Grid_V"] = gv
    n_zero = int((gv <= 1.0).sum())
    pct_zero = 100.0 * n_zero / max(len(gv), 1)
    if pct_zero >= 50.0:
        F.append(_finding(
            "grid_zero", "critical",
            f"gridVoltage reads 0 V on {n_zero} of {len(gv)} rows ({pct_zero:.0f}%) — treated as real outages",
            "0 V taken at face value (grid down).",
            f"A {pct_zero:.0f}% outage rate is implausible for a live site; the Expected Bill counts grid-up rows only.",
            "Is gridVoltage = 0 a real outage, or 'not reported this cycle'?"))

    if "statusCode" in merged.columns and merged["statusCode"].notna().any():
        sc = merged["statusCode"].astype(str).str.strip().str.upper()
        agree = float((sc.eq("L") == (gv > 150.0)).mean())
        F.append(_finding(
            "status_code", "info" if agree >= 0.98 else "warning",
            f"gridVoltage and the hybrid's statusCode agree on grid up/down for {agree:.1%} of rows",
            "Grid status comes from the voltage band; statusCode is only a cross-check.",
            f"statusCode counts: {sc.value_counts().to_dict()}. Disagreements are usually single-row switchover transients.",
            "Please confirm the statusCode letters (L = line, B = battery, C = charging, S = standby)."))

    # ---- 7. Battery chemistry and pack voltage ---------------------------------------------------
    chem = _num(merged["batteryChemistry"]).dropna()
    uniq = sorted(chem.unique().tolist())
    if uniq:
        chem_val = int(chem.mode().iloc[0])
        label = CHEMISTRY_LABELS.get(chem_val, f"unknown ({chem_val})")
        if len(uniq) == 1:
            F.append(_finding("chemistry", "info", f"Battery chemistry detected: {label}",
                              f"Read from `batteryChemistry` = {chem_val}.",
                              f"batteryChemistry is constant at {chem_val} across all telemetry rows.", value=chem_val))
        else:
            F.append(_finding("chemistry", "critical", "This device reports MORE THAN ONE battery chemistry",
                              f"Used the most frequent value ({chem_val} = {label}).",
                              f"batteryChemistry value counts: {chem.value_counts().to_dict()}.",
                              "Did the battery change mid-window, or is this field unreliable?", value=chem_val))

    bv = _num(merged["batteryVoltage"])
    bv = bv[bv > 5.0]
    if len(bv) >= 5:
        med = float(bv.median())
        nom = 12 if med < 17 else (24 if med < 36 else (48 if med < 64 else None))
        F.append(_finding(
            "battery_pack", "info" if nom in (24, 48) else "warning",
            f"Battery pack voltage: {nom} V system" if nom else "Battery pack voltage not recognised",
            f"Battery preset matched on {nom} V." if nom in (24, 48) else "Pick the battery preset manually.",
            f"batteryVoltage median {med:.1f} V (range {bv.min():.1f}–{bv.max():.1f} V) over {len(bv)} rows.",
            "Confirm the installed battery's nameplate Ah — it cannot be derived from telemetry.", value=nom))

    if "batteryType" in merged.columns:
        bt = sorted(_num(merged["batteryType"]).dropna().unique().tolist())
        if bt and set(bt) - {0.0, 1.0}:
            F.append(_finding(
                "battery_type_field", "info", "`batteryType` is NOT the chemistry flag — chemistry comes from batteryChemistry",
                "Chemistry read from `batteryChemistry`; `batteryType` ignored.",
                f"batteryType values seen here: {bt}. Only batteryChemistry fits '0 = Lead-Acid / 1 = Lithium'.",
                "What do batteryType's values mean?"))

    # ---- 8. batterySoc — the only SOC source, used directly ---------------------------------------
    soc = _num(merged["batterySoc"])
    valid = soc.where(soc > 0).dropna()
    out["SOC_pct"] = soc.where(soc > 0).clip(lower=0, upper=100)
    if valid.empty:
        F.append(_finding(
            "soc_source", "critical", "No usable batterySoc in this file",
            "Battery flows will read zero (no SOC change can be observed).",
            "batterySoc is absent or zero on every telemetry row.",
            "Is batterySoc reported for this battery/BMS?"))
    else:
        dis = _num(merged.get("batteryDischargingCurrent", zero)).fillna(0.0)
        big = dis >= 10.0
        flat = bool(big.any()) and float(soc[big].max() - soc[big].min()) <= 1.0 and \
            float(valid.quantile(0.95) - valid.quantile(0.05)) <= 2.0
        F.append(_finding(
            "soc_source", "warning" if flat else "info",
            "batterySoc does not yet respond to discharge — used as-is" if flat else "Battery energy read directly from batterySoc",
            "Battery kWh per row = ΔbatterySoc × rated capacity." + (
                " Battery-to-load will read low until batterySoc is calibrated." if flat else ""),
            f"batterySoc range {valid.min():.0f}–{valid.max():.0f}%."
            + (f" It stays there even on the {int(big.sum())} rows where the battery discharges at 10 A or more "
               f"(up to {dis.max():.0f} A)." if flat else ""),
            "When will batterySoc calibration be complete?" if flat else None))
        last = valid.index[-1]
        status = _num(merged.get("batteryStatus", pd.Series(np.nan, index=merged.index)))
        ichg = _num(merged.get("batteryChargingCurrent", pd.Series(np.nan, index=merged.index)))
        vbat = _num(merged["batteryVoltage"])
        state = {0: "idle", 1: "discharging", 2: "charging"}.get(
            int(status.loc[last]) if pd.notna(status.loc[last]) else -1, "unknown")
        F.append(_finding(
            "battery_now", "info", f"Battery charged to {float(soc.loc[last]):.0f}% at the latest reading",
            "Current charge level = most recent batterySoc reading.",
            f"Latest reading {out.loc[last, 'Timestamp']:%d %b %Y %H:%M} local time; battery {state}"
            + (f", {vbat.loc[last]:.1f} V" if pd.notna(vbat.loc[last]) else "")
            + (f", charging {ichg.loc[last]:.0f} A" if pd.notna(ichg.loc[last]) and ichg.loc[last] > 0 else "") + ".",
            value=float(soc.loc[last]), flat=flat))

    # ---- 9. Cumulative counters (would give an independent cross-check) --------------------------
    cum_cols = [c for c in ["cumulativeLoad", "cumulativeGridPower", "cumulativeGridExport", "pvEnergyToday",
                            "pvEnergyTotal", "batteryCumulativeChargeEnergy", "batteryCumulativeDischargeEnergy"]
                if c in merged.columns]
    all_zero = [c for c in cum_cols if float(_num(merged[c]).fillna(0).abs().sum()) == 0.0]
    if all_zero:
        F.append(_finding(
            "cumulative_zero", "warning", f"{len(all_zero)} cumulative energy counter(s) read zero throughout",
            "No independent cross-check available on the computed kWh totals.",
            f"All-zero counters: {', '.join(all_zero)}. These would let us validate power × time integration.",
            "Will these counters start reporting real values?"))

    # ---- 10. Hybrid standby draw (the hybrid inverter's own consumption) -------------------------
    # gridPower is the hybrid's AC input, not the utility meter. On grid-up rows:
    #   standby = median(gridPower − load − battery charging power)
    if all(c in merged.columns for c in ["gridPower", "batteryChargingCurrent"]):
        gp = _num(merged["gridPower"]).fillna(0.0)
        ld = out["Load_W"]
        chg_w = _num(merged["batteryVoltage"]).fillna(0.0) * _num(merged["batteryChargingCurrent"]).fillna(0.0)
        gu = (gv >= 150.0) & (gp > 0)
        if int(gu.sum()) >= 20:
            standby_w = float((gp - ld - chg_w)[gu].clip(lower=0).median())
            F.append(_finding(
                "hybrid_standby", "info", f"Hybrid standby draw measured: ≈ {standby_w:.0f} W",
                "Pre-filled into the sidebar's hybrid standby input (billed on grid-up rows, PV surplus first).",
                f"Median of (gridPower − load − battery charging power) over {int(gu.sum())} grid-up rows. "
                f"gridPower itself is NOT used as the utility import.",
                "Are all household circuits behind the hybrid? If some bypass it, load is understated.",
                standby_w=standby_w))

    # ---- 11. Span and cadence ----------------------------------------------------------------------
    tsv = out["Timestamp"].dropna().sort_values()
    if len(tsv) > 1:
        span_days = (tsv.max() - tsv.min()).total_seconds() / 86400.0
        gaps_min = tsv.diff().dt.total_seconds().div(60).dropna()
        if span_days < 28:
            F.append(_finding(
                "short_span", "critical", f"Only {span_days:.2f} days of data — too short for a billing simulation",
                "Calculation still runs, but bill/saving figures are NOT a monthly result.",
                f"Data spans {tsv.min():%Y-%m-%d %H:%M} to {tsv.max():%Y-%m-%d %H:%M}; a partial window never "
                f"climbs into the higher monthly slabs.",
                "Can we get 30+ continuous days, ideally aligned to the meter's billing-cycle start?"))
        n_big = int((gaps_min > GAP_SHORT_MAX_MIN).sum())
        if n_big:
            F.append(_finding(
                "big_gaps", "warning", f"{n_big} gap(s) longer than {GAP_SHORT_MAX_MIN:.0f} minutes",
                "Rows after such a gap get dt_hours = 0, so they contribute no energy.",
                f"Median cadence {gaps_min.median():.2f} min, worst gap {gaps_min.max():.0f} min.",
                "Device dropouts, connectivity loss, or genuine power-down?"))

    # ---- 12. Constant-value (test rig) detection --------------------------------------------------
    if len(out) >= 5 and all(out[c].nunique(dropna=True) <= 1 for c in ["Load_W", "PV_W", "Grid_V"]):
        F.append(_finding(
            "constant_values", "critical", "Every telemetry value is identical on every row — looks like a test rig",
            "Data passed through unchanged, but results from it are meaningless.",
            f"Load, PV and grid voltage each hold one constant value across all {len(out)} rows.",
            "Is this a bench/test unit?"))

    return out, F


def finding(findings: "list[dict]", key: str) -> "dict | None":
    return next((f for f in findings if f["key"] == key), None)


def detected_battery_preset(findings: "list[dict]") -> "str | None":
    """batteryChemistry + measured pack voltage -> BATTERY_PRESETS key."""
    chem = (finding(findings, "chemistry") or {}).get("value")
    volt = (finding(findings, "battery_pack") or {}).get("value")
    if chem is not None and volt is not None and (int(chem), int(volt)) in PRESET_BY_PACK:
        return PRESET_BY_PACK[(int(chem), int(volt))]
    if chem is not None and CHEMISTRY_LABELS.get(int(chem)):
        return next((k for k in BATTERY_PRESETS if k.startswith(CHEMISTRY_LABELS[int(chem)])), None)
    return None


# =====================================================================
# PER-ROW ENERGY TABLE
# =====================================================================
def prepare_rows(rows: pd.DataFrame, grid_v_min: float, grid_v_max: float, cycle_days: int,
                 cycle_start_date=None) -> pd.DataFrame:
    """Add dt_hours, kW/kWh, grid status, time of day and billing-cycle id to the adapted rows."""
    df = rows.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    for c in ["PV_W", "Load_W", "Grid_V"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["PV_W", "Load_W", "Grid_V"]).reset_index(drop=True)

    # Each row's real elapsed time since the previous row. The first row has none, so it borrows the
    # second row's interval.
    dt_raw_h = df["Timestamp"].diff().dt.total_seconds() / 3600.0
    if len(dt_raw_h) > 1 and pd.notna(dt_raw_h.iloc[1]):
        dt_raw_h.iloc[0] = dt_raw_h.iloc[1]
    elif len(dt_raw_h):
        dt_raw_h.iloc[0] = 0.0
    dt_min = dt_raw_h * 60.0
    df["gap_flag"] = np.where(dt_min <= GAP_NORMAL_MAX_MIN, "normal",
                              np.where(dt_min <= GAP_SHORT_MAX_MIN, "gap_interpolated", "gap_excluded"))
    df["dt_hours"] = np.where(df["gap_flag"] == "gap_excluded", 0.0, dt_raw_h)

    df["PV_kW"] = (df["PV_W"] / 1000.0).clip(lower=0)
    df["Load_kW"] = (df["Load_W"] / 1000.0).clip(lower=0)
    df["PV_kWh"] = (df["PV_kW"] * df["dt_hours"]).clip(lower=0)
    df["Load_kWh"] = (df["Load_kW"] * df["dt_hours"]).clip(lower=0)
    df["Grid_Status"] = ((df["Grid_V"] >= grid_v_min) & (df["Grid_V"] <= grid_v_max)).astype(int)
    df["t_of_day"] = df["Timestamp"].dt.time

    # Billing cycle: anchored to the real bill-cycle start date if given, else the first day in the file.
    anchor = pd.Timestamp(cycle_start_date).normalize() if cycle_start_date is not None \
        else df["Timestamp"].min().normalize()
    df["cycle_id"] = np.floor((df["Timestamp"].dt.normalize() - anchor).dt.days / cycle_days).astype(int)
    return df


# =====================================================================
# ENGINES
# =====================================================================
class _Ledger:
    """Row-by-row billing shared by both engines (slab position is cycle-cumulative, never reset daily)."""

    def __init__(self, tariff: dict):
        self.tariff = tariff
        self.baseline_bill = self.actual_bill_gross = 0.0
        self.base_cum = self.act_cum = 0.0
        self.cycle = None
        self.rows = {k: [] for k in ("cycle_ids", "grid_import", "pv_export", "actual_bill",
                                     "dates", "baseline_bill", "tod", "outage_bill", "cum_before")}

    def start_row(self, row):
        if row.cycle_id != self.cycle:
            self.base_cum = self.act_cum = 0.0
            self.cycle = row.cycle_id
        self.r_import = self.r_export = self.r_bill = self.r_base = self.r_tod = self.r_outage = 0.0
        self.cum_before = self.base_cum
        # "No solar, no battery" counterfactual: grid-up rows only (a plain house is dark and billed
        # nothing during an outage).
        self.rate_baseline = effective_rate(row.t_of_day, self.base_cum, self.tariff)
        if row.Grid_Status:
            self.r_base = self.rate_baseline * row.Load_kWh
            self.baseline_bill += self.r_base
            self.base_cum += row.Load_kWh

    def import_kwh(self, t, kwh: float) -> float:
        """Bill `kwh` of real grid import at the current cycle position; returns the rate used."""
        rate = effective_rate(t, self.act_cum, self.tariff)
        amt = rate * kwh
        self.actual_bill_gross += amt
        self.r_bill += amt
        self.act_cum += kwh
        self.r_import += kwh
        return rate

    def end_row(self, row):
        r = self.rows
        r["cycle_ids"].append(row.cycle_id)
        r["grid_import"].append(self.r_import)
        r["pv_export"].append(self.r_export)
        r["actual_bill"].append(self.r_bill)
        r["dates"].append(pd.Timestamp(row.Timestamp).normalize())
        r["baseline_bill"].append(self.r_base)
        r["tod"].append(self.r_tod)
        r["outage_bill"].append(self.r_outage)
        r["cum_before"].append(self.cum_before)

    def result(self) -> dict:
        r = self.rows
        return dict(
            baseline_bill=self.baseline_bill, actual_bill_gross=self.actual_bill_gross,
            row_cycle_ids=r["cycle_ids"], row_grid_import=r["grid_import"], row_pv_export=r["pv_export"],
            row_actual_bill=r["actual_bill"], row_dates=r["dates"], row_baseline_bill=r["baseline_bill"],
            row_tod=r["tod"], row_outage_bill=r["outage_bill"], row_cum_before=r["cum_before"],
        )


def simulate_grid_tie(df: pd.DataFrame, tariff: dict) -> dict:
    """No battery. Grid up: PV serves load, surplus exports, deficit imports. Grid down: inverter is off."""
    L = _Ledger(tariff)
    pv_to_load = export = grid_import = unserved = opportunity_loss = 0.0
    for row in df.itertuples(index=False):
        L.start_row(row)
        pv, load = row.PV_kWh, row.Load_kWh
        if row.Grid_Status:
            u = min(pv, load)
            pv_to_load += u
            export += pv - u
            L.r_export += pv - u
            imp = load - u
            if imp > 0:
                L.import_kwh(row.t_of_day, imp)
                grid_import += imp
        else:
            unserved += load
            opportunity_loss += pv
        L.end_row(row)

    total_pv, total_demand = float(df["PV_kWh"].sum()), float(df["Load_kWh"].sum())
    return dict(
        mode="grid_tie", total_pv=total_pv, opportunity_loss=opportunity_loss, notional_loss=0.0,
        pv_available=total_pv - opportunity_loss, total_demand=total_demand, load_served=total_demand - unserved,
        pv_to_load=pv_to_load, pv_export=export, pv_to_battery=0.0, batt_to_load=0.0,
        grid_import=grid_import, grid_to_battery=0.0, unserved=unserved, outage_served=0.0,
        hybrid_standby=0.0, hybrid_standby_grid=0.0, tod_kwh=0.0, outage_bill_notional=0.0, tod_optimization=0.0, **L.result(),
    )


def sustained_discharge_rows(soc_pct: np.ndarray, grid_up: np.ndarray, min_drop_pct: float) -> np.ndarray:
    """
    True on rows that belong to a REAL, sustained grid-up discharge — not ±1-2% SOC noise.
    An episode starts where batterySoc falls below the previous reading (grid up) and lasts while the grid stays
    up and SOC stays below that starting level. Small up-ticks inside the episode are tolerated; climbing back to
    the starting level ends it. The episode counts only if SOC fell by at least `min_drop_pct` in total, so a
    98 → 97 → 98 wobble is ignored while a steady 80 → 79 → 79 → 78 evening discharge is kept.
    """
    n = len(soc_pct)
    flag = np.zeros(n, dtype=bool)
    i = 1
    while i < n:
        if grid_up[i] and soc_pct[i] < soc_pct[i - 1] - 1e-9:
            start_level, low, j = soc_pct[i - 1], soc_pct[i], i
            while j < n and grid_up[j] and soc_pct[j] < start_level - 1e-9:
                low = min(low, soc_pct[j])
                j += 1
            if start_level - low >= min_drop_pct - 1e-9:
                flag[i:j] = True
            i = j
        else:
            i += 1
    return flag


def simulate_hybrid_live_soc(df: pd.DataFrame, tariff: dict, battery: dict, standby_w: float = 0.0,
                             tod_min_drop_pct: float = 2.0) -> dict:
    """
    Hybrid result driven DIRECTLY by batterySoc (column SOC_pct). Per row:
        ΔkWh = ΔSOC% / 100 × rated capacity
        ΔSOC < 0  -> battery discharged: delivered to load (× discharge efficiency), capped at the load
                     PV couldn't cover.
        ΔSOC > 0  -> battery charged: from PV surplus first (÷ charge efficiency), the rest from the grid.
        ΔSOC = 0  -> battery idle: PV serves load, surplus exports, deficit imports.
    Missing readings carry the last value forward (= no change). The hybrid's own standby draw is the only
    other input: served by PV surplus first, then billed from the grid, on grid-up rows.
    TOD Optimization: battery energy delivered to load while the grid is UP, inside the TOD window, and only
    during a sustained discharge (see sustained_discharge_rows) — the telemetry signature of a Smart Hybrid
    choosing to run on battery. A Dumb Hybrid never discharges with the grid up, so it scores ₹0.
    Charge is tracked in PV-origin / grid-origin buckets so TOD credit for grid-charged energy counts only
    the rate difference (that energy was already billed once).
    """
    capacity_kwh = battery["capacity_kwh"]
    usable_kwh = capacity_kwh * battery["dod"]
    charge_eff = discharge_eff = float(np.sqrt(battery["efficiency"])) if battery["efficiency"] > 0 else 0.0
    standby_kw = max(float(standby_w or 0.0), 0.0) / 1000.0

    soc_pct = pd.to_numeric(df["SOC_pct"], errors="coerce").clip(lower=0, upper=100).ffill().bfill().fillna(0.0)
    soc_kwh = soc_pct / 100.0 * capacity_kwh
    delta_kwh = soc_kwh.diff().fillna(0.0).to_numpy()
    sustained = sustained_discharge_rows(soc_pct.to_numpy(), df["Grid_Status"].to_numpy() == 1, tod_min_drop_pct)

    soc_pv = float(soc_kwh.iloc[0]) if len(df) else 0.0   # starting charge counted as PV-origin
    soc_grid = grid_cost_basis = 0.0

    def discharge_buckets(gross):
        nonlocal soc_pv, soc_grid
        from_pv = min(soc_pv, gross)
        from_grid = gross - from_pv
        soc_pv = max(soc_pv - from_pv, 0.0)
        soc_grid = max(soc_grid - from_grid, 0.0)
        return from_pv, from_grid

    L = _Ledger(tariff)
    pv_to_load = export = batt_to_load = grid_import = unserved = 0.0
    pv_to_battery = notional_loss = outage_served = grid_to_battery = hybrid_standby = hybrid_standby_grid = 0.0
    tod_optimization = outage_bill_notional = tod_kwh = 0.0

    for i, row in enumerate(df.itertuples(index=False)):
        L.start_row(row)
        pv, load, t, d = row.PV_kWh, row.Load_kWh, row.t_of_day, float(delta_kwh[i])

        if row.Grid_Status:
            if standby_kw > 0:
                aux = standby_kw * float(row.dt_hours)
                aux_pv = min(aux, max(pv - load, 0.0))
                pv -= aux_pv
                hybrid_standby += aux
                if aux - aux_pv > 0:
                    L.import_kwh(t, aux - aux_pv)
                    grid_import += aux - aux_pv
                    hybrid_standby_grid += aux - aux_pv

            u = min(pv, load)
            pv_to_load += u
            deficit, surplus = max(load - pv, 0.0), max(pv - load, 0.0)

            if d < -1e-9:                                   # observed discharge
                gross = min(-d, usable_kwh)
                from_pv, from_grid = discharge_buckets(gross)
                delivered = gross * discharge_eff
                used = min(delivered, deficit) if deficit > 0 else 0.0
                scale = used / delivered if delivered > 1e-12 else 0.0
                batt_to_load += used
                rem = max(deficit - used, 0.0)
                if rem > 0:
                    L.import_kwh(t, rem)
                    grid_import += rem
                if used > 0 and sustained[i] and in_window(t, tariff["tod_start"], tariff["tod_end"]):
                    tod_kwh += used
                    r_now = effective_rate(t, L.act_cum, tariff)
                    L.r_tod = from_pv * discharge_eff * scale * r_now + \
                        from_grid * discharge_eff * scale * (r_now - grid_cost_basis)
                    tod_optimization += L.r_tod
                export += surplus
                L.r_export += surplus

            elif d > 1e-9:                                  # observed charge: PV surplus first, then grid
                charge = min(d, usable_kwh)
                from_pv = min(charge, surplus * charge_eff)
                from_grid = max(charge - from_pv, 0.0)
                soc_pv += from_pv
                pv_used = from_pv / charge_eff if charge_eff > 0 else 0.0
                pv_to_battery += pv_used
                export += max(surplus - pv_used, 0.0)
                L.r_export += max(surplus - pv_used, 0.0)
                if from_grid > 0:
                    pull = from_grid / charge_eff if charge_eff > 0 else 0.0
                    rate = L.import_kwh(t, pull)
                    grid_to_battery += pull
                    grid_import += pull
                    new_grid = soc_grid + from_grid
                    grid_cost_basis = (soc_grid * grid_cost_basis + from_grid * rate) / new_grid if new_grid > 1e-12 else 0.0
                    soc_grid = new_grid
                if deficit > 0:
                    L.import_kwh(t, deficit)
                    grid_import += deficit

            else:                                           # battery idle
                export += surplus
                L.r_export += surplus
                if deficit > 0:
                    L.import_kwh(t, deficit)
                    grid_import += deficit

        else:                                               # outage: no grid leg
            u = min(pv, load)
            pv_to_load += u
            load_rem, pv_rem = max(load - u, 0.0), max(pv - u, 0.0)
            served = u
            if d < -1e-9:
                gross = min(-d, usable_kwh)
                discharge_buckets(gross)
                used = min(gross * discharge_eff, load_rem)
                batt_to_load += used
                load_rem = max(load_rem - used, 0.0)
                served += used
                notional_loss += pv_rem
            elif d > 1e-9:
                from_pv = min(min(d, usable_kwh), pv_rem * charge_eff)
                soc_pv += from_pv
                pv_used = from_pv / charge_eff if charge_eff > 0 else 0.0
                pv_to_battery += pv_used
                notional_loss += max(pv_rem - pv_used, 0.0)
            else:
                notional_loss += pv_rem
            outage_served += served
            L.r_outage = served * L.rate_baseline
            outage_bill_notional += L.r_outage
            unserved += load_rem

        L.end_row(row)

    total_pv, total_demand = float(df["PV_kWh"].sum()), float(df["Load_kWh"].sum())
    return dict(
        mode="hybrid", total_pv=total_pv, opportunity_loss=0.0, notional_loss=notional_loss,
        pv_available=total_pv - notional_loss, total_demand=total_demand, load_served=total_demand - unserved,
        pv_to_load=pv_to_load, pv_export=export, pv_to_battery=pv_to_battery, batt_to_load=batt_to_load,
        grid_import=grid_import, grid_to_battery=grid_to_battery, unserved=unserved, outage_served=outage_served,
        hybrid_standby=hybrid_standby, hybrid_standby_grid=hybrid_standby_grid, tod_kwh=tod_kwh,
        sustained_rows=int(sustained.sum()),
        outage_bill_notional=outage_bill_notional, tod_optimization=tod_optimization,
        final_soc_pct=float(soc_pct.iloc[-1]) if len(df) else 0.0,
        final_soc=float(soc_kwh.iloc[-1]) if len(df) else 0.0, capacity_kwh=capacity_kwh,
        **L.result(),
    )


# =====================================================================
# FINANCIALS
# =====================================================================
def _fiscal_year(d: pd.Timestamp) -> int:
    """Indian financial year label: Apr 2026 – Mar 2027 -> 2026."""
    return d.year if d.month >= 4 else d.year - 1


def compute_financials(res: dict, tariff: dict, net_metering: bool, export_rate: float) -> dict:
    """
    Expected (baseline) bill : no-solar counterfactual, grid-up rows only.
    Actual bill              : Non-net-metering — the live per-row bill of real grid import.
                               Net metering — per billing cycle, export offsets import 1:1. Any surplus export
                               goes into a BANK that keeps carrying forward cycle after cycle and offsets later
                               cycles' import first. The remaining net import is re-billed from 0 through the slabs
                               (+ wheeling/FPPA, + duty). At each financial-year end (31 March) whatever is still
                               banked is bought back at the state's year-end rate and the bank resets to 0.
    TOD Optimization         : sustained grid-up battery discharge in the TOD window (already inside Base − Actual).
    Outage Savings           : energy served during outages at the rate that would have applied.
    Solar Savings            : (Base − Actual) − TOD Optimization
    Solar Earning            : non-net-metering — full export × export rate; net metering — year-end buyback.
    Net Savings              : Solar Savings + Outage Savings + TOD Optimization (Solar Earning shown separately)
    """
    rows = pd.DataFrame({"cycle_id": res["row_cycle_ids"], "date": res["row_dates"],
                         "imp": res["row_grid_import"], "exp": res["row_pv_export"]})
    cyc = rows.groupby("cycle_id").agg(start=("date", "min"), end=("date", "max"),
                                       import_kwh=("imp", "sum"), export_kwh=("exp", "sum")).reset_index()
    buyback_rate = float(tariff.get("yearend_buyback", 0.0))

    if net_metering:
        bank, recs, payout = 0.0, [], 0.0
        prev_fy = None
        for c in cyc.itertuples(index=False):
            fy = _fiscal_year(pd.Timestamp(c.start))
            paid_kwh = 0.0
            if prev_fy is not None and fy != prev_fy and bank > 0:     # a 31 March passed: settle the bank
                paid_kwh, payout, bank = bank, payout + bank * buyback_rate, 0.0
            prev_fy = fy
            bank_in = bank
            offset_now = min(c.import_kwh, c.export_kwh)                # same-cycle 1:1 netting
            from_bank = min(c.import_kwh - offset_now, bank)            # banked units used next
            net_kwh = c.import_kwh - offset_now - from_bank
            bank = bank - from_bank + (c.export_kwh - offset_now)       # this cycle's surplus is banked
            recs.append(dict(cycle=int(c.cycle_id), start=c.start, end=c.end, import_kwh=c.import_kwh,
                             export_kwh=c.export_kwh, fy_buyback_kwh=paid_kwh, bank_in_kwh=bank_in,
                             offset_same_cycle_kwh=offset_now, used_from_bank_kwh=from_bank,
                             net_billed_kwh=net_kwh, bank_out_kwh=bank,
                             bill=slab_bill_for_lump_kwh(net_kwh, tariff)))
        table = pd.DataFrame(recs)
        actual_bill = float(table["bill"].sum())
        solar_earning = payout
        settled = float((table["offset_same_cycle_kwh"] + table["used_from_bank_kwh"]).sum())
        res["nm_bank_end_kwh"] = bank
        res["nm_bank_end_value"] = bank * buyback_rate
        res["nm_buyback_kwh"] = float(table["fy_buyback_kwh"].sum())
        res["nm_cycles"] = table
    else:
        actual_bill = res["actual_bill_gross"]
        solar_earning = float(cyc["export_kwh"].sum()) * export_rate
        settled = 0.0
        res["nm_bank_end_kwh"] = res["nm_bank_end_value"] = res["nm_buyback_kwh"] = 0.0
        res["nm_cycles"] = None

    res.update(
        actual_bill=actual_bill,
        outage_savings=res["outage_bill_notional"],
        solar_savings=(res["baseline_bill"] - actual_bill) - res["tod_optimization"],
        solar_earning=solar_earning,
        settlement_kwh=settled,
        export_kwh=float(cyc["export_kwh"].sum()),
        net_metering=net_metering,
    )
    res["net_savings"] = res["solar_savings"] + res["outage_savings"] + res["tod_optimization"]
    return res


def compute_daily_earning(res: dict, slabs: list) -> pd.DataFrame:
    """The same three savings per calendar day, from row values already priced at the cycle-cumulative
    slab position. Net-metering settlement is not split across days (it only settles per cycle)."""
    rows = pd.DataFrame({
        "date": res["row_dates"], "baseline": res["row_baseline_bill"], "actual": res["row_actual_bill"],
        "tod": res["row_tod"], "outage_bill": res["row_outage_bill"], "cum_before": res["row_cum_before"],
    })
    daily = rows.groupby("date").agg(
        baseline_bill=("baseline", "sum"), actual_bill=("actual", "sum"), tod_optimization=("tod", "sum"),
        outage_savings=("outage_bill", "sum"), cycle_units_at_day_start=("cum_before", "min"),
        cycle_units_at_day_end=("cum_before", "max"),
    ).reset_index()
    daily["solar_savings"] = (daily["baseline_bill"] - daily["actual_bill"]) - daily["tod_optimization"]
    daily["today_earning"] = daily["solar_savings"] + daily["tod_optimization"] + daily["outage_savings"]
    daily["slab_rate_at_day_start"] = daily["cycle_units_at_day_start"].apply(lambda u: marginal_slab_rate(u, slabs))
    return daily.sort_values("date").reset_index(drop=True)


# =====================================================================
# OUTAGE DISTRIBUTION
# =====================================================================
def compute_outage_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per contiguous Grid_Status == 0 episode; durations from real dt_hours."""
    is_out = (df["Grid_Status"] == 0).to_numpy()
    ts, solar, dt_h = df["Timestamp"].to_numpy(), df["solar_hr"].to_numpy(), df["dt_hours"].to_numpy()
    episodes, i, n = [], 0, len(is_out)
    while i < n:
        if not is_out[i]:
            i += 1
            continue
        j = i
        while j < n and is_out[j]:
            j += 1
        start = pd.Timestamp(ts[i])
        end = pd.Timestamp(ts[j]) if j < n else pd.Timestamp(ts[j - 1])   # first grid-up reading after it
        seg_dt, seg_solar = dt_h[i:j], solar[i:j]
        episodes.append(dict(start=start, end=end, n_rows=j - i, duration_min=float(seg_dt.sum()) * 60.0,
                             solar_min=float(seg_dt[seg_solar].sum()) * 60.0,
                             offsolar_min=float(seg_dt[~seg_solar].sum()) * 60.0, start_hour=start.hour))
        i = j
    return pd.DataFrame(episodes)


def render_outage_distribution(df: pd.DataFrame):
    ep = compute_outage_episodes(df)
    if ep.empty:
        st.success("No outage episodes in this file.")
        return
    total_min = ep["duration_min"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Outage Episodes", f"{len(ep):,}")
    m2.metric("Total Outage Duration", f"{total_min / 60:,.1f} h")
    m3.metric("Longest Episode", f"{ep['duration_min'].max() / 60:,.2f} h")
    m4.metric("Average Episode", f"{ep['duration_min'].mean():,.1f} min")
    if total_min > 0:
        pct_solar = ep["solar_min"].sum() / total_min * 100
        st.caption(f"☀️ Solar hours: {ep['solar_min'].sum() / 60:,.2f} h ({pct_solar:.1f}%) · "
                   f"🌙 Off-solar hours: {ep['offsolar_min'].sum() / 60:,.2f} h ({100 - pct_solar:.1f}%) of outage time.")
    bins = [0, 15, 30, 60, 180, 360, 1440, np.inf]
    labels = ["<15m", "15-30m", "30-60m", "1-3h", "3-6h", "6-24h", ">24h"]
    ep["bucket"] = pd.cut(ep["duration_min"], bins=bins, labels=labels, right=False)
    st.markdown("**Episodes by duration (minutes, split by window)**")
    st.bar_chart(ep.groupby("bucket", observed=False)[["solar_min", "offsolar_min"]].sum()
                 .reindex(labels).fillna(0.0)
                 .rename(columns={"solar_min": "Solar-hours minutes", "offsolar_min": "Off-solar-hours minutes"}))
    st.markdown("**Hour of day each episode starts**")
    st.bar_chart(ep["start_hour"].value_counts().reindex(range(24), fill_value=0).sort_index().rename("Episodes"))
    with st.expander(f"All {len(ep)} episode(s)"):
        show = ep[["start", "end", "duration_min", "solar_min", "offsolar_min"]].copy()
        show["start"] = show["start"].dt.strftime("%Y-%m-%d %H:%M")
        show["end"] = show["end"].dt.strftime("%Y-%m-%d %H:%M")
        show.columns = ["Start", "Grid back", "Duration (min)", "Solar-hours (min)", "Off-solar-hours (min)"]
        st.dataframe(show.round(1), use_container_width=True)


# =====================================================================
# BACKUP-TIME ESTIMATOR (standalone calculator)
# =====================================================================
def estimate_backup_time_hours(load_watts, nominal_voltage, capacity_ah, dod, voltage_sag=1.0,
                               peukert_exponent=1.0, peukert_reference_hours=None, system_efficiency=1.0) -> dict:
    """Current = W ÷ (V × sag); lead-acid capacity Peukert-derated above the reference current;
    backup = capacity ÷ current × DoD × system efficiency."""
    avg_v = nominal_voltage * voltage_sag
    if avg_v <= 0 or load_watts <= 0:
        return dict(avg_v=avg_v, current_a=0.0, effective_ah=capacity_ah,
                    plain_h=float("inf"), dod_h=float("inf"), final_h=float("inf"))
    current = load_watts / avg_v
    eff_ah = capacity_ah
    if peukert_reference_hours and peukert_exponent != 1.0 and capacity_ah > 0:
        ref = capacity_ah / peukert_reference_hours
        if current > ref:
            eff_ah = capacity_ah / (current / ref) ** (peukert_exponent - 1)
    plain = eff_ah / current
    return dict(avg_v=avg_v, current_a=current, effective_ah=eff_ah, plain_h=plain,
                dod_h=plain * dod, final_h=plain * dod * system_efficiency)


# =====================================================================
# UI
# =====================================================================
st.title("☀️ Solar Savings Simulator")
st.caption("Grid-Tie vs Hybrid, row by row from the device's own telemetry. Battery energy comes straight "
           "from batterySoc — nothing about the battery is simulated.")

with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload the device workbook (.xlsx with sheets Device + Battery)", type=["xlsx"])
    st.caption("Device: " + ", ".join(f"`{c}`" for c in REQUIRED_DEVICE_FIELDS)
               + ".  \nBattery: " + ", ".join(f"`{c}`" for c in REQUIRED_BATTERY_FIELDS) + ".")
    tz_shift = st.number_input(
        "Timestamp shift from UTC (hours)", -12.0, 14.0, 5.5, 0.5,
        help="receivedAt is stored in UTC; 5.5 = IST. The findings panel checks the shift against daylight.")
    trim_comm = st.checkbox(
        "Trim the install/commissioning stretch at the start", value=True,
        help="Drops the leading rows before load first appears on the hybrid output.")

rows, findings = None, []
if uploaded:
    try:
        _dev, _bat = read_site_workbook(uploaded)
        rows, findings = adapt_site_data(_dev, _bat, tz_shift_hours=float(tz_shift),
                                         trim_commissioning=bool(trim_comm))
    except Exception as e:
        st.error(f"Could not read the workbook: {e}")
        st.stop()

_detected = detected_battery_preset(findings)
_standby = finding(findings, "hybrid_standby")
_battery_now = finding(findings, "battery_now")

with st.sidebar:
    st.header("2. Battery")
    _presets = list(BATTERY_PRESETS)
    battery_choice = st.selectbox(
        "Battery", _presets, index=_presets.index(_detected) if _detected else 0,
        help="Pre-selected from batteryChemistry and the measured pack voltage. The Ah rating is the fleet "
             "variant for that chemistry and voltage — confirm it with the installer.")
    battery = BATTERY_PRESETS[battery_choice]
    st.caption(f"Rated {battery['capacity_kwh']} kWh (1% batterySoc = {battery['capacity_kwh'] * 10:.0f} Wh) · "
               f"DoD {battery['dod']:.0%} · round-trip efficiency {battery['efficiency']:.0%}"
               + (" · detected from the data" if battery_choice == _detected else ""))
    hybrid_standby_w = st.number_input(
        "Hybrid standby draw (W)", min_value=0.0,
        value=round(float(_standby["standby_w"]), 1) if _standby else 0.0, step=5.0,
        help="The hybrid inverter's own consumption on grid-up rows (PV surplus first, then grid). "
             "Pre-filled from gridPower when the file allows it.")

    st.header("3. Grid Status")
    grid_v_min, grid_v_max = st.slider("Grid is up when gridVoltage is within (V)", 0, 300, (180, 260))

    st.header("4. Tariff (residential, FY 2026-27)")
    tariff_state = st.selectbox("State / DISCOM", list(TARIFF_PRESETS), index=0)
    _tp = TARIFF_PRESETS[tariff_state]
    st.caption(_tp["note"])
    _t1, _t2, _t3 = st.columns(3)
    wheeling = _t1.number_input("Wheeling (₹/unit)", min_value=0.0, value=float(_tp["wheeling"]), step=0.05)
    fppa = _t2.number_input("FPPA / FAC (₹/unit)", value=float(_tp["fppa"]), step=0.05,
                            help="Monthly fuel/power-purchase adjustment; changes every month — enter the current bill's value.")
    duty_pct = _t3.number_input("Electricity duty (%)", min_value=0.0, value=float(_tp["duty_pct"]), step=0.5)
    sanctioned_kw = st.number_input("Sanctioned load (kW)", min_value=0.5, value=2.0, step=0.5,
                                    help="Only for the fixed charge, which is the same with and without solar.")
    fixed_per_month = _tp["fixed_flat"] + _tp["fixed_per_kw"] * sanctioned_kw
    cycle_days = st.slider("Billing cycle length (days)", 1, 90, int(_tp["cycle_days"]))
    use_real_cycle_start = st.checkbox("Anchor billing cycles to the customer's real bill date", value=False)
    cycle_start_date = st.date_input("Real billing-cycle start date", value=None) if use_real_cycle_start else None
    _c1, _c2 = st.columns(2)
    solar_start = _c1.time_input("Solar hours start", _tp["solar_start"])
    solar_end = _c2.time_input("Solar hours end", _tp["solar_end"])
    rebate_on = st.checkbox("Apply solar-hour ToD rebate", value=bool(_tp["rebate_default"]),
                            help="MSEDCL: ₹0.80/unit in 09:00–17:00 for LT-domestic with a ToD/smart meter. UP: none.")
    rebate_rate = st.number_input("Solar-hour rebate (₹/unit)", min_value=0.0, value=float(_tp["rebate_rate"]), step=0.05)
    _c3, _c4 = st.columns(2)
    tod_start = _c3.time_input("TOD / peak window start", _tp["tod_start"])
    tod_end = _c4.time_input("TOD / peak window end", _tp["tod_end"])
    surcharge_on = st.checkbox("Apply peak surcharge in the TOD window", value=bool(_tp["surcharge_default"]),
                               help="Residential in both states: none for FY 2026-27.")
    tod_pct = st.number_input("Peak surcharge (% on the energy charge)", min_value=0.0,
                              value=float(_tp["surcharge_pct"]), step=1.0)
    tod_min_drop = st.number_input(
        "TOD Optimization: minimum sustained SOC drop (%)", min_value=1.0, max_value=20.0, value=2.0, step=1.0,
        help="Battery discharge in the TOD window (grid up) counts as TOD Optimization only when batterySoc keeps "
             "falling by at least this much without climbing back — so ±1–2% sensor wobble is ignored.")

    st.header("5. Net Metering")
    net_metering_enabled = st.radio(
        "Settlement mode", ["Net Metering — 1:1 with carry-forward bank", "Non-Net Metering — export sold"], index=0,
    ).startswith("Net Metering")
    yearend_buyback = st.number_input(
        "Year-end buyback of banked units (₹/unit)", min_value=0.0, value=float(_tp["yearend_buyback"]), step=0.10,
        disabled=not net_metering_enabled,
        help="Units still in the bank on 31 March are bought back at this rate and the bank resets. "
             "UP (RSPV 2019): ₹2/kWh. Maharashtra: MERC generic tariff (₹2.90 was FY 2021-22) — confirm.")
    export_rate = st.number_input("Non-net-metering export price (₹/unit)", min_value=0.0, value=3.0, step=0.10,
                                  disabled=net_metering_enabled,
                                  help="Gross metering / net billing feed-in price — confirm with the DISCOM.")
    tariff = dict(slabs=_tp["slabs"], wheeling=wheeling, fppa=fppa, duty_pct=duty_pct,
                  solar_start=solar_start, solar_end=solar_end, tod_start=tod_start, tod_end=tod_end,
                  tod_pct=tod_pct, surcharge_on=surcharge_on, rebate_rate=rebate_rate, rebate_on=rebate_on,
                  yearend_buyback=yearend_buyback)


def render_backup_estimator(battery: dict):
    with st.expander("⏱️ Backup time estimator (Watts → hours)"):
        la = battery["chemistry"] == 0
        st.caption("Current = W ÷ (V × sag); lead-acid capacity is Peukert-derated "
                   f"(k = {LEAD_ACID_PEUKERT_EXPONENT}, reference {LEAD_ACID_PEUKERT_REFERENCE_HOURS} h); "
                   "backup = capacity ÷ current × DoD × system efficiency.")
        b1, b2, b3, b4 = st.columns(4)
        load_w = b1.number_input("Backup load (W)", min_value=1.0, value=1500.0, step=50.0)
        volts = b2.number_input("Nominal voltage (V)", min_value=1.0, value=float(battery["nominal_v"]), step=1.0)
        ah = b3.number_input("Capacity (Ah)", min_value=1.0,
                             value=round(battery["capacity_kwh"] * 1000 / battery["nominal_v"], 0), step=1.0)
        dod = b4.number_input("DoD (%)", min_value=1.0, max_value=100.0, value=battery["dod"] * 100, step=1.0) / 100
        sys_eff = st.slider("System / inverter efficiency (%)", 50, 100, int(SYSTEM_EFFICIENCY * 100)) / 100
        r = estimate_backup_time_hours(
            load_w, volts, ah, dod, voltage_sag=LEAD_ACID_VOLTAGE_SAG if la else LITHIUM_VOLTAGE_SAG,
            peukert_exponent=LEAD_ACID_PEUKERT_EXPONENT if la else 1.0,
            peukert_reference_hours=LEAD_ACID_PEUKERT_REFERENCE_HOURS if la else None, system_efficiency=sys_eff)
        m1, m2, m3 = st.columns(3)
        m1.metric("Current drawn", f"{r['current_a']:.2f} A")
        m2.metric("Backup (after DoD)", f"{r['dod_h']:.2f} h")
        m3.metric("Backup (after DoD + losses)", f"{r['final_h']:.2f} h")
        st.caption(f"{load_w:.0f} W ÷ {r['avg_v']:.2f} V = {r['current_a']:.2f} A · effective capacity "
                   f"{r['effective_ah']:.1f} Ah · {r['plain_h']:.2f} h × {dod:.0%} × {sys_eff:.0%} = {r['final_h']:.2f} h")


if rows is None:
    st.info("⬅️ Upload the device workbook in the sidebar to run the calculation.")
    render_backup_estimator(battery)
    st.stop()

# ---------------- Battery right now ----------------
if _battery_now:
    st.header("🔋 Battery right now")
    _b1, _b2 = st.columns([1, 2])
    _b1.metric("Charged to (batterySoc)", f"{_battery_now['value']:.0f}%")
    _b2.markdown(_battery_now["evidence"])
    if _battery_now.get("flat"):
        _b2.caption("⚠️ batterySoc is still being calibrated and barely moves on discharge. It is used as-is, "
                    "so battery-to-load will read low until calibration is complete.")

# ---------------- Findings ----------------
_sev = {"critical": 0, "warning": 1, "info": 2}
_icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
_crit = sum(f["severity"] == "critical" for f in findings)
_warn = sum(f["severity"] == "warning" for f in findings)
st.header("🔎 Data findings")
if _crit:
    st.error(f"🔴 {_crit} unresolved finding(s) can materially change the results — treat the numbers as provisional.")
if _warn:
    st.warning(f"🟠 {_warn} further judgement call(s) — review below.")
if not _crit and not _warn:
    st.success("✅ No data issues needing a decision.")
for f in sorted(findings, key=lambda f: _sev[f["severity"]]):
    with st.expander(f"{_icon[f['severity']]} {f['title']}", expanded=f["severity"] == "critical"):
        st.markdown(f"**What the app did:** {f['decision']}")
        st.markdown(f"**Evidence in this file:** {f['evidence']}")
        if f.get("question"):
            st.markdown(f"**❓ Open question for the IoT / platform team:** {f['question']}")
if _detected and _detected != battery_choice:
    st.warning(f"⚠️ The data points to **{_detected}**, but **{battery_choice}** is selected.")
st.divider()

df = prepare_rows(rows, grid_v_min, grid_v_max, cycle_days, cycle_start_date)
if df.empty:
    st.error("No valid rows left after cleaning.")
    st.stop()
df["solar_hr"] = df["t_of_day"].apply(lambda t: in_window(t, solar_start, solar_end))

# ---------------- Load matching (sizing tool) ----------------
# Optionally replace the measured load with a flat two-level profile (solar hours / off-solar hours),
# keeping the real PV, grid timing and batterySoc.
_solar_h = float(df.loc[df["solar_hr"], "dt_hours"].sum())
_night_h = float(df.loc[~df["solar_hr"], "dt_hours"].sum())
_kw_solar = float(df.loc[df["solar_hr"], "Load_kWh"].sum() / _solar_h) if _solar_h > 0 else 0.0
_kw_night = float(df.loc[~df["solar_hr"], "Load_kWh"].sum() / _night_h) if _night_h > 0 else 0.0
with st.sidebar:
    st.header("6. Load Matching (sizing tool)")
    st.caption(f"This file's average load: {_kw_solar:.2f} kW in solar hours, {_kw_night:.2f} kW otherwise.")
    load_matching_on = st.checkbox("Replace the measured load with a flat profile", value=False)
    if load_matching_on:
        lm_solar = st.slider("Average load in solar hours (kW)", 0.0, round(max(5.0, _kw_solar * 3), 2), round(_kw_solar, 2), 0.01)
        lm_night = st.slider("Average load off solar hours (kW)", 0.0, round(max(5.0, _kw_night * 3), 2), round(_kw_night, 2), 0.01)
if load_matching_on:
    df["Load_kWh"] = np.where(df["solar_hr"], lm_solar * df["dt_hours"], lm_night * df["dt_hours"])
    st.info(f"⚙️ Load Matching is on: load is a flat {lm_solar:.2f} kW (solar hours) / {lm_night:.2f} kW (otherwise). "
            "PV, grid timing and batterySoc are the real readings.")

# ---------------- Data summary ----------------
st.header("🧪 Data summary")
_days = (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 86400
_n_out = int((df["Grid_Status"] == 0).sum())
s1, s2, s3, s4 = st.columns(4)
s1.metric("Load (kWh)", f"{df['Load_kWh'].sum():,.2f}")
s2.metric("PV (kWh)", f"{df['PV_kWh'].sum():,.2f}")
s3.metric("Rows / span", f"{len(df):,} / {_days:.1f} days")
s4.metric("Outage rows", f"{_n_out:,} ({df.loc[df['Grid_Status'] == 0, 'dt_hours'].sum():.1f} h)")
st.caption(
    f"{df['Timestamp'].min():%Y-%m-%d %H:%M} → {df['Timestamp'].max():%Y-%m-%d %H:%M} local time. "
    "`PV_kWh = PV_W / 1000 × Δt` (PV_W = pvVoltageMeasured × pvCurrentMeasured, or pvPower when pvConnected = 1); "
    "`Load_kWh = activePowerOutput / 1000 × Δt`; Δt = each row's real elapsed time "
    f"(gaps of {GAP_NORMAL_MAX_MIN:.0f}–{GAP_SHORT_MAX_MIN:.0f} min: {int((df['gap_flag'] == 'gap_interpolated').sum())} row(s); "
    f"over {GAP_SHORT_MAX_MIN:.0f} min, counted as zero energy: {int((df['gap_flag'] == 'gap_excluded').sum())} row(s)). "
    f"Grid up when {grid_v_min} ≤ gridVoltage ≤ {grid_v_max} V; billing cycle {cycle_days} days.")
with st.expander(f"🔌 Outage distribution ({_n_out:,} outage rows)"):
    render_outage_distribution(df)
with st.expander("Preview cleaned rows (first 20)"):
    st.dataframe(df[["Timestamp", "dt_hours", "gap_flag", "PV_kWh", "Load_kWh", "Grid_V", "Grid_Status",
                     "SOC_pct", "cycle_id"]].head(20), use_container_width=True)

# ---------------- Run ----------------
results = {
    "grid_tie": simulate_grid_tie(df, tariff),
    "hybrid": simulate_hybrid_live_soc(df, tariff, battery, standby_w=hybrid_standby_w, tod_min_drop_pct=tod_min_drop),
}
for _r in results.values():
    compute_financials(_r, tariff, net_metering_enabled, export_rate)

st.header("📊 Grid-Tie vs Hybrid")
st.dataframe(pd.DataFrame(
    {MODE_LABELS[m]: [round(results[m][k], 2) for k in METRIC_ORDER] for m in MODES},
    index=[METRIC_LABELS[k] for k in METRIC_ORDER]), use_container_width=True)
st.caption("Household demand is the same for both. Grid-Tie switches off during an outage, so its unserved load "
           "is the whole outage-time load; the hybrid serves what PV and the battery (per batterySoc) cover.")
_days_span = max((df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 86400, 0.0)
st.caption(f"Fixed charge ₹{fixed_per_month:,.0f}/month (≈ ₹{fixed_per_month * _days_span / 30:,.2f} for this "
           f"{_days_span:.1f}-day window) is the same with or without solar, so it is left out of both bills and "
           "never changes a saving. Every per-unit price above includes wheeling, FPPA and electricity duty.")

if net_metering_enabled:
    st.subheader("🏦 Net-metering bank (carry-forward)")
    st.caption("Each cycle: export first offsets that cycle's import 1:1; import still left is offset from the bank; "
               "any extra export goes into the bank and carries forward to every later cycle. On 31 March the bank "
               f"is bought back at ₹{yearend_buyback:.2f}/unit and resets.")
    for m in MODES:
        t = results[m]["nm_cycles"]
        st.markdown(f"**{MODE_LABELS[m]}** — bank at the end of the data: {results[m]['nm_bank_end_kwh']:.2f} kWh "
                    f"(worth ₹{results[m]['nm_bank_end_value']:.2f} at year-end buyback; not yet paid)")
        show = t.copy()
        show["start"] = pd.to_datetime(show["start"]).dt.strftime("%Y-%m-%d")
        show["end"] = pd.to_datetime(show["end"]).dt.strftime("%Y-%m-%d")
        st.dataframe(show.round(2).rename(columns={
            "cycle": "Cycle", "start": "From", "end": "To", "import_kwh": "Import (kWh)", "export_kwh": "Export (kWh)",
            "fy_buyback_kwh": "Bought back at FY end (kWh)", "bank_in_kwh": "Bank at start (kWh)",
            "offset_same_cycle_kwh": "Offset same cycle (kWh)", "used_from_bank_kwh": "Used from bank (kWh)",
            "net_billed_kwh": "Net billed (kWh)", "bank_out_kwh": "Bank carried forward (kWh)", "bill": "Bill (₹)"}),
            use_container_width=True, hide_index=True)

# ---------------- Daily breakdown ----------------
st.header("📅 Today Earning — daily breakdown")
st.caption("Solar Savings + TOD Optimization + Outage Savings per calendar day, priced at the running billing-cycle "
           "slab position (never reset at midnight). Net-metering settlement is not split across days.")
_te_label = st.selectbox("Mode", [MODE_LABELS[m] for m in MODES], index=1)
_te_mode = next(m for m in MODES if MODE_LABELS[m] == _te_label)
daily = compute_daily_earning(results[_te_mode], tariff["slabs"])
st.bar_chart(daily.set_index("date")[["solar_savings", "tod_optimization", "outage_savings"]])
_show = daily.copy()
_show["date"] = _show["date"].dt.strftime("%Y-%m-%d")
st.dataframe(_show[["date", "solar_savings", "tod_optimization", "outage_savings", "today_earning",
                    "cycle_units_at_day_start", "cycle_units_at_day_end", "slab_rate_at_day_start"]].round(2).rename(columns={
    "date": "Date", "solar_savings": "Solar Savings (₹)", "tod_optimization": "TOD Optimization (₹)",
    "outage_savings": "Outage Savings (₹)", "today_earning": "Today Earning (₹)",
    "cycle_units_at_day_start": "Cycle units at day start (kWh)", "cycle_units_at_day_end": "Cycle units at day end (kWh)",
    "slab_rate_at_day_start": "Slab rate at day start (₹/kWh)"}), use_container_width=True)

# ---------------- Formula & arithmetic audit ----------------
st.header("🔍 Formula & arithmetic audit")
st.caption("Each metric: how it is computed for each mode, the formula, and the numbers from this file.")


def f2(x):
    return f"{x:,.2f}"


def audit(key: str, explanation: str, formula: str, arithmetic):
    with st.expander(METRIC_LABELS[key]):
        st.markdown(explanation)
        st.latex(formula)
        for m in MODES:
            st.markdown(f"**{MODE_LABELS[m]}**: {arithmetic(m, results[m])}")


audit("pv_available",
      "- **Grid-Tie**: PV during an outage is lost (**opportunity loss**) — the inverter is off.\n"
      "- **Hybrid**: PV during an outage is lost only when neither the load nor the battery (per batterySoc) "
      "took it (**notional loss**).",
      r"PV_{available} = PV_{total} - Loss",
      lambda m, r: f"{f2(r['total_pv'])} − {f2(r['opportunity_loss'] if m == 'grid_tie' else r['notional_loss'])} "
                   f"= **{f2(r['pv_available'])} kWh**")

audit("total_demand",
      "What the house drew, from `activePowerOutput` (real active power). The same for both modes.",
      r"Demand = \sum_i \frac{activePowerOutput_i}{1000}\times\Delta t_i",
      lambda m, r: f"**{f2(r['total_demand'])} kWh**")

audit("load_served",
      "Demand minus the load nobody could serve (outage rows only).",
      r"Load_{served} = Demand - Unserved",
      lambda m, r: f"{f2(r['total_demand'])} − {f2(r['unserved'])} = **{f2(r['load_served'])} kWh**")

audit("pv_to_load",
      "- **Grid-Tie**: grid-up rows only.\n- **Hybrid**: every row — the hybrid stays on during outages.",
      r"PV_{to\ load} = \sum_i \min(PV_i,\ Load_i)",
      lambda m, r: f"**{f2(r['pv_to_load'])} kWh**")

audit("pv_export",
      "Grid-up rows only (nothing can be exported into a dead grid, so outages affect neither mode). For the hybrid, "
      "PV surplus first covers the hybrid's own standby draw and any battery charging (a batterySoc rise); only the "
      "rest is exported.",
      r"PV_{export} = \sum_{Grid\ up} \max(PV_i - Load_i - Standby_{PV,i} - PV_{to\ battery,i},\ 0)",
      lambda m, r: f"**{f2(r['pv_export'])} kWh**" + (
          f" (standby taken from PV surplus: {f2(r['hybrid_standby'] - r['hybrid_standby_grid'])} kWh; "
          f"PV into battery: {f2(r['pv_to_battery'])} kWh)" if m == "hybrid" else ""))

audit("batt_to_load",
      "- **Grid-Tie**: no battery.\n"
      "- **Hybrid**: every drop in batterySoc is energy the battery gave out: ΔSOC% × rated capacity × "
      "discharge efficiency η_d, capped at the load PV could not cover in that row. η_d = η_c = √(round-trip "
      "efficiency): lead-acid √0.80 = 0.894, lithium √0.95 = 0.975.",
      r"Batt_{to\ load} = \sum_{\Delta SOC_i<0} \min\Big(\tfrac{-\Delta SOC_i}{100}\,C_{rated}\,\eta_d,\ Load_i - PV_i\Big)",
      lambda m, r: "0 (no battery)" if m == "grid_tie" else
      f"**{f2(r['batt_to_load'])} kWh** (last batterySoc {r['final_soc_pct']:.0f}% = {f2(r['final_soc'])} of "
      f"{f2(r['capacity_kwh'])} kWh rated)")

audit("grid_to_battery",
      "- **Grid-Tie**: no battery.\n"
      "- **Hybrid**: every rise in batterySoc is charging. PV surplus in that row is credited first; whatever the "
      "surplus can't explain came from the grid (÷ charge efficiency η_c) and is billed.",
      r"Grid_{to\ battery} = \sum_{\Delta SOC_i>0} \frac{\max\big(\tfrac{\Delta SOC_i}{100}C_{rated} - (PV_i-Load_i)^{+}\eta_c,\ 0\big)}{\eta_c}",
      lambda m, r: "0 (no battery)" if m == "grid_tie" else f"**{f2(r['grid_to_battery'])} kWh**")

audit("grid_import",
      "Grid-up rows only: load not covered by PV or the battery, plus grid charging of the battery, plus (hybrid) "
      "the hybrid's own standby draw not covered by PV surplus.",
      r"Grid_{import} = \sum_{Grid\ up}\big(Load_i - PV_{to\ load,i} - Batt_{to\ load,i}\big) + Grid_{to\ battery} + Standby_{grid}",
      lambda m, r: f"**{f2(r['grid_import'])} kWh**" + (
          f" (includes standby; total standby {f2(r['hybrid_standby'])} kWh)" if m == "hybrid" else ""))

audit("unserved",
      "- **Grid-Tie**: all load during outages.\n- **Hybrid**: outage load left after PV and the battery's observed discharge.",
      r"Unserved = \sum_{Grid\ down} \max(Load_i - PV_{to\ load,i} - Batt_{to\ load,i},\ 0)",
      lambda m, r: f"**{f2(r['unserved'])} kWh**")

audit("baseline_bill",
      "What the house would pay with no solar and no battery: every grid-up row's load priced at the all-in rate for "
      "that moment — slab energy charge for the cycle's units so far (× peak surcharge if on) + wheeling + FPPA − "
      "solar-hour rebate, then × (1 + electricity duty). Outage rows are excluded (a plain house is dark and billed "
      "nothing then). Fixed charges are left out because they are the same with solar. Same for both modes.",
      r"rate = \big(E_{slab}(cum)\,(1+s_{peak}) + W + FPPA - R_{solar}\big)(1 + ED)\qquad Expected = \sum_{Grid\ up} Load_i \times rate_i",
      lambda m, r: f"**₹{f2(r['baseline_bill'])}**")

audit("actual_bill",
      "- **Non-net-metering**: the per-row bill of every kWh really imported (load, battery charging, standby).\n"
      "- **Net metering**: per cycle, export offsets import 1:1, then the bank (surplus carried forward from earlier "
      "cycles) offsets what is left; the remaining net import is billed from 0 through the slabs + wheeling + FPPA, "
      "× (1 + duty). No ToD survives netting (a net meter reports one number). Surplus export goes into the bank.",
      r"Actual = \begin{cases}\sum_i Import_i \times rate_i & \text{non-net}\\ \sum_{cycles} Bill\big(\max(Import - Export - Bank,\ 0)\big) & \text{net}\end{cases}",
      lambda m, r: f"**₹{f2(r['actual_bill'])}** for {f2(r['grid_import'])} kWh imported"
                   + (" (net-metering settled)" if r["net_metering"] else ""))

audit("solar_savings",
      "The bill reduction, minus the part booked as TOD Optimization (so nothing is counted twice).",
      r"Solar_{savings} = (Expected - Actual) - TOD_{opt}",
      lambda m, r: f"(₹{f2(r['baseline_bill'])} − ₹{f2(r['actual_bill'])}) − ₹{f2(r['tod_optimization'])} "
                   f"= **₹{f2(r['solar_savings'])}**")

audit("tod_optimization",
      "Battery energy delivered to load while the grid is **up**, inside the TOD window, during a **sustained** "
      "discharge: batterySoc keeps falling by at least the sidebar threshold (default 2%) without climbing back, so "
      "±1–2% sensor wobble does not count. This is what a Smart Hybrid does; a Dumb Hybrid never discharges with the "
      "grid up, so it scores ₹0. Valued at the rate avoided at that moment; grid-charged energy only gets the "
      "difference from what it cost to charge.",
      r"TOD_{opt} = \sum_{TOD,\ Grid\ up}\big[E_{PV}\,r_{now} + E_{grid}\,(r_{now} - r_{charge})\big]",
      lambda m, r: "₹0 (no battery)" if m == "grid_tie" else
      f"{f2(r['tod_kwh'])} kWh of sustained grid-up discharge in the window ({r['sustained_rows']} rows in sustained "
      f"discharge episodes overall) = **₹{f2(r['tod_optimization'])}**")

audit("outage_savings",
      "Energy the system actually served during outages (PV + battery), valued at the rate the house would have "
      "paid at that time and cycle position. Grid-Tie is off during outages, so it is ₹0.",
      r"Outage_{savings} = \sum_{Grid\ down}(PV_{to\ load,i} + Batt_{to\ load,i})\times rate(t_i,\ cum_i)",
      lambda m, r: f"{f2(r['outage_served'])} kWh served → **₹{f2(r['outage_savings'])}**")

audit("solar_earning",
      "Reported separately (not in Net Savings). **Non-net-metering**: all exported kWh × the export price. "
      "**Net metering**: export is already used inside the Actual Bill; the only cash is the 31-March buyback of "
      "units still in the bank.",
      r"Solar\ Earning = \begin{cases}Export \times ExportPrice & \text{non-net}\\ Bank_{31\,Mar} \times Buyback & \text{net}\end{cases}",
      lambda m, r: f"{f2(r['export_kwh'])} kWh × ₹{export_rate:.2f} = **₹{f2(r['solar_earning'])}**"
      if not r["net_metering"] else
      f"{f2(r['nm_buyback_kwh'])} kWh bought back at FY end = **₹{f2(r['solar_earning'])}**; "
      f"{f2(r['settlement_kwh'])} kWh offset against import; {f2(r['nm_bank_end_kwh'])} kWh still in the bank")

audit("net_savings",
      "Solar Savings + Outage Savings + TOD Optimization.",
      r"Net = Solar_{savings} + Outage_{savings} + TOD_{opt}",
      lambda m, r: f"₹{f2(r['solar_savings'])} + ₹{f2(r['outage_savings'])} + ₹{f2(r['tod_optimization'])} "
                   f"= **₹{f2(r['net_savings'])}**")

st.divider()
render_backup_estimator(battery)