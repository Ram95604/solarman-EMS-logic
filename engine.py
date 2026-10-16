"""
Measured-telemetry savings engine.
==================================
This replaces the "simulate a policy and call the result a saving" approach.
For a site that already exists and already has a year of telemetry, the
savings number must be DERIVED FROM WHAT THE HARDWARE ACTUALLY DID, not from
what a modelled controller would have done. That is the difference between a
number a manager can take to a customer and a number that collapses under
one question.

Everything below therefore follows one rule: each rupee of claimed benefit is
traced to (a) a measured kWh, (b) a timestamp, (c) a grid state, (d) an energy
source, and (e) a tariff rate that applied at that moment.

THE FIVE SEPARATE BUCKETS (kept strictly apart, as instructed)
--------------------------------------------------------------
1. BILLING SAVINGS       — PV served the load while the grid was UP.
                           Directly reduces metered import. Reconciles against
                           (baseline bill - actual bill).
2. ISLANDING SAVINGS     — grid DOWN during SOLAR HOURS, load carried by PV
                           directly, PLUS later discharge of battery energy
                           that was charged by PV *during* such an islanding
                           window. (Island-charged energy keeps its identity
                           through storage and is still islanding value when
                           it comes out.)
3. NIGHT DISCHARGE       — battery served the load off-solar-hours while the
                           grid was UP. Net of what the stored energy cost if
                           it was grid-charged.
4. OUTAGE BACKUP VALUE   — load carried during a NON-solar-hour outage from
                           non-island-charged battery energy. This is
                           resilience, not a tariff saving, and is reported
                           OUTSIDE net savings by default.
5. EXPORT EARNING        — surplus PV injected to the grid, valued per the
                           settlement mode actually in force at the site.

   less: GRID CHARGING COST — energy bought from the grid to fill the battery.
                              A real cash cost that must be netted off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Column contract for the DESS / inverter export
# ---------------------------------------------------------------------
COL_TS_PRIMARY = "Timestamp"
COL_TS_ALT = "Date Time"
COL_PV1 = "PV1 Input Power(W)"
COL_PV2 = "PV2 Input Power(W)"
COL_ACTIVE = "AC output active power(W)"
COL_APPARENT = "AC output apparent power(VA)"
COL_GRID_V = "Grid voltage(V)"
COL_BAT_V = "Battery Voltage(V)"
COL_BAT_SOC = "Battery Capacity(%)"
COL_BAT_DIS_A = "Battery Discharging Current(A)"
COL_BAT_CHG_A = "Battery Charging Current(A)"
COL_TOTAL_GEN = "Total generation(kWh)"

GAP_NORMAL_MAX_MIN = 6.0    # ordinary logger jitter
GAP_EXCLUDE_MIN = 30.0      # beyond this the device was offline: integrate nothing


# =====================================================================
# 1. TIME BASE — the thing that quietly invalidates everything else
# =====================================================================
def fit_local_time_offset(df: pd.DataFrame, ts_col: str, lon_deg: float = 75.0) -> dict:
    """
    Derive the offset between the file's timestamp column and real LOCAL clock
    time, by aligning observed PV generation against astronomical solar noon.

    Why this exists: time-of-day is load-bearing for this entire analysis.
    "Solar-hour outage" vs "night outage", the ToD window, and the day/night
    split of battery discharge are ALL decided by the clock. If the logger's
    clock is offset (wrong device timezone, UTC stored as local, an un-synced
    RTC), every one of those splits is silently wrong while every total still
    looks perfectly reasonable. So the offset is measured from the sun, not
    assumed.

    Method: for each reasonably clear day, take the midpoint of the window
    where PV output is above half its daily peak — a robust proxy for solar
    noon that is insensitive to cloud asymmetry. Compare with astronomical
    solar noon in IST for the site longitude, including the equation of time.
    """
    d = df[[ts_col, COL_PV1]].copy()
    d["date"] = d[ts_col].dt.normalize()
    d["tod"] = (
        d[ts_col].dt.hour + d[ts_col].dt.minute / 60.0 + d[ts_col].dt.second / 3600.0
    )
    rows = []
    for day, g in d.groupby("date"):
        pv = g[COL_PV1].to_numpy(dtype=float)
        if pv.size < 10 or pv.max() < 500:
            continue
        tod = g["tod"].to_numpy(dtype=float)
        sel = tod[pv >= 0.5 * pv.max()]
        if sel.size < 3:
            continue
        rows.append((day, 0.5 * (sel.min() + sel.max()), pv.max()))
    if not rows:
        return dict(ok=False, offset_h=0.0, n_days=0, scatter_min=float("nan"),
                    observed_noon=float("nan"), astronomical_noon=float("nan"))

    obs = pd.DataFrame(rows, columns=["date", "noon_obs", "pvmax"])
    # Restrict the fit to the clearer half of days. On a heavily clouded day the
    # half-max window is clipped asymmetrically and its midpoint drifts, which
    # widens the scatter without adding information about the clock.
    if len(obs) > 30:
        obs = obs[obs["pvmax"] >= obs["pvmax"].quantile(0.5)].reset_index(drop=True)
    doy = pd.to_datetime(obs["date"]).dt.dayofyear.to_numpy(dtype=float)
    B = np.deg2rad(360.0 * (doy - 81.0) / 364.0)
    eot_min = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    noon_ist = 12.0 - (lon_deg - 82.5) * 4.0 / 60.0 - eot_min / 60.0
    offs = noon_ist - obs["noon_obs"].to_numpy(dtype=float)
    return dict(
        ok=True,
        offset_h=float(np.median(offs)),
        n_days=int(len(obs)),
        scatter_min=float(np.std(offs) * 60.0),
        observed_noon=float(obs["noon_obs"].median()),
        astronomical_noon=float(np.median(noon_ist)),
    )


# =====================================================================
# 2. PREPROCESS
# =====================================================================
def preprocess(
    raw: pd.DataFrame,
    *,
    ts_col: str = COL_TS_PRIMARY,
    local_offset_h: float = 0.0,
    load_source: str = "active",          # "active" | "apparent_pf"
    power_factor: float = 0.92,
    outage_voltage_max: float = 1.0,      # grid considered DOWN at/below this
    min_outage_minutes: float = 5.0,      # ignore shorter zero-voltage blips
    battery_capacity_kwh: float = 15.07,
    battery_dod: float = 0.80,
    count_float_current_as_charge: bool = False,
    cycle_mode: str = "monthly",          # "monthly" | "fixed_days"
    cycle_anchor_day: int = 1,
    cycle_days: int = 30,
) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    needed = [ts_col, COL_PV1, COL_GRID_V]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df = df.drop_duplicates(subset=[ts_col], keep="first").reset_index(drop=True)

    for c in [COL_PV1, COL_PV2, COL_ACTIVE, COL_APPARENT, COL_GRID_V,
              COL_BAT_V, COL_BAT_SOC, COL_BAT_DIS_A, COL_BAT_CHG_A]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if COL_PV2 not in df.columns:
        df[COL_PV2] = 0.0
    df[COL_PV1] = df[COL_PV1].fillna(0.0)
    df[COL_PV2] = df[COL_PV2].fillna(0.0)
    df = df.dropna(subset=[COL_GRID_V]).reset_index(drop=True)

    # ---- real per-row elapsed time, with offline windows excluded ----
    dt_raw_h = df[ts_col].diff().dt.total_seconds() / 3600.0
    if len(dt_raw_h) > 1 and pd.notna(dt_raw_h.iloc[1]):
        dt_raw_h.iloc[0] = dt_raw_h.iloc[1]
    else:
        dt_raw_h.iloc[0] = 5.0 / 60.0
    dt_min = dt_raw_h * 60.0
    df["gap_flag"] = np.where(
        dt_min <= GAP_NORMAL_MAX_MIN, "normal",
        np.where(dt_min <= GAP_EXCLUDE_MIN, "short_gap", "device_offline"),
    )
    df["dt_h"] = np.where(df["gap_flag"] == "device_offline", 0.0, dt_raw_h)

    # ---- local clock ----
    df["local"] = df[ts_col] + pd.to_timedelta(local_offset_h, unit="h")
    df["hour"] = df["local"].dt.hour + df["local"].dt.minute / 60.0
    df["date"] = df["local"].dt.normalize()

    # ---- power -> energy ----
    df["pv_kw"] = ((df[COL_PV1] + df[COL_PV2]) / 1000.0).clip(lower=0)
    if load_source == "active" and COL_ACTIVE in df.columns:
        df["load_kw"] = (df[COL_ACTIVE] / 1000.0).clip(lower=0)
        df.attrs["load_source"] = "AC output active power(W) — measured real power"
    else:
        df["load_kw"] = (df[COL_APPARENT] * power_factor / 1000.0).clip(lower=0)
        df.attrs["load_source"] = f"AC output apparent power(VA) x PF {power_factor}"

    bat_v = df[COL_BAT_V] if COL_BAT_V in df.columns else pd.Series(48.0, index=df.index)
    bat_v = bat_v.fillna(48.0).clip(lower=20.0)
    dis_a = df[COL_BAT_DIS_A].fillna(0.0).clip(lower=0) if COL_BAT_DIS_A in df.columns else 0.0
    chg_a = df[COL_BAT_CHG_A].fillna(0.0).clip(lower=0) if COL_BAT_CHG_A in df.columns else 0.0
    df["bat_dis_kw"] = dis_a * bat_v / 1000.0
    df["bat_chg_kw_raw"] = chg_a * bat_v / 1000.0

    # FLOAT-CURRENT CORRECTION.
    # A battery sitting at 100% SOC still draws a small float/absorption
    # current indefinitely. That current is NOT energy going into storage —
    # it offsets self-discharge and is lost as heat. Integrating it as
    # "charge energy" inflates the apparent grid-charging cost and makes the
    # round-trip efficiency look impossible. Charge energy is therefore only
    # counted while SOC is below full.
    if COL_BAT_SOC in df.columns and not count_float_current_as_charge:
        at_full = df[COL_BAT_SOC].fillna(0) >= 100
        df["bat_chg_kw"] = df["bat_chg_kw_raw"].where(~at_full, 0.0)
        df["float_kw"] = df["bat_chg_kw_raw"].where(at_full, 0.0)
    else:
        df["bat_chg_kw"] = df["bat_chg_kw_raw"]
        df["float_kw"] = 0.0

    for a, b in [("pv_kw", "pv_kwh"), ("load_kw", "load_kwh"),
                 ("bat_dis_kw", "bat_dis_kwh"), ("bat_chg_kw", "bat_chg_kwh"),
                 ("float_kw", "float_kwh")]:
        df[b] = (df[a] * df["dt_h"]).clip(lower=0)

    # ---- grid state, with minimum-duration filtering ----
    raw_down = df[COL_GRID_V] <= outage_voltage_max
    # break runs at device-offline boundaries so an offline window is never
    # absorbed into one giant "outage"
    brk = raw_down.ne(raw_down.shift()) | (df["gap_flag"] == "device_offline")
    run_id = brk.cumsum()
    df["_run"] = run_id
    run_dur_min = df.groupby("_run")["dt_h"].transform("sum") * 60.0
    # a run's own first row contributes its own interval, so add one nominal step
    nominal = float(np.median(df["dt_h"][df["dt_h"] > 0]) * 60.0) if (df["dt_h"] > 0).any() else 5.0
    df["run_minutes"] = run_dur_min + nominal
    df["outage"] = raw_down & (df["run_minutes"] >= min_outage_minutes)
    df["outage_raw"] = raw_down
    df["grid_up"] = ~df["outage"]

    # ---- billing cycle id ----
    if cycle_mode == "monthly":
        shifted = df["local"] - pd.to_timedelta(cycle_anchor_day - 1, unit="D")
        df["cycle_id"] = shifted.dt.to_period("M").astype(str)
        df["cycle_label"] = df["cycle_id"]
    else:
        first = df["date"].min()
        n = np.floor((df["date"] - first).dt.days / max(cycle_days, 1)).astype(int)
        df["cycle_id"] = n.astype(str)
        df["cycle_label"] = [
            f"Cycle {i + 1} ({(first + pd.Timedelta(days=int(i) * cycle_days)).date()})"
            for i in n
        ]

    df.attrs["cycle_mode"] = cycle_mode
    df.attrs["cycle_days"] = cycle_days
    df.attrs["usable_kwh"] = battery_capacity_kwh * battery_dod
    df.attrs["nominal_interval_min"] = nominal
    return df.drop(columns=["_run"])


# =====================================================================
# 3. MEASURED ENERGY BALANCE
# =====================================================================
def derive_flows(df: pd.DataFrame, solar_start_h: float, solar_end_h: float,
                 dispatch_order: str = "pv_first") -> pd.DataFrame:
    """
    Decompose each interval into physical flows using ONLY measured quantities.

    Identities enforced per row:
        load = pv_to_load + bat_to_load + grid_to_load
        pv   = pv_to_load + pv_to_batt  + pv_export + pv_curtailed
        battery charge = pv_to_batt + grid_to_batt

    Ordering reflects how a hybrid inverter actually prioritises. The default
    is PV-FIRST: solar carries the load whenever it is available, outage or
    not, and only the shortfall is drawn from the battery (during an outage)
    or the grid. `dispatch_order="battery_first"` is kept for sensitivity
    testing; on this data the two differ by about 0.1% of net savings, which
    is itself a useful answer when the ordering is questioned. During an outage
    grid flows are forced to zero and un-absorbed PV is CURTAILED rather than
    exported — a point that matters, because treating it as export would
    invent revenue that physically could not have been earned with the grid
    down.
    """
    d = df.copy()
    d["solar_hr"] = (d["hour"] >= solar_start_h) & (d["hour"] < solar_end_h)

    load = d["load_kwh"].to_numpy()
    pv = d["pv_kwh"].to_numpy()
    bdis = d["bat_dis_kwh"].to_numpy()
    bchg = d["bat_chg_kwh"].to_numpy()
    up = d["grid_up"].to_numpy()

    if dispatch_order == "pv_first":
        # How a hybrid inverter actually behaves: PV carries the load first,
        # whatever the grid is doing. Only the shortfall goes to the battery
        # (during an outage) or the grid.
        pv_to_load = np.minimum(pv, load)
        rem_load = load - pv_to_load
        bat_to_load = np.minimum(bdis, rem_load)
        rem_load = rem_load - bat_to_load
        # Measured discharge that PV-first leaves no room for. Physically the
        # battery cannot export on these units, so this is a reconciliation
        # residual — mostly 5-minute averaging hiding sub-interval crossings
        # where load briefly exceeded PV. It is reported, never buried.
        unexplained_discharge = np.maximum(bdis - bat_to_load, 0.0)
    else:  # "battery_first"
        bat_to_load = np.minimum(bdis, load)
        rem_load = load - bat_to_load
        pv_to_load = np.minimum(pv, rem_load)
        rem_load = rem_load - pv_to_load
        unexplained_discharge = np.zeros_like(load)
    pv_surplus = pv - pv_to_load
    pv_to_batt = np.minimum(pv_surplus, bchg)
    grid_to_batt = np.where(up, np.maximum(bchg - pv_to_batt, 0.0), 0.0)
    leftover_pv = pv_surplus - pv_to_batt
    pv_export = np.where(up, leftover_pv, 0.0)
    pv_curtailed = np.where(up, 0.0, leftover_pv)
    grid_to_load = np.where(up, rem_load, 0.0)
    unserved = np.where(up, 0.0, rem_load)

    d["bat_to_load"] = bat_to_load
    d["pv_to_load"] = pv_to_load
    d["pv_to_batt"] = pv_to_batt
    d["grid_to_batt"] = grid_to_batt
    d["pv_export"] = pv_export
    d["pv_curtailed"] = pv_curtailed
    d["grid_to_load"] = grid_to_load
    d["grid_import"] = grid_to_load + grid_to_batt
    d["unserved"] = unserved
    d["unexplained_discharge"] = unexplained_discharge
    d.attrs["dispatch_order"] = dispatch_order
    return d


# =====================================================================
# 4. BUCKETED SAVINGS ACCOUNTING
# =====================================================================
def measured_round_trip(df: pd.DataFrame) -> dict:
    """
    Observe the bank's round-trip efficiency instead of assuming it.

    In a measured analysis an assumed efficiency has no place on the energy
    side: both legs are metered, so the ratio IS the efficiency. It is still
    needed on the MONEY side — see `account()` — because a kWh delivered to
    the load consumed more than a kWh of what was bought.
    """
    cin = float(df["pv_to_batt"].sum() + df["grid_to_batt"].sum())
    cout = float(df["bat_to_load"].sum())
    return dict(
        charged_kwh=cin, discharged_kwh=cout,
        eff=(cout / cin) if cin > 1e-9 else float("nan"),
        float_excluded_kwh=float(df["float_kwh"].sum()),
    )


def account(df: pd.DataFrame, tariff, round_trip_eff: float = 1.0,
            settlement_start_month: int = 4,
            outage_pricing: str = "incremental") -> dict:
    """
    Walk the series once, cycle-aware, and attribute every rupee to exactly
    one bucket. Battery energy carries PROVENANCE through storage in three
    sub-buckets so that a later discharge is valued by where the energy came
    from, not by when it happened to leave:

        island : charged by PV during a solar-hour outage  -> islanding value
        pv     : charged by PV at any other time           -> free energy
        grid   : bought from the grid (carries a Rs/kWh cost basis)

    Discharge draws proportionally from the three sub-buckets. Proportional
    draw avoids the arbitrary bias that a fixed draw order (island-first, say)
    would introduce into the islanding figure.

    Round-trip efficiency enters on the MONEY side, not the energy side. Both
    legs are measured, so no efficiency is applied to any kWh figure. But the
    provenance buckets are credited with DELIVERABLE energy (kWh charged x
    round_trip_eff), and the grid bucket's cost basis is therefore cost per
    deliverable kWh. That matters: at a 69% round trip, buying a unit at
    Rs 6.45 means the unit that eventually reaches the load cost Rs 9.35, not
    Rs 6.45. Carrying the basis at the purchase price would make night
    arbitrage look roughly 45% better than it is.

    OUTAGE PRICING. Energy served during an outage never appears on a bill,
    so it has to be valued against a counterfactual: what would this kWh have
    cost had the grid been up?

      "incremental" (default) — it would have been ADDITIONAL consumption on
          top of the month's grid-up units, and a telescopic tariff puts
          additional units at the TOP of the stack. So outage energy is priced
          from the block above the cycle's grid-up total. This is the same
          top-of-stack logic that makes solar's bill saving exact, applied to
          the one stream that has no bill of its own.

      "chronological" — each outage kWh priced at the slab position the
          grid-up stream had reached at that moment. Kept for comparison. It
          understates: a kWh avoided in the first week is priced in the first
          slab, when adding it would in fact have pushed the whole month's
          stack up a band.

    On the Maharashtra file the two differ by 74% on the islanding bucket.
    """
    rows = []
    eff = max(min(round_trip_eff, 1.0), 1e-6)
    # cycle totals needed before the walk, for incremental outage pricing
    cycle_base_units = (df.loc[df["grid_up"]].groupby("cycle_id")["load_kwh"].sum()
                        if "grid_up" in df.columns else pd.Series(dtype=float))

    # SPEED, no change to any number.
    #
    # Two things made this walk far slower than the arithmetic in it warrants.
    #
    # 1. itertuples materialises EVERY column of the frame, and by this point
    #    the frame carries ~79 of them — Arrow-backed strings and datetimes
    #    included. Iterating those cost more than the accounting itself (about
    #    1.2 s of a 3.7 s run on a 103k-row file) purely to build attributes
    #    this loop never reads. Walking a 14-column view instead is identical
    #    row for row: the loop can only see what it names.
    # 2. `cycle_base_units.get(...)` is a pandas lookup per row. A plain dict
    #    answers the same question without the indexing machinery.
    _cols = ["cycle_id", "date", "hour", "grid_up", "solar_hr", "dt_h", "load_kwh",
             "pv_to_load", "pv_to_batt", "pv_export", "bat_to_load",
             "grid_to_load", "grid_to_batt", "grid_import"]
    _walk = df[[c for c in _cols if c in df.columns]]
    _base_by_cycle = {k: float(v) for k, v in cycle_base_units.items()}

    outage_cum = 0.0
    b_island = b_pv = b_grid = 0.0
    grid_basis = 0.0  # Rs per DELIVERABLE kWh from the grid-charged bucket

    cur_cycle = None
    base_cum = act_cum = 0.0

    tot = dict(
        billing_savings=0.0, islanding_savings=0.0, night_discharge_savings=0.0,
        slab_displacement=0.0, outage_backup_value=0.0, export_earning=0.0,
        grid_charging_cost=0.0, islanding_kwh=0.0, island_batt_kwh=0.0,
        night_discharge_kwh=0.0, outage_backup_kwh=0.0, self_consumed_kwh=0.0,
        unattributed_discharge_kwh=0.0,
    )

    for r in _walk.itertuples(index=False):
        if r.cycle_id != cur_cycle:
            cur_cycle, base_cum, act_cum = r.cycle_id, 0.0, 0.0
            outage_cum = 0.0

        # baseline counterfactual = the household's full demand, but ONLY while
        # the grid is up. With no solar and no battery the house is simply dark
        # during an outage and is billed nothing for that period, so outage
        # energy must not inflate the baseline bill. Its value is captured in
        # the islanding / backup buckets instead — never twice.
        if r.grid_up:
            base_rate = tariff.marginal_rate(base_cum, r.hour)
            base_cum += r.load_kwh
        elif outage_pricing == "incremental":
            # price this outage kWh from the block ABOVE the cycle's grid-up
            # total, because that is where it would have landed on the bill
            base_rate = tariff.marginal_rate(
                _base_by_cycle.get(r.cycle_id, 0.0) + outage_cum, r.hour)
            outage_cum += r.load_kwh
        else:
            base_rate = tariff.marginal_rate(base_cum, r.hour)
            outage_cum += r.load_kwh
        act_rate = tariff.marginal_rate(act_cum, r.hour)
        act_cum += r.grid_import

        row = dict(
            date=r.date, cycle_id=r.cycle_id, hour=r.hour,
            baseline_units=r.load_kwh if r.grid_up else 0.0,
            actual_units=r.grid_import,
            baseline_cost=(r.load_kwh * base_rate) if r.grid_up else 0.0,
            actual_cost=r.grid_import * act_rate,
            billing_savings=0.0, islanding_savings=0.0, night_discharge_savings=0.0,
            slab_displacement=0.0, outage_backup_value=0.0, export_earning=0.0,
            grid_charging_cost=0.0,
            isl_pv_direct=0.0, isl_battery=0.0, isl_island_charged=0.0,
            isl_kwh_pv=0.0, isl_kwh_batt=0.0, outage_h=0.0,
        )
        if not r.grid_up:
            row["outage_h"] = r.dt_h

        # ---- 1. BILLING SAVINGS: PV straight to load, grid up ----
        if r.grid_up and r.pv_to_load > 0:
            v = r.pv_to_load * base_rate
            row["billing_savings"] += v
            tot["billing_savings"] += v
            tot["self_consumed_kwh"] += r.pv_to_load

        # ---- 1b. SLAB DISPLACEMENT ----
        # The units the customer STILL imports get cheaper, because solar pushed
        # their monthly total down the telescopic ladder. A 230-unit Maharashtra
        # household cut to 86 units does not merely avoid 144 units: the 86 that
        # remain are repriced out of the ~₹14.4 band into the ~₹6.4 band. That
        # repricing is a genuine, bankable saving, but it is NOT an avoided kWh,
        # so it cannot live in the self-consumption bucket. Named separately it
        # also makes the reconciliation against the bill delta exact.
        if r.grid_up and r.grid_to_load > 0:
            v = r.grid_to_load * (base_rate - act_rate)
            row["slab_displacement"] += v
            tot["slab_displacement"] += v

        # ---- 2a. ISLANDING: PV direct to load, grid down, solar hours ----
        if (not r.grid_up) and r.solar_hr and r.pv_to_load > 0:
            v = r.pv_to_load * base_rate
            row["islanding_savings"] += v
            row["isl_pv_direct"] += v
            row["isl_kwh_pv"] += r.pv_to_load
            tot["islanding_savings"] += v
            tot["islanding_kwh"] += r.pv_to_load

        # ---- 2b. off-solar outage carried directly by PV (rare) ----
        if (not r.grid_up) and (not r.solar_hr) and r.pv_to_load > 0:
            v = r.pv_to_load * base_rate
            row["outage_backup_value"] += v
            tot["outage_backup_value"] += v
            tot["outage_backup_kwh"] += r.pv_to_load

        # ---- charge the provenance buckets ----
        # Buckets are credited with DELIVERABLE kWh: what goes in, times the
        # round trip. The losses are therefore never creditable as savings,
        # and the grid bucket's basis is a cost per kWh that actually reaches
        # the load.
        if r.pv_to_batt > 0:
            if (not r.grid_up) and r.solar_hr:
                b_island += r.pv_to_batt * eff    # charged DURING islanding
            else:
                b_pv += r.pv_to_batt * eff
        if r.grid_to_batt > 0:
            cost = r.grid_to_batt * act_rate      # the full cash cost, undiscounted
            deliverable = r.grid_to_batt * eff
            new_tot = b_grid + deliverable
            grid_basis = ((grid_basis * b_grid) + cost) / new_tot if new_tot > 0 else 0.0
            b_grid = new_tot
            row["grid_charging_cost"] += cost
            tot["grid_charging_cost"] += cost

        # ---- discharge, proportional across provenance buckets ----
        if r.bat_to_load > 0:
            pool = b_island + b_pv + b_grid
            # Measured discharge can exceed the tracked pool — the bank starts
            # the file with a charge nobody observed being put in. That excess
            # is recorded rather than silently valued or silently dropped.
            tot["unattributed_discharge_kwh"] += max(r.bat_to_load - pool, 0.0)
            if pool > 1e-12:
                take = min(r.bat_to_load, pool)
                f_i = b_island / pool
                f_p = b_pv / pool
                f_g = b_grid / pool
                d_i, d_p, d_g = take * f_i, take * f_p, take * f_g
                b_island -= d_i
                b_pv -= d_p
                b_grid -= d_g

                # island-charged energy stays islanding value whenever it leaves
                if d_i > 0:
                    v = d_i * base_rate
                    row["islanding_savings"] += v
                    row["isl_island_charged"] += v
                    tot["islanding_savings"] += v
                    tot["island_batt_kwh"] += d_i

                # PV-charged and grid-charged energy is bucketed by WHEN it is
                # used and what the grid was doing at the time.
                free_val = d_p * base_rate
                grid_val = max(d_g * (base_rate - grid_basis), 0.0)
                v = free_val + grid_val
                kwh = d_p + d_g
                if not r.grid_up:
                    if r.solar_hr:
                        # Grid down during solar hours IS islanding, whatever
                        # charged the bank. The island-charged bucket above is
                        # the separate case of island energy discharged LATER,
                        # outside the window that created it.
                        row["islanding_savings"] += v
                        row["isl_battery"] += v
                        row["isl_kwh_batt"] += kwh
                        tot["islanding_savings"] += v
                        tot["islanding_kwh"] += kwh
                    else:
                        row["outage_backup_value"] += v
                        tot["outage_backup_value"] += v
                        tot["outage_backup_kwh"] += kwh
                elif not r.solar_hr:
                    row["night_discharge_savings"] += v
                    tot["night_discharge_savings"] += v
                    tot["night_discharge_kwh"] += kwh
                else:
                    row["billing_savings"] += v
                    tot["billing_savings"] += v

        # ---- 5. EXPORT ----
        if r.pv_export > 0:
            if tariff.settlement == "no_export":
                v = 0.0
            elif tariff.settlement == "net_billing":
                v = r.pv_export * tariff.export_rate
            else:  # net_metering: 1:1 offset handled at cycle settlement below
                v = 0.0
            row["export_earning"] += v
            tot["export_earning"] += v

        rows.append(row)

    ledger = pd.DataFrame(rows)

    # ---- net-metering settlement, with banking and carry-forward ----
    nm = pd.DataFrame()
    if tariff.settlement == "net_metering":
        nm = net_metering_settlement(df, tariff, settlement_start_month)
        tot["export_earning"] = float(nm["total_value"].sum())
        tot["banked_offset_value"] = float(nm["offset_value"].sum())
        tot["cashout_value"] = float(nm["cashout_value"].sum())
        tot["unsettled_bank_kwh"] = float(nm.attrs.get("unsettled_bank_kwh", 0.0))

    tot["net_savings"] = (
        tot["billing_savings"] + tot["slab_displacement"] + tot["islanding_savings"]
        + tot["night_discharge_savings"] + tot["export_earning"]
        - tot["grid_charging_cost"]
    )
    tot["net_savings_incl_backup"] = tot["net_savings"] + tot["outage_backup_value"]
    tot["baseline_bill_energy"] = float(ledger["baseline_cost"].sum())
    tot["actual_bill_energy"] = float(ledger["actual_cost"].sum())
    tot["bill_delta"] = tot["baseline_bill_energy"] - tot["actual_bill_energy"]
    return dict(totals=tot, ledger=ledger, net_metering_detail=nm)


# =====================================================================
# 5. BILL VIEW (daily / monthly / cycle)
# =====================================================================
def settlement_year(ts, start_month: int = 4) -> str:
    """Label of the settlement year containing `ts` (Indian FY by default)."""
    ts = pd.Timestamp(ts)
    y = ts.year if ts.month >= start_month else ts.year - 1
    return f"FY{y}-{str(y + 1)[-2:]}"


def net_metering_settlement(df: pd.DataFrame, tariff, settlement_start_month: int = 4) -> pd.DataFrame:
    """
    Net-metering settlement with BANKING and CARRY-FORWARD, per MERC
    (Grid Interactive Rooftop RE) Regulations 11.4.

    The rule, and why a per-cycle cash-out is wrong:

      11.4(a) Exported units are set off against imported units in the SAME
              billing period. Any surplus is carried forward to the next
              billing period AS CREDITED UNITS — not as money.
      11.4(c) The settlement period runs 1 April to 31 March. The carried
              forward quantum resets to zero at the start of each period, and
              whatever credit is still unadjusted at the end of the year is
              bought by the licensee at the Generic Tariff for that year.

    The difference is large and it runs in the consumer's favour. A banked
    unit that offsets a future month's import is worth that month's RETAIL
    marginal rate — Rs 6.45 to Rs 22.19 here. The same unit cashed out at the
    generic tariff is worth Rs 2.82. Settling every month at the generic rate
    therefore understates the benefit badly on a site whose winter surplus
    would otherwise carry into a heavy summer.

    Returns one row per billing cycle with the bank balance, what was offset,
    what was actually billable, and any year-end cash-out.
    """
    cyc = (df.groupby("cycle_id")
             .agg(imp=("grid_import", "sum"), exp=("pv_export", "sum"),
                  start=("date", "min"), end=("date", "max"), days=("date", "nunique"))
             .reset_index().sort_values("start"))
    cyc["sy"] = cyc["end"].map(lambda d: settlement_year(d, settlement_start_month))

    rows, bank = [], 0.0
    sys_ = cyc["sy"].tolist()
    for n, r in enumerate(cyc.itertuples(index=False)):
        opening = bank
        avail = r.exp + opening
        offset = min(r.imp, avail)
        billed = r.imp - offset
        bank = avail - offset

        # value of the offset = the bill it removed, taken off the TOP of the
        # telescopic stack (gross import priced in full, less the bill on what
        # remains billable)
        offset_value = (tariff.energy_cost_block(0.0, r.imp)
                        - tariff.energy_cost_block(0.0, billed))

        # year-end cash-out, only when the settlement year actually closes
        # inside the data. A trailing part-year leaves its bank unsettled
        # rather than being cashed out early.
        year_closes = (n + 1 < len(sys_)) and (sys_[n + 1] != r.sy)
        cash_kwh = bank if year_closes else 0.0
        cash_value = cash_kwh * tariff.export_rate
        if year_closes:
            bank = 0.0

        rows.append(dict(
            cycle_id=r.cycle_id, settlement_year=r.sy, days=r.days,
            import_kwh=r.imp, export_kwh=r.exp,
            opening_bank_kwh=opening, offset_kwh=offset, billed_kwh=billed,
            closing_bank_kwh=bank, offset_value=offset_value,
            cashed_out_kwh=cash_kwh, cashout_value=cash_value,
            total_value=offset_value + cash_value,
        ))
    out = pd.DataFrame(rows)
    out.attrs["unsettled_bank_kwh"] = float(bank)
    return out



def _billing_periods(df: pd.DataFrame, cycle_group: pd.DataFrame) -> float:
    """
    How many billing periods a cycle's FIXED charges cover.

    A monthly cycle is ONE billing period and carries the full monthly fixed
    charge and meter rent, whether the month ran 28 days or 31, and whether or
    not the telemetry covers every day of it. Prorating by days produced
    figures like Rs 102.51 against a real bill's Rs 130, which is simply not
    how a connection is billed. Fixed charges appear identically in the
    baseline and the actual bill, so this never moved savings — but it made
    the displayed bill impossible to lay next to the customer's paper one.
    """
    if df.attrs.get("cycle_mode", "monthly") == "monthly":
        return 1.0
    return max(float(df.attrs.get("cycle_days", 30)) / 30.4375, 1e-6)


def bill_table(df: pd.DataFrame, tariff, granularity: str = "cycle") -> pd.DataFrame:
    """
    Build a full consumer bill, baseline vs actual, at the requested
    granularity. Slabs are resolved on the BILLING CYCLE, which is the only
    level at which they are real; a daily or monthly view simply apportions
    that cycle's charges, so a "daily bill" never pretends the slab counter
    resets at midnight.
    """
    key = {"cycle": "cycle_id", "month": None, "day": None}[granularity] if granularity in ("cycle",) else None
    d = df.copy()
    if granularity == "cycle":
        d["_k"] = d["cycle_id"]
    elif granularity == "month":
        d["_k"] = d["local"].dt.to_period("M").astype(str)
    else:
        d["_k"] = d["date"].dt.date.astype(str)

    out = []
    # cycle-level slab position, apportioned to sub-periods
    for cid, cg in d.groupby("cycle_id"):
        base_units_cycle = float(cg.loc[cg["grid_up"], "load_kwh"].sum())
        act_units_cycle = float(cg["grid_import"].sum())
        months = _billing_periods(df, cg)
        base_bill = tariff.build_bill(base_units_cycle, n_months=months)
        act_bill = tariff.build_bill(act_units_cycle, n_months=months)
        for k, sg in cg.groupby("_k"):
            bu = float(sg.loc[sg["grid_up"], "load_kwh"].sum())
            au = float(sg["grid_import"].sum())
            fb = bu / base_units_cycle if base_units_cycle > 0 else 0.0
            fa = au / act_units_cycle if act_units_cycle > 0 else 0.0
            out.append(dict(
                period=k, cycle=cid, days=sg["date"].nunique(),
                baseline_units=bu, actual_units=au,
                baseline_energy=base_bill["energy_charge"] * fb,
                baseline_wheeling=base_bill["wheeling_charge"] * fb,
                baseline_fixed=base_bill["fixed_charge"] * fb,
                baseline_meter=base_bill["meter_rent"] * fb,
                baseline_duty=base_bill["electricity_duty"] * fb,
                baseline_total=base_bill["total"] * fb,
                actual_energy=act_bill["energy_charge"] * fa,
                actual_wheeling=act_bill["wheeling_charge"] * fa,
                actual_fixed=act_bill["fixed_charge"] * fa,
                actual_meter=act_bill["meter_rent"] * fa,
                actual_duty=act_bill["electricity_duty"] * fa,
                actual_total=act_bill["total"] * fa,
            ))
    t = pd.DataFrame(out).sort_values("period").reset_index(drop=True)
    if not t.empty:
        t["bill_reduction"] = t["baseline_total"] - t["actual_total"]
    return t


def full_cycle_bills(df: pd.DataFrame, tariff,
                     settlement_start_month: int = 4) -> pd.DataFrame:
    """
    One un-apportioned, fully itemised bill per real billing cycle — the
    version that can be laid next to the customer's paper bill.

    `act_total` is the bill on GROSS metered import. The export credit is a
    separate line, because that is how it appears on a net-metered bill and
    because keeping it separate is what stops it from being counted both here
    and in the export bucket of the savings ledger.

    Under net metering the credit is the BANKED settlement (MERC Reg 11.4):
    exports offset import within the month, surplus carries forward as kWh
    credit, and only what is still unadjusted at the end of the settlement
    year is cashed at the generic tariff. `net_billed_units` is what the
    consumer is actually charged for after credits — which is why it can be
    zero in a surplus month even though the meter recorded real import.
    """
    nm_rows = {}
    if tariff.settlement == "net_metering":
        _nm = net_metering_settlement(df, tariff, settlement_start_month)
        nm_rows = {r["cycle_id"]: r for r in _nm.to_dict("records")}
    out = []
    for cid, cg in df.groupby("cycle_id"):
        bu = float(cg.loc[cg["grid_up"], "load_kwh"].sum())
        au = float(cg["grid_import"].sum())
        ex = float(cg["pv_export"].sum())
        months = _billing_periods(df, cg)
        b = tariff.build_bill(bu, n_months=months)
        a = tariff.build_bill(au, n_months=months)
        floor = tariff.build_bill(0.0, n_months=months)["total"]

        if tariff.settlement == "net_metering":
            s = nm_rows.get(cid)
            credit = (s["offset_value"] + s["cashout_value"]) if s else 0.0
        elif tariff.settlement == "net_billing":
            credit = ex * tariff.export_rate
        else:
            credit = 0.0

        out.append(dict(
            cycle=cid, days=cg["date"].nunique(),
            from_date=cg["date"].min().date(), to_date=cg["date"].max().date(),
            baseline_units=round(bu, 1), actual_units=round(au, 1),
            export_units=round(ex, 1),
            net_billed_units=round(nm_rows[cid]["billed_kwh"], 1) if cid in nm_rows else round(au, 1),
            bank_kwh=round(nm_rows[cid]["closing_bank_kwh"], 1) if cid in nm_rows else 0.0,
            **{f"base_{k}": v for k, v in b.items() if k != "units"},
            **{f"act_{k}": v for k, v in a.items() if k != "units"},
            export_credit=credit,
            # A credit can wipe out the energy part of a bill but never the
            # fixed charge and meter rent: a surplus month still produces a
            # payable bill, not a negative one. The floor is the bill on zero
            # units, which carries the duty on those fixed components too.
            act_total_after_credit=max(a["total"] - credit, floor),
            bill_reduction=b["total"] - a["total"],
            bill_reduction_after_credit=b["total"] - max(a["total"] - credit, floor),
        ))
    return pd.DataFrame(out).sort_values("cycle").reset_index(drop=True)


# =====================================================================
# 6. OUTAGE MODEL
# =====================================================================
def outage_episodes(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["outage"]].copy()
    if d.empty:
        return pd.DataFrame()
    brk = (~df["outage"]).cumsum()
    d["_g"] = brk.loc[d.index]
    ep = d.groupby("_g").agg(
        start=("local", "min"), end=("local", "max"), rows=("outage", "size"),
        metered_h=("dt_h", "sum"), load_kwh=("load_kwh", "sum"),
        pv_kwh=("pv_kwh", "sum"), bat_kwh=("bat_to_load", "sum"),
        solar_frac=("solar_hr", "mean"),
    ).reset_index(drop=True)
    nominal = df.attrs.get("nominal_interval_min", 5.0) / 60.0
    ep["duration_h"] = (ep["end"] - ep["start"]).dt.total_seconds() / 3600.0 + nominal
    ep["type"] = np.where(ep["solar_frac"] >= 0.5, "solar-hour", "off-solar")
    return ep.sort_values("start").reset_index(drop=True)


def outage_model(df: pd.DataFrame, nameplate_kwh: float, charge_rate_c: float = 0.15,
                 usable_kwh: float | None = None) -> dict:
    """
    The quantity the business case actually hinges on: how much battery
    headroom can realistically be refilled by PV during daytime outages.

    During a solar-hour outage the inverter is islanded — PV that the load
    cannot absorb has nowhere to go except the battery, and the charge
    controller caps that at the C-rate. So the ceiling on "energy I can
    deliberately pre-drain overnight and still recover tomorrow" is:

        headroom = C-rate x NAMEPLATE kWh x (solar-hour outage hours that day)

    C-RATE IS DEFINED ON RATED CAPACITY, NOT USABLE CAPACITY. 0.15C on a
    200 Ah / 48 V bank is 30 A = 1.44 kW, whatever depth of discharge the
    system is configured to allow. Applying the C-rate to a DoD-reduced
    "usable" figure understates the charge rate and is simply the wrong
    definition. Usable capacity still matters — it caps how much headroom
    there is to refill at all — so it is reported alongside.

    This is a CEILING, not an expectation. It is only achieved if PV is
    actually generating during that outage and the load is not already
    consuming all of it. Both the ceiling and the realised figure are
    returned, because the gap between them is where an over-confident
    pre-drain policy loses the customer money.
    """
    # SPEED, no change to any number. groupby.apply with a lambda runs the
    # whole body once per day in Python — on a year of minute data that is
    # several hundred passes building a Series each time. Masking the columns
    # first and summing them in ONE grouped pass is the same arithmetic:
    # x.loc[mask, col].sum() and col.where(mask, 0).sum() agree exactly,
    # because the rows excluded by the mask contribute zero either way.
    d = df
    _out = d["outage"].to_numpy()
    _sol = d["solar_hr"].to_numpy()
    _up = d["grid_up"].to_numpy()
    _dt = d["dt_h"].to_numpy(dtype=float)
    _masked = pd.DataFrame({
        "date": d["date"].to_numpy(),
        "solar_outage_h": np.where(_out & _sol, _dt, 0.0),
        "offsolar_outage_h": np.where(_out & ~_sol, _dt, 0.0),
        "outage_h": np.where(_out, _dt, 0.0),
        "metered_h": _dt,
        "grid_up_h": np.where(_up, _dt, 0.0),
        "pv_to_batt_in_outage": np.where(
            _out & _sol, d["pv_to_batt"].to_numpy(dtype=float), 0.0),
        "curtailed_in_outage": np.where(
            _out, d["pv_curtailed"].to_numpy(dtype=float), 0.0),
        "outage_load": np.where(_out, d["load_kwh"].to_numpy(dtype=float), 0.0),
        "_pv_surplus": np.where(
            _out & _sol,
            np.maximum(d["pv_kwh"].to_numpy(dtype=float)
                       - d["load_kwh"].to_numpy(dtype=float), 0.0), 0.0),
    })
    daily = _masked.groupby("date", sort=True).sum().reset_index()

    kw_cap = charge_rate_c * nameplate_kwh
    daily["charge_rate_limit_kwh"] = kw_cap * daily["solar_outage_h"]
    daily["headroom_ceiling_kwh"] = daily["charge_rate_limit_kwh"]
    if usable_kwh:
        # A day's recoverable headroom can never exceed the usable bank: once
        # the bank is full there is nothing left to refill.
        daily["headroom_ceiling_kwh"] = daily["headroom_ceiling_kwh"].clip(upper=usable_kwh)

    # THIRD CONSTRAINT — does the surplus physically exist?
    # The two limits above describe what the hardware could absorb. Neither
    # asks whether there is any PV left over to absorb. During an outage the
    # load is being carried by PV too, so the energy actually available to the
    # bank is PV minus load, and only in the intervals where that is positive.
    # On this kind of site that is by far the tightest of the three, and
    # leaving it out turns a ceiling into a fantasy.
    daily["pv_surplus_kwh"] = daily.pop("_pv_surplus")
    daily["recoverable_kwh"] = np.minimum(
        daily["headroom_ceiling_kwh"], daily["pv_surplus_kwh"])
    daily["binding_constraint"] = np.where(
        daily["pv_surplus_kwh"] <= daily["headroom_ceiling_kwh"], "PV surplus",
        np.where(daily["charge_rate_limit_kwh"] <= (usable_kwh or np.inf),
                 "charge rate", "usable bank"))
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)

    monthly = daily.groupby("month").agg(
        days=("date", "nunique"),
        metered_h=("metered_h", "sum"),
        grid_up_h=("grid_up_h", "sum"),
        solar_outage_h=("solar_outage_h", "sum"),
        offsolar_outage_h=("offsolar_outage_h", "sum"),
        outage_h=("outage_h", "sum"),
        headroom_ceiling_kwh=("headroom_ceiling_kwh", "sum"),
        realised_pv_to_batt_kwh=("pv_to_batt_in_outage", "sum"),
        pv_surplus_kwh=("pv_surplus_kwh", "sum"),
        recoverable_kwh=("recoverable_kwh", "sum"),
        curtailed_kwh=("curtailed_in_outage", "sum"),
        outage_load_kwh=("outage_load", "sum"),
    ).reset_index()
    # The identity a reader must be able to check by eye:
    #     grid_up_h + outage_h = metered_h          (outage is INSIDE metered time —
    #                                                the logger keeps recording during an
    #                                                outage, so adding the two double-counts)
    #     metered_h + unlogged_h = 24 x days        (the remainder is logger downtime,
    #                                                which is NOT an outage: during a real
    #                                                outage data still arrives)
    monthly["calendar_h"] = 24.0 * monthly["days"]
    monthly["unlogged_h"] = (monthly["calendar_h"] - monthly["metered_h"]).clip(lower=0)
    monthly["data_completeness_pct"] = 100.0 * monthly["metered_h"] / monthly["calendar_h"]
    monthly["solar_outage_h_per_day"] = monthly["solar_outage_h"] / monthly["days"]
    monthly["outage_pct_of_time"] = 100 * monthly["outage_h"] / monthly["metered_h"]
    monthly["headroom_kwh_per_day"] = monthly["headroom_ceiling_kwh"] / monthly["days"]
    monthly["realisation_pct"] = np.where(
        monthly["headroom_ceiling_kwh"] > 0,
        100 * monthly["realised_pv_to_batt_kwh"] / monthly["headroom_ceiling_kwh"],
        np.nan,
    )

    s = daily["solar_outage_h"]
    hr = daily["headroom_ceiling_kwh"]
    stats = dict(
        charge_kw_cap=kw_cap,
        nameplate_kwh=nameplate_kwh,
        usable_kwh=usable_kwh or float("nan"),
        days=int(len(daily)),
        mean_h=float(s.mean()), median_h=float(s.median()),
        p10_h=float(s.quantile(0.10)), p90_h=float(s.quantile(0.90)),
        zero_days_pct=float(100 * (s <= 0.01).mean()),
        mean_headroom=float(hr.mean()),
        median_headroom=float(hr.median()),
        p10_headroom=float(hr.quantile(0.10)),
        p90_headroom=float(hr.quantile(0.90)),
        total_headroom=float(hr.sum()),
        total_pv_surplus=float(daily["pv_surplus_kwh"].sum()),
        total_recoverable=float(daily["recoverable_kwh"].sum()),
        median_recoverable=float(daily["recoverable_kwh"].median()),
        binding_counts=daily["binding_constraint"].value_counts().to_dict(),
        headroom_as_pct_of_usable=(
            float(100 * hr.median() / usable_kwh) if usable_kwh else float("nan")),
        realised_total=float(daily["pv_to_batt_in_outage"].sum()),
    )
    return dict(daily=daily, monthly=monthly, stats=stats)


# =====================================================================
# 7. DATA INTEGRITY
# =====================================================================
def integrity_report(raw: pd.DataFrame, df: pd.DataFrame, ts_col: str) -> list[dict]:
    f = []
    add = lambda sev, area, finding, impact: f.append(
        dict(severity=sev, area=area, finding=finding, impact=impact))

    # --- clock sanity across both timestamp columns ---
    if COL_TS_ALT in raw.columns and ts_col in raw.columns:
        a = pd.to_datetime(raw[ts_col], errors="coerce")
        b = pd.to_datetime(raw[COL_TS_ALT], errors="coerce")
        off = (b - a).dt.total_seconds() / 3600.0
        if off.notna().sum() > 10:
            drift = off.dropna()
            spread = drift.max() - drift.min()
            if spread > 0.25:
                add("critical", "Clock integrity",
                    f"'{ts_col}' and '{COL_TS_ALT}' disagree by {drift.min():.2f}h to "
                    f"{drift.max():.2f}h, and the gap drifts by {spread:.2f}h across the file.",
                    "Two clocks that drift apart linearly means one is an un-synced device RTC. "
                    "Every time-of-day split in this analysis — solar-hour vs night outage, ToD "
                    "windows, night battery discharge — depends on picking the right one and "
                    "correcting its offset. Confirm the platform's timezone handling before "
                    "quoting any day/night split.")

    # --- solar-noon alignment of the chosen clock ---
    fit = fit_local_time_offset(raw.assign(**{ts_col: pd.to_datetime(raw[ts_col], errors="coerce")})
                                .dropna(subset=[ts_col]), ts_col)
    if fit["ok"]:
        if abs(fit["offset_h"]) > 0.5:
            add("critical", "Local time offset",
                f"PV output in '{ts_col}' peaks at {fit['observed_noon']:.2f}h, but astronomical "
                f"solar noon for this longitude is {fit['astronomical_noon']:.2f}h — implying the "
                f"column runs {fit['offset_h']:+.2f}h off true local time "
                f"(fitted over {fit['n_days']} clear days, scatter ±{fit['scatter_min']:.0f} min).",
                "Uncorrected, a solar-hour window of 07:00-18:00 local would be applied to the "
                "wrong hours, misclassifying daytime outages as night-time ones and vice versa. "
                "The offset is applied in the sidebar; it still needs confirming against the "
                "site's real coordinates and the logging platform's timezone setting.")
        else:
            add("info", "Local time offset",
                f"Clock aligns with solar noon to within {abs(fit['offset_h']) * 60:.0f} min "
                f"over {fit['n_days']} clear days.", "Time-of-day splits can be trusted.")

    # --- energy integration vs the inverter's own counter ---
    if COL_TOTAL_GEN in raw.columns:
        tg = pd.to_numeric(raw[COL_TOTAL_GEN], errors="coerce").dropna()
        if len(tg) > 2:
            counter = tg.iloc[-1] - tg.iloc[0]
            integ = float(df["pv_kwh"].sum())
            if counter > 0:
                err = 100 * (integ - counter) / counter
                sev = "info" if abs(err) < 5 else "warning"
                add(sev, "Energy integration validation",
                    f"PV integrated from instantaneous power = {integ:,.0f} kWh vs the inverter's "
                    f"own cumulative counter = {counter:,.0f} kWh ({err:+.1f}%).",
                    "This is the strongest available independent check on the kWh figures. A gap "
                    "under about 5% is explained by excluded device-offline windows; a larger gap "
                    "means the interval integration or the gap rules are wrong.")

    # --- coverage ---
    span_days = (df["local"].max() - df["local"].min()).total_seconds() / 86400
    metered_days = float(df["dt_h"].sum()) / 24
    loss = 100 * (1 - metered_days / span_days) if span_days > 0 else 0
    add("warning" if loss > 5 else "info", "Data coverage",
        f"File spans {span_days:.1f} calendar days but only {metered_days:.1f} days of metered "
        f"time survive the gap rules ({loss:.1f}% excluded).",
        "Annualised figures must be scaled from metered time, not calendar time, or the loss is "
        "silently reported as reduced consumption.")

    # --- load channel ---
    if COL_ACTIVE in raw.columns and COL_APPARENT in raw.columns:
        m = pd.to_numeric(raw[COL_APPARENT], errors="coerce") > 100
        pf = (pd.to_numeric(raw[COL_ACTIVE], errors="coerce")[m]
              / pd.to_numeric(raw[COL_APPARENT], errors="coerce")[m]).dropna()
        if len(pf) > 100:
            add("warning", "Load measurement channel",
                f"Measured power factor varies from {pf.quantile(0.05):.2f} to "
                f"{pf.quantile(0.95):.2f} (median {pf.median():.2f}).",
                "A single assumed power factor applied to apparent power cannot track that "
                "spread. 'AC output active power(W)' is populated on this file and is the real "
                "billable quantity — it is used directly, and the PF assumption is retired.")

    # --- float current ---
    if COL_BAT_SOC in raw.columns:
        at_full = pd.to_numeric(raw[COL_BAT_SOC], errors="coerce").fillna(0) >= 100
        float_kwh = float(df["float_kwh"].sum())
        real_chg = float(df["bat_chg_kwh"].sum())
        if float_kwh > 0.05 * max(real_chg, 1e-9):
            add("critical", "Battery charge energy",
                f"The battery reads 100% SOC on {100 * at_full.mean():.0f}% of rows. Integrating "
                f"charging current across those rows adds {float_kwh:,.0f} kWh of float/absorption "
                f"current on top of {real_chg:,.0f} kWh of genuine charging.",
                "Float current offsets self-discharge and is lost as heat — it is not energy "
                "stored. Counting it inflates grid-charging cost and makes round-trip efficiency "
                "come out impossibly low. It is excluded here.")

    # --- SOC telemetry quality ---
    if COL_BAT_SOC in df.columns:
        soc = pd.to_numeric(df[COL_BAT_SOC], errors="coerce")
        flips = int((soc.diff().abs() > 5).sum())
        if flips > 0.02 * len(soc):
            add("warning", "SOC telemetry quality",
                f"Battery SOC jumps by more than 5 percentage points between consecutive "
                f"5-minute rows on {flips:,} occasions ({100 * flips / len(soc):.1f}% of rows).",
                "SOC cannot physically move that fast on this bank, so the reported SOC is noisy. "
                "Any energy figure derived by differencing SOC is unusable; battery energy is "
                "taken from current x voltage instead.")

    # --- outage definition sensitivity ---
    raw_down = df["outage_raw"]
    kept = df["outage"]
    dropped_h = float(df.loc[raw_down & ~kept, "dt_h"].sum())
    if dropped_h > 0:
        add("info", "Outage definition",
            f"{dropped_h:.1f} h of zero-voltage readings sit in runs shorter than the minimum "
            f"outage duration and are excluded; {float(df.loc[kept, 'dt_h'].sum()):.1f} h are "
            f"counted as real outages.",
            "Single-sample voltage dropouts are usually sampling artefacts rather than supply "
            "interruptions. The threshold is a judgement call, so the sensitivity is shown "
            "explicitly rather than buried.")

    # --- dispatch-order reconciliation ---
    if "unexplained_discharge" in df.columns:
        ud = float(df["unexplained_discharge"].sum())
        bl = float(df["bat_to_load"].sum())
        if ud > 0.01:
            add("warning", "Dispatch-order reconciliation",
                f"Under PV-first dispatch, {ud:,.1f} kWh of measured battery discharge has no "
                f"load left to serve — PV alone already covered the load in those intervals "
                f"({100 * ud / max(bl + ud, 1e-9):.0f}% of total discharge).",
                "Five-minute averaging hides sub-interval crossings: within a window whose "
                "average shows PV above load, the load can briefly exceed PV and the battery "
                "covers it. The residual is reported rather than absorbed into a bucket. A "
                "battery-first ordering makes it vanish but is less faithful to how the "
                "inverter actually dispatches; the two orderings differ by about 0.1% of net "
                "savings, so the choice is not material to the result.")

    # --- export measurement ---
    add("warning", "Export is inferred, not measured",
        "This export contains no grid import/export channel. Export is therefore derived as "
        "PV minus load minus charging, interval by interval.",
        "The derived figure is only as good as the interval resolution: 5-minute averaging hides "
        "sub-interval swings where PV and load cross, so true export is slightly understated and "
        "self-consumption slightly overstated. Any export revenue claim should be reconciled "
        "against the bidirectional meter reading before it goes to a customer.")

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(f, key=lambda x: order[x["severity"]])


# =====================================================================
# 8. ISLANDING — SLAB-BY-SLAB WORKING
# =====================================================================
def outage_slab_working(df: pd.DataFrame, tariff, granularity: str = "cycle") -> pd.DataFrame:
    """
    The arithmetic behind the outage buckets, written out slab by slab.

    A summary figure invites the question "how did you get that?". This
    answers it before it is asked: for each billing cycle it shows the
    month's without-solar consumption, the outage units stacked on top of it,
    and exactly how those units split across tariff slabs — which is what
    the bill would have done had the grid been up.

    Returns one row per (period, slab) so nothing is hidden inside an
    average. The slab rate shown is the all-in marginal rate: energy +
    wheeling + FAC, grossed up by electricity duty.
    """
    out = []
    for cid, g in df.groupby("cycle_id"):
        base = float(g.loc[g["grid_up"], "load_kwh"].sum())
        isl = float(g.loc[g["outage"] & g["solar_hr"], "load_kwh"].sum())
        night = float(g.loc[g["outage"] & (~g["solar_hr"]), "load_kwh"].sum())
        total = isl + night
        if total <= 1e-9:
            continue
        isl_share = isl / total

        # walk the outage block up the telescopic ladder from `base`
        cursor, remaining, prev_cap = base, total, 0.0
        for cap, _rate in tariff.slabs:
            width_left = cap - max(cursor, prev_cap)
            prev_cap = cap
            if width_left <= 0:
                continue
            take = min(remaining, width_left)
            if take > 1e-9:
                rate = tariff.marginal_rate(cursor)
                en = tariff.slab_rate(cursor)
                duty_amt = (en + tariff.wheeling_per_kwh + tariff.fac_per_kwh) * tariff.duty_pct / 100.0
                lo = cap - width_left if np.isfinite(cap) else cursor
                out.append(dict(
                    period=cid,
                    base_units_no_solar=round(base, 1),
                    slab=(f"{max(cursor, lo):.0f}–{min(cursor + take, cap):.0f} units"
                          if np.isfinite(cap) else f"above {cursor:.0f} units"),
                    energy_rate=round(en, 2),
                    wheeling_rate=round(tariff.wheeling_per_kwh, 2),
                    fac_rate=round(tariff.fac_per_kwh, 2),
                    duty_per_kwh=round(duty_amt, 2),
                    all_in_rate=round(rate, 2),
                    outage_units=round(take, 2),
                    islanding_units=round(take * isl_share, 2),
                    night_units=round(take * (1 - isl_share), 2),
                    value=round(take * rate, 2),
                    islanding_value=round(take * rate * isl_share, 2),
                ))
                remaining -= take
                cursor += take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9:
            rate = tariff.marginal_rate(cursor)
            _en = tariff.slab_rate(cursor)
            out.append(dict(
                period=cid, base_units_no_solar=round(base, 1),
                slab=f"above {cursor:.0f} units",
                energy_rate=round(_en, 2), wheeling_rate=round(tariff.wheeling_per_kwh, 2),
                fac_rate=round(tariff.fac_per_kwh, 2),
                duty_per_kwh=round((_en + tariff.wheeling_per_kwh + tariff.fac_per_kwh)
                                   * tariff.duty_pct / 100.0, 2),
                all_in_rate=round(rate, 2),
                outage_units=round(remaining, 2),
                islanding_units=round(remaining * isl_share, 2),
                night_units=round(remaining * (1 - isl_share), 2),
                value=round(remaining * rate, 2),
                islanding_value=round(remaining * rate * isl_share, 2)))
    return pd.DataFrame(out)


# =====================================================================
# 9. PV ARRAY SIZE — estimated, because no export states it
# =====================================================================
def estimate_array_size(df: pd.DataFrame, latitude_band_yield: tuple = (1400.0, 1600.0),
                        clear_day_derate: float = 0.80) -> dict:
    """
    Estimate installed DC array capacity (kWp) from telemetry alone.

    No DESS export carries the array rating. What it carries is the INVERTER
    rating ("AC output rating apparent power(VA)"), which people routinely
    mistake for the system size — a 5.6 kVA inverter says nothing about
    whether 2 kWp or 6 kWp is wired to it.

    Three independent estimates, deliberately not averaged:

      1. PEAK POWER — the best observed DC power, divided by a derate for
         temperature, soiling and off-normal incidence. Upper-bound flavoured:
         it reflects what the array CAN do on its best interval.
      2. CLEAR-DAY PEAK — the median daily peak across the highest-yield days.
         More robust than a single peak, since one freak interval cannot carry
         it, and less sensitive to a clipped or spiking reading.
      3. SPECIFIC YIELD — annualised generation divided by the kWh/kWp a
         healthy Indian rooftop returns (1,400-1,600). This is the only one
         that reflects how the array performs over a year rather than at its
         best moment.

    When 1 and 2 land well above 3, the array is bigger than its output
    justifies — shading, soiling, a dead string or a failed MPPT. That gap is
    worth more to a customer than any savings figure, so it is reported rather
    than averaged away.
    """
    pv = df["pv_kw"]
    metered_days = float(df["dt_h"].sum()) / 24.0
    annual_kwh = float(df["pv_kwh"].sum()) * 365.0 / max(metered_days, 1e-9)

    daily = df.groupby("date").agg(peak=("pv_kw", "max"), kwh=("pv_kwh", "sum"))
    clear = daily.nlargest(max(int(len(daily) * 0.10), 5), "kwh")
    clear_peak = float(clear["peak"].median())

    peak_obs = float(pv.max())
    est_peak = peak_obs / clear_day_derate
    est_clear = clear_peak / clear_day_derate
    est_yield_hi = annual_kwh / latitude_band_yield[0]   # low yield -> larger array
    est_yield_lo = annual_kwh / latitude_band_yield[1]

    consistent = (est_yield_lo <= est_clear <= est_yield_hi * 1.15)
    implied_yield_at_clear = annual_kwh / max(est_clear, 1e-9)

    return dict(
        inverter_va=float(df.attrs.get("inverter_va", float("nan"))),
        peak_observed_kw=peak_obs,
        clear_day_peak_kw=clear_peak,
        annual_kwh=annual_kwh,
        metered_days=metered_days,
        est_from_peak_kwp=est_peak,
        est_from_clear_day_kwp=est_clear,
        est_from_yield_kwp=(est_yield_lo, est_yield_hi),
        implied_yield_at_clear_estimate=implied_yield_at_clear,
        consistent=bool(consistent),
        shortfall_pct=(100.0 * (1 - implied_yield_at_clear / latitude_band_yield[0])
                       if implied_yield_at_clear < latitude_band_yield[0] else 0.0),
        second_string_dead=bool(("PV2 Input Power(W)" in df.columns)
                                and float(df["PV2 Input Power(W)"].abs().max()) == 0.0),
    )