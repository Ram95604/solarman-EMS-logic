"""
State retail tariff definitions + bill build-up.
=================================================
Every number here is traceable to a published tariff order. Nothing is a
round-number guess, because the whole credibility of the savings analysis
rests on these rates.

DESIGN RULE THAT MATTERS MOST
----------------------------
A saving is only a saving if the rupee would actually have been spent.
Therefore:

  * MARGINAL (variable) charges  -> used to value avoided energy.
        energy slab rate + wheeling + FAC, then grossed up by electricity duty.
  * FIXED charges               -> NEVER used to value avoided energy.
        Fixed charge, meter rent and the per-kW demand charge are paid whether
        or not the solar system exists, so avoiding a kWh does not avoid them.
        They appear in the DISPLAYED BILL (both the baseline bill and the
        actual bill) and therefore cancel out in the delta. Including them in
        a per-kWh "savings rate" is the single most common way a residential
        solar savings model gets inflated.

So: `marginal_rate()` drives savings. `build_bill()` drives the bill shown on
screen. They are deliberately different functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INF = float("inf")


# =====================================================================
# TARIFF CATALOGUE
# =====================================================================
# Each slab list is TELESCOPIC: (upper_bound_units, Rs_per_kWh). A consumer at
# 250 units in Maharashtra pays 100 units at 3.96 and 150 units at 10.80 — not
# all 250 at 10.80. Indian residential tariffs for MSEDCL and UPPCL LMV-1 are
# both telescopic; a non-telescopic ("whole consumption at the slab you land
# in") reading would overstate the bill, and therefore overstate savings.

STATE_TARIFFS: dict[str, dict] = {
    "Maharashtra — MSEDCL LT-I(B) Residential": dict(
        code="MH",
        slabs=[(100, 3.96), (300, 10.80), (500, 15.03), (INF, 17.53)],
        wheeling_per_kwh=1.60,
        fac_per_kwh=0.00,
        duty_pct=16.0,
        # Maharashtra Electricity Duty Act 2016, s.3(1)(a): duty is levied on
        # "consumption charges". s.2(1)(e) defines those as the licensee's
        # charges for energy consumed, EXPRESSLY INCLUDING fixed charges such
        # as demand charges, plus energy charges, FAC and reliability charges.
        # So fixed charges are inside the duty base by statute.
        #
        # Meter rent is named neither way. The definition's list is open
        # ("such as"), but meter rent is a charge for equipment rather than
        # for energy consumed, so it is left OUT of the base by default. This
        # is the one genuinely unsettled component — a real bill settles it.
        duty_on_fixed=True,
        duty_on_meter=False,
        fixed_options={
            "Single phase (sanctioned load ≤ 5 kW)": 130.0,
            "Three phase (sanctioned load > 5 kW)": 435.0,
        },
        fixed_per_kw_month=0.0,
        meter_rent_month=30.0,
        # MERC generic tariff for SURPLUS rooftop solar injected to the grid.
        export_rate_default=2.82,
        has_tod=False,
        source=(
            "MERC Order, Case No. 75 of 2025 (MSEDCL MYT mid-term review), "
            "tariff effective 01-Apr-2026. Surplus rooftop injection tariff "
            "₹2.82/kWh for FY 2026-27. Duty base per Maharashtra Electricity "
            "Duty Act 2016, s.3(1)(a) and s.2(1)(e)."
        ),
        notes=(
            "Wheeling charge ₹1.60/kWh is a VARIABLE charge in Maharashtra — it "
            "scales with units consumed, so avoided units DO avoid it. It must be "
            "inside the marginal savings rate. Electricity duty at 16% then applies "
            "on top, so the true avoided cost of one unit in the first slab is "
            "(3.96 + 1.60) x 1.16 = ₹6.45, not ₹3.96. FAC is notified monthly and "
            "defaults to 0 here — set it from the customer's actual bill."
        ),
    ),
    "Uttar Pradesh — UPPCL LMV-1 Domestic (Urban, metered)": dict(
        code="UP-U",
        slabs=[(150, 5.50), (300, 6.00), (INF, 6.50)],
        wheeling_per_kwh=0.00,  # UP LMV-1 retail tariff is bundled; no separate wheeling
        fac_per_kwh=0.00,
        duty_pct=5.0,
        duty_on_fixed=True,
        duty_on_meter=False,
        fixed_options={},
        fixed_per_kw_month=110.0,  # Rs per kW of sanctioned load per month
        meter_rent_month=15.0,     # single-phase meter; verify against the bill
        export_rate_default=2.82,  # placeholder — see notes
        has_tod=False,
        source=(
            "UPERC Rate Schedule for FY 2026-27 (order dated 02-Jul-2026); LMV-1 "
            "domestic urban metered rates unchanged since 01-Oct-2024. Duty base per "
            "UP Electricity (Duty) Act 1952, s.2(h)."
        ),
        notes=(
            "Electricity duty follows the UP Electricity (Duty) Act 1952. Section 2(h) "
            "defines the 'rate charged' on which duty is levied as including, for a two-part "
            "tariff, 'the fixed charge and also the unit-charge' — and expressly EXCLUDING "
            "'hire for meter or service line'. So duty applies to the fixed charge but not to "
            "meter rent, the same split as Maharashtra, and here the statute names meter rent "
            "directly rather than leaving it to inference. "
            "UP LMV-1 has NO separate wheeling charge — the retail slab rate is "
            "bundled, so the marginal rate is simply slab x 1.05 (5% duty). The "
            "fixed charge is ₹110 per kW of SANCTIONED LOAD per month, not a flat "
            "monthly figure, so it must be entered per site. Export settlement rate "
            "is a placeholder: confirm against the applicable UPERC net-metering "
            "regulation before quoting any export earning for a UP site."
        ),
    ),
    "Uttar Pradesh — UPPCL LMV-1 Domestic (Rural, metered)": dict(
        code="UP-R",
        slabs=[(100, 3.35), (150, 3.85), (300, 5.00), (INF, 5.50)],
        wheeling_per_kwh=0.00,
        fac_per_kwh=0.00,
        duty_pct=5.0,
        duty_on_fixed=True,
        duty_on_meter=False,
        fixed_options={},
        fixed_per_kw_month=90.0,
        meter_rent_month=15.0,
        export_rate_default=2.82,
        has_tod=False,
        source=("UPERC Rate Schedule FY 2026-27, LMV-1 rural metered (non-life-line); "
                "duty base per UP Electricity (Duty) Act 1952, s.2(h)."),
        notes=(
            "Rural LMV-1 rates are materially lower than urban — the SAME physical "
            "solar system saves roughly half as much money on a rural UP connection "
            "as on a Maharashtra one. Life-line consumers (≤1 kW, ≤100 units) sit on "
            "a separate subsidised schedule not modelled here."
        ),
    ),
    # ================= HIGHEST-TARIFF STATES =================
    "Tamil Nadu — TNPDCL LT-IA Domestic": dict(
        code="TN", region="Highest 5",
        slabs=[(200, 4.95), (250, 6.65), (300, 8.80), (400, 9.95), (500, 11.05), (INF, 12.15)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.00,
        duty_pct=0.0, duty_on_fixed=False, duty_on_meter=False,
        fixed_options={}, fixed_per_kw_month=0.0, meter_rent_month=0.0,
        export_rate_default=2.82, has_tod=False, verified=True,
        source="TNERC Tariff Order No. 6 of 2025, in force 01-Jul-2025 (secondary source).",
        notes=(
            "NO fixed charge and NO electricity duty on LT-IA domestic — the 5% duty "
            "applies to commercial LT-V only. So the marginal rate IS the slab rate. "
            "TWO TRAPS: (1) TNPDCL bills BI-MONTHLY; the slab bands above are monthly "
            "equivalents, so a two-month total must be halved before the ladder is "
            "applied or every household lands several slabs too high. (2) The free-units "
            "scheme (G.O.(Ms) No.50, 10-May-2026) gives 200 free units per two-month "
            "cycle at or below 500 units, else 100 — which removes the saving entirely "
            "for low-consumption households. Neither is modelled here."
        ),
    ),
    "Andhra Pradesh — APERC LT-I Domestic": dict(
        code="AP", region="Highest 5",
        slabs=[(30, 1.90), (75, 3.00), (125, 4.50), (225, 6.00), (400, 8.75), (INF, 9.75)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.06,
        duty_pct=0.0, duty_on_fixed=False, duty_on_meter=False,
        fixed_options={}, fixed_per_kw_month=10.0, meter_rent_month=40.0,
        export_rate_default=2.82, has_tod=False, verified=True,
        source="APERC retail supply tariff FY 2026-27, one schedule across APSPDCL, APCPDCL and APEPDCL (secondary source; no order number published).",
        notes=(
            "Electricity duty here is a FLAT ₹0.06/unit, not a percentage, so it is "
            "carried in levy_per_kwh and there is no duty-base question. Six slabs — the "
            "steepest early ladder of any state modelled: the rate quintuples between 30 "
            "and 400 units. The ₹40/month entered as meter rent is the mid-point of the "
            "₹25–55 customer charge, which actually varies with consumption."
        ),
    ),
    "West Bengal — WBSEDCL Domestic (Urban, DM-U)": dict(
        code="WB", region="Highest 5",
        slabs=[(34, 5.04), (60, 6.33), (100, 7.12), (150, 7.52), (300, 7.69), (INF, 9.22)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.00,
        duty_pct=10.0, duty_on_fixed=True, duty_on_meter=False,
        fixed_options={}, fixed_per_kw_month=30.0, meter_rent_month=0.0,
        export_rate_default=2.82, has_tod=False, verified=False,
        source="WBERC schedule FY 2025-26 (secondary source). RATES verified; SLAB BOUNDARIES are an estimate.",
        notes=(
            "SLAB BOUNDARIES ARE NOT VERIFIED. The six RATES (5.04 / 6.33 / 7.12 / 7.52 / "
            "7.69 / 9.22) come from the source, but only the first band (34 units) and the "
            "last (above 300) are published. The four middle boundaries here are an "
            "estimate of the usual WBERC structure and WILL be wrong for some consumption "
            "levels. Do not bill a customer or quote a saving from this schedule until the "
            "WBERC order itself is obtained. The 10% duty applies to energy + fixed + fuel "
            "surcharge together, the same base as Maharashtra. MVCA (the monthly variable "
            "cost adjustment) is revised monthly by WBERC and is NOT included — set it via "
            "the FAC field from the customer's own bill. Fixed charge is ₹30 per kVA per "
            "month, entered here against sanctioned load in kW, which is a simplification."
        ),
    ),
    "Chhattisgarh — CSPDCL LV-1 Domestic": dict(
        code="CG", region="Highest 5",
        slabs=[(100, 4.40), (200, 4.50), (400, 6.00), (600, 7.00), (INF, 8.80)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.10,
        duty_pct=11.0, duty_on_fixed=False, duty_on_meter=False,
        fixed_options={}, fixed_per_kw_month=20.0, meter_rent_month=0.0,
        export_rate_default=2.82, has_tod=False, verified=True,
        source="CSERC tariff order FY 2026-27, in force 01-Jul-2026 (secondary source).",
        notes=(
            "Duty is 11% of ENERGY CHARGE PLUS FPPA ONLY — it does NOT touch the fixed "
            "charge, unlike Maharashtra and West Bengal. A separate duty cess of ₹0.10/unit "
            "sits on top and is carried in levy_per_kwh. "
            "TWO TRAPS: (1) FPPA has been running above 12% (+12.52% on September 2026 "
            "bills); at that level it moves the marginal rate by about a rupee, so set it "
            "in the FAC field rather than leaving it at zero. (2) The fixed charge is also "
            "a MINIMUM PAYABLE at zero consumption — a net-zero solar household still pays "
            "it, so it can never be modelled as avoidable. "
            "The first two slabs are nearly flat (4.40 then 4.50), so the telescopic effect "
            "only begins past 200 units. ₹20/kW applies to the first 5 kW; 5–10 kW is ₹30 "
            "and above 10 kW is ₹50, not modelled here."
        ),
    ),
    # ================= LOWEST-TARIFF STATES =================
    "Goa — Goa Electricity Dept. LTDS-II Domestic": dict(
        code="GA", region="Lowest 5",
        slabs=[(100, 2.10), (200, 3.10), (300, 4.15), (400, 5.45), (INF, 6.60)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.25,
        duty_pct=0.0, duty_on_fixed=False, duty_on_meter=False,
        fixed_options={}, fixed_per_kw_month=25.0, meter_rent_month=0.0,
        export_rate_default=2.82, has_tod=False, verified=True,
        source="JERC schedule FY 2026-27, Goa Electricity Department (secondary source).",
        notes=(
            "levy_per_kwh of ₹0.25 is Electricity Duty ₹0.20 plus Public Lighting Duty "
            "₹0.05, both flat per unit — there is no percentage duty. FPPA was +12.82% on "
            "August 2026 bills (down from +19.24% in July), large enough that it must be "
            "set in the FAC field. Lifeline LTDS-I (₹1.50 flat, ₹20/month) applies only "
            "below 250 W load and 50 units, and let-out portions and homestays fall under "
            "LTDS-III (₹3.15–6.70) — neither is modelled here."
        ),
    ),
    "Himachal Pradesh — HPSEBL Domestic": dict(
        code="HP", region="Lowest 5",
        slabs=[(125, 5.44), (INF, 5.89)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.10,
        duty_pct=3.0, duty_on_fixed=False, duty_on_meter=False,
        fixed_options={"Ordinary domestic": 85.0, "Lifeline / tribal & difficult areas": 55.0},
        fixed_per_kw_month=0.0, meter_rent_month=0.0,
        export_rate_default=2.82, has_tod=False, verified=True,
        source="HPERC order FY 2026-27, in force 01-Apr-2026 (secondary source).",
        notes=(
            "Only two slabs, 45 paise apart — this tariff is FLAT rather than cheap, so "
            "there is almost no slab-displacement saving to capture here. Duty is 3% of "
            "the ENERGY CHARGE ONLY. The ₹0.10/unit in levy_per_kwh is the milk cess. "
            "FPPA was ₹0.34/unit on June 2026 bills. "
            "Worth knowing when reading the market map: HPSEBL reports roughly 3,000 rural "
            "outage hours a year against 20–55 urban, a ratio near 100x that looks more "
            "like a reporting-method difference than a physical one."
        ),
    ),
    "Jammu & Kashmir — JPDCL / KPDCL Domestic (metered)": dict(
        code="JK", region="Lowest 5",
        slabs=[(200, 2.30), (400, 4.00), (INF, 4.35)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.00,
        duty_pct=0.0, duty_on_fixed=False, duty_on_meter=False,
        fixed_options={}, fixed_per_kw_month=8.0, meter_rent_month=0.0,
        export_rate_default=2.82, has_tod=False, verified=False,
        source="JERC for the UT of J&K and Ladakh, JPDCL & KPDCL ARR and Tariff FY 2025-26 (secondary source).",
        notes=(
            "DUTY IS SET TO ZERO BECAUSE NO FIGURE IS PUBLISHED, not because none is "
            "levied — the source states duty applies 'in addition' without quantifying it. "
            "Every rate shown here is therefore an UNDERSTATEMENT of the true marginal "
            "cost. Fixed charges are rounded up to the next half kW on a real bill, which "
            "this does not do. A BPL schedule (₹1.40 for the first 30 units) and a "
            "per-house slab scaling rule for group housing on one connection are not "
            "modelled."
        ),
    ),
    "Sikkim — Energy & Power Dept. Domestic": dict(
        code="SK", region="Lowest 5",
        slabs=[(50, 3.40), (150, 5.46), (INF, 6.16)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.00,
        duty_pct=0.0, duty_on_fixed=False, duty_on_meter=False,
        fixed_options={}, fixed_per_kw_month=0.0, meter_rent_month=45.0,
        export_rate_default=2.82, has_tod=False, verified=False,
        source="Sikkim Energy & Power Department FY 2025-26 (secondary source, internally inconsistent).",
        notes=(
            "TREAT THIS SCHEDULE AS UNKNOWN RATHER THAN CHEAP. The published source gives "
            "a 301–400 rate LOWER than its 151–300 rate, which no telescopic schedule "
            "would do, and publishes no slab at all above 400 units. The rate above 150 "
            "units here simply carries the 151–300 figure forward, which is an assumption, "
            "not a source. A separate headline range of ₹1.60–4.60 quoted elsewhere does "
            "not reconcile with these figures either. No duty, FPPA or meter rent is "
            "published; the ₹45 shown is the mid-point of the ₹40–50 customer charge. "
            "Do not quote any number from this state without the regulator's own order."
        ),
    ),
    "Dadra & Nagar Haveli and Daman & Diu — DNHDDPDCL LTDS-II": dict(
        code="DD", region="Lowest 5",
        slabs=[(100, 1.85), (200, 2.50), (300, 3.15), (400, 3.25), (INF, 3.80)],
        wheeling_per_kwh=0.00, fac_per_kwh=0.00, levy_per_kwh=0.00,
        duty_pct=0.0, duty_on_fixed=False, duty_on_meter=False,
        fixed_options={}, fixed_per_kw_month=10.0, meter_rent_month=0.0,
        export_rate_default=2.82, has_tod=False, verified=True,
        source="JERC Table 10-1, DNHDDPDCL LTDS-II, in force 01-Apr-2026 (secondary source).",
        notes=(
            "The cheapest top slab in the country and the flattest ladder — just 2.05x "
            "from the first slab to the last, against 4.4x in Maharashtra. An avoided unit "
            "here is worth ₹3.80 against ₹22.19 in Maharashtra, a factor of 5.8. Duty is "
            "not published and is set to zero, so the rates shown are if anything an "
            "understatement. FPPA applies but no verified rate is published."
        ),
    ),
}

DEFAULT_STATE = "Maharashtra — MSEDCL LT-I(B) Residential"


# =====================================================================
# TARIFF OBJECT
# =====================================================================
@dataclass
class Tariff:
    """A fully resolved tariff for one site, with user overrides applied."""

    name: str
    slabs: list[tuple[float, float]]
    wheeling_per_kwh: float
    fac_per_kwh: float
    duty_pct: float
    duty_on_fixed: bool
    duty_on_meter: bool
    fixed_month: float            # flat Rs/month component (MH style)
    fixed_per_kw_month: float     # Rs/kW/month component (UP style)
    sanctioned_kw: float
    meter_rent_month: float
    export_rate: float
    settlement: str               # "net_metering" | "net_billing" | "no_export"
    # Flat per-unit levies that are NOT a percentage of anything: Andhra's
    # ₹0.06/unit duty, Chhattisgarh's ₹0.10 cess, Goa's ₹0.20 duty + ₹0.05
    # public-lighting duty, Himachal's ₹0.10 milk cess. They scale with units,
    # so they belong in the marginal rate; they are not a percentage, so they
    # sit OUTSIDE the duty gross-up rather than inside it.
    levy_per_kwh: float = 0.0
    # False where the schedule itself is not trustworthy — unpublished slab
    # boundaries (West Bengal), unquantified duty (J&K), or internally
    # contradictory published rates (Sikkim). The UI warns on these.
    verified: bool = True
    region: str = ""
    source: str = ""
    notes: str = ""
    # Optional residential ToD (off by default — neither MSEDCL LT-I(B) nor
    # UPPCL LMV-1 applies a mandatory residential ToD at present).
    tod_on: bool = False
    tod_start_h: float = 18.0
    tod_end_h: float = 22.0
    tod_pct: float = 0.0

    # ---------- marginal (savings) side ----------
    def slab_rate(self, cum_units: float) -> float:
        """Energy-charge rate (Rs/kWh) applying to the NEXT unit."""
        c = max(cum_units, 0.0)
        for cap, rate in self.slabs:
            if c < cap - 1e-9:
                return rate
        return self.slabs[-1][1]

    def marginal_rate(self, cum_units: float, hour_of_day: float | None = None) -> float:
        """
        All-in avoided cost of the NEXT kWh, Rs/kWh.

            (energy slab + wheeling + FAC) x (1 + duty%)

        Fixed charge / meter rent / per-kW demand charge are deliberately
        EXCLUDED — see the module docstring. This is the only rate that should
        ever multiply an avoided-kWh figure.
        """
        energy = self.slab_rate(cum_units)
        if self.tod_on and hour_of_day is not None and _in_window(
            hour_of_day, self.tod_start_h, self.tod_end_h
        ):
            energy *= 1.0 + self.tod_pct / 100.0
        variable = energy + self.wheeling_per_kwh + self.fac_per_kwh
        return variable * (1.0 + self.duty_pct / 100.0) + self.levy_per_kwh

    def avg_marginal_rate_for_block(self, start_units: float, kwh: float) -> float:
        """Average all-in rate across a block of kWh starting at `start_units`
        — used when a lump quantity crosses a slab boundary."""
        if kwh <= 1e-12:
            return self.marginal_rate(start_units)
        return self.energy_cost_block(start_units, kwh) / kwh

    def energy_cost_block(self, start_units: float, kwh: float) -> float:
        """Rupee cost of consuming `kwh` starting from `start_units` already
        consumed this cycle, marginal components only."""
        if kwh <= 0:
            return 0.0
        remaining, cost, cursor = kwh, 0.0, max(start_units, 0.0)
        prev_cap = 0.0
        for cap, rate in self.slabs:
            width_left = cap - max(cursor, prev_cap)
            prev_cap = cap
            if width_left <= 0:
                continue
            take = min(remaining, width_left)
            cost += take * (rate + self.wheeling_per_kwh + self.fac_per_kwh)
            remaining -= take
            cursor += take
            if remaining <= 1e-12:
                break
        if remaining > 1e-12:  # beyond the last finite cap
            last = self.slabs[-1][1]
            cost += remaining * (last + self.wheeling_per_kwh + self.fac_per_kwh)
        return cost * (1.0 + self.duty_pct / 100.0) + kwh * self.levy_per_kwh

    # ---------- bill side ----------
    def telescopic_energy_charge(self, units: float) -> float:
        """Pure energy charge for `units`, telescopic, no wheeling/duty."""
        if units <= 0:
            return 0.0
        remaining, charge, prev_cap = units, 0.0, 0.0
        for cap, rate in self.slabs:
            width = cap - prev_cap
            take = min(remaining, width)
            if take > 0:
                charge += take * rate
                remaining -= take
            prev_cap = cap
            if remaining <= 1e-12:
                break
        if remaining > 1e-12:
            charge += remaining * self.slabs[-1][1]
        return charge

    def build_bill(self, units: float, n_months: float = 1.0) -> dict:
        """
        Full consumer bill build-up for `units` consumed over `n_months`
        billing periods. Returns every line item so the total is auditable
        against a real bill.

        NOTE on slabs vs multi-month: slabs are a PER-BILLING-CYCLE concept.
        This function assumes `units` belongs to ONE cycle. For a multi-month
        view, call it per cycle and sum the totals — never sum the units and
        call it once, which would push the consumer into a higher slab than
        they ever actually occupied.
        """
        units = max(units, 0.0)
        energy = self.telescopic_energy_charge(units)
        wheeling = units * self.wheeling_per_kwh
        fac = units * self.fac_per_kwh
        fixed = (self.fixed_month + self.fixed_per_kw_month * self.sanctioned_kw) * n_months
        meter = self.meter_rent_month * n_months

        duty_base = (energy + wheeling + fac
                     + (fixed if self.duty_on_fixed else 0.0)
                     + (meter if self.duty_on_meter else 0.0))
        duty = duty_base * self.duty_pct / 100.0
        levy = units * self.levy_per_kwh
        subtotal = energy + wheeling + fac + fixed + meter
        return dict(
            units=units,
            energy_charge=energy,
            wheeling_charge=wheeling,
            fac_charge=fac,
            fixed_charge=fixed,
            meter_rent=meter,
            subtotal=subtotal,
            electricity_duty=duty,
            flat_levy=levy,
            total=subtotal + duty + levy,
            effective_rate=(subtotal + duty + levy) / units if units > 0 else 0.0,
        )


def _in_window(h: float, start: float, end: float) -> bool:
    """Hour-of-day window membership, handling windows that wrap midnight."""
    if start <= end:
        return start <= h < end
    return h >= start or h < end


def build_tariff(
    state_name: str,
    *,
    sanctioned_kw: float = 5.0,
    fixed_option: str | None = None,
    settlement: str = "net_metering",
    export_rate: float | None = None,
    fac_per_kwh: float | None = None,
    duty_pct: float | None = None,
    duty_on_fixed: bool | None = None,
    duty_on_meter: bool | None = None,
    slab_override: list[tuple[float, float]] | None = None,
    wheeling_override: float | None = None,
    levy_override: float | None = None,
    tod_on: bool = False,
    tod_start_h: float = 18.0,
    tod_end_h: float = 22.0,
    tod_pct: float = 0.0,
) -> Tariff:
    spec = STATE_TARIFFS[state_name]
    fixed_month = 0.0
    if spec["fixed_options"]:
        key = fixed_option or next(iter(spec["fixed_options"]))
        fixed_month = spec["fixed_options"][key]
    return Tariff(
        name=state_name,
        slabs=slab_override or list(spec["slabs"]),
        wheeling_per_kwh=(
            spec["wheeling_per_kwh"] if wheeling_override is None else wheeling_override
        ),
        fac_per_kwh=spec["fac_per_kwh"] if fac_per_kwh is None else fac_per_kwh,
        duty_pct=spec["duty_pct"] if duty_pct is None else duty_pct,
        duty_on_fixed=(spec["duty_on_fixed"] if duty_on_fixed is None else duty_on_fixed),
        duty_on_meter=(spec["duty_on_meter"] if duty_on_meter is None else duty_on_meter),
        fixed_month=fixed_month,
        fixed_per_kw_month=spec["fixed_per_kw_month"],
        sanctioned_kw=sanctioned_kw,
        meter_rent_month=spec["meter_rent_month"],
        export_rate=(
            spec["export_rate_default"] if export_rate is None else export_rate
        ),
        settlement=settlement,
        levy_per_kwh=(spec.get("levy_per_kwh", 0.0) if levy_override is None else levy_override),
        verified=spec.get("verified", True),
        region=spec.get("region", ""),
        source=spec["source"],
        notes=spec["notes"],
        tod_on=tod_on,
        tod_start_h=tod_start_h,
        tod_end_h=tod_end_h,
        tod_pct=tod_pct,
    )