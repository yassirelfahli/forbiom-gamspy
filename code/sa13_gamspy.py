"""
GAMSPy translation of the FORBIOM-ON model in code/FORBIOM_ON.gms.

Economic flows are million m3, prices and costs are $/m3, and the objective
is million dollars. Prices stay on the exogenous 3 percent path. Random draws
use the GAMS uniform generator with seed 3141, in the same order as the GAMS
loops (harvest-cost shocks, then demand shocks).

Arc, mill, and demand inequalities are enforced as variable bounds. They are
the same constraints as arcCapEq, millCapacityEq, and demandCapEq. Writing
them as bounds keeps the generated model within the GAMSPy demonstration
license (2,000 rows and 2,000 columns) without relaxing the program.

Set GAMSPY_LICENSE to a full GAMS license file if one is available. Otherwise
the script uses the license bundled with gamspy_base.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import gamspy as gp
from gamspy import (
    Alias,
    Container,
    Domain,
    Equation,
    Model,
    Options,
    Ord,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
)
from gamspy.math import uniform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
GAMS_SUMMARY = ROOT / "results" / "gams_summary_public_table.csv"


def resolve_license() -> str | None:
    """Prefer GAMSPY_LICENSE, then the license shipped with gamspy_base."""
    env = os.environ.get("GAMSPY_LICENSE")
    if env and Path(env).is_file():
        return env
    try:
        import gamspy_base
    except ImportError:
        return None
    bundled = Path(gamspy_base.__file__).resolve().parent / "gamslice.txt"
    return str(bundled) if bundled.is_file() else None
SEED = 3141

REGIONS = ["TroutLake", "Abitibi", "Algoma", "Kenora", "Nipissing", "Algonquin"]
PRODUCTS = ["CLT", "Glulam", "Lumber", "Pulpwood"]
YEARS = [str(y) for y in range(2020, 2031)]
TREES = [
    "Spruce",
    "JackPine",
    "WhiteBirch",
    "TolerantHdwds",
    "WhitePine",
    "RedPine",
    "Aspen",
]
SITES = ["SC1", "SC2", "SC3"]

SPECIES_MIX = {
    "TroutLake": dict(Spruce=0.50, JackPine=0.30, WhiteBirch=0.10, TolerantHdwds=0.00, WhitePine=0.00, RedPine=0.00, Aspen=0.10),
    "Abitibi": dict(Spruce=0.60, JackPine=0.20, WhiteBirch=0.10, TolerantHdwds=0.00, WhitePine=0.00, RedPine=0.00, Aspen=0.10),
    "Kenora": dict(Spruce=0.40, JackPine=0.40, WhiteBirch=0.10, TolerantHdwds=0.00, WhitePine=0.00, RedPine=0.05, Aspen=0.05),
    "Algoma": dict(Spruce=0.10, JackPine=0.05, WhiteBirch=0.10, TolerantHdwds=0.40, WhitePine=0.30, RedPine=0.05, Aspen=0.00),
    "Nipissing": dict(Spruce=0.10, JackPine=0.05, WhiteBirch=0.10, TolerantHdwds=0.30, WhitePine=0.40, RedPine=0.05, Aspen=0.00),
    "Algonquin": dict(Spruce=0.05, JackPine=0.00, WhiteBirch=0.10, TolerantHdwds=0.50, WhitePine=0.30, RedPine=0.05, Aspen=0.00),
}
INITIAL_STOCK = {
    "TroutLake": 28.99948,
    "Abitibi": 107.77522,
    "Algoma": 54.88489,
    "Kenora": 49.74619,
    "Nipissing": 45.26169,
    "Algonquin": 36.72526,
}
REGION_SITE = {
    "TroutLake": "SC2",
    "Abitibi": "SC2",
    "Algoma": "SC1",
    "Kenora": "SC3",
    "Nipissing": "SC1",
    "Algonquin": "SC1",
}
TRANSPORT = {
    ("TroutLake", "Abitibi"): 38, ("TroutLake", "Algoma"): 42, ("TroutLake", "Kenora"): 100,
    ("TroutLake", "Nipissing"): 14, ("TroutLake", "Algonquin"): 9,
    ("Abitibi", "TroutLake"): 26, ("Abitibi", "Algoma"): 35, ("Abitibi", "Kenora"): 75,
    ("Abitibi", "Nipissing"): 25, ("Abitibi", "Algonquin"): 32,
    ("Algoma", "TroutLake"): 31, ("Algoma", "Abitibi"): 28, ("Algoma", "Kenora"): 60,
    ("Algoma", "Nipissing"): 31, ("Algoma", "Algonquin"): 38,
    ("Kenora", "TroutLake"): 79, ("Kenora", "Abitibi"): 61, ("Kenora", "Algoma"): 49,
    ("Kenora", "Nipissing"): 79, ("Kenora", "Algonquin"): 87,
    ("Nipissing", "TroutLake"): 14, ("Nipissing", "Abitibi"): 27, ("Nipissing", "Algoma"): 31,
    ("Nipissing", "Kenora"): 79, ("Nipissing", "Algonquin"): 8,
    ("Algonquin", "TroutLake"): 9, ("Algonquin", "Abitibi"): 34, ("Algonquin", "Algoma"): 38,
    ("Algonquin", "Kenora"): 87, ("Algonquin", "Nipissing"): 8,
}

YIELD_FACTOR = {"CLT": 0.82, "Glulam": 0.85, "Lumber": 0.90, "Pulpwood": 0.98}
LUMBER_INPUT = {"CLT": 1.20, "Glulam": 1.10, "Lumber": 0.0, "Pulpwood": 0.0}
BASE_PRICE = {"CLT": 700.0, "Glulam": 575.0, "Lumber": 375.0, "Pulpwood": 75.0}
TIER = {"CLT": 2, "Glulam": 2, "Lumber": 1, "Pulpwood": 1}
MARKUP = {"CLT": 150.0, "Glulam": 120.0, "Lumber": 20.0, "Pulpwood": -80.0}
BASE_PROC = {1: 100.0, 2: 50.0}
EMISSION_FACTOR = {"CLT": 0.15, "Glulam": 0.17, "Lumber": 0.065, "Pulpwood": 0.01}
STORAGE_FACTOR = {"CLT": 0.7, "Glulam": 0.7, "Lumber": 0.4, "Pulpwood": 0.07}
MILL_CAP = {"Lumber": 10.0, "Pulpwood": 10.0, "CLT": 2.5, "Glulam": 2.5}
ARC_CAP = 20.0
HARVEST_COST = 38.0
CARBON_PRICE0 = 170.0
MAX_HARVEST_RATE = 0.01

CP_VALUE = {
    "cp0": 0, "cp50": 50, "cp100": 100, "cp170": 170,
    "cp250": 250, "cp350": 350, "cp500": 500,
}
CP_ORDER = ["cp0", "cp50", "cp100", "cp170", "cp250", "cp350", "cp500"]
DEM_ORDER = ["Baseline", "D50", "D100", "D200", "D300", "D400", "DU10"]
DEM_CLT = {
    "Baseline": 1.0, "D50": 1.5, "D100": 2.0, "D200": 3.0,
    "D300": 4.0, "D400": 5.0, "DU10": 1.1,
}
PR_ORDER = ["Pbase", "PD5", "PI5", "PD10", "PI10"]
PR_LUMBER = {"Pbase": 1.00, "PD5": 0.95, "PI5": 1.05, "PD10": 0.90, "PI10": 1.10}
CM_ORDER = ["CM_D100_PD5", "CM_D100_PI10", "CM_D200_PD10"]
CM_CLT = {"CM_D100_PD5": 2.0, "CM_D100_PI10": 2.0, "CM_D200_PD10": 3.0}
CM_LUMBER = {"CM_D100_PD5": 0.95, "CM_D100_PI10": 1.10, "CM_D200_PD10": 0.90}

METRICS = [
    "Obj", "Zprofit", "Zemission", "CarbonStorage", "NetCarbon", "NetCarbDel",
    "NetCarbPct", "CarbonCost", "Revenue", "HarvestCost", "ProcCost", "TransCost",
    "TotalHarvest", "TotalProduction", "TotalSales",
]


def build_yield() -> dict[tuple[str, str, int], float]:
    """Plonski gross merchantable volume (m3/ha), including the GAMS proxies."""
    raw = {
        "WhitePine": {30: 76, 40: 157, 50: 242, 60: 315, 70: 372, 80: 414},
        "JackPine": {30: 89, 40: 155, 50: 196, 60: 223, 70: 238, 80: 243},
        "TolerantHdwds": {30: 36, 40: 68, 50: 100, 60: 130, 70: 158, 80: 185},
    }
    yld: dict[tuple[str, str, int], float] = {}
    for tree, curve in raw.items():
        for age, vol in curve.items():
            yld[(tree, "SC1", age)] = float(vol)
    for tree in TREES:
        for age in range(0, 151):
            sc1 = yld.get((tree, "SC1", age), 0.0)
            yld[(tree, "SC2", age)] = sc1 * 0.80
            yld[(tree, "SC3", age)] = sc1 * 0.60
    for site in SITES:
        for age in range(0, 151):
            yld[("RedPine", site, age)] = yld.get(("WhitePine", site, age), 0.0)
            yld[("Spruce", site, age)] = yld.get(("JackPine", site, age), 0.0)
            yld[("Aspen", site, age)] = yld.get(("JackPine", site, age), 0.0) * 0.9
            yld[("WhiteBirch", site, age)] = yld.get(("JackPine", site, age), 0.0) * 0.8
    return yld


def growth_shock(ir: int, iy: int) -> float:
    """GAMS: 0.88 + 0.24 * mod(ord(r)*31 + ord(y)*17 + 7, 97) / 97."""
    return 0.88 + 0.24 * (((ir + 1) * 31 + (iy + 1) * 17 + 7) % 97) / 97.0


def prepare_data(demand_shock: pd.DataFrame, harvest_shock: pd.DataFrame) -> dict:
    yld = build_yield()
    area = {}
    cai = {}
    for r in REGIONS:
        site = REGION_SITE[r]
        vol30 = sum(SPECIES_MIX[r][t] * yld[(t, site, 30)] for t in TREES)
        area[r] = INITIAL_STOCK[r] * 1_000_000.0 / vol30
        vol_step = sum(
            SPECIES_MIX[r][t] * (yld[(t, site, 40)] - yld[(t, site, 30)]) for t in TREES
        )
        cai[r] = vol_step / 10.0 * area[r] / 1_000_000.0

    growth_rows = []
    for ir, r in enumerate(REGIONS):
        for iy, y in enumerate(YEARS):
            growth_rows.append((r, y, cai[r] * growth_shock(ir, iy)))

    sust = {r: MAX_HARVEST_RATE * INITIAL_STOCK[r] for r in REGIONS}
    base_demand = {r: {} for r in REGIONS}
    for r in REGIONS:
        base_demand[r]["Lumber"] = 0.20 * sust[r] * YIELD_FACTOR["Lumber"]
        base_demand[r]["Pulpwood"] = 0.08 * sust[r] * YIELD_FACTOR["Pulpwood"]
        base_demand[r]["CLT"] = (
            0.035 * sust[r] * YIELD_FACTOR["Lumber"] / LUMBER_INPUT["CLT"]
        )
        base_demand[r]["Glulam"] = (
            0.030 * sust[r] * YIELD_FACTOR["Lumber"] / LUMBER_INPUT["Glulam"]
        )

    dshock = demand_shock.set_index(["r", "p", "y"])["value"]
    hshock = harvest_shock.set_index(["r", "y"])["value"]

    demand_rows = []
    price_rows = []
    proc_rows = []
    hcost_rows = []
    for iy, y in enumerate(YEARS):
        price_factor = 1.03 ** iy
        proc_factor = 0.98 ** iy
        for p in PRODUCTS:
            price_rows.append((p, y, BASE_PRICE[p] * price_factor))
            proc_rows.append((p, y, (BASE_PROC[TIER[p]] + MARKUP[p]) * proc_factor))
        for r in REGIONS:
            hcost_rows.append((r, y, HARVEST_COST * float(hshock.loc[(r, y)])))
            for p in PRODUCTS:
                demand_rows.append(
                    (r, p, y, base_demand[r][p] * float(dshock.loc[(r, p, y)]))
                )

    return {
        "sust": sust,
        "base_demand": base_demand,
        "growth_rows": growth_rows,
        "demand_rows": demand_rows,
        "price_rows": price_rows,
        "proc_rows": proc_rows,
        "hcost_rows": hcost_rows,
    }


def _solve(model: Model, label: str) -> None:
    model.solve(solver="CPLEX")
    status = str(model.status)
    if "Optimal" not in status:
        raise RuntimeError(f"{label} finished with status {status}")
    print(f"  {label:22s}  obj = {model.objective_value:,.4f}   {status}")


def _sync_sales_cap(sales: Variable, demand_cap: Parameter, r, p, y) -> None:
    sales.up[r, p, y] = demand_cap[r, p, y]


def _metrics(s: dict) -> dict[str, float]:
    return {
        "Obj": float(s["obj"].toValue()),
        "Zprofit": float(s["z_profit"].toValue()),
        "Zemission": float(s["z_emission"].toValue()),
        "CarbonStorage": float(s["z_storage"].toValue()),
        "NetCarbon": float(s["z_net"].toValue()),
        "CarbonCost": float(s["carbon_cost"].toValue()),
        "Revenue": float(s["revenue"].toValue()),
        "HarvestCost": float(s["harvest_cost_tot"].toValue()),
        "ProcCost": float(s["proc_cost_tot"].toValue()),
        "TransCost": float(s["trans_cost_tot"].toValue()),
        "TotalHarvest": float(s["harvest_rw"].records["level"].sum()),
        "TotalProduction": float(s["production"].records["level"].sum()),
        "TotalSales": float(s["sales"].records["level"].sum()),
    }


def _mt_tables(s: dict) -> tuple[pd.Series, pd.Series]:
    """Mass-timber quantity (1000 m3) and CLT+Glulam sales revenue (million $)."""
    prod = s["production"].records
    mt = prod[prod["p"].isin(["CLT", "Glulam"])]
    qty = mt.groupby("r", sort=False)["level"].sum() * 1000.0

    sales = s["sales"].records
    price = s["base_price_y"].records.rename(columns={"value": "price"})
    merged = sales.merge(price, on=["p", "y"], how="left")
    merged = merged[merged["p"].isin(["CLT", "Glulam"])]
    merged["rev"] = merged["price"] * merged["level"]
    rev = merged.groupby("r", sort=False)["rev"].sum()
    qty = qty.reindex(REGIONS).fillna(0.0)
    rev = rev.reindex(REGIONS).fillna(0.0)
    return qty, rev


def _add_net_carbon(summary: dict[str, dict[str, float]], reference: str) -> None:
    base = summary[reference]["NetCarbon"]
    for name, row in summary.items():
        delta = row["NetCarbon"] - base
        row["NetCarbDel"] = delta
        row["NetCarbPct"] = 100.0 * delta / abs(base) if abs(base) > 1e-6 else 0.0


def _write_region_report(path: Path, title: str, order: list[str], rows: dict, extra) -> None:
    lines = [title, ""]
    for name in order:
        qty, rev, obj = rows[name]
        lines.append(f"Scenario: {name}")
        lines.append(extra(name, qty, rev, obj))
        for reg in REGIONS:
            lines.append(
                f"  {reg:12s} Qty {qty[reg]:12.2f} Rev {rev[reg]:12.4f}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_scenarios(s: dict) -> None:
    r, p, y = s["r"], s["p"], s["y"]
    demand_cap = s["demand_cap"]
    demand_base = s["demand_cap_base"]
    price = s["base_price_y"]
    price_path = s["base_price_path"]
    carbon_price = s["carbon_price"]

    def reset_prices() -> None:
        price[p, y] = price_path[p, y]

    def set_demand_mult(mult_clt: float) -> None:
        # Lumber pull share is 0, so only CLT and Glulam caps move.
        demand_cap[r, p, y] = demand_base[r, p, y]
        demand_cap[r, "CLT", y] = demand_base[r, "CLT", y] * mult_clt
        demand_cap[r, "Glulam", y] = demand_base[r, "Glulam", y] * mult_clt
        _sync_sales_cap(s["sales"], demand_cap, r, p, y)

    def set_lumber_mult(mult: float) -> None:
        price[p, y] = price_path[p, y]
        price["Lumber", y] = price_path["Lumber", y] * mult

    print("=== Baseline solve (carbon price 170, harvest ceiling on) ===")
    _solve(s["model_on"], "baseline")
    base_metrics = _metrics(s)
    print(
        "  Zprofit {Zprofit:.4f}  emissions {Zemission:.4f}  "
        "storage {CarbonStorage:.4f}  harvest {TotalHarvest:.4f}".format(**base_metrics)
    )

    print("=== Demand sensitivity (harvest ceiling off) ===")
    dem_summary: dict[str, dict[str, float]] = {}
    dem_rows = {}
    harvest_rows = []
    for name in DEM_ORDER:
        set_demand_mult(DEM_CLT[name])
        _solve(s["model_off"], name)
        dem_summary[name] = _metrics(s)
        qty, rev = _mt_tables(s)
        dem_rows[name] = (qty, rev, dem_summary[name]["Obj"])
        harv = s["harvest_rw"].records
        for _, row in harv.iterrows():
            harvest_rows.append(
                (name, DEM_CLT[name], row["r"], row["y"], float(row["level"]))
            )
    _add_net_carbon(dem_summary, "Baseline")

    print("=== Restore baseline and run lumber-price sensitivity ===")
    set_demand_mult(1.0)
    reset_prices()
    carbon_price.setRecords(CARBON_PRICE0)
    _solve(s["model_on"], "restore")

    pr_summary = {}
    pr_rows = {}
    for name in PR_ORDER:
        set_demand_mult(1.0)
        set_lumber_mult(PR_LUMBER[name])
        _solve(s["model_on"], name)
        pr_summary[name] = _metrics(s)
        qty, rev = _mt_tables(s)
        pr_rows[name] = (qty, rev, pr_summary[name]["Obj"])

    print("=== Coupled demand and lumber-price sensitivity ===")
    cm_summary = {}
    cm_rows = {}
    for name in CM_ORDER:
        set_demand_mult(CM_CLT[name])
        set_lumber_mult(CM_LUMBER[name])
        _solve(s["model_on"], name)
        cm_summary[name] = _metrics(s)
        qty, rev = _mt_tables(s)
        cm_rows[name] = (qty, rev, cm_summary[name]["Obj"])

    print("=== Carbon-price sensitivity ===")
    set_demand_mult(1.0)
    reset_prices()
    cp_summary: dict[str, dict[str, float]] = {}
    for name in CP_ORDER:
        carbon_price.setRecords(float(CP_VALUE[name]))
        _solve(s["model_on"], name)
        cp_summary[name] = _metrics(s)
    _add_net_carbon(cp_summary, "cp170")

    write_outputs(dem_summary, dem_rows, pr_rows, cm_rows, cp_summary, harvest_rows)
    compare_with_gams(cp_summary, dem_summary)


def write_outputs(dem_summary, dem_rows, pr_rows, cm_rows, cp_summary, harvest_rows) -> None:
    pub_lines = [
        "Block,Scenario,Obj_MUSD,Zemission_Mt,CarbonStorage_Mt,NetCarbon_Mt,"
        "NetCarbDelta_Mt,NetCarbPct_vs_ref,TotalHarvest_Mm3"
    ]
    for name in CP_ORDER:
        row = cp_summary[name]
        pub_lines.append(
            "Carbon,{name},{Obj:.4f},{Zemission:.4f},{CarbonStorage:.4f},"
            "{NetCarbon:.4f},{NetCarbDel:.4f},{NetCarbPct:.4f},{TotalHarvest:.4f}".format(
                name=name, **row
            )
        )
    for name in DEM_ORDER:
        row = dem_summary[name]
        pub_lines.append(
            "Demand,{name},{Obj:.4f},{Zemission:.4f},{CarbonStorage:.4f},"
            "{NetCarbon:.4f},{NetCarbDel:.4f},{NetCarbPct:.4f},{TotalHarvest:.4f}".format(
                name=name, **row
            )
        )
    (OUT / "summary_public_table.csv").write_text("\n".join(pub_lines) + "\n", encoding="utf-8")

    def dem_extra(name, qty, rev, obj):
        return (
            f"  CLT mult {DEM_CLT[name]:8.4f}  Obj {obj:12.4f}  "
            f"TotQty(1000m3) {qty.sum():12.2f}  TotRev(M$) {rev.sum():12.4f}"
        )

    def pr_extra(name, qty, rev, obj):
        return (
            f"  LumberMult {PR_LUMBER[name]:8.4f}  Obj {obj:12.4f}  "
            f"TotQty(1000m3) {qty.sum():12.2f}  TotRev(M$) {rev.sum():12.4f}"
        )

    def cm_extra(name, qty, rev, obj):
        return (
            f"  CLTdemMult {CM_CLT[name]:8.4f}  LumberMult {CM_LUMBER[name]:8.4f}  "
            f"Obj {obj:12.4f}  TotQty {qty.sum():12.2f}  TotRev {rev.sum():12.4f}"
        )

    _write_region_report(
        OUT / "demand_sensitivity_report.txt",
        "Demand sensitivity - mass timber (CLT+Glulam), 6 regions, sum over years\n"
        "Qty = 1000 m3 ; Rev = MILLION $",
        DEM_ORDER,
        dem_rows,
        dem_extra,
    )
    _write_region_report(
        OUT / "price_sensitivity_report.txt",
        "Lumber price sensitivity - mass timber (CLT+Glulam), 6 regions, sum over years\n"
        "Qty = 1000 m3 ; Rev = MILLION $",
        PR_ORDER,
        pr_rows,
        pr_extra,
    )
    _write_region_report(
        OUT / "coupled_sensitivity_report.txt",
        "Coupled sensitivity - MT demand (CLT+Glulam) x lumber price, 6 regions",
        CM_ORDER,
        cm_rows,
        cm_extra,
    )

    harv = pd.DataFrame(
        harvest_rows, columns=["demScen", "CLT_cap_mult", "Region", "Year", "Harvest_Mm3_per_year"]
    )
    harv.to_csv(OUT / "regional_harvest_by_year_demand_scenarios_long.csv", index=False)
    print(f"Wrote reports under {OUT}")


def compare_with_gams(cp_summary, dem_summary) -> None:
    if not GAMS_SUMMARY.exists():
        print(f"No GAMS summary at {GAMS_SUMMARY}")
        return
    import re

    expected = []
    for line in GAMS_SUMMARY.read_text(encoding="utf-8").splitlines():
        if "Carbon" not in line and "Demand" not in line:
            continue
        if "Block" in line or "_NOTE" in line:
            continue
        block = "Carbon" if "Carbon" in line else "Demand"
        name_match = re.search(
            r"cp\d+|Baseline|D50|D100|D200|D300|D400|DU10", line
        )
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
        if name_match is None or len(nums) < 7:
            continue
        expected.append((block, name_match.group(0), nums[:7]))

    cols = ["Obj", "Zemission", "CarbonStorage", "NetCarbon", "NetCarbDel", "NetCarbPct", "TotalHarvest"]
    print("\nComparison with the GAMS summary (absolute difference)")
    print(f"{'Block':8s} {'Scenario':10s} " + " ".join(f"{c:>12s}" for c in cols))
    worst = 0.0
    for block, name, nums in expected:
        src = cp_summary if block == "Carbon" else dem_summary
        got = [src[name][c] for c in cols]
        diffs = [abs(a - b) for a, b in zip(got, nums)]
        worst = max(worst, max(diffs))
        print(
            f"{block:8s} {name:10s} "
            + " ".join(f"{d:12.4f}" for d in diffs)
        )
    print(f"Largest absolute gap: {worst:.6f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("TEMP", ".")) / "forbiom_gamspy"
    work.mkdir(parents=True, exist_ok=True)

    m = Container(
        options=Options(seed=SEED, license=resolve_license()),
        working_directory=str(work),
    )

    r = Set(m, "r", records=REGIONS, description="Regions")
    p = Set(m, "p", records=PRODUCTS, description="Products")
    y = Set(m, "y", records=YEARS, description="Years")
    rr = Alias(m, "rr", r)

    # Same draw order as the GAMS LOOPs, default execseed 3141.
    harvest_cost_shock = Parameter(m, "harvestCostShock", domain=[r, y])
    demand_shock = Parameter(m, "demandShock", domain=[r, p, y])
    harvest_cost_shock[r, y] = uniform(0.90, 1.10)
    demand_shock[r, p, y] = uniform(0.88, 1.12)

    data = prepare_data(demand_shock.records.copy(), harvest_cost_shock.records.copy())
    print("Sustainable harvest proxy (Mm3/yr):")
    for reg, val in data["sust"].items():
        print(f"  {reg:12s} {val:.6f}")

    growth_rw = Parameter(m, "growthRW", domain=[r, y], records=data["growth_rows"])
    demand_cap_base = Parameter(
        m, "demandCapBase", domain=[r, p, y], records=data["demand_rows"]
    )
    demand_cap = Parameter(
        m, "demandCap_y", domain=[r, p, y], records=data["demand_rows"]
    )
    base_price_path = Parameter(
        m, "basePricePath", domain=[p, y], records=data["price_rows"]
    )
    base_price_y = Parameter(
        m, "basePrice_y", domain=[p, y], records=data["price_rows"]
    )
    processing_cost_y = Parameter(
        m, "processingCost_y", domain=[p, y], records=data["proc_rows"]
    )
    harvest_cost_y = Parameter(
        m, "harvestCost_y", domain=[r, y], records=data["hcost_rows"]
    )
    init_stock = Parameter(
        m, "initStockRW", domain=[r], records=[(reg, INITIAL_STOCK[reg]) for reg in REGIONS]
    )
    max_harvest = Parameter(
        m, "maxHarvestRate", domain=[r], records=[(reg, MAX_HARVEST_RATE) for reg in REGIONS]
    )
    yield_factor = Parameter(
        m, "yieldFactor", domain=[p], records=list(YIELD_FACTOR.items())
    )
    lumber_input = Parameter(
        m, "lumberInputPerUnit", domain=[p], records=list(LUMBER_INPUT.items())
    )
    emission_factor = Parameter(
        m, "carbonEmissionFactor", domain=[p], records=list(EMISSION_FACTOR.items())
    )
    storage_factor = Parameter(
        m, "carbonStorageFactor", domain=[p], records=list(STORAGE_FACTOR.items())
    )
    transport_cost = Parameter(
        m,
        "transportCost",
        domain=[r, rr, p],
        records=[(i, j, prod, cost) for (i, j), cost in TRANSPORT.items() for prod in PRODUCTS],
    )
    transport_ef = Parameter(
        m,
        "transportEmissionFactor",
        domain=[r, rr, p],
        records=[(i, j, prod, 0.018) for (i, j) in TRANSPORT for prod in PRODUCTS],
    )
    mill_cap = Parameter(
        m,
        "millCap",
        domain=[r, p, y],
        records=[(reg, prod, yr, MILL_CAP[prod]) for reg in REGIONS for prod in PRODUCTS for yr in YEARS],
    )
    arc_cap = Parameter(
        m,
        "arcCap",
        domain=[r, rr, p, y],
        records=[
            (i, j, prod, yr, ARC_CAP)
            for (i, j) in TRANSPORT
            for prod in PRODUCTS
            for yr in YEARS
        ],
    )
    carbon_price = Parameter(m, "carbonPrice", records=CARBON_PRICE0)

    # ------------------------------------------------------------------
    # Variables. Indexed carbon accounts are substituted into the scalar
    # emission and storage identities (same algebra as carbonEmitEq /
    # carbonStoreEq summed over r,p,y).
    # ------------------------------------------------------------------
    stock_rw = Variable(m, "stockRW", "positive", domain=[r, y])
    harvest_rw = Variable(m, "harvestRW", "positive", domain=[r, y])
    production = Variable(m, "production", "positive", domain=[r, p, y])
    sales = Variable(m, "sales", "positive", domain=[r, p, y])
    exports = Variable(m, "exports", "positive", domain=[r, rr, p, y])

    z_profit = Variable(m, "Zprofit")
    z_emission = Variable(m, "Zemission")
    z_storage = Variable(m, "Zstorage")
    z_net = Variable(m, "ZnetCarbon")
    carbon_cost = Variable(m, "carbonCost")
    obj = Variable(m, "obj")
    revenue = Variable(m, "revenue")
    harvest_cost_tot = Variable(m, "harvestCostTot")
    proc_cost_tot = Variable(m, "procCostTot")
    trans_cost_tot = Variable(m, "transCostTot")

    exports.fx[r, rr, p, y].where[r.sameAs(rr)] = 0
    exports.up[r, rr, p, y].where[~r.sameAs(rr)] = arc_cap[r, rr, p, y]
    production.up[r, p, y] = mill_cap[r, p, y]
    sales.up[r, p, y] = demand_cap[r, p, y]
    stock_rw.lo[r, y].where[y.val >= 2025] = 0.30 * init_stock[r]

    # ------------------------------------------------------------------
    # Equations
    # ------------------------------------------------------------------
    stock_balance = Equation(m, "stockBalanceRW", domain=[r, y])
    sustainability = Equation(m, "sustainabilityRW", domain=[r, y])
    roundwood_use = Equation(m, "roundwoodUseEq", domain=[r, y])
    lumber_balance = Equation(m, "lumberBalance", domain=[r, y])
    pulp_balance = Equation(m, "pulpwoodBalance", domain=[r, y])
    clt_balance = Equation(m, "cltBalance", domain=[r, y])
    glulam_balance = Equation(m, "glulamBalance", domain=[r, y])
    emission_eq = Equation(m, "emissionEq")
    storage_eq = Equation(m, "storageEq")
    net_carbon_eq = Equation(m, "netCarbonEq")
    carbon_cost_eq = Equation(m, "eCarbonCost")
    revenue_eq = Equation(m, "revenueEq")
    harvest_cost_eq = Equation(m, "harvestCostEq")
    proc_cost_eq = Equation(m, "procCostEq")
    trans_cost_eq = Equation(m, "transCostEq")
    profit_eq = Equation(m, "profitEq")
    objective_eq = Equation(m, "scalarObjective")

    stock_balance[r, y] = stock_rw[r, y] == (
        (init_stock[r] - harvest_rw[r, y]).where[Ord(y) == 1]
        + (stock_rw[r, y - 1] + growth_rw[r, y] - harvest_rw[r, y]).where[Ord(y) > 1]
    )
    sustainability[r, y] = harvest_rw[r, y] <= (
        (max_harvest[r] * init_stock[r]).where[Ord(y) == 1]
        + (max_harvest[r] * (stock_rw[r, y - 1] + growth_rw[r, y])).where[Ord(y) > 1]
    )
    roundwood_use[r, y] = (
        production[r, "Lumber", y] / yield_factor["Lumber"]
        + production[r, "Pulpwood", y] / yield_factor["Pulpwood"]
        <= harvest_rw[r, y]
    )

    def inflow(prod: str):
        return Sum(rr.where[~r.sameAs(rr)], exports[rr, r, prod, y])

    def outflow(prod: str):
        return Sum(rr.where[~r.sameAs(rr)], exports[r, rr, prod, y])

    lumber_balance[r, y] = (
        production[r, "Lumber", y] + inflow("Lumber")
        == sales[r, "Lumber", y]
        + outflow("Lumber")
        + lumber_input["CLT"] * production[r, "CLT", y]
        + lumber_input["Glulam"] * production[r, "Glulam", y]
    )
    pulp_balance[r, y] = (
        production[r, "Pulpwood", y] + inflow("Pulpwood")
        == sales[r, "Pulpwood", y] + outflow("Pulpwood")
    )
    clt_balance[r, y] = (
        production[r, "CLT", y] + inflow("CLT")
        == sales[r, "CLT", y] + outflow("CLT")
    )
    glulam_balance[r, y] = (
        production[r, "Glulam", y] + inflow("Glulam")
        == sales[r, "Glulam", y] + outflow("Glulam")
    )

    emission_eq[...] = z_emission == Sum(
        (r, p, y),
        emission_factor[p] * production[r, p, y]
        + Sum(
            rr.where[~r.sameAs(rr)],
            transport_ef[r, rr, p] * exports[r, rr, p, y],
        ),
    )
    storage_eq[...] = z_storage == Sum(
        (r, p, y), storage_factor[p] * production[r, p, y]
    )
    net_carbon_eq[...] = z_net == z_emission - z_storage
    carbon_cost_eq[...] = carbon_cost == carbon_price * z_emission
    revenue_eq[...] = revenue == Sum((r, p, y), base_price_y[p, y] * sales[r, p, y])
    harvest_cost_eq[...] = harvest_cost_tot == Sum(
        (r, y), harvest_cost_y[r, y] * harvest_rw[r, y]
    )
    proc_cost_eq[...] = proc_cost_tot == Sum(
        (r, p, y), processing_cost_y[p, y] * production[r, p, y]
    )
    trans_cost_eq[...] = trans_cost_tot == Sum(
        Domain(r, rr, p, y).where[~r.sameAs(rr)],
        transport_cost[r, rr, p] * exports[r, rr, p, y],
    )
    profit_eq[...] = z_profit == revenue - harvest_cost_tot - proc_cost_tot - trans_cost_tot
    objective_eq[...] = obj == z_profit - carbon_cost

    core = [
        stock_balance,
        roundwood_use,
        lumber_balance,
        pulp_balance,
        clt_balance,
        glulam_balance,
        emission_eq,
        storage_eq,
        net_carbon_eq,
        carbon_cost_eq,
        revenue_eq,
        harvest_cost_eq,
        proc_cost_eq,
        trans_cost_eq,
        profit_eq,
        objective_eq,
    ]
    model_on = Model(
        m,
        name="ForestCP",
        equations=core + [sustainability],
        problem=Problem.LP,
        sense=Sense.MAX,
        objective=obj,
    )
    model_off = Model(
        m,
        name="ForestCP_open",
        equations=core,
        problem=Problem.LP,
        sense=Sense.MAX,
        objective=obj,
    )

    symbols = dict(
        r=r, p=p, y=y, rr=rr,
        demand_cap=demand_cap,
        demand_cap_base=demand_cap_base,
        base_price_y=base_price_y,
        base_price_path=base_price_path,
        sales=sales,
        production=production,
        exports=exports,
        harvest_rw=harvest_rw,
        obj=obj,
        z_profit=z_profit,
        z_emission=z_emission,
        z_storage=z_storage,
        z_net=z_net,
        carbon_cost=carbon_cost,
        carbon_price=carbon_price,
        revenue=revenue,
        harvest_cost_tot=harvest_cost_tot,
        proc_cost_tot=proc_cost_tot,
        trans_cost_tot=trans_cost_tot,
        model_on=model_on,
        model_off=model_off,
    )
    run_scenarios(symbols)


if __name__ == "__main__":
    main()
