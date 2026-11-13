"""生成可复现的合成实时资料。

脚本只生成“业务记录”，不参与估计。所有输出都用固定随机种子重算，
便于审计 data/realtime-*.jsonl 的每个数值。

覆盖的发布情形
--------------
* 正常月度发布：期末后 25 天首次发布，之后 +33 天、+68 天各修订一次。
* 延迟发布：ind_prod 2023-05 首次发布推迟到期末后 75 天。
* 历史大修订：GDP 2024-Q3 首次发布偏低约 6%，第三次修订回到真值；
  ind_prod 2024-02 首次发布偏高约 15%。
* 月末跨季度：2023-03 的月度资料在 2023-04-25 才发布；决策日放在
  2023-03-31、2023-04-01 与 2023-04-26，检验“期末已到、资料未到”。
* 整列长期缺报：ind_price 2022 全年没有任何发布；ind_prod 2020-04 至
  2020-06 没有任何发布。
* 已报告为零：ind_orders 2023-06 的发布值就是 0，保留为 reported_zero，
  不当缺报、不当未发布。
* 成熟版本缺失：2025-Q4 只有两次修订，成熟口径下不可评估，不用最后
  一个版本顶替。
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 20261114

SERIES = [
    {"id": "gdp", "frequency": "quarter", "aggregation": "flow", "unit": "million_CNY"},
    {"id": "ind_prod", "frequency": "month", "aggregation": "flow", "unit": "index_points"},
    {"id": "ind_stock", "frequency": "month", "aggregation": "stock", "unit": "million_CNY"},
    {"id": "ind_price", "frequency": "month", "aggregation": "average", "unit": "index"},
    {"id": "ind_orders", "frequency": "month", "aggregation": "flow", "unit": "orders"},
]


def month_list(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    end_year, end_month = int(last[:4]), int(last[5:])
    result = []
    while (year, month) <= (end_year, end_month):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result


def month_end(month: str) -> date:
    year, m = int(month[:4]), int(month[5:])
    if m == 12:
        return date(year, 12, 31)
    return date(year, m + 1, 1) - timedelta(days=1)


def quarter_of(month: str) -> str:
    return f"{month[:4]}-Q{(int(month[5:]) - 1) // 3 + 1}"


def quarter_end(quarter: str) -> date:
    year, q = int(quarter[:4]), int(quarter[-1])
    return month_end(f"{year:04d}-{3 * q:02d}")


MONTHS = month_list("2019-01", "2025-12")
QUARTERS = []
for month in MONTHS:
    quarter = quarter_of(month)
    if not QUARTERS or QUARTERS[-1] != quarter:
        QUARTERS.append(quarter)

SEASON_MONTH = [
    0.030, 0.022, 0.010, 0.000, -0.010, -0.022,
    -0.030, -0.020, 0.000, 0.012, 0.024, 0.036,
]


def synthetic_truth():
    """含“当季不可预测冲击”的月度—季度共同因子过程。

    * 月度活动 = 平滑趋势 + 月度季节 + 月度噪声 + 当季冲击（季度内相同）；
    * 季度目标 = 三个月活动之和的相对水平。
    当季冲击只影响当前季度，目标自身历史看不到它，但季度内已发布的
    月份可以部分揭示它——这正是实时监测要用的信息，也是纯目标基线
    无法提前获得的信息。
    """
    rng = random.Random(SEED)
    ar = 0.0
    monthly = {}
    quarter_shock: dict[str, float] = {}
    for index, month in enumerate(MONTHS):
        quarter = quarter_of(month)
        if quarter not in quarter_shock:
            # 当季冲击：季度内三个月共享，下一季度重新抽取。
            quarter_shock[quarter] = rng.gauss(0.0, 0.012)
        ar = 0.55 * ar + rng.gauss(0.0, 0.008)
        activity = (
            4.62
            + 0.0042 * index
            + SEASON_MONTH[int(month[5:]) - 1]
            + ar / 8
            + quarter_shock[quarter]
        )
        monthly[month] = {
            "activity": activity,
            "prod": 100 * math.exp(activity + rng.gauss(0.0, 0.004)),
            "stock": 210 * math.exp(0.0025 * index + 0.010 * ar + quarter_shock[quarter])
            + 6 * math.sin(2 * math.pi * (int(month[5:]) - 1) / 12),
            "price": 100
            * math.exp(0.0016 * index + 0.6 * quarter_shock[quarter])
            * math.exp(rng.gauss(0.0, 0.0015)),
            "orders": max(
                0.0,
                820
                + 160 * math.sin(2 * math.pi * (index % 12) / 12)
                + 3200 * quarter_shock[quarter]
                + 18 * ar / 0.008
                + rng.gauss(0.0, 20),
            ),
        }
    quarterly = {}
    for quarter in QUARTERS:
        months = [f"{quarter[:4]}-{3 * (int(quarter[-1]) - 1) + i:02d}" for i in (1, 2, 3)]
        base = sum(math.exp(monthly[m]["activity"]) for m in months)
        # 季度值由月度共同因子决定，只加很小的测量噪声。
        value = 42.0 * base * math.exp(rng.gauss(0.0, 0.0015))
        quarterly[quarter] = value
    return monthly, quarterly


def rev_values(true_value: float, first_error: float, rng: random.Random) -> dict[int, float]:
    return {
        1: true_value * (1 + first_error),
        2: true_value * (1 + first_error * 0.35),
        3: true_value,
    }


def main() -> None:
    monthly, quarterly = synthetic_truth()
    rng = random.Random(SEED + 1)
    releases: list[dict] = []
    counter = 0

    def add(series, period, released, revision, value, note=""):
        nonlocal counter
        counter += 1
        releases.append(
            {
                "id": f"rel{counter:04d}",
                "series": series,
                "period": period,
                "released": released.isoformat(),
                "revision": revision,
                "value": round(float(value), 4),
                "_note": note,
            }
        )

    # ---- 月度指标 -------------------------------------------------
    for series in ("ind_prod", "ind_stock", "ind_price", "ind_orders"):
        for month in MONTHS:
            if series == "ind_price" and month[:4] == "2022":
                continue  # 整列长期缺报
            if series == "ind_prod" and "2020-04" <= month <= "2020-06":
                continue  # 连续三个月无任何发布
            truth = monthly[month][{"ind_prod": "prod", "ind_stock": "stock",
                                    "ind_price": "price", "ind_orders": "orders"}[series]]
            error = 0.004 if series != "ind_stock" else 0.002
            if series == "ind_orders":
                error = 0.03
            if series == "ind_orders" and month == "2023-06":
                values = {1: 0.0, 2: 0.0, 3: 0.0}  # 真实且已报告的零
            else:
                values = rev_values(truth, rng.gauss(0.0, error), rng)
            if series == "ind_prod" and month == "2024-02":
                values = {1: truth * 1.15, 2: truth * 1.02, 3: truth}
            first_day = month_end(month) + timedelta(days=25)
            if series == "ind_prod" and month == "2023-05":
                first_day = month_end(month) + timedelta(days=75)  # 延迟发布
            add(series, month, first_day, 1, values[1])
            add(series, month, first_day + timedelta(days=33), 2, values[2])
            add(series, month, first_day + timedelta(days=68), 3, values[3])

    # ---- 季度目标 -------------------------------------------------
    for quarter in QUARTERS:
        truth = quarterly[quarter]
        first_error = rng.gauss(0.0, 0.008)
        values = rev_values(truth, first_error, rng)
        if quarter == "2024-Q3":
            values = {1: truth * 0.94, 2: truth * 0.985, 3: truth}  # 历史大修订
        if quarter == "2025-Q4":
            values = {1: values[1], 2: values[2]}  # 成熟版本缺失
        first_day = quarter_end(quarter) + timedelta(days=40)
        if quarter == "2024-Q1":
            first_day += timedelta(days=30)  # 首次发布延迟
        add("gdp", quarter, first_day, 1, values[1])
        add("gdp", quarter, first_day + timedelta(days=33), 2, values[2])
        if 3 in values:
            add("gdp", quarter, first_day + timedelta(days=68), 3, values[3])

    # ---- 决策日 ---------------------------------------------------
    decisions: list[dict] = []
    targets = [q for q in QUARTERS if q >= "2023-Q1"]

    def add_decision(day: date, target: str):
        decisions.append(
            {
                "id": f"dec{len(decisions) + 1:03d}",
                "day": day.isoformat(),
                "target": f"gdp:{target}",
            }
        )

    for quarter in targets:
        end = quarter_end(quarter)
        offsets = [-10, 15, 33, 50]
        for offset in offsets:
            add_decision(end + timedelta(days=offset), quarter)
    # 月末跨季度：期末当天、次日，以及 3 月资料发布前后各做一次决策。
    add_decision(date(2023, 3, 31), "2023-Q1")
    add_decision(date(2023, 4, 1), "2023-Q1")
    add_decision(date(2023, 4, 26), "2023-Q1")
    # 2025-Q4 成熟版本缺失，保留决策以便检验冻结口径。
    add_decision(date(2026, 1, 20), "2025-Q4")

    def write(table: str, rows: list[dict], fields: list[str]) -> dict:
        path = DATA / f"realtime-{table}.jsonl"
        lines = []
        for row in rows:
            clean = {field: row[field] for field in fields}
            lines.append(json.dumps(clean, ensure_ascii=False, sort_keys=True))
        content = ("\n".join(lines) + "\n").encode("utf-8")
        path.write_bytes(content)
        return {
            "file": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": len(rows),
            "fields": fields,
        }

    series_fields = ["id", "frequency", "aggregation", "unit"]
    release_fields = ["id", "series", "period", "released", "revision", "value"]
    decision_fields = ["id", "day", "target"]
    manifest = {
        "schema": 1,
        "tables": {
            "series": write("series", SERIES, series_fields),
            "releases": write("releases", releases, release_fields),
            "decisions": write("decisions", decisions, decision_fields),
        },
    }
    (DATA / "realtime.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"月度 {len(MONTHS)}，季度 {len(QUARTERS)}，发布 {len(releases)}，决策 {len(decisions)}")


if __name__ == "__main__":
    main()
