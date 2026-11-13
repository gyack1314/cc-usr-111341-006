"""生成 docs/report.json 与 docs/report.md 的可复现报告。

只读取已登记资料，不写回 data/；输出固定随机种子下的指标、分解与
场景说明，供评审逐项核对。
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nowcast_data.evaluation import (  # noqa: E402
    BacktestSpec,
    backtest,
    build_decision_record,
    release_calendar,
    shapley_decomposition,
)
from nowcast_data.io import load  # noqa: E402
from nowcast_data.model import Config, estimate, fit_params  # noqa: E402
from nowcast_data.periods import parse_day  # noqa: E402
from nowcast_data.vintages import (  # noqa: E402
    snapshot,
    validate_decisions,
    validate_releases,
    validate_series,
    visible_versions,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CONFIG = Config(draws=400, seed=20261114)
SPEC = BacktestSpec(maturity_revision=3, interval_level=0.80)
SCENARIO_DECISIONS = [
    ("dec001", "dec002", "期初未见 2 月资料；2 月发布后"),
    ("dec002", "dec003", "2 月资料 + 1 月修订；3 月资料发布后"),
    ("dec003", "dec004", "3 月资料齐全；目标首次发布前"),
    ("dec004", "dec005", "目标首次发布（target_version 主导）"),
]


def _replay_sample(decision, metas, releases):
    """只保留目标季度相关单元，避免整份重放记录把报告撑得过大。"""
    record = build_decision_record(decision, metas, releases, config=CONFIG)
    months = [
        f"{decision.target[:4]}-{3 * (int(decision.target[-1]) - 1) + i:02d}"
        for i in (1, 2, 3)
    ]
    keep = {f"{decision.target_series}:{decision.target}"}
    keep |= {f"{series}:{month}" for series in metas for month in months}
    record["quarter_cells"] = {
        key: cell for key, cell in record["visible_cells"].items() if key in keep
    }
    record.pop("visible_cells")
    return record


def _bundle():
    tables = load(ROOT / "data" / "realtime.json")
    metas = validate_series(tables["series"])
    releases = validate_releases(tables["releases"], metas)
    decisions = validate_decisions(tables["decisions"], metas)
    return metas, releases, decisions


def _round(value, digits=4):
    if isinstance(value, float):
        return round(value, digits)
    return value


def _clean(obj, digits=4):
    if isinstance(obj, dict):
        return {k: _clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v, digits) for v in obj]
    return _round(obj, digits)


def main() -> None:
    metas, releases, decisions = _bundle()
    bt = backtest(metas, releases, decisions, CONFIG, SPEC)
    by_id = {d.id: d for d in decisions}
    decomp = []
    for a, b, label in SCENARIO_DECISIONS:
        if (by_id[a].target_series, by_id[a].target) != (
            by_id[b].target_series,
            by_id[b].target,
        ):
            continue
        result = shapley_decomposition(metas, releases, by_id[a], by_id[b], CONFIG)
        decomp.append({"label": label, "from": a, "to": b, "result": result})

    scenarios = {}
    # 延迟发布：ind_prod 2023-05 首次发布推迟到期末后 75 天。
    day = parse_day("2023-08-02")
    view = visible_versions(releases, day)
    params = fit_params(metas, releases, day, "gdp", CONFIG)
    est = estimate(params, view, "2023-Q2", day, include_inputs=False)
    prod = [r for r in releases if r.series == "ind_prod" and r.period == "2023-05"]
    scenarios["late_release"] = {
        "case": "ind_prod 2023-05 延迟发布",
        "first_release_date": sorted(r.released for r in prod)[0].isoformat(),
        "decision_day": day.isoformat(),
        "month_state": est.months["ind_prod"],
        "bridge_present": any(
            c.kind == "bridge" and c.name.endswith("ind_prod") for c in est.components
        ),
        "flags": est.flags,
    }
    # 历史大修订：GDP 2024-Q3 首次发布偏低 6%。
    day = parse_day("2024-11-02")
    params = fit_params(metas, releases, day, "gdp", CONFIG)
    view = visible_versions(releases, day)
    gdp_releases = sorted(
        [
            r
            for r in releases
            if r.series == "gdp" and r.period == "2024-Q3"
        ],
        key=lambda r: (r.released, r.revision),
    )
    first = estimate(
        params, view, "2024-Q3", day, measure="first_release", include_inputs=False
    )
    mature = estimate(
        params, view, "2024-Q3", day, measure="mature", include_inputs=False
    )

    def _covered(est, actual_value):
        if est.interval is None or actual_value is None:
            return None
        return est.interval[0] <= actual_value <= est.interval[1]

    scenarios["large_revision"] = {
        "case": "GDP 2024-Q3 首次发布偏低 6%",
        "versions": [
            {"revision": r.revision, "released": r.released.isoformat(), "value": r.value}
            for r in gdp_releases
        ],
        "estimate_first_release": {
            "point": first.point,
            "interval": list(first.interval) if first.interval else None,
            "actual": gdp_releases[0].value,
            "covered": _covered(first, gdp_releases[0].value),
        },
        "estimate_mature": {
            "point": mature.point,
            "interval": list(mature.interval) if mature.interval else None,
            "actual": gdp_releases[-1].value,
            "covered": _covered(mature, gdp_releases[-1].value),
        },
        "revision_drift": params.revision.drift,
        "revision_sigma": params.revision.sigma,
        "revision_n": params.revision.n,
    }
    # 月末跨季度：3 月资料 4 月 25 日发布。
    cross = []
    for day_str in ("2023-03-31", "2023-04-01", "2023-04-26"):
        day = parse_day(day_str)
        params = fit_params(metas, releases, day, "gdp", CONFIG)
        view = visible_versions(releases, day)
        est = estimate(params, view, "2023-Q1", day, include_inputs=False)
        cells = snapshot(
            metas, releases, day, periods={"ind_prod": ["2023-03"]}
        )
        cross.append(
            {
                "day": day_str,
                "march_state": cells[("ind_prod", "2023-03")].state,
                "point": est.point,
                "status": est.status,
                "interval": list(est.interval) if est.interval else None,
            }
        )
    scenarios["month_end_cross_quarter"] = {
        "case": "2023-03 月度资料在 2023-04-25 发布",
        "checks": cross,
    }
    # 整列长期缺报：ind_price 2022 全年无发布。
    day = parse_day("2022-12-20")
    params = fit_params(metas, releases, day, "gdp", CONFIG)
    indicator = params.indicators["ind_price"]
    gap_view = visible_versions(releases, day)
    gap_est = estimate(params, gap_view, "2022-Q4", day, include_inputs=False)
    scenarios["long_gap_column"] = {
        "case": "ind_price 2022 全年无任何发布",
        "decision_day": day.isoformat(),
        "visible_months_2022": sorted(
            p for p in indicator.raw_levels if p.startswith("2022")
        ),
        "bridge_fitted": indicator.bridge is not None,
        "ar_fitted": indicator.ar is not None,
        "bridge_component_present": any(
            c.kind == "bridge" and c.name.endswith("ind_price")
            for c in gap_est.components
        ),
        "notes": gap_est.flags,
    }
    # 到期未发布：期末已过但决策日仍无发布。
    day = parse_day("2023-03-31")
    params = fit_params(metas, releases, day, "gdp", CONFIG)
    view = visible_versions(releases, day)
    est = estimate(params, view, "2023-Q1", day, include_inputs=False)
    scenarios["period_ended_not_published"] = {
        "case": "2023-Q1 已于 3 月 31 日结束，决策日仍无任何目标发布",
        "target_state": est.target_state,
        "status": est.status,
        "formal": est.formal,
        "months": est.months,
        "point": est.point,
    }

    report = {
        "generated_from": "data/realtime.json",
        "config": {"full": CONFIG.key(), "baseline": bt["config"]["target_only"]},
        "spec": bt["spec"],
        "decisions_total": bt["decisions_total"],
        "calendar": release_calendar(metas, releases),
        "metrics": bt["metrics"],
        "decompositions": [
            {
                "label": item["label"],
                "from": item["from"],
                "to": item["to"],
                "status": item["result"].get("status"),
                "point_from": item["result"].get("point_from"),
                "point_to": item["result"].get("point_to"),
                "total_change": item["result"].get("total_change"),
                "contributions": item["result"].get("contributions"),
                "residual": item["result"].get("residual"),
                "sequential": item["result"].get("sequential", {}).get("deltas"),
                "sequential_residual": item["result"].get("sequential_residual"),
                        "interaction_max_abs": item["result"].get("interaction", {}).get(
                    "max_abs"
                ),
                "interaction_pairwise": item["result"].get("interaction", {}).get(
                    "pairwise"
                ),
                "missing_scenarios": item["result"].get("missing_scenarios"),
                "new_releases": item["result"].get("new_releases"),
                "revised_releases": item["result"].get("revised_releases"),
                "target_version_changes": item["result"].get(
                    "target_version_changes"
                ),
            }
            for item in decomp
        ],
        "scenarios": scenarios,
        "replay_sample": _replay_sample(
            by_id["dec003"], metas, releases
        ),
    }
    (DOCS / "report.json").write_text(
        json.dumps(_clean(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    redacted = json.loads(json.dumps(_clean(report), ensure_ascii=False))
    redacted.pop("calendar", None)
    redacted.pop("replay_sample", None)
    (DOCS / "report-summary.json").write_text(
        json.dumps(redacted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (DOCS / "report.md").write_text(render(report), encoding="utf-8")
    print("写出 docs/report.json, docs/report-summary.json, docs/report.md")


def _pct(x):
    return f"{x*100:.1f}%" if x is not None else "—"


def render(report: dict) -> str:
    lines = ["# 实时监测回测报告（自动生成）", ""]
    lines.append(
        "本文件由 `python tools/make_report.py` 生成，数据来自 `data/realtime.json`；"
        "所有数字在固定种子下可重算。方法说明见 `docs/methodology.md`。"
    )
    lines.append("")
    lines.append("## 冻结口径")
    lines.append("")
    spec = report["spec"]
    lines.append(f"- 首次发布口径：`{spec['criteria']['first_release']}`")
    lines.append(f"- 成熟口径：`{spec['criteria']['mature']}`")
    lines.append(f"- 可见性：`{spec['visibility']}`")
    lines.append(f"- 参数拟合：{spec['parameter_fit']}")
    nominal = report["metrics"]["criterion"]["first_release"]["full"]["n"]
    lines.append(
        f"- 名义区间水平：{_pct(0.80)}；实测覆盖率在下方表格逐项报告"
        f"（首次发布口径 n={nominal}）"
    )
    lines.append("")
    lines.append("## 两个口径的汇总指标")
    lines.append("")
    lines.append(
        "| 口径 | 模型 | n | MAE | RMSE | 偏差 | MAPE | 区间覆盖率 | 平均区间宽度 | 相对基线宽度 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for criterion, models in report["metrics"]["criterion"].items():
        for model, block in models.items():
            lines.append(
                f"| {criterion} | {model} | {block['n']} | "
                f"{block['mae']:.1f} | {block['rmse']:.1f} | {block['bias']:.1f} | "
                f"{_pct(block['mape'])} | {_pct(block['coverage'])} | "
                f"{block['avg_width']:.1f} | "
                f"{block['width_ratio_to_baseline']:.3f} |"
            )
    lines.append("")
    lines.append("## 误差随提前期变化（首次发布口径）")
    lines.append("")
    lines.append("| 提前期（日历天） | 模型 | n | MAE | 覆盖率 | 平均区间宽度 | 相对基线宽度 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for bucket, models in report["metrics"]["by_horizon"]["first_release"].items():
        for model, block in models.items():
            lines.append(
                f"| {bucket} | {model} | {block['n']} | {block['mae']:.1f} | "
                f"{_pct(block['coverage'])} | {block['avg_width']:.1f} | "
                f"{block['width_ratio_to_baseline']:.3f} |"
            )
    lines.append("")
    lines.append("## 相邻决策日的四源分解（Shapley）")
    lines.append("")
    lines.append(
        "| 区间 | 总变动 | 新发布 | 旧数据修订 | 参数重估 | 目标版本 | Shapley 残差 | 链式残差 | 交互最大绝对值 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in report["decompositions"]:
        if item["status"] != "ok":
            continue
        c = item["contributions"]
        lines.append(
            f"| {item['from']} → {item['to']}（{item['label']}） | "
            f"{item['total_change']:.1f} | {c['new_data']:.1f} | "
            f"{c['revisions']:.1f} | {c['reestimate']:.1f} | "
            f"{c['target_version']:.1f} | {item['residual']:.2e} | "
            f"{item['sequential_residual']:.2e} | "
            f"{item['interaction_max_abs']:.2e} |"
        )
    lines.append("")
    lines.append("## 场景行为")
    lines.append("")
    lines.append("### 延迟发布")
    scenario = report["scenarios"]["late_release"]
    lines.append(
        f"- {scenario['case']}；首次发布日 {scenario['first_release_date']}，"
        f"决策日 {scenario['decision_day']}。目标季度各月状态："
        f"{scenario['month_state']}；桥接分量存在：{scenario['bridge_present']}。"
    )
    lines.append("")
    lines.append("### 历史大修订")
    scenario = report["scenarios"]["large_revision"]
    lines.append(f"- {scenario['case']}")
    lines.append("")
    lines.append("| 修订 | 发布日期 | 值 |")
    lines.append("| --- | --- | --- |")
    for version in scenario["versions"]:
        lines.append(f"| {version['revision']} | {version['released']} | {version['value']:.1f} |")
    lines.append("")
    first = scenario["estimate_first_release"]
    mature = scenario["estimate_mature"]
    lines.append(
        f"- 首次发布口径点估计 {first['point']:.1f}"
        f"（区间 {first['interval'][0]:.1f}–{first['interval'][1]:.1f}），"
        f"实际首次发布 {first['actual']:.1f}，"
        f"覆盖={'是' if first['covered'] else '否'}。"
        f"成熟口径点估计 {mature['point']:.1f}"
        f"（区间 {mature['interval'][0]:.1f}–{mature['interval'][1]:.1f}），"
        f"实际成熟值 {mature['actual']:.1f}，"
        f"覆盖={'是' if mature['covered'] else '否'}。"
    )
    lines.append("")
    lines.append(
        "这次 −6% 的首次发布属于个别离群修订；训练期的平均首次→成熟漂移只有 "
        f"{scenario['revision_drift']*100:.2f}%，无法预测它。首次发布口径因此落在"
        "区间之外——这正是“不能靠全时期调参掩盖历史失败”的例子；成熟口径不受影响。"
    )
    lines.append("")
    lines.append("### 月末跨季度")
    scenario = report["scenarios"]["month_end_cross_quarter"]
    lines.append(f"- {scenario['case']}")
    lines.append("")
    lines.append("| 决策日 | 2023-03 状态 | 点估计 | 状态 |")
    lines.append("| --- | --- | --- | --- |")
    for check in scenario["checks"]:
        point = f"{check['point']:.1f}" if check["point"] is not None else "—"
        lines.append(
            f"| {check['day']} | {check['march_state']} | {point} | {check['status']} |"
        )
    lines.append("")
    lines.append(
        "3 月 31 日与 4 月 1 日之间没有任何新发布，点估计与区间完全相同——这是"
        "正确行为：跨季度并不自动带来新信息，只有真正可见的发布才更新估计。"
        "4 月 26 日 3 月资料首次可见后，状态从 pending 变为 observed，估计随之更新。"
    )
    lines.append("")
    lines.append("### 整列长期缺报")
    scenario = report["scenarios"]["long_gap_column"]
    lines.append(
        f"- {scenario['case']}；决策日 {scenario['decision_day']}。"
        f"2022 年可见月份：{scenario['visible_months_2022'] or '无'}；"
        f"桥接可识别：{scenario['bridge_fitted']}；"
        f"本次估计实际启用该桥接分量：{scenario['bridge_component_present']}；"
        f"AR 可识别：{scenario['ar_fitted']}。"
    )
    notes = [n for n in scenario["notes"] if scenario["case"].split()[0] in n]
    if notes:
        lines.append(f"- 相关提示：{'；'.join(notes)}")
    lines.append("")
    lines.append("### 期末已到但尚未发布")
    scenario = report["scenarios"]["period_ended_not_published"]
    lines.append(
        f"- {scenario['case']}；目标状态 {scenario['target_state']}，"
        f"估计状态 {scenario['status']}，正式估计={scenario['formal']}。"
    )
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```sh")
    lines.append("python -I tools/make_realtime_data.py")
    lines.append("python tools/make_report.py")
    lines.append("python -m unittest discover -s tests -v")
    lines.append("```")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
