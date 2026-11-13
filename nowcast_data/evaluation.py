"""修订分解、回测与重放记录。

分解约定
--------

同一季度在两个相邻决策日 d1、d2 之间的点估计变动，按四个来源归因：

1. ``new_data``：d2 才首次可见的指标发布（不含目标自身的发布）。
2. ``revisions``：d1 已可见、d2 值发生变化的指标发布（旧数据修订）。
3. ``reestimate``：参数由 d1 截止日样本重估为 d2 截止日样本。
4. ``target_version``：目标指标自身的可见版本发生变化（新发布或修订）。

采用精确 Shapley 值：对四个来源的 16 个开关组合分别估价，再对每个来源
求边际贡献的加权平均。Shapley 值对交互是对称的，且四个值严格加总到
``V(全部) - V(空集)``。同时输出固定顺序的链式分解：链式增量也严格加总，
但交互被记在较后加入的来源上；两种约定都公开，便于审计。

回测口径
--------

* 首次发布值 = 修订序号 1 的值，无论它何时发布。
* 成熟版本 = 修订序号等于 ``maturity_revision`` 的值；若该修订不存在，
  该季度在成熟口径下记为不可用，不用“最后一个版本”替代。这样不会挑
  有利版本。
* 估计只使用决策日可见版本；评估使用决策日之后的发布，仅用于打分。
"""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from datetime import date
from statistics import mean, median
from typing import Iterable

from .model import Config, Estimate, ParamSet, estimate, fit_params
from .periods import (
    HORIZONS,
    horizon_bucket,
    horizon_days,
    period_end,
    quarter_months,
    shift_period,
)
from .vintages import Decision, Release, SeriesMeta, snapshot, visible_versions

DECOMPOSITION_FACTORS = ["new_data", "revisions", "reestimate", "target_version"]
SEQUENTIAL_ORDER = ["new_data", "revisions", "reestimate", "target_version"]


# ---------------------------------------------------------------- 版本读取


def release_timeline(
    releases: Iterable[Release], series: str, period: str
) -> list[Release]:
    return sorted(
        (r for r in releases if r.series == series and r.period == period),
        key=lambda r: (r.released, r.revision),
    )


def release_versions(
    releases: Iterable[Release], series: str, period: str, day: date
) -> list[Release]:
    """决策日路径上的全部版本（按时间顺序）。"""
    return [r for r in release_timeline(releases, series, period) if r.released <= day]


def actual_value(
    releases: Iterable[Release],
    series: str,
    period: str,
    criterion: str,
    maturity_revision: int,
) -> Release | None:
    """冻结口径下的实际值：first_release 或恰好第 maturity_revision 次修订。"""
    versions = {
        r.revision: r for r in releases if r.series == series and r.period == period
    }
    if criterion == "first_release":
        return versions.get(1)
    if criterion == "mature":
        return versions.get(maturity_revision)
    raise ValueError(f"未知的评估口径: {criterion}")


# ---------------------------------------------------------------- 重放


def build_decision_record(
    decision: Decision,
    metas: dict[str, SeriesMeta],
    releases: list[Release],
    *,
    config: Config | None = None,
) -> dict:
    """单个决策日的可逐日重放记录。"""
    config = config or Config()
    view = visible_versions(releases, decision.day)
    cells = snapshot(
        metas,
        releases,
        decision.day,
        periods={
            series_id: quarter_months(decision.target)
            for series_id, meta in metas.items()
            if meta.frequency == "month"
        }
        | {
            decision.target_series: [
                shift_period(decision.target, -1),
                decision.target,
                shift_period(decision.target, -2),
                shift_period(decision.target, -3),
            ]
        },
        grace_days=config.missing_after_days,
    )
    target_state = next(
        cell.state for cell in cells.values()
        if cell.series == decision.target_series and cell.period == decision.target
    )
    return {
        "decision": decision.to_dict(),
        "visibility_rule": "released <= day（当日发布视为当日可得）",
        "target_state": target_state,
        "visible_cells": {
            f"{sid}:{period}": cell.to_dict()
            for (sid, period), cell in sorted(cells.items())
        },
        "used_releases": [
            {
                "series": rel.series,
                "period": rel.period,
                "release_id": rel.id,
                "revision": rel.revision,
                "released": rel.released.isoformat(),
                "value": rel.value,
            }
            for key, rel in sorted(view.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        ],
        "target_versions": [
            {
                "release_id": rel.id,
                "revision": rel.revision,
                "released": rel.released.isoformat(),
                "value": rel.value,
            }
            for rel in release_versions(
                releases, decision.target_series, decision.target, decision.day
            )
        ],
        "inputs_digest": _digest(view),
    }


def _digest(view: dict[tuple[str, str], Release]) -> str:
    digest = hashlib.sha256()
    for key in sorted(view):
        rel = view[key]
        digest.update(
            f"{rel.series}|{rel.period}|{rel.released.isoformat()}|"
            f"{rel.revision}|{rel.value!r}\n".encode("utf-8")
        )
    return digest.hexdigest()


def release_calendar(
    metas: dict[str, SeriesMeta], releases: list[Release]
) -> dict:
    """发布日历：每个指标每次实际发布，以及首发布滞后统计。"""
    calendar = {}
    for series_id, meta in metas.items():
        rows = []
        lags = []
        for rel in sorted(
            (r for r in releases if r.series == series_id),
            key=lambda r: (r.period, r.released, r.revision),
        ):
            lag = (rel.released - period_end(rel.period)).days
            if rel.revision == 1:
                lags.append(lag)
            rows.append(
                {
                    "period": rel.period,
                    "release_id": rel.id,
                    "revision": rel.revision,
                    "released": rel.released.isoformat(),
                    "value": rel.value,
                    "lag_days": lag,
                }
            )
        calendar[series_id] = {
            "frequency": meta.frequency,
            "aggregation": meta.aggregation,
            "unit": meta.unit,
            "first_release_lag_median_days": int(round(median(lags))) if lags else None,
            "first_release_lag_max_days": max(lags) if lags else None,
            "releases": rows,
        }
    return calendar


# ---------------------------------------------------------------- 场景估价


def _scenario_view(
    d1: dict[tuple[str, str], Release],
    d2: dict[tuple[str, str], Release],
    target_series: str,
    factors: set[str],
) -> dict[tuple[str, str], Release]:
    """按来源开关构造中间资料视图。

    * 新可见指标发布只在 ``new_data`` 打开时加入；
    * 指标旧版本的值只在 ``revisions`` 打开时更新；
    * 目标指标的任何新发布或修订只在 ``target_version`` 打开时加入。
    """
    view: dict[tuple[str, str], Release] = dict(d1)
    keys = set(d1) | set(d2)
    for key in sorted(keys):
        old, new = d1.get(key), d2.get(key)
        is_target = key[0] == target_series
        if old is None and new is not None:
            if is_target and "target_version" in factors:
                view[key] = new
            elif not is_target and "new_data" in factors:
                view[key] = new
            continue
        if old is not None and new is not None and old.id != new.id:
            if is_target and "target_version" in factors:
                view[key] = new
            elif not is_target and "revisions" in factors:
                view[key] = new
    return view


def _scenario_releases(
    releases: list[Release],
    d1: dict[tuple[str, str], Release],
    d2: dict[tuple[str, str], Release],
    target_series: str,
    factors: set[str],
    d1_day: date,
) -> list[Release]:
    """与 ``_scenario_view`` 一致的完整版本记录，供重估参数使用。

    规则：d1 之前发布的版本一律保留；d1 之后、d2 之前发布的版本，只有
    对应来源被打开时才保留。这样“旧数据修订”不会经由参数泄漏到关闭
    该来源的情景里。

    注意 ``reestimate`` 不是资料开关：它决定这些可见资料是否重新拟合
    参数。资料集合由 new_data / revisions / target_version 三个来源决定。
    """
    selected = []
    for rel in releases:
        key = (rel.series, rel.period)
        if rel.released <= d1_day:
            selected.append(rel)
            continue
        if key not in d2:
            continue
        if rel.released > d2[key].released:
            continue
        if key[0] == target_series:
            if "target_version" in factors:
                selected.append(rel)
        elif key not in d1:
            if "new_data" in factors:
                selected.append(rel)
        elif "revisions" in factors:
            selected.append(rel)
    return selected


def _scenario_estimate(
    metas: dict[str, SeriesMeta],
    releases: list[Release],
    d1: dict[tuple[str, str], Release],
    d2: dict[tuple[str, str], Release],
    decision: Decision,
    base_params: ParamSet,
    factors: set[str],
    config: Config,
    measure: str,
    d1_day: date,
) -> Estimate | None:
    view = _scenario_view(d1, d2, decision.target_series, factors)
    if "reestimate" in factors:
        # 用该情景自己的资料集合在 d2 截点重估参数。
        try:
            params = fit_params(
                metas,
                _scenario_releases(
                    releases, d1, d2, decision.target_series, factors, d1_day
                ),
                decision.day,
                decision.target_series,
                config,
            )
        except ValueError:
            params = base_params
    else:
        params = base_params
    try:
        return estimate(
            params,
            view,
            decision.target,
            decision.day,
            measure=measure,
            simulate=False,
            include_inputs=False,
        )
    except ValueError:
        return None


def _point(est: Estimate | None) -> float | None:
    return est.point if est is not None else None


def shapley_decomposition(
    metas: dict[str, SeriesMeta],
    releases: list[Release],
    decision_d1: Decision,
    decision_d2: Decision,
    config: Config,
    *,
    measure: str = "first_release",
) -> dict:
    """四个来源的精确 Shapley 分解与固定顺序链式分解。"""
    if (decision_d1.target_series, decision_d1.target) != (
        decision_d2.target_series,
        decision_d2.target,
    ):
        raise ValueError("两个决策日的目标必须相同")
    if decision_d2.day <= decision_d1.day:
        raise ValueError("第二个决策日必须晚于第一个")

    d1_view = visible_versions(releases, decision_d1.day)
    d2_view = visible_versions(releases, decision_d2.day)
    try:
        base_params = fit_params(
            metas, releases, decision_d1.day, decision_d1.target_series, config
        )
    except ValueError as exc:
        return {
            "status": "unavailable",
            "reason": f"第一个决策日无法拟合参数: {exc}",
            "factors": DECOMPOSITION_FACTORS,
            "convention": _convention(),
        }

    weights = _shapley_weights(len(DECOMPOSITION_FACTORS))
    cache: dict[frozenset[str], float | None] = {}
    for subset_size in range(len(DECOMPOSITION_FACTORS) + 1):
        for subset in itertools.combinations(DECOMPOSITION_FACTORS, subset_size):
            key = frozenset(subset)
            est = _scenario_estimate(
                metas, releases, d1_view, d2_view, decision_d2, base_params,
                set(subset), config, measure, decision_d1.day,
            )
            cache[key] = _point(est)
    if any(cache[frozenset(DECOMPOSITION_FACTORS)] is None for _ in [0]):
        return {
            "status": "unavailable",
            "reason": "全因素情景无法产生点估计",
            "factors": DECOMPOSITION_FACTORS,
            "convention": _convention(),
        }
    full = cache[frozenset(DECOMPOSITION_FACTORS)]
    empty = cache[frozenset()]
    if empty is None:
        return {
            "status": "unavailable",
            "reason": "第一个决策日的基准估计不可用（拒绝或目标已观测）",
            "factors": DECOMPOSITION_FACTORS,
            "convention": _convention(),
        }
    contributions = {factor: 0.0 for factor in DECOMPOSITION_FACTORS}
    missing_subsets = 0
    for factor in DECOMPOSITION_FACTORS:
        for subset_size in range(len(DECOMPOSITION_FACTORS)):
            for subset in itertools.combinations(
                [f for f in DECOMPOSITION_FACTORS if f != factor], subset_size
            ):
                key_without = frozenset(subset)
                key_with = key_without | {factor}
                if cache[key_without] is None or cache[key_with] is None:
                    missing_subsets += 1
                    continue
                delta = cache[key_with] - cache[key_without]
                contributions[factor] += weights[len(subset)] * delta
    total = full - empty
    assigned = sum(contributions.values())

    chain = _chain_decomposition(cache, empty)
    interactions = _interaction_report(cache)
    changes = _release_changes(d1_view, d2_view, decision_d1.target_series)
    return {
        "status": "ok" if missing_subsets == 0 else "partial",
        "target": f"{decision_d2.target_series}:{decision_d2.target}",
        "measure": measure,
        "day_from": decision_d1.day.isoformat(),
        "day_to": decision_d2.day.isoformat(),
        "point_from": empty,
        "point_to": full,
        "total_change": total,
        "contributions": contributions,
        "contribution_share": {
            factor: (value / total if abs(total) > 1e-12 else None)
            for factor, value in contributions.items()
        },
        "assigned_total": assigned,
        "residual": total - assigned,
        "missing_scenarios": missing_subsets,
        "sequential": chain,
        "sequential_residual": total - sum(chain["deltas"].values()),
        "interaction": interactions,
        "new_releases": changes["new"],
        "revised_releases": changes["revisions"],
        "target_version_changes": changes["target"],
        "convention": _convention(),
        "visibility_rule": "released <= day（当日发布视为当日可得）",
    }


def _shapley_weights(n: int) -> list[float]:
    return [
        math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
        for s in range(n)
    ]


def _chain_decomposition(cache: dict[frozenset[str], float | None], empty: float) -> dict:
    current = empty
    deltas = {}
    steps = []
    active: set[str] = set()
    for factor in SEQUENTIAL_ORDER:
        active.add(factor)
        value = cache[frozenset(active)]
        if value is None:
            return {
                "status": "unavailable",
                "order": SEQUENTIAL_ORDER,
                "deltas": {},
                "reason": f"步骤 {factor} 无法估价",
            }
        deltas[factor] = value - current
        steps.append(
            {"factor": factor, "level": value, "delta": value - current}
        )
        current = value
    return {
        "status": "ok",
        "order": SEQUENTIAL_ORDER,
        "deltas": deltas,
        "steps": steps,
        "final": current,
        "rule": "按公开顺序逐个打开来源，交互记入较后打开的来源",
    }


def _interaction_report(cache: dict[frozenset[str], float | None]) -> dict:
    interactions = {}
    for a, b in itertools.combinations(DECOMPOSITION_FACTORS, 2):
        both = cache[frozenset({a, b})]
        one = cache[frozenset({a})]
        other = cache[frozenset({b})]
        base = cache[frozenset()]
        if None in (both, one, other, base):
            continue
        interactions[f"{a}*{b}"] = both - one - other + base
    magnitudes = [abs(v) for v in interactions.values()]
    return {
        "pairwise": interactions,
        "max_abs": max(magnitudes) if magnitudes else 0.0,
        "shapley_note": "Shapley 把交互对称分摊到相关来源；链式把交互记给较后者",
    }


def _release_changes(
    d1: dict[tuple[str, str], Release],
    d2: dict[tuple[str, str], Release],
    target_series: str,
) -> dict:
    new, revisions, target = [], [], []
    for key in sorted(set(d1) | set(d2)):
        old, now = d1.get(key), d2.get(key)
        entry = {
            "series": key[0],
            "period": key[1],
            "old": None
            if old is None
            else {
                "release_id": old.id,
                "revision": old.revision,
                "released": old.released.isoformat(),
                "value": old.value,
            },
            "new": None
            if now is None
            else {
                "release_id": now.id,
                "revision": now.revision,
                "released": now.released.isoformat(),
                "value": now.value,
            },
        }
        if key[0] == target_series:
            if old is None or (now is not None and old.id != now.id):
                target.append(entry)
        elif old is None and now is not None:
            new.append(entry)
        elif now is not None and old.id != now.id:
            revisions.append(entry)
    return {"new": new, "revisions": revisions, "target": target}


def _convention() -> dict:
    return {
        "method": "exact_shapley",
        "factors": DECOMPOSITION_FACTORS,
        "sequential_order": SEQUENTIAL_ORDER,
        "visibility_rule": "released <= day（当日发布视为当日可得）",
        "reestimate_rule": "来源 reestimate 打开时，用该情景自己的可见视图重估参数；关闭时沿用 d1 参数",
        "additivity": "Shapley 四项加总 = 全因素点估计 − 基准点估计；链式增量同样加总，交互记入较后来源",
        "partial_scenarios": "任一侧估计被拒绝或目标已观测时按不可用计，不插补",
    }


# ---------------------------------------------------------------- 回测


@dataclass(frozen=True)
class BacktestSpec:
    """冻结的回测口径；输出中逐项记录，事后不随结果修改。"""

    first_release: str = "revision==1"
    mature: str = "revision=={maturity}"
    maturity_revision: int = 3
    visibility: str = "released <= decision_day"
    parameter_fit: str = "可见版本，fit_day = decision_day"
    interval_level: float = 0.80
    skip_observed_target: bool = True
    models: tuple[str, ...] = ("full", "target_only")

    def to_dict(self) -> dict:
        return {
            "criteria": {
                "first_release": self.first_release,
                "mature": self.mature.format(maturity=self.maturity_revision),
            },
            "visibility": self.visibility,
            "parameter_fit": self.parameter_fit,
            "interval_level": self.interval_level,
            "skip_observed_target": self.skip_observed_target,
            "models": list(self.models),
            "maturity_revision": self.maturity_revision,
        }


def baseline_config(config: Config) -> Config:
    """只用目标自身历史的基线。

    保留目标自身的季节效应与修订历史（都属于目标自身资料），只关闭
    月度桥接。这样比较的是“多一批月度指标”带来的增益，而不是把基线
    故意做差。
    """
    return Config(
        seasonal=config.seasonal,
        transform=config.transform,
        bridge=False,
        revision_adjust=config.revision_adjust,
        interval_level=config.interval_level,
        draws=config.draws,
        seed=config.seed,
        missing_after_days=config.missing_after_days,
        structural_break_z=config.structural_break_z,
        min_target_quarters=config.min_target_quarters,
        revision_maturity=config.revision_maturity,
    )


def _relative_error(pred: float, actual: float) -> float | None:
    if actual == 0:
        return None
    return pred / actual - 1.0


def backtest(
    metas: dict[str, SeriesMeta],
    releases: list[Release],
    decisions: list[Decision],
    config: Config,
    spec: BacktestSpec | None = None,
) -> dict:
    """逐决策日滚动回测；两个评估口径各自冻结。"""
    spec = spec or BacktestSpec(interval_level=config.interval_level,
                                maturity_revision=config.revision_maturity)
    records = []
    baseline_cfg = baseline_config(config)
    for decision in sorted(decisions, key=lambda d: (d.day, d.id)):
        view = visible_versions(releases, decision.day)
        actuals = {
            criterion: actual_value(
                releases, decision.target_series, decision.target,
                criterion, spec.maturity_revision,
            )
            for criterion in ("first_release", "mature")
        }
        row = {
            "decision_id": decision.id,
            "day": decision.day.isoformat(),
            "target": f"{decision.target_series}:{decision.target}",
            "horizon_days": horizon_days(decision.day, decision.target),
            "horizon_bucket": horizon_bucket(horizon_days(decision.day, decision.target)),
            "actual": {
                criterion: (
                    None
                    if rel is None
                    else {
                        "release_id": rel.id,
                        "revision": rel.revision,
                        "released": rel.released.isoformat(),
                        "value": rel.value,
                    }
                )
                for criterion, rel in actuals.items()
            },
            "models": {},
        }
        for model_name, model_config in (
            ("full", config), ("target_only", baseline_cfg)
        ):
            try:
                params = fit_params(
                    metas, releases, decision.day, decision.target_series,
                    model_config,
                )
            except ValueError as exc:
                row["models"][model_name] = {
                    "status": "refused", "reason": str(exc), "formal": False,
                }
                continue
            model_entry = {}
            for measure, criterion in (
                ("first_release", "first_release"), ("mature", "mature")
            ):
                est = estimate(
                    params, view, decision.target, decision.day,
                    measure=measure,
                    include_inputs=False,
                )
                entry = est.to_dict(include_inputs=False)
                entry["measure"] = measure
                entry["actual"] = row["actual"].get(criterion)
                entry["evaluation_criterion"] = criterion
                entry["error"] = _errors(est, actuals[criterion])
                model_entry[measure] = entry
            row["models"][model_name] = model_entry
        records.append(row)
    return {
        "spec": spec.to_dict(),
        "config": {
            "full": config.key(),
            "target_only": baseline_cfg.key(),
            "interval_level": config.interval_level,
            "maturity_revision": spec.maturity_revision,
            "seed": config.seed,
        },
        "decisions_total": len(records),
        "records": records,
        "metrics": metrics(records, spec),
    }


def _errors(est: Estimate, actual: Release | None) -> dict:
    if actual is None or est.point is None:
        return {"available": False, "reason": "缺少实际值或点估计"}
    error = est.point - actual.value
    relative = _relative_error(est.point, actual.value)
    covered = None
    width = None
    if est.interval is not None:
        covered = est.interval[0] <= actual.value <= est.interval[1]
        width = est.interval[1] - est.interval[0]
    return {
        "available": True,
        "absolute": error,
        "relative": relative,
        "covered": covered,
        "width": width,
        "actual_release_id": actual.id,
        "actual_revision": actual.revision,
    }


def metrics(records: list[dict], spec: BacktestSpec) -> dict:
    result: dict = {
        "criterion": {},
        "by_horizon": {},
        "formal_only": {},
        "notes": [],
    }
    for criterion in ("first_release", "mature"):
        for model_name in spec.models:
            cells = []
            for row in records:
                model = row["models"].get(model_name)
                if not isinstance(model, dict):
                    continue
                entry = model.get(criterion)
                if not entry:
                    continue
                cells.append((row, entry, _eligibility(entry, spec)))
            result["criterion"].setdefault(criterion, {})[model_name] = _metric_block(
                cells
            )
    # 正式估计口径：只统计点估计状态为 ok 的记录，用于对比“何时敢报正式值”。
    for criterion in ("first_release", "mature"):
        for model_name in spec.models:
            cells = []
            for row in records:
                model = row["models"].get(model_name)
                if not isinstance(model, dict):
                    continue
                entry = model.get(criterion)
                if not entry or entry.get("status") != "ok":
                    continue
                cells.append((row, entry, "formal"))
            result["formal_only"].setdefault(criterion, {})[model_name] = (
                _metric_block(cells)
            )
    for criterion in ("first_release", "mature"):
        for bucket in HORIZONS:
            for model_name in spec.models:
                cells = []
                for row in records:
                    if row["horizon_bucket"] != bucket:
                        continue
                    model = row["models"].get(model_name)
                    if not isinstance(model, dict):
                        continue
                    entry = model.get(criterion)
                    if not entry:
                        continue
                    cells.append((row, entry, _eligibility(entry, spec)))
                block = _metric_block(cells)
                if block["n"] == 0 and not block["refused"]:
                    continue
                result["by_horizon"].setdefault(criterion, {}).setdefault(
                    bucket, {}
                )[model_name] = block
    # 每个提前期内部也报告宽度相对基线，便于看增益是否随提前期变化。
    for criterion, buckets in result["by_horizon"].items():
        for blocks in buckets.values():
            full = blocks.get("full")
            base = blocks.get("target_only")
            if full and base and full["avg_width"] and base["avg_width"]:
                full["width_ratio_to_baseline"] = full["avg_width"] / base["avg_width"]
                base["width_ratio_to_baseline"] = 1.0
    # 区间宽度相对基线；只在同口径、同记录集合上比较，避免挑样本。
    for criterion in ("first_release", "mature"):
        full = result["criterion"][criterion].get("full")
        base = result["criterion"][criterion].get("target_only")
        if full and base and full["avg_width"] and base["avg_width"]:
            full["width_ratio_to_baseline"] = full["avg_width"] / base["avg_width"]
            base["width_ratio_to_baseline"] = 1.0
    result["notes"].append(
        "正式估计口径（formal_only）只含 status=ok 的记录；宽区间与拒绝不计入覆盖率，"
        "但单独计数，避免用降级掩盖失败"
    )
    return result


def _eligibility(entry: dict, spec: BacktestSpec) -> str:
    if entry.get("status") == "observed_target" and spec.skip_observed_target:
        return "skipped_target_observed"
    if entry.get("status") == "refused":
        return "refused"
    if entry.get("status") == "wide":
        return "wide"
    if entry.get("status") in {"ok", "point_only"}:
        return "formal"
    return "other"


def _metric_block(cells: list[tuple[dict, dict, str]]) -> dict:
    block = {
        "n": 0,
        "refused": 0,
        "wide": 0,
        "skipped_target_observed": 0,
        "formal_n": 0,
        "mae": None,
        "rmse": None,
        "bias": None,
        "mape": None,
        "median_relative": None,
        "coverage": None,
        "coverage_n": 0,
        "avg_width": None,
        "width_ratio_to_baseline": None,
    }
    errors = []
    relatives = []
    cover = []
    widths = []
    for _row, entry, eligibility in cells:
        if eligibility == "refused":
            block["refused"] += 1
            continue
        if eligibility == "skipped_target_observed":
            block["skipped_target_observed"] += 1
            continue
        if eligibility == "wide":
            block["wide"] += 1
        if eligibility == "formal":
            block["formal_n"] += 1
        error = entry.get("error") or {}
        if not error.get("available"):
            continue
        block["n"] += 1
        errors.append(error["absolute"])
        if error.get("relative") is not None:
            relatives.append(error["relative"])
        if error.get("covered") is not None:
            cover.append(1.0 if error["covered"] else 0.0)
            if error.get("width") is not None:
                widths.append(error["width"])
    if errors:
        block["mae"] = mean(abs(e) for e in errors)
        block["rmse"] = math.sqrt(mean(e * e for e in errors))
        block["bias"] = mean(errors)
    if relatives:
        block["mape"] = mean(abs(r) for r in relatives)
        block["median_relative"] = median(relatives)
    if cover:
        block["coverage"] = sum(cover) / len(cover)
        block["coverage_n"] = len(cover)
    if widths:
        block["avg_width"] = mean(widths)
    return block


__all__ = [
    "DECOMPOSITION_FACTORS",
    "SEQUENTIAL_ORDER",
    "BacktestSpec",
    "build_decision_record",
    "release_calendar",
    "release_timeline",
    "release_versions",
    "actual_value",
    "shapley_decomposition",
    "backtest",
    "baseline_config",
    "metrics",
]
