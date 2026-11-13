"""混频、参差发布条件下的实时估计与区间模拟。

方法概要（细节见 docs/methodology.md）：

* **目标**：尚未发布的季度指标，用决策日可见的月度指标做桥接。
* **变换**：默认同比（YoY），由同比差分吸收季节；样本不足时退回环比
  并做加法季节调整。切换规则只看训练截止日可见的配对数量。
* **参差发布**：缺报不补成最终矩阵。月度缺口从缺口前最后一个可见月
  逐月外推；季度聚合只使用三个月齐全的季度（stock 只看季末月）。
  外推使用“去季节月增长 + AR(1) + 季节效应”，不使用缺口后的资料回填。
* **季度连接**：flow 求和、stock 取季末月、average 取三月均值。
* **训练截点**：季节效应、AR 系数、标准化参数、桥接系数、修订漂移
  全部由 ``fit_day`` 可见版本估计；决策日的资料视图可以晚于训练截点，
  以便把“资料更新”与“参数重估”分开。
* **区间**：Monte Carlo 合成锚定季度修订噪声、月度外推误差、桥接残差、
  参数不确定性与目标修订噪声。样本不足或结构变化时降级为宽区间，
  没有基期时拒绝正式估计。
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import mean, median, pstdev
from typing import Iterable

from . import linalg
from .periods import (
    horizon_days,
    is_month,
    parse_period,
    period_end,
    quarter_months,
    shift_period,
)
from .vintages import Release, SeriesMeta, visible_versions


@dataclass(frozen=True)
class Config:
    """模型与评估的冻结参数。

    默认值事前设定、不随回测期调整。``transform="auto"`` 只在训练截止日
    配对数量足够时选择同比；不参考任何目标结果。
    """

    transform: str = "auto"  # auto | qoq | yoy
    min_yoy_pairs: int = 12
    seasonal: bool = True
    seasonal_min_cycles: int = 3
    min_target_quarters: int = 8
    min_bridge_pairs: int = 8
    min_month_pairs: int = 12
    bridge_max_horizon: int = 1
    revision_maturity: int = 3
    interval_level: float = 0.80
    draws: int = 400
    seed: int = 20261114
    bridge: bool = True
    revision_adjust: bool = True
    structural_break_z: float = 3.0
    max_indicator_gap_months: int = 6
    missing_after_days: int = 120
    wide_inflation: float = 4.0
    min_months_visible: int = 3
    fallback_growth_sd: float = 0.05
    fallback_anchor_sd: float = 0.01
    fallback_revision_sd: float = 0.02

    def key(self) -> str:
        return (
            f"transform={self.transform};bridge={int(self.bridge)};"
            f"seasonal={int(self.seasonal)};rev={int(self.revision_adjust)};"
            f"maturity={self.revision_maturity};draws={self.draws};"
            f"seed={self.seed}"
        )


@dataclass
class ArFit:
    c: float
    phi: float
    sigma2: float
    n: int
    var_c: float
    var_phi: float
    cov_c_phi: float
    fallback: bool = False
    note: str = ""

    def covariance(self) -> list[list[float]]:
        return [[self.var_c, self.cov_c_phi], [self.cov_c_phi, self.var_phi]]


@dataclass
class BridgeFit:
    slope: float
    intercept: float
    sigma2: float
    n: int
    x_mean: float
    x_std: float
    sxx: float
    var_slope: float
    var_intercept: float
    cov: float
    pairs: list[tuple[str, float, float]] = field(default_factory=list)

    def prediction_variance(self, z: float) -> float:
        return self.sigma2 * (1 + 1 / self.n + z * z / self.sxx)

    def slope_variance(self, x_var: float) -> float:
        return self.var_slope * x_var / max(self.x_std ** 2, 1e-12)


@dataclass
class IndicatorParams:
    series: str
    meta: SeriesMeta
    mode: str
    growth_kind: str
    effects: dict[int, float] | None
    raw_levels: dict[str, float]
    growth: dict[str, float]
    ar: ArFit | None
    bridge: BridgeFit | None
    observed_quarters: list[str]
    flags: list[str] = field(default_factory=list)


@dataclass
class RevisionStats:
    drift: float
    sigma: float
    n: int
    pairs: list[dict] = field(default_factory=list)
    fallback: bool = True


@dataclass
class ParamSet:
    target: SeriesMeta
    fit_day: date
    config: Config
    growth_kind: str
    mode: str
    effects: dict[int, float] | None
    raw_levels: dict[str, float]
    growth: dict[str, float]
    ar: ArFit | None
    indicators: dict[str, IndicatorParams]
    revision: RevisionStats
    break_detected: bool
    flags: list[str]

    def summary(self) -> dict:
        return {
            "fit_day": self.fit_day.isoformat(),
            "growth_kind": self.growth_kind,
            "target_mode": self.mode,
            "seasonal_effects": (
                {str(k): round(v, 6) for k, v in self.effects.items()}
                if self.effects
                else None
            ),
            "target_growth_n": len(self.growth),
            "target_ar": _ar_summary(self.ar),
            "structural_break": self.break_detected,
            "revision": {
                "drift": self.revision.drift,
                "sigma": self.revision.sigma,
                "n": self.revision.n,
                "fallback": self.revision.fallback,
            },
            "indicators": {
                sid: {
                    "mode": p.mode,
                    "growth_kind": p.growth_kind,
                    "seasonal": bool(p.effects),
                    "raw_months": len(p.raw_levels),
                    "growth_n": len(p.growth),
                    "ar": _ar_summary(p.ar),
                    "bridge": (
                        {
                            "slope": p.bridge.slope,
                            "intercept": p.bridge.intercept,
                            "n": p.bridge.n,
                            "sigma2": p.bridge.sigma2,
                            "x_mean": p.bridge.x_mean,
                            "x_std": p.bridge.x_std,
                        }
                        if p.bridge
                        else None
                    ),
                    "aggregation": p.meta.aggregation,
                    "flags": p.flags,
                }
                for sid, p in self.indicators.items()
            },
            "flags": self.flags,
        }


def _ar_summary(ar: ArFit | None) -> dict | None:
    if ar is None:
        return None
    return {
        "c": ar.c,
        "phi": ar.phi,
        "sigma2": ar.sigma2,
        "n": ar.n,
        "fallback": ar.fallback,
        "note": ar.note,
    }


# ---------------------------------------------------------------- 基础变换


def _growth(prev: float, cur: float, mode: str) -> float:
    if mode == "log":
        return math.log(cur) - math.log(prev)
    return cur - prev


def _level(base: float, growth: float, mode: str) -> float:
    if mode == "log":
        return base * math.exp(growth)
    return base + growth


def _period_index(period: str) -> int:
    kind, year, index = parse_period(period)
    return year * (4 if kind == "quarter" else 12) + index


def _quarter_index(period: str) -> int:
    _, year, index = parse_period(period)
    return year * 4 + index - 1


def _month_index(period: str) -> int:
    _, year, month = parse_period(period)
    return year * 12 + month - 1


def _season_index(period: str) -> int:
    _, _, index = parse_period(period)
    return index


def _raw_growth(levels: dict[str, float], lag: int, mode: str) -> dict[str, float]:
    """只对相邻且两期都有值的观测计算增长率，缺口不插值。"""
    result = {}
    for period in sorted(levels, key=_period_index):
        previous = shift_period(period, -lag)
        if previous in levels:
            result[period] = _growth(levels[previous], levels[period], mode)
    return result


def _pair_count(levels: dict[str, float], lag: int) -> int:
    return sum(1 for p in levels if shift_period(p, -lag) in levels)


def _choose_growth_kind(
    levels: dict[str, float], config: Config
) -> tuple[str, list[str]]:
    flags: list[str] = []
    if config.transform == "yoy":
        kind = "yoy"
    elif config.transform == "qoq":
        kind = "qoq"
    elif config.transform == "auto":
        kind = "yoy" if _pair_count(levels, 4) >= config.min_yoy_pairs else "qoq"
    else:
        raise ValueError(f"未知的 transform: {config.transform}")
    if kind == "yoy":
        if _pair_count(levels, 4) < 4:
            flags.append("同比可用配对不足 4 个，退回环比并做季节调整")
            kind = "qoq"
        else:
            flags.append("采用同比变换，季节影响由同比差分吸收")
    return kind, flags


def _seasonal_effects(
    growth: dict[str, float], min_cycles: int
) -> dict[int, float] | None:
    """估算加法季节效应：同月/同季平均增长率减去总体平均增长率。

    只使用训练截止日可见的相邻增长率；每个季节位置至少 min_cycles 个
    观测才估计，否则返回 None（不做季节调整）。中心化后均值为 0。
    季节效应作用在增长率上，不作用在水平上，避免趋势污染因子。
    """
    if not growth:
        return None
    groups: dict[int, list[float]] = {}
    for period, value in growth.items():
        groups.setdefault(_season_index(period), []).append(value)
    width = 12 if any(is_month(p) for p in growth) else 4
    if len(groups) != width or any(
        len(values) < min_cycles for values in groups.values()
    ):
        return None
    overall = mean(v for values in groups.values() for v in values)
    effects = {i: mean(values) - overall for i, values in groups.items()}
    centre = mean(effects.values())
    return {i: value - centre for i, value in effects.items()}


def _effect_for(period: str, effects: dict[int, float] | None) -> float:
    if not effects:
        return 0.0
    return effects.get(_season_index(period), 0.0)


def _deseasonal_growth(
    growth: dict[str, float], effects: dict[int, float] | None
) -> dict[str, float]:
    if not effects:
        return dict(growth)
    return {p: g - _effect_for(p, effects) for p, g in growth.items()}


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - 1)


def _fit_ar(
    growth: dict[str, float], periods: list[str], min_pairs: int
) -> ArFit | None:
    """AR(1)：只用相邻且都有值的期。"""
    xs, ys = [], []
    for period in periods:
        if period not in growth:
            continue
        previous = shift_period(period, -1)
        if previous in growth:
            xs.append(growth[previous])
            ys.append(growth[period])
    n = len(xs)
    if n < 2:
        return None
    xbar, ybar = mean(xs), mean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return ArFit(ybar, 0.0, _variance(ys), n, 0.0, 0.0, 0.0, True, "自变量无变化")
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    intercept = ybar - slope * xbar
    residuals = [y - intercept - slope * x for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    sigma2 = sum(r * r for r in residuals) / dof
    var_slope = sigma2 / sxx
    var_intercept = sigma2 * (1 / n + xbar * xbar / sxx)
    cov = -sigma2 * xbar / sxx
    fallback = n < min_pairs
    note = "" if not fallback else f"样本仅 {n} 对，低于 {min_pairs}"
    phi = max(min(slope, 0.98), -0.98)
    return ArFit(
        intercept, phi, sigma2, n, var_intercept, var_slope, cov, fallback, note
    )


def _fit_bridge(
    target_growth: dict[str, float],
    x_growth: dict[str, float],
    min_pairs: int,
) -> BridgeFit | None:
    """标准化后的单变量桥接回归：target = a + b·z(x)。"""
    pairs = [
        (period, x_growth[period], target_growth[period])
        for period in sorted(target_growth, key=_period_index)
        if period in x_growth
    ]
    if len(pairs) < max(3, min_pairs):
        return None
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    xbar = mean(xs)
    ybar = mean(ys)
    x_std = pstdev(xs) if len(xs) > 1 else 0.0
    if x_std <= 1e-12:
        return None
    zs = [(x - xbar) / x_std for x in xs]
    sxx = sum(z * z for z in zs)
    if sxx <= 1e-12:
        return None
    slope = sum(z * (y - ybar) for z, y in zip(zs, ys)) / sxx
    residuals = [y - ybar - slope * z for z, y in zip(zs, ys)]
    dof = max(len(pairs) - 2, 1)
    sigma2 = sum(r * r for r in residuals) / dof
    return BridgeFit(
        slope=slope,
        intercept=ybar,
        sigma2=sigma2,
        n=len(pairs),
        x_mean=xbar,
        x_std=x_std,
        sxx=sxx,
        var_slope=sigma2 / sxx,
        var_intercept=sigma2 / len(pairs),
        cov=0.0,
        pairs=pairs,
    )


def _quarters_covering(levels: dict[str, float]) -> list[str]:
    quarters = {
        f"{period[:4]}-Q{(int(period[5:]) - 1) // 3 + 1}"
        for period in levels
        if is_month(period)
    }
    return sorted(quarters, key=_quarter_index)


def _quarterly_aggregate(
    levels: dict[str, float], quarter: str, rule: str
) -> float | None:
    months = quarter_months(quarter)
    if rule == "stock":
        return levels.get(months[-1])
    values = [levels.get(m) for m in months]
    if any(v is None for v in values):
        return None
    if rule == "flow":
        return sum(values)
    return mean(values)


def _fit_revision(
    releases: Iterable[Release],
    target_series: str,
    config: Config,
) -> RevisionStats:
    """用训练截止日可见的修订历史估计首次发布→成熟版本的漂移与噪声。

    必须使用完整版本记录（同一参考期的多个 revision），不能用
    ``visible_versions`` 的单版本视图，否则永远凑不出首次→成熟配对。
    """
    by_period: dict[str, dict[int, Release]] = {}
    for rel in releases:
        if rel.series == target_series:
            by_period.setdefault(rel.period, {})[rel.revision] = rel
    pairs = []
    for period, versions in sorted(
        by_period.items(), key=lambda kv: _quarter_index(kv[0])
    ):
        first = versions.get(1)
        mature = versions.get(config.revision_maturity)
        if first is None or mature is None or mature.revision == first.revision:
            continue
        if first.value == 0:
            continue
        pairs.append(
            {
                "period": period,
                "first": first.value,
                "mature": mature.value,
                "first_release_id": first.id,
                "mature_release_id": mature.id,
                "relative": mature.value / first.value - 1.0,
            }
        )
    if not pairs:
        return RevisionStats(0.0, config.fallback_revision_sd, 0, [], True)
    relatives = [p["relative"] for p in pairs]
    centre = median(relatives)
    spread = median([abs(v - centre) for v in relatives]) * 1.4826
    if spread <= 0:
        spread = (
            pstdev(relatives) if len(relatives) > 1 else config.fallback_revision_sd
        )
    return RevisionStats(centre, max(spread, 1e-4), len(pairs), pairs, False)


def _detect_break(growth: dict[str, float], z_limit: float) -> bool:
    """训练截止日的两段均值漂移检验。

    至少 12 个增长观测、两段各至少 6 个才检验；尺度用全样本 MAD，
    避免把一次大修订误判成结构变化。规则事前设定。
    """
    values = [growth[p] for p in sorted(growth, key=_period_index)]
    if len(values) < 12:
        return False
    half = len(values) // 2
    first, second = values[:half], values[half:]
    if len(first) < 6 or len(second) < 6:
        return False
    centre = median(values)
    mad = median([abs(v - centre) for v in values]) * 1.4826
    if mad <= 1e-12:
        return False
    se = mad * math.sqrt(1 / len(first) + 1 / len(second))
    return abs(mean(second) - mean(first)) > z_limit * se


# ---------------------------------------------------------------- 参数拟合


def fit_params(
    metas: dict[str, SeriesMeta],
    releases: Iterable[Release] | dict[tuple[str, str], Release],
    fit_day: date,
    target_series: str,
    config: Config,
    *,
    inclusive: bool = True,
) -> ParamSet:
    """只用训练截止日可见版本拟合全部参数。

    ``releases`` 可以是完整发布记录（推荐：修订历史可同时看到首次与成熟
    版本），也可以是已裁剪的 ``(series, period) -> Release`` 视图。
    """
    target_meta = metas[target_series]
    raw_releases = (
        list(releases.values()) if isinstance(releases, dict) else list(releases)
    )
    view = visible_versions(raw_releases, fit_day, inclusive=inclusive)
    visible_raw = [
        r
        for r in raw_releases
        if (r.released <= fit_day if inclusive else r.released < fit_day)
    ]
    flags: list[str] = []
    target_raw = {
        period: rel.value
        for (sid, period), rel in view.items()
        if sid == target_series
    }
    if not target_raw:
        raise ValueError(f"训练截止日没有目标 {target_series} 的可见值")
    growth_kind, transform_flags = _choose_growth_kind(target_raw, config)
    flags.extend(transform_flags)
    mode = "log" if all(v > 0 for v in target_raw.values()) else "diff"
    lag = 1 if growth_kind == "qoq" else 4
    raw_growth = _raw_growth(target_raw, lag, mode)

    if config.seasonal and growth_kind == "qoq":
        effects = _seasonal_effects(raw_growth, config.seasonal_min_cycles)
        if effects is None:
            flags.append("目标季节效应样本不足，未做季节调整")
    else:
        effects = None
    target_growth = _deseasonal_growth(raw_growth, effects)
    ar = _fit_ar(target_growth, sorted(target_raw), config.min_bridge_pairs)
    if ar is None:
        flags.append("目标自身历史不足，无法拟合 AR(1)")
    elif ar.fallback:
        flags.append(ar.note)

    indicators: dict[str, IndicatorParams] = {}
    for sid, meta in metas.items():
        if sid == target_series or meta.frequency != "month":
            continue
        raw = {
            period: rel.value
            for (series, period), rel in view.items()
            if series == sid
        }
        ind_flags: list[str] = []
        ind_mode = "log" if raw and all(v > 0 for v in raw.values()) else "diff"
        monthly_raw_growth = _raw_growth(raw, 1, ind_mode)
        # 季节效应只在环比口径下需要；同比口径由差分吸收季节。
        if config.seasonal and growth_kind == "qoq":
            ind_effects = _seasonal_effects(
                monthly_raw_growth, config.seasonal_min_cycles
            )
            if ind_effects is None:
                ind_flags.append("月度季节效应样本不足，未做季节调整")
        else:
            ind_effects = None
        ind_growth = _deseasonal_growth(monthly_raw_growth, ind_effects)
        monthly_ar = _fit_ar(ind_growth, sorted(raw), config.min_month_pairs)
        if monthly_ar is None:
            ind_flags.append("月度自身历史不足，缺报只能做随机游走外推")
        elif monthly_ar.fallback:
            ind_flags.append("月度 AR 样本偏少：" + monthly_ar.note)
        aggregates = {}
        for quarter in _quarters_covering(raw):
            value = _quarterly_aggregate(raw, quarter, meta.aggregation)
            if value is not None:
                aggregates[quarter] = value
        x_growth = _raw_growth(aggregates, lag, ind_mode)
        bridge = (
            _fit_bridge(target_growth, x_growth, config.min_bridge_pairs)
            if config.bridge
            else None
        )
        if config.bridge and bridge is None:
            ind_flags.append(
                f"与目标的完整季度配对不足 {config.min_bridge_pairs}，桥接不可识别"
            )
        indicators[sid] = IndicatorParams(
            series=sid,
            meta=meta,
            mode=ind_mode,
            growth_kind=growth_kind,
            effects=ind_effects,
            raw_levels=raw,
            growth=ind_growth,
            ar=monthly_ar,
            bridge=bridge,
            observed_quarters=sorted(aggregates, key=_quarter_index),
            flags=ind_flags,
        )

    revision = _fit_revision(visible_raw, target_series, config)
    if revision.fallback:
        flags.append("没有可见的目标首次→成熟修订历史，成熟版本区间放宽")
    break_detected = _detect_break(target_growth, config.structural_break_z)
    if break_detected:
        flags.append("目标历史检测到结构变化，区间放大且桥接降权")
    if config.bridge and not any(p.bridge for p in indicators.values()):
        flags.append("没有可用的桥接指标，退化为目标自身历史基线")
    return ParamSet(
        target=target_meta,
        fit_day=fit_day,
        config=config,
        growth_kind=growth_kind,
        mode=mode,
        effects=effects,
        raw_levels=target_raw,
        growth=target_growth,
        ar=ar,
        indicators=indicators,
        revision=revision,
        break_detected=break_detected,
        flags=flags,
    )


# ---------------------------------------------------------------- 估计


@dataclass
class Component:
    name: str
    weight: float
    growth: float | None
    variance: float | None
    kind: str
    detail: dict = field(default_factory=dict)


@dataclass
class Estimate:
    target_series: str
    target_period: str
    day: date
    status: str
    formal: bool
    flags: list[str]
    point: float | None
    interval: tuple[float, float] | None
    growth_point: float | None
    growth_sd: float | None
    components: list[Component]
    target_state: str
    measure: str
    horizon_quarters: int | None
    last_observed_period: str | None
    months: dict[str, dict[str, str]]
    inputs_digest: str
    input_counts: dict[str, int]
    used_releases: list[dict]
    params: ParamSet

    def to_dict(self, include_inputs: bool = True) -> dict:
        payload = {
            "target": f"{self.target_series}:{self.target_period}",
            "day": self.day.isoformat(),
            "status": self.status,
            "formal": self.formal,
            "target_state": self.target_state,
            "measure": self.measure,
            "point": self.point,
            "interval": list(self.interval) if self.interval else None,
            "interval_level": self.params.config.interval_level,
            "growth_point": self.growth_point,
            "growth_sd": self.growth_sd,
            "horizon_quarters": self.horizon_quarters,
            "horizon_days": horizon_days(self.day, self.target_period),
            "last_observed_period": self.last_observed_period,
            "flags": self.flags,
            "components": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "weight": c.weight,
                    "growth": c.growth,
                    "variance": c.variance,
                    "detail": c.detail,
                }
                for c in self.components
            ],
            "months": self.months,
            "inputs_digest": self.inputs_digest,
            "input_counts": self.input_counts,
            "params": self.params.summary(),
        }
        if include_inputs:
            payload["used_releases"] = self.used_releases
        return payload


def _vintage_digest(view: dict[tuple[str, str], Release]) -> str:
    digest = hashlib.sha256()
    for key in sorted(view):
        rel = view[key]
        digest.update(
            f"{rel.series}|{rel.period}|{rel.released.isoformat()}|"
            f"{rel.revision}|{rel.value!r}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _used_releases(view: dict[tuple[str, str], Release]) -> list[dict]:
    return [
        {
            "series": rel.series,
            "period": rel.period,
            "release_id": rel.id,
            "revision": rel.revision,
            "released": rel.released.isoformat(),
            "value": rel.value,
        }
        for _, rel in sorted(
            view.items(), key=lambda kv: (kv[0][0], _period_index(kv[0][1]))
        )
    ]


def _monthly_extrapolation_var(
    params: IndicatorParams, data_levels: dict[str, float], month: str
) -> float:
    """单月外推方差；已可见月份为 0。

    缺口可以出现在样本中间（整列长期缺报之后再恢复发布），因此以该月
    之前最近的可见月为锚，而不是全局最后可见月。
    """
    if month in data_levels or params.ar is None:
        return 0.0
    anchor = _last_visible_before(data_levels, month)
    if anchor is None:
        return 0.0
    steps = _month_index(month) - _month_index(anchor)
    if steps <= 0:
        return 0.0
    phi = params.ar.phi
    return max(params.ar.sigma2, 0.0) * sum(phi ** (2 * j) for j in range(steps))


def _aggregate_variance(
    params: IndicatorParams,
    data_levels: dict[str, float],
    quarter: str,
) -> float:
    """按聚合规则把单月外推方差合成为季度方差。"""
    months = quarter_months(quarter)
    variances = [_monthly_extrapolation_var(params, data_levels, m) for m in months]
    if params.meta.aggregation == "stock":
        return variances[-1]
    if params.meta.aggregation == "average":
        return sum(variances) / 9.0
    return sum(variances)


def _baseline_forecast(
    params: ParamSet,
    data_raw: dict[str, float],
    target_period: str,
) -> tuple[dict | None, list[str]]:
    """目标自身历史基线；返回累计原始增长（含季节效应）与方差。"""
    kind = params.growth_kind
    mode = params.mode
    visible = sorted(
        (p for p in data_raw if _quarter_index(p) < _quarter_index(target_period)),
        key=_quarter_index,
    )
    if not visible:
        return None, ["目标没有早于目标季度的可见历史"]
    last = visible[-1]
    h = _quarter_index(target_period) - _quarter_index(last)
    if kind == "yoy":
        base_period = shift_period(target_period, -4)
        if base_period not in data_raw:
            return None, [f"同比变换需要 {base_period} 的可见目标值，但该期不可见"]
        online = [target_period]
        steps = h
    else:
        base_period = last
        online = [shift_period(last, k) for k in range(1, h + 1)]
        steps = h
    base_level = data_raw[base_period]
    effects = params.effects if kind == "qoq" else None
    g0 = params.growth.get(last)
    if g0 is None:
        g0 = (
            params.growth[sorted(params.growth, key=_period_index)[-1]]
            if params.growth
            else 0.0
        )
    point, gradient = _ar_forecast(params.ar, g0, steps, kind)
    point += sum(_effect_for(p, effects) for p in online)
    variance = _ar_variance(params.ar, gradient, steps, kind, params.config)
    anchor_sigma = (
        params.revision.sigma
        if not params.revision.fallback
        else params.config.fallback_anchor_sd
    )
    if mode == "log":
        anchor_var = anchor_sigma ** 2
    else:
        anchor_var = (anchor_sigma * abs(base_level)) ** 2
    return (
        {
            "last": last,
            "base_period": base_period,
            "base_level": base_level,
            "h": h,
            "steps": steps,
            "growth": point,
            # 分量方差只含增长过程误差；锚定季度修订噪声由模拟单独加入，
            # 它同等地作用于每个分量，不应影响分量权重。
            "variance": variance,
            "g0": g0,
            "gradient": gradient,
            "anchor_sigma": anchor_sigma,
            "anchor_var": anchor_var,
            "online_periods": online,
            "seasonal_sum": sum(_effect_for(p, effects) for p in online),
        },
        [],
    )


def _ar_point(c: float, phi: float, g0: float, steps: int, kind: str) -> float:
    if kind == "yoy":
        value = g0
        for _ in range(steps):
            value = c + phi * value
        return value
    total = 0.0
    value = g0
    for _ in range(steps):
        value = c + phi * value
        total += value
    return total


def _ar_forecast(
    ar: ArFit | None, g0: float, steps: int, kind: str
) -> tuple[float, list[float]]:
    if ar is None:
        return 0.0, [0.0, 0.0]
    point = _ar_point(ar.c, ar.phi, g0, steps, kind)
    eps = 1e-5
    d_c = (
        _ar_point(ar.c + eps, ar.phi, g0, steps, kind)
        - _ar_point(ar.c - eps, ar.phi, g0, steps, kind)
    ) / (2 * eps)
    d_phi = (
        _ar_point(ar.c, ar.phi + eps, g0, steps, kind)
        - _ar_point(ar.c, ar.phi - eps, g0, steps, kind)
    ) / (2 * eps)
    return point, [d_c, d_phi]


def _ar_variance(
    ar: ArFit | None,
    gradient: list[float],
    steps: int,
    kind: str,
    config: Config,
) -> float:
    if ar is None:
        return (config.fallback_growth_sd ** 2) * (steps if kind == "qoq" else 1)
    phi = ar.phi
    if kind == "yoy":
        innovation = sum(ar.sigma2 * phi ** (2 * j) for j in range(steps))
    else:
        innovation = sum(
            ar.sigma2 * sum(phi ** j for j in range(k)) ** 2
            for k in range(1, steps + 1)
        )
    covariance = ar.covariance()
    parameter = sum(
        gradient[i] * covariance[i][j] * gradient[j]
        for i in range(2)
        for j in range(2)
    )
    # 样本不足时由 estimate() 标记 wide，并由模拟统一放大；此处不重复放大。
    return max(innovation + parameter, 1e-10)


def _simulate_monthly(
    params: IndicatorParams,
    data_levels: dict[str, float],
    needed: list[str],
    *,
    rng: random.Random | None = None,
    sample_params: bool = True,
) -> dict[str, float] | None:
    """返回 needed 各月水平：可见值，或从最后可见月逐月外推。

    外推公式（对数或差分口径下形式一致）：
        g_next = c + phi · g_deseason(上一月) + season(next) + shock
        level_next = level_last · exp(g_next)（log）或 level_last + g_next
    缺口之后的观测不会被用来回填缺口；每个缺口月只抽一次新息。
    """
    if not data_levels:
        return None
    ar = params.ar
    c = ar.c if ar else 0.0
    phi = ar.phi if ar else 0.0
    sigma = math.sqrt(max(ar.sigma2, 0.0)) if ar else 0.0
    if rng is not None and sample_params and ar is not None and ar.n >= 4:
        c, phi = linalg.sample_normal_vector(rng, [ar.c, ar.phi], ar.covariance())
        phi = max(min(phi, 0.98), -0.98)
    wanted = sorted(set(needed), key=_month_index)
    first_wanted = wanted[0]
    # 只从最早需要月份的锚点开始推进，避免重复模拟整段历史。
    anchor = _last_visible_before(data_levels, first_wanted)
    if anchor is None and first_wanted not in data_levels:
        return None
    start = anchor or first_wanted
    end = wanted[-1]
    values: dict[str, float] = {}
    month = start
    while _month_index(month) <= _month_index(end):
        if month in data_levels:
            values[month] = data_levels[month]
        else:
            previous = shift_period(month, -1)
            base = values.get(previous)
            if base is None:
                return None
            g_prev = _growth_from_levels(values, previous, params.mode)
            if g_prev is None:
                g_prev = 0.0
            if params.effects:
                g_prev -= _effect_for(previous, params.effects)
            season = _effect_for(month, params.effects) if params.effects else 0.0
            shock = rng.gauss(0.0, sigma) if (rng is not None and sigma > 0) else 0.0
            try:
                values[month] = _level(
                    base, c + phi * g_prev + season + shock, params.mode
                )
            except (ValueError, OverflowError):
                return None
        month = shift_period(month, 1)
    return values if all(m in values for m in wanted) else None


def _growth_from_levels(
    levels: dict[str, float], period: str, mode: str
) -> float | None:
    previous = shift_period(period, -1)
    if period in levels and previous in levels:
        try:
            return _growth(levels[previous], levels[period], mode)
        except ValueError:
            return None
    return None


def _last_visible_before(levels: dict[str, float], month: str) -> str | None:
    candidates = [p for p in levels if _month_index(p) < _month_index(month)]
    return max(candidates, key=_month_index) if candidates else None


def _bridge_component(
    params: ParamSet,
    indicator: IndicatorParams,
    view: dict[tuple[str, str], Release],
    target_period: str,
    h: int,
) -> tuple[Component | None, list[str]]:
    config = params.config
    if not config.bridge:
        return None, []
    if h > config.bridge_max_horizon:
        return None, [
            f"{indicator.series} 桥接仅支持 {config.bridge_max_horizon} 步内，当前 {h} 步"
        ]
    bridge = indicator.bridge
    if bridge is None:
        return None, []
    data_levels = {
        period: rel.value
        for (series, period), rel in view.items()
        if series == indicator.series
    }
    if len(data_levels) < config.min_months_visible:
        return None, [
            f"{indicator.series} 决策日可见月度资料不足 {config.min_months_visible} 个"
        ]
    last_month = max(data_levels, key=_month_index)
    gap = _month_index(quarter_months(target_period)[-1]) - _month_index(last_month)
    if gap > config.max_indicator_gap_months:
        return None, [
            f"{indicator.series} 最新可见月份比目标季度末早 {gap} 个月，超过缺口上限"
        ]
    lag = 4 if params.growth_kind == "yoy" else 1
    other_period = shift_period(target_period, -lag)
    needed_months = [
        m for q in (other_period, target_period) for m in quarter_months(q)
    ]
    values = _simulate_monthly(indicator, data_levels, needed_months)
    if values is None:
        return None, [f"{indicator.series} 缺少外推锚点，无法形成季度聚合"]
    target_agg = _quarterly_aggregate(values, target_period, indicator.meta.aggregation)
    other_agg = _quarterly_aggregate(values, other_period, indicator.meta.aggregation)
    if target_agg is None or other_agg is None:
        return None, [f"{indicator.series} 季度聚合不完整"]
    try:
        x_growth = _growth(other_agg, target_agg, indicator.mode)
    except ValueError:
        return None, [f"{indicator.series} 变换定义域不合法"]
    z = (x_growth - bridge.x_mean) / bridge.x_std
    comp_growth = bridge.intercept + bridge.slope * z
    x_var = _aggregate_variance(indicator, data_levels, target_period) + (
        _aggregate_variance(indicator, data_levels, other_period)
        if params.growth_kind == "qoq"
        else 0.0
    )
    variance = bridge.prediction_variance(z) + bridge.slope_variance(x_var)
    return (
        Component(
            name=f"bridge:{indicator.series}",
            kind="bridge",
            weight=0.0,
            growth=comp_growth,
            variance=max(variance, 1e-10),
            detail={
                "x_growth": x_growth,
                "z": z,
                "n_pairs": bridge.n,
                "slope": bridge.slope,
                "gap_months": gap,
                "aggregation": indicator.meta.aggregation,
                "x_var": x_var,
                "last_month": last_month,
            },
        ),
        [],
    )


def estimate(
    params: ParamSet,
    view: dict[tuple[str, str], Release],
    target_period: str,
    day: date,
    *,
    measure: str = "first_release",
    simulate: bool = True,
    include_inputs: bool = True,
) -> Estimate:
    """在决策日 ``day`` 用可见版本 ``view`` 估计目标季度。

    ``params`` 可来自更早（或同一）训练截止日，从而把“资料变了”与
    “参数重新估计了”分开。``view`` 必须已经是决策日可见的版本集合。
    """
    if measure not in {"first_release", "mature"}:
        raise ValueError(f"未知的 measure: {measure}")
    config = params.config
    target_series = params.target.id
    flags = list(params.flags)
    target_cells = {
        period: rel for (sid, period), rel in view.items() if sid == target_series
    }
    data_raw = {p: rel.value for p, rel in target_cells.items()}
    digest = _vintage_digest(view)
    counts: dict[str, int] = {}
    for (sid, _), _rel in view.items():
        counts[sid] = counts.get(sid, 0) + 1
    months = _month_states(params, view, target_period, day)
    target_state = _target_state(target_cells, target_period, day, config)

    if target_state == "observed":
        rel = target_cells[target_period]
        return Estimate(
            target_series=target_series,
            target_period=target_period,
            day=day,
            status="observed_target",
            formal=False,
            flags=_dedupe(flags + ["目标季度已有可见发布，估计转为观测值"]),
            point=rel.value,
            interval=(rel.value, rel.value),
            growth_point=None,
            growth_sd=0.0,
            components=[],
            target_state=target_state,
            measure=measure,
            horizon_quarters=0,
            last_observed_period=target_period,
            months=months,
            inputs_digest=digest,
            input_counts=counts,
            used_releases=_used_releases(view),
            params=params,
        )

    baseline, notes = _baseline_forecast(params, data_raw, target_period)
    flags.extend(notes)
    if baseline is None:
        return _refused(
            params, target_period, day, measure, flags, digest, counts,
            months, target_state, view,
        )
    h = baseline["h"]
    wide = False
    if params.ar is None:
        wide = True
        flags.append("目标自身历史不足，基线方差取宽泛先验")
    elif params.ar.fallback:
        wide = True
    if params.break_detected:
        wide = True
        flags.append("结构变化提示：区间按宽口径放大，桥接权重减半")
    if len(params.growth) < config.min_target_quarters:
        wide = True
        flags.append(
            f"目标增长观测仅 {len(params.growth)} 个，低于正式估计门槛 "
            f"{config.min_target_quarters}"
        )
    components = [
        Component(
            name="target_ar",
            kind="baseline",
            weight=0.0,
            growth=baseline["growth"],
            variance=baseline["variance"],
            detail={
                "h": h,
                "last": baseline["last"],
                "base_period": baseline["base_period"],
                "g0": baseline["g0"],
                "phi": params.ar.phi if params.ar else None,
                "growth_kind": params.growth_kind,
                "anchor_sigma": baseline["anchor_sigma"],
                "seasonal_sum": baseline["seasonal_sum"],
            },
        )
    ]
    bridge_notes: list[str] = []
    if config.bridge:
        for indicator in params.indicators.values():
            component, notes = _bridge_component(
                params, indicator, view, target_period, h
            )
            bridge_notes.extend(notes)
            if component is not None:
                components.append(component)
    flags.extend(bridge_notes)

    valid = [c for c in components if c.growth is not None and c.variance is not None]
    bridges = [c for c in valid if c.kind == "bridge"]
    if not bridges:
        flags.append("无可用桥接分量，仅使用目标自身历史")
    if len(valid) == 1:
        valid[0].weight = 1.0
    else:
        variances = [c.variance for c in valid]
        floor = 0.05 * median(variances)
        raw_weights = []
        for component in valid:
            weight = 1.0 / (component.variance + floor)
            if component.kind == "bridge" and params.break_detected:
                weight *= 0.5
            raw_weights.append(weight)
        total = sum(raw_weights)
        for component, weight in zip(valid, raw_weights):
            component.weight = weight / total

    growth_point = sum(c.weight * c.growth for c in components)
    point = _level(baseline["base_level"], growth_point, params.mode)
    point, revision_note = _apply_revision(point, params.revision, config, measure)
    if revision_note:
        flags.append(revision_note)

    interval = None
    growth_sd = None
    if simulate:
        draws = _simulate(
            params=params,
            view=view,
            baseline=baseline,
            components=components,
            target_period=target_period,
            measure=measure,
            config=config,
            wide=wide,
        )
        if draws:
            draws.sort()
            lo = linalg.quantile(draws, (1 - config.interval_level) / 2)
            hi = linalg.quantile(draws, 1 - (1 - config.interval_level) / 2)
            interval = (lo, hi)
            growth_sd = pstdev(draws) if len(draws) > 1 else None
    if wide:
        flags.append("区间按宽口径放大，不视为正式估计")
    status = "wide" if wide and simulate else ("ok" if simulate else "point_only")
    return Estimate(
        target_series=target_series,
        target_period=target_period,
        day=day,
        status=status,
        formal=status == "ok",
        flags=_dedupe(flags),
        point=point,
        interval=interval,
        growth_point=growth_point,
        growth_sd=growth_sd,
        components=components,
        target_state=target_state,
        measure=measure,
        horizon_quarters=h,
        last_observed_period=baseline["last"],
        months=months,
        inputs_digest=digest,
        input_counts=counts,
        used_releases=_used_releases(view) if include_inputs else [],
        params=params,
    )


def _target_state(
    target_cells: dict[str, Release],
    target_period: str,
    day: date,
    config: Config,
) -> str:
    if target_period in target_cells:
        return (
            "reported_zero" if target_cells[target_period].value == 0 else "observed"
        )
    end = period_end(target_period)
    if end > day:
        return "future"
    if day > end + timedelta(days=config.missing_after_days):
        return "missing"
    return "pending"


def _month_states(
    params: ParamSet,
    view: dict[tuple[str, str], Release],
    target_period: str,
    day: date,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for sid in params.indicators:
        row = {}
        for month in quarter_months(target_period):
            rel = view.get((sid, month))
            if rel is not None:
                row[month] = "reported_zero" if rel.value == 0 else "observed"
            elif period_end(month) > day:
                row[month] = "future"
            elif day > period_end(month) + timedelta(
                days=params.config.missing_after_days
            ):
                row[month] = "missing"
            else:
                row[month] = "pending"
        result[sid] = row
    return result


def _apply_revision(
    point: float, revision: RevisionStats, config: Config, measure: str
) -> tuple[float, str]:
    """把“潜在成熟值”换算到指定评估口径。

    模型把潜在值视为成熟版本：成熟口径不加漂移；首次发布口径除以
    (1+drift)，drift 为训练期估计的“成熟/首次−1”。没有修订历史时
    按无漂移处理，噪声在模拟中放宽。
    """
    if not config.revision_adjust or revision.fallback:
        note = (
            "没有可见修订历史，首次/成熟口径按无漂移处理且区间放宽"
            if revision.fallback
            else ""
        )
        return point, note
    if measure == "first_release":
        return point / (1 + revision.drift), ""
    return point, ""


def _dedupe(items: list[str]) -> list[str]:
    seen, result = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _refused(
    params: ParamSet,
    target_period: str,
    day: date,
    measure: str,
    flags: list[str],
    digest: str,
    counts: dict[str, int],
    months: dict[str, dict[str, str]],
    target_state: str,
    view: dict[tuple[str, str], Release],
) -> Estimate:
    return Estimate(
        target_series=params.target.id,
        target_period=target_period,
        day=day,
        status="refused",
        formal=False,
        flags=_dedupe(flags + ["没有可用基期，停止正式估计"]),
        point=None,
        interval=None,
        growth_point=None,
        growth_sd=None,
        components=[],
        target_state=target_state,
        measure=measure,
        horizon_quarters=None,
        last_observed_period=None,
        months=months,
        inputs_digest=digest,
        input_counts=counts,
        used_releases=_used_releases(view),
        params=params,
    )


# ---------------------------------------------------------------- 模拟


def _simulate(
    *,
    params: ParamSet,
    view: dict[tuple[str, str], Release],
    baseline: dict,
    components: list[Component],
    target_period: str,
    measure: str,
    config: Config,
    wide: bool,
) -> list[float]:
    seed_material = f"{_vintage_digest(view)}|{target_period}|{measure}|{config.seed}"
    rng = random.Random(
        int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    )
    inflation = config.wide_inflation if wide else 1.0
    bridge_components = [
        c for c in components if c.kind == "bridge" and c.weight > 0
    ]
    indicator_data = {
        c.name.split(":", 1)[1]: (
            params.indicators[c.name.split(":", 1)[1]],
            {
                period: rel.value
                for (series, period), rel in view.items()
                if series == c.name.split(":", 1)[1]
            },
        )
        for c in bridge_components
    }
    draws: list[float] = []
    for _ in range(config.draws):
        sampled = {"target_ar": _draw_baseline(params, baseline, inflation, rng)}
        anchor_noise = (
            rng.gauss(0.0, baseline["anchor_sigma"] * math.sqrt(inflation))
            if baseline["anchor_sigma"] > 0
            else 0.0
        )
        for component in bridge_components:
            sid = component.name.split(":", 1)[1]
            indicator, data_levels = indicator_data[sid]
            sampled[component.name] = _draw_bridge(
                params, indicator, data_levels, target_period, inflation, rng
            )
        weighted = 0.0
        for component in components:
            if component.weight <= 0:
                continue
            value = sampled.get(component.name)
            if value is None:
                value = component.growth
            if value is None:
                continue
            weighted += component.weight * value
        try:
            if params.mode == "log":
                level = _level(
                    baseline["base_level"], weighted + anchor_noise, params.mode
                )
            else:
                level = (
                    baseline["base_level"]
                    + weighted
                    + anchor_noise * abs(baseline["base_level"])
                )
        except (ValueError, OverflowError):
            continue
        if params.revision.fallback:
            level *= 1 + rng.gauss(0.0, 0.02 * math.sqrt(inflation))
        elif measure == "first_release":
            # level 此时是成熟口径；首次发布相对成熟版本再叠加漂移与噪声。
            level *= (1 + rng.gauss(0.0, params.revision.sigma)) / (
                1 + params.revision.drift
            )
            if inflation > 1:
                level *= 1 + rng.gauss(
                    0.0, params.revision.sigma * (math.sqrt(inflation) - 1)
                )
        if math.isfinite(level) and level > 0:
            draws.append(level)
    return draws


def _draw_baseline(
    params: ParamSet, baseline: dict, inflation: float, rng: random.Random
) -> float:
    ar = params.ar
    steps = baseline["steps"]
    g0 = baseline["g0"]
    kind = params.growth_kind
    seasonal = baseline["seasonal_sum"]
    if ar is None:
        shock = rng.gauss(
            0.0, params.config.fallback_growth_sd * math.sqrt(inflation)
        )
        return shock * (math.sqrt(steps) if kind == "qoq" else 1.0) + seasonal
    c, phi = ar.c, ar.phi
    if ar.n >= 4:
        c, phi = linalg.sample_normal_vector(rng, [ar.c, ar.phi], ar.covariance())
        phi = max(min(phi, 0.98), -0.98)
    sigma = math.sqrt(max(ar.sigma2, 0.0) * inflation)
    if kind == "yoy":
        value = g0
        for _ in range(steps):
            value = c + phi * value + (rng.gauss(0.0, sigma) if sigma > 0 else 0.0)
        return value
    total = 0.0
    value = g0
    for _ in range(steps):
        value = c + phi * value + (rng.gauss(0.0, sigma) if sigma > 0 else 0.0)
        total += value
    return total + seasonal


def _draw_bridge(
    params: ParamSet,
    indicator: IndicatorParams,
    data_levels: dict[str, float],
    target_period: str,
    inflation: float,
    rng: random.Random,
) -> float | None:
    bridge = indicator.bridge
    lag = 4 if params.growth_kind == "yoy" else 1
    other_period = shift_period(target_period, -lag)
    needed = [m for q in (other_period, target_period) for m in quarter_months(q)]
    values = _simulate_monthly(
        indicator, data_levels, needed, rng=rng, sample_params=True
    )
    if values is None:
        return None
    target_agg = _quarterly_aggregate(values, target_period, indicator.meta.aggregation)
    other_agg = _quarterly_aggregate(
        values, other_period, indicator.meta.aggregation
    )
    if target_agg is None or other_agg is None:
        return None
    try:
        x_growth = _growth(other_agg, target_agg, indicator.mode)
    except ValueError:
        return None
    z = (x_growth - bridge.x_mean) / bridge.x_std
    slope, intercept = linalg.sample_normal_vector(
        rng,
        [bridge.slope, bridge.intercept],
        [[bridge.var_slope, bridge.cov], [bridge.cov, bridge.var_intercept]],
    )
    residual_sd = math.sqrt(max(bridge.sigma2, 0.0) * inflation)
    residual = rng.gauss(0.0, residual_sd) if residual_sd > 0 else 0.0
    return intercept + slope * z + residual


__all__ = [
    "Config",
    "ArFit",
    "BridgeFit",
    "IndicatorParams",
    "RevisionStats",
    "ParamSet",
    "Component",
    "Estimate",
    "fit_params",
    "estimate",
]
