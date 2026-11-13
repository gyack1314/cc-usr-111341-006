"""发布版本库与决策日快照。

关键约定：

* 决策日 d 的默认可见集合是 ``released <= d``（当日发布视为当日可得）。
  如需严格早于决策日，可把 ``inclusive`` 设为 False。
* 同一 (series, period) 在决策日只允许一个版本：按 (released, revision)
  取最后一条。
* 观测期结束不等于已经发布。没有可见发布时快照保留状态，区分
  future / pending / missing，而不是把面板补成最终完整矩阵。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median
from typing import Iterable, Mapping

from .io import DataError
from .periods import (
    finite_number,
    is_quarter,
    parse_day,
    parse_period,
    period_end,
)

AGGREGATIONS = {"flow", "stock", "average"}
FREQUENCIES = {"month", "quarter"}

# 状态取值：
#   observed       决策日已有可见发布，值为当时版本
#   reported_zero  决策日已有可见发布，值为 0（与缺报分开保留）
#   pending        参考期已结束，但决策日尚无可见发布
#   future         参考期尚未结束，决策日不可能有结果
#   missing        参考期已结束且超过宽限期仍无任何发布记录
STATE_ORDER = ["observed", "reported_zero", "pending", "future", "missing"]


@dataclass(frozen=True)
class SeriesMeta:
    id: str
    frequency: str
    aggregation: str
    unit: str


@dataclass(frozen=True)
class Release:
    id: str
    series: str
    period: str
    released: date
    revision: int
    value: float


@dataclass(frozen=True)
class CellState:
    """决策日视角下单个 (series, period) 的资料状态。"""

    series: str
    period: str
    state: str
    value: float | None
    release_id: str | None
    revision: int | None
    released: date | None
    expected_release: date | None
    calendar_basis: int
    note: str = ""

    @property
    def observed(self) -> bool:
        return self.state in {"observed", "reported_zero"}

    def to_dict(self) -> dict:
        return {
            "series": self.series,
            "period": self.period,
            "state": self.state,
            "value": self.value,
            "release_id": self.release_id,
            "revision": self.revision,
            "released": self.released.isoformat() if self.released else None,
            "expected_release": (
                self.expected_release.isoformat() if self.expected_release else None
            ),
            "calendar_basis": self.calendar_basis,
            "note": self.note,
        }


def validate_series(rows: Iterable[Mapping]) -> dict[str, SeriesMeta]:
    metas: dict[str, SeriesMeta] = {}
    for row in rows:
        series_id = row["id"]
        if series_id in metas:
            raise DataError(f"series.id 重复: {series_id}")
        frequency = row["frequency"]
        aggregation = row["aggregation"]
        unit = row["unit"]
        if frequency not in FREQUENCIES:
            raise DataError(f"{series_id} 的 frequency 无效: {frequency}")
        if aggregation not in AGGREGATIONS:
            raise DataError(f"{series_id} 的 aggregation 无效: {aggregation}")
        if not isinstance(unit, str) or not unit.strip():
            raise DataError(f"{series_id} 的 unit 无效")
        metas[series_id] = SeriesMeta(series_id, frequency, aggregation, unit)
    return metas


def validate_releases(
    rows: Iterable[Mapping], metas: Mapping[str, SeriesMeta]
) -> list[Release]:
    releases: list[Release] = []
    seen_ids: set[str] = set()
    seen_versions: set[tuple[str, str, int]] = set()
    for row in rows:
        rid = row["id"]
        if not isinstance(rid, str) or not rid or rid in seen_ids:
            raise DataError(f"releases.id 为空或重复: {rid!r}")
        seen_ids.add(rid)
        series_id = row["series"]
        if series_id not in metas:
            raise DataError(f"releases.series 引用不存在: {series_id}")
        period = row["period"]
        kind, _, _ = parse_period(period)
        if kind != metas[series_id].frequency:
            raise DataError(f"{rid} 的参考期频率与指标 {series_id} 不符")
        revision = row["revision"]
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise DataError(f"{rid} 的 revision 必须是从 1 开始的整数")
        version_key = (series_id, period, revision)
        if version_key in seen_versions:
            raise DataError(f"{series_id} {period} 的修订序号重复: {revision}")
        seen_versions.add(version_key)
        value = row["value"]
        if not finite_number(value):
            raise DataError(f"{rid} 的 value 必须是有限数值")
        released = parse_day(row["released"])
        releases.append(Release(rid, series_id, period, released, revision, float(value)))
    return releases


@dataclass(frozen=True)
class Decision:
    id: str
    day: date
    target: str
    target_series: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "day": self.day.isoformat(),
            "target": self.target,
            "target_series": self.target_series,
        }


def split_target(
    target: str, metas: Mapping[str, SeriesMeta] | None = None
) -> tuple[str, str]:
    """把 ``gdp:2026-Q1`` 拆成 (系列, 参考期)。

    兼容只写参考期 ``2026-Q1`` 的旧式目标：此时要求资料中恰有一个季度
    频率指标，否则必须显式写明指标前缀，避免悄悄挑一个目标。
    """
    if not isinstance(target, str):
        raise ValueError("decisions.target 必须是字符串")
    if ":" in target:
        series_id, period = target.split(":", 1)
        if not series_id or not period:
            raise ValueError(f"decisions.target 无效: {target}")
        parse_period(period)
        return series_id, period
    parse_period(target)
    if metas is None:
        raise ValueError(f"decisions.target 缺少指标前缀: {target}")
    quarter_ids = [sid for sid, meta in metas.items() if meta.frequency == "quarter"]
    if len(quarter_ids) != 1:
        raise ValueError(
            f"decisions.target 未写明指标且季度指标不唯一: {target}"
        )
    return quarter_ids[0], target


def validate_decisions(
    rows: Iterable[Mapping], metas: Mapping[str, SeriesMeta]
) -> list[Decision]:
    decisions: list[Decision] = []
    seen: set[str] = set()
    for row in rows:
        did = row["id"]
        if not isinstance(did, str) or not did or did in seen:
            raise DataError(f"decisions.id 为空或重复: {did!r}")
        seen.add(did)
        day = parse_day(row["day"])
        series_id, period = split_target(row["target"], metas)
        if series_id not in metas:
            raise DataError(f"decisions.target 引用不存在的指标: {series_id}")
        meta = metas[series_id]
        kind, _, _ = parse_period(period)
        if kind != meta.frequency:
            raise DataError(f"{did} 的目标频率与指标 {series_id} 不符")
        if not is_quarter(period):
            raise DataError(f"{did} 的目标必须是季度参考期: {period}")
        if meta.frequency != "quarter":
            raise DataError(f"{did} 的目标指标必须是季度频率")
        decisions.append(Decision(did, day, period, series_id))
    return decisions


def _normalise_publish_lag(
    releases: Iterable[Release], metas: Mapping[str, SeriesMeta]
) -> dict[tuple[str, int], int]:
    """每个指标按已发布历史估计“期末→首次发布”的滞后天数中位数。

    估计只使用传入的 releases（调用方负责按决策日裁剪），避免把未来
    发布日历泄漏到当时的期望发布日里。
    """
    lags: dict[str, list[int]] = {}
    for release in releases:
        if release.revision != 1:
            continue
        lags.setdefault(release.series, []).append(
            (release.released - period_end(release.period)).days
        )
    result: dict[tuple[str, int], int] = {}
    for series_id, meta in metas.items():
        series_lags = lags.get(series_id)
        if not series_lags:
            # 没有首次发布历史时给一个保守的日历基准，并标记为日历推测。
            default = 45 if meta.frequency == "month" else 105
            result[(series_id, 0)] = default
            continue
        result[(series_id, 0)] = int(round(median(series_lags)))
    return result

def _expected_release(
    series_id: str, period: str, lags: Mapping[tuple[str, int], int]
) -> date | None:
    lag = lags.get((series_id, 0))
    if lag is None:
        return None
    end = period_end(period)
    return date.fromordinal(end.toordinal() + lag)


def visible_versions(
    releases: Iterable[Release], day: date, inclusive: bool = True
) -> dict[tuple[str, str], Release]:
    """决策日可见版本：每个 (series, period) 按 (released, revision) 取最后一条。"""
    chosen: dict[tuple[str, str], Release] = {}
    for release in releases:
        if inclusive:
            if release.released > day:
                continue
        else:
            if release.released >= day:
                continue
        key = (release.series, release.period)
        current = chosen.get(key)
        if current is None or (release.released, release.revision) >= (
            current.released,
            current.revision,
        ):
            chosen[key] = release
    return chosen


def snapshot(
    metas: Mapping[str, SeriesMeta],
    releases: Iterable[Release],
    day: date,
    periods: Mapping[str, Iterable[str]] | None = None,
    *,
    inclusive: bool = True,
    grace_days: int = 120,
) -> dict[tuple[str, str], CellState]:
    """构建设策日快照。

    periods 可指定每个指标要检查的参考期；缺省时使用该指标所有出现过的
    参考期，加上“目标季度应含而未出现的月份”。
    """
    releases = list(releases)
    visible = visible_versions(releases, day, inclusive=inclusive)
    # 期望发布日只用决策日可见的首次发布历史，避免未来日历泄漏。
    visible_releases = [
        r
        for r in releases
        if (r.released <= day if inclusive else r.released < day)
    ]
    lags = _normalise_publish_lag(visible_releases, metas)
    all_periods: dict[str, set[str]] = {sid: set() for sid in metas}
    for release in releases:
        all_periods.setdefault(release.series, set()).add(release.period)
    if periods:
        for series_id, items in periods.items():
            all_periods.setdefault(series_id, set()).update(items)

    cells: dict[tuple[str, str], CellState] = {}
    for series_id, period_set in all_periods.items():
        meta = metas.get(series_id)
        if meta is None:
            continue
        for period in sorted(period_set, key=lambda p: parse_period(p)[1:]):
            release = visible.get((series_id, period))
            end = period_end(period)
            expected = _expected_release(series_id, period, lags)
            if release is not None:
                state = "reported_zero" if release.value == 0 else "observed"
                note = "决策日可见版本"
                cells[(series_id, period)] = CellState(
                    series_id,
                    period,
                    state,
                    release.value,
                    release.id,
                    release.revision,
                    release.released,
                    expected,
                    (day - end).days,
                    note,
                )
                continue
            if end > day:
                state = "future"
                note = "参考期尚未结束"
            elif expected is not None and day < expected + timedelta(days=grace_days):
                state = "pending"
                note = "已过期末但决策日尚未见发布"
            elif day <= end + timedelta(days=grace_days):
                state = "pending"
                note = "已过期末，处于通常发布滞后内"
            else:
                state = "missing"
                note = "超过宽限期仍无任何发布记录"
            cells[(series_id, period)] = CellState(
                series_id,
                period,
                state,
                None,
                None,
                None,
                None,
                expected,
                (day - end).days,
                note,
            )
    return cells


__all__ = [
    "AGGREGATIONS",
    "FREQUENCIES",
    "STATE_ORDER",
    "SeriesMeta",
    "Release",
    "CellState",
    "Decision",
    "validate_series",
    "validate_releases",
    "validate_decisions",
    "visible_versions",
    "snapshot",
    "split_target",
]
