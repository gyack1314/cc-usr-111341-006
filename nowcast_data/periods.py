"""参考期、日历与提前期的纯标准库工具。

参考期只有两种写法：月度 ``YYYY-MM`` 与季度 ``YYYY-Qn``。
本模块不做任何推断，只负责把参考期映射到真实日历边界。
"""
from __future__ import annotations

import calendar
import math
import re
from datetime import date, timedelta

_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_QUARTER = re.compile(r"^(\d{4})-Q([1-4])$")


def parse_period(period: str) -> tuple[str, int, int]:
    """返回 (frequency, year, index)；月度 index 为 1..12，季度为 1..4。"""
    if not isinstance(period, str):
        raise ValueError("参考期必须是字符串")
    match = _MONTH.match(period)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if not 1 <= month <= 12:
            raise ValueError(f"月度参考期无效: {period}")
        return "month", year, month
    match = _QUARTER.match(period)
    if match:
        return "quarter", int(match.group(1)), int(match.group(2))
    raise ValueError(f"参考期格式无效: {period}")


def is_month(period: str) -> bool:
    return parse_period(period)[0] == "month"


def is_quarter(period: str) -> bool:
    return parse_period(period)[0] == "quarter"


def period_start(period: str) -> date:
    kind, year, index = parse_period(period)
    if kind == "month":
        return date(year, index, 1)
    return date(year, 3 * index - 2, 1)


def period_end(period: str) -> date:
    kind, year, index = parse_period(period)
    if kind == "month":
        return date(year, index, calendar.monthrange(year, index)[1])
    month = 3 * index
    return date(year, month, calendar.monthrange(year, month)[1])


def shift_period(period: str, steps: int) -> str:
    kind, year, index = parse_period(period)
    if kind == "month":
        absolute = year * 12 + (index - 1) + steps
        new_year, new_month = divmod(absolute, 12)
        return f"{new_year:04d}-{new_month + 1:02d}"
    absolute = year * 4 + (index - 1) + steps
    new_year, new_quarter = divmod(absolute, 4)
    return f"{new_year:04d}-Q{new_quarter + 1}"


def month_to_quarter(period: str) -> str:
    _, year, month = parse_period(period)
    return f"{year:04d}-Q{(month - 1) // 3 + 1}"


def quarter_months(period: str) -> list[str]:
    _, year, quarter = parse_period(period)
    if _MONTH.match(period):
        raise ValueError(f"不是季度参考期: {period}")
    return [f"{year:04d}-{3 * (quarter - 1) + month:02d}" for month in (1, 2, 3)]


def quarter_of_date(day: date) -> str:
    return f"{day.year:04d}-Q{(day.month - 1) // 3 + 1}"


def month_of_date(day: date) -> str:
    return f"{day.year:04d}-{day.month:02d}"


def parse_day(day: str) -> date:
    if not isinstance(day, str):
        raise ValueError("日期必须是 ISO 字符串")
    try:
        parsed = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError(f"日期格式无效: {day}") from exc
    if parsed.isoformat() != day:
        raise ValueError(f"日期格式无效: {day}")
    return parsed


def age_days(day: date, period: str) -> int:
    """参考期结束日至决策日的天数；结束日当天为 0。"""
    return (day - period_end(period)).days


def horizon_days(day: date, target_period: str) -> int:
    """决策日距目标参考期结束日的日历天数，可为负数。"""
    return (period_end(target_period) - day).days


HORIZONS = ["lead>30", "lead1-30", "after0-30", "after31-60", "after61+"]


def horizon_bucket(days: int) -> str:
    """按决策日距目标期末的日历天数分组（正数为提前）。"""
    if days > 30:
        return "lead>30"
    if days >= 1:
        return "lead1-30"
    if days >= -30:
        return "after0-30"
    if days >= -60:
        return "after31-60"
    return "after61+"


def months_span(day: date, period: str) -> float:
    """近似日历月份数，仅用于报告。"""
    return (period_end(period) - day).days / 30.4375


def finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def date_add_days(day: date, days: int) -> date:
    return day + timedelta(days=days)


__all__ = [
    "HORIZONS",
    "horizon_bucket",
    "parse_period",
    "period_start",
    "period_end",
    "shift_period",
    "quarter_months",
    "month_to_quarter",
    "quarter_of_date",
    "month_of_date",
    "parse_day",
    "age_days",
    "horizon_days",
    "months_span",
    "finite_number",
]
