"""时点可见性、参差发布与聚合规则的单元测试。"""
import unittest
from datetime import date

from nowcast_data.io import DataError, load
from nowcast_data.periods import (
    horizon_bucket,
    month_to_quarter,
    parse_day,
    period_end,
    quarter_months,
)
from nowcast_data.vintages import (
    Release,
    SeriesMeta,
    snapshot,
    validate_decisions,
    validate_releases,
    validate_series,
    visible_versions,
)


def _release(rid, series, period, released, revision, value):
    return Release(rid, series, period, parse_day(released), revision, float(value))


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.metas = {
            "gdp": SeriesMeta("gdp", "quarter", "flow", "million_CNY"),
            "web": SeriesMeta("web", "month", "flow", "transactions"),
            "staff": SeriesMeta("staff", "month", "stock", "people"),
        }
        self.releases = [
            _release("r1", "web", "2026-01", "2026-02-10", 1, 100),
            _release("r2", "web", "2026-01", "2026-03-10", 2, 120),
            _release("r3", "staff", "2026-02", "2026-03-15", 1, 40),
            _release("r4", "gdp", "2025-Q4", "2026-02-01", 1, 300),
        ]

    def test_decision_day_only_sees_published_versions(self):
        view = visible_versions(self.releases, date(2026, 2, 15))
        self.assertEqual(set(view), {("web", "2026-01"), ("gdp", "2025-Q4")})
        self.assertEqual(view[("web", "2026-01")].value, 100)
        view = visible_versions(self.releases, date(2026, 3, 20))
        self.assertEqual(view[("web", "2026-01")].value, 120)
        self.assertEqual(view[("web", "2026-01")].revision, 2)

    def test_same_day_release_visibility_is_configurable(self):
        inclusive = visible_versions(self.releases, date(2026, 2, 10))
        exclusive = visible_versions(
            self.releases, date(2026, 2, 10), inclusive=False
        )
        self.assertIn(("web", "2026-01"), inclusive)
        self.assertNotIn(("web", "2026-01"), exclusive)

    def test_period_end_not_equal_to_publication(self):
        # 2 月已结束，但 3 月 1 日还没有发布；必须显式检查这些月份。
        cells = snapshot(
            self.metas,
            self.releases,
            date(2026, 3, 1),
            periods={"web": ["2026-01", "2026-02"], "staff": ["2026-02"]},
        )
        self.assertEqual(cells[("web", "2026-02")].state, "pending")
        self.assertIsNone(cells[("web", "2026-02")].value)
        self.assertEqual(cells[("staff", "2026-02")].state, "pending")
        # 1 月的第二次修订在 3 月 10 日才可见。
        self.assertEqual(cells[("web", "2026-01")].revision, 1)

    def test_reported_zero_snapshot(self):
        cells = snapshot(
            self.metas,
            self.releases,
            date(2026, 2, 15),
            periods={"staff": ["2026-02"]},
        )
        self.assertEqual(cells[("staff", "2026-02")].state, "future")
        cells = snapshot(
            self.metas,
            self.releases,
            date(2027, 9, 1),
            periods={"web": ["2026-07"]},
        )
        # 1 月之后没有任何 web 发布，超过宽限期后应标 missing，而不是 0。
        self.assertEqual(cells[("web", "2026-07")].state, "missing")

    def test_reported_zero_is_not_missing(self):
        releases = self.releases + [
            _release("r0", "web", "2026-02", "2026-03-10", 1, 0),
        ]
        cells = snapshot(
            self.metas, releases, date(2026, 3, 20), periods={"web": ["2026-02"]}
        )
        self.assertEqual(cells[("web", "2026-02")].state, "reported_zero")
        self.assertEqual(cells[("web", "2026-02")].value, 0.0)

    def test_expected_release_uses_only_visible_history(self):
        cells = snapshot(
            self.metas,
            self.releases,
            date(2026, 2, 15),
            periods={"web": ["2026-02"]},
        )
        cell = cells[("web", "2026-02")]
        self.assertEqual(cell.expected_release, date(2026, 3, 10))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.metas = {
            "gdp": SeriesMeta("gdp", "quarter", "flow", "million_CNY"),
            "web": SeriesMeta("web", "month", "flow", "transactions"),
        }

    def _row(self, **overrides):
        row = {
            "id": "x",
            "series": "web",
            "period": "2026-01",
            "released": "2026-02-10",
            "revision": 1,
            "value": 1.0,
        }
        row.update(overrides)
        return row

    def test_revision_must_be_positive_integer(self):
        with self.assertRaisesRegex(DataError, "revision"):
            validate_releases([self._row(revision=0)], self.metas)

    def test_non_finite_value_rejected(self):
        with self.assertRaisesRegex(DataError, "有限"):
            validate_releases([self._row(value=float("nan"))], self.metas)

    def test_frequency_mismatch_rejected(self):
        with self.assertRaisesRegex(DataError, "频率"):
            validate_releases(
                [self._row(period="2026-Q1")], self.metas
            )

    def test_target_frequency_must_be_quarter(self):
        with self.assertRaises((DataError, ValueError)):
            validate_decisions(
                [{"id": "d", "day": "2026-02-15", "target": "gdp:2026-01"}],
                self.metas,
            )

    def test_duplicate_version_rejected(self):
        with self.assertRaisesRegex(DataError, "修订序号"):
            validate_releases(
                [self._row(id="a"), self._row(id="b")], self.metas
            )


class CalendarTests(unittest.TestCase):
    def test_month_and_quarter_boundaries(self):
        self.assertEqual(period_end("2024-02"), date(2024, 2, 29))
        self.assertEqual(period_end("2024-Q1"), date(2024, 3, 31))
        self.assertEqual(
            quarter_months("2024-Q1"), ["2024-01", "2024-02", "2024-03"]
        )
        self.assertEqual(month_to_quarter("2024-03"), "2024-Q1")

    def test_horizon_buckets_are_ordered(self):
        self.assertEqual(horizon_bucket(45), "lead>30")
        self.assertEqual(horizon_bucket(10), "lead1-30")
        self.assertEqual(horizon_bucket(0), "after0-30")
        self.assertEqual(horizon_bucket(-40), "after31-60")
        self.assertEqual(horizon_bucket(-100), "after61+")


class RegisteredDataTests(unittest.TestCase):
    def test_normal_release_versions_are_as_registered(self):
        tables = load("data/normal.json")
        metas = validate_series(tables["series"])
        releases = validate_releases(tables["releases"], metas)
        self.assertEqual(len(releases), 4)
        first = visible_versions(releases, date(2026, 2, 15))
        self.assertEqual(first[("web", "2026-01")].value, 100)
        second = visible_versions(releases, date(2026, 3, 20))
        self.assertEqual(second[("web", "2026-01")].value, 120)


if __name__ == "__main__":
    unittest.main()
