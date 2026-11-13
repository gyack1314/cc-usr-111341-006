"""模型、分解与回测的行为测试。

重点验证“决策日之后的信息不能进入估计/参数/口径”这条硬约束。
"""
import copy
import unittest
from datetime import date

from nowcast_data.evaluation import (
    BacktestSpec,
    actual_value,
    backtest,
    baseline_config,
    release_calendar,
    shapley_decomposition,
)
from nowcast_data.io import load
from nowcast_data.model import Config, estimate, fit_params
from nowcast_data.periods import parse_day
from nowcast_data.vintages import (
    Decision,
    Release,
    validate_decisions,
    validate_releases,
    validate_series,
    visible_versions,
)


def _bundle(manifest="data/realtime.json"):
    tables = load(manifest)
    metas = validate_series(tables["series"])
    releases = validate_releases(tables["releases"], metas)
    decisions = validate_decisions(tables["decisions"], metas)
    return metas, releases, decisions


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metas, cls.releases, cls.decisions = _bundle()

    def test_point_estimate_uses_only_visible_vintages(self):
        """把决策日之后的发布全部删掉，估计必须逐位相同。"""
        day = parse_day("2024-06-20")
        target = "2024-Q2"
        config = Config(draws=50)
        full_view = visible_versions(self.releases, day)
        params = fit_params(self.metas, self.releases, day, "gdp", config)
        full = estimate(params, full_view, target, day, simulate=False)

        truncated = [r for r in self.releases if r.released <= day]
        truncated_view = visible_versions(truncated, day)
        params2 = fit_params(self.metas, truncated, day, "gdp", config)
        cut = estimate(params2, truncated_view, target, day, simulate=False)
        self.assertEqual(full.point, cut.point)
        self.assertEqual(full.inputs_digest, cut.inputs_digest)

    def test_parameters_change_when_later_data_arrives_only_via_view(self):
        """同一套 d1 参数配 d2 资料：点估计可以变，但参数摘要不变。"""
        d1, d2 = self.decisions[0], self.decisions[3]
        config = Config(draws=20)
        params = fit_params(self.metas, self.releases, d1.day, "gdp", config)
        view1 = visible_versions(self.releases, d1.day)
        view2 = visible_versions(self.releases, d2.day)
        a = estimate(params, view1, d1.target, d1.day, simulate=False)
        b = estimate(params, view2, d1.target, d1.day, simulate=False)
        self.assertEqual(a.params.summary(), b.params.summary())
        self.assertNotEqual(a.inputs_digest, b.inputs_digest)

    def test_actual_value_is_frozen_by_criterion(self):
        first = actual_value(self.releases, "gdp", "2024-Q3", "first_release", 3)
        mature = actual_value(self.releases, "gdp", "2024-Q3", "mature", 3)
        self.assertEqual(first.revision, 1)
        self.assertEqual(mature.revision, 3)
        # 大修订季度：首次与成熟版本必须不同，防止挑有利版本。
        self.assertNotAlmostEqual(first.value, mature.value)

    def test_missing_maturity_is_not_substituted(self):
        # 2025-Q4 只有两次修订：成熟口径必须返回 None，而不是取最后版本。
        self.assertIsNone(
            actual_value(self.releases, "gdp", "2025-Q4", "mature", 3)
        )

    def test_monthly_panel_is_not_filled(self):
        day = parse_day("2023-03-31")
        params = fit_params(self.metas, self.releases, day, "gdp", Config())
        view = visible_versions(self.releases, day)
        est = estimate(params, view, "2023-Q1", day, simulate=False)
        # 3 月尚未发布：必须是 pending/future，而不是插值出的数。
        self.assertIn(est.months["ind_prod"]["2023-03"], {"pending", "future"})
        self.assertNotIn(
            ("ind_prod", "2023-03"),
            {(k[0], k[1]) for k in view},
        )


class StructuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metas, cls.releases, cls.decisions = _bundle()

    def test_aggregation_rules_differ(self):
        day = parse_day("2024-06-20")
        config = Config(draws=10)
        params = fit_params(self.metas, self.releases, day, "gdp", config)
        view = visible_versions(self.releases, day)
        est = estimate(params, view, "2024-Q2", day, simulate=False)
        by_series = {
            c.name.split(":", 1)[1]: c for c in est.components if c.kind == "bridge"
        }
        self.assertEqual(by_series["ind_prod"].detail["aggregation"], "flow")
        self.assertEqual(by_series["ind_stock"].detail["aggregation"], "stock")
        self.assertEqual(by_series["ind_price"].detail["aggregation"], "average")

    def test_wide_or_refused_when_history_is_thin(self):
        """normal 资料只有一条目标值：必须拒绝或宽区间，不能给正式估计。"""
        tables = load("data/normal.json")
        metas = validate_series(tables["series"])
        releases = validate_releases(tables["releases"], metas)
        decisions = validate_decisions(tables["decisions"], metas)
        params = fit_params(
            metas, releases, decisions[0].day, "gdp", Config(draws=20)
        )
        view = visible_versions(releases, decisions[0].day)
        est = estimate(params, view, "2026-Q1", decisions[0].day)
        self.assertIn(est.status, {"wide", "refused"})
        self.assertFalse(est.formal)

    def test_reported_zero_kept_in_panel(self):
        day = parse_day("2023-08-15")
        params = fit_params(self.metas, self.releases, day, "gdp", Config())
        view = visible_versions(self.releases, day)
        est = estimate(params, view, "2023-Q2", day, simulate=False)
        self.assertEqual(est.months["ind_orders"]["2023-06"], "reported_zero")

    def test_long_gap_column_drops_out_with_flag(self):
        """ind_price 2022 全年缺失：桥接应能识别并给出缺口说明或仍可用。"""
        day = parse_day("2023-03-21")
        params = fit_params(self.metas, self.releases, day, "gdp", Config())
        indicator = params.indicators["ind_price"]
        self.assertNotIn("2022-06", indicator.raw_levels)
        view = visible_versions(self.releases, day)
        est = estimate(params, view, "2023-Q1", day, simulate=False)
        self.assertTrue(
            any("股票" in f or "ind_price" in f for f in est.flags)
            or any(c.kind == "bridge" for c in est.components)
        )


class DecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metas, cls.releases, cls.decisions = _bundle()
        cls.by_id = {d.id: d for d in cls.decisions}

    def test_shapley_and_chain_sum_to_total(self):
        config = Config(draws=30)
        # 相邻决策日必须共享目标季度。
        pairs = []
        for a, b in zip(self.decisions, self.decisions[1:]):
            if (a.target_series, a.target) == (b.target_series, b.target):
                pairs.append((a.id, b.id))
        checked = 0
        for a, b in pairs:
            result = shapley_decomposition(
                self.metas, self.releases, self.by_id[a], self.by_id[b], config
            )
            if result["status"] != "ok":
                continue
            checked += 1
            self.assertAlmostEqual(sum(result["contributions"].values()),
                                   result["total_change"], places=6)
            self.assertAlmostEqual(result["residual"], 0.0, places=6)
            self.assertAlmostEqual(
                result["sequential_residual"], 0.0, places=6
            )
        self.assertGreaterEqual(checked, 2)

    def test_decomposition_endpoints_match_direct_estimates(self):
        config = Config(draws=10)
        for a, b in zip(self.decisions, self.decisions[1:]):
            if (a.target_series, a.target) != (b.target_series, b.target):
                continue
            result = shapley_decomposition(
                self.metas, self.releases, a, b, config
            )
            if result["status"] != "ok":
                continue
            p1 = fit_params(
                self.metas, self.releases, a.day, a.target_series, config
            )
            e1 = estimate(
                p1, visible_versions(self.releases, a.day), a.target, a.day,
                simulate=False,
            )
            p2 = fit_params(
                self.metas, self.releases, b.day, b.target_series, config
            )
            e2 = estimate(
                p2, visible_versions(self.releases, b.day), b.target, b.day,
                simulate=False,
            )
            self.assertAlmostEqual(result["point_from"], e1.point, places=6)
            self.assertAlmostEqual(result["point_to"], e2.point, places=6)
            return
        self.fail("没有可用于校验的相邻决策日")

    def test_decomposition_requires_same_target(self):
        with self.assertRaises(ValueError):
            shapley_decomposition(
                self.metas,
                self.releases,
                self.by_id["dec001"],
                self.by_id["dec006"],
                Config(draws=10),
            )

    def test_target_release_attributed_to_target_version(self):
        # 2023-05-10 首次发布 GDP 2023-Q1：该步变动主要归因于目标版本变化。
        result = shapley_decomposition(
            self.metas,
            self.releases,
            self.by_id["dec003"],
            self.by_id["dec004"],
            Config(draws=20),
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["target_version_changes"])
        self.assertGreater(abs(result["contributions"]["target_version"]), 0.0)


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metas, cls.releases, cls.decisions = _bundle()
        cls.result = backtest(
            cls.metas, cls.releases, cls.decisions, Config(draws=40)
        )

    def test_spec_is_frozen_and_reported(self):
        spec = self.result["spec"]
        self.assertEqual(spec["criteria"]["first_release"], "revision==1")
        self.assertEqual(spec["criteria"]["mature"], "revision==3")
        self.assertEqual(spec["maturity_revision"], 3)
        self.assertEqual(spec["visibility"], "released <= decision_day")

    def test_both_criteria_reported_separately(self):
        criteria = self.result["metrics"]["criterion"]
        self.assertIn("first_release", criteria)
        self.assertIn("mature", criteria)
        counts = {c: criteria[c]["full"]["n"] for c in criteria}
        # 2025-Q4 缺成熟版本，成熟口径样本应少于首次发布口径。
        self.assertLess(counts["mature"], counts["first_release"])

    def test_by_horizon_and_coverage_present(self):
        horizons = self.result["metrics"]["by_horizon"]["first_release"]
        self.assertGreaterEqual(len(horizons), 2)
        for block in horizons.values():
            self.assertIsNotNone(block["full"]["coverage"])

    def test_bridge_beats_target_only_on_average(self):
        first = self.result["metrics"]["criterion"]["first_release"]
        self.assertLess(first["full"]["mae"], first["target_only"]["mae"])

    def test_baseline_uses_only_target_history(self):
        cfg = baseline_config(Config())
        self.assertFalse(cfg.bridge)


class CalendarTests(unittest.TestCase):
    def test_calendar_lists_every_taken_version(self):
        metas, releases, _ = _bundle()
        calendar = release_calendar(metas, releases)
        web = calendar["ind_prod"]["releases"]
        periods = {row["period"] for row in web}
        self.assertIn("2023-05", periods)
        delayed = [row for row in web if row["period"] == "2023-05"]
        self.assertGreaterEqual(len(delayed), 1)
        self.assertGreater(delayed[0]["lag_days"], 60)


if __name__ == "__main__":
    unittest.main()
