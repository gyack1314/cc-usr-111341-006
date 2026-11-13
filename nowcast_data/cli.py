"""命令行入口：实时估计、分解、回测与发布日历。

子命令
------
``check``      检查资料完整性（沿用原有 identity 检查）
``estimate``   一个决策日的实时估计
``decompose``  两个相邻决策日之间的变动归因
``backtest``   逐决策日回测两个冻结口径
``calendar``   发布日历与每次实际取用版本
``replay``     输出某决策日的逐日重放记录（可见单元与版本）

所有子命令都接受 ``--inclusive/--exclusive`` 控制“决策日当天发布是否
可得”，默认 ``--inclusive``（released <= day）。
"""
from __future__ import annotations

import argparse
import json
import sys

from .evaluation import (
    BacktestSpec,
    backtest,
    build_decision_record,
    release_calendar,
    shapley_decomposition,
)
from .io import DataError, load
from .model import Config, estimate, fit_params
from .periods import parse_day
from .vintages import (
    validate_decisions,
    validate_releases,
    validate_series,
    visible_versions,
)


def _config(args) -> Config:
    return Config(
        transform=args.transform,
        draws=args.draws,
        interval_level=args.level,
        revision_maturity=args.maturity,
        seed=args.seed,
    )


def _load_bundle(manifest: str):
    tables = load(manifest)
    metas = validate_series(tables["series"])
    releases = validate_releases(tables["releases"], metas)
    decisions = validate_decisions(tables["decisions"], metas)
    return metas, releases, decisions


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_default))


def _default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def cmd_check(args) -> int:
    from .domain import inspect_tables

    tables = load(args.manifest)
    payload = inspect_tables(tables)
    if "releases" in tables:
        try:
            metas = validate_series(tables["series"])
            payload["releases_checked"] = len(
                validate_releases(tables["releases"], metas)
            )
            payload["decisions_checked"] = len(
                validate_decisions(tables["decisions"], metas)
            )
        except (DataError, ValueError) as exc:
            payload["releases_checked"] = False
            payload["releases_error"] = str(exc)
    _print(payload)
    return 0


def cmd_estimate(args) -> int:
    metas, releases, decisions = _load_bundle(args.manifest)
    config = _config(args)
    inclusive = args.visibility == "inclusive"
    if args.decision_id:
        decision = next((d for d in decisions if d.id == args.decision_id), None)
        if decision is None:
            raise SystemExit(f"找不到决策日: {args.decision_id}")
        day, target, target_series = decision.day, decision.target, decision.target_series
    elif args.day and args.target:
        day = parse_day(args.day)
        if ":" in args.target:
            target_series, target = args.target.split(":", 1)
        else:
            quarter_ids = [s for s, m in metas.items() if m.frequency == "quarter"]
            if len(quarter_ids) != 1:
                raise SystemExit("目标未写指标前缀且季度指标不唯一")
            target_series, target = quarter_ids[0], args.target
    else:
        raise SystemExit("需要 --decision-id，或同时给出 --day 与 --target")

    view = visible_versions(releases, day, inclusive=inclusive)
    params = fit_params(
        metas, releases, day, target_series, config, inclusive=inclusive
    )
    result = estimate(
        params,
        view,
        target,
        day,
        measure=args.measure,
        simulate=not args.point_only,
        include_inputs=not args.no_inputs,
    )
    payload = {
        "target": f"{target_series}:{target}",
        "decision_day": day.isoformat(),
        "visibility": args.visibility,
        "estimate": result.to_dict(include_inputs=not args.no_inputs),
        "review": _review(result),
    }
    if args.with_record:
        payload["record"] = build_decision_record(
            _as_decision(day, target, target_series, args.decision_id or "adhoc"),
            metas,
            releases,
            config=config,
        )
    _print(payload)
    return 0


def _as_decision(day, target, target_series, decision_id):
    from .vintages import Decision

    return Decision(decision_id, day, target, target_series)


def _review(est) -> dict:
    """人可读的“这次估计为什么动/为什么不能正式报”。"""
    return {
        "summary": (
            "已见目标发布，转为观测值"
            if est.status == "observed_target"
            else (
                "没有可用基期，停止正式估计"
                if est.status == "refused"
                else ("正式估计" if est.formal else "宽区间，不作为正式估计")
            )
        ),
        "monthly_panel": est.months,
        "component_weights": {
            c.name: round(c.weight, 4) for c in est.components if c.weight > 0
        },
        "flags": est.flags,
    }


def cmd_decompose(args) -> int:
    metas, releases, decisions = _load_bundle(args.manifest)
    config = _config(args)
    by_id = {decision.id: decision for decision in decisions}
    d1 = by_id.get(args.from_decision)
    d2 = by_id.get(args.to_decision)
    if d1 is None or d2 is None:
        raise SystemExit("--from-decision 与 --to-decision 都必须存在")
    payload = shapley_decomposition(
        metas, releases, d1, d2, config, measure=args.measure
    )
    _print(payload)
    return 0


def cmd_backtest(args) -> int:
    metas, releases, decisions = _load_bundle(args.manifest)
    config = _config(args)
    spec = BacktestSpec(
        interval_level=args.level, maturity_revision=args.maturity
    )
    if args.decision_ids:
        wanted = set(args.decision_ids.split(","))
        decisions = [d for d in decisions if d.id in wanted]
    result = backtest(metas, releases, decisions, config, spec)
    payload = {
        "spec": result["spec"],
        "config": result["config"],
        "decisions_total": result["decisions_total"],
        "metrics": result["metrics"],
    }
    if args.include_records:
        payload["records"] = result["records"]
    _print(payload)
    return 0


def cmd_calendar(args) -> int:
    metas, releases, _decisions = _load_bundle(args.manifest)
    _print(release_calendar(metas, releases))
    return 0


def cmd_replay(args) -> int:
    metas, releases, decisions = _load_bundle(args.manifest)
    config = _config(args)
    wanted = (
        [d for d in decisions if d.id in set(args.decision_ids.split(","))]
        if args.decision_ids
        else decisions
    )
    if not wanted:
        raise SystemExit("没有匹配的决策日")
    _print(
        [
            build_decision_record(decision, metas, releases, config=config)
            for decision in wanted
        ]
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="滞后发布指标的实时估计、修订分解与回测"
    )
    parser.add_argument("--version", action="version", version="nowcast 1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument(
            "--visibility",
            choices=["inclusive", "exclusive"],
            default="inclusive",
            help="inclusive=当日发布当日可得；exclusive=必须早于决策日",
        )

    def model_args(p):
        p.add_argument("--transform", choices=["auto", "qoq", "yoy"], default="auto")
        p.add_argument("--draws", type=int, default=400)
        p.add_argument("--level", type=float, default=0.80)
        p.add_argument("--maturity", type=int, default=3)
        p.add_argument("--seed", type=int, default=20261114)
        p.add_argument(
            "--measure",
            choices=["first_release", "mature"],
            default="first_release",
        )

    p = sub.add_parser("check", help="检查资料完整性与身份")
    p.add_argument("manifest")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("estimate", help="一个决策日的实时估计")
    p.add_argument("manifest")
    p.add_argument("--decision-id")
    p.add_argument("--day")
    p.add_argument("--target")
    p.add_argument("--point-only", action="store_true", help="跳过区间模拟")
    p.add_argument("--no-inputs", action="store_true", help="不输出逐条取用版本")
    p.add_argument("--with-record", action="store_true", help="附逐日重放记录")
    common(p)
    model_args(p)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("decompose", help="两个决策日之间的变动归因")
    p.add_argument("manifest")
    p.add_argument("--from-decision", required=True)
    p.add_argument("--to-decision", required=True)
    model_args(p)
    p.set_defaults(func=cmd_decompose)

    p = sub.add_parser("backtest", help="逐决策日回测两个冻结口径")
    p.add_argument("manifest")
    p.add_argument("--decision-ids", help="逗号分隔，缺省为全部")
    p.add_argument("--include-records", action="store_true")
    model_args(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("calendar", help="发布日历与首发布滞后")
    p.add_argument("manifest")
    p.set_defaults(func=cmd_calendar)

    p = sub.add_parser("replay", help="逐日重放记录")
    p.add_argument("manifest")
    p.add_argument("--decision-ids", help="逗号分隔，缺省为全部")
    model_args(p)
    p.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DataError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
