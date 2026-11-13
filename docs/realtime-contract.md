# 资料约定（实时监测扩展）

本文件补充 `docs/data-contract.md`。原有三张表 `series`、`releases`、
`decisions` 的字段与身份规则不变；实时监测模块只读取这三张表，不另建
“已对齐面板”或“最终矩阵”文件。

## 参考期与状态

- `period` 只能是 `YYYY-MM`（月度）或 `YYYY-Qn`（季度），且必须与
  `series.frequency` 一致。
- `released` 是该版本真正可得的日期；同一 `(series, period)` 可以有多条
  记录，用 `revision` 从 1 开始递增区分。**修订序号重复视为资料错误。**
- 决策日 `d` 的可见版本是 `released <= d`（当天发布视为可得）中
  `(released, revision)` 最大的一条。`--visibility exclusive` 可改成
  `released < d`。
- **观测期结束不等于已发布。** 没有可见记录的单元保持 `pending`
  （已过期末）或 `future`（期末未到）；超过宽限期仍无任何记录标为
  `missing`。三种状态都不会被填成 0。
- 已报告为零保留为 `reported_zero`，与缺报、未发布分开。

## 指标定义

- `frequency`：`month` 或 `quarter`。
- `aggregation`：`flow`（按期汇总，季度=三月之和）、`stock`（期末存量，
  季度=季末月）、`average`（平均值，季度=三月均值）。不同 `unit`
  不得直接相加。
- 季度目标必须显式写成 `series:period`（如 `gdp:2026-Q1`）。只写
  `2026-Q1` 时，只有当资料中恰有一个季度指标才被接受。

## 决策日

- `day` 是决策日（ISO 日期）。`target` 是决策日要估计的参考期。
- 同一目标可以有多个相邻决策日；分解与回测都以决策日排序。
- 决策日晚于目标首次发布时，模块返回观测值并把状态标为
  `observed_target`，不会伪装成“预测”。

## 完整性

- `data/realtime.json` 是演示用的合成资料包，由
  `tools/make_realtime_data.py` 以固定种子生成；manifest 内的 SHA256、
  字节数与行数绑定规则与 `data/normal.json` 相同。
- 不得把 `normal`、`boundary`、`realtime` 三个资料包拼接使用：它们是
  相互独立的登记单元，id 可能重复。
- 重新生成 `data/realtime-*.jsonl` 后必须同步更新 `data/realtime.json`
  的摘要，否则 `load()` 会拒绝读取。
