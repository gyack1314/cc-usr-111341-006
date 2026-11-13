# 滞后发布指标的实时经济监测与修订分解

本项目保存业务分析所需的本地资料，并提供**实时监测模块**：用决策日
真正可得的月度资料估计尚未发布的季度指标，解释每次估计更新由什么
新消息引起，并在冻结口径下回测。全部实现只使用 Python 标准库。

原有读取入口仍然保留（完整性、身份与登记关系检查）；新增的估计、
分解、回测只读取已登记资料，不写回 `data/`。

## 快速开始

```sh
# 1. 资料检查
python -m nowcast_data.cli check data/normal.json

# 2. 一个决策日的实时估计
python -m nowcast_data.cli estimate data/realtime.json --decision-id dec003

# 3. 相邻决策日的四源归因
python -m nowcast_data.cli decompose data/realtime.json \
  --from-decision dec002 --to-decision dec003

# 4. 冻结口径回测（两个口径分别报告）
python -m nowcast_data.cli backtest data/realtime.json

# 5. 发布日历与首发布滞后
python -m nowcast_data.cli calendar data/realtime.json

# 6. 逐日重放记录
python -m nowcast_data.cli replay data/realtime.json --decision-ids dec003,dec004
```

常用开关：

- `--visibility inclusive|exclusive`：决策日当天发布是否可得，默认
  `inclusive`（`released <= day`）；
- `--measure first_release|mature`：估计输出对应哪个评估口径；
- `--transform auto|qoq|yoy`：默认 `auto` 只在训练截点配对足够时用同比；
- `--draws`、`--level`、`--maturity`、`--seed`：区间模拟与口径参数，
  默认值即冻结值。

## 设计要点

- **时点可见**：决策日只看 `released <= day` 的版本；参数、季节效应、
  标准化参数与桥接系数全部由训练截点可见版本估计。
- **不补全面板**：缺报、暂未发布与已报告为零分别保留；月度缺口从最后
  可见月按 AR(1) 逐月外推，季度聚合只使用齐全的季度。
- **混频连接**：`flow` 求和、`stock` 取季末月、`average` 取三月均值。
- **修订分解**：新发布数据、旧数据修订、参数重估、目标版本变化四项，
  用精确 Shapley 分摊交互并保证加总；同时给出固定顺序链式分解。
- **冻结回测**：首次发布值与第 3 次修订分别评估，成熟版本缺失时记为
  不可用而非取最后版本；按提前期报告误差与覆盖率。
- **允许降级**：样本不足、结构变化或无桥接时输出宽区间或拒绝正式
  估计，不靠全时期调参掩盖历史失败。

方法细节、识别约束与已知局限见 [`docs/methodology.md`](docs/methodology.md)；
字段与状态约定见 [`docs/realtime-contract.md`](docs/realtime-contract.md)；
自动生成的指标、分解与场景报告见 [`docs/report.md`](docs/report.md)。

## 重新生成资料与报告

`data/realtime.json` 是演示用合成资料（固定种子），覆盖延迟发布、历史
大修订、月末跨季度、整列长期缺报、已报告为零与成熟版本缺失等情形：

```sh
python -I tools/make_realtime_data.py   # 重新生成 data/realtime-*.jsonl 与清单
python tools/make_report.py             # 写出 docs/report.{json,md}
```

## 测试

```sh
python -m unittest discover -s tests -v
python -m compileall -q nowcast_data tests tools
```

共 39 个测试，覆盖：时点可见性与同日发布规则、期末未发布/缺报/为零三态、
修订版本校验、聚合规则、决策后信息不泄漏（删除未来发布后点估计与区间
逐位相同）、四源分解加总闭合且端点等于直接估计、冻结口径与成熟版本缺失、
按提前期覆盖率、整列长期缺报与已报告为零。

## 容器

```sh
docker build -t nowcast-data .
docker run --rm nowcast-data
```
