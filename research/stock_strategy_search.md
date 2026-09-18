# 正股策略开仓期权：持续研究记录

更新：2026-09-18。研究代码、逐笔账本和失败状态都已保留。本记录会随后续实验更新。

## 目前能支持什么

**尚未找到跨日期、采样时点都稳定的扣费盈利策略。** 旧样本中最显眼的是早盘一分钟强变动后的反转，但它对检查秒数、持仓筛选和日期很敏感。不能再把旧样本中的盈利称作稳定优势。

本轮确实发现了两份此前尚未生成特征的新录制：9月16、17日。在生成特征前冻结三条规则，随后三条规则在新增两天的合计损益都为负。新增结果没有用于修改那次冻结规则。之后的价差、持仓容量等研究已经知道这些结果，因此不能再将这两天称作全新留出样本。

## 数据和成交口径

- 原14个特征日期：8月24日至9月15日的14个可用QQQ交易日；前5日校准，8月31日至9月4日的5日筛选，9月8、11、14、15日检查。旧日期此前已反复探索。
- 新增9月16、17日原始录制由本工程生成特征，原始文件只读。本地输出在 `data/`，原有档案数据仍从外部目录读取。
- 买入策略每笔一张QQQ当日到期期权，信号时固定合约，延迟1秒，以ask加1bp买、bid减1bp卖，每张每边手续费0.65美元。通常持有60、180、300秒。
- 严格入场要求报价年龄≤1秒且在信号后更新；退出原合约，缺报价最多等30秒。未平仓保留资金并另算零回收压力损失，不能删掉。
- 阈值、模型只用此前已结束日期。检查同一天多个相关信号不等于增加独立日期。

## 搜索顺序及结果

| 研究 | 配置/比较 | 发现 |
|---|---|---|
| 正股规则库 | 12规则×3持有期×4条件，共144基础组合 | 唯一较突出的前段候选是早盘60秒强变动反转、持有300秒 |
| 严格报价 | 早盘36组合重放 | 原候选92笔、旧9日净收益+420.65美元，但仅4天盈利 |
| 机制修正 | 路径效率过滤、等反转确认、两者结合 | 没有得到更稳定的候选 |
| 正股幅度预测 | Ridge/浅树×3期限×2成本门槛 | 12组合均未通过前段筛选 |
| 直接预测期权净收益 | 正股特征、过去5日训练，12组合 | 无候选；保留35个无法退出训练情景的压力标签 |
| 合约选择 | 平值、实值1/2美元、虚值1美元×3期限 | 实值1美元五分钟方案前段很好，旧后4天全部亏损 |
| 执行敏感性 | 报价时效、2/5秒延迟、额外滑点 | 原反转后4天的盈利在5秒延迟下转负；898次原始报价核对通过 |
| 正股触发退出 | 4种提前退出与固定300秒比较 | 只止损旧9日+750.13美元；过早止盈反而削掉收益 |
| 检查时点 | 第0/15/30/45秒，1/5秒延迟 | 固定持有仅第30秒较好；止损方案也不跨相位稳定 |
| 逐秒事件 | 首次越界、强变动消退、消退且短线转向 | 首次越界失败；消退后入场旧后段几乎持平；转向过滤依赖单日 |
| 新录制冻结检验 | 三条固定规则，新增2日 | 三条规则合计均亏损，详见下表 |
| 有保护腿信用价差 | 3类正股事件×180/300秒 | 六组合前5日全部亏损，每套平均亏5.36～5.75美元 |
| 持仓容量 | 1/2/4张、每天统一1000美元，四相位 | 原第30秒两张方案新增2日只赚3.12美元，仍不稳定 |
| 较长正股结构 | 成交VWAP、十五分钟趋势、区间回测，10组合 | 无候选；唯一前段略盈利的三分钟趋势回撤依赖最好的一天 |
| 补齐日历 | 固定三规则，补入9月9/10日并重算历史窗口 | 原反转在补齐两日盈利，但止损与消退反转亏损，16/17日仍不佳 |

表内包含基准重放与敏感性检查，不能简单相加作为独立试验数量，也不能将不同组合的损益相加当一个可交易组合。

## 新录制对冻结规则的检验

| 规则 | 9月16日净收益 | 9月17日净收益 | 两日笔数 | 两日净收益 |
|---|---:|---:|---:|---:|
| 早盘反转，固定五分钟 | +51.52 | -74.95 | 16 | -23.44 |
| 早盘反转，正股不利0.5D止损 | -30.89 | -38.29 | 21 | -69.17 |
| 强变动消退后反转，固定五分钟 | -26.95 | -68.57 | 31 | -95.52 |

D为信号时过去一分钟正股价格变化的绝对值。金额为美元，已扣费用。没有无法退出的持仓；两日正股新鲜报价覆盖率分别99.966%、99.731%，三个组合合计136次入场/退出原始报价核对通过。

[冻结协议](new_dates_protocol.md)；[冻结记录](../results/research/new_dates_frozen/freeze.json)；[完整新增日期报告](../results/research/new_dates_validation_20260918_074309_080451/report.md)。

## 目前定位到的两个问题

**检查时点和仓位限制会改变实际成交的信号集合。** 原固定五分钟方案，在旧9日第0/15/30/45秒入场的净收益依次为-334.58、-727.59、+420.65、-405.56美元。正股止损方案同顺序为-113.42、-805.44、+750.13、+465.20美元。不能只报告第30秒。

**有小幅正股方向优势，仍可能覆盖不了期权成本。** 新增两日34个早盘反转意向信号的正股五分钟按日平均有利变化约1.08bps，但实际16笔可成交、不重叠仓位的有利变化平均只有约0.16bps。实际期权每笔中价变化贡献+0.97美元，价差成本1.09、手续费1.30、滑点0.04，最终每笔约-1.46美元。意向信号有重叠，不能把全部信号均值当作组合收益。

这不能简单归因为时间价值：同一批实际交易中，正股有利价格变化与期权中价损益的相关性很高，但这只是事后价格响应关系，不能据此预测下一次正股变化，也没有分离Delta、Gamma和隐含波动变化。

[方向与成交诊断](../results/research/stock_option_bridge_20260918_074940_253963/report.md)；[采样时点报告](../results/research/reversal_phase_20260918_073148_964518/report.md)；[持仓容量报告](../results/research/position_capacity_20260918_075402_459790/report.md)。

## 重跑方法

设置 `SF_CLOUD` 为外部档案目录，使用安装了本工程依赖的Python。为避免生成带机器路径的字节码，可设置 `PYTHONDONTWRITEBYTECODE=1`。

档案在研究中仍有新增文件。早期脚本会扫描目录中的当前日期；完整复核旧输出时须保持其 `source_manifest` 中的原日期集合。较新的价差、VWAP、丰富特征模型和日历检验使用本工程的固定输入清单，忽略后来添加或重复的日期，并拒绝哈希不符的文件。

```bash
python -m research_engine.analysis.stock_strategy_search --data-root "$SF_CLOUD/data" --context all
python -m research_engine.analysis.stock_strategy_search --data-root "$SF_CLOUD/data" --context morning --strict-entry
python -m research_engine.analysis.stock_prediction_search --data-root "$SF_CLOUD/data"
python -m research_engine.analysis.option_target_search --data-root "$SF_CLOUD/data"
python -m research_engine.analysis.event_reversal_search --data-root "$SF_CLOUD/data" --mode recovery
python -m research_engine.analysis.event_reversal_search --data-root "$SF_CLOUD/data" --mode recovery_turn
python -m research_engine.analysis.vertical_spread_search --archive-data "$SF_CLOUD/data"
python -m research_engine.analysis.long_context_search --archive-data "$SF_CLOUD/data"
python -m research_engine.analysis.complete_calendar_validation --archive-data "$SF_CLOUD/data"
python -m research_engine.analysis.rich_stock_model_search --archive-data "$SF_CLOUD/data"
```

新增数据构建与已冻结规则重放：

```bash
python -m research_engine.cli --recordings-root "$SF_CLOUD/recordings" build-features --date 2026-09-16
python -m research_engine.cli --recordings-root "$SF_CLOUD/recordings" build-features --date 2026-09-17
python -m research_engine.analysis.new_dates_validation --archive-data "$SF_CLOUD/data" --recordings-root "$SF_CLOUD/recordings"
```

需要引用旧输出的研究，使用对应参数指定目录；不会自动挑最新的有利结果：

```bash
MORNING=results/research/stock_strategy_morning_20260918_030530_761255
STRICT=results/research/stock_strategy_morning_20260918_030642_960403
ADAPTIVE=results/research/adaptive_exit_20260918_072903_231525
PHASE=results/research/reversal_phase_20260918_073148_964518
FRESH=results/research/new_dates_validation_20260918_074309_080451
python -m research_engine.analysis.reversal_refinement --data-root "$SF_CLOUD/data" --baseline-root "$MORNING"
python -m research_engine.analysis.contract_selection --data-root "$SF_CLOUD/data" --baseline-root "$MORNING"
python -m research_engine.analysis.reversal_execution_audit --data-root "$SF_CLOUD/data" --baseline-root "$MORNING" --strict-root "$STRICT"
python -m research_engine.analysis.adaptive_exit_search --data-root "$SF_CLOUD/data" --baseline-root "$MORNING"
python -m research_engine.analysis.reversal_phase_audit --data-root "$SF_CLOUD/data" --adaptive-root "$ADAPTIVE"
python -m research_engine.analysis.position_capacity_search --archive-data "$SF_CLOUD/data" --phase-root "$PHASE" --new-root "$FRESH"
python -m research_engine.analysis.stock_option_bridge --archive-data "$SF_CLOUD/data" --phase-root "$PHASE" --new-root "$FRESH"
python -m research_engine.analysis.search_artifact_audit --data-root "$SF_CLOUD/data"
```

结果在 `results/research/`：`study.json` 保存参数、数据与代码哈希；`daily.csv` 保存逐日结果；`opportunities` 保存所有成交和失败状态；`protocol.md` 保存当时协议。共用执行函数后来增加能力时，旧版本存入 `source_snapshots/<sha256>/`，旧报告的哈希仍可核对。源码快照与不同方案重复的交易不能当作额外样本。

## 补齐日期与校验记录

9月9、10日是在后续研究时新补入的。采用固定三条规则、补齐后逐日向前校准：

| 规则 | 补齐9/10日净收益 | 重算后16/17日净收益 |
|---|---:|---:|
| 早盘反转固定五分钟 | +278.37 | -19.77 |
| 同入场加正股止损 | -188.89 | -58.80 |
| 强变动消退后反转 | -83.27 | -120.56 |

与第一次冻结检验的不同来自此前5日校准窗口变化，旧结果没有覆盖。补齐日期的策略选择已看到其后其他日期，因此不称为严格时间外验证。[补齐日历报告](../results/research/complete_calendar_20260918_080827_599609/report.md)。

19组研究的账本、日期、资金和代码版本核对已通过，涉及954,966条机会记录、12,416次新增原始报价核对。记录包含不同策略及重复研究，不能把35,036条完成记录当作独立交易样本。[校验明细](../results/research/search_validation_20260918_080622_129504/validation.json)。后续新增研究另行校验。

固定16日清单见 [stock_search_dataset.json](stock_search_dataset.json)，包含补齐日期的18日清单见 [complete_calendar_dataset.json](complete_calendar_dataset.json)。原冻结检验另保存其当时5个校准日期，后续补入文件不会自动改变旧检验。

## 后续调查

继续检查更完整的正股盘口、OFI和成交信息是否能直接预测期权净收益。保持协议先固定、所有失败保留、筛选与后续日期分开，并明确新增日期何时已经被查看。
