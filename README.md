# micro_alpha_lab

QQQ and 0DTE options microstructure research: recording ingestion, causal feature
engineering, forward labels, day-split model training, analysis, and backtesting.
The training toolkit was migrated from the external `sf_cloud` archive.
Its archived recordings and derived Parquet files are used by path, not copied
into this repository. Set `SF_CLOUD` to the archive directory on your machine.

## Quick start with the existing archive

```bash
export SF_CLOUD=/path/to/sf_cloud
python -m pip install -e '.[test]'
python -m pytest
python -m research_engine.cli train --data-root "$SF_CLOUD/data" --model logistic
python -m research_engine.cli backtest --data-root "$SF_CLOUD/data" --model logistic
```

`train` prints walk-forward fold metrics. `backtest` writes predictions, metrics,
and a report under `results/research/`. The `--data-root` option is read-only:
it is available on `train`, `analyze`, and `backtest`, and points to a directory
containing `features/` and `labels/` Parquet files.

To build fresh local features from the archived recordings:

```bash
python -m research_engine.cli --recordings-root "$SF_CLOUD/recordings" build-features --all
python -m research_engine.cli train --model logistic
```

Fresh normalized events, snapshots, features, and labels go under local `data/`.
See [RESEARCH_ENGINE.md](RESEARCH_ENGINE.md) for the data format, methods,
limitations, and all commands.

An initial [QQQ and 0DTE option quote study](research/spot_option_volatility.md)
compares trailing underlying volatility with subsequent option mid-price changes.

The follow-up [causal volatility and execution study](research/causal_volatility.md)
sets thresholds using previous dates and simulates a fixed call/put straddle
with bid/ask fills, latency, commissions, and unresolved-position audits:

```bash
python -m research_engine.analysis.causal_volatility --data-root "$SF_CLOUD/data"
```

This study requires both `features/` and full-contract `normalized/` Parquet files.

The [directional factor study](research/directional_factors.md) then tests three
fixed order-flow factors against future 30-second QQQ returns, simulates long
CALL/PUT trades after full costs, and independently compares all volatility
states with a high-volatility filter:

```bash
python -m research_engine.analysis.directional_factors --data-root "$SF_CLOUD/data"
```

It preserves missing quotes and unresolved positions, keeps future evaluation
labels separate from decisions, and records daily uncertainty and filter
sensitivity. The six tested policies all had negative average net trade PnL
on the nine historical evaluation dates; this is not a new held-out test.

The next [event and option-response study](research/event_response.md) follows a
[fixed protocol](research/event_response_protocol.md): pressure breakouts and
exhaustion, same-contract option response gaps, and option-flow confirmation,
each with independent 30-second and 180-second positions:

```bash
python -m research_engine.analysis.event_response --data-root "$SF_CLOUD/data"
```

The response-gap candidate had positive average net trade PnL on a small sample,
but its profit depended on one date and most quoted gaps did not persist for a
second. Reports include latency diagnostics, daily uncertainty, and profit
concentration; this does not establish a stable trading edge.

The ongoing [underlying-strategy search](research/stock_strategy_search.md)
tests minute-scale option trades using stock patterns, past-date models,
contract choices, adaptive exits, sampling times, protected spreads and
position limits. It includes the separately frozen September 16–17 replay,
full unsuccessful results, raw-quote audits, and reproducible commands.
The previously promising morning reversal did not sustain net profitability
on those two added dates; no stable trading edge has been established.

```bash
python -m research_engine.analysis.stock_strategy_search --data-root "$SF_CLOUD/data" --context morning --strict-entry
python -m research_engine.analysis.option_target_search --data-root "$SF_CLOUD/data"
```
