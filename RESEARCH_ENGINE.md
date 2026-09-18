# QQQ Microstructure Research Engine (research MVP)

This repository contains the research toolkit migrated from the Signalforge
offsite archive. The archive's raw recordings and
derived Parquet files are separate from this repository. Raw recordings are
opened read-only. By default, newly derived files are written under local
`data/` and reports under local `results/research/`.

## Installation

Python 3.8+:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

Optional models: `.venv/bin/python -m pip install -e '.[lightgbm]'` or
`'.[xgboost]'`. Logistic regression works with the required dependencies.
The active configuration is `research_config.yaml`; pass `--config PATH`
before the subcommand to use another config.

## Commands

```bash
.venv/bin/python -m research_engine.cli inspect --input recordings/live_20260915_132007.jsonl.gz
.venv/bin/python -m research_engine.cli inspect --date 2026-08-26 --output results/research/quality_2026-08-26.json
.venv/bin/python -m research_engine.cli build-features --date 2026-09-15
.venv/bin/python -m research_engine.cli build-features --all
.venv/bin/python -m research_engine.cli analyze --feature depth_imbalance --label future_ret_30s
.venv/bin/python -m research_engine.cli train --model logistic
.venv/bin/python -m research_engine.cli train --model lightgbm
.venv/bin/python -m research_engine.cli backtest --strategy ml_probability --mode underlying
.venv/bin/python -m research_engine.cli backtest --strategy ml_probability --mode option
.venv/bin/python -m research_engine.cli run-all
```

To use the archive without copying its data, set `SF_CLOUD` to its directory.
Pass `--recordings-root "$SF_CLOUD/recordings"` **before** `inspect`,
`build-features`, or `run-all`. For `train`, `analyze`, or `backtest`, pass
`--data-root "$SF_CLOUD/data"` **after** the subcommand. That option
reads existing features and labels; reports still go to this repository's
`results/research/`. For example:

```bash
export SF_CLOUD=/path/to/sf_cloud
.venv/bin/python -m research_engine.cli train --data-root "$SF_CLOUD/data" --model logistic
.venv/bin/python -m research_engine.cli --recordings-root "$SF_CLOUD/recordings" build-features --date 2026-09-15
```

`run-all` builds/updates all dates, computes statistics and rules, trains
expanding day-split baselines, runs the backtest, and writes reports under
`results/research/YYYYMMDD_HHMMSS/`. Before there are at least
`train_days + validation_days + test_days` distinct usable dates, training
raises an explicit error. `--stride 10` (default for training) samples one
grid row every ten seconds to limit overlap and memory; these rows are *not*
randomly shuffled and each split is by whole trading day.

Derived files:

```text
data/normalized/YYYY-MM-DD/events.parquet     # original events, normalized UTC times
data/normalized/YYYY-MM-DD/*.parquet           # per-capture parts if a day has two captures
data/snapshots/YYYY-MM-DD.parquet              # 1s QQQ best-book grid
data/features/YYYY-MM-DD.parquet               # causal QQQ/0DTE features
data/labels/YYYY-MM-DD.parquet                 # future-only labels, separate file
results/research/YYYYMMDD_HHMMSS/             # JSON, CSV, Markdown, model predictions
```

The streaming gzip reader writes normalized events to Parquet in batches of
100,000. Feature processing loads one date at a time, uses vectorized
bucketed windows/as-of joins, and writes one Parquet per date. Source hashes
come from recording `.source.sha256` sidecars when present; feature caches
check source hash, config hash and feature version. Existing raw files are
never rewritten. Duplicate prints are retained by default; `dedup_mode:
exact_event` is opt-in and can require substantial memory for large files.

## Causality and validation

Each normalized record contains distinct `event_ts`, `received_at`, and
`available_at = max(event_ts, received_at)` in UTC. Depth has no venue `ts`,
so `event_ts` is explicitly set to `received_at` and the provenance column
reads `received_fallback`. A snapshot at time `t` consumes only events with
`available_at <= t`; a trade between grid timestamps enters the next grid
bucket. It cannot leak backward into the preceding second. The relative
ATM chain is selected from the QQQ spot observable at `t`, and option returns
are marked missing on contract changes. Chain-flow strike distance uses the
last **completed** spot snapshot. Price, volatility, OFI, flow and options
all use past or current inputs; forward-return/direction/MFE/MAE calculations
are isolated in `research_engine/labels` and separately stored.

Labels at a horizon without a full future window are null; they are never
filled using the next trading day. A QQQ mid from depth older than five
seconds is masked in labels by default (`labels.max_mid_age_seconds`).
Quantile analysis preserves tied values
(a heavily zero-inflated feature can yield fewer than 10 distinct bins), and
correlation outputs both pooled and per-day IC statistics. Labels and entry
mid prices are research targets, **not** executable fill assumptions.

The backtest enters on the next grid row, buys at ask, sells at bid (or
shorts QQQ at bid and covers at ask), adds configured slippage, and charges
the configured two-sided option commission at multiplier 100. For options,
the exit uses the original contract even if ATM switches. A depth quote older
than five seconds is refused by default (`execution.max_quote_age_seconds`
can change this). Overlapping positions are not permitted. Stops and profit
targets are assessed on executable exit quotes. Entries too close to session
end to complete the configured holding period are skipped. If the holding deadline has
no fresh bid/ask, the simulator waits up to 30 seconds for the next fresh
book. Entries lacking an executable exit in that window are excluded from
realized-trade metrics and require audit before production use. This remains a snapshot
simulation; queue priority, partial fills, early assignment, halts,
transaction taxes, market impact and intrasecond excursions are not modeled.

## Source data limitations

- `header`, `contracts`, and `bar` appear in real JSONL despite examples
  showing only `trade`, `quote`, `depth`. Metadata is used to index sessions;
  stale/replayed minute bars are intentionally excluded from tick features.
- `2026-08-21` holds SPY rather than QQQ; QQQ research ignores it.
- Small early-hours diagnostic captures for Aug 25/26 and a cross-day file
  named Sep 7 with Sep 8 contracts are excluded from regular QQQ dates;
  `inspect --input` can analyze them. On Aug 26 the 13:20 and 14:12 main
  captures are *both* included without deduplication.
- Individual identical-looking trade records may be distinct executions:
  the feed has no exchange trade id/sequence. `inspect` reports suspected
  duplicates and out-of-order delivery; default ingestion retains both.
- Exchange timestamps of quote/trade can lag reception; depth provides only
  reception timestamps. Source lines are not strictly sorted by either.
  Recording-time availability is the conservative live-information proxy.
- Depth often contains only L1 or a missing side even though the archive
  README mentions 40 levels. Normalized Parquet preserves price/size arrays
  (not the auxiliary fourth tuple element); the complete tuples remain in
  the untouched raw gzip. L1/L5/L10 sizes and imbalance use the levels
  actually present, not synthetic unobserved levels.
- Book states with bid > ask are counted by `inspect` and excluded from
  derived price features. Backtests also reject stale or incomplete sides.
- The MVP uses a fixed 09:30–16:00 New York session; half-day calendars,
  exchange-level market status, delayed labels at the close, per-trade
  aggressor classification fidelity and true exchange-time depth are not yet
  validated. Treat significance estimates cautiously because adjacent
  1-second examples and 30-second labels overlap heavily.
