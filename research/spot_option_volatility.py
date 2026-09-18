"""Exploratory QQQ/0DTE quote study on non-overlapping 30-second intervals."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', required=True, type=Path,
                    help='directory containing features/*.parquet')
parser.add_argument('--max-quote-age-seconds', type=float, default=5)
args = parser.parse_args()
ROOT = args.data_root / 'features'
MAX_AGE = args.max_quote_age_seconds
COLS = [
    'timestamp', 'qqq_mid', 'qqq_depth_age_seconds', 'vol_30s',
    'call_atm_option_bid', 'call_atm_option_ask', 'call_atm_option_mid',
    'call_atm_option_strike', 'call_atm_option_depth_age_seconds',
    'put_atm_option_bid', 'put_atm_option_ask', 'put_atm_option_mid',
    'put_atm_option_strike', 'put_atm_option_depth_age_seconds',
]


def build_intervals(path):
    d = pd.read_parquet(path, columns=COLS)
    start = d.iloc[np.arange(30, len(d) - 30, 30)].reset_index(drop=True)
    end = d.iloc[np.arange(60, len(d), 30)].reset_index(drop=True)
    past_fresh = d['qqq_depth_age_seconds'].le(MAX_AGE).rolling(31, min_periods=31).sum().eq(31)
    past_fresh = past_fresh.iloc[np.arange(30, len(d) - 30, 30)].reset_index(drop=True)
    result = pd.DataFrame({
        'date': path.stem,
        'half_hour': ((start['timestamp'].dt.tz_convert('America/New_York').dt.hour * 60 +
                       start['timestamp'].dt.tz_convert('America/New_York').dt.minute - 570) // 30),
        'past_vol': start['vol_30s'],
        'd_s': end['qqq_mid'] - start['qqq_mid'],
    })
    fresh_spot = (past_fresh & start['qqq_mid'].gt(0) & end['qqq_mid'].gt(0) &
                  start['qqq_depth_age_seconds'].le(MAX_AGE) &
                  end['qqq_depth_age_seconds'].le(MAX_AGE) &
                  (end['timestamp'] - start['timestamp']).eq(pd.Timedelta(seconds=30)))
    result['fresh_spot'] = fresh_spot
    for side in ('call', 'put'):
        base = f'{side}_atm_option_'
        fresh = (start[base + 'bid'].gt(0) & end[base + 'bid'].gt(0) &
                 start[base + 'ask'].ge(start[base + 'bid']) &
                 end[base + 'ask'].ge(end[base + 'bid']) &
                 start[base + 'depth_age_seconds'].le(MAX_AGE) &
                 end[base + 'depth_age_seconds'].le(MAX_AGE) &
                 start[base + 'mid'].ge(0.05) &
                 end[base + 'mid'].ge(0.05) &
                 start[base + 'strike'].eq(end[base + 'strike']))
        result[f'{side}_valid'] = fresh_spot & fresh
        result[f'd_{side}'] = end[base + 'mid'] - start[base + 'mid']
        result[f'{side}_spread'] = end[base + 'ask'] - end[base + 'bid']
    result['both_valid'] = result['call_valid'] & result['put_valid'] & (
        start['call_atm_option_strike'].eq(start['put_atm_option_strike']))
    return result


def bootstrap_daily(values, seed=42):
    values = np.asarray(values, float)
    rng = np.random.RandomState(seed)
    sampled = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return np.percentile(sampled, [2.5, 97.5])


paths = sorted(ROOT.glob('*.parquet'))
if not paths:
    parser.error(f'no feature files found in {ROOT}')
frames = [build_intervals(p) for p in paths]
d = pd.concat(frames, ignore_index=True)
print('days', d['date'].nunique(), 'intervals', len(d))
print('valid spot/call/put/both', *(int(d[c].sum()) for c in ('fresh_spot', 'call_valid', 'put_valid', 'both_valid')))

for side in ('call', 'put'):
    v = d.loc[d[f'{side}_valid'], ['date', 'd_s', f'd_{side}']].dropna()
    slope, intercept = np.polyfit(v['d_s'], v[f'd_{side}'], 1)
    daily = v.groupby('date').apply(lambda g: np.polyfit(g['d_s'], g[f'd_{side}'], 1)[0])
    print(side, 'n', len(v), 'slope', slope, 'intercept', intercept,
          'corr', v['d_s'].corr(v[f'd_{side}']), 'daily_slope_mean', daily.mean(),
          'daily_slope_ci', bootstrap_daily(daily.values))

v = d.loc[d['both_valid'], ['date', 'half_hour', 'past_vol', 'd_s',
                             'd_call', 'd_put', 'call_spread', 'put_spread']].dropna()
v['d_straddle'] = v['d_call'] + v['d_put']
v['abs_option_change'] = (v['d_call'].abs() + v['d_put'].abs()) / 2
v['abs_spot_change'] = v['d_s'].abs()
v['option_minus_half_spot'] = v['abs_option_change'] - 0.5 * v['abs_spot_change']
for label, mask in (
    ('down_20c', v['d_s'] <= -0.20),
    ('quiet_5c', v['d_s'].abs() < 0.05),
    ('up_20c', v['d_s'] >= 0.20),
):
    w = v.loc[mask]
    print(label, 'n', len(w), 'mean_dS', w['d_s'].mean(), 'mean_dCall', w['d_call'].mean(),
          'mean_dPut', w['d_put'].mean(), 'mean_dStraddle', w['d_straddle'].mean(),
          'mean_absOption', w['abs_option_change'].mean())

# Compare the top and bottom 25% of trailing QQQ volatility within the same
# day and half-hour segment. The next 30-second outcome does not overlap with
# the trailing 30-second input window.
v['vol_rank'] = v.groupby(['date', 'half_hour'])['past_vol'].rank(pct=True)
v['vol_group'] = np.where(v['vol_rank'] <= 0.25, 'low',
                          np.where(v['vol_rank'] >= 0.75, 'high', 'middle'))
tails = v[v['vol_group'].isin(['low', 'high'])]
for metric in ('abs_spot_change', 'abs_option_change', 'option_minus_half_spot', 'd_straddle',
               'call_spread', 'put_spread'):
    means = tails.groupby(['date', 'vol_group'])[metric].mean().unstack()
    diff = means['high'] - means['low']
    print('lead', metric, 'n_low', int((tails['vol_group'] == 'low').sum()),
          'n_high', int((tails['vol_group'] == 'high').sum()),
          'low', tails.loc[tails['vol_group'] == 'low', metric].mean(),
          'high', tails.loc[tails['vol_group'] == 'high', metric].mean(),
          'daily_diff_mean', diff.mean(), 'daily_diff_ci', bootstrap_daily(diff.values),
          'days_high_gt_low', int((diff > 0).sum()))

print('per_day_option_change')
print(tails.groupby(['date', 'vol_group'])['abs_option_change'].mean().unstack().to_string())
