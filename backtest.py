import pandas as pd
import numpy as np

def _ensure_weights_df(weights_by_date, all_dates, assets):
    wdf = pd.DataFrame(index=all_dates, columns=assets, dtype=float)
    for d, w in sorted(weights_by_date.items()):
        if d not in wdf.index:
            eligible = [idx for idx in wdf.index if idx >= d]
            if not eligible:
                continue
            start = eligible[0]
        else:
            start = d
        for a in assets:
            if a in w.index:
                wdf.at[start, a] = w.at[a]
    wdf = wdf.ffill().fillna(0.0)
    return wdf

def run_backtest(price_or_returns: pd.DataFrame,
                    weights_by_date: dict,
                    is_price: bool = True,
                    init_nav: float = 1.0,
                    trade_cost: float = 0.0003):
    df = price_or_returns.copy().sort_index()
    if is_price:
        returns = df.pct_change().fillna(0.0)
    else:
        returns = df.copy()

    dates = returns.index
    assets = returns.columns.tolist()
    weights_df = _ensure_weights_df(weights_by_date, dates, assets)

    nav = pd.Series(index=dates, dtype=float)
    port_ret = pd.Series(index=dates, dtype=float)
    turnover = pd.Series(index=dates, dtype=float)
    prev_weights = pd.Series(0.0, index=assets)
    prev_nav = init_nav
    trade_logs = []

    for i, date in enumerate(dates):
        target_w = weights_df.loc[date]
        turnover_i = float(np.abs(target_w - prev_weights).sum())
        cost = turnover_i * trade_cost
        r = float((target_w * returns.loc[date]).sum())
        nav_i = prev_nav * (1 + r) * (1 - cost)
        nav.at[date] = nav_i
        port_ret.at[date] = r
        turnover.at[date] = turnover_i
        trade_logs.append({'date': date, 'turnover': turnover_i, 'cost': cost, 'nav_before': prev_nav, 'nav_after': nav_i})
        prev_weights = target_w
        prev_nav = nav_i

    trades_df = pd.DataFrame(trade_logs).set_index('date')
    return {'returns': port_ret, 'nav': nav, 'turnover': turnover, 'trades': trades_df, 'weights': weights_df}
