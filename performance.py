import numpy as np
import pandas as pd

def annualize_return(r_monthly: pd.Series):
    valid = r_monthly.dropna()
    if len(valid) == 0:
        return np.nan
    return (1 + valid).prod() ** (12.0 / len(valid)) - 1

def annualize_vol(r_monthly: pd.Series):
    valid = r_monthly.dropna()
    if len(valid) <= 1:
        return np.nan
    return valid.std(ddof=1) * (12 ** 0.5)

def max_drawdown(nav: pd.Series):
    roll_max = nav.cummax()
    drawdown = nav / roll_max - 1.0
    if drawdown.empty:
        return np.nan
    return drawdown.min()

def perf_report(r_monthly: pd.Series, rf: float = 0.0):
    nav = (1 + r_monthly.fillna(0)).cumprod()
    ann_ret = annualize_return(r_monthly)
    ann_vol = annualize_vol(r_monthly)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol and not np.isnan(ann_vol) and ann_vol > 0 else np.nan
    mdd = max_drawdown(nav)
    return {'annual_return': ann_ret, 'annual_vol': ann_vol, 'sharpe': sharpe, 'max_drawdown': mdd, 'nav': nav}

