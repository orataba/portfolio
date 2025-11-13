import pandas as pd

def forecast_returns(beta: pd.DataFrame, factor_forecast: pd.Series) -> pd.Series:
    beta = beta.copy()
    f = factor_forecast.copy()
    if 'const' in beta.columns and 'const' not in f.index:
        f['const'] = 1.0
    common = beta.columns.intersection(f.index)
    if len(common) == 0:
        raise ValueError("No overlapping factors between beta and factor_forecast")
    mu_vals = beta[common].to_numpy() @ f.loc[common].to_numpy()
    return pd.Series(mu_vals, index=beta.index)
