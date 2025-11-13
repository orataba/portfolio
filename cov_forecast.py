import numpy as np
import pandas as pd
from scipy.stats import norm

# 计算因子协方差矩阵
def ewma_cov(X: pd.DataFrame, lam: float = 0.97, demean: bool = True) -> pd.DataFrame:
    X = X.dropna()
    if X.shape[0] == 0:
        return None
    if demean:
        X = X - X.mean()
    S = np.zeros((X.shape[1], X.shape[1]))
    for t in range(X.shape[0]):
        x = X.iloc[t].to_numpy().reshape(-1,1)
        if t == 0:
            S = x @ x.T
        else:
            S = lam * S + (1 - lam) * (x @ x.T)
    return pd.DataFrame(S, index=X.columns, columns=X.columns)

# 计算残差方差
def ewma_var(series: pd.Series, lam: float = 0.94, min_var: float = 1e-8) -> float:
    s = series.dropna()
    if s.empty:
        return min_var
    s2 = None
    for i, val in enumerate(s):
        if i == 0:
            s2 = val**2
        else:
            s2 = lam * s2 + (1 - lam) * (val**2)
    return float(max(s2, min_var))

# 计算合并后的总方差
def compose_asset_cov(beta: pd.DataFrame,
                      Sigma_F: pd.DataFrame,
                      residuals: pd.DataFrame,
                      lambda_resid: float = 0.94,
                      jitter: float = 1e-8):
    beta_filled = beta.fillna(0.0)
    assets = beta_filled.index.tolist()
    # 残差方差
    sigma_eps = pd.DataFrame(0.0, index=assets, columns=assets)
    for a in assets:
        if a not in residuals.columns:
            var = 1e-8
        else:
            var = ewma_var(residuals[a], lam=lambda_resid)
        sigma_eps.loc[a, a] = var
    # 因子方差
    if Sigma_F is None:
        Sigma_F = pd.DataFrame(np.eye(beta_filled.shape[1]) * 1e-8, index=beta_filled.columns, columns=beta_filled.columns)

    beta_mat = beta_filled.to_numpy()
    Sigma_F_mat = Sigma_F.reindex(index=beta_filled.columns, columns=beta_filled.columns).fillna(0.0).to_numpy()
    Sigma_factor_component = beta_mat @ Sigma_F_mat @ beta_mat.T
    Sigma_asset = pd.DataFrame(Sigma_factor_component, index=assets, columns=assets) + sigma_eps
    Sigma_asset += np.eye(Sigma_asset.shape[0]) * jitter
    return Sigma_asset, pd.DataFrame(Sigma_factor_component, index=assets, columns=assets), sigma_eps

def compute_cvar(returns_df: pd.DataFrame, conf_level: float = 0.95) -> pd.Series:
    cvar = {}
    for col in returns_df.columns:
        r = returns_df[col].dropna()
        if r.empty:
            cvar[col] = np.nan
            continue
        var = np.quantile(r, 1 - conf_level)
        tail = r[r <= var]
        if len(tail) == 0:
            cvar[col] = float(var)  # fallback: VaR
        else:
            cvar[col] = float(tail.mean())
    return pd.Series(cvar, name=f"CVaR_{int(conf_level*100)}")

def forecast_risk(beta: pd.DataFrame,
                  Sigma_F: pd.DataFrame,
                  residuals: pd.DataFrame,
                  asset_history_returns: pd.DataFrame = None,
                  lambda_resid: float = 0.94,
                  cvar_conf_level: float = 0.95,
                  jitter: float = 1e-8):

    Sigma_asset, Sigma_factor_component, Sigma_eps = compose_asset_cov(beta, Sigma_F, residuals, lambda_resid=lambda_resid, jitter=jitter)
    cvar_residual = compute_cvar(residuals, conf_level=cvar_conf_level)
    cvar_asset_hist = None
    if asset_history_returns is not None:
        cvar_asset_hist = compute_cvar(asset_history_returns, conf_level=cvar_conf_level)

    return {
        'Sigma_asset': Sigma_asset,
        'Sigma_factor_component': Sigma_factor_component,
        'Sigma_eps': Sigma_eps,
        'cvar_residual': cvar_residual,
        'cvar_asset_hist': cvar_asset_hist
    }
