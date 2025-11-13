import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from typing import Dict


def _add_const(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    Xc.insert(0, 'const', 1.0)
    return Xc


def fit_cross_sectional(asset_returns: pd.DataFrame,
                        macro_factors: pd.DataFrame,
                        model: str = 'ridge',
                        alpha: float = 1.0,
                        add_const: bool = True,
                        min_obs: int = 24) -> Dict:

    common = asset_returns.index.intersection(macro_factors.index)
    asset_returns = asset_returns.loc[common].sort_index()
    macro_factors = macro_factors.loc[common].sort_index()

    X = macro_factors.copy()
    if add_const:
        X = _add_const(X)

    factors = X.columns.tolist()
    assets = asset_returns.columns.tolist()

    betas = pd.DataFrame(index=assets, columns=factors, dtype=float)
    residuals = pd.DataFrame(index=asset_returns.index, columns=assets, dtype=float)
    models = {}

    for asset in assets:
        y = asset_returns[asset]
        mask = (~y.isna()) & (~X.isna().any(axis=1))
        if mask.sum() < min_obs:
            models[asset] = None
            continue
        X_fit = X.loc[mask]
        y_fit = y.loc[mask]

        if model == 'ridge':
            clf = Ridge(alpha=alpha, fit_intercept=False)
            clf.fit(X_fit.to_numpy(), y_fit.to_numpy())
            coef = pd.Series(clf.coef_, index=factors)
            pred = pd.Series(clf.predict(X_fit.to_numpy()), index=X_fit.index)
            resid = y_fit - pred
            betas.loc[asset, :] = coef
            residuals.loc[X_fit.index, asset] = resid
            models[asset] = clf
        else:
            ols = sm.OLS(y_fit.to_numpy(), X_fit.to_numpy()).fit()
            coef = pd.Series(ols.params, index=factors)
            pred = pd.Series(ols.predict(X_fit.to_numpy()), index=X_fit.index)
            resid = y_fit - pred
            betas.loc[asset, :] = coef
            residuals.loc[X_fit.index, asset] = resid
            models[asset] = ols

    return {'beta': betas, 'residuals': residuals, 'models': models}
