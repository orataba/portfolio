import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import utils, factor_regression, cov_forecast, return_forecast, weight_optimizer, backtest, performance
import matplotlib.pyplot as plt
import seaborn as sns

#配置
DATA_DIR = "data"
OUTPUT_DIR = "output"
WINDOW = 105
LAMBDA_RESID = 0.90
LAMBDA_FACTOR = 0.90
REGRESSION_MODEL = 'ridge'
RIDGE_ALPHA = 1.0
ADD_CONST = True
MIN_OBS = 24
TRADE_COST = 0.0002
LONG_ONLY = True
MAX_SINGLE = 0.5
MINI_SINGLE = 0.02
CVAR_CONF_LEVEL = 0.95
IS_PRICE = False

def plot_nav_and_weights(nav, weights_df, outdir):
    utils.ensure_dir(outdir)
    plt.figure(figsize=(10,5))
    nav.plot(title="Portfolio NAV")
    plt.ylabel("NAV")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pnl_nav.png"))
    plt.close()

    plt.figure(figsize=(12,6))
    sns.heatmap(weights_df.T, cmap="YlGnBu", cbar_kws={'label': 'weight'})
    plt.title("Weights Heatmap (assets x time)")
    plt.xlabel("Date")
    plt.ylabel("Asset")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "weights_heatmap.png"))
    plt.close()

def run():
    logger = utils.init_logger(os.path.join(OUTPUT_DIR, "pipeline.log"))
    utils.ensure_dir(OUTPUT_DIR)
    logger.info("Loading data...")
    macro = utils.load_csv(os.path.join(DATA_DIR, "macro_factors.csv"))
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro)
    macro = pd.DataFrame(macro_scaled, index=macro.index, columns=macro.columns)
    assets = utils.load_csv(os.path.join(DATA_DIR, "asset_returns.csv"))
    macro = utils.ensure_numeric(macro)
    assets = utils.ensure_numeric(assets)
    logger.info(f"Data loaded. macro shape {macro.shape}, assets shape {assets.shape}")

    dates = assets.index
    n_dates = len(dates)
    intermediates = {}
    weights_by_date = {}

    logger.info(f"Begin rolling pipeline with window={WINDOW}. Total dates: {n_dates}")
    # 每个调仓日，根据前面WINDOW天数的数据训练，月初调仓
    for t_idx in range(WINDOW, n_dates):
        rebalance_date = dates[t_idx]
        train_slice = slice(t_idx - WINDOW, t_idx)
        train_dates = dates[train_slice]
        logger.info(f"Rebalance {rebalance_date}; training from {train_dates[0]} to {train_dates[-1]}")

        asset_train = assets.loc[train_dates]
        macro_train = macro.loc[train_dates]

        # 1) 回归拟合
        fit = factor_regression.fit_cross_sectional(asset_train, macro_train, model=REGRESSION_MODEL, alpha=RIDGE_ALPHA, add_const=ADD_CONST, min_obs=MIN_OBS)
        beta = fit['beta']
        residuals = fit['residuals']  # historical residuals inside window

        # 2) 因子协方差
        Sigma_F = cov_forecast.ewma_cov(macro_train, lam=LAMBDA_FACTOR)

        # 3) 风险计算，假设风险稳定
        risk = cov_forecast.forecast_risk(beta=beta, Sigma_F=Sigma_F, residuals=residuals, asset_history_returns=asset_train, lambda_resid=LAMBDA_RESID, cvar_conf_level=CVAR_CONF_LEVEL)
        Sigma_asset = risk['Sigma_asset']
        Sigma_eps = risk['Sigma_eps']
        cvar_residual = risk['cvar_residual']  # CVaR of residuals
        cvar_asset_hist = risk['cvar_asset_hist']  # CVaR of asset history (optional)

        # 4) 回测使用本月数据，实盘应使用分析师预期数据
        factor_next = macro.loc[rebalance_date].copy()
        if ADD_CONST and 'const' not in factor_next.index:
            factor_next['const'] = 1.0
        mu = return_forecast.forecast_returns(beta, factor_next)

        # 5) 最优权重配置
        try:
            w = weight_optimizer.max_sharpe_cvxpy(mu, Sigma_asset, long_only=LONG_ONLY, weight_sum_to_one=True, bounds=None, max_single=MAX_SINGLE,  mini_single=MINI_SINGLE)
        except Exception as e:
            logger.warning(f"Optimizer failed at {rebalance_date}: {e}. Using equal-weight fallback.")
            w = pd.Series(1.0/len(mu), index=mu.index)
        # 6) save intermediates (including CVaR)
        intermediates[rebalance_date] = {
            'beta': beta,
            'Sigma_F': Sigma_F,
            'Sigma_asset': Sigma_asset,
            'Sigma_eps': Sigma_eps,
            'mu': mu,
            'weights': w,
            'cvar_residual': cvar_residual,
            'cvar_asset_hist': cvar_asset_hist
        }
        weights_by_date[rebalance_date] = w

    # 储存中间数据和权重
    logger.info("Saving outputs...")
    utils.save_pickle(intermediates, os.path.join(OUTPUT_DIR, "processed_intermediates.pkl"))
    utils.save_pickle(weights_by_date, os.path.join(OUTPUT_DIR, "weights_by_date.pkl"))
    logger.info(f"Saved {len(weights_by_date)} weight entries and intermediates.")

    # Backtest
    logger.info("Running backtest...")
    assets=assets.iloc[WINDOW-1:]
    backtest_res = backtest.run_backtest(assets, weights_by_date, is_price=IS_PRICE, init_nav=1.0, trade_cost=TRADE_COST)
    perf = performance.perf_report(backtest_res['returns'])
    logger.info(f"Backtest performance: annual_return={perf['annual_return']:.4f}, annual_vol={perf['annual_vol']:.4f}, sharpe={perf['sharpe']:.4f}, mdd={perf['max_drawdown']:.4%}")

    # 储存回测结果/可视化
    utils.save_pickle(backtest_res, os.path.join(OUTPUT_DIR, "backtest_result.pkl"))
    weights_df = pd.DataFrame(weights_by_date).T.sort_index()
    weights_df.to_csv(os.path.join(OUTPUT_DIR, "weights_by_date.csv"))
    backtest_res['returns'].to_csv(os.path.join(OUTPUT_DIR, "portfolio_monthly_returns.csv"))
    perf['nav'].to_csv(os.path.join(OUTPUT_DIR, "portfolio_nav.csv"))

    plot_nav_and_weights(perf['nav'], backtest_res['weights'], OUTPUT_DIR)
    logger.info("Pipeline completed. All artifacts saved to output/")

    #输出CVAR
    cvar_resid_df = pd.DataFrame({d: intermediates[d]['cvar_residual'] for d in intermediates}).T.sort_index()
    cvar_resid_df.to_csv(os.path.join(OUTPUT_DIR, "cvar_residuals_by_date.csv"))
    if any(intermediates[d]['cvar_asset_hist'] is not None for d in intermediates):
        cvar_asset_df = pd.DataFrame({d: intermediates[d]['cvar_asset_hist'] for d in intermediates}).T.sort_index()
        cvar_asset_df.to_csv(os.path.join(OUTPUT_DIR, "cvar_asset_hist_by_date.csv"))

    return {'intermediates': intermediates, 'weights_by_date': weights_by_date, 'backtest': backtest_res, 'perf': perf}

if __name__ == "__main__":
    run()
