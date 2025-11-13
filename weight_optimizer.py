import numpy as np
import pandas as pd
import cvxpy as cp

def max_sharpe_cvxpy(mu: pd.Series, Sigma: pd.DataFrame,
                     long_only: bool = True, weight_sum_to_one: bool = True,
                     bounds: dict = None, max_single: float = 0.2, mini_single: float = -0.2, rf: float = 0.0):
    assets = mu.index.tolist()
    n = len(assets)
    mu_vec = mu.to_numpy()
    Sigma_mat = Sigma.to_numpy()
    try:
        L = np.linalg.cholesky(Sigma_mat)
    except np.linalg.LinAlgError:
        jitter = 1e-8
        L = np.linalg.cholesky(Sigma_mat + np.eye(n) * jitter)
    w = cp.Variable(n)
    constraints = []
    if weight_sum_to_one:
        constraints.append(cp.sum(w) == 1)
    constraints.append(w >= mini_single)
    if long_only:
        constraints.append(w >= 0)
    if bounds is not None:
        for i, a in enumerate(assets):
            lb, ub = bounds.get(a, (None, None))
            if lb is not None:
                constraints.append(w[i] >= lb)
            if ub is not None:
                constraints.append(w[i] <= ub)
    if max_single is not None:
        constraints.append(w <= max_single)

    constraints.append(cp.norm(L @ w, 2) <= 1)
    excess_ret = mu_vec - rf
    objective = cp.Maximize(excess_ret @ w)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        raise RuntimeError("cvxpy failed to find solution")
    return pd.Series(w.value, index=assets)
