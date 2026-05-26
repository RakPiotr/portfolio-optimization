import gurobipy as gp
from gurobipy import GRB


def mean_variance_model(expected_returns, covariance_matrix, max_weight=0.10, risk_aversion=1.0):
    n = len(expected_returns)

    model = gp.Model("portfolio_optimization")

    # Decision variables: portfolio weights
    w = model.addVars(
        n,
        lb=0.0,              # no short selling
        ub=max_weight,       # max 10% per asset
        name="w"
    )

    # Fully invested: sum of weights = 1
    model.addConstr(
        gp.quicksum(w[i] for i in range(n)) == 1.0,
        name="fully_invested"
    )

    # Portfolio expected return
    portfolio_return = gp.quicksum(
        expected_returns[i] * w[i]
        for i in range(n)
    )

    # Portfolio variance
    portfolio_variance = gp.quicksum(
        covariance_matrix[i, j] * w[i] * w[j]
        for i in range(n)
        for j in range(n)
    )

    # Objective: maximize return - risk penalty
    model.setObjective(
        portfolio_return - risk_aversion * portfolio_variance,
        GRB.MAXIMIZE
    )

    return model, w