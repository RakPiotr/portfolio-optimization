import pandas as pd

from data_loader import DataLoader
from optimizer import mean_variance_model
from pso_optimizer import particle_swarm_optimization


def solve_with_gurobi(expected_returns, covariance_matrix, max_weight, risk_aversion):
    model, variables = mean_variance_model(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
    )
    model.setParam("OutputFlag", 0)
    model.optimize()

    return [variables[i].X for i in range(len(expected_returns))], model.ObjVal


def summarize_portfolio(tickers, weights, expected_returns, covariance_matrix, risk_aversion):
    weights = pd.Series(weights, index=tickers)
    expected_return = float(weights.to_numpy() @ expected_returns)
    variance = float(weights.to_numpy() @ covariance_matrix @ weights.to_numpy())
    objective = expected_return - risk_aversion * variance

    return {
        "objective": objective,
        "expected_return": expected_return,
        "variance": variance,
        "selected_assets": int((weights > 1e-6).sum()),
    }


def main():
    max_weight = 0.10
    risk_aversion = 1.0

    loader = DataLoader()

    gurobi_weights, _ = solve_with_gurobi(
        loader.expected_returns,
        loader.covariance_matrix,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
    )

    pso_result = particle_swarm_optimization(
        loader.expected_returns,
        loader.covariance_matrix,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
        swarm_size=80,
        iterations=1000,
        seed=42,
    )

    gurobi_summary = summarize_portfolio(
        loader.tickers,
        gurobi_weights,
        loader.expected_returns,
        loader.covariance_matrix,
        risk_aversion,
    )
    pso_summary = summarize_portfolio(
        loader.tickers,
        pso_result.weights,
        loader.expected_returns,
        loader.covariance_matrix,
        risk_aversion,
    )

    comparison = pd.DataFrame(
        [gurobi_summary, pso_summary],
        index=["Gurobi", "PSO"],
    )
    comparison["gap_to_gurobi"] = comparison.loc["Gurobi", "objective"] - comparison["objective"]

    print("\nObjective comparison")
    print(comparison)

    top_pso = (
        pd.DataFrame(
            {
                "ticker": loader.tickers,
                "weight": pso_result.weights,
            }
        )
        .query("weight > 1e-6")
        .sort_values("weight", ascending=False)
        .head(20)
    )

    print("\nTop PSO holdings")
    print(top_pso.to_string(index=False))


if __name__ == "__main__":
    main()
