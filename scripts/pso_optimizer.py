from dataclasses import dataclass

import numpy as np


@dataclass
class PSOResult:
    weights: np.ndarray
    objective_value: float
    expected_return: float
    variance: float
    history: list[float]


def portfolio_objective(weights, expected_returns, covariance_matrix, risk_aversion=1.0):
    """Return mean-variance objective value for one portfolio."""
    expected_return = float(weights @ expected_returns)
    variance = float(weights @ covariance_matrix @ weights)
    return expected_return - risk_aversion * variance


def _evaluate_swarm(positions, expected_returns, covariance_matrix, risk_aversion):
    returns = positions @ expected_returns
    variances = np.einsum("ij,jk,ik->i", positions, covariance_matrix, positions)
    return returns - risk_aversion * variances


def project_to_bounded_simplex(values, max_weight=0.10, total_weight=1.0):
    """
    Project values onto {w: sum(w) = total_weight, 0 <= w_i <= max_weight}.

    This keeps PSO particles feasible after each velocity update.
    """
    values = np.asarray(values, dtype=float)
    n_assets = values.size

    if total_weight < 0:
        raise ValueError("total_weight must be non-negative")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    if n_assets * max_weight + 1e-12 < total_weight:
        raise ValueError("No feasible portfolio: n_assets * max_weight < total_weight")

    lower = values.min() - max_weight
    upper = values.max()

    for _ in range(100):
        theta = (lower + upper) / 2.0
        projected = np.clip(values - theta, 0.0, max_weight)

        if projected.sum() > total_weight:
            lower = theta
        else:
            upper = theta

    projected = np.clip(values - upper, 0.0, max_weight)

    # Remove tiny numerical drift while preserving bounds.
    difference = total_weight - projected.sum()
    if abs(difference) > 1e-12:
        if difference > 0:
            free = projected < max_weight - 1e-12
            if free.any():
                projected[free] += difference / free.sum()
        else:
            free = projected > 1e-12
            if free.any():
                projected[free] += difference / free.sum()
        projected = np.clip(projected, 0.0, max_weight)

    return projected


def _initial_positions(rng, swarm_size, n_assets, max_weight):
    raw_positions = rng.dirichlet(np.ones(n_assets), size=swarm_size)
    return np.array(
        [project_to_bounded_simplex(position, max_weight=max_weight) for position in raw_positions]
    )


def particle_swarm_optimization(
    expected_returns,
    covariance_matrix,
    max_weight=0.10,
    risk_aversion=1.0,
    swarm_size=60,
    iterations=500,
    inertia=0.70,
    cognitive=1.50,
    social=1.50,
    max_velocity=0.05,
    seed=None,
):
    """
    Maximize the mean-variance portfolio objective using Particle Swarm Optimization.

    Each particle is a candidate portfolio. After every move, its weights are projected
    back to the feasible set: fully invested, long-only, and capped by max_weight.
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)
    n_assets = expected_returns.size

    if covariance_matrix.shape != (n_assets, n_assets):
        raise ValueError("covariance_matrix must have shape (n_assets, n_assets)")
    if swarm_size <= 0:
        raise ValueError("swarm_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)

    positions = _initial_positions(rng, swarm_size, n_assets, max_weight)
    velocities = rng.uniform(-max_velocity, max_velocity, size=(swarm_size, n_assets))

    personal_best_positions = positions.copy()
    personal_best_scores = _evaluate_swarm(
        positions, expected_returns, covariance_matrix, risk_aversion
    )

    best_index = int(np.argmax(personal_best_scores))
    global_best_position = personal_best_positions[best_index].copy()
    global_best_score = float(personal_best_scores[best_index])
    history = [global_best_score]

    for _ in range(iterations):
        r1 = rng.random(size=(swarm_size, n_assets))
        r2 = rng.random(size=(swarm_size, n_assets))

        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best_positions - positions)
            + social * r2 * (global_best_position - positions)
        )
        velocities = np.clip(velocities, -max_velocity, max_velocity)

        positions = positions + velocities
        positions = np.array(
            [project_to_bounded_simplex(position, max_weight=max_weight) for position in positions]
        )

        scores = _evaluate_swarm(positions, expected_returns, covariance_matrix, risk_aversion)
        improved = scores > personal_best_scores
        personal_best_positions[improved] = positions[improved]
        personal_best_scores[improved] = scores[improved]

        best_index = int(np.argmax(personal_best_scores))
        if personal_best_scores[best_index] > global_best_score:
            global_best_score = float(personal_best_scores[best_index])
            global_best_position = personal_best_positions[best_index].copy()

        history.append(global_best_score)

    expected_return = float(global_best_position @ expected_returns)
    variance = float(global_best_position @ covariance_matrix @ global_best_position)

    return PSOResult(
        weights=global_best_position,
        objective_value=global_best_score,
        expected_return=expected_return,
        variance=variance,
        history=history,
    )
