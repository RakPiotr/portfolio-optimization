# Mean-Variance Portfolio Optimization

## 1. Introduction

Portfolio optimization is the process of selecting asset weights according to a chosen investment objective and a set of constraints.

Different portfolio optimization problems can have different goals. For example, an investor may want to:

- maximize expected return,
- minimize investment risk,
- balance expected return and risk,
- limit exposure to individual assets,
- enforce diversification rules.

This project focuses on **mean-variance portfolio optimization**.

The goal is to construct a portfolio that balances:

- expected return,
- investment risk.

---
## 2. Asset Returns

An asset return measures how much an asset price changes over time.

For price $P_t$, the simple return is:

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

Historical returns are used to estimate expected future returns.

For each asset, the expected return is calculated as the average historical return:

$$
\mu_i = \frac{1}{T} \sum_{t=1}^{T} r_{i,t}
$$

These expected returns form the vector $\mu$, which is used in the optimization model.

---

## 3. Investment Risk and Covariance Matrix

In this project, investment risk is measured by **portfolio variance**.

Portfolio variance is:

$$
\sigma_p^2 = w^T \Sigma w
$$

Where $\Sigma$ is the covariance matrix of asset returns.

The covariance matrix shows how assets move together.

- Positive covariance means assets tend to move in the same direction.
- Negative covariance means assets tend to move in opposite directions.
- Low covariance means assets are weakly related.

This is important because portfolio risk depends not only on individual asset risk, but also on how assets interact with each other.

Diversification can reduce total portfolio risk when assets do not move perfectly together.

---

## 4. Portfolio Return

The expected portfolio return is the weighted average of the expected returns of all assets:

$$
\mu_p = w^T \mu
$$

Or:

$$
\mu_p = \sum_{i=1}^{n} \mu_i w_i
$$

A larger weight in an asset with a higher expected return increases the expected return of the portfolio.

---

## 5. Optimization Objective and Risk Aversion

The model maximizes:

$$
w^T \mu - \lambda w^T \Sigma w
$$

The first part, $w^T \mu$, rewards expected return.

The second part, $\lambda w^T \Sigma w$, penalizes risk.

The parameter $\lambda$ controls how important risk is in the model.

- A low $\lambda$ creates a more aggressive portfolio.
- A high $\lambda$ creates a more conservative portfolio.

This means the model balances expected return against variance depending on the investor’s risk preference.

---

## 6. Constraints

The optimization problem is solved with the following constraints.

First, the full portfolio must be invested:

$$
\sum_{i=1}^{n} w_i = 1
$$

Second, short selling is not allowed:

$$
w_i \ge 0
$$

Third, each asset has a maximum weight of 10%:

$$
w_i \le 0.10
$$

This prevents the optimizer from putting too much capital into one asset and helps improve diversification.

The final optimization problem is:

$$
\max_w \left( w^T \mu - \lambda w^T \Sigma w \right)
$$

Subject to:

$$
\sum_{i=1}^{n} w_i = 1
$$

$$
0 \le w_i \le 0.10
$$

---

## 7. Interpretation of the Results

The output of the optimization is a set of portfolio weights.

Each weight shows what percentage of the portfolio should be invested in a specific asset.

For example, if an asset has a weight of `0.08`, then 8% of the portfolio is invested in that asset.

The selected portfolio is the one that gives the best value of the objective function while satisfying all constraints.

Because each asset is limited to a maximum weight of 10%, the portfolio must be spread across multiple assets.

---

## 8. Dataset and Data Collection

The project uses historical market data to estimate expected returns and the covariance matrix.

Instead of collecting data manually from Yahoo Finance, a ready dataset from Kaggle was used.

The dataset contains S&P 500 company data and covers more than 25 years of historical market prices starting from January 2000.

This period includes major market events such as:

- the dot-com bubble collapse,
- the 2008 global financial crisis.

These events are useful because they show how assets behaved during difficult market conditions.

However, the dataset has survivorship bias because it does not include companies that went bankrupt, were delisted, or were removed from the S&P 500.

---

## 9. Limitations

This model has several limitations.

Expected returns and the covariance matrix are estimated from historical data, so they may not accurately predict future performance.

The model is also sensitive to small changes in input data. Small differences in expected returns or covariances can lead to different optimal portfolios.

Another limitation is that variance is used as the measure of risk. In reality, investors may also care about downside risk, drawdowns, liquidity, or transaction costs.

Finally, the model does not directly maximize the Sharpe ratio. It uses a mean-variance objective that balances return and risk through the risk-aversion parameter.

---

## 10. Conclusion

This project implements a mean-variance portfolio optimization model.

The model selects portfolio weights by maximizing expected return while penalizing variance.

The portfolio must satisfy three main rules:

- all capital is invested,
- short selling is not allowed,
- each asset is limited to a maximum weight of 10%.

The result is a diversified long-only portfolio that balances expected profitability and investment risk.