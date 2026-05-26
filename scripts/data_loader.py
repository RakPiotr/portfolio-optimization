import pandas as pd

class DataLoader:
    CSV_PATH = "data/SP500_Historical_Data.csv"

    def __init__(
        self,
        start_date=None,
        end_date=None
    ):
        self.df = pd.read_csv(self.CSV_PATH)

        # Create price matrix
        self.prices = self.df.pivot(
            index="Date",
            columns="Ticker",
            values="Adj Close"
        )

        # Ensure datetime index
        self.prices.index = pd.to_datetime(self.prices.index)

        # Filter date range
        self.prices = self.prices.loc[start_date:end_date]

        # Daily returns in %
        self.returns = self.prices.pct_change() * 100

        # For dropping NaN values
        self.returns = self.prices.pct_change().iloc[1:]
        self.returns = self.returns.dropna(axis=1)

        # Optimization inputs
        self.expected_returns = self.returns.mean().to_numpy()

        self.covariance_matrix = self.returns.cov().to_numpy()

        self.tickers = self.returns.columns.to_list()