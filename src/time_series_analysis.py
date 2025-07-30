import numpy as np
from statsmodels.tsa.stattools import adfuller

class TimeSeriesAnalyzer:
    def __init__(self, df):
        self.df = df

    def compute_log_returns(self):
        self.df['LogReturn'] = np.log(self.df['Price']).diff()
        return self.df

    def test_stationarity(self, column='LogReturn'):
        result = adfuller(self.df[column].dropna())
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }