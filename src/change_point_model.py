import pymc3 as pm
import numpy as np

class ChangePointModel:
    def __init__(self, series):
        self.series = series.dropna().values

    def fit(self):
        n = len(self.series)
        with pm.Model() as model:
            tau = pm.DiscreteUniform('tau', lower=0, upper=n-1)
            mu1 = pm.Normal('mu1', mu=np.mean(self.series), sigma=np.std(self.series))
            mu2 = pm.Normal('mu2', mu=np.mean(self.series), sigma=np.std(self.series))
            sigma = pm.HalfNormal('sigma', sigma=1)
            mu = pm.math.switch(tau >= np.arange(n), mu1, mu2)
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=self.series)
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)
        self.trace = trace
        return trace