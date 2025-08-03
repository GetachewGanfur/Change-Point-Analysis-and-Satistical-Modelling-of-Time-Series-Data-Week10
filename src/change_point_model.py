import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats

class EnhancedChangePointModel:
    def __init__(self, data, dates=None):
        self.data = np.array(data)
        self.dates = dates
        self.n = len(self.data)
        self.model = None
        self.trace = None
        
    def fit(self, n_samples=2000, tune=1000, chains=2):
        """Fit Bayesian change point model"""
        with pm.Model() as model:
            # Change point location
            tau = pm.DiscreteUniform('tau', lower=10, upper=self.n-10)
            
            # Parameters before and after change point
            mu1 = pm.Normal('mu1', mu=0, sigma=1)
            mu2 = pm.Normal('mu2', mu=0, sigma=1)
            sigma1 = pm.HalfNormal('sigma1', sigma=1)
            sigma2 = pm.HalfNormal('sigma2', sigma=1)
            
            # Switch function
            idx = np.arange(self.n)
            mu = pm.math.switch(tau >= idx, mu1, mu2)
            sigma = pm.math.switch(tau >= idx, sigma1, sigma2)
            
            # Likelihood
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=self.data)
            
            # Sample
            self.trace = pm.sample(n_samples, tune=tune, chains=chains, 
                                 return_inferencedata=True, random_seed=42)
            
        self.model = model
        return self.trace
    
    def get_change_point_summary(self):
        """Get change point summary statistics"""
        tau_samples = self.trace.posterior['tau'].values.flatten()
        
        summary = {
            'mode_tau': int(stats.mode(tau_samples)[0][0]),
            'mean_tau': np.mean(tau_samples),
            'median_tau': np.median(tau_samples),
            'tau_95_hdi': az.hdi(tau_samples, hdi_prob=0.95)
        }
        
        if self.dates is not None:
            summary['mode_date'] = self.dates.iloc[summary['mode_tau']]
            summary['mean_date'] = self.dates.iloc[int(summary['mean_tau'])]
            summary['median_date'] = self.dates.iloc[int(summary['median_tau'])]
            
        return summary
    
    def get_parameter_summary(self):
        """Get parameter summary statistics"""
        return az.summary(self.trace, var_names=['mu1', 'mu2', 'sigma1', 'sigma2'])
    
    def check_convergence(self):
        """Check model convergence"""
        rhat = az.rhat(self.trace)
        ess = az.ess(self.trace)
        
        return {
            'converged': all(rhat[var].max() < 1.1 for var in rhat.data_vars),
            'rhat': rhat,
            'ess': ess
        }
    
    def plot_results(self, figsize=(14, 10)):
        """Plot comprehensive results"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Data with change point
        tau_mode = int(stats.mode(self.trace.posterior['tau'].values.flatten())[0][0])
        
        axes[0, 0].plot(self.data, 'b-', alpha=0.7, linewidth=0.8)
        axes[0, 0].axvline(tau_mode, color='red', linestyle='--', linewidth=2, 
                          label=f'Change Point (τ={tau_mode})')
        axes[0, 0].set_title('Data with Detected Change Point')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Change point posterior
        tau_samples = self.trace.posterior['tau'].values.flatten()
        axes[0, 1].hist(tau_samples, bins=50, alpha=0.7, density=True)
        axes[0, 1].axvline(tau_mode, color='red', linestyle='--', 
                          label=f'Mode = {tau_mode}')
        axes[0, 1].set_title('Change Point Posterior Distribution')
        axes[0, 1].set_xlabel('τ (Change Point Index)')
        axes[0, 1].legend()
        
        # Parameter comparison
        mu1_samples = self.trace.posterior['mu1'].values.flatten()
        mu2_samples = self.trace.posterior['mu2'].values.flatten()
        
        axes[1, 0].hist(mu1_samples, bins=50, alpha=0.5, label='μ₁ (before)', density=True)
        axes[1, 0].hist(mu2_samples, bins=50, alpha=0.5, label='μ₂ (after)', density=True)
        axes[1, 0].set_title('Mean Parameters')
        axes[1, 0].legend()
        
        # Volatility comparison
        sigma1_samples = self.trace.posterior['sigma1'].values.flatten()
        sigma2_samples = self.trace.posterior['sigma2'].values.flatten()
        
        axes[1, 1].hist(sigma1_samples, bins=50, alpha=0.5, label='σ₁ (before)', density=True)
        axes[1, 1].hist(sigma2_samples, bins=50, alpha=0.5, label='σ₂ (after)', density=True)
        axes[1, 1].set_title('Volatility Parameters')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()