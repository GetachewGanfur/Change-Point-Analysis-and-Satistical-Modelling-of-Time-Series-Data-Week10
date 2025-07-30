"""
Bayesian Change Point Analysis models for Brent oil price data.

This module implements various change point detection models using PyMC3
to identify structural breaks in oil price time series.
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import theano.tensor as tt
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianChangePointModel:
    """
    Bayesian Change Point Analysis model for detecting structural breaks.
    
    This class implements a Bayesian approach to change point detection
    using PyMC3 for Markov Chain Monte Carlo sampling.
    """
    
    def __init__(self, data: pd.Series, model_type: str = 'mean_change'):
        """
        Initialize the change point model.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data to analyze
        model_type : str
            Type of change point model ('mean_change', 'variance_change', 'both')
        """
        self.data = np.array(data)
        self.n_observations = len(self.data)
        self.model_type = model_type
        self.model = None
        self.trace = None
        self.summary = None
        
        logger.info(f"Initialized {model_type} change point model with {self.n_observations} observations")
    
    def build_mean_change_model(self) -> pm.Model:
        """
        Build a change point model for detecting changes in mean.
        
        Returns:
        --------
        pm.Model
            PyMC3 model for change point detection
        """
        with pm.Model() as model:
            # Prior for the change point location (uniform over all possible points)
            tau = pm.DiscreteUniform('tau', lower=0, upper=self.n_observations - 1)
            
            # Priors for the means before and after the change point
            mu_1 = pm.Normal('mu_1', mu=np.mean(self.data), sigma=np.std(self.data))
            mu_2 = pm.Normal('mu_2', mu=np.mean(self.data), sigma=np.std(self.data))
            
            # Prior for the standard deviation (assumed constant)
            sigma = pm.HalfNormal('sigma', sigma=np.std(self.data))
            
            # Create time index
            idx = np.arange(self.n_observations)
            
            # Switch function to select appropriate mean
            mu = tt.switch(tau >= idx, mu_1, mu_2)
            
            # Likelihood
            likelihood = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=self.data)
            
        logger.info("Built mean change point model")
        return model
    
    def build_variance_change_model(self) -> pm.Model:
        """
        Build a change point model for detecting changes in variance.
        
        Returns:
        --------
        pm.Model
            PyMC3 model for variance change point detection
        """
        with pm.Model() as model:
            # Prior for the change point location
            tau = pm.DiscreteUniform('tau', lower=0, upper=self.n_observations - 1)
            
            # Prior for the mean (assumed constant)
            mu = pm.Normal('mu', mu=np.mean(self.data), sigma=np.std(self.data))
            
            # Priors for the standard deviations before and after change point
            sigma_1 = pm.HalfNormal('sigma_1', sigma=np.std(self.data))
            sigma_2 = pm.HalfNormal('sigma_2', sigma=np.std(self.data))
            
            # Create time index
            idx = np.arange(self.n_observations)
            
            # Switch function to select appropriate variance
            sigma = tt.switch(tau >= idx, sigma_1, sigma_2)
            
            # Likelihood
            likelihood = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=self.data)
            
        logger.info("Built variance change point model")
        return model
    
    def build_combined_change_model(self) -> pm.Model:
        """
        Build a change point model for detecting changes in both mean and variance.
        
        Returns:
        --------
        pm.Model
            PyMC3 model for combined change point detection
        """
        with pm.Model() as model:
            # Prior for the change point location
            tau = pm.DiscreteUniform('tau', lower=0, upper=self.n_observations - 1)
            
            # Priors for the means before and after the change point
            mu_1 = pm.Normal('mu_1', mu=np.mean(self.data), sigma=np.std(self.data))
            mu_2 = pm.Normal('mu_2', mu=np.mean(self.data), sigma=np.std(self.data))
            
            # Priors for the standard deviations before and after change point
            sigma_1 = pm.HalfNormal('sigma_1', sigma=np.std(self.data))
            sigma_2 = pm.HalfNormal('sigma_2', sigma=np.std(self.data))
            
            # Create time index
            idx = np.arange(self.n_observations)
            
            # Switch functions to select appropriate parameters
            mu = tt.switch(tau >= idx, mu_1, mu_2)
            sigma = tt.switch(tau >= idx, sigma_1, sigma_2)
            
            # Likelihood
            likelihood = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=self.data)
            
        logger.info("Built combined change point model")
        return model
    
    def fit_model(self, draws: int = 2000, tune: int = 2000, chains: int = 2, 
                  target_accept: float = 0.95) -> az.InferenceData:
        """
        Fit the change point model using MCMC sampling.
        
        Parameters:
        -----------
        draws : int
            Number of MCMC samples to draw
        tune : int
            Number of tuning samples
        chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance rate for NUTS sampler
            
        Returns:
        --------
        az.InferenceData
            ArviZ inference data object with sampling results
        """
        # Build the appropriate model
        if self.model_type == 'mean_change':
            self.model = self.build_mean_change_model()
        elif self.model_type == 'variance_change':
            self.model = self.build_variance_change_model()
        elif self.model_type == 'both':
            self.model = self.build_combined_change_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Sample from the model
        with self.model:
            logger.info(f"Starting MCMC sampling with {draws} draws, {tune} tune, {chains} chains")
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
                random_seed=42
            )
        
        # Generate model summary
        self.summary = az.summary(self.trace)
        
        logger.info("MCMC sampling completed successfully")
        return self.trace
    
    def get_change_point_probability(self) -> np.ndarray:
        """
        Calculate the probability of a change point at each time index.
        
        Returns:
        --------
        np.ndarray
            Probability of change point at each time index
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before calculating probabilities")
        
        tau_samples = self.trace.posterior['tau'].values.flatten()
        
        # Calculate probability for each time point
        probabilities = np.zeros(self.n_observations)
        for i in range(self.n_observations):
            probabilities[i] = np.mean(tau_samples == i)
        
        return probabilities
    
    def get_most_probable_change_point(self) -> Tuple[int, float]:
        """
        Get the most probable change point location and its probability.
        
        Returns:
        --------
        Tuple[int, float]
            Index of most probable change point and its probability
        """
        probabilities = self.get_change_point_probability()
        most_probable_idx = np.argmax(probabilities)
        max_probability = probabilities[most_probable_idx]
        
        return most_probable_idx, max_probability
    
    def get_parameter_estimates(self) -> Dict:
        """
        Get posterior estimates of model parameters.
        
        Returns:
        --------
        Dict
            Dictionary containing parameter estimates
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before getting parameter estimates")
        
        estimates = {}
        
        for var in self.trace.posterior.data_vars:
            if var != 'y_obs':  # Skip observed data
                estimates[var] = {
                    'mean': float(self.trace.posterior[var].mean()),
                    'std': float(self.trace.posterior[var].std()),
                    'hdi_95': az.hdi(self.trace.posterior[var], hdi_prob=0.95).values.tolist()
                }
        
        return estimates
    
    def check_convergence(self) -> Dict:
        """
        Check MCMC convergence diagnostics.
        
        Returns:
        --------
        Dict
            Convergence diagnostics including R-hat and ESS
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before checking convergence")
        
        diagnostics = {
            'r_hat': az.rhat(self.trace).to_dict(),
            'ess_bulk': az.ess(self.trace, method='bulk').to_dict(),
            'ess_tail': az.ess(self.trace, method='tail').to_dict()
        }
        
        # Check if all R-hat values are close to 1
        r_hat_ok = all(
            r_hat < 1.1 for var_dict in diagnostics['r_hat'].values() 
            for r_hat in var_dict.values() if not np.isnan(r_hat)
        )
        
        diagnostics['convergence_ok'] = r_hat_ok
        
        return diagnostics


class MultipleChangePointModel:
    """
    Model for detecting multiple change points in a time series.
    """
    
    def __init__(self, data: pd.Series, max_change_points: int = 3):
        """
        Initialize multiple change point model.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        max_change_points : int
            Maximum number of change points to detect
        """
        self.data = np.array(data)
        self.n_observations = len(self.data)
        self.max_change_points = max_change_points
        self.model = None
        self.trace = None
        
        logger.info(f"Initialized multiple change point model with max {max_change_points} change points")
    
    def build_model(self) -> pm.Model:
        """
        Build model for detecting multiple change points.
        
        Returns:
        --------
        pm.Model
            PyMC3 model for multiple change point detection
        """
        with pm.Model() as model:
            # Number of change points (between 0 and max_change_points)
            n_changepoints = pm.DiscreteUniform('n_changepoints', 
                                               lower=0, 
                                               upper=self.max_change_points)
            
            # Change point locations (sorted)
            tau_raw = pm.DiscreteUniform('tau_raw', 
                                        lower=0, 
                                        upper=self.n_observations - 1,
                                        shape=self.max_change_points)
            
            # Sort the change points
            tau = tt.sort(tau_raw)
            
            # Parameters for each segment
            mu = pm.Normal('mu', mu=np.mean(self.data), sigma=np.std(self.data),
                          shape=self.max_change_points + 1)
            sigma = pm.HalfNormal('sigma', sigma=np.std(self.data),
                                 shape=self.max_change_points + 1)
            
            # Create segment indicators
            idx = np.arange(self.n_observations)
            
            # Determine which segment each observation belongs to
            segment = tt.zeros(self.n_observations, dtype='int32')
            for i in range(self.max_change_points):
                segment += (idx > tau[i]) * (i < n_changepoints)
            
            # Select appropriate parameters for each observation
            mu_obs = mu[segment]
            sigma_obs = sigma[segment]
            
            # Likelihood
            likelihood = pm.Normal('y_obs', mu=mu_obs, sigma=sigma_obs, 
                                  observed=self.data)
            
        logger.info("Built multiple change point model")
        return model
    
    def fit_model(self, draws: int = 2000, tune: int = 2000, chains: int = 2) -> az.InferenceData:
        """
        Fit the multiple change point model.
        
        Parameters:
        -----------
        draws : int
            Number of MCMC samples
        tune : int
            Number of tuning samples
        chains : int
            Number of MCMC chains
            
        Returns:
        --------
        az.InferenceData
            Sampling results
        """
        self.model = self.build_model()
        
        with self.model:
            logger.info("Starting MCMC sampling for multiple change point model")
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                return_inferencedata=True,
                random_seed=42
            )
        
        logger.info("Multiple change point model fitted successfully")
        return self.trace


def detect_change_points(data: pd.Series, 
                        model_type: str = 'mean_change',
                        multiple: bool = False,
                        **kwargs) -> Dict:
    """
    Convenience function to detect change points in time series data.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    model_type : str
        Type of change point model
    multiple : bool
        Whether to detect multiple change points
    **kwargs
        Additional arguments for model fitting
        
    Returns:
    --------
    Dict
        Results including change point locations and model diagnostics
    """
    if multiple:
        model = MultipleChangePointModel(data, **kwargs)
    else:
        model = BayesianChangePointModel(data, model_type)
    
    # Fit the model
    trace = model.fit_model(**kwargs)
    
    # Get results
    results = {
        'model': model,
        'trace': trace,
        'convergence': model.check_convergence() if not multiple else None
    }
    
    if not multiple:
        cp_idx, cp_prob = model.get_most_probable_change_point()
        results.update({
            'change_point_index': cp_idx,
            'change_point_probability': cp_prob,
            'change_point_probabilities': model.get_change_point_probability(),
            'parameter_estimates': model.get_parameter_estimates()
        })
    
    return results