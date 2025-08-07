"""
Change Point Model Module
=========================

Implements change point detection models for analyzing structural breaks
in Brent oil price data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import warnings
from scipy.stats import ttest_ind

try:
    import pymc3 as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC3/ArviZ not available. Bayesian methods disabled.")

class ChangePointModel:
    """
    Change point detection model for identifying structural breaks in oil prices.
    """
    
    def __init__(self, data: pd.DataFrame, method: str = 'pelt'):
        self.data = data
        self.method = method
        self.change_points = []
        self.model_results = {}
    
    def detect_change_points(self, penalty: float = 10.0) -> Dict:
        """Detect change points in the time series."""
        try:
            if self.method == 'pelt':
                results = self._pelt_detection(penalty)
            elif self.method == 'binseg':
                results = self._binary_segmentation(penalty)
            else:
                results = self._sliding_window_detection()
            
            self.model_results = results
            logging.info(f"Detected {len(results.get('change_points', []))} change points")
            return results
            
        except Exception as e:
            logging.error(f"Change point detection failed: {str(e)}")
            raise
    
    def _pelt_detection(self, penalty: float) -> Dict:
        """PELT (Pruned Exact Linear Time) change point detection."""
        try:
            prices = self.data['Price'].values
            n = len(prices)
            
            if n < 20:
                return {'change_points': [], 'change_dates': [], 'method': 'PELT', 'penalty': penalty}
            
            # Find multiple change points
            change_points = []
            min_segment = 10
            
            for i in range(min_segment, n - min_segment):
                segment1 = prices[:i]
                segment2 = prices[i:]
                
                # Cost based on variance reduction
                full_var = np.var(prices)
                split_var = (len(segment1) * np.var(segment1) + len(segment2) * np.var(segment2)) / n
                improvement = full_var - split_var
                
                if improvement > penalty / 1000:  # Scaled penalty
                    change_points.append(i)
            
            # Filter nearby change points
            if change_points:
                filtered_cps = [change_points[0]]
                for cp in change_points[1:]:
                    if cp - filtered_cps[-1] > min_segment:
                        filtered_cps.append(cp)
                change_points = filtered_cps[:5]  # Limit to top 5
            
            return {
                'change_points': change_points,
                'change_dates': [self.data.iloc[cp]['Date'] for cp in change_points],
                'method': 'PELT',
                'penalty': penalty
            }
            
        except Exception as e:
            logging.error(f"PELT detection failed: {str(e)}")
            return {'change_points': [], 'change_dates': [], 'method': 'PELT'}
    
    def _binary_segmentation(self, penalty: float) -> Dict:
        """Binary segmentation change point detection."""
        try:
            prices = self.data['Price'].values
            change_points = []
            
            def find_best_split(start: int, end: int, depth: int = 0) -> List[int]:
                if end - start < 20 or depth > 3:  # Limit recursion
                    return []
                
                best_split = None
                best_improvement = 0
                
                for split in range(start + 10, end - 10):
                    seg1 = prices[start:split]
                    seg2 = prices[split:end]
                    full_seg = prices[start:end]
                    
                    improvement = np.var(full_seg) - (np.var(seg1) + np.var(seg2))
                    
                    if improvement > best_improvement and improvement > penalty / 100:
                        best_improvement = improvement
                        best_split = split
                
                if best_split is None:
                    return []
                
                # Recursively find more splits
                left_splits = find_best_split(start, best_split, depth + 1)
                right_splits = find_best_split(best_split, end, depth + 1)
                
                return left_splits + [best_split] + right_splits
            
            change_points = find_best_split(0, len(prices))
            change_points = sorted(list(set(change_points)))  # Remove duplicates and sort
            
            return {
                'change_points': change_points,
                'change_dates': [self.data.iloc[cp]['Date'] for cp in change_points],
                'method': 'Binary Segmentation',
                'penalty': penalty
            }
            
        except Exception as e:
            logging.error(f"Binary segmentation failed: {str(e)}")
            return {'change_points': [], 'change_dates': [], 'method': 'Binary Segmentation'}
    
    def _sliding_window_detection(self) -> Dict:
        """Sliding window change point detection."""
        try:
            prices = self.data['Price'].values
            window_size = min(50, len(prices) // 10)
            change_points = []
            
            for i in range(window_size, len(prices) - window_size):
                before = prices[i-window_size:i]
                after = prices[i:i+window_size]
                
                t_stat, p_value = ttest_ind(before, after)
                
                if p_value < 0.01:
                    change_points.append(i)
            
            # Remove nearby change points
            filtered_cps = []
            for cp in change_points:
                if not filtered_cps or cp - filtered_cps[-1] > window_size:
                    filtered_cps.append(cp)
            
            return {
                'change_points': filtered_cps,
                'change_dates': [self.data.iloc[cp]['Date'] for cp in filtered_cps],
                'method': 'Sliding Window',
                'window_size': window_size
            }
            
        except Exception as e:
            logging.error(f"Sliding window detection failed: {str(e)}")
            return {}

class BayesianChangePointModel:
    """
    Bayesian change point detection using PyMC3 for Brent oil price time series.
    """
    def __init__(self, series, dates=None):
        if not BAYESIAN_AVAILABLE:
            raise ImportError("PyMC3/ArviZ required for Bayesian analysis")
        
        self.series = np.asarray(series.dropna())
        self.dates = dates.iloc[series.dropna().index] if dates is not None else None
        self.model = None
        self.trace = None

    def build_model(self):
        n = len(self.series)
        with pm.Model() as model:
            tau = pm.DiscreteUniform('tau', lower=int(n*0.1), upper=int(n*0.9))
            mu1 = pm.Normal('mu1', mu=np.mean(self.series), sigma=np.std(self.series))
            mu2 = pm.Normal('mu2', mu=np.mean(self.series), sigma=np.std(self.series))
            sigma1 = pm.HalfNormal('sigma1', sigma=np.std(self.series))
            sigma2 = pm.HalfNormal('sigma2', sigma=np.std(self.series))
            
            idx = np.arange(n)
            mu = pm.math.switch(tau >= idx, mu1, mu2)
            sigma = pm.math.switch(tau >= idx, sigma1, sigma2)
            
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=self.series)
        self.model = model
        return model

    def fit(self, draws=1000, tune=500, chains=2):
        if self.model is None:
            self.build_model()
        with self.model:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.trace = pm.sample(draws=draws, tune=tune, chains=chains, 
                                     return_inferencedata=True, progressbar=False)
        return self.trace

    def get_change_point_summary(self):
        if self.trace is None:
            raise ValueError("Model must be fit before summarizing.")
        tau_samples = self.trace.posterior['tau'].values.flatten()
        mode_tau = int(np.median(tau_samples))
        ci_lower = int(np.percentile(tau_samples, 5))
        ci_upper = int(np.percentile(tau_samples, 95))
        
        result = {
            'mode_tau': mode_tau,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'probability': len(tau_samples) / len(tau_samples)  # Always 1 for fitted model
        }
        
        if self.dates is not None:
            result['mode_date'] = str(self.dates.iloc[mode_tau])
            result['ci_lower_date'] = str(self.dates.iloc[ci_lower])
            result['ci_upper_date'] = str(self.dates.iloc[ci_upper])
        
        return result

    def get_regime_parameters(self):
        """Get parameters for each regime."""
        if self.trace is None:
            raise ValueError("Model must be fit before getting parameters.")
        
        summary = az.summary(self.trace, var_names=['mu1', 'mu2', 'sigma1', 'sigma2'])
        return {
            'regime1_mean': summary.loc['mu1', 'mean'],
            'regime1_std': summary.loc['sigma1', 'mean'],
            'regime2_mean': summary.loc['mu2', 'mean'],
            'regime2_std': summary.loc['sigma2', 'mean']
        }

def correlate_with_events(change_points: List, events_data: pd.DataFrame, 
                         tolerance_days: int = 30) -> Dict:
    """
    Correlate detected change points with major events.
    """
    correlations = []
    
    for cp_date in change_points:
        cp_date = pd.to_datetime(cp_date)
        
        for _, event in events_data.iterrows():
            event_date = pd.to_datetime(event['Date'])
            days_diff = abs((cp_date - event_date).days)
            
            if days_diff <= tolerance_days:
                correlations.append({
                    'change_point_date': cp_date,
                    'event_date': event_date,
                    'event_description': event.get('Description', 'Unknown'),
                    'days_difference': days_diff
                })
    
    return {
        'correlations': correlations,
        'correlation_rate': len(correlations) / len(change_points) if change_points else 0
    }

def calculate_price_impact(data: pd.DataFrame, change_points: List, 
                          window_days: int = 30) -> Dict:
    """
    Calculate price impact around change points.
    """
    impacts = []
    
    for cp_idx in change_points:
        if cp_idx < window_days or cp_idx >= len(data) - window_days:
            continue
            
        before_prices = data.iloc[cp_idx-window_days:cp_idx]['Price']
        after_prices = data.iloc[cp_idx:cp_idx+window_days]['Price']
        
        before_mean = before_prices.mean()
        after_mean = after_prices.mean()
        
        impacts.append({
            'change_point_index': cp_idx,
            'absolute_change': after_mean - before_mean,
            'percentage_change': ((after_mean - before_mean) / before_mean) * 100
        })
    
    return {
        'impacts': impacts,
        'avg_percentage_change': np.mean([i['percentage_change'] for i in impacts]) if impacts else 0
    }