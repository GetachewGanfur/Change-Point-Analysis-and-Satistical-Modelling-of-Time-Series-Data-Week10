"""
Dashboard Module for Brent Oil Price Analysis
============================================

Provides interactive visualization and analysis tools for the change point analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class BrentOilDashboard:
    """
    Interactive dashboard for Brent oil price change point analysis.
    
    Provides visualization tools and analysis capabilities for exploring
    oil price data, events, and change points.
    """
    
    def __init__(self, oil_data: pd.DataFrame, events_data: pd.DataFrame):
        """
        Initialize dashboard with data.
        
        Args:
            oil_data (pd.DataFrame): Brent oil price data
            events_data (pd.DataFrame): Events dataset
        """
        self.oil_data = oil_data
        self.events_data = events_data
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_price_series_with_events(self, start_date: Optional[str] = None, 
                                    end_date: Optional[str] = None):
        """
        Plot oil price series with major events.
        
        Args:
            start_date (str, optional): Start date for plot
            end_date (str, optional): End date for plot
        """
        # Filter data by date range
        plot_data = self.oil_data.copy()
        if start_date:
            plot_data = plot_data[plot_data['date'] >= start_date]
        if end_date:
            plot_data = plot_data[plot_data['date'] <= end_date]
        
        # Filter events by date range
        plot_events = self.events_data.copy()
        if start_date:
            plot_events = plot_events[plot_events['date'] >= start_date]
        if end_date:
            plot_events = plot_events[plot_events['date'] <= end_date]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Price series
        ax1.plot(plot_data['date'], plot_data['price'], linewidth=1, alpha=0.8, label='Oil Price')
        
        # Add event markers
        colors = {'Geopolitical': 'red', 'OPEC Decision': 'blue', 'Economic': 'green'}
        for _, event in plot_events.iterrows():
            ax1.axvline(x=event['date'], color=colors[event['category']], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        ax1.set_title('Brent Oil Prices with Major Events', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price (USD per barrel)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(['Oil Price'] + list(colors.keys()), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Event timeline
        for _, event in plot_events.iterrows():
            ax2.scatter(event['date'], 0, color=colors[event['category']], s=100, alpha=0.8)
            ax2.annotate(event['event'][:25] + '...', 
                        xy=(event['date'], 0), xytext=(5, 10),
                        textcoords='offset points', fontsize=8, rotation=45)
        
        ax2.set_title('Event Timeline', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_properties(self):
        """Plot comprehensive time series properties analysis."""
        # Calculate returns
        returns = self.oil_data['price'].pct_change().dropna()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price series
        axes[0, 0].plot(self.oil_data['date'], self.oil_data['price'], linewidth=1, alpha=0.8)
        axes[0, 0].set_title('Price Series', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[0, 1].hist(returns, bins=50, alpha=0.7, density=True)
        axes[0, 1].set_title('Returns Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        axes[1, 0].plot(self.oil_data['date'][1:], rolling_vol, linewidth=1, alpha=0.8)
        axes[1, 0].set_title('Rolling Volatility (30-day)', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Annualized Volatility')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Returns over time
        axes[1, 1].plot(self.oil_data['date'][1:], returns, linewidth=0.5, alpha=0.7)
        axes[1, 1].set_title('Returns Over Time', fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Returns')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_change_points_comparison(self, change_points_results: Dict):
        """
        Plot comparison of different change point detection methods.
        
        Args:
            change_points_results (Dict): Results from different detection methods
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot price series with change points
        ax1.plot(self.oil_data['date'], self.oil_data['price'], 'b-', alpha=0.7, 
                label='Oil Price', linewidth=1)
        
        # Add change points from different methods
        colors = {'pelt': 'red', 'binseg': 'green', 'window': 'orange'}
        for method, result in change_points_results.items():
            if 'change_dates' in result:
                for date in result['change_dates']:
                    ax1.axvline(x=date, color=colors[method], 
                               linestyle='--', alpha=0.8, linewidth=2, label=f'{method.upper()}')
        
        ax1.set_title('Change Point Detection Methods Comparison', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Compare number of change points
        method_counts = [len(change_points_results[m].get('change_points', [])) 
                        for m in change_points_results.keys()]
        method_names = list(change_points_results.keys())
        ax2.bar(method_names, method_counts, color=[colors.get(m, 'gray') for m in method_names])
        ax2.set_title('Number of Change Points by Method', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics.
        
        Returns:
            Dict: Summary statistics
        """
        stats = {
            'data_period': {
                'start_date': self.oil_data['date'].min(),
                'end_date': self.oil_data['date'].max(),
                'total_days': len(self.oil_data),
                'years_covered': (self.oil_data['date'].max() - self.oil_data['date'].min()).days / 365.25
            },
            'price_statistics': {
                'min_price': self.oil_data['price'].min(),
                'max_price': self.oil_data['price'].max(),
                'mean_price': self.oil_data['price'].mean(),
                'median_price': self.oil_data['price'].median(),
                'std_price': self.oil_data['price'].std()
            },
            'events_summary': {
                'total_events': len(self.events_data),
                'geopolitical_events': len(self.events_data[self.events_data['category'] == 'Geopolitical']),
                'opec_events': len(self.events_data[self.events_data['category'] == 'OPEC Decision']),
                'economic_events': len(self.events_data[self.events_data['category'] == 'Economic']),
                'high_impact_events': len(self.events_data[self.events_data['impact'] == 'High']),
                'medium_impact_events': len(self.events_data[self.events_data['impact'] == 'Medium'])
            }
        }
        
        return stats
    
    def print_summary_report(self):
        """Print comprehensive summary report."""
        stats = self.generate_summary_statistics()
        
        print("=" * 60)
        print("BRENT OIL PRICE ANALYSIS - SUMMARY REPORT")
        print("=" * 60)
        
        print(f"\nDATA PERIOD:")
        print(f"  Start Date: {stats['data_period']['start_date'].strftime('%Y-%m-%d')}")
        print(f"  End Date: {stats['data_period']['end_date'].strftime('%Y-%m-%d')}")
        print(f"  Total Days: {stats['data_period']['total_days']:,}")
        print(f"  Years Covered: {stats['data_period']['years_covered']:.1f}")
        
        print(f"\nPRICE STATISTICS:")
        print(f"  Minimum Price: ${stats['price_statistics']['min_price']:.2f}")
        print(f"  Maximum Price: ${stats['price_statistics']['max_price']:.2f}")
        print(f"  Mean Price: ${stats['price_statistics']['mean_price']:.2f}")
        print(f"  Median Price: ${stats['price_statistics']['median_price']:.2f}")
        print(f"  Standard Deviation: ${stats['price_statistics']['std_price']:.2f}")
        
        print(f"\nEVENTS SUMMARY:")
        print(f"  Total Events: {stats['events_summary']['total_events']}")
        print(f"  Geopolitical Events: {stats['events_summary']['geopolitical_events']}")
        print(f"  OPEC Decisions: {stats['events_summary']['opec_events']}")
        print(f"  Economic Events: {stats['events_summary']['economic_events']}")
        print(f"  High Impact Events: {stats['events_summary']['high_impact_events']}")
        print(f"  Medium Impact Events: {stats['events_summary']['medium_impact_events']}")
        
        print("\n" + "=" * 60)