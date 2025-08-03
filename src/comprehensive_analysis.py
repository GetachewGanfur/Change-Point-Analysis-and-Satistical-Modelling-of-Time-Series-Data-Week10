import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from event_manager import EventManager
from change_point_model import EnhancedChangePointModel
from time_series_analysis import TimeSeriesAnalyzer

class ComprehensiveAnalysis:
    def __init__(self, data_path):
        """Initialize comprehensive analysis with data path"""
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)
        self.event_manager = EventManager()
        self.df = None
        self.model = None
        
    def load_and_prepare_data(self):
        """Load and prepare the data for analysis"""
        print("Loading and preparing data...")
        self.df = self.data_loader.load_data()
        
        # Handle date format inconsistencies in the CSV
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df = self.df.dropna(subset=['Date'])
        
        # Sort by date
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['Date'].min()} to {self.df['Date'].max()}")
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("\nPerforming Exploratory Data Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Price over time
        axes[0, 0].plot(self.df['Date'], self.df['Price'], linewidth=0.8)
        axes[0, 0].set_title('Brent Oil Prices Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price (USD/barrel)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Log returns
        axes[0, 1].plot(self.df['Date'], self.df['Log_Returns'], linewidth=0.5, alpha=0.7)
        axes[0, 1].set_title('Log Returns')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Log Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Price distribution
        axes[1, 0].hist(self.df['Price'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Price Distribution')
        axes[1, 0].set_xlabel('Price (USD/barrel)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Rolling volatility
        rolling_std = self.df['Log_Returns'].rolling(window=30).std()
        axes[1, 1].plot(self.df['Date'], rolling_std, linewidth=0.8)
        axes[1, 1].set_title('30-Day Rolling Volatility')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        print("\nSummary Statistics:")
        print(self.df[['Price', 'Log_Returns']].describe())
        
    def run_change_point_analysis(self, series_type='log_returns', n_samples=2000):
        """Run change point analysis on the specified series"""
        print(f"\nRunning Change Point Analysis on {series_type}...")
        
        if series_type == 'log_returns':
            series = self.df['Log_Returns'].dropna()
            dates = self.df['Date'][1:]  # Skip first date due to diff
        elif series_type == 'price':
            series = self.df['Price']
            dates = self.df['Date']
        else:
            raise ValueError("series_type must be 'log_returns' or 'price'")
        
        # Initialize and fit model
        self.model = EnhancedChangePointModel(series, dates)
        trace = self.model.fit(n_samples=n_samples, tune=1000, chains=2)
        
        # Check convergence
        convergence = self.model.check_convergence()
        print(f"Model converged: {convergence['converged']}")
        
        # Get change point summary
        cp_summary = self.model.get_change_point_summary()
        print(f"\nChange Point Analysis Results:")
        print(f"Most likely change point (mode): Index {cp_summary['mode_tau']}")
        if 'mode_date' in cp_summary:
            print(f"Most likely change point date: {cp_summary['mode_date']}")
        
        # Plot results
        self.model.plot_results()
        
        return cp_summary
    
    def associate_with_events(self, change_point_summary, days_threshold=60):
        """Associate detected change points with historical events"""
        print(f"\nAssociating change points with historical events (±{days_threshold} days)...")
        
        if 'mode_date' not in change_point_summary:
            print("No date information available for change point")
            return None
        
        change_date = change_point_summary['mode_date']
        
        # Find events near the change point
        nearby_events = self.event_manager.find_nearest_event(change_date, days_threshold)
        
        if len(nearby_events) > 0:
            print(f"\nEvents near change point ({change_date}):")
            for _, event in nearby_events.head(5).iterrows():
                print(f"- {event['date'].strftime('%Y-%m-%d')}: {event['event']} "
                      f"({event['category']}, {event['impact']} impact) "
                      f"[{event['days_diff']} days difference]")
        else:
            print(f"No major events found within {days_threshold} days of the change point")
        
        return nearby_events
    
    def quantify_impact(self, change_point_summary):
        """Quantify the impact of the change point"""
        print("\nQuantifying Change Point Impact...")
        
        if self.model is None:
            print("No model available. Run change point analysis first.")
            return None
        
        # Get parameter summaries
        param_summary = self.model.get_parameter_summary()
        
        mu1_mean = param_summary.loc['mu1', 'mean']
        mu2_mean = param_summary.loc['mu2', 'mean']
        
        # Calculate impact metrics
        absolute_change = mu2_mean - mu1_mean
        relative_change = (mu2_mean - mu1_mean) / abs(mu1_mean) * 100
        
        print(f"Mean before change point (μ₁): {mu1_mean:.4f}")
        print(f"Mean after change point (μ₂): {mu2_mean:.4f}")
        print(f"Absolute change: {absolute_change:.4f}")
        print(f"Relative change: {relative_change:.2f}%")
        
        # Variance analysis
        if 'sigma1' in param_summary.index and 'sigma2' in param_summary.index:
            sigma1_mean = param_summary.loc['sigma1', 'mean']
            sigma2_mean = param_summary.loc['sigma2', 'mean']
            
            variance_change = (sigma2_mean**2 - sigma1_mean**2) / sigma1_mean**2 * 100
            print(f"Variance change: {variance_change:.2f}%")
        
        return {
            'mu1': mu1_mean,
            'mu2': mu2_mean,
            'absolute_change': absolute_change,
            'relative_change': relative_change
        }
    
    def create_event_timeline(self, start_year=2000, end_year=2022):
        """Create a timeline visualization with events and price data"""
        print(f"\nCreating event timeline from {start_year} to {end_year}...")
        
        # Filter data for the specified period
        start_date = pd.to_datetime(f'{start_year}-01-01')
        end_date = pd.to_datetime(f'{end_year}-12-31')
        
        period_data = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)]
        period_events = self.event_manager.get_events_in_period(start_date, end_date)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot price data
        ax1.plot(period_data['Date'], period_data['Price'], linewidth=1, color='blue', alpha=0.7)
        ax1.set_ylabel('Price (USD/barrel)')
        ax1.set_title('Brent Oil Prices with Major Events')
        ax1.grid(True, alpha=0.3)
        
        # Add event markers
        colors = {'Conflict': 'red', 'Economic': 'orange', 'OPEC': 'green', 
                 'Geopolitical': 'purple', 'Natural Disaster': 'brown'}
        
        for _, event in period_events.iterrows():
            color = colors.get(event['category'], 'gray')
            ax1.axvline(event['date'], color=color, linestyle='--', alpha=0.7, linewidth=1)
            
            # Add text annotation for high impact events
            if event['impact'] == 'High':
                ax1.annotate(event['event'][:30] + '...', 
                           xy=(event['date'], ax1.get_ylim()[1]), 
                           xytext=(10, -10), textcoords='offset points',
                           rotation=45, fontsize=8, alpha=0.8)
        
        # Plot log returns
        ax2.plot(period_data['Date'], period_data['Log_Returns'], linewidth=0.5, color='red', alpha=0.7)
        ax2.set_ylabel('Log Returns')
        ax2.set_xlabel('Date')
        ax2.set_title('Log Returns')
        ax2.grid(True, alpha=0.3)
        
        # Add event markers to returns plot
        for _, event in period_events.iterrows():
            color = colors.get(event['category'], 'gray')
            ax2.axvline(event['date'], color=color, linestyle='--', alpha=0.7, linewidth=1)
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], color=color, linestyle='--', label=category) 
                          for category, color in colors.items()]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, change_point_summary, impact_analysis, nearby_events):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE CHANGE POINT ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nDATA OVERVIEW:")
        print(f"- Dataset: Brent Oil Prices")
        print(f"- Period: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"- Total observations: {len(self.df)}")
        
        print(f"\nCHANGE POINT DETECTION:")
        if 'mode_date' in change_point_summary:
            print(f"- Most probable change point: {change_point_summary['mode_date']}")
        print(f"- Change point index: {change_point_summary['mode_tau']}")
        
        print(f"\nIMPACT QUANTIFICATION:")
        if impact_analysis:
            print(f"- Mean before change: {impact_analysis['mu1']:.4f}")
            print(f"- Mean after change: {impact_analysis['mu2']:.4f}")
            print(f"- Absolute change: {impact_analysis['absolute_change']:.4f}")
            print(f"- Relative change: {impact_analysis['relative_change']:.2f}%")
        
        print(f"\nEVENT ASSOCIATION:")
        if nearby_events is not None and len(nearby_events) > 0:
            closest_event = nearby_events.iloc[0]
            print(f"- Closest event: {closest_event['event']}")
            print(f"- Event date: {closest_event['date']}")
            print(f"- Event category: {closest_event['category']}")
            print(f"- Impact level: {closest_event['impact']}")
            print(f"- Days from change point: {closest_event['days_diff']}")
        else:
            print("- No major events found near the change point")
        
        print(f"\nCONCLUSION:")
        if nearby_events is not None and len(nearby_events) > 0:
            closest_event = nearby_events.iloc[0]
            direction = "increase" if impact_analysis['absolute_change'] > 0 else "decrease"
            print(f"The analysis detected a significant structural break in Brent oil prices")
            print(f"around {change_point_summary.get('mode_date', 'the detected time point')}.")
            print(f"This change point shows a {direction} in the mean level of ")
            print(f"{abs(impact_analysis['relative_change']):.1f}%, which appears to be")
            print(f"associated with '{closest_event['event']}' that occurred")
            print(f"{closest_event['days_diff']} days from the change point.")
        
        print("="*60)
    
    def run_full_analysis(self, series_type='log_returns', n_samples=1500):
        """Run the complete analysis workflow"""
        print("Starting Comprehensive Brent Oil Price Change Point Analysis")
        print("="*60)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Exploratory data analysis
        self.exploratory_data_analysis()
        
        # Step 3: Run change point analysis
        change_point_summary = self.run_change_point_analysis(series_type, n_samples)
        
        # Step 4: Associate with events
        nearby_events = self.associate_with_events(change_point_summary)
        
        # Step 5: Quantify impact
        impact_analysis = self.quantify_impact(change_point_summary)
        
        # Step 6: Create event timeline
        self.create_event_timeline()
        
        # Step 7: Generate report
        self.generate_report(change_point_summary, impact_analysis, nearby_events)
        
        return {
            'change_point_summary': change_point_summary,
            'impact_analysis': impact_analysis,
            'nearby_events': nearby_events,
            'model': self.model
        }