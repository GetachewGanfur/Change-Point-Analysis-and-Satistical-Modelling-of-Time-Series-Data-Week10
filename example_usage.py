"""
Example Usage of Brent Oil Price Change Point Analysis

This script demonstrates how to use the analysis components individually
for more targeted analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import DataLoader
from event_manager import EventManager
from enhanced_change_point_model import EnhancedChangePointModel

def example_basic_analysis():
    """Example of basic change point analysis"""
    print("🔍 Basic Change Point Analysis Example")
    print("-" * 40)
    
    # Load data
    loader = DataLoader("data/BrentOilPrices.csv")
    df = loader.load_data()
    
    print(f"Loaded {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")
    
    # Prepare log returns (more stationary)
    log_returns = df['Log_Returns'].dropna()
    dates = df['Date'][1:]  # Skip first date due to diff
    
    # Fit change point model
    model = EnhancedChangePointModel(log_returns, dates)
    print("Fitting Bayesian change point model...")
    trace = model.fit(n_samples=1000, tune=500)  # Reduced for quick demo
    
    # Get results
    summary = model.get_change_point_summary()
    print(f"\nChange point detected at: {summary['mode_date']}")
    
    # Plot results
    model.plot_results()
    
    return model, summary

def example_event_analysis():
    """Example of event analysis"""
    print("\n🗓️  Event Analysis Example")
    print("-" * 40)
    
    # Initialize event manager
    event_manager = EventManager()
    
    # Get all events
    all_events = event_manager.get_all_events()
    print(f"Total events in database: {len(all_events)}")
    
    # Get high impact events
    high_impact = event_manager.get_events_by_impact('High')
    print(f"High impact events: {len(high_impact)}")
    
    # Show recent events
    recent_events = all_events[all_events['date'] >= '2020-01-01']
    print(f"\nRecent events (2020+):")
    for _, event in recent_events.iterrows():
        print(f"- {event['date'].strftime('%Y-%m-%d')}: {event['event']}")
    
    return event_manager

def example_targeted_analysis():
    """Example of analyzing a specific time period"""
    print("\n🎯 Targeted Analysis Example (2008 Financial Crisis)")
    print("-" * 50)
    
    # Load data
    loader = DataLoader("data/BrentOilPrices.csv")
    df = loader.load_data()
    
    # Focus on 2007-2009 period (Financial Crisis)
    start_date = pd.to_datetime('2007-01-01')
    end_date = pd.to_datetime('2009-12-31')
    
    crisis_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    print(f"Crisis period data: {len(crisis_data)} records")
    
    # Analyze this specific period
    log_returns = crisis_data['Log_Returns'].dropna()
    dates = crisis_data['Date'][1:]
    
    model = EnhancedChangePointModel(log_returns, dates)
    print("Analyzing financial crisis period...")
    trace = model.fit(n_samples=800, tune=400)
    
    summary = model.get_change_point_summary()
    print(f"Change point during crisis: {summary['mode_date']}")
    
    # Check for nearby events
    event_manager = EventManager()
    nearby_events = event_manager.find_nearest_event(summary['mode_date'], 90)
    
    if len(nearby_events) > 0:
        print(f"Nearby events:")
        for _, event in nearby_events.head(3).iterrows():
            print(f"- {event['date']}: {event['event']} ({event['days_diff']} days)")
    
    return model, summary

def example_multiple_series_comparison():
    """Example comparing different transformations"""
    print("\n📊 Multiple Series Comparison")
    print("-" * 40)
    
    # Load data
    loader = DataLoader("data/BrentOilPrices.csv")
    df = loader.load_data()
    
    # Take a subset for faster computation
    recent_data = df[df['Date'] >= '2010-01-01'].copy()
    
    print(f"Analyzing recent data: {len(recent_data)} records")
    
    # Compare different series
    series_types = {
        'Price': recent_data['Price'],
        'Log Price': recent_data['Log_Price'],
        'Returns': recent_data['Returns'].dropna(),
        'Log Returns': recent_data['Log_Returns'].dropna()
    }
    
    results = {}
    
    for name, series in series_types.items():
        if len(series) < 100:  # Skip if too few observations
            continue
            
        print(f"\nAnalyzing {name}...")
        model = EnhancedChangePointModel(series)
        
        try:
            trace = model.fit(n_samples=500, tune=250)  # Quick fit
            summary = model.get_change_point_summary()
            results[name] = summary['mode_tau']
            print(f"Change point at index: {summary['mode_tau']}")
        except Exception as e:
            print(f"Failed to analyze {name}: {e}")
    
    print(f"\nComparison of change points:")
    for name, cp in results.items():
        print(f"- {name}: Index {cp}")
    
    return results

def main():
    """Run all examples"""
    print("🛢️  BRENT OIL ANALYSIS EXAMPLES")
    print("=" * 50)
    
    try:
        # Example 1: Basic analysis
        model1, summary1 = example_basic_analysis()
        
        # Example 2: Event analysis
        event_manager = example_event_analysis()
        
        # Example 3: Targeted analysis
        model2, summary2 = example_targeted_analysis()
        
        # Example 4: Multiple series comparison
        comparison_results = example_multiple_series_comparison()
        
        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Run main_analysis.py for comprehensive analysis")
        print("2. Modify parameters for different time periods")
        print("3. Add your own events to the event database")
        print("4. Experiment with different priors in the model")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        print("Make sure all dependencies are installed and data file exists")

if __name__ == "__main__":
    main()