"""
Data loading utilities for Brent oil price analysis.

This module provides functions to load, clean, and prepare oil price data
and geopolitical events data for change point analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_brent_data(file_path: str = '../data/raw/brent_oil_prices.csv') -> pd.DataFrame:
    """
    Load Brent oil price data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing Brent oil price data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date and Price columns
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        
        # Convert date column - handle multiple date formats
        if 'Date' in df.columns:
            try:
                # Try day-month-year format first (e.g., 20-May-87)
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
            except:
                try:
                    # Try standard date formats
                    df['Date'] = pd.to_datetime(df['Date'])
                except:
                    logger.error("Could not parse date column")
                    raise ValueError("Unable to parse date column")
        
        # Ensure Price column exists and is numeric
        if 'Price' not in df.columns:
            raise ValueError("Price column not found in data")
        
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Sort by date and reset index
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Remove any rows with missing data
        df = df.dropna()
        
        logger.info(f"Data processed: {len(df)} observations from {df['Date'].min()} to {df['Date'].max()}")
        
        return df
        
    except FileNotFoundError:
        logger.warning(f"Data file not found at {file_path}. Creating sample data.")
        return create_sample_brent_data()
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_sample_brent_data() -> pd.DataFrame:
    """
    Create realistic sample Brent oil price data for demonstration.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with simulated oil price data
    """
    logger.info("Creating sample Brent oil price data...")
    
    # Create date range from May 1987 to September 2022
    dates = pd.date_range(start='1987-05-20', end='2022-09-30', freq='D')
    n = len(dates)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate realistic oil price series with trends and volatility
    # Base price trend with multiple regimes
    t = np.linspace(0, 1, n)
    
    # Long-term trend with cycles
    base_trend = 40 + 30 * np.sin(2 * np.pi * t * 3) + 20 * t
    
    # Add volatility clustering (GARCH-like behavior)
    volatility = 0.015 + 0.01 * np.sin(20 * np.pi * t)**2
    
    # Generate price series with AR(1) component
    prices = np.zeros(n)
    prices[0] = base_trend[0]
    
    for i in range(1, n):
        # AR(1) component with time-varying volatility
        shock = np.random.normal(0, volatility[i])
        prices[i] = 0.999 * prices[i-1] + 0.001 * base_trend[i] + shock * prices[i-1]
    
    # Add major price shocks at specific points (simulating crises)
    shock_points = [
        (int(n * 0.05), -0.2),   # 1990 Gulf War
        (int(n * 0.3), -0.4),    # 1998 Asian Crisis
        (int(n * 0.6), -0.6),    # 2008 Financial Crisis
        (int(n * 0.75), -0.5),   # 2014 Oil Glut
        (int(n * 0.93), -0.7),   # 2020 COVID Crisis
    ]
    
    for point, magnitude in shock_points:
        # Create temporary shock
        for j in range(point, min(point + 60, n)):  # 60-day shock period
            shock_factor = np.exp(magnitude * np.exp(-(j - point) / 20))
            prices[j] *= shock_factor
    
    # Ensure prices don't go below realistic minimum
    prices = np.maximum(prices, 10)
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    logger.info(f"Created sample data with {len(df)} observations")
    return df


def load_events_data(file_path: str = '../data/events/major_events.csv') -> pd.DataFrame:
    """
    Load major geopolitical events data.
    
    Parameters:
    -----------
    file_path : str
        Path to the events CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with events data
    """
    try:
        events_df = pd.read_csv(file_path)
        events_df['Date'] = pd.to_datetime(events_df['Date'])
        logger.info(f"Loaded {len(events_df)} events from {file_path}")
        return events_df
        
    except FileNotFoundError:
        logger.warning(f"Events file not found at {file_path}. Creating default events.")
        return create_default_events()


def create_default_events() -> pd.DataFrame:
    """
    Create default major events dataset.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with major geopolitical events
    """
    major_events = [
        {'Date': '1990-08-02', 'Event': 'Iraq invades Kuwait', 'Category': 'Geopolitical Conflict', 'Expected_Impact': 'Price Increase'},
        {'Date': '1991-01-17', 'Event': 'Gulf War begins', 'Category': 'Geopolitical Conflict', 'Expected_Impact': 'Price Increase'},
        {'Date': '1997-07-02', 'Event': 'Asian Financial Crisis begins', 'Category': 'Economic Crisis', 'Expected_Impact': 'Price Decrease'},
        {'Date': '1998-12-28', 'Event': 'OPEC production cuts', 'Category': 'OPEC Decision', 'Expected_Impact': 'Price Increase'},
        {'Date': '2001-09-11', 'Event': '9/11 terrorist attacks', 'Category': 'Geopolitical Shock', 'Expected_Impact': 'Price Increase'},
        {'Date': '2003-03-20', 'Event': 'Iraq War begins', 'Category': 'Geopolitical Conflict', 'Expected_Impact': 'Price Increase'},
        {'Date': '2008-09-15', 'Event': 'Lehman Brothers collapse', 'Category': 'Economic Crisis', 'Expected_Impact': 'Price Decrease'},
        {'Date': '2010-12-17', 'Event': 'Arab Spring begins (Tunisia)', 'Category': 'Geopolitical Unrest', 'Expected_Impact': 'Price Increase'},
        {'Date': '2011-03-11', 'Event': 'Fukushima nuclear disaster', 'Category': 'Energy Crisis', 'Expected_Impact': 'Price Increase'},
        {'Date': '2014-11-27', 'Event': 'OPEC decides not to cut production', 'Category': 'OPEC Decision', 'Expected_Impact': 'Price Decrease'},
        {'Date': '2016-02-16', 'Event': 'Oil production freeze agreement', 'Category': 'OPEC Decision', 'Expected_Impact': 'Price Increase'},
        {'Date': '2018-05-08', 'Event': 'US withdraws from Iran nuclear deal', 'Category': 'Economic Sanctions', 'Expected_Impact': 'Price Increase'},
        {'Date': '2020-03-06', 'Event': 'OPEC+ agreement collapses', 'Category': 'OPEC Decision', 'Expected_Impact': 'Price Decrease'},
        {'Date': '2020-03-11', 'Event': 'WHO declares COVID-19 pandemic', 'Category': 'Global Health Crisis', 'Expected_Impact': 'Price Decrease'},
        {'Date': '2022-02-24', 'Event': 'Russia invades Ukraine', 'Category': 'Geopolitical Conflict', 'Expected_Impact': 'Price Increase'},
    ]
    
    events_df = pd.DataFrame(major_events)
    events_df['Date'] = pd.to_datetime(events_df['Date'])
    
    logger.info(f"Created default events dataset with {len(events_df)} events")
    return events_df


def prepare_data_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare oil price data for change point analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw oil price data
        
    Returns:
    --------
    pd.DataFrame
        Processed data ready for analysis
    """
    data = df.copy()
    
    # Calculate log prices and returns
    data['Log_Price'] = np.log(data['Price'])
    data['Log_Returns'] = data['Log_Price'].diff()
    
    # Calculate other features
    data['Price_Change'] = data['Price'].diff()
    data['Price_Pct_Change'] = data['Price'].pct_change()
    
    # Rolling volatility (30-day window)
    data['Volatility_30d'] = data['Log_Returns'].rolling(window=30).std()
    
    # Remove missing values
    data = data.dropna()
    
    # Create time index
    data['Time_Index'] = range(len(data))
    
    logger.info(f"Prepared data for analysis: {data.shape}")
    return data


def load_and_prepare_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare all data for the analysis.
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Prepared oil price data and events data
    """
    # Load oil price data
    oil_data = load_brent_data()
    
    # Load events data
    events_data = load_events_data()
    
    # Prepare oil data for analysis
    prepared_data = prepare_data_for_analysis(oil_data)
    
    logger.info("All data loaded and prepared successfully")
    
    return prepared_data, events_data