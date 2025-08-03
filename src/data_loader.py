import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, price_path):
        if not isinstance(price_path, str) or not price_path:
            raise ValueError("price_path must be a non-empty string")
        self.price_path = price_path

    def load_data(self):
        """Load and preprocess Brent oil price data"""
        # Load the data with proper date parsing
        df = pd.read_csv(self.price_path)
        
        # Handle different date formats in the CSV
        def parse_date(date_str):
            try:
                # Try format like '20-May-87'
                return pd.to_datetime(date_str, format='%d-%b-%y')
            except:
                try:
                    # Try format like 'May 20, 2020'
                    return pd.to_datetime(date_str)
                except:
                    # Return NaT for unparseable dates
                    return pd.NaT
        
        df['Date'] = df['Date'].apply(parse_date)
        
        # Remove rows with unparseable dates
        df = df.dropna(subset=['Date'])
        
        # Sort by date and reset index
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Convert Price to numeric, handling any potential issues
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Remove any rows with missing values
        df = df.dropna()
        
        # Add additional useful columns
        df['Log_Price'] = np.log(df['Price'])
        df['Returns'] = df['Price'].pct_change()
        df['Log_Returns'] = df['Log_Price'].diff()
        
        return df
    
    def get_summary_stats(self, df):
        """Get summary statistics for the dataset"""
        return {
            'total_records': len(df),
            'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
            'price_stats': df['Price'].describe(),
            'return_stats': df['Returns'].dropna().describe()
        }