"""
Flask backend application for Brent Oil Price Change Point Analysis Dashboard.

This application provides REST APIs for serving analysis results, 
change point detection data, and event correlations to the React frontend.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.data_loader import load_and_prepare_all_data, load_events_data
from src.models.change_point_models import detect_change_points

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store data
oil_data = None
events_data = None
analysis_results = None


def initialize_data():
    """Initialize data and run basic analysis on startup."""
    global oil_data, events_data, analysis_results
    
    try:
        # Load data
        oil_data, events_data = load_and_prepare_all_data()
        logger.info("Data loaded successfully")
        
        # Run basic change point analysis (this might take a while)
        logger.info("Running change point analysis...")
        sample_data = oil_data.sample(n=min(1000, len(oil_data)))  # Use sample for faster demo
        
        analysis_results = detect_change_points(
            sample_data['Log_Returns'], 
            model_type='mean_change',
            draws=500,  # Reduced for faster startup
            tune=500
        )
        
        logger.info("Initial analysis completed")
        
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
        # Create mock data for development
        create_mock_data()


def create_mock_data():
    """Create mock data for development purposes."""
    global oil_data, events_data, analysis_results
    
    # Create mock oil data
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    prices = 50 + np.cumsum(np.random.normal(0, 2, len(dates)))
    prices = np.maximum(prices, 10)  # Ensure positive prices
    
    oil_data = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Log_Price': np.log(prices),
        'Log_Returns': np.diff(np.log(prices), prepend=np.log(prices[0])),
        'Time_Index': range(len(dates))
    })
    
    # Create mock events data
    events_data = pd.DataFrame({
        'Date': pd.to_datetime(['2020-03-11', '2020-12-01', '2021-06-15', '2022-02-24']),
        'Event': ['COVID-19 Pandemic', 'OPEC+ Meeting', 'Oil Recovery', 'Russia-Ukraine Conflict'],
        'Category': ['Health Crisis', 'OPEC Decision', 'Economic Recovery', 'Geopolitical Conflict'],
        'Expected_Impact': ['Price Decrease', 'Price Increase', 'Price Increase', 'Price Increase']
    })
    
    # Mock analysis results
    analysis_results = {
        'change_point_index': 300,
        'change_point_probability': 0.85,
        'parameter_estimates': {
            'mu_1': {'mean': -0.001, 'std': 0.002},
            'mu_2': {'mean': 0.003, 'std': 0.002}
        }
    }
    
    logger.info("Mock data created for development")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': oil_data is not None
    })


@app.route('/api/data/oil-prices', methods=['GET'])
def get_oil_prices():
    """Get oil price data with optional date filtering."""
    if oil_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Get query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = request.args.get('limit', type=int)
    
    # Filter data
    data = oil_data.copy()
    
    if start_date:
        data = data[data['Date'] >= start_date]
    if end_date:
        data = data[data['Date'] <= end_date]
    if limit:
        data = data.tail(limit)
    
    # Convert to JSON-serializable format
    result = {
        'data': data[['Date', 'Price', 'Log_Returns']].to_dict('records'),
        'total_records': len(data),
        'date_range': {
            'start': data['Date'].min().isoformat() if not data.empty else None,
            'end': data['Date'].max().isoformat() if not data.empty else None
        }
    }
    
    # Convert datetime objects to strings
    for record in result['data']:
        if 'Date' in record:
            record['Date'] = record['Date'].isoformat() if hasattr(record['Date'], 'isoformat') else str(record['Date'])
    
    return jsonify(result)


@app.route('/api/data/events', methods=['GET'])
def get_events():
    """Get geopolitical events data."""
    if events_data is None:
        return jsonify({'error': 'Events data not loaded'}), 500
    
    # Get query parameters
    category = request.args.get('category')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Filter events
    data = events_data.copy()
    
    if category:
        data = data[data['Category'] == category]
    if start_date:
        data = data[data['Date'] >= start_date]
    if end_date:
        data = data[data['Date'] <= end_date]
    
    # Convert to JSON-serializable format
    result = {
        'events': data.to_dict('records'),
        'categories': events_data['Category'].unique().tolist(),
        'total_events': len(data)
    }
    
    # Convert datetime objects to strings
    for event in result['events']:
        if 'Date' in event:
            event['Date'] = event['Date'].isoformat() if hasattr(event['Date'], 'isoformat') else str(event['Date'])
    
    return jsonify(result)


@app.route('/api/analysis/change-points', methods=['GET'])
def get_change_points():
    """Get change point analysis results."""
    if analysis_results is None:
        return jsonify({'error': 'Analysis not completed'}), 500
    
    # Get query parameters
    model_type = request.args.get('model_type', 'mean_change')
    confidence = request.args.get('confidence', 0.95, type=float)
    
    # Prepare results
    result = {
        'change_points': [],
        'model_type': model_type,
        'confidence_level': confidence,
        'summary': {}
    }
    
    if 'change_point_index' in analysis_results:
        cp_date = oil_data.iloc[analysis_results['change_point_index']]['Date']
        
        result['change_points'].append({
            'index': analysis_results['change_point_index'],
            'date': cp_date.isoformat() if hasattr(cp_date, 'isoformat') else str(cp_date),
            'probability': analysis_results['change_point_probability'],
            'type': model_type
        })
        
        result['summary'] = {
            'total_change_points': 1,
            'highest_probability': analysis_results['change_point_probability'],
            'parameter_estimates': analysis_results.get('parameter_estimates', {})
        }
    
    return jsonify(result)


@app.route('/api/analysis/correlations', methods=['GET'])
def get_event_correlations():
    """Get correlations between change points and events."""
    if analysis_results is None or events_data is None:
        return jsonify({'error': 'Analysis or events data not available'}), 500
    
    correlations = []
    
    if 'change_point_index' in analysis_results:
        cp_date = oil_data.iloc[analysis_results['change_point_index']]['Date']
        
        # Find events within +/- 30 days of change point
        time_window = timedelta(days=30)
        
        for _, event in events_data.iterrows():
            event_date = event['Date']
            time_diff = abs((cp_date - event_date).days)
            
            if time_diff <= 30:
                correlations.append({
                    'change_point_date': cp_date.isoformat() if hasattr(cp_date, 'isoformat') else str(cp_date),
                    'event_date': event_date.isoformat() if hasattr(event_date, 'isoformat') else str(event_date),
                    'event_name': event['Event'],
                    'event_category': event['Category'],
                    'expected_impact': event['Expected_Impact'],
                    'time_difference_days': time_diff,
                    'change_point_probability': analysis_results['change_point_probability']
                })
    
    result = {
        'correlations': correlations,
        'total_correlations': len(correlations),
        'analysis_summary': {
            'time_window_days': 30,
            'methodology': 'Simple time-based correlation within 30-day window'
        }
    }
    
    return jsonify(result)


@app.route('/api/analysis/run', methods=['POST'])
def run_analysis():
    """Run change point analysis with custom parameters."""
    global analysis_results
    
    if oil_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:
        # Get parameters from request
        params = request.get_json() or {}
        model_type = params.get('model_type', 'mean_change')
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        sample_size = params.get('sample_size', 1000)
        
        # Filter data if date range specified
        data = oil_data.copy()
        if start_date:
            data = data[data['Date'] >= start_date]
        if end_date:
            data = data[data['Date'] <= end_date]
        
        # Sample data for performance
        if len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
        
        logger.info(f"Running {model_type} analysis on {len(data)} observations")
        
        # Run analysis
        analysis_results = detect_change_points(
            data['Log_Returns'], 
            model_type=model_type,
            draws=params.get('draws', 500),
            tune=params.get('tune', 500)
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Analysis completed successfully',
            'parameters': params,
            'data_points_analyzed': len(data)
        })
        
    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    """Get summary statistics of the data."""
    if oil_data is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    summary = {
        'oil_prices': {
            'total_observations': len(oil_data),
            'date_range': {
                'start': oil_data['Date'].min().isoformat(),
                'end': oil_data['Date'].max().isoformat()
            },
            'price_statistics': {
                'mean': float(oil_data['Price'].mean()),
                'std': float(oil_data['Price'].std()),
                'min': float(oil_data['Price'].min()),
                'max': float(oil_data['Price'].max()),
                'median': float(oil_data['Price'].median())
            },
            'volatility': {
                'daily_volatility': float(oil_data['Log_Returns'].std()),
                'annualized_volatility': float(oil_data['Log_Returns'].std() * np.sqrt(252))
            }
        },
        'events': {
            'total_events': len(events_data),
            'categories': events_data['Category'].value_counts().to_dict(),
            'expected_impacts': events_data['Expected_Impact'].value_counts().to_dict()
        } if events_data is not None else None
    }
    
    return jsonify(summary)


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize data on startup
    initialize_data()
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )