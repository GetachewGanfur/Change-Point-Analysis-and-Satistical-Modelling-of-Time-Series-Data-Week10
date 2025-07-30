# Setup Guide - Brent Oil Price Change Point Analysis

This guide will help you set up and run the Brent Oil Price Change Point Analysis project.

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd brent-oil-analysis
```

### 2. Set Up Python Environment

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your configuration
# Most default values should work for local development
```

### 4. Prepare Data Directory

```bash
# Create necessary directories
mkdir -p data/raw data/processed data/events logs results

# If you have Brent oil price data, place it in data/raw/brent_oil_prices.csv
# Expected format: Date (day-month-year), Price (USD per barrel)
# Example: 20-May-87,18.63

# The application will create sample data if no real data is found
```

### 5. Run the Backend API

```bash
cd dashboard/backend
python app.py
```

The Flask backend will start on `http://localhost:5000`

### 6. Set Up and Run the Frontend

In a new terminal:

```bash
cd dashboard/frontend

# Install Node.js dependencies
npm install

# Start the React development server
npm start
```

The React frontend will start on `http://localhost:3000`

## Data Analysis Workflow

### 1. Jupyter Notebooks

The analysis can be run using Jupyter notebooks:

```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_change_point_analysis.ipynb
# 3. notebooks/03_event_correlation.ipynb
```

### 2. Python Scripts

You can also run analysis using Python scripts:

```bash
# Run data loading and preparation
python -c "
from src.utils.data_loader import load_and_prepare_all_data
data, events = load_and_prepare_all_data()
print(f'Loaded {len(data)} price observations and {len(events)} events')
"

# Run change point analysis
python -c "
from src.utils.data_loader import load_and_prepare_all_data
from src.models.change_point_models import detect_change_points
data, events = load_and_prepare_all_data()
results = detect_change_points(data['Log_Returns'].head(1000))
print('Analysis completed')
"
```

## API Endpoints

Once the backend is running, you can access these endpoints:

- `GET /api/health` - Health check
- `GET /api/data/oil-prices` - Get oil price data
- `GET /api/data/events` - Get geopolitical events
- `GET /api/data/summary` - Get data summary statistics
- `GET /api/analysis/change-points` - Get change point analysis results
- `GET /api/analysis/correlations` - Get event-change point correlations
- `POST /api/analysis/run` - Run new analysis with custom parameters

## Dashboard Features

The React dashboard provides:

1. **Interactive Price Charts**: Visualize oil price trends with events overlay
2. **Change Point Analysis**: View detected structural breaks with probabilities
3. **Event Correlation**: Explore relationships between events and price changes
4. **Custom Analysis**: Run analysis with different parameters and date ranges
5. **Data Export**: Download analysis results and visualizations

## Configuration

### Analysis Parameters

Edit `config/config.yaml` to customize:

- MCMC sampling parameters (draws, tune, chains)
- Time windows for event correlation
- Chart display settings
- Data file paths

### Environment Variables

Key environment variables in `.env`:

- `REACT_APP_API_URL`: Backend API URL for frontend
- `DEMO_MODE`: Enable faster settings for development
- `LOG_LEVEL`: Logging verbosity
- `MCMC_DRAWS`: Number of MCMC samples (increase for better accuracy)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies installed
2. **Port already in use**: Change ports in configuration files
3. **CORS errors**: Ensure Flask-CORS is installed and configured
4. **Slow analysis**: Reduce MCMC parameters or enable demo mode
5. **Data not loading**: Check file paths and format in configuration

### Performance Optimization

For large datasets:

1. Set `DEMO_MODE=true` for faster startup
2. Reduce `MCMC_DRAWS` and `MCMC_TUNE` in configuration
3. Use data sampling for initial exploration
4. Consider running analysis on server with more resources

### Debugging

Enable detailed logging:

```bash
# Set in .env file
LOG_LEVEL=DEBUG

# Or set environment variable
export LOG_LEVEL=DEBUG
```

Check log files in `logs/` directory for detailed error information.

## Production Deployment

For production deployment:

1. Set `FLASK_ENV=production` in environment
2. Use production-grade WSGI server (e.g., Gunicorn)
3. Build React app for production: `npm run build`
4. Set up reverse proxy (e.g., Nginx)
5. Configure proper logging and monitoring
6. Set up SSL certificates for HTTPS

Example production commands:

```bash
# Backend with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 dashboard.backend.app:app

# Frontend build
cd dashboard/frontend
npm run build
```

## Additional Resources

- [PyMC3 Documentation](https://docs.pymc.io/en/v3/)
- [React Documentation](https://reactjs.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Material-UI Documentation](https://mui.com/)

## Getting Help

If you encounter issues:

1. Check this setup guide
2. Review log files for error messages
3. Ensure all dependencies are correctly installed
4. Verify data file formats and paths
5. Check that all required ports are available