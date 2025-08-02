# Brent Oil Price Change Point Analysis

## Project Overview

This project implements a comprehensive Bayesian change point analysis of Brent oil prices to identify structural breaks and their relationship with major geopolitical and economic events. The analysis helps investors, policymakers, and energy companies understand how significant events impact oil price dynamics.

## Business Objective

The main goal is to study how important events affect Brent oil prices, focusing on:
- Political decisions and conflicts in oil-producing regions
- Global economic sanctions
- Changes in OPEC policies
- Other major geopolitical events

The analysis provides clear insights to help stakeholders understand and react to price changes better.

## Key Features

### 1. Comprehensive Data Analysis
- **Data Loading**: Robust data preprocessing with proper date parsing
- **Exploratory Data Analysis**: Complete EDA with price trends, returns analysis, and volatility clustering
- **Time Series Analysis**: Stationarity tests, decomposition, and autocorrelation analysis

### 2. Bayesian Change Point Detection
- **Multiple Model Types**: Mean shift, volatility shift, and combined models
- **PyMC3 Implementation**: Robust Bayesian inference with MCMC sampling
- **Model Diagnostics**: Convergence checks, trace plots, and posterior analysis

### 3. Event Correlation Analysis
- **Historical Events Database**: Comprehensive database of major events affecting oil prices
- **Event Matching**: Automatic correlation of change points with nearby events
- **Impact Assessment**: Quantitative analysis of price changes and volatility shifts

### 4. Comprehensive Reporting
- **Executive Summary**: High-level insights for decision makers
- **Technical Details**: Statistical evidence and model diagnostics
- **Business Implications**: Strategic recommendations for different stakeholders
- **Visualizations**: Interactive plots and comprehensive charts

## Project Structure

```
├── data/
│   └── BrentOilPrices.csv          # Historical Brent oil price data
├── src/
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── event_manager.py            # Historical events management
│   ├── eda.py                      # Exploratory data analysis
│   ├── time_series_analysis.py     # Time series analysis
│   ├── change_point_model.py       # Bayesian change point models
│   ├── results_interpreter.py      # Results interpretation and reporting
│   └── main_analysis.py           # Main analysis orchestration
├── notebooks/
│   └── 01_analysis.ipynb          # Jupyter notebook for analysis
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone <https://github.com/GetachewGanfur/Change-Point-Analysis-and-Satistical-Modelling-of-Time-Series-Data-Week10.git>
cd Change-Point-Analysis-and-Statistical-Modelling-of-Time-Series-Data-Week10
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete analysis:

```python
from src.main_analysis import BrentOilAnalysis

# Create analysis instance
analysis = BrentOilAnalysis()

# Run complete analysis
results = analysis.run_complete_analysis(model_type='mean_shift')
```

### Step-by-Step Analysis

```python
from src.main_analysis import BrentOilAnalysis

# Initialize
analysis = BrentOilAnalysis()

# Step 1: Load data
analysis.load_and_prepare_data()

# Step 2: Initialize components
analysis.initialize_components()

# Step 3: Run exploratory analysis
analysis.run_exploratory_analysis()

# Step 4: Run time series analysis
ts_stats = analysis.run_time_series_analysis()

# Step 5: Run change point analysis
trace = analysis.run_change_point_analysis(model_type='mean_shift')

# Step 6: Generate report
analysis.generate_comprehensive_report()
```

### Multiple Model Comparison

```python
# Compare different model types
multi_results = analysis.run_multiple_models()
```

## Key Components

### 1. Data Loader (`data_loader.py`)
- Handles Brent oil price data loading
- Proper date parsing and preprocessing
- Calculates returns and log returns
- Provides summary statistics

### 2. Event Manager (`event_manager.py`)
- Comprehensive database of historical events
- Categories: Conflict, Economic, OPEC, Geopolitical, Natural Disaster
- Impact levels: High, Medium, Low
- Event matching and correlation analysis

### 3. EDA (`eda.py`)
- Price series visualization
- Returns analysis and distribution
- Volatility clustering analysis
- Stationarity tests

### 4. Time Series Analysis (`time_series_analysis.py`)
- Comprehensive time series analysis
- Decomposition (trend, seasonal, residual)
- Autocorrelation analysis
- Regime change detection

### 5. Change Point Model (`change_point_model.py`)
- Bayesian change point detection using PyMC3
- Multiple model types: mean shift, volatility shift, both
- Model diagnostics and convergence checks
- Posterior analysis and visualization

### 6. Results Interpreter (`results_interpreter.py`)
- Comprehensive results interpretation
- Event correlation analysis
- Business impact assessment
- Executive reporting

## Model Types

### 1. Mean Shift Model
- Detects changes in the mean price level
- Useful for identifying structural breaks in price levels
- Parameters: μ₁ (before), μ₂ (after), σ (volatility)

### 2. Volatility Shift Model
- Detects changes in price volatility
- Useful for identifying periods of increased uncertainty
- Parameters: μ (mean), σ₁ (before), σ₂ (after)

### 3. Combined Model
- Detects both mean and volatility changes
- Most comprehensive but computationally intensive
- Parameters: μ₁, μ₂, σ₁, σ₂

## Output and Reports

### 1. Executive Summary
- High-level findings and key insights
- Impact assessment and magnitude
- Strategic recommendations

### 2. Technical Details
- Statistical evidence and model diagnostics
- Change point confidence intervals
- Parameter comparisons and significance

### 3. Business Implications
- Market impact analysis
- Risk assessment
- Strategic considerations

### 4. Visualizations
- Time series with change points
- Parameter comparisons
- Event correlations
- Model diagnostics

## Key Insights

The analysis typically reveals:

1. **Structural Breaks**: Significant changes in oil price dynamics
2. **Event Correlations**: Links between geopolitical events and price changes
3. **Volatility Regimes**: Periods of high and low market uncertainty
4. **Policy Implications**: Impact of OPEC decisions and economic sanctions

## Assumptions and Limitations

### Statistical Assumptions
- Normal distribution for price changes
- Independence of observations (may not hold in practice)
- Linear relationships (may miss non-linear effects)

### Causal Inference Limitations
- Correlation does not imply causation
- Multiple events may occur simultaneously
- Market efficiency may incorporate events before official announcements

### Model Limitations
- Single change point assumption
- Fixed parameter structure
- May miss gradual changes

## Future Enhancements

1. **Multiple Change Points**: Extend to detect multiple structural breaks
2. **Advanced Models**: Implement Markov-switching and VAR models
3. **Real-time Analysis**: Develop streaming analysis capabilities
4. **Machine Learning**: Incorporate ML models for event prediction
5. **Interactive Dashboard**: Create web-based visualization platform

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or contributions, please contact the project maintainers.

---

**Note**: This analysis is for educational and research purposes. Investment decisions should not be based solely on this analysis. Always consult with financial professionals for investment advice.