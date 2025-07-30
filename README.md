# Brent Oil Price Change Point Analysis

## Business Objective

This project analyzes how important events affect Brent oil prices through change point analysis and statistical modeling. We focus on detecting changes in oil prices and associating them with major events like political decisions, conflicts in oil-producing regions, global economic sanctions, and OPEC policy changes.

## Project Overview

As data scientists at Birhan Energies, we provide data-driven insights to help stakeholders in the energy sector navigate market complexities. This analysis delivers actionable intelligence for investors, policymakers, and energy companies.

### Key Goals
- Identify key events that significantly impacted Brent oil prices over the past decade
- Measure the quantitative impact of these events on price changes
- Provide clear, data-driven insights for investment strategies, policy development, and operational planning

## Project Structure

```
├── data/
│   ├── raw/                    # Raw Brent oil price data
│   ├── processed/              # Cleaned and preprocessed data
│   └── events/                 # Major geopolitical events dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_change_point_analysis.ipynb
│   └── 03_event_correlation.ipynb
├── src/
│   ├── analysis/               # Data analysis modules
│   ├── models/                 # Change point models
│   └── utils/                  # Utility functions
├── dashboard/
│   ├── backend/                # Flask API
│   └── frontend/               # React dashboard
├── docs/                       # Documentation
└── config/                     # Configuration files
```

## Dataset

The dataset contains historical Brent oil prices with daily prices from May 20, 1987, to September 30, 2022.

### Data Fields
- **Date**: Date of recorded price (day-month-year format)
- **Price**: Brent oil price in USD per barrel

## Methodology

### 1. Bayesian Change Point Analysis
- Use PyMC3 for Bayesian modeling
- Detect statistically significant structural breaks
- Identify change points in mean price and volatility

### 2. Event Correlation
- Research and compile major geopolitical events
- Associate detected change points with specific events
- Quantify impact of events on price behavior

### 3. Interactive Dashboard
- Flask backend for API services
- React frontend for data visualization
- Real-time exploration of event impacts

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd brent-oil-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Analysis
1. Start with data exploration: `notebooks/01_data_exploration.ipynb`
2. Run change point analysis: `notebooks/02_change_point_analysis.ipynb`
3. Analyze event correlations: `notebooks/03_event_correlation.ipynb`

### Dashboard
1. Start the Flask backend:
```bash
cd dashboard/backend
python app.py
```

2. Start the React frontend:
```bash
cd dashboard/frontend
npm install
npm start
```

## Key Features

- **Change Point Detection**: Bayesian models to identify structural breaks
- **Event Analysis**: Correlation of price changes with geopolitical events
- **Interactive Visualization**: User-friendly dashboard for data exploration
- **Statistical Validation**: Rigorous statistical methods for insight generation

## Learning Outcomes

### Skills
- Change Point Analysis & Interpretation
- Statistical Reasoning
- Bayesian Modeling with PyMC3
- Analytical Storytelling with Data

### Knowledge
- Probability distributions and model selection
- Bayesian inference and MCMC
- Model comparison techniques
- Policy analysis and interpretation

## Contributing

Please read the contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.