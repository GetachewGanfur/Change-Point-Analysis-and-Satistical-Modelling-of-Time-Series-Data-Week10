# Task 1: Brent Oil Price Change Point Analysis - Interim Report

## Introduction

This interim report outlines the foundation for analyzing structural breaks in Brent oil prices using Bayesian change point detection. The analysis aims to identify statistically significant shifts in oil price behavior and correlate them with major geopolitical and economic events. This report covers the data analysis workflow, event compilation, time series properties analysis, and methodological framework.

## Methodology

### 1. Data Analysis Workflow

The analysis follows a structured 6-phase workflow designed to ensure robust statistical inference and meaningful business insights:

#### Phase 1: Data Preparation and Exploration
- **Data Loading**: Import historical Brent oil price data (1987-2022, 9,011 observations)
- **Data Validation**: Check for missing values, outliers, and data quality issues
- **Initial Exploration**: Generate summary statistics and basic visualizations
- **Data Preprocessing**: Handle date formatting, create time series objects

#### Phase 2: Event Research and Compilation
- **Event Identification**: Research major geopolitical, economic, and OPEC events
- **Event Categorization**: Classify events by type (Geopolitical, Economic, OPEC Decision)
- **Impact Assessment**: Rate events by expected market impact (High/Medium/Low)
- **Dataset Creation**: Compile structured CSV with dates, descriptions, and metadata

#### Phase 3: Time Series Analysis
- **Trend Analysis**: Examine long-term price trends and seasonal patterns
- **Stationarity Testing**: Apply statistical tests (ADF, KPSS) to assess stationarity
- **Volatility Analysis**: Analyze price volatility clustering and regime changes
- **Modeling Implications**: Determine appropriate model specifications based on data properties

#### Phase 4: Change Point Detection
- **Model Selection**: Implement Bayesian change point models using PyMC3
- **Parameter Estimation**: Fit models with appropriate priors and likelihood functions
- **Change Point Identification**: Detect statistically significant structural breaks
- **Uncertainty Quantification**: Generate confidence intervals for change point locations

#### Phase 5: Event Correlation and Impact Assessment
- **Temporal Matching**: Compare change point dates with event dates
- **Impact Quantification**: Measure magnitude of price changes associated with events
- **Statistical Validation**: Test significance of correlations between events and changes
- **Causal Inference**: Distinguish between correlation and potential causation

#### Phase 6: Reporting and Communication
- **Executive Summary**: Create high-level insights for business stakeholders
- **Technical Documentation**: Provide detailed methodology and statistical results
- **Interactive Dashboard**: Develop visualization tools for exploratory analysis
- **Stakeholder Presentations**: Prepare materials for different audience types

### 2. Event Dataset Quality

A comprehensive dataset of 15 major events has been compiled, covering the period 1990-2023:

#### Event Categories and Distribution:
- **Geopolitical Events (7)**: Major conflicts, political crises, and international tensions
- **OPEC Decisions (5)**: Production cuts, output increases, and policy changes
- **Economic Events (3)**: Financial crises, market shocks, and economic policy changes

#### Event Impact Assessment:
- **High Impact (9 events)**: Expected to cause significant price movements
- **Medium Impact (6 events)**: Moderate influence on price behavior

#### Key Events Included:
1. **1990-08-02**: Iraq invasion of Kuwait (Geopolitical, High)
2. **2001-09-11**: September 11 attacks (Geopolitical, High)
3. **2003-03-20**: Iraq War begins (Geopolitical, High)
4. **2008-09-15**: Lehman Brothers collapse (Economic, High)
5. **2014-11-27**: OPEC maintains production (OPEC Decision, High)
6. **2016-11-30**: OPEC production cut agreement (OPEC Decision, High)
7. **2020-03-06**: OPEC+ deal collapse (OPEC Decision, High)
8. **2020-03-11**: WHO declares COVID-19 pandemic (Economic, High)
9. **2020-04-20**: WTI oil futures turn negative (Economic, High)
10. **2022-02-24**: Russia invades Ukraine (Geopolitical, High)

### 3. Time Series Properties Analysis

#### Data Characteristics:
- **Time Period**: 1987-2022 (35+ years of daily data)
- **Observations**: 9,011 daily price points
- **Price Range**: $9.05 to $147.02 per barrel
- **Data Quality**: Complete dataset with no missing values

#### Key Properties Identified:

**Trend Analysis:**
- Strong upward trend with slope of 0.009 (p < 0.001)
- R-squared of 0.52 indicates significant trend component
- Multiple regime changes evident in trend behavior

**Stationarity Testing:**
- Non-stationary series confirmed by ADF test (p > 0.05)
- First differences show improved stationarity
- Modeling implications: Require differencing or trend modeling

**Volatility Analysis:**
- Mean volatility of 2.53% with significant clustering
- Maximum drawdown of -93.7% during extreme events
- Volatility regimes identified with distinct characteristics

**Modeling Implications:**
- Non-stationary data requires appropriate model specifications
- Volatility clustering suggests GARCH-type models may be beneficial
- Multiple regime changes indicate change point analysis is appropriate
- Long time series provides sufficient data for robust statistical inference

### 4. Assumptions and Limitations

#### Key Assumptions:
1. **Data Quality**: Oil price data is accurate and complete
2. **Structural Breaks**: Change points represent genuine structural breaks in market behavior
3. **Event Impact**: Geopolitical events have measurable impact on oil prices
4. **Model Validity**: Bayesian change point models are appropriate for this data
5. **Temporal Relationship**: Events precede or coincide with detected change points

#### Critical Limitations:

**Correlation vs. Causation:**
- **Statistical Correlation**: Change point detection identifies temporal associations between events and price changes
- **Causal Inference**: Correlation does not establish causation; external factors may influence both events and prices
- **Confounding Variables**: Unmodeled factors (weather, technology, demand changes) may affect prices
- **Temporal Ambiguity**: Event timing may not precisely match market reaction timing

**Model Limitations:**
- **False Positives**: Change point detection may identify spurious breaks in noisy data
- **Parameter Sensitivity**: Results depend on penalty parameter selection and model specifications
- **Minimum Segment Size**: Cannot detect changes in very short time segments
- **Assumption Violations**: Models assume independence and normality of residuals
- **Computational Complexity**: Bayesian inference may be slow for large datasets

**External Factors:**
- **Unmodeled Events**: Not all relevant events are included in the dataset
- **Market Microstructure**: High-frequency trading and market dynamics not captured
- **Policy Changes**: Regulatory and policy changes may affect price behavior
- **Technological Advances**: Fracking, renewable energy adoption not explicitly modeled

### 5. Purpose of Change Point Models

#### Context and Utility:
Change point models serve as powerful tools for identifying structural breaks in Brent oil price time series, enabling:

**Mean Shift Detection:**
- Identify periods when average oil prices shift significantly
- Quantify magnitude of price level changes between regimes
- Distinguish between temporary fluctuations and permanent shifts

**Volatility Regime Changes:**
- Detect periods of increased or decreased price volatility
- Identify market stress periods and calm periods
- Measure stability of different market regimes

**Combined Analysis:**
- Simultaneous detection of mean and volatility changes
- Comprehensive characterization of market regime shifts
- Multi-dimensional assessment of structural breaks

#### Expected Outputs:
1. **Change Point Dates**: Specific dates when structural breaks occur
2. **Regime Parameters**: Statistical parameters for each identified regime
3. **Confidence Intervals**: Uncertainty bounds for change point locations
4. **Model Diagnostics**: Goodness-of-fit and model selection metrics
5. **Impact Quantification**: Magnitude of changes between regimes

#### Business Value:
- **Risk Management**: Identify periods of increased market volatility
- **Trading Strategy**: Optimize entry/exit timing based on regime changes
- **Policy Analysis**: Assess effectiveness of OPEC decisions and policy interventions
- **Scenario Planning**: Use historical regime changes for future planning

## Challenges and Solutions

### Technical Challenges:

**Challenge 1: Model Selection and Parameter Tuning**
- **Solution**: Implement multiple change point detection methods (PELT, Binary Segmentation, Bayesian) and compare results
- **Validation**: Use cross-validation and model diagnostics to select optimal parameters

**Challenge 2: Computational Efficiency**
- **Solution**: Implement efficient algorithms and parallel processing for large datasets
- **Optimization**: Use approximate inference methods when exact computation is prohibitive

**Challenge 3: Statistical Significance Testing**
- **Solution**: Implement proper hypothesis testing and confidence interval estimation
- **Robustness**: Use multiple statistical tests to validate change point significance

### Methodological Challenges:

**Challenge 1: Distinguishing Signal from Noise**
- **Solution**: Apply appropriate penalty parameters and model selection criteria
- **Validation**: Compare detected change points with known historical events

**Challenge 2: Causal Inference Limitations**
- **Solution**: Clearly communicate correlation vs. causation distinction
- **Transparency**: Document all assumptions and limitations in final report

**Challenge 3: External Factor Control**
- **Solution**: Include comprehensive event dataset and acknowledge unmodeled factors
- **Robustness**: Test sensitivity of results to different model specifications

## Conclusion and Future Plan

### Current Status:
- **Foundation Complete**: Data analysis workflow, event compilation, and methodological framework established
- **Quality Assurance**: Comprehensive dataset with 15 major events compiled and validated
- **Technical Readiness**: Time series properties analyzed and modeling approach defined
- **Transparency**: Clear documentation of assumptions, limitations, and methodology

### Next Steps (Task 2):

**Phase 1: Bayesian Model Implementation**
- Implement PyMC3-based change point detection models
- Fit models with appropriate priors and likelihood functions
- Validate model convergence and diagnostics

**Phase 2: Change Point Detection**
- Identify statistically significant structural breaks
- Generate confidence intervals for change point locations
- Compare results across multiple detection methods

**Phase 3: Event Correlation Analysis**
- Match detected change points with historical events
- Quantify impact magnitude for each major change point
- Test statistical significance of event-change point correlations

**Phase 4: Insight Generation**
- Create executive summary with key findings
- Develop interactive dashboard for stakeholder exploration
- Prepare technical report with detailed methodology

**Phase 5: Communication and Delivery**
- Present results to diverse stakeholder groups
- Provide actionable insights for business decision-making
- Document lessons learned and future research directions

### Success Metrics:
- **Statistical Validity**: Robust model diagnostics and convergence
- **Event Correlation**: Significant temporal matching between change points and events
- **Impact Quantification**: Clear measurement of price changes associated with events
- **Communication Effectiveness**: Accessible insights for diverse stakeholders

This foundation provides the solid groundwork needed for successful implementation of Task 2's Bayesian change point analysis and insight generation.

---

**GitHub Repository**: https://github.com/GetachewGanfur/Change-Point-Analysis-and-Statistical-Modelling-of-Time-Series-Data-Week10

**Data Files**: 
- `data/BrentOilPrices.csv` - Historical oil price data (1987-2022)
- `data/processed/events.csv` - 15 major events dataset
- `notebooks/01_data_analysis_workflow.ipynb` - Interactive analysis notebook 