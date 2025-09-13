# Social Cohesion Monitoring System

A comprehensive Python-based application for identifying areas of low social cohesion and rising tensions, and prioritizing interventions for local stakeholders. The system provides early warning capabilities, sentiment mapping, intervention recommendations, and community engagement simulation tools.

## Quick Start

**Get up and running in 5 minutes:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the system
python example.py

# 3. Launch web dashboard
python main.py dashboard
```

**What you'll see:**
- Complete system demonstration with sample data
- Early warning analysis with risk scoring
- Interactive maps showing social trust levels
- Evidence-based intervention recommendations
- Community engagement impact simulation
- Alert system testing and configuration

**Ready to explore?** Jump to [Installation](#installation) or [Usage](#usage) sections below.

## Project Overview

**Goal**: Create a data-driven tool to identify areas of low social cohesion and rising tensions, and prioritize interventions for local stakeholders.

**Users**: Local authorities, community groups, government policymakers.

**Focus**: Geographic granularity (e.g., local authority/MSOA level).

## Key Features

### 1. Early Warning System for Social Tension
- **Automated risk scoring** using anomaly detection and clustering models
- **Real-time monitoring** of social indicators
- **SMS/Email alerts** for local authorities
- **Interpretable results** showing which factors drive risk in each area

### 2. Sentiment & Social Trust Mapping
- **Spatial mapping** of trust, deprivation, crime, and economic uncertainty
- **Correlation analysis** dashboard showing factor relationships
- **Interactive maps** with trust levels and community cohesion indicators
- **GIS integration** for geographical analysis

### 3. Intervention Effectiveness Tool
- **Case-based recommendations** based on similar historical cases
- **Evidence-based suggestions** with success metrics
- **Cost-effectiveness analysis** for different intervention types
- **Success factor identification** for optimal outcomes

### 4. Community Engagement Simulator
- **"What-if" analysis** for intervention impacts
- **Budget optimization** for maximum impact
- **Outcome prediction** using machine learning models
- **Scenario comparison** tools

### 5. Alert Management System
- **Automated notifications** for high-risk areas
- **Multi-channel alerts** (SMS, Email)
- **Configurable thresholds** and alert frequency
- **Alert history** and reporting

## Data Sources

### Current Implementation
The system currently uses **synthetic sample data** for demonstration purposes. This includes:
- **Sample MSOA data** with realistic characteristics
- **Generated social indicators** (trust, cohesion, sentiment)
- **Simulated intervention outcomes** based on research
- **Sample IMD data** when real data isn't available

### Real Data Integration
To integrate real data sources, replace the sample data generation functions with actual data connectors:

#### **Indices of Multiple Deprivation (IMD) 2019**
- **Source**: [GOV.UK](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)
- **Format**: Excel files with LSOA-level data
- **Integration**: Update `src/imd_connector.py` with real data URLs
- **Usage**: Automatically aggregates LSOA data to MSOA level

#### **ONS Census Data 2021**
- **Source**: [Office for National Statistics](https://www.ons.gov.uk/census)
- **Format**: CSV/Excel files with MSOA-level statistics
- **Integration**: Add new connector in `src/` directory
- **Usage**: Population, demographics, housing data

#### **Community Life Survey**
- **Source**: [GOV.UK](https://www.gov.uk/government/collections/community-life-survey)
- **Format**: SPSS/Excel files with survey responses
- **Integration**: Create `src/community_life_connector.py`
- **Usage**: Social trust, community cohesion, volunteering

#### **Crime Statistics**
- **Source**: [Police.uk](https://data.police.uk/) or [ONS Crime Survey](https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice)
- **Format**: CSV files with geographic crime data
- **Integration**: Add crime data connector
- **Usage**: Safety indicators, crime rates by area

#### **Economic Indicators**
- **Source**: [ONS Labour Market](https://www.ons.gov.uk/employmentandlabourmarket)
- **Format**: CSV files with employment/unemployment data
- **Integration**: Add economic data connector
- **Usage**: Employment rates, income data, economic uncertainty

### Data Integration Guide

1. **Download real data** from official sources
2. **Create data connector** in `src/` directory
3. **Update sample data functions** to use real data
4. **Test with small datasets** first
5. **Validate data quality** and geographic matching
6. **Update documentation** with data sources and update frequencies

## Installation

### Prerequisites
- **Python 3.7+** (tested with Python 3.8-3.11)
- **Internet connection** for data downloads
- **~500MB disk space** for data and models

### Quick Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd "Find locales"
   ```

2. **Create virtual environment (recommended)**
   ```bash
   # Using conda
   conda create -n social-cohesion python=3.9
   conda activate social-cohesion
   
   # Or using venv
   python -m venv social-cohesion
   source social-cohesion/bin/activate  # On Windows: social-cohesion\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python main.py status
   ```

### Optional: Data Source Configuration

The system can use either real data sources or dummy data for demonstration. Configure data sources in your `.env` file:

```bash
# Copy example file
cp env.example .env

# Configure data sources (set to 'true' for real data, 'false' for dummy data)
IMD_DATA_USE_REAL_DATA=false
ONS_CENSUS_USE_REAL_DATA=false
COMMUNITY_LIFE_SURVEY_USE_REAL_DATA=false
CRIME_DATA_USE_REAL_DATA=false
ECONOMIC_DATA_USE_REAL_DATA=false

# Alert System Configuration
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Twilio SMS (optional)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+447700900123
```

### Troubleshooting Installation

**Common Issues:**

1. **Python not found**: Ensure Python is in your PATH or use full path
2. **Package conflicts**: Use virtual environment
3. **Permission errors**: Run as administrator or use `--user` flag
4. **Missing dependencies**: Some packages may need system libraries

**Windows-specific:**
```bash
# If pip not found
python -m pip install -r requirements.txt

# If conda not found
# Install Miniconda/Anaconda first
```

**macOS/Linux-specific:**
```bash
# May need system dependencies
sudo apt-get install python3-dev  # Ubuntu/Debian
brew install python3              # macOS
```

## Usage

### Command Line Interface

#### System Status:
```bash
python main.py status
```

#### Early Warning System:
```bash
# Run full analysis
python main.py warning analyze --areas 100

# Get risk profile for specific MSOA
python main.py warning profile --msoa-code "E02000001"
```

#### Sentiment Mapping:
```bash
# Generate sentiment mapping
python main.py sentiment map --areas 50

# Get sentiment profile
python main.py sentiment profile --msoa-code "E02000001"
```

#### Intervention Recommendations:
```bash
# Get recommendations
python main.py intervention recommend --recommendations 5

# Analyze specific intervention
python main.py intervention analyze --intervention-type "Community Events Program"
```

#### Engagement Simulator:
```bash
# Optimize intervention mix
python main.py simulator optimize --budget 1000 --target overall
```

#### Alert Management:
```bash
# Test alert system
python main.py alerts test

# Get alert summary
python main.py alerts summary --hours 24
```

#### Data Source Configuration:
```bash
# Check data source status
python main.py data status

# Configure IMD data to use real data
python main.py data configure --source imd_data --use-real-data

# Configure Community Life Survey to use dummy data
python main.py data configure --source community_life_survey --use-dummy-data
```

#### Original MSOA Lookup:
```bash
# Look up by postcode
python main.py msoa lookup --postcode "SW1A 1AA"

# Look up by MSOA code
python main.py msoa lookup --msoa-code "E02000001"
```

### Streamlit Dashboard

Launch the interactive web dashboard:
```bash
python main.py dashboard
```

Or directly:
```bash
streamlit run streamlit_app.py
```

The dashboard provides:
- **System Overview** with key metrics and visualizations
- **Early Warning System** with risk analysis and alerts
- **Sentiment Mapping** with interactive maps and correlation analysis
- **Intervention Recommendations** with evidence-based suggestions
- **Engagement Simulator** with what-if analysis and optimization
- **Alert Management** with configuration and testing tools

### Python API

```python
from src.early_warning_system import EarlyWarningSystem
from src.sentiment_mapping import SentimentMapping
from src.intervention_tool import InterventionTool
from src.engagement_simulator import EngagementSimulator

# Early Warning System
ew_system = EarlyWarningSystem()
results = ew_system.run_full_analysis()

# Sentiment Mapping
sm_system = SentimentMapping()
trust_data = sm_system.run_full_analysis()

# Intervention Tool
int_tool = InterventionTool()
recommendations = int_tool.run_full_analysis()

# Engagement Simulator
simulator = EngagementSimulator()
optimization = simulator.optimize_intervention_mix(baseline_area, budget=1000)
```

## Project Structure

```
├── main.py                      # Enhanced CLI interface
├── streamlit_app.py             # Streamlit dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── example.py                   # Example usage script
├── .env                         # Environment variables (optional)
├── src/
│   ├── __init__.py
│   ├── data_aggregator.py       # Original MSOA data aggregation
│   ├── msoa_search.py           # MSOA search and lookup
│   ├── imd_connector.py         # IMD data connector
│   ├── early_warning_system.py  # Early warning and risk detection
│   ├── sentiment_mapping.py     # Sentiment and trust mapping
│   ├── intervention_tool.py     # Intervention recommendations
│   ├── engagement_simulator.py  # Community engagement simulation
│   └── alert_system.py          # Alert management system
└── data/                        # Local data storage (auto-created)
```

## Technical Stack

- **Python 3.7+** - Core programming language
- **scikit-learn** - Machine learning for anomaly detection and clustering
- **Streamlit** - Web dashboard interface
- **Plotly** - Interactive visualizations
- **Folium** - Interactive mapping
- **GeoPandas** - Geospatial data processing
- **Pandas/NumPy** - Data manipulation and analysis
- **Click** - Command-line interface
- **Twilio** - SMS notifications
- **SMTP** - Email notifications

## Key Metrics and Indicators

### Risk Scoring
- **Overall Risk Score** (0-1 scale)
- **Risk Levels**: Low, Medium, High, Critical
- **Anomaly Detection** using Isolation Forest
- **Clustering Analysis** for area grouping

### Social Cohesion Indicators
- **Social Trust Score** (1-10 scale)
- **Community Cohesion Score** (1-10 scale)
- **Sentiment Score** (1-10 scale)
- **Volunteer Participation Rate** (0-50%)

### Intervention Metrics
- **Success Score** based on historical outcomes
- **Cost Effectiveness** (improvement per £1000)
- **Expected Improvements** in trust, cohesion, sentiment
- **Evidence Base** from similar cases

## Alert System

The system can send automated alerts when:
- **Critical risk areas** are detected
- **Anomalous patterns** are identified
- **Threshold breaches** occur

Alert channels:
- **Email** for detailed reports
- **SMS** for urgent notifications
- **Configurable thresholds** and frequency limits

## Future Enhancements

- **Real-time data integration** from social media and surveys
- **Additional data sources** (census, health, education)
- **Advanced ML models** for prediction
- **Mobile application** for field workers
- **API endpoints** for third-party integration
- **Multi-nation support** (Wales, Scotland, Northern Ireland)

## Requirements

- **Python 3.7+**
- **Internet connection** for data downloads
- **~500MB disk space** for data and models
- **Optional**: Email/SMS credentials for alerts

## License

This project is open source. Please ensure you comply with the terms of use for the underlying data sources.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to:
- Add support for additional data sources
- Improve machine learning models
- Enhance visualization capabilities
- Add new intervention types
- Improve documentation

## Testing the System

### Quick Test
```bash
# Run the complete demo
python example.py

# Launch web dashboard
python main.py dashboard
```

### Individual Component Tests
```bash
# Test each component
python main.py warning analyze --areas 50
python main.py sentiment map --areas 30
python main.py intervention recommend --recommendations 3
python main.py simulator optimize --budget 500 --target trust
python main.py alerts test
```

### Expected Output
The system should show:
- All components initialized successfully
- Sample data generated and processed
- Risk analysis completed
- Intervention recommendations generated
- Impact simulation completed

## Troubleshooting

### Common Issues and Solutions

#### 1. **Maps Not Displaying/Flashing**
**Problem**: Maps appear briefly then disappear
**Solution**: 
```bash
# Install missing package
pip install streamlit-folium

# Or use alternative visualization
# The system automatically falls back to Plotly maps
```

#### 2. **IMD Data Download Errors**
**Problem**: "404 Client Error" when downloading IMD data
**Solution**: This is expected - the system uses sample data for demonstration
```bash
# For real IMD data, download manually from:
# https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
```

#### 3. **Python Environment Issues**
**Problem**: "Python was not found" or "pip not recognized"
**Solution**:
```bash
# Windows
python -m pip install -r requirements.txt
C:\Users\username\miniforge3\envs\envname\python.exe main.py status

# Activate conda environment
conda activate your-env-name
```

#### 4. **Streamlit Dashboard Issues**
**Problem**: Dashboard won't launch or shows errors
**Solution**:
```bash
# Update Streamlit
pip install streamlit --upgrade

# Clear cache
streamlit cache clear

# Use different port
streamlit run streamlit_app.py --server.port 8502
```

#### 5. **Alert System Not Working**
**Problem**: Email/SMS alerts not sending
**Solution**:
```bash
# Check configuration
python main.py alerts test

# Verify .env file exists and has correct credentials
# For Gmail: Use App Password, not regular password
# For Twilio: Verify account SID and auth token
```

#### 6. **Memory/Performance Issues**
**Problem**: System runs slowly or crashes
**Solution**:
```bash
# Reduce data size
python main.py warning analyze --areas 20
python main.py sentiment map --areas 20

# Check available memory
# Close other applications
```

### Getting Help

1. **Check System Status**
   ```bash
   python main.py status
   ```

2. **Run Diagnostics**
   ```bash
   python example.py  # Complete system test
   ```

3. **Check Logs**
   - Look for error messages in terminal output
   - Check `data/` folder for any error logs

4. **Common Error Messages**
   - `ModuleNotFoundError`: Missing package - run `pip install -r requirements.txt`
   - `PermissionError`: Run as administrator or use virtual environment
   - `ConnectionError`: Check internet connection for data downloads
   - `ValueError`: Invalid input parameters - check command syntax

## Support

For questions or support, please:
1. **Check this documentation** and troubleshooting section
2. **Run the example script** to verify system functionality
3. **Check system status** with `python main.py status`
4. **Submit an issue** with:
   - Operating system and Python version
   - Complete error message
   - Steps to reproduce the issue
   - Output from `python main.py status`

## Data Sources and Attribution

- **Indices of Multiple Deprivation 2019**: [GOV.UK](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)
- **ONS Postcode Directory**: [Office for National Statistics](https://www.ons.gov.uk/)
- **Community Life Survey**: [GOV.UK](https://www.gov.uk/government/collections/community-life-survey)
- **Postcode Data**: [postcodes.io](https://postcodes.io/)

---

**Built for local authorities, community groups, and policymakers to enhance social cohesion and community resilience.**
