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
 OR

 streamlit run streamlit_app.py
```

**What you'll see:**
- Complete system demonstration with sample data
- Early warning analysis with risk scoring
- Interactive maps showing social trust levels
- Evidence-based intervention recommendations
- Community engagement impact simulation
- Real social trust data from Good Neighbours survey

##  Key Features

### 1. Early Warning System
- **Automated risk scoring** using anomaly detection and clustering models
- **Real-time monitoring** of social indicators
- **SMS/Email alerts** for local authorities
- **Interpretable results** showing which factors drive risk in each area

### 2. Sentiment & Social Trust Mapping
- **Spatial mapping** of trust, deprivation, crime, and economic uncertainty
- **Correlation analysis** dashboard showing factor relationships
- **Interactive maps** with trust levels and community cohesion indicators
- **GIS integration** for geographical analysis

### 3. Good Neighbours Trust Data
- **Real social trust data** from Good Neighbours survey
- **MSOA-level analysis** of net trust scores
- **Trust vs caution** relationship analysis
- **Top/lowest trust areas** identification
- **Interactive visualizations** and distribution analysis

### 4. Intervention Effectiveness Tool
- **Case-based recommendations** based on similar historical cases
- **Evidence-based suggestions** with success metrics
- **Cost-effectiveness analysis** for different intervention types
- **Success factor identification** for optimal outcomes

### 5. Community Engagement Simulator
- **"What-if" analysis** for intervention impacts
- **Budget optimization** for maximum impact
- **Outcome prediction** using machine learning models
- **Scenario comparison** tools

### 6. Alert Management System
- **Automated notifications** for high-risk areas
- **Multi-channel alerts** (SMS, Email)
- **Configurable thresholds** and alert frequency
- **Alert history** and reporting

## 📁 Data Sources

### Real Data Integration
The system integrates real data sources alongside synthetic sample data:

#### **Indices of Multiple Deprivation (IMD) 2019**
- **Source**: [GOV.UK](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)
- **Format**: Excel files with LSOA-level data
- **Integration**: `src/imd_connector.py` - simplified connector for real data
- **Usage**: Automatically aggregates LSOA data to MSOA level

#### **Good Neighbours Social Trust Data**
- **Source**: Local survey data
- **Format**: Excel file with MSOA-level social trust scores
- **Integration**: `src/good_neighbours_connector.py` - dedicated connector
- **Usage**: Net trust analysis, trust vs caution relationships
- **Columns**: MSOA_code, MSOA_name, always_trust OR usually_trust, usually_careful OR almost_always_careful, Net_trust

#### **Other Data Sources** (Sample data available)
- **ONS Census Data** - Population, demographics, housing data
- **Community Life Survey** - Social trust, community cohesion, volunteering
- **Crime Statistics** - Safety indicators, crime rates by area
- **Economic Indicators** - Employment rates, income data, economic uncertainty

## 🛠️ Installation

### Prerequisites
- **Python 3.7+** (tested with Python 3.8-3.11)
- **Internet connection** for data downloads
- **~500MB disk space** for data and models

### Quick Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd "Social Cohesion Tool"
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

### Data Configuration

Configure data sources in your `.env` file:

```bash
# Copy example file
cp env.example .env

# Configure data sources (set to 'true' for real data, 'false' for dummy data)
IMD_DATA_USE_REAL_DATA=true
GOOD_NEIGHBOURS_USE_REAL_DATA=true
ONS_CENSUS_USE_REAL_DATA=false
COMMUNITY_LIFE_SURVEY_USE_REAL_DATA=false
CRIME_DATA_USE_REAL_DATA=false
ECONOMIC_DATA_USE_REAL_DATA=false

# Alert System Configuration (optional)
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Twilio SMS (optional)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+447700900123
```

## 💻 Usage

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

#### Good Neighbours Trust Data:
```bash
# Get trust data summary
python main.py trust summary

# Look up specific MSOA
python main.py trust lookup --msoa-code "E02000001"

# Show top trust areas
python main.py trust top --top 10

# Show lowest trust areas
python main.py trust lowest --bottom 10
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

# Configure Good Neighbours data
python main.py data configure --source good_neighbours --use-real-data
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
- **Good Neighbours Trust Data** with real social trust analysis
- **Intervention Recommendations** with evidence-based suggestions
- **Engagement Simulator** with what-if analysis and optimization
- **Alert Management** with configuration and testing tools

### Python API

```python
from src.early_warning_system import EarlyWarningSystem
from src.sentiment_mapping import SentimentMapping
from src.good_neighbours_connector import GoodNeighboursConnector
from src.intervention_tool import InterventionTool
from src.engagement_simulator import EngagementSimulator

# Early Warning System
ew_system = EarlyWarningSystem()
results = ew_system.run_full_analysis()

# Sentiment Mapping
sm_system = SentimentMapping()
trust_data = sm_system.run_full_analysis()

# Good Neighbours Social Trust Data
gn_connector = GoodNeighboursConnector()
summary = gn_connector.get_social_trust_summary()
msoa_data = gn_connector.get_social_trust_for_msoa("E02000001")

# Intervention Tool
int_tool = InterventionTool()
recommendations = int_tool.run_full_analysis()

# Engagement Simulator
simulator = EngagementSimulator()
optimization = simulator.optimize_intervention_mix(baseline_area, budget=1000)
```

## 📂 Project Structure

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
│   ├── imd_connector.py         # IMD data connector (simplified)
│   ├── good_neighbours_connector.py # Good Neighbours trust data connector
│   ├── early_warning_system.py  # Early warning and risk detection
│   ├── sentiment_mapping.py     # Sentiment and trust mapping
│   ├── intervention_tool.py     # Intervention recommendations
│   ├── engagement_simulator.py  # Community engagement simulation
│   ├── alert_system.py          # Alert management system
│   └── data_config.py           # Data source configuration
└── data/                        # Local data storage (auto-created)
    ├── IMD2019_Index_of_Multiple_Deprivation.xlsx
    └── good_neighbours_full_data_by_msoa.xlsx
```

## 🔧 Technical Stack

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

## 📈 Key Metrics and Indicators

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

### Good Neighbours Trust Metrics
- **Net Trust Score** (trust minus caution)
- **Always/Usually Trust** percentage
- **Usually/Almost Always Careful** percentage
- **Trust Distribution** analysis

### Intervention Metrics
- **Success Score** based on historical outcomes
- **Cost Effectiveness** (improvement per £1000)
- **Expected Improvements** in trust, cohesion, sentiment
- **Evidence Base** from similar cases

## 🚨 Alert System

The system can send automated alerts when:
- **Critical risk areas** are detected
- **Anomalous patterns** are identified
- **Threshold breaches** occur

Alert channels:
- **Email** for detailed reports
- **SMS** for urgent notifications
- **Configurable thresholds** and frequency limits

## 🧪 Testing the System

### Quick Test
```bash
# Run the complete demo
python example.py

# Launch web dashboard
streamlit run streamlit_app.py
```

### Individual Component Tests
```bash
# Test each component
python main.py warning analyze --areas 50
python main.py sentiment map --areas 30
python main.py trust summary
python main.py trust top --top 5
python main.py intervention recommend --recommendations 3
python main.py simulator optimize --budget 500 --target trust
python main.py alerts test
```

### Expected Output
The system should show:
- All components initialized successfully
- Sample data generated and processed
- Real IMD and Good Neighbours data loaded
- Risk analysis completed
- Intervention recommendations generated
- Impact simulation completed

## 🔧 Troubleshooting

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

#### 2. **Data File Not Found**
**Problem**: "File not found" errors for IMD or Good Neighbours data
**Solution**: 
```bash
# Ensure data files are in the data/ folder:
# - data/IMD2019_Index_of_Multiple_Deprivation.xlsx
# - data/good_neighbours_full_data_by_msoa.xlsx
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

## 📊 Data Sources and Attribution

- **Indices of Multiple Deprivation 2019**: [GOV.UK](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)
- **Good Neighbours Social Trust Data**: Local survey data
- **ONS Postcode Directory**: [Office for National Statistics](https://www.ons.gov.uk/)
- **Community Life Survey**: [GOV.UK](https://www.gov.uk/government/collections/community-life-survey)
- **Postcode Data**: [postcodes.io](https://postcodes.io/)

## 🔮 Future Enhancements

- **Real-time data integration** from social media and surveys
- **Additional data sources** (census, health, education)
- **Advanced ML models** for prediction
- **Mobile application** for field workers
- **API endpoints** for third-party integration
- **Multi-nation support** (Wales, Scotland, Northern Ireland)

## 📄 License

This project is open source. Please ensure you comply with the terms of use for the underlying data sources.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to:
- Add support for additional data sources
- Improve machine learning models
- Enhance visualization capabilities
- Add new intervention types
- Improve documentation

## 📞 Support

For questions or support, please:
1. **Check this documentation** and troubleshooting section
2. **Run the example script** to verify system functionality
3. **Check system status** with `python main.py status`
4. **Submit an issue** with:
   - Operating system and Python version
   - Complete error message
   - Steps to reproduce the issue
   - Output from `python main.py status`

---

**Built for local authorities, community groups, and policymakers to enhance social cohesion and community resilience.**