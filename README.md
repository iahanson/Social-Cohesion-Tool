# Social Cohesion Monitoring System

A comprehensive Python-based application for identifying areas of low social cohesion and rising tensions, and prioritizing interventions for local stakeholders. The system provides early warning capabilities, sentiment mapping, intervention recommendations, community engagement simulation tools, **GenAI-powered text analysis** for social cohesion issues, **interactive mapping with choropleth visualization**, **local news analysis**, and **unemployment data integration**.

## Quick Start

**Get up and running in 5 minutes:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (optional)
cp env_template.txt .env
# Edit .env with your Azure OpenAI credentials for GenAI features

# 3. Test the system
python example.py

# 4. Launch web dashboard
python main.py dashboard
# OR
streamlit run streamlit_app.py
```

**What you'll see:**
- Complete system demonstration with sample data
- Early warning analysis with risk scoring
- Interactive maps showing social trust levels with choropleth visualization
- Evidence-based intervention recommendations
- Community engagement impact simulation
- Real social trust data from Good Neighbours survey
- **GenAI text analysis** for social cohesion issues
- **Unified data connector** for streamlined data access
- **Local news analysis** with sentiment mapping
- **Unemployment data integration** for economic indicators
- **Similar LADs analysis** based on risk scores
- **Interactive UK map** with clickable LADs and risk visualization

## Key Features

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

### 4. 🤖 GenAI Text Analysis
- **Multi-provider support** (Azure OpenAI GPT-4.1-mini, AWS Bedrock Claude Sonnet 4)
- **Social cohesion issue detection** from text inputs
- **Automatic locality mapping** to MSOA codes
- **Severity classification** (Low, Medium, High, Critical)
- **Text similarity analysis** using embeddings
- **Batch processing** for multiple texts
- **Export capabilities** (JSON, CSV, summary)
- **Local news sentiment analysis** with social cohesion relevance scoring

### 5. Intervention Effectiveness Tool
- **Case-based recommendations** based on similar historical cases
- **Evidence-based suggestions** with success metrics
- **Cost-effectiveness analysis** for different intervention types
- **Success factor identification** for optimal outcomes

### 6. Community Engagement Simulator
- **"What-if" analysis** for intervention impacts
- **Budget optimization** for maximum impact
- **Outcome prediction** using machine learning models
- **Scenario comparison** tools

### 7. Interactive Mapping & Visualization
- **Interactive UK map** with clickable LAD boundaries
- **Choropleth visualization** showing risk levels and social indicators
- **Local news coverage mapping** with origin and reference locations
- **Risk assessment visualization** with color-coded areas
- **Click-to-select LAD** functionality for detailed analysis

### 8. Local News Analysis
- **News article processing** from local news sources
- **Location mapping** for originating and referenced places
- **Social cohesion relevance** analysis using GenAI
- **Sentiment analysis** (positive, negative, neutral)
- **Theme extraction** and keyword analysis
- **Geographic coverage** visualization

### 9. Unemployment Data Integration
- **Real unemployment statistics** from ONS data
- **LAD-level unemployment rates** and population counts
- **Economic indicator integration** into risk assessment
- **Trend analysis** and comparative metrics

### 10. Similar LADs Analysis
- **Risk-based similarity** calculation using Early Warning System
- **Top 10 most similar LADs** identification
- **Comparative analysis** of risk factors
- **Evidence-based recommendations** for similar areas

### 11. Alert Management System
- **Automated notifications** for high-risk areas
- **Multi-channel alerts** (SMS, Email)
- **Configurable thresholds** and alert frequency
- **Alert history** and reporting

## Installation

### Prerequisites
- **Python 3.7+** (tested with Python 3.8-3.11)
- **Internet connection** for data downloads
- **~1GB disk space** for data, models, and cached data
- **Azure OpenAI account** (optional, for GenAI features)
- **AWS Bedrock access** (optional, for Claude Sonnet 4 GenAI features)

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

4. **Configure environment variables (optional)**
   ```bash
   cp env_template.txt .env
   # Edit .env with your configuration
   ```

5. **Verify installation**
   ```bash
   python main.py status
   ```

## Usage

### Command Line Interface

#### System Status:
```bash
python main.py status
```

#### MSOA Data Lookup:
```bash
# Lookup by postcode
python main.py msoa lookup --postcode "SW1A 1AA"

# Lookup by MSOA code
python main.py msoa lookup --msoa-code "E02000001"
```

#### GenAI Text Analysis:
```bash
# Analyze single text
python main.py genai analyze --text "Residents report feeling unsafe in the area"

# Map locality to MSOA
python main.py genai map-locality --locality "Kensington"

# Calculate text similarity
python main.py genai similarity --text1 "Safety concerns" --text2 "Crime issues"
```

#### Trust Data Analysis:
```bash
# Get trust data summary
python main.py trust summary

# Show top trust areas
python main.py trust top --top 10
```

### Streamlit Dashboard

Launch the interactive web dashboard:
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
- **🤖 GenAI Text Analysis** with comprehensive text processing capabilities

## Data Sources

The system integrates real data sources through a **unified data connector**:

- **Indices of Multiple Deprivation (IMD) 2019** - [GOV.UK](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)
- **Good Neighbours Social Trust Data** - Local survey data
- **ONS Census Data** - Population, demographics, housing data
- **Community Life Survey** - Social trust, community cohesion, volunteering
- **Unemployment Data** - ONS unemployment statistics by LAD
- **Local News Data** - Scraped local news articles with location mapping
- **Crime Statistics** - Safety indicators, crime rates by area
- **Economic Indicators** - Employment rates, income data, economic uncertainty

## Project Structure

```
├── main.py                      # Enhanced CLI interface
├── streamlit_app.py             # Streamlit dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── example.py                   # Example usage script
├── env_template.txt             # Environment variables template
├── docs/                        # 📁 Supporting documentation
│   ├── SETUP.md                 # Detailed setup instructions
│   ├── CONFIGURATION.md         # Environment configuration guide
│   ├── TROUBLESHOOTING.md       # Common issues and solutions
│   └── API_REFERENCE.md         # Complete API documentation
├── src/
│   ├── unified_data_connector.py    # Unified data connector
│   ├── locality_mapper.py           # Enhanced locality mapping
│   ├── genai_text_analyzer.py      # GenAI text analysis
│   ├── aws_bedrock_client.py       # AWS Bedrock integration
│   ├── early_warning_system.py     # Early warning system
│   ├── sentiment_mapping.py        # Sentiment mapping
│   ├── intervention_tool.py        # Intervention recommendations
│   ├── engagement_simulator.py     # Engagement simulation
│   ├── alert_system.py             # Alert management
│   ├── community_life_survey_connector.py # Community Life Survey data
│   ├── unemployment_connector.py   # Unemployment data connector
│   ├── local_news_connector.py     # Local news data connector
│   ├── lsoa_msoa_mapper.py         # Geographic mapping utilities
│   └── data_config.py              # Data configuration
└── data/                        # Local data storage
    ├── IMD2019_Index_of_Multiple_Deprivation.xlsx
    ├── good_neighbours_full_data_by_msoa.xlsx
    ├── Community_Life_Survey_2023_24.xlsx
    ├── unmenploymentSept25.xls
    ├── england_local_news_batch100_full_completed.csv
    ├── Local_Authority_Districts_May_2023.csv
    ├── Census_population_2022.xlsx
    ├── Census21 areaLookupTable.xlsx
    ├── lsoa_msoa_mapping.json
    └── msoa_population_cache.json
```

## Documentation

- **[Setup Guide](docs/SETUP.md)** - Detailed installation and configuration
- **[Configuration](docs/CONFIGURATION.md)** - Environment variables and settings
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[API Reference](docs/API_REFERENCE.md)** - Complete command and API documentation

## Testing

```bash
# Run the complete demo
python example.py

# Launch web dashboard
streamlit run streamlit_app.py

# Test individual components
python main.py status
python main.py lookup --postcode "SW1A 1AA"
python main.py genai analyze --text "Test text for analysis"
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to:
- Add support for additional data sources
- Improve machine learning models
- Enhance visualization capabilities
- Add new intervention types
- Improve GenAI analysis capabilities

## Support

For questions or support:
1. **Check the [troubleshooting guide](docs/TROUBLESHOOTING.md)**
2. **Run the example script** to verify system functionality
3. **Check system status** with `python main.py status`
4. **Submit an issue** with complete error details

---

**Built for local authorities, community groups, and policymakers to enhance social cohesion and community resilience through data-driven insights and GenAI-powered analysis.**