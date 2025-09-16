# Setup Guide

This guide provides detailed installation and setup instructions for the Social Cohesion Monitoring System.

## Prerequisites

### System Requirements
- **Python 3.7+** (tested with Python 3.8-3.11)
- **Internet connection** for data downloads
- **~500MB disk space** for data and models
- **Azure OpenAI account** (optional, for GenAI features)

### Operating System Support
- **Windows 10/11** (tested)
- **macOS 10.14+** (tested)
- **Linux Ubuntu 18.04+** (tested)

## Installation Methods

### Method 1: Conda (Recommended)

1. **Install Miniconda or Anaconda**
   - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)
   - Follow installation instructions for your OS

2. **Create environment**
   ```bash
   conda create -n social-cohesion python=3.9
   conda activate social-cohesion
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Virtual Environment

1. **Create virtual environment**
   ```bash
   python -m venv social-cohesion
   
   # Windows
   social-cohesion\Scripts\activate
   
   # macOS/Linux
   source social-cohesion/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Method 3: System Python (Not Recommended)

```bash
pip install -r requirements.txt
```

## Data Setup

### Required Data Files

Place these files in the `data/` directory:

1. **IMD Data**
   - File: `IMD2019_Index_of_Multiple_Deprivation.xlsx`
   - Source: [GOV.UK IMD 2019](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)
   - Sheet: `IMD2019`

2. **Good Neighbours Data**
   - File: `good_neighbours_full_data_by_msoa.xlsx`
   - Source: Local survey data
   - Columns: MSOA_code, MSOA_name, always_trust OR usually_trust, usually_careful OR almost_always_careful, Net_trust

3. **Community Life Survey Data**
   - File: `Community_Life_Survey_2023_24.xlsx`
   - Source: [GOV.UK Community Life Survey](https://www.gov.uk/government/statistics/community-life-survey-2023-24)
   - Multiple sheets with social trust and community cohesion data

4. **Unemployment Data**
   - File: `unmenploymentSept25.xls`
   - Source: ONS unemployment statistics
   - Columns: Geography Code (Column B), Geography Name (Column A), People Looking for Work (Column E), Unemployment Proportion (Column H)

5. **Local News Data**
   - File: `england_local_news_batch100_full_completed.csv`
   - Source: Scraped local news articles
   - Columns: local_authority_district, brief_description, referenced_place, url, source

6. **Geographic Data**
   - File: `Local_Authority_Districts_May_2023.csv`
   - Source: ONS geographic boundaries
   - Columns: LAD24CD, LAD24NM, LAT, LONG for mapping

7. **Population Data**
   - File: `Census_population_2022.xlsx`
   - Source: ONS Census data
   - LSOA-level population statistics

8. **Area Lookup Data**
   - File: `Census21 areaLookupTable.xlsx`
   - Source: ONS Census lookup tables
   - LSOA to MSOA mapping data

### Data Directory Structure

```
data/
├── IMD2019_Index_of_Multiple_Deprivation.xlsx
├── good_neighbours_full_data_by_msoa.xlsx
├── Community_Life_Survey_2023_24.xlsx
├── unmenploymentSept25.xls
├── england_local_news_batch100_full_completed.csv
├── Local_Authority_Districts_May_2023.csv
├── Census_population_2022.xlsx
├── Census21 areaLookupTable.xlsx
├── lsoa_msoa_mapping.json
├── msoa_population_cache.json
└── alert_log.json
```

## Verification

### Test Installation

1. **Check system status**
   ```bash
   python main.py status
   ```

2. **Run example script**
   ```bash
   python example.py
   ```

3. **Launch dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

### Expected Output

The system should show:
- ✅ All components initialized successfully
- ✅ Sample data generated and processed
- ✅ Real IMD and Good Neighbours data loaded
- ✅ Community Life Survey data processed
- ✅ Unemployment data integrated
- ✅ Local news data mapped and analyzed
- ✅ Population data cached and aggregated
- ✅ Risk analysis completed
- ✅ Intervention recommendations generated
- ✅ Impact simulation completed
- ✅ Interactive maps rendered successfully

## Next Steps

1. **Configure environment variables** - See [Configuration Guide](CONFIGURATION.md)
2. **Set up GenAI features** - See [Configuration Guide](CONFIGURATION.md#genai-setup)
3. **Configure alerts** - See [Configuration Guide](CONFIGURATION.md#alert-setup)
4. **Test all features** - See [Troubleshooting Guide](TROUBLESHOOTING.md)

## Common Setup Issues

### Python Not Found
```bash
# Windows - use full path
C:\Users\username\miniforge3\envs\envname\python.exe main.py status

# Activate conda environment first
conda activate social-cohesion
```

### Permission Errors
```bash
# Run as administrator (Windows)
# Or use virtual environment
python -m venv social-cohesion
```

### Missing Dependencies
```bash
# Update pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# If specific package fails
pip install package-name --no-cache-dir
```

### Data Files Not Found
- Ensure files are in `data/` directory
- Check file names match exactly
- Verify file permissions (readable)

## Uninstallation

### Remove Environment
```bash
# Conda
conda remove -n social-cohesion --all

# Virtual environment
# Simply delete the directory
rm -rf social-cohesion  # macOS/Linux
rmdir /s social-cohesion  # Windows
```

### Clean Data
```bash
# Remove data directory
rm -rf data/
```

## Advanced Setup

### Development Setup

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd "Social Cohesion Tool"
   ```

2. **Install in development mode**
   ```bash
   pip install -e .
   ```

3. **Install development dependencies**
   ```bash
   pip install pytest black flake8
   ```

### Docker Setup (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "streamlit_app.py"]
```

### Production Setup

1. **Use production environment**
   ```bash
   conda create -n social-cohesion-prod python=3.9
   conda activate social-cohesion-prod
   ```

2. **Install production dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure production settings**
   - Set `DEBUG=false` in `.env`
   - Use production Azure OpenAI endpoints
   - Configure production alert settings

## Support

If you encounter issues during setup:
1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Verify all prerequisites are met
3. Try the verification steps above
4. Submit an issue with complete error details
