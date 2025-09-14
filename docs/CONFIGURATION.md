# Configuration Guide

This guide covers all configuration options for the Social Cohesion Monitoring System.

## Environment Variables

### Basic Configuration

Create a `.env` file in the project root:

```bash
cp env_template.txt .env
```

### Data Source Configuration

```bash
# IMD Data (Indices of Multiple Deprivation)
IMD_DATA_USE_REAL_DATA=true
IMD_DATA_FILE_PATH=data/IMD2019_Index_of_Multiple_Deprivation.xlsx

# Good Neighbours Social Trust Data
GOOD_NEIGHBOURS_USE_REAL_DATA=true
GOOD_NEIGHBOURS_FILE_PATH=data/good_neighbours_full_data_by_msoa.xlsx

# ONS Census Data
ONS_CENSUS_USE_REAL_DATA=false
ONS_CENSUS_URL=https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/middlesuperoutputareamidyearpopulationestimates/mid2020/sape22dt1amid2020on2021geography.xlsx

# Community Life Survey
COMMUNITY_LIFE_SURVEY_USE_REAL_DATA=false
COMMUNITY_LIFE_SURVEY_URL=https://www.gov.uk/government/statistics/community-life-survey-2020-21

# Crime Data
CRIME_DATA_USE_REAL_DATA=false
CRIME_DATA_URL=https://data.police.uk/data/

# Economic Data
ECONOMIC_DATA_USE_REAL_DATA=false
ECONOMIC_DATA_URL=https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/employmentbyoccupationemp04

# Postcode Data (always real)
POSTCODE_DATA_USE_REAL_DATA=true
POSTCODE_DATA_URL=https://api.postcodes.io/
```

## GenAI Setup

### Azure OpenAI Configuration

```bash
# Azure OpenAI Service Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Model Configuration
AZURE_OPENAI_MODEL=gpt-4.1-mini
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### Getting Azure OpenAI Credentials

1. **Create Azure Account**
   - Sign up at [azure.microsoft.com](https://azure.microsoft.com/)

2. **Create OpenAI Resource**
   - Go to Azure Portal
   - Create new "OpenAI" resource
   - Choose region and pricing tier

3. **Get Credentials**
   - Copy the endpoint URL
   - Generate API key from Keys section
   - Note the resource name

### Alternative Models

```bash
# Alternative model options:
AZURE_OPENAI_MODEL=gpt-4o-mini
AZURE_OPENAI_MODEL=gpt-4o
AZURE_OPENAI_MODEL=gpt-35-turbo

# Alternative embedding models:
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

## Alert System Setup

### Email Configuration

```bash
# Email Configuration
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

#### Gmail Setup

1. **Enable 2-Factor Authentication**
2. **Generate App Password**
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. **Use App Password** (not regular password)

#### Other Email Providers

```bash
# Outlook/Hotmail
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587

# Yahoo
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587

# Custom SMTP
SMTP_SERVER=your-smtp-server.com
SMTP_PORT=587
```

### SMS Configuration (Twilio)

```bash
# Twilio SMS Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+447700900123
```

#### Twilio Setup

1. **Create Twilio Account**
   - Sign up at [twilio.com](https://www.twilio.com/)

2. **Get Credentials**
   - Copy Account SID from dashboard
   - Copy Auth Token from dashboard
   - Purchase phone number

3. **Configure Phone Number**
   - Use international format (+44 for UK)
   - Verify number if required

## System Configuration

### Debug and Logging

```bash
# Debug mode
DEBUG=false

# Log level
LOG_LEVEL=INFO

# Data cache settings
CACHE_ENABLED=true
CACHE_TTL_HOURS=24
```

### Performance Settings

```bash
# Memory settings
MAX_MEMORY_USAGE=80  # Percentage

# Processing settings
BATCH_SIZE=100
MAX_CONCURRENT_REQUESTS=5

# Timeout settings
REQUEST_TIMEOUT=30  # seconds
```

## Configuration Examples

### Development Environment

```bash
# Development with dummy data
IMD_DATA_USE_REAL_DATA=false
GOOD_NEIGHBOURS_USE_REAL_DATA=false
AZURE_OPENAI_API_KEY=your_dev_key_here
AZURE_OPENAI_ENDPOINT=https://your-dev-resource.openai.azure.com/
DEBUG=true
LOG_LEVEL=DEBUG
```

### Production Environment

```bash
# Production with real data
IMD_DATA_USE_REAL_DATA=true
GOOD_NEIGHBOURS_USE_REAL_DATA=true
AZURE_OPENAI_API_KEY=your_production_key
AZURE_OPENAI_ENDPOINT=https://your-production-resource.openai.azure.com/
EMAIL_USERNAME=alerts@yourdomain.com
EMAIL_PASSWORD=your_email_password
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
DEBUG=false
LOG_LEVEL=WARNING
```

### Testing Environment

```bash
# Testing configuration
IMD_DATA_USE_REAL_DATA=true
GOOD_NEIGHBOURS_USE_REAL_DATA=true
AZURE_OPENAI_API_KEY=your_test_key
AZURE_OPENAI_ENDPOINT=https://your-test-resource.openai.azure.com/
DEBUG=true
LOG_LEVEL=INFO
CACHE_ENABLED=false
```

## Validation

### Test Configuration

```bash
# Test all configurations
python main.py status

# Test specific components
python main.py alerts test
python main.py genai analyze --text "Test configuration"
```

### Configuration Validation

The system validates configuration on startup:

- ✅ **Data sources** - File existence and format
- ✅ **Azure OpenAI** - API key and endpoint connectivity
- ✅ **Email settings** - SMTP server connectivity
- ✅ **SMS settings** - Twilio credentials validation
- ✅ **File permissions** - Read/write access

## Security Considerations

### API Keys
- **Never commit** `.env` files to version control
- **Use environment variables** in production
- **Rotate keys** regularly
- **Use least privilege** access

### Data Protection
- **Encrypt sensitive data** at rest
- **Use HTTPS** for all API calls
- **Implement access controls** for data files
- **Regular backups** of configuration

### Network Security
- **Firewall rules** for outbound connections
- **VPN access** for sensitive environments
- **Monitor API usage** for anomalies

## Troubleshooting Configuration

### Common Issues

#### Azure OpenAI Not Working
```bash
# Check API key format
AZURE_OPENAI_API_KEY=sk-...  # Should start with sk-

# Verify endpoint format
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Test connectivity
python -c "import openai; print('OpenAI available')"
```

#### Email Not Sending
```bash
# Check SMTP settings
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Verify credentials
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password  # Not regular password
```

#### Data Files Not Loading
```bash
# Check file paths
IMD_DATA_FILE_PATH=data/IMD2019_Index_of_Multiple_Deprivation.xlsx
GOOD_NEIGHBOURS_FILE_PATH=data/good_neighbours_full_data_by_msoa.xlsx

# Verify file permissions
ls -la data/
```

### Configuration Testing

```bash
# Test individual components
python main.py alerts test
python main.py genai analyze --text "Configuration test"

# Check data loading
python main.py lookup --msoa-code "E02000001"

# Verify dashboard
streamlit run streamlit_app.py
```

## Advanced Configuration

### Custom Data Sources

```python
# Add custom data source
from src.unified_data_connector import UnifiedDataConnector

connector = UnifiedDataConnector()
# Add custom data loading logic
```

### Custom Models

```bash
# Use different ML models
MODEL_TYPE=custom
MODEL_PATH=models/custom_model.pkl
```

### Custom Alerts

```python
# Custom alert configuration
ALERT_THRESHOLDS={
    "risk_score": 0.8,
    "trust_score": 0.3,
    "sentiment_score": 0.2
}
```

## Support

For configuration issues:
1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Verify all environment variables
3. Test individual components
4. Submit an issue with configuration details (excluding sensitive data)
