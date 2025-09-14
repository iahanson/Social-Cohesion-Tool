# Troubleshooting Guide

This guide helps you resolve common issues with the Social Cohesion Monitoring System.

## Quick Diagnostics

### System Status Check
```bash
python main.py status
```

### Complete System Test
```bash
python example.py
```

### Individual Component Tests
```bash
python main.py lookup --postcode "SW1A 1AA"
python main.py genai analyze --text "Test text"
python main.py alerts test
```

## Common Issues and Solutions

### 1. GenAI Features Not Working

#### Problem: GenAI text analysis shows configuration errors
**Symptoms:**
- "Azure OpenAI configuration error" messages
- GenAI commands fail with authentication errors
- Text analysis returns empty results

**Solutions:**
```bash
# 1. Check Azure OpenAI configuration in .env file
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL=gpt-4.1-mini
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# 2. Verify API key format
# Should start with 'sk-' for OpenAI format
# Or be a valid Azure OpenAI key

# 3. Test connectivity
python -c "
import os
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version='2024-02-15-preview',
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
)
print('Connection successful')
"

# 4. Check endpoint format
# Should be: https://your-resource-name.openai.azure.com/
# Not: https://your-resource-name.openai.azure.com/v1/
```

#### Problem: GenAI analysis returns no results
**Solutions:**
```bash
# 1. Check text input
python main.py genai analyze --text "Residents report safety concerns in the area"

# 2. Try different text
python main.py genai analyze --text "Community tension and conflict issues"

# 3. Check model availability
# Ensure your Azure OpenAI resource has the required models deployed
```

### 2. Maps Not Displaying/Flashing

#### Problem: Maps appear briefly then disappear
**Symptoms:**
- Maps load then immediately disappear
- "streamlit-folium not installed" warning
- Maps show as blank areas

**Solutions:**
```bash
# 1. Install missing package
pip install streamlit-folium

# 2. Update Streamlit
pip install streamlit --upgrade

# 3. Clear Streamlit cache
streamlit cache clear

# 4. Use different port
streamlit run streamlit_app.py --server.port 8502

# 5. Check browser compatibility
# Try different browser (Chrome, Firefox, Edge)
```

#### Problem: Maps show as static images
**Solutions:**
```bash
# 1. Install folium dependencies
pip install folium streamlit-folium

# 2. Check JavaScript enabled in browser
# 3. Disable ad blockers temporarily
# 4. Try incognito/private browsing mode
```

### 3. Data File Not Found

#### Problem: "File not found" errors for IMD or Good Neighbours data
**Symptoms:**
- "File not found" error messages
- Data loading fails
- Empty results in analysis

**Solutions:**
```bash
# 1. Check file locations
ls -la data/
# Should show:
# IMD2019_Index_of_Multiple_Deprivation.xlsx
# good_neighbours_full_data_by_msoa.xlsx

# 2. Verify file names (case-sensitive)
# Exact names required:
# IMD2019_Index_of_Multiple_Deprivation.xlsx
# good_neighbours_full_data_by_msoa.xlsx

# 3. Check file permissions
chmod 644 data/*.xlsx

# 4. Verify file format
# Files should be valid Excel files (.xlsx)
# Not CSV or other formats

# 5. Check .env configuration
IMD_DATA_FILE_PATH=data/IMD2019_Index_of_Multiple_Deprivation.xlsx
GOOD_NEIGHBOURS_FILE_PATH=data/good_neighbours_full_data_by_msoa.xlsx
```

#### Problem: Data loads but shows empty results
**Solutions:**
```bash
# 1. Check Excel file structure
# IMD file should have sheet named 'IMD2019'
# Good Neighbours file should have columns:
# MSOA_code, MSOA_name, always_trust OR usually_trust, 
# usually_careful OR almost_always_careful, Net_trust

# 2. Verify data format
python -c "
import pandas as pd
df = pd.read_excel('data/IMD2019_Index_of_Multiple_Deprivation.xlsx', sheet_name='IMD2019')
print(df.columns.tolist())
print(df.head())
"
```

### 4. Python Environment Issues

#### Problem: "Python was not found" or "pip not recognized"
**Solutions:**
```bash
# Windows - use full path
C:\Users\username\miniforge3\envs\envname\python.exe main.py status

# Activate conda environment first
conda activate social-cohesion

# Or use python -m pip
python -m pip install -r requirements.txt

# Check Python installation
python --version
pip --version
```

#### Problem: Package installation fails
**Solutions:**
```bash
# 1. Update pip first
pip install --upgrade pip

# 2. Install with no cache
pip install -r requirements.txt --no-cache-dir

# 3. Install packages individually
pip install streamlit
pip install plotly
pip install pandas
pip install scikit-learn

# 4. Use conda for problematic packages
conda install pandas numpy scikit-learn
pip install streamlit plotly folium
```

### 5. Streamlit Dashboard Issues

#### Problem: Dashboard won't launch or shows errors
**Solutions:**
```bash
# 1. Update Streamlit
pip install streamlit --upgrade

# 2. Clear cache
streamlit cache clear

# 3. Use different port
streamlit run streamlit_app.py --server.port 8502

# 4. Check for port conflicts
netstat -an | grep 8501

# 5. Run with debug mode
streamlit run streamlit_app.py --logger.level debug

# 6. Check browser console for JavaScript errors
# Press F12 in browser and check Console tab
```

#### Problem: Dashboard loads but features don't work
**Solutions:**
```bash
# 1. Check browser compatibility
# Use Chrome, Firefox, or Edge (latest versions)

# 2. Disable browser extensions
# Try incognito/private mode

# 3. Clear browser cache
# Ctrl+Shift+Delete (Windows) or Cmd+Shift+Delete (Mac)

# 4. Check JavaScript errors
# Press F12 and check Console tab for errors
```

### 6. Alert System Not Working

#### Problem: Email/SMS alerts not sending
**Solutions:**
```bash
# 1. Test alert system
python main.py alerts test

# 2. Check email configuration
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password  # Not regular password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# 3. For Gmail - use App Password
# Go to Google Account → Security → 2-Step Verification → App passwords
# Generate password for "Mail"

# 4. Check Twilio configuration
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+447700900123

# 5. Test SMTP connection
python -c "
import smtplib
smtp = smtplib.SMTP('smtp.gmail.com', 587)
smtp.starttls()
smtp.login('your_email@gmail.com', 'your_app_password')
print('SMTP connection successful')
smtp.quit()
"
```

### 7. Performance Issues

#### Problem: System runs slowly or uses too much memory
**Solutions:**
```bash
# 1. Reduce data size for testing
python main.py warning analyze --areas 50  # Instead of 1000

# 2. Clear cache
streamlit cache clear

# 3. Check memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"

# 4. Use smaller datasets
# Set USE_REAL_DATA=false for testing

# 5. Optimize browser
# Close other tabs and applications
```

### 8. CLI Commands Not Working

#### Problem: Commands fail with "command not found" or errors
**Solutions:**
```bash
# 1. Check command syntax
python main.py --help
python main.py genai --help

# 2. Verify file paths
python main.py lookup --postcode "SW1A 1AA"
python main.py lookup --msoa-code "E02000001"

# 3. Check required parameters
python main.py genai analyze --text "Required text here"

# 4. Use full paths if needed
python /full/path/to/main.py status
```

## Error Messages and Solutions

### Common Error Messages

#### `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install streamlit
```

#### `PermissionError: [Errno 13] Permission denied`
```bash
# Run as administrator (Windows)
# Or use virtual environment
python -m venv social-cohesion
```

#### `ConnectionError: Failed to establish connection`
```bash
# Check internet connection
# Verify API endpoints are accessible
# Check firewall settings
```

#### `ValueError: Invalid input parameters`
```bash
# Check command syntax
python main.py --help
# Verify parameter formats
```

#### `FileNotFoundError: [Errno 2] No such file or directory`
```bash
# Check file paths in .env
# Verify files exist in data/ directory
# Check file permissions
```

#### `Azure OpenAI Error: Invalid API key`
```bash
# Verify API key in .env file
# Check key format and validity
# Ensure resource has required models deployed
```

## Getting Help

### Before Submitting an Issue

1. **Run diagnostics**
   ```bash
   python main.py status
   python example.py
   ```

2. **Check logs**
   - Look for error messages in terminal output
   - Check browser console (F12) for JavaScript errors
   - Check `data/` folder for any error logs

3. **Gather information**
   - Operating system and version
   - Python version (`python --version`)
   - Complete error message
   - Steps to reproduce the issue
   - Output from `python main.py status`

### Submitting an Issue

When submitting an issue, include:

1. **System Information**
   - OS: Windows 10/11, macOS, Linux
   - Python version: 3.8, 3.9, 3.10, 3.11
   - Installation method: conda, pip, system

2. **Error Details**
   - Complete error message
   - Command that caused the error
   - Expected vs actual behavior

3. **Configuration**
   - `.env` file contents (remove sensitive data)
   - Data files present in `data/` directory
   - Network connectivity status

4. **Troubleshooting Attempted**
   - Steps already tried
   - Results of diagnostic commands
   - Any workarounds found

### Emergency Workarounds

#### If Nothing Works
```bash
# 1. Fresh installation
conda create -n social-cohesion-new python=3.9
conda activate social-cohesion-new
pip install -r requirements.txt

# 2. Use example data only
# Set all USE_REAL_DATA=false in .env

# 3. Minimal test
python -c "import streamlit; print('Streamlit works')"
python -c "import pandas; print('Pandas works')"
```

#### If Dashboard Won't Load
```bash
# Use command line only
python main.py status
python main.py lookup --postcode "SW1A 1AA"
python main.py genai analyze --text "Test analysis"
```

## Prevention

### Regular Maintenance

1. **Update dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Clear caches**
   ```bash
   streamlit cache clear
   ```

3. **Check disk space**
   ```bash
   df -h  # Linux/Mac
   dir    # Windows
   ```

4. **Monitor logs**
   - Check for recurring errors
   - Monitor performance metrics
   - Update configurations as needed

### Best Practices

1. **Use virtual environments**
2. **Keep data files backed up**
3. **Test configurations before production use**
4. **Monitor API usage and limits**
5. **Regular security updates**
