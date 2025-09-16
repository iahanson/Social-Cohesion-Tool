# 📰 Local News Analyzer Integration

## Overview

The Local News Analyzer is now integrated into the Social Cohesion Tool as a new tab that provides real-time analysis of local news for social trust and community sentiment.

## Features

### 🗺️ Interactive Coverage Map
- **Click anywhere on the map** to analyze local news for that area
- Shows coverage areas for 8 major UK news sources
- Red markers indicate news source locations
- Blue circles show coverage radius for each source

### 📊 Real-time Analysis
When you click on a location, the system:
1. **Finds the closest news source** covering that area
2. **Scrapes the most recent article** from that source
3. **Analyzes sentiment** (-1 to +1 scale)
4. **Calculates trust score** (1-10 scale)
5. **Identifies community topics** (housing, transport, crime, etc.)
6. **Shows trust indicators** (positive/negative mentions)

### 📄 Article Display
- Article title and preview
- Direct link to full article
- Source name and distance from clicked location
- Analysis timestamp

### 🏷️ Topic Analysis
Identifies and counts mentions of:
- **Housing**: homes, rent, property, development
- **Transport**: bus, train, traffic, roads
- **Education**: schools, universities, students
- **Healthcare**: hospitals, NHS, medical services
- **Crime & Safety**: police, crime, security
- **Economy**: jobs, employment, business
- **Environment**: green spaces, pollution, climate
- **Local Government**: council, mayor, policy

## How to Use

### 1. Access the News Analyzer
- Open the Social Cohesion Tool Streamlit app
- Select **"📰 Local News Analyzer"** from the sidebar

### 2. Analyze a Location
**Option A: Click on Map**
- Click anywhere on the interactive map
- Wait for analysis to complete (may take 30-60 seconds)
- View results in the right panel

**Option B: Quick City Analysis**
- Use the quick buttons for major cities:
  - 📍 London
  - 📍 Manchester
  - 📍 Birmingham
  - 📍 Glasgow

### 3. Interpret Results

**Sentiment Score:**
- `> 0.1`: 🟢 Positive community sentiment
- `-0.1 to 0.1`: 🟡 Neutral sentiment
- `< -0.1`: 🔴 Negative community sentiment

**Trust Score:**
- `7-10`: 🟢 High community trust
- `4-7`: 🟡 Moderate community trust
- `1-4`: 🔴 Low community trust

## Coverage Areas

The system covers these UK regions:

| Source | Location | Coverage Radius |
|--------|----------|----------------|
| Manchester Evening News | Manchester | 50km |
| Birmingham Live | Birmingham | 40km |
| Liverpool Echo | Liverpool | 45km |
| Yorkshire Evening Post | Leeds | 35km |
| Bristol Post | Bristol | 30km |
| Chronicle Live | Newcastle | 40km |
| The Herald | Glasgow | 60km |
| Wales Online | Cardiff | 50km |

## Technical Details

### Files Added
- `src/local_news_analyzer.py` - Main analyzer class
- `src/news_sources_finder.py` - News source finder utility
- `src/news_scraper_analyzer.py` - Complete scraping system
- `test_news_integration.py` - Integration test script

### Dependencies
- `requests` - Web scraping
- `beautifulsoup4` - HTML parsing
- `geopy` - Distance calculations
- `folium` - Interactive maps
- `streamlit-folium` - Streamlit map integration
- `pandas` - Data handling
- `plotly` - Charts and visualizations

### Error Handling
The system gracefully handles:
- No news sources in area
- Website blocking/timeouts
- Network connectivity issues
- Empty or invalid articles

## Limitations

1. **Web Scraping Constraints**
   - Some news sites block automated access
   - Rate limiting may cause delays
   - Article structure varies by site

2. **Coverage Limitations**
   - Currently covers major UK cities only
   - Rural areas may not have local sources
   - Limited to English-language sources

3. **Analysis Accuracy**
   - Simple keyword-based sentiment analysis
   - May miss context or sarcasm
   - Trust indicators are heuristic-based

## Future Enhancements

### Short Term
- Add RSS feed integration for more reliable article access
- Expand coverage to more local news sources
- Improve sentiment analysis with ML models

### Long Term
- Historical news trend analysis
- Multi-source article comparison
- Integration with social media sentiment
- Real-time news alerts for areas

## Testing

Run the integration test:
```bash
python3 test_news_integration.py
```

Expected output:
- ✅ News analyzer working
- ✅ Source detection working
- ✅ Map generation working
- Coverage map saved as `test_coverage_map.html`

## Usage Examples

### Example 1: Analyzing Manchester
1. Click on Manchester on the map
2. System finds Manchester Evening News (0km away)
3. Scrapes recent article about local transport issues
4. Shows sentiment: 0.0 (neutral), trust: 5.0/10
5. Top topic: Transport (4 mentions)

### Example 2: Analyzing Birmingham
1. Click Birmingham area
2. Birmingham Live covers the location
3. Article about community housing development
4. Positive sentiment (+0.3), high trust (7.2/10)
5. Topics: Housing, Local Government

## Troubleshooting

**"No news sources found"**
- You clicked in an area without coverage
- Try clicking closer to a major city

**"Could not retrieve article"**
- Website may be blocking access
- Try a different location
- Check internet connectivity

**Map not loading**
- Ensure `streamlit-folium` is installed
- Refresh the page
- Check browser console for errors

**Analysis taking too long**
- News sites may have slow response times
- Analysis includes web scraping which takes time
- Wait up to 60 seconds for results

---

## Integration Complete! 🎉

The Local News Analyzer is now fully integrated into the Social Cohesion Tool. Click on any area of the map to get instant analysis of local news sentiment and community topics.

**Ready to use:** Select "📰 Local News Analyzer" from the sidebar to start analyzing local news!