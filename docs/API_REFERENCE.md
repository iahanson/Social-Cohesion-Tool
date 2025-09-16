# API Reference

Complete reference for all commands and APIs in the Social Cohesion Monitoring System.

## Command Line Interface

### System Commands

#### `python main.py status`
Check system status and component health.

**Output:**
- Data source status
- Component initialization status
- Configuration validation results

#### `python main.py --help`
Show available commands and options.

### MSOA Data Commands

#### `python main.py msoa lookup`
Look up MSOA data by postcode or MSOA code.

**Options:**
- `--postcode, -p`: UK postcode to lookup
- `--msoa-code, -m`: MSOA code to lookup
- `--output, -o`: Output format (console, json, csv)

**Examples:**
```bash
python main.py msoa lookup --postcode "SW1A 1AA"
python main.py msoa lookup --msoa-code "E02000001"
python main.py msoa lookup --postcode "SW1A 1AA" --output json
```

**Output:**
- MSOA information
- IMD data (if available)
- Good Neighbours trust data (if available)
- Geographic information

#### `python main.py msoa sources`
List available data sources.

**Examples:**
```bash
python main.py msoa sources
```

**Output:**
- List of available data sources
- Data source descriptions

### GenAI Text Analysis Commands

#### `python main.py genai analyze`
Analyze text for social cohesion issues using GenAI.

**Options:**
- `--text, -t`: Text to analyze
- `--file, -f`: File containing text to analyze
- `--source, -s`: Source of the text (survey, social_media, report, interview, feedback, other)
- `--output, -o`: Output format (console, json, csv, summary)

**Examples:**
```bash
python main.py genai analyze --text "Residents report feeling unsafe in the area"
python main.py genai analyze --file survey_responses.txt --source survey
python main.py genai analyze --text "Community tension issues" --output json
```

**Output:**
- Identified social cohesion issues
- Severity levels (Low, Medium, High, Critical)
- Confidence scores
- Localities found and mapped to MSOAs
- Recommendations

#### `python main.py genai map-locality`
Map a locality to MSOA code.

**Options:**
- `--locality, -l`: Locality to map to MSOA

**Examples:**
```bash
python main.py genai map-locality --locality "Kensington"
python main.py genai map-locality --locality "SW1A 1AA"
python main.py genai map-locality --locality "Hyde Park"
```

**Output:**
- MSOA code
- Local authority
- Region
- Confidence score

#### `python main.py genai search-localities`
Search for localities matching a query.

**Options:**
- `--query, -q`: Search query for localities

**Examples:**
```bash
python main.py genai search-localities --query "park"
python main.py genai search-localities --query "station"
python main.py genai search-localities --query "market"
```

**Output:**
- List of matching localities
- MSOA codes
- Local authorities
- Confidence scores

#### `python main.py genai validate-msoa`
Validate an MSOA code.

**Options:**
- `--msoa-code, -m`: MSOA code to validate

**Examples:**
```bash
python main.py genai validate-msoa --msoa-code E02000001
```

**Output:**
- Validation result
- Local authority information
- Region information

#### `python main.py genai similarity`
Calculate similarity between two texts using embeddings.

**Options:**
- `--text1, -t1`: First text for similarity comparison
- `--text2, -t2`: Second text for similarity comparison

**Examples:**
```bash
python main.py genai similarity --text1 "Safety concerns" --text2 "Crime issues"
```

**Output:**
- Similarity score (0-1)
- Interpretation (Very Similar, Moderately Similar, etc.)
- Confidence percentage

#### `python main.py genai embed`
Generate embedding for a text.

**Options:**
- `--text, -t`: Text to generate embedding for
- `--output, -o`: Output format (console, json)

**Examples:**
```bash
python main.py genai embed --text "Community residents feel unsafe"
python main.py genai embed --text "Sample text" --output json
```

**Output:**
- Embedding vector
- Model information
- Dimensions count
- Statistics (mean, std dev, min)

#### `python main.py genai batch-analyze`
Analyze multiple texts from a file.

**Options:**
- `--file, -f`: File containing multiple texts to analyze
- `--source, -s`: Source identifier for the batch
- `--output, -o`: Output format (json, csv, summary)

**Examples:**
```bash
python main.py genai batch-analyze --file multiple_texts.txt
python main.py genai batch-analyze --file survey_batch.txt --source survey
```

**Output:**
- Batch analysis results
- Summary statistics
- Export files (JSON/CSV)

### Trust Data Commands

#### `python main.py trust summary`
Get social trust data summary.

**Output:**
- Total MSOAs
- Average trust scores
- Trust distribution
- Key statistics

#### `python main.py trust lookup`
Get social trust data for a specific MSOA.

**Options:**
- `--msoa-code, -m`: MSOA code to look up

**Examples:**
```bash
python main.py trust lookup --msoa-code "E02000001"
```

**Output:**
- MSOA name
- Net trust score
- Trust percentages
- Interpretation

#### `python main.py trust top`
Show MSOAs with highest social trust.

**Options:**
- `--top, -t`: Number of top trust areas to show

**Examples:**
```bash
python main.py trust top --top 10
```

**Output:**
- Ranked list of highest trust MSOAs
- Trust scores and percentages
- MSOA codes and names

#### `python main.py trust lowest`
Show MSOAs with lowest social trust.

**Options:**
- `--bottom, -b`: Number of lowest trust areas to show

**Examples:**
```bash
python main.py trust lowest --bottom 10
```

**Output:**
- Ranked list of lowest trust MSOAs
- Trust scores and percentages
- MSOA codes and names

### Data Management Commands

#### `python main.py refresh-population-cache`
Refresh the population data cache for faster loading.

**Examples:**
```bash
python main.py refresh-population-cache
```

**Output:**
- Cache refresh status
- Data processing results
- Performance metrics

#### `python main.py refresh-community-survey-data`
Refresh the Community Life Survey data cache.

**Examples:**
```bash
python main.py refresh-community-survey-data
```

**Output:**
- Survey data processing status
- Question extraction results
- Data validation results

### Local News Commands

#### `python main.py news analyze --lad "Kensington and Chelsea"`
Analyze local news for a specific LAD.

**Options:**
- `--lad, -l`: Local Authority District name
- `--output, -o`: Output format (console, json, csv)

**Examples:**
```bash
python main.py news analyze --lad "Kensington and Chelsea"
python main.py news analyze --lad "Barnsley Borough Council" --output json
```

**Output:**
- News articles for the LAD
- Sentiment analysis results
- Social cohesion relevance scores
- Geographic coverage analysis

#### `python main.py news summary`
Get summary of local news data.

**Examples:**
```bash
python main.py news summary
```

**Output:**
- Total articles processed
- Geographic coverage statistics
- Sentiment distribution
- Key themes identified

### Unemployment Commands

#### `python main.py unemployment lookup --lad "Kensington and Chelsea"`
Look up unemployment data for a specific LAD.

**Options:**
- `--lad, -l`: Local Authority District name
- `--output, -o`: Output format (console, json, csv)

**Examples:**
```bash
python main.py unemployment lookup --lad "Kensington and Chelsea"
python main.py unemployment lookup --lad "Barnsley Borough Council" --output json
```

**Output:**
- Unemployment statistics
- People looking for work
- Unemployment proportion
- Geographic information

#### `python main.py unemployment summary`
Get summary of unemployment data.

**Examples:**
```bash
python main.py unemployment summary
```

**Output:**
- Total areas monitored
- Average unemployment rate
- Total people looking for work
- Statistical summaries

### Early Warning System Commands

#### `python main.py warning analyze`
Run early warning analysis.

**Options:**
- `--areas`: Number of areas to analyze

**Examples:**
```bash
python main.py warning analyze --areas 100
```

**Output:**
- Risk scores for areas
- Risk level classifications
- Anomaly detection results
- Risk factor analysis

#### `python main.py warning profile`
Get risk profile for specific MSOA.

**Options:**
- `--msoa-code`: MSOA code to analyze

**Examples:**
```bash
python main.py warning profile --msoa-code "E02000001"
```

**Output:**
- Risk score
- Risk factors
- Recommendations
- Historical trends

### Sentiment Mapping Commands

#### `python main.py sentiment map`
Generate sentiment mapping.

**Options:**
- `--areas`: Number of areas to map

**Examples:**
```bash
python main.py sentiment map --areas 50
```

**Output:**
- Sentiment scores
- Geographic distribution
- Correlation analysis
- Interactive maps

#### `python main.py sentiment profile`
Get sentiment profile for specific MSOA.

**Options:**
- `--msoa-code`: MSOA code to analyze

**Examples:**
```bash
python main.py sentiment profile --msoa-code "E02000001"
```

**Output:**
- Sentiment score
- Component analysis
- Trends and patterns
- Recommendations

### Intervention Commands

#### `python main.py intervention recommend`
Get intervention recommendations.

**Options:**
- `--recommendations`: Number of recommendations to generate

**Examples:**
```bash
python main.py intervention recommend --recommendations 5
```

**Output:**
- Recommended interventions
- Success scores
- Cost-effectiveness analysis
- Evidence base

#### `python main.py intervention analyze`
Analyze specific intervention.

**Options:**
- `--intervention-type`: Type of intervention to analyze

**Examples:**
```bash
python main.py intervention analyze --intervention-type "Community Events Program"
```

**Output:**
- Intervention details
- Success metrics
- Cost analysis
- Implementation guidance

### Engagement Simulator Commands

#### `python main.py simulator optimize`
Optimize intervention mix.

**Options:**
- `--budget`: Budget constraint
- `--target`: Optimization target (overall, trust, cohesion, sentiment)

**Examples:**
```bash
python main.py simulator optimize --budget 1000 --target overall
```

**Output:**
- Optimal intervention mix
- Expected outcomes
- Budget allocation
- Impact projections

### Alert Commands

#### `python main.py alerts test`
Test alert system.

**Output:**
- Alert system status
- Configuration validation
- Test message results

#### `python main.py alerts summary`
Get alert summary.

**Options:**
- `--hours`: Time period for summary

**Examples:**
```bash
python main.py alerts summary --hours 24
```

**Output:**
- Alert history
- Trigger statistics
- Performance metrics

## Python API

### Unified Data Connector

```python
from src.unified_data_connector import UnifiedDataConnector, MSOADataResult

# Initialize connector
connector = UnifiedDataConnector()

# Get data for single MSOA
results = connector.get_msoa_data("E02000001")
for source, result in results.items():
    if result.success:
        print(f"{source}: {result.data}")
    else:
        print(f"{source}: {result.error_message}")

# Get data for multiple MSOAs
msoa_codes = ["E02000001", "E02000002", "E02000003"]
all_results = connector.get_multiple_msoas_data(msoa_codes)

# Get top performing MSOAs
top_areas = connector.get_top_performing_msoas('net_trust', 10, 'good_neighbours')

# Get data summary
summary = connector.get_data_summary()

# Export data
json_data = connector.export_data(msoa_codes, 'json')
csv_data = connector.export_data(msoa_codes, 'csv')
```

### GenAI Text Analyzer

```python
from src.genai_text_analyzer import GenAITextAnalyzer, TextAnalysisResult, SocialCohesionIssue

# Initialize analyzer
analyzer = GenAITextAnalyzer()

# Analyze single text
result = analyzer.analyze_text("Community residents report safety concerns", "survey")

# Access results
print(f"Total issues: {result.total_issues}")
print(f"Critical issues: {result.critical_issues}")
for issue in result.issues:
    print(f"Issue: {issue.issue_type} - {issue.severity}")
    print(f"Description: {issue.description}")
    print(f"Confidence: {issue.confidence}")

# Analyze multiple texts
texts = [
    ("Text 1", "source1", "id1"),
    ("Text 2", "source2", "id2")
]
results = analyzer.analyze_multiple_texts(texts)

# Generate embeddings
embedding = analyzer.generate_single_embedding("Sample text")
embeddings = analyzer.generate_embeddings(["Text 1", "Text 2"])

# Calculate similarity
similarity = analyzer.calculate_text_similarity("Text 1", "Text 2")

# Find similar issues
similar_issues = analyzer.find_similar_issues("target text", existing_issues)

# Export results
json_export = analyzer.export_results(results, 'json')
csv_export = analyzer.export_results(results, 'csv')
summary_export = analyzer.export_results(results, 'summary')
```

### Locality Mapper

```python
from src.locality_mapper import LocalityMapper, LocalityInfo

# Initialize mapper
mapper = LocalityMapper()

# Map locality to MSOA
locality_info = mapper.map_locality("Kensington")
if locality_info:
    print(f"MSOA: {locality_info.msoa_code}")
    print(f"Local Authority: {locality_info.local_authority}")
    print(f"Confidence: {locality_info.confidence}")

# Postcode to MSOA
msoa_info = mapper.postcode_to_msoa("SW1A 1AA")
if msoa_info:
    print(f"MSOA Code: {msoa_info['msoa_code']}")
    print(f"Local Authority: {msoa_info['local_authority']}")

# Search localities
search_results = mapper.search_localities("park")
for result in search_results:
    print(f"{result.name} -> {result.msoa_code}")

# Validate MSOA
is_valid = mapper.validate_msoa_code("E02000001")
msoa_details = mapper.get_msoa_details("E02000001")

# Get all MSOA codes
all_codes = mapper.get_all_msoa_codes()
local_authorities = mapper.get_local_authorities()

# Export mapping data
json_data = mapper.export_mapping_data('json')
csv_data = mapper.export_mapping_data('csv')
```

### Early Warning System

```python
from src.early_warning_system import EarlyWarningSystem

# Initialize system
ew_system = EarlyWarningSystem()

# Run full analysis
results = ew_system.run_full_analysis()

# Get risk profile for MSOA
risk_profile = ew_system.get_risk_profile("E02000001")

# Get risk factors
risk_factors = ew_system.get_risk_factors(data, "E02000001")
```

### Sentiment Mapping

```python
from src.sentiment_mapping import SentimentMapping

# Initialize system
sm_system = SentimentMapping()

# Run full analysis
results = sm_system.run_full_analysis()

# Get sentiment profile
profile = sm_system.get_sentiment_profile("E02000001")

# Analyze area profile
area_profile = sm_system.analyze_area_profile(data, "E02000001")
```

### Intervention Tool

```python
from src.intervention_tool import InterventionTool

# Initialize tool
int_tool = InterventionTool()

# Run full analysis
recommendations = int_tool.run_full_analysis()

# Get recommendations for MSOA
msoa_recommendations = int_tool.get_recommendations_for_msoa("E02000001")

# Analyze intervention
analysis = int_tool.analyze_intervention("Community Events Program")
```

### Engagement Simulator

```python
from src.engagement_simulator import EngagementSimulator

# Initialize simulator
simulator = EngagementSimulator()

# Optimize intervention mix
optimization = simulator.optimize_intervention_mix(baseline_area, budget=1000)

# Simulate intervention impact
impact = simulator.simulate_intervention_impact(intervention, baseline_area)
```

### Alert System

```python
from src.alert_system import AlertSystem

# Initialize system
alert_system = AlertSystem()

# Send alert
alert_system.send_alert("E02000001", "high_risk", "Critical risk detected")

# Test alert system
alert_system.test_alert_system()

# Get alert summary
summary = alert_system.get_alert_summary(hours=24)
```

## Data Structures

### MSOADataResult
```python
@dataclass
class MSOADataResult:
    msoa_code: str
    msoa_name: str
    data_source: str
    data: Dict[str, Any]
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
```

### TextAnalysisResult
```python
@dataclass
class TextAnalysisResult:
    text_id: str
    source: str
    timestamp: datetime
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    issues: List[SocialCohesionIssue]
    localities_found: List[Dict[str, str]]
    summary: str
    recommendations: List[str]
```

### SocialCohesionIssue
```python
@dataclass
class SocialCohesionIssue:
    issue_type: str
    severity: str  # Low, Medium, High, Critical
    description: str
    confidence: float  # 0-1
    location_mentioned: Optional[str] = None
    msoa_code: Optional[str] = None
    local_authority: Optional[str] = None
    keywords: List[str] = None
    context: str = ""
```

### LocalityInfo
```python
@dataclass
class LocalityInfo:
    name: str
    type: str  # postcode, borough, ward, landmark, area
    msoa_code: str
    local_authority: str
    region: str
    coordinates: Optional[Tuple[float, float]] = None
    confidence: float = 1.0
```

## Error Handling

### Common Exceptions

- `FileNotFoundError`: Data files not found
- `ConnectionError`: API connection failed
- `ValueError`: Invalid input parameters
- `ImportError`: Missing dependencies
- `PermissionError`: File access denied

### Error Response Format

```python
{
    "success": False,
    "error": "Error message",
    "error_code": "ERROR_CODE",
    "timestamp": "2024-01-01T00:00:00Z",
    "details": {
        "component": "component_name",
        "operation": "operation_name",
        "parameters": {...}
    }
}
```

## Rate Limits

### Azure OpenAI
- **GPT-4.1-mini**: 10,000 tokens per minute
- **text-embedding-3-large**: 1,000 requests per minute
- **Batch processing**: Recommended for large volumes

### Postcodes.io API
- **Free tier**: 1,000 requests per day
- **Rate limit**: 100 requests per minute

### Recommendations
- Use batch processing for multiple texts
- Implement retry logic with exponential backoff
- Cache results when possible
- Monitor usage and implement throttling
