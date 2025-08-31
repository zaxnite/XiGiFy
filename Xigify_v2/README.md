# XiGiFy

A Python-based tool that analyzes structured IoT sensor data from smart rooms using Google's Gemini AI to detect energy inefficiencies, system anomalies, behavioral patterns, and actionable insights.

## Overview

XiGiFy leverages the power of Google's Gemini AI to transform raw IoT sensor data into meaningful insights for smart room environments. Whether you're managing a smart home, office building, or industrial facility, XiGiFy helps you optimize energy consumption, detect system issues, and understand usage patterns.

## Features

- **Energy Efficiency Analysis**: Calculate efficiency scores (0-100) based on kWh consumption and waste indicators
- **Comfort & Wellbeing Assessment**: Analyze occupancy-device alignment and runtime stability
- **Predictive Analysis**: Detect power drift, pattern anomalies, and seasonal deviations
- **Smart Preprocessing**: Transform raw sensor data into structured ML-ready metrics
- **AI-Powered Insights**: Generate deterministic, actionable recommendations using Google Gemini AI
- **Dual Report Types**: Customer-friendly and technical internal reports
- **Comprehensive Metrics**: Cross-sensor ratios, run-length statistics, and health monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zaxnite/XiGiFy.git
cd XiGiFy
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Google Gemini AI API key:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Project Structure

```
XiGiFy/
├── preprocess.py            # Data preprocessing and analysis
├── xigify.py                # AI interpretation with Gemini
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── day_*.json             # Sample sensor data files
├── Reports-Customer/      # Customer-facing reports
└── Reports-Internal/      # Technical internal reports
```

## Usage

### Step 1: Data Preprocessing

Process raw IoT sensor data to extract meaningful metrics:

```bash
python preprocess.py
```

This will:
- Load sensor data from JSON files
- Calculate energy efficiency scores
- Analyze comfort and wellbeing metrics
- Generate predictive flags and health assessments
- Output processed data with `_preprocessed.json` suffix

### Step 2: AI Interpretation

Generate AI-powered insights using Google Gemini:

```bash
# Generate customer report
python xigify.py --day day_5 --report-type customer

# Generate internal technical report  
python xigify.py --day day_5 --report-type internal
```

### Advanced Usage

```python
# Direct preprocessing in Python
from preprocess import preprocess_sensor_data, load_json_data

# Load and process data
data = load_json_data("day_5.json")
processed = preprocess_sensor_data(data)

# Access analysis results
efficiency = processed['energy_efficiency_analysis']['efficiency_score']
comfort = processed['comfort_wellbeing_analysis']['comfort_score']
health_events = processed['sensor_health_analysis']['total_health_events']
```

## Data Format

XiGiFy processes structured IoT sensor data in JSON format. The expected structure includes:

```json
{
  "room_id": "living_room_01",
  "summary_window": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-01T23:59:59Z"
  },
  "raw_stream_derived_metrics": {
    "energy_proxy_metrics": {
      "light_kwh": 2.4,
      "ac_kwh": 8.7,
      "fan_kwh": 1.2
    },
    "event_counts_and_timing": {
      "light": {"on_count": 12, "off_count": 11},
      "ac": {"on_count": 8, "off_count": 7}
    },
    "manual_override_counters": [
      {"device": "ac", "count": 3},
      {"device": "light", "count": 1}
    ],
    "sensor_health_events": [
      {"sensor": "temperature", "event": "battery_low"}
    ]
  },
  "ml_abstracted_metrics": {
    "usage_features": {
      "occupancy_detected_pct": 0.45,
      "light_on_pct": 0.38,
      "ac_on_pct": 0.62
    },
    "cross_sensor_ratios": {
      "ac_on_while_unoccupied_pct": 0.12,
      "light_on_while_unoccupied_pct": 0.08,
      "ac_on_while_window_open_pct": 0.05
    },
    "run_length_statistics": {
      "ac": {"avg_min": 120, "variance_min2": 2400},
      "occupancy": {"avg_min": 45, "variance_min2": 900}
    }
  }
}
```

## Analysis Outputs

### Energy Efficiency Analysis
- **Efficiency Score**: 0-100 scale based on consumption and waste
- **Total Energy Consumption**: kWh usage breakdown by device
- **Waste Indicators**: Unoccupied usage and inefficient patterns
- **Device Runtime Ratios**: Percentage of time each device was active

### Comfort & Wellbeing Analysis
- **Comfort Score**: 0-100 scale based on occupancy alignment
- **Alignment Metrics**: Device usage vs occupancy correlation
- **Runtime Stability**: Consistency of AC and heating patterns

### Predictive Analysis
- **Drift Detection**: Power consumption trend analysis
- **Pattern Flags**: Rapid cycling and stuck sensor detection
- **Seasonal Analysis**: Weekly and monthly usage comparisons
- **Health Monitoring**: Battery and signal quality assessment

## Command Line Options

```bash
python xigify.py [options]

Options:
  --day DAY                Day identifier (default: day_5)
  --report-type {customer,internal}  Report type (default: customer)
  --input-dir DIR          Input directory for JSON files (default: .)
  --output-dir DIR         Output directory for reports (default: .)
  --max-retries N          API retry attempts (default: 3)
  --model MODEL            Gemini model to use (default: gemini-1.5-flash)
  --log-level LEVEL        Logging level (default: INFO)
```