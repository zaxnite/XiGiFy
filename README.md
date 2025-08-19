Simplified IoT Sensor Data Analysis with Google Gemini API

This project provides a Python-based workflow to analyze structured IoT sensor data from JSON files using Google Gemini API. The script extracts insights about energy usage, anomalies, sensor health, and behavioral patterns in smart buildings, generating both internal and customer-facing reports.

Table of Contents

Features

Requirements

Installation

Usage

Command-Line Arguments

Output

Logging

JSON Data Format

Error Handling

License

Features

Validates JSON sensor data structure and required fields.

Preprocesses and summarizes key metrics:

Device usage percentages (light, AC, fan, occupancy, doors/windows).

Energy consumption (kWh) and efficiency metrics.

Device activity cycles and runtime statistics.

Anomaly detection and sensor health events.

Generates professional reports using Google Gemini AI:

Customer report – user-friendly, actionable, and simplified.

Internal report – detailed, technical, with calculations and metrics references.

Retry logic for API calls and robust error handling.

Saves both preprocessed data and AI analysis into timestamped text files.

Requirements

Python 3.10+

Python packages (install via pip):

pip install google-generativeai python-dotenv


A .env file containing your Gemini API key:

GEMINI_API_KEY=your_api_key_here

Installation

Clone the repository:

git clone https://github.com/yourusername/iot-gemini-analysis.git
cd iot-gemini-analysis


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Create a .env file in the project root and add your Gemini API key.

Usage

Run the script via command line:

python iot_analysis.py --day day_4 --report-type customer --input-dir ./data/json --output-dir ./data/reports


The script will:

Load the specified JSON file (e.g., day_4.json).

Validate its structure.

Preprocess key metrics for analysis.

Send data to Google Gemini API for AI-powered analysis.

Print the report in the console.

Save the report and preprocessed data in the appropriate output folder.

Command-Line Arguments
Argument	Description	Default
--day	Identifier for the JSON file (e.g., day_4).	day_4
--report-type	Type of report: customer or internal.	customer
--input-dir	Directory containing input JSON files.	Current dir
--output-dir	Base directory for output reports.	Current dir
--max-retries	Maximum number of retries for Gemini API calls.	3
--model	Google Gemini model to use (e.g., gemini-1.5-flash).	gemini-1.5-flash
--log-level	Logging level: DEBUG, INFO, WARNING, ERROR.	INFO
Output

Report Folder Structure:

./Reports-Customer/day_4_customer.txt
./Reports-Internal/day_4_internal.txt


Each report includes:

Preprocessed sensor data summary.

AI-generated analysis.

Timestamp and source JSON file reference.

Example sections in a customer report:

🚀 Executive Summary

🌡️ How is the room performing?

💡 Energy Waste

⚠️ Any Warning Signs

🕒 How is the room being used?

🔗 Sensors & Controls

✅ Actionable Improvements

Logging

Logs are written to iot_analysis.log and also displayed in the console.

Supports multiple levels: DEBUG, INFO, WARNING, ERROR.

JSON Data Format

The input JSON must include the following top-level fields:

room_id

summary_window (start & end timestamps)

ml_abstracted_metrics:

usage_features

cross_sensor_ratios

run_length_statistics

raw_stream_derived_metrics:

event_counts_and_timing

energy_proxy_metrics

Optional: sensor_health_events, manual_override_counters

Validation: Missing required fields will terminate the analysis with an error message.

Error Handling

Missing API key → exits with error.

Invalid/missing JSON file → exits with error.

Gemini API failures → retries with exponential backoff (configurable via --max-retries).

Unhandled exceptions are logged and terminate the program gracefully.

License

This project is licensed under the MIT License. See LICENSE
 for details.