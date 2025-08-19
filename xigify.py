"""
XiGiFy

This script analyzes structured IoT sensor data from JSON files using Google's Gemini API
to extract insights about energy inefficiencies, anomalies, and behavioral patterns.
"""

import json
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = 'gemini-1.5-flash'
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1
DEFAULT_DAY = 'day_4'
#DEFAULT_REPORT_TYPE = 'internal'
DEFAULT_REPORT_TYPE = 'customer'
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_PRECISION = 1
DEFAULT_PERCENTAGE_PRECISION = 1

# File and output formatting constants
REPORT_SEPARATOR = "=" * 100
SECTION_SEPARATOR = "-" * 50
LOG_FILE_NAME = 'iot_analysis.log'
JSON_FILE_SUFFIX = '.json'

# Report formatting constants
UNKNOWN_VALUE = "Unknown"
NOT_AVAILABLE = "N/A"
NO_HEALTH_ISSUES = "No sensor health issues detected"
NO_MANUAL_OVERRIDES = "No manual overrides detected"

# Validation constants
REQUIRED_TOP_LEVEL_FIELDS = [
    "room_id",
    "summary_window", 
    "ml_abstracted_metrics",
    "raw_stream_derived_metrics"
]

REQUIRED_ML_METRICS_FIELDS = [
    "usage_features",
    "cross_sensor_ratios", 
    "run_length_statistics"
]

REQUIRED_RAW_METRICS_FIELDS = [
    "event_counts_and_timing",
    "energy_proxy_metrics"
]

# Time formatting constants
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_NAME),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_json_structure(data: Dict[str, Any]) -> bool:
    """
    Validate that the JSON has the required structure for IoT sensor data.
    
    Args:
        data: Parsed JSON data
        
    Returns:
        True if valid structure, False otherwise
    """
    # Check top-level fields
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Check ml_abstracted_metrics structure
    ml_metrics = data.get("ml_abstracted_metrics", {})
    for field in REQUIRED_ML_METRICS_FIELDS:
        if field not in ml_metrics:
            logger.error(f"Missing required ml_abstracted_metrics field: {field}")
            return False
    
    # Check raw_stream_derived_metrics structure
    raw_metrics = data.get("raw_stream_derived_metrics", {})
    for field in REQUIRED_RAW_METRICS_FIELDS:
        if field not in raw_metrics:
            logger.error(f"Missing required raw_stream_derived_metrics field: {field}")
            return False
    
    logger.info("JSON structure validation passed")
    return True

def safe_percentage(value: Optional[float]) -> str:
    """Safely format percentage values"""
    if value is None:
        return NOT_AVAILABLE
    return f"{value * 100:.{DEFAULT_PERCENTAGE_PRECISION}f}%"

def safe_float(value: Optional[Union[int, float]], precision: int = DEFAULT_PRECISION) -> str:
    """Safely format float values"""
    if value is None:
        return NOT_AVAILABLE
    return f"{float(value):.{precision}f}"

def safe_int(value: Optional[int]) -> str:
    """Safely format integer values"""
    if value is None:
        return NOT_AVAILABLE
    return str(value)

def preprocess_sensor_data(data: Dict[str, Any]) -> str:
    """
    Extract and format key metrics from the sensor data for analysis.
    
    Args:
        data: Raw JSON sensor data
        
    Returns:
        Formatted string summary of key metrics
    """
    room_id = data.get("room_id", UNKNOWN_VALUE)
    window = data.get("summary_window", {})
    start_time = window.get("start", UNKNOWN_VALUE)
    end_time = window.get("end", UNKNOWN_VALUE)
    
    # Extract ML metrics
    ml_metrics = data.get("ml_abstracted_metrics", {})
    usage = ml_metrics.get("usage_features", {})
    cross_ratios = ml_metrics.get("cross_sensor_ratios", {})
    run_stats = ml_metrics.get("run_length_statistics", {})
    drift_data = ml_metrics.get("rolling_statistics_and_drift", {})
    deviation_flags = ml_metrics.get("pattern_deviation_flags", {})
    seasonal = ml_metrics.get("seasonal_pattern_summaries", {})
    intervals = ml_metrics.get("inter_event_distributions", {})
    
    # Extract raw metrics
    raw_metrics = data.get("raw_stream_derived_metrics", {})
    event_counts = raw_metrics.get("event_counts_and_timing", {})
    energy = raw_metrics.get("energy_proxy_metrics", {})
    health_events = raw_metrics.get("sensor_health_events", [])
    manual_overrides = raw_metrics.get("manual_override_counters", [])
    
    # Calculate total energy
    total_energy = sum(v for v in energy.values() if isinstance(v, (int, float)))
    
    # Format the summary
    summary = f"""
ROOM: {room_id}
TIME PERIOD: {start_time} to {end_time}

USAGE PATTERNS:
- Light on: {safe_percentage(usage.get('light_on_pct'))}
- AC on: {safe_percentage(usage.get('ac_on_pct'))}
- Fan on: {safe_percentage(usage.get('fan_on_pct'))}
- Occupancy detected: {safe_percentage(usage.get('occupancy_detected_pct'))}
- Window open: {safe_percentage(usage.get('window_open_pct'))}
- Door open: {safe_percentage(usage.get('door_open_pct'))}

EFFICIENCY METRICS:
- AC on while unoccupied: {safe_percentage(cross_ratios.get('ac_on_while_unoccupied_pct'))}
- Light on while unoccupied: {safe_percentage(cross_ratios.get('light_on_while_unoccupied_pct'))}
- AC on while window open: {safe_percentage(cross_ratios.get('ac_on_while_window_open_pct'))}

ENERGY CONSUMPTION:
- Light: {safe_float(energy.get('light_kwh'))} kWh
- AC: {safe_float(energy.get('ac_kwh'))} kWh
- Fan: {safe_float(energy.get('fan_kwh'))} kWh
- Total: {safe_float(total_energy)} kWh

DEVICE ACTIVITY:
- Light cycles: {safe_int(event_counts.get('light', {}).get('on_count'))} on/{safe_int(event_counts.get('light', {}).get('off_count'))} off
- AC cycles: {safe_int(event_counts.get('ac', {}).get('on_count'))} on/{safe_int(event_counts.get('ac', {}).get('off_count'))} off
- Fan cycles: {safe_int(event_counts.get('fan', {}).get('on_count'))} on/{safe_int(event_counts.get('fan', {}).get('off_count'))} off
- Door events: {safe_int(event_counts.get('door', {}).get('open_count'))} opens
- Occupancy events: {safe_int(event_counts.get('occupancy', {}).get('detected_count'))} detections

RUN LENGTH STATISTICS:
- Light avg runtime: {safe_float(run_stats.get('light', {}).get('avg_min'))} min
- AC avg runtime: {safe_float(run_stats.get('ac', {}).get('avg_min'))} min
- Fan avg runtime: {safe_float(run_stats.get('fan', {}).get('avg_min'))} min
- Occupancy avg duration: {safe_float(run_stats.get('occupancy', {}).get('avg_min'))} min

POWER DRIFT & ANOMALIES:
- AC power average: {safe_float(drift_data.get('ac_power_avg_w5m'))}W
- AC power drift detected: {drift_data.get('ac_power_drift_flag', False)}
- Light power average: {safe_float(drift_data.get('light_power_avg_w5m'))}W
- Light power drift detected: {drift_data.get('light_power_drift_flag', False)}

ANOMALY FLAGS:
- AC rapid cycling: {deviation_flags.get('ac_rapid_cycle_flag', False)}
- Light stuck: {deviation_flags.get('light_stuck_flag', False)}
- Occupancy sensor stuck: {deviation_flags.get('occupancy_sensor_stuck_flag', False)}

SEASONAL PATTERNS:
- AC vs weekly pattern: {safe_float(seasonal.get('ac_vs_weekly_pct'))}% deviation
- Light vs weekly pattern: {safe_float(seasonal.get('light_vs_weekly_pct'))}% deviation

INTERVAL STATISTICS (p50/p90/p99):
- Door open intervals: {safe_int(intervals.get('door_open_intervals_s', {}).get('p50'))}/{safe_int(intervals.get('door_open_intervals_s', {}).get('p90'))}/{safe_int(intervals.get('door_open_intervals_s', {}).get('p99'))} seconds
- Occupancy intervals: {safe_int(intervals.get('occupancy_intervals_s', {}).get('p50'))}/{safe_int(intervals.get('occupancy_intervals_s', {}).get('p90'))}/{safe_int(intervals.get('occupancy_intervals_s', {}).get('p99'))} seconds

SENSOR HEALTH:
"""
    
    if health_events:
        for event in health_events:
            summary += f"- {event.get('sensor', UNKNOWN_VALUE)} sensor: {event.get('event', UNKNOWN_VALUE)} at {event.get('timestamp', UNKNOWN_VALUE)}\n"
    else:
        summary += f"- {NO_HEALTH_ISSUES}\n"
    
    summary += "\nMANUAL OVERRIDES:\n"
    if manual_overrides:
        for override in manual_overrides:
            summary += f"- {override.get('device', UNKNOWN_VALUE)}: {safe_int(override.get('count'))} overrides\n"
    else:
        summary += f"- {NO_MANUAL_OVERRIDES}\n"
    
    return summary

def get_analysis_prompt(preprocessed_data: str, report_type: str) -> str:
    """
    Get the appropriate analysis prompt based on report type.
    
    Args:
        preprocessed_data: Formatted sensor data summary
        report_type: Type of report ('customer' or 'internal')
        
    Returns:
        Formatted prompt string
    """
    if report_type == "internal":
        return f"""You are an expert IoT data analyst specializing in smart building systems and energy efficiency.  
            Analyze the following sensor data from a smart room and generate a comprehensive, accurate, and actionable report.

            SENSOR DATA:
            {preprocessed_data}

            Please provide a detailed analysis covering these areas. Always follow this exact structure, in order, and do not omit any section:

            ## ENERGY INEFFICIENCIES
            Identify quantifiable energy waste patterns using only the provided data:
            - Devices running when the room is unoccupied (e.g., AC or lights on during 0% occupancy)
            - HVAC operation conflicting with environmental conditions (e.g., AC on while windows are open)
            - Excessive device cycling (based on on/off counts)
            - Power consumption anomalies (using rolling power averages and drift flags)

            For any energy waste calculation:
            - Use the formula: (device_energy_kWh) × (inefficiency_percentage)
            - Show all steps explicitly (e.g., 8.1 kWh × 0.05 = 0.405 kWh of wasted energy)
            - Reference the exact JSON field used (e.g., "cross_sensor_ratios.ac_on_while_unoccupied_pct")

            ## ANOMALIES & SYSTEM ISSUES
            Detect potential system problems using anomaly flags and statistical metrics:
            - Sensor malfunctions or stuck readings (check pattern_deviation_flags and sensor_health_events)
            - Rapid cycling of devices (review pattern_deviation_flags and on/off counts)
            - Power drift issues (use rolling_statistics_and_drift flags and values)
            - Unusual runtime patterns (compare avg_min across devices using run_length_statistics)
            - Statistical outliers in timing data (analyze inter_event_distributions p99 values)

            Always distinguish between flagged anomalies (boolean flags) and inferred patterns from counts or durations.

            ## BEHAVIORAL PATTERNS
            Analyze human and operational behaviors based strictly on available metrics:
            - Peak usage times and patterns → If no time-stamped data is present, state: "Not available"
            - Occupancy correlation with device usage → Compare occupancy_detected_pct with light_on_pct, ac_on_pct; avoid assuming causality
            - Door/window opening patterns → Use open_count and window_open_pct; comment on frequency and duration via p50/p90/p99
            - Seasonal or weekly deviations → Use seasonal_pattern_summaries fields only; report values directly

            Never infer granular timing (e.g., "morning peak") without explicit time-series data.

            ## CROSS-SENSOR CORRELATIONS
            Examine relationships between sensor types:
            - Occupancy vs device activation → Compare occupancy_detected_pct with light_on_pct, ac_on_pct; reference cross_sensor_ratios if available
            - Environmental factors (window/door) vs HVAC usage → Use window_open_pct and ac_on_while_window_open_pct
            - Manual override patterns → Report manual_override_counters exactly as listed; if empty, state "None recorded"

            Only discuss correlations supported by direct metrics. Do not assume synchronization or causation.

            ## STATISTICAL INSIGHTS
            Interpret the statistical distributions and variance data:
            - P99 interval outliers and their implications → Convert seconds to minutes/hours; explain what long intervals may indicate
            - Variance patterns in runtime data → Use variance_min2 or variance_s2 to compute standard deviation where possible (e.g., √150.2 ≈ 12.26 min); compare across devices
            - Event frequency analysis → Use on/off counts and open/close events to assess traffic or usage intensity

            If variance or interval data is missing, state "Not available".

            ## ACTIONABLE RECOMMENDATIONS
            Provide specific, implementable suggestions based solely on observed data:
            - Energy savings opportunities → Focus on reducing waste identified in ENERGY INEFFICIENCIES
            - System optimization → Suggest investigations into high cycle counts, long runtimes, or outlier intervals
            - Maintenance needs → Only if sensor_health_events contain entries or anomaly flags are true
            - Automation improvements → Recommend rules like occupancy-based shutoffs or window-linked AC disable
            - User behavior modifications → Suggest feedback or training if manual overrides or avoidable waste are present

            Ensure every recommendation ties directly to a finding in prior sections.

            ### STRICT RULES FOR ANALYSIS
            - Use only the fields present in the provided JSON. Do not invent, assume, or hallucinate data.
            - If a required metric for a section is missing, state clearly: **"Not available"**.
            - All numerical insights must reference the exact field name and value from the JSON (e.g., "light_on_pct = 0.35").
            - When calculating wasted energy or other derived values, show the full formula and arithmetic (e.g., 1.5 kWh × 0.02 = 0.03 kWh).
            - Never claim temporal patterns (e.g., "usage peaks in the afternoon") unless timestamps are provided — they are not in this dataset.
            - Maintain a professional, analytical tone. Avoid speculation or vague language.
            - Format the response using clear section headers (##), bullet points, and concise explanations.
            - Preserve all section titles exactly as defined above — do not merge, rename, or skip any.

            Now generate the report accordingly.
        """
    elif report_type == "customer":  # customer report
        return f"""You are a smart building efficiency advisor. Your role is to turn sensor data into a clear, engaging report for facility managers, property owners, and building occupants.

                    Focus on outcomes that matter to people: **comfort, savings, convenience, and sustainability**. Use simple, professional, and friendly language—like explaining over coffee. Avoid jargon, formulas, acronyms, and technical units unless absolutely necessary. Translate numbers into real-world terms (e.g., cost or relatable usage).

                    **SENSOR DATA:**
                    {preprocessed_data}

                    Follow this structure exactly, in order:

                    ## 🚀 EXECUTIVE SUMMARY
                    - A 1-2 sentence summary of the room's overall performance.
                    - The top 1-2 actionable recommendations and their estimated impact (e.g., "Automating cooling could save an estimated AED 200 annually.").
                    If data is missing or analysis is not possible: **"Not available"**.


                    ## 🌡️ HOW IS THE ROOM PERFORMING?
                    - Quick snapshot of comfort (temperature, air quality).
                    - Was energy used wisely when empty?
                    - Any open windows/doors while cooling ran?
                    - Any red flags or positive trends?
                    - **Optional Benchmark:** How does this room's performance compare to the building average or similar rooms? (e.g., "This room uses 15% less energy during unoccupied hours than the building average.")
                    If data is missing: **"Not available"**.


                    ## 💡 ENERGY WASTE: WHERE IS POWER BEING WASTED?
                    - Lights, cooling, or devices left on in empty room?
                    - Cooling running while windows open?
                    - Devices turning on/off too often?
                    For each issue:
                    - Explain in plain terms (e.g., "Cooling ran 3 hours after room was empty").
                    - Estimate likely cost/impact (e.g., "May add AED 45 to the bill").
                    - Reference data clearly (e.g., "Lights on 60% of time, room occupied 35%").

                    ## ⚠️ ANY WARNING SIGNS IN THE SYSTEM?
                    - Unusual sensor/device behavior (stuck values, frequent cycling).
                    - Equipment running much longer than expected.
                    - Possible comfort or cost risks.
                    - **Specific instructions:** Note any contradictory or "stale" sensor data (e.g., a temperature sensor that hasn't changed its reading in 24 hours).
                    If none: **"No system issues detected."**

                    ## 🕒 HOW IS THE ROOM BEING USED?
                    - Occupancy frequency and patterns.
                    - Do lights/cooling respond to entry/exit?
                    - Are windows/doors opened often?
                    - Any daily/weekly usage trends?
                    If patterns can't be determined: **"Usage timing patterns cannot be determined from this data."**

                    ## 🔗 WHAT'S HAPPENING WITH SENSORS AND CONTROLS?
                    - Do systems work in sync (lights/cooling with occupancy, AC with windows)?
                    - Are manual overrides frequent?
                    - Note if automation could improve.

                    ## ✅ WHAT CAN BE IMPROVED? (ACTIONABLE NEXT STEPS)
                    Provide 3–5 specific, practical recommendations tied directly to the data. For each suggestion, clearly state the anticipated outcome (e.g., "This will save an estimated AED 45 per month" or "This will improve occupant comfort by reducing temperature swings."). Examples:
                    - Automate cooling to turn off when empty.
                    - Check/replace a faulty sensor.
                    - Encourage staff to close windows before using AC.
                    - Link AC to window sensors for automatic shutoff.
                    Keep suggestions positive and solution-focused.

                    ### CUSTOMER REPORT RULES
                    - Use the exact headers above.
                    - Never invent data; use **"Not available"** if missing.
                    - Keep paragraphs short; use bullet points for actions.
                    - Use emojis only in section headers.
                    - Always stay clear, professional, and customer-friendly.
                    - **Audience Note:** For reports intended for occupants, focus more on comfort and convenience, and less on detailed cost analysis unless it's a direct action they can take.

                    """
    else:
        return "Invalid report type"

def initialize_gemini_model(api_key: str, model_name: str):
    """Initialize the Gemini AI model"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized Gemini model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return None

def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load and parse JSON file from disk.
    
    Returns:
        Parsed JSON data as dictionary, or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            logger.info(f"Successfully loaded JSON file: {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return None

def analyze_with_gemini(model, preprocessed_data: str, report_type: str, max_retries: int) -> Optional[str]:
    """
    Send preprocessed sensor data to Gemini API for analysis with retry logic.
    
    Args:
        model: Gemini model instance
        preprocessed_data: Formatted sensor data summary
        report_type: Type of report to generate
        max_retries: Maximum retry attempts
        
    Returns:
        Gemini's analysis response, or None if error
    """
    if not model:
        logger.error("Gemini model not initialized")
        return None
    
    prompt = get_analysis_prompt(preprocessed_data, report_type)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending data to Gemini API (attempt {attempt + 1}/{max_retries})")
            response = model.generate_content(prompt)
            
            if response and response.text:
                logger.info("Successfully received analysis from Gemini API")
                return response.text
            else:
                logger.warning("Empty response from Gemini API")
                
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Final attempt failed: {e}")
                return None
            
            delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    return None

def get_output_file_path(day: str, report_type: str, output_dir: Path) -> Path:
    """
    Generate output file path based on report type and day.
    
    Args:
        day: Day identifier
        report_type: Type of report ('customer' or 'internal')
        output_dir: Base output directory
        
    Returns:
        Path object for output file
    """
    if report_type == "customer":
        folder_name = "Reports-Customer"
        filename = f"{day}_customer.txt"
    elif report_type == "internal":
        folder_name = "Reports-Internal"
        filename = f"{day}_internal.txt"
    else:
        # Fallback
        folder_name = "Reports"
        filename = f"{day}_{report_type}.txt"
    
    return output_dir / folder_name / filename

def save_analysis_to_file(analysis: str, preprocessed_data: str, output_path: Path, json_file_path: Path, report_type: str) -> bool:
    """
    Save the analysis results and preprocessed data to a text file.
    
    Args:
        analysis: Gemini's analysis response
        preprocessed_data: The preprocessed sensor data
        output_path: Path where to save the file
        json_file_path: Source JSON file path
        report_type: Type of report
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(REPORT_SEPARATOR + "\n")
            file.write("IoT SENSOR DATA ANALYSIS REPORT\n")
            file.write(REPORT_SEPARATOR + "\n")
            file.write(f"Generated on: {timestamp}\n")
            file.write(f"Source file: {json_file_path}\n")
            file.write(f"Report type: {report_type}\n")
            file.write(REPORT_SEPARATOR + "\n\n")
            
            file.write("PREPROCESSED SENSOR DATA:\n")
            file.write(SECTION_SEPARATOR + "\n")
            file.write(preprocessed_data)
            file.write("\n\n")
            
            file.write("GEMINI AI ANALYSIS:\n")
            file.write(SECTION_SEPARATOR + "\n")
            file.write(analysis)
            file.write("\n\n")
            
            file.write(REPORT_SEPARATOR + "\n")
            file.write("End of Report\n")
            file.write(REPORT_SEPARATOR + "\n")
        
        logger.info(f"Analysis results saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving analysis to file: {e}")
        return False

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze IoT sensor data using Google Gemini API'
    )
    parser.add_argument(
        '--day', 
        default=DEFAULT_DAY,
        help=f'Day identifier for the JSON file (default: {DEFAULT_DAY})'
    )
    parser.add_argument(
        '--report-type',
        choices=['customer', 'internal'],
        default=DEFAULT_REPORT_TYPE,
        help=f'Type of report to generate (default: {DEFAULT_REPORT_TYPE})'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('.'),
        help='Directory containing input JSON files (default: current directory)'
    )
    parser.add_argument(
        '--output-dir', 
        type=Path,
        default=Path('.'),
        help='Base directory for output folders (default: current directory)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f'Maximum retry attempts for API calls (default: {DEFAULT_MAX_RETRIES})'
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'Gemini model to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=DEFAULT_LOG_LEVEL,
        help=f'Set logging level (default: {DEFAULT_LOG_LEVEL})'
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function to orchestrate the IoT data analysis workflow"""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print(REPORT_SEPARATOR)
    print("SIMPLIFIED IoT SENSOR DATA ANALYSIS WITH GOOGLE GEMINI")
    print(REPORT_SEPARATOR)
    print()
    
    # Validate API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Set up file paths
    json_file_path = args.input_dir / f"{args.day}{JSON_FILE_SUFFIX}"
    output_file_path = get_output_file_path(args.day, args.report_type, args.output_dir)
    
    try:
        # Step 1: Initialize model
        logger.info("Step 1: Initializing Gemini model...")
        model = initialize_gemini_model(api_key, args.model)
        if not model:
            sys.exit(1)
        
        # Step 2: Load JSON file
        logger.info("Step 2: Loading JSON file...")
        sensor_data = load_json_file(json_file_path)
        if sensor_data is None:
            sys.exit(1)
        
        # Step 3: Validate structure
        logger.info("Step 3: Validating JSON structure...")
        if not validate_json_structure(sensor_data):
            sys.exit(1)
        
        # Step 4: Preprocess data
        logger.info("Step 4: Preprocessing sensor data...")
        preprocessed_data = preprocess_sensor_data(sensor_data)
        logger.info("Data preprocessing completed")
        
        # Step 5: Analyze with Gemini
        logger.info("Step 5: Analyzing with Gemini API...")
        analysis = analyze_with_gemini(model, preprocessed_data, args.report_type, args.max_retries)
        if analysis is None:
            sys.exit(1)
        
        # Step 6: Display results
        print("\n" + REPORT_SEPARATOR)
        print("GEMINI ANALYSIS RESULTS")
        print(REPORT_SEPARATOR)
        print()
        print(analysis)
        print()
        print(REPORT_SEPARATOR)
        
        # Step 7: Save results to file
        logger.info("Step 6: Saving analysis results...")
        save_success = save_analysis_to_file(analysis, preprocessed_data, output_file_path, json_file_path, args.report_type)
        
        if save_success:
            logger.info("Analysis completed successfully!")
            print(REPORT_SEPARATOR)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"Report saved to: {output_file_path}")
            print(REPORT_SEPARATOR)
            sys.exit(0)
        else:
            logger.warning("Analysis completed but could not save results to file")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Analysis workflow failed: {e}")
        print(REPORT_SEPARATOR)
        print("ANALYSIS FAILED - CHECK LOGS FOR DETAILS")
        print(REPORT_SEPARATOR)
        sys.exit(1)

if __name__ == "__main__":
    main()