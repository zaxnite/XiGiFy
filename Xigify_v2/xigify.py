#!/usr/bin/env python3
"""
Deterministic AI Interpretation Engine for IoT Sensor Data
Generates consistent, minimal-variation interpretations using Google Gemini API
"""

import json
import logging
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai


# Load environment variables
load_dotenv()

# Configuration constants
DEFAULT_DAY = "day_5"
DEFAULT_REPORT_TYPE = "customer"
DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2
DEFAULT_LOG_LEVEL = "INFO"
JSON_FILE_SUFFIX = "_preprocessed.json"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Report separators
REPORT_SEPARATOR = "=" * 80
SECTION_SEPARATOR = "-" * 60


def generate_inference_guidelines(data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Generate deterministic inference guidelines for the model to create rules.
    Provides clear logic for rule generation while allowing text variation.
    
    Args:
        data: Extracted interpretation data
        
    Returns:
        Tuple of (energy_inference_guidelines, comfort_inference_guidelines)
    """
    
    # Extract key metrics for inference guidelines
    efficiency_score = data.get('efficiency_score', 0.0) * 100  # Convert to percentage
    comfort_score = data.get('comfort_score', 0.0) * 100
    device_ratios = data.get('device_ratios', {})
    ac_ratio = device_ratios.get('ac', 0.0) * 100
    light_ratio = device_ratios.get('light', 0.0) * 100
    fan_ratio = device_ratios.get('fan', 0.0) * 100
    occupancy_light_alignment = data.get('occupancy_light_alignment', 0.0)
    ac_runtime_stability = data.get('ac_runtime_stability', 0.0)
    manual_overrides = data.get('manual_overrides', 0)
    
    # ENERGY EFFICIENCY INFERENCE GUIDELINES
    energy_guidelines = []
    
    # Guideline 1: AC Optimization Logic
    if efficiency_score < 60:
        if ac_ratio > 70:
            energy_guidelines.append(f"INFERENCE: AC usage at {ac_ratio:.1f}% is excessive for low efficiency ({efficiency_score:.1f}%). Generate HIGH priority rule for immediate AC runtime reduction through scheduling and occupancy controls.")
        elif ac_ratio > 50:
            energy_guidelines.append(f"INFERENCE: AC usage at {ac_ratio:.1f}% with low efficiency ({efficiency_score:.1f}%). Generate HIGH priority rule for AC optimization with smart controls.")
        else:
            energy_guidelines.append(f"INFERENCE: Low efficiency ({efficiency_score:.1f}%) despite moderate AC usage ({ac_ratio:.1f}%). Generate HIGH priority rule for comprehensive energy management.")
    elif efficiency_score < 80:
        if ac_ratio > 60:
            energy_guidelines.append(f"INFERENCE: AC usage at {ac_ratio:.1f}% with moderate efficiency ({efficiency_score:.1f}%). Generate MEDIUM priority rule for AC scheduling optimization.")
        elif ac_ratio > 40:
            energy_guidelines.append(f"INFERENCE: Moderate AC usage ({ac_ratio:.1f}%) and efficiency ({efficiency_score:.1f}%). Generate MEDIUM priority rule for AC efficiency improvements.")
        else:
            energy_guidelines.append(f"INFERENCE: Good efficiency ({efficiency_score:.1f}%) with low AC usage ({ac_ratio:.1f}%). Generate MEDIUM priority rule for monitoring and preventive measures.")
    else:
        energy_guidelines.append(f"INFERENCE: Excellent efficiency ({efficiency_score:.1f}%) with AC usage at {ac_ratio:.1f}%. Generate LOW priority rule for maintaining current performance.")
    
    # Guideline 2: Lighting Optimization Logic  
    if occupancy_light_alignment < 60:
        if light_ratio > 50:
            energy_guidelines.append(f"INFERENCE: High lighting usage ({light_ratio:.1f}%) with poor alignment ({occupancy_light_alignment:.1f}%). Generate HIGH priority rule for occupancy sensor installation.")
        else:
            energy_guidelines.append(f"INFERENCE: Poor occupancy alignment ({occupancy_light_alignment:.1f}%) despite moderate lighting usage ({light_ratio:.1f}%). Generate MEDIUM priority rule for lighting automation.")
    elif occupancy_light_alignment < 80:
        energy_guidelines.append(f"INFERENCE: Moderate occupancy alignment ({occupancy_light_alignment:.1f}%) needs improvement. Generate MEDIUM priority rule to exceed 80% alignment.")
    else:
        energy_guidelines.append(f"INFERENCE: Excellent occupancy alignment ({occupancy_light_alignment:.1f}%). Generate LOW priority rule for maintaining performance.")
    
    # Guideline 3: System Optimization Logic
    if efficiency_score < 60:
        if fan_ratio > 30:
            energy_guidelines.append(f"INFERENCE: Low efficiency ({efficiency_score:.1f}%) with notable fan usage ({fan_ratio:.1f}%). Generate MEDIUM priority rule for fan strategy review.")
        else:
            energy_guidelines.append(f"INFERENCE: Very low efficiency ({efficiency_score:.1f}%) requires systematic approach. Generate HIGH priority rule for holistic energy management.")
    elif efficiency_score < 80:
        if fan_ratio > 40:
            energy_guidelines.append(f"INFERENCE: Moderate efficiency ({efficiency_score:.1f}%) with high fan usage ({fan_ratio:.1f}%). Generate MEDIUM priority rule for fan optimization.")
        else:
            energy_guidelines.append(f"INFERENCE: Moderate efficiency ({efficiency_score:.1f}%) baseline established. Generate LOW priority rule for periodic reviews.")
    else:
        energy_guidelines.append(f"INFERENCE: Excellent efficiency ({efficiency_score:.1f}%). Generate LOW priority rule for continuing current practices with quarterly reviews.")
    
    # COMFORT & WELLBEING INFERENCE GUIDELINES
    comfort_guidelines = []
    
    # Guideline 1: Temperature Control Logic
    if comfort_score < 60:
        if ac_runtime_stability < 0.7:
            comfort_guidelines.append(f"INFERENCE: Poor comfort ({comfort_score:.1f}%) with unstable AC ({ac_runtime_stability:.1f}). Generate HIGH priority rule for temperature control stabilization.")
        else:
            comfort_guidelines.append(f"INFERENCE: Poor comfort ({comfort_score:.1f}%) despite stable AC ({ac_runtime_stability:.1f}). Generate HIGH priority rule for comfort issue investigation.")
    elif comfort_score < 80:
        if ac_runtime_stability < 0.8:
            comfort_guidelines.append(f"INFERENCE: Moderate comfort ({comfort_score:.1f}%) with suboptimal AC stability ({ac_runtime_stability:.1f}). Generate MEDIUM priority rule for AC fine-tuning.")
        else:
            comfort_guidelines.append(f"INFERENCE: Moderate comfort ({comfort_score:.1f}%) with stable AC ({ac_runtime_stability:.1f}). Generate MEDIUM priority rule for minor comfort improvements.")
    else:
        comfort_guidelines.append(f"INFERENCE: Excellent comfort ({comfort_score:.1f}%) with stable AC ({ac_runtime_stability:.1f}). Generate LOW priority rule for maintaining performance.")
    
    # Guideline 2: Occupancy Response Logic
    if manual_overrides > 10:
        if occupancy_light_alignment < 70:
            comfort_guidelines.append(f"INFERENCE: High manual overrides ({manual_overrides}) with poor occupancy detection ({occupancy_light_alignment:.1f}%). Generate HIGH priority rule for override reduction through improved detection.")
        else:
            comfort_guidelines.append(f"INFERENCE: High manual overrides ({manual_overrides}) despite good alignment. Generate MEDIUM priority rule for automation enhancement and user training.")
    elif manual_overrides > 5:
        comfort_guidelines.append(f"INFERENCE: Moderate manual overrides ({manual_overrides}) indicate automation gaps. Generate MEDIUM priority rule for monitoring and preventive improvements.")
    elif occupancy_light_alignment < 75:
        comfort_guidelines.append(f"INFERENCE: Low overrides but suboptimal alignment ({occupancy_light_alignment:.1f}%). Generate MEDIUM priority rule to exceed 75% alignment.")
    else:
        comfort_guidelines.append(f"INFERENCE: Excellent occupancy response ({occupancy_light_alignment:.1f}% alignment, {manual_overrides} overrides). Generate LOW priority rule for maintaining performance.")
    
    # Guideline 3: System Maintenance Logic
    if comfort_score < 60 or efficiency_score < 60:
        comfort_guidelines.append(f"INFERENCE: Low performance scores (comfort: {comfort_score:.1f}%, efficiency: {efficiency_score:.1f}%). Generate HIGH priority rule for immediate system calibration.")
    elif comfort_score < 80:
        comfort_guidelines.append(f"INFERENCE: Moderate comfort score ({comfort_score:.1f}%). Generate MEDIUM priority rule for quarterly assessments and minor adjustments.")
    else:
        comfort_guidelines.append(f"INFERENCE: Excellent comfort performance ({comfort_score:.1f}%). Generate LOW priority rule for bi-annual monitoring.")
    
    # Format guidelines for model consumption
    energy_inference_text = "\n".join([f"ENERGY_RULE_{i+1}_LOGIC: {guideline}" for i, guideline in enumerate(energy_guidelines[:3])])
    comfort_inference_text = "\n".join([f"COMFORT_RULE_{i+1}_LOGIC: {guideline}" for i, guideline in enumerate(comfort_guidelines[:3])])
    
    return energy_inference_text, comfort_inference_text


def get_deterministic_prompt_template(report_type: str) -> str:
    """
    Get deterministic prompt template based on report type.
    
    Args:
        report_type: 'customer' or 'internal'
        
    Returns:
        Structured prompt template with minimal variation points
    """
    
    base_structure = """ANALYZE THIS BUILDING SENSOR DATA DETERMINISTICALLY.
USE ONLY THE PROVIDED METRICS. BE PRECISE AND CONSISTENT.

ROOM_ID: {room_id}
ANALYSIS_PERIOD: {analysis_period}
DATA_SUMMARY: {data_summary}

SECTION A - ENERGY EFFICIENCY INTERPRETATION:
Efficiency Score: {efficiency_score}
Total Energy Consumption: {total_energy_kwh} kWh
Energy Waste Indicators: {waste_indicators}
Device Runtime Ratios: {device_ratios}

INTERPRET: Based ONLY on these efficiency metrics, provide exactly 2 sentences describing:
1. Overall energy performance level (excellent/good/fair/poor)
2. Primary waste source if efficiency_score < 0.7

SECTION B - COMFORT & WELLBEING INTERPRETATION:
Comfort Score: {comfort_score}
Occupancy-Light Alignment: {occupancy_light_alignment}%
AC Runtime Stability: {ac_runtime_stability}
Temperature Variation: {temp_variation}

INTERPRET: Based ONLY on these comfort metrics, provide exactly 2 sentences describing:
1. Overall comfort level (optimal/adequate/suboptimal/poor)
2. Primary comfort issue if comfort_score < 0.7

SECTION C - PREDICTIVE & PRESCRIPTIVE INSIGHTS:
Drift Detection Flags: {drift_flags}
Pattern Trends: {pattern_trends}
Manual Override Count: {manual_overrides}
System Health Events: {system_health_events}

INTERPRET: Based ONLY on these predictive metrics, provide exactly 2 sentences describing:
1. System stability status (stable/minor-issues/significant-issues)
2. Recommended action priority (none/low/medium/high)

SECTION D - DETERMINISTIC ENERGY EFFICIENCY RULES:
Based on the following inference logic, GENERATE exactly 3 actionable rules with natural language variation but consistent priorities and logic:

{energy_inference_guidelines}

Requirements:
- Follow the priority levels (HIGH/MEDIUM/LOW) indicated in each inference
- Include specific metrics mentioned in the inference logic  
- Use natural language but maintain consistent recommendations for same data patterns
- Each rule should be actionable and specific to the inferred conditions

SECTION E - DETERMINISTIC COMFORT & WELLBEING RULES:
Based on the following inference logic, GENERATE exactly 3 actionable rules with natural language variation but consistent priorities and logic:

{comfort_inference_guidelines}

Requirements:
- Follow the priority levels (HIGH/MEDIUM/LOW) indicated in each inference
- Include specific metrics mentioned in the inference logic
- Use natural language but maintain consistent recommendations for same data patterns  
- Each rule should be actionable and specific to the inferred conditions"""

    if report_type == "customer":
        return base_structure + """

OUTPUT FORMAT REQUIREMENTS:
- Use simple, non-technical language
- Focus on outcomes (comfort, savings, convenience)
- Avoid technical jargon and sensor names
- Structure as: ENERGY_EFFICIENCY: [2 sentences] | COMFORT_WELLBEING: [2 sentences] | PREDICTIVE_INSIGHTS: [2 sentences] | ENERGY_RULES: [3 rules] | COMFORT_RULES: [3 rules]
- Be deterministic - same input should produce identical output
- Temperature=0, deterministic=true"""
    
    else:  # internal
        return base_structure + """

OUTPUT FORMAT REQUIREMENTS:
- Use technical precision and specific metrics
- Include quantitative references where available
- Reference specific sensor types and thresholds
- Structure as: ENERGY_EFFICIENCY: [2 sentences] | COMFORT_WELLBEING: [2 sentences] | PREDICTIVE_INSIGHTS: [2 sentences] | ENERGY_RULES: [3 rules] | COMFORT_RULES: [3 rules]
- Be deterministic - same input should produce identical output
- Temperature=0, deterministic=true"""


def extract_interpretation_data(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract specific data points needed for deterministic interpretation.
    
    Args:
        sensor_data: Processed JSON data
        
    Returns:
        Extracted data dictionary
    """
    try:
        # Extract basic info
        room_id = sensor_data.get("room_id", "Unknown")
        summary_window = sensor_data.get("summary_window", {})
        analysis_period = f"{summary_window.get('start', 'Unknown')} to {summary_window.get('end', 'Unknown')}"
        
        # Extract energy efficiency data
        energy_section = sensor_data.get("energy_efficiency_analysis", {})
        efficiency_score = energy_section.get("efficiency_score", 0.0)
        total_energy = energy_section.get("total_energy_kwh", 0.0)
        waste_indicators = energy_section.get("waste_indicators", {})
        
        # Extract comfort data
        comfort_section = sensor_data.get("comfort_wellbeing_analysis", {})
        comfort_score = comfort_section.get("comfort_score", 0.0)
        alignment_metrics = comfort_section.get("alignment_metrics", {})
        occupancy_light_alignment = alignment_metrics.get("occupancy_light_ratio", 0.0) * 100
        
        # Extract ML abstracted metrics for AC stability approximation
        ml_metrics = sensor_data.get("ml_abstracted_metrics", {})
        runtime_stats = ml_metrics.get("run_length_statistics", {})
        ac_stats = runtime_stats.get("ac", {})
        ac_variance = ac_stats.get("variance_min2", 0.0)
        ac_stability = max(0, 1.0 - (ac_variance / 1000))  # Normalize variance to stability score
        
        # Temperature variation - use occupancy variance as proxy for environmental stability
        occupancy_stats = runtime_stats.get("occupancy", {})
        temp_variation = min(1.0, occupancy_stats.get("variance_min2", 0.0) / 500)  # Normalize to 0-1
        
        # Extract predictive data
        predictive_section = sensor_data.get("predictive_prescriptive_analysis", {})
        drift_flags = predictive_section.get("predictive_flags", {})
        
        # Pattern trends from rolling statistics
        rolling_stats = ml_metrics.get("rolling_statistics_and_drift", {})
        pattern_trends = {
            "ac_power_drift": rolling_stats.get("ac_power_drift_flag", False),
            "light_power_drift": rolling_stats.get("light_power_drift_flag", False)
        }
        
        # Manual overrides
        override_breakdown = predictive_section.get("override_breakdown", {})
        manual_overrides = sum(override_breakdown.values()) if isinstance(override_breakdown, dict) else 0
        
        # System health
        sensor_health = sensor_data.get("sensor_health_analysis", {})
        system_health_events = sensor_health.get("total_health_events", 0)
        
        # Calculate device ratios from usage features
        usage_features = ml_metrics.get("usage_features", {})
        device_ratios = {
            "light": usage_features.get("light_on_pct", 0.0),
            "ac": usage_features.get("ac_on_pct", 0.0),
            "fan": usage_features.get("fan_on_pct", 0.0),
            "occupancy": usage_features.get("occupancy_detected_pct", 0.0)
        }
        
        # Create summary from aggregated events
        aggregated_events = sensor_data.get("aggregated_events", {})
        total_events = sum(device.get("total_cycles", 0) for device in aggregated_events.values())
        data_summary = f"Room: {room_id}, Total Device Cycles: {total_events}, Energy: {total_energy}kWh"
        
        return {
            "room_id": room_id,
            "analysis_period": analysis_period,
            "data_summary": data_summary,
            "efficiency_score": round(efficiency_score, 3),
            "total_energy_kwh": round(total_energy, 2),
            "waste_indicators": waste_indicators,
            "device_ratios": device_ratios,
            "comfort_score": round(comfort_score, 3),
            "occupancy_light_alignment": round(occupancy_light_alignment, 1),
            "ac_runtime_stability": round(ac_stability, 3),
            "temp_variation": round(temp_variation, 3),
            "drift_flags": drift_flags,
            "pattern_trends": pattern_trends,
            "manual_overrides": manual_overrides,
            "system_health_events": system_health_events
        }
        
    except Exception as e:
        logger.error(f"Error extracting interpretation data: {e}")
        return {}


def create_deterministic_prompt(data: Dict[str, Any], report_type: str) -> str:
    """
    Create deterministic prompt from extracted data.
    
    Args:
        data: Extracted interpretation data
        report_type: 'customer' or 'internal'
        
    Returns:
        Formatted prompt string
    """
    # Generate deterministic inference guidelines for the model
    energy_inference_guidelines, comfort_inference_guidelines = generate_inference_guidelines(data)
    
    # Add inference guidelines to data for template formatting
    data_with_guidelines = data.copy()
    data_with_guidelines['energy_inference_guidelines'] = energy_inference_guidelines
    data_with_guidelines['comfort_inference_guidelines'] = comfort_inference_guidelines
    
    template = get_deterministic_prompt_template(report_type)
    
    try:
        return template.format(**data_with_guidelines)
    except KeyError as e:
        logger.error(f"Missing data field for prompt: {e}")
        return ""


def initialize_gemini_model(api_key: str, model_name: str):
    """Initialize the Gemini AI model with deterministic settings"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                max_output_tokens=500,
                candidate_count=1
            )
        )
        logger.info(f"Initialized deterministic Gemini model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return None


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse JSON file from disk."""
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


def interpret_with_gemini(model, prompt: str, max_retries: int) -> Optional[str]:
    """
    Send deterministic prompt to Gemini API for interpretation.
    
    Args:
        model: Gemini model instance
        prompt: Deterministic prompt string
        max_retries: Maximum retry attempts
        
    Returns:
        Gemini's interpretation response, or None if error
    """
    if not model:
        logger.error("Gemini model not initialized")
        return None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending deterministic prompt to Gemini API (attempt {attempt + 1}/{max_retries})")
            response = model.generate_content(prompt)
            
            if response and response.text:
                logger.info("Successfully received interpretation from Gemini API")
                return response.text.strip()
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


def parse_interpretation_response(response: str) -> Dict[str, str]:
    """
    Parse structured interpretation response into sections including rules.
    
    Args:
        response: Raw Gemini response
        
    Returns:
        Dictionary with parsed sections including energy and comfort rules
    """
    sections = {
        "energy_efficiency": "",
        "comfort_wellbeing": "",
        "predictive_insights": "",
        "energy_rules": "",
        "comfort_rules": ""
    }
    
    try:
        # Look for the structured format with rules
        expected_sections = ["ENERGY_EFFICIENCY:", "COMFORT_WELLBEING:", "PREDICTIVE_INSIGHTS:", 
                           "ENERGY_RULES:", "COMFORT_RULES:"]
        
        if all(section in response for section in expected_sections):
            parts = response.split("|")
            
            for part in parts:
                part = part.strip()
                if part.startswith("ENERGY_EFFICIENCY:"):
                    sections["energy_efficiency"] = part.replace("ENERGY_EFFICIENCY:", "").strip()
                elif part.startswith("COMFORT_WELLBEING:"):
                    sections["comfort_wellbeing"] = part.replace("COMFORT_WELLBEING:", "").strip()
                elif part.startswith("PREDICTIVE_INSIGHTS:"):
                    sections["predictive_insights"] = part.replace("PREDICTIVE_INSIGHTS:", "").strip()
                elif part.startswith("ENERGY_RULES:"):
                    rules_text = part.replace("ENERGY_RULES:", "").strip()
                    # Format rules with line breaks - split on rule patterns and clean up
                    formatted_rules = format_rules_for_display(rules_text, "ENERGY_RULE_")
                    sections["energy_rules"] = formatted_rules
                elif part.startswith("COMFORT_RULES:"):
                    rules_text = part.replace("COMFORT_RULES:", "").strip()
                    # Format rules with line breaks - split on rule patterns and clean up
                    formatted_rules = format_rules_for_display(rules_text, "COMFORT_RULE_")
                    sections["comfort_rules"] = formatted_rules
        else:
            # Fallback: try to extract individual sections from response
            lines = response.split('\n')
            current_section = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("ENERGY_EFFICIENCY:"):
                    if current_section:
                        if current_section in ["energy_rules", "comfort_rules"]:
                            rule_prefix = "ENERGY_RULE_" if current_section == "energy_rules" else "COMFORT_RULE_"
                            sections[current_section] = format_rules_for_display(' '.join(current_content), rule_prefix)
                        else:
                            sections[current_section] = ' '.join(current_content)
                    current_section = "energy_efficiency"
                    current_content = [line.replace("ENERGY_EFFICIENCY:", "").strip()]
                elif line.startswith("COMFORT_WELLBEING:"):
                    if current_section:
                        if current_section in ["energy_rules", "comfort_rules"]:
                            rule_prefix = "ENERGY_RULE_" if current_section == "energy_rules" else "COMFORT_RULE_"
                            sections[current_section] = format_rules_for_display(' '.join(current_content), rule_prefix)
                        else:
                            sections[current_section] = ' '.join(current_content)
                    current_section = "comfort_wellbeing"
                    current_content = [line.replace("COMFORT_WELLBEING:", "").strip()]
                elif line.startswith("PREDICTIVE_INSIGHTS:"):
                    if current_section:
                        if current_section in ["energy_rules", "comfort_rules"]:
                            rule_prefix = "ENERGY_RULE_" if current_section == "energy_rules" else "COMFORT_RULE_"
                            sections[current_section] = format_rules_for_display(' '.join(current_content), rule_prefix)
                        else:
                            sections[current_section] = ' '.join(current_content)
                    current_section = "predictive_insights"
                    current_content = [line.replace("PREDICTIVE_INSIGHTS:", "").strip()]
                elif line.startswith("ENERGY_RULES:"):
                    if current_section:
                        if current_section in ["energy_rules", "comfort_rules"]:
                            rule_prefix = "ENERGY_RULE_" if current_section == "energy_rules" else "COMFORT_RULE_"
                            sections[current_section] = format_rules_for_display(' '.join(current_content), rule_prefix)
                        else:
                            sections[current_section] = ' '.join(current_content)
                    current_section = "energy_rules"
                    current_content = [line.replace("ENERGY_RULES:", "").strip()]
                elif line.startswith("COMFORT_RULES:"):
                    if current_section:
                        if current_section in ["energy_rules", "comfort_rules"]:
                            rule_prefix = "ENERGY_RULE_" if current_section == "energy_rules" else "COMFORT_RULE_"
                            sections[current_section] = format_rules_for_display(' '.join(current_content), rule_prefix)
                        else:
                            sections[current_section] = ' '.join(current_content)
                    current_section = "comfort_rules"
                    current_content = [line.replace("COMFORT_RULES:", "").strip()]
                elif current_section and line:
                    current_content.append(line)
            
            # Don't forget the last section
            if current_section:
                if current_section in ["energy_rules", "comfort_rules"]:
                    rule_prefix = "ENERGY_RULE_" if current_section == "energy_rules" else "COMFORT_RULE_"
                    sections[current_section] = format_rules_for_display(' '.join(current_content), rule_prefix)
                else:
                    sections[current_section] = ' '.join(current_content)
            
            # If still no structured content, treat as fallback
            if not any(sections.values()):
                sections["energy_efficiency"] = response
                
    except Exception as e:
        logger.warning(f"Error parsing interpretation response: {e}")
        sections["energy_efficiency"] = response
    
    return sections


def format_rules_for_display(rules_text: str, rule_prefix: str) -> str:
    """
    Format rules text to display each rule on a separate line.
    Fixed to avoid splitting on decimal numbers.
    """
    if not rules_text or not rules_text.strip():
        return ""
    
    try:
        import re
        
        rules_text = rules_text.strip()
        
        # Method 1: Split on single digits followed by period at word boundaries
        # This avoids decimal numbers like "37.92%" by ensuring we only split on
        # patterns like " 1. ", " 2. ", " 3. " at the beginning or after spaces
        rule_pattern = r'(?:^|\s)(\d)\.\s+'
        parts = re.split(rule_pattern, rules_text)
        
        if len(parts) > 2:  # We have actual rule splits
            rules = []
            # Skip the first part (before first rule) and process pairs
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    rule_num = parts[i]
                    content = parts[i + 1].strip()
                    if content:
                        # Clean up the content - remove trailing periods
                        content = re.sub(r'\.$', '', content)
                        rules.append(f"Rule {rule_num}: {content}")
            
            if rules:
                return '\n'.join(rules)
        
        # Method 2: If that didn't work, try priority-based splitting
        priority_pattern = r'(HIGH|MEDIUM|LOW)\s*PRIORITY:'
        if priority_pattern in rules_text.upper():
            parts = re.split(priority_pattern, rules_text, flags=re.IGNORECASE)
            rules = []
            rule_num = 1
            
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    priority = parts[i].strip()
                    content = parts[i + 1].strip()
                    if content:
                        # Remove leading punctuation
                        content = re.sub(r'^[:\-\.\s]+', '', content)
                        content = re.sub(r'\.$', '', content)  # Remove trailing period
                        rules.append(f"Rule {rule_num}: {priority} PRIORITY: {content}")
                        rule_num += 1
            
            if rules:
                return '\n'.join(rules)
        
        # Final fallback
        return f"Rule 1: {rules_text}"
        
    except Exception as e:
        logger.warning(f"Error formatting rules: {e}")
        return f"Rule 1: {rules_text.strip()}"


def save_interpretation_results(
    interpretation_data: Dict[str, Any],
    raw_response: str,
    parsed_sections: Dict[str, str],
    output_path: Path,
    source_file: Path,
    report_type: str
) -> bool:
    """
    Save interpretation results to text file in the specified format, including rules.
    
    Args:
        interpretation_data: Extracted data used for interpretation
        raw_response: Raw Gemini response
        parsed_sections: Parsed interpretation sections (now includes rules)
        output_path: Path to save file
        source_file: Source JSON file path
        report_type: Report type (customer/internal)
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(REPORT_SEPARATOR + "\n")
            file.write("DETERMINISTIC AI INTERPRETATION REPORT\n")
            file.write(REPORT_SEPARATOR + "\n")
            file.write(f"Generated on: {timestamp}\n")
            file.write(f"Source file: {source_file}\n")
            file.write(f"Report type: {report_type}\n")
            file.write(f"Room ID: {interpretation_data.get('room_id', 'Unknown')}\n")
            file.write(f"Analysis Period: {interpretation_data.get('analysis_period', 'Unknown')}\n")
            file.write(REPORT_SEPARATOR + "\n\n")
            
            file.write("INTERPRETATION DATA SUMMARY:\n")
            file.write(SECTION_SEPARATOR + "\n")
            file.write(f"Efficiency Score: {interpretation_data.get('efficiency_score', 0)}\n")
            file.write(f"Total Energy: {interpretation_data.get('total_energy_kwh', 0)} kWh\n")
            file.write(f"Comfort Score: {interpretation_data.get('comfort_score', 0)}\n")
            file.write(f"Manual Overrides: {interpretation_data.get('manual_overrides', 0)}\n")
            file.write(f"System Health Events: {interpretation_data.get('system_health_events', 0)}\n")
            file.write("\n")
            
            file.write("DETERMINISTIC INTERPRETATION:\n")
            file.write(SECTION_SEPARATOR + "\n")
            file.write("ENERGY EFFICIENCY ANALYSIS:\n")
            file.write(parsed_sections["energy_efficiency"] + "\n\n")
            
            file.write("COMFORT & WELLBEING ANALYSIS:\n")
            file.write(parsed_sections["comfort_wellbeing"] + "\n\n")
            
            file.write("PREDICTIVE INSIGHTS ANALYSIS:\n")
            file.write(parsed_sections["predictive_insights"] + "\n\n")
            
            # Add the rules sections with proper formatting
            file.write("DETERMINISTIC RULES:\n")
            file.write(SECTION_SEPARATOR + "\n")
            
            file.write("ENERGY EFFICIENCY RULES:\n")
            if parsed_sections["energy_rules"]:
                # Split by newlines and format each rule properly
                for line in parsed_sections["energy_rules"].split('\n'):
                    line = line.strip()
                    if line:
                        # Don't add bullet if the line already starts with "Rule"
                        if line.startswith('Rule '):
                            file.write(f"  • {line}\n")
                        else:
                            file.write(f"  • {line}\n")
            else:
                file.write("  • No energy efficiency rules generated.\n")
            file.write("\n")
            
            file.write("COMFORT & WELLBEING RULES:\n")
            if parsed_sections["comfort_rules"]:
                # Split by newlines and format each rule properly
                for line in parsed_sections["comfort_rules"].split('\n'):
                    line = line.strip()
                    if line:
                        # Don't add bullet if the line already starts with "Rule"
                        if line.startswith('Rule '):
                            file.write(f"  • {line}\n")
                        else:
                            file.write(f"  • {line}\n")
            else:
                file.write("  • No comfort & wellbeing rules generated.\n")
            file.write("\n")
            
            file.write(REPORT_SEPARATOR + "\n")
            file.write("End of Interpretation Report\n")
            file.write(REPORT_SEPARATOR + "\n")
        
        logger.info(f"Interpretation results saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving interpretation results: {e}")
        return False


def get_output_file_path(day: str, report_type: str, output_dir: Path) -> Path:
    """Generate output file path for interpretation results."""
    if report_type == "customer":
        folder_name = "Reports-Customer"
        filename = f"{day}_customer.txt"
    elif report_type == "internal":
        folder_name = "Reports-Internal"
        filename = f"{day}_internal.txt"
    else:
        folder_name = "Reports"
        filename = f"{day}_{report_type}.txt"
    
    return output_dir / folder_name / filename


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate deterministic AI interpretations of IoT sensor data'
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
        help=f'Type of interpretation to generate (default: {DEFAULT_REPORT_TYPE})'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('.'),
        help='Directory containing preprocessed JSON files (default: current directory)'
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
    """Main function to orchestrate the deterministic interpretation workflow"""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print(REPORT_SEPARATOR)
    print("DETERMINISTIC AI INTERPRETATION ENGINE")
    print("IoT Sensor Data Analysis with Google Gemini")
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
        # Step 1: Initialize deterministic model
        logger.info("Step 1: Initializing deterministic Gemini model...")
        model = initialize_gemini_model(api_key, args.model)
        if not model:
            sys.exit(1)
        
        # Step 2: Load processed JSON file
        logger.info("Step 2: Loading processed JSON file...")
        sensor_data = load_json_file(json_file_path)
        if sensor_data is None:
            sys.exit(1)
        
        # Step 3: Extract interpretation data
        logger.info("Step 3: Extracting interpretation data...")
        interpretation_data = extract_interpretation_data(sensor_data)
        if not interpretation_data:
            logger.error("Failed to extract interpretation data")
            sys.exit(1)
        
        # Step 4: Create deterministic prompt
        logger.info("Step 4: Creating deterministic prompt...")
        prompt = create_deterministic_prompt(interpretation_data, args.report_type)
        if not prompt:
            logger.error("Failed to create deterministic prompt")
            sys.exit(1)
        
        # Step 5: Generate interpretation
        logger.info("Step 5: Generating deterministic interpretation...")
        raw_response = interpret_with_gemini(model, prompt, args.max_retries)
        if raw_response is None:
            sys.exit(1)
        
        # Step 6: Parse interpretation response
        logger.info("Step 6: Parsing interpretation response...")
        parsed_sections = parse_interpretation_response(raw_response)
        
        # Step 7: Display results
        print("\n" + REPORT_SEPARATOR)
        print("DETERMINISTIC INTERPRETATION RESULTS")
        print(REPORT_SEPARATOR)
        print()
        print("ENERGY EFFICIENCY:")
        print(parsed_sections["energy_efficiency"])
        print()
        print("COMFORT & WELLBEING:")
        print(parsed_sections["comfort_wellbeing"])
        print()
        print("PREDICTIVE INSIGHTS:")
        print(parsed_sections["predictive_insights"])
        print()
        print("ENERGY EFFICIENCY RULES:")
        if parsed_sections["energy_rules"]:
            print(parsed_sections["energy_rules"])
        else:
            print("No energy efficiency rules generated.")
        print()
        print("COMFORT & WELLBEING RULES:")
        if parsed_sections["comfort_rules"]:
            print(parsed_sections["comfort_rules"])
        else:
            print("No comfort & wellbeing rules generated.")
        print()
        print(REPORT_SEPARATOR)
        
        # Step 8: Save results
        logger.info("Step 8: Saving interpretation results...")
        save_success = save_interpretation_results(
            interpretation_data, raw_response, parsed_sections,
            output_file_path, json_file_path, args.report_type
        )
        
        if save_success:
            logger.info("Deterministic interpretation completed successfully!")
            print(REPORT_SEPARATOR)
            print("INTERPRETATION COMPLETED SUCCESSFULLY!")
            print(f"Results saved to: {output_file_path}")
            print(REPORT_SEPARATOR)
            sys.exit(0)
        else:
            logger.warning("Interpretation completed but could not save results")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Interpretation workflow failed: {e}")
        print(REPORT_SEPARATOR)
        print("INTERPRETATION FAILED - CHECK LOGS FOR DETAILS")
        print(REPORT_SEPARATOR)
        sys.exit(1)


if __name__ == "__main__":
    main()