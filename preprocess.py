import json
import math
from typing import Dict,Any

def calculate_energy_efficiency_score(data: Dict[str, Any]) -> float:
    """
    Calculate energy efficiency score (0-100) based on kWh consumption,
    usage percentages, and waste indicators.
    """
    usage_features = data['ml_abstracted_metrics']['usage_features']
    energy_metrics = data['raw_stream_derived_metrics']['energy_proxy_metrics']
    cross_sensor = data['ml_abstracted_metrics']['cross_sensor_ratios']
    
    # Total energy consumption
    total_kwh = energy_metrics['light_kwh'] + energy_metrics['ac_kwh'] + energy_metrics['fan_kwh']
    
    # Normalize energy score (lower is better, max expected ~15 kWh per day)
    energy_score = max(0, 100 - (total_kwh / 15.0) * 100)
    
    # Usage efficiency (higher occupancy vs device usage is better)
    occupancy_pct = usage_features['occupancy_detected_pct']
    avg_device_usage = (usage_features['light_on_pct'] + usage_features['ac_on_pct']) / 2
    
    if occupancy_pct > 0:
        usage_efficiency = min(100, (avg_device_usage / occupancy_pct) * 100)
    else:
        usage_efficiency = 0
    
    # Waste penalty
    waste_penalty = (cross_sensor['ac_on_while_unoccupied_pct'] * 50 + 
                    cross_sensor['light_on_while_unoccupied_pct'] * 30 +
                    cross_sensor['ac_on_while_window_open_pct'] * 70)
    
    # Final score (weighted average)
    final_score = (energy_score * 0.4 + usage_efficiency * 0.4) - waste_penalty
    return max(0, min(100, final_score))

def calculate_total_energy_consumption(data: Dict[str, Any]) -> float:
    """Calculate total energy consumption in kWh."""
    energy_metrics = data['raw_stream_derived_metrics']['energy_proxy_metrics']
    return energy_metrics['light_kwh'] + energy_metrics['ac_kwh'] + energy_metrics['fan_kwh']

def calculate_waste_indicators(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate waste indicators from cross-sensor ratios."""
    cross_sensor = data['ml_abstracted_metrics']['cross_sensor_ratios']
    return {
        'ac_waste_unoccupied': cross_sensor['ac_on_while_unoccupied_pct'],
        'light_waste_unoccupied': cross_sensor['light_on_while_unoccupied_pct'],
        'ac_waste_window_open': cross_sensor['ac_on_while_window_open_pct'],
        'total_waste_score': (cross_sensor['ac_on_while_unoccupied_pct'] * 2 + 
                             cross_sensor['light_on_while_unoccupied_pct'] +
                             cross_sensor['ac_on_while_window_open_pct'] * 3)
    }

def analyze_energy_efficiency(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform complete energy efficiency analysis."""
    return {
        'efficiency_score': round(calculate_energy_efficiency_score(data), 2),
        'total_energy_kwh': round(calculate_total_energy_consumption(data), 2),
        'waste_indicators': calculate_waste_indicators(data),
        'energy_breakdown': {
            'light_kwh': data['raw_stream_derived_metrics']['energy_proxy_metrics']['light_kwh'],
            'ac_kwh': data['raw_stream_derived_metrics']['energy_proxy_metrics']['ac_kwh'],
            'fan_kwh': data['raw_stream_derived_metrics']['energy_proxy_metrics']['fan_kwh']
        }
    }

def calculate_comfort_score(data: Dict[str, Any]) -> float:
    """
    Calculate comfort score (0-100) based on occupancy-device alignment
    and AC runtime stability.
    """
    usage_features = data['ml_abstracted_metrics']['usage_features']
    run_length = data['ml_abstracted_metrics']['run_length_statistics']
    
    # Occupancy-light alignment (higher is better)
    occupancy_pct = usage_features['occupancy_detected_pct']
    light_pct = usage_features['light_on_pct']
    
    if occupancy_pct > 0:
        light_alignment = min(100, (light_pct / occupancy_pct) * 100)
    else:
        light_alignment = 50  # Neutral if no occupancy
    
    # AC runtime stability (lower variance relative to average is better)
    ac_avg = run_length['ac']['avg_min']
    ac_variance = run_length['ac']['variance_min2']
    
    if ac_avg > 0:
        ac_stability = max(0, 100 - (math.sqrt(ac_variance) / ac_avg) * 100)
    else:
        ac_stability = 100
    
    # Occupancy consistency
    occupancy_avg = run_length['occupancy']['avg_min']
    occupancy_variance = run_length['occupancy']['variance_min2']
    
    if occupancy_avg > 0:
        occupancy_consistency = max(0, 100 - (math.sqrt(occupancy_variance) / occupancy_avg) * 100)
    else:
        occupancy_consistency = 50
    
    # Weighted comfort score
    comfort_score = (light_alignment * 0.4 + ac_stability * 0.3 + occupancy_consistency * 0.3)
    return max(0, min(100, comfort_score))

def calculate_alignment_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate key alignment metrics for comfort analysis."""
    usage_features = data['ml_abstracted_metrics']['usage_features']
    cross_sensor = data['ml_abstracted_metrics']['cross_sensor_ratios']
    
    occupancy_pct = usage_features['occupancy_detected_pct']
    
    return {
        'occupancy_light_ratio': (usage_features['light_on_pct'] / occupancy_pct 
                                 if occupancy_pct > 0 else 0),
        'occupancy_ac_ratio': (usage_features['ac_on_pct'] / occupancy_pct 
                              if occupancy_pct > 0 else 0),
        'unoccupied_device_usage': cross_sensor['ac_on_while_unoccupied_pct'] + 
                                  cross_sensor['light_on_while_unoccupied_pct'],
        'window_ac_conflict': cross_sensor['ac_on_while_window_open_pct']
    }

def analyze_comfort_wellbeing(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform complete comfort and wellbeing analysis."""
    return {
        'comfort_score': round(calculate_comfort_score(data), 2),
        'alignment_metrics': calculate_alignment_metrics(data),
        'occupancy_stats': {
            'detected_pct': data['ml_abstracted_metrics']['usage_features']['occupancy_detected_pct'],
            'avg_duration_min': data['ml_abstracted_metrics']['run_length_statistics']['occupancy']['avg_min'],
            'variance_min2': data['ml_abstracted_metrics']['run_length_statistics']['occupancy']['variance_min2']
        }
    }

def check_predictive_flags(data: Dict[str, Any]) -> Dict[str, bool]:
    """Check various predictive flags for potential issues."""
    pattern_flags = data['ml_abstracted_metrics']['pattern_deviation_flags']
    rolling_stats = data['ml_abstracted_metrics']['rolling_statistics_and_drift']
    seasonal = data['ml_abstracted_metrics']['seasonal_pattern_summaries']
    
    return {
        'power_drift_detected': (rolling_stats['ac_power_drift_flag'] or 
                                rolling_stats['light_power_drift_flag']),
        'pattern_anomaly_detected': (pattern_flags['ac_rapid_cycle_flag'] or
                                    pattern_flags['light_stuck_flag'] or
                                    pattern_flags['occupancy_sensor_stuck_flag']),
        'seasonal_deviation_high': (seasonal['ac_vs_weekly_pct'] > 10 or
                                   seasonal['light_vs_weekly_pct'] > 10),
        'ac_usage_trending_up': seasonal['ac_vs_weekly_pct'] > 5
    }

def calculate_total_override_count(data: Dict[str, Any]) -> int:
    """Calculate total manual override count across all devices."""
    overrides = data['raw_stream_derived_metrics']['manual_override_counters']
    return sum(override['count'] for override in overrides)

def check_frequent_overrides(data: Dict[str, Any]) -> bool:
    """Check if there are frequent manual overrides (>3 total)."""
    return calculate_total_override_count(data) > 3

def analyze_predictive_prescriptive(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform predictive and prescriptive analysis."""
    flags = check_predictive_flags(data)
    total_overrides = calculate_total_override_count(data)
    
    return {
        'predictive_flags': flags,
        'frequent_overrides': check_frequent_overrides(data),
        'total_override_count': total_overrides,
        'override_breakdown': {
            override['device']: override['count'] 
            for override in data['raw_stream_derived_metrics']['manual_override_counters']
        },
        'risk_score': sum([
            flags['power_drift_detected'] * 20,
            flags['pattern_anomaly_detected'] * 30,
            flags['seasonal_deviation_high'] * 15,
            (total_overrides > 3) * 25,
            (total_overrides > 5) * 10
        ])
    }

def analyze_sensor_health(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze sensor health status."""
    health_events = data['raw_stream_derived_metrics']['sensor_health_events']
    
    battery_low_sensors = []
    signal_degradation_sensors = []
    
    for event in health_events:
        if event['event'] == 'battery_low':
            battery_low_sensors.append(event['sensor'])
        elif event['event'] == 'signal_degradation':
            signal_degradation_sensors.append(event['sensor'])
    
    return {
        'battery_low_flag': len(battery_low_sensors) > 0,
        'signal_degradation_flag': len(signal_degradation_sensors) > 0,
        'any_health_issue': len(health_events) > 0,
        'affected_sensors': {
            'battery_low': battery_low_sensors,
            'signal_degradation': signal_degradation_sensors
        },
        'total_health_events': len(health_events),
        'health_events_detail': health_events
    }

def calculate_aggregated_events(data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Calculate total on/off events per device."""
    event_counts = data['raw_stream_derived_metrics']['event_counts_and_timing']
    
    aggregated = {}
    
    for device, counts in event_counts.items():
        if device in ['light', 'ac', 'fan']:
            aggregated[device] = {
                'total_on_events': counts['on_count'],
                'total_off_events': counts['off_count'],
                'total_cycles': min(counts['on_count'], counts['off_count'])
            }
        elif device == 'window':
            aggregated[device] = {
                'total_open_events': counts['open_count'],
                'total_close_events': counts['close_count'],
                'total_cycles': min(counts['open_count'], counts['close_count'])
            }
        elif device == 'door':
            aggregated[device] = {
                'total_open_events': counts['open_count'],
                'total_close_events': counts['close_count'],
                'total_cycles': min(counts['open_count'], counts['close_count'])
            }
        elif device == 'occupancy':
            aggregated[device] = {
                'total_detected_events': counts['detected_count'],
                'total_cleared_events': counts['cleared_count'],
                'total_cycles': min(counts['detected_count'], counts['cleared_count'])
            }
    
    return aggregated

def preprocess_sensor_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main preprocessing function that transforms raw sensor data into 
    structured format with analysis sections.
    """
    # Preserve all original data
    preprocessed = data.copy()
    
    # Add new analysis sections
    preprocessed['energy_efficiency_analysis'] = analyze_energy_efficiency(data)
    preprocessed['comfort_wellbeing_analysis'] = analyze_comfort_wellbeing(data)
    preprocessed['predictive_prescriptive_analysis'] = analyze_predictive_prescriptive(data)
    preprocessed['sensor_health_analysis'] = analyze_sensor_health(data)
    preprocessed['aggregated_events'] = calculate_aggregated_events(data)
    
    return preprocessed

def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json_data(data: Dict[str, Any], file_path: str) -> None:
    """Save JSON data to file with proper formatting."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def process_single_file(input_path: str) -> Dict[str, Any]:
    """Process a single JSON file and save with _preprocessed suffix."""
    raw_data = load_json_data(input_path)
    preprocessed_data = preprocess_sensor_data(raw_data)
    
    # Generate output filename with _preprocessed suffix
    if input_path.endswith('.json'):
        output_path = input_path[:-5] + '_preprocessed.json'
    else:
        output_path = input_path + '_preprocessed.json'
    
    save_json_data(preprocessed_data, output_path)
    return preprocessed_data

def process_json_string(json_string: str) -> str:
    """
    Process JSON string directly and return preprocessed JSON string.
    """
    raw_data = json.loads(json_string)
    preprocessed_data = preprocess_sensor_data(raw_data)
    return json.dumps(preprocessed_data, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    # Example: Process a single file
    try:
        processed = process_single_file("day_1.json")
        print(f"Processed day_1.json -> day_1_preprocessed.json")
        print(f"Efficiency Score: {processed['energy_efficiency_analysis']['efficiency_score']}")
        print(f"Comfort Score: {processed['comfort_wellbeing_analysis']['comfort_score']}")
        print(f"Risk Score: {processed['predictive_prescriptive_analysis']['risk_score']}")
    except FileNotFoundError:
        print("File day_1.json not found")
    except Exception as e:
        print(f"Processing failed: {e}")
    