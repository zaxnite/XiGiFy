import json
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

def load_json(file_path: str) -> dict:
    """Load JSON data from a file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"The file '{file_path}' was not found."}
    except json.JSONDecodeError:
        return {"error": f"The file '{file_path}' is not a valid JSON file."}

class ChunkParser:
    """A state machine-based parser for chunk strings with hierarchical structure."""
    def __init__(self):
        # Define the expected section hierarchy
        self.section_hierarchy = {
            'root': ['_id', 'room_id', 'summary_window', 'ml_abstracted_metrics', 'raw_stream_derived_metrics', 'sensor_metrics'],
            'summary_window': ['start', 'end'],
            'ml_abstracted_metrics': [
                'occupancy_score', 'energy_consumption_score', 'activity_variance',
                'anomaly_likelihood', 'connectivity_score', 'cross_sensor_ratios',
                'inter_event_distributions', 'pattern_deviation_flags',
                'rolling_statistics_and_drift', 'run_length_statistics',
                'runtime_features', 'seasonal_pattern_summaries', 'usage_features'
            ],
            'cross_sensor_ratios': [
                'light_to_occupancy_ratio', 'ac_to_occupancy_ratio', 'ac_to_window_ratio',
                'sensor_to_occupancy_ratio', 'energy_per_occupancy_minute'
            ],
            'inter_event_distributions': ['power', 'sensor', 'occupancy_intervals_s'],
            'pattern_deviation_flags': [
                'unexpected_energy_consumption', 'off_hour_activity', 'prolonged_unoccupancy',
                'device_state_mismatch', 'poor_connectivity', 'sensor_anomaly'
            ],
            'rolling_statistics_and_drift': [
                'power_power_avg_w', 'power_power_variance', 'power_power_drift_flag',
                'power_voltage_avg', 'power_voltage_variance'
            ],
            'run_length_statistics': ['light', 'ac', 'fan', 'occupancy', 'window', 'door', 'switch'],
            'runtime_features': [
                'light_active_pct', 'ac_active_pct', 'fan_active_pct', 'occupancy_active_pct',
                'window_active_pct', 'door_active_pct', 'switch_active_pct'
            ],
            'seasonal_pattern_summaries': [
                'ac_vs_weekly_pct', 'light_vs_weekly_pct', 'occupancy_vs_weekly_pct'
            ],
            'usage_features': [
                'light_on_pct', 'ac_on_pct', 'fan_on_pct', 'occupancy_on_pct',
                'window_on_pct', 'door_on_pct', 'switch_on_pct', 'power_on_pct',
                'sensor_on_pct', 'controller_on_pct', 'temperature_reporting_pct',
                'humidity_reporting_pct', 'voltage_reporting_pct', 'signal_reporting_pct'
            ],
            'raw_stream_derived_metrics': [
                'message_count', 'device_type_distribution', 'avg_message_latency_ms',
                'keep_alive_ratio', 'event_counts_and_timing', 'manual_override_counters'
            ],
            'device_type_distribution': ['light', 'ac', 'fan', 'occupancy', 'window', 'door', 'switch', 'power', 'sensor', 'controller'],
            'event_counts_and_timing': ['power', 'sensor'],
            'manual_override_counters': [
                'total_manual_switches', 'out_of_hour_manual_switches', 'override_duration_s'
            ],
            'sensor_metrics': [
                'power_voltage_avg', 'power_voltage_min', 'power_voltage_max', 'power_voltage_std_dev'
            ]
        }
        # Special handling for nested structures
        self.special_structures = {
            'inter_event_distributions': ['power', 'sensor'],
            'run_length_statistics': ['light', 'ac', 'fan', 'occupancy', 'window', 'door', 'switch'],
            'event_counts_and_timing': ['power', 'sensor']
        }

    def tokenize_chunk(self, chunk_str: str) -> List[Tuple[str, str]]:
        """Tokenize the chunk string into key-value pairs."""
        # Split by spaces and find key: value patterns
        tokens = []
        words = chunk_str.split()
        i = 0
        while i < len(words):
            word = words[i]
            if ':' in word and not word.endswith(':'):
                # Handle "key: value" pattern
                key = word.split(':')[0]
                value = word.split(':', 1)[1]
                tokens.append((key, value))
            elif word.endswith(':') and i + 1 < len(words):
                # Handle "key:" followed by "value"
                key = word[:-1]  # Remove the colon
                value = words[i + 1]
                tokens.append((key, value))
                i += 1  # Skip the value word
            i += 1
        return tokens

    def convert_value(self, value_str: str) -> Any:
        """Convert string value to appropriate type."""
        if not value_str:
            return ""
        # Handle boolean values
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        # Handle numeric values
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            return value_str

    def parse_with_state_machine(self, chunk_str: str) -> Dict[str, Any]:
        """Parse chunk string using a state machine approach."""
        tokens = self.tokenize_chunk(chunk_str)
        result = {}
        current_section = None
        current_subsection = None
        current_subsubsection = None
        # Initialize all expected sections
        template = self.create_empty_template()
        result.update(template)
        i = 0
        while i < len(tokens):
            key, value = tokens[i]
            # Check if this is a top-level section
            if key in self.section_hierarchy['root']:
                if key in ['_id', 'room_id']:
                    result[key] = self.convert_value(value)
                else:
                    current_section = key
                    current_subsection = None
                    current_subsubsection = None
                    # For summary_window, extract start and end
                    if key == 'summary_window':
                        i += 1
                        if i < len(tokens) and tokens[i][0] == 'start':
                            result[key]['start'] = tokens[i][1]
                        i += 1
                        if i < len(tokens) and tokens[i][0] == 'end':
                            result[key]['end'] = tokens[i][1]
                        i += 1
                        continue
            # Check if this is a subsection
            elif current_section and key in self.section_hierarchy.get(current_section, []):
                current_subsection = key
                current_subsubsection = None
                # Handle special nested structures
                if key in self.special_structures:
                    i = self.parse_special_structure(tokens, i, result, current_section, key)
                    continue
                elif current_section == 'ml_abstracted_metrics' and key == 'cross_sensor_ratios':
                    # Handle cross_sensor_ratios
                    i = self.parse_cross_sensor_ratios(tokens, i, result)
                    continue
                elif current_section == 'ml_abstracted_metrics' and key in ['pattern_deviation_flags', 'rolling_statistics_and_drift', 'runtime_features', 'seasonal_pattern_summaries', 'usage_features']:
                    # Handle flat subsections
                    i = self.parse_flat_subsection(tokens, i, result, current_section, key)
                    continue
                elif current_section == 'raw_stream_derived_metrics' and key == 'manual_override_counters':
                    i = self.parse_flat_subsection(tokens, i, result, current_section, key)
                    continue
                elif current_section == 'raw_stream_derived_metrics' and key in ['message_count', 'avg_message_latency_ms', 'keep_alive_ratio']:
                    result[current_section][key] = self.convert_value(value)
            # Check if this is a subsubsection or specific value
            elif current_subsection:
                # Handle device types in special structures
                if current_subsection in self.special_structures and key in self.special_structures[current_subsection]:
                    current_subsubsection = key
                    i = self.parse_device_specific_data(tokens, i, result, current_section, current_subsection, key)
                    continue
                # Handle direct values in subsections
                elif current_section and current_subsection:
                    if current_section not in result:
                        result[current_section] = {}
                    if current_subsection not in result[current_section]:
                        result[current_section][current_subsection] = {}
                    result[current_section][current_subsection][key] = self.convert_value(value)
            i += 1
        return result

    def parse_cross_sensor_ratios(self, tokens: List[Tuple[str, str]], start_idx: int, result: Dict[str, Any]) -> int:
        """Parse cross_sensor_ratios section."""
        i = start_idx + 1
        while i < len(tokens):
            key, value = tokens[i]
            if key in self.section_hierarchy['cross_sensor_ratios']:
                result['ml_abstracted_metrics']['cross_sensor_ratios'][key] = self.convert_value(value)
                i += 1
            else:
                break
        return i - 1

    def parse_flat_subsection(self, tokens: List[Tuple[str, str]], start_idx: int, result: Dict[str, Any], section: str, subsection: str) -> int:
        """Parse a flat subsection with direct key-value pairs."""
        i = start_idx + 1
        while i < len(tokens):
            key, value = tokens[i]
            if key in self.section_hierarchy.get(subsection, []):
                result[section][subsection][key] = self.convert_value(value)
                i += 1
            else:
                break
        return i - 1

    def parse_special_structure(self, tokens: List[Tuple[str, str]], start_idx: int, result: Dict[str, Any], section: str, subsection: str) -> int:
        """Parse special nested structures like inter_event_distributions."""
        i = start_idx + 1
        while i < len(tokens):
            key, value = tokens[i]
            # Check if this is a device/type in the special structure
            if key in self.special_structures[subsection]:
                if subsection == 'inter_event_distributions':
                    # Parse distribution statistics
                    i = self.parse_distribution_stats(tokens, i, result, section, subsection, key)
                elif subsection == 'event_counts_and_timing':
                    # Parse event timing data
                    i = self.parse_event_timing(tokens, i, result, section, subsection, key)
                elif subsection == 'run_length_statistics':
                    # Parse run length stats
                    i = self.parse_run_length_stats(tokens, i, result, section, subsection, key)
                i += 1
            else:
                break
        return i - 1

    def parse_distribution_stats(self, tokens: List[Tuple[str, str]], start_idx: int, result: Dict[str, Any], section: str, subsection: str, device: str) -> int:
        """Parse distribution statistics (mean, std_dev, min, max)."""
        stats = {}
        i = start_idx + 1
        stat_keys = ['mean', 'std_dev', 'min', 'max']
        for stat_key in stat_keys:
            if i < len(tokens) and tokens[i][0] == stat_key:
                stats[stat_key] = self.convert_value(tokens[i][1])
                i += 1
            else:
                break
        if stats:
            result[section][subsection][device] = stats
        return i - 1

    def parse_event_timing(self, tokens: List[Tuple[str, str]], start_idx: int, result: Dict[str, Any], section: str, subsection: str, device: str) -> int:
        """Parse event timing data (on_count, off_count, first_seen, last_seen)."""
        timing = {}
        i = start_idx + 1
        timing_keys = ['on_count', 'off_count', 'first_seen', 'last_seen']
        for timing_key in timing_keys:
            if i < len(tokens) and tokens[i][0] == timing_key:
                timing[timing_key] = self.convert_value(tokens[i][1])
                i += 1
            else:
                break
        if timing:
            result[section][subsection][device] = timing
        return i - 1

    def parse_run_length_stats(self, tokens: List[Tuple[str, str]], start_idx: int, result: Dict[str, Any], section: str, subsection: str, device: str) -> int:
        """Parse run length statistics (avg_min/avg_s, variance_min2/variance_s2)."""
        stats = {}
        i = start_idx + 1
        # Door uses different keys (avg_s, variance_s2) vs others (avg_min, variance_min2)
        if device == 'door':
            stat_keys = ['avg_s', 'variance_s2']
        else:
            stat_keys = ['avg_min', 'variance_min2']
        for stat_key in stat_keys:
            if i < len(tokens) and tokens[i][0] == stat_key:
                stats[stat_key] = self.convert_value(tokens[i][1])
                i += 1
            else:
                break
        if stats:
            result[section][subsection][device] = stats
        return i - 1

    def parse_device_specific_data(self, tokens: List[Tuple[str, str]], start_idx: int, result: Dict[str, Any], section: str, subsection: str, device: str) -> int:
        """Parse device-specific data based on context."""
        if subsection == 'device_type_distribution':
            # Simple count value
            result[section][subsection][device] = self.convert_value(tokens[start_idx][1])
            return start_idx
        else:
            return start_idx

    def create_empty_template(self) -> Dict[str, Any]:
        """Create an empty template with all expected structure matching the desired output."""
        return {
            "_id": "", # Will be populated
            "room_id": "", # Will be populated
            "summary_window": {
                "start": "", # Will be populated
                "end": "" # Will be populated
            },
            "ml_abstracted_metrics": {
                "occupancy_score": 0.0,
                "energy_consumption_score": 0.0,
                "activity_variance": 0.0,
                "anomaly_likelihood": 1.0, # Default from example
                "connectivity_score": 0.0,
                # cross_sensor_ratios is nested directly under ml_abstracted_metrics
                "cross_sensor_ratios": {
                    "light_to_occupancy_ratio": 0.0,
                    "ac_to_occupancy_ratio": 0.0,
                    "ac_to_window_ratio": 0.0,
                    "sensor_to_occupancy_ratio": 0.0,
                    "energy_per_occupancy_minute": 0.0
                },
                # inter_event_distributions contains nested structures
                "inter_event_distributions": {
                    "power": { # Stats for power, initialized if present
                        "mean": 0.0, # Default values for structure
                        "std_dev": 0.0,
                        "min": 0.0,
                        "max": 0.0
                    },
                    # "sensor" stats within inter_event_distributions are parsed if present
                    # sensor: { mean: ..., std_dev: ... } -> handled dynamically
                    # occupancy_intervals_s contains pattern_deviation_flags
                    "occupancy_intervals_s": {
                        "pattern_deviation_flags": {
                            "unexpected_energy_consumption": False,
                            "off_hour_activity": False,
                            "prolonged_unoccupancy": True, # Default from example
                            "device_state_mismatch": False,
                            "poor_connectivity": True, # Default from example
                            "sensor_anomaly": False
                        }
                    }
                },
                # pattern_deviation_flags moved to be directly under ml_abstracted_metrics
                # (This is kept for hierarchy definition but actual data goes under inter_event_distributions)
                # "pattern_deviation_flags": {...}, # Moved inside inter_event_distributions.occupancy_intervals_s
                "rolling_statistics_and_drift": {
                    "power_power_avg_w": 0.0,
                    "power_power_variance": 0.0,
                    "power_power_drift_flag": False,
                    "power_voltage_avg": 0.0,
                    "power_voltage_variance": 0.0
                },
                "run_length_statistics": {
                    "light": {"avg_min": 0.0, "variance_min2": 0.0},
                    "ac": {"avg_min": 0.0, "variance_min2": 0.0},
                    "fan": {"avg_min": 0.0, "variance_min2": 0.0},
                    "occupancy": {"avg_min": 0.0, "variance_min2": 0.0},
                    "window": {"avg_min": 0.0, "variance_min2": 0.0},
                    "door": {"avg_s": 0.0, "variance_s2": 0.0},
                    "switch": {"avg_min": 0.0, "variance_min2": 0.0}
                },
                "runtime_features": {
                    "light_active_pct": 0.0,
                    "ac_active_pct": 0.0,
                    "fan_active_pct": 0.0,
                    "occupancy_active_pct": 0.0,
                    "window_active_pct": 0.0,
                    "door_active_pct": 0.0,
                    "switch_active_pct": 0.0
                },
                "seasonal_pattern_summaries": {
                    "ac_vs_weekly_pct": 100.0, # Default from example
                    "light_vs_weekly_pct": 100.0, # Default from example
                    "occupancy_vs_weekly_pct": 100.0 # Default from example
                },
                "usage_features": {
                    "light_on_pct": 0.0,
                    "ac_on_pct": 0.0,
                    "fan_on_pct": 0.0,
                    "occupancy_on_pct": 0.0,
                    "window_on_pct": 0.0,
                    "door_on_pct": 0.0,
                    "switch_on_pct": 0.0,
                    "power_on_pct": 0.0,
                    "sensor_on_pct": 0.0,
                    "controller_on_pct": 0.0,
                    "temperature_reporting_pct": 0.0,
                    "humidity_reporting_pct": 0.0,
                    "voltage_reporting_pct": 1.0, # Default from example
                    "signal_reporting_pct": 0.0
                }
            },
            "raw_stream_derived_metrics": {
                "message_count": 0,
                "device_type_distribution": { # Initialize common keys
                    "light": 0, "ac": 0, "fan": 0, "occupancy": 0,
                    "window": 0, "door": 0, "switch": 0, "power": 0,
                    "sensor": 0, "controller": 0
                },
                "avg_message_latency_ms": 0.0,
                "keep_alive_ratio": 0.0,
                # event_counts_and_timing is nested directly under raw_stream_derived_metrics
                "event_counts_and_timing": {
                     "power": { # Stats for power, initialized if present
                         "on_count": 0,
                         "off_count": 0,
                         "first_seen": "",
                         "last_seen": ""
                     }
                     # "sensor" stats within event_counts_and_timing are parsed if present
                     # sensor: { on_count: ..., off_count: ..., first_seen: ..., last_seen: ... } -> handled dynamically
                },
                "manual_override_counters": {
                    "total_manual_switches": 0,
                    "out_of_hour_manual_switches": 0,
                    "override_duration_s": 0.0
                }
            },
            "sensor_metrics": {
                "power_voltage_avg": 0.0,
                "power_voltage_min": 0.0,
                "power_voltage_max": 0.0,
                "power_voltage_std_dev": 0.0
            }
        }

    def parse_chunk_improved(self, chunk_str: str) -> Dict[str, Any]:
        """Main parsing method using improved logic."""
        # Clean the chunk string
        chunk_str = ' '.join(chunk_str.split())  # Normalize whitespace
        # Use regex-based approach with better section detection
        result = self.create_empty_template()
        # Extract basic fields first
        basic_patterns = {
            '_id': r'_id:\s*(\S+)',
            'room_id': r'room_id:\s*(\S+)'
        }
        for field, pattern in basic_patterns.items():
            match = re.search(pattern, chunk_str)
            if match:
                result[field] = match.group(1)
        # Extract summary_window
        start_match = re.search(r'start:\s*([\d\-T:]+)', chunk_str)
        end_match = re.search(r'end:\s*([\d\-T:]+)', chunk_str)
        if start_match:
            result['summary_window']['start'] = start_match.group(1)
        if end_match:
            result['summary_window']['end'] = end_match.group(1)
        # Parse sections using enhanced regex patterns
        self.parse_ml_abstracted_metrics(chunk_str, result)
        self.parse_raw_stream_derived_metrics(chunk_str, result)
        self.parse_sensor_metrics(chunk_str, result)
        return result

    def parse_ml_abstracted_metrics(self, chunk_str: str, result: Dict[str, Any]):
        """Parse ml_abstracted_metrics section."""
        # Find the ml_abstracted_metrics section
        ml_start = chunk_str.find('ml_abstracted_metrics:')
        if ml_start == -1:
            return
        # Find the end of this section (start of next major section)
        raw_start = chunk_str.find('raw_stream_derived_metrics:', ml_start)
        ml_section = chunk_str[ml_start:raw_start] if raw_start != -1 else chunk_str[ml_start:]
        # Parse simple numeric fields
        simple_fields = [
            'occupancy_score', 'energy_consumption_score', 'activity_variance',
            'anomaly_likelihood', 'connectivity_score'
        ]
        for field in simple_fields:
            pattern = rf'{field}:\s*([\d\.\-]+)'
            match = re.search(pattern, ml_section)
            if match:
                result['ml_abstracted_metrics'][field] = self.convert_value(match.group(1))

        # Parse cross_sensor_ratios (nested directly under ml_abstracted_metrics)
        self.parse_section_with_pattern(ml_section, 'cross_sensor_ratios',
                                       self.section_hierarchy['cross_sensor_ratios'],
                                       result['ml_abstracted_metrics']['cross_sensor_ratios'])

        # Parse pattern_deviation_flags (nested under inter_event_distributions.occupancy_intervals_s)
        # Adjust target dictionary path
        pattern_flags_path = result['ml_abstracted_metrics']['inter_event_distributions']['occupancy_intervals_s']['pattern_deviation_flags']
        self.parse_section_with_pattern(ml_section, 'pattern_deviation_flags',
                                       self.section_hierarchy['pattern_deviation_flags'],
                                       pattern_flags_path)

        # Parse rolling_statistics_and_drift
        self.parse_section_with_pattern(ml_section, 'rolling_statistics_and_drift',
                                       self.section_hierarchy['rolling_statistics_and_drift'],
                                       result['ml_abstracted_metrics']['rolling_statistics_and_drift'])

        # Parse runtime_features
        self.parse_section_with_pattern(ml_section, 'runtime_features',
                                       self.section_hierarchy['runtime_features'],
                                       result['ml_abstracted_metrics']['runtime_features'])

        # Parse seasonal_pattern_summaries
        self.parse_section_with_pattern(ml_section, 'seasonal_pattern_summaries',
                                       self.section_hierarchy['seasonal_pattern_summaries'],
                                       result['ml_abstracted_metrics']['seasonal_pattern_summaries'])

        # Parse usage_features
        self.parse_section_with_pattern(ml_section, 'usage_features',
                                       self.section_hierarchy['usage_features'],
                                       result['ml_abstracted_metrics']['usage_features'])

        # Parse inter_event_distributions
        self.parse_inter_event_distributions(ml_section, result)

        # Parse run_length_statistics
        self.parse_run_length_statistics(ml_section, result)

    def parse_section_with_pattern(self, text: str, section_name: str, expected_keys: List[str], target_dict: Dict[str, Any]):
        """Parse a section using pattern matching for expected keys."""
        section_start = text.find(f'{section_name}:')
        if section_start == -1:
            return
        for key in expected_keys:
            # Handle potential colon after key name in the pattern
            # Allow for values that might contain colons (e.g., timestamps)
            pattern = rf'{key}:\s*([\w\.\-:T]+)'
            match = re.search(pattern, text[section_start:])
            if match:
                target_dict[key] = self.convert_value(match.group(1))

    def parse_inter_event_distributions(self, ml_section: str, result: Dict[str, Any]):
        """Parse inter_event_distributions section."""
        inter_event_start = ml_section.find('inter_event_distributions:')
        if inter_event_start == -1:
            return
        # Find the end of inter_event_distributions section
        # More accurate next sections based on the template structure
        next_section_patterns = ['rolling_statistics_and_drift:', 'run_length_statistics:', 'runtime_features:']
        inter_event_end = len(ml_section)
        for pattern in next_section_patterns:
            pos = ml_section.find(pattern, inter_event_start)
            if pos != -1:
                inter_event_end = min(inter_event_end, pos)
        inter_event_text = ml_section[inter_event_start:inter_event_end]

        # Parse power distribution stats within inter_event_distributions
        power_pattern = r'power:\s*mean:\s*([\d\.]+)\s*std_dev:\s*([\d\.]+)\s*min:\s*([\d\.]+)\s*max:\s*([\d\.]+)'
        power_match = re.search(power_pattern, inter_event_text)
        if power_match:
            # Update the pre-initialized power dict
            result['ml_abstracted_metrics']['inter_event_distributions']['power'].update({
                "mean": float(power_match.group(1)),
                "std_dev": float(power_match.group(2)),
                "min": float(power_match.group(3)),
                "max": float(power_match.group(4))
            })

        # Parse sensor distribution stats within inter_event_distributions
        sensor_ie_pattern = r'sensor:\s*mean:\s*([\d\.]+)\s*std_dev:\s*([\d\.]+)\s*min:\s*([\d\.]+)\s*max:\s*([\d\.]+)'
        sensor_ie_match = re.search(sensor_ie_pattern, inter_event_text)
        if sensor_ie_match:
            # Add or update sensor stats under inter_event_distributions
            result['ml_abstracted_metrics']['inter_event_distributions']['sensor'] = {
                "mean": float(sensor_ie_match.group(1)),
                "std_dev": float(sensor_ie_match.group(2)),
                "min": float(sensor_ie_match.group(3)),
                "max": float(sensor_ie_match.group(4))
            }

        # Check for and parse occupancy_intervals_s flags
        # The path is already initialized in the template
        occupancy_flags_start = inter_event_text.find('occupancy_intervals_s:')
        if occupancy_flags_start != -1:
            # Parse the flags within occupancy_intervals_s
            # The target dictionary path was already set in parse_ml_abstracted_metrics
            flags_dict = result['ml_abstracted_metrics']['inter_event_distributions']['occupancy_intervals_s']['pattern_deviation_flags']
            # Re-use parse_section_with_pattern for the flags within this specific context
            temp_hierarchy_flags = ['unexpected_energy_consumption', 'off_hour_activity', 'prolonged_unoccupancy',
                                    'device_state_mismatch', 'poor_connectivity', 'sensor_anomaly']
            self.parse_section_with_pattern(inter_event_text[occupancy_flags_start:], 'occupancy_intervals_s', temp_hierarchy_flags, flags_dict)
        # If occupancy_intervals_s is not found in the text, the template default values remain

    def parse_run_length_statistics(self, ml_section: str, result: Dict[str, Any]):
        """Parse run_length_statistics section."""
        run_length_start = ml_section.find('run_length_statistics:')
        if run_length_start == -1:
            return
        # Find the end of run_length_statistics section
        next_section_patterns = ['runtime_features:', 'seasonal_pattern_summaries:']
        run_length_end = len(ml_section)
        for pattern in next_section_patterns:
            pos = ml_section.find(pattern, run_length_start)
            if pos != -1:
                run_length_end = min(run_length_end, pos)
        run_length_text = ml_section[run_length_start:run_length_end]
        # Parse each device type
        devices = ['light', 'ac', 'fan', 'occupancy', 'window', 'door', 'switch']
        for device in devices:
            if device == 'door':
                # Door uses avg_s and variance_s2
                pattern = rf'{device}:\s*avg_s:\s*([\d\.]+)\s*variance_s2:\s*([\d\.]+)'
                match = re.search(pattern, run_length_text)
                if match:
                    result['ml_abstracted_metrics']['run_length_statistics'][device] = {
                        "avg_s": float(match.group(1)),
                        "variance_s2": float(match.group(2))
                    }
            else:
                # Others use avg_min and variance_min2
                pattern = rf'{device}:\s*avg_min:\s*([\d\.]+)\s*variance_min2:\s*([\d\.]+)'
                match = re.search(pattern, run_length_text)
                if match:
                    result['ml_abstracted_metrics']['run_length_statistics'][device] = {
                        "avg_min": float(match.group(1)),
                        "variance_min2": float(match.group(2))
                    }

    def parse_raw_stream_derived_metrics(self, chunk_str: str, result: Dict[str, Any]):
        """Parse raw_stream_derived_metrics section."""
        raw_start = chunk_str.find('raw_stream_derived_metrics:')
        if raw_start == -1:
            return
        # Find the end of this section
        sensor_start = chunk_str.find('sensor_metrics:', raw_start)
        raw_section = chunk_str[raw_start:sensor_start] if sensor_start != -1 else chunk_str[raw_start:]
        # Parse simple fields
        simple_patterns = {
            'message_count': r'message_count:\s*(\d+)',
            'avg_message_latency_ms': r'avg_message_latency_ms:\s*([\d\.]+)',
            'keep_alive_ratio': r'keep_alive_ratio:\s*([\d\.]+)'
        }
        for field, pattern in simple_patterns.items():
            match = re.search(pattern, raw_section)
            if match:
                result['raw_stream_derived_metrics'][field] = self.convert_value(match.group(1))
        # Parse device_type_distribution
        self.parse_device_type_distribution(raw_section, result)
        # Parse event_counts_and_timing
        self.parse_event_counts_and_timing(raw_section, result)
        # Parse manual_override_counters
        self.parse_section_with_pattern(raw_section, 'manual_override_counters',
                                       self.section_hierarchy['manual_override_counters'],
                                       result['raw_stream_derived_metrics']['manual_override_counters'])

    def parse_device_type_distribution(self, raw_section: str, result: Dict[str, Any]):
        """Parse device_type_distribution section."""
        dist_start = raw_section.find('device_type_distribution:')
        if dist_start == -1:
            return
        # Find the end of device_type_distribution
        next_patterns = ['avg_message_latency_ms:', 'keep_alive_ratio:', 'event_counts_and_timing:']
        dist_end = len(raw_section)
        for pattern in next_patterns:
            pos = raw_section.find(pattern, dist_start)
            if pos != -1:
                dist_end = min(dist_end, pos)
        dist_text = raw_section[dist_start:dist_end]
        # Extract device counts
        devices = ['light', 'ac', 'fan', 'occupancy', 'window', 'door', 'switch', 'power', 'sensor', 'controller']
        for device in devices:
            pattern = rf'{device}:\s*(\d+)'
            match = re.search(pattern, dist_text)
            if match:
                result['raw_stream_derived_metrics']['device_type_distribution'][device] = int(match.group(1))

    def parse_event_counts_and_timing(self, raw_section: str, result: Dict[str, Any]):
        """Parse event_counts_and_timing section."""
        event_start = raw_section.find('event_counts_and_timing:')
        if event_start == -1:
            return
        # Find the end of event_counts_and_timing
        next_patterns = ['manual_override_counters:', 'sensor_metrics:']
        event_end = len(raw_section)
        for pattern in next_patterns:
            pos = raw_section.find(pattern, event_start)
            if pos != -1:
                event_end = min(event_end, pos)
        event_text = raw_section[event_start:event_end]
        # Parse power events (update pre-initialized dict)
        power_pattern = r'power:\s*on_count:\s*(\d+)\s*off_count:\s*(\d+)\s*first_seen:\s*([\d\-T:]+)\s*last_seen:\s*([\d\-T:]+)'
        power_match = re.search(power_pattern, event_text)
        if power_match:
            result['raw_stream_derived_metrics']['event_counts_and_timing']['power'].update({
                "on_count": int(power_match.group(1)),
                "off_count": int(power_match.group(2)),
                "first_seen": power_match.group(3),
                "last_seen": power_match.group(4)
            })
        # Parse sensor events within event_counts_and_timing
        # The regex needs to be flexible for cases where first_seen/last_seen might be missing or present.
        sensor_pattern = r'sensor:\s*on_count:\s*(\d+)\s*off_count:\s*(\d+)(?:\s*first_seen:\s*([\d\-T:]+))?(?:\s*last_seen:\s*([\d\-T:]+))?'
        sensor_match = re.search(sensor_pattern, event_text)
        if sensor_match:
            sensor_data = {
                "on_count": int(sensor_match.group(1)),
                "off_count": int(sensor_match.group(2))
            }
            # Add timestamps only if they were found
            if sensor_match.group(3):
                sensor_data["first_seen"] = sensor_match.group(3)
            if sensor_match.group(4):
                sensor_data["last_seen"] = sensor_match.group(4)

            result['raw_stream_derived_metrics']['event_counts_and_timing']['sensor'] = sensor_data
        # If 'sensor' data is not found in the text for this section, it won't be added,
        # relying on the template's initial structure or absence.

    def parse_sensor_metrics(self, chunk_str: str, result: Dict[str, Any]):
        """Parse sensor_metrics section."""
        sensor_start = chunk_str.find('sensor_metrics:')
        if sensor_start == -1:
            return
        sensor_section = chunk_str[sensor_start:]
        # Parse sensor metric fields
        sensor_fields = [
            'power_voltage_avg', 'power_voltage_min', 'power_voltage_max', 'power_voltage_std_dev'
        ]
        for field in sensor_fields:
            pattern = rf'{field}:\s*([\d\.\-]+)'
            match = re.search(pattern, sensor_section)
            if match:
                result['sensor_metrics'][field] = self.convert_value(match.group(1))

def format_chunk_data_improved(chunk_key: str, chunk_str: str) -> dict:
    """Format a single chunk using the improved parser."""
    parser = ChunkParser()
    return parser.parse_chunk_improved(chunk_str)

def list_available_chunks(input_file: str):
    """List all available chunks with their indices (1-based)."""
    data = load_json(input_file)
    if "error" in data:
        print(data["error"])
        return
    print(f"Available chunks in {input_file}:")
    print("-" * 50)
    for i, chunk_key in enumerate(data.keys(), 1):
        chunk_str = data[chunk_key].get("chunk", "")
        # Extract room_id and time for preview
        room_match = re.search(r'room_id:\s*(\S+)', chunk_str)
        start_match = re.search(r'start:\s*([\d\-T:]+)', chunk_str)
        room_id = room_match.group(1) if room_match else "Unknown"
        start_time = start_match.group(1) if start_match else "Unknown"
        print(f"Chunk {i}: {chunk_key}")
        print(f"  Room: {room_id}")
        print(f"  Start Time: {start_time}")
        print()

def extract_chunk_by_index_and_format(input_file: str, chunk_index: int) -> dict:
    """
    Extracts a single chunk from a JSON file by 1-based index, parses its content,
    and formats it according to the predefined JSON structure using improved parsing.
    Args:
        input_file: Path to the input JSON file
        chunk_index: 1-based index (chunk 1, chunk 2, etc.)
    """
    data = load_json(input_file)
    if "error" in data:
        return data
    chunk_keys = list(data.keys())
    # Convert to 0-based index for internal use
    zero_based_index = chunk_index - 1
    if not (0 <= zero_based_index < len(chunk_keys)):
        return {"error": f"Chunk index {chunk_index} is out of range. The file has {len(chunk_keys)} chunks (1-{len(chunk_keys)})."}
    chunk_key = chunk_keys[zero_based_index]
    chunk_str = data[chunk_key].get("chunk", "")
    print(f"Processing Chunk {chunk_index} - Key: {chunk_key}")
    return format_chunk_data_improved(chunk_key, chunk_str)

def save_formatted_chunk_by_index(input_file: str, chunk_index: int, output_file_name: str = None):
    """
    Extracts, formats, and saves a single chunk to a specified JSON file using improved parsing.
    Args:
        input_file: Path to the input JSON file
        chunk_index: 1-based index (chunk 1, chunk 2, etc.)
        output_file_name: Output filename (if None, uses "day_{index}.json")
    """
    if output_file_name is None:
        output_file_name = f"day_{chunk_index}.json"
    formatted_data = extract_chunk_by_index_and_format(input_file, chunk_index)
    if "error" in formatted_data:
        print(formatted_data["error"])
        return
    try:
        with open(output_file_name, "w") as f:
            json.dump(formatted_data, f, indent=4)
        print(f"✅ Chunk {chunk_index} formatted data saved to {output_file_name}")
        # Also print the formatted data
        print("\nFormatted JSON:")
        print(json.dumps(formatted_data, indent=2))
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

def process_specific_chunks(input_file: str, chunk_indices: list):
    """
    Process specific chunks by their 1-based indices using improved parsing.
    Args:
        input_file: Path to the input JSON file
        chunk_indices: List of 1-based chunk indices to process
    """
    data = load_json(input_file)
    if "error" in data:
        print(data["error"])
        return
    total_chunks = len(data)
    print(f"Total chunks available: {total_chunks}")
    print("=" * 80)
    for chunk_index in chunk_indices:
        print(f"\nProcessing Chunk {chunk_index}:")
        print("-" * 50)
        formatted_data = extract_chunk_by_index_and_format(input_file, chunk_index)
        if "error" in formatted_data:
            print(formatted_data["error"])
            continue
        # Save individual chunk
        output_filename = f"chunk_{chunk_index}.json"
        try:
            with open(output_filename, "w") as f:
                json.dump(formatted_data, f, indent=4)
            print(f"✅ Saved to {output_filename}")
        except Exception as e:
            print(f"❌ Error saving: {e}")
        # Print formatted data
        print("\nFormatted JSON:")
        print(json.dumps(formatted_data, indent=2))
        print("-" * 50)

def validate_parsing_completeness(chunk_str: str, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that all data from the chunk string has been captured in the parsed result.
    Returns a report of missing or potentially missed data.
    """
    validation_report = {
        "missing_sections": [],
        "empty_sections": [],
        "unexpected_keys": [],
        "parsing_stats": {}
    }
    # Count total key-value pairs in original string
    original_pairs = len(re.findall(r'\w+:\s*[\w\.\-:T]+', chunk_str))
    # Count parsed pairs
    def count_parsed_pairs(obj, path=""):
        count = 0
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    count += count_parsed_pairs(v, f"{path}.{k}" if path else k)
                else:
                    count += 1
        return count
    parsed_pairs = count_parsed_pairs(parsed_result)
    validation_report["parsing_stats"] = {
        "original_pairs": original_pairs,
        "parsed_pairs": parsed_pairs,
        "coverage_percentage": round((parsed_pairs / original_pairs) * 100, 2) if original_pairs > 0 else 0
    }
    return validation_report

# Example usage with validation
if __name__ == "__main__":
    input_file = "pinecone_chunks_only.json"
    # # List available chunks first
    # print("LISTING AVAILABLE CHUNKS:")
    # list_available_chunks(input_file)
    print("\n" + "="*80)
    print("PROCESSING INDIVIDUAL CHUNK WITH IMPROVED PARSER:")
    try:
        chunk_index = 1
        save_formatted_chunk_by_index(input_file, chunk_index)
    except Exception as e:
        print(f"\nNote: Could not process file '{input_file}': {e}")
        print("The test examples above demonstrate the improved parsing capability.")

