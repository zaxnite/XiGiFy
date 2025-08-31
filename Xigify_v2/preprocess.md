# Sensor Data Processing & Analysis – Function Documentation

This module processes raw IoT sensor data (lights, AC, fan, occupancy, etc.) into structured insights. It calculates **energy efficiency, comfort, predictive risk, sensor health, and aggregated events**.

---

## 🔋 Energy Efficiency Functions

### `calculate_energy_efficiency_score(data: Dict[str, Any]) -> float`

* **Purpose**: Computes an energy efficiency score (0–100).
* **Logic**:

  * Calculates total kWh (light, AC, fan).
  * Normalizes consumption (lower is better).
  * Evaluates device usage efficiency against occupancy.
  * Applies penalties for waste (AC/lights on while unoccupied, AC on with window open).
  * Produces weighted final score.
* **Returns**: Energy efficiency score (0–100).

---

### `calculate_total_energy_consumption(data: Dict[str, Any]) -> float`

* **Purpose**: Sums energy consumption from light, AC, and fan.
* **Returns**: Total kWh used.

---

### `calculate_waste_indicators(data: Dict[str, Any]) -> Dict[str, float]`

* **Purpose**: Quantifies waste behavior.
* **Metrics**:

  * AC running unoccupied
  * Lights on unoccupied
  * AC on with window open
  * Combined waste score
* **Returns**: Dictionary of waste indicators.

---

### `analyze_energy_efficiency(data: Dict[str, Any]) -> Dict[str, Any]`

* **Purpose**: Full energy efficiency analysis wrapper.
* **Outputs**:

  * Efficiency score
  * Total energy (kWh)
  * Waste breakdown
  * Device-specific energy usage

---

## 🌡️ Comfort & Wellbeing Functions

### `calculate_comfort_score(data: Dict[str, Any]) -> float`

* **Purpose**: Computes comfort score (0–100).
* **Logic**:

  * Light usage vs occupancy alignment.
  * AC runtime stability (variance vs average).
  * Occupancy consistency.
* **Returns**: Weighted comfort score.

---

### `calculate_alignment_metrics(data: Dict[str, Any]) -> Dict[str, float]`

* **Purpose**: Measures alignment between occupancy and device usage.
* **Metrics**:

  * Occupancy-to-light ratio
  * Occupancy-to-AC ratio
  * Device usage while unoccupied
  * AC usage with window open

---

### `analyze_comfort_wellbeing(data: Dict[str, Any]) -> Dict[str, Any]`

* **Purpose**: Full comfort and wellbeing analysis.
* **Outputs**:

  * Comfort score
  * Alignment metrics
  * Occupancy statistics (avg duration, variance)

---

## 🔮 Predictive & Prescriptive Functions

### `check_predictive_flags(data: Dict[str, Any]) -> Dict[str, bool]`

* **Purpose**: Detects predictive anomalies.
* **Flags**:

  * Power drift
  * Rapid cycling or sensor stuck patterns
  * Seasonal deviations
  * AC usage trending upward

---

### `calculate_total_override_count(data: Dict[str, Any]) -> int`

* **Purpose**: Sums all manual overrides (light, AC, etc.).

---

### `check_frequent_overrides(data: Dict[str, Any]) -> bool`

* **Purpose**: Detects excessive manual overrides (>3).

---

### `analyze_predictive_prescriptive(data: Dict[str, Any]) -> Dict[str, Any]`

* **Purpose**: Prescriptive analysis combining predictive flags + overrides.
* **Outputs**:

  * Predictive anomaly flags
  * Override stats
  * Risk score (weighted by issues severity)

---

## 🩺 Sensor Health Functions

### `analyze_sensor_health(data: Dict[str, Any]) -> Dict[str, Any]`

* **Purpose**: Checks health of sensors.
* **Detects**:

  * Low battery
  * Signal degradation
* **Outputs**:

  * Flags for issues
  * Affected sensors
  * Total health events + detailed log

---

## 📊 Event Aggregation

### `calculate_aggregated_events(data: Dict[str, Any]) -> Dict[str, Dict[str, int]]`

* **Purpose**: Counts ON/OFF (or open/close/detect) cycles.
* **Devices**: Light, AC, Fan, Window, Door, Occupancy.
* **Returns**: Per-device event summary.

---

## 🔄 Preprocessing & Data Handling

### `preprocess_sensor_data(data: Dict[str, Any]) -> Dict[str, Any]`

* **Purpose**: Main entrypoint for analysis.
* **Adds**:

  * Energy efficiency analysis
  * Comfort wellbeing analysis
  * Predictive-prescriptive analysis
  * Sensor health analysis
  * Aggregated events

---

### `load_json_data(file_path: str) -> Dict[str, Any]`

* **Purpose**: Loads raw JSON file.

### `save_json_data(data: Dict[str, Any], file_path: str) -> None`

* **Purpose**: Saves JSON with formatting.

### `process_single_file(input_path: str) -> Dict[str, Any]`

* **Purpose**: Processes one JSON file.
* **Steps**:

  * Load raw file
  * Preprocess data
  * Save output with `_preprocessed.json` suffix

### `process_json_string(json_string: str) -> str`

* **Purpose**: Processes JSON data from a string directly.
* **Returns**: Preprocessed JSON string.

---

## ▶️ Example Usage

When run directly:

* Processes `day_5.json` (if available).
* Prints **Efficiency Score, Comfort Score, Risk Score**.
* Handles missing file or unexpected errors gracefully.

