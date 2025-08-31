# 🧠 Deterministic AI Interpretation Engine – Function Documentation

This script processes **preprocessed IoT sensor data** and generates **deterministic, minimal-variation interpretations** using the **Google Gemini API**.
It ensures **consistency across runs** while adapting outputs for **customer** or **internal reports**.

---

## ⚙️ Core Inference Functions

### `generate_inference_guidelines(data: Dict[str, Any]) -> Tuple[str, str>`

* **Purpose**: Produces **deterministic inference guidelines** (energy + comfort).
* **Logic**:

  * Uses efficiency score, comfort score, device ratios, occupancy alignment, AC stability, and overrides.
  * Assigns **HIGH / MEDIUM / LOW priority** rules.
* **Returns**:

  * `energy_inference_text` – 3 formatted energy guidelines
  * `comfort_inference_text` – 3 formatted comfort guidelines

---

### `get_deterministic_prompt_template(report_type: str) -> str`

* **Purpose**: Provides **structured prompt template** for Gemini AI.
* **Modes**:

  * `customer`: Simple, outcome-focused, no technical jargon.
  * `internal`: Technical, includes metrics and thresholds.
* **Returns**: Deterministic, structured prompt with placeholders.

---

### `extract_interpretation_data(sensor_data: Dict[str, Any]) -> Dict[str, Any]`

* **Purpose**: Extracts **key metrics** from preprocessed sensor JSON.
* **Extracts**:

  * Room ID & analysis window
  * Efficiency score, kWh, waste indicators
  * Comfort score, occupancy-light alignment, AC stability, temp variation
  * Predictive flags & drift trends
  * Overrides & sensor health
  * Device usage ratios
* **Returns**: Compact dictionary with normalized values.

---

### `create_deterministic_prompt(data: Dict[str, Any], report_type: str) -> str`

* **Purpose**: Builds a **final deterministic prompt** for Gemini.
* **Steps**:

  * Calls `generate_inference_guidelines()`
  * Adds guidelines into extracted data
  * Fills `get_deterministic_prompt_template()`
* **Returns**: Fully formatted prompt string.

---

## 🤖 Gemini AI Integration

### `initialize_gemini_model(api_key: str, model_name: str)`

* **Purpose**: Configures Gemini with **deterministic settings**.
* **Config**:

  * `temperature=0.0`, `top_p=1.0`, `top_k=1`, single candidate.
* **Returns**: Gemini `GenerativeModel` instance (or `None` on failure).

---

### `interpret_with_gemini(model, prompt: str, max_retries: int) -> Optional[str]`

* **Purpose**: Sends deterministic prompt to Gemini API.
* **Features**:

  * Retries with exponential backoff on errors.
  * Logs success/failure states.
* **Returns**: Gemini response text (or `None` if all retries fail).

---

## 📑 Response Parsing & Formatting

### `parse_interpretation_response(response: str) -> Dict[str, str]`

* **Purpose**: Parses structured Gemini response into **sections**.
* **Sections Extracted**:

  * Energy Efficiency
  * Comfort Wellbeing
  * Predictive Insights
  * Energy Rules
  * Comfort Rules
* **Fallback**: If structured format missing, extracts line-by-line.

---

### `format_rules_for_display(rules_text: str, rule_prefix: str) -> str`

* **Purpose**: Formats rules into **clean multi-line list**.
* **Fixes**:

  * Avoids splitting on decimals (e.g., `37.5%`).
  * Supports numbered or priority-based rules.
* **Returns**: Rules as `Rule 1: ...`, `Rule 2: ...`.

---

### `save_interpretation_results(...) -> bool`

* **Purpose**: Writes **final interpretation report** to file.
* **Report Includes**:

  * Metadata (timestamp, room ID, source file, report type)
  * Extracted data summary
  * AI interpretation (energy, comfort, predictive)
  * Deterministic rules (energy + comfort)
* **Returns**: `True` if saved, else `False`.

---

## 📂 File Handling & CLI

### `load_json_file(file_path: Path) -> Optional[Dict[str, Any]]`

* **Purpose**: Loads sensor JSON from disk.
* **Returns**: Parsed dictionary or `None` on error.

---

### `get_output_file_path(day: str, report_type: str, output_dir: Path) -> Path`

* **Purpose**: Generates report path based on day + report type.
* **Returns**: Path object (e.g., `Reports-Customer/day_5_customer.txt`).

---

### `parse_arguments() -> argparse.Namespace`

* **Purpose**: Parses CLI arguments.
* **Args**:

  * `--day`: Day ID (default: `day_5`)
  * `--report-type`: `customer` or `internal`
  * `--input-dir`: Directory for input JSON
  * `--output-dir`: Directory for reports
  * `--max-retries`: Retry attempts for Gemini API

---

## 🧭 Execution Flow

1. Load preprocessed JSON file (`*_preprocessed.json`).
2. Extract interpretation data (`extract_interpretation_data`).
3. Generate deterministic prompt (`create_deterministic_prompt`).
4. Send to Gemini API (`interpret_with_gemini`).
5. Parse structured response (`parse_interpretation_response`).
6. Save formatted report (`save_interpretation_results`).

---

⚡ This design ensures **identical output for identical input**, making it suitable for **auditable IoT energy and comfort analysis reports**.

