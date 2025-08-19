````markdown
# XiGiFy 

A Python-based tool that analyzes structured IoT sensor data from smart rooms using Google's Gemini AI to detect energy inefficiencies, anomalies, behavioral patterns, and actionable insights.

---

## Features
- Validates IoT sensor data structure.
- Summarizes key metrics:
  - Device usage (lights, AC, fan, occupancy, doors/windows)  
  - Energy consumption and efficiency indicators  
  - Device activity cycles and runtime statistics  
  - Anomaly detection and sensor health events  
- AI-generated reports:
  - **Customer reports** – simplified, action-focused  
  - **Internal reports** – detailed, technical  
- Saves timestamped reports and input data for traceability.

---

## Getting Started
1. Clone the repo:
   ```bash
   git clone https://github.com/zaxnite/XiGiFy.git
   cd XiGiFy
````

2. (Optional) Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Add a `.env` file with:

   ```text
   GEMINI_API_KEY=your_api_key_here
   ```

---

## Usage

Run the analysis:

```bash
python xigify.py --day day_4 --report-type customer --input-dir . --output-dir .
```

**Workflow:**

1. Load JSON (e.g., `day_4.json`).
2. Validate and preprocess data.
3. Analyze with Gemini AI.
4. Save report in the appropriate folder.

---

## Command-Line Options

| Flag            | Description                                      | Default            |
| --------------- | ------------------------------------------------ | ------------------ |
| `--day`         | Select JSON file (e.g., `day_4`)                 | `day_4`            |
| `--report-type` | `customer` or `internal` report                  | `customer`         |
| `--input-dir`   | Directory with JSON files                        | Current directory  |
| `--output-dir`  | Directory for generated reports                  | Current directory  |
| `--max-retries` | Max retries for Gemini API calls                 | `3`                |
| `--model`       | Gemini model to use                              | `gemini-1.5-flash` |
| `--log-level`   | Logging level (`DEBUG`, `INFO`, `WARNING`, etc.) | `INFO`             |

---

## Outputs

Reports are saved in:

```
./Reports-Customer/day_4_customer.txt
./Reports-Internal/day_4_internal.txt
```

---

## Logging

* Logs are stored in `iot_analysis.log` and shown in console.
* Verbosity adjustable with `--log-level`.

---

## Data Format

JSON input must contain:

* `room_id`
* `summary_window` → `start`, `end`
* `ml_abstracted_metrics` → usage features, cross-sensor ratios, run-length statistics
* `raw_stream_derived_metrics` → event counts, energy metrics, etc.

---

## Error Handling

* Invalid API key → exits with message
* Malformed JSON → error + exit
* API failures → retries with backoff
* Unexpected errors are logged and handled gracefully

---

## Contributing

1. Fork the repo and create a branch (`git checkout -b feature-name`)
2. Commit with clear messages
3. Open a pull request

---

## License

MIT License. See [LICENSE](LICENSE).

```

