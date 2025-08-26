// XiGiFy Preprocessing Functions for n8n
// Translation of Python preprocessing functions to JavaScript

// Constants
const DEFAULT_PRECISION = 1;
const DEFAULT_PERCENTAGE_PRECISION = 1;

// Report formatting constants
const UNKNOWN_VALUE = "Unknown";
const NOT_AVAILABLE = "N/A";
const NO_HEALTH_ISSUES = "No sensor health issues detected";
const NO_MANUAL_OVERRIDES = "No manual overrides detected";

// Validation constants
const REQUIRED_TOP_LEVEL_FIELDS = [
    "room_id",
    "summary_window", 
    "ml_abstracted_metrics",
    "raw_stream_derived_metrics"
];

const REQUIRED_ML_METRICS_FIELDS = [
    "usage_features",
    "cross_sensor_ratios", 
    "run_length_statistics"
];

const REQUIRED_RAW_METRICS_FIELDS = [
    "event_counts_and_timing",
    "energy_proxy_metrics"
];

/**
 * Validate that the JSON has the required structure for IoT sensor data.
 * @param {Object} data - Parsed JSON data
 * @returns {Object} - n8n result object with validation status and message
 */
function validateJsonStructure(data) {
    const errors = [];
    
    // Check top-level fields
    for (const field of REQUIRED_TOP_LEVEL_FIELDS) {
        if (!(field in data)) {
            errors.push(`Missing required field: ${field}`);
        }
    }
    
    // Check ml_abstracted_metrics structure
    const mlMetrics = data.ml_abstracted_metrics || {};
    for (const field of REQUIRED_ML_METRICS_FIELDS) {
        if (!(field in mlMetrics)) {
            errors.push(`Missing required ml_abstracted_metrics field: ${field}`);
        }
    }
    
    // Check raw_stream_derived_metrics structure
    const rawMetrics = data.raw_stream_derived_metrics || {};
    for (const field of REQUIRED_RAW_METRICS_FIELDS) {
        if (!(field in rawMetrics)) {
            errors.push(`Missing required raw_stream_derived_metrics field: ${field}`);
        }
    }
    
    return {
        isValid: errors.length === 0,
        errors: errors,
        message: errors.length === 0 ? "JSON structure validation passed" : "Validation failed"
    };
}

/**
 * Safely format percentage values
 * @param {number|null|undefined} value 
 * @returns {string}
 */
function safePercentage(value) {
    if (value === null || value === undefined) {
        return NOT_AVAILABLE;
    }
    return `${(value * 100).toFixed(DEFAULT_PERCENTAGE_PRECISION)}%`;
}

/**
 * Safely format float values
 * @param {number|null|undefined} value 
 * @param {number} precision 
 * @returns {string}
 */
function safeFloat(value, precision = DEFAULT_PRECISION) {
    if (value === null || value === undefined) {
        return NOT_AVAILABLE;
    }
    return parseFloat(value).toFixed(precision);
}

/**
 * Safely format integer values
 * @param {number|null|undefined} value 
 * @returns {string}
 */
function safeInt(value) {
    if (value === null || value === undefined) {
        return NOT_AVAILABLE;
    }
    return value.toString();
}

/**
 * Extract and format key metrics from the sensor data for analysis.
 * @param {Object} data - Raw JSON sensor data
 * @returns {Object} - n8n result object with formatted summary and metadata
 */
function preprocessSensorData(data) {
    try {
        const roomId = data.room_id || UNKNOWN_VALUE;
        const window = data.summary_window || {};
        const startTime = window.start || UNKNOWN_VALUE;
        const endTime = window.end || UNKNOWN_VALUE;
        
        // Extract ML metrics
        const mlMetrics = data.ml_abstracted_metrics || {};
        const usage = mlMetrics.usage_features || {};
        const crossRatios = mlMetrics.cross_sensor_ratios || {};
        const runStats = mlMetrics.run_length_statistics || {};
        const driftData = mlMetrics.rolling_statistics_and_drift || {};
        const deviationFlags = mlMetrics.pattern_deviation_flags || {};
        const seasonal = mlMetrics.seasonal_pattern_summaries || {};
        const intervals = mlMetrics.inter_event_distributions || {};
        
        // Extract raw metrics
        const rawMetrics = data.raw_stream_derived_metrics || {};
        const eventCounts = rawMetrics.event_counts_and_timing || {};
        const energy = rawMetrics.energy_proxy_metrics || {};
        const healthEvents = rawMetrics.sensor_health_events || [];
        const manualOverrides = rawMetrics.manual_override_counters || [];
        
        // Calculate total energy
        const totalEnergy = Object.values(energy).reduce((sum, val) => {
            return typeof val === 'number' ? sum + val : sum;
        }, 0);
        
        // Format the summary
        let summary = `
ROOM: ${roomId}
TIME PERIOD: ${startTime} to ${endTime}

USAGE PATTERNS:
- Light on: ${safePercentage(usage.light_on_pct)}
- AC on: ${safePercentage(usage.ac_on_pct)}
- Fan on: ${safePercentage(usage.fan_on_pct)}
- Occupancy detected: ${safePercentage(usage.occupancy_detected_pct)}
- Window open: ${safePercentage(usage.window_open_pct)}
- Door open: ${safePercentage(usage.door_open_pct)}

EFFICIENCY METRICS:
- AC on while unoccupied: ${safePercentage(crossRatios.ac_on_while_unoccupied_pct)}
- Light on while unoccupied: ${safePercentage(crossRatios.light_on_while_unoccupied_pct)}
- AC on while window open: ${safePercentage(crossRatios.ac_on_while_window_open_pct)}

ENERGY CONSUMPTION:
- Light: ${safeFloat(energy.light_kwh)} kWh
- AC: ${safeFloat(energy.ac_kwh)} kWh
- Fan: ${safeFloat(energy.fan_kwh)} kWh
- Total: ${safeFloat(totalEnergy)} kWh

DEVICE ACTIVITY:
- Light cycles: ${safeInt(eventCounts.light?.on_count)} on/${safeInt(eventCounts.light?.off_count)} off
- AC cycles: ${safeInt(eventCounts.ac?.on_count)} on/${safeInt(eventCounts.ac?.off_count)} off
- Fan cycles: ${safeInt(eventCounts.fan?.on_count)} on/${safeInt(eventCounts.fan?.off_count)} off
- Door events: ${safeInt(eventCounts.door?.open_count)} opens
- Occupancy events: ${safeInt(eventCounts.occupancy?.detected_count)} detections

RUN LENGTH STATISTICS:
- Light avg runtime: ${safeFloat(runStats.light?.avg_min)} min
- AC avg runtime: ${safeFloat(runStats.ac?.avg_min)} min
- Fan avg runtime: ${safeFloat(runStats.fan?.avg_min)} min
- Occupancy avg duration: ${safeFloat(runStats.occupancy?.avg_min)} min

POWER DRIFT & ANOMALIES:
- AC power average: ${safeFloat(driftData.ac_power_avg_w5m)}W
- AC power drift detected: ${driftData.ac_power_drift_flag || false}
- Light power average: ${safeFloat(driftData.light_power_avg_w5m)}W
- Light power drift detected: ${driftData.light_power_drift_flag || false}

ANOMALY FLAGS:
- AC rapid cycling: ${deviationFlags.ac_rapid_cycle_flag || false}
- Light stuck: ${deviationFlags.light_stuck_flag || false}
- Occupancy sensor stuck: ${deviationFlags.occupancy_sensor_stuck_flag || false}

SEASONAL PATTERNS:
- AC vs weekly pattern: ${safeFloat(seasonal.ac_vs_weekly_pct)}% deviation
- Light vs weekly pattern: ${safeFloat(seasonal.light_vs_weekly_pct)}% deviation

INTERVAL STATISTICS (p50/p90/p99):
- Door open intervals: ${safeInt(intervals.door_open_intervals_s?.p50)}/${safeInt(intervals.door_open_intervals_s?.p90)}/${safeInt(intervals.door_open_intervals_s?.p99)} seconds
- Occupancy intervals: ${safeInt(intervals.occupancy_intervals_s?.p50)}/${safeInt(intervals.occupancy_intervals_s?.p90)}/${safeInt(intervals.occupancy_intervals_s?.p99)} seconds

SENSOR HEALTH:
`;
        
        if (healthEvents.length > 0) {
            for (const event of healthEvents) {
                summary += `- ${event.sensor || UNKNOWN_VALUE} sensor: ${event.event || UNKNOWN_VALUE} at ${event.timestamp || UNKNOWN_VALUE}\n`;
            }
        } else {
            summary += `- ${NO_HEALTH_ISSUES}\n`;
        }
        
        summary += "\nMANUAL OVERRIDES:\n";
        if (manualOverrides.length > 0) {
            for (const override of manualOverrides) {
                summary += `- ${override.device || UNKNOWN_VALUE}: ${safeInt(override.count)} overrides\n`;
            }
        } else {
            summary += `- ${NO_MANUAL_OVERRIDES}\n`;
        }
        
        return {
            success: true,
            summary: summary,
            metadata: {
                roomId: roomId,
                startTime: startTime,
                endTime: endTime,
                totalEnergy: totalEnergy,
                processedAt: new Date().toISOString()
            }
        };
        
    } catch (error) {
        return {
            success: false,
            error: error.message,
            summary: null,
            metadata: null
        };
    }
}

/**
 * Get the appropriate analysis prompt based on report type.
 * @param {string} preprocessedData - Formatted sensor data summary
 * @param {string} reportType - Type of report ('customer' or 'internal')
 * @returns {Object} - n8n result object with prompt and metadata
 */
function getAnalysisPrompt(preprocessedData, reportType) {
    if (!preprocessedData) {
        return {
            success: false,
            error: "Preprocessed data is required",
            prompt: null
        };
    }
    
    if (!['customer', 'internal'].includes(reportType)) {
        return {
            success: false,
            error: "Invalid report type. Must be 'customer' or 'internal'",
            prompt: null
        };
    }
    
    let prompt;
    
    if (reportType === "internal") {
        prompt = `You are an expert IoT data analyst specializing in smart building systems and energy efficiency.  
            Analyze the following sensor data from a smart room and generate a comprehensive, accurate, and actionable report.

            SENSOR DATA:
            ${preprocessedData}

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
        `;
    } else { // customer report
        prompt = `You are a smart building efficiency advisor. Your role is to turn sensor data into a clear, engaging report for facility managers, property owners, and building occupants.

                    Focus on outcomes that matter to people: **comfort, savings, convenience, and sustainability**. Use simple, professional, and friendly language—like explaining over coffee. Avoid jargon, formulas, acronyms, and technical units unless absolutely necessary. Translate numbers into real-world terms (e.g., cost or relatable usage).

                    **SENSOR DATA:**
                    ${preprocessedData}

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

                    `;
    }
    
    return {
        success: true,
        prompt: prompt,
        reportType: reportType,
        generatedAt: new Date().toISOString()
    };
}

// n8n Export - Main execution function
function executePreprocessing() {
    // Get input data from n8n context
    const inputData = $input.all();
    
    if (!inputData || inputData.length === 0) {
        return {
            error: "No input data provided",
            success: false
        };
    }
    
    const results = [];
    
    for (let i = 0; i < inputData.length; i++) {
        const item = inputData[i];
        const sensorData = item.json;
        
        // Validate the JSON structure
        const validation = validateJsonStructure(sensorData);
        
        if (!validation.isValid) {
            results.push({
                json: {
                    success: false,
                    error: "Validation failed",
                    validationErrors: validation.errors,
                    itemIndex: i
                }
            });
            continue;
        }
        
        // Preprocess the sensor data
        const preprocessResult = preprocessSensorData(sensorData);
        
        if (!preprocessResult.success) {
            results.push({
                json: {
                    success: false,
                    error: preprocessResult.error,
                    itemIndex: i
                }
            });
            continue;
        }
        
        // Generate prompts for both report types
        const internalPrompt = getAnalysisPrompt(preprocessResult.summary, 'internal');
        const customerPrompt = getAnalysisPrompt(preprocessResult.summary, 'customer');
        
        results.push({
            json: {
                success: true,
                itemIndex: i,
                preprocessed: {
                    summary: preprocessResult.summary,
                    metadata: preprocessResult.metadata
                },
                prompts: {
                    internal: internalPrompt,
                    customer: customerPrompt
                },
                validation: validation
            }
        });
    }
    
    return results;
}

// For n8n: Execute the preprocessing
return executePreprocessing();