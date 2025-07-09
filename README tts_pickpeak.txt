
================================================================================
Thermal Probe Peak Picker (tts_pickpeak.py)
================================================================================

Version: 1.0 - Interactive Peak Detection for Temperature Time Series

Author: Timothy Wu
Created: 7/8/2025
Last Updated: 7/8/2025

This tool identifies temperature peaks and troughs in shallow and deep thermal 
probe data using multiple detection algorithms. It features an interactive 
Dash-based GUI and outputs formatted data for further analysis.

================================================================================
INSTALLATION REQUIREMENTS
================================================================================

1. PYTHON VERSION
   - Python 3.7 or higher
   - Download from: https://www.python.org/downloads/

2. REQUIRED LIBRARIES
   Install all required packages using pip:

   pip install dash>=2.9.0 plotly pandas numpy scipy PyWavelets

   Alternatively, install libraries individually:
   pip install dash>=2.9.0      # Web application framework
   pip install plotly           # Interactive plotting
   pip install pandas           # Data manipulation
   pip install numpy            # Numerical computations
   pip install scipy            # Scientific computing
   pip install PyWavelets       # Wavelet analysis

3. VERIFY INSTALLATION
   Open Python and run:
   import dash, plotly, pandas, numpy, scipy, pywt
   print("All libraries installed successfully!")

================================================================================
QUICK START GUIDE
================================================================================

1. SAVE AND RUN
   - Save `tts_pickpeak.py` to a directory of your choice
   - Open terminal/command prompt and run:
     python tts_pickpeak.py

2. INTERFACE LAUNCH
   - Open your browser and go to: http://127.0.0.1:8050/

3. ANALYSIS WORKFLOW
   - Upload a filtered CSV file containing `WaterDay`, `Shallow.Temp.Filt`, `Deep.Temp.Filt`
   - Select a peak detection method (Combined is default)
   - Click "Detect Peaks" to begin
   - Use manual tools to refine selections
   - Run "Check Errors" to validate
   - Export peak and formatted data using the export buttons

================================================================================
PARAMETER FILE CONFIGURATION (tts_pickpeak.par)
================================================================================

The parameter file allows you to save and load application settings. Create a
file named `tts_pickpeak.par` in the same folder as `tts_pickpeak.py`.

EXAMPLE PARAMETER FILE:
-----------------------
[PARAMETERS]
data_file = 
sensor_spacing = 0.18
target_period = 1.0
search_tolerance = 0.1
slope_threshold = 0.001
min_distance = 20
prominence_factor = 0.15
output_folder = peak_analysis_output
peak_method = combined
shallow_color = blue
deep_color = red
peak_size = 12
trough_size = 10
line_width = 2

KEY SETTINGS:
-------------
- sensor_spacing: Distance between sensors (in meters)
- target_period: Time between expected peaks (in days)
- search_tolerance: Allowed deviation for bootstrap detection (days)
- slope_threshold: Minimum slope (°C/min) for derivative method
- prominence_factor: Sensitivity of peak prominence (0-1 scale)
- peak_method: Detection algorithm (e.g., combined, wavelet, scipy, etc.)
- output_folder: Directory for output files

================================================================================
INPUT DATA FORMAT REQUIREMENTS
================================================================================

CSV FILE STRUCTURE:
Your CSV file should include these columns:

WaterDay,Shallow.Temp.Filt,Deep.Temp.Filt
312.000000000,0.007140,0.018250
312.000694302,0.001050,0.015070
...

COLUMN DESCRIPTIONS:
- WaterDay: Decimal day from start of water year (e.g., 312.35)
- Shallow.Temp.Filt: Filtered temperature from shallow probe
- Deep.Temp.Filt: Filtered temperature from deep probe


================================================================================
PEAK DETECTION METHODS (DETAILED)
================================================================================

This application offers six distinct peak detection approaches, each tailored to 
different data characteristics and analysis goals. You can switch between them 
via the “Peak Detection Method” dropdown in the interface.

--------------------------------------------------------------------------------
1. SciPy find_peaks 
--------------------------------------------------------------------------------

HOW IT WORKS:
- Utilizes `scipy.signal.find_peaks` with custom adaptive parameters
- Removes DC offset by centering temperature data around zero
- Computes peak prominence using standard deviation of signal
- Dynamically relaxes criteria if too few peaks are found

BEST FOR:
- Clean data with relatively consistent peak heights

KEY PARAMETERS:
- `distance`: Minimum number of samples between peaks
- `prominence`: Minimum prominence value for valid peaks

EXAMPLE:
```python
from scipy.signal import find_peaks
peaks, _ = find_peaks(data, distance=min_distance, prominence=prom_value)
```

--------------------------------------------------------------------------------
2. Wavelet-based Detection
--------------------------------------------------------------------------------

HOW IT WORKS:
- Performs Continuous Wavelet Transform (CWT) using the Mexican hat wavelet
- Extracts peaks as ridgelines in the wavelet coefficient space across scales
- Captures features missed by fixed-window methods
- Scales are tuned to expected peak periodicity

BEST FOR:
- Noisy datasets or when peak widths vary significantly

ALGORITHM EXCERPT:
```python
coefficients = pywt.cwt(data, widths, 'mexh')[0]
peaks = identify_ridgelines(coefficients)
```

NOTES:
- Insensitive to minor noise
- Computationally heavier than other methods
- Will take some time to run

--------------------------------------------------------------------------------
3. Custom Derivative Method
--------------------------------------------------------------------------------

HOW IT WORKS:
- Smooths signal using a Gaussian filter
- Calculates first and second derivatives
- Detects zero-crossings in the first derivative
- Uses second derivative to confirm local max/min classification

PEAK LOGIC:
- Peak: First derivative transitions + → −
- Trough: First derivative transitions − → +

BEST FOR:
- Gradual peak transitions, noisy or smoothed data

ADAPTATIONS:
- Gaussian smoothing kernel size scales with signal length
- Optional slope threshold filters weak transitions

ADVANTAGES:
- High selectivity with noise rejection
- Can identify subtle but real thermal responses

--------------------------------------------------------------------------------
4. Prominence-based Detection
--------------------------------------------------------------------------------

HOW IT WORKS:
- Calculates data range: `max(data) - min(data)`
- Applies `prominence_factor` (0–1) to set detection threshold
- Filters all peaks below threshold

FORMULA:
```
min_prominence = (max(data) - min(data)) × prominence_factor
```

BEST FOR:
- Clean data with strong thermal signal variation

PROS:
- Simple and fast
- Easy to tune with just one slider

CONS:
- May miss small but meaningful peaks if threshold is too high

--------------------------------------------------------------------------------
5. Combined Methods (Most Sensitive)
--------------------------------------------------------------------------------

HOW IT WORKS:
- Merges outputs from multiple techniques:
  1. Local maxima (baseline detection)
  2. Multiple runs of `scipy.find_peaks` with varying prominence (0.05–0.3)
  3. Unconstrained detection (no distance filter)
  4. All results merged and de-duplicated

- Final peaks filtered by user-specified minimum distance

BEST FOR:
- Exploratory work or ensuring completeness
- When you're unsure which algorithm works best

ADVANTAGE:
- Least likely to miss a peak
- Good starting point for refining with manual edits or bootstrap

--------------------------------------------------------------------------------
6. Bootstrap from Manual Selection
--------------------------------------------------------------------------------

HOW IT WORKS:
- User manually identifies at least 2–3 “trusted” peaks
- Tool estimates average period and propagates timing pattern forward/backward
- Searches for local maxima around expected times within a tolerance window

PROCESS:
1. User clicks to add 2–3 peaks
2. App computes:
   ```
   avg_period = mean(diff(manual_peak_times))
   next_peak = last_peak + avg_period
   ```
3. Looks within ±`search_tolerance` days for the actual peak

BEST FOR:
- Strongly periodic signals
- When auto-detection struggles due to subtle patterns

NOTES:
- Bootstrap adapts to your selected time spacing
- Great for low-noise, low-variance datasets

--------------------------------------------------------------------------------
RECOMMENDATION SUMMARY
--------------------------------------------------------------------------------

| Method           | Best For                             | Sensitivity | Speed  |
|------------------|---------------------------------------|-------------|--------|
| SciPy            | Clean, high-quality data              | Medium      | Fast   |
| Wavelet          | Noisy data, irregular peak shapes     | High        | Medium |
| Derivative       | Smooth transitions, subtle changes    | High        | Medium |
| Prominence       | Strong, obvious peaks                 | Medium      | Fast   |
| Combined         | Exploratory or max sensitivity        | Very High   | Slow   |
| Bootstrap        | Periodic data, user-assisted patterns | High        | Medium |


================================================================================
OUTPUT FILES AND FORMATS
================================================================================

1. BASIC PEAK CSV (e.g., PR24_TP04-SD1-pick.csv)
WaterDay,Temp.Filt,Depth
312.731100378,1.51357,Shallow
312.822053986,0.68311,Deep

2. FORMATTED OUTPUT (e.g., PR24_TP04-SD1-pickr.dAf)
PHASE SHIFT AND AMPLITUDE RATIO DATA FILE: PEAKPICKER OUTPUT
------------------------------------------------------------
0.180 is the relative distance (in m) between sensors.
------------------------------------------------------------
Data_Year    Water_Day    Ad_As    A_Uncertainty    Phase_Shift(days)    f_Uncertainty
2024    312.77657689    0.45131852    1.00000e-05    0.09095361    0.00100000

3. NAMING CONVENTIONS
- Input: PR24_TP04-SD1-filtr.csv
- Peak Output: PR24_TP04-SD1-pick.csv
- Formatted Output: PR24_TP04-SD1-pickr.dAf

================================================================================
TROUBLESHOOTING
================================================================================

1. NO PEAKS DETECTED
   - Lower prominence factor
   - Use combined method
   - Check that data is filtered and uploaded correctly

2. TOO MANY PEAKS
   - Raise prominence or min distance
   - Use exclusion ranges to skip noisy zones

3. NEGATIVE PHASE SHIFTS
   - Sensor labels may be swapped
   - Manual peak order may be wrong
   - Run error checker for guidance

4. APP NOT STARTING
   - Check for required packages: pip list
   - Confirm Python version: python --version
   - Ensure port 8050 is free or change Dash server port

================================================================================
INTERPRETATION AND QUALITY CONTROL
================================================================================

PHYSICAL EXPECTATIONS:
- Shallow peaks first (phase shift positive)
- Deep amplitudes < shallow amplitudes
- Peak spacing consistent with daily forcing

QUALITY CHECKS:
- Alternating shallow → deep → shallow
- Amplitude ratio Ad/As < 1.0
- Positive phase shifts (e.g., 0.05–0.3 days)

ERROR TYPES:
- Alternation: Repeating same depth
- Amplitude: Deep > Shallow
- Phase Shift: Negative or too large
- Unpaired: No matching trough near peak

================================================================================
LICENSE AND CITATION
================================================================================

This software is provided for research and educational purposes.
If using in published research, please cite appropriately.

Purpose: Temperature time-series peak picking for UCSC Hydrology

The software is provided as-is without warranty. Users are responsible for
validating results and ensuring appropriateness for their specific applications.

================================================================================
