================================================================================
Temperature Time-Series Frequency Analysis and Filtering Tool (tts_frefil_alpha.py)
================================================================================

Version: 2.0 - Full Specification Compliance with Folder Management

Author: Timothy Wu
Created: 7/3/2025
Last Updated: 7/3/2025

This tool provides interactive frequency analysis and filtering capabilities for 
temperature time-series data, specifically designed for analyzing diurnal cycles 
and other frequency components in shallow and deep temperature measurements.

================================================================================
INSTALLATION REQUIREMENTS
================================================================================

1. PYTHON VERSION
   - Python 3.7 or higher (recommended: Python 3.9+)
   - Download from: https://www.python.org/downloads/

2. REQUIRED LIBRARIES
   Install using pip (copy and paste the commands below):

   # Core libraries (REQUIRED)
   pip install numpy pandas scipy plotly dash dash-bootstrap-components

   # Optional but recommended for multitaper analysis
   pip install mne

   # If you get errors, try installing with specific versions:
   pip install numpy==1.24.3 pandas==2.0.3 scipy==1.11.1 plotly==5.15.0 dash==2.11.1 dash-bootstrap-components==1.4.1 mne==1.4.2

3. INSTALLATION COMMANDS (All at once)
   
   For Windows (Command Prompt or PowerShell):
   pip install numpy pandas scipy plotly dash dash-bootstrap-components mne

   For macOS/Linux (Terminal):
   pip3 install numpy pandas scipy plotly dash dash-bootstrap-components mne

   For Anaconda users:
   conda install numpy pandas scipy plotly
   pip install dash dash-bootstrap-components mne

4. VERIFY INSTALLATION
   Open Python and run:
   import numpy, pandas, scipy, plotly, dash, dash_bootstrap_components
   print("All libraries installed successfully!")

================================================================================
QUICK START GUIDE
================================================================================

1. PREPARE YOUR DATA (Use tts_merpar.py)
   - Ensure your CSV file has columns: WaterDay, TempShallow, TempDeep
   - Alternative column names (Shallow.Temp, Deep.Temp) are automatically recognized
   - Example data format:
     WaterDay,TempShallow,TempDeep
     312.000000,20.246000,21.199000
     312.013889,20.150000,21.151000
     ...

2. CONFIGURE PARAMETERS (Optional)
   - Edit tts_frefil.par file (see PARAMETER FILE section below)
   - Or use default values and adjust in the web interface

3. RUN THE APPLICATION
   - Double-click tts_frefil.py OR
   - Open command prompt/terminal in the script folder and run:
     python tts_frefil.py
   
4. OPEN WEB INTERFACE
   - The application will start a local web server
   - Open your web browser and go to: http://127.0.0.1:8050
   - You should see the interactive interface

5. BASIC WORKFLOW
   a) Upload your CSV data file
   b) Adjust PSD analysis parameters if needed
   c) Click "Update PSD" to view frequency spectrum
   d) Use "Enter Frequency Selection Mode" to click on plot and select frequency band
   e) Adjust filter parameters as needed
   f) Click "Apply Filter" to process data
   g) Review filtered results in the bottom plot
   h) Click "Save Filtered Data" to export results

================================================================================
PARAMETER FILE CONFIGURATION (tts_frefil.par)
================================================================================

The parameter file allows you to set default values and specify file locations.
Create a file named "tts_frefil.par" in the same folder as tts_frefil.py.

EXAMPLE PARAMETER FILE:
-----------------------
# Temperature Time-Series Frequency Filter Parameters
filename_composite_data = temperature_composite_data.csv
data_interval_minutes = 20
time_bandwidth_parameter = 4
wd_lower_limit = -1
wd_upper_limit = -1
start_band_pass = 0.8
end_band_pass = 1.2
ramp_fraction = 0.1
filter_order = 6
resample_interval = 20
output_folder = Filter Data

PARAMETER DESCRIPTIONS:
-----------------------
- filename_composite_data: Name of your input CSV file
- data_interval_minutes: Time between measurements (e.g., 20 for 20-minute intervals)
- time_bandwidth_parameter: Controls frequency resolution (2-10, default: 4)
- wd_lower_limit/wd_upper_limit: Water Day range to analyze (-1 = use all data)
- start_band_pass/end_band_pass: Frequency range for filtering (day^-1)
- ramp_fraction: Filter smoothness (0.0-0.5, 0.1=10% taper, 0.5=50% taper)
- filter_order: Filter sharpness (2-20, higher = sharper but more ringing)
- resample_interval: Output data interval (usually same as input)
- output_folder: Folder name where filtered files will be saved

================================================================================
DETAILED USAGE GUIDE
================================================================================

1. DATA UPLOAD SECTION
   - Drag and drop your CSV file OR click to browse
   - Supported formats: CSV with WaterDay, TempShallow, TempDeep columns
   - File is automatically validated and loaded

2. PSD ANALYSIS PARAMETERS
   - Sampling Frequency: Calculated from data interval (e.g., 72 per day for 20-min intervals)
   - WD Limits: Restrict analysis to specific time range
   - PSD Method: Choose analysis method:
     * Welch (Standard): Good general-purpose method
     * Welch (Smoothed): Smoother spectra, better for visualization
     * Multitaper: Most robust method (requires MNE library)
     * MATLAB MTM Style: Compatible with MATLAB spectrum.mtm (best)

3. FREQUENCY SELECTION
   - Click "Enter Frequency Selection Mode"
   - Click on the PSD plot to select lower frequency
   - Click again to select upper frequency
   - Frequencies are automatically sorted (low to high)
   - Click "Exit Frequency Selection Mode" when done

4. FILTER PARAMETERS
   - Filter Type: Butterworth (recommended), Chebyshev I/II, Elliptic
   - Filter Order: Higher = sharper cutoff (6 is good default)
   - Low/High Frequency: Band pass limits (day^-1)
   - Ramp Fraction: Transition smoothness (0.1-0.5 recommended)
   - Trend Removal: 
     * None: No detrending
     * DC Offset (recommended): Remove mean value
     * Linear/Polynomial: Remove longer-term trends
     * High-pass Filter: Remove very low frequencies
     * Moving Average: Remove slow variations

5. OUTPUT AND EXPORT
   - Output files are saved in the configured output folder
   - Files use specification-compliant format: WaterDay, Shallow.Temp.Filt, Deep.Temp.Filt
   - Parameters can be saved for future use

================================================================================
TROUBLESHOOTING
================================================================================

COMMON PROBLEMS AND SOLUTIONS:

1. "ModuleNotFoundError: No module named 'X'"
   - Solution: Install missing library with: pip install [library_name]
   - For MNE specifically: pip install mne

2. "Permission denied" when saving files
   - Solution: Check that output folder is writable
   - Try running as administrator (Windows) or with sudo (Mac/Linux)

3. "Address already in use" error
   - Solution: Close other instances of the application
   - Or change port by editing the last line: app.run(debug=True, port=8051)

4. Browser doesn't open automatically
   - Solution: Manually open browser and go to http://127.0.0.1:8050

5. PSD plot shows errors
   - Check data format (must have WaterDay, TempShallow, TempDeep columns)
   - Ensure data contains numeric values only
   - Try different PSD method if one fails

6. Filter fails to apply
   - Check that low frequency < high frequency
   - Ensure frequencies are within valid range (< Nyquist frequency)
   - Try lower filter order if getting errors

7. "Multitaper method not available"
   - Install MNE library: pip install mne
   - Use alternative methods if MNE installation fails

================================================================================
FEATURES AND FUNCTIONALITY
================================================================================

SPECTRAL ANALYSIS:
- Multiple PSD methods (Welch, Multitaper, MATLAB-style)
- Interactive parameter adjustment
- Real-time frequency spectrum visualization
- Data clipping for focused analysis

FILTER DESIGN:
- Specification-compliant ramp implementation
- Multiple filter types (Butterworth, Chebyshev, Elliptic)
- Comprehensive parameter validation
- Filter characteristics display

TREND REMOVAL:
- DC offset correction (addresses specification concerns)
- Linear and polynomial detrending
- High-pass filtering options
- Moving average trend removal

OUTPUT MANAGEMENT:
- Automatic folder creation
- Specification-compliant file format
- Filter history tracking
- Parameter export/import

VISUALIZATION:
- Interactive PSD plots with zoom/pan
- Raw data display
- Filtered data comparison
- Filter band visualization

================================================================================
FILE FORMATS
================================================================================

INPUT FILE FORMAT:
WaterDay,TempShallow,TempDeep
312.000000,20.246000,21.199000
312.013889,20.150000,21.151000
...

OUTPUT FILE FORMAT (Specification Compliant):
WaterDay,Shallow.Temp.Filt,Deep.Temp.Filt
312.000000000,0.007140,0.018250
312.000694302,0.001050,0.015070
...

PARAMETER FILE FORMAT (.par):
parameter_name = value
# Comments start with #

================================================================================
ADVANCED USAGE
================================================================================

BATCH PROCESSING:
- Use parameter files to standardize analysis across multiple datasets
- Modify input filename in .par file for different data files
- Save different parameter sets for different analysis types

CUSTOM ANALYSIS:
- Adjust segment length parameters for different data characteristics
- Use different time-bandwidth parameters for frequency/time resolution trade-offs
- Experiment with filter orders for optimal results

QUALITY CONTROL:
- Check filter validation warnings
- Review filter characteristics table
- Compare raw and filtered data plots
- Monitor edge effects in filtered results

INTEGRATION:
- Export parameters as JSON for integration with other tools
- Use filter history for reproducible analysis
- Document analysis workflow with saved parameters

================================================================================
EXAMPLE WORKFLOWS
================================================================================

WORKFLOW 1: DIURNAL CYCLE ANALYSIS
1. Load temperature data with ~20-minute intervals
2. Use Multitaper PSD method for robust analysis
3. Select frequency band around 0.8-1.2 day^-1
4. Apply Butterworth filter, order 6
5. Use DC offset removal
6. Export filtered data for amplitude/phase analysis

WORKFLOW 2: MULTI-DAY PATTERN ANALYSIS
1. Load longer-term temperature dataset
2. Set WD limits to focus on specific period
3. Use wider frequency band (e.g., 0.5-2.0 day^-1)
4. Apply polynomial trend removal
5. Use higher filter order for sharper cutoff
6. Compare multiple filter parameter sets

WORKFLOW 3: QUALITY ASSESSMENT
1. Load data and check for gaps/anomalies in raw plot
2. Use Welch (Smoothed) method for clean spectral visualization
3. Identify dominant frequency peaks
4. Apply narrow-band filter around main peak
5. Assess filter performance in filtered data plot
6. Adjust parameters based on validation warnings

================================================================================
TECHNICAL NOTES
================================================================================

FREQUENCY UNITS:
- All frequencies in day^-1 (cycles per day)
- For diurnal cycles: ~1 day^-1
- Nyquist frequency = (sampling frequency) / 2

FILTER DESIGN:
- Uses scipy.signal.filtfilt for zero-phase filtering
- Ramp implementation follows specification Table 2
- Filter characteristics displayed in specification format

DATA VALIDATION:
- Automatic column name recognition
- Parameter range checking
- Edge effect warnings
- Sampling frequency verification

PERFORMANCE:
- Optimized for datasets up to ~50,000 points
- Real-time parameter updates
- Efficient PSD computation
- Memory-conscious data handling

================================================================================
SUPPORT AND CONTACT
================================================================================

For questions, issues, or feature requests:
1. Check this README for common solutions
2. Verify all libraries are installed correctly
3. Test with example data to isolate issues
4. Document error messages and steps to reproduce

Version History:
- v1.0: Initial implementation
- v2.0: Full specification compliance, folder management, enhanced validation

================================================================================
LICENSE AND CITATION
================================================================================

This software is provided for research and educational purposes.
If using in published research, please cite appropriately.

Purpose: Temperature time-series analysis for UCSC Hydrology 

The software is provided as-is without warranty. Users are responsible for
validating results and ensuring appropriateness for their specific applications.

================================================================================