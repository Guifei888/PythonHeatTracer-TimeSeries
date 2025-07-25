================================================================================
Temperature Time-Series Merger and Processor (tts_mergpars.py)
================================================================================

Version: 2.0 - Enhanced with Adaptive Zoom and Improved User Interface

Author: Timothy Wu
Created: 6/26/2025
Last Updated: 7/16/2025

This tool processes temperature data from shallow and deep water temperature loggers,
merges them based on water day calculations, and provides an interactive interface
for selecting and exporting data intervals with advanced visualization features.

================================================================================
INSTALLATION REQUIREMENTS
================================================================================

1. PYTHON VERSION
   - Python 3.7 or higher (recommended: Python 3.9+)
   - Download from: https://www.python.org/downloads/

2. REQUIRED LIBRARIES
   Install using pip (copy and paste the commands below):

   # Core libraries (REQUIRED)
   pip install pandas numpy matplotlib

   # If you get errors, try installing with specific versions:
   pip install pandas==2.0.3 numpy==1.24.3 matplotlib==3.7.2

3. INSTALLATION COMMANDS (All at once)
   
   For Windows (Command Prompt or PowerShell):
   pip install pandas numpy matplotlib

   For macOS/Linux (Terminal):
   pip3 install pandas numpy matplotlib

   For Anaconda users:
   conda install pandas numpy matplotlib

4. VERIFY INSTALLATION
   Open Python and run:
   import pandas, numpy, matplotlib
   print("All libraries installed successfully!")

================================================================================
QUICK START GUIDE
================================================================================

1. PREPARE YOUR DATA
   - Ensure you have two CSV files: shallow and deep temperature logger data
   - Files should contain date/time and temperature columns
   - Example format:
     ID,DateTime,Temperature
     1,10/01/23 12:00:00 AM,15.25
     2,10/01/23 12:15:00 AM,15.30
     ...

2. CONFIGURE PARAMETERS
   - Create tts_mergpars.par file (see PARAMETER FILE section below)
   - Set input filenames, water year, and processing options

3. RUN THE APPLICATION
   - Double-click tts_mergpars_alpha.py OR
   - Open command prompt/terminal in the script folder and run:
     python tts_mergpars_alpha.py
   
4. INTERACTIVE WORKFLOW
   - View merged temperature data in interactive plot
   - Use zoom/pan tools for detailed inspection
   - Click and drag to select data intervals
   - Edit selection bounds using text boxes
   - Save selected intervals as CSV and WYO files

5. COMPLETION
   - Click "Done" to create full-record files
   - All output saved to configured folder with comprehensive logging

================================================================================
PARAMETER FILE CONFIGURATION (tts_mergpars.par)
================================================================================

The parameter file allows you to set processing options and file locations.
Create a file named "tts_mergpars.par" in the same folder as tts_mergpars_alpha.py.

EXAMPLE PARAMETER FILE:
-----------------------
# TTS_MERGPARS Parameter File
# Lines starting with # or ; are comments

# Input Files (required)
filename_shallow = shallow_temp_data.csv
filename_deep = deep_temp_data.csv

# Processing Parameters (required)
water_year = 2024
data_interval_min = 15.0

# Optional Parameters
convert_f_to_c = 0              # 1 to convert Fahrenheit to Celsius, 0 to keep original
gap_threshold_factor = 1.5      # Factor for detecting data gaps
output_folder = processed       # Output directory name
plot_relative = 0               # 1 for relative plotting, 0 for absolute water days

PARAMETER DESCRIPTIONS:
-----------------------
- filename_shallow: Filename of shallow temperature data CSV (REQUIRED)
- filename_deep: Filename of deep temperature data CSV (REQUIRED)
- water_year: Water year for analysis (Oct 1 to Sep 30) (REQUIRED)
- data_interval_min: Expected data logging interval in minutes (REQUIRED)
- convert_f_to_c: Convert Fahrenheit to Celsius (1=yes, 0=no, default=0)
- gap_threshold_factor: Multiplier for gap detection threshold (default=1.5)
- output_folder: Output directory name (default=processed)
- plot_relative: Use relative water day plotting (1=yes, 0=no, default=0)

================================================================================
INPUT DATA FORMAT REQUIREMENTS
================================================================================

CSV FILE STRUCTURE:
Your temperature data CSV files must contain at least 3 columns:
1. Column 1: Record number or ID (can be any value, will be ignored)
2. Column 2: Date/time string (see supported formats below)
3. Column 3: Temperature value (numeric)

SUPPORTED DATE/TIME FORMATS:
The script automatically detects and parses these datetime formats:
- MM/DD/YY HH:MM:SS AM/PM (e.g., 12/25/23 2:30:45 PM)
- MM/DD/YY HH:MM:SS (e.g., 12/25/23 14:30:45)
- MM/DD/YYYY HH:MM:SS AM/PM (e.g., 12/25/2023 2:30:45 PM)
- MM/DD/YYYY HH:MM:SS (e.g., 12/25/2023 14:30:45)
- YYYY-MM-DD HH:MM:SS (e.g., 2023-12-25 14:30:45)
- MM/DD/YY HH:MM (e.g., 12/25/23 14:30)
- MM/DD/YYYY HH:MM (e.g., 12/25/2023 14:30)

EXAMPLE CSV FORMAT:
ID,DateTime,Temperature
1,10/01/23 12:00:00 AM,15.25
2,10/01/23 12:15:00 AM,15.30
3,10/01/23 12:30:00 AM,15.28
4,10/01/23 12:45:00 AM,15.32
...

================================================================================
DETAILED USAGE GUIDE
================================================================================

1. SCRIPT STARTUP
   The script performs these operations automatically:
   - Loads parameters from tts_mergpars.par
   - Validates required parameters and files
   - Creates output directory if needed
   - Sets up comprehensive logging

2. DATA LOADING PHASE
   - Reads shallow and deep temperature CSV files
   - Detects and validates date/time formats
   - Reports parsing success and any errors
   - Identifies data gaps based on expected interval

   Enhanced Debug Output:
   "Debug: Showing first 10 date strings to verify format detection..."
   "✓ Successfully parsed using format: %m/%d/%y %I:%M:%S %p"
   "✓ All displayed date strings were successfully parsed and included"

3. WATER YEAR PROCESSING
   - Converts datetime to water day format (Oct 1 = Day 0)
   - Handles leap years automatically (366 vs 365 days)
   - Validates data falls within specified water year
   - Reports any pre-season or post-season data

4. DATA MERGING
   - Aligns shallow and deep datasets by water day
   - Uses nearest-neighbor matching for timestamp alignment
   - Reports merged dataset statistics
   - Identifies any anomalous data

5. INTERACTIVE VISUALIZATION
   Enhanced Features (v2.0):
   - Adaptive zoom-responsive axis ticks
   - Decimal water day precision when zoomed in
   - Smart temperature axis formatting (even degrees)
   - Real-time selection bound editing
   - Improved visual feedback

6. DATA SELECTION METHODS
   Method 1 - Graphical Selection:
   - Click and drag on plot to select interval
   - Visual rectangle shows selection bounds
   - Automatic text box updates

   Method 2 - Precise Text Entry:
   - Edit Start WD and End WD values directly
   - Selection rectangle updates in real-time
   - Fine-tune bounds with decimal precision

7. FILE EXPORT
   - User controls exact filenames (no automatic suffixes added)
   - Leave start/end water day fields empty to save full dataset
   - Creates three files per save: CSV, shallow WYO, deep WYO
   - Maintains comprehensive processing history

================================================================================
INTERACTIVE CONTROLS AND FEATURES
================================================================================

PLOT INTERACTION:
- Click and drag: Select data interval on the plot
- Zoom: Use matplotlib toolbar for detailed inspection
- Pan: Navigate through large datasets
- Adaptive ticks: Automatic adjustment based on zoom level

ZOOM-RESPONSIVE FEATURES:
- Time axis shows decimal water days when zoomed (0.1, 0.2, 0.5 day intervals)
- Temperature axis adjusts precision (0.1°C when zoomed, 2°C for overview)
- Grid lines adapt to zoom level for optimal readability
- Water year boundaries remain visible at all zoom levels

CONTROL PANEL:
- Start WD: Start water day for selection (editable, leave empty for full dataset)
- End WD: End water day for selection (editable, leave empty for full dataset)
- Filename: Base filename for output files (full user control, no automatic suffixes)
- Save: Export current selection or full dataset with chosen filename
- Clear: Remove all selection rectangles from plot
- Done: Finish processing and create full-record files

TEXT BOX ENHANCEMENTS:
- Real-time rectangle updates when editing values
- Click to position cursor and edit
- Standard matplotlib text editing (character-by-character)
- Immediate visual feedback for bound changes

================================================================================
OUTPUT FILES AND FORMATS
================================================================================

DIRECTORY STRUCTURE:
your_project_folder/
├── tts_mergpars_alpha.py
├── tts_mergpars.par
├── shallow_data.csv
├── deep_data.csv
└── processed/                  ← Output folder (configurable)
    ├── user_specified_name.csv
    ├── user_specified_name-S.wyo
    ├── user_specified_name-D.wyo
    ├── basename_full.csv
    ├── basename-S_full.wyo
    ├── basename-D_full.wyo
    └── basename_timestamp.log

FOR EACH SAVED DATASET:
- {user_filename}.csv - Composite CSV with both temperatures
- {user_filename}-S.wyo - Shallow temperature in WYO format
- {user_filename}-D.wyo - Deep temperature in WYO format

FULL DATASET FILES (created when clicking "Done"):
- {basename}_full.csv - Complete merged dataset
- {basename}-S_full.wyo - Complete shallow temperature data
- {basename}-D_full.wyo - Complete deep temperature data

LOG FILE:
- {basename}_{timestamp}.log - Complete processing log with all console output

FILE FORMAT SPECIFICATIONS:

CSV Format (Specification Compliant):
WaterDay,Shallow.Temp,Deep.Temp
0.000000,15.25000,14.82000
0.010417,15.30000,14.85000
0.020833,15.28000,14.80000
...

WYO Format (Water Year Output):
# Year	WaterDay	Temperature	DepthID
2023	0.00000	15.25000	1
2023	0.01042	15.30000	1
2023	0.02083	15.28000	1
...

================================================================================
WATER YEAR CONCEPTS AND CALCULATIONS
================================================================================

WATER YEAR DEFINITION:
- Water Year 2024: October 1, 2023 to September 30, 2024
- Water Day 0: October 1 (start of water year)
- Water Day 365/366: September 30 (end of water year)

LEAP YEAR HANDLING:
The script automatically handles leap years correctly:
- Water Year 2024: 366 days (contains Feb 29, 2024)
- Water Year 2025: 365 days (no leap day)
- February 29th determination based on calendar year containing the leap day

WATER DAY CALCULATION:
- Precise decimal calculations for fractional days
- Example: Day 45.25 = 45 days + 6 hours after water year start
- Consistent with hydrological modeling standards

YEAR ASSIGNMENT IN WYO FILES:
- Water Days 0-364/365: Assigned to previous calendar year
- Water Days 365/366+: Assigned to current water year
- Ensures proper temporal continuity in WYO format

================================================================================
TROUBLESHOOTING
================================================================================

COMMON PROBLEMS AND SOLUTIONS:

1. "Parameter file not found"
   Error: tts_mergpars.par not found.
   Solution: Create tts_mergpars.par in the same directory as the script

2. "CSV file not found"
   Error: Shallow file 'data.csv' not found
   Solution: Check filename spelling in parameter file and ensure CSV files exist

3. "No valid data found"
   Error: no valid data found in filename.csv
   Solution: Check CSV format, ensure date/time strings are in supported format

4. "Could not parse date"
   Warning: Could not parse date '25/12/23 14:30:00' on line 5
   Solution: Ensure MM/DD/YY format (not DD/MM/YY), check supported formats

5. Import errors
   ModuleNotFoundError: No module named 'pandas'
   Solution: Install required packages: pip install pandas numpy matplotlib

6. Display issues on headless systems
   TclTK Error: no display
   Solution: Use system with GUI support or set up X11 forwarding for SSH

7. Memory issues with large datasets
   Solution: Increase system memory or split large datasets into smaller chunks

8. Interactive plot not responding
   Solution: Check matplotlib backend, ensure GUI libraries are installed

DATA QUALITY DIAGNOSTICS:

Gap Detection:
- Automatically detects gaps larger than interval × gap_threshold_factor
- Reports gap locations and durations
- Helps identify logger malfunctions or data transmission issues

Water Year Validation:
- Flags data outside specified water year
- Identifies pre-season data (negative water days)
- Identifies post-season data (water day > 365/366)
- Ensures temporal consistency

Date Format Validation:
- Shows successful parsing examples in debug output
- Reports specific parsing failures with line numbers
- Suggests format corrections for failed dates

================================================================================
ENHANCED FEATURES (VERSION 2.0)
================================================================================

ADAPTIVE VISUALIZATION:
- Zoom-responsive tick formatting for both axes
- Time axis shows decimal precision when zoomed (0.1, 0.2, 0.5 days)
- Temperature axis shows appropriate increments (0.1°C to 5°C)
- Grid lines adapt automatically to zoom level

USER INTERFACE IMPROVEMENTS:
- Real-time selection rectangle updates
- Editable selection bounds via text boxes
- Visual feedback for saved vs. current selections
- Enhanced plot styling and clarity

FILENAME MANAGEMENT:
- Complete user control over output filenames (no automatic suffixes)
- Ability to save full dataset without interval selection
- Intelligent base name generation from input files
- Consistent file organization

DEBUG AND VALIDATION:
- Clear explanations of date parsing process
- Enhanced error reporting with solutions
- Comprehensive data quality diagnostics
- Detailed processing logs

PERFORMANCE OPTIMIZATIONS:
- Efficient memory usage for large datasets
- Responsive interactive plotting
- Optimized data processing algorithms
- Minimal computational overhead

================================================================================
EXAMPLE WORKFLOWS
================================================================================

WORKFLOW 1: SINGLE INTERVAL EXTRACTION
1. Load temperature data for water year 2024
2. Zoom to period of interest (e.g., days 100-150)
3. Fine-tune selection using text boxes (e.g., 120.5 to 135.2)
4. Save with descriptive filename: "SpringPeak_2024"
5. Results: SpringPeak_2024.csv, SpringPeak_2024-S.wyo, SpringPeak_2024-D.wyo

WORKFLOW 2: FULL DATASET EXPORT
1. Load temperature data for water year 2024
2. Leave start/end water day fields empty
3. Enter desired filename: "FullYear_2024"
4. Click Save to export complete dataset
5. Results: FullYear_2024.csv, FullYear_2024-S.wyo, FullYear_2024-D.wyo

WORKFLOW 3: MULTIPLE INTERVAL COMPARISON
1. Load full water year dataset
2. Select first interval (e.g., winter period), save as: "Winter_Dec_Jan"
3. Select second interval (e.g., summer period), save as: "Summer_Jul_Aug"
4. Select third interval (e.g., storm event), save as: "Storm_Event_March"
5. Compare intervals using generated files

WORKFLOW 4: QUALITY ASSESSMENT WORKFLOW
1. Load data and examine full time series
2. Check console output for gap reports and parsing issues
3. Zoom to suspicious periods for detailed inspection
4. Export problem periods for further analysis
5. Document data quality issues in processing log

WORKFLOW 5: SEASONAL ANALYSIS PREPARATION
1. Load annual temperature dataset
2. Export seasonal intervals:
   - Fall: Days 0-90, save as: "Fall_OctNovDec"
   - Winter: Days 91-180, save as: "Winter_JanFebMar"
   - Spring: Days 181-270, save as: "Spring_AprMayJun"
   - Summer: Days 271-365, save as: "Summer_JulAugSep"
3. Use consistent naming for downstream analysis

WORKFLOW 6: EVENT-FOCUSED EXTRACTION
1. Load data and zoom to event period
2. Use precise text entry for exact timing
3. Extract pre-event baseline, save as: "Baseline_Pre"
4. Extract event period, save as: "Event_Peak"
5. Extract recovery period, save as: "Recovery_Post"
6. Maintain temporal continuity for analysis

================================================================================
TECHNICAL SPECIFICATIONS
================================================================================

DATA PROCESSING:
- Pandas-based data manipulation for reliability
- Numpy for efficient numerical computations
- Automatic datetime parsing with multiple format support
- Nearest-neighbor timestamp alignment for merging

WATER YEAR CALCULATIONS:
- Precise decimal day calculations
- Automatic leap year detection and handling
- Temporal consistency validation
- Standard hydrological conventions

INTERACTIVE PLOTTING:
- Matplotlib-based visualization with enhanced widgets
- Real-time plot updates and responsive controls
- Adaptive axis formatting based on zoom level
- Professional-quality output suitable for publications

FILE I/O:
- Robust CSV parsing with error handling
- Multiple output format support (CSV, WYO)
- Comprehensive logging with timestamps
- Automatic directory creation and management

PERFORMANCE:
- Optimized for datasets up to 100,000 records
- Memory-efficient data structures
- Responsive user interface
- Minimal processing latency

COMPATIBILITY:
- Cross-platform support (Windows, macOS, Linux)
- Python 3.7+ compatibility
- Standard library dependencies where possible
- Graceful degradation for missing optional features

================================================================================
INTEGRATION WITH OTHER TOOLS
================================================================================

DOWNSTREAM ANALYSIS:
- Output files compatible with tts_frefil.py for frequency analysis
- WYO format compatible with hydrological modeling software
- CSV format suitable for statistical analysis packages
- Maintains data precision for numerical modeling

UPSTREAM DATA SOURCES:
- Compatible with most temperature logger formats
- Flexible date/time parsing for various instrument outputs
- Handles both Fahrenheit and Celsius temperature units
- Accommodates different sampling intervals

WORKFLOW INTEGRATION:
- Parameter files enable batch processing
- Log files provide audit trail for reproducible analysis
- Consistent file naming supports automated workflows
- JSON parameter export for integration with other tools

================================================================================
SUPPORT AND DEVELOPMENT
================================================================================

REPORTING ISSUES:
When reporting problems, please include:
1. Complete error message and traceback
2. Sample of your CSV data format (first 10 lines)
3. Contents of your tts_mergpars.par parameter file
4. Python version and operating system information
5. Steps to reproduce the issue

KNOWN LIMITATIONS:
- Text box editing uses standard matplotlib functionality (no Tab navigation)
- Very large datasets (>500,000 records) may require increased memory
- Interactive plotting requires GUI-capable system
- Date parsing assumes MM/DD/YY format (not DD/MM/YY)

FUTURE ENHANCEMENTS:
- Web-based interface for improved user experience
- Batch processing capabilities for multiple files
- Advanced quality control and gap-filling options
- Integration with real-time data acquisition systems

================================================================================
LICENSE AND CITATION
================================================================================

This software is provided for research and educational purposes.
If using in published research, please cite appropriately.

Purpose: Temperature time-series processing for UCSC Hydrology

The software is provided as-is without warranty. Users are responsible for
validating results and ensuring appropriateness for their specific applications.

================================================================================
CHANGELOG AND VERSION HISTORY
================================================================================

Last Updated: July 3, 2025

Enhancement Suggestions from: Dr. Andrew T. Fisher, Ethan Yan 

VERSION 2.0 ENHANCEMENTS (July 3, 2025):

MAJOR USER INTERFACE IMPROVEMENTS:
 Adaptive Zoom-Responsive Axis Ticks
   - Time axis now shows decimal water days when zoomed in (0.1, 0.2, 0.5 day intervals)
   - Automatic tick density adjustment based on visible range
   - Maintains readability at all zoom levels from full year to sub-daily resolution
   - Enhanced grid formatting for professional presentation

 Smart Temperature Axis Formatting
   - Even degree increments for normal temperature ranges (2°C intervals)
   - 0.1°C precision when zoomed to small temperature ranges
   - Automatic adjustment from 0.1°C to 5°C intervals based on visible range
   - Professional °C labeling on all major ticks

 Editable Selection Bounds
   - Real-time rectangle updates when editing Start WD/End WD text boxes
   - Two-way synchronization between graphical selection and text input
   - Fine-tune selections with decimal precision (e.g., 45.25 to 47.80)
   - Immediate visual feedback for bound changes

 User-Controlled Filename Management
   - Complete user control over output filenames (no automatic suffixes)
   - Ability to save full dataset by leaving start/end fields empty
   - Cleaner file organization with user-specified naming
   - Maintains necessary -S and -D suffixes for WYO files only

DATA PROCESSING ENHANCEMENTS:
 Enhanced Debug Output and Validation
   - Clarified "Raw date string" messages with success indicators
   - Added ✓ checkmarks to show successful date format parsing
   - Clear explanation that displayed dates are included in processing
   - Better distinction between debug information and actual warnings

 Improved Data Quality Reporting
   - Enhanced gap detection with clear explanations
   - Better water year validation with specific recommendations
   - Comprehensive processing statistics and summaries
   - Professional logging with timestamps and processing duration

VISUAL AND INTERACTION IMPROVEMENTS:
 Enhanced Selection Visualization
   - Color-coded selection states (blue=current, green=saved)
   - Persistent selection rectangles with improved styling
   - Better visual feedback for saved vs. active selections
   - Cleaner plot aesthetics with professional styling

Stability and Error Handling
   - Removed problematic Tab navigation due to matplotlib limitations
   - Removed unreliable double-click text selection features
   - Enhanced error handling with specific solution guidance
   - Improved crash prevention and graceful error recovery

TECHNICAL IMPROVEMENTS:
 Performance Optimizations
   - Efficient axis management with AdaptiveAxisManager class
   - Optimized real-time rectangle updates
   - Reduced computational overhead for interactive operations
   - Better memory management for large datasets

 Code Architecture Enhancements
   - Modular axis management system
   - Improved event handling architecture
   - Better separation of concerns in UI components
   - Enhanced maintainability and extensibility

FEATURES ATTEMPTED BUT REMOVED (due to matplotlib limitations):
 Tab Navigation Between Text Boxes
   - Initial implementation caused crashes due to matplotlib widget focus issues
   - Removed for stability - users can click to select text boxes instead

 Advanced Text Selection (Double-click, Drag-select)
   - matplotlib TextBox widgets have inherent limitations for rich text editing
   - Removed to prevent errors - standard click-to-edit functionality maintained

BACKWARD COMPATIBILITY:
 All existing parameter files work unchanged
 All output file formats remain identical
 Command-line usage and workflow preserved
 Full compatibility with downstream analysis tools

VERSION 1.0 (Original Implementation - June 26, 2025):
- Basic temperature data loading and parsing
- Water year calculations and leap year handling
- Interactive matplotlib plotting with basic selection
- CSV and WYO file output formats
- Parameter file configuration system
- Comprehensive logging and error reporting

SUMMARY OF VERSION 2.0 IMPACT:
The Version 2.0 enhancements significantly improve the user experience while 
maintaining full backward compatibility. Key improvements focus on:

1. USABILITY: Adaptive ticks make zooming and detailed inspection much more effective
2. PRECISION: Decimal water day editing enables exact interval specification  
3. CLARITY: Enhanced debug output eliminates user confusion about data processing
4. CONTROL: User-specified filenames provide better file organization
5. STABILITY: Removal of problematic features ensures reliable operation

These improvements make tts_mergpars_alpha.py a more professional and user-friendly
tool for temperature time-series processing while preserving all core functionality
and ensuring seamless integration with existing workflows and downstream analysis tools.

================================================================================