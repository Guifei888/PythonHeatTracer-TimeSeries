#!/usr/bin/env python3
"""
TTS_MERGPARS Dash Application - Temperature Time Series Merger and Processor

A modern web interface for processing temperature data from shallow and deep water 
temperature loggers, with interactive visualization and flexible export options.

Author: Converted from original TTS_MERGPARS.py by Timothy Wu
Created: 2025
"""

import os
import sys
import re
import csv
import base64
import io
import json
import zipfile
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class TeeLogger:
    """
    Custom logger that writes output to both console and a log file.
    
    This class redirects stdout to capture all print statements and save them
    to a log file while still displaying them on the console.
    """
    
    def __init__(self, log_file):
        """
        Initialize the logger.
        
        Args:
            log_file (str): Path to the log file to create
        """
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', encoding='utf-8')
        self.start_time = datetime.now()
        
        # Write header information to log file
        self._write_log_header()

    def _write_log_header(self):
        """Write header information to the log file."""
        self.log_file.write(f"=== TTS_MERGPARS Dash Application Processing Log ===\n")
        self.log_file.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Command: {' '.join(sys.argv)}\n")
        self.log_file.write("=" * 60 + "\n\n")
        self.log_file.flush()

    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write

    def flush(self):
        """Flush both terminal and log file."""
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        """Close the logger and write footer information."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        footer = f"\n\n{'=' * 60}\n"
        footer += f"Processing session completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"Total session duration: {duration}\n"
        footer += f"Log file: {self.log_file.name}\n"
        footer += "=" * 60 + "\n"
        
        self.write(footer)
        self.log_file.close()

def create_log_filename():
    """Create a timestamped log filename in dedicated log directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'logs')  # Changed to 'logs' for clarity
    
    # Create log directory if it doesn't exist  
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"📁 Created dedicated log directory: {log_dir}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"tts_mergpars_session_{timestamp}.log"
    full_log_path = os.path.join(log_dir, log_filename)
    
    print(f"📝 Log file will be saved to: {full_log_path}")
    return full_log_path

# Initialize logging system
log_file_path = create_log_filename()
logger = TeeLogger(log_file_path)
sys.stdout = logger  # Redirect stdout to logger

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "TTS MERGPARS - Temperature Data Processor"

print("=" * 60)
print("TTS MERGPARS - Temperature Time Series Merger and Processor")
print("=" * 60)
print(f"🚀 Application starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📝 Session logging to: {log_file_path}")
print(f"📁 Log directory: logs/ (separate from processed data)")
print(f"📁 Data output directory: processed/ (configurable)")
print("💡 All terminal output will be saved to the session log file")
print("=" * 60)

# Global variables for data storage
global_data = {
    'shallow_df': None,
    'deep_df': None,
    'merged_df': None,
    'parameters': {},
    'processing_log': [],
    'waterday_selection_mode': False,
    'selected_waterdays': []
}

# ============================================================================
# UTILITY FUNCTIONS (from original script)
# ============================================================================

def is_leap_year(year):
    """
    Check if a given year is a leap year using the Gregorian calendar rules.
    
    This function is fully future-proof and handles all edge cases including:
    - Standard leap years (divisible by 4)
    - Century years (divisible by 100 are NOT leap years)  
    - Millennium years (divisible by 400 ARE leap years)
    
    Examples:
    - 2024: LEAP (divisible by 4, not century)
    - 2025: NOT LEAP 
    - 2100: NOT LEAP (century year, not divisible by 400)
    - 2400: LEAP (divisible by 400)
    
    Args:
        year (int): Any calendar year (past, present, or future)
        
    Returns:
        bool: True if leap year, False otherwise
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_water_year_length(water_year):
    """
    Get the length of a water year in days.
    
    Water Year runs from October 1 to September 30.
    For Water Year N, we check if calendar year N has a leap day (Feb 29).
    
    Examples:
    - Water Year 2024: Oct 1, 2023 → Sep 30, 2024
      Contains Feb 29, 2024, so check if 2024 is leap → 366 days
    - Water Year 2025: Oct 1, 2024 → Sep 30, 2025  
      Contains Feb 29, 2025 (if it existed), so check if 2025 is leap → 365 days
    
    This function is future-proof for any year including:
    - 2100 (NOT leap - century rule)
    - 2400 (LEAP - millennium rule)  
    - Any future year following Gregorian calendar rules
    
    Args:
        water_year (int): The water year (any past, present, or future year)
        
    Returns:
        int: 366 if leap year, 365 if not
    """
    calendar_year_with_feb = water_year
    return 366 if is_leap_year(calendar_year_with_feb) else 365

def parse_par_content(content):
    """
    Parse a flat parameter file with key=value pairs.
    
    The parameter file format supports:
    - Comments starting with # or ;
    - Key=value pairs
    - Inline comments after values
    
    Args:
        content (str): Content of parameter file
        
    Returns:
        dict: Dictionary of parameter key-value pairs
    """
    params = {}
    print(f"\n=== Parsing Parameter File Content ===")
    
    for line_num, line in enumerate(content.split('\n'), 1):
        line = line.strip()
        print(f"Line {line_num}: '{line}'")
        
        # Skip empty lines and comments
        if not line or line.startswith('#') or line.startswith(';'):
            print(f"  -> Skipped (empty or comment)")
            continue
            
        if '=' in line:
            key, value = line.split('=', 1)
            
            # Clean up key (remove non-alphanumeric, convert to lowercase)
            original_key = key.strip()
            clean_key = re.sub(r'\W+', '_', key.strip().lower())
            
            # Remove inline comments from value
            value = value.split(';', 1)[0].split('#', 1)[0].strip()
            
            params[clean_key] = value
            print(f"  -> Parsed: '{original_key}' = '{value}' (key: '{clean_key}')")
        else:
            print(f"  -> Skipped (no '=' found)")
    
    print(f"✅ Successfully parsed {len(params)} parameters")
    for key, value in params.items():
        print(f"   {key} = {value}")
    
    return params

def load_default_parameters():
    """Load default parameters from .par file in script directory."""
    print("\n=== Loading Default Parameters ===")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        par_file = os.path.join(script_dir, 'tts_mergpars.par')
        
        if os.path.exists(par_file):
            print(f"✓ Found parameter file: {par_file}")
            with open(par_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✓ File content preview:")
            for i, line in enumerate(content.split('\n')[:10], 1):
                print(f"   Line {i}: {line}")
            if len(content.split('\n')) > 10:
                print(f"   ... ({len(content.split('\n'))} total lines)")
            
            params = parse_par_content(content)
            print(f"✓ Loaded {len(params)} parameters from .par file")
            return params
        else:
            print(f"⚠️ No parameter file found at: {par_file}")
            print("Using built-in default values")
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        import traceback
        traceback.print_exc()
    
    # Return default values if file not found
    defaults = {
        'filename_shallow': 'PR24_TP04-S.csv',
        'filename_deep': 'PR24_TP04-D.csv',
        'water_year': '2024',
        'data_interval_min': '20.0',
        'convert_f_to_c': '0',
        'gap_threshold_factor': '1.5',
        'output_folder': 'processed',
        'plot_relative': '0'
    }
    print("✓ Using built-in default parameters:")
    for key, value in defaults.items():
        print(f"  {key} = {value}")
    return defaults

def calculate_water_day_columns(df, water_year):
    """Add water day and water year columns to DataFrame."""
    water_year_start = datetime(water_year - 1, 10, 1)
    
    def calc_water_day(dt):
        delta = dt - water_year_start
        return delta.total_seconds() / 86400.0
    
    def get_water_year_for_date(dt):
        if dt.month >= 10:
            return dt.year + 1
        else:
            return dt.year
    
    df['WaterDay'] = df['DateTime'].apply(calc_water_day)
    df['WaterYear'] = df['DateTime'].apply(get_water_year_for_date)
    return df

def read_logger_data_from_content(content, filename, label, convert_fahrenheit=False, 
                                 interval_min=15, gap_factor=1.5):
    """
    Read and parse temperature logger data from file content.
    
    This function handles multiple datetime formats and performs data validation.
    It also detects gaps in the data based on expected sampling interval.
    
    Args:
        content (str): File content as string
        filename (str): Name of file for logging
        label (str): Description for logging (e.g., 'Shallow', 'Deep')
        convert_fahrenheit (bool): Whether to convert Fahrenheit to Celsius
        interval_min (float): Expected data interval in minutes
        gap_factor (float): Factor to determine gap threshold
        
    Returns:
        tuple: (DataFrame with DateTime and Temp columns, valid_records, error_count, total_lines)
        
    Raises:
        ValueError: If no valid data found
    """
    print(f"Reading {label} data from: {filename}")
    
    # Initialize counters for processing statistics
    rows = []
    line_count = 0
    error_count = 0
    
    # Supported datetime formats (most common first for efficiency)
    date_formats = [
        '%m/%d/%y %I:%M:%S %p',  # 12-hour with AM/PM
        '%m/%d/%y %H:%M:%S',     # 24-hour
        '%m/%d/%Y %I:%M:%S %p',  # 4-digit year with AM/PM
        '%m/%d/%Y %H:%M:%S',     # 4-digit year 24-hour
        '%Y-%m-%d %H:%M:%S',     # ISO format
        '%m/%d/%y %H:%M',        # No seconds
        '%m/%d/%Y %H:%M'         # 4-digit year no seconds
    ]
    
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        line_count += 1
        parts = line.split(',')
        
        # Skip header lines or lines with insufficient columns
        if len(parts) < 3:
            if line_count <= 3:  # Allow for headers in first few lines
                continue
            error_count += 1
            continue
        
        # Extract date and temperature strings
        date_str = parts[1].strip()
        temp_str = parts[2].strip()
        
        # Debug: Show first few raw date strings
        if line_count <= 10 and date_str and not date_str.startswith('Date'):
            print(f"  Raw date string (line {line_num}): '{date_str}'")
        
        # Try to parse datetime with multiple formats
        parsed_datetime = None
        for date_format in date_formats:
            try:
                parsed_datetime = datetime.strptime(date_str, date_format)
                break
            except ValueError:
                continue
        
        if not parsed_datetime:
            error_count += 1
            if error_count <= 5:  # Show first few errors
                print(f"  Warning: Could not parse date '{date_str}' on line {line_num}")
            continue
        
        # Parse temperature value
        try:
            temperature = float(temp_str)
        except ValueError:
            error_count += 1
            if error_count <= 5:
                print(f"  Warning: Could not parse temperature '{temp_str}' on line {line_num}")
            continue
        
        # Convert Fahrenheit to Celsius if requested
        if convert_fahrenheit:
            temperature = (temperature - 32.0) / 1.8
        
        rows.append((parsed_datetime, temperature))
    
    # Validate that we have data
    if not rows:
        raise ValueError(f"No valid data found in {filename}")
    
    print(f"  Loaded {len(rows)} valid records, {error_count} errors from {line_count} lines")
    
    # Create DataFrame and sort by datetime
    df = pd.DataFrame(rows, columns=['DateTime', 'Temp']).sort_values('DateTime')
    df = df.reset_index(drop=True)
    
    # Detect gaps in the data
    detect_data_gaps(df, interval_min, gap_factor)
    
    return df, len(rows), error_count, line_count

def detect_data_gaps(df, interval_min, gap_factor):
    """
    Detect and report gaps in time series data.
    
    Args:
        df (pandas.DataFrame): DataFrame with DateTime column
        interval_min (float): Expected interval in minutes
        gap_factor (float): Factor to determine gap threshold
        
    Returns:
        list: List of gap information dictionaries
    """
    time_diffs = df['DateTime'].diff().dt.total_seconds() / 60.0
    gap_threshold = interval_min * gap_factor
    gaps = time_diffs > gap_threshold
    
    gap_info = []
    gap_count = gaps.sum()
    if gap_count > 0:
        print(f"  Found {gap_count} gaps > {gap_threshold:.1f} minutes:")
        for idx in df.index[gaps]:
            gap_size = time_diffs.iloc[idx]
            gap_info.append({
                'index': idx,
                'datetime': df.iloc[idx]['DateTime'],
                'gap_minutes': gap_size
            })
            print(f"    Gap at {df.iloc[idx]['DateTime']}: {gap_size:.1f} minutes")
    
    return gap_info

def validate_water_year_data(df, label, water_year):
    """
    Validate data falls within the specified water year and report issues.
    
    Args:
        df (pandas.DataFrame): DataFrame with WaterYear and WaterDay columns
        label (str): Description for logging
        water_year (int): Expected water year
        
    Returns:
        bool: True if all data is within the specified water year
    """
    water_year_length = get_water_year_length(water_year)
    outside_wy = df[df['WaterYear'] != water_year]
    
    if len(outside_wy) > 0:
        print(f"\n*** WARNING: {label} data contains {len(outside_wy)} records outside Water Year {water_year} ***")
        
        # Group and report by water year
        for wy in sorted(outside_wy['WaterYear'].unique()):
            wy_data = outside_wy[outside_wy['WaterYear'] == wy]
            wd_range = (wy_data['WaterDay'].min(), wy_data['WaterDay'].max())
            date_range = (wy_data['DateTime'].min(), wy_data['DateTime'].max())
            
            print(f"  Water Year {wy}: {len(wy_data)} records")
            print(f"    WaterDay range: {wd_range[0]:.5f} to {wd_range[1]:.5f}")
            print(f"    Date range: {date_range[0]} to {date_range[1]}")
            
            # Flag pre-season and post-season data
            if wd_range[0] < 0:
                print(f"    *** Contains PRE-SEASON data (negative WaterDay values) ***")
            if wd_range[1] > water_year_length:
                print(f"    *** Contains POST-SEASON data (WaterDay > {water_year_length}) ***")
    
    return len(outside_wy) == 0

def report_anomalous_data(merged_df):
    """
    Report pre-season and post-season data in merged dataset.
    
    Args:
        merged_df (pandas.DataFrame): Merged temperature data with WaterDay and WaterYear columns
    """
    # Get water year from the data (assuming all data is from same water year)
    water_year = merged_df['WaterYear'].iloc[0] if len(merged_df) > 0 else None
    
    if water_year is None:
        print("Warning: Could not determine water year for anomaly detection")
        return
    
    water_year_length = get_water_year_length(water_year)
    
    # Check for negative WaterDay values (pre-season)
    negative_wd = merged_df[merged_df['WaterDay'] < 0]
    if len(negative_wd) > 0:
        print(f"\nFound {len(negative_wd)} records with negative WaterDay (pre-season data)")
        print(f"  WaterDay range: {negative_wd['WaterDay'].min():.5f} to {negative_wd['WaterDay'].max():.5f}")

    # Check for post-season data (WaterDay > water_year_length)
    post_season = merged_df[merged_df['WaterDay'] > water_year_length]
    if len(post_season) > 0:
        print(f"\nFound {len(post_season)} records with WaterDay > {water_year_length} (post-season data)")
        print(f"  WaterDay range: {post_season['WaterDay'].min():.5f} to {post_season['WaterDay'].max():.5f}")
        print(f"  These likely belong to the next water year")

def create_default_filename(shallow_filename, deep_filename, water_year):
    """
    Create a composite base name from input filenames in the correct format.
    
    Attempts to find common patterns or prefixes in the filenames.
    Output format should be like: PR24_TP04SD (not PR24_TP04_WY2024)
    
    Args:
        shallow_filename (str): Shallow data filename
        deep_filename (str): Deep data filename
        water_year (int): Water year for naming
        
    Returns:
        str: Base name for output files
    """
    if shallow_filename and deep_filename:
        # Extract base names without extensions
        sh_base = os.path.splitext(shallow_filename)[0]
        dp_base = os.path.splitext(deep_filename)[0]
        
        # Handle common naming patterns like filename-S and filename-D
        if sh_base.lower().endswith('-s') and dp_base.lower().endswith('-d'):
            # Use the common part + SD (for Shallow-Deep combined)
            return sh_base[:-2] + 'SD'
        
        # Find common prefix
        common_prefix = os.path.commonprefix([sh_base, dp_base]).rstrip('_-')
        if common_prefix and len(common_prefix) > 2:
            # Add SD suffix to indicate combined shallow-deep data
            return f"{common_prefix}SD"
        else:
            return f"TempDataSD"
    else:
        return f"TempDataSD"

def merge_temperature_data(df_shallow, df_deep):
    """
    Merge shallow and deep temperature data based on water day alignment.
    
    Uses pandas merge_asof to find the nearest deep temperature reading
    for each shallow temperature reading.
    
    Args:
        df_shallow (pandas.DataFrame): Shallow temperature data
        df_deep (pandas.DataFrame): Deep temperature data
        
    Returns:
        pandas.DataFrame: Merged data with T_shallow and T_deep columns
    """
    print('\n=== Merging Data ===')
    
    # Prepare data for merging (ensure sorted by WaterDay)
    shallow_sorted = df_shallow[['DateTime', 'WaterDay', 'Temp', 'WaterYear']].sort_values('WaterDay')
    deep_sorted = df_deep[['WaterDay', 'Temp']].sort_values('WaterDay')
    
    # Use merge_asof to align timestamps (shallow as primary timeline)
    merged = pd.merge_asof(
        shallow_sorted,
        deep_sorted,
        on='WaterDay',
        suffixes=('_sh', '_dp'),
        direction='nearest'
    )
    
    # Rename columns for clarity
    merged.columns = ['DateTime', 'WaterDay', 'T_shallow', 'WaterYear', 'T_deep']
    
    # Remove any rows with missing data
    merged = merged.dropna()
    
    print(f"Merged dataset: {len(merged)} records")
    print(f"Date range: {merged['DateTime'].min()} to {merged['DateTime'].max()}")
    print(f"WaterDay range: {merged['WaterDay'].min():.5f} to {merged['WaterDay'].max():.5f}")
    
    # Report any anomalous data
    report_anomalous_data(merged)
    
    return merged

def save_files_to_folder(merged_df, filename_base, folder_name, water_year):
    """
    Save CSV and WYO files directly to the specified folder.
    
    Args:
        merged_df: DataFrame with merged temperature data
        filename_base: Base filename (literal, no additions)
        folder_name: Folder name to save files in
        water_year: Water year for WYO file calculations
        
    Returns:
        tuple: (success, file_paths, message)
    """
    try:
        # Create folder if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_folder_path = os.path.join(script_dir, folder_name)
        
        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)
            print(f"📁 Created output folder: {full_folder_path}")
        else:
            print(f"📁 Using existing folder: {full_folder_path}")
        
        # Prepare file paths (use exact filename without additions)
        csv_path = os.path.join(full_folder_path, f"{filename_base}.csv")
        shallow_wyo_path = os.path.join(full_folder_path, f"{filename_base}-S.wyo")
        deep_wyo_path = os.path.join(full_folder_path, f"{filename_base}-D.wyo")
        
        print(f"💾 Saving files:")
        print(f"   1. {os.path.basename(csv_path)} (CSV format)")
        print(f"   2. {os.path.basename(shallow_wyo_path)} (WYO - Shallow)")
        print(f"   3. {os.path.basename(deep_wyo_path)} (WYO - Deep)")
        
        # Save CSV file with EXACT formatting: WaterDay (6 decimals), Temps (5 decimals)
        csv_data = pd.DataFrame({
            'WaterDay': merged_df['WaterDay'].apply(lambda x: f"{x:.6f}"),
            'Shallow.Temp': merged_df['T_shallow'].apply(lambda x: f"{x:.5f}"),
            'Deep.Temp': merged_df['T_deep'].apply(lambda x: f"{x:.5f}")
        })
        csv_data.to_csv(csv_path, index=False)
        print(f"   ✅ CSV saved: {len(csv_data)} records with precision formatting")
        
        # Save WYO files
        create_wyo_file(merged_df, 'T_shallow', shallow_wyo_path, water_year)
        print(f"   ✅ Shallow WYO saved: {len(merged_df)} records")
        
        create_wyo_file(merged_df, 'T_deep', deep_wyo_path, water_year)
        print(f"   ✅ Deep WYO saved: {len(merged_df)} records")
        
        file_paths = [csv_path, shallow_wyo_path, deep_wyo_path]
        
        return True, file_paths, f"Files saved to {folder_name}/"
        
    except Exception as e:
        print(f"❌ Error in save_files_to_folder: {str(e)}")
        return False, [], f"Error saving files: {str(e)}"

def create_wyo_file(df, temp_column, filepath, water_year):
    """
    Create WYO format file for a single temperature column.
    
    Args:
        df: DataFrame with water day and temperature data
        temp_column: Column name for temperature data
        filepath: Full path for output file
        water_year: Water year for calculations
    """
    # Water year starts October 1 of previous calendar year
    water_year_start = datetime(water_year - 1, 10, 1)
    print(f"   🗓️ Water Year {water_year} starts: {water_year_start.strftime('%Y-%m-%d')}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Year\tWaterDay\tTemperature\tDepthID\n")
        
        # Track year transitions for reporting
        year_counts = {}
        sample_entries = []
        
        for _, row in df.iterrows():
            # Calculate the actual calendar date for this water day
            calendar_date = water_year_start + timedelta(days=row['WaterDay'])
            # Use the calendar year of the actual date
            calendar_year = calendar_date.year
            
            # Track statistics
            year_counts[calendar_year] = year_counts.get(calendar_year, 0) + 1
            
            # Collect sample entries for verification
            if len(sample_entries) < 5 or len(sample_entries) < 10 and row['WaterDay'] > 300:
                sample_entries.append((row['WaterDay'], calendar_date.strftime('%Y-%m-%d'), calendar_year))
            
            f.write(f"{calendar_year}\t{row['WaterDay']:.5f}\t{row[temp_column]:.5f}\t1\n")
        
        # Report year distribution
        print(f"   📊 Calendar year distribution:")
        for year in sorted(year_counts.keys()):
            print(f"      {year}: {year_counts[year]} records")
        
        # Show sample entries for verification
        print(f"   🔍 Sample water day → calendar date mappings:")
        for wd, date_str, year in sample_entries:
            print(f"      WD {wd:.1f} → {date_str} (year {year})")

# ============================================================================
# DASH LAYOUT COMPONENTS
# ============================================================================

def create_header():
    """Create the application header."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("TTS MERGPARS", className="display-4 text-primary"),
                html.P("Temperature Time Series Merger and Processor", 
                      className="lead text-muted"),
                html.Hr()
            ])
        ])
    ], fluid=True)

def create_parameter_panel():
    """Create the parameter configuration panel."""
    default_params = load_default_parameters()
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Configuration Parameters", className="mb-0"),
            html.P(f"💡 Default values loaded from tts_mergpars.par", 
                   className="text-muted small mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Water Year"),
                    dbc.Input(
                        id="water-year-input",
                        type="number",
                        value=int(default_params.get('water_year', 2024)),
                        min=1900,
                        max=2100
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Data Interval (minutes)"),
                    dbc.Input(
                        id="interval-input",
                        type="number",
                        value=float(default_params.get('data_interval_min', 20.0)),
                        min=1,
                        max=1440,
                        step=0.1
                    )
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Gap Threshold Factor"),
                    dbc.Input(
                        id="gap-factor-input",
                        type="number",
                        value=float(default_params.get('gap_threshold_factor', 1.5)),
                        min=1.0,
                        max=10.0,
                        step=0.1
                    )
                ], width=6),
                dbc.Col([
                    dbc.Checklist(
                        options=[{"label": "Convert °F to °C", "value": 1}],
                        value=[1] if int(default_params.get('convert_f_to_c', 0)) else [],
                        id="temp-conversion-check",
                        inline=True
                    ),
                    dbc.Checklist(
                        options=[{"label": "Plot Relative to Start", "value": 1}],
                        value=[1] if int(default_params.get('plot_relative', 0)) else [],
                        id="plot-relative-check",
                        inline=True
                    ),
                    dbc.Checklist(
                        options=[{"label": "Show Temperature Difference", "value": 1}],
                        value=[],  # Not enabled by default
                        id="show-temp-diff-check",
                        inline=True
                    )
                ], width=6)
            ])
        ])
    ], className="mb-4")

def create_file_upload_section():
    """Create the file upload section."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Data Files", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Load Parameters File (.par)"),
                    dcc.Upload(
                        id='par-file-upload',
                        children=dbc.Button(
                            "Upload .par File",
                            color="secondary",
                            outline=True,
                            className="w-100"
                        ),
                        multiple=False,
                        accept='.par'
                    ),
                    html.Div(id="par-file-status", className="mt-2")
                ], width=12, className="mb-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Shallow Temperature Data (.csv)"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Upload(
                                id='shallow-file-upload',
                                children=dbc.Button(
                                    "Upload Shallow CSV",
                                    color="primary",
                                    outline=True,
                                    className="w-100"
                                ),
                                multiple=False,
                                accept='.csv'
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Button(
                                "Load from .par",
                                id="load-shallow-btn",
                                color="info",
                                outline=True,
                                size="sm",
                                className="w-100",
                                title="Load file specified in .par file"
                            )
                        ], width=4)
                    ]),
                    html.Div(id="shallow-file-status", className="mt-2")
                ], width=6),
                
                dbc.Col([
                    dbc.Label("Deep Temperature Data (.csv)"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Upload(
                                id='deep-file-upload',
                                children=dbc.Button(
                                    "Upload Deep CSV",
                                    color="primary",
                                    outline=True,
                                    className="w-100"
                                ),
                                multiple=False,
                                accept='.csv'
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Button(
                                "Load from .par",
                                id="load-deep-btn",
                                color="info",
                                outline=True,
                                size="sm",
                                className="w-100",
                                title="Load file specified in .par file"
                            )
                        ], width=4)
                    ]),
                    html.Div(id="deep-file-status", className="mt-2")
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Process Data",
                        id="process-button",
                        color="success",
                        size="lg",
                        className="w-100 mt-3",
                        disabled=True
                    )
                ], width=12)
            ])
        ])
    ], className="mb-4")

def create_visualization_section():
    """Create the main visualization section."""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.H5("Interactive Data Visualization", className="mb-0")
                ], width=6),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("Select Water Days", id="waterday-select-btn", color="primary", size="sm",
                                 title="Click to select water day range by clicking two points on the plot"),
                        dbc.Button("Reset Selection", id="reset-selection-btn", color="outline-secondary", size="sm",
                                 title="Clear current water day selection")
                    ], className="float-end")
                ], width=6)
            ])
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="main-plot",
                style={'height': '800px'},
                config={'displayModeBar': True, 'displaylogo': False}
            ),
            
            # Water Day Selection Controls
            dbc.Row([
                dbc.Col([
                    html.Div(id="waterday-selection-indicator", className="text-muted small"),
                    html.Div(id="selection-info", className="mt-2")
                ], width=8),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Start WD", className="small"),
                            dbc.Input(
                                id="start-wd-input",
                                type="number",
                                size="sm",
                                step=0.001,
                                placeholder="Start",
                                style={'font-size': '0.875rem'}
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("End WD", className="small"),
                            dbc.Input(
                                id="end-wd-input",
                                type="number",
                                size="sm",
                                step=0.001,
                                placeholder="End",
                                style={'font-size': '0.875rem'}
                            )
                        ], width=6)
                    ]),
                    dbc.Button(
                        "Apply Range",
                        id="apply-wd-range-btn",
                        color="primary",
                        size="sm",
                        className="w-100 mt-2",
                        title="Apply manually entered water day range"
                    )
                ], width=4)
            ], className="mt-2"),
            
            # Export section
            dbc.Row([
                dbc.Col([
                    dbc.Label("Export Filename"),
                    dbc.Input(id="export-filename", placeholder="Enter filename (no extension)")
                ], width=4),
                dbc.Col([
                    dbc.Label("Folder Name"),
                    dbc.Input(id="folder-name", value="processed", placeholder="Folder to save files")
                ], width=4),
                dbc.Col([
                    dbc.Label("Actions"),
                    dbc.Button("Save Files", id="save-files-btn", 
                             color="success", size="sm", className="w-100",
                             title="Save CSV and WYO files to folder")
                ], width=4)
            ], className="mt-3")
        ])
    ], className="mb-4")

def create_summary_section():
    """Create data summary section."""
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Data Summary", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Processing Log"),
                    html.Div(id="processing-log", 
                            style={'height': '200px', 'overflow-y': 'auto',
                                  'border': '1px solid #dee2e6', 'padding': '10px',
                                  'background-color': '#f8f9fa'})
                ], width=6),
                dbc.Col([
                    html.H6("Data Statistics"),
                    html.Div(id="data-statistics")
                ], width=6)
            ]),
            
            # File save status
            dbc.Row([
                dbc.Col([
                    html.Div(id="file-save-status", className="mt-3")
                ], width=12)
            ])
        ])
    ], className="mb-4")

# Main layout
app.layout = dbc.Container([
    create_header(),
    
    # Configuration Parameters and Data Files (Row 1)
    dbc.Row([
        dbc.Col([
            create_parameter_panel()
        ], width=6),
        dbc.Col([
            create_file_upload_section()
        ], width=6)
    ], className="mb-4"),
    
    # Interactive Data Visualization (Row 2)
    dbc.Row([
        dbc.Col([
            create_visualization_section()
        ], width=12)
    ], className="mb-4"),
    
    # Data Summary (Row 3)  
    dbc.Row([
        dbc.Col([
            create_summary_section()
        ], width=12)
    ]),
    
    # Store components for data persistence
    dcc.Store(id='shallow-data-store'),
    dcc.Store(id='deep-data-store'),
    dcc.Store(id='merged-data-store'),
    dcc.Store(id='current-parameters-store'),  # Store for current .par parameters
    dcc.Store(id='parameters-store'),
    dcc.Store(id='waterday-selection-state', data='inactive'),
    dcc.Store(id='selected-waterdays-store', data=[])
    
], fluid=True)

# ============================================================================
# DASH CALLBACKS
# ============================================================================

# Initialize current parameters store on startup
@app.callback(
    Output('current-parameters-store', 'data', allow_duplicate=True),
    Input('main-plot', 'id'),  # Dummy input to trigger on app load
    prevent_initial_call='initial_duplicate'
)
def initialize_parameters_store(dummy):
    """Initialize the parameters store with default values."""
    return load_default_parameters()

# Initialize processing log with default parameter info
@app.callback(
    Output('processing-log', 'children', allow_duplicate=True),
    Input('main-plot', 'id'),  # Dummy input to trigger on app load
    prevent_initial_call='initial_duplicate'
)
def initialize_processing_log(dummy):
    """Initialize processing log with default parameters info."""
    default_params = load_default_parameters()
    
    # Check if default .par file was found
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        par_file = os.path.join(script_dir, 'tts_mergpars.par')
        
        if os.path.exists(par_file):
            log_entries = [
                html.P("✓ Loaded default parameters from tts_mergpars.par", style={'color': 'green'}),
                html.P(f"📁 Output folder: {default_params.get('output_folder', 'processed')}", style={'color': 'blue'}),
                html.P(f"🗓️ Water Year: {default_params.get('water_year', 2024)}", style={'color': 'blue'}),
                html.P(f"📄 Expected shallow file: {default_params.get('filename_shallow', 'N/A')}", style={'color': 'blue'}),
                html.P(f"📄 Expected deep file: {default_params.get('filename_deep', 'N/A')}", style={'color': 'blue'}),
                html.P("💡 Use 'Load from .par' buttons to auto-load expected files", style={'color': 'gray'}),
                html.P("📤 Ready to upload or load CSV files and process data", style={'color': 'gray'})
            ]
        else:
            log_entries = [
                html.P("ℹ️ Using built-in default parameters (no tts_mergpars.par found)", style={'color': 'orange'}),
                html.P(f"📁 Output folder: {default_params.get('output_folder', 'processed')}", style={'color': 'blue'}),
                html.P("📤 Ready to upload CSV files and process data", style={'color': 'gray'})
            ]
    except:
        log_entries = [html.P("ℹ️ App initialized with default parameters", style={'color': 'gray'})]
    
    return html.Div(log_entries)

@app.callback(
    [Output('water-year-input', 'value'),
     Output('interval-input', 'value'),
     Output('gap-factor-input', 'value'),
     Output('temp-conversion-check', 'value'),
     Output('plot-relative-check', 'value'),
     Output('folder-name', 'value'),
     Output('par-file-status', 'children'),
     Output('processing-log', 'children', allow_duplicate=True),
     Output('current-parameters-store', 'data')],
    Input('par-file-upload', 'contents'),
    [State('par-file-upload', 'filename'),
     State('processing-log', 'children')],
    prevent_initial_call=True
)
def update_parameters_from_par_file(contents, filename, current_log):
    """Update parameter inputs when .par file is uploaded."""
    if contents is None:
        raise PreventUpdate
    
    try:
        print(f"\n=== Processing uploaded .par file: {filename} ===")
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        
        print(f"📄 File content preview:")
        for i, line in enumerate(decoded.split('\n')[:5], 1):
            print(f"   Line {i}: {line}")
        
        params = parse_par_content(decoded)
        
        print(f"🔧 Updating parameters from .par file:")
        for key, value in params.items():
            print(f"   {key} = {value}")
        
        # Add success message to log
        new_log_entry = [
            html.P(f"✓ Successfully loaded parameters from {filename}", style={'color': 'green'}),
            html.P(f"📄 Shallow file expected: {params.get('filename_shallow', 'N/A')}", style={'color': 'blue'}),
            html.P(f"📄 Deep file expected: {params.get('filename_deep', 'N/A')}", style={'color': 'blue'}),
            html.P("💡 Use 'Load from .par' buttons to auto-load these files", style={'color': 'gray'})
        ]
        par_status = dbc.Alert(f"✓ Parameters loaded from {filename}", color="success", dismissable=True)
        
        if isinstance(current_log, dict) and 'props' in current_log and 'children' in current_log['props']:
            updated_log = current_log['props']['children'] + new_log_entry
        else:
            updated_log = [current_log] + new_log_entry if current_log else new_log_entry
        
        return (
            int(params.get('water_year', 2024)),
            float(params.get('data_interval_min', 20.0)),
            float(params.get('gap_threshold_factor', 1.5)),
            [1] if int(params.get('convert_f_to_c', 0)) else [],
            [1] if int(params.get('plot_relative', 0)) else [],
            params.get('output_folder', 'processed'),
            par_status,
            html.Div(updated_log),
            params  # Store the parameters
        )
    except Exception as e:
        print(f"❌ Error loading .par file: {e}")
        import traceback
        traceback.print_exc()
        
        # Add error to log
        new_log_entry = [html.P(f"✗ Error loading .par file: {str(e)}", style={'color': 'red'})]
        par_status = dbc.Alert(f"✗ Error loading {filename}: {str(e)}", color="danger", dismissable=True)
        
        if isinstance(current_log, dict) and 'props' in current_log and 'children' in current_log['props']:
            updated_log = current_log['props']['children'] + new_log_entry
        else:
            updated_log = new_log_entry
        
        return no_update, no_update, no_update, no_update, no_update, no_update, par_status, html.Div(updated_log), no_update

@app.callback(
    [Output('shallow-file-status', 'children'),
     Output('shallow-data-store', 'data')],
    Input('shallow-file-upload', 'contents'),
    [State('shallow-file-upload', 'filename'),
     State('temp-conversion-check', 'value'),
     State('interval-input', 'value'),
     State('gap-factor-input', 'value')]
)
def process_shallow_file(contents, filename, convert_temp, interval, gap_factor):
    """Process uploaded shallow temperature file."""
    if contents is None:
        return "", None
    
    print(f"\n=== Processing Shallow File Upload ===")
    print(f"📁 File: {filename}")
    print(f"🔄 Convert F to C: {len(convert_temp) > 0 if convert_temp else False}")
    print(f"⏱️ Expected interval: {interval} minutes")
    print(f"🔍 Gap threshold factor: {gap_factor}")
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        
        convert_f = len(convert_temp) > 0 if convert_temp else False
        
        df, valid_records, errors, total_lines = read_logger_data_from_content(
            decoded, filename, "Shallow", convert_f, interval, gap_factor
        )
        
        gaps = detect_data_gaps(df, interval, gap_factor)
        
        print(f"✅ Shallow file processing complete:")
        print(f"   📊 {valid_records} valid records from {total_lines} total lines")
        print(f"   ❌ {errors} parsing errors")
        print(f"   🕳️ {len(gaps)} data gaps detected")
        print(f"   📈 Data range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        status = dbc.Alert([
            html.Strong("✓ Shallow file loaded successfully"),
            html.Br(),
            f"Records: {valid_records}, Errors: {errors}, Gaps: {len(gaps)}"
        ], color="success")
        
        return status, df.to_json(date_format='iso', orient='split')
        
    except Exception as e:
        print(f"❌ Error processing shallow file: {e}")
        status = dbc.Alert([
            html.Strong("✗ Error loading shallow file"),
            html.Br(),
            str(e)
        ], color="danger")
        return status, None

@app.callback(
    [Output('deep-file-status', 'children'),
     Output('deep-data-store', 'data')],
    Input('deep-file-upload', 'contents'),
    [State('deep-file-upload', 'filename'),
     State('temp-conversion-check', 'value'),
     State('interval-input', 'value'),
     State('gap-factor-input', 'value')]
)
def process_deep_file(contents, filename, convert_temp, interval, gap_factor):
    """Process uploaded deep temperature file."""
    if contents is None:
        return "", None
    
    print(f"\n=== Processing Deep File Upload ===")
    print(f"📁 File: {filename}")
    print(f"🔄 Convert F to C: {len(convert_temp) > 0 if convert_temp else False}")
    print(f"⏱️ Expected interval: {interval} minutes")
    print(f"🔍 Gap threshold factor: {gap_factor}")
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        
        convert_f = len(convert_temp) > 0 if convert_temp else False
        
        df, valid_records, errors, total_lines = read_logger_data_from_content(
            decoded, filename, "Deep", convert_f, interval, gap_factor
        )
        
        gaps = detect_data_gaps(df, interval, gap_factor)
        
        print(f"✅ Deep file processing complete:")
        print(f"   📊 {valid_records} valid records from {total_lines} total lines")
        print(f"   ❌ {errors} parsing errors")
        print(f"   🕳️ {len(gaps)} data gaps detected")
        print(f"   📈 Data range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        status = dbc.Alert([
            html.Strong("✓ Deep file loaded successfully"),
            html.Br(),
            f"Records: {valid_records}, Errors: {errors}, Gaps: {len(gaps)}"
        ], color="success")
        
        return status, df.to_json(date_format='iso', orient='split')
        
    except Exception as e:
        print(f"❌ Error processing deep file: {e}")
        status = dbc.Alert([
            html.Strong("✗ Error loading deep file"),
            html.Br(),
            str(e)
        ], color="danger")
        return status, None

@app.callback(
    [Output('shallow-file-status', 'children', allow_duplicate=True),
     Output('shallow-data-store', 'data', allow_duplicate=True)],
    Input('load-shallow-btn', 'n_clicks'),
    [State('current-parameters-store', 'data'),
     State('temp-conversion-check', 'value'),
     State('interval-input', 'value'),
     State('gap-factor-input', 'value')],
    prevent_initial_call=True
)
def load_shallow_from_par(n_clicks, stored_params, convert_temp, interval, gap_factor):
    """Load shallow temperature file specified in .par file."""
    if n_clicks is None:
        raise PreventUpdate
    
    print(f"\n=== Loading Shallow File from .par ===")
    
    try:
        # Use stored parameters or fall back to loading from file
        if stored_params:
            current_params = stored_params
            print("📋 Using parameters from uploaded .par file")
        else:
            current_params = load_default_parameters()
            print("📋 Using default parameters (no .par file uploaded)")
        
        shallow_filename = current_params.get('filename_shallow', '')
        
        if not shallow_filename:
            print(f"❌ No shallow filename specified in parameters")
            status = dbc.Alert("⚠️ No shallow filename found in parameters", color="warning")
            return status, None
        
        # Check if file exists in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, shallow_filename)
        
        if not os.path.exists(file_path):
            print(f"❌ Shallow file not found: {file_path}")
            status = dbc.Alert(f"✗ File not found: {shallow_filename}", color="danger")
            return status, None
        
        print(f"📁 Loading shallow file: {file_path}")
        print(f"🔄 Convert F to C: {len(convert_temp) > 0 if convert_temp else False}")
        print(f"⏱️ Expected interval: {interval} minutes")
        print(f"🔍 Gap threshold factor: {gap_factor}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        convert_f = len(convert_temp) > 0 if convert_temp else False
        
        df, valid_records, errors, total_lines = read_logger_data_from_content(
            file_content, shallow_filename, "Shallow", convert_f, interval, gap_factor
        )
        
        gaps = detect_data_gaps(df, interval, gap_factor)
        
        print(f"✅ Shallow file processing complete:")
        print(f"   📊 {valid_records} valid records from {total_lines} total lines")
        print(f"   ❌ {errors} parsing errors")
        print(f"   🕳️ {len(gaps)} data gaps detected")
        print(f"   📈 Data range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        status = dbc.Alert([
            html.Strong(f"✓ Loaded from .par: {shallow_filename}"),
            html.Br(),
            f"Records: {valid_records}, Errors: {errors}, Gaps: {len(gaps)}"
        ], color="success")
        
        return status, df.to_json(date_format='iso', orient='split')
        
    except Exception as e:
        print(f"❌ Error loading shallow file from .par: {e}")
        import traceback
        traceback.print_exc()
        status = dbc.Alert([
            html.Strong("✗ Error loading shallow file from .par"),
            html.Br(),
            str(e)
        ], color="danger")
        return status, None

@app.callback(
    [Output('deep-file-status', 'children', allow_duplicate=True),
     Output('deep-data-store', 'data', allow_duplicate=True)],
    Input('load-deep-btn', 'n_clicks'),
    [State('current-parameters-store', 'data'),
     State('temp-conversion-check', 'value'),
     State('interval-input', 'value'),
     State('gap-factor-input', 'value')],
    prevent_initial_call=True
)
def load_deep_from_par(n_clicks, stored_params, convert_temp, interval, gap_factor):
    """Load deep temperature file specified in .par file."""
    if n_clicks is None:
        raise PreventUpdate
    
    print(f"\n=== Loading Deep File from .par ===")
    
    try:
        # Use stored parameters or fall back to loading from file
        if stored_params:
            current_params = stored_params
            print("📋 Using parameters from uploaded .par file")
        else:
            current_params = load_default_parameters()
            print("📋 Using default parameters (no .par file uploaded)")
        
        deep_filename = current_params.get('filename_deep', '')
        
        if not deep_filename:
            print(f"❌ No deep filename specified in parameters")
            status = dbc.Alert("⚠️ No deep filename found in parameters", color="warning")
            return status, None
        
        # Check if file exists in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, deep_filename)
        
        if not os.path.exists(file_path):
            print(f"❌ Deep file not found: {file_path}")
            status = dbc.Alert(f"✗ File not found: {deep_filename}", color="danger")
            return status, None
        
        print(f"📁 Loading deep file: {file_path}")
        print(f"🔄 Convert F to C: {len(convert_temp) > 0 if convert_temp else False}")
        print(f"⏱️ Expected interval: {interval} minutes")
        print(f"🔍 Gap threshold factor: {gap_factor}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        convert_f = len(convert_temp) > 0 if convert_temp else False
        
        df, valid_records, errors, total_lines = read_logger_data_from_content(
            file_content, deep_filename, "Deep", convert_f, interval, gap_factor
        )
        
        gaps = detect_data_gaps(df, interval, gap_factor)
        
        print(f"✅ Deep file processing complete:")
        print(f"   📊 {valid_records} valid records from {total_lines} total lines")
        print(f"   ❌ {errors} parsing errors")
        print(f"   🕳️ {len(gaps)} data gaps detected")
        print(f"   📈 Data range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        status = dbc.Alert([
            html.Strong(f"✓ Loaded from .par: {deep_filename}"),
            html.Br(),
            f"Records: {valid_records}, Errors: {errors}, Gaps: {len(gaps)}"
        ], color="success")
        
        return status, df.to_json(date_format='iso', orient='split')
        
    except Exception as e:
        print(f"❌ Error loading deep file from .par: {e}")
        import traceback
        traceback.print_exc()
        status = dbc.Alert([
            html.Strong("✗ Error loading deep file from .par"),
            html.Br(),
            str(e)
        ], color="danger")
        return status, None

@app.callback(
    Output('process-button', 'disabled'),
    [Input('shallow-data-store', 'data'),
     Input('deep-data-store', 'data')]
)
def enable_process_button(shallow_data, deep_data):
    """Enable process button when both files are loaded."""
    return not (shallow_data is not None and deep_data is not None)

@app.callback(
    [Output('merged-data-store', 'data'),
     Output('processing-log', 'children'),
     Output('main-plot', 'figure'),
     Output('data-statistics', 'children'),
     Output('export-filename', 'value')],
    Input('process-button', 'n_clicks'),
    [State('shallow-data-store', 'data'),
     State('deep-data-store', 'data'),
     State('water-year-input', 'value'),
     State('plot-relative-check', 'value'),
     State('show-temp-diff-check', 'value'),
     State('shallow-file-upload', 'filename'),
     State('deep-file-upload', 'filename')],
    prevent_initial_call=True
)
def process_and_merge_data(
    n_clicks,           
    shallow_data,       
    deep_data,          
    water_year,         
    plot_relative,      
    show_temp_diff,     
    shallow_filename,   
    deep_filename       
):
    """
    Process and merge the temperature data with comprehensive validation.
    """
    if n_clicks is None or shallow_data is None or deep_data is None:
        return None, "", {}, "", ""
    
    try:
        # Handle None filenames (fallback to default names) 
        shallow_fname: str = shallow_filename if shallow_filename is not None else "shallow_data.csv"
        deep_fname: str = deep_filename if deep_filename is not None else "deep_data.csv"
            
        # Load data from stores
        df_shallow = pd.read_json(StringIO(shallow_data), orient='split')
        df_deep = pd.read_json(StringIO(deep_data), orient='split')
        
        # Convert DateTime columns
        df_shallow['DateTime'] = pd.to_datetime(df_shallow['DateTime'])
        df_deep['DateTime'] = pd.to_datetime(df_deep['DateTime'])
        
        print('\n=== Processing Water Years ===')
        print(f"Water Year: {water_year} ({'LEAP' if is_leap_year(water_year) else 'NON-LEAP'} - {get_water_year_length(water_year)} days)")
        
        # Calculate water day columns
        df_shallow = calculate_water_day_columns(df_shallow, water_year)
        df_deep = calculate_water_day_columns(df_deep, water_year)
        
        # Validate water year data
        print(f"\nValidating data for Water Year {water_year}:")
        shallow_valid = validate_water_year_data(df_shallow, 'Shallow', water_year)
        deep_valid = validate_water_year_data(df_deep, 'Deep', water_year)
        
        # Merge data
        merged_df = merge_temperature_data(df_shallow, df_deep)
        
        # Create processing log with comprehensive information
        log_entries = [
            f"✓ Processed {len(df_shallow)} shallow records from {shallow_fname}",
            f"✓ Processed {len(df_deep)} deep records from {deep_fname}", 
            f"✓ Merged to {len(merged_df)} records",
            f"✓ Water Year: {water_year} ({'LEAP' if is_leap_year(water_year) else 'NON-LEAP'} - {get_water_year_length(water_year)} days)",
            f"✓ Date range: {merged_df['DateTime'].min()} to {merged_df['DateTime'].max()}",
            f"✓ Water Day range: {merged_df['WaterDay'].min():.5f} to {merged_df['WaterDay'].max():.5f}"
        ]
        
        # Add validation warnings to log
        if not shallow_valid:
            log_entries.append("⚠️ Shallow data contains records outside specified water year")
        if not deep_valid:
            log_entries.append("⚠️ Deep data contains records outside specified water year")
        
        # Check for anomalous data
        negative_wd = merged_df[merged_df['WaterDay'] < 0]
        if len(negative_wd) > 0:
            log_entries.append(f"⚠️ Found {len(negative_wd)} records with negative WaterDay (pre-season data)")
            log_entries.append(f"   WaterDay range: {negative_wd['WaterDay'].min():.5f} to {negative_wd['WaterDay'].max():.5f}")
        
        post_season = merged_df[merged_df['WaterDay'] > get_water_year_length(water_year)]
        if len(post_season) > 0:
            log_entries.append(f"⚠️ Found {len(post_season)} records with WaterDay > {get_water_year_length(water_year)} (post-season data)")
            log_entries.append(f"   WaterDay range: {post_season['WaterDay'].min():.5f} to {post_season['WaterDay'].max():.5f}")
            log_entries.append("   These likely belong to the next water year")
        
        # Add comprehensive summary like the original
        log_entries.extend([
            f"📊 Temperature Summary:",
            f"   Shallow: {merged_df['T_shallow'].mean():.2f} ± {merged_df['T_shallow'].std():.2f} °C",
            f"   Deep: {merged_df['T_deep'].mean():.2f} ± {merged_df['T_deep'].std():.2f} °C",
            f"   Difference: {(merged_df['T_shallow'] - merged_df['T_deep']).mean():.2f} ± {(merged_df['T_shallow'] - merged_df['T_deep']).std():.2f} °C"
        ])
        
        log_div = html.Div([html.P(entry) for entry in log_entries])
        
        # Create main plot
        plot_relative_bool = len(plot_relative) > 0 if plot_relative else False
        show_temp_diff_bool = len(show_temp_diff) > 0 if show_temp_diff else False
        fig = create_main_plot(merged_df, water_year, plot_relative_bool, show_temp_diff_bool)
        
        # Create statistics
        stats = create_data_statistics(merged_df, df_shallow, df_deep)
        
        # Create default filename based on input files and water year (correct format)
        default_filename = create_default_filename(shallow_fname, deep_fname, water_year)
        
        return merged_df.to_json(date_format='iso', orient='split'), log_div, fig, stats, default_filename
        
    except Exception as e:
        print(f"Error in process_and_merge_data: {e}")
        import traceback
        traceback.print_exc()
        error_log = html.Div([html.P(f"✗ Error: {str(e)}", style={'color': 'red'})])
        return None, error_log, {}, "", ""

def create_main_plot(merged_df, water_year, plot_relative=False, show_temp_diff=False):
    """Create the main temperature plot with enhanced features."""
    wd_values = merged_df['WaterDay'].values
    offset = wd_values.min() if plot_relative else 0.0
    wd_plot = wd_values - offset
    
    fig = go.Figure()
    
    # Add temperature traces
    fig.add_trace(go.Scatter(
        x=wd_plot,
        y=merged_df['T_shallow'],
        mode='lines',
        name='Shallow',
        line=dict(color='blue', width=1),
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=wd_plot,
        y=merged_df['T_deep'],
        mode='lines',
        name='Deep',
        line=dict(color='red', width=1),
        opacity=0.8
    ))
    
    # Conditionally add temperature difference
    if show_temp_diff:
        temp_diff = merged_df['T_shallow'] - merged_df['T_deep']
        fig.add_trace(go.Scatter(
            x=wd_plot,
            y=temp_diff,
            mode='lines',
            name='Difference (S-D)',
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.6,
            yaxis='y2'
        ))
    
    # Calculate data ranges for proper scaling
    x_min, x_max = wd_plot.min(), wd_plot.max()
    x_range = x_max - x_min
    x_padding = max(x_range * 0.02, 0.1)
    
    y_min = min(merged_df['T_shallow'].min(), merged_df['T_deep'].min())
    y_max = max(merged_df['T_shallow'].max(), merged_df['T_deep'].max())
    y_range = y_max - y_min
    y_padding = max(y_range * 0.05, 0.5)
    
    # Enhanced tick spacing based on data range
    data_range_days = x_max - x_min
    if data_range_days <= 30:
        dtick = 5
    elif data_range_days <= 90:
        dtick = 10
    elif data_range_days <= 180:
        dtick = 20
    else:
        dtick = 50
    
    # Layout configuration
    layout_config = {
        'title': f'Temperature Data - Water Year {water_year} (Click to select Water Day range)',
        'xaxis_title': 'Water Day (relative to start)' if plot_relative else 'Water Day',
        'yaxis_title': 'Temperature (°C)',
        'xaxis': dict(
            range=[x_min - x_padding, x_max + x_padding],
            autorange=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick=dtick
        ),
        'yaxis': dict(
            range=[y_min - y_padding, y_max + y_padding],
            autorange=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        'hovermode': 'x unified',
        'showlegend': True,
        'height': 800,
        'plot_bgcolor': 'white'
    }
    
    # Add secondary y-axis only if showing temperature difference
    if show_temp_diff:
        temp_diff = merged_df['T_shallow'] - merged_df['T_deep']
        diff_min, diff_max = temp_diff.min(), temp_diff.max()
        diff_range = diff_max - diff_min
        diff_padding = max(diff_range * 0.1, 0.1) if diff_range > 0 else 1.0
        
        layout_config['yaxis2'] = dict(
            title='Temperature Difference (°C)',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[diff_min - diff_padding, diff_max + diff_padding],
            autorange=False
        )
    
    # Add water year boundaries
    water_year_length = get_water_year_length(water_year)
    if not plot_relative:
        x_range_with_padding = [x_min - x_padding, x_max + x_padding]
        
        if x_range_with_padding[0] <= 0 <= x_range_with_padding[1]:
            wy_start_date = datetime(water_year - 1, 10, 1)
            fig.add_vline(x=0, line=dict(color='green', dash='dash', width=2),
                         annotation_text=f"WY Start ({wy_start_date.strftime('%b %d, %Y')})", 
                         annotation_position="top")
        
        if x_range_with_padding[0] <= water_year_length <= x_range_with_padding[1]:
            wy_end_date = datetime(water_year, 9, 30)
            fig.add_vline(x=water_year_length, line=dict(color='orange', dash='dash', width=2),
                         annotation_text=f"WY End ({wy_end_date.strftime('%b %d, %Y')}) - Day {water_year_length}", 
                         annotation_position="top")
    
    # Apply layout
    fig.update_layout(layout_config)
    
    return fig

def create_data_statistics(merged_df, df_shallow, df_deep):
    """Create comprehensive data statistics display."""
    stats = []
    
    # Basic statistics
    stats.extend([
        html.P(f"Merged Records: {len(merged_df)}"),
        html.P(f"Shallow Records: {len(df_shallow)}"),
        html.P(f"Deep Records: {len(df_deep)}"),
        html.Hr()
    ])
    
    # Temperature statistics
    if len(merged_df) > 0:
        shallow_stats = merged_df['T_shallow'].describe()
        deep_stats = merged_df['T_deep'].describe()
        diff_stats = (merged_df['T_shallow'] - merged_df['T_deep']).describe()
        
        stats.extend([
            html.H6("Temperature Statistics (°C)"),
            html.P(f"Shallow: {shallow_stats['mean']:.2f} ± {shallow_stats['std']:.2f} °C"),
            html.P(f"  Range: {shallow_stats['min']:.2f} to {shallow_stats['max']:.2f} °C"),
            html.P(f"Deep: {deep_stats['mean']:.2f} ± {deep_stats['std']:.2f} °C"),
            html.P(f"  Range: {deep_stats['min']:.2f} to {deep_stats['max']:.2f} °C"),
            html.P(f"Difference (S-D): {diff_stats['mean']:.2f} ± {diff_stats['std']:.2f} °C"),
            html.P(f"  Range: {diff_stats['min']:.2f} to {diff_stats['max']:.2f} °C"),
            html.Hr()
        ])
        
        # Data quality indicators
        time_diffs = merged_df['DateTime'].diff().dt.total_seconds() / 60.0
        avg_interval = time_diffs.mean()
        water_year = merged_df['WaterYear'].iloc[0] if len(merged_df) > 0 else None
        
        stats.extend([
            html.H6("Data Quality"),
            html.P(f"Average Interval: {avg_interval:.1f} minutes"),
            html.P(f"Coverage: {(merged_df['WaterDay'].max() - merged_df['WaterDay'].min()):.1f} days"),
        ])
        
        # Water year validation info
        if water_year:
            water_year_length = get_water_year_length(water_year)
            negative_wd = merged_df[merged_df['WaterDay'] < 0]
            post_season = merged_df[merged_df['WaterDay'] > water_year_length]
            
            if len(negative_wd) > 0:
                stats.append(html.P(f"⚠️ Pre-season data: {len(negative_wd)} records", style={'color': 'orange'}))
            if len(post_season) > 0:
                stats.append(html.P(f"⚠️ Post-season data: {len(post_season)} records", style={'color': 'orange'}))
            
            in_season = merged_df[(merged_df['WaterDay'] >= 0) & (merged_df['WaterDay'] <= water_year_length)]
            stats.append(html.P(f"✓ In-season data: {len(in_season)} records", style={'color': 'green'}))
    
    return html.Div(stats)

# Water day selection callbacks
@app.callback(
    [Output('waterday-selection-state', 'data'),
     Output('waterday-selection-indicator', 'children'),
     Output('waterday-select-btn', 'children'),
     Output('waterday-select-btn', 'color')],
    [Input('waterday-select-btn', 'n_clicks')]
)
def toggle_waterday_selection(n_clicks):
    """Toggle water day selection mode."""
    if n_clicks is None:
        return 'inactive', "", "Select Water Days", 'primary'
    
    if n_clicks % 2 == 1:  # Odd clicks = active
        global_data['waterday_selection_mode'] = True
        global_data['selected_waterdays'] = []
        status = "🎯 Water Day Selection Mode ACTIVE: Click on the plot to select start and end water days"
        return 'active', status, "Exit Selection Mode", 'warning'
    else:  # Even clicks = inactive
        global_data['waterday_selection_mode'] = False
        return 'inactive', "", "Select Water Days", 'primary'

# Combined callback for handling both plot clicks and manual input
@app.callback(
    [Output('selected-waterdays-store', 'data'),
     Output('selection-info', 'children'),
     Output('main-plot', 'figure', allow_duplicate=True),
     Output('start-wd-input', 'value'),
     Output('end-wd-input', 'value')],
    [Input('main-plot', 'clickData'),
     Input('reset-selection-btn', 'n_clicks'),
     Input('apply-wd-range-btn', 'n_clicks')],
    [State('waterday-selection-state', 'data'),
     State('selected-waterdays-store', 'data'),
     State('merged-data-store', 'data'),
     State('water-year-input', 'value'),
     State('plot-relative-check', 'value'),
     State('show-temp-diff-check', 'value'),
     State('main-plot', 'figure'),
     State('start-wd-input', 'value'),
     State('end-wd-input', 'value')],
    prevent_initial_call=True
)
def handle_waterday_selection(click_data, reset_clicks, apply_clicks, selection_state, selected_waterdays, 
                             merged_data, water_year, plot_relative, show_temp_diff, current_fig,
                             start_wd_manual, end_wd_manual):
    """Handle water day selection via plot clicks or manual input."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'reset-selection-btn' and reset_clicks:
        # Reset selection but preserve zoom
        print(f"\n🔄 Water Day Selection Reset")
        global_data['selected_waterdays'] = []
        
        if merged_data:
            merged_df = pd.read_json(StringIO(merged_data), orient='split')
            merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
            
            plot_relative_bool = len(plot_relative) > 0 if plot_relative else False
            show_temp_diff_bool = len(show_temp_diff) > 0 if show_temp_diff else False
            fig = create_main_plot(merged_df, water_year, plot_relative_bool, show_temp_diff_bool)
            
            # Preserve zoom level from current figure
            if current_fig and 'layout' in current_fig:
                layout = current_fig['layout']
                if 'xaxis' in layout and 'range' in layout['xaxis']:
                    fig.update_layout(xaxis_range=layout['xaxis']['range'])
                    print(f"🔍 Preserved zoom: X-axis {layout['xaxis']['range']}")
                if 'yaxis' in layout and 'range' in layout['yaxis']:
                    fig.update_layout(yaxis_range=layout['yaxis']['range'])
            
            print(f"✅ Selection cleared, zoom preserved")
            return [], "", fig, None, None
        else:
            return [], "", no_update, None, None
    
    elif trigger_id == 'apply-wd-range-btn' and apply_clicks:
        # Manual water day range input
        if start_wd_manual is not None and end_wd_manual is not None and merged_data:
            print(f"\n📝 Manual Water Day Range Input")
            print(f"   Start WD: {start_wd_manual}")
            print(f"   End WD: {end_wd_manual}")
            
            new_selection = sorted([start_wd_manual, end_wd_manual])
            start_wd, end_wd = new_selection[0], new_selection[1]
            
            merged_df = pd.read_json(StringIO(merged_data), orient='split')
            merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
            
            # Calculate selection info
            mask = (merged_df['WaterDay'] >= start_wd) & (merged_df['WaterDay'] <= end_wd)
            selected_records = mask.sum()
            
            if selected_records > 0:
                selected_df = merged_df[mask]
                date_start = selected_df['DateTime'].min().strftime('%Y-%m-%d')
                date_end = selected_df['DateTime'].max().strftime('%Y-%m-%d')
                duration_days = end_wd - start_wd
                
                info_text = f"📊 Manual Selection: Water Days {start_wd:.3f} to {end_wd:.3f} ({duration_days:.1f} days, {selected_records} records)"
                info_text += f"\n📅 Date Range: {date_start} to {date_end}"
                
                print(f"   Duration: {duration_days:.1f} days")
                print(f"   Records: {selected_records}")
                print(f"   Date range: {date_start} to {date_end}")
            else:
                info_text = f"⚠️ No records in selected range: {start_wd:.3f} to {end_wd:.3f}"
                print(f"⚠️ No records in selected range")
            
            # Update global state
            global_data['selected_waterdays'] = new_selection
            
            # Update plot
            plot_relative_bool = len(plot_relative) > 0 if plot_relative else False
            show_temp_diff_bool = len(show_temp_diff) > 0 if show_temp_diff else False
            fig = create_main_plot(merged_df, water_year, plot_relative_bool, show_temp_diff_bool)
            
            # PRESERVE ZOOM LEVEL from current figure
            if current_fig and 'layout' in current_fig:
                layout = current_fig['layout']
                if 'xaxis' in layout and 'range' in layout['xaxis']:
                    fig.update_layout(xaxis_range=layout['xaxis']['range'])
                if 'yaxis' in layout and 'range' in layout['yaxis']:
                    fig.update_layout(yaxis_range=layout['yaxis']['range'])
            
            # Add selection visualization
            if plot_relative_bool:
                offset = merged_df['WaterDay'].min()
                start_plot = start_wd - offset
                end_plot = end_wd - offset
            else:
                start_plot, end_plot = start_wd, end_wd
            
            # Add shaded region
            fig.add_vrect(x0=start_plot, x1=end_plot, 
                          fillcolor="yellow", opacity=0.3,
                          line_width=0)
            
            # Add vertical lines
            fig.add_vline(x=start_plot, line_dash="dash", line_color="green", line_width=3,
                         annotation_text=f"Start: {start_wd:.3f}", 
                         annotation_position="top left",
                         annotation_yshift=10)
            fig.add_vline(x=end_plot, line_dash="dash", line_color="red", line_width=3,
                         annotation_text=f"End: {end_wd:.3f}", 
                         annotation_position="top right",
                         annotation_yshift=10)
            
            return new_selection, info_text, fig, start_wd, end_wd
        else:
            raise PreventUpdate
    
    elif (trigger_id == 'main-plot' and selection_state == 'active' and 
          click_data is not None and 'points' in click_data and len(click_data['points']) > 0):
        
        waterday_clicked = click_data['points'][0]['x']
        
        # Handle relative plotting offset
        if merged_data:
            merged_df = pd.read_json(StringIO(merged_data), orient='split')
            merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
            
            plot_relative_bool = len(plot_relative) > 0 if plot_relative else False
            if plot_relative_bool:
                # Convert relative position back to absolute water day
                offset = merged_df['WaterDay'].min()
                waterday_clicked += offset
                print(f"🎯 Click detected: Relative {click_data['points'][0]['x']:.3f} → Absolute {waterday_clicked:.3f}")
            else:
                print(f"🎯 Click detected: Water Day {waterday_clicked:.3f}")
        
        # Update selected water days (same logic as before)
        if len(selected_waterdays) == 0:
            new_selection = [waterday_clicked]
            info_text = f"📍 First point selected: Water Day {waterday_clicked:.3f}. Click again to set end point."
            print(f"📍 First point selected: WD {waterday_clicked:.3f}")
            
            start_wd_display = waterday_clicked
            end_wd_display = None
            
        elif len(selected_waterdays) == 1:
            new_selection = selected_waterdays + [waterday_clicked]
            freqs_sorted = sorted(new_selection)
            start_wd, end_wd = freqs_sorted[0], freqs_sorted[1]
            
            print(f"📍 Second point selected: WD {waterday_clicked:.3f}")
            print(f"📊 Selection range: WD {start_wd:.3f} to {end_wd:.3f}")
            
            # Calculate selection info
            if merged_data:
                mask = (merged_df['WaterDay'] >= start_wd) & (merged_df['WaterDay'] <= end_wd)
                selected_records = mask.sum()
                
                if selected_records > 0:
                    selected_df = merged_df[mask]
                    date_start = selected_df['DateTime'].min().strftime('%Y-%m-%d')
                    date_end = selected_df['DateTime'].max().strftime('%Y-%m-%d')
                    duration_days = end_wd - start_wd
                    
                    info_text = f"📊 Selected: Water Days {start_wd:.3f} to {end_wd:.3f} ({duration_days:.1f} days, {selected_records} records)"
                    info_text += f"\n📅 Date Range: {date_start} to {date_end}"
                    
                    print(f"   Duration: {duration_days:.1f} days")
                    print(f"   Records: {selected_records}")
                    print(f"   Date range: {date_start} to {date_end}")
                else:
                    info_text = f"⚠️ No records in selected range: {start_wd:.3f} to {end_wd:.3f}"
                    print(f"⚠️ No records in selected range")
            else:
                info_text = f"📊 Selected: Water Days {start_wd:.3f} to {end_wd:.3f}"
                
            start_wd_display = start_wd
            end_wd_display = end_wd
            
        else:
            # Reset to single selection
            new_selection = [waterday_clicked]
            info_text = f"📍 New selection started: Water Day {waterday_clicked:.3f}. Click again to set end point."
            print(f"🔄 New selection started: WD {waterday_clicked:.3f}")
            
            start_wd_display = waterday_clicked
            end_wd_display = None
        
        # Update global state
        global_data['selected_waterdays'] = new_selection
        
        # Update plot with selection visualization and PRESERVE ZOOM
        if merged_data:
            plot_relative_bool = len(plot_relative) > 0 if plot_relative else False
            show_temp_diff_bool = len(show_temp_diff) > 0 if show_temp_diff else False
            fig = create_main_plot(merged_df, water_year, plot_relative_bool, show_temp_diff_bool)
            
            # PRESERVE ZOOM LEVEL from current figure
            if current_fig and 'layout' in current_fig:
                layout = current_fig['layout']
                if 'xaxis' in layout and 'range' in layout['xaxis']:
                    fig.update_layout(xaxis_range=layout['xaxis']['range'])
                if 'yaxis' in layout and 'range' in layout['yaxis']:
                    fig.update_layout(yaxis_range=layout['yaxis']['range'])
                print(f"🔍 Zoom preserved during selection")
            
            # Add selection visualization
            if len(new_selection) == 1:
                # FIRST CLICK: Show single vertical line
                waterday_for_plot = new_selection[0]
                
                # Apply offset for relative plotting
                if plot_relative_bool:
                    offset = merged_df['WaterDay'].min()
                    waterday_plot = waterday_for_plot - offset
                else:
                    waterday_plot = waterday_for_plot
                
                # Add single vertical line for first click
                fig.add_vline(x=waterday_plot, line_dash="dash", line_color="blue", line_width=3,
                             annotation_text=f"Start: {waterday_for_plot:.3f}", 
                             annotation_position="top left",
                             annotation_yshift=10)
                print(f"🎨 Added start line visualization")
                
            elif len(new_selection) >= 2:
                # SECOND CLICK: Show shaded region between two points
                freqs_sorted = sorted(new_selection)
                start_wd, end_wd = freqs_sorted[0], freqs_sorted[1]
                
                # Apply offset for relative plotting
                if plot_relative_bool:
                    offset = merged_df['WaterDay'].min()
                    start_plot = start_wd - offset
                    end_plot = end_wd - offset
                else:
                    start_plot, end_plot = start_wd, end_wd
                
                # Add shaded region
                fig.add_vrect(x0=start_plot, x1=end_plot, 
                              fillcolor="yellow", opacity=0.3,
                              line_width=0)
                
                # Add vertical lines
                fig.add_vline(x=start_plot, line_dash="dash", line_color="green", line_width=3,
                             annotation_text=f"Start: {start_wd:.3f}", 
                             annotation_position="top left",
                             annotation_yshift=10)
                fig.add_vline(x=end_plot, line_dash="dash", line_color="red", line_width=3,
                             annotation_text=f"End: {end_wd:.3f}", 
                             annotation_position="top right",
                             annotation_yshift=10)
                print(f"🎨 Added selection range visualization")
            
            return new_selection, info_text, fig, start_wd_display, end_wd_display
        else:
            return new_selection, info_text, no_update, start_wd_display, end_wd_display
    
    raise PreventUpdate

# Save files callback
@app.callback(
    Output('file-save-status', 'children'),
    Input('save-files-btn', 'n_clicks'),
    [State('merged-data-store', 'data'),
     State('export-filename', 'value'),
     State('folder-name', 'value'),
     State('water-year-input', 'value'),
     State('selected-waterdays-store', 'data')],
    prevent_initial_call=True
)
def save_files_to_folder_callback(n_clicks, merged_data, filename, folder_name, water_year, selected_waterdays):
    """Save CSV and WYO files directly to folder."""
    if n_clicks is None or merged_data is None:
        raise PreventUpdate
    
    print(f"\n=== File Export Operation ===")
    print(f"🗂️ Target folder: {folder_name or 'processed'}")
    print(f"📝 Filename base: {filename or 'auto-generated'}")
    print(f"📅 Water year: {water_year}")
    
    try:
        # Load merged data
        merged_df = pd.read_json(StringIO(merged_data), orient='split')
        merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
        
        print(f"📊 Original dataset: {len(merged_df)} records")
        print(f"📈 Data range: WD {merged_df['WaterDay'].min():.3f} to {merged_df['WaterDay'].max():.3f}")
        
        # Apply water day selection if any
        if len(selected_waterdays) >= 2:
            freqs_sorted = sorted(selected_waterdays)
            start_wd, end_wd = freqs_sorted[0], freqs_sorted[1]
            mask = (merged_df['WaterDay'] >= start_wd) & (merged_df['WaterDay'] <= end_wd)
            export_df = merged_df[mask]
            
            if len(export_df) == 0:
                print(f"❌ No data in selected range: WD {start_wd:.3f} to {end_wd:.3f}")
                return dbc.Alert("⚠️ No data in selected water day range", color="warning")
            
            selection_info = f" (Water Days {start_wd:.3f} to {end_wd:.3f})"
            filename_suffix = f"_WD{start_wd:.1f}-{end_wd:.1f}"
            print(f"🎯 Selection applied: WD {start_wd:.3f} to {end_wd:.3f}")
            print(f"📊 Selected dataset: {len(export_df)} records")
        else:
            export_df = merged_df
            selection_info = " (Full dataset)"
            filename_suffix = "_full"  # Use "_full" suffix for complete dataset
            print(f"📋 Exporting full dataset: {len(export_df)} records")
        
        # Use provided filename or default
        if not filename or filename.strip() == "":
            filename_base = f"TempDataSD{filename_suffix}"
            print(f"🏷️ Auto-generated filename: {filename_base}")
        else:
            filename_base = f"{filename.strip()}{filename_suffix}"
            print(f"🏷️ User filename: {filename_base}")
        
        # Use provided folder or default
        if not folder_name or folder_name.strip() == "":
            folder_name = "processed"
        
        print(f"💾 Saving files with format:")
        print(f"   CSV: WaterDay (6 decimals), Temperatures (5 decimals)")
        print(f"   WYO: Year, WaterDay, Temperature, DepthID format")
        
        # Save files
        success, file_paths, message = save_files_to_folder(export_df, filename_base, folder_name, water_year)
        
        if success:
            # Create success message with file details
            file_list = [os.path.basename(path) for path in file_paths]
            
            print(f"✅ Export successful{selection_info}")
            print(f"📁 Files saved to: {folder_name}/")
            for i, (path, name) in enumerate(zip(file_paths, file_list), 1):
                file_size = os.path.getsize(path) / 1024  # KB
                print(f"   {i}. {name} ({file_size:.1f} KB)")
            print(f"📊 Total records exported: {len(export_df)}")
            
            alert_content = [
                html.Strong(f"✓ Files saved successfully{selection_info}"),
                html.Br(),
                html.Small(f"Folder: {folder_name}/"),
                html.Br(),
                html.Small(f"Files: {', '.join(file_list)}"),
                html.Br(),
                html.Small(f"Records exported: {len(export_df)}")
            ]
            
            return dbc.Alert(alert_content, color="success", dismissable=True)
        else:
            print(f"❌ Export failed: {message}")
            return dbc.Alert(f"✗ {message}", color="danger", dismissable=True)
            
    except Exception as e:
        print(f"❌ Export error: {str(e)}")
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"✗ Error saving files: {str(e)}", color="danger", dismissable=True)

# Callback to refresh plot when temperature difference option changes
@app.callback(
    Output('main-plot', 'figure', allow_duplicate=True),
    Input('show-temp-diff-check', 'value'),
    [State('merged-data-store', 'data'),
     State('water-year-input', 'value'),
     State('plot-relative-check', 'value'),
     State('main-plot', 'figure'),
     State('selected-waterdays-store', 'data')],
    prevent_initial_call=True
)
def update_plot_for_temp_diff(show_temp_diff, merged_data, water_year, plot_relative, current_fig, selected_waterdays):
    """Update plot when temperature difference option changes."""
    if merged_data is None:
        raise PreventUpdate
    
    try:
        merged_df = pd.read_json(StringIO(merged_data), orient='split')
        merged_df['DateTime'] = pd.to_datetime(merged_df['DateTime'])
        
        plot_relative_bool = len(plot_relative) > 0 if plot_relative else False
        show_temp_diff_bool = len(show_temp_diff) > 0 if show_temp_diff else False
        
        # Create new plot with updated temperature difference setting
        fig = create_main_plot(merged_df, water_year, plot_relative_bool, show_temp_diff_bool)
        
        # PRESERVE ZOOM LEVEL from current figure
        if current_fig and 'layout' in current_fig:
            layout = current_fig['layout']
            if 'xaxis' in layout and 'range' in layout['xaxis']:
                fig.update_layout(xaxis_range=layout['xaxis']['range'])
            if 'yaxis' in layout and 'range' in layout['yaxis']:
                fig.update_layout(yaxis_range=layout['yaxis']['range'])
        
        # Restore any water day selection visualization
        if len(selected_waterdays) >= 1:
            if len(selected_waterdays) == 1:
                # Show single vertical line for first click
                waterday_for_plot = selected_waterdays[0]
                
                # Apply offset for relative plotting
                if plot_relative_bool:
                    offset = merged_df['WaterDay'].min()
                    waterday_plot = waterday_for_plot - offset
                else:
                    waterday_plot = waterday_for_plot
                
                fig.add_vline(x=waterday_plot, line_dash="dash", line_color="blue", line_width=3,
                             annotation_text=f"Start: {waterday_for_plot:.3f}", 
                             annotation_position="top left",
                             annotation_yshift=10)
                
            elif len(selected_waterdays) >= 2:
                # Show shaded region between two points
                freqs_sorted = sorted(selected_waterdays)
                start_wd, end_wd = freqs_sorted[0], freqs_sorted[1]
                
                # Apply offset for relative plotting
                if plot_relative_bool:
                    offset = merged_df['WaterDay'].min()
                    start_plot = start_wd - offset
                    end_plot = end_wd - offset
                else:
                    start_plot, end_plot = start_wd, end_wd
                
                # Add shaded region
                fig.add_vrect(x0=start_plot, x1=end_plot, 
                              fillcolor="yellow", opacity=0.3,
                              line_width=0)
                
                # Add vertical lines
                fig.add_vline(x=start_plot, line_dash="dash", line_color="green", line_width=3,
                             annotation_text=f"Start: {start_wd:.3f}", 
                             annotation_position="top left",
                             annotation_yshift=10)
                fig.add_vline(x=end_plot, line_dash="dash", line_color="red", line_width=3,
                             annotation_text=f"End: {end_wd:.3f}", 
                             annotation_position="top right",
                             annotation_yshift=10)
        
        return fig
        
    except Exception as e:
        print(f"Error updating plot for temp diff: {e}")
        raise PreventUpdate

# ============================================================================
# RUN APPLICATION
# ============================================================================

def cleanup_on_exit():
    """Clean up logging when application exits."""
    try:
        if logger:
            print(f"\n🏁 Application shutting down...")
            print(f"📝 Closing log file: {log_file_path}")
            sys.stdout = logger.terminal  # Restore original stdout
            logger.close()
    except:
        pass

# Register cleanup function
import atexit
atexit.register(cleanup_on_exit)

# Also handle Ctrl+C gracefully
import signal

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print(f"\n\n🛑 Received interrupt signal (Ctrl+C)")
    cleanup_on_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    print("🌐 Starting web application...")
    print("🔗 Open your browser to: http://127.0.0.1:8050")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        app.run(debug=True, port=8050)
    except KeyboardInterrupt:
        print(f"\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
    finally:
        cleanup_on_exit()