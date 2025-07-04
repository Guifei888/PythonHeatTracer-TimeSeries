#!/usr/bin/env python3
"""
TTS_MERPAR.py - Temperature Time Series Merger and Processor

This script processes temperature data from shallow and deep water temperature loggers,
merges them based on water day calculations, and provides an interactive interface
for selecting and exporting data intervals.

Main Features:
- Reads temperature data from CSV files (shallow and deep loggers)
- Converts data to water year format (Oct 1 - Sep 30)
- Merges datasets based on timestamp alignment
- Interactive plotting with interval selection
- Exports data in multiple formats (CSV, WYO)

Enhanced Features (v2.0):
- Adaptive zoom-responsive axis ticks
- Editable selection bounds via text boxes
- Improved user interface with better text handling
- Cleaner filename management
- Enhanced temperature axis formatting
- Better debug output explanations

Author: Timothy Wu
Created: 6/26/2025
Last Updated: 7/3/2025
Version: 2.0

Requirements:
- pandas, numpy, matplotlib
- Input parameter file: tts_merpar.par
- Two CSV files with temperature data

Usage:
    python tts_merpar.py
    
The script looks for tts_merpar.par in the current directory or script directory.
"""

import os
import sys
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
from matplotlib.widgets import SpanSelector, Button, TextBox
from datetime import datetime


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
        self.log_file.write(f"=== TTS_MERPAR Processing Log ===\n")
        self.log_file.write(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"Command: {' '.join(sys.argv)}\n")
        self.log_file.write("=" * 50 + "\n\n")
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
        
        footer = f"\n\n{'=' * 50}\n"
        footer += f"Processing completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"Total duration: {duration}\n"
        footer += f"Log file: {self.log_file.name}\n"
        
        self.write(footer)
        self.log_file.close()


class AdaptiveAxisManager:
    """
    Manages adaptive tick formatting for both time and temperature axes.
    Automatically adjusts tick density and precision based on zoom level.
    """
    
    def __init__(self, ax, wd_values, offset, plot_relative):
        self.ax = ax
        self.wd_values = wd_values
        self.offset = offset
        self.plot_relative = plot_relative
        self.setup_axis_callbacks()
    
    def setup_axis_callbacks(self):
        """Connect callbacks to axis limit changes for adaptive ticking."""
        self.ax.callbacks.connect('xlim_changed', self.on_xlims_change)
        self.ax.callbacks.connect('ylim_changed', self.on_ylims_change)
    
    def on_xlims_change(self, ax):
        """Handle X-axis (time) limit changes for adaptive time ticks."""
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        
        # Determine appropriate tick intervals based on visible range
        if x_range <= 1:           # Less than 1 day
            major_interval = 0.2   # Every 4.8 hours
            minor_interval = 0.1   # Every 2.4 hours
            precision = 1
        elif x_range <= 3:         # Less than 3 days  
            major_interval = 0.5   # Every 12 hours
            minor_interval = 0.25  # Every 6 hours
            precision = 1
        elif x_range <= 7:         # Less than 1 week
            major_interval = 1     # Daily
            minor_interval = 0.5   # Every 12 hours
            precision = 1
        elif x_range <= 30:        # Less than 30 days
            major_interval = 5     # Every 5 days
            minor_interval = 1     # Daily
            precision = 0
        elif x_range <= 90:        # Less than 3 months
            major_interval = 10    # Every 10 days
            minor_interval = 5     # Every 5 days
            precision = 0
        elif x_range <= 180:       # Less than 6 months
            major_interval = 20    # Every 20 days
            minor_interval = 5     # Every 5 days
            precision = 0
        else:                      # Full year or more
            major_interval = 50    # Every 50 days
            minor_interval = 10    # Every 10 days
            precision = 0
        
        # Create tick arrays
        major_start = np.ceil(x_min / major_interval) * major_interval
        major_ticks = np.arange(major_start, x_max + major_interval, major_interval)
        
        minor_start = np.ceil(x_min / minor_interval) * minor_interval
        minor_ticks = np.arange(minor_start, x_max + minor_interval, minor_interval)
        
        # Remove overlapping ticks
        minor_ticks = minor_ticks[~np.isin(minor_ticks, major_ticks)]
        
        # Apply ticks to plot
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        
        # Format labels with appropriate precision
        major_labels = []
        for tick in major_ticks:
            if precision == 0:
                if self.plot_relative:
                    major_labels.append(f'{int(tick)}')
                else:
                    if tick == 0:
                        major_labels.append('0\n(Oct 1)')
                    elif tick == 365 or tick == 366:
                        major_labels.append(f'{int(tick)}\n(Sep 30)')
                    else:
                        major_labels.append(f'{int(tick)}')
            else:
                major_labels.append(f'{tick:.{precision}f}')
        
        ax.set_xticklabels(major_labels)
        
        # Style the ticks
        ax.tick_params(axis='x', which='major', labelsize=10, length=8, width=1.5)
        ax.tick_params(axis='x', which='minor', length=4, width=1)
    
    def on_ylims_change(self, ax):
        """Handle Y-axis (temperature) limit changes for adaptive temperature ticks."""
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Determine appropriate tick intervals for temperature
        if y_range <= 1:           # Less than 1°C range
            major_interval = 0.2   # Every 0.2°C
            minor_interval = 0.1   # Every 0.1°C
            precision = 1
        elif y_range <= 2:         # Less than 2°C range
            major_interval = 0.5   # Every 0.5°C
            minor_interval = 0.1   # Every 0.1°C
            precision = 1
        elif y_range <= 5:         # Less than 5°C range
            major_interval = 1     # Every 1°C
            minor_interval = 0.5   # Every 0.5°C
            precision = 1
        elif y_range <= 20:        # Less than 20°C range
            major_interval = 2     # Every 2°C (even degrees as requested)
            minor_interval = 1     # Every 1°C
            precision = 0
        else:                      # Large range
            major_interval = 5     # Every 5°C
            minor_interval = 1     # Every 1°C
            precision = 0
        
        # Create tick arrays
        major_start = np.ceil(y_min / major_interval) * major_interval
        major_ticks = np.arange(major_start, y_max + major_interval, major_interval)
        
        minor_start = np.ceil(y_min / minor_interval) * minor_interval
        minor_ticks = np.arange(minor_start, y_max + minor_interval, minor_interval)
        
        # Remove overlapping ticks
        minor_ticks = minor_ticks[~np.isin(minor_ticks, major_ticks)]
        
        # Apply ticks to plot
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        
        # Format temperature labels
        if precision == 0:
            major_labels = [f'{int(tick)}°C' for tick in major_ticks]
        else:
            major_labels = [f'{tick:.{precision}f}°C' for tick in major_ticks]
        
        ax.set_yticklabels(major_labels)
        
        # Style the ticks
        ax.tick_params(axis='y', which='major', labelsize=10, length=8, width=1.5)
        ax.tick_params(axis='y', which='minor', length=4, width=1)


# Note: Removed EnhancedTextBox class due to matplotlib limitations
# Tab navigation and advanced text selection are not reliably supported
# in matplotlib's widget system. Using standard TextBox instead.


def is_leap_year(year):
    """
    Check if a given year is a leap year.
    
    Args:
        year (int): Year to check
        
    Returns:
        bool: True if leap year, False otherwise
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_water_year_length(water_year):
    """
    Get the length of a water year in days.
    
    FIXED: Now checks the calendar year that contains February 29th.
    For water year 2024 (Oct 1, 2023 - Sep 30, 2024), we need to check
    if calendar year 2024 is a leap year.
    
    Args:
        water_year (int): The water year
        
    Returns:
        int: 366 if leap year, 365 if not
    """
    # For a water year, February 29th falls in the same calendar year
    # as the water year end date (September 30th)
    calendar_year_with_feb = water_year
    return 366 if is_leap_year(calendar_year_with_feb) else 365


def create_log_filename(shallow_fn, deep_fn, out_dir):
    """
    Create a timestamped log filename based on input files.
    
    Args:
        shallow_fn (str): Shallow data filename
        deep_fn (str): Deep data filename  
        out_dir (str): Output directory
        
    Returns:
        str: Full path to log file
    """
    base_name = make_base_name(shallow_fn, deep_fn)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{base_name}_{timestamp}.log"
    return os.path.join(out_dir, log_filename)


def parse_par(path):
    """
    Parse a flat parameter file with key=value pairs.
    
    The parameter file format supports:
    - Comments starting with # or ;
    - Key=value pairs
    - Inline comments after values
    
    Args:
        path (str): Path to parameter file
        
    Returns:
        dict: Dictionary of parameter key-value pairs
    """
    params = {}
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#') or line.startswith(';'):
                    continue
                    
                if '=' in line:
                    key, value = line.split('=', 1)
                    
                    # Clean up key (remove non-alphanumeric, convert to lowercase)
                    key = re.sub(r'\W+', '_', key.strip().lower())
                    
                    # Remove inline comments from value
                    value = value.split(';', 1)[0].split('#', 1)[0].strip()
                    
                    params[key] = value
                    
    except FileNotFoundError:
        print(f"Warning: Parameter file {path} not found")
        
    return params


def load_parameters():
    """
    Load and validate parameters from tts_merpar.par file.
    
    Searches for the parameter file in current directory and script directory.
    
    Returns:
        dict: Validated parameters
        
    Raises:
        SystemExit: If parameter file not found or required parameters missing
    """
    # Search for parameter file in current directory and script directory
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    par_file = None
    
    for directory in search_dirs:
        potential_path = os.path.join(directory, 'tts_merpar.par')
        if os.path.isfile(potential_path):
            par_file = potential_path
            break

    if not par_file:
        sys.exit('Error: tts_merpar.par not found.')

    print(f"Loading parameters from: {par_file}")
    params = parse_par(par_file)

    # Validate required parameters
    required_keys = ['filename_shallow', 'filename_deep', 'water_year', 'data_interval_min']
    for key in required_keys:
        if key not in params:
            sys.exit(f"Error: '{key}' missing in {par_file}")

    # Parse and validate parameter values
    try:
        validated_params = {
            'fn_shallow': params['filename_shallow'],
            'fn_deep': params['filename_deep'],
            'water_year': int(params['water_year']),
            'interval_min': float(params['data_interval_min']),
            'convert_f_to_c': bool(int(params.get('convert_f_to_c', '0'))),
            'gap_factor': float(params.get('gap_threshold_factor', '1.5')),
            'output_folder': params.get('output_folder', 'processed'),
            'plot_relative': bool(int(params.get('plot_relative', '0')))
        }
    except ValueError as e:
        sys.exit(f"Error parsing parameters: {e}")
    
    return validated_params


def read_logger_data(filename, label, search_dirs, convert_fahrenheit=False, 
                    interval_min=15, gap_factor=1.5):
    """
    Read and parse temperature logger data from CSV file.
    
    This function handles multiple datetime formats and performs data validation.
    It also detects gaps in the data based on expected sampling interval.
    
    Args:
        filename (str): Name of CSV file to read
        label (str): Description for logging (e.g., 'Shallow', 'Deep')
        search_dirs (list): Directories to search for the file
        convert_fahrenheit (bool): Whether to convert Fahrenheit to Celsius
        interval_min (float): Expected data interval in minutes
        gap_factor (float): Factor to determine gap threshold
        
    Returns:
        pandas.DataFrame: DataFrame with DateTime and Temp columns
        
    Raises:
        SystemExit: If file not found or no valid data found
    """
    # Find the file in search directories
    file_path = None
    for directory in search_dirs:
        potential_path = os.path.join(directory, filename)
        if os.path.isfile(potential_path):
            file_path = potential_path
            break
    
    if not file_path:
        sys.exit(f"Error: {label} file '{filename}' not found in {search_dirs}")
    
    print(f"Reading {label} data from: {file_path}")
    
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
    
    print(f"  Debug: Showing first 10 date strings to verify format detection...")
    
    # Read and parse CSV file
    with open(file_path, newline='', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        
        for line_num, line in enumerate(csv_reader, 1):
            line_count += 1
            
            # Skip header lines or lines with insufficient columns
            if len(line) < 3:
                if line_count <= 3:  # Allow for headers in first few lines
                    continue
                error_count += 1
                continue
            
            # Extract date and temperature strings
            date_str = line[1].strip()
            temp_str = line[2].strip()
            
            # Debug: Show first few raw date strings (ENHANCED: Better explanation)
            if line_count <= 10 and date_str and not date_str.startswith('Date'):
                print(f"  Raw date string (line {line_num}): '{date_str}' -> Checking format compatibility...")
            
            # Try to parse datetime with multiple formats
            parsed_datetime = None
            for date_format in date_formats:
                try:
                    parsed_datetime = datetime.strptime(date_str, date_format)
                    if line_count <= 10 and date_str and not date_str.startswith('Date'):
                        print(f"    ✓ Successfully parsed using format: {date_format}")
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
        sys.exit(f"Error: no valid data found in {filename}")
    
    print(f"  ✓ Successfully loaded {len(rows)} valid records")
    print(f"  ✓ All displayed date strings were successfully parsed and included")
    if error_count > 0:
        print(f"  ! Skipped {error_count} invalid records from {line_count} total lines")
    
    # Create DataFrame and sort by datetime
    df = pd.DataFrame(rows, columns=['DateTime', 'Temp']).sort_values('DateTime')
    df = df.reset_index(drop=True)
    
    # Detect gaps in the data
    _detect_data_gaps(df, interval_min, gap_factor)
    
    return df


def _detect_data_gaps(df, interval_min, gap_factor):
    """
    Detect and report gaps in time series data.
    
    Args:
        df (pandas.DataFrame): DataFrame with DateTime column
        interval_min (float): Expected interval in minutes
        gap_factor (float): Factor to determine gap threshold
    """
    time_diffs = df['DateTime'].diff().dt.total_seconds() / 60.0
    gap_threshold = interval_min * gap_factor
    gaps = time_diffs > gap_threshold
    
    gap_count = gaps.sum()
    if gap_count > 0:
        print(f"  Found {gap_count} gaps > {gap_threshold:.1f} minutes:")
        for idx in df.index[gaps]:
            gap_size = time_diffs.iloc[idx]
            print(f"    Gap at {df.iloc[idx]['DateTime']}: {gap_size:.1f} minutes")


def calculate_water_day_columns(df, water_year):
    """
    Add water day and water year columns to DataFrame.
    
    Water year runs from October 1 to September 30.
    Water day 0 = October 1 of the previous calendar year.
    
    FIXED: Now properly handles leap years by checking the calendar year
    that contains February 29th.
    
    Args:
        df (pandas.DataFrame): DataFrame with DateTime column
        water_year (int): The water year to calculate for
        
    Returns:
        pandas.DataFrame: DataFrame with added WaterDay and WaterYear columns
    """
    water_year_start = datetime(water_year - 1, 10, 1)
    
    def calc_water_day(dt):
        """Calculate water day from datetime, accounting for leap years."""
        delta = dt - water_year_start
        return delta.total_seconds() / 86400.0
    
    def get_water_year_for_date(dt):
        """Get the correct water year for a given date."""
        if dt.month >= 10:  # October, November, December
            return dt.year + 1
        else:  # January through September
            return dt.year
    
    # Add calculated columns
    df['WaterDay'] = df['DateTime'].apply(calc_water_day)
    df['WaterYear'] = df['DateTime'].apply(get_water_year_for_date)
    
    return df


def validate_water_year_data(df, label, water_year):
    """
    Validate data falls within the specified water year and report issues.
    
    FIXED: Now uses dynamic water year length for validation.
    
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
    _report_anomalous_data(merged)
    
    return merged


def _report_anomalous_data(merged_df):
    """
    Report pre-season and post-season data in merged dataset.
    
    FIXED: Now uses dynamic water year length for post-season detection.
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


def make_base_name(shallow_fn, deep_fn):
    """
    Create a composite base name from input filenames.
    
    Attempts to find common patterns or prefixes in the filenames.
    
    Args:
        shallow_fn (str): Shallow data filename
        deep_fn (str): Deep data filename
        
    Returns:
        str: Base name for output files
    """
    sh_base = os.path.splitext(os.path.basename(shallow_fn))[0]
    dp_base = os.path.splitext(os.path.basename(deep_fn))[0]
    
    # Handle common naming patterns like filename-S and filename-D
    if sh_base.lower().endswith('-s') and dp_base.lower().endswith('-d'):
        return sh_base[:-2] + '-SD'
    
    # Find common prefix
    common_prefix = os.path.commonprefix([sh_base, dp_base]).rstrip('_-')
    if common_prefix:
        return common_prefix + '_SD'
    else:
        return f"{sh_base}_{dp_base}"


def setup_interactive_plot(merged_data, base_name, water_year, plot_relative=False):
    """
    Create enhanced interactive matplotlib plot for data visualization and interval selection.
    
    Args:
        merged_data (pandas.DataFrame): Merged temperature data
        base_name (str): Base name for output files
        water_year (int): Water year for labeling
        plot_relative (bool): Whether to plot relative to data start
        
    Returns:
        tuple: (figure, axis, selector, text_boxes, buttons, selection_state, axis_manager)
    """
    print('\n=== Starting Interactive Selection ===')
    
    # Prepare plot data
    wd_values = merged_data['WaterDay'].values
    offset = wd_values.min() if plot_relative else 0.0
    wd_plot = wd_values - offset
    
    # Create figure with space for controls
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.25)
    
    # Plot temperature data
    ax.plot(wd_plot, merged_data['T_shallow'], color='tab:blue', alpha=0.7, 
            linewidth=1, label='Shallow')
    ax.plot(wd_plot, merged_data['T_deep'], color='tab:red', alpha=0.7, 
            linewidth=1, label='Deep')
    
    # Add water year boundary lines
    _add_water_year_boundaries(ax, water_year, plot_relative)
    
    # Configure plot appearance (without setting ticks - adaptive manager will handle)
    _configure_plot_appearance(ax, wd_values, offset, base_name, water_year, plot_relative)
    
    # Create adaptive axis manager for zoom-responsive ticks
    axis_manager = AdaptiveAxisManager(ax, wd_values, offset, plot_relative)
    
    # Create interactive elements
    selector = _create_span_selector(ax, offset)
    text_boxes, buttons = _create_control_widgets(base_name)
    
    # Selection state for tracking user interactions
    selection_state = {
        'start': None, 
        'end': None, 
        'filename': f"{base_name}_interval",
        'selection_rectangles': [],
        'current_rect': None
    }
    
    # Initial axis setup to trigger adaptive ticks
    axis_manager.on_xlims_change(ax)
    axis_manager.on_ylims_change(ax)
    
    return fig, ax, selector, text_boxes, buttons, selection_state, axis_manager


def _add_water_year_boundaries(ax, water_year, plot_relative):
    """
    Add vertical lines marking water year boundaries.
    
    FIXED: Now shows correct end boundary for leap years.
    """
    wy_start_date = datetime(water_year - 1, 10, 1)  # Oct 1 of previous year
    wy_end_date = datetime(water_year, 9, 30)        # Sep 30 of water year
    water_year_length = get_water_year_length(water_year)
    
    ax.axvline(0, color='green', linestyle='--', alpha=0.8, linewidth=2, 
              label=f'WY Start ({wy_start_date.strftime("%b %d, %Y")})')
    ax.axvline(water_year_length, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
              label=f'WY End ({wy_end_date.strftime("%b %d, %Y")}) - Day {water_year_length}')
    
    print(f"Added water year boundary lines (Length: {water_year_length} days)")


def _configure_plot_appearance(ax, wd_values, offset, base_name, water_year, plot_relative):
    """Configure the overall appearance of the plot."""
    # Set axis limits
    if plot_relative:
        ax.set_xlim(0, wd_values.max() - offset)
        ax.set_xlabel('Water Day (relative to start)', fontsize=12)
    else:
        ax.set_xlim(wd_values.min(), wd_values.max())
        ax.set_xlabel('Water Day', fontsize=12)

    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Temperature Data - {base_name} (Water Year {water_year})', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.3)


def _create_span_selector(ax, offset):
    """Create and configure the span selector for interactive selection."""
    def on_select(x_start, x_end):
        """Handle span selection callback."""
        # This will be connected to the main selection handler
        pass
    
    selector = SpanSelector(
        ax, on_select, 'horizontal',
        useblit=True,
        props=dict(alpha=0.3, facecolor='gray')
    )
    
    return selector


def _create_control_widgets(base_name):
    """Create text boxes and buttons for user interaction."""
    # Create axes for widgets
    ax_start = plt.axes([0.1, 0.15, 0.15, 0.04])      # Start WD box
    ax_end = plt.axes([0.1, 0.10, 0.15, 0.04])        # End WD box  
    ax_filename = plt.axes([0.25, 0.02, 0.4, 0.04])   # Filename box
    ax_save = plt.axes([0.67, 0.13, 0.08, 0.04])      # Save button
    ax_clear = plt.axes([0.76, 0.13, 0.08, 0.04])     # Clear button
    ax_done = plt.axes([0.85, 0.13, 0.08, 0.04])      # Done button
    
    # Create standard widgets (matplotlib TextBox has inherent limitations)
    tb_start = TextBox(ax_start, 'Start WD: ')
    tb_end = TextBox(ax_end, 'End WD: ')
    tb_filename = TextBox(ax_filename, 'Filename: ')
    btn_save = Button(ax_save, 'Save')
    btn_clear = Button(ax_clear, 'Clear')
    btn_done = Button(ax_done, 'Done')
    
    # Set default filename
    default_filename = f"{base_name}_interval"
    tb_filename.set_val(default_filename)
    
    text_boxes = {
        'start': tb_start,
        'end': tb_end,
        'filename': tb_filename
    }
    
    buttons = {
        'save': btn_save,
        'clear': btn_clear,
        'done': btn_done
    }
    
    return text_boxes, buttons


def write_composite_csv(df_subset, filename):
    """
    Write composite CSV file with formatted temperature data.
    
    Args:
        df_subset (pandas.DataFrame): Data subset to write
        filename (str): Output filename
    """
    df_output = pd.DataFrame({
        'WaterDay': df_subset['WaterDay'].map(lambda x: f"{x:.6f}"),
        'Shallow.Temp': df_subset['T_shallow'].map(lambda x: f"{x:.5f}"),
        'Deep.Temp': df_subset['T_deep'].map(lambda x: f"{x:.5f}")
    })
    
    df_output.to_csv(filename, index=False)
    print(f"Composite CSV saved: {filename}")


def write_wyo_file(df_subset, temp_column, filename, water_year):
    """
    Write WYO format file with proper water year handling.
    
    FIXED: Now properly handles leap year transitions.
    
    Args:
        df_subset (pandas.DataFrame): Data subset to write
        temp_column (str): Column name for temperature data
        filename (str): Output filename
        water_year (int): Water year for calculations
    """
    water_year_length = get_water_year_length(water_year)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Year\tWaterDay\tTemperature\tDepthID\n")
        
        for _, row in df_subset.iterrows():
            # Determine year based on water day and leap year status
            if row['WaterDay'] < water_year_length:
                year = water_year - 1  # Previous calendar year
            else:
                year = water_year      # Current water year
            
            f.write(f"{year}\t{row['WaterDay']:.5f}\t{row[temp_column]:.5f}\t1\n")
    
    print(f"WYO file saved: {filename} (Water Year Length: {water_year_length} days)")


def save_interval(event, merged_data, selection_state, text_boxes, out_dir, base_name, water_year):
    """
    Save the selected interval to CSV and WYO files with user-controlled filenames.
    
    ENHANCED: No automatic suffix addition - user controls filename completely.
    
    Args:
        event: Button click event
        merged_data (pandas.DataFrame): Full merged dataset
        selection_state (dict): Current selection state
        text_boxes (dict): Text input widgets
        out_dir (str): Output directory
        base_name (str): Base filename (for WYO files only)
        water_year (int): Water year for WYO files
    """
    try:
        # Get values from text boxes
        start_wd = float(text_boxes['start'].text)
        end_wd = float(text_boxes['end'].text)
        filename_base = text_boxes['filename'].text.strip()
        
        # Validate inputs
        if start_wd >= end_wd:
            print("Error: Start WaterDay must be less than End WaterDay")
            return
            
        # Filter data to selected interval
        mask = (merged_data['WaterDay'] >= start_wd) & (merged_data['WaterDay'] <= end_wd)
        df_subset = merged_data[mask].copy()
        
        if len(df_subset) == 0:
            print(f"Warning: No data found in interval {start_wd:.5f} to {end_wd:.5f}")
            return
        
        # ENHANCED: Use exact filename from user - no automatic suffix
        # Ensure .csv extension
        if not filename_base.endswith('.csv'):
            filename_base += '.csv'
        
        csv_filename = os.path.join(out_dir, filename_base)
        
        # For WYO files, derive names from CSV filename
        csv_base = os.path.splitext(filename_base)[0]  # Remove .csv
        shallow_wyo = os.path.join(out_dir, f"{csv_base}-S.wyo")
        deep_wyo = os.path.join(out_dir, f"{csv_base}-D.wyo")
        
        # Write files
        write_composite_csv(df_subset, csv_filename)
        write_wyo_file(df_subset, 'T_shallow', shallow_wyo, water_year)
        write_wyo_file(df_subset, 'T_deep', deep_wyo, water_year)
        
        print(f"✓ Saved interval: WD {start_wd:.5f} to {end_wd:.5f}")
        print(f"  Records: {len(df_subset)}")
        print(f"  Files: {os.path.basename(csv_filename)}, {os.path.basename(shallow_wyo)}, {os.path.basename(deep_wyo)}")
        
        # Move current selection to saved selections and change color
        if 'current_rect' in selection_state and selection_state['current_rect']:
            rect = selection_state['current_rect']
            rect.set_facecolor('lightgreen')
            rect.set_alpha(0.15)
            rect.set_edgecolor('green')
            selection_state['selection_rectangles'].append(rect)
            selection_state['current_rect'] = None
            plt.gcf().canvas.draw()
        
    except ValueError as e:
        print(f"Error: Invalid input values - {e}")
    except Exception as e:
        print(f"Error saving interval: {e}")


def finish_processing(event, merged_data, out_dir, base_name, water_year, fig, logger):
    """
    Create full-record WYO files and finish processing.
    
    Args:
        event: Button click event
        merged_data (pandas.DataFrame): Full merged dataset
        out_dir (str): Output directory
        base_name (str): Base filename
        water_year (int): Water year for files
        fig: Matplotlib figure to close
        logger: TeeLogger instance to close
    """
    print('\n=== Creating Full Record Files ===')
    
    try:
        # Create full composite CSV file
        full_csv = os.path.join(out_dir, f"{base_name}_full.csv")
        write_composite_csv(merged_data, full_csv)
        
        # Create full WYO files
        shallow_full = os.path.join(out_dir, f"{base_name}-S_full.wyo")
        deep_full = os.path.join(out_dir, f"{base_name}-D_full.wyo")
        
        write_wyo_file(merged_data, 'T_shallow', shallow_full, water_year)
        write_wyo_file(merged_data, 'T_deep', deep_full, water_year)
        
        print('Processing complete!')
        print(f"All files saved to: {out_dir}")
        
    except Exception as e:
        print(f"Error creating full record files: {e}")
    
    finally:
        # Clean up
        plt.close(fig)
        sys.stdout = logger.terminal  # Restore original stdout
        logger.close()
        print(f"Log saved to: {logger.log_file.name}")


def setup_event_handlers(selector, text_boxes, buttons, merged_data, selection_state, 
                        out_dir, base_name, water_year, fig, logger, ax):
    """
    Connect all event handlers for interactive plot with enhanced text box integration.
    
    Args:
        selector: SpanSelector widget
        text_boxes (dict): Text input widgets
        buttons (dict): Button widgets
        merged_data (pandas.DataFrame): Full dataset
        selection_state (dict): Selection state tracking
        out_dir (str): Output directory
        base_name (str): Base filename
        water_year (int): Water year
        fig: Matplotlib figure
        logger: TeeLogger instance
        ax: Plot axis
    """
    def update_selection_rectangle(start_wd, end_wd):
        """Update or create selection rectangle based on text box values."""
        # Remove previous current rectangle if it exists
        if 'current_rect' in selection_state and selection_state['current_rect']:
            selection_state['current_rect'].remove()
        
        # Create new rectangle
        y_min, y_max = ax.get_ylim()
        rect = Rectangle((start_wd, y_min), end_wd - start_wd, y_max - y_min, 
                        alpha=0.2, facecolor='lightblue', edgecolor='blue', 
                        linewidth=2, linestyle='--')
        ax.add_patch(rect)
        selection_state['current_rect'] = rect
        fig.canvas.draw()
    
    def on_span_select(x_start, x_end):
        """Handle span selection from plot."""
        selection_state['start'] = x_start
        selection_state['end'] = x_end
    
        # Update text boxes
        text_boxes['start'].set_val(f"{x_start:.5f}")
        text_boxes['end'].set_val(f"{x_end:.5f}")
    
        # Update selection rectangle
        update_selection_rectangle(x_start, x_end)
        
        print(f"Selected interval: WD {x_start:.5f} to {x_end:.5f}")
    
    def on_textbox_change(text_widget):
        """Handle text box changes to update selection rectangle."""
        try:
            start_text = text_boxes['start'].text.strip()
            end_text = text_boxes['end'].text.strip()
            
            if start_text and end_text:
                start_wd = float(start_text)
                end_wd = float(end_text)
                
                if start_wd < end_wd:
                    selection_state['start'] = start_wd
                    selection_state['end'] = end_wd
                    update_selection_rectangle(start_wd, end_wd)
                    print(f"Updated selection from text boxes: WD {start_wd:.5f} to {end_wd:.5f}")
        except ValueError:
            # Invalid values in text boxes - ignore
            pass
    
    def on_save_click(event):
        """Handle save button click."""
        save_interval(event, merged_data, selection_state, text_boxes, 
                     out_dir, base_name, water_year)
    
    def on_clear_click(event):
        """Handle clear button click - removes all selection rectangles."""
        # Get all Rectangle patches from the axes and remove them
        patches_to_remove = []
        for patch in ax.patches:
            if isinstance(patch, Rectangle):
                patches_to_remove.append(patch)
    
        for patch in patches_to_remove:
            patch.remove()
    
        # Clear the selection state
        selection_state['selection_rectangles'] = []
        selection_state['current_rect'] = None
    
        # Clear text boxes
        text_boxes['start'].set_val('')
        text_boxes['end'].set_val('')
    
        fig.canvas.draw()
        print(f"Cleared {len(patches_to_remove)} selection rectangles")

    def on_done_click(event):
        """Handle done button click."""
        finish_processing(event, merged_data, out_dir, base_name, 
                         water_year, fig, logger)
    
    # Connect event handlers
    selector.onselect = on_span_select
    buttons['save'].on_clicked(on_save_click)
    buttons['clear'].on_clicked(on_clear_click)  
    buttons['done'].on_clicked(on_done_click)
    
    # Connect text box change events (using standard TextBox functionality)
    text_boxes['start'].on_text_change(lambda text: on_textbox_change(text_boxes['start']))
    text_boxes['end'].on_text_change(lambda text: on_textbox_change(text_boxes['end']))


def main():
    """
    Main function to orchestrate the entire processing workflow.
    """
    print("=== TTS_MERPAR: Temperature Time Series Merger and Processor v2.0 ===")
    print("Enhanced with adaptive zoom, editable selections, and improved stability\n")
    
    try:
        # Load and validate parameters
        params = load_parameters()
        
        # Create output directory relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, params['output_folder'])
        os.makedirs(out_dir, exist_ok=True)
        
        # Set up logging
        log_filename = create_log_filename(params['fn_shallow'], params['fn_deep'], out_dir)
        logger = TeeLogger(log_filename)
        sys.stdout = logger  # Redirect stdout to logger
        
        # Print processing parameters
        print(f"Shallow file: {params['fn_shallow']}")
        print(f"Deep file: {params['fn_deep']}")
        print(f"Water year: {params['water_year']}")
        print(f"Data interval: {params['interval_min']} minutes")
        print(f"Convert F to C: {params['convert_f_to_c']}")
        print(f"Gap threshold factor: {params['gap_factor']}")
        print(f"Plot relative: {params['plot_relative']}")
        
        # Define search directories for data files
        search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
        
        # Read temperature data files
        print('\n=== Reading Data Files ===')
        df_shallow = read_logger_data(
            params['fn_shallow'], 'Shallow', search_dirs,
            params['convert_f_to_c'], params['interval_min'], params['gap_factor']
        )
        
        df_deep = read_logger_data(
            params['fn_deep'], 'Deep', search_dirs,
            params['convert_f_to_c'], params['interval_min'], params['gap_factor']
        )
        
        # Calculate water day columns
        print('\n=== Processing Water Years ===')
        df_shallow = calculate_water_day_columns(df_shallow, params['water_year'])
        df_deep = calculate_water_day_columns(df_deep, params['water_year'])
        
        # Validate water year data
        validate_water_year_data(df_shallow, 'Shallow', params['water_year'])
        validate_water_year_data(df_deep, 'Deep', params['water_year'])
        
        # Merge datasets
        merged_data = merge_temperature_data(df_shallow, df_deep)
        
        # Create base name for output files
        base_name = make_base_name(params['fn_shallow'], params['fn_deep'])
        
        # Set up enhanced interactive plot
        fig, ax, selector, text_boxes, buttons, selection_state, axis_manager = setup_interactive_plot(
            merged_data, base_name, params['water_year'], params['plot_relative']
        )
        
        # Connect enhanced event handlers
        setup_event_handlers(
            selector, text_boxes, buttons, merged_data, selection_state,
            out_dir, base_name, params['water_year'], fig, logger, ax
        )
        
        # Display enhanced instructions and show plot
        print("\n=== Enhanced Interactive Selection Instructions ===")
        print("1. Click and drag on the plot to select a data interval")
        print("2. OR manually edit Start WD/End WD values to adjust selection bounds")
        print("3. Modify filename as desired (no automatic suffixes added)")
        print("4. Click 'Save' to export the selected interval")
        print("5. Repeat for additional intervals")
        print("6. Click 'Done' when finished to create full-record files")
        print("7. Zoom in/out to see adaptive tick formatting on both axes")
        print("\nEnhanced Features:")
        print("• Adaptive ticks adjust automatically when zooming")
        print("• Edit selection bounds via text boxes")
        print("• Temperature axis shows even degree increments")
        print("• User controls exact filenames (no automatic suffixes)")
        print("• Improved date format detection and validation")
        print("\nNote: Text editing uses standard matplotlib functionality.")
        print("Click in text boxes to edit, use mouse to position cursor.")
        print(f"\nNote: All WYO files will use Water Year {params['water_year']} in the Year column.")
        
        # Warning messages for anomalous data
        if len(merged_data[merged_data['WaterDay'] < 0]) > 0:
            print("Warning: Negative WaterDay values indicate pre-season data.")
        if len(merged_data[merged_data['WaterDay'] > 365]) > 0:
            print("Warning: WaterDay > 365 indicates data from the next water year.")
        
        # Show the enhanced interactive plot
        plt.show()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


# Script entry point
if __name__ == "__main__":
    main()
