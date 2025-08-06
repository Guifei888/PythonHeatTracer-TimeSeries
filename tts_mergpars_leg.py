#!/usr/bin/env python3
"""
Filename: tts_mergpars.py
Title: Temperature Time-Series Merger and Processor (Legacy Version)

Description:
    Merges temperature logger data (shallow + deep) into a unified dataset.
    Converts time to water year format and supports interval selection and export.
    Primarily used for command-line workflows and pre-Dash automation.

Features:
    - Reads CSV temperature data
    - Water year conversion (Oct 1 - Sep 30)
    - Gap detection and interval validation
    - Interactive plot for selecting intervals
    - Output in CSV and WYO formats
    - Parameter file input: tts_mergpars.par
    - Custom TeeLogger for console + file output

Author: Timothy Wu  
Created: 2025-06-26  
Last Modified: 2025-07-16  
Version: 1.0

Requirements:
    - pandas, numpy, matplotlib
    - CSV input files + parameter file (`tts_mergpars.par`)

Usage:
    python tts_mergpars.py
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
        self.log_file.write(f"=== TTS_MERGPARS Processing Log ===\n")
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
    Load and validate parameters from tts_mergpars.par file.
    
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
        potential_path = os.path.join(directory, 'tts_mergpars.par')
        if os.path.isfile(potential_path):
            par_file = potential_path
            break

    if not par_file:
        sys.exit('Error: tts_mergpars.par not found.')

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
        sys.exit(f"Error: no valid data found in {filename}")
    
    print(f"  Loaded {len(rows)} valid records, {error_count} errors from {line_count} lines")
    
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
    Create interactive matplotlib plot for data visualization and interval selection.
    
    Args:
        merged_data (pandas.DataFrame): Merged temperature data
        base_name (str): Base name for output files
        water_year (int): Water year for labeling
        plot_relative (bool): Whether to plot relative to data start
        
    Returns:
        tuple: (figure, axis, selector, text_boxes, buttons, selection_state)
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
    
    # Set up enhanced tick marks
    _setup_enhanced_ticks(ax, wd_values, offset, plot_relative)
    
    # Configure plot appearance
    _configure_plot_appearance(ax, wd_values, offset, base_name, water_year, plot_relative)
    
    # Create interactive elements
    selector = _create_span_selector(ax, offset)
    text_boxes, buttons = _create_control_widgets(base_name)
    
    # Selection state for tracking user interactions
    selection_state = {
        'start': None, 
        'end': None, 
        'filename': f"{base_name}",
        'interval_count': 0,
        'selection_rectangles': []  
    }
    
    return fig, ax, selector, text_boxes, buttons, selection_state


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


def _setup_enhanced_ticks(ax, wd_values, offset, plot_relative):
    """Set up enhanced tick marks based on data range."""
    if plot_relative:
        data_min, data_max = 0, wd_values.max() - offset
    else:
        data_min, data_max = wd_values.min(), wd_values.max()
    
    data_range = data_max - data_min
    
    # Determine tick intervals based on data range
    if data_range <= 30:      # Less than 30 days
        major_interval, minor_interval = 5, 1
    elif data_range <= 90:    # Less than 3 months
        major_interval, minor_interval = 10, 1
    elif data_range <= 180:   # Less than 6 months
        major_interval, minor_interval = 20, 5
    else:                     # Full year or more
        major_interval, minor_interval = 50, 10
    
    # Create tick arrays
    major_start = np.ceil(data_min / major_interval) * major_interval
    major_ticks = np.arange(major_start, data_max + major_interval, major_interval)
    
    minor_start = np.ceil(data_min / minor_interval) * minor_interval
    minor_ticks = np.arange(minor_start, data_max + minor_interval, minor_interval)
    
    # Remove overlapping ticks
    minor_ticks = minor_ticks[~np.isin(minor_ticks, major_ticks)]
    
    # Apply ticks to plot
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    
    # Format major tick labels
    if plot_relative:
        major_labels = [f'{int(tick)}' for tick in major_ticks]
    else:
        major_labels = []
        for tick in major_ticks:
            if tick == 0:
                major_labels.append('0\n(Oct 1)')
            elif tick == 365:
                major_labels.append('365\n(Sep 30)')
            else:
                major_labels.append(f'{int(tick)}')
    
    ax.set_xticklabels(major_labels)
    
    # Style the ticks
    ax.tick_params(axis='x', which='major', labelsize=10, length=8, width=1.5)
    ax.tick_params(axis='x', which='minor', length=4, width=1)
    
    print(f"Set up ticks: Major every {major_interval} days, Minor every {minor_interval} day(s)")


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
    
    # Create widgets
    tb_start = TextBox(ax_start, 'Start WD: ')
    tb_end = TextBox(ax_end, 'End WD: ')
    tb_filename = TextBox(ax_filename, 'Filename: ')
    btn_save = Button(ax_save, 'Save')
    btn_clear = Button(ax_clear, 'Clear')
    btn_done = Button(ax_done, 'Done')
    
    # Set default filename
    default_filename = f"{base_name}"
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
    Save the selected interval to CSV and WYO files.
    If no start/end water day is specified, saves the full dataset.
    
    Args:
        event: Button click event
        merged_data (pandas.DataFrame): Full merged dataset
        selection_state (dict): Current selection state
        text_boxes (dict): Text input widgets
        out_dir (str): Output directory
        base_name (str): Base filename
        water_year (int): Water year for WYO files
    """
    try:
        # Get filename from text box
        filename_base = text_boxes['filename'].text.strip()
        if not filename_base:
            print("Error: Please specify a filename")
            return
        
        # Get start/end values from text boxes
        start_text = text_boxes['start'].text.strip()
        end_text = text_boxes['end'].text.strip()
        
        # Check if both start and end are specified
        if start_text and end_text:
            # Parse water day values for interval selection
            try:
                start_wd = float(start_text)
                end_wd = float(end_text)
            except ValueError:
                print("Error: Start and End WaterDay must be valid numbers")
                return
            
            # Validate interval
            if start_wd >= end_wd:
                print("Error: Start WaterDay must be less than End WaterDay")
                return
                
            # Filter data to selected interval
            mask = (merged_data['WaterDay'] >= start_wd) & (merged_data['WaterDay'] <= end_wd)
            df_subset = merged_data[mask].copy()
            
            if len(df_subset) == 0:
                print(f"Warning: No data found in interval {start_wd:.5f} to {end_wd:.5f}")
                return
                
            print(f"Saving interval: WD {start_wd:.5f} to {end_wd:.5f}")
            
        else:
            # No start/end specified - save full dataset
            df_subset = merged_data.copy()
            start_wd = merged_data['WaterDay'].min()
            end_wd = merged_data['WaterDay'].max()
            print(f"No interval specified - saving full dataset: WD {start_wd:.5f} to {end_wd:.5f}")
        
        # Create output filenames (user controls exact names)
        csv_filename = os.path.join(out_dir, f"{filename_base}.csv")
        shallow_wyo = os.path.join(out_dir, f"{filename_base}-S.wyo")
        deep_wyo = os.path.join(out_dir, f"{filename_base}-D.wyo")
        
        # Write files
        write_composite_csv(df_subset, csv_filename)
        write_wyo_file(df_subset, 'T_shallow', shallow_wyo, water_year)
        write_wyo_file(df_subset, 'T_deep', deep_wyo, water_year)
        
        # Increment counter for user reference
        selection_state['interval_count'] += 1
        
        print(f"Saved dataset #{selection_state['interval_count']}:")
        print(f"  Records: {len(df_subset)}")
        print(f"  Files: {os.path.basename(csv_filename)}, {os.path.basename(shallow_wyo)}, {os.path.basename(deep_wyo)}")
        
    except ValueError as e:
        print(f"Error: Invalid input values - {e}")
    except Exception as e:
        print(f"Error saving data: {e}")


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
                        out_dir, base_name, water_year, fig, logger):
    """
    Connect all event handlers for interactive plot.
    
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
    """
    def on_span_select(x_start, x_end):
        """Handle span selection from plot."""
        selection_state['start'] = x_start
        selection_state['end'] = x_end
    
        # Update text boxes
        text_boxes['start'].set_val(f"{x_start:.5f}")
        text_boxes['end'].set_val(f"{x_end:.5f}")
    
        # Remove previous current rectangle if it exists
        if 'current_rect' in selection_state and selection_state['current_rect']:
            selection_state['current_rect'].remove()
    
        # Add persistent rectangle for current selection
        y_min, y_max = fig.axes[0].get_ylim()
        rect = Rectangle((x_start, y_min), x_end - x_start, y_max - y_min, 
                        alpha=0.2, facecolor='lightblue', edgecolor='blue', 
                        linewidth=2, linestyle='--')
        fig.axes[0].add_patch(rect)
        selection_state['current_rect'] = rect
        fig.canvas.draw()
    
        print(f"Selected interval: WD {x_start:.5f} to {x_end:.5f}")
    
    def on_save_click(event):
        """Handle save button click."""
        save_interval(event, merged_data, selection_state, text_boxes, 
                     out_dir, base_name, water_year)
        # Move current selection to saved selections and change color
        if 'current_rect' in selection_state and selection_state['current_rect']:
            rect = selection_state['current_rect']
            rect.set_facecolor('lightgreen')
            rect.set_alpha(0.15)
            selection_state['selection_rectangles'].append(rect)
            selection_state['current_rect'] = None
            fig.canvas.draw()
    
    def on_clear_click(event):
        """Handle clear button click - removes all selection rectangles."""
        # Get all Rectangle patches from the axes and remove them
        ax = fig.axes[0]
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


def main():
    """
    Main function to orchestrate the entire processing workflow.
    """
    print("=== TTS_MERGPARS: Temperature Time Series Merger and Processor ===\n")
    
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
        
        # Set up interactive plot
        fig, ax, selector, text_boxes, buttons, selection_state = setup_interactive_plot(
            merged_data, base_name, params['water_year'], params['plot_relative']
        )
        
        # Add this debug line
        print(f"Debug: buttons keys = {list(buttons.keys())}")

        # Connect event handlers
        setup_event_handlers(
            selector, text_boxes, buttons, merged_data, selection_state,
            out_dir, base_name, params['water_year'], fig, logger
        )
        
        # Display instructions and show plot
        print("\n=== Interactive Selection Instructions ===")
        print("1. Click and drag on the plot to select a data interval")
        print("2. Adjust start/end values in text boxes if needed")
        print("3. Modify filename as desired")
        print("4. Click 'Save' to export the selected interval")
        print("   - Leave start/end fields empty to save the full dataset")
        print("5. Repeat for additional intervals with different filenames")
        print("6. Click 'Done' when finished to create full-record files")
        print(f"\nNote: All WYO files will use Water Year {params['water_year']} in the Year column.")
        print("Note: User has complete control over filenames - no automatic suffixes added.")
        
        # Warning messages for anomalous data
        if len(merged_data[merged_data['WaterDay'] < 0]) > 0:
            print("Warning: Negative WaterDay values indicate pre-season data.")
        if len(merged_data[merged_data['WaterDay'] > 365]) > 0:
            print("Warning: WaterDay > 365 indicates data from the next water year.")
        
        # Show the interactive plot
        plt.show()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


# Script entry point
if __name__ == "__main__":
    main()

''' <-- remove these comments to test 
"""
Leap Year Logic Verification for TTS_MERGPARS

This script tests and verifies the leap year logic to ensure it works correctly
for water year calculations.
"""

from datetime import datetime, timedelta

def is_leap_year(year):
    """Check if a given year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_water_year_length(water_year):
    """
    Get the length of a water year in days.
    
    For water year calculations:
    - Water Year 2024 runs from Oct 1, 2023 to Sep 30, 2024
    - February 29th (leap day) falls in calendar year 2024
    - So we check if calendar year 2024 is a leap year
    """
    calendar_year_with_feb = water_year
    return 366 if is_leap_year(calendar_year_with_feb) else 365

def verify_water_year_calculation(water_year):
    """
    Verify water year calculation by actually counting the days.
    """
    start_date = datetime(water_year - 1, 10, 1)  # Oct 1 of previous year
    end_date = datetime(water_year, 9, 30)        # Sep 30 of water year
    
    # Calculate actual number of days
    actual_days = (end_date - start_date).days + 1  # +1 to include both start and end dates
    
    # Get our calculated length
    calculated_length = get_water_year_length(water_year)
    
    return actual_days, calculated_length

def test_leap_year_logic():
    """
    Test the leap year logic with various scenarios.
    """
    print("=== LEAP YEAR LOGIC VERIFICATION ===\n")
    
    # Test basic leap year function
    print("1. Testing basic leap year function:")
    test_years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 1900, 2000]
    for year in test_years:
        is_leap = is_leap_year(year)
        print(f"   {year}: {'LEAP' if is_leap else 'NOT LEAP'}")
    
    print("\n2. Testing water year length calculations:")
    print("   Water Year | Feb 29 in | Expected | Calculated | Actual | Match?")
    print("   -----------|-----------|----------|------------|--------|-------")
    
    water_years_to_test = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]
    
    for wy in water_years_to_test:
        feb_year = wy  # February falls in the same calendar year as water year
        expected = 366 if is_leap_year(feb_year) else 365
        calculated = get_water_year_length(wy)
        actual, _ = verify_water_year_calculation(wy)
        match = "✓" if expected == calculated == actual else "✗"
        
        print(f"   {wy:>10} | {feb_year:>8} | {expected:>8} | {calculated:>10} | {actual:>6} | {match:>6}")
    
    print("\n3. Detailed verification for key years:")
    
    # Test Water Year 2024 (leap year)
    print("\n   Water Year 2024 (LEAP YEAR):")
    print("   - Starts: Oct 1, 2023")
    print("   - Ends: Sep 30, 2024")
    print("   - Contains: Feb 29, 2024 (leap day)")
    actual_2024, calc_2024 = verify_water_year_calculation(2024)
    print(f"   - Calculated length: {calc_2024} days")
    print(f"   - Actual length: {actual_2024} days")
    print(f"   - Correct: {'✓' if calc_2024 == actual_2024 == 366 else '✗'}")
    
    # Test Water Year 2025 (not leap year)
    print("\n   Water Year 2025 (NOT LEAP YEAR):")
    print("   - Starts: Oct 1, 2024")
    print("   - Ends: Sep 30, 2025")
    print("   - No leap day in 2025")
    actual_2025, calc_2025 = verify_water_year_calculation(2025)
    print(f"   - Calculated length: {calc_2025} days")
    print(f"   - Actual length: {actual_2025} days")
    print(f"   - Correct: {'✓' if calc_2025 == actual_2025 == 365 else '✗'}")
    
    print("\n4. Testing water day calculations:")
    
    # Test specific dates in Water Year 2024
    test_dates_2024 = [
        (datetime(2023, 10, 1), "WY 2024 Start"),
        (datetime(2024, 2, 28), "Day before leap day"),
        (datetime(2024, 2, 29), "Leap day"),
        (datetime(2024, 3, 1), "Day after leap day"),
        (datetime(2024, 9, 30), "WY 2024 End"),
    ]
    
    print("\n   Water Year 2024 test dates:")
    water_year_start_2024 = datetime(2023, 10, 1)
    
    for test_date, description in test_dates_2024:
        water_day = (test_date - water_year_start_2024).total_seconds() / 86400.0
        print(f"   {test_date.strftime('%Y-%m-%d')}: Water Day {water_day:7.2f} ({description})")
    
    # Test specific dates in Water Year 2025
    test_dates_2025 = [
        (datetime(2024, 10, 1), "WY 2025 Start"),
        (datetime(2025, 2, 28), "Last day of Feb (no leap day)"),
        (datetime(2025, 3, 1), "March 1"),
        (datetime(2025, 9, 30), "WY 2025 End"),
    ]
    
    print("\n   Water Year 2025 test dates:")
    water_year_start_2025 = datetime(2024, 10, 1)
    
    for test_date, description in test_dates_2025:
        water_day = (test_date - water_year_start_2025).total_seconds() / 86400.0
        print(f"   {test_date.strftime('%Y-%m-%d')}: Water Day {water_day:7.2f} ({description})")
    
    print("\n5. Edge case testing:")
    
    # Test century years (tricky leap year rules)
    century_years = [1900, 2000, 2100, 2400]
    print("\n   Century year leap year tests:")
    for year in century_years:
        is_leap = is_leap_year(year)
        # Verify with Python's built-in logic
        try:
            datetime(year, 2, 29)
            python_says_leap = True
        except ValueError:
            python_says_leap = False
        
        match = "✓" if is_leap == python_says_leap else "✗"
        print(f"   {year}: Our function = {'LEAP' if is_leap else 'NOT LEAP'}, "
              f"Python built-in = {'LEAP' if python_says_leap else 'NOT LEAP'} {match}")
    
    print("\n=== VERIFICATION COMPLETE ===")
    print("\nSUMMARY:")
    print("- The leap year logic correctly identifies leap years using the standard rules")
    print("- Water year length calculation properly accounts for February 29th")
    print("- Water Year 2024 correctly shows 366 days (includes Feb 29, 2024)")
    print("- Water Year 2025 correctly shows 365 days (no leap day)")
    print("- Century year edge cases are handled correctly")

def test_wyo_file_logic():
    """
    Test the logic used in WYO file writing to ensure year assignments are correct.
    """
    print("\n=== WYO FILE YEAR ASSIGNMENT LOGIC TEST ===\n")
    
    # Test Water Year 2024 (leap year)
    water_year = 2024
    water_year_length = get_water_year_length(water_year)
    
    print(f"Water Year {water_year} (Length: {water_year_length} days)")
    print("Water Day | Assigned Calendar Year | Reasoning")
    print("----------|------------------------|----------")
    
    test_water_days = [0, 50, 100, 150, 200, 250, 300, 350, 365, 366, 400]
    
    for wd in test_water_days:
        if wd < water_year_length:
            assigned_year = water_year - 1  # Previous calendar year
            reasoning = f"< {water_year_length} days, so {water_year-1}"
        else:
            assigned_year = water_year      # Current water year
            reasoning = f">= {water_year_length} days, so {water_year}"
        
        print(f"{wd:>9} | {assigned_year:>22} | {reasoning}")
    
    print(f"\nThis means:")
    print(f"- Water Days 0-{water_year_length-1} get assigned to calendar year {water_year-1}")
    print(f"- Water Days {water_year_length}+ get assigned to calendar year {water_year}")

if __name__ == "__main__":
    test_leap_year_logic()
    test_wyo_file_logic()
remove these comments to test ---> '''