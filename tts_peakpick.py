"""
Filename: tts_peakpick.py
Title: Thermal Probe Peak Picker - Enhanced Peak-Trough Algorithm and Logging

Description:
    Graphical Dash-based tool for detecting and validating peaks/troughs in thermal probe data.
    Designed to analyze shallow and deep temperature series for streambed flow and heat transport.

Features:
    - Multiple detection algorithms (SciPy, Wavelet, Prominence, Derivative)
    - Manual peak editing + error flagging
    - Time-range exclusion and data quality checks
    - Peak-trough amplitude calculation and pairing
    - Enhanced session logger (with timestamped log files)
    - Supports parameter persistence via `.par` files
    - Custom export options and UI layout tuning
    - User-selectable year for output files

Enhanced in v1.4:
    - User-selectable year input for export files
    - Amplitude ratio and phase delay analysis
    - Session-aware log management
    - Improved robustness and UI experience

Author: Timothy Wu  
Created: 2025-07-08  
Last Updated: 2025-08-07  
Version: 1.4

Requirements:
    - dash, plotly, pandas, numpy, scipy, PyWavelets

Usage:
    python tts_peakpick.py
"""


# ===========================
# IMPORTS
# ===========================
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH, no_update
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pywt
from datetime import datetime, timedelta
import base64
import io
from io import StringIO
import json
import os
import configparser
from pathlib import Path
import re
import logging
import sys
from logging.handlers import RotatingFileHandler

# ===========================
# ENHANCED LOGGING SYSTEM
# ===========================

class SessionLogger:
    """
    Enhanced logging system for the thermal probe peak picker application.
    Provides both file logging and improved terminal output.
    """
    
    def __init__(self, script_dir):
        self.script_dir = Path(script_dir)
        self.log_dir = self.script_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session timestamp
        self.session_start = datetime.now()
        session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Set up logging
        self.logger = logging.getLogger('TTS_PeakPicker')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler for this session
        log_file = self.log_dir / f'tts_session_{session_id}.log'
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB max file size
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler for terminal output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Redirect stdout to also capture any remaining print statements
        self.original_stdout = sys.stdout
        sys.stdout = LogCapture(self.logger, self.original_stdout)
        
        # Log session start
        self.logger.info("="*80)
        self.logger.info(f"TTS PEAK PICKER SESSION STARTED")
        self.logger.info(f"Session ID: {session_id}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Script directory: {script_dir}")
        self.logger.info(f"All terminal output captured to log file")
        self.logger.info("="*80)
        
        # Store reference for cleanup
        self.session_id = session_id
        self.log_file = log_file
    
    def info(self, message, category="GENERAL"):
        """Log info message with category."""
        self.logger.info(f"[{category}] {message}")
    
    def warning(self, message, category="WARNING"):
        """Log warning message with category."""
        self.logger.warning(f"[{category}] {message}")
    
    def error(self, message, category="ERROR"):
        """Log error message with category."""
        self.logger.error(f"[{category}] {message}")
    
    def debug(self, message, category="DEBUG"):
        """Log debug message with category."""
        self.logger.debug(f"[{category}] {message}")
    
    def data_loaded(self, filename, num_points, time_span, issues=None):
        """Log data loading events."""
        self.info(f"Data loaded: {filename}", "DATA")
        self.info(f"  └─ Points: {num_points:,}, Time span: {time_span:.1f} days", "DATA")
        if issues:
            for issue in issues:
                self.warning(f"  └─ {issue}", "DATA_QUALITY")
    
    def peak_detection(self, method, shallow_count, deep_count, params=None):
        """Log peak detection events."""
        self.info(f"Peak detection: {method}", "PEAKS")
        self.info(f"  └─ Found: {shallow_count} shallow, {deep_count} deep peaks", "PEAKS")
        if params:
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            self.info(f"  └─ Parameters: {param_str}", "PEAKS")
    
    def error_check(self, error_count, error_types=None):
        """Log error checking results."""
        if error_count == 0:
            self.info("Error check: ✅ No errors detected", "ERRORS")
        else:
            self.warning(f"Error check: ⚠️ {error_count} errors detected", "ERRORS")
            if error_types:
                for error_type, count in error_types.items():
                    self.warning(f"  └─ {error_type}: {count} errors", "ERRORS")
    
    def export_data(self, output_dir, files_created):
        """Log data export events."""
        self.info(f"Data exported to: {output_dir}", "EXPORT")
        for file_info in files_created:
            self.info(f"  └─ {file_info}", "EXPORT")
    
    def user_action(self, action, details=None):
        """Log user actions."""
        message = f"User action: {action}"
        if details:
            message += f" ({details})"
        self.info(message, "USER")
    
    def session_end(self):
        """Log session end and restore stdout."""
        duration = datetime.now() - self.session_start
        self.info("="*80)
        self.info(f"SESSION ENDED - Duration: {duration}")
        self.info(f"Log saved to: {self.log_file}")
        self.info("="*80)
        
        # Restore original stdout
        sys.stdout = self.original_stdout
    
    def cleanup_old_logs(self, keep_days=30):
        """Clean up log files older than specified days."""
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            cleaned_count = 0
            
            for log_file in self.log_dir.glob('tts_session_*.log*'):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        log_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    self.warning(f"Could not clean up log file {log_file}: {e}", "CLEANUP")
            
            if cleaned_count > 0:
                self.info(f"Cleaned up {cleaned_count} old log files", "CLEANUP")
                
        except Exception as e:
            self.warning(f"Log cleanup failed: {e}", "CLEANUP")

class LogCapture:
    """Capture stdout to both console and log file."""
    
    def __init__(self, logger, original_stdout):
        self.logger = logger
        self.original_stdout = original_stdout
    
    def write(self, message):
        # Write to original stdout (console)
        self.original_stdout.write(message)
        
        # Also log to file if it's not just whitespace and not from our logger
        if message.strip() and not message.startswith('2025-'):  # Avoid double-logging our own messages
            # Clean up the message and log it
            clean_message = message.strip()
            if clean_message:
                self.logger.info(f"[STDOUT] {clean_message}")
    
    def flush(self):
        self.original_stdout.flush()

# ===========================
# GLOBAL CONFIGURATION
# ===========================

# Get the directory where this script is located
# This ensures all file operations are relative to the script location
try:
    SCRIPT_DIR = Path(__file__).parent.absolute()
except NameError:
    # Fallback for interactive environments
    SCRIPT_DIR = Path.cwd()

# Initialize logger
logger = SessionLogger(SCRIPT_DIR)

# Default parameter file path
PARAM_FILE = SCRIPT_DIR / 'tts_peakpick.par'

# ===========================
# PEAK-TROUGH AMPLITUDE CALCULATION
# ===========================

def calculate_peak_trough_amplitude(peak_indices, trough_indices, peak_data, trough_data, time_array, max_time_diff=0.5):
    """
    Calculate amplitude using peak-trough pairs.
    
    For each peak, finds the closest trough and calculates amplitude as (peak - trough) / 2.
    This method is more accurate for asymmetric waves and cases with subtle trends.
    
    Parameters:
    -----------
    peak_indices : array
        Indices of peak locations
    trough_indices : array
        Indices of trough locations  
    peak_data : array
        Temperature data for peaks
    trough_data : array
        Temperature data for troughs
    time_array : array
        Time values for all data points
    max_time_diff : float, default=0.5
        Maximum time difference (days) to consider for peak-trough pairing
    
    Returns:
    --------
    dict : Contains amplitude information for each peak
    """
    amplitudes = {}
    
    if len(peak_indices) == 0:
        logger.warning("No peaks provided for amplitude calculation", "AMPLITUDE")
        return amplitudes
    
    logger.info(f"Calculating peak-trough amplitudes for {len(peak_indices)} peaks", "AMPLITUDE")
    
    for i, peak_idx in enumerate(peak_indices):
        if peak_idx >= len(time_array):
            logger.warning(f"Peak index {peak_idx} out of range", "AMPLITUDE")
            continue
            
        peak_time = time_array[peak_idx]
        peak_value = peak_data[peak_idx]
        
        # Find closest trough within time window
        closest_trough_idx = None
        closest_trough_value = None
        min_time_diff = float('inf')
        
        for trough_idx in trough_indices:
            if trough_idx >= len(time_array):
                continue
                
            trough_time = time_array[trough_idx]
            time_diff = abs(trough_time - peak_time)
            
            if time_diff <= max_time_diff and time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_trough_idx = trough_idx
                closest_trough_value = trough_data[trough_idx]
        
        # Calculate amplitude
        if closest_trough_idx is not None:
            # Peak-trough method: A = (peak - trough) / 2
            amplitude = (peak_value - closest_trough_value) / 2
            method = "peak-trough"
            trough_time = time_array[closest_trough_idx]
            
            logger.debug(f"Peak {i+1}: Peak={peak_value:.3f} at day {peak_time:.2f}, "
                        f"Trough={closest_trough_value:.3f} at day {trough_time:.2f}, "
                        f"Amplitude={(peak_value - closest_trough_value)/2:.3f}", "AMPLITUDE")
        else:
            # Fallback to peak-only method
            amplitude = abs(peak_value)
            method = "peak-only"
            closest_trough_idx = None
            
            logger.warning(f"Peak {i+1}: No trough found within {max_time_diff} days, "
                          f"using peak-only method (A={amplitude:.3f})", "AMPLITUDE")
        
        amplitudes[peak_idx] = {
            'peak_value': peak_value,
            'peak_time': peak_time,
            'trough_idx': closest_trough_idx,
            'trough_value': closest_trough_value,
            'trough_time': time_array[closest_trough_idx] if closest_trough_idx is not None else None,
            'amplitude': amplitude,
            'method': method,
            'time_diff': min_time_diff if min_time_diff != float('inf') else None
        }
    
    # Log summary
    peak_trough_count = sum(1 for amp in amplitudes.values() if amp['method'] == 'peak-trough')
    peak_only_count = len(amplitudes) - peak_trough_count
    
    logger.info(f"Amplitude calculation summary:", "AMPLITUDE")
    logger.info(f"  └─ Peak-trough pairs: {peak_trough_count}", "AMPLITUDE")
    logger.info(f"  └─ Peak-only fallback: {peak_only_count}", "AMPLITUDE")
    
    return amplitudes

def calculate_amplitude_ratio_enhanced(shallow_amplitudes, deep_amplitudes, time_tolerance=0.5):
    """
    Calculate amplitude ratios using enhanced peak-trough method.
    
    Parameters:
    -----------
    shallow_amplitudes : dict
        Amplitude data for shallow peaks from calculate_peak_trough_amplitude
    deep_amplitudes : dict
        Amplitude data for deep peaks from calculate_peak_trough_amplitude
    time_tolerance : float, default=0.5
        Maximum time difference (days) for pairing shallow and deep peaks
    
    Returns:
    --------
    list : List of dictionaries containing amplitude ratio data
    """
    ratio_data = []
    
    logger.info(f"Calculating enhanced amplitude ratios", "AMPLITUDE")
    logger.info(f"  └─ Shallow peaks: {len(shallow_amplitudes)}, Deep peaks: {len(deep_amplitudes)}", "AMPLITUDE")
    
    for shallow_idx, shallow_data in shallow_amplitudes.items():
        shallow_time = shallow_data['peak_time']
        shallow_amp = shallow_data['amplitude']
        
        # Find corresponding deep peak (should come after shallow peak)
        best_deep_idx = None
        best_time_diff = float('inf')
        
        for deep_idx, deep_data in deep_amplitudes.items():
            deep_time = deep_data['peak_time']
            
            # Look for deep peaks that come after shallow peak
            if deep_time > shallow_time:
                time_diff = deep_time - shallow_time
                if time_diff <= time_tolerance and time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_deep_idx = deep_idx
        
        if best_deep_idx is not None:
            deep_data = deep_amplitudes[best_deep_idx]
            deep_amp = deep_data['amplitude']
            
            # Calculate amplitude ratio: Ar = Ad/As
            if abs(shallow_amp) > 1e-10:  # Avoid division by very small numbers
                ar = deep_amp / shallow_amp
                
                ratio_info = {
                    'shallow_idx': shallow_idx,
                    'deep_idx': best_deep_idx,
                    'shallow_time': shallow_time,
                    'deep_time': deep_data['peak_time'],
                    'shallow_amp': shallow_amp,
                    'deep_amp': deep_amp,
                    'amplitude_ratio': ar,
                    'phase_shift': best_time_diff,
                    'shallow_method': shallow_data['method'],
                    'deep_method': deep_data['method'],
                    'shallow_trough_time': shallow_data['trough_time'],
                    'deep_trough_time': deep_data['trough_time']
                }
                
                ratio_data.append(ratio_info)
                
                logger.debug(f"Paired: Shallow={shallow_amp:.3f} ({shallow_data['method']}) "
                           f"at day {shallow_time:.2f} with Deep={deep_amp:.3f} "
                           f"({deep_data['method']}) at day {deep_data['peak_time']:.2f}, "
                           f"Ar={ar:.3f}, Phase={best_time_diff:.3f} days", "AMPLITUDE")
            else:
                logger.warning(f"Shallow amplitude too small ({shallow_amp:.6f}) "
                             f"for reliable ratio calculation at day {shallow_time:.2f}", "AMPLITUDE")
        else:
            logger.warning(f"No corresponding deep peak found for shallow peak "
                         f"at day {shallow_time:.2f}", "AMPLITUDE")
    
    logger.info(f"Created {len(ratio_data)} amplitude ratio pairs", "AMPLITUDE")
    return ratio_data

# ===========================
# DATA VALIDATION FUNCTIONS
# ===========================

def validate_data_quality(df):
    """
    Validate the quality of loaded data and provide warnings.
    """
    issues = []
    
    # Check for required columns
    required_cols = ['WaterDay', 'Shallow.Temp.Filt', 'Deep.Temp.Filt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        issues.append(f"❌ Missing required columns: {missing_cols}")
        return issues, False  # Critical error
    
    # Check data length
    if len(df) < 100:
        issues.append(f"⚠️ Very short dataset ({len(df)} points). Peak detection may be unreliable.")
    
    # Check for NaN values
    nan_counts = df[required_cols].isnull().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        issues.append(f"⚠️ Found {total_nans} NaN values: {nan_counts.to_dict()}")
    
    # Check time series continuity
    time_diffs = np.diff(df['WaterDay'].values)
    if len(time_diffs) > 0:
        avg_dt = np.mean(time_diffs)
        large_gaps = time_diffs > avg_dt * 5
        if np.any(large_gaps):
            gap_count = np.sum(large_gaps)
            issues.append(f"⚠️ Found {gap_count} large time gaps (>5x average spacing)")
    
    # Check temperature ranges
    for col in ['Shallow.Temp.Filt', 'Deep.Temp.Filt']:
        temp_data = df[col].dropna()
        if len(temp_data) > 0:
            temp_range = temp_data.max() - temp_data.min()
            if temp_range < 0.1:
                issues.append(f"⚠️ Very small temperature range in {col}: {temp_range:.3f}°C")
            if temp_range > 50:
                issues.append(f"⚠️ Very large temperature range in {col}: {temp_range:.1f}°C")
    
    # Check for constant values
    for col in ['Shallow.Temp.Filt', 'Deep.Temp.Filt']:
        temp_data = df[col].dropna()
        if len(temp_data) > 10:
            std_dev = np.std(temp_data)
            if std_dev < 0.01:
                issues.append(f"⚠️ Nearly constant values in {col} (std = {std_dev:.4f})")
    
    return issues, True  # Data is usable

def load_and_validate_csv(file_path_or_content, filename=None):
    """
    Load CSV data with comprehensive validation and error reporting.
    """
    try:
        # Handle both file paths and content
        if isinstance(file_path_or_content, str) and len(file_path_or_content) < 500:
            # Assume it's a file path
            if os.path.exists(file_path_or_content):
                df = pd.read_csv(file_path_or_content)
                if filename is None:
                    filename = os.path.basename(file_path_or_content)
                logger.info(f"Loading CSV file: {file_path_or_content}", "DATA")
            else:
                raise FileNotFoundError(f"File not found: {file_path_or_content}")
        else:
            # Assume it's file content
            if isinstance(file_path_or_content, bytes):
                content = file_path_or_content.decode('utf-8')
            else:
                content = file_path_or_content
            df = pd.read_csv(StringIO(content))
            logger.info(f"Loading uploaded CSV content: {filename or 'unknown'}", "DATA")
    
    except Exception as e:
        error_msg = f"Error reading CSV file: {str(e)}"
        logger.error(error_msg, "DATA")
        return None, error_msg, []
    
    # Validate data quality
    issues, is_usable = validate_data_quality(df)
    
    if not is_usable:
        error_msg = f"Data validation failed: {'; '.join(issues)}"
        logger.error(error_msg, "DATA")
        return None, error_msg, issues
    
    # Success
    success_msg = f"✅ Successfully loaded {len(df)} data points"
    if issues:
        success_msg += f" (with {len(issues)} warnings)"
    
    # Log data loading with enhanced information
    time_span = df['WaterDay'].max() - df['WaterDay'].min()
    logger.data_loaded(filename or 'uploaded_file', len(df), time_span, issues)
    
    return df, success_msg, issues

def validate_parameters(params):
    """
    Validate parameter values and provide sensible defaults.
    """
    validated = params.copy()
    warnings = []
    
    # Sensor spacing validation
    if 'sensor_spacing' in validated:
        spacing = validated['sensor_spacing']
        if spacing <= 0:
            warnings.append("Sensor spacing must be positive, using default 0.18m")
            validated['sensor_spacing'] = 0.18
        elif spacing > 10:
            warnings.append(f"Very large sensor spacing ({spacing}m), check if correct")
    
    # Target period validation
    if 'target_period' in validated:
        period = validated['target_period']
        if period <= 0:
            warnings.append("Target period must be positive, using default 1.0 day")
            validated['target_period'] = 1.0
        elif period > 10:
            warnings.append(f"Very large target period ({period} days), check if correct")
    
    # Prominence factor validation
    if 'prominence_factor' in validated:
        prom = validated['prominence_factor']
        if prom < 0 or prom > 1:
            warnings.append(f"Prominence factor should be 0-1, got {prom}, clamping to valid range")
            validated['prominence_factor'] = max(0, min(1, prom))
    
    # Error checking limits validation
    if 'ar_tolerance' in validated:
        ar_tol = validated['ar_tolerance']
        if ar_tol < 0 or ar_tol > 0.5:
            warnings.append(f"Amplitude ratio tolerance should be 0-0.5, got {ar_tol}")
            validated['ar_tolerance'] = max(0, min(0.5, ar_tol))
    
    if 'phase_min_limit' in validated and 'phase_max_limit' in validated:
        if validated['phase_min_limit'] >= validated['phase_max_limit']:
            warnings.append("Phase min limit >= max limit, swapping values")
            validated['phase_min_limit'], validated['phase_max_limit'] = \
                validated['phase_max_limit'], validated['phase_min_limit']
    
    if warnings:
        for warning in warnings:
            logger.warning(warning, "PARAMS")
    
    return validated, warnings

# ===========================
# PARAMETER FILE FUNCTIONS
# ===========================

def load_param_file(filepath=None):
    """
    Load parameters from .par configuration file.
    
    The .par file uses ConfigParser format with a [PARAMETERS] section
    containing all configuration values.
    
    Parameters:
    -----------
    filepath : Path or str, optional
        Path to parameter file. If None, uses default PARAM_FILE location.
    
    Returns:
    --------
    dict : Dictionary containing all parameters with appropriate type conversions
    """
    if filepath is None:
        filepath = PARAM_FILE
    
    # Default parameters with sensible initial values
    params = {
        'data_file': '',  # Input data file path
        'sensor_spacing': 0.18,  # Distance between sensors in meters
        'target_period': 1.0,  # Expected period between peaks (days)
        'search_tolerance': 0.1,  # Window for searching peaks (days)
        'slope_threshold': 0.001,  # Derivative threshold for peak detection (degC/min)
        'min_distance': 20,  # Minimum samples between peaks
        'prominence_factor': 0.15,  # Lower default for better sensitivity
        'output_folder': 'peak_analysis_output',  # Output directory name
        'peak_method': 'manual',  # Default to manual selection
        'shallow_color': 'blue',  # Color for shallow temperature trace
        'deep_color': 'red',  # Color for deep temperature trace
        'peak_size': 12,  # Size of peak markers
        'trough_size': 10,  # Size of trough markers
        'line_width': 2,  # Width of temperature trace lines
        'ar_tolerance': 0.001,  # Amplitude ratio tolerance for error checking
        'phase_max_limit': 1.0,  # Maximum phase shift limit (days)
        'phase_min_limit': 0.001,  # Minimum phase shift limit (days)
        'amplitude_method': 'peak-trough'  # New: amplitude calculation method
    }
    
    # Load from file if it exists
    if os.path.exists(filepath):
        logger.info(f"Loading parameters from: {filepath}", "PARAMS")
        config = configparser.ConfigParser()
        config.read(filepath)
        
        if 'PARAMETERS' in config:
            # Update defaults with file values
            for key in params:
                if key in config['PARAMETERS']:
                    # Convert to appropriate type based on parameter
                    if key in ['sensor_spacing', 'target_period', 'search_tolerance', 
                              'slope_threshold', 'prominence_factor', 'ar_tolerance', 
                              'phase_max_limit', 'phase_min_limit']:
                        params[key] = float(config['PARAMETERS'][key])
                    elif key in ['min_distance', 'peak_size', 'trough_size', 'line_width']:
                        params[key] = int(config['PARAMETERS'][key])
                    else:
                        params[key] = config['PARAMETERS'][key]
        
        logger.info(f"Loaded {len(params)} parameters", "PARAMS")
    else:
        logger.warning(f"Parameter file not found: {filepath}, using defaults", "PARAMS")
    
    # Validate parameters
    validated_params, warnings = validate_parameters(params)
    
    return validated_params

def save_param_file(params, filepath=None):
    """
    Save parameters to .par configuration file.
    
    Parameters:
    -----------
    params : dict
        Dictionary of parameters to save
    filepath : Path or str, optional
        Path to save file. If None, uses default PARAM_FILE location.
    """
    if filepath is None:
        filepath = PARAM_FILE
    
    config = configparser.ConfigParser()
    config['PARAMETERS'] = {}
    
    # Convert all parameters to strings for ConfigParser
    for key, value in params.items():
        config['PARAMETERS'][key] = str(value)
    
    # Write to file
    with open(filepath, 'w') as f:
        config.write(f)
    
    logger.info(f"Saved parameters to: {filepath}", "PARAMS")

def ensure_output_dir(folder_name):
    """
    Create output directory if it doesn't exist.
    
    Parameters:
    -----------
    folder_name : str
        Name of folder to create in script directory
    
    Returns:
    --------
    Path : Path object pointing to the output directory
    """
    output_path = SCRIPT_DIR / folder_name
    output_path.mkdir(exist_ok=True)
    logger.info(f"Output directory ready: {output_path}", "FILES")
    return output_path

# Load initial parameters
initial_params = load_param_file()

# ===========================
# INITIALIZE DASH APPLICATION
# ===========================
app = dash.Dash(__name__)
logger.info("Dash application initialized", "APP")

# ===========================
# ENHANCED PEAK DETECTION ALGORITHMS
# ===========================

def scipy_find_peaks_method_improved(data, distance=None, prominence=None, height=None):
    """
    Improved scipy find_peaks method with better error handling and parameter tuning.
    """
    if len(data) < 3:
        logger.warning("Data too short for peak detection", "PEAKS")
        return np.array([], dtype=int)
    
    # Remove any DC offset by subtracting mean
    data_centered = data - np.mean(data)
    
    # Handle NaN values
    if np.any(np.isnan(data_centered)):
        logger.warning("NaN values detected in data, interpolating...", "PEAKS")
        valid_mask = ~np.isnan(data_centered)
        if np.sum(valid_mask) < 3:
            return np.array([], dtype=int)
        
        # Simple linear interpolation for NaN values
        from scipy import interpolate
        valid_indices = np.where(valid_mask)[0]
        f = interpolate.interp1d(valid_indices, data_centered[valid_mask], 
                               kind='linear', fill_value='extrapolate')
        data_centered = f(np.arange(len(data_centered)))
    
    # Calculate adaptive prominence if not provided
    if prominence is None:
        data_std = np.std(data_centered)
        data_range = np.max(data_centered) - np.min(data_centered)
        # Use minimum of std-based and range-based prominence
        prominence = min(data_std * 0.1, data_range * 0.05)
        logger.debug(f"Auto-calculated prominence: {prominence:.4f}", "PEAKS")
    
    # Ensure minimum prominence
    if prominence <= 0:
        prominence = np.std(data_centered) * 0.05
    
    # Build kwargs dictionary for find_peaks
    kwargs = {}
    
    if distance is not None and distance > 1:
        kwargs['distance'] = int(distance)
        
    if prominence is not None and prominence > 0:
        kwargs['prominence'] = prominence
    
    # Don't use height restriction by default
    if height is not None:
        kwargs['height'] = height
    
    try:
        # Find peaks with relaxed parameters
        peaks, properties = find_peaks(data_centered, **kwargs)
        
        # If we found too few peaks, try with more relaxed parameters
        if len(peaks) < 3 and distance is not None and distance > 5:
            logger.info(f"Too few peaks ({len(peaks)}), relaxing distance constraint...", "PEAKS")
            kwargs_relaxed = kwargs.copy()
            kwargs_relaxed['distance'] = max(3, distance // 2)
            peaks_relaxed, _ = find_peaks(data_centered, **kwargs_relaxed)
            
            if len(peaks_relaxed) > len(peaks):
                peaks = peaks_relaxed
                logger.info(f"Relaxed detection found {len(peaks)} peaks", "PEAKS")
        
        # Final fallback: find all local maxima with minimal constraints
        if len(peaks) < 2:
            logger.info("Very few peaks found, using local maxima fallback...", "PEAKS")
            all_maxima = []
            for i in range(1, len(data) - 1):
                if (data_centered[i] > data_centered[i-1] and 
                    data_centered[i] > data_centered[i+1] and
                    abs(data_centered[i]) > prominence):
                    all_maxima.append(i)
            
            if len(all_maxima) > len(peaks):
                logger.info(f"Local maxima fallback found {len(all_maxima)} peaks", "PEAKS")
                peaks = np.array(all_maxima, dtype=int)
        
    except Exception as e:
        logger.error(f"Error in scipy peak detection: {e}", "PEAKS")
        return np.array([], dtype=int)
    
    # Validate peak indices
    valid_peaks = peaks[(peaks >= 0) & (peaks < len(data))]
    
    if len(valid_peaks) != len(peaks):
        logger.warning(f"Removed {len(peaks) - len(valid_peaks)} invalid peak indices", "PEAKS")
    
    return valid_peaks.astype(int)

def wavelet_peak_detection(data, widths=None, min_snr=1):
    """
    Continuous Wavelet Transform peak detection using PyWavelets.
    """
    if widths is None:
        # Adaptive width selection based on data length
        widths = np.arange(5, min(100, len(data)//5), 2)
    
    # Perform CWT with Mexican hat wavelet
    coeffs = []
    for width in widths:
        wavelet_data, _ = pywt.cwt(data, [width], 'mexh')
        coeffs.append(wavelet_data[0])
    
    cwt_matrix = np.array(coeffs)
    
    # Find ridgelines in the CWT matrix
    ridge_lines = []
    for i in range(cwt_matrix.shape[1]):
        column = cwt_matrix[:, i]
        # Check if signal is strong enough
        if np.max(np.abs(column)) > min_snr * np.std(cwt_matrix):
            ridge_lines.append(i)
    
    # Refine peaks using local maxima
    peaks = np.array([], dtype=int)
    if len(ridge_lines) > 0:
        # Check for local maxima at ridge locations
        potential_peaks = []
        for idx in ridge_lines:
            if 1 < idx < len(data) - 1:
                if data[idx] > data[idx-1] and data[idx] > data[idx+1]:
                    potential_peaks.append(idx)
        
        if len(potential_peaks) > 0:
            peaks = np.array(potential_peaks, dtype=int)
    
    logger.debug(f"Wavelet detection found {len(peaks)} peaks", "PEAKS")
    return peaks

def derivative_peak_detection(data, time, slope_threshold=None, min_peak_distance=10):
    """
    Custom derivative-based peak and trough detection.
    """
    # Remove DC offset
    data_centered = data - np.mean(data)
    
    # Smooth the data to reduce noise effects
    # Use adaptive sigma based on data length
    sigma = max(2, min(5, len(data) // 100))
    smoothed_data = gaussian_filter1d(data_centered, sigma=sigma)
    
    # Calculate derivative
    dt = np.gradient(time)
    # Avoid division by zero
    dt[dt == 0] = np.min(dt[dt > 0])
    dT_dt = np.gradient(smoothed_data) / dt
    
    # Also calculate second derivative for better peak detection
    d2T_dt2 = np.gradient(dT_dt) / dt
    
    # Find all local maxima and minima
    peaks = []
    troughs = []
    
    # Look for points where derivative crosses zero
    for i in range(2, len(data) - 2):
        # Peak: derivative changes from positive to negative
        # AND second derivative is negative (concave down)
        if (dT_dt[i-1] > 0 and dT_dt[i+1] < 0) or \
           (abs(dT_dt[i]) < abs(dT_dt[i-1]) and abs(dT_dt[i]) < abs(dT_dt[i+1]) and d2T_dt2[i] < 0):
            # Verify it's a local maximum in the original data
            if data[i] >= data[i-1] and data[i] >= data[i+1]:
                peaks.append(i)
        
        # Trough: derivative changes from negative to positive
        # AND second derivative is positive (concave up)
        elif (dT_dt[i-1] < 0 and dT_dt[i+1] > 0) or \
             (abs(dT_dt[i]) < abs(dT_dt[i-1]) and abs(dT_dt[i]) < abs(dT_dt[i+1]) and d2T_dt2[i] > 0):
            # Verify it's a local minimum in the original data
            if data[i] <= data[i-1] and data[i] <= data[i+1]:
                troughs.append(i)
    
    # Remove peaks that are too close together
    if len(peaks) > 1:
        peaks = np.array(peaks)
        keep = np.ones(len(peaks), dtype=bool)
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] < min_peak_distance:
                # Keep the higher peak
                if data[peaks[i]] < data[peaks[i-1]]:
                    keep[i] = False
                else:
                    keep[i-1] = False
        peaks = peaks[keep]
    
    # Same for troughs
    if len(troughs) > 1:
        troughs = np.array(troughs)
        keep = np.ones(len(troughs), dtype=bool)
        for i in range(1, len(troughs)):
            if troughs[i] - troughs[i-1] < min_peak_distance:
                # Keep the lower trough
                if data[troughs[i]] > data[troughs[i-1]]:
                    keep[i] = False
                else:
                    keep[i-1] = False
        troughs = troughs[keep]
    
    # Ensure we return integer arrays
    peaks = np.array(peaks, dtype=int) if len(peaks) > 0 else np.array([], dtype=int)
    troughs = np.array(troughs, dtype=int) if len(troughs) > 0 else np.array([], dtype=int)
    
    logger.info(f"Derivative method found {len(peaks)} peaks and {len(troughs)} troughs", "PEAKS")
    
    return peaks, troughs

def combined_peak_detection(data, time, target_period=1.0, min_peak_distance=10):
    """
    Combined peak detection using multiple methods for robustness.
    """
    all_peaks = []
    
    # Method 1: Simple local maxima
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            all_peaks.append(i)
    
    # Method 2: SciPy with various prominence values
    data_centered = data - np.mean(data)
    data_std = np.std(data_centered)
    
    for prom_factor in [0.05, 0.1, 0.15, 0.2, 0.3]:
        try:
            peaks, _ = find_peaks(data_centered, prominence=data_std * prom_factor, distance=min_peak_distance)
            all_peaks.extend(peaks)
        except:
            pass
    
    # Method 3: Find peaks without any constraints
    try:
        peaks_no_constraint, _ = find_peaks(data_centered)
        all_peaks.extend(peaks_no_constraint)
    except:
        pass
    
    # Method 4: Find peaks with very minimal constraints
    try:
        peaks_minimal, _ = find_peaks(data, distance=5)
        all_peaks.extend(peaks_minimal)
    except:
        pass
    
    # Remove duplicates and sort
    all_peaks = sorted(list(set(all_peaks)))
    
    # Filter peaks that are too close together
    if len(all_peaks) > 1:
        filtered_peaks = [all_peaks[0]]
        for i in range(1, len(all_peaks)):
            if all_peaks[i] - filtered_peaks[-1] >= min_peak_distance:
                filtered_peaks.append(all_peaks[i])
            else:
                # Keep the higher peak
                if data[all_peaks[i]] > data[filtered_peaks[-1]]:
                    filtered_peaks[-1] = all_peaks[i]
        
        all_peaks = filtered_peaks
    
    logger.info(f"Combined method found {len(all_peaks)} peaks", "PEAKS")
    
    return np.array(all_peaks, dtype=int) if len(all_peaks) > 0 else np.array([], dtype=int)

def prominence_based_detection(data, prominence_factor=0.1):
    """
    Peak detection based on prominence relative to signal amplitude.
    """
    # Remove DC offset
    data_centered = data - np.mean(data)
    
    # Calculate prominence threshold
    data_range = np.max(data_centered) - np.min(data_centered)
    min_prominence = data_range * prominence_factor
    
    # Try to find peaks with current prominence
    peaks, _ = find_peaks(data_centered, prominence=min_prominence, distance=10)
    
    # If too few peaks, relax the constraint
    if len(peaks) < 5:
        min_prominence = data_range * prominence_factor * 0.5
        peaks, _ = find_peaks(data_centered, prominence=min_prominence, distance=10)
        logger.info(f"Relaxed prominence to {min_prominence:.4f}, found {len(peaks)} peaks", "PEAKS")
    
    if len(peaks) == 0:
        return np.array([], dtype=int)
    return peaks.astype(int)

def bootstrap_peak_detection_improved(data, time, initial_peaks, period=1.0, tolerance=0.2):
    """
    Improved bootstrap peak detection with better error handling.
    """
    if len(initial_peaks) < 1:
        logger.error("Need at least 1 initial peak for bootstrap", "PEAKS")
        return np.array([], dtype=int)
    
    if len(data) < 10:
        logger.error("Data too short for bootstrap", "PEAKS")
        return np.array(initial_peaks, dtype=int)
    
    # Sort initial peaks and validate indices
    initial_peaks = [p for p in initial_peaks if 0 <= p < len(data)]
    if len(initial_peaks) == 0:
        return np.array([], dtype=int)
    
    initial_peaks = sorted(initial_peaks)
    
    # Calculate periods from initial peaks if we have more than one
    if len(initial_peaks) >= 2:
        peak_times = time[initial_peaks]
        periods = np.diff(peak_times)
        # Use median period for robustness
        avg_period = np.median(periods) if len(periods) > 0 else period
        # Filter out outlier periods
        valid_periods = periods[np.abs(periods - avg_period) < avg_period * 0.5]
        if len(valid_periods) > 0:
            avg_period = np.mean(valid_periods)
    else:
        avg_period = period
    
    logger.info(f"Bootstrap: Starting with {len(initial_peaks)} peaks, avg period = {avg_period:.3f} days", "PEAKS")
    
    all_peaks = list(initial_peaks)
    
    # Forward propagation with improved search
    last_peak_time = time[initial_peaks[-1]]
    last_peak_idx = initial_peaks[-1]
    
    search_attempts = 0
    max_attempts = int((time[-1] - last_peak_time) / avg_period) + 5
    
    while last_peak_idx < len(data) - 10 and search_attempts < max_attempts:
        search_attempts += 1
        expected_time = last_peak_time + avg_period
        
        # Adaptive tolerance based on data quality
        adaptive_tolerance = min(tolerance, avg_period * 0.3)
        
        search_start = max(0, np.searchsorted(time, expected_time - adaptive_tolerance))
        search_end = min(len(data), np.searchsorted(time, expected_time + adaptive_tolerance))
        
        if search_start < len(data) and search_end > search_start and search_end - search_start > 2:
            window_data = data[search_start:search_end]
            
            # Find the highest peak in the window
            local_peak = np.argmax(window_data)
            peak_idx = search_start + local_peak
            
            # Validate it's a true local maximum in the broader context
            validation_start = max(0, peak_idx - 3)
            validation_end = min(len(data), peak_idx + 4)
            
            if (peak_idx >= validation_start + 1 and peak_idx < validation_end - 1 and
                data[peak_idx] >= np.max(data[validation_start:validation_end]) * 0.95):
                
                if peak_idx not in all_peaks:
                    all_peaks.append(peak_idx)
                    last_peak_time = time[peak_idx]
                    last_peak_idx = peak_idx
                    logger.debug(f"Bootstrap forward: added peak at day {last_peak_time:.2f}", "PEAKS")
                else:
                    break
            else:
                # No valid peak found, try next expected location
                last_peak_time = expected_time
                last_peak_idx = search_end
        else:
            break
    
    # Backward propagation with similar improvements
    first_peak_time = time[initial_peaks[0]]
    first_peak_idx = initial_peaks[0]
    
    search_attempts = 0
    max_attempts = int((first_peak_time - time[0]) / avg_period) + 5
    
    while first_peak_idx > 10 and search_attempts < max_attempts:
        search_attempts += 1
        expected_time = first_peak_time - avg_period
        
        if expected_time < time[0]:
            break
        
        adaptive_tolerance = min(tolerance, avg_period * 0.3)
        
        search_start = max(0, np.searchsorted(time, expected_time - adaptive_tolerance))
        search_end = min(len(data), np.searchsorted(time, expected_time + adaptive_tolerance))
        
        if search_start < len(data) and search_end > search_start and search_end - search_start > 2:
            window_data = data[search_start:search_end]
            
            local_peak = np.argmax(window_data)
            peak_idx = search_start + local_peak
            
            # Validate it's a true local maximum
            validation_start = max(0, peak_idx - 3)
            validation_end = min(len(data), peak_idx + 4)
            
            if (peak_idx >= validation_start + 1 and peak_idx < validation_end - 1 and
                data[peak_idx] >= np.max(data[validation_start:validation_end]) * 0.95):
                
                if peak_idx not in all_peaks:
                    all_peaks.insert(0, peak_idx)
                    first_peak_time = time[peak_idx]
                    first_peak_idx = peak_idx
                    logger.debug(f"Bootstrap backward: added peak at day {first_peak_time:.2f}", "PEAKS")
                else:
                    break
            else:
                first_peak_time = expected_time
                first_peak_idx = search_start
        else:
            break
    
    final_peaks = sorted(list(set(all_peaks)))
    logger.info(f"Bootstrap: Found {len(final_peaks)} total peaks", "PEAKS")
    
    return np.array(final_peaks, dtype=int)

# Legacy function name for compatibility
def bootstrap_peak_detection(data, time, initial_peaks, period=1.0, tolerance=0.2):
    return bootstrap_peak_detection_improved(data, time, initial_peaks, period, tolerance)

# Legacy function name for compatibility
def scipy_find_peaks_method(data, distance=None, prominence=None, height=None):
    return scipy_find_peaks_method_improved(data, distance, prominence, height)

# ===========================
# ENHANCED ERROR CHECKING FUNCTIONS WITH PEAK-TROUGH SUPPORT
# ===========================

def check_alternation_improved(shallow_peak_times, deep_peak_times):
    """
    Improved alternation checking with better logic.
    """
    errors = []
    
    if len(shallow_peak_times) == 0 or len(deep_peak_times) == 0:
        return errors
    
    # Combine and sort all peaks with their types
    all_peaks = []
    for t in shallow_peak_times:
        all_peaks.append((t, 'shallow'))
    for t in deep_peak_times:
        all_peaks.append((t, 'deep'))
    
    all_peaks.sort(key=lambda x: x[0])
    
    logger.debug(f"Alternation Check: Found {len(all_peaks)} total peaks", "ERRORS")
    for i, (time, depth) in enumerate(all_peaks[:10]):  # Show first 10
        logger.debug(f"  Peak {i+1}: {depth} at day {time:.2f}", "ERRORS")
    
    # Check for alternation violations (consecutive peaks of same type)
    consecutive_violations = []
    for i in range(1, len(all_peaks)):
        if all_peaks[i][1] == all_peaks[i-1][1]:
            # Found consecutive peaks of same type
            consecutive_violations.append({
                'type': 'alternation',
                'time': all_peaks[i][0],  # Time of second peak in violation
                'message': f'Two consecutive {all_peaks[i][1]} peaks at days {all_peaks[i-1][0]:.2f} and {all_peaks[i][0]:.2f}'
            })
    
    errors.extend(consecutive_violations)
    
    # Also check for missing expected alternation patterns
    # If we have significantly more of one type than the other
    shallow_count = len(shallow_peak_times)
    deep_count = len(deep_peak_times)
    
    if abs(shallow_count - deep_count) > 2:  # Allow small differences
        dominant_type = 'shallow' if shallow_count > deep_count else 'deep'
        missing_type = 'deep' if shallow_count > deep_count else 'shallow'
        
        errors.append({
            'type': 'alternation',
            'time': (shallow_peak_times[0] + deep_peak_times[0]) / 2 if len(shallow_peak_times) > 0 and len(deep_peak_times) > 0 else 0,
            'message': f'Imbalanced peak counts: {shallow_count} shallow vs {deep_count} deep peaks. Missing {missing_type} peaks?'
        })
    
    logger.debug(f"Found {len(errors)} alternation errors", "ERRORS")
    return errors

def check_amplitude_and_phase_enhanced(shallow_peak_indices, deep_peak_indices, 
                                      shallow_data, deep_data, time_array,
                                      shallow_trough_indices, deep_trough_indices,
                                      amplitude_method='peak-trough',
                                      ar_tolerance=0.001, phase_min_limit=0.001, phase_max_limit=1.0):
    """
    Enhanced amplitude and phase checking with support for peak-trough amplitude calculation.
    
    Parameters:
    -----------
    shallow_peak_indices, deep_peak_indices : array
        Indices of peaks in the data
    shallow_data, deep_data : array
        Temperature data
    time_array : array
        Time values
    shallow_trough_indices, deep_trough_indices : array
        Indices of troughs in the data
    amplitude_method : str, default='peak-trough'
        Method for amplitude calculation: 'peak-only' or 'peak-trough'
    ar_tolerance : float
        Tolerance for amplitude ratio checking
    phase_min_limit, phase_max_limit : float
        Phase shift limits in days
    
    Returns:
    --------
    list : List of error dictionaries
    """
    errors = []
    
    if len(shallow_peak_indices) == 0 or len(deep_peak_indices) == 0:
        logger.debug("Amplitude/Phase Check: No peaks to check", "ERRORS")
        return errors
    
    logger.info(f"Amplitude/Phase Check: {len(shallow_peak_indices)} shallow, {len(deep_peak_indices)} deep peaks", "ERRORS")
    logger.info(f"Using amplitude method: {amplitude_method}", "ERRORS")
    
    # Calculate amplitudes using the specified method
    if amplitude_method == 'peak-trough' and len(shallow_trough_indices) > 0 and len(deep_trough_indices) > 0:
        # Use enhanced peak-trough method
        shallow_amplitudes = calculate_peak_trough_amplitude(
            shallow_peak_indices, shallow_trough_indices, 
            shallow_data, shallow_data, time_array
        )
        deep_amplitudes = calculate_peak_trough_amplitude(
            deep_peak_indices, deep_trough_indices,
            deep_data, deep_data, time_array
        )
        
        # Get amplitude ratio data
        ratio_data = calculate_amplitude_ratio_enhanced(shallow_amplitudes, deep_amplitudes)
        
        # Check each ratio for errors
        for ratio_info in ratio_data:
            s_time = ratio_info['shallow_time']
            ar = ratio_info['amplitude_ratio']
            phase_shift = ratio_info['phase_shift']
            
            # Check amplitude ratio constraints
            if ar >= (1.0 - ar_tolerance):
                errors.append({
                    'type': 'amplitude_ratio_high',
                    'time': s_time,
                    'message': f'Ar = {ar:.4f} >= {1.0 - ar_tolerance:.3f} at day {s_time:.2f} '
                              f'(Method: {ratio_info["shallow_method"]}/{ratio_info["deep_method"]})'
                })
            elif ar <= ar_tolerance:
                errors.append({
                    'type': 'amplitude_ratio_low', 
                    'time': s_time,
                    'message': f'Ar = {ar:.4f} <= {ar_tolerance:.3f} at day {s_time:.2f} '
                              f'(Method: {ratio_info["shallow_method"]}/{ratio_info["deep_method"]})'
                })
            
            # Check phase shift constraints
            if phase_shift <= phase_min_limit:
                errors.append({
                    'type': 'phase_too_small',
                    'time': s_time,
                    'message': f'Phase shift {phase_shift:.4f} <= {phase_min_limit:.3f} days at day {s_time:.2f}'
                })
            elif phase_shift > phase_max_limit:
                errors.append({
                    'type': 'phase_too_large',
                    'time': s_time,
                    'message': f'Phase shift {phase_shift:.3f} > {phase_max_limit:.1f} days at day {s_time:.2f}'
                })
    
    else:
        # Fallback to original peak-only method
        logger.info("Using fallback peak-only amplitude method", "ERRORS")
        
        # Get peak times and amplitudes
        shallow_times = time_array[shallow_peak_indices]
        shallow_amps = shallow_data[shallow_peak_indices]
        deep_times = time_array[deep_peak_indices]
        deep_amps = deep_data[deep_peak_indices]
        
        # Improved pairing algorithm: find closest deep peak for each shallow peak
        valid_pairs = 0
        for i, (s_time, s_amp) in enumerate(zip(shallow_times, shallow_amps)):
            
            # Find the closest deep peak that comes AFTER this shallow peak
            future_deep_mask = deep_times > s_time
            
            if np.any(future_deep_mask):
                # Get future deep peaks and their distances
                future_deep_times = deep_times[future_deep_mask]
                future_deep_amps = deep_amps[future_deep_mask]
                
                # Find the closest one (should be the next expected peak)
                time_differences = future_deep_times - s_time
                closest_idx = np.argmin(time_differences)
                
                d_time = future_deep_times[closest_idx]
                d_amp = future_deep_amps[closest_idx]
                
                valid_pairs += 1
                
                # Calculate amplitude ratio: Ar = Ad/As
                if abs(s_amp) > 1e-10:  # Avoid division by very small numbers
                    ar = abs(d_amp) / abs(s_amp)  # Use absolute values for amplitude comparison
                    
                    # Check amplitude ratio constraints
                    if ar >= (1.0 - ar_tolerance):
                        errors.append({
                            'type': 'amplitude_ratio_high',
                            'time': s_time,
                            'message': f'Ar = {ar:.4f} >= {1.0 - ar_tolerance:.3f} at day {s_time:.2f} (As={s_amp:.3f}, Ad={d_amp:.3f})'
                        })
                    elif ar <= ar_tolerance:
                        errors.append({
                            'type': 'amplitude_ratio_low', 
                            'time': s_time,
                            'message': f'Ar = {ar:.4f} <= {ar_tolerance:.3f} at day {s_time:.2f} (As={s_amp:.3f}, Ad={d_amp:.3f})'
                        })
                    
                    # Check phase shift constraints
                    phase_shift = d_time - s_time
                    if phase_shift <= phase_min_limit:
                        errors.append({
                            'type': 'phase_too_small',
                            'time': s_time,
                            'message': f'Phase shift {phase_shift:.4f} <= {phase_min_limit:.3f} days at day {s_time:.2f}'
                        })
                    elif phase_shift > phase_max_limit:
                        errors.append({
                            'type': 'phase_too_large',
                            'time': s_time,
                            'message': f'Phase shift {phase_shift:.3f} > {phase_max_limit:.1f} days at day {s_time:.2f}'
                        })
                    
                    # Debug info for first few pairs
                    if valid_pairs <= 3:
                        logger.debug(f"Pair {valid_pairs}: Shallow {s_time:.2f} -> Deep {d_time:.2f}, Ar={ar:.3f}, Phase={phase_shift:.3f}", "ERRORS")
                else:
                    errors.append({
                        'type': 'amplitude_ratio_low',
                        'time': s_time,
                        'message': f'Shallow amplitude too small ({s_amp:.6f}) for reliable ratio calculation at day {s_time:.2f}'
                    })
    
    logger.info(f"Processed amplitude/phase check, found {len(errors)} errors", "ERRORS")
    return errors

def check_mismatched_peaks_improved(shallow_peak_indices, deep_peak_indices, time_array, tolerance=0.5):
    """
    Improved mismatch checking with better tolerance handling.
    """
    errors = []
    
    if len(shallow_peak_indices) == 0 and len(deep_peak_indices) == 0:
        return errors
    
    # Get peak times
    shallow_times = time_array[shallow_peak_indices] if len(shallow_peak_indices) > 0 else np.array([])
    deep_times = time_array[deep_peak_indices] if len(deep_peak_indices) > 0 else np.array([])
    
    logger.debug(f"Mismatch Check: {len(shallow_times)} shallow, {len(deep_times)} deep peaks", "ERRORS")
    
    # Check for isolated shallow peaks (no deep peak within reasonable time)
    unmatched_shallow = 0
    for s_time in shallow_times:
        # Look for deep peaks within tolerance window AFTER this shallow peak
        future_deep_mask = (deep_times > s_time) & (deep_times <= s_time + tolerance)
        nearby_deep = np.sum(future_deep_mask)
        
        if nearby_deep == 0:
            unmatched_shallow += 1
            errors.append({
                'type': 'unmatched_shallow',
                'time': s_time,
                'message': f'No deep peak within {tolerance:.1f} days after shallow peak at day {s_time:.2f}'
            })
        elif nearby_deep > 1:
            # Multiple deep peaks following one shallow peak
            deep_matches = deep_times[future_deep_mask]
            errors.append({
                'type': 'multiple_matches',
                'time': s_time,
                'message': f'{nearby_deep} deep peaks within {tolerance:.1f} days of shallow peak at day {s_time:.2f}: {deep_matches}'
            })
    
    # Check for isolated deep peaks (no shallow peak before them)
    unmatched_deep = 0
    for d_time in deep_times:
        # Look for shallow peaks within tolerance window BEFORE this deep peak
        preceding_shallow_mask = (shallow_times < d_time) & (shallow_times >= d_time - tolerance)
        nearby_shallow = np.sum(preceding_shallow_mask)
        
        if nearby_shallow == 0:
            unmatched_deep += 1
            errors.append({
                'type': 'unmatched_deep',
                'time': d_time,
                'message': f'No shallow peak within {tolerance:.1f} days before deep peak at day {d_time:.2f}'
            })
    
    logger.debug(f"{unmatched_shallow} unmatched shallow, {unmatched_deep} unmatched deep peaks", "ERRORS")
    return errors

# Wrapper functions for compatibility
def check_alternation(shallow_peak_times, deep_peak_times):
    return check_alternation_improved(shallow_peak_times, deep_peak_times)

def check_amplitude_and_phase(shallow_peak_indices, deep_peak_indices, 
                             shallow_data, deep_data, time_array,
                             ar_tolerance=0.001, phase_min_limit=0.001, phase_max_limit=1.0):
    # For backward compatibility, use peak-only method
    return check_amplitude_and_phase_enhanced(
        shallow_peak_indices, deep_peak_indices, 
        shallow_data, deep_data, time_array,
        np.array([], dtype=int), np.array([], dtype=int),  # Empty trough arrays
        'peak-only', ar_tolerance, phase_min_limit, phase_max_limit
    )

def check_mismatched_peaks(shallow_peak_indices, deep_peak_indices, time_array, tolerance=0.5):
    return check_mismatched_peaks_improved(shallow_peak_indices, deep_peak_indices, time_array, tolerance)

# ===========================
# DASH LAYOUT
# ===========================
app.title = "TTS PEAKPICK - Temperature Time-Series Peak Picker"

app.layout = html.Div([
    # Application header
    html.H1("Thermal Probe Peak Picker - Enhanced Version with Logging and Peak-Trough Algorithm", style={'textAlign': 'center'}),
    
    # Parameter file controls section
    html.Div([
        html.Label("Parameter File: "),
        html.Span(str(PARAM_FILE), id='param-file-path'),
        html.Button('Load Parameters', id='load-param-file-button', n_clicks=0, 
                   style={'marginLeft': '10px'}),
        html.Button('Save to .par', id='save-param-file-button', n_clicks=0, 
                   style={'marginLeft': '10px'}),
        html.Button('Auto-Load CSV from .par', id='auto-load-csv-button', n_clicks=0, 
                   style={'marginLeft': '10px', 'backgroundColor': '#4CAF50', 'color': 'white'}),
        html.Span(id='param-file-status', style={'marginLeft': '10px'})
    ], style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}),
    
    # File upload section
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '98%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    
    # Current file display
    html.Div(id='current-file-display', style={'margin': '10px', 'padding': '10px', 
                                              'backgroundColor': '#e6f3ff', 'borderRadius': '5px'}),
    
    # Main graph - DATA VERIFICATION (READ-ONLY)
    html.Div([
        html.H3("Data Verification (Read-Only)", style={'margin': '10px 0 5px 0'}),
        html.P("This graph shows your loaded data for verification. Use the Interactive Peak Picking graph below for editing.", 
               style={'margin': '0 10px 10px 10px', 'fontSize': '0.9em', 'color': 'gray'}),
        dcc.Graph(id='main-graph', style={'height': '400px', 'margin': '10px'})
    ]),
    
    # Peak detection method selection - MOVED BELOW GRAPH
    html.Div([
        html.H3("Peak Detection Method"),
        html.Label("Detection Method:"),
        dcc.Dropdown(
            id='peak-method-dropdown',
            options=[
                {'label': 'Manual Selection (Default)', 'value': 'manual'},
                {'label': 'SciPy Find Peaks', 'value': 'scipy'},
                {'label': 'Wavelet-based Detection', 'value': 'wavelet'},
                {'label': 'Custom Derivative Method', 'value': 'derivative'},
                {'label': 'Prominence-based Detection', 'value': 'prominence'},
                {'label': 'Combined Methods (Most Sensitive)', 'value': 'combined'},
                {'label': 'Bootstrap from Manual Selection', 'value': 'bootstrap'}
            ],
            value=initial_params['peak_method'],
            style={'width': '400px'}
        )
    ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd'}),
    
    # Detection parameters section - MOVED BELOW METHOD SELECTION
    html.Div([
        html.H3("Detection Parameters"),
        # Row 1 of parameters
        html.Div([
            html.Div([
                html.Label("Sensor Spacing (m):"),
                dcc.Input(id='sensor-spacing', type='number', 
                         value=initial_params['sensor_spacing'], step=0.01),
            ], style={'display': 'inline-block', 'margin': '5px'}),
            html.Div([
                html.Label("Target Period (days):"),
                dcc.Input(id='target-period', type='number', 
                         value=initial_params['target_period'], step=0.1),
            ], style={'display': 'inline-block', 'margin': '5px'}),
            html.Div([
                html.Label("Search Tolerance (days):"),
                dcc.Input(id='search-tolerance', type='number', 
                         value=initial_params['search_tolerance'], step=0.01),
            ], style={'display': 'inline-block', 'margin': '5px'}),
        ]),
        # Row 2 of parameters
        html.Div([
            html.Div([
                html.Label("Slope Threshold (degC/min):"),
                dcc.Input(id='slope-threshold', type='number', 
                         value=initial_params['slope_threshold'], step=0.0001),
            ], style={'display': 'inline-block', 'margin': '5px'}),
            html.Div([
                html.Label("Min Peak Distance (samples):"),
                dcc.Input(id='min-distance', type='number', 
                         value=initial_params['min_distance'], step=1),
            ], style={'display': 'inline-block', 'margin': '5px'}),
            html.Div([
                html.Label("Prominence Factor (0-1):"),
                dcc.Input(id='prominence-factor', type='number', 
                         value=initial_params['prominence_factor'], min=0, max=1, step=0.05),
                html.Span(" (Lower = more peaks)", style={'fontSize': '0.9em', 'color': 'gray', 'marginLeft': '5px'})
            ], style={'display': 'inline-block', 'margin': '5px'}),
        ]),
        # NEW: Amplitude calculation method selection
        html.Div([
            html.H4("Amplitude Calculation Method", style={'color': '#2E86AB', 'margin': '10px 0 5px 0'}),
            html.Label("Amplitude Calculation:"),
            dcc.Dropdown(
                id='amplitude-method',
                options=[
                    {'label': 'Peak-Trough Method (Recommended): A = (Peak - Trough) / 2', 'value': 'peak-trough'},
                    {'label': 'Peak-Only Method (Original): A = Peak Value', 'value': 'peak-only'}
                ],
                value=initial_params.get('amplitude_method', 'peak-trough'),
                style={'width': '500px', 'margin': '5px 0'}
            ),
            html.P("Peak-Trough method provides more accurate amplitudes for asymmetric waves and cases with subtle trends.", 
                   style={'fontSize': '0.9em', 'color': 'gray', 'margin': '5px 0'})
        ], style={'backgroundColor': '#f0f8ff', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'}),
        # Error checking parameters
        html.Div([
            html.H4("Error Checking Limits"),
            html.Div([
                html.Label("Amplitude Ratio Tolerance:"),
                dcc.Input(id='ar-tolerance', type='number', 
                         value=initial_params.get('ar_tolerance', 0.001), step=0.001, min=0),
                html.Span(" (for Ar >= 1-tolerance or <= tolerance)", style={'fontSize': '0.8em', 'color': 'gray'})
            ], style={'display': 'inline-block', 'margin': '5px'}),
            html.Div([
                html.Label("Max Phase Shift (days):"),
                dcc.Input(id='phase-max-limit', type='number', 
                         value=initial_params.get('phase_max_limit', 1.0), step=0.1, min=0),
            ], style={'display': 'inline-block', 'margin': '5px'}),
            html.Div([
                html.Label("Min Phase Shift (days):"),
                dcc.Input(id='phase-min-limit', type='number', 
                         value=initial_params.get('phase_min_limit', 0.001), step=0.001, min=0),
            ], style={'display': 'inline-block', 'margin': '5px'}),
        ], style={'backgroundColor': '#fff9e6', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'}),
        # Action buttons
        html.Div([
            html.Button('Detect Peaks', id='detect-button', n_clicks=0),
            html.Button('Clear Manual Selections', id='clear-manual-button', n_clicks=0, 
                       style={'marginLeft': '10px'}),
            html.Button('Clear Troughs', id='clear-troughs-button', n_clicks=0, 
                       style={'marginLeft': '10px', 'backgroundColor': '#ff9800', 'color': 'white'}),
            html.Button('Toggle Peak Visibility', id='toggle-visibility-button', n_clicks=0, 
                       style={'marginLeft': '10px', 'backgroundColor': '#9c27b0', 'color': 'white'}),
        ], style={'marginTop': '10px'}),
        # Tips for better detection
        html.Div([
            html.P("💡 Tips: For better peak detection and management:", style={'margin': '5px 0', 'fontWeight': 'bold'}),
            html.Ul([
                html.Li("Manual Selection: Click on peaks to add them manually (most accurate)"),
                html.Li("If automated methods miss peaks: Reduce Prominence Factor (e.g., 0.1 or 0.05)"),
                html.Li("For noisy data: Use 'Custom Derivative Method' or 'Combined Methods'"),
                html.Li("Bootstrap: Add a few manual peaks first, then use 'Bootstrap' to find similar patterns"),
                html.Li("Peak-Trough Method: Better for asymmetric waves and low amplitude signals"),
                html.Li("All detection methods allow manual editing - add/remove peaks as needed"),
                html.Li("Use 'Clear Troughs' to remove unwanted trough markers"),
                html.Li("Use 'Toggle Peak Visibility' if peaks become transparent or hard to see")
            ], style={'fontSize': '0.9em', 'margin': '5px 0'})
        ], style={'backgroundColor': '#f0f8ff', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'})
    ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd'}),
    
    # Interactive controls section - IMPROVED
    html.Div([
        html.H3("Interactive Peak Picking"),
        html.P("Use this graph for adding/removing peaks. Zoom and pan will be preserved during editing.", 
               style={'margin': '5px 0', 'fontSize': '0.9em', 'color': 'gray'}),
        
        # Interactive peak picking graph
        dcc.Graph(id='interactive-graph', style={'height': '700px', 'margin': '10px'}),
        
        html.Div([
            html.Label("Click Mode:"),
            dcc.RadioItems(
                id='interaction-mode',
                options=[
                    {'label': 'View Only (Pan/Zoom)', 'value': 'view'},
                    {'label': 'Add Shallow Peaks', 'value': 'add_shallow'},
                    {'label': 'Add Deep Peaks', 'value': 'add_deep'},
                    {'label': 'Remove ANY Peaks', 'value': 'remove'},
                    {'label': 'Exclude Time Range', 'value': 'exclude'},
                    {'label': 'Show All Local Maxima', 'value': 'show_all'}
                ],
                value='add_shallow',  # Default to adding peaks
                inline=True
            )
        ]),
        html.Div([
            html.P("📌 Click on the graph to:", style={'fontWeight': 'bold', 'margin': '5px 0'}),
            html.Ul([
                html.Li("Add Shallow/Deep Peaks: Click near temperature peaks"),
                html.Li("Remove Peaks: Click near any peak (detected or manual) to remove it"),
                html.Li("View Mode: Standard pan/zoom behavior (zoom state preserved)"),
                html.Li("Use 'Clear Troughs' button to remove all trough markers"),
                html.Li("Use 'Toggle Peak Visibility' if peaks become hard to see"),
                html.Li("Check Errors: Red ❌ markers show error locations on graph"),
                html.Li("Peak-Trough Method: Uses trough data for better amplitude calculation")
            ], style={'fontSize': '0.9em', 'margin': '5px 0'})
        ], style={'backgroundColor': '#f0f8ff', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'}),
        # Time range exclusion controls
        html.Div([
            html.Label("Exclude Time Range (only when mode is 'Exclude Time Range'):"),
            html.Div([
                dcc.Input(id='exclude-start', type='number', placeholder='Start Day', 
                         style={'width': '100px', 'marginRight': '10px'}),
                dcc.Input(id='exclude-end', type='number', placeholder='End Day', 
                         style={'width': '100px', 'marginRight': '10px'}),
                html.Button('Add Exclusion', id='add-exclusion-button', n_clicks=0),
                html.Button('Clear Exclusions', id='clear-exclusions-button', n_clicks=0, 
                           style={'marginLeft': '10px'})
            ], style={'marginTop': '5px'})
        ], id='exclusion-controls', style={'marginTop': '10px'}),
        html.Div(id='selection-info', style={'marginTop': '10px'})
    ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd'}),
    
    # Error display section - ENHANCED WITH VISUAL MARKERS INFO
    html.Div([
        html.H3("Error Analysis", style={'color': '#d32f2f'}),
        html.P("Visual error markers appear on the interactive graph above. Click 'Check Errors' to see detailed analysis.", 
               style={'margin': '5px 0', 'fontSize': '0.9em', 'color': 'gray'}),
        html.Div(id='error-display', style={'marginTop': '10px'})
    ], style={'margin': '10px', 'padding': '10px', 'border': '2px solid #d32f2f', 'borderRadius': '5px'}),
    
    # Export section with filename customization and YEAR INPUT
    html.Div([
        html.H3("Export Data"),
        html.Div([
            html.Label("Output Folder Name:"),
            dcc.Input(id='output-folder', type='text', 
                     value=initial_params['output_folder'], 
                     style={'width': '300px', 'marginLeft': '10px'}),
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Peak Data Filename:"),
            dcc.Input(id='peak-filename', type='text', placeholder='peak_picks.csv', 
                     value='peak_picks.csv', style={'width': '200px', 'marginLeft': '10px'}),
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Formatted Data Filename:"),
            dcc.Input(id='formatted-filename', type='text', placeholder='peak_picks_formatted.csv', 
                     value='peak_picks_formatted.csv', style={'width': '200px', 'marginLeft': '10px'}),
        ], style={'marginBottom': '10px'}),
        # NEW: Year input field
        html.Div([
            html.Label("Data Year for .dAf file:"),
            dcc.Input(id='data-year', type='number', 
                     value=datetime.now().year, 
                     min=1950, max=2150, step=1,
                     style={'width': '100px', 'marginLeft': '10px'}),
            html.Span(" (Used in formatted output file)", style={'fontSize': '0.9em', 'color': 'gray', 'marginLeft': '5px'})
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Button('Check Errors', id='check-errors-button', n_clicks=0,
                       style={'backgroundColor': '#ff9800', 'color': 'white'}),
            html.Button('Export Peak Data', id='export-button', n_clicks=0, style={'marginLeft': '10px'}),
            html.Button('Save Parameters', id='save-params-button', n_clicks=0, style={'marginLeft': '10px'}),
            html.Button('Load Parameters', id='load-params-button', n_clicks=0, style={'marginLeft': '10px'})
        ]),
        html.Div(id='export-status', style={'marginTop': '10px'})
    ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd'}),
    
    # Hidden storage divs for state management
    html.Div(id='stored-data', style={'display': 'none'}),
    html.Div(id='stored-peaks', style={'display': 'none'}),
    html.Div(id='manual-selections-store', style={'display': 'none'}, 
             children=json.dumps({'shallow_peaks': [], 'deep_peaks': [], 'excluded_ranges': []})),
    html.Div(id='excluded-ranges-store', style={'display': 'none'}),
    html.Div(id='current-filename', style={'display': 'none'}),
    html.Div(id='peak-visibility-state', style={'display': 'none'}, children='visible')
])

# ===========================
# CALLBACKS
# ===========================

@app.callback(
    [Output('stored-data', 'children'),
     Output('main-graph', 'figure'),
     Output('current-filename', 'children'),
     Output('current-file-display', 'children')],
    [Input('upload-data', 'contents'),
     Input('auto-load-csv-button', 'n_clicks')],
    [State('upload-data', 'filename')]
)
def upload_or_auto_load_file(contents, auto_load_clicks, filename):
    """
    Handle file upload or auto-load from .par file and create initial visualization.
    """
    triggered = ctx.triggered_id
    
    if triggered == 'auto-load-csv-button' and auto_load_clicks > 0:
        logger.user_action("Auto-load CSV from .par file")
        # Auto-load from .par file
        params = load_param_file()
        data_file = params.get('data_file', '')
        
        if data_file:
            # Try multiple paths: absolute, relative to script dir, and just filename
            possible_paths = [
                data_file,  # As specified in .par file
                SCRIPT_DIR / data_file,  # Relative to script directory
                SCRIPT_DIR / os.path.basename(data_file)  # Just filename in script dir
            ]
            
            file_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        df, success_msg, issues = load_and_validate_csv(str(path))
                        if df is not None:
                            filename = os.path.basename(str(path))
                            file_loaded = True
                            logger.info(f"Successfully auto-loaded file from: {path}", "DATA")
                            if issues:
                                logger.warning("Data validation issues found during auto-load", "DATA")
                                for issue in issues:
                                    logger.warning(f"  {issue}", "DATA_QUALITY")
                            break
                    except Exception as e:
                        logger.error(f"Error auto-loading {path}: {e}", "DATA")
                        continue
            
            if not file_loaded:
                error_msg = f"Could not find data file: {data_file}"
                logger.error(f"Auto-load failed: {error_msg}", "DATA")
                logger.info(f"Searched paths: {[str(p) for p in possible_paths]}", "DATA")
                return None, go.Figure().add_annotation(text=error_msg, x=0.5, y=0.5), None, \
                       html.Div(f"❌ {error_msg}", style={'color': 'red'})
        else:
            error_msg = "No data file specified in .par file"
            logger.error(error_msg, "DATA")
            return None, go.Figure().add_annotation(text=error_msg, x=0.5, y=0.5), None, \
                   html.Div(f"❌ {error_msg}", style={'color': 'red'})
    
    elif contents is not None:
        logger.user_action("Manual file upload", f"filename: {filename}")
        # Manual upload
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            # Use improved loading function
            df, success_msg, issues = load_and_validate_csv(decoded.decode('utf-8'), filename)
            if df is None:
                logger.error(f"Manual upload failed: {success_msg}", "DATA")
                return None, go.Figure().add_annotation(text=success_msg, x=0.5, y=0.5), None, \
                       html.Div(f"❌ {success_msg}", style={'color': 'red'})
        except Exception as e:
            error_msg = f"Error reading uploaded file: {e}"
            logger.error(error_msg, "DATA")
            return None, go.Figure().add_annotation(text=error_msg, x=0.5, y=0.5), None, \
                   html.Div(f"❌ {error_msg}", style={'color': 'red'})
    else:
        return None, go.Figure(), None, html.Div("No file loaded")
    
    # Create initial plot (READ-ONLY VIEW)
    fig = go.Figure()
    
    # Get colors from parameters
    params = load_param_file()
    
    # Add shallow temperature trace
    fig.add_trace(go.Scatter(
        x=df['WaterDay'],
        y=df['Shallow.Temp.Filt'],
        mode='lines',
        name='Shallow',
        line=dict(color=params['shallow_color'], width=params['line_width'])
    ))
    
    # Add deep temperature trace
    fig.add_trace(go.Scatter(
        x=df['WaterDay'],
        y=df['Deep.Temp.Filt'],
        mode='lines',
        name='Deep',
        line=dict(color=params['deep_color'], width=params['line_width'])
    ))
    
    # Configure plot layout (NO CLICKING - just viewing)
    fig.update_layout(
        title=f'Data Verification View: {filename} (Read-Only)',
        xaxis_title='Water Day',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        dragmode='zoom',  # Only allow zooming
        selectdirection='d',  # 'd' for diagonal (not 'diagonal')
        height=400  # Smaller height since it's just for verification
    )
    
    # Disable trace selection behavior that causes transparency issues
    fig.update_traces(selectedpoints=[], unselected={'marker': {'opacity': 1.0}})
    
    # Calculate data statistics for display
    data_points = len(df)
    time_span = df['WaterDay'].max() - df['WaterDay'].min()
    sampling_rate = data_points / time_span if time_span > 0 else 0
    
    # Create file info display
    info_elements = [
        html.H4(f"📁 Current File: {filename}", style={'color': 'green', 'margin': '5px 0'}),
        html.P(f"Data points: {data_points:,} | Time span: {time_span:.1f} days | "
               f"Sampling rate: ~{sampling_rate:.1f} points/day", 
               style={'margin': '5px 0', 'fontSize': '0.9em'})
    ]
    
    # Add data quality issues if any
    if 'issues' in locals() and issues:
        info_elements.append(html.Div([
            html.H5("Data Quality Issues:", style={'color': 'orange', 'margin': '5px 0'}),
            html.Ul([html.Li(issue) for issue in issues], 
                   style={'fontSize': '0.8em', 'margin': '0'})
        ]))
    
    file_info = html.Div(info_elements)
    
    # Store data as JSON and filename
    return df.to_json(date_format='iso', orient='split'), fig, filename, file_info

# Add callback to populate interactive graph when data is loaded
@app.callback(
    Output('interactive-graph', 'figure'),
    Input('stored-data', 'children'),
    prevent_initial_call=True
)
def populate_interactive_graph(stored_data):
    """
    Populate the interactive graph when data is first loaded.
    """
    if stored_data is None:
        return go.Figure()
    
    # Load data
    df = pd.read_json(StringIO(stored_data), orient='split')
    
    # Extract data arrays
    water_day = df['WaterDay'].values
    shallow_temp = df['Shallow.Temp.Filt'].values
    deep_temp = df['Deep.Temp.Filt'].values
    
    # Create interactive plot
    fig = go.Figure()
    
    # Get current params for colors
    current_params = load_param_file()
    
    # Add temperature traces
    fig.add_trace(go.Scatter(
        x=water_day, y=shallow_temp,
        mode='lines', name='Shallow',
        line=dict(color=current_params['shallow_color'], width=current_params['line_width'])
    ))
    
    fig.add_trace(go.Scatter(
        x=water_day, y=deep_temp,
        mode='lines', name='Deep',
        line=dict(color=current_params['deep_color'], width=current_params['line_width'])
    ))
    
    # Configure plot layout
    fig.update_layout(
        title='Interactive Peak Picking - Click to Add/Remove Peaks',
        xaxis_title='Water Day',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        clickmode='event+select',
        selectdirection='d',  # 'd' for diagonal selection
        dragmode='zoom',  # Default to zoom mode
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=700
    )
    
    # Disable trace selection behavior that causes transparency issues
    fig.update_traces(selectedpoints=[], unselected={'marker': {'opacity': 1.0}})
    
    logger.info("Interactive graph populated with temperature data", "UI")
    
    return fig

@app.callback(
    [Output('stored-peaks', 'children'),
     Output('interactive-graph', 'figure', allow_duplicate=True),
     Output('manual-selections-store', 'children')],
    [Input('detect-button', 'n_clicks'),
     Input('interactive-graph', 'clickData'),
     Input('add-exclusion-button', 'n_clicks'),
     Input('clear-exclusions-button', 'n_clicks'),
     Input('clear-troughs-button', 'n_clicks')],
    [State('stored-data', 'children'),
     State('peak-method-dropdown', 'value'),
     State('target-period', 'value'),
     State('search-tolerance', 'value'),
     State('slope-threshold', 'value'),
     State('min-distance', 'value'),
     State('prominence-factor', 'value'),
     State('interaction-mode', 'value'),
     State('manual-selections-store', 'children'),
     State('exclude-start', 'value'),
     State('exclude-end', 'value'),
     State('sensor-spacing', 'value'),
     State('interactive-graph', 'figure'),
     State('peak-visibility-state', 'children')],
    prevent_initial_call=True
)
def detect_peaks_and_interact(n_clicks, click_data, add_exclusion_clicks, 
                            clear_exclusions_clicks, clear_troughs_clicks, stored_data, method, target_period, 
                            tolerance, slope_threshold, min_distance, prominence_factor, 
                            interaction_mode, manual_store, exclude_start, exclude_end, 
                            sensor_spacing, current_figure, visibility_state):
    """
    Enhanced callback for peak detection and interaction with zoom state preservation and trough management.
    """
    if stored_data is None:
        return None, go.Figure(), json.dumps({'shallow_peaks': [], 'deep_peaks': [], 'excluded_ranges': []})
    
    # Ensure visibility_state is properly set and define related variables
    if visibility_state is None:
        visibility_state = 'visible'
    
    # Define peak visibility variables based on state
    if visibility_state == 'visible':
        peak_opacity = 1.0
        peak_visible = True
    else:
        peak_opacity = 0.3
        peak_visible = True  # Keep visible but with reduced opacity
    
    # Load data
    df = pd.read_json(StringIO(stored_data), orient='split')
    
    # Extract data arrays
    water_day = df['WaterDay'].values
    shallow_temp = df['Shallow.Temp.Filt'].values
    deep_temp = df['Deep.Temp.Filt'].values
    
    # Preserve current zoom state
    current_xaxis_range = None
    current_yaxis_range = None
    if current_figure and 'layout' in current_figure:
        if 'xaxis' in current_figure['layout'] and 'range' in current_figure['layout']['xaxis']:
            current_xaxis_range = current_figure['layout']['xaxis']['range']
        if 'yaxis' in current_figure['layout'] and 'range' in current_figure['layout']['yaxis']:
            current_yaxis_range = current_figure['layout']['yaxis']['range']
    
    # Handle manual selections - ensure proper initialization
    if manual_store:
        try:
            manual_selections = json.loads(manual_store)
            # Ensure all required keys exist
            if 'shallow_peaks' not in manual_selections:
                manual_selections['shallow_peaks'] = []
            if 'deep_peaks' not in manual_selections:
                manual_selections['deep_peaks'] = []
            if 'excluded_ranges' not in manual_selections:
                manual_selections['excluded_ranges'] = []
        except:
            manual_selections = {'shallow_peaks': [], 'deep_peaks': [], 'excluded_ranges': []}
    else:
        manual_selections = {'shallow_peaks': [], 'deep_peaks': [], 'excluded_ranges': []}
    
    # Keep track of all detected peaks for removal functionality
    detected_peaks = {'shallow_peaks': [], 'deep_peaks': []}
    
    # Initialize force_clear_troughs flag
    force_clear_troughs = False
    
    # Determine which input triggered the callback
    triggered = ctx.triggered_id
    
    # Log user actions
    if triggered == 'detect-button':
        logger.user_action(f"Peak detection using {method} method")
    elif triggered == 'interactive-graph':
        logger.user_action(f"Graph interaction in {interaction_mode} mode")
    elif triggered == 'clear-troughs-button':
        logger.user_action("Clear all troughs")
    elif triggered == 'add-exclusion-button':
        logger.user_action(f"Add time exclusion: {exclude_start}-{exclude_end}")
    elif triggered == 'clear-exclusions-button':
        logger.user_action("Clear all time exclusions")
    
    # Handle exclusion range additions
    if triggered == 'add-exclusion-button' and exclude_start is not None and exclude_end is not None:
        manual_selections['excluded_ranges'].append([exclude_start, exclude_end])
        logger.info(f"Added exclusion range: {exclude_start:.2f} to {exclude_end:.2f} days", "USER")
    
    # Handle clearing exclusions
    if triggered == 'clear-exclusions-button':
        manual_selections['excluded_ranges'] = []
        logger.info("Cleared all time exclusions", "USER")
    
    # Handle clearing troughs
    if triggered == 'clear-troughs-button':
        # Force troughs to be empty by setting a flag
        force_clear_troughs = True
        logger.info("Clearing all trough markers", "USER")
    else:
        force_clear_troughs = False
    
    # Handle click interactions for manual peak selection/removal
    if triggered == 'interactive-graph' and click_data and interaction_mode != 'view':
        click_point = click_data['points'][0]
        click_x = click_point['x']
        
        # Find nearest data point
        nearest_idx = int(np.argmin(np.abs(water_day - click_x)))
        
        if interaction_mode == 'add_shallow':
            if nearest_idx not in manual_selections['shallow_peaks']:
                manual_selections['shallow_peaks'].append(nearest_idx)
                logger.info(f"Added manual shallow peak at index {nearest_idx}, day {water_day[nearest_idx]:.2f}", "USER")
            else:
                logger.warning(f"Shallow peak already exists at index {nearest_idx}", "USER")
                
        elif interaction_mode == 'add_deep':
            if nearest_idx not in manual_selections['deep_peaks']:
                manual_selections['deep_peaks'].append(nearest_idx)
                logger.info(f"Added manual deep peak at index {nearest_idx}, day {water_day[nearest_idx]:.2f}", "USER")
            else:
                logger.warning(f"Deep peak already exists at index {nearest_idx}", "USER")
                
        elif interaction_mode == 'remove':
            # ENHANCED REMOVAL: Remove from ANY peaks (manual or detected)
            removed = False
            
            # First check manual selections
            if nearest_idx in manual_selections['shallow_peaks']:
                manual_selections['shallow_peaks'].remove(nearest_idx)
                logger.info(f"Removed manual shallow peak at index {nearest_idx}", "USER")
                removed = True
            
            if nearest_idx in manual_selections['deep_peaks']:
                manual_selections['deep_peaks'].remove(nearest_idx)
                logger.info(f"Removed manual deep peak at index {nearest_idx}", "USER")
                removed = True
            
            if not removed:
                # If not found in manual, we need to get detected peaks and store exclusions
                # This requires running detection first to see what would be detected
                logger.info(f"Peak at index {nearest_idx} not found in manual selections", "USER")
                # We'll handle this by detecting peaks first, then checking for removal
    
    # Detect peaks based on selected method (skip if only manual and no detection button pressed)
    shallow_troughs = np.array([], dtype=int)
    deep_troughs = np.array([], dtype=int)
    
    if method != 'manual' or triggered == 'detect-button':
        
        # Log peak detection parameters
        detection_params = {
            'method': method,
            'target_period': target_period,
            'tolerance': tolerance,
            'prominence_factor': prominence_factor,
            'min_distance': min_distance
        }
        
        if method == 'scipy':
            # Estimate distance parameter based on target period
            samples_per_day = len(water_day) / (water_day[-1] - water_day[0])
            min_distance_samples = int(target_period * samples_per_day * 0.5)
            
            # Calculate prominence based on data std and user factor
            if prominence_factor is None:
                prominence_factor = 0.3
            
            # For peak detection, use centered data for better results
            shallow_centered = shallow_temp - np.mean(shallow_temp)
            deep_centered = deep_temp - np.mean(deep_temp)
            
            # Calculate prominence
            shallow_prominence = np.std(shallow_centered) * prominence_factor
            deep_prominence = np.std(deep_centered) * prominence_factor
            
            # Detect peaks using improved method
            detected_peaks['shallow_peaks'] = scipy_find_peaks_method_improved(shallow_temp, 
                                                  distance=max(min_distance_samples, min_distance),
                                                  prominence=shallow_prominence,
                                                  height=None)
            detected_peaks['deep_peaks'] = scipy_find_peaks_method_improved(deep_temp, 
                                               distance=max(min_distance_samples, min_distance),
                                               prominence=deep_prominence,
                                               height=None)
            
            # Find troughs by inverting the signal
            shallow_troughs = scipy_find_peaks_method_improved(-shallow_temp, 
                                                    distance=max(min_distance_samples, min_distance),
                                                    prominence=shallow_prominence,
                                                    height=None)
            deep_troughs = scipy_find_peaks_method_improved(-deep_temp, 
                                                 distance=max(min_distance_samples, min_distance),
                                                 prominence=deep_prominence,
                                                 height=None)
            
        elif method == 'wavelet':
            detected_peaks['shallow_peaks'] = wavelet_peak_detection(shallow_temp)
            detected_peaks['deep_peaks'] = wavelet_peak_detection(deep_temp)
            shallow_troughs = wavelet_peak_detection(-shallow_temp)
            deep_troughs = wavelet_peak_detection(-deep_temp)
            
        elif method == 'derivative':
            detected_peaks['shallow_peaks'], shallow_troughs = derivative_peak_detection(
                shallow_temp, water_day, slope_threshold, min_distance)
            detected_peaks['deep_peaks'], deep_troughs = derivative_peak_detection(
                deep_temp, water_day, slope_threshold, min_distance)
            
        elif method == 'prominence':
            detected_peaks['shallow_peaks'] = prominence_based_detection(shallow_temp, prominence_factor)
            detected_peaks['deep_peaks'] = prominence_based_detection(deep_temp, prominence_factor)
            shallow_troughs = prominence_based_detection(-shallow_temp, prominence_factor)
            deep_troughs = prominence_based_detection(-deep_temp, prominence_factor)
            
        elif method == 'combined':
            # Use combined method for maximum sensitivity
            samples_per_day = len(water_day) / (water_day[-1] - water_day[0])
            min_distance_samples = int(target_period * samples_per_day * 0.5)
            
            detected_peaks['shallow_peaks'] = combined_peak_detection(
                shallow_temp, water_day, target_period, 
                max(min_distance_samples, min_distance))
            detected_peaks['deep_peaks'] = combined_peak_detection(
                deep_temp, water_day, target_period, 
                max(min_distance_samples, min_distance))
            
            # For troughs, invert the signal
            shallow_troughs = combined_peak_detection(
                -shallow_temp, water_day, target_period, 
                max(min_distance_samples, min_distance))
            deep_troughs = combined_peak_detection(
                -deep_temp, water_day, target_period, 
                max(min_distance_samples, min_distance))
            
        elif method == 'bootstrap':
            # Use manual selections as seeds
            if len(manual_selections['shallow_peaks']) >= 2:
                detected_peaks['shallow_peaks'] = bootstrap_peak_detection_improved(
                    shallow_temp, water_day, 
                    np.array(manual_selections['shallow_peaks'], dtype=int), 
                    target_period, tolerance)
            else:
                # If not enough manual peaks, just use what we have
                detected_peaks['shallow_peaks'] = np.array(manual_selections['shallow_peaks'], dtype=int)
                
            if len(manual_selections['deep_peaks']) >= 2:
                detected_peaks['deep_peaks'] = bootstrap_peak_detection_improved(
                    deep_temp, water_day, 
                    np.array(manual_selections['deep_peaks'], dtype=int), 
                    target_period, tolerance)
            else:
                # If not enough manual peaks, just use what we have
                detected_peaks['deep_peaks'] = np.array(manual_selections['deep_peaks'], dtype=int)
            
            # For bootstrap mode, we don't detect troughs automatically
            shallow_troughs = np.array([], dtype=int)
            deep_troughs = np.array([], dtype=int)
        
        # Log peak detection results
        logger.peak_detection(
            method, 
            len(detected_peaks['shallow_peaks']), 
            len(detected_peaks['deep_peaks']), 
            detection_params
        )
    
    # Handle forced trough clearing
    if force_clear_troughs:
        shallow_troughs = np.array([], dtype=int)
        deep_troughs = np.array([], dtype=int)
        logger.info("Troughs cleared by user request", "USER")
    
    # Handle removal of detected peaks if in remove mode and click happened
    if triggered == 'interactive-graph' and interaction_mode == 'remove' and click_data:
        click_point = click_data['points'][0]
        click_x = click_point['x']
        nearest_idx = int(np.argmin(np.abs(water_day - click_x)))
        
        # Check if the clicked point is near any detected peaks
        tolerance_samples = 10  # Allow clicking within 10 samples of peak
        
        # Check detected shallow peaks
        if len(detected_peaks['shallow_peaks']) > 0:
            distances = np.abs(detected_peaks['shallow_peaks'] - nearest_idx)
            if np.any(distances <= tolerance_samples):
                closest_peak_idx = detected_peaks['shallow_peaks'][np.argmin(distances)]
                # Add to exclusion list (we'll filter out later)
                if 'excluded_detected_peaks' not in manual_selections:
                    manual_selections['excluded_detected_peaks'] = {'shallow': [], 'deep': []}
                if closest_peak_idx not in manual_selections['excluded_detected_peaks']['shallow']:
                    manual_selections['excluded_detected_peaks']['shallow'].append(int(closest_peak_idx))
                    logger.info(f"Excluded detected shallow peak at index {closest_peak_idx}", "USER")
        
        # Check detected deep peaks
        if len(detected_peaks['deep_peaks']) > 0:
            distances = np.abs(detected_peaks['deep_peaks'] - nearest_idx)
            if np.any(distances <= tolerance_samples):
                closest_peak_idx = detected_peaks['deep_peaks'][np.argmin(distances)]
                # Add to exclusion list
                if 'excluded_detected_peaks' not in manual_selections:
                    manual_selections['excluded_detected_peaks'] = {'shallow': [], 'deep': []}
                if closest_peak_idx not in manual_selections['excluded_detected_peaks']['deep']:
                    manual_selections['excluded_detected_peaks']['deep'].append(int(closest_peak_idx))
                    logger.info(f"Excluded detected deep peak at index {closest_peak_idx}", "USER")
    
    # Combine detected and manual peaks, excluding removed detected peaks
    if method == 'manual':
        # Manual mode: only use manually selected peaks
        shallow_peaks = np.array(manual_selections['shallow_peaks'], dtype=int)
        deep_peaks = np.array(manual_selections['deep_peaks'], dtype=int)
    else:
        # Other methods: combine detected and manual, excluding removed detected peaks
        all_shallow = list(detected_peaks['shallow_peaks']) + manual_selections['shallow_peaks']
        all_deep = list(detected_peaks['deep_peaks']) + manual_selections['deep_peaks']
        
        # Remove excluded detected peaks
        if 'excluded_detected_peaks' in manual_selections:
            all_shallow = [p for p in all_shallow if p not in manual_selections['excluded_detected_peaks'].get('shallow', [])]
            all_deep = [p for p in all_deep if p not in manual_selections['excluded_detected_peaks'].get('deep', [])]
        
        shallow_peaks = np.unique(np.array(all_shallow, dtype=int)) if len(all_shallow) > 0 else np.array([], dtype=int)
        deep_peaks = np.unique(np.array(all_deep, dtype=int)) if len(all_deep) > 0 else np.array([], dtype=int)
    
    # Apply exclusion ranges - remove peaks within excluded time ranges
    if len(manual_selections.get('excluded_ranges', [])) > 0:
        for exclusion in manual_selections['excluded_ranges']:
            start_day, end_day = exclusion
            # Remove shallow peaks in excluded range
            if len(shallow_peaks) > 0:
                shallow_mask = (water_day[shallow_peaks] < start_day) | (water_day[shallow_peaks] > end_day)
                shallow_peaks = shallow_peaks[shallow_mask]
            # Remove deep peaks in excluded range
            if len(deep_peaks) > 0:
                deep_mask = (water_day[deep_peaks] < start_day) | (water_day[deep_peaks] > end_day)
                deep_peaks = deep_peaks[deep_mask]
            # Remove shallow troughs in excluded range
            if len(shallow_troughs) > 0:
                shallow_trough_mask = (water_day[shallow_troughs] < start_day) | (water_day[shallow_troughs] > end_day)
                shallow_troughs = shallow_troughs[shallow_trough_mask]
            # Remove deep troughs in excluded range
            if len(deep_troughs) > 0:
                deep_trough_mask = (water_day[deep_troughs] < start_day) | (water_day[deep_troughs] > end_day)
                deep_troughs = deep_troughs[deep_trough_mask]
    
    # Ensure all peak indices are integers and valid
    if isinstance(shallow_peaks, (list, np.ndarray)) and len(shallow_peaks) > 0:
        shallow_peaks = np.array([int(p) for p in shallow_peaks if 0 <= p < len(water_day)], dtype=int)
    else:
        shallow_peaks = np.array([], dtype=int)
        
    if isinstance(deep_peaks, (list, np.ndarray)) and len(deep_peaks) > 0:
        deep_peaks = np.array([int(p) for p in deep_peaks if 0 <= p < len(water_day)], dtype=int)
    else:
        deep_peaks = np.array([], dtype=int)
        
    if isinstance(shallow_troughs, (list, np.ndarray)) and len(shallow_troughs) > 0:
        shallow_troughs = np.array([int(t) for t in shallow_troughs if 0 <= t < len(water_day)], dtype=int)
    else:
        shallow_troughs = np.array([], dtype=int)
        
    if isinstance(deep_troughs, (list, np.ndarray)) and len(deep_troughs) > 0:
        deep_troughs = np.array([int(t) for t in deep_troughs if 0 <= t < len(water_day)], dtype=int)
    else:
        deep_troughs = np.array([], dtype=int)
    
    # Create updated plot
    fig = go.Figure()
    
    # Get current params for colors
    current_params = load_param_file()
    
    # Add temperature traces
    fig.add_trace(go.Scatter(
        x=water_day, y=shallow_temp,
        mode='lines', name='Shallow',
        line=dict(color=current_params['shallow_color'], width=current_params['line_width'])
    ))
    
    fig.add_trace(go.Scatter(
        x=water_day, y=deep_temp,
        mode='lines', name='Deep',
        line=dict(color=current_params['deep_color'], width=current_params['line_width'])
    ))
    
    # Add peaks with controlled visibility
    if len(shallow_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[shallow_peaks], y=shallow_temp[shallow_peaks],
            mode='markers', name=f'Shallow Peaks ({len(shallow_peaks)})',
            marker=dict(color='darkblue', size=current_params['peak_size'], 
                       symbol='circle-open', line=dict(width=2)),
            opacity=peak_opacity,
            visible=peak_visible
        ))
    
    if len(deep_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[deep_peaks], y=deep_temp[deep_peaks],
            mode='markers', name=f'Deep Peaks ({len(deep_peaks)})',
            marker=dict(color='darkred', size=current_params['peak_size'], 
                       symbol='circle-open', line=dict(width=2)),
            opacity=peak_opacity,
            visible=peak_visible
        ))
    
    # Add troughs with controlled visibility
    if len(shallow_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[shallow_troughs], y=shallow_temp[shallow_troughs],
            mode='markers', name=f'Shallow Troughs ({len(shallow_troughs)})',
            marker=dict(color='navy', size=current_params['trough_size'], 
                       symbol='square-open', line=dict(width=2)),
            opacity=peak_opacity,
            visible=peak_visible
        ))
    
    if len(deep_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[deep_troughs], y=deep_temp[deep_troughs],
            mode='markers', name=f'Deep Troughs ({len(deep_troughs)})',
            marker=dict(color='maroon', size=current_params['trough_size'], 
                       symbol='square-open', line=dict(width=2)),
            opacity=peak_opacity,
            visible=peak_visible
        ))
    
    # Add manual selections with different symbols (show them separately if method is not manual)
    if method != 'manual':
        manual_shallow_indices = [int(idx) for idx in manual_selections['shallow_peaks'] if idx < len(water_day)]
        manual_deep_indices = [int(idx) for idx in manual_selections['deep_peaks'] if idx < len(water_day)]
        
        if len(manual_shallow_indices) > 0:
            fig.add_trace(go.Scatter(
                x=water_day[manual_shallow_indices], 
                y=shallow_temp[manual_shallow_indices],
                mode='markers', 
                name=f'Manual Shallow ({len(manual_shallow_indices)})',
                marker=dict(color='cyan', size=14, symbol='star', line=dict(width=2, color='darkblue')),
                opacity=peak_opacity,
                visible=peak_visible
            ))
        
        if len(manual_deep_indices) > 0:
            fig.add_trace(go.Scatter(
                x=water_day[manual_deep_indices], 
                y=deep_temp[manual_deep_indices],
                mode='markers', 
                name=f'Manual Deep ({len(manual_deep_indices)})',
                marker=dict(color='orange', size=14, symbol='star', line=dict(width=2, color='darkred')),
                opacity=peak_opacity,
                visible=peak_visible
            ))
    
    # Add excluded ranges as shaded regions
    for exclusion in manual_selections.get('excluded_ranges', []):
        start_day, end_day = exclusion
        fig.add_vrect(
            x0=start_day, x1=end_day,
            fillcolor="gray", opacity=0.3,
            layer="below", line_width=0,
            annotation_text=f"Excluded: {start_day:.1f}-{end_day:.1f}",
            annotation_position="top left"
        )
        
    # Update layout with preserved zoom state and disabled selection
    fig.update_layout(
        title='Interactive Peak Picking - Click to Add/Remove Peaks',
        xaxis_title='Water Day',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        clickmode='event+select',
        selectdirection='d',  # 'd' for diagonal selection
        dragmode='zoom',  # Default to zoom mode
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=700
    )
    
    # Disable trace selection behavior that causes transparency issues
    fig.update_traces(selectedpoints=[], unselected={'marker': {'opacity': 1.0}})
    
    # Add a subtitle showing the current mode
    mode_text = {
        'view': 'View Only Mode - Pan/Zoom enabled, zoom state preserved',
        'add_shallow': 'Click to Add Shallow Peaks (Blue)',
        'add_deep': 'Click to Add Deep Peaks (Red)',
        'remove': 'Click to Remove ANY Peaks (Detected or Manual)',
        'exclude': 'Use controls below to exclude time ranges',
        'show_all': 'Showing all local maxima/minima as gray dots'
    }
    if interaction_mode in mode_text:
        fig.add_annotation(
            text=f"Mode: {mode_text[interaction_mode]}",
            xref="paper", yref="paper",
            x=0.5, y=1.05, showarrow=False,
            font=dict(size=14, color="green" if interaction_mode != 'view' else "black")
        )
    
    # Show all local maxima/minima if requested
    if interaction_mode == 'show_all':
        # Find all local maxima in shallow data
        all_shallow_max = []
        for i in range(1, len(shallow_temp) - 1):
            if shallow_temp[i] > shallow_temp[i-1] and shallow_temp[i] > shallow_temp[i+1]:
                all_shallow_max.append(i)
        
        # Find all local maxima in deep data
        all_deep_max = []
        for i in range(1, len(deep_temp) - 1):
            if deep_temp[i] > deep_temp[i-1] and deep_temp[i] > deep_temp[i+1]:
                all_deep_max.append(i)
        
        # Add as small gray dots with controlled visibility
        if len(all_shallow_max) > 0:
            maxima_opacity = 0.7 if visibility_state == 'visible' else 0.2
            fig.add_trace(go.Scatter(
                x=water_day[all_shallow_max], 
                y=shallow_temp[all_shallow_max],
                mode='markers', 
                name=f'All Shallow Maxima ({len(all_shallow_max)})',
                marker=dict(color='gray', size=10, symbol='circle'),
                opacity=maxima_opacity,
                visible=peak_visible
            ))
        
        if len(all_deep_max) > 0:
            maxima_opacity = 0.7 if visibility_state == 'visible' else 0.2
            fig.add_trace(go.Scatter(
                x=water_day[all_deep_max], 
                y=deep_temp[all_deep_max],
                mode='markers', 
                name=f'All Deep Maxima ({len(all_deep_max)})',
                marker=dict(color='darkgray', size=10, symbol='circle'),
                opacity=maxima_opacity,
                visible=peak_visible
            ))
    
    # PRESERVE ZOOM STATE
    if current_xaxis_range:
        fig.update_layout(xaxis=dict(range=current_xaxis_range))
    if current_yaxis_range:
        fig.update_layout(yaxis=dict(range=current_yaxis_range))
    
    # Store peaks data
    peaks_data = {
        'shallow_peaks': [int(p) for p in shallow_peaks],  # Convert to regular Python int
        'deep_peaks': [int(p) for p in deep_peaks],
        'shallow_troughs': [int(t) for t in shallow_troughs],
        'deep_troughs': [int(t) for t in deep_troughs]
    }
    
    # Enhanced logging summary
    logger.info("=== Peak Detection Summary ===", "PEAKS")
    logger.info(f"Method: {method}", "PEAKS")
    logger.info(f"Total shallow peaks: {len(peaks_data['shallow_peaks'])} (including {len(manual_selections['shallow_peaks'])} manual)", "PEAKS")
    logger.info(f"Total deep peaks: {len(peaks_data['deep_peaks'])} (including {len(manual_selections['deep_peaks'])} manual)", "PEAKS")
    logger.info(f"Shallow troughs: {len(peaks_data['shallow_troughs'])}", "PEAKS")
    logger.info(f"Deep troughs: {len(peaks_data['deep_troughs'])}", "PEAKS")
    logger.info(f"Excluded ranges: {len(manual_selections.get('excluded_ranges', []))}", "PEAKS")
    logger.info(f"Zoom preserved: x={current_xaxis_range is not None}, y={current_yaxis_range is not None}", "UI")
    logger.info(f"Visibility state: {visibility_state}", "UI")
    logger.info("==============================", "PEAKS")
    
    # Ensure manual selections are also regular ints
    manual_selections_clean = {
        'shallow_peaks': [int(p) for p in manual_selections['shallow_peaks']],
        'deep_peaks': [int(p) for p in manual_selections['deep_peaks']],
        'excluded_ranges': manual_selections.get('excluded_ranges', []),
        'excluded_detected_peaks': manual_selections.get('excluded_detected_peaks', {'shallow': [], 'deep': []})
    }
    
    return json.dumps(peaks_data), fig, json.dumps(manual_selections_clean)

@app.callback(
    Output('exclusion-controls', 'style'),
    Input('interaction-mode', 'value')
)
def toggle_exclusion_controls(mode):
    """Show/hide time range exclusion controls based on interaction mode."""
    if mode == 'exclude':
        return {'marginTop': '10px', 'display': 'block'}
    else:
        return {'marginTop': '10px', 'display': 'none'}

@app.callback(
    Output('selection-info', 'children'),
    Input('manual-selections-store', 'children')
)
def update_selection_info(manual_store):
    """Update display of manual selection counts."""
    if manual_store:
        selections = json.loads(manual_store)
        info_items = [
            html.P(f"Manual Shallow Peaks: {len(selections['shallow_peaks'])}"),
            html.P(f"Manual Deep Peaks: {len(selections['deep_peaks'])}"),
            html.P(f"Excluded Time Ranges: {len(selections.get('excluded_ranges', []))}"),
        ]
        
        # Show excluded detected peaks if any
        excluded_detected = selections.get('excluded_detected_peaks', {'shallow': [], 'deep': []})
        if excluded_detected['shallow'] or excluded_detected['deep']:
            info_items.append(html.P(f"Excluded Detected Peaks: {len(excluded_detected['shallow'])} shallow, {len(excluded_detected['deep'])} deep", 
                                   style={'color': 'orange'}))
        
        # List excluded ranges
        if selections.get('excluded_ranges'):
            for i, (start, end) in enumerate(selections['excluded_ranges']):
                info_items.append(html.P(f"  Range {i+1}: Day {start:.1f} - {end:.1f}", 
                                       style={'marginLeft': '20px', 'fontSize': '0.9em'}))
        
        return html.Div(info_items)
    return "No manual selections"

# ===========================
# ENHANCED ERROR CHECKING CALLBACK WITH VISUAL MARKERS AND PEAK-TROUGH SUPPORT
# ===========================

@app.callback(
    [Output('error-display', 'children'),
     Output('interactive-graph', 'figure', allow_duplicate=True)],
    Input('check-errors-button', 'n_clicks'),
    [State('stored-data', 'children'),
     State('stored-peaks', 'children'),
     State('interactive-graph', 'figure'),
     State('sensor-spacing', 'value'),
     State('ar-tolerance', 'value'),
     State('phase-max-limit', 'value'),
     State('phase-min-limit', 'value'),
     State('amplitude-method', 'value')],
    prevent_initial_call=True
)
def check_errors_with_visual_markers_enhanced(n_clicks, stored_data, stored_peaks, current_fig, sensor_spacing,
                                            ar_tolerance, phase_max_limit, phase_min_limit, amplitude_method):
    """
    Enhanced error checking with visual error markers on the graph and peak-trough amplitude support.
    """
    if n_clicks == 0 or stored_data is None or stored_peaks is None:
        return '', current_fig if current_fig else go.Figure()
    
    logger.info("=== ERROR CHECKING WITH VISUAL MARKERS STARTED ===", "ERRORS")
    logger.user_action("Error checking", f"using {amplitude_method} amplitude method")
    
    # Load data
    df = pd.read_json(StringIO(stored_data), orient='split')
    peaks_data = json.loads(stored_peaks)
    
    water_day = df['WaterDay'].values
    shallow_temp = df['Shallow.Temp.Filt'].values
    deep_temp = df['Deep.Temp.Filt'].values
    
    # Extract peak indices (these should include both detected AND manual peaks)
    shallow_peak_indices = np.array(peaks_data.get('shallow_peaks', []), dtype=int)
    deep_peak_indices = np.array(peaks_data.get('deep_peaks', []), dtype=int)
    shallow_trough_indices = np.array(peaks_data.get('shallow_troughs', []), dtype=int)
    deep_trough_indices = np.array(peaks_data.get('deep_troughs', []), dtype=int)
    
    logger.info(f"Data has {len(water_day)} points, time range {water_day[0]:.2f} to {water_day[-1]:.2f} days", "ERRORS")
    logger.info(f"Found {len(shallow_peak_indices)} shallow peaks, {len(deep_peak_indices)} deep peaks", "ERRORS")
    logger.info(f"Found {len(shallow_trough_indices)} shallow troughs, {len(deep_trough_indices)} deep troughs", "ERRORS")
    logger.info(f"Using amplitude method: {amplitude_method}", "ERRORS")
    
    # Validate peak indices
    valid_shallow = shallow_peak_indices[(shallow_peak_indices >= 0) & (shallow_peak_indices < len(water_day))]
    valid_deep = deep_peak_indices[(deep_peak_indices >= 0) & (deep_peak_indices < len(water_day))]
    valid_shallow_troughs = shallow_trough_indices[(shallow_trough_indices >= 0) & (shallow_trough_indices < len(water_day))]
    valid_deep_troughs = deep_trough_indices[(deep_trough_indices >= 0) & (deep_trough_indices < len(water_day))]
    
    if len(valid_shallow) != len(shallow_peak_indices):
        logger.warning(f"Removed {len(shallow_peak_indices) - len(valid_shallow)} invalid shallow peak indices", "ERRORS")
    if len(valid_deep) != len(deep_peak_indices):
        logger.warning(f"Removed {len(deep_peak_indices) - len(valid_deep)} invalid deep peak indices", "ERRORS")
    
    shallow_peak_indices = valid_shallow
    deep_peak_indices = valid_deep
    shallow_trough_indices = valid_shallow_troughs
    deep_trough_indices = valid_deep_troughs
    
    if len(shallow_peak_indices) > 0:
        shallow_peak_times = water_day[shallow_peak_indices]
        logger.debug(f"Shallow peaks at days: {shallow_peak_times[:5]}", "ERRORS")  # Show first 5
    if len(deep_peak_indices) > 0:
        deep_peak_times = water_day[deep_peak_indices]
        logger.debug(f"Deep peaks at days: {deep_peak_times[:5]}", "ERRORS")  # Show first 5
    
    # Run error checks with enhanced functions
    errors = []
    
    # Check 1: Mismatched peaks
    logger.info("--- Checking for mismatched peaks ---", "ERRORS")
    if len(shallow_peak_indices) > 0 or len(deep_peak_indices) > 0:
        mismatch_errors = check_mismatched_peaks(shallow_peak_indices, deep_peak_indices, water_day, tolerance=0.5)
        errors.extend(mismatch_errors)
    
    # Check 2: Alternation
    logger.info("--- Checking peak alternation ---", "ERRORS")
    if len(shallow_peak_indices) > 0 and len(deep_peak_indices) > 0:
        shallow_times = water_day[shallow_peak_indices]
        deep_times = water_day[deep_peak_indices]
        alternation_errors = check_alternation(shallow_times, deep_times)
        errors.extend(alternation_errors)
    
    # Check 3: Enhanced amplitude and phase with user-defined limits and method selection
    logger.info("--- Checking amplitude ratios and phase shifts ---", "ERRORS")
    logger.info(f"Using limits - Ar tolerance: {ar_tolerance}, Phase: {phase_min_limit}-{phase_max_limit} days", "ERRORS")
    if len(shallow_peak_indices) > 0 and len(deep_peak_indices) > 0:
        amp_phase_errors = check_amplitude_and_phase_enhanced(
            shallow_peak_indices, deep_peak_indices, 
            shallow_temp, deep_temp, water_day,
            shallow_trough_indices, deep_trough_indices,
            amplitude_method, ar_tolerance, phase_min_limit, phase_max_limit
        )
        errors.extend(amp_phase_errors)
    
    logger.info(f"=== ERROR CHECKING COMPLETED: {len(errors)} total errors ===", "ERRORS")
    
    # Log error summary by type for detailed tracking
    error_types = {}
    for error in errors:
        error_type = error['type']
        if error_type not in error_types:
            error_types[error_type] = 0
        error_types[error_type] += 1
    
    logger.error_check(len(errors), error_types)
    
    # Create enhanced error display with debugging info
    if len(errors) == 0:
        error_display = html.Div([
            html.H4("✅ No errors detected!", style={'color': 'green'}),
            html.P(f"Checked {len(shallow_peak_indices)} shallow and {len(deep_peak_indices)} deep peaks using {amplitude_method} method", 
                   style={'fontSize': '0.9em', 'color': 'gray'})
        ])
    else:
        # Group errors by type for better display
        error_groups = {}
        for error in errors:
            error_type = error['type']
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(error)
        
        error_sections = []
        error_colors = {
            'unmatched_shallow': '#ff6b6b',      # Red
            'unmatched_deep': '#ff6b6b',         # Red  
            'multiple_matches': '#ff9500',       # Orange
            'alternation': '#ff9500',            # Orange
            'amplitude_ratio_high': '#ffd93d',   # Yellow
            'amplitude_ratio_low': '#ffd93d',    # Yellow
            'phase_too_small': '#74c0fc',        # Light blue
            'phase_too_large': '#74c0fc'         # Light blue
        }
        
        # Add debug summary
        debug_info = html.Div([
            html.H5("Debug Information:", style={'color': '#666', 'marginTop': '10px'}),
            html.P(f"• Total peaks analyzed: {len(shallow_peak_indices)} shallow + {len(deep_peak_indices)} deep"),
            html.P(f"• Total troughs available: {len(shallow_trough_indices)} shallow + {len(deep_trough_indices)} deep"),
            html.P(f"• Amplitude calculation method: {amplitude_method}"),
            html.P(f"• Error tolerance settings: Ar ≤ {ar_tolerance:.3f} or ≥ {1-ar_tolerance:.3f}, Phase: {phase_min_limit:.3f}-{phase_max_limit:.1f} days"),
            html.P(f"• Time range: {water_day[0]:.1f} to {water_day[-1]:.1f} days ({len(water_day)} data points)"),
        ], style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px', 'fontSize': '0.9em'})
        
        error_sections.append(debug_info)
        
        for error_type, type_errors in error_groups.items():
            color = error_colors.get(error_type, '#666666')
            
            type_section = html.Div([
                html.H5(f"{error_type.replace('_', ' ').title()} ({len(type_errors)} errors):", 
                       style={'color': color, 'marginBottom': '5px'}),
                html.Ul([
                    html.Li(f"Day {error['time']:.2f}: {error['message']}")
                    for error in type_errors[:10]  # Limit to first 10 errors of each type
                ] + ([html.Li(f"... and {len(type_errors)-10} more", style={'fontStyle': 'italic'})] 
                     if len(type_errors) > 10 else []),
                style={'fontSize': '0.9em', 'marginBottom': '10px'})
            ])
            error_sections.append(type_section)
        
        error_display = html.Div([
            html.H4(f"⚠️ {len(errors)} errors detected:", style={'color': 'red'}),
            html.Div(error_sections)
        ])
    
    # ADD VISUAL ERROR MARKERS TO THE GRAPH - ENHANCED VERSION
    if current_fig and len(errors) > 0:
        fig = go.Figure(current_fig)
        
        # Group errors by type for different marker styles
        error_times_by_type = {}
        for error in errors:
            error_type = error['type']
            if error_type not in error_times_by_type:
                error_times_by_type[error_type] = []
            error_times_by_type[error_type].append(error['time'])
        
        # Enhanced Y-range calculation with proper numeric handling
        y_values = []
        try:
            # Collect all y values from temperature traces only (avoid marker traces)
            for trace in fig.data:
                if (hasattr(trace, 'y') and len(trace.y) > 0 and 
                    hasattr(trace, 'name') and 
                    ('Shallow' in str(trace.name) or 'Deep' in str(trace.name)) and
                    'Peak' not in str(trace.name) and 'Trough' not in str(trace.name)):
                    
                    # Convert to numeric values and filter out non-numeric
                    for y_val in trace.y:
                        try:
                            numeric_val = float(y_val)
                            if not np.isnan(numeric_val) and not np.isinf(numeric_val):
                                y_values.append(numeric_val)
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            continue
            
            # Fallback: use original data if trace extraction fails
            if len(y_values) < 10:
                logger.warning("Few y-values extracted from traces, using original data", "ERRORS")
                y_values = list(shallow_temp) + list(deep_temp)
                # Convert to numeric and filter
                numeric_y_values = []
                for val in y_values:
                    try:
                        numeric_val = float(val)
                        if not np.isnan(numeric_val) and not np.isinf(numeric_val):
                            numeric_y_values.append(numeric_val)
                    except (ValueError, TypeError):
                        continue
                y_values = numeric_y_values
            
            if len(y_values) > 0:
                y_min = float(min(y_values))
                y_max = float(max(y_values))
                y_range = y_max - y_min
                
                # Ensure we have a reasonable range
                if y_range <= 0:
                    y_range = 1.0  # Default range
                    
                error_y_position = y_max + 0.05 * y_range
                logger.debug(f"Y-range calculated: min={y_min:.3f}, max={y_max:.3f}, range={y_range:.3f}", "ERRORS")
            else:
                # Ultimate fallback if no numeric values found
                logger.warning("No valid numeric y-values found, using default positioning", "ERRORS")
                y_min, y_max, y_range = 0, 20, 20
                error_y_position = 22
                
        except Exception as e:
            logger.error(f"Error in Y-range calculation: {e}", "ERRORS")
            # Emergency fallback
            y_min, y_max, y_range = 0, 20, 20
            error_y_position = 22
        
        # Define error marker styles
        error_marker_styles = {
            'unmatched_shallow': {'color': '#ff0000', 'symbol': 'x', 'size': 15, 'name': 'Unmatched Shallow'},
            'unmatched_deep': {'color': '#cc0000', 'symbol': 'x', 'size': 15, 'name': 'Unmatched Deep'},
            'multiple_matches': {'color': '#ff6600', 'symbol': 'triangle-up', 'size': 12, 'name': 'Multiple Matches'},
            'alternation': {'color': '#ff9900', 'symbol': 'diamond', 'size': 12, 'name': 'Alternation Error'},
            'amplitude_ratio_high': {'color': '#ffcc00', 'symbol': 'square', 'size': 10, 'name': 'High Amplitude Ratio'},
            'amplitude_ratio_low': {'color': '#ffdd44', 'symbol': 'square', 'size': 10, 'name': 'Low Amplitude Ratio'},
            'phase_too_small': {'color': '#0099ff', 'symbol': 'triangle-down', 'size': 10, 'name': 'Phase Too Small'},
            'phase_too_large': {'color': '#0066cc', 'symbol': 'triangle-down', 'size': 10, 'name': 'Phase Too Large'}
        }
        
        # Add error markers for each type
        for error_type, times in error_times_by_type.items():
            if error_type in error_marker_styles:
                style = error_marker_styles[error_type]
                try:
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=[error_y_position] * len(times),
                        mode='markers',
                        name=f"{style['name']} ({len(times)})",
                        marker=dict(
                            color=style['color'],
                            size=style['size'],
                            symbol=style['symbol'],
                            line=dict(width=2, color='black')
                        ),
                        hovertemplate=f"{style['name']}<br>Day: %{{x:.2f}}<extra></extra>",
                        showlegend=True
                    ))
                    logger.debug(f"Added {len(times)} {style['name']} markers at y={error_y_position:.2f}", "ERRORS")
                except Exception as e:
                    logger.error(f"Error adding {error_type} markers: {e}", "ERRORS")
        
        # Update the Y-axis range to accommodate error markers
        try:
            new_y_min = y_min - 0.02 * y_range
            new_y_max = error_y_position + 0.05 * y_range
            fig.update_layout(yaxis=dict(range=[new_y_min, new_y_max]))
            logger.debug(f"Updated Y-axis range: [{new_y_min:.2f}, {new_y_max:.2f}]", "ERRORS")
        except Exception as e:
            logger.error(f"Error updating Y-axis range: {e}", "ERRORS")
        
        # Add annotation explaining error markers
        try:
            fig.add_annotation(
                text=f"❌ Error markers shown above data ({len(errors)} errors found, {amplitude_method} method)",
                xref="paper", yref="paper",
                x=0.02, y=0.98, showarrow=False,
                font=dict(size=12, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
        except Exception as e:
            logger.error(f"Error adding annotation: {e}", "ERRORS")
        
        return error_display, fig
    else:
        return error_display, current_fig if current_fig else go.Figure()

@app.callback(
    Output('export-status', 'children'),
    Input('export-button', 'n_clicks'),
    [State('stored-data', 'children'),
     State('stored-peaks', 'children'),
     State('output-folder', 'value'),
     State('peak-filename', 'value'),
     State('formatted-filename', 'value'),
     State('sensor-spacing', 'value'),
     State('current-filename', 'children'),
     State('amplitude-method', 'value'),
     State('data-year', 'value')],  # NEW: Add data-year state
    prevent_initial_call=True
)
def export_data_enhanced(n_clicks, stored_data, stored_peaks, output_folder, 
                        peak_filename, formatted_filename, sensor_spacing, original_filename, 
                        amplitude_method, data_year):  # NEW: Add data_year parameter
    """
    Export peak data to CSV files with improved formatting and peak-trough amplitude support.
    Now uses user-specified year instead of filename parsing.
    """
    if n_clicks == 0 or stored_data is None or stored_peaks is None:
        return ''
    
    logger.user_action("Data export", f"using {amplitude_method} amplitude method, year={data_year}")
    
    # Create output directory
    output_path = ensure_output_dir(output_folder)
    
    # Load data
    df = pd.read_json(StringIO(stored_data), orient='split')
    peaks_data = json.loads(stored_peaks)
    
    water_day = df['WaterDay'].values
    shallow_temp = df['Shallow.Temp.Filt'].values
    deep_temp = df['Deep.Temp.Filt'].values
    
    # Create peak data for export
    export_data = []
    
    # Add shallow peaks
    for idx in peaks_data['shallow_peaks']:
        if isinstance(idx, (int, np.integer)) and 0 <= idx < len(water_day):
            export_data.append({
                'WaterDay': float(water_day[idx]),
                'Temp.Filt': float(shallow_temp[idx]),
                'Depth': 'Shallow'
            })
    
    # Add deep peaks
    for idx in peaks_data['deep_peaks']:
        if isinstance(idx, (int, np.integer)) and 0 <= idx < len(water_day):
            export_data.append({
                'WaterDay': float(water_day[idx]),
                'Temp.Filt': float(deep_temp[idx]),
                'Depth': 'Deep'
            })
    
    # Sort by WaterDay
    export_df = pd.DataFrame(export_data).sort_values('WaterDay')
    
    # Save to file with custom filename
    peak_filepath = output_path / peak_filename
    export_df.to_csv(peak_filepath, index=False)
    
    logger.info(f"Exported {len(export_df)} peaks to {peak_filepath}", "EXPORT")
    
    # Enhanced amplitude ratio and phase shift calculation with peak-trough support
    shallow_peaks = np.array(peaks_data.get('shallow_peaks', []), dtype=int)
    deep_peaks = np.array(peaks_data.get('deep_peaks', []), dtype=int)
    shallow_troughs = np.array(peaks_data.get('shallow_troughs', []), dtype=int)
    deep_troughs = np.array(peaks_data.get('deep_troughs', []), dtype=int)
    
    ar_ps_data = []
    
    if len(shallow_peaks) > 0 and len(deep_peaks) > 0:
        
        if amplitude_method == 'peak-trough' and len(shallow_troughs) > 0 and len(deep_troughs) > 0:
            # Use enhanced peak-trough method
            logger.info("Using peak-trough amplitude calculation for export", "EXPORT")
            
            # Calculate amplitudes using the peak-trough method
            shallow_amplitudes = calculate_peak_trough_amplitude(
                shallow_peaks, shallow_troughs, 
                shallow_temp, shallow_temp, water_day
            )
            deep_amplitudes = calculate_peak_trough_amplitude(
                deep_peaks, deep_troughs,
                deep_temp, deep_temp, water_day
            )
            
            # Get amplitude ratio data
            ratio_data = calculate_amplitude_ratio_enhanced(shallow_amplitudes, deep_amplitudes)
            
            # Convert to export format using user-specified year
            for ratio_info in ratio_data:
                # Use the year from the user input (data_year)
                year = data_year if data_year is not None else datetime.now().year
                
                # Validate year is reasonable (1950-2150)
                if year < 1950 or year > 2150:
                    year = datetime.now().year
                    logger.warning(f"Invalid year {data_year}, using current year {year}", "EXPORT")
                
                ar_ps_data.append({
                    'Data_Year': year,
                    'Water_Day': float(ratio_info['shallow_time'] + ratio_info['phase_shift'] / 2),
                    'Ad_As': ratio_info['amplitude_ratio'],
                    'A_Uncertainty': 1e-5,
                    'Phase_Shift(days)': ratio_info['phase_shift'],
                    'f_Uncertainty': 0.001
                })
                
        else:
            # Use original peak-only method
            logger.info("Using peak-only amplitude calculation for export", "EXPORT")
            
            for s_idx in shallow_peaks:
                if 0 <= s_idx < len(water_day):
                    s_time = water_day[s_idx]
                    s_peak_amp = shallow_temp[s_idx]
                    s_amp = s_peak_amp  # Use peak value
                    
                    # Find corresponding deep peak
                    deep_times = water_day[deep_peaks]
                    deep_after_mask = deep_times > s_time
                    
                    if np.any(deep_after_mask):
                        # Get the first deep peak after shallow
                        deep_after_indices = deep_peaks[deep_after_mask]
                        d_idx = deep_after_indices[0]
                        
                        if 0 <= d_idx < len(water_day):
                            d_time = water_day[d_idx]
                            d_peak_amp = deep_temp[d_idx]
                            d_amp = d_peak_amp  # Use peak value
                            
                            # Calculate amplitude ratio and phase shift
                            ar = float(d_amp / s_amp) if s_amp != 0 else 0
                            ps = float(d_time - s_time)
                            
                            # Use the year from the user input (data_year)
                            year = data_year if data_year is not None else datetime.now().year
                            
                            # Validate year is reasonable (1950-2150)
                            if year < 1950 or year > 2150:
                                year = datetime.now().year
                                logger.warning(f"Invalid year {data_year}, using current year {year}", "EXPORT")
                            
                            ar_ps_data.append({
                                'Data_Year': year,
                                'Water_Day': float((s_time + d_time) / 2),
                                'Ad_As': ar,
                                'A_Uncertainty': 1e-5,
                                'Phase_Shift(days)': ps,
                                'f_Uncertainty': 0.001
                            })
    
    formatted_filepath = None
    files_created = [f"Peak data: {peak_filename} ({len(export_df)} peaks)"]
    
    if len(ar_ps_data) > 0:
        # Create header for formatted file using only ASCII characters
        header_lines = [
            "PHASE SHIFT AND AMPLITUDE RATIO DATA FILE: PEAKPICKER OUTPUT",
            "----------------------------------------------------------------",
            f"{sensor_spacing:.3f} is the relative distance (in m) between sensors.",
            f"Amplitude calculation method: {amplitude_method}",
            f"Data year: {data_year} (user-specified)",
            "----------------------------------------------------------------",
            ""
        ]
        
        ar_ps_df = pd.DataFrame(ar_ps_data)
        
        # Save with header and tab separation
        formatted_filepath = output_path / formatted_filename
        with open(formatted_filepath, 'w') as f:
            for line in header_lines:
                f.write(line + '\n')
            # Write column headers
            f.write("Data_Year\tWater_Day\tAd_As\tA_Uncertainty\tPhase_Shift(days)\tf_Uncertainty\n")
            # Write data with proper formatting
            for _, row in ar_ps_df.iterrows():
                f.write(f"{int(row['Data_Year'])}\t")
                f.write(f"{row['Water_Day']:.8f}\t")
                f.write(f"{row['Ad_As']:.8f}\t")
                f.write(f"{row['A_Uncertainty']:.8e}\t")
                f.write(f"{row['Phase_Shift(days)']:.8f}\t")
                f.write(f"{row['f_Uncertainty']:.8f}\n")
        
        files_created.append(f"Formatted data: {formatted_filename} ({len(ar_ps_data)} ratio pairs, {amplitude_method} method, year={data_year})")
        logger.info(f"Exported {len(ar_ps_data)} amplitude ratio/phase shift pairs using {amplitude_method} method with year {data_year}", "EXPORT")
    
    # Log export summary
    logger.export_data(str(output_path), files_created)
    
    return html.Div([
        html.P(f"✓ Exported {len(export_df)} peaks to {peak_filepath}"),
        html.P(f"✓ Exported {len(ar_ps_data)} amplitude ratio/phase shift pairs to {formatted_filepath} using {amplitude_method} method with year {data_year}" 
               if formatted_filepath else "No valid peak pairs found for formatted output"),
        html.P(f"Files saved in: {output_path}")
    ], style={'color': 'green'})

# Additional callbacks for parameter management
@app.callback(
    [Output('target-period', 'value'),
     Output('search-tolerance', 'value'),
     Output('slope-threshold', 'value'),
     Output('min-distance', 'value'),
     Output('prominence-factor', 'value'),
     Output('peak-method-dropdown', 'value'),
     Output('sensor-spacing', 'value'),
     Output('output-folder', 'value'),
     Output('ar-tolerance', 'value'),
     Output('phase-max-limit', 'value'),
     Output('phase-min-limit', 'value'),
     Output('amplitude-method', 'value'),
     Output('param-file-status', 'children')],
    Input('load-param-file-button', 'n_clicks'),
    prevent_initial_call=True
)
def load_parameters_from_file(n_clicks):
    """Load parameters from .par file and update UI."""
    if n_clicks > 0:
        logger.user_action("Load parameters from .par file")
        params = load_param_file()
        return (params.get('target_period', 1.0),
               params.get('search_tolerance', 0.1),
               params.get('slope_threshold', 0.001),
               params.get('min_distance', 20),
               params.get('prominence_factor', 0.15),
               params.get('peak_method', 'manual'),
               params.get('sensor_spacing', 0.18),
               params.get('output_folder', 'peak_analysis_output'),
               params.get('ar_tolerance', 0.001),
               params.get('phase_max_limit', 1.0),
               params.get('phase_min_limit', 0.001),
               params.get('amplitude_method', 'peak-trough'),
               html.Span("✓ Parameters loaded from .par file", style={'color': 'green'}))
    return no_update

@app.callback(
    Output('param-file-status', 'children', allow_duplicate=True),
    Input('save-param-file-button', 'n_clicks'),
    [State('target-period', 'value'),
     State('search-tolerance', 'value'),
     State('slope-threshold', 'value'),
     State('min-distance', 'value'),
     State('prominence-factor', 'value'),
     State('peak-method-dropdown', 'value'),
     State('sensor-spacing', 'value'),
     State('output-folder', 'value'),
     State('ar-tolerance', 'value'),
     State('phase-max-limit', 'value'),
     State('phase-min-limit', 'value'),
     State('amplitude-method', 'value'),
     State('current-filename', 'children')],
    prevent_initial_call=True
)
def save_parameters_to_file(n_clicks, period, tolerance, slope, distance, prominence, 
                           method, spacing, output_folder, ar_tol, phase_max, phase_min, 
                           amplitude_method, data_filename):
    """Save current parameters to .par file."""
    if n_clicks > 0:
        logger.user_action("Save parameters to .par file")
        params = {
            'target_period': period,
            'search_tolerance': tolerance,
            'slope_threshold': slope,
            'min_distance': distance,
            'prominence_factor': prominence,
            'peak_method': method,
            'sensor_spacing': spacing,
            'output_folder': output_folder,
            'ar_tolerance': ar_tol,
            'phase_max_limit': phase_max,
            'phase_min_limit': phase_min,
            'amplitude_method': amplitude_method,
            'data_file': data_filename or ''
        }
        save_param_file(params)
        return html.Span("✓ Parameters saved to .par file", style={'color': 'green'})
    return ''

# Additional UI callbacks
@app.callback(
    [Output('peak-visibility-state', 'children'),
     Output('interactive-graph', 'figure', allow_duplicate=True)],
    Input('toggle-visibility-button', 'n_clicks'),
    [State('peak-visibility-state', 'children'),
     State('interactive-graph', 'figure')],
    prevent_initial_call=True
)
def toggle_peak_visibility(n_clicks, current_state, current_figure):
    """
    Toggle peak visibility between visible and semi-transparent.
    """
    if n_clicks == 0:
        return current_state if current_state is not None else 'visible', current_figure
    
    logger.user_action("Toggle peak visibility")
    
    # Ensure current_state is not None
    if current_state is None:
        current_state = 'visible'
    
    # Toggle state
    new_state = 'hidden' if current_state == 'visible' else 'visible'
    
    # Update figure if it exists
    if current_figure is None:
        return new_state, go.Figure()
    
    fig = go.Figure(current_figure)
    
    if fig.data:
        for trace in fig.data:
            # Only affect traces that are peaks/troughs (not temperature lines)
            if ('Peak' in str(trace.name) or 'Trough' in str(trace.name) or 
                'Manual' in str(trace.name) or 'Maxima' in str(trace.name)):
                
                if new_state == 'visible':
                    # Make peaks fully visible
                    if 'Maxima' in str(trace.name):
                        trace.opacity = 0.7  # Keep maxima slightly transparent
                    else:
                        trace.opacity = 1.0
                    trace.visible = True
                else:
                    # Make peaks semi-transparent
                    if 'Maxima' in str(trace.name):
                        trace.opacity = 0.2
                    else:
                        trace.opacity = 0.3
                    trace.visible = True
    
    logger.info(f"Peak visibility changed to: {new_state}", "UI")
    return new_state, fig

@app.callback(
    Output('manual-selections-store', 'children', allow_duplicate=True),
    Input('clear-manual-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_manual_selections(n_clicks):
    """Clear all manual peak selections and exclusions."""
    if n_clicks > 0:
        logger.user_action("Clear all manual selections")
        empty_selections = {'shallow_peaks': [], 'deep_peaks': [], 'excluded_ranges': [], 
                           'excluded_detected_peaks': {'shallow': [], 'deep': []}}
        logger.info("Cleared all manual selections and exclusions", "USER")
        return json.dumps(empty_selections)
    return no_update

@app.callback(
    [Output('peak-filename', 'value'),
     Output('formatted-filename', 'value')],
    Input('current-filename', 'children'),
    prevent_initial_call=True
)
def suggest_filenames(original_filename):
    """
    Suggest output filenames based on input filename.
    
    Follows the naming convention:
    - Input: PR24_TP04-SD1-filtr.csv
    - Peak output: PR24_TP04-SD1-pick.csv
    - Formatted output: PR24_TP04-SD1-pickr.dAf
    """
    if original_filename:
        # Remove extension and add suffixes
        base_name = os.path.splitext(original_filename)[0]
        # Remove '-filtr' if present
        base_name = base_name.replace('-filtr', '')
        
        peak_filename = f"{base_name}-pick.csv"
        formatted_filename = f"{base_name}-pickr.dAf"
        
        logger.info(f"Suggested filenames based on {original_filename}: {peak_filename}, {formatted_filename}", "FILES")
        
        return peak_filename, formatted_filename
    
    return "peak_picks.csv", "peak_picks_formatted.csv"

# NEW: Callback to auto-populate year from filename if present
@app.callback(
    Output('data-year', 'value'),
    Input('current-filename', 'children'),
    prevent_initial_call=True
)
def suggest_year_from_filename(original_filename):
    """
    Try to extract year from filename and suggest it, but user can still override.
    This maintains the old functionality as a helpful default.
    """
    if original_filename:
        # Try multiple patterns for year extraction (same logic as before)
        # Pattern 1: Look for 4-digit year (e.g., PR2024_...)
        year_match_4digit = re.search(r'(19|20|21)\d{2}', original_filename)
        if year_match_4digit:
            suggested_year = int(year_match_4digit.group(0))
            logger.info(f"Suggested year {suggested_year} from filename pattern", "FILES")
            return suggested_year
        else:
            # Pattern 2: Look for 2-digit year after PR (e.g., PR24_...)
            year_match_2digit = re.search(r'PR(\d{2})_', original_filename)
            if year_match_2digit:
                two_digit_year = int(year_match_2digit.group(1))
                # Smart century detection
                current_year = datetime.now().year
                current_century = (current_year // 100) * 100
                current_year_2digit = current_year % 100
                
                # If 2-digit year is more than 50 years in future, assume previous century
                if two_digit_year > current_year_2digit + 50:
                    suggested_year = current_century - 100 + two_digit_year
                else:
                    suggested_year = current_century + two_digit_year
                
                logger.info(f"Suggested year {suggested_year} from 2-digit pattern ({two_digit_year})", "FILES")
                return suggested_year
            else:
                # Pattern 3: Look for year in WY or water year format
                wy_match = re.search(r'WY(\d{2,4})', original_filename)
                if wy_match:
                    wy_year = wy_match.group(1)
                    if len(wy_year) == 2:
                        # Same smart century detection for WY format
                        two_digit_year = int(wy_year)
                        current_year = datetime.now().year
                        current_century = (current_year // 100) * 100
                        current_year_2digit = current_year % 100
                        
                        if two_digit_year > current_year_2digit + 50:
                            suggested_year = current_century - 100 + two_digit_year
                        else:
                            suggested_year = current_century + two_digit_year
                    else:
                        suggested_year = int(wy_year)
                    
                    logger.info(f"Suggested year {suggested_year} from WY pattern", "FILES")
                    return suggested_year
    
    # If no year found in filename, keep current year as default
    return no_update

# ===========================
# MAIN EXECUTION WITH ENHANCED LOGGING
# ===========================

if __name__ == '__main__':
    # Enhanced startup information
    logger.info("="*80, "APP")
    logger.info("TTS PEAK PICKER - ENHANCED VERSION WITH LOGGING, PEAK-TROUGH ALGORITHM, AND USER YEAR INPUT", "APP")
    logger.info("="*80, "APP")
    logger.info(f"Parameter file location: {PARAM_FILE}", "CONFIG")
    logger.info(f"Output will be saved to: {SCRIPT_DIR / initial_params['output_folder']}", "CONFIG")
    logger.info(f"Log directory: {logger.log_dir}", "CONFIG")
    logger.info(f"Session log file: {logger.log_file}", "CONFIG")
    
    # Create default .par file if it doesn't exist
    if not PARAM_FILE.exists():
        logger.info(f"Creating default parameter file: {PARAM_FILE}", "CONFIG")
        default_params = {
            'data_file': '',
            'sensor_spacing': 0.18,
            'target_period': 1.0,
            'search_tolerance': 0.1,
            'slope_threshold': 0.001,
            'min_distance': 20,
            'prominence_factor': 0.15,  # Lower for better sensitivity
            'output_folder': 'peak_analysis_output',
            'peak_method': 'manual',  # Default to manual
            'shallow_color': 'blue',
            'deep_color': 'red',
            'peak_size': 12,
            'trough_size': 10,
            'line_width': 2,
            'ar_tolerance': 0.001,
            'phase_max_limit': 1.0,
            'phase_min_limit': 0.001,
            'amplitude_method': 'peak-trough'  # Default to peak-trough method
        }
        save_param_file(default_params, PARAM_FILE)
        
        # Also create a sample .par file with comments using only ASCII characters
        sample_content = """# Thermal Probe Peak Picker Parameter File - Enhanced Version with User Year Input
# tts_peakpick.par
#
# This file contains initial parameters for the peak picking analysis.
# Parameters can be modified within the tool and saved back to this file.
#
# New in this version:
# - Enhanced logging system with session tracking
# - Peak-trough amplitude calculation for better accuracy
# - Visual error markers on graphs
# - Improved peak detection algorithms
# - User-selectable year input for export files
#
# Format: parameter_name = value
# Lines starting with # are comments

[PARAMETERS]
# Input data file (can be selected via GUI or auto-loaded)
data_file = 

# Sensor spacing in meters (used in formatted output file)
sensor_spacing = 0.18

# Desired time period between peaks (days)
# Common values: 1.0 (daily), 0.5 (semi-daily), 2.0 (bi-daily)
target_period = 1.0

# Timing threshold for searching for next peak (days)
# This defines the window around the expected peak location
search_tolerance = 0.1

# Slope threshold for identifying peaks and troughs (degC/min)
# Lower values = more sensitive detection
slope_threshold = 0.001

# Minimum distance between peaks (in samples)
min_distance = 20

# Prominence factor (0-1) for peak detection
# Lower values = more peaks detected
# Recommended: 0.05-0.15 for noisy data, 0.2-0.3 for clean data
prominence_factor = 0.15

# Output folder name (created in same directory as script)
output_folder = peak_analysis_output

# Default peak detection method
# Options: manual, scipy, wavelet, derivative, prominence, combined, bootstrap
# 'manual' is most accurate, 'combined' finds the most peaks automatically
peak_method = manual

# NEW: Amplitude calculation method
# Options: peak-trough, peak-only
# peak-trough: More accurate for asymmetric waves (A = (Peak - Trough) / 2)
# peak-only: Original method using peak values only
amplitude_method = peak-trough

# Error checking limits
# Amplitude ratio tolerance for flagging Ar >= 1-tolerance or <= tolerance
ar_tolerance = 0.001

# Maximum allowed phase shift (days)
phase_max_limit = 1.0

# Minimum allowed phase shift (days)  
phase_min_limit = 0.001

# Display colors and sizes
shallow_color = blue
deep_color = red
peak_size = 12
trough_size = 10
line_width = 2
"""
        
        # Write sample file with comments
        sample_file = SCRIPT_DIR / 'tts_peakpick_sample.par'
        with open(sample_file, 'w') as f:
            f.write(sample_content)
        logger.info(f"Created sample parameter file: {sample_file}", "CONFIG")
    
    # Clean up old log files
    logger.cleanup_old_logs(keep_days=30)
    
    logger.info("Key Features - Enhanced Version with User Year Input:", "FEATURES")
    logger.info("✅ USER YEAR INPUT: Manual year selection for .dAf export files", "FEATURES")
    logger.info("✅ COMPREHENSIVE LOGGING: Session tracking with detailed logs in logs/ folder", "FEATURES")
    logger.info("✅ PEAK-TROUGH AMPLITUDE: More accurate A = (Peak - Trough) / 2 calculation", "FEATURES")
    logger.info("✅ VISUAL ERROR MARKERS: Red ❌ marks show exact error locations on graph", "FEATURES")
    logger.info("✅ TWO GRAPHS: Data verification (read-only) + Interactive peak picking", "FEATURES")
    logger.info("✅ ENHANCED auto-load: Searches script directory and relative paths", "FEATURES")
    logger.info("✅ ZOOM PRESERVATION: Graph maintains zoom level after peak edits", "FEATURES")
    logger.info("✅ CLEAR TROUGHS: Orange button to remove all trough markers", "FEATURES")
    logger.info("✅ TOGGLE VISIBILITY: Purple button to control peak opacity", "FEATURES")
    logger.info("✅ Enhanced peak removal: can remove ANY peaks (detected or manual)", "FEATURES")
    logger.info("✅ Amplitude method selection: Choose between peak-trough or peak-only", "FEATURES")
    logger.info("✅ Comprehensive error checking with visual feedback", "FEATURES")
    logger.info("✅ Improved robustness and data validation", "FEATURES")
    logger.info("✅ Enhanced parameter validation with warnings", "FEATURES")
    logger.info("✅ Better peak detection algorithms with fallback methods", "FEATURES")
    
    logger.info("Year Input Features:", "FEATURES")
    logger.info("📅 User can manually specify data year for .dAf export", "FEATURES")
    logger.info("📅 Year auto-suggested from filename but user can override", "FEATURES")
    logger.info("📅 Year validation (1950-2150) with fallback to current year", "FEATURES")
    logger.info("📅 Year included in export status and log messages", "FEATURES")
    logger.info("📅 No more dependency on filename parsing for year determination", "FEATURES")
    
    logger.info("Logging Features:", "FEATURES")
    logger.info("📋 Session tracking with timestamps and unique session IDs", "FEATURES")
    logger.info("📋 Detailed user action logging for reproducibility", "FEATURES")
    logger.info("📋 Peak detection parameter and result logging", "FEATURES")
    logger.info("📋 Error checking summaries with counts by type", "FEATURES")
    logger.info("📋 Data export tracking with file information", "FEATURES")
    logger.info("📋 Automatic log cleanup (keeps 30 days by default)", "FEATURES")
    
    logger.info("Peak-Trough Algorithm Features:", "FEATURES")
    logger.info("🔬 More accurate amplitude calculation for asymmetric waves", "FEATURES")
    logger.info("🔬 Better handling of signals with subtle trends", "FEATURES")
    logger.info("🔬 Automatic peak-trough pairing within configurable time windows", "FEATURES")
    logger.info("🔬 Fallback to peak-only method when no troughs are available", "FEATURES")
    logger.info("🔬 Enhanced error checking with amplitude method information", "FEATURES")
    logger.info("🔬 Export files include amplitude calculation method details", "FEATURES")
    
    logger.info("Error Visualization Features:", "FEATURES")
    logger.info("❌ Red ❌ marks for unmatched peaks", "FEATURES")
    logger.info("🔺 Orange triangles for alternation errors", "FEATURES")
    logger.info("⬜ Yellow squares for amplitude ratio issues", "FEATURES")
    logger.info("🔻 Blue triangles for phase shift problems", "FEATURES")
    logger.info("ℹ️ Hover information showing error details", "FEATURES")
    logger.info("📊 Automatic Y-axis adjustment to show error markers", "FEATURES")
    
    logger.info("Usage:", "USAGE")
    logger.info("1. Load your thermal probe data (CSV with WaterDay, Shallow.Temp.Filt, Deep.Temp.Filt)", "USAGE")
    logger.info("2. Set the data year in the textbox (auto-suggested from filename)", "USAGE")
    logger.info("3. Choose amplitude calculation method (peak-trough recommended)", "USAGE")
    logger.info("4. Use manual peak selection or automated detection methods", "USAGE")
    logger.info("5. Click 'Check Errors' to see visual error markers on the graph", "USAGE")
    logger.info("6. Fix errors by adding/removing peaks or adjusting parameters", "USAGE")
    logger.info("7. Export your results with the specified year included in .dAf file", "USAGE")
    logger.info("8. Review session logs in the logs/ folder for detailed analysis", "USAGE")
    
    logger.info("Starting enhanced web application at http://127.0.0.1:8052/", "APP")
    logger.info("Press Ctrl+C to stop the application", "APP")
    
    # Verify logging is working by testing different output methods
    logger.info("Testing logging capture:", "TEST")
    logger.info("✅ Logger.info() captured", "TEST")
    print("✅ print() statement captured via stdout intercept")  # This will be captured by LogCapture
    
    try:
        app.run(debug=False, host='127.0.0.1', port=8052)
    except KeyboardInterrupt:
        logger.info("Application stopped by user", "APP")
    finally:
        logger.session_end()
