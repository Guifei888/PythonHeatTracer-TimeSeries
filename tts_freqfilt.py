#!/usr/bin/env python3
"""
Filename: tts_freqfilt.py
Title: Temperature Time-Series Frequency Analysis and Filtering Tool

Description:
    Advanced interactive analysis and filtering tool for temperature time-series data.
    Designed to identify frequency components such as diurnal signals in shallow and deep probes.
    Includes spectral methods, filtering, and high-resolution resampling.

Features:
    - Power Spectral Density (PSD) visualization: Welch, Multitaper, and smoothed options
    - Graphical frequency band selection
    - Multiple filter types (Butterworth, etc.) with ramped transitions
    - Real-time filter preview and result comparison
    - High-resolution resampling (e.g., 20-min → 1-min)
    - Parameter file support (.par format)
    - Output export and parameter persistence
    - Robust, rotating logging system (terminal + UTF-8 logs)
    - FIXED: Zoom preservation when applying filters

Enhanced in v2.2.1:
    - Fixed zoom reset issue when applying filters
    - Preserved user zoom/pan state during filter application

Author: Timothy Wu  
Created: 2025-07-03  
Last Updated: 2025-08-13  
Version: 2.2.1

Usage:
    python tts_freqfilt.py
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from scipy import signal
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
import os
import json
from datetime import datetime
import base64
import io
import re
import tkinter as tk
from tkinter import filedialog
import logging
import sys
from logging.handlers import RotatingFileHandler

# Debug flag for troubleshooting callbacks
DEBUG_CALLBACKS = True

# Try to import mne for multitaper support
try:
    from mne.time_frequency import psd_array_multitaper
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


def setup_logging():
    """Set up comprehensive logging system with proper encoding"""
    # Get script directory for log folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'log')
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")
    
    # Create logger
    logger = logging.getLogger('tts_freqfilt')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler for detailed logs (with rotation and UTF-8 encoding)
    detailed_log_file = os.path.join(log_dir, 'tts_freqfilt_detailed.log')
    file_handler = RotatingFileHandler(
        detailed_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # File handler for operation logs (user-friendly, UTF-8 encoding)
    operation_log_file = os.path.join(log_dir, 'tts_freqfilt_operations.log')
    operation_handler = RotatingFileHandler(
        operation_log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    operation_handler.setLevel(logging.INFO)
    operation_handler.setFormatter(simple_formatter)
    
    # Console handler with encoding error handling
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
            except UnicodeEncodeError:
                # Convert Unicode characters to ASCII equivalents
                msg = self.format(record)
                # Replace problematic Unicode characters
                msg = msg.replace('⁻¹', '^-1')
                msg = msg.replace('→', '->')
                msg = msg.replace('✓', '[OK]')
                msg = msg.replace('✗', '[X]')
                msg = msg.replace('•', '*')
                msg = msg.replace('↓', '[v]')
                print(msg) 
    
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(operation_handler)
    logger.addHandler(console_handler)
    
    # Log startup information
    logger.info("="*70)
    logger.info("Temperature Time-Series Frequency Analysis & Filtering Tool v2.2.1")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Detailed log: {detailed_log_file}")
    logger.info(f"Operations log: {operation_log_file}")
    logger.info("="*70)
    
    return logger


# Initialize logging
logger = setup_logging()


class TemperatureFrequencyAnalyzer:
    def __init__(self):
        self.data = None
        self.filtered_data = None
        self.sampling_freq = 72  # measurements per day (default)
        self.analysis_data = None  # Stores clipped data for analysis
        self.last_filter_info = None  # Store filter characteristics
        
        logger.info("Initializing TemperatureFrequencyAnalyzer")
        
        self.psd_params = {
            'method': 'matlab_mtm',  # Default to MATLAB MTM
            'matlab_nfft': 2048,
            'nperseg_divider': 4,
            'nperseg_max': 520,
            'window': 'tukey',
            'noverlap': None,
            'nfft': None,
            'detrend': 'constant',
            'return_onesided': True,
            'scaling': 'density',
            'average': 'mean',
            # Multitaper specific - Default TBW to 4 to match R/MATLAB
            'time_bandwidth': 4,
            'adaptive': True,
            'low_bias': True,
            # Smoothing specific
            'smooth_sigma': 2
        }
        
        self.filter_params = {
            'f_low': 0.8,
            'f_high': 1.2, 
            'filter_type': 'butter',
            'filter_order': 3,  # Changed default from 6 to 3
            'ramp_fraction': 0.1,  # Use the value from user's .par file
            'original_interval_minutes': 20,  
            'resample_interval_minutes': 1,  # DEFAULT CHANGED: Now defaults to 1-minute for 20:1 improvement
            'trend_removal': 'dc'  # Default to DC removal as recommended
        }
        self.frequency_selection_mode = False
        self.selected_frequencies = []
        self.filter_history = []
        
        # Parameter file attributes
        self.input_filename = 'temperature_composite_data.csv'
        self.output_folder = 'Filter Data'  # New: configurable output folder
        self.time_bandwidth = 4
        self.wd_lower = -1
        self.wd_upper = -1
        
        # Search for parameter file in current directory and script directory
        search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
        par_file = None
        
        for directory in search_dirs:
            potential_path = os.path.join(directory, 'tts_freqfilt.par')
            if os.path.isfile(potential_path):
                par_file = potential_path
                break
        
        if par_file:
            logger.info(f"Loading parameters from: {par_file}")
            self.load_parameters(par_file)
        else:
            logger.warning("No tts_freqfilt.par found in current or script directory. Using defaults.")
            logger.info("IMPORTANT: Default resampling set to 1-minute intervals for 20:1 resolution improvement.")
            
    def check_default_file_exists(self):
        """Check if default input file exists in multiple locations"""
        if not hasattr(self, 'input_filename') or not self.input_filename:
            return False
            
        search_paths = [
            self.input_filename,  # As specified
            os.path.join(os.getcwd(), self.input_filename),  # Current working directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.input_filename)  # Script directory
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                logger.debug(f"Default file found at: {path}")
                return True
        
        logger.debug(f"Default file '{self.input_filename}' not found in any search location")
        return False
        
    def parse_par(self, path):
        """Parse a flat parameter file with key=value pairs."""
        params = {}
        logger.debug(f"Parsing parameter file: {path}")
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
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
                        logger.debug(f"Line {line_num}: {key} = {value}")
                        
        except FileNotFoundError:
            logger.error(f"Parameter file {path} not found")
            
        except Exception as e:
            logger.error(f"Error parsing parameter file {path}: {e}")
            
        logger.info(f"Successfully parsed {len(params)} parameters from {path}")
        return params

    def load_parameters(self, filepath):
        """Load parameters from .par file"""
        logger.info(f"Loading parameters from: {filepath}")
        params = self.parse_par(filepath)
        
        # Map parameter file keys to internal parameter names
        if 'filename_composite_data' in params:
            self.input_filename = params['filename_composite_data']
            logger.info(f"Set input filename to: {self.input_filename}")
        
        # NEW: Load output folder parameter
        if 'output_folder' in params:
            self.output_folder = params['output_folder']
            logger.info(f"Set output folder to: {self.output_folder}")
        
        if 'data_interval_minutes' in params:
            try:
                self.filter_params['original_interval_minutes'] = int(params['data_interval_minutes'])
                logger.info(f"Set original interval to: {self.filter_params['original_interval_minutes']} minutes")
                # IMPROVED: Better default handling for resample_interval
                if 'resample_interval' in params:
                    self.filter_params['resample_interval_minutes'] = int(params['resample_interval'])
                    logger.info(f"Set resample interval to: {self.filter_params['resample_interval_minutes']} minutes")
                else:
                    # If not specified, default to 1-minute for high resolution
                    logger.info("resample_interval not specified in .par file, defaulting to 1-minute for 20:1 improvement")
                    self.filter_params['resample_interval_minutes'] = 1
            except ValueError as e:
                logger.error(f"Error parsing data_interval_minutes: {e}")
        
        if 'time_bandwidth_parameter' in params:
            try:
                self.time_bandwidth = float(params['time_bandwidth_parameter'])
                self.psd_params['time_bandwidth'] = float(params['time_bandwidth_parameter'])
                logger.info(f"Set time-bandwidth parameter to: {self.time_bandwidth}")
            except ValueError as e:
                logger.error(f"Error parsing time_bandwidth_parameter: {e}")
        
        if 'wd_lower_limit' in params:
            try:
                self.wd_lower = float(params['wd_lower_limit'])
                logger.info(f"Set WD lower limit to: {self.wd_lower}")
            except ValueError as e:
                logger.error(f"Error parsing wd_lower_limit: {e}")
        
        if 'wd_upper_limit' in params:
            try:
                self.wd_upper = float(params['wd_upper_limit'])
                logger.info(f"Set WD upper limit to: {self.wd_upper}")
            except ValueError as e:
                logger.error(f"Error parsing wd_upper_limit: {e}")
        
        if 'start_band_pass' in params:
            try:
                self.filter_params['f_low'] = float(params['start_band_pass'])
                logger.info(f"Set filter low frequency to: {self.filter_params['f_low']} day^-1")
            except ValueError as e:
                logger.error(f"Error parsing start_band_pass: {e}")
        
        if 'end_band_pass' in params:
            try:
                self.filter_params['f_high'] = float(params['end_band_pass'])
                logger.info(f"Set filter high frequency to: {self.filter_params['f_high']} day^-1")
            except ValueError as e:
                logger.error(f"Error parsing end_band_pass: {e}")
        
        if 'ramp_fraction' in params:
            try:
                self.filter_params['ramp_fraction'] = float(params['ramp_fraction'])
                logger.info(f"Set ramp fraction to: {self.filter_params['ramp_fraction']}")
            except ValueError as e:
                logger.error(f"Error parsing ramp_fraction: {e}")
        
        if 'filter_order' in params:
            try:
                self.filter_params['filter_order'] = int(params['filter_order'])
                logger.info(f"Set filter order to: {self.filter_params['filter_order']}")
            except ValueError as e:
                logger.error(f"Error parsing filter_order: {e}")
        
        # IMPROVED: Better handling of resample_interval parameter
        if 'resample_interval' in params:
            try:
                resample_val = int(params['resample_interval'])
                self.filter_params['resample_interval_minutes'] = resample_val
                
                # Provide user feedback about resampling
                original_interval = self.filter_params['original_interval_minutes']
                if resample_val < original_interval:
                    improvement_factor = original_interval / resample_val
                    logger.info(f"Resampling enabled - {original_interval}-minute -> {resample_val}-minute data")
                    logger.info(f"Resolution improvement: {improvement_factor:.1f}:1")
                elif resample_val == original_interval:
                    logger.info(f"No resampling - maintaining {resample_val}-minute intervals")
                else:
                    logger.info(f"Downsampling - {original_interval}-minute -> {resample_val}-minute data")
            except ValueError as e:
                logger.error(f"Error parsing resample_interval: {e}")
        
        # Calculate sampling frequency from data interval
        if 'data_interval_minutes' in params:
            try:
                interval = int(params['data_interval_minutes'])
                self.sampling_freq = (24 * 60) / interval
                logger.info(f"Calculated sampling frequency: {self.sampling_freq:.2f} samples/day")
            except ValueError as e:
                logger.error(f"Error calculating sampling frequency: {e}")
        
        logger.info("Parameters loaded successfully")
        return True

    def save_parameters_file(self, filename='tts_freqfilt.par'):
        """Save current parameters to .par file"""
        logger.info(f"Saving parameters to file: {filename}")
        try:
            with open(filename, 'w') as f:
                f.write("# Temperature Time-Series Frequency Filter Parameters\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("#\n")
                f.write("# Format: parameter_name = value\n")
                f.write("# Lines starting with # are comments\n\n")
                
                f.write("# ===== General =====\n")
                f.write(f"filename_composite_data = {getattr(self, 'input_filename', 'temperature_composite_data.csv')}\n")
                f.write(f"data_interval_minutes = {self.filter_params['original_interval_minutes']}\n")
                f.write(f"time_bandwidth_parameter = {self.psd_params.get('time_bandwidth', 4)}\n\n")
                
                f.write("# ===== Spectral Analysis =====\n")
                f.write(f"wd_lower_limit = {getattr(self, 'wd_lower', -1)}\n")
                f.write(f"wd_upper_limit = {getattr(self, 'wd_upper', -1)}\n\n")
                
                f.write("# ===== Filter =====\n")
                f.write(f"start_band_pass = {self.filter_params['f_low']}\n")
                f.write(f"end_band_pass = {self.filter_params['f_high']}\n")
                f.write(f"ramp_fraction = {self.filter_params['ramp_fraction']}\n")
                f.write(f"filter_order = {self.filter_params['filter_order']}\n")
                f.write(f"# IMPORTANT: Set resample_interval = 1 for 20:1 resolution improvement\n")
                f.write(f"# Use 1 for maximum resolution, 20 for no resampling\n")
                f.write(f"resample_interval = {self.filter_params['resample_interval_minutes']}\n\n")
                
                f.write("# ===== Output =====\n")
                f.write(f"# Output folder name for filtered data files\n")
                f.write(f"output_folder = {self.output_folder}\n")
            
            logger.info(f"Parameters successfully saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving parameter file: {str(e)}")
            return False

    def create_output_folder(self):
        """Create output folder if it doesn't exist"""
        try:
            # Use the output folder as is (could be relative or absolute)
            if not os.path.isabs(self.output_folder):
                # If relative, make it relative to script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                full_output_path = os.path.join(script_dir, self.output_folder)
            else:
                full_output_path = self.output_folder
            
            if not os.path.exists(full_output_path):
                os.makedirs(full_output_path)
                logger.info(f"Created output folder: {full_output_path}")
            else:
                logger.debug(f"Output folder already exists: {full_output_path}")
            
            # Update the output_folder to use the full path
            self.output_folder = full_output_path
            return True
        except Exception as e:
            logger.error(f"Error creating output folder: {str(e)}")
            return False
    
    def set_output_folder(self, folder_path):
        """Set a new output folder"""
        if folder_path and os.path.isdir(folder_path):
            self.output_folder = folder_path
            logger.info(f"Output folder updated to: {self.output_folder}")
            return True
        logger.warning(f"Invalid output folder path: {folder_path}")
        return False
        
    def load_data(self, filepath_or_contents, filename=None):
        """Load temperature data from CSV file or uploaded content with better error handling"""
        logger.info(f"Loading data from: {filename or 'uploaded content'}")
        try:
            if isinstance(filepath_or_contents, str) and os.path.exists(filepath_or_contents):
                # File path provided
                logger.debug(f"Loading from file path: {filepath_or_contents}")
                self.data = pd.read_csv(filepath_or_contents)
            else:
                # Uploaded content (base64 encoded)
                logger.debug("Loading from uploaded content")
                if ',' not in filepath_or_contents:
                    raise ValueError("Invalid file content format")
                    
                content_type, content_string = filepath_or_contents.split(',', 1)
                decoded = base64.b64decode(content_string)
                
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        content_str = decoded.decode(encoding)
                        self.data = pd.read_csv(io.StringIO(content_str))
                        logger.debug(f"Successfully decoded with {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode file with any standard encoding")
            
            logger.debug(f"Initial data shape: {self.data.shape}")
            logger.debug(f"Columns found: {list(self.data.columns)}")
            
            # Validate required columns
            required_cols = ['WaterDay', 'TempShallow', 'TempDeep']
            if not all(col in self.data.columns for col in required_cols):
                # Try alternative column names
                alt_names = {
                    'Shallow.Temp': 'TempShallow',
                    'Deep.Temp': 'TempDeep'
                }
                for old_name, new_name in alt_names.items():
                    if old_name in self.data.columns:
                        self.data.rename(columns={old_name: new_name}, inplace=True)
                        logger.info(f"Renamed column: {old_name} -> {new_name}")
            
            # Ensure we have the required columns
            if not all(col in self.data.columns for col in required_cols):
                error_msg = f"CSV must contain columns: {required_cols}. Found: {list(self.data.columns)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Apply WD limits if set
            if self.wd_lower > 0 or self.wd_upper > 0:
                original_length = len(self.data)
                if self.wd_lower > 0:
                    self.data = self.data[self.data['WaterDay'] >= self.wd_lower]
                if self.wd_upper > 0:
                    self.data = self.data[self.data['WaterDay'] <= self.wd_upper]
                logger.info(f"Applied WD limits: {self.wd_lower} to {self.wd_upper}")
                logger.info(f"Data reduced from {original_length} to {len(self.data)} records")
                
            # Log data statistics
            wd_min, wd_max = self.data['WaterDay'].min(), self.data['WaterDay'].max()
            temp_shallow_range = (self.data['TempShallow'].min(), self.data['TempShallow'].max())
            temp_deep_range = (self.data['TempDeep'].min(), self.data['TempDeep'].max())
            
            logger.info(f"Data loaded successfully: {len(self.data)} records")
            logger.info(f"WaterDay range: {wd_min:.3f} to {wd_max:.3f}")
            logger.info(f"TempShallow range: {temp_shallow_range[0]:.3f}C to {temp_shallow_range[1]:.3f}C")
            logger.info(f"TempDeep range: {temp_deep_range[0]:.3f}C to {temp_deep_range[1]:.3f}C")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.data = None
            return False
    
    def remove_trend_advanced(self, data_col, trend_type='none', return_trend=False):
        """Enhanced trend removal with multiple options to address DC offset issues"""
        logger.debug(f"Removing trend using method: {trend_type}")
        data_clean = np.array(data_col)
        trend = np.zeros_like(data_clean)
        
        if trend_type == 'none':
            detrended = data_clean
        elif trend_type == 'dc':
            # Remove DC offset (mean)
            mean_val = np.mean(data_clean)
            trend = np.full_like(data_clean, mean_val)
            detrended = data_clean - trend
            logger.debug(f"Removed DC offset: {mean_val:.6f}")
        elif trend_type == 'linear':
            # Remove linear trend
            x = np.arange(len(data_clean))
            coeffs = np.polyfit(x, data_clean, 1)
            trend = np.polyval(coeffs, x)
            detrended = data_clean - trend
            logger.debug(f"Removed linear trend: slope={coeffs[0]:.6f}, intercept={coeffs[1]:.6f}")
        elif trend_type == 'polynomial':
            # Remove polynomial trend (order 2)
            x = np.arange(len(data_clean))
            coeffs = np.polyfit(x, data_clean, 2)
            trend = np.polyval(coeffs, x)
            detrended = data_clean - trend
            logger.debug(f"Removed polynomial trend (order 2)")
        elif trend_type == 'highpass':
            # High-pass filter to remove low-frequency drift
            from scipy.signal import butter, filtfilt
            nyquist = self.sampling_freq / 2
            cutoff = 0.1 / nyquist  # 0.1 day^-1 normalized
            b, a = butter(3, cutoff, btype='high')
            detrended = filtfilt(b, a, data_clean)
            trend = data_clean - detrended
            logger.debug(f"Applied high-pass filter with cutoff: {0.1} day^-1")
        elif trend_type == 'moving_average':
            # Remove moving average trend (window = 3 days)
            window_size = int(3 * self.sampling_freq)  # 3 days
            from scipy.ndimage import uniform_filter1d
            trend = uniform_filter1d(data_clean, size=window_size, mode='nearest')
            detrended = data_clean - trend
            logger.debug(f"Removed moving average trend with window: {window_size} samples (3 days)")
        else:
            detrended = data_clean
        
        if return_trend:
            return detrended, trend
        return detrended

    def validate_filter_parameters(self):
        """Validate filter parameters according to specifications"""
        logger.debug("Validating filter parameters")
        warnings = []
        errors = []
        
        if self.data is None:
            errors.append("No data loaded for validation")
            return {'valid': False, 'warnings': warnings, 'errors': errors}
        
        data_length = len(self.data)
        data_duration_days = data_length / self.sampling_freq
        
        # Check filter order validity first
        filter_order = self.filter_params.get('filter_order')
        if filter_order is None:
            errors.append("Filter order is required")
            return {'valid': False, 'warnings': warnings, 'errors': errors}
        
        if filter_order < 2:
            errors.append("Filter order must be at least 2 for bandpass filters")
            return {'valid': False, 'warnings': warnings, 'errors': errors}
        
        # Calculate filter impulse response duration
        filter_duration_days = filter_order / (self.sampling_freq * 2)
        
        # Check edge effect issues (specs mention 3-day minimum)
        min_edge_days = 3
        if filter_duration_days > min_edge_days:
            warnings.append(f"Filter order {filter_order} may cause edge effects. "
                           f"Consider reducing order or ensuring data extends {filter_duration_days:.1f} days "
                           f"beyond analysis period.")
        
        if data_duration_days < 2 * filter_duration_days + 1:
            warnings.append(f"Data length ({data_duration_days:.1f} days) may be too short for "
                           f"filter order {filter_order}. Consider longer data or lower order.")
        
        # Check frequency band validity
        f_low = self.filter_params.get('f_low')
        f_high = self.filter_params.get('f_high')
        
        if f_low is None or f_high is None:
            errors.append("Low and high frequencies are required")
            return {'valid': False, 'warnings': warnings, 'errors': errors}
        
        nyquist = self.sampling_freq / 2
        
        if f_low >= f_high:
            errors.append("Low frequency must be less than high frequency")
        
        if f_high >= nyquist:
            errors.append(f"High frequency ({f_high:.3f}) must be less than Nyquist frequency ({nyquist:.3f})")
        
        if f_low <= 0:
            errors.append("Low frequency must be positive")
        
        # Check ramp fraction
        ramp_fraction = self.filter_params.get('ramp_fraction')
        if ramp_fraction is None:
            errors.append("Ramp fraction is required")
        elif ramp_fraction < 0 or ramp_fraction > 0.5:
            errors.append("Ramp fraction must be between 0 and 0.5")
        
        # Warn about diurnal frequency optimization
        if f_low > 1.2 or f_high < 0.8:
            warnings.append("Filter band may not capture diurnal (daily) cycles optimally. "
                           "Consider including 0.8-1.2 day^-1 range.")
        
        # NEW: Check resampling configuration and warn about resolution
        original_interval = self.filter_params.get('original_interval_minutes', 20)
        resample_interval = self.filter_params.get('resample_interval_minutes', 1)
        
        if resample_interval == original_interval:
            warnings.append(f"No resampling configured. Consider setting resample_interval to 1 minute "
                           f"for {original_interval}:1 resolution improvement in peak/trough detection.")
        elif resample_interval < original_interval:
            improvement = original_interval / resample_interval
            if improvement >= 10:
                warnings.append(f"High resolution resampling: {improvement:.1f}:1 improvement configured. "
                               f"This will enhance peak/trough detection accuracy.")
        
        validation_result = {
            'valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors,
            'filter_duration_days': filter_duration_days,
            'data_duration_days': data_duration_days
        }
        
        logger.debug(f"Validation result: {len(errors)} errors, {len(warnings)} warnings")
        return validation_result

    def design_filter_with_ramp(self, f_low, f_high, fs, filter_type='butter', order=6, ramp_fraction=0.1):
        """Design bandpass filter with explicit ramp/taper implementation matching specifications"""
        logger.info(f"Designing {filter_type} filter: {f_low:.3f}-{f_high:.3f} day^-1, order={order}, ramp={ramp_fraction}")
        
        nyquist = fs / 2
        
        # Calculate ramp frequencies based on ramp_fraction
        band_width = f_high - f_low
        ramp_width = band_width * ramp_fraction
        
        # Define the four critical frequencies as per Table 2
        f_start_ramp = f_low - ramp_width/2      # Start ramp up
        f_start_pass = f_low + ramp_width/2      # Start full pass  
        f_end_pass = f_high - ramp_width/2       # End full pass, start ramp down
        f_end_ramp = f_high + ramp_width/2       # End ramp down
        
        # Normalize by Nyquist frequency
        f_start_ramp_norm = f_start_ramp / nyquist
        f_start_pass_norm = f_start_pass / nyquist  
        f_end_pass_norm = f_end_pass / nyquist
        f_end_ramp_norm = f_end_ramp / nyquist
        
        # Ensure frequencies are within valid range [0, 1]
        f_start_ramp_norm = max(0.01, f_start_ramp_norm)
        f_end_ramp_norm = min(0.99, f_end_ramp_norm)
        
        logger.debug(f"Normalized frequencies: {f_start_ramp_norm:.6f} to {f_end_ramp_norm:.6f}")
        
        if filter_type == 'butter':
            b, a = signal.butter(order, [f_start_ramp_norm, f_end_ramp_norm], btype='band')
        elif filter_type == 'cheby1':
            b, a = signal.cheby1(order, 1, [f_start_ramp_norm, f_end_ramp_norm], btype='band')
        elif filter_type == 'cheby2':
            b, a = signal.cheby2(order, 40, [f_start_ramp_norm, f_end_ramp_norm], btype='band')
        elif filter_type == 'ellip':
            b, a = signal.ellip(order, 1, 40, [f_start_ramp_norm, f_end_ramp_norm], btype='band')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Store filter characteristics for user information
        filter_info = {
            'f_start_ramp': f_start_ramp,
            'f_start_pass': f_start_pass,
            'f_end_pass': f_end_pass,
            'f_end_ramp': f_end_ramp,
            'ramp_width': ramp_width,
            'pass_width': f_end_pass - f_start_pass
        }
        
        logger.debug(f"Filter designed successfully with ramp from {f_start_ramp:.6f} to {f_end_ramp:.6f} day^-1")
        return b, a, filter_info
    
    def compute_psd_welch(self, data_col, custom_params):
        """Compute PSD using Welch's method with proper mathematical corrections"""
        logger.debug("Computing PSD using Welch method")
        
        # Ensure all parameters have valid values (not None)
        params = custom_params.copy()
        params['nperseg_divider'] = params.get('nperseg_divider') or 4
        params['nperseg_max'] = params.get('nperseg_max') or 520
        
        # Calculate nperseg based on user formula
        nperseg = min(len(data_col) // int(params['nperseg_divider']), int(params['nperseg_max']))
        
        # Ensure nperseg is at least a reasonable minimum
        nperseg = max(nperseg, 16)
        
        logger.debug(f"Welch method: nperseg={nperseg} from data length: {len(data_col)}")
        
        # Handle optional parameters - only pass them if they have valid values
        welch_params = {
            'x': data_col,
            'fs': self.sampling_freq,
            'window': params.get('window', 'tukey'),
            'nperseg': nperseg,
            'scaling': params.get('scaling', 'density'),
            'average': params.get('average', 'mean'),
            'return_onesided': params.get('return_onesided', True),
            'detrend': params.get('detrend', 'constant')
        }
        
        # Add optional parameters only if they have non-None values
        if params.get('noverlap') is not None:
            welch_params['noverlap'] = int(params['noverlap'])
        
        if params.get('nfft') is not None:
            welch_params['nfft'] = int(params['nfft'])
        
        # Call Welch method with parameters
        frequencies, psd = signal.welch(**welch_params)
        
        logger.debug(f"Welch PSD computed: {len(frequencies)} frequency bins, max_freq={frequencies.max():.3f}")
        
        return frequencies, psd
    
    def compute_psd_welch_smooth(self, data_col, custom_params):
        """Compute PSD using Welch's method with heavy smoothing for R-like output"""
        logger.debug("Computing PSD using Welch smooth method")
        params = custom_params.copy()
        
        data_len = len(data_col)
        
        # Check if user wants to use the divider/max formula for smooth method too
        use_user_formula = params.get('use_divider_for_smooth', False)
        
        if use_user_formula:
            # Use the same calculation as standard Welch
            nperseg_divider = params.get('nperseg_divider', 4)
            nperseg_max = params.get('nperseg_max', 520)
            
            if nperseg_divider is None:
                nperseg_divider = 4
            if nperseg_max is None:
                nperseg_max = 520
                
            nperseg = min(data_len // int(nperseg_divider), int(nperseg_max))
            nperseg = max(nperseg, 16)
            
            logger.debug(f"Smooth method using user formula: nperseg = min({data_len} // {nperseg_divider}, {nperseg_max}) = {nperseg}")
        else:
            # Use the original logic for optimal smoothness
            if data_len > 10000:
                nperseg = 2048
            elif data_len > 5000:
                nperseg = 1024
            elif data_len > 2000:
                nperseg = 512
            else:
                nperseg = min(data_len // 4, 256)
            
            logger.debug(f"Smooth method using optimal logic for {data_len} points: nperseg = {nperseg}")
        
        # Very high overlap for smoothness
        noverlap = int(nperseg * 0.95)
        
        # Match R's high nfft
        nfft = min(100000, data_len * 10)  # Very high, but not larger than needed
        
        welch_params = {
            'x': data_col,
            'fs': self.sampling_freq,
            'window': ('tukey', 0.25),  # Tukey window with low taper for smoothness
            'nperseg': nperseg,
            'noverlap': noverlap,
            'nfft': nfft,
            'scaling': params.get('scaling', 'density'),
            'detrend': params.get('detrend', 'constant'),
            'return_onesided': params.get('return_onesided', True)
        }
        
        # Remove average parameter as it might not be supported in all scipy versions
        frequencies, psd = signal.welch(**welch_params)
        
        # Apply multiple smoothing passes for ultra-smooth output
        sigma = params.get('smooth_sigma', 3)
        
        # First pass - moderate smoothing
        psd_smooth = gaussian_filter1d(psd, sigma=sigma)
        
        # Second pass - stronger smoothing in log space
        log_psd = np.log10(np.maximum(psd_smooth, 1e-10))
        log_psd_smooth = gaussian_filter1d(log_psd, sigma=sigma*2)
        psd_smooth = 10**log_psd_smooth
        
        # Optional: Reduce high-frequency noise further
        if len(frequencies) > 1000:
            # Apply extra smoothing to high frequencies
            high_freq_start = len(frequencies) // 2
            psd_smooth[high_freq_start:] = gaussian_filter1d(psd_smooth[high_freq_start:], sigma=sigma*3)
        
        logger.debug(f"Applied smoothing with sigma={sigma}")
        return frequencies, psd_smooth
    
    def compute_psd_multitaper(self, data_col, custom_params):
        """Compute PSD using multitaper method with proper zero-padding to match R implementation"""
        if not MNE_AVAILABLE:
            logger.warning("Multitaper method not available. Using corrected MATLAB-style instead.")
            return self.compute_psd_matlab_style_corrected(data_col, custom_params)
        
        logger.debug("Computing PSD using multitaper method")
        
        # Multitaper parameters
        bandwidth = custom_params.get('time_bandwidth', 4)
        adaptive = custom_params.get('adaptive', True)
        low_bias = custom_params.get('low_bias', True)
        nfft = custom_params.get('matlab_nfft', 2048)  # Use the same nfft as MATLAB method
        
        # Zero-pad data to match nfft length before MNE processing
        data_len = len(data_col)
        if nfft > data_len:
            # Zero-pad to nfft length
            padded_data = np.pad(data_col, (0, nfft - data_len), 'constant')
            logger.debug(f"Multitaper: Zero-padded data from {data_len} to {nfft} samples")
        else:
            padded_data = data_col[:nfft]  # Truncate if data is longer than nfft
            logger.debug(f"Multitaper: Truncated data from {data_len} to {nfft} samples")
        
        logger.debug(f"Multitaper method: bandwidth={bandwidth}, nfft={nfft}, adaptive={adaptive}")
        
        # Compute multitaper PSD with zero-padded data
        psd, freqs = psd_array_multitaper(
            padded_data.reshape(1, -1),  # MNE expects 2D array
            sfreq=self.sampling_freq,
            bandwidth=bandwidth,
            adaptive=adaptive,  # This enables Thomson's adaptive weighting
            low_bias=low_bias,
            n_jobs=1,
            verbose=False
        )
        
        logger.debug(f"Multitaper PSD computed: {len(freqs)} frequency bins")
        return freqs, psd[0]  # Return 1D array
    
    def compute_psd_matlab_style_corrected(self, data_col, custom_params):
        """Compute PSD using DPSS tapers with proper mathematical corrections to match MATLAB spectrum.mtm behavior"""
        from scipy.signal.windows import dpss as scipy_dpss
        from scipy.signal import periodogram
        
        logger.debug("Computing PSD using MATLAB-style multitaper method")
        
        # MATLAB-style parameters
        tbw = custom_params.get('time_bandwidth', 4)
        nfft = custom_params.get('matlab_nfft', 2048)
        
        # Check if user wants to override nfft with their segment parameters
        use_user_segments = custom_params.get('use_divider_for_matlab', False)
        if use_user_segments:
            nperseg_divider = custom_params.get('nperseg_divider', 4)
            nperseg_max = custom_params.get('nperseg_max', 2048)
            if nperseg_divider and nperseg_max:
                user_nfft = min(len(data_col) // int(nperseg_divider), int(nperseg_max))
                nfft = max(user_nfft, 512)
                logger.debug(f"MATLAB MTM using user nfft: {nfft}")
        
        # Prevent silent truncation - ensure nfft >= data length
        N = len(data_col)
        if nfft < N:
            logger.warning(f"nfft ({nfft}) < data length ({N}). Increasing nfft to {N} to avoid truncation.")
            nfft = N
        
        # Calculate number of tapers (MATLAB uses K = 2*NW - 1)
        n_tapers = int(2 * tbw - 1)
        
        logger.debug(f"MATLAB MTM Corrected: N={N}, tbw={tbw}, n_tapers={n_tapers}, nfft={nfft}")
        
        # Generate DPSS (Discrete Prolate Spheroidal Sequences) tapers with eigenvalues
        try:
            # Get eigenvalues for adaptive weighting
            tapers, eigenvalues = scipy_dpss(N, tbw, n_tapers, return_ratios=True)
            
            # Handle scipy DPSS output shape properly
            if tapers.ndim == 1:
                # Single taper case
                tapers = tapers.reshape(1, -1)
            elif tapers.shape[0] == N:
                # Transpose if tapers are [N, n_tapers] instead of [n_tapers, N]
                tapers = tapers.T
                
            logger.debug(f"Generated {n_tapers} DPSS tapers with shape {tapers.shape}")
            logger.debug(f"Eigenvalues: {eigenvalues[:min(3, len(eigenvalues))]}")
            
            # Ensure we have the right number of tapers
            if tapers.shape[0] != n_tapers:
                logger.warning(f"Expected {n_tapers} tapers, got {tapers.shape[0]}")
                n_tapers = min(n_tapers, tapers.shape[0])
                eigenvalues = eigenvalues[:n_tapers]
                
        except Exception as e:
            logger.error(f"Failed to generate DPSS tapers: {e}")
            # Fallback to approximation
            tapers = self._create_dpss_approximation(N, tbw, n_tapers)
            eigenvalues = np.ones(n_tapers)  # Equal weights as fallback
            logger.debug(f"Using fallback tapers with shape {tapers.shape}")
        
        # Initialize arrays for storing individual taper PSDs
        taper_psds = []
        
        # Proper DPSS taper normalization - pass taper to periodogram
        for k in range(n_tapers):
            # Get the k-th taper
            taper = tapers[k, :]  # Shape should be [n_tapers, N]
            
            # Verify taper shape matches data
            if len(taper) != len(data_col):
                logger.error(f"Taper {k} length {len(taper)} doesn't match data length {len(data_col)}")
                continue
            
            # Pass the taper directly to periodogram for proper normalization
            freqs, psd_k = periodogram(
                data_col,                    # Use original data, not pre-tapered
                fs=self.sampling_freq,
                window=taper,                # Pass the DPSS taper here for proper normalization
                nfft=nfft,
                scaling='density',
                detrend='constant',
                return_onesided=True
            )
            
            taper_psds.append(psd_k)
        
        if not taper_psds:
            raise ValueError("No valid taper PSDs computed")
        
        # Convert to numpy array for easier manipulation
        taper_psds = np.array(taper_psds)  # Shape: [n_tapers, n_freq_bins]
        
        logger.debug(f"Computed {len(taper_psds)} taper PSDs with shape {taper_psds.shape}")
        
        # Apply Thomson's adaptive weighting using eigenvalues
        if len(eigenvalues) == len(taper_psds):
            # Reshape eigenvalues to [n_tapers, 1] for proper broadcasting
            eigenvalues_reshaped = eigenvalues[:len(taper_psds)].reshape(-1, 1)
            
            # Eigenvalue-weighted average with proper broadcasting
            weighted_sum = np.sum(eigenvalues_reshaped * taper_psds, axis=0)
            weight_sum = np.sum(eigenvalues[:len(taper_psds)])
            psd_final = weighted_sum / weight_sum
            
            logger.debug(f"Applied eigenvalue weighting: sum(eigenvalues)={weight_sum:.3f}")
        else:
            # Fallback to simple average
            psd_final = np.mean(taper_psds, axis=0)
            logger.warning(f"Used simple averaging (eigenvalue mismatch: {len(eigenvalues)} vs {len(taper_psds)})")
        
        return freqs, psd_final

    def _create_dpss_approximation(self, N, tbw, n_tapers):
        """Create approximate DPSS tapers using sine tapers when scipy.signal.windows.dpss is not available"""
        logger.debug(f"Creating DPSS approximation: N={N}, tbw={tbw}, n_tapers={n_tapers}")
        tapers = np.zeros((n_tapers, N))
        
        for k in range(n_tapers):
            # Create sine taper approximation
            if k == 0:
                # First taper - similar to a Tukey window
                taper = np.ones(N)
                # Apply cosine taper at ends
                taper_length = int(N * 0.1)  # 10% taper
                if taper_length > 0:
                    taper_window = 0.5 * (1 - np.cos(np.pi * np.arange(taper_length) / taper_length))
                    taper[:taper_length] *= taper_window
                    taper[-taper_length:] *= taper_window[::-1]
            else:
                # Higher order tapers - use sine functions
                taper = np.sqrt(2/N) * np.sin((k * np.pi * (np.arange(N) + 0.5)) / N)
                # Apply additional windowing for better frequency concentration
                window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
                taper *= window
            
            # Normalize
            tapers[k] = taper / np.sqrt(np.sum(taper**2))
        
        return tapers

    def compute_psd(self, data_col=None, use_clipped=True, custom_params=None):
        """Compute Power Spectral Density using selected method with mathematical corrections"""
        # Determine which dataset to use
        if use_clipped and hasattr(self, 'analysis_data') and self.analysis_data is not None:
            data_source = self.analysis_data
        else:
            data_source = self.data
        
        if data_col is None:
            return None, None
        
        # If data_col is a string (column name), extract from data_source
        if isinstance(data_col, str) and data_source is not None:
            data_col = data_source[data_col].values
        
        if data_col is None or len(data_col) == 0:
            return np.array([]), np.array([])
        
        # Use custom parameters if provided, otherwise use defaults
        params = custom_params if custom_params else self.psd_params.copy()
        
        # Select method
        method = params.get('method', 'matlab_mtm')
        
        logger.debug(f"Computing PSD using method: {method} for {len(data_col)} data points")
        
        if method == 'multitaper':
            return self.compute_psd_multitaper(data_col, params)
        elif method == 'welch_smooth':
            return self.compute_psd_welch_smooth(data_col, params)
        elif method == 'matlab_mtm':
            return self.compute_psd_matlab_style_corrected(data_col, params)
        else:  # Default to standard Welch
            return self.compute_psd_welch(data_col, params)

    def format_output_data(self):
        """Format filtered data according to exact specification format"""
        if self.filtered_data is None:
            return None
        
        logger.debug("Formatting output data to specification format")
        
        # Create output dataframe with exact column names from specification
        output_data = pd.DataFrame()
        
        # WaterDay with 9 decimal places precision (as shown in spec example)
        output_data['WaterDay'] = self.filtered_data['WaterDay'].round(9)
        
        # Filtered temperature columns with exact naming from specification
        output_data['Shallow.Temp.Filt'] = self.filtered_data['TempShallow_Filt'].round(6)
        output_data['Deep.Temp.Filt'] = self.filtered_data['TempDeep_Filt'].round(6)
        
        logger.debug(f"Formatted output data: {len(output_data)} rows, columns: {list(output_data.columns)}")
        return output_data

    def apply_filter_with_validation(self, update_params=None):
        """Enhanced filter application with validation and proper trend handling"""
        if self.data is None:
            error_msg = "No data loaded"
            logger.error(error_msg)
            return False, error_msg

        if update_params:
            self.filter_params.update(update_params)
            logger.debug(f"Updated filter parameters: {update_params}")

        # Validate parameters first
        validation = self.validate_filter_parameters()
        if not validation['valid']:
            error_msg = f"Validation failed: {'; '.join(validation['errors'])}"
            logger.error(error_msg)
            return False, error_msg

        try:
            logger.info("Applying filter to temperature data")
            
            # Remove trend first if requested
            shallow_data, shallow_trend = self.remove_trend_advanced(
                self.data['TempShallow'].values,
                self.filter_params['trend_removal'],
                return_trend=True
            )
            deep_data, deep_trend = self.remove_trend_advanced(
                self.data['TempDeep'].values,
                self.filter_params['trend_removal'],
                return_trend=True
            )

            # Design filter with ramp implementation
            b, a, filter_info = self.design_filter_with_ramp(
                self.filter_params['f_low'],
                self.filter_params['f_high'],
                self.sampling_freq,
                self.filter_params['filter_type'],
                self.filter_params['filter_order'],
                self.filter_params['ramp_fraction']
            )

            # Apply zero-phase filtering
            logger.debug("Applying zero-phase filtering")
            shallow_filt = signal.filtfilt(b, a, shallow_data)
            deep_filt = signal.filtfilt(b, a, deep_data)

            # ENHANCED: Handle resampling with improved user feedback
            original_interval = self.filter_params['original_interval_minutes']
            resample_interval = self.filter_params['resample_interval_minutes']
            
            if resample_interval != original_interval:
                # Calculate the new length based on the interval ratio
                original_len = len(self.data)
                ratio = original_interval / resample_interval
                new_length = int(original_len * ratio)
                
                # Generate a new, correctly scaled time axis
                start_time = self.data['WaterDay'].iloc[0]
                end_time = self.data['WaterDay'].iloc[-1]
                new_time_axis = np.linspace(start_time, end_time, num=new_length)
                
                # Resample all data columns to the new length
                def resample_array(arr):
                    return signal.resample(arr, new_length)
                
                # Build the new DataFrame with aligned data
                self.filtered_data = pd.DataFrame({
                    'WaterDay': new_time_axis,
                    'TempShallow': resample_array(self.data['TempShallow'].values),
                    'TempDeep': resample_array(self.data['TempDeep'].values),
                    'TempShallow_Filt': resample_array(shallow_filt),
                    'TempDeep_Filt': resample_array(deep_filt),
                    'TempShallow_Trend': resample_array(shallow_trend),
                    'TempDeep_Trend': resample_array(deep_trend)
                })
                
                # Enhanced user feedback
                improvement_factor = ratio
                logger.info("RESAMPLING APPLIED:")
                logger.info(f"  Original: {original_len} points at {original_interval}-minute intervals")
                logger.info(f"  New: {new_length} points at {resample_interval}-minute intervals")
                logger.info(f"  Resolution improvement: {improvement_factor:.1f}:1")
                logger.info(f"  Time range preserved: {start_time:.3f} to {end_time:.3f} WD")
                
            else:
                # No resampling - use original data structure
                self.filtered_data = self.data.copy()
                self.filtered_data['TempShallow_Filt'] = shallow_filt
                self.filtered_data['TempDeep_Filt'] = deep_filt
                self.filtered_data['TempShallow_Trend'] = shallow_trend
                self.filtered_data['TempDeep_Trend'] = deep_trend
                
                logger.info(f"NO RESAMPLING: Maintaining original {len(self.data)} points at {original_interval}-minute intervals")

            # Store filter info for reference
            self.last_filter_info = filter_info
            
            # Enhanced success message
            success_msg = "Filter applied successfully."
            if resample_interval != original_interval:
                improvement = original_interval / resample_interval
                success_msg += f" Resolution improved {improvement:.1f}:1 ({original_interval}->{resample_interval} min)."
            
            if validation['warnings']:
                success_msg += f" {'; '.join(validation['warnings'])}"
                
            logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"Filtering failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        
    def resample_data(self, data, original_interval_minutes, target_interval_minutes):
        """Resample data using scipy.signal.resample"""
        from scipy.signal import resample
        
        logger.debug(f"Resampling data from {original_interval_minutes} to {target_interval_minutes} minute intervals")
        
        # Calculate resampling ratio
        ratio = original_interval_minutes / target_interval_minutes
        new_length = int(len(data) * ratio)
        
        if new_length != len(data):
            resampled = resample(data, new_length)
            logger.debug(f"Resampled from {len(data)} to {new_length} points")
            return resampled
        else:
            return data

    def save_filtered_data_spec_format(self, filename):
        """Save filtered data in exact specification format with folder management"""
        logger.info(f"Saving filtered data to: {filename}")
        
        if self.filtered_data is not None:
            # Create output folder if it doesn't exist (this will set the full path)
            if not self.create_output_folder():
                error_msg = "Failed to create output folder"
                logger.error(error_msg)
                return False, None
            
            # Now self.output_folder contains the full path, so just join with filename
            full_path = os.path.join(self.output_folder, filename)
            
            output_data = self.format_output_data()
            
            if output_data is not None:
                # Save with exact formatting - no index, specific precision
                output_data.to_csv(full_path, index=False, float_format='%.6f')
                
                # Enhanced metadata for filter history
                resampling_info = {
                    'original_interval_min': self.filter_params['original_interval_minutes'],
                    'resample_interval_min': self.filter_params['resample_interval_minutes'],
                    'resolution_improvement': self.filter_params['original_interval_minutes'] / self.filter_params['resample_interval_minutes']
                }
                
                # Add to filter history with additional metadata
                self.filter_history.append({
                    'filename': filename,
                    'full_path': full_path,
                    'timestamp': datetime.now().isoformat(),
                    'parameters': self.filter_params.copy(),
                    'resampling_info': resampling_info,
                    'data_range': {
                        'start_wd': float(output_data['WaterDay'].min()),
                        'end_wd': float(output_data['WaterDay'].max()),
                        'n_points': len(output_data)
                    },
                    'filter_validation': self.validate_filter_parameters()
                })
                
                logger.info(f"Filtered data saved to: {full_path}")
                if resampling_info['resolution_improvement'] > 1:
                    logger.info(f"Resolution improvement: {resampling_info['resolution_improvement']:.1f}:1")
                return True, full_path
        
        error_msg = "No filtered data available to save"
        logger.error(error_msg)
        return False, None
        
    def save_filtered_data(self, filename):
        """Legacy method for backward compatibility"""
        success, _ = self.save_filtered_data_spec_format(filename)
        return success

    def clip_data_for_analysis(self, wd_lower=None, wd_upper=None):
        """Clip data based on WaterDay limits for spectral analysis"""
        if self.data is None:
            return {'success': False, 'message': 'No data loaded'}
        
        logger.info(f"Clipping data for analysis: WD {wd_lower} to {wd_upper}")
        
        # Update stored limits
        if wd_lower is not None:
            self.wd_lower = wd_lower
        if wd_upper is not None:
            self.wd_upper = wd_upper
        
        # Create a copy of the data for analysis
        self.analysis_data = self.data.copy()
        
        # Apply clipping if limits are set
        original_length = len(self.analysis_data)
        
        if self.wd_lower > 0:
            self.analysis_data = self.analysis_data[self.analysis_data['WaterDay'] >= self.wd_lower]
            logger.debug(f"Applied lower WD limit: {self.wd_lower}")
        
        if self.wd_upper > 0:
            self.analysis_data = self.analysis_data[self.analysis_data['WaterDay'] <= self.wd_upper]
            logger.debug(f"Applied upper WD limit: {self.wd_upper}")
        
        clipped_length = len(self.analysis_data)
        
        # Calculate actual WD range
        if clipped_length > 0:
            actual_wd_min = self.analysis_data['WaterDay'].min()
            actual_wd_max = self.analysis_data['WaterDay'].max()
        else:
            actual_wd_min = actual_wd_max = 0
        
        logger.info(f"Data clipped from {original_length} to {clipped_length} records")
        logger.info(f"Actual WD range: {actual_wd_min:.3f} to {actual_wd_max:.3f}")
        
        return {
            'success': True,
            'original_length': original_length,
            'clipped_length': clipped_length,
            'actual_wd_range': (actual_wd_min, actual_wd_max),
            'message': f'Data clipped from {original_length} to {clipped_length} records'
        }


# Initialize the analyzer
logger.info("Initializing Temperature Frequency Analyzer")
analyzer = TemperatureFrequencyAnalyzer()

# Create Dash app with proper configuration
logger.info("Creating Dash application")
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP], 
                suppress_callback_exceptions=True)
app.title = "TTS FREQFILT - Temperature Time-Series Frequency Analysis and Filtering"

# Configure for better error handling
app.config.suppress_callback_exceptions = True

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Temperature Time-Series Frequency Analysis & Filtering", 
                   className="text-center mb-4"),
            html.Hr(),
        ])
    ]),
    
    # File upload section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Data Upload", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            # File upload area
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                        ], width=8),
                        dbc.Col([
                            # Default file section
                            html.Div([
                                html.Small(f"Default file: {analyzer.input_filename}", 
                                         className="text-muted d-block"),
                                dbc.Button("Load Default File", id='load-default-btn', 
                                         color="success" if analyzer.check_default_file_exists() else "secondary", 
                                         size="sm", className="mt-1 me-2"),
                                dbc.Button("Browse Different File", id='browse-file-btn', 
                                         color="primary", size="sm", className="mt-1")
                            ])
                        ], width=4)
                    ]),
                    html.Div(id='upload-status'),
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Raw Data Plot - immediately after upload
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='raw-data-plot-top', style={'height': '400px'})
        ], width=12)
    ], className="mb-4"),
    
    # PSD Analysis Parameters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("PSD Analysis Parameters", className="card-title"),
                    
                    # Basic parameters row
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Sampling Frequency (per day):"),
                            dbc.Input(id='sampling-freq', type='number', value=analyzer.sampling_freq, min=1, max=1000)
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Original Sample Interval (minutes):"),
                            dbc.Input(id='original-interval', type='number', 
                                    value=analyzer.filter_params['original_interval_minutes'], min=1, max=1440)
                        ], width=3),
                        dbc.Col([
                            dbc.Label("WD Lower Limit:"),
                            dbc.Input(id='wd-lower-limit', type='number', 
                                    value=analyzer.wd_lower, step=0.1,
                                    placeholder="-1 for no limit")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("WD Upper Limit:"),
                            dbc.Input(id='wd-upper-limit', type='number', 
                                    value=analyzer.wd_upper, step=0.1,
                                    placeholder="-1 for no limit")
                        ], width=3)
                    ]),
                    
                    html.Hr(),
                    
                    # PSD Method Selection
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("PSD Method:"),
                            dcc.Dropdown(
                                id='psd-method',
                                options=[
                                    {'label': 'Multitaper (MATLAB style)', 'value': 'matlab_mtm'},
                                    {'label': 'Multitaper (R style)' + (' - Not Available' if not MNE_AVAILABLE else ''), 
                                    'value': 'multitaper', 'disabled': not MNE_AVAILABLE},
                                    {'label': 'Welch (Standard)', 'value': 'welch'},
                                    {'label': 'Welch (Smoothed)', 'value': 'welch_smooth'}
                                ],
                                value='matlab_mtm'
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Apply Data Clipping", id='apply-clip-btn', 
                                    color='warning', className="me-2"),
                            dbc.Button("Enter Frequency Selection Mode", id='freq-select-btn', 
                                    color='info', className="me-2"),
                            dbc.Button("Reset View", id='reset-view-btn', color='secondary', className="me-2"),
                        ], width=6)
                    ], className="mb-3"),
                    
                    # Dynamic method-specific parameters
                    html.Div(id='method-specific-params'),
                    
                    html.Div(id='clip-status', className="mt-2")
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    # Power Spectral Density Plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='psd-plot', style={'height': '400px'})
        ], width=12)
    ], className="mb-4"),

    # Filter Parameters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Filter Parameters", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Filter Type:"),
                            dcc.Dropdown(
                                id='filter-type',
                                options=[
                                    {'label': 'Butterworth', 'value': 'butter'},
                                    {'label': 'Chebyshev I', 'value': 'cheby1'},
                                    {'label': 'Chebyshev II', 'value': 'cheby2'},
                                    {'label': 'Elliptic', 'value': 'ellip'}
                                ],
                                value='butter'
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Filter Order:"),
                            dbc.Input(id='filter-order', type='number', 
                                    value=analyzer.filter_params['filter_order'], min=2, max=20),
                            html.Small("Minimum 2 for bandpass filters. Higher = sharper cutoff but more ringing.", 
                                     className="text-muted")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Low Frequency (day^-1):"),
                            dbc.Input(id='f-low', type='number', 
                                    value=analyzer.filter_params['f_low'], min=0.01, max=10, step=0.01)
                        ], width=3),
                        dbc.Col([
                            dbc.Label("High Frequency (day^-1):"),
                            dbc.Input(id='f-high', type='number', 
                                    value=analyzer.filter_params['f_high'], min=0.01, max=10, step=0.01)
                        ], width=3)
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Ramp Fraction:"),
                            dbc.Input(id='ramp-fraction', type='number', 
                                    value=analyzer.filter_params['ramp_fraction'], min=0, max=0.5, step=0.01)
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Resample Interval (minutes):"),
                            dbc.Input(id='resample-interval', type='number', 
                                    value=analyzer.filter_params['resample_interval_minutes'], min=1, max=1440),
                            html.Small("Set to 1 for 20:1 resolution improvement", className="text-muted")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Trend Removal:"),
                            dcc.Dropdown(
                                id='trend-removal',
                                options=[
                                    {'label': 'None', 'value': 'none'},
                                    {'label': 'DC Offset (recommended)', 'value': 'dc'},
                                    {'label': 'Linear Trend', 'value': 'linear'},
                                    {'label': 'Polynomial Trend', 'value': 'polynomial'},
                                    {'label': 'High-pass Filter', 'value': 'highpass'},
                                    {'label': 'Moving Average', 'value': 'moving_average'}
                                ],
                                value='dc'
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Button("Apply Filter", id='apply-filter-btn', color='primary'),
                        ], width=3)
                    ]),
                    
                    # Filter validation display
                    html.Div(id='filter-validation-status', className="mt-3"),
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    # Filtered Data Plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='filtered-data-plot', style={'height': '400px'})
        ], width=12)
    ], className="mb-4"),
    
    # Raw Data Plot for comparison
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='raw-data-plot', style={'height': '400px'})
        ], width=12)
    ], className="mb-4"),
    
    # Hidden div to store x-axis range for synchronization
    html.Div(id='x-axis-range-store', style={'display': 'none'}),

    # Export Parameters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Export Parameters", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Output Filename:"),
                            dbc.Input(id='output-filename', type='text', 
                                    placeholder="filtered_temperature_data.csv",
                                    value="filtered_temperature_data.csv")
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Output Folder:"),
                            dbc.InputGroup([
                                dbc.Input(id='output-folder-display', type='text', 
                                        value=analyzer.output_folder, disabled=True),
                                dbc.Button("Change", id='change-folder-btn', color='secondary')
                            ])
                        ], width=5),
                        dbc.Col([
                            dbc.Button("Save Filtered Data", id='save-data-btn', 
                                     color='success', className="me-2 mt-4"),
                            dbc.Button("Save Parameters", id='save-params-btn', 
                                     color='secondary', className="mt-4"),
                        ], width=3)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Filter Characteristics Display
    dbc.Row([
        dbc.Col([
            html.Div(id='filter-characteristics-card')
        ], width=12)
    ], className="mb-4"),
    
    # Status and frequency selection info
    dbc.Row([
        dbc.Col([
            html.Div(id='frequency-selection-status'),
            html.Div(id='filter-history-display'),
            html.Div(id='param-save-status')
        ], width=12)
    ], className="mb-4"),
    
    # Hidden div to store data
    html.Div(id='data-store', style={'display': 'none'}),
    html.Div(id='frequency-selection-state', children='inactive', style={'display': 'none'}),
    html.Div(id='output-folder-store', children=analyzer.output_folder, style={'display': 'none'}),
    
    # Download components (only for parameters now)
    dcc.Download(id="download-parameters"),
    
], fluid=True)


# Callback for loading default file
@app.callback(
    [Output('upload-status', 'children'),
     Output('data-store', 'children')],
    [Input('upload-data', 'contents'),
     Input('load-default-btn', 'n_clicks'),
     Input('browse-file-btn', 'n_clicks')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def upload_or_load_default(contents, load_clicks, browse_clicks, filename):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return "", "no-data"
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id == 'load-default-btn' and load_clicks:
            if DEBUG_CALLBACKS:
                print("Loading default file requested")
            # Load default file from parameters - search in multiple locations
            search_paths = [
                analyzer.input_filename,  # As specified (could be relative or absolute)
                os.path.join(os.getcwd(), analyzer.input_filename),  # Current working directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), analyzer.input_filename)  # Script directory
            ]
            
            file_found = False
            actual_path = None
            for path in search_paths:
                if os.path.exists(path):
                    file_found = True
                    actual_path = path
                    break
            
            if file_found:
                success = analyzer.load_data(actual_path)
                if success:
                    status = dbc.Alert(f"Successfully loaded default file: {analyzer.input_filename}", color="success")
                    logger.info(f"Default file loaded successfully from: {actual_path}")
                    return status, "data-loaded"
                else:
                    status = dbc.Alert(f"Error loading default file: {analyzer.input_filename}", color="danger")
                    return status, "no-data"
            else:
                status = dbc.Alert(f"Default file not found: {analyzer.input_filename}", color="warning")
                logger.warning(f"Default file not found: {analyzer.input_filename}")
                return status, "no-data"
        
        elif triggered_id == 'browse-file-btn' and browse_clicks:
            if DEBUG_CALLBACKS:
                print("Browse file dialog requested")
            # Open file browser for different file
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            file_path = filedialog.askopenfilename(
                title="Select Temperature Data File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            root.destroy()
            
            if file_path:
                success = analyzer.load_data(file_path)
                if success:
                    status = dbc.Alert(f"Successfully loaded: {os.path.basename(file_path)}", color="success")
                    logger.info(f"File loaded successfully from browser: {file_path}")
                    return status, "data-loaded"
                else:
                    status = dbc.Alert(f"Error loading file: {os.path.basename(file_path)}", color="danger")
                    return status, "no-data"
            else:
                if DEBUG_CALLBACKS:
                    print("File browser cancelled")
                return "", "no-data"
        
        elif triggered_id == 'upload-data' and contents is not None:
            if DEBUG_CALLBACKS:
                print(f"File upload requested: {filename}")
            # Load uploaded file
            success = analyzer.load_data(contents, filename)
            if success:
                status = dbc.Alert(f"Successfully loaded {filename}", color="success")
                return status, "data-loaded"
            else:
                status = dbc.Alert("Error loading file. Please check format.", color="danger")
                return status, "no-data"
        
        return "", "no-data"
        
    except Exception as e:
        print(f"Upload callback error: {e}")
        import traceback
        traceback.print_exc()
        
        error_status = dbc.Alert(f"Error loading file: {str(e)}", color="danger")
        return error_status, "no-data"


# Callback for method-specific parameters
@app.callback(
    Output('method-specific-params', 'children'),
    [Input('psd-method', 'value')],
    prevent_initial_call=False
)
def update_method_params(method):
    # Add comprehensive None checking
    if method is None:
        method = 'matlab_mtm'  # Safe default
    
    try:
        if method == 'matlab_mtm':
            return dbc.Row([
                dbc.Col([
                    dbc.Label("Time-Bandwidth (tbw):"),
                    dbc.Input(
                        id={'type': 'method-param', 'param': 'matlab-tbw'}, 
                        type='number', 
                        value=analyzer.psd_params.get('time_bandwidth', 4),
                        min=2, max=10, step=1
                    ),
                    html.Small("MATLAB tbw parameter - use 4 to match R/MATLAB", className="text-muted")
                ], width=4),
                dbc.Col([
                    dbc.Label("NFFT:"),
                    dbc.Input(
                        id={'type': 'method-param', 'param': 'matlab-nfft'}, 
                        type='number', 
                        value=analyzer.psd_params.get('matlab_nfft', 2048), 
                        min=512, max=8192, step=512
                    ),
                    html.Small("FFT length - higher values give better frequency resolution", className="text-muted")
                ], width=4),
                dbc.Col([
                    dbc.Checklist(
                        id={'type': 'method-param', 'param': 'use-divider-matlab'},
                        options=[{"label": "Use segment length for NFFT", "value": "use_formula"}],
                        value=["use_formula"] if analyzer.psd_params.get('use_divider_for_matlab', False) else [],
                    ),
                    html.Small("Check to calculate NFFT from segment formula", className="text-muted")
                ], width=4)
            ])
        elif method == 'multitaper':
            return dbc.Row([
                dbc.Col([
                    dbc.Label("Time-Bandwidth:"),
                    dbc.Input(
                        id={'type': 'method-param', 'param': 'time-bandwidth'}, 
                        type='number', 
                        value=4,
                        min=2, max=10, step=0.5
                    ),
                    html.Small("Higher values give better frequency resolution but more smoothing", className="text-muted")
                ], width=6)
            ])
        elif method in ['welch', 'welch_smooth']:
            # Show Welch parameters only when Welch methods are selected
            welch_specific = []
            
            if method == 'welch':
                welch_specific = [
                    html.H5("Welch Method Parameters", className="mt-3 mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Segment Length = min(data_length/divider, max_value)"),
                        ], width=12)
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Divider:"),
                            dbc.Input(id='nperseg-divider', type='number', 
                                    value=analyzer.psd_params['nperseg_divider'], 
                                    min=2, max=20, step=1)
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Max Value:"),
                            dbc.Input(id='nperseg-max', type='number', 
                                    value=analyzer.psd_params['nperseg_max'], 
                                    min=64, step=64)
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Window Type:"),
                            dcc.Dropdown(
                                id='window-type',
                                options=[
                                    {'label': 'Tukey', 'value': 'tukey'},
                                    {'label': 'Hann', 'value': 'hann'},
                                    {'label': 'Hamming', 'value': 'hamming'},
                                    {'label': 'Blackman', 'value': 'blackman'},
                                    {'label': 'Bartlett', 'value': 'bartlett'},
                                    {'label': 'Cosine', 'value': 'cosine'},
                                    {'label': 'Boxcar (None)', 'value': 'boxcar'}
                                ],
                                value=analyzer.psd_params['window']
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Current nperseg:"),
                            html.Div(id='current-nperseg', 
                                className="form-control bg-light")
                        ], width=2)
                    ])
                ]
            elif method == 'welch_smooth':
                welch_specific = [
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Smoothing Sigma:"),
                            dbc.Input(
                                id={'type': 'method-param', 'param': 'smooth-sigma'}, 
                                type='number', 
                                value=analyzer.psd_params.get('smooth_sigma', 2), 
                                min=0.5, max=10, step=0.5
                            ),
                            html.Small("Higher values give smoother spectra", className="text-muted")
                        ], width=6)
                    ])
                ]
            
            return html.Div(welch_specific)
        else:
            return html.Div()
    except Exception as e:
        print(f"Error in update_method_params: {e}")
        return html.Div("Error loading method parameters")
    

# Update the current-nperseg display when method is Welch
@app.callback(
    Output('current-nperseg', 'children'),
    [Input('nperseg-divider', 'value'),
     Input('nperseg-max', 'value'),
     Input('data-store', 'children'),
     Input('apply-clip-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_nperseg_display(divider, max_val, data_status, clip_clicks):
    try:
        # Check if we're in the context where these inputs exist
        ctx = dash.callback_context
        
        # If the inputs don't exist (not Welch method), return empty
        if divider is None and max_val is None:
            return ""
        
        if data_status != "data-loaded" or analyzer.data is None:
            return "No data"
        
        # Handle None values - use defaults if not provided
        if divider is None:
            divider = 4
        if max_val is None:
            max_val = 520
        
        # Calculate based on current data length (use clipped data if available)
        if hasattr(analyzer, 'analysis_data') and analyzer.analysis_data is not None:
            data_length = len(analyzer.analysis_data)
            data_source = "clipped"
        else:
            data_length = len(analyzer.data)
            data_source = "full"
        
        nperseg = min(data_length // int(divider), int(max_val))
        nperseg = max(nperseg, 16)  # Minimum value
        
        return f"{nperseg} ({data_source} data: {data_length} pts)"
    
    except Exception as e:
        print(f"Error in nperseg callback: {e}")
        return "Error"


@app.callback(
    [Output('frequency-selection-state', 'children'),
     Output('frequency-selection-status', 'children'),
     Output('freq-select-btn', 'children'),
     Output('freq-select-btn', 'color')],
    [Input('freq-select-btn', 'n_clicks')]
)
def toggle_frequency_selection(n_clicks):
    if n_clicks is None:
        return 'inactive', "", "Enter Frequency Selection Mode", 'info'
    
    if n_clicks % 2 == 1:  # Odd clicks = active
        analyzer.frequency_selection_mode = True
        analyzer.selected_frequencies = []
        logger.info("Frequency selection mode activated")
        status = dbc.Alert("Frequency Selection Mode ACTIVE: Click on the PSD plot to select low and high frequency bounds", 
                          color="warning")
        return 'active', status, "Exit Frequency Selection Mode", 'warning'
    else:  # Even clicks = inactive
        analyzer.frequency_selection_mode = False
        logger.info("Frequency selection mode deactivated")
        return 'inactive', "", "Enter Frequency Selection Mode", 'info'


# Callback for changing output folder
@app.callback(
    [Output('output-folder-display', 'value'),
     Output('output-folder-store', 'children')],
    [Input('change-folder-btn', 'n_clicks')],
    [State('output-folder-store', 'children')],
    prevent_initial_call=True
)
def change_output_folder(n_clicks, current_folder):
    if n_clicks:
        logger.info("Output folder change requested")
        # Create a simple tkinter dialog for folder selection
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring to front
        
        folder_selected = filedialog.askdirectory(
            initialdir=current_folder,
            title="Select Output Folder"
        )
        
        root.destroy()
        
        if folder_selected:
            analyzer.set_output_folder(folder_selected)
            logger.info(f"Output folder changed to: {folder_selected}")
            return folder_selected, folder_selected
    
    return current_folder, current_folder


# PSD plot callback
@app.callback(
    [Output('psd-plot', 'figure'),
     Output('f-low', 'value'),
     Output('f-high', 'value')],
    [Input('data-store', 'children'),
     Input('sampling-freq', 'value'),
     Input('psd-plot', 'clickData'),
     Input('reset-view-btn', 'n_clicks'),
     Input('psd-method', 'value'),
     Input({'type': 'method-param', 'param': ALL}, 'value')],
    [State('frequency-selection-state', 'children'),
     State({'type': 'method-param', 'param': ALL}, 'id')],
    prevent_initial_call=False
)
def update_psd_plot(data_status, sampling_freq, click_data, reset_clicks, psd_method,
                   method_param_values, freq_selection_state, method_param_ids):
    
    try:
        # Default return values
        default_fig = go.Figure().add_annotation(
            text="Please upload data first", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        default_f_low = analyzer.filter_params.get('f_low', 0.8)
        default_f_high = analyzer.filter_params.get('f_high', 1.2)
        
        # Early return if no data
        if data_status != "data-loaded" or analyzer.data is None:
            return default_fig, default_f_low, default_f_high
        
        # Handle None values with safe defaults
        sampling_freq = sampling_freq or analyzer.sampling_freq or 72
        psd_method = psd_method or 'matlab_mtm'
        freq_selection_state = freq_selection_state or 'inactive'
        
        if DEBUG_CALLBACKS:
            print(f"PSD Plot Update: method={psd_method}, sampling_freq={sampling_freq}")
        
        # Update analyzer parameters safely
        analyzer.sampling_freq = sampling_freq
        analyzer.psd_params['method'] = psd_method
        
        # Handle method-specific parameters with None checks
        if method_param_values and method_param_ids:
            for value, id_dict in zip(method_param_values, method_param_ids):
                if value is not None and id_dict is not None:
                    param_name = id_dict.get('param')
                    if param_name == 'matlab-tbw':
                        analyzer.psd_params['time_bandwidth'] = float(value)
                    elif param_name == 'matlab-nfft':
                        analyzer.psd_params['matlab_nfft'] = int(value)
                    elif param_name == 'time-bandwidth':
                        analyzer.psd_params['time_bandwidth'] = float(value)
                    elif param_name == 'smooth-sigma':
                        analyzer.psd_params['smooth_sigma'] = float(value)
                    elif param_name == 'use-divider-smooth':
                        analyzer.psd_params['use_divider_for_smooth'] = 'use_formula' in value if isinstance(value, list) else False
                    elif param_name == 'use-divider-matlab':
                        analyzer.psd_params['use_divider_for_matlab'] = 'use_formula' in value if isinstance(value, list) else False
        
        # Get parameters based on current method
        custom_params = analyzer.psd_params.copy()
        
        # Handle frequency selection clicks
        f_low_new = default_f_low
        f_high_new = default_f_high

        if (freq_selection_state == 'active' and click_data is not None and 
            'points' in click_data and len(click_data['points']) > 0):
            
            freq_clicked = click_data['points'][0]['x']
            logger.info(f"Frequency selected: {freq_clicked:.5f} day^-1")
            
            if len(analyzer.selected_frequencies) == 0:
                analyzer.selected_frequencies = [freq_clicked]
                f_low_new = freq_clicked
                f_high_new = freq_clicked
            elif len(analyzer.selected_frequencies) == 1:
                analyzer.selected_frequencies.append(freq_clicked)
                freqs_sorted = sorted(analyzer.selected_frequencies)
                f_low_new = freqs_sorted[0]
                f_high_new = freqs_sorted[1]
                logger.info(f"Frequency band selected: {f_low_new:.5f} - {f_high_new:.5f} day^-1")
            else:
                analyzer.selected_frequencies = [freq_clicked]
                f_low_new = freq_clicked
                f_high_new = freq_clicked
        
        # Compute PSD with error handling
        try:
            if hasattr(analyzer, 'analysis_data') and analyzer.analysis_data is not None:
                freq_shallow, psd_shallow = analyzer.compute_psd('TempShallow', use_clipped=True, custom_params=custom_params)
                freq_deep, psd_deep = analyzer.compute_psd('TempDeep', use_clipped=True, custom_params=custom_params)
                if DEBUG_CALLBACKS:
                    print("Computed PSD using clipped data")
            else:
                freq_shallow, psd_shallow = analyzer.compute_psd('TempShallow', use_clipped=False, custom_params=custom_params)
                freq_deep, psd_deep = analyzer.compute_psd('TempDeep', use_clipped=False, custom_params=custom_params)
                if DEBUG_CALLBACKS:
                    print("Computed PSD using full data")
        except Exception as psd_error:
            print(f"Error computing PSD: {psd_error}")
            error_fig = go.Figure().add_annotation(
                text=f"Error computing PSD: {str(psd_error)}", 
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            return error_fig, f_low_new, f_high_new
            
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=freq_shallow, y=psd_shallow,
            mode='lines', name='Shallow Temp',
            line=dict(color='blue', width=2),
            hovertemplate='Frequency: %{x:.5f} day^-1<br>Power: %{y:.5e}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=freq_deep, y=psd_deep,
            mode='lines', name='Deep Temp',
            line=dict(color='red', width=2),
            hovertemplate='Frequency: %{x:.5f} day^-1<br>Power: %{y:.5e}<extra></extra>'
        ))
        
        # Add the shaded rectangle for the filter band
        fig.add_vrect(x0=f_low_new, x1=f_high_new, 
                      fillcolor="yellow", opacity=0.2,
                      line_width=0)
        
        # Add vertical lines with annotations
        fig.add_vline(x=f_low_new, 
                      line_dash="dash", 
                      line_color="green", 
                      line_width=2,
                      annotation_text=f"Low: {f_low_new:.5f}",
                      annotation_position="top left",
                      annotation_yshift=10)
        
        fig.add_vline(x=f_high_new, 
                      line_dash="dash", 
                      line_color="orange", 
                      line_width=2,
                      annotation_text=f"High: {f_high_new:.5f}",
                      annotation_position="top right",
                      annotation_yshift=10)
        
        # Add filter band annotation
        if f_low_new != f_high_new:
            fig.add_annotation(
                x=(f_low_new + f_high_new) / 2,
                y=1,
                yref="paper",
                text="Filter Band",
                showarrow=False,
                yshift=30,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        # Add method annotation with parameters
        method_text = {
            'welch': 'Welch Method',
            'welch_smooth': 'Welch (Smoothed)',
            'multitaper': 'Multitaper (R style)',
            'matlab_mtm': f'Multitaper (MATLAB style) - tbw={analyzer.psd_params.get("time_bandwidth", 4)}'
        }.get(psd_method, 'Welch Method')
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Method: {method_text}",
            showarrow=False,
            bgcolor="lightgray",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title="Power Spectral Density (Click to select frequency bounds in selection mode)",
            xaxis_title="Frequency (day^-1)",
            yaxis_title="Power Spectral Density",
            yaxis_type="log",
            xaxis=dict(range=[0, 4]),
            hovermode='x unified',
            template='plotly_white',
            margin=dict(t=80)
        )
        
        return fig, f_low_new, f_high_new
        
    except Exception as e:
        print(f"Critical error in PSD plot callback: {e}")
        import traceback
        traceback.print_exc()
        
        error_fig = go.Figure().add_annotation(
            text=f"Critical error: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return error_fig, analyzer.filter_params.get('f_low', 0.8), analyzer.filter_params.get('f_high', 1.2)


@app.callback(
    [Output('filter-validation-status', 'children'),
     Output('filter-characteristics-card', 'children')],
    [Input('f-low', 'value'),
     Input('f-high', 'value'),
     Input('filter-order', 'value'),
     Input('ramp-fraction', 'value'),
     Input('resample-interval', 'value'),  # NEW: Include resample interval
     Input('original-interval', 'value')],  # NEW: Include original interval
    [State('data-store', 'children')]
)
def update_filter_validation(f_low, f_high, filter_order, ramp_fraction, resample_interval, original_interval, data_status):
    if data_status != "data-loaded" or not all([f_low, f_high, filter_order, ramp_fraction]):
        return "", ""
    
    # Update analyzer parameters for validation
    analyzer.filter_params.update({
        'f_low': f_low,
        'f_high': f_high,
        'filter_order': filter_order,
        'ramp_fraction': ramp_fraction,
        'resample_interval_minutes': resample_interval or 1,
        'original_interval_minutes': original_interval or 20
    })
    
    validation = analyzer.validate_filter_parameters()
    
    # Create validation status display
    validation_alerts = []
    
    if validation['errors']:
        for error in validation['errors']:
            validation_alerts.append(dbc.Alert(f"Error: {error}", color="danger"))
    
    if validation['warnings']:
        for warning in validation['warnings']:
            validation_alerts.append(dbc.Alert(f"Warning: {warning}", color="warning"))
    
    if not validation['errors'] and not validation['warnings']:
        validation_alerts.append(dbc.Alert("Filter parameters validated successfully", color="success"))
    
    # Create filter band details (following Table 2 format from specs)
    band_width = f_high - f_low
    ramp_width = band_width * ramp_fraction
    
    f_start_ramp = f_low - ramp_width/2
    f_start_pass = f_low + ramp_width/2
    f_end_pass = f_high - ramp_width/2
    f_end_ramp = f_high + ramp_width/2
    
    # NEW: Add resampling information
    original_int = original_interval or 20
    resample_int = resample_interval or 1
    improvement_factor = original_int / resample_int
    
    # Create table similar to Table 2 in specifications
    band_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Frequency (day^-1)"),
                html.Th("Normalize by Day"),
                html.Th("Normalize by Nyquist"),
                html.Th("Purpose")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(f"{f_start_ramp:.5f}"),
                html.Td(f"{1/f_start_ramp:.5f}" if f_start_ramp > 0 else "N/A"),
                html.Td(f"{f_start_ramp/(analyzer.sampling_freq/2):.5f}"),
                html.Td("Start ramp up")
            ]),
            html.Tr([
                html.Td(f"{f_start_pass:.5f}"),
                html.Td(f"{1/f_start_pass:.5f}" if f_start_pass > 0 else "N/A"),
                html.Td(f"{f_start_pass/(analyzer.sampling_freq/2):.5f}"),
                html.Td("Start full pass")
            ]),
            html.Tr([
                html.Td(f"{f_end_pass:.5f}"),
                html.Td(f"{1/f_end_pass:.5f}" if f_end_pass > 0 else "N/A"),
                html.Td(f"{f_end_pass/(analyzer.sampling_freq/2):.5f}"),
                html.Td("End full pass, start ramp down")
            ]),
            html.Tr([
                html.Td(f"{f_end_ramp:.5f}"),
                html.Td(f"{1/f_end_ramp:.5f}" if f_end_ramp > 0 else "N/A"),
                html.Td(f"{f_end_ramp/(analyzer.sampling_freq/2):.5f}"),
                html.Td("End ramp down")
            ])
        ])
    ], bordered=True, striped=True, size="sm")
    
    # NEW: Resampling information section
    resampling_color = "success" if improvement_factor > 1 else "warning" if improvement_factor == 1 else "info"
    resampling_text = f"Resolution improvement: {improvement_factor:.1f}:1 ({original_int}->{resample_int} min)"
    
    filter_characteristics_card = dbc.Card([
        dbc.CardHeader("Filter Characteristics (Specification Compliant)"),
        dbc.CardBody([
            html.H6("Frequency Band Details:"),
            band_table,
            html.Hr(),
            html.P(f"Total band width: {band_width:.5f} day^-1"),
            html.P(f"Ramp width: {ramp_width:.5f} day^-1 ({ramp_fraction*100:.1f}% of band)"),
            html.P(f"Pass width: {f_end_pass - f_start_pass:.5f} day^-1"),
            html.P(f"Estimated filter duration: {validation.get('filter_duration_days', 0):.5f} days"),
            html.Hr(),
            html.H6("Resampling Configuration:"),
            dbc.Alert([
                html.Strong("Resampling: "),
                resampling_text,
                html.Br(),
                html.Small("Higher resolution enables better peak/trough detection and uncertainty analysis.")
            ], color=resampling_color)
        ])
    ])
    
    return validation_alerts, filter_characteristics_card


# Raw data plot at top
@app.callback(
    Output('raw-data-plot-top', 'figure'),
    [Input('data-store', 'children')]
)
def update_raw_data_plot_top(data_status):
    if data_status != "data-loaded" or analyzer.data is None:
        return go.Figure().add_annotation(text="Please upload data first", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    logger.debug("Updating raw data plot (top)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=analyzer.data['WaterDay'], y=analyzer.data['TempShallow'],
        mode='lines', name='Shallow Temp',
        line=dict(color='blue', width=2),
        hovertemplate='Day: %{x:.5f}<br>Temperature: %{y:.5f}C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=analyzer.data['WaterDay'], y=analyzer.data['TempDeep'],
        mode='lines', name='Deep Temp',
        line=dict(color='red', width=2),
        hovertemplate='Day: %{x:.5f}<br>Temperature: %{y:.5f}C<extra></extra>'
    ))
    
    # Add title with WD limits if applied
    title = "Raw Temperature Data"
    if analyzer.wd_lower > 0 or analyzer.wd_upper > 0:
        title += f" (WD: {analyzer.wd_lower} to {analyzer.wd_upper})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Water Day",
        yaxis_title="Temperature (C)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


# Update raw data plot callback - ALWAYS use original data, never change
@app.callback(
    Output('raw-data-plot', 'figure'),
    [Input('data-store', 'children'),
     Input('filtered-data-plot', 'relayoutData')]
)
def update_raw_data_plot(data_status, filtered_relayout):
    if data_status != "data-loaded" or analyzer.data is None:
        return go.Figure().add_annotation(text="Please upload data first", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    logger.debug("Updating raw data plot (comparison)")
    
    fig = go.Figure()
    
    # ALWAYS use original data - never modified by filtering
    fig.add_trace(go.Scatter(
        x=analyzer.data['WaterDay'], y=analyzer.data['TempShallow'],
        mode='lines', name='Shallow Temp',
        line=dict(color='blue', width=2),
        hovertemplate='Day: %{x:.5f}<br>Temperature: %{y:.5f}C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=analyzer.data['WaterDay'], y=analyzer.data['TempDeep'],
        mode='lines', name='Deep Temp',
        line=dict(color='red', width=2),
        hovertemplate='Day: %{x:.5f}<br>Temperature: %{y:.5f}C<extra></extra>'
    ))
    
    # Simple x-axis range matching
    if analyzer.filtered_data is not None and len(analyzer.filtered_data) > 0:
        # Default to filtered data range
        x_min = analyzer.filtered_data['WaterDay'].min()
        x_max = analyzer.filtered_data['WaterDay'].max()
        xaxis_range = [x_min, x_max]
        
        # Override with user zoom/pan if available
        if filtered_relayout and 'xaxis.range[0]' in filtered_relayout and 'xaxis.range[1]' in filtered_relayout:
            xaxis_range = [filtered_relayout['xaxis.range[0]'], filtered_relayout['xaxis.range[1]']]
    else:
        # No filtered data - use full range
        xaxis_range = [analyzer.data['WaterDay'].min(), analyzer.data['WaterDay'].max()]
    
    fig.update_layout(
        title="Raw Temperature Data (Comparison)",
        xaxis_title="Water Day",
        yaxis_title="Temperature (C)",
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(range=xaxis_range)
    )
    
    return fig


# FIXED: Filtered data plot callback with zoom preservation
@app.callback(
    Output('filtered-data-plot', 'figure'),
    [Input('apply-filter-btn', 'n_clicks')],
    [State('f-low', 'value'),
     State('f-high', 'value'),
     State('filter-type', 'value'),
     State('filter-order', 'value'),
     State('ramp-fraction', 'value'),
     State('resample-interval', 'value'),  
     State('original-interval', 'value'),  
     State('trend-removal', 'value'),
     State('data-store', 'children'),
     State('filtered-data-plot', 'relayoutData')]  # FIXED: Added this to preserve zoom
)
def update_filtered_plot(n_clicks, f_low, f_high, filter_type, filter_order, 
                       ramp_fraction, resample_interval, original_interval, 
                       trend_removal, data_status, current_layout):
    if data_status != "data-loaded" or analyzer.data is None:
        return go.Figure().add_annotation(text="Please upload data first", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    if n_clicks is None:
        return go.Figure().add_annotation(text="Click 'Apply Filter' to see filtered data", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    logger.info("Applying filter and updating filtered data plot")
    
    # Update filter parameters and apply filter
    update_params = {
        'f_low': f_low,
        'f_high': f_high,
        'filter_type': filter_type,
        'filter_order': filter_order,
        'ramp_fraction': ramp_fraction,
        'resample_interval_minutes': resample_interval,  
        'original_interval_minutes': original_interval, 
        'trend_removal': trend_removal
    }
    
    success, message = analyzer.apply_filter_with_validation(update_params)
    
    if not success:
        return go.Figure().add_annotation(text=f"Error applying filter: {message}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=analyzer.filtered_data['WaterDay'], y=analyzer.filtered_data['TempShallow_Filt'],
        mode='lines', name='Shallow Temp',
        line=dict(color='blue', width=2),
        hovertemplate='Day: %{x:.5f}<br>Temperature: %{y:.5f}C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=analyzer.filtered_data['WaterDay'], y=analyzer.filtered_data['TempDeep_Filt'],
        mode='lines', name='Deep Temp',
        line=dict(color='red', width=2),
        hovertemplate='Day: %{x:.5f}<br>Temperature: %{y:.5f}C<extra></extra>'
    ))
    
    # FIXED: Preserve user's zoom level if available
    if current_layout and 'xaxis.range[0]' in current_layout and 'xaxis.range[1]' in current_layout:
        # User has zoomed/panned - preserve their view
        xaxis_range = [current_layout['xaxis.range[0]'], current_layout['xaxis.range[1]']]
        logger.debug(f"Preserving user zoom: {xaxis_range}")
    else:
        # No previous zoom - use full data range
        x_min = analyzer.filtered_data['WaterDay'].min()
        x_max = analyzer.filtered_data['WaterDay'].max()
        xaxis_range = [x_min, x_max]
        logger.debug(f"Using full range: {xaxis_range}")
    
    # Enhanced title with resampling info
    improvement = original_interval / resample_interval if resample_interval and resample_interval > 0 else 1
    title = f"Filtered Temperature Data (Band: {f_low:.5f}-{f_high:.5f} day^-1, {filter_type}, Order: {filter_order}"
    if improvement > 1:
        title += f", {improvement:.1f}:1 resampling)"
    else:
        title += ")"
    
    fig.update_layout(
        title=title,
        xaxis_title="Water Day",
        yaxis_title="Filtered Temperature (C)",
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(range=xaxis_range)  # FIXED: Use preserved or default range
    )
    
    return fig


# Save filtered data callback
@app.callback(
    [Output('filter-history-display', 'children')],
    [Input("save-data-btn", "n_clicks")],
    [State('output-filename', 'value'),
     State('output-folder-store', 'children')]
)
def save_filtered_data(n_clicks, filename, output_folder):
    if n_clicks is None:
        return [""]
    
    logger.info(f"Save filtered data requested: {filename}")
    
    if analyzer.filtered_data is not None:
        # Update the output folder if it has changed
        if output_folder and output_folder != analyzer.output_folder:
            analyzer.output_folder = output_folder
        
        # Use custom filename if provided
        if not filename or filename.strip() == "":
            filename = "filtered_temperature_data.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Save to history and get formatted data (only save to output folder, no browser download)
        success, full_path = analyzer.save_filtered_data_spec_format(filename)
        
        if success:
            # Update filter history display with enhanced resampling info
            history_cards = []
            for i, entry in enumerate(analyzer.filter_history):
                # Extract resampling info
                resampling_info = entry.get('resampling_info', {})
                improvement = resampling_info.get('resolution_improvement', 1)
                
                card = dbc.Card([
                    dbc.CardBody([
                        html.H6(f"File: {entry['filename']}", className="card-title"),
                        html.P([
                            f"Band: {entry['parameters']['f_low']:.5f}-{entry['parameters']['f_high']:.5f} day^-1, ",
                            f"Filter: {entry['parameters']['filter_type']}, ",
                            f"Order: {entry['parameters']['filter_order']}, ",
                            f"Trend: {entry['parameters']['trend_removal']}"
                        ], className="card-text small"),
                        html.P([
                            f"Resampling: {improvement:.1f}:1 improvement ",
                            f"({resampling_info.get('original_interval_min', 20)}->{resampling_info.get('resample_interval_min', 1)} min)"
                        ], className="card-text small text-info"),
                        html.P(f"Saved: {entry['timestamp'][:19]}", className="card-text small text-muted"),
                        html.P(f"Location: {entry['full_path']}", className="card-text small text-info")
                    ])
                ], color="light", outline=True, style={"margin-bottom": "5px"})
                history_cards.append(card)
            
            # Enhanced success message
            improvement_text = ""
            if len(analyzer.filter_history) > 0:
                latest_entry = analyzer.filter_history[-1]
                resampling_info = latest_entry.get('resampling_info', {})
                improvement = resampling_info.get('resolution_improvement', 1)
                if improvement > 1:
                    improvement_text = f" Resolution: {improvement:.1f}:1 improvement applied."
            
            success_alert = dbc.Alert(
                f"Data saved successfully to: {full_path}{improvement_text}",
                color="success",
                dismissable=True
            )
            
            history_display = html.Div([
                success_alert,
                html.H5("Filter History:"),
                html.Div(history_cards)
            ]) if history_cards else html.Div([success_alert])
            
            return [history_display]
        else:
            error_alert = dbc.Alert("Error saving data", color="danger", dismissable=True)
            return [error_alert]
    
    logger.warning("No filtered data available to save")
    return [""]


@app.callback(
    Output("download-parameters", "data"),
    [Input("save-params-btn", "n_clicks")],
    prevent_initial_call=True
)
def download_parameters(n_clicks):
    logger.info("Download parameters requested")
    params = {
        'filter_parameters': analyzer.filter_params,
        'psd_parameters': analyzer.psd_params,
        'sampling_frequency': analyzer.sampling_freq,
        'filter_history': analyzer.filter_history,
        'output_folder': analyzer.output_folder,
        'timestamp': datetime.now().isoformat()
    }
    return dict(content=json.dumps(params, indent=2), filename="filter_parameters.json")


@app.callback(
    Output('param-save-status', 'children'),
    [Input('save-params-btn', 'n_clicks')],
    [State('f-low', 'value'),
     State('f-high', 'value'),
     State('filter-order', 'value'),
     State('ramp-fraction', 'value'),
     State('resample-interval', 'value'),
     State('original-interval', 'value'),
     State('sampling-freq', 'value')],
    prevent_initial_call=True
)
def save_parameters_to_file(n_clicks, f_low, f_high, order, ramp, resample, original, sampling_freq):
    if n_clicks:
        logger.info("Save parameters to file requested")
        # Update analyzer parameters with current UI values
        analyzer.filter_params.update({
            'f_low': f_low,
            'f_high': f_high,
            'filter_order': order,
            'ramp_fraction': ramp,
            'resample_interval_minutes': resample,
            'original_interval_minutes': original
        })
        analyzer.sampling_freq = sampling_freq
        
        # Save to file
        if analyzer.save_parameters_file():
            return dbc.Alert("Parameters saved to tts_freqfilt.par", color="success", dismissable=True)
        else:
            return dbc.Alert("Error saving parameters", color="danger", dismissable=True)
    
    return ""


@app.callback(
    Output('clip-status', 'children'),
    [Input('apply-clip-btn', 'n_clicks')],
    [State('wd-lower-limit', 'value'),
     State('wd-upper-limit', 'value'),
     State('data-store', 'children')]
)
def apply_data_clipping(n_clicks, wd_lower, wd_upper, data_status):
    if n_clicks is None or data_status != "data-loaded":
        return ""
    
    logger.info(f"Apply data clipping requested: WD {wd_lower} to {wd_upper}")
    
    # Apply clipping
    result = analyzer.clip_data_for_analysis(wd_lower, wd_upper)
    
    if result['success']:
        if result['original_length'] == result['clipped_length']:
            message = "No clipping applied (limits include all data)"
            color = "info"
        else:
            message = (f"Data clipped: {result['original_length']} -> {result['clipped_length']} records. "
                      f"WD range: {result['actual_wd_range'][0]:.5f} to {result['actual_wd_range'][1]:.5f}")
            color = "success"
        
        return dbc.Alert(message, color=color, dismissable=True)
    else:
        return dbc.Alert(result['message'], color="danger", dismissable=True)


# Store method parameters when they change
@app.callback(
    Output('method-specific-params', 'id'),  # Dummy output
    [Input({'type': 'method-param', 'param': ALL}, 'value')],
    [State({'type': 'method-param', 'param': ALL}, 'id')],
    prevent_initial_call=True
)
def update_method_specific_params(values, ids):
    """Capture and store method-specific parameters"""
    try:
        if not values or not ids or len(values) != len(ids):
            return 'method-specific-params'
            
        for value, id_dict in zip(values, ids):
            if value is not None and id_dict is not None and isinstance(id_dict, dict):
                param_name = id_dict.get('param')
                if param_name == 'time-bandwidth':
                    analyzer.psd_params['time_bandwidth'] = float(value)
                    logger.debug(f"Updated time_bandwidth to {value}")
                elif param_name == 'smooth-sigma':
                    analyzer.psd_params['smooth_sigma'] = float(value)
                    logger.debug(f"Updated smooth_sigma to {value}")
                elif param_name == 'matlab-tbw':
                    analyzer.psd_params['time_bandwidth'] = float(value)
                    logger.debug(f"Updated matlab tbw to {value}")
                elif param_name == 'matlab-nfft':
                    analyzer.psd_params['matlab_nfft'] = int(value)
                    logger.debug(f"Updated matlab nfft to {value}")
                elif param_name == 'use-divider-smooth':
                    analyzer.psd_params['use_divider_for_smooth'] = 'use_formula' in value if isinstance(value, list) else False
                elif param_name == 'use-divider-matlab':
                    analyzer.psd_params['use_divider_for_matlab'] = 'use_formula' in value if isinstance(value, list) else False
    except Exception as e:
        print(f"Error updating method parameters: {e}")
    
    return 'method-specific-params'


if __name__ == '__main__':
    logger.info("="*70)
    logger.info("Temperature Time-Series Frequency Analysis & Filtering Tool v2.2.1")
    logger.info("="*70)
    
    # Check for MNE availability
    if MNE_AVAILABLE:
        logger.info("[OK] MNE library available - Multitaper method enabled")
    else:
        logger.warning("[X] MNE library not found - Multitaper method disabled")
        logger.info("  To enable multitaper support, install MNE: pip install mne")
    
    logger.info("="*70)
    
    # Check for parameter file
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    par_file = None
    
    for directory in search_dirs:
        potential_path = os.path.join(directory, 'tts_freqfilt.par')
        if os.path.isfile(potential_path):
            par_file = potential_path
            break
    
    if par_file:
        logger.info(f"Found parameter file: {par_file}")
    else:
        logger.warning("No parameter file found. Using default values.")
        logger.info("Create 'tts_freqfilt.par' in current or script directory.")
    
    logger.info("="*70)
    
    # Get script directory for output folder display
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(analyzer.output_folder):
        expected_output_path = os.path.join(script_dir, analyzer.output_folder)
    else:
        expected_output_path = analyzer.output_folder
    
    logger.info(f"Output folder configured: {analyzer.output_folder}")
    logger.info(f"Full output path: {expected_output_path}")
    
    if not os.path.exists(expected_output_path):
        logger.info("Output folder will be created when first file is saved.")
    else:
        logger.info("Output folder exists and ready for use.")
    
    # Check for default input file
    if hasattr(analyzer, 'input_filename') and analyzer.input_filename:
        if analyzer.check_default_file_exists():
            # Find the actual path
            search_paths = [
                analyzer.input_filename,  # As specified (could be relative or absolute)
                os.path.join(os.getcwd(), analyzer.input_filename),  # Current working directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), analyzer.input_filename)  # Script directory
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    actual_path = path
                    break
            
            logger.info(f"[OK] Default input file found: {analyzer.input_filename}")
            logger.info(f"  Full path: {actual_path}")
            logger.info("  You can click 'Load Default File' button to load it.")
        else:
            logger.warning(f"[X] Default input file not found: {analyzer.input_filename}")
            logger.info("  Searched in:")
            search_paths = [
                analyzer.input_filename,
                os.path.join(os.getcwd(), analyzer.input_filename),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), analyzer.input_filename)
            ]
            for path in search_paths:
                logger.info(f"    - {path}")
            logger.info("  Place the file in the same directory as the script or update the .par file.")
    
    logger.info("="*70)
    
    # Enhanced startup message about resampling
    current_resample = analyzer.filter_params['resample_interval_minutes']
    current_original = analyzer.filter_params['original_interval_minutes']
    improvement = current_original / current_resample
    
    logger.info("RESAMPLING CONFIGURATION:")
    logger.info(f"  Original interval: {current_original} minutes")
    logger.info(f"  Resample interval: {current_resample} minutes")
    if improvement > 1:
        logger.info(f"  [OK] Resolution improvement: {improvement:.1f}:1 (Enhanced peak/trough detection)")
    elif improvement == 1:
        logger.info("  * No resampling configured (same interval)")
        logger.info("  * TIP: Set resample_interval = 1 in .par file for 20:1 improvement")
    else:
        logger.info(f"  [v] Downsampling configured: {1/improvement:.1f}:1")
    
    logger.info("="*70)
    logger.info("ZOOM PRESERVATION FIX APPLIED:")
    logger.info("  - Apply Filter button now preserves your zoom level")
    logger.info("  - No more automatic zoom reset when filtering")
    logger.info("  - Enhanced user experience for detailed peak analysis")
    logger.info("="*70)
    logger.info("Starting web application...")
    logger.info("Open your browser to http://127.0.0.1:8051")
    logger.info("Press Ctrl+C to stop the application")
    logger.info("="*70)
    
    app.run(debug=True, host='127.0.0.1', port=8051)
