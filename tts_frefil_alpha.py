#!/usr/bin/env python3
"""
tts_frefil.py - Temperature Time-Series Frequency Analysis and Filtering

This script provides interactive frequency analysis and filtering capabilities for 
temperature time-series data, specifically designed for analyzing diurnal cycles 
and other frequency components in shallow and deep temperature measurements.

Key Features:
- Interactive power spectral density (PSD) visualization
- Multiple PSD methods: Welch, Multitaper, and Welch with smoothing
- Graphical frequency band selection for filtering
- Multiple filter types and parameters with proper ramp implementation
- Real-time filter preview and comparison
- Export of filtered data and processing parameters
- Parameter file support for saving/loading analysis settings
- Comprehensive validation and specification compliance
- Automatic output folder management

Enhanced Features (v2.0):
- Specification-compliant filter ramp implementation
- Enhanced trend removal with DC offset correction
- Comprehensive parameter validation
- Filter characteristics display matching specifications
- Exact output format compliance
- Advanced edge effect handling
- Configurable output folders from parameter file

Author: Timothy Wu
Created: 7/3/2025
Last Updated: 7/3/2025

Version: 2.0 - Full Specification Compliance with Folder Management
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

# Try to import mne for multitaper support
try:
    from mne.time_frequency import psd_array_multitaper
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not installed. Multitaper method will not be available.")
    print("Install with: pip install mne")


class TemperatureFrequencyAnalyzer:
    def __init__(self):
        self.data = None
        self.filtered_data = None
        self.sampling_freq = 72  # measurements per day (default)
        self.analysis_data = None  # Stores clipped data for analysis
        self.last_filter_info = None  # Store filter characteristics
        
        self.psd_params = {
            'method': 'welch',
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
            'filter_order': 6,
            'ramp_fraction': 0.1,  # Use the value from user's .par file
            'original_interval_minutes': 20,  
            'resample_interval_minutes': 20,  
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
            potential_path = os.path.join(directory, 'tts_frefil.par')
            if os.path.isfile(potential_path):
                par_file = potential_path
                break
        
        if par_file:
            print(f"Loading parameters from: {par_file}")
            self.load_parameters(par_file)
        else:
            print("No tts_frefil.par found in current or script directory. Using defaults.")
        
    def parse_par(self, path):
        """Parse a flat parameter file with key=value pairs."""
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

    def load_parameters(self, filepath):
        """Load parameters from .par file"""
        params = self.parse_par(filepath)
        
        # Map parameter file keys to internal parameter names
        if 'filename_composite_data' in params:
            self.input_filename = params['filename_composite_data']
        
        # NEW: Load output folder parameter
        if 'output_folder' in params:
            self.output_folder = params['output_folder']
        
        if 'data_interval_minutes' in params:
            try:
                self.filter_params['original_interval_minutes'] = int(params['data_interval_minutes'])
                self.filter_params['resample_interval_minutes'] = int(params.get('resample_interval', params['data_interval_minutes']))
            except ValueError:
                pass
        
        if 'time_bandwidth_parameter' in params:
            try:
                self.time_bandwidth = float(params['time_bandwidth_parameter'])
                self.psd_params['time_bandwidth'] = float(params['time_bandwidth_parameter'])
            except ValueError:
                pass
        
        if 'wd_lower_limit' in params:
            try:
                self.wd_lower = float(params['wd_lower_limit'])
            except ValueError:
                pass
        
        if 'wd_upper_limit' in params:
            try:
                self.wd_upper = float(params['wd_upper_limit'])
            except ValueError:
                pass
        
        if 'start_band_pass' in params:
            try:
                self.filter_params['f_low'] = float(params['start_band_pass'])
            except ValueError:
                pass
        
        if 'end_band_pass' in params:
            try:
                self.filter_params['f_high'] = float(params['end_band_pass'])
            except ValueError:
                pass
        
        if 'ramp_fraction' in params:
            try:
                self.filter_params['ramp_fraction'] = float(params['ramp_fraction'])
            except ValueError:
                pass
        
        if 'filter_order' in params:
            try:
                self.filter_params['filter_order'] = int(params['filter_order'])
            except ValueError:
                pass
        
        if 'resample_interval' in params:
            try:
                self.filter_params['resample_interval_minutes'] = int(params['resample_interval'])
            except ValueError:
                pass
        
        # Calculate sampling frequency from data interval
        if 'data_interval_minutes' in params:
            try:
                interval = int(params['data_interval_minutes'])
                self.sampling_freq = (24 * 60) / interval
            except ValueError:
                pass
        
        print(f"Parameters loaded successfully")
        print(f"Output folder set to: {self.output_folder}")
        return True

    def save_parameters_file(self, filename='tts_frefil.par'):
        """Save current parameters to .par file"""
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
                f.write(f"resample_interval = {self.filter_params['resample_interval_minutes']}\n\n")
                
                f.write("# ===== Output =====\n")
                f.write(f"# Output folder name for filtered data files\n")
                f.write(f"output_folder = {self.output_folder}\n")
            
            print(f"Parameters saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving parameter file: {str(e)}")
            return False

    def create_output_folder(self):
        """Create output folder if it doesn't exist (relative to script location)"""
        try:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Create full path relative to script directory
            full_output_path = os.path.join(script_dir, self.output_folder)
            
            if not os.path.exists(full_output_path):
                os.makedirs(full_output_path)
                print(f"Created output folder: {full_output_path}")
            
            # Update the output_folder to use the full path
            self.output_folder = full_output_path
            return True
        except Exception as e:
            print(f"Error creating output folder: {str(e)}")
            return False
        
    def load_data(self, filepath_or_contents, filename=None):
        """Load temperature data from CSV file or uploaded content"""
        try:
            if isinstance(filepath_or_contents, str) and os.path.exists(filepath_or_contents):
                # File path provided
                self.data = pd.read_csv(filepath_or_contents)
            else:
                # Uploaded content (base64 encoded)
                content_type, content_string = filepath_or_contents.split(',')
                decoded = base64.b64decode(content_string)
                self.data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
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
            
            # Ensure we have the required columns
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            # Apply WD limits if set
            if self.wd_lower > 0 or self.wd_upper > 0:
                original_length = len(self.data)
                if self.wd_lower > 0:
                    self.data = self.data[self.data['WaterDay'] >= self.wd_lower]
                if self.wd_upper > 0:
                    self.data = self.data[self.data['WaterDay'] <= self.wd_upper]
                print(f"Applied WD limits: {self.wd_lower} to {self.wd_upper}")
                print(f"Data reduced from {original_length} to {len(self.data)} records")
                
            print(f"Data loaded successfully: {len(self.data)} records")
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def remove_trend_advanced(self, data_col, trend_type='none', return_trend=False):
        """Enhanced trend removal with multiple options to address DC offset issues"""
        data_clean = np.array(data_col)
        trend = np.zeros_like(data_clean)
        
        if trend_type == 'none':
            detrended = data_clean
        elif trend_type == 'dc':
            # Remove DC offset (mean)
            trend = np.full_like(data_clean, np.mean(data_clean))
            detrended = data_clean - trend
        elif trend_type == 'linear':
            # Remove linear trend
            x = np.arange(len(data_clean))
            coeffs = np.polyfit(x, data_clean, 1)
            trend = np.polyval(coeffs, x)
            detrended = data_clean - trend
        elif trend_type == 'polynomial':
            # Remove polynomial trend (order 2)
            x = np.arange(len(data_clean))
            coeffs = np.polyfit(x, data_clean, 2)
            trend = np.polyval(coeffs, x)
            detrended = data_clean - trend
        elif trend_type == 'highpass':
            # High-pass filter to remove low-frequency drift
            from scipy.signal import butter, filtfilt
            nyquist = self.sampling_freq / 2
            cutoff = 0.1 / nyquist  # 0.1 day⁻¹ normalized
            b, a = butter(3, cutoff, btype='high')
            detrended = filtfilt(b, a, data_clean)
            trend = data_clean - detrended
        elif trend_type == 'moving_average':
            # Remove moving average trend (window = 3 days)
            window_size = int(3 * self.sampling_freq)  # 3 days
            from scipy.ndimage import uniform_filter1d
            trend = uniform_filter1d(data_clean, size=window_size, mode='nearest')
            detrended = data_clean - trend
        else:
            detrended = data_clean
        
        if return_trend:
            return detrended, trend
        return detrended

    def validate_filter_parameters(self):
        """Validate filter parameters according to specifications"""
        warnings = []
        errors = []
        
        if self.data is None:
            errors.append("No data loaded for validation")
            return {'valid': False, 'warnings': warnings, 'errors': errors}
        
        data_length = len(self.data)
        data_duration_days = data_length / self.sampling_freq
        
        # Calculate filter impulse response duration
        filter_order = self.filter_params['filter_order']
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
        f_low = self.filter_params['f_low']
        f_high = self.filter_params['f_high']
        nyquist = self.sampling_freq / 2
        
        if f_low >= f_high:
            errors.append("Low frequency must be less than high frequency")
        
        if f_high >= nyquist:
            errors.append(f"High frequency ({f_high:.3f}) must be less than Nyquist frequency ({nyquist:.3f})")
        
        if f_low <= 0:
            errors.append("Low frequency must be positive")
        
        # Check ramp fraction
        ramp_fraction = self.filter_params['ramp_fraction']
        if ramp_fraction < 0 or ramp_fraction > 0.5:
            errors.append("Ramp fraction must be between 0 and 0.5")
        
        # Warn about diurnal frequency optimization
        if f_low > 1.2 or f_high < 0.8:
            warnings.append("Filter band may not capture diurnal (daily) cycles optimally. "
                           "Consider including 0.8-1.2 day⁻¹ range.")
        
        return {
            'valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors,
            'filter_duration_days': filter_duration_days,
            'data_duration_days': data_duration_days
        }

    def design_filter_with_ramp(self, f_low, f_high, fs, filter_type='butter', order=6, ramp_fraction=0.1):
        """Design bandpass filter with explicit ramp/taper implementation matching specifications"""
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
        
        return b, a, filter_info
    
    def compute_psd_welch(self, data_col, custom_params):
        """Compute PSD using Welch's method with proper mathematical corrections"""
        
        # Ensure all parameters have valid values (not None)
        params = custom_params.copy()
        params['nperseg_divider'] = params.get('nperseg_divider') or 4
        params['nperseg_max'] = params.get('nperseg_max') or 520
        
        # Calculate nperseg based on user formula
        nperseg = min(len(data_col) // int(params['nperseg_divider']), int(params['nperseg_max']))
        
        # Ensure nperseg is at least a reasonable minimum
        nperseg = max(nperseg, 16)
        
        print(f"Welch method: nperseg={nperseg} from data length: {len(data_col)}")
        
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
        
        print(f"Welch PSD computed: {len(frequencies)} frequency bins, max_freq={frequencies.max():.3f}")
        
        return frequencies, psd
    
    def compute_psd_welch_smooth(self, data_col, custom_params):
        """Compute PSD using Welch's method with heavy smoothing for R-like output"""
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
            
            print(f"Smooth method using user formula: nperseg = min({data_len} // {nperseg_divider}, {nperseg_max}) = {nperseg}")
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
            
            print(f"Smooth method using optimal logic for {data_len} points: nperseg = {nperseg}")
        
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
        
        return frequencies, psd_smooth
    
    def compute_psd_multitaper(self, data_col, custom_params):
        """Compute PSD using multitaper method with proper zero-padding to match R implementation"""
        if not MNE_AVAILABLE:
            print("Warning: Multitaper method not available. Using corrected MATLAB-style instead.")
            return self.compute_psd_matlab_style_corrected(data_col, custom_params)
        
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
            print(f"Multitaper: Zero-padded data from {data_len} to {nfft} samples")
        else:
            padded_data = data_col[:nfft]  # Truncate if data is longer than nfft
            print(f"Multitaper: Truncated data from {data_len} to {nfft} samples")
        
        print(f"Multitaper method: bandwidth={bandwidth}, nfft={nfft}, adaptive={adaptive}")
        
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
        
        return freqs, psd[0]  # Return 1D array
    
    def compute_psd_matlab_style_corrected(self, data_col, custom_params):
        """Compute PSD using DPSS tapers with proper mathematical corrections to match MATLAB spectrum.mtm behavior"""
        from scipy.signal.windows import dpss as scipy_dpss
        from scipy.signal import periodogram
        
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
                print(f"MATLAB MTM using user nfft: {nfft}")
        
        # Prevent silent truncation - ensure nfft >= data length
        N = len(data_col)
        if nfft < N:
            print(f"WARNING: nfft ({nfft}) < data length ({N}). Increasing nfft to {N} to avoid truncation.")
            nfft = N
        
        # Calculate number of tapers (MATLAB uses K = 2*NW - 1)
        n_tapers = int(2 * tbw - 1)
        
        print(f"MATLAB MTM Corrected: N={N}, tbw={tbw}, n_tapers={n_tapers}, nfft={nfft}")
        
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
                
            print(f"Generated {n_tapers} DPSS tapers with shape {tapers.shape}")
            print(f"Eigenvalues: {eigenvalues[:min(3, len(eigenvalues))]}")
            
            # Ensure we have the right number of tapers
            if tapers.shape[0] != n_tapers:
                print(f"Warning: Expected {n_tapers} tapers, got {tapers.shape[0]}")
                n_tapers = min(n_tapers, tapers.shape[0])
                eigenvalues = eigenvalues[:n_tapers]
                
        except Exception as e:
            print(f"Failed to generate DPSS tapers: {e}")
            # Fallback to approximation
            tapers = self._create_dpss_approximation(N, tbw, n_tapers)
            eigenvalues = np.ones(n_tapers)  # Equal weights as fallback
            print(f"Using fallback tapers with shape {tapers.shape}")
        
        # Initialize arrays for storing individual taper PSDs
        taper_psds = []
        
        # Proper DPSS taper normalization - pass taper to periodogram
        for k in range(n_tapers):
            # Get the k-th taper
            taper = tapers[k, :]  # Shape should be [n_tapers, N]
            
            # Verify taper shape matches data
            if len(taper) != len(data_col):
                print(f"Error: Taper {k} length {len(taper)} doesn't match data length {len(data_col)}")
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
        
        print(f"Computed {len(taper_psds)} taper PSDs with shape {taper_psds.shape}")
        
        # Apply Thomson's adaptive weighting using eigenvalues
        if len(eigenvalues) == len(taper_psds):
            # Reshape eigenvalues to [n_tapers, 1] for proper broadcasting
            eigenvalues_reshaped = eigenvalues[:len(taper_psds)].reshape(-1, 1)
            
            # Eigenvalue-weighted average with proper broadcasting
            weighted_sum = np.sum(eigenvalues_reshaped * taper_psds, axis=0)
            weight_sum = np.sum(eigenvalues[:len(taper_psds)])
            psd_final = weighted_sum / weight_sum
            
            print(f"Applied eigenvalue weighting: sum(eigenvalues)={weight_sum:.3f}")
        else:
            # Fallback to simple average
            psd_final = np.mean(taper_psds, axis=0)
            print(f"Used simple averaging (eigenvalue mismatch: {len(eigenvalues)} vs {len(taper_psds)})")
        
        return freqs, psd_final

    def _create_dpss_approximation(self, N, tbw, n_tapers):
        """Create approximate DPSS tapers using sine tapers when scipy.signal.windows.dpss is not available"""
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
        method = params.get('method', 'welch')
        
        print(f"Computing PSD using method: {method}")
        
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
        
        # Create output dataframe with exact column names from specification
        output_data = pd.DataFrame()
        
        # WaterDay with 9 decimal places precision (as shown in spec example)
        output_data['WaterDay'] = self.filtered_data['WaterDay'].round(9)
        
        # Filtered temperature columns with exact naming from specification
        output_data['Shallow.Temp.Filt'] = self.filtered_data['TempShallow_Filt'].round(6)
        output_data['Deep.Temp.Filt'] = self.filtered_data['TempDeep_Filt'].round(6)
        
        return output_data

    def apply_filter_with_validation(self, update_params=None):
        """Enhanced filter application with validation and proper trend handling"""
        if self.data is None:
            return False, "No data loaded"

        if update_params:
            self.filter_params.update(update_params)

        # Validate parameters first
        validation = self.validate_filter_parameters()
        if not validation['valid']:
            return False, f"Validation failed: {'; '.join(validation['errors'])}"

        try:
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
            shallow_filt = signal.filtfilt(b, a, shallow_data)
            deep_filt = signal.filtfilt(b, a, deep_data)

            # Apply resampling if requested
            if self.filter_params['resample_interval_minutes'] != self.filter_params['original_interval_minutes']:
                shallow_filt = self.resample_data(
                    shallow_filt,
                    self.filter_params['original_interval_minutes'],
                    self.filter_params['resample_interval_minutes']
                )
                deep_filt = self.resample_data(
                    deep_filt,
                    self.filter_params['original_interval_minutes'],
                    self.filter_params['resample_interval_minutes']
                )

                # Resample other data
                new_time = self.resample_data(
                    self.data['WaterDay'].values,
                    self.filter_params['original_interval_minutes'],
                    self.filter_params['resample_interval_minutes']
                )

                self.filtered_data = pd.DataFrame({
                    'WaterDay': new_time,
                    'TempShallow': self.resample_data(
                        shallow_data,
                        self.filter_params['original_interval_minutes'],
                        self.filter_params['resample_interval_minutes']
                    ),
                    'TempDeep': self.resample_data(
                        deep_data,
                        self.filter_params['original_interval_minutes'],
                        self.filter_params['resample_interval_minutes']
                    ),
                    'TempShallow_Filt': shallow_filt,
                    'TempDeep_Filt': deep_filt,
                    'TempShallow_Trend': self.resample_data(
                        shallow_trend,
                        self.filter_params['original_interval_minutes'],
                        self.filter_params['resample_interval_minutes']
                    ),
                    'TempDeep_Trend': self.resample_data(
                        deep_trend,
                        self.filter_params['original_interval_minutes'],
                        self.filter_params['resample_interval_minutes']
                    )
                })
            else:
                self.filtered_data = self.data.copy()
                self.filtered_data['TempShallow_Filt'] = shallow_filt
                self.filtered_data['TempDeep_Filt'] = deep_filt
                self.filtered_data['TempShallow_Trend'] = shallow_trend
                self.filtered_data['TempDeep_Trend'] = deep_trend

            # Store filter info for reference
            self.last_filter_info = filter_info
            
            return True, f"Filter applied successfully. {'; '.join(validation['warnings']) if validation['warnings'] else ''}"

        except Exception as e:
            return False, f"Filtering failed: {str(e)}"
        
    def resample_data(self, data, original_interval_minutes, target_interval_minutes):
        """Resample data using scipy.signal.resample"""
        from scipy.signal import resample
        
        # Calculate resampling ratio
        ratio = original_interval_minutes / target_interval_minutes
        new_length = int(len(data) * ratio)
        
        if new_length != len(data):
            return resample(data, new_length)
        else:
            return data

    def save_filtered_data_spec_format(self, filename):
        """Save filtered data in exact specification format with folder management"""
        if self.filtered_data is not None:
            # Create output folder if it doesn't exist (this will set the full path)
            if not self.create_output_folder():
                return False
            
            # Now self.output_folder contains the full path, so just join with filename
            full_path = os.path.join(self.output_folder, filename)
            
            output_data = self.format_output_data()
            
            if output_data is not None:
                # Save with exact formatting - no index, specific precision
                output_data.to_csv(full_path, index=False, float_format='%.6f')
                
                # Add to filter history with additional metadata
                self.filter_history.append({
                    'filename': filename,
                    'full_path': full_path,
                    'timestamp': datetime.now().isoformat(),
                    'parameters': self.filter_params.copy(),
                    'data_range': {
                        'start_wd': float(output_data['WaterDay'].min()),
                        'end_wd': float(output_data['WaterDay'].max()),
                        'n_points': len(output_data)
                    },
                    'filter_validation': self.validate_filter_parameters()
                })
                
                print(f"Filtered data saved to: {full_path}")
                return True
        return False
        
    def save_filtered_data(self, filename):
        """Legacy method for backward compatibility"""
        return self.save_filtered_data_spec_format(filename)

    def clip_data_for_analysis(self, wd_lower=None, wd_upper=None):
        """Clip data based on WaterDay limits for spectral analysis"""
        if self.data is None:
            return {'success': False, 'message': 'No data loaded'}
        
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
        
        if self.wd_upper > 0:
            self.analysis_data = self.analysis_data[self.analysis_data['WaterDay'] <= self.wd_upper]
        
        clipped_length = len(self.analysis_data)
        
        # Calculate actual WD range
        if clipped_length > 0:
            actual_wd_min = self.analysis_data['WaterDay'].min()
            actual_wd_max = self.analysis_data['WaterDay'].max()
        else:
            actual_wd_min = actual_wd_max = 0
        
        return {
            'success': True,
            'original_length': original_length,
            'clipped_length': clipped_length,
            'actual_wd_range': (actual_wd_min, actual_wd_max),
            'message': f'Data clipped from {original_length} to {clipped_length} records'
        }


# Initialize the analyzer
analyzer = TemperatureFrequencyAnalyzer()

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                    html.Div(id='upload-status'),
                    html.Div([
                        html.Small(f"Default file from parameters: {analyzer.input_filename}", 
                                 className="text-muted"),
                        html.Br(),
                        html.Small(f"Output folder: {analyzer.output_folder}", 
                                 className="text-info")
                    ])
                ])
            ])
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
                                    {'label': 'Welch (Standard)', 'value': 'welch'},
                                    {'label': 'Welch (Smoothed)', 'value': 'welch_smooth'},
                                    {'label': 'Multitaper (R-style)' + (' - Not Available' if not MNE_AVAILABLE else ''), 
                                    'value': 'multitaper', 'disabled': not MNE_AVAILABLE},
                                    {'label': 'MATLAB MTM Style', 'value': 'matlab_mtm'}
                                ],
                                value=analyzer.psd_params['method']
                            )
                        ], width=6),
                        dbc.Col([
                            html.Div(id='method-specific-params')
                        ], width=6)
                    ], className="mb-3"),
                    
                    html.H5("Welch Method Parameters", className="mt-3 mb-3"),
                    
                    # Segment length parameters
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
                            dbc.Label("Overlap:"),
                            dbc.Input(id='noverlap', type='number', 
                                    placeholder="Default: nperseg//2",
                                    min=0)
                        ], width=2),
                        dbc.Col([
                            dbc.Label("FFT Length:"),
                            dbc.Input(id='nfft', type='number', 
                                    placeholder="Default: nperseg",
                                    min=0)
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Current nperseg:"),
                            html.Div(id='current-nperseg', 
                                className="form-control bg-light")
                        ], width=2)
                    ]),
                    
                    # Advanced parameters
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Detrend:"),
                            dcc.Dropdown(
                                id='detrend',
                                options=[
                                    {'label': 'Constant (DC removal)', 'value': 'constant'},
                                    {'label': 'Linear', 'value': 'linear'},
                                    {'label': 'None', 'value': False}
                                ],
                                value=analyzer.psd_params['detrend']
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Return One-sided:"),
                            dcc.Dropdown(
                                id='return-onesided',
                                options=[
                                    {'label': 'Yes', 'value': True},
                                    {'label': 'No', 'value': False}
                                ],
                                value=analyzer.psd_params['return_onesided']
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Scaling:"),
                            dcc.Dropdown(
                                id='scaling',
                                options=[
                                    {'label': 'Density', 'value': 'density'},
                                    {'label': 'Spectrum', 'value': 'spectrum'}
                                ],
                                value=analyzer.psd_params['scaling']
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Average:"),
                            dcc.Dropdown(
                                id='average',
                                options=[
                                    {'label': 'Mean', 'value': 'mean'},
                                    {'label': 'Median', 'value': 'median'}
                                ],
                                value=analyzer.psd_params['average']
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Button("Update PSD", id='update-psd-btn', 
                                    color='success', className="mt-4"),
                        ], width=3)
                    ], className="mt-3"),
                    
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Apply Data Clipping", id='apply-clip-btn', 
                                    color='warning', className="me-2"),
                            dbc.Button("Enter Frequency Selection Mode", id='freq-select-btn', 
                                    color='info', className="me-2"),
                            dbc.Button("Reset View", id='reset-view-btn', color='secondary', className="me-2"),
                        ])
                    ]),
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
                                    value=analyzer.filter_params['filter_order'], min=2, max=20)
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Low Frequency (day⁻¹):"),
                            dbc.Input(id='f-low', type='number', 
                                    value=analyzer.filter_params['f_low'], min=0.01, max=10, step=0.01)
                        ], width=3),
                        dbc.Col([
                            dbc.Label("High Frequency (day⁻¹):"),
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
                                    value=analyzer.filter_params['resample_interval_minutes'], min=1, max=1440)
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
                                value='dc'  # Default to DC removal as recommended
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

    # Filter Characteristics Display
    dbc.Row([
        dbc.Col([
            html.Div(id='filter-characteristics-card')
        ], width=12)
    ], className="mb-4"),

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
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Save Filtered Data", id='save-data-btn', 
                                     color='success', className="me-2"),
                            dbc.Button("Save Parameters", id='save-params-btn', 
                                     color='secondary'),
                        ], width=6)
                    ]),
                    html.Div([
                        html.Small(f"Files will be saved to folder: {analyzer.output_folder}", 
                                 className="text-info mt-2")
                    ])
                ])
            ])
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
    
    # Raw Data Plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='raw-data-plot', style={'height': '400px'})
        ], width=12)
    ], className="mb-4"),
    
    # Filtered Data Plot
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='filtered-data-plot', style={'height': '400px'})
        ], width=12)
    ]),
    
    # Hidden div to store data
    html.Div(id='data-store', style={'display': 'none'}),
    html.Div(id='frequency-selection-state', children='inactive', style={'display': 'none'}),
    
    # Download components
    dcc.Download(id="download-filtered-data"),
    dcc.Download(id="download-parameters"),
    
], fluid=True)


# Callback for method-specific parameters
@app.callback(
    Output('method-specific-params', 'children'),
    [Input('psd-method', 'value')],
    prevent_initial_call=False
)
def update_method_params(method):
    if method == 'multitaper':
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
    elif method == 'welch_smooth':
        return dbc.Row([
            dbc.Col([
                dbc.Label("Smoothing Sigma:"),
                dbc.Input(
                    id={'type': 'method-param', 'param': 'smooth-sigma'}, 
                    type='number', 
                    value=analyzer.psd_params.get('smooth_sigma', 2), 
                    min=0.5, max=10, step=0.5
                ),
                html.Small("Higher values give smoother spectra", className="text-muted")
            ], width=6),
            dbc.Col([
                dbc.Checklist(
                    id={'type': 'method-param', 'param': 'use-divider-smooth'},
                    options=[{"label": "Use segment length formula", "value": "use_formula"}],
                    value=["use_formula"] if analyzer.psd_params.get('use_divider_for_smooth', False) else [],
                ),
                html.Small("Check to use divider/max instead of optimal segments", className="text-muted")
            ], width=6)
        ])
    elif method == 'matlab_mtm':
        return dbc.Row([
            dbc.Col([
                dbc.Label("Time-Bandwidth (tbw):"),
                dbc.Input(
                    id={'type': 'method-param', 'param': 'matlab-tbw'}, 
                    type='number', 
                    value=4,
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
                html.Small("FFT length - will be increased if < data length", className="text-muted")
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
    else:  # welch method
        return html.Div([
            html.Small("Standard Welch method uses all segment length parameters above", 
                      className="text-info")
        ])
    
@app.callback(
    Output('method-specific-params', 'id'),  # Dummy output
    [Input({'type': 'method-param', 'param': ALL}, 'value')],
    [State({'type': 'method-param', 'param': ALL}, 'id')],
    prevent_initial_call=True
)
def update_method_specific_params(values, ids):
    """Capture and store method-specific parameters"""
    if values and ids:
        for value, id_dict in zip(values, ids):
            param_name = id_dict['param']
            if value is not None:
                # Map the parameter names to analyzer storage
                if param_name == 'time-bandwidth':
                    analyzer.psd_params['time_bandwidth'] = float(value)
                    print(f"Updated time_bandwidth to: {value}")
                elif param_name == 'smooth-sigma':
                    analyzer.psd_params['smooth_sigma'] = float(value)
                    print(f"Updated smooth_sigma to: {value}")
                elif param_name == 'matlab-tbw':
                    analyzer.psd_params['time_bandwidth'] = float(value)
                    print(f"Updated MATLAB tbw to: {value}")
                elif param_name == 'matlab-nfft':
                    analyzer.psd_params['matlab_nfft'] = int(value)
                    print(f"Updated MATLAB nfft to: {value}")
                elif param_name == 'use-divider-smooth':
                    analyzer.psd_params['use_divider_for_smooth'] = 'use_formula' in value if value else False
                    print(f"Updated use_divider_for_smooth to: {analyzer.psd_params['use_divider_for_smooth']}")
                elif param_name == 'use-divider-matlab':
                    analyzer.psd_params['use_divider_for_matlab'] = 'use_formula' in value if value else False
                    print(f"Updated use_divider_for_matlab to: {analyzer.psd_params['use_divider_for_matlab']}")
    
    return 'method-specific-params'

@app.callback(
    [Output('upload-status', 'children'),
     Output('data-store', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def upload_file(contents, filename):
    if contents is not None:
        success = analyzer.load_data(contents, filename)
        if success:
            status = dbc.Alert(f"Successfully loaded {filename}", color="success")
            return status, "data-loaded"
        else:
            status = dbc.Alert("Error loading file. Please check format.", color="danger")
            return status, "no-data"
    return "", "no-data"


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
        status = dbc.Alert("Frequency Selection Mode ACTIVE: Click on the PSD plot to select low and high frequency bounds", 
                          color="warning")
        return 'active', status, "Exit Frequency Selection Mode", 'warning'
    else:  # Even clicks = inactive
        analyzer.frequency_selection_mode = False
        return 'inactive', "", "Enter Frequency Selection Mode", 'info'

@app.callback(
    Output('current-nperseg', 'children'),
    [Input('nperseg-divider', 'value'),
     Input('nperseg-max', 'value'),
     Input('data-store', 'children'),
     Input('apply-clip-btn', 'n_clicks')]
)
def update_nperseg_display(divider, max_val, data_status, clip_clicks):
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

@app.callback(
    [Output('psd-plot', 'figure'),
     Output('f-low', 'value'),
     Output('f-high', 'value')],
    [Input('data-store', 'children'),
     Input('sampling-freq', 'value'),
     Input('psd-plot', 'clickData'),
     Input('reset-view-btn', 'n_clicks'),
     Input('update-psd-btn', 'n_clicks'),
     Input({'type': 'method-param', 'param': ALL}, 'value')],
    [State('frequency-selection-state', 'children'),
     State('psd-method', 'value'),
     State('nperseg-divider', 'value'),
     State('nperseg-max', 'value'),
     State('window-type', 'value'),
     State('noverlap', 'value'),
     State('nfft', 'value'),
     State('detrend', 'value'),
     State('return-onesided', 'value'),
     State('scaling', 'value'),
     State('average', 'value'),
     State({'type': 'method-param', 'param': ALL}, 'id')]
)
def update_psd_plot(data_status, sampling_freq, click_data, reset_clicks, update_clicks,
                   method_param_values, freq_selection_state, psd_method, nperseg_divider, nperseg_max, window_type,
                   noverlap, nfft, detrend, return_onesided, scaling, average, method_param_ids):
    
    ctx = dash.callback_context
    if ctx.triggered:
        print(f"Triggered by: {ctx.triggered[0]['prop_id']}")
        print(f"Update button clicks: {update_clicks}")
    
    if data_status != "data-loaded" or analyzer.data is None:
        return (go.Figure().add_annotation(text="Please upload data first", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False),
                analyzer.filter_params['f_low'], analyzer.filter_params['f_high'])
    
    # Update method-specific parameters if they exist
    if method_param_values and method_param_ids:
        for value, id_dict in zip(method_param_values, method_param_ids):
            if value is not None:
                param_name = id_dict['param']
                if param_name == 'matlab-tbw':
                    analyzer.psd_params['time_bandwidth'] = float(value)
                    print(f"Applied MATLAB tbw: {value}")
                elif param_name == 'matlab-nfft':
                    analyzer.psd_params['matlab_nfft'] = int(value)
                    print(f"Applied MATLAB nfft: {value}")
                elif param_name == 'time-bandwidth':
                    analyzer.psd_params['time_bandwidth'] = float(value)
                elif param_name == 'smooth-sigma':
                    analyzer.psd_params['smooth_sigma'] = float(value)
                elif param_name == 'use-divider-smooth':
                    analyzer.psd_params['use_divider_for_smooth'] = 'use_formula' in value if value else False
                elif param_name == 'use-divider-matlab':
                    analyzer.psd_params['use_divider_for_matlab'] = 'use_formula' in value if value else False
    
    # Update analyzer parameters
    analyzer.sampling_freq = sampling_freq
    
    # Get method-specific parameters from analyzer's stored values
    time_bandwidth = analyzer.psd_params.get('time_bandwidth', 4)
    smooth_sigma = analyzer.psd_params.get('smooth_sigma', 2)
    matlab_nfft = analyzer.psd_params.get('matlab_nfft', 2048)
    
    # Debug output for MATLAB method
    if psd_method == 'matlab_mtm':
        print(f"MATLAB MTM Method - Using tbw={time_bandwidth}, nfft={matlab_nfft}")
    
    # Handle frequency selection clicks
    f_low_new = analyzer.filter_params['f_low']
    f_high_new = analyzer.filter_params['f_high']

    if (freq_selection_state == 'active' and click_data is not None and 
        'points' in click_data and len(click_data['points']) > 0):
        
        freq_clicked = click_data['points'][0]['x']
        
        # If this is the first click or we already have 2 frequencies, start fresh
        if len(analyzer.selected_frequencies) == 0:
            analyzer.selected_frequencies = [freq_clicked]
            f_low_new = freq_clicked
            f_high_new = freq_clicked  # Temporary, will be updated on second click
        elif len(analyzer.selected_frequencies) == 1:
            analyzer.selected_frequencies.append(freq_clicked)
            # Sort the frequencies
            freqs_sorted = sorted(analyzer.selected_frequencies)
            f_low_new = freqs_sorted[0]
            f_high_new = freqs_sorted[1]
        else:
            # Start over with new selection
            analyzer.selected_frequencies = [freq_clicked]
            f_low_new = freq_clicked
            f_high_new = freq_clicked
    
    # Update PSD parameters
    custom_params = {
        'method': psd_method or 'welch',
        'nperseg_divider': nperseg_divider if nperseg_divider is not None else 4,
        'nperseg_max': nperseg_max if nperseg_max is not None else 520,
        'window': window_type if window_type is not None else 'tukey',
        'noverlap': noverlap,
        'nfft': nfft,
        'detrend': detrend if detrend is not None else 'constant',
        'return_onesided': return_onesided if return_onesided is not None else True,
        'scaling': scaling if scaling is not None else 'density',
        'average': average if average is not None else 'mean',
        'time_bandwidth': time_bandwidth,
        'smooth_sigma': smooth_sigma,
        'matlab_nfft': matlab_nfft,
        'use_divider_for_smooth': analyzer.psd_params.get('use_divider_for_smooth', False),
        'use_divider_for_matlab': analyzer.psd_params.get('use_divider_for_matlab', False)
    }
    
    # Store the updated parameters
    analyzer.psd_params.update(custom_params)
    
    # Compute PSD with custom parameters
    try:
        if hasattr(analyzer, 'analysis_data') and analyzer.analysis_data is not None:
            freq_shallow, psd_shallow = analyzer.compute_psd('TempShallow', use_clipped=True, custom_params=custom_params)
            freq_deep, psd_deep = analyzer.compute_psd('TempDeep', use_clipped=True, custom_params=custom_params)
        else:
            freq_shallow, psd_shallow = analyzer.compute_psd('TempShallow', use_clipped=False, custom_params=custom_params)
            freq_deep, psd_deep = analyzer.compute_psd('TempDeep', use_clipped=False, custom_params=custom_params)
    except Exception as e:
        print(f"Error computing PSD: {e}")
        return (go.Figure().add_annotation(text=f"Error computing PSD: {str(e)}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False),
                f_low_new, f_high_new)
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=freq_shallow, y=psd_shallow,
        mode='lines', name='Shallow Temp',
        line=dict(color='blue', width=2),
        hovertemplate='Frequency: %{x:.3f} day⁻¹<br>Power: %{y:.2e}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=freq_deep, y=psd_deep,
        mode='lines', name='Deep Temp',
        line=dict(color='red', width=2),
        hovertemplate='Frequency: %{x:.3f} day⁻¹<br>Power: %{y:.2e}<extra></extra>'
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
                  annotation_text=f"Low: {f_low_new:.3f}",
                  annotation_position="top left",
                  annotation_yshift=10)
    
    fig.add_vline(x=f_high_new, 
                  line_dash="dash", 
                  line_color="orange", 
                  line_width=2,
                  annotation_text=f"High: {f_high_new:.3f}",
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
        'multitaper': 'Multitaper Method',
        'matlab_mtm': f'MATLAB MTM (tbw={time_bandwidth}, nfft={matlab_nfft})'
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
        xaxis_title="Frequency (day⁻¹)",
        yaxis_title="Power Spectral Density",
        yaxis_type="log",
        xaxis=dict(range=[0, 4]),
        hovermode='x unified',
        template='plotly_white',
        margin=dict(t=80)
    )
    
    return fig, f_low_new, f_high_new

@app.callback(
    [Output('filter-validation-status', 'children'),
     Output('filter-characteristics-card', 'children')],
    [Input('f-low', 'value'),
     Input('f-high', 'value'),
     Input('filter-order', 'value'),
     Input('ramp-fraction', 'value')],
    [State('data-store', 'children')]
)
def update_filter_validation(f_low, f_high, filter_order, ramp_fraction, data_status):
    if data_status != "data-loaded" or not all([f_low, f_high, filter_order, ramp_fraction]):
        return "", ""
    
    # Update analyzer parameters for validation
    analyzer.filter_params.update({
        'f_low': f_low,
        'f_high': f_high,
        'filter_order': filter_order,
        'ramp_fraction': ramp_fraction
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
    
    # Create table similar to Table 2 in specifications
    band_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Frequency (day⁻¹)"),
                html.Th("Normalize by Day"),
                html.Th("Normalize by Nyquist"),
                html.Th("Purpose")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(f"{f_start_ramp:.3f}"),
                html.Td(f"{1/f_start_ramp:.8f}" if f_start_ramp > 0 else "N/A"),
                html.Td(f"{f_start_ramp/(analyzer.sampling_freq/2):.8f}"),
                html.Td("Start ramp up")
            ]),
            html.Tr([
                html.Td(f"{f_start_pass:.3f}"),
                html.Td(f"{1/f_start_pass:.8f}" if f_start_pass > 0 else "N/A"),
                html.Td(f"{f_start_pass/(analyzer.sampling_freq/2):.8f}"),
                html.Td("Start full pass")
            ]),
            html.Tr([
                html.Td(f"{f_end_pass:.3f}"),
                html.Td(f"{1/f_end_pass:.8f}" if f_end_pass > 0 else "N/A"),
                html.Td(f"{f_end_pass/(analyzer.sampling_freq/2):.8f}"),
                html.Td("End full pass, start ramp down")
            ]),
            html.Tr([
                html.Td(f"{f_end_ramp:.3f}"),
                html.Td(f"{1/f_end_ramp:.8f}" if f_end_ramp > 0 else "N/A"),
                html.Td(f"{f_end_ramp/(analyzer.sampling_freq/2):.8f}"),
                html.Td("End ramp down")
            ])
        ])
    ], bordered=True, striped=True, size="sm")
    
    filter_characteristics_card = dbc.Card([
        dbc.CardHeader("Filter Characteristics (Specification Compliant)"),
        dbc.CardBody([
            html.H6("Frequency Band Details:"),
            band_table,
            html.Hr(),
            html.P(f"Total band width: {band_width:.3f} day⁻¹"),
            html.P(f"Ramp width: {ramp_width:.3f} day⁻¹ ({ramp_fraction*100:.1f}% of band)"),
            html.P(f"Pass width: {f_end_pass - f_start_pass:.3f} day⁻¹"),
            html.P(f"Estimated filter duration: {validation.get('filter_duration_days', 0):.2f} days")
        ])
    ])
    
    return validation_alerts, filter_characteristics_card

@app.callback(
    Output('raw-data-plot', 'figure'),
    [Input('data-store', 'children')]
)
def update_raw_data_plot(data_status):
    if data_status != "data-loaded" or analyzer.data is None:
        return go.Figure().add_annotation(text="Please upload data first", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=analyzer.data['WaterDay'], y=analyzer.data['TempShallow'],
        mode='lines', name='Shallow Temp',
        line=dict(color='blue', width=1),
        hovertemplate='Day: %{x:.1f}<br>Temperature: %{y:.2f}°C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=analyzer.data['WaterDay'], y=analyzer.data['TempDeep'],
        mode='lines', name='Deep Temp',
        line=dict(color='red', width=1),
        hovertemplate='Day: %{x:.1f}<br>Temperature: %{y:.2f}°C<extra></extra>'
    ))
    
    # Add title with WD limits if applied
    title = "Raw Temperature Data"
    if analyzer.wd_lower > 0 or analyzer.wd_upper > 0:
        title += f" (WD: {analyzer.wd_lower} to {analyzer.wd_upper})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Water Day",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


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
     State('data-store', 'children')]
)
def update_filtered_plot(n_clicks, f_low, f_high, filter_type, filter_order, 
                       ramp_fraction, resample_interval, original_interval, 
                       trend_removal, data_status):
    if data_status != "data-loaded" or analyzer.data is None:
        return go.Figure().add_annotation(text="Please upload data first", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    if n_clicks is None:
        return go.Figure().add_annotation(text="Click 'Apply Filter' to see filtered data", 
                                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
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
        mode='lines', name='Shallow Temp (Filtered)',
        line=dict(color='blue', width=2),
        hovertemplate='Day: %{x:.1f}<br>Temperature: %{y:.2f}°C<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=analyzer.filtered_data['WaterDay'], y=analyzer.filtered_data['TempDeep_Filt'],
        mode='lines', name='Deep Temp (Filtered)',
        line=dict(color='red', width=2),
        hovertemplate='Day: %{x:.1f}<br>Temperature: %{y:.2f}°C<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Filtered Temperature Data (Band: {f_low}-{f_high} day⁻¹, {filter_type}, Order: {filter_order})",
        xaxis_title="Water Day",
        yaxis_title="Filtered Temperature (°C)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


@app.callback(
    [Output("download-filtered-data", "data"),
     Output('filter-history-display', 'children')],
    [Input("save-data-btn", "n_clicks")],
    [State('output-filename', 'value')]
)
def download_filtered_data(n_clicks, filename):
    if n_clicks is None:
        return None, ""
    
    if analyzer.filtered_data is not None:
        # Use custom filename if provided
        if not filename or filename.strip() == "":
            filename = "filtered_temperature_data.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Save to history and get formatted data
        success = analyzer.save_filtered_data_spec_format(filename)
        
        if success:
            # Update filter history display
            history_cards = []
            for i, entry in enumerate(analyzer.filter_history):
                card = dbc.Card([
                    dbc.CardBody([
                        html.H6(f"File: {entry['filename']}", className="card-title"),
                        html.P([
                            f"Band: {entry['parameters']['f_low']:.3f}-{entry['parameters']['f_high']:.3f} day⁻¹, ",
                            f"Filter: {entry['parameters']['filter_type']}, ",
                            f"Order: {entry['parameters']['filter_order']}, ",
                            f"Trend: {entry['parameters']['trend_removal']}"
                        ], className="card-text small"),
                        html.P(f"Saved: {entry['timestamp'][:19]}", className="card-text small text-muted"),
                        html.P(f"Location: {entry['full_path']}", className="card-text small text-info")
                    ])
                ], color="light", outline=True, style={"margin-bottom": "5px"})
                history_cards.append(card)
            
            history_display = html.Div([
                html.H5("Filter History:"),
                html.Div(history_cards)
            ]) if history_cards else ""
            
            # Get the formatted output data for download
            output_data = analyzer.format_output_data()
            return (dcc.send_data_frame(output_data.to_csv, filename, index=False),
                    history_display)
    
    return None, ""


@app.callback(
    Output("download-parameters", "data"),
    [Input("save-data-btn", "n_clicks")],
    prevent_initial_call=True
)
def download_parameters(n_clicks):
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
            return dbc.Alert("Parameters saved to tts_frefil.par", color="success", dismissable=True)
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
    
    # Apply clipping
    result = analyzer.clip_data_for_analysis(wd_lower, wd_upper)
    
    if result['success']:
        if result['original_length'] == result['clipped_length']:
            message = "No clipping applied (limits include all data)"
            color = "info"
        else:
            message = (f"Data clipped: {result['original_length']} → {result['clipped_length']} records. "
                      f"WD range: {result['actual_wd_range'][0]:.2f} to {result['actual_wd_range'][1]:.2f}")
            color = "success"
        
        return dbc.Alert(message, color=color, dismissable=True)
    else:
        return dbc.Alert(result['message'], color="danger", dismissable=True)

# Optional: Load data automatically if specified in parameters
@app.callback(
    Output('upload-data', 'children'),
    [Input('data-store', 'children')],
    prevent_initial_call=True
)
def auto_load_data(data_status):
    if data_status == "no-data" and hasattr(analyzer, 'input_filename'):
        # Check if the file exists
        if os.path.exists(analyzer.input_filename):
            # Try to load it automatically
            if analyzer.load_data(analyzer.input_filename):
                return html.Div([
                    'File loaded from parameters: ',
                    html.B(analyzer.input_filename)
                ])
    
    return html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ])


if __name__ == '__main__':
    print("Temperature Time-Series Frequency Analysis & Filtering Tool")
    print("="*60)
    
    # Check for MNE availability
    if MNE_AVAILABLE:
        print("✓ MNE library available - Multitaper method enabled")
    else:
        print("✗ MNE library not found - Multitaper method disabled")
        print("  To enable multitaper support, install MNE: pip install mne")
    
    print("="*60)
    
    # Check for parameter file
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    par_file = None
    
    for directory in search_dirs:
        potential_path = os.path.join(directory, 'tts_frefil.par')
        if os.path.isfile(potential_path):
            par_file = potential_path
            break
    
    if par_file:
        print(f"Found parameter file: {par_file}")
    else:
        print("No parameter file found. Using default values.")
        print("Create 'tts_frefil.par' in current or script directory.")
    
    print("="*60)
    
    # Get script directory for output folder display
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expected_output_path = os.path.join(script_dir, analyzer.output_folder)
    
    print(f"Output folder configured: {analyzer.output_folder}")
    print(f"Full output path will be: {expected_output_path}")
    
    if not os.path.exists(expected_output_path):
        print(f"Output folder will be created when first file is saved.")
    else:
        print(f"Output folder exists and ready for use.")
    print("="*60)
    print("Starting web application...")
    print("Open your browser to http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the application")
    print("="*60)
    
    app.run(debug=True, host='127.0.0.1', port=8050)
        