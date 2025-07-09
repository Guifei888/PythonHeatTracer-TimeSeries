"""
Thermal Probe Peak Picker Application
=====================================
A comprehensive tool for analyzing temperature time series data from thermal probes
to identify peaks and troughs in shallow and deep temperature measurements.

This application is designed for hydrological research, specifically for analyzing
streambed temperature patterns to understand water flow and heat transport.

Key Features:
- Multiple peak detection algorithms (SciPy, Wavelet, Derivative, Prominence)
- Interactive manual peak selection and editing
- Time range exclusion capabilities
- Automatic error checking for peak quality
- Parameter persistence via .par configuration files
- Customizable output filenames and folders

Requirements:
- dash >= 2.9.0 (for allow_duplicate=True support)
- plotly (for interactive graphing)
- pandas (for data manipulation)
- numpy (for numerical operations)
- scipy (for signal processing)
- PyWavelets (for wavelet analysis)

Author: Timothy Wu
Created: 7/8/2025
Last Updated: 7/8/2025
Version: 1.0
Date: 2025
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
from datetime import datetime
import base64
import io
from io import StringIO
import json
import os
import configparser
from pathlib import Path
import re

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

# Default parameter file path
PARAM_FILE = SCRIPT_DIR / 'tts_pickpeak.par'

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
        'slope_threshold': 0.001,  # Derivative threshold for peak detection (°C/min)
        'min_distance': 20,  # Minimum samples between peaks
        'prominence_factor': 0.15,  # Lower default for better sensitivity
        'output_folder': 'peak_analysis_output',  # Output directory name
        'peak_method': 'combined',  # Default to most sensitive method
        'shallow_color': 'blue',  # Color for shallow temperature trace
        'deep_color': 'red',  # Color for deep temperature trace
        'peak_size': 12,  # Size of peak markers
        'trough_size': 10,  # Size of trough markers
        'line_width': 2  # Width of temperature trace lines
    }
    
    # Load from file if it exists
    if os.path.exists(filepath):
        config = configparser.ConfigParser()
        config.read(filepath)
        
        if 'PARAMETERS' in config:
            # Update defaults with file values
            for key in params:
                if key in config['PARAMETERS']:
                    # Convert to appropriate type based on parameter
                    if key in ['sensor_spacing', 'target_period', 'search_tolerance', 
                              'slope_threshold', 'prominence_factor']:
                        params[key] = float(config['PARAMETERS'][key])
                    elif key in ['min_distance', 'peak_size', 'trough_size', 'line_width']:
                        params[key] = int(config['PARAMETERS'][key])
                    else:
                        params[key] = config['PARAMETERS'][key]
    
    return params

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
    return output_path

# Load initial parameters
initial_params = load_param_file()

# ===========================
# INITIALIZE DASH APPLICATION
# ===========================
app = dash.Dash(__name__)

# ===========================
# PEAK DETECTION ALGORITHMS
# ===========================

def scipy_find_peaks_method(data, distance=None, prominence=None, height=None):
    """
    Standard scipy find_peaks method with adaptive parameters.
    
    This method uses SciPy's find_peaks function with intelligent defaults
    based on the data characteristics.
    
    Parameters:
    -----------
    data : array-like
        Temperature data to analyze
    distance : int, optional
        Minimum distance between peaks in samples
    prominence : float, optional
        Minimum prominence of peaks. If None, calculated from data std
    height : float, optional
        Minimum height of peaks
    
    Returns:
    --------
    array : Indices of detected peaks
    """
    # Remove any DC offset by subtracting mean
    data_centered = data - np.mean(data)
    
    # Calculate adaptive prominence if not provided
    if prominence is None:
        data_std = np.std(data_centered)
        # Use a lower threshold to catch more peaks
        prominence = data_std * 0.1  # Very low threshold
    
    # Build kwargs dictionary for find_peaks
    kwargs = {}
    
    if distance is not None and distance > 1:
        kwargs['distance'] = distance
        
    if prominence is not None and prominence > 0:
        kwargs['prominence'] = prominence
    
    # Don't use height restriction by default as it can miss valid peaks
    if height is not None:
        kwargs['height'] = height
    
    # Find peaks with relaxed parameters
    peaks, properties = find_peaks(data_centered, **kwargs)
    
    # If we found too few peaks, try again with even more relaxed parameters
    if len(peaks) < 10 and distance is not None:  # Expecting more peaks
        # Try without distance constraint
        kwargs_relaxed = kwargs.copy()
        if 'distance' in kwargs_relaxed:
            kwargs_relaxed['distance'] = max(1, distance // 2)
        peaks_relaxed, _ = find_peaks(data_centered, **kwargs_relaxed)
        
        if len(peaks_relaxed) > len(peaks):
            peaks = peaks_relaxed
            print(f"Relaxed distance constraint, found {len(peaks)} peaks")
    
    # If still too few, try with minimal constraints
    if len(peaks) < 5:
        # Just find all local maxima
        all_maxima = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                all_maxima.append(i)
        
        if len(all_maxima) > len(peaks):
            print(f"Using all local maxima: {len(all_maxima)} peaks")
            # Filter by distance if needed
            if distance is not None and distance > 1 and len(all_maxima) > 1:
                filtered = [all_maxima[0]]
                for i in range(1, len(all_maxima)):
                    if all_maxima[i] - filtered[-1] >= distance:
                        filtered.append(all_maxima[i])
                peaks = np.array(filtered, dtype=int)
            else:
                peaks = np.array(all_maxima, dtype=int)
    
    # Return as integer array
    if len(peaks) == 0:
        return np.array([], dtype=int)
    elif isinstance(peaks, list):
        return np.array(peaks, dtype=int)
    else:
        return peaks.astype(int)

def wavelet_peak_detection(data, widths=None, min_snr=1):
    """
    Continuous Wavelet Transform peak detection using PyWavelets.
    
    This method uses the Mexican hat wavelet to identify peaks across
    multiple scales, making it robust to noise and varying peak widths.
    
    Parameters:
    -----------
    data : array-like
        Temperature data to analyze
    widths : array-like, optional
        Range of widths for wavelet transform
    min_snr : float
        Minimum signal-to-noise ratio for peak detection
    
    Returns:
    --------
    array : Indices of detected peaks
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
    
    return peaks

def derivative_peak_detection(data, time, slope_threshold=None, min_peak_distance=10):
    """
    Custom derivative-based peak and trough detection.
    
    This method finds peaks and troughs by analyzing the derivative
    of the signal, looking for zero-crossings with sign changes.
    
    Parameters:
    -----------
    data : array-like
        Temperature data to analyze
    time : array-like
        Time values corresponding to data points
    slope_threshold : float, optional
        Threshold for derivative. If None, calculated adaptively
    min_peak_distance : int
        Minimum distance between peaks in samples
    
    Returns:
    --------
    tuple : (peak_indices, trough_indices) as integer arrays
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
    
    print(f"Derivative method found {len(peaks)} peaks and {len(troughs)} troughs")
    
    return peaks, troughs

def combined_peak_detection(data, time, target_period=1.0, min_peak_distance=10):
    """
    Combined peak detection using multiple methods for robustness.
    
    This method combines results from multiple peak detection algorithms
    to ensure no peaks are missed.
    
    Parameters:
    -----------
    data : array-like
        Temperature data to analyze
    time : array-like
        Time values corresponding to data points
    target_period : float
        Expected period between peaks (days)
    min_peak_distance : int
        Minimum distance between peaks in samples
    
    Returns:
    --------
    array : Indices of detected peaks
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
    
    print(f"Combined method found {len(all_peaks)} peaks")
    
    return np.array(all_peaks, dtype=int) if len(all_peaks) > 0 else np.array([], dtype=int)


def prominence_based_detection(data, prominence_factor=0.1):
    """
    Peak detection based on prominence relative to signal amplitude.
    
    This method finds peaks that stand out by a certain fraction
    of the overall signal range.
    
    Parameters:
    -----------
    data : array-like
        Temperature data to analyze
    prominence_factor : float
        Fraction of data range to use as minimum prominence
    
    Returns:
    --------
    array : Indices of detected peaks
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
        print(f"Relaxed prominence to {min_prominence:.4f}, found {len(peaks)} peaks")
    
    if len(peaks) == 0:
        return np.array([], dtype=int)
    return peaks.astype(int)

def bootstrap_peak_detection(data, time, initial_peaks, period=1.0, tolerance=0.2):
    """
    Bootstrap peak detection from user-selected initial peaks.
    
    This method uses manually selected peaks to establish a pattern,
    then propagates that pattern forward and backward through the data.
    
    Parameters:
    -----------
    data : array-like
        Temperature data to analyze
    time : array-like
        Time values corresponding to data points
    initial_peaks : array-like
        Indices of manually selected peaks
    period : float
        Expected period between peaks (days)
    tolerance : float
        Search window around expected peak location (days)
    
    Returns:
    --------
    array : Complete set of peak indices including bootstrapped peaks
    """
    if len(initial_peaks) < 2:
        # If less than 2 peaks, return what we have
        return np.array(initial_peaks, dtype=int)
    
    # Sort initial peaks
    initial_peaks = sorted(initial_peaks)
    
    # Calculate average period from initial peaks
    peak_times = time[initial_peaks]
    periods = np.diff(peak_times)
    avg_period = np.mean(periods) if len(periods) > 0 else period
    
    print(f"Bootstrap: Starting with {len(initial_peaks)} peaks, avg period = {avg_period:.3f} days")
    
    all_peaks = list(initial_peaks)
    
    # Forward propagation
    last_peak_time = peak_times[-1]
    last_peak_idx = initial_peaks[-1]
    
    while last_peak_idx < len(data) - 10:
        # Look for next peak around expected time
        expected_time = last_peak_time + avg_period
        search_start = max(0, np.searchsorted(time, expected_time - tolerance))
        search_end = min(len(data), np.searchsorted(time, expected_time + tolerance))
        
        if search_start < len(data) and search_end > search_start:
            window_data = data[search_start:search_end]
            if len(window_data) > 0:
                # Find local maximum in search window
                local_peak = np.argmax(window_data)
                peak_idx = search_start + local_peak
                
                # Only add if it's a true local maximum
                if peak_idx > 0 and peak_idx < len(data) - 1:
                    if data[peak_idx] >= data[peak_idx - 1] and data[peak_idx] >= data[peak_idx + 1]:
                        if peak_idx not in all_peaks:
                            all_peaks.append(peak_idx)
                            last_peak_time = time[peak_idx]
                            last_peak_idx = peak_idx
                        else:
                            break
                    else:
                        break
                else:
                    break
            else:
                break
        else:
            break
    
    # Backward propagation
    first_peak_time = peak_times[0]
    first_peak_idx = initial_peaks[0]
    
    while first_peak_idx > 10:
        expected_time = first_peak_time - avg_period
        search_start = max(0, np.searchsorted(time, expected_time - tolerance))
        search_end = min(len(data), np.searchsorted(time, expected_time + tolerance))
        
        if search_start < len(data) and search_end > search_start:
            window_data = data[search_start:search_end]
            if len(window_data) > 0:
                local_peak = np.argmax(window_data)
                peak_idx = search_start + local_peak
                
                # Only add if it's a true local maximum
                if peak_idx > 0 and peak_idx < len(data) - 1:
                    if data[peak_idx] >= data[peak_idx - 1] and data[peak_idx] >= data[peak_idx + 1]:
                        if peak_idx not in all_peaks:
                            all_peaks.insert(0, peak_idx)
                            first_peak_time = time[peak_idx]
                            first_peak_idx = peak_idx
                        else:
                            break
                    else:
                        break
                else:
                    break
            else:
                break
        else:
            break
    
    print(f"Bootstrap: Found {len(all_peaks)} total peaks")
    
    return np.array(sorted(all_peaks), dtype=int)

# ===========================
# ERROR CHECKING FUNCTIONS
# ===========================

def check_alternation(shallow_peaks_times, deep_peaks_times):
    """
    Check if peaks alternate properly between shallow and deep.
    
    Proper alternation means shallow → deep → shallow → deep pattern.
    This is essential for valid heat transport analysis.
    
    Parameters:
    -----------
    shallow_peaks_times : array-like
        Time values of shallow peaks
    deep_peaks_times : array-like
        Time values of deep peaks
    
    Returns:
    --------
    list : List of error dictionaries with type, time, and message
    """
    errors = []
    
    # Combine and sort all peaks with their types
    all_peaks = []
    for t in shallow_peaks_times:
        all_peaks.append((t, 'shallow'))
    for t in deep_peaks_times:
        all_peaks.append((t, 'deep'))
    
    all_peaks.sort(key=lambda x: x[0])
    
    # Check for alternation violations
    for i in range(1, len(all_peaks)):
        if all_peaks[i][1] == all_peaks[i-1][1]:
            errors.append({
                'type': 'alternation',
                'time': all_peaks[i][0],
                'message': f'Two {all_peaks[i][1]} peaks in a row at time {all_peaks[i][0]:.2f}'
            })
    
    return errors

def check_amplitude_and_phase(shallow_peaks, deep_peaks, shallow_data, deep_data, 
                             shallow_times, deep_times, tolerance=0.2):
    """
    Check amplitude and phase shift constraints.
    
    Valid peaks should have:
    - Shallow amplitude > Deep amplitude (As > Ad)
    - Positive phase shift (deep peak after shallow peak)
    - Reasonable phase shift (typically < 1.5 days)
    
    Parameters:
    -----------
    shallow_peaks : array-like
        Indices of shallow peaks
    deep_peaks : array-like
        Indices of deep peaks
    shallow_data : array-like
        Shallow temperature data
    deep_data : array-like
        Deep temperature data
    shallow_times : array-like
        Time values for data points
    deep_times : array-like
        Time values for data points (usually same as shallow_times)
    tolerance : float
        Not used in current implementation
    
    Returns:
    --------
    list : List of error dictionaries
    """
    errors = []
    
    # Check each shallow peak
    for s_idx in shallow_peaks:
        s_time = shallow_times[s_idx]
        s_amp = shallow_data[s_idx]
        
        # Find the next deep peak after this shallow peak
        deep_after = deep_peaks[deep_times[deep_peaks] > s_time]
        
        if len(deep_after) > 0:
            # Get the first deep peak after shallow
            d_idx = deep_after[0]
            d_time = deep_times[d_idx]
            d_amp = deep_data[d_idx]
            
            # Check amplitude constraint: As > Ad
            if s_amp <= d_amp:
                errors.append({
                    'type': 'amplitude',
                    'time': s_time,
                    'message': f'Shallow amplitude ({s_amp:.3f}) ≤ Deep amplitude ({d_amp:.3f}) at time {s_time:.2f}'
                })
            
            # Check phase shift is positive and reasonable
            phase_shift = d_time - s_time
            if phase_shift <= 0:
                errors.append({
                    'type': 'phase',
                    'time': s_time,
                    'message': f'Negative phase shift at time {s_time:.2f}'
                })
            elif phase_shift > 1.5:  # More than 1.5 days is suspicious
                errors.append({
                    'type': 'phase',
                    'time': s_time,
                    'message': f'Phase shift too large ({phase_shift:.2f} days) at time {s_time:.2f}'
                })
    
    return errors

def check_peak_trough_pairing(peaks, troughs, times, tolerance=0.5):
    """
    Check that each peak has a matching trough within tolerance.
    
    For proper amplitude calculation, each peak should be paired
    with a trough occurring within approximately half a period.
    
    Parameters:
    -----------
    peaks : array-like
        Indices of peaks
    troughs : array-like
        Indices of troughs
    times : array-like
        Time values for data points
    tolerance : float
        Maximum time difference for peak-trough pairing (days)
    
    Returns:
    --------
    list : List of error dictionaries
    """
    errors = []
    
    for p_idx in peaks:
        p_time = times[p_idx]
        
        # Find troughs within tolerance
        trough_times = times[troughs]
        nearby_troughs = troughs[np.abs(trough_times - p_time) <= tolerance]
        
        if len(nearby_troughs) == 0:
            errors.append({
                'type': 'pairing',
                'time': p_time,
                'message': f'No trough found within {tolerance} days of peak at time {p_time:.2f}'
            })
    
    return errors

# ===========================
# DASH LAYOUT
# ===========================

app.layout = html.Div([
    # Application header
    html.H1("Thermal Probe Peak Picker", style={'textAlign': 'center'}),
    
    # Parameter file controls section
    html.Div([
        html.Label("Parameter File: "),
        html.Span(str(PARAM_FILE), id='param-file-path'),
        html.Button('Load Parameters', id='load-param-file-button', n_clicks=0, 
                   style={'marginLeft': '10px'}),
        html.Button('Save to .par', id='save-param-file-button', n_clicks=0, 
                   style={'marginLeft': '10px'}),
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
    
    # Peak detection method selection
    html.Div([
        html.Label("Peak Detection Method:"),
        dcc.Dropdown(
            id='peak-method-dropdown',
            options=[
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
    ], style={'margin': '10px'}),
    
    # Detection parameters section
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
                html.Label("Slope Threshold (°C/min):"),
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
        # Action buttons
        html.Div([
            html.Button('Detect Peaks', id='detect-button', n_clicks=0),
            html.Button('Clear Manual Selections', id='clear-manual-button', n_clicks=0, 
                       style={'marginLeft': '10px'}),
            html.Button('Apply Realtime', id='realtime-toggle', n_clicks=0, 
                       style={'marginLeft': '10px'})
        ], style={'marginTop': '10px'}),
        # Tips for better detection
        html.Div([
            html.P("💡 Tips: If peaks are missed, try:", style={'margin': '5px 0', 'fontWeight': 'bold'}),
            html.Ul([
                html.Li("Reduce Prominence Factor (e.g., 0.1 or 0.05)"),
                html.Li("Reduce Min Peak Distance"),
                html.Li("Use 'Combined Methods' for maximum sensitivity"),
                html.Li("Use 'Custom Derivative Method' for noisy data"),
                html.Li("Switch to 'Show All Local Maxima' mode to see all possible peaks"),
                html.Li("Manually add missed peaks and use 'Bootstrap' method")
            ], style={'fontSize': '0.9em', 'margin': '5px 0'})
        ], style={'backgroundColor': '#f0f8ff', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'})
    ], style={'margin': '10px', 'padding': '10px', 'border': '1px solid #ddd'}),
    
    # Interactive controls section
    html.Div([
        html.H3("Interactive Controls"),
        html.Div([
            html.Label("Mode:"),
            dcc.RadioItems(
                id='interaction-mode',
                options=[
                    {'label': 'View Only', 'value': 'view'},
                    {'label': 'Add Shallow Peaks', 'value': 'add_shallow'},
                    {'label': 'Add Deep Peaks', 'value': 'add_deep'},
                    {'label': 'Remove Peaks', 'value': 'remove'},
                    {'label': 'Exclude Time Range', 'value': 'exclude'},
                    {'label': 'Show All Local Maxima', 'value': 'show_all'}
                ],
                value='view',
                inline=True
            )
        ]),
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
    
    # Main graph
    dcc.Graph(id='main-graph', style={'height': '600px'}),
    
    # Error display section
    html.Div(id='error-display', style={'margin': '10px', 'padding': '10px'}),
    
    # Export section with filename customization
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
        html.Div([
            html.Button('Check Errors', id='check-errors-button', n_clicks=0),
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
    dcc.Interval(id='realtime-interval', interval=1000, disabled=True)
])

# ===========================
# CALLBACKS
# ===========================

@app.callback(
    Output('stored-data', 'children'),
    Output('main-graph', 'figure'),
    Output('current-filename', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_file(contents, filename):
    """
    Handle file upload and create initial visualization.
    
    This callback processes uploaded CSV files containing filtered
    temperature data and creates the initial plot.
    """
    if contents is None:
        return None, go.Figure(), None
    
    # Decode uploaded file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Read CSV data
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Create initial plot
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
        
        # Configure plot layout
        fig.update_layout(
            title=f'Filtered Temperature Data: {filename}',
            xaxis_title='Water Day',
            yaxis_title='Temperature (°C)',
            hovermode='x unified',
            clickmode='event+select'
        )
        
        # Store data as JSON and filename
        return df.to_json(date_format='iso', orient='split'), fig, filename
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, go.Figure(), None

@app.callback(
    Output('stored-peaks', 'children'),
    Output('main-graph', 'figure', allow_duplicate=True),
    Output('manual-selections-store', 'children'),
    [Input('detect-button', 'n_clicks'),
     Input('realtime-interval', 'n_intervals'),
     Input('main-graph', 'clickData'),
     Input('add-exclusion-button', 'n_clicks'),
     Input('clear-exclusions-button', 'n_clicks')],
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
     State('sensor-spacing', 'value')],
    prevent_initial_call=True
)
def detect_peaks_and_interact(n_clicks, n_intervals, click_data, add_exclusion_clicks, 
                            clear_exclusions_clicks, stored_data, method, target_period, 
                            tolerance, slope_threshold, min_distance, prominence_factor, 
                            interaction_mode, manual_store, exclude_start, exclude_end, sensor_spacing):
    """
    Main callback for peak detection and interaction.
    
    This comprehensive callback handles:
    - Automatic peak detection using selected algorithm
    - Manual peak selection/deselection
    - Time range exclusions
    - Real-time parameter updates
    - Plot updates with detected and manual peaks
    """
    if stored_data is None:
        return None, go.Figure(), None
    
    # Load data
    df = pd.read_json(StringIO(stored_data), orient='split')
    
    # Extract data arrays
    water_day = df['WaterDay'].values
    shallow_temp = df['Shallow.Temp.Filt'].values
    deep_temp = df['Deep.Temp.Filt'].values
    
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
    
    # Determine which input triggered the callback
    triggered = ctx.triggered_id
    
    # Handle exclusion range additions
    if triggered == 'add-exclusion-button' and exclude_start is not None and exclude_end is not None:
        manual_selections['excluded_ranges'].append([exclude_start, exclude_end])
    
    # Handle clearing exclusions
    if triggered == 'clear-exclusions-button':
        manual_selections['excluded_ranges'] = []
    
    # Handle click interactions for manual peak selection
    if triggered == 'main-graph' and click_data and interaction_mode != 'view':
        click_point = click_data['points'][0]
        click_x = click_point['x']
        click_y = click_point.get('y', 0)
        
        # Find nearest data point
        nearest_idx = int(np.argmin(np.abs(water_day - click_x)))
        
        # Get the y-values at this index for both traces
        shallow_y = shallow_temp[nearest_idx]
        deep_y = deep_temp[nearest_idx]
        
        # For debugging
        print(f"Click at x={click_x:.2f}, y={click_y:.3f}")
        print(f"Nearest point: idx={nearest_idx}, day={water_day[nearest_idx]:.2f}")
        print(f"Shallow temp={shallow_y:.3f}, Deep temp={deep_y:.3f}")
        
        if interaction_mode == 'add_shallow':
            if nearest_idx not in manual_selections['shallow_peaks']:
                manual_selections['shallow_peaks'].append(nearest_idx)
                print(f"Added shallow peak at index {nearest_idx}, day {water_day[nearest_idx]:.2f}, temp {shallow_y:.3f}")
            else:
                print(f"Shallow peak already exists at index {nearest_idx}")
                
        elif interaction_mode == 'add_deep':
            if nearest_idx not in manual_selections['deep_peaks']:
                manual_selections['deep_peaks'].append(nearest_idx)
                print(f"Added deep peak at index {nearest_idx}, day {water_day[nearest_idx]:.2f}, temp {deep_y:.3f}")
            else:
                print(f"Deep peak already exists at index {nearest_idx}")
                
        elif interaction_mode == 'remove':
            removed = False
            
            # Remove from manual selections first
            if nearest_idx in manual_selections['shallow_peaks']:
                manual_selections['shallow_peaks'].remove(nearest_idx)
                print(f"Removed manual shallow peak at index {nearest_idx}")
                removed = True
            
            if nearest_idx in manual_selections['deep_peaks']:
                manual_selections['deep_peaks'].remove(nearest_idx)
                print(f"Removed manual deep peak at index {nearest_idx}")
                removed = True
            
            if not removed:
                # Check if it's near a detected peak by looking at actual peak arrays
                # This helps user understand what peaks are detected vs manual
                print(f"No manual peak found at index {nearest_idx} to remove")
    
    # Detect peaks based on selected method
    shallow_troughs = np.array([], dtype=int)
    deep_troughs = np.array([], dtype=int)
    
    if method == 'scipy':
        # Estimate distance parameter based on target period
        samples_per_day = len(water_day) / (water_day[-1] - water_day[0])
        min_distance_samples = int(target_period * samples_per_day * 0.5)  # Reduced from 0.7 for better sensitivity
        
        # Calculate prominence based on data std and user factor
        if prominence_factor is None:
            prominence_factor = 0.3
        
        # For peak detection, use centered data for better results
        shallow_centered = shallow_temp - np.mean(shallow_temp)
        deep_centered = deep_temp - np.mean(deep_temp)
        
        # Calculate prominence
        shallow_prominence = np.std(shallow_centered) * prominence_factor
        deep_prominence = np.std(deep_centered) * prominence_factor
        
        print(f"SciPy detection parameters:")
        print(f"  Min distance: {max(min_distance_samples, min_distance)} samples")
        print(f"  Shallow prominence: {shallow_prominence:.4f}")
        print(f"  Deep prominence: {deep_prominence:.4f}")
        
        # Detect peaks
        shallow_peaks = scipy_find_peaks_method(shallow_temp, 
                                              distance=max(min_distance_samples, min_distance),
                                              prominence=shallow_prominence,
                                              height=None)
        deep_peaks = scipy_find_peaks_method(deep_temp, 
                                           distance=max(min_distance_samples, min_distance),
                                           prominence=deep_prominence,
                                           height=None)
        
        print(f"Found {len(shallow_peaks)} shallow peaks, {len(deep_peaks)} deep peaks")
        
        # Find troughs by inverting the signal
        shallow_troughs = scipy_find_peaks_method(-shallow_temp, 
                                                distance=max(min_distance_samples, min_distance),
                                                prominence=shallow_prominence,
                                                height=None)
        deep_troughs = scipy_find_peaks_method(-deep_temp, 
                                             distance=max(min_distance_samples, min_distance),
                                             prominence=deep_prominence,
                                             height=None)
        
        print(f"Found {len(shallow_troughs)} shallow troughs, {len(deep_troughs)} deep troughs")
        
    elif method == 'wavelet':
        shallow_peaks = wavelet_peak_detection(shallow_temp)
        deep_peaks = wavelet_peak_detection(deep_temp)
        shallow_troughs = wavelet_peak_detection(-shallow_temp)
        deep_troughs = wavelet_peak_detection(-deep_temp)
        
    elif method == 'derivative':
        shallow_peaks, shallow_troughs = derivative_peak_detection(
            shallow_temp, water_day, slope_threshold, min_distance)
        deep_peaks, deep_troughs = derivative_peak_detection(
            deep_temp, water_day, slope_threshold, min_distance)
        
    elif method == 'prominence':
        shallow_peaks = prominence_based_detection(shallow_temp, prominence_factor)
        deep_peaks = prominence_based_detection(deep_temp, prominence_factor)
        shallow_troughs = prominence_based_detection(-shallow_temp, prominence_factor)
        deep_troughs = prominence_based_detection(-deep_temp, prominence_factor)
        
    elif method == 'combined':
        # Use combined method for maximum sensitivity
        samples_per_day = len(water_day) / (water_day[-1] - water_day[0])
        min_distance_samples = int(target_period * samples_per_day * 0.5)
        
        shallow_peaks = combined_peak_detection(
            shallow_temp, water_day, target_period, 
            max(min_distance_samples, min_distance))
        deep_peaks = combined_peak_detection(
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
            shallow_peaks = bootstrap_peak_detection(
                shallow_temp, water_day, 
                np.array(manual_selections['shallow_peaks'], dtype=int), 
                target_period, tolerance)
        else:
            # If not enough manual peaks, just use what we have
            shallow_peaks = np.array(manual_selections['shallow_peaks'], dtype=int)
            
        if len(manual_selections['deep_peaks']) >= 2:
            deep_peaks = bootstrap_peak_detection(
                deep_temp, water_day, 
                np.array(manual_selections['deep_peaks'], dtype=int), 
                target_period, tolerance)
        else:
            # If not enough manual peaks, just use what we have
            deep_peaks = np.array(manual_selections['deep_peaks'], dtype=int)
        
        # For bootstrap mode, we don't detect troughs automatically
        shallow_troughs = np.array([], dtype=int)
        deep_troughs = np.array([], dtype=int)
    
    # Merge with manual selections if not in bootstrap mode
    if method != 'bootstrap':
        if len(manual_selections['shallow_peaks']) > 0:
            print(f"Manual shallow peaks: {manual_selections['shallow_peaks']}")
            print(f"Detected shallow peaks: {len(shallow_peaks)}")
            # Combine and remove duplicates
            all_shallow = np.concatenate([shallow_peaks, np.array(manual_selections['shallow_peaks'], dtype=int)])
            shallow_peaks = np.unique(all_shallow)
            print(f"Merged shallow peaks: {len(shallow_peaks)} total")
        
        if len(manual_selections['deep_peaks']) > 0:
            print(f"Manual deep peaks: {manual_selections['deep_peaks']}")
            print(f"Detected deep peaks: {len(deep_peaks)}")
            # Combine and remove duplicates
            all_deep = np.concatenate([deep_peaks, np.array(manual_selections['deep_peaks'], dtype=int)])
            deep_peaks = np.unique(all_deep)
            print(f"Merged deep peaks: {len(deep_peaks)} total")
    
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
    
    # Add peaks with darker colors
    if len(shallow_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[shallow_peaks], y=shallow_temp[shallow_peaks],
            mode='markers', name=f'Shallow Peaks ({len(shallow_peaks)})',
            marker=dict(color='darkblue', size=current_params['peak_size'], 
                       symbol='circle-open', line=dict(width=2))
        ))
    
    if len(deep_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[deep_peaks], y=deep_temp[deep_peaks],
            mode='markers', name=f'Deep Peaks ({len(deep_peaks)})',
            marker=dict(color='darkred', size=current_params['peak_size'], 
                       symbol='circle-open', line=dict(width=2))
        ))
    
    # Add troughs with darker colors
    if len(shallow_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[shallow_troughs], y=shallow_temp[shallow_troughs],
            mode='markers', name=f'Shallow Troughs ({len(shallow_troughs)})',
            marker=dict(color='navy', size=current_params['trough_size'], 
                       symbol='square-open', line=dict(width=2))
        ))
    
    if len(deep_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[deep_troughs], y=deep_temp[deep_troughs],
            mode='markers', name=f'Deep Troughs ({len(deep_troughs)})',
            marker=dict(color='maroon', size=current_params['trough_size'], 
                       symbol='square-open', line=dict(width=2))
        ))
    
    # Add manual selections with different symbols (show them separately from detected peaks)
    manual_shallow_indices = [int(idx) for idx in manual_selections['shallow_peaks'] if idx < len(water_day)]
    manual_deep_indices = [int(idx) for idx in manual_selections['deep_peaks'] if idx < len(water_day)]
    
    print(f"Displaying {len(manual_shallow_indices)} manual shallow peaks: {manual_shallow_indices[:5]}...")
    print(f"Displaying {len(manual_deep_indices)} manual deep peaks: {manual_deep_indices[:5]}...")
    
    if len(manual_shallow_indices) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[manual_shallow_indices], 
            y=shallow_temp[manual_shallow_indices],
            mode='markers', 
            name=f'Manual Shallow ({len(manual_shallow_indices)})',
            marker=dict(color='cyan', size=14, symbol='star', line=dict(width=2, color='darkblue'))
        ))
    
    if len(manual_deep_indices) > 0:
        fig.add_trace(go.Scatter(
            x=water_day[manual_deep_indices], 
            y=deep_temp[manual_deep_indices],
            mode='markers', 
            name=f'Manual Deep ({len(manual_deep_indices)})',
            marker=dict(color='orange', size=14, symbol='star', line=dict(width=2, color='darkred'))
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
        
    # Update layout
    fig.update_layout(
        title='Peak Detection Results',
        xaxis_title='Water Day',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        clickmode='event+select',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add a subtitle showing the current mode
    mode_text = {
        'view': 'View Only Mode',
        'add_shallow': 'Click to Add Shallow Peaks (Blue)',
        'add_deep': 'Click to Add Deep Peaks (Red)',
        'remove': 'Click to Remove Peaks',
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
        
        # Add as small gray dots
        if len(all_shallow_max) > 0:
            fig.add_trace(go.Scatter(
                x=water_day[all_shallow_max], 
                y=shallow_temp[all_shallow_max],
                mode='markers', 
                name=f'All Shallow Maxima ({len(all_shallow_max)})',
                marker=dict(color='gray', size=10, symbol='circle'),
                opacity=0.7
            ))
        
        if len(all_deep_max) > 0:
            fig.add_trace(go.Scatter(
                x=water_day[all_deep_max], 
                y=deep_temp[all_deep_max],
                mode='markers', 
                name=f'All Deep Maxima ({len(all_deep_max)})',
                marker=dict(color='darkgray', size=10, symbol='circle'),
                opacity=0.7
            ))
    
    # Store peaks data
    peaks_data = {
        'shallow_peaks': [int(p) for p in shallow_peaks],  # Convert to regular Python int
        'deep_peaks': [int(p) for p in deep_peaks],
        'shallow_troughs': [int(t) for t in shallow_troughs],
        'deep_troughs': [int(t) for t in deep_troughs]
    }
    
    # Debug summary
    print(f"\n=== Peak Detection Summary ===")
    print(f"Method: {method}")
    print(f"Total shallow peaks: {len(peaks_data['shallow_peaks'])} (including {len(manual_selections['shallow_peaks'])} manual)")
    print(f"Total deep peaks: {len(peaks_data['deep_peaks'])} (including {len(manual_selections['deep_peaks'])} manual)")
    print(f"Excluded ranges: {len(manual_selections.get('excluded_ranges', []))}")
    print(f"==============================\n")
    
    # Ensure manual selections are also regular ints
    manual_selections_clean = {
        'shallow_peaks': [int(p) for p in manual_selections['shallow_peaks']],
        'deep_peaks': [int(p) for p in manual_selections['deep_peaks']],
        'excluded_ranges': manual_selections.get('excluded_ranges', [])
    }
    
    return json.dumps(peaks_data), fig, json.dumps(manual_selections_clean)

@app.callback(
    Output('realtime-interval', 'disabled'),
    Input('realtime-toggle', 'n_clicks'),
    State('realtime-interval', 'disabled')
)
def toggle_realtime(n_clicks, disabled):
    """Toggle real-time parameter updates."""
    if n_clicks > 0:
        return not disabled
    return disabled

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
            html.P(f"Excluded Time Ranges: {len(selections.get('excluded_ranges', []))}")
        ]
        
        # List excluded ranges
        if selections.get('excluded_ranges'):
            for i, (start, end) in enumerate(selections['excluded_ranges']):
                info_items.append(html.P(f"  Range {i+1}: Day {start:.1f} - {end:.1f}", 
                                       style={'marginLeft': '20px', 'fontSize': '0.9em'}))
        
        return html.Div(info_items)
    return "No manual selections"

@app.callback(
    Output('error-display', 'children'),
    Output('main-graph', 'figure', allow_duplicate=True),
    Input('check-errors-button', 'n_clicks'),
    [State('stored-data', 'children'),
     State('stored-peaks', 'children'),
     State('main-graph', 'figure'),
     State('sensor-spacing', 'value')],
    prevent_initial_call=True
)
def check_errors(n_clicks, stored_data, stored_peaks, current_fig, sensor_spacing):
    """
    Check for errors in detected peaks and display results.
    
    This callback runs all error checking functions and displays
    the results both as text and as visual markers on the plot.
    """
    if n_clicks == 0 or stored_data is None or stored_peaks is None:
        return '', current_fig
    
    # Load data
    df = pd.read_json(StringIO(stored_data), orient='split')
    peaks_data = json.loads(stored_peaks)
    
    water_day = df['WaterDay'].values
    shallow_temp = df['Shallow.Temp.Filt'].values
    deep_temp = df['Deep.Temp.Filt'].values
    
    shallow_peaks = np.array(peaks_data['shallow_peaks'])
    deep_peaks = np.array(peaks_data['deep_peaks'])
    shallow_troughs = np.array(peaks_data['shallow_troughs'])
    deep_troughs = np.array(peaks_data['deep_troughs'])
    
    # Run error checks
    errors = []
    
    # Check alternation
    if len(shallow_peaks) > 0 and len(deep_peaks) > 0:
        alternation_errors = check_alternation(
            water_day[shallow_peaks], water_day[deep_peaks])
        errors.extend(alternation_errors)
    
    # Check amplitude and phase
    if len(shallow_peaks) > 0 and len(deep_peaks) > 0:
        amp_phase_errors = check_amplitude_and_phase(
            shallow_peaks, deep_peaks, shallow_temp, deep_temp, water_day, water_day)
        errors.extend(amp_phase_errors)
    
    # Check peak-trough pairing
    if len(shallow_peaks) > 0 and len(shallow_troughs) > 0:
        pairing_errors_shallow = check_peak_trough_pairing(
            shallow_peaks, shallow_troughs, water_day)
        errors.extend(pairing_errors_shallow)
        
    if len(deep_peaks) > 0 and len(deep_troughs) > 0:
        pairing_errors_deep = check_peak_trough_pairing(
            deep_peaks, deep_troughs, water_day)
        errors.extend(pairing_errors_deep)
    
    # Create error display
    if len(errors) == 0:
        error_display = html.Div([
            html.H4("✓ No errors detected!", style={'color': 'green'})
        ])
    else:
        error_list = []
        for error in errors:
            error_list.append(html.Li(
                f"{error['type'].upper()}: {error['message']}"
            ))
        
        error_display = html.Div([
            html.H4(f"⚠ {len(errors)} errors detected:", style={'color': 'red'}),
            html.Ul(error_list)
        ])
    
    # Update figure with error markers
    fig = go.Figure(current_fig)
    
    # Add error markers
    if len(errors) > 0:
        error_times = [e['time'] for e in errors]
        error_types = [e['type'] for e in errors]
        
        # Find y-values for error markers (at the top of the plot)
        y_max = max(np.max(shallow_temp), np.max(deep_temp))
        
        fig.add_trace(go.Scatter(
            x=error_times,
            y=[y_max * 1.1] * len(error_times),
            mode='markers',
            name='Errors',
            marker=dict(color='red', size=15, symbol='x', line=dict(width=2)),
            text=[f"{t}: {errors[i]['message']}" for i, t in enumerate(error_types)],
            hoverinfo='text'
        ))
    
    return error_display, fig

@app.callback(
    Output('export-status', 'children'),
    Input('export-button', 'n_clicks'),
    [State('stored-data', 'children'),
     State('stored-peaks', 'children'),
     State('output-folder', 'value'),
     State('peak-filename', 'value'),
     State('formatted-filename', 'value'),
     State('sensor-spacing', 'value'),
     State('current-filename', 'children')],
    prevent_initial_call=True
)
def export_data(n_clicks, stored_data, stored_peaks, output_folder, 
               peak_filename, formatted_filename, sensor_spacing, original_filename):
    """
    Export peak data to CSV files.
    
    Creates two output files:
    1. Basic peak data with WaterDay, Temperature, and Depth
    2. Formatted data with amplitude ratios and phase shifts
    """
    if n_clicks == 0 or stored_data is None or stored_peaks is None:
        return ''
    
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
    
    # Calculate amplitude ratios and phase shifts for second output file
    shallow_peaks = np.array(peaks_data.get('shallow_peaks', []), dtype=int)
    deep_peaks = np.array(peaks_data.get('deep_peaks', []), dtype=int)
    shallow_troughs = np.array(peaks_data.get('shallow_troughs', []), dtype=int)
    deep_troughs = np.array(peaks_data.get('deep_troughs', []), dtype=int)
    
    ar_ps_data = []
    
    if len(shallow_peaks) > 0 and len(deep_peaks) > 0:
        for s_idx in shallow_peaks:
            if 0 <= s_idx < len(water_day):
                s_time = water_day[s_idx]
                s_peak_amp = shallow_temp[s_idx]
                
                # Find corresponding trough for this peak (if available)
                s_trough_amp = None
                if len(shallow_troughs) > 0:
                    # Find trough within ~0.5 days
                    trough_times = water_day[shallow_troughs]
                    time_diffs = np.abs(trough_times - s_time)
                    nearby_troughs = shallow_troughs[time_diffs <= 0.5]
                    if len(nearby_troughs) > 0:
                        # Use the closest trough
                        closest_trough = nearby_troughs[np.argmin(time_diffs[time_diffs <= 0.5])]
                        s_trough_amp = shallow_temp[closest_trough]
                
                # Calculate shallow amplitude (using trough if available)
                if s_trough_amp is not None:
                    s_amp = (s_peak_amp - s_trough_amp) / 2
                else:
                    s_amp = s_peak_amp  # Use peak value if no trough found
                
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
                        
                        # Find corresponding deep trough
                        d_trough_amp = None
                        if len(deep_troughs) > 0:
                            # Find trough within ~0.5 days
                            deep_trough_times = water_day[deep_troughs]
                            time_diffs = np.abs(deep_trough_times - d_time)
                            nearby_troughs = deep_troughs[time_diffs <= 0.5]
                            if len(nearby_troughs) > 0:
                                closest_trough = nearby_troughs[np.argmin(time_diffs[time_diffs <= 0.5])]
                                d_trough_amp = deep_temp[closest_trough]
                        
                        # Calculate deep amplitude (using trough if available)
                        if d_trough_amp is not None:
                            d_amp = (d_peak_amp - d_trough_amp) / 2
                        else:
                            d_amp = d_peak_amp  # Use peak value if no trough found
                        
                        # Calculate amplitude ratio and phase shift
                        ar = float(d_amp / s_amp) if s_amp != 0 else 0
                        ps = float(d_time - s_time)
                        
                        # Extract year from original filename if possible
                        year = datetime.now().year  # Default to current year
                        if original_filename:
                            # Try multiple patterns for year extraction
                            # Pattern 1: Look for 4-digit year (e.g., PR2024_...)
                            year_match_4digit = re.search(r'(19|20|21)\d{2}', original_filename)
                            if year_match_4digit:
                                year = int(year_match_4digit.group(0))
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
                                        year = current_century - 100 + two_digit_year
                                    else:
                                        year = current_century + two_digit_year
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
                                                year = current_century - 100 + two_digit_year
                                            else:
                                                year = current_century + two_digit_year
                                        else:
                                            year = int(wy_year)
                        
                        print(f"Extracted year: {year} from filename: {original_filename}")
                        
                        # Validate year is reasonable (1950-2150)
                        if year < 1950 or year > 2150:
                            print(f"Warning: Extracted year {year} seems unreasonable, using current year")
                            year = datetime.now().year
                        
                        ar_ps_data.append({
                            'Data_Year': year,
                            'Water_Day': float((s_time + d_time) / 2),
                            'Ad_As': ar,
                            'A_Uncertainty': 1e-5,
                            'Phase_Shift(days)': ps,
                            'f_Uncertainty': 0.001
                        })
    
    formatted_filepath = None
    if len(ar_ps_data) > 0:
        # Create header for formatted file
        header_lines = [
            "PHASE SHIFT AND AMPLITUDE RATIO DATA FILE: PEAKPICKER OUTPUT",
            "----------------------------------------------------------------",
            f"{sensor_spacing:.3f} is the relative distance (in m) between sensors.",
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
    
    return html.Div([
        html.P(f"✓ Exported {len(export_df)} peaks to {peak_filepath}"),
        html.P(f"✓ Exported {len(ar_ps_data)} amplitude ratio/phase shift pairs to {formatted_filepath}" 
               if formatted_filepath else "No valid peak pairs found for formatted output"),
        html.P(f"Files saved in: {output_path}")
    ], style={'color': 'green'})

@app.callback(
    [Output('target-period', 'value'),
     Output('search-tolerance', 'value'),
     Output('slope-threshold', 'value'),
     Output('min-distance', 'value'),
     Output('prominence-factor', 'value'),
     Output('peak-method-dropdown', 'value'),
     Output('sensor-spacing', 'value'),
     Output('output-folder', 'value'),
     Output('param-file-status', 'children')],
    Input('load-param-file-button', 'n_clicks'),
    prevent_initial_call=True
)
def load_parameters_from_file(n_clicks):
    """Load parameters from .par file and update UI."""
    if n_clicks > 0:
        params = load_param_file()
        return (params.get('target_period', 1.0),
               params.get('search_tolerance', 0.1),
               params.get('slope_threshold', 0.001),
               params.get('min_distance', 20),
               params.get('prominence_factor', 0.3),
               params.get('peak_method', 'scipy'),
               params.get('sensor_spacing', 0.18),
               params.get('output_folder', 'peak_analysis_output'),
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
     State('current-filename', 'children')],
    prevent_initial_call=True
)
def save_parameters_to_file(n_clicks, period, tolerance, slope, distance, prominence, 
                           method, spacing, output_folder, data_filename):
    """Save current parameters to .par file."""
    if n_clicks > 0:
        params = {
            'target_period': period,
            'search_tolerance': tolerance,
            'slope_threshold': slope,
            'min_distance': distance,
            'prominence_factor': prominence,
            'peak_method': method,
            'sensor_spacing': spacing,
            'output_folder': output_folder,
            'data_file': data_filename or ''
        }
        save_param_file(params)
        return html.Span("✓ Parameters saved to .par file", style={'color': 'green'})
    return ''

@app.callback(
    [Output('target-period', 'value', allow_duplicate=True),
     Output('search-tolerance', 'value', allow_duplicate=True),
     Output('slope-threshold', 'value', allow_duplicate=True),
     Output('min-distance', 'value', allow_duplicate=True),
     Output('prominence-factor', 'value', allow_duplicate=True),
     Output('peak-method-dropdown', 'value', allow_duplicate=True)],
    Input('load-params-button', 'n_clicks'),
    prevent_initial_call=True
)
def load_parameters(n_clicks):
    """Load parameters from JSON file (legacy compatibility)."""
    if n_clicks > 0:
        try:
            with open('peak_picker_params.json', 'r') as f:
                params = json.load(f)
            return (params.get('target_period', 1.0),
                   params.get('search_tolerance', 0.1),
                   params.get('slope_threshold', 0.001),
                   params.get('min_distance', 20),
                   params.get('prominence_factor', 0.3),
                   params.get('method', 'scipy'))
        except FileNotFoundError:
            # Return defaults if file doesn't exist
            return 1.0, 0.1, 0.001, 20, 0.3, 'scipy'
    return no_update

@app.callback(
    Output('export-status', 'children', allow_duplicate=True),
    Input('save-params-button', 'n_clicks'),
    [State('target-period', 'value'),
     State('search-tolerance', 'value'),
     State('slope-threshold', 'value'),
     State('min-distance', 'value'),
     State('prominence-factor', 'value'),
     State('peak-method-dropdown', 'value')],
    prevent_initial_call=True
)
def save_parameters(n_clicks, period, tolerance, slope, distance, prominence, method):
    """Save parameters to JSON file (legacy compatibility)."""
    if n_clicks > 0:
        params = {
            'target_period': period,
            'search_tolerance': tolerance,
            'slope_threshold': slope,
            'min_distance': distance,
            'prominence_factor': prominence,
            'method': method
        }
        # Save to JSON file for compatibility
        with open('peak_picker_params.json', 'w') as f:
            json.dump(params, f)
        return html.Div("✓ Parameters saved!", style={'color': 'green'})
    return ''

@app.callback(
    Output('manual-selections-store', 'children', allow_duplicate=True),
    Input('clear-manual-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_manual_selections(n_clicks):
    """Clear all manual peak selections and exclusions."""
    if n_clicks > 0:
        empty_selections = {'shallow_peaks': [], 'deep_peaks': [], 'excluded_ranges': []}
        print("Cleared all manual selections and exclusions")
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
        
        return peak_filename, formatted_filename
    
    return "peak_picks.csv", "peak_picks_formatted.csv"

# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == '__main__':
    print(f"\nThermal Probe Peak Picker")
    print(f"========================")
    print(f"Parameter file location: {PARAM_FILE}")
    print(f"Output will be saved to: {SCRIPT_DIR / initial_params['output_folder']}")
    
    # Create default .par file if it doesn't exist
    if not PARAM_FILE.exists():
        print(f"Creating default parameter file: {PARAM_FILE}")
        default_params = {
            'data_file': '',
            'sensor_spacing': 0.18,
            'target_period': 1.0,
            'search_tolerance': 0.1,
            'slope_threshold': 0.001,
            'min_distance': 20,
            'prominence_factor': 0.15,  # Lower for better sensitivity
            'output_folder': 'peak_analysis_output',
            'peak_method': 'combined',  # Most sensitive method
            'shallow_color': 'blue',
            'deep_color': 'red',
            'peak_size': 12,
            'trough_size': 10,
            'line_width': 2,
            'default_year': datetime.now().year  # Current year as default
        }
        save_param_file(default_params, PARAM_FILE)
        
        # Also create a sample .par file with comments
        sample_content = """# Thermal Probe Peak Picker Parameter File
# tts_pickpeak.par
#
# This file contains initial parameters for the peak picking analysis.
# Parameters can be modified within the tool and saved back to this file.
#
# Format: parameter_name = value
# Lines starting with # are comments

[PARAMETERS]
# Input data file (can be selected via GUI)
data_file = 

# Sensor spacing in meters (used in formatted output file)
sensor_spacing = 0.18

# Desired time period between peaks (days)
# Common values: 1.0 (daily), 0.5 (semi-daily), 2.0 (bi-daily)
target_period = 1.0

# Timing threshold for searching for next peak (days)
# This defines the window around the expected peak location
search_tolerance = 0.1

# Slope threshold for identifying peaks and troughs (°C/min)
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
# Options: scipy, wavelet, derivative, prominence, combined, bootstrap
# 'combined' is most sensitive and finds the most peaks
peak_method = combined

# Display colors and sizes
shallow_color = blue
deep_color = red
peak_size = 12
trough_size = 10
line_width = 2
"""
        
        # Write sample file with comments
        sample_file = SCRIPT_DIR / 'tts_pickpeak_sample.par'
        with open(sample_file, 'w') as f:
            f.write(sample_content)
        print(f"Created sample parameter file: {sample_file}")
    
    print(f"\nStarting web application at http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the application")
    
    app.run(debug=True)