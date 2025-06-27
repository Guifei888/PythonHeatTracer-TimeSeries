# TTS_MERPAR - Temperature Time Series Merger and Processor

A Python application for processing, merging, and analyzing temperature data from shallow and deep water temperature loggers with interactive data selection and export capabilities.

## Overview

TTS_MERPAR processes temperature time series data by:
- Reading CSV files from temperature loggers
- Converting data to water year format (October 1 - September 30)
- Merging shallow and deep temperature datasets
- Providing interactive visualization for data interval selection
- Exporting selected intervals in multiple formats (CSV, WYO)

## Requirements

### Python Version
- Python 3.7 or higher

### Required Python Packages
```bash
pip install pandas numpy matplotlib
```

Or install all at once:
```bash
pip install pandas numpy matplotlib
```

### System Requirements
- Operating System: Windows, macOS, or Linux
- Memory: At least 4GB RAM recommended for large datasets
- Display: GUI support required for interactive plotting

## Installation

1. **Download the script**
   ```bash
   # Download tts_merpar_alpha.py to your working directory
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib
   ```

3. **Verify installation**
   ```bash
   python -c "import pandas, numpy, matplotlib; print('All dependencies installed successfully')"
   ```

## File Structure

Your working directory should contain:
```
your_project_folder/
├── tts_merpar_alpha.py          # Main script
├── tts_merpar.par               # Parameter file (required)
├── your_shallow_data.csv        # Shallow temperature data
├── your_deep_data.csv           # Deep temperature data
└── processed/                   # Output folder (created automatically)
```

## Configuration

### Parameter File (tts_merpar.par)

Create a parameter file named `tts_merpar.par` in the same directory as the script:

```ini
# TTS_MERPAR Parameter File
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
```

### Parameter Descriptions

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `filename_shallow` | Yes | Filename of shallow temperature data CSV | `temp_shallow.csv` |
| `filename_deep` | Yes | Filename of deep temperature data CSV | `temp_deep.csv` |
| `water_year` | Yes | Water year for analysis (Oct 1 to Sep 30) | `2024` |
| `data_interval_min` | Yes | Expected data logging interval in minutes | `15.0` |
| `convert_f_to_c` | No | Convert Fahrenheit to Celsius (1=yes, 0=no) | `0` |
| `gap_threshold_factor` | No | Multiplier for gap detection threshold | `1.5` |
| `output_folder` | No | Output directory name | `processed` |
| `plot_relative` | No | Use relative water day plotting (1=yes, 0=no) | `0` |

## Input Data Format

### CSV File Requirements

Your temperature data CSV files must contain at least 3 columns:
1. **Column 1**: Record number or ID (ignored)
2. **Column 2**: Date/time string
3. **Column 3**: Temperature value

### Supported Date/Time Formats

The script automatically detects these datetime formats:
- `MM/DD/YY HH:MM:SS AM/PM` (e.g., `12/25/23 2:30:45 PM`)
- `MM/DD/YY HH:MM:SS` (e.g., `12/25/23 14:30:45`)
- `MM/DD/YYYY HH:MM:SS AM/PM` (e.g., `12/25/2023 2:30:45 PM`)
- `MM/DD/YYYY HH:MM:SS` (e.g., `12/25/2023 14:30:45`)
- `YYYY-MM-DD HH:MM:SS` (e.g., `2023-12-25 14:30:45`)
- `MM/DD/YY HH:MM` (e.g., `12/25/23 14:30`)
- `MM/DD/YYYY HH:MM` (e.g., `12/25/2023 14:30`)

### Example CSV Format
```csv
ID,DateTime,Temperature
1,10/01/23 12:00:00 AM,15.25
2,10/01/23 12:15:00 AM,15.30
3,10/01/23 12:30:00 AM,15.28
...
```

## Usage

### Basic Usage

1. **Prepare your files**
   - Place CSV data files in the working directory
   - Create `tts_merpar.par` parameter file
   - Ensure the script `tts_merpar_alpha.py` is in the same directory

2. **Run the script**
   ```bash
   python tts_merpar_alpha.py
   ```

3. **Interactive selection**
   - The script will display an interactive plot
   - Click and drag to select data intervals
   - Use the control buttons to save intervals
   - Click "Done" when finished

### Step-by-Step Workflow

1. **Script starts and loads parameters**
   ```
   === TTS_MERPAR: Temperature Time Series Merger and Processor ===
   Loading parameters from: /path/to/tts_merpar.par
   ```

2. **Data loading and validation**
   ```
   === Reading Data Files ===
   Reading Shallow data from: shallow_temp_data.csv
   Reading Deep data from: deep_temp_data.csv
   ```

3. **Water year processing**
   ```
   === Processing Water Years ===
   === Merging Data ===
   Merged dataset: 35040 records
   ```

4. **Interactive plot appears**
   - Temperature data plotted vs. water day
   - Water year boundaries marked
   - Control panel at bottom

5. **Data selection**
   - Click and drag on plot to select intervals
   - Adjust values in text boxes if needed
   - Click "Save" to export interval
   - Repeat for multiple intervals

6. **Completion**
   - Click "Done" to create full-record files
   - All files saved to output directory

## Interactive Controls

### Plot Interaction
- **Click and drag**: Select data interval on the plot
- **Zoom**: Use matplotlib toolbar for zooming/panning
- **Legend**: Toggle data series visibility

### Control Panel
- **Start WD**: Start water day for selection
- **End WD**: End water day for selection  
- **Filename**: Base filename for output files
- **Save**: Export current selection
- **Clear**: Remove all selection rectangles
- **Done**: Finish processing and create full files

## Output Files

The script creates several output files in the specified output directory:

### For Each Selected Interval
- `{filename}_int01.csv` - Composite CSV with both temperatures
- `{filename}-S_int01.wyo` - Shallow temperature in WYO format
- `{filename}-D_int01.wyo` - Deep temperature in WYO format

### Full Dataset Files
- `{filename}_full.csv` - Complete merged dataset
- `{filename}-S_full.wyo` - Complete shallow temperature data
- `{filename}-D_full.wyo` - Complete deep temperature data

### Log File
- `{filename}_{timestamp}.log` - Processing log with all console output

### File Format Details

**CSV Format**:
```csv
WaterDay,Shallow.Temp,Deep.Temp
0.000000,15.25000,14.82000
0.010417,15.30000,14.85000
```

**WYO Format**:
```
# Year	WaterDay	Temperature	DepthID
2023	0.00000	15.25000	1
2023	0.01042	15.30000	1
```

## Water Year Concepts

### Water Year Definition
- **Water Year 2024**: October 1, 2023 to September 30, 2024
- **Water Day 0**: October 1 (start of water year)
- **Water Day 365/366**: September 30 (end of water year)

### Leap Year Handling
- Water years containing February 29th have 366 days
- Water Year 2024 is a leap year (366 days)
- Water Year 2025 is not a leap year (365 days)

## Troubleshooting

### Common Issues

**1. "Parameter file not found"**
```
Error: tts_merpar.par not found.
```
- **Solution**: Create `tts_merpar.par` in the same directory as the script

**2. "CSV file not found"**
```
Error: Shallow file 'data.csv' not found
```
- **Solution**: Check filename spelling in parameter file and ensure CSV files exist

**3. "No valid data found"**
```
Error: no valid data found in filename.csv
```
- **Solution**: Check CSV format, ensure date/time strings are in supported format

**4. "Could not parse date"**
```
Warning: Could not parse date '25/12/23 14:30:00' on line 5
```
- **Solution**: Check date format, ensure MM/DD/YY format (not DD/MM/YY)

**5. Import errors**
```
ModuleNotFoundError: No module named 'pandas'
```
- **Solution**: Install required packages: `pip install pandas numpy matplotlib`

**6. Display issues on headless systems**
```
TclTK Error: no display
```
- **Solution**: Use a system with GUI support or set up X11 forwarding for SSH

### Data Quality Issues

**Gap Detection**:
- The script automatically detects gaps in data
- Gaps larger than `data_interval_min * gap_threshold_factor` are reported
- Review gap reports to ensure data quality

**Water Year Validation**:
- Data outside the specified water year is flagged
- Pre-season data (negative water days) indicates data from previous water year
- Post-season data (water day > 365/366) indicates data from next water year

### Performance Tips

**Large Datasets**:
- For datasets > 100,000 records, processing may take several minutes
- Increase system memory if encountering memory errors
- Consider splitting very large datasets into smaller chunks

**Interactive Plotting**:
- Use plot zoom/pan tools for detailed inspection
- Clear selections periodically to improve plot performance
- Save intervals incrementally rather than making many selections at once

## Advanced Usage

### Custom Filename Patterns
The script creates intelligent base names from input filenames:
- `temp-S.csv` + `temp-D.csv` → `temp-SD`
- `site1_shallow.csv` + `site1_deep.csv` → `site1_SD`
- `shallow.csv` + `deep.csv` → `shallow_deep`

### Multiple Water Years
To process multiple water years:
1. Create separate parameter files for each water year
2. Run the script multiple times with different parameter files
3. Organize output folders by water year

### Batch Processing
For automated processing without interactive selection:
```python
# Modify the script to skip interactive plotting
# and export full datasets directly
```

## Support and Development

### Script Information
- **Author**: Timothy Wu
- **Created**: June 26, 2025
- **Version**: Alpha
- **Language**: Python 3.7+

### Reporting Issues
When reporting issues, please include:
1. Complete error message
2. Sample of your CSV data format
3. Contents of your parameter file
4. Python version and operating system

### Contributing
This is research software. Contact the author for feature requests or modifications.

## License

This software is provided as-is for research and educational purposes. Modify and distribute as needed for your research applications.