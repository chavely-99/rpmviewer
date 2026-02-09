# 🏁 Pit Road Tach Analyzer

Upload pit road video recordings → Extract RPM data via OCR → Add custom tach markers → Analyze time spent in each zone.

## Features

- **Video OCR Processing**: Extracts RPM readings from screen recordings of digital tachometers
- **Parallel Processing**: 12-thread processing for fast video analysis
- **Custom Tach Markers**: Add unlimited tach lines with custom labels, colors, and line styles
- **Zone Analysis**: Detailed breakdown of time spent in each RPM zone
- **Interactive Charts**: Hover, zoom, and explore RPM traces with Plotly
- **CSV Export**: Download both raw RPM data and zone analysis

## Use Cases

- Analyze pit road RPM compliance
- Compare driver pit road performance
- Review tach light adherence (1 RED, 2 RED, etc.)
- Generate reports for crew meetings

## Quick Start

1. Upload your pit road video (MP4, MOV, AVI, MKV)
2. Click "Process Video" to extract RPM data
3. Add tach markers in the sidebar
4. View zone analysis and export results

## Requirements

- Python 3.10+
- Tesseract OCR (for text recognition)
- FFmpeg (for video processing)

### Local Installation

```bash
# Install Python packages
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt install tesseract-ocr ffmpeg

# Install system dependencies (macOS)
brew install tesseract ffmpeg

# Install system dependencies (Windows)
# Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# FFmpeg from: https://ffmpeg.org/download.html
```

## Running Locally

```bash
streamlit run TachAnalyzer_Streamlit.py
```

## Deployment

This app is configured for deployment on [Streamlit Cloud](https://streamlit.io/cloud):

- `TachAnalyzer_Streamlit.py` - Main application
- `requirements.txt` - Python dependencies
- `packages.txt` - System dependencies (Tesseract, FFmpeg)

## How It Works

1. **Frame Extraction**: Extracts frames from video at 10 FPS using FFmpeg
2. **Color Detection**: Identifies cyan/blue text pixels (common for digital tach displays)
3. **OCR**: Tesseract reads RPM values from detected text regions
4. **Data Cleaning**: Removes outliers, fixes truncated readings, trims paused sections
5. **Visualization**: Plots RPM trace with custom tach markers
6. **Analysis**: Calculates time spent in each zone between markers

## Technical Details

- **Multi-strategy OCR**: Three different color detection strategies for robustness
- **Parallel Processing**: ThreadPoolExecutor with 12 workers
- **Data Validation**: Range checking (10-9999 RPM), outlier removal, truncation fixes
- **Light Theme**: High-contrast design for readability

## License

Internal tool for Trackhouse Racing analysis.

---

🏁 Built for weekly pit road analysis
