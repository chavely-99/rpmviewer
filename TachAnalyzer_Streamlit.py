#!/usr/bin/env python3
"""
Pit Road Tach Analyzer - Streamlit App
======================================
Upload pit road video → Extract RPM → Add tach markers → Analyze zones

Requirements:
    pip install streamlit plotly pytesseract pillow numpy pandas
    apt install tesseract-ocr ffmpeg  (or brew install on Mac)
"""

import streamlit as st
import tempfile
import subprocess
import shutil
import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import pytesseract
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Configure Tesseract path for Windows
if sys.platform == 'win32':
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


# ============================================================================
# VIDEO PROCESSING FUNCTIONS
# ============================================================================

def detect_tach_roi(video_path):
    """Detect tach display region from first few frames for efficient cropping.
    Returns (x, y, w, h) bounding box or None if not detected."""
    tmpdir = tempfile.mkdtemp(prefix="tach_roi_")
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-frames:v", "5",
            str(Path(tmpdir) / "roi_%03d.png"),
            "-y", "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)
        frames = sorted(Path(tmpdir).glob("roi_*.png"))

        strategies = [
            lambda a: (a[:,:,0] < 80) & (a[:,:,2] > 60) & \
                      (a[:,:,2].astype(int) - a[:,:,0].astype(int) > 30),
            lambda a: (a[:,:,0] < 100) & (a[:,:,1] > 150) & (a[:,:,2] > 150),
            lambda a: (a[:,:,0] < 150) & (a[:,:,1] > 100) & (a[:,:,2] > 100) & \
                      ((a[:,:,1].astype(int) + a[:,:,2].astype(int) - a[:,:,0].astype(int)) > 200),
        ]

        for frame_path in frames:
            arr = np.array(Image.open(frame_path))
            for strategy in strategies:
                mask = strategy(arr)
                if mask.sum() < 10:
                    continue
                ys, xs = np.where(mask)
                # Generous padding around detected text region
                pad = 60
                x1 = max(0, int(xs.min()) - pad)
                y1 = max(0, int(ys.min()) - pad)
                x2 = min(arr.shape[1], int(xs.max()) + pad)
                y2 = min(arr.shape[0], int(ys.max()) + pad)
                w, h = x2 - x1, y2 - y1
                if w > 20 and h > 10:
                    return (x1, y1, w, h)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def extract_frames(video_path, fps=10, crop_roi=None):
    """Extract frames from video, optionally cropped to tach ROI."""
    tmpdir = tempfile.mkdtemp(prefix="tach_")
    vf_filters = []
    if crop_roi:
        x, y, w, h = crop_roi
        vf_filters.append(f"crop={w}:{h}:{x}:{y}")
    else:
        vf_filters.append("scale=-2:720")  # Fallback: scale to 720p
    vf_filters.append(f"fps={fps}")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", ",".join(vf_filters),
        str(Path(tmpdir) / "frame_%05d.png"),
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(Path(tmpdir).glob("frame_*.png"))
    return tmpdir, frames


def find_unique_frames(frames, threshold=0.5):
    """Find frames that visually changed from predecessor. Skip static duplicates.
    Returns list of (original_index, frame_path) for unique frames.
    Low threshold (0.5) ensures single-digit RPM changes aren't missed."""
    if not frames:
        return []
    unique = [(0, frames[0])]
    prev_arr = np.array(Image.open(frames[0]), dtype=np.float32)
    for i in range(1, len(frames)):
        curr_arr = np.array(Image.open(frames[i]), dtype=np.float32)
        diff = np.mean(np.abs(curr_arr - prev_arr))
        if diff > threshold:
            unique.append((i, frames[i]))
            prev_arr = curr_arr
    return unique


def ocr_frame(frame_path):
    """Extract RPM value from a single frame."""
    img = Image.open(frame_path)
    arr = np.array(img)

    # Try multiple detection strategies
    strategies = [
        # Darker cyan (R<80, B dominant) — most precise
        lambda a: (a[:,:,0] < 80) & (a[:,:,2] > 60) & \
                  (a[:,:,2].astype(int) - a[:,:,0].astype(int) > 30),
        # Bright saturated cyan (R low, G+B high)
        lambda a: (a[:,:,0] < 100) & (a[:,:,1] > 150) & (a[:,:,2] > 150),
        # Broader cyan catch-all
        lambda a: (a[:,:,0] < 150) & (a[:,:,1] > 100) & (a[:,:,2] > 100) & \
                  ((a[:,:,1].astype(int) + a[:,:,2].astype(int) - a[:,:,0].astype(int)) > 200),
    ]

    for strategy in strategies:
        mask = strategy(arr)
        if mask.sum() < 10:
            continue

        binary = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
        binary[mask] = 255

        # Crop to text region with small padding
        ys, xs = np.where(mask)
        pad = 3
        y1 = max(0, ys.min() - pad)
        y2 = min(arr.shape[0], ys.max() + pad + 1)
        x1 = max(0, xs.min() - pad)
        x2 = min(arr.shape[1], xs.max() + pad + 1)
        cropped = binary[y1:y2, x1:x2]

        crop_img = Image.fromarray(cropped)
        w, h = crop_img.size
        if w < 3 or h < 3:
            continue

        # Scale up for better OCR
        scale = max(4, min(8, 200 // max(w, 1)))
        crop_img = crop_img.resize((w * scale, h * scale), Image.NEAREST)

        text = pytesseract.image_to_string(
            crop_img,
            config='--psm 7 -c tessedit_char_whitelist=0123456789,'
        )
        text = text.strip().replace(',', '').replace('.', '').replace(' ', '')

        match = re.search(r'\d+', text)
        if match:
            val = int(match.group())
            if 10 <= val <= 9999:
                return val  # Early return on success - don't try other strategies

    return None


def trim_leading_flat(data, threshold=50, min_flat_samples=10):
    """Remove leading section where RPM is constant (paused playback)."""
    if len(data) < 20:
        return data

    first_val = data[0][1]
    flat_end = 0
    for i in range(1, len(data)):
        if abs(data[i][1] - first_val) > threshold:
            flat_end = i
            break

    if flat_end > min_flat_samples:
        trimmed = data[flat_end:]
        t0 = trimmed[0][0]
        return [[round(t - t0, 2), r] for t, r in trimmed]
    return data


def fix_truncated(data):
    """Fix 3-digit values that are truncated 4-digit readings."""
    for i in range(len(data)):
        t, r = data[i]
        if 100 <= r <= 600:
            neighbors = []
            for j in range(max(0, i-5), min(len(data), i+6)):
                if j != i and data[j][1] is not None and data[j][1] >= 1000:
                    neighbors.append(data[j][1])
            if neighbors and abs(r * 10 - np.mean(neighbors)) < 500:
                data[i][1] = r * 10
            else:
                data[i][1] = None  # can't fix
    return [[t, r] for t, r in data if r is not None]


def remove_outliers(data):
    """Remove points that jump wildly from both neighbors."""
    if len(data) < 3:
        return data
    cleaned = [list(d) for d in data]
    for i in range(1, len(cleaned) - 1):
        prev = cleaned[i-1][1]
        curr = cleaned[i][1]
        nxt = cleaned[i+1][1]
        if abs(curr - prev) > 800 and abs(curr - nxt) > 800:
            cleaned[i][1] = int((prev + nxt) / 2)
    return cleaned


def ocr_frame_with_time(args):
    """Wrapper for parallel processing - returns (time, rpm)."""
    frame_path, frame_idx, fps = args
    rpm = ocr_frame(frame_path)
    time_sec = round(frame_idx / fps, 2)
    return (time_sec, rpm)


def process_video(video_path, fps=10, progress_callback=None):
    """Full pipeline: video → cleaned RPM data.
    Uses auto-ROI detection, frame dedup, and parallel OCR for speed."""

    # Phase 1: Detect tach display region
    if progress_callback:
        progress_callback("Detecting tach display region...", 0.05)
    roi = detect_tach_roi(video_path)

    # Phase 2: Extract frames (cropped to ROI if detected)
    if progress_callback:
        roi_msg = f"ROI found ({roi[2]}x{roi[3]}px) — cropping" if roi else "Full frame mode"
        progress_callback(f"Extracting frames... {roi_msg}", 0.10)
    tmpdir, frames = extract_frames(video_path, fps, crop_roi=roi)
    total_frames = len(frames)

    # Phase 3: Skip duplicate frames (tach doesn't change every frame)
    if progress_callback:
        progress_callback(f"Deduplicating {total_frames} frames...", 0.15)
    unique = find_unique_frames(frames)
    skipped = total_frames - len(unique)

    if progress_callback:
        progress_callback(f"{len(unique)} unique frames (skipped {skipped} duplicates)", 0.20)

    # Phase 4: OCR only unique frames in parallel
    frame_args = [(path, idx, fps) for idx, path in unique]
    num_workers = 16

    raw_data = []
    valid = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunk_size = max(5, len(frame_args) // 20)
        for i in range(0, len(frame_args), chunk_size):
            chunk = frame_args[i:i+chunk_size]
            results = list(executor.map(ocr_frame_with_time, chunk))

            for time_sec, rpm in results:
                if rpm is not None:
                    raw_data.append([time_sec, rpm])
                    valid += 1

            if progress_callback:
                progress = 0.20 + ((i + len(chunk)) / len(frame_args)) * 0.65
                progress_callback(
                    f"OCR: {i+len(chunk)}/{len(unique)} unique frames ({valid} valid)",
                    progress
                )

    # Cleanup temp files
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Phase 5: Clean data
    if progress_callback:
        progress_callback("Cleaning data...", 0.90)

    raw_data.sort(key=lambda x: x[0])
    data = trim_leading_flat(raw_data)
    data = fix_truncated(data)
    data = remove_outliers(data)

    if progress_callback:
        progress_callback("Complete!", 1.0)

    return data


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Pit Road Tach Analyzer",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for lighter, readable theme
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stMetric { background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #d1d5db; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stMetric label { color: #6b7280 !important; font-size: 0.75rem !important; font-weight: 600 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1f2937 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
    h1 { font-family: 'Courier New', monospace; color: #1f2937; letter-spacing: 1px; font-weight: 800; }
    h2, h3 { color: #374151; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid #d1d5db;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rpm_data' not in st.session_state:
    st.session_state.rpm_data = None
if 'tach_lines' not in st.session_state:
    st.session_state.tach_lines = []

# Header
st.markdown("# 🏁 PIT ROAD TACH ANALYZER")
st.markdown("**Upload pit road video → Extract RPM → Add tach markers → Analyze zones**")
st.divider()

# Sidebar - File Upload & Processing
with st.sidebar:
    st.header("📹 Video Input")

    uploaded_file = st.file_uploader(
        "Upload pit road video",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Screen recording of digital RPM display"
    )

    fps = st.slider("Extraction FPS", 5, 30, 8, help="Higher = more accurate but slower")

    if uploaded_file:
        if st.button("🚀 Process Video", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(msg, val):
                    status_text.text(msg)
                    progress_bar.progress(val)

                rpm_data = process_video(tmp_path, fps=fps, progress_callback=update_progress)
                st.session_state.rpm_data = rpm_data
                # Keep existing tach_lines so they persist across video uploads

                st.success(f"✅ Extracted {len(rpm_data)} RPM samples!")
                st.rerun()

            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    st.divider()

    # Tach Line Controls (available even before video upload)
    st.header("📍 Tach Markers")
    st.caption("Add markers before or after uploading video")

    if st.button("➕ Add Tach Line"):
        colors = ['#ff3b3b','#ff6b2b','#ffaa00','#ffe03b','#4ade80','#00d4ff','#a78bfa','#f472b6']
        color = colors[len(st.session_state.tach_lines) % len(colors)]
        st.session_state.tach_lines.append({
            'label': f'Line {len(st.session_state.tach_lines) + 1}',
            'rpm': 3800,
            'color': color,
            'line_style': 'solid'  # solid, dash, dot, dashdot
        })
        st.rerun()

    # Edit existing lines
    to_remove = []
    for i, line in enumerate(st.session_state.tach_lines):
        # Ensure line_style exists for backwards compatibility
        if 'line_style' not in line:
            line['line_style'] = 'solid'

        with st.expander(f"📍 {line['label']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_label = st.text_input("Label", line['label'], key=f"label_{i}")
                line['label'] = new_label
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    to_remove.append(i)

            new_rpm = st.number_input("RPM", 0, 10000, line['rpm'], step=50, key=f"rpm_{i}")
            line['rpm'] = new_rpm

            col_color, col_style = st.columns(2)
            with col_color:
                new_color = st.color_picker("Color", line['color'], key=f"color_{i}")
                line['color'] = new_color

            with col_style:
                style_options = {
                    'Solid': 'solid',
                    'Dashed': 'dash',
                    'Dotted': 'dot',
                    'Dash-Dot': 'dashdot'
                }
                current_style_label = [k for k, v in style_options.items() if v == line['line_style']][0]
                new_style_label = st.selectbox(
                    "Line Style",
                    options=list(style_options.keys()),
                    index=list(style_options.keys()).index(current_style_label),
                    key=f"style_{i}"
                )
                line['line_style'] = style_options[new_style_label]

    # Remove marked lines
    for i in reversed(to_remove):
        st.session_state.tach_lines.pop(i)
        st.rerun()

# Main Content
if st.session_state.rpm_data is None:
    st.info("👈 Upload a pit road video to get started")
else:
    rpm_data = st.session_state.rpm_data
    df = pd.DataFrame(rpm_data, columns=['Time', 'RPM'])

    # Quick Stats Bar at Top
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Peak RPM", f"{df['RPM'].max():,}", delta=None)
    with col2:
        st.metric("Min RPM", f"{df['RPM'].min():,}")
    with col3:
        st.metric("Avg RPM", f"{int(df['RPM'].mean()):,}")
    with col4:
        st.metric("Duration", f"{df['Time'].max():.1f}s")
    with col5:
        st.metric("Samples", f"{len(df):,}")

    st.divider()

    # Use tabs for organized content
    if st.session_state.tach_lines:
        tab1, tab2, tab3 = st.tabs(["📈 RPM Trace", "⏱️ Zone Analysis", "📥 Export Data"])
    else:
        tab1, tab3 = st.tabs(["📈 RPM Trace", "📥 Export Data"])
        tab2 = None

    # TAB 1: RPM Trace
    with tab1:
        fig = go.Figure()

        # Main RPM trace
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df['RPM'],
            mode='lines',
            name='RPM',
            line=dict(color='#3b82f6', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='<b>%{x:.1f}s</b><br>%{y:,} RPM<extra></extra>'
        ))

        # Add tach lines
        for line in st.session_state.tach_lines:
            # Ensure line_style exists for backwards compatibility
            line_style = line.get('line_style', 'solid')
            fig.add_trace(go.Scatter(
                x=[df['Time'].min(), df['Time'].max()],
                y=[line['rpm'], line['rpm']],
                mode='lines',
                name=line['label'],
                line=dict(color=line['color'], width=3, dash=line_style),
                hovertemplate=f"<b>{line['label']}</b><br>{line['rpm']:,} RPM<extra></extra>"
            ))

        # Calculate auto-zoom range (filter outliers using percentiles)
        rpm_p1 = df['RPM'].quantile(0.01)   # 1st percentile
        rpm_p99 = df['RPM'].quantile(0.99)  # 99th percentile
        y_min = max(0, rpm_p1 - 1000)       # Add 1000 RPM padding below
        y_max = rpm_p99 + 1000              # Add 1000 RPM padding above

        # Layout
        fig.update_layout(
            template='plotly_white',
            height=600,
            xaxis_title="Time (seconds)",
            yaxis_title="RPM",
            yaxis=dict(range=[y_min, y_max]),  # Auto-zoom with padding
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Courier New', color='#1f2937', size=12),
            margin=dict(l=60, r=40, t=60, b=60)
        )

        st.plotly_chart(fig, use_container_width=True, key='main_trace')

        if not st.session_state.tach_lines:
            st.info("👈 Add tach markers in the sidebar to analyze zones")

    # TAB 2: Zone Analysis (only if tach lines exist)
    if tab2 is not None:
        with tab2:
            sorted_lines = sorted(st.session_state.tach_lines, key=lambda x: x['rpm'])
            dt = df['Time'].iloc[1] - df['Time'].iloc[0] if len(df) > 1 else 0.1

            zones = []

            # Below first line
            below_count = len(df[df['RPM'] < sorted_lines[0]['rpm']])
            zones.append({
                'Zone': f"Below {sorted_lines[0]['label']}",
                'RPM Range': f"< {sorted_lines[0]['rpm']:,}",
                'Samples': below_count,
                'Time (s)': below_count * dt,
                'Percentage': (below_count / len(df)) * 100,
                'Color': '#4a5270'
            })

            # Between lines
            for i in range(len(sorted_lines) - 1):
                lower = sorted_lines[i]
                upper = sorted_lines[i + 1]
                between_count = len(df[(df['RPM'] >= lower['rpm']) & (df['RPM'] < upper['rpm'])])
                zones.append({
                    'Zone': f"{lower['label']} → {upper['label']}",
                    'RPM Range': f"{lower['rpm']:,} - {upper['rpm']:,}",
                    'Samples': between_count,
                    'Time (s)': between_count * dt,
                    'Percentage': (between_count / len(df)) * 100,
                    'Color': lower['color']
                })

            # Above last line
            above_count = len(df[df['RPM'] >= sorted_lines[-1]['rpm']])
            zones.append({
                'Zone': f"Above {sorted_lines[-1]['label']}",
                'RPM Range': f"> {sorted_lines[-1]['rpm']:,}",
                'Samples': above_count,
                'Time (s)': above_count * dt,
                'Percentage': (above_count / len(df)) * 100,
                'Color': sorted_lines[-1]['color']
            })

            zone_df = pd.DataFrame(zones)

            # Key Insights Summary
            st.markdown("### 🎯 Key Insights")
            col1, col2, col3 = st.columns(3)

            max_zone = zone_df.loc[zone_df['Percentage'].idxmax()]
            with col1:
                st.metric(
                    "Most Time Spent",
                    max_zone['Zone'],
                    f"{max_zone['Percentage']:.1f}%"
                )

            with col2:
                total_above_first = zone_df[zone_df['Zone'] != zone_df.iloc[0]['Zone']]['Time (s)'].sum()
                st.metric(
                    f"Time Above {sorted_lines[0]['label']}",
                    f"{total_above_first:.1f}s",
                    f"{(total_above_first/df['Time'].max()*100):.1f}%"
                )

            with col3:
                if len(sorted_lines) > 1:
                    red_zone = zone_df.iloc[-1]
                    st.metric(
                        f"Time in {red_zone['Zone']}",
                        f"{red_zone['Time (s)']:.1f}s",
                        f"{red_zone['Percentage']:.1f}%"
                    )

            st.divider()

            # Visual breakdown
            st.markdown("### 📊 Zone Breakdown")

            # Horizontal bar chart
            fig_zones = go.Figure()

            for _, zone in zone_df.iterrows():
                fig_zones.add_trace(go.Bar(
                    y=[zone['Zone']],
                    x=[zone['Time (s)']],
                    name=zone['Zone'],
                    orientation='h',
                    marker=dict(color=zone['Color']),
                    text=f"{zone['Time (s)']:.1f}s ({zone['Percentage']:.1f}%)",
                    textposition='inside',
                    textfont=dict(size=14, family='Courier New', color='white'),
                    hovertemplate=f"<b>{zone['Zone']}</b><br>" +
                                 f"Time: {zone['Time (s)']:.1f}s<br>" +
                                 f"Percentage: {zone['Percentage']:.1f}%<br>" +
                                 f"Samples: {zone['Samples']}<extra></extra>"
                ))

            fig_zones.update_layout(
                template='plotly_white',
                height=max(300, len(zones) * 80),
                xaxis_title="Time (seconds)",
                yaxis_title="",
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Courier New', color='#1f2937', size=12),
                margin=dict(l=200, r=40, t=20, b=60)
            )

            st.plotly_chart(fig_zones, use_container_width=True, key='zone_chart')

            # Data table
            st.markdown("### 📋 Detailed Breakdown")
            st.dataframe(
                zone_df[['Zone', 'RPM Range', 'Time (s)', 'Percentage']].style.format({
                    'Time (s)': '{:.1f}',
                    'Percentage': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

    # TAB 3: Export
    with tab3:
        st.markdown("### 📥 Export Options")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Raw RPM Trace")
            st.markdown(f"**{len(df):,}** samples • **{df['Time'].max():.1f}s** duration")
            st.download_button(
                "⬇️ Download RPM Trace CSV",
                df.to_csv(index=False),
                "rpm_trace.csv",
                "text/csv",
                key='download-raw',
                use_container_width=True
            )
            with st.expander("Preview Data"):
                st.dataframe(df.head(100), use_container_width=True)

        with col2:
            if st.session_state.tach_lines:
                st.markdown("#### Zone Analysis")
                sorted_lines = sorted(st.session_state.tach_lines, key=lambda x: x['rpm'])
                dt = df['Time'].iloc[1] - df['Time'].iloc[0] if len(df) > 1 else 0.1

                # Recreate zone_df for export
                zones = []
                below_count = len(df[df['RPM'] < sorted_lines[0]['rpm']])
                zones.append({
                    'Zone': f"Below {sorted_lines[0]['label']}",
                    'RPM Range': f"< {sorted_lines[0]['rpm']:,}",
                    'Time (s)': below_count * dt,
                    'Percentage': (below_count / len(df)) * 100,
                })
                for i in range(len(sorted_lines) - 1):
                    lower = sorted_lines[i]
                    upper = sorted_lines[i + 1]
                    between_count = len(df[(df['RPM'] >= lower['rpm']) & (df['RPM'] < upper['rpm'])])
                    zones.append({
                        'Zone': f"{lower['label']} → {upper['label']}",
                        'RPM Range': f"{lower['rpm']:,} - {upper['rpm']:,}",
                        'Time (s)': between_count * dt,
                        'Percentage': (between_count / len(df)) * 100,
                    })
                above_count = len(df[df['RPM'] >= sorted_lines[-1]['rpm']])
                zones.append({
                    'Zone': f"Above {sorted_lines[-1]['label']}",
                    'RPM Range': f"> {sorted_lines[-1]['rpm']:,}",
                    'Time (s)': above_count * dt,
                    'Percentage': (above_count / len(df)) * 100,
                })
                zone_df = pd.DataFrame(zones)

                st.markdown(f"**{len(zones)}** zones defined")
                st.download_button(
                    "⬇️ Download Zone Analysis CSV",
                    zone_df.to_csv(index=False),
                    "tach_zone_analysis.csv",
                    "text/csv",
                    key='download-csv',
                    use_container_width=True
                )
                with st.expander("Preview Data"):
                    st.dataframe(zone_df, use_container_width=True)
            else:
                st.info("Add tach markers to export zone analysis")
