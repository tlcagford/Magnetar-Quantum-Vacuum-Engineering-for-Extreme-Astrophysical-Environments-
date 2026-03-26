"""
Stellaris QED Explorer v6.1 – High Contrast Theme
Dark background | Light text | Clear readability
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, hbar, e, m_e, alpha
from scipy.ndimage import gaussian_filter
import warnings
import time
import json
import pandas as pd
from PIL import Image

# Optional imports
try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

try:
    import pydicom
    HAS_DICOM = True
except ImportError:
    HAS_DICOM = False

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Stellaris QED Explorer v6.1",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# HIGH CONTRAST DARK THEME - Fully readable
st.markdown("""
<style>
    /* Main background - dark */
    [data-testid="stAppViewContainer"] { 
        background: #0a0a1a;
    }
    /* Sidebar - dark blue */
    [data-testid="stSidebar"] { 
        background: #0f0f1f;
        border-right: 2px solid #00aaff;
    }
    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label {
        color: #ffffff !important;
    }
    /* Main title */
    .stTitle, h1, h2, h3 {
        color: #00aaff !important;
        font-weight: bold;
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #00aaff !important;
        font-size: 2rem;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #cccccc !important;
    }
    [data-testid="stMetricDelta"] {
        color: #88ff88 !important;
    }
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1a1a2a;
        border: 2px dashed #00aaff;
        border-radius: 10px;
        padding: 20px;
    }
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }
    /* Info, warning, success boxes */
    .stInfo {
        background-color: #1a2a3a;
        border-left: 4px solid #00aaff;
        color: #ffffff;
    }
    .stWarning {
        background-color: #3a2a1a;
        border-left: 4px solid #ffaa00;
        color: #ffffff;
    }
    .stSuccess {
        background-color: #1a3a2a;
        border-left: 4px solid #00ffaa;
        color: #ffffff;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a2a;
        color: #ffffff !important;
    }
    .streamlit-expanderContent {
        background-color: #0f0f1f;
        color: #ffffff;
    }
    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] button {
        color: #cccccc;
        background-color: #1a1a2a;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #00aaff;
        border-bottom: 2px solid #00aaff;
    }
    /* Code blocks */
    code {
        color: #88ff88;
        background-color: #2a2a3a;
    }
    /* Captions */
    .stCaption {
        color: #aaaaaa !important;
    }
    /* Slider labels */
    .stSlider label {
        color: #ffffff !important;
    }
    /* Number input */
    .stNumberInput label {
        color: #ffffff !important;
    }
    /* Select box */
    .stSelectbox label {
        color: #ffffff !important;
    }
    /* Checkbox */
    .stCheckbox label {
        color: #ffffff !important;
    }
    /* Markdown text */
    .stMarkdown p, .stMarkdown li {
        color: #dddddd !important;
    }
    /* Latex equations */
    .stMarkdown .katex {
        color: #88ff88 !important;
    }
    .stMarkdown .katex-html {
        color: #88ff88 !important;
    }
    /* Download button */
    .stDownloadButton button {
        background-color: #00aaff;
        color: #ffffff !important;
        border-radius: 8px;
        font-weight: bold;
    }
    .stDownloadButton button:hover {
        background-color: #0088cc;
    }
    /* Primary button */
    .stButton button {
        background-color: #00aaff;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)  # 4.4e13 G
alpha_fine = 1/137.036


# ── FILE LOADING FUNCTIONS ─────────────────────────────────────────────

def load_file(uploaded_file):
    """Universal file loader"""
    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()
    data_bytes = uploaded_file.read()
    
    # FITS files
    if ext in ['fits', 'fit', 'fts']:
        if HAS_ASTROPY:
            with fits.open(io.BytesIO(data_bytes)) as hdul:
                data = hdul[0].data.astype(np.float32)
                if len(data.shape) > 2:
                    data = data[0] if data.shape[0] < data.shape[1] else data[:, :, 0]
                return {'data': data, 'type': 'fits'}
        else:
            return None
    
    # DICOM files
    elif ext in ['dcm', 'dicom']:
        if HAS_DICOM:
            ds = pydicom.dcmread(io.BytesIO(data_bytes))
            data = ds.pixel_array.astype(np.float32)
            return {'data': data, 'type': 'dicom'}
        else:
            return None
    
    # CSV files
    elif ext == 'csv':
        df = pd.read_csv(io.BytesIO(data_bytes))
        data = df.values.astype(np.float32)
        return {'data': data, 'type': 'csv', 'columns': df.columns.tolist()}
    
    # Image files
    elif ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']:
        img = Image.open(io.BytesIO(data_bytes))
        if img.mode != 'L':
            img = img.convert('L')
        data = np.array(img, dtype=np.float32)
        return {'data': data, 'type': 'image'}
    
    # NumPy files
    elif ext == 'npy':
        data = np.load(io.BytesIO(data_bytes))
        return {'data': data, 'type': 'numpy'}
    
    else:
        return None


def normalize_data(data):
    """Normalize to [0,1]"""
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    if data.max() > data.min():
        return (data - data.min()) / (data.max() - data.min())
    return data


def apply_pdp_to_data(data, omega, fringe, brightness=1.2):
    """Apply PDP physics"""
    h, w = data.shape
    result = data.copy()
    
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    soliton = np.exp(-r**2 / (2 * (0.2 * (50/fringe))**2))
    wave = np.sin(2 * np.pi * fringe * r / 50) * soliton
    
    mixing = omega * 0.5
    result = data * (1 - mixing * 0.5)
    result = result + wave * mixing * 0.5
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    return result, wave, soliton


# ── PLOTTING FUNCTIONS ─────────────────────────────────────────────

def plot_data_with_overlay(data, title, cmap='inferno'):
    """Plot data with dark background"""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color='#00aaff', fontsize=12)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_color('white')
    fig.tight_layout()
    return fig


def plot_magnetar_field(B_surface, R_ns, inclination):
    """Magnetar field visualization"""
    resolution = 40
    r = np.linspace(R_ns, 5*R_ns, resolution)
    theta = np.linspace(0, np.pi, resolution)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    theta_rad = np.radians(inclination)
    B0 = B_surface * (R_ns / (R + 1e-9))**3
    B_r = 2 * B0 * np.cos(Theta + theta_rad)
    B_theta = B0 * np.sin(Theta + theta_rad)
    
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)
    U = B_r * np.sin(Theta) + B_theta * np.cos(Theta)
    V = B_r * np.cos(Theta) - B_theta * np.sin(Theta)
    
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-9)
    V_norm = V / (magnitude + 1e-9)
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    skip = 2
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              U_norm[::skip, ::skip], V_norm[::skip, ::skip],
              alpha=0.7, color='#00aaff', width=0.008)
    
    ax.add_patch(Circle((0, 0), R_ns, color='#ff4444', alpha=0.9))
    ax.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.set_xlim(-5*R_ns, 5*R_ns)
    ax.set_ylim(-5*R_ns, 5*R_ns)
    ax.set_xlabel('x (km)', color='white')
    ax.set_ylabel('y (km)', color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'Magnetar Field | B = {B_surface:.1e} G', color='#00aaff')
    
    return fig


def plot_dark_photon_conversion(B, epsilon, m_dark):
    """Dark photon conversion plot"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    L = np.logspace(-2, 2, 500)
    if m_dark <= 0:
        P = (epsilon * B / 1e15)**2 * np.ones_like(L)
    else:
        hbar_ev_s = 6.582e-16
        c_km_s = 3e5
        conversion_length = 4 * 1e18 * hbar_ev_s * c_km_s / (m_dark**2)
        P = (epsilon * B / 1e15)**2 * np.sin(np.pi * L / conversion_length)**2
    P = np.clip(P, 0, 1)
    
    ax.semilogx(L, P, '#00aaff', linewidth=2)
    ax.set_xlabel('Propagation Length (km)', color='white')
    ax.set_ylabel('Conversion Probability', color='white')
    ax.set_title(f'γ ↔ A\' Conversion | ε = {epsilon:.1e}', color='#00aaff')
    ax.grid(True, alpha=0.3, color='#888888')
    ax.set_ylim(0, 1)
    ax.tick_params(colors='white')
    
    return fig


def plot_kerr_geodesic(a_spin):
    """Kerr geodesic plot"""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    r_horizon = 1 + np.sqrt(1 - a_spin**2)
    circle = Circle((0, 0), r_horizon, color='#555555', alpha=0.6)
    ax.add_patch(circle)
    ax.text(0, 0, 'BH', color='white', ha='center', va='center', fontsize=10)
    
    if a_spin <= 0.999:
        r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r_photon * np.cos(theta), r_photon * np.sin(theta), 
                '#ff8888', linewidth=2, linestyle='--', alpha=0.7, label='Photon Sphere')
    
    for impact in [6, 8, 10]:
        t = np.linspace(0, 50, 300)
        r = 12 * np.exp(-t/35) + r_horizon + 0.5
        phi = (impact/10) * np.sin(t/25)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        ax.plot(x, y, '#88ff88', linewidth=1.2, alpha=0.6)
    
    ax.set_aspect('equal')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel('x/M', color='white')
    ax.set_ylabel('y/M', color='white')
    ax.set_title(f'Kerr Spacetime | a/M = {a_spin:.3f}', color='#00aaff')
    ax.legend()
    ax.grid(True, alpha=0.3, color='#888888')
    ax.tick_params(colors='white')
    
    return fig


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Stellaris QED")
    st.markdown("*Quantum Vacuum Engineering*")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 **Drop files here**",
        type=['fits', 'fit', 'fts', 'csv', 'dcm', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'npy'],
        help="FITS | CSV | DICOM | Images | NumPy"
    )
    
    st.markdown("---")
    st.markdown("### ⚛️ Physics Parameters")
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, 0.65, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, 55, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    
    st.markdown("---")
    st.markdown("### 🌌 Magnetar Parameters")
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("Kinetic Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin a/M", 0.0, 0.998, 0.9)
    
    st.markdown("---")
    st.latex(r"P_{\gamma\to A'} = \left(\frac{\varepsilon B}{m_{A'}}\right)^2\sin^2\left(\frac{m_{A'}^2 L}{4\omega}\right)")
    
    st.caption("Tony Ford Model | v6.1 - High Contrast")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Stellaris QED Explorer")
st.markdown("*Quantum Vacuum Engineering for Extreme Astrophysical Environments*")
st.markdown("---")

# Metrics
B_ratio = B_surface / B_crit
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("B / B_crit", f"{B_ratio:.2e}")
with col2:
    st.metric("Max γ→A' P", f"{(epsilon * B_surface / 1e15)**2:.2e}")
with col3:
    st.metric("Dark Photon Mass", f"{m_dark:.1e} eV")
with col4:
    st.metric("Ω Entanglement", f"{omega:.2f}")

if B_ratio > 1:
    st.warning(f"⚠️ **Super-critical field!** B/B_crit = {B_ratio:.2e} | QED effects dominate.")


# ── FILE PROCESSING ─────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner(f"Loading {uploaded_file.name}..."):
        file_data = load_file(uploaded_file)
    
    if file_data is not None:
        st.success(f"✅ Loaded: {uploaded_file.name} | Type: {file_data['type'].upper()}")
        
        data = normalize_data(file_data['data'])
        
        # Resize if too large
        MAX_SIZE = 512
        if data.shape[0] > MAX_SIZE or data.shape[1] > MAX_SIZE:
            from skimage.transform import resize
            data = resize(data, (MAX_SIZE, MAX_SIZE), preserve_range=True)
            data = normalize_data(data)
        
        # Apply PDP physics
        with st.spinner("Applying PDP physics..."):
            result, wave, soliton = apply_pdp_to_data(data, omega, fringe, brightness)
        
        st.markdown("### 📊 Processed Data")
        
        col_orig, col_result = st.columns(2)
        with col_orig:
            fig = plot_data_with_overlay(data, "Original Data", 'gray')
            st.pyplot(fig)
            plt.close(fig)
        
        with col_result:
            fig = plot_data_with_overlay(result, f"PDP Entangled (Ω={omega:.2f})", 'inferno')
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("### ⚛️ PDP Components")
        col_wave, col_soliton = st.columns(2)
        
        with col_wave:
            fig = plot_data_with_overlay(wave, "Dark Photon Field", 'plasma')
            st.pyplot(fig)
            plt.close(fig)
        
        with col_soliton:
            fig = plot_data_with_overlay(soliton, "FDM Soliton Core", 'hot')
            st.pyplot(fig)
            plt.close(fig)
        
        # Download
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
            ax.imshow(result, cmap='inferno')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            plt.close(fig)
            st.download_button("📸 Download as PNG", buf.getvalue(), "processed.png")
        
        with col_d2:
            buf = io.BytesIO()
            np.save(buf, result)
            st.download_button("📊 Download as NPY", buf.getvalue(), "processed.npy")

else:
    st.info("""
    ## 📁 **Drop a file to begin**
    
    **Supported formats:**
    - 🔭 **FITS** - Astronomical images (HST, JWST, etc.)
    - 🏥 **DICOM** - Medical imaging data
    - 📊 **CSV** - Tabular scientific data
    - 🖼️ **Images** - PNG, JPG, TIFF, BMP
    - 📦 **NumPy** - .npy arrays
    
    **What happens:**
    1. Your data is normalized and processed
    2. **PDP physics** applies entanglement effects
    3. **Dark photon conversion** simulated
    4. Download enhanced results
    """)
    
    with st.expander("📖 Supported File Types"):
        st.markdown("""
        | Format | Extension | Use Case |
        |--------|-----------|----------|
        | **FITS** | .fits, .fit, .fts | Astronomy (HST, JWST, ground-based) |
        | **DICOM** | .dcm | Medical imaging, research scans |
        | **CSV** | .csv | Tabular data, simulation outputs |
        | **Images** | .png, .jpg, .tif | Photographs, scans, renderings |
        | **NumPy** | .npy | Saved arrays, simulation data |
        """)

# Physics Tabs
st.markdown("---")
st.markdown("### 🔬 Quantum Vacuum Physics")

tab1, tab2, tab3 = st.tabs(["🌌 Magnetar Field", "🕳️ Dark Photons", "🌀 Kerr Geodesics"])

with tab1:
    fig = plot_magnetar_field(B_surface, 10, 0)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig = plot_dark_photon_conversion(B_surface, epsilon, m_dark)
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig = plot_kerr_geodesic(a_spin)
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v6.1** | High Contrast Theme | Full File Support | Tony Ford Model")
