"""
Stellaris QED Explorer v6.0 – Full File Support
Drag & Drop | FITS | CSV | DICOM | Images | Real Data Processing
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, hbar, e, m_e, alpha
from scipy.ndimage import gaussian_filter, zoom
import warnings
import time
import json
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import base64
import hashlib

# Optional imports with fallbacks
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
    page_title="Stellaris QED Explorer v6.0",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom CSS for drag-and-drop styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #f0f4f8 0%, #e6ecf3 100%); }
[data-testid="stSidebar"] { background: #ffffff; border-right: 2px solid #00aaff; }
.stTitle, h1, h2, h3 { color: #0066cc; }
[data-testid="stMetricValue"] { color: #0066cc; }
[data-testid="stFileUploader"] { background-color: #e6f0ff; border: 2px dashed #0066cc; border-radius: 10px; padding: 20px; }
.drag-drop-text { color: #0066cc; font-size: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)  # 4.4e13 G
alpha_fine = 1/137.036


# ── FILE LOADING FUNCTIONS ─────────────────────────────────────────────

def load_file(uploaded_file):
    """Universal file loader for all science formats"""
    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()
    data_bytes = uploaded_file.read()
    
    # FITS files (astronomy)
    if ext in ['fits', 'fit', 'fts']:
        if HAS_ASTROPY:
            with fits.open(io.BytesIO(data_bytes)) as hdul:
                data = hdul[0].data.astype(np.float32)
                if len(data.shape) > 2:
                    data = data[0] if data.shape[0] < data.shape[1] else data[:, :, 0]
                header = dict(hdul[0].header) if hasattr(hdul[0], 'header') else {}
                return {'data': data, 'header': header, 'type': 'fits'}
        else:
            st.error("Astropy not installed. Install with: pip install astropy")
            return None
    
    # DICOM files (medical imaging)
    elif ext in ['dcm', 'dicom']:
        if HAS_DICOM:
            ds = pydicom.dcmread(io.BytesIO(data_bytes))
            data = ds.pixel_array.astype(np.float32)
            return {'data': data, 'header': dict(ds), 'type': 'dicom'}
        else:
            st.error("pydicom not installed. Install with: pip install pydicom")
            return None
    
    # CSV files (tabular data)
    elif ext == 'csv':
        df = pd.read_csv(io.BytesIO(data_bytes))
        return {'data': df.values, 'columns': df.columns.tolist(), 'type': 'csv', 'df': df}
    
    # Image files
    elif ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']:
        img = Image.open(io.BytesIO(data_bytes))
        if img.mode != 'L':
            img = img.convert('L')
        data = np.array(img, dtype=np.float32)
        return {'data': data, 'type': 'image', 'mode': img.mode}
    
    # Numpy files
    elif ext == 'npy':
        data = np.load(io.BytesIO(data_bytes))
        return {'data': data, 'type': 'numpy'}
    
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


def normalize_data(data):
    """Normalize data to [0,1] range"""
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    if data.max() > data.min():
        return (data - data.min()) / (data.max() - data.min())
    return data


# ── PHYSICS FUNCTIONS ─────────────────────────────────────────────

def magnetar_dipole_field(B_surface, R_ns, r, theta, inclination=0):
    """Magnetar dipole field"""
    B0 = B_surface * (R_ns / (r + 1e-9))**3
    theta_rad = np.radians(theta + inclination)
    B_r = 2 * B0 * np.cos(theta_rad)
    B_theta = B0 * np.sin(theta_rad)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    return B_r, B_theta, B_mag


def dark_photon_conversion(B, L, epsilon, m_dark, omega=1e18):
    """Dark photon conversion probability"""
    if m_dark <= 0:
        return (epsilon * B / 1e15)**2 * np.ones_like(L)
    hbar_ev_s = 6.582e-16
    c_km_s = 3e5
    conversion_length = 4 * omega * hbar_ev_s * c_km_s / (m_dark**2)
    P = (epsilon * B / 1e15)**2 * np.sin(np.pi * L / conversion_length)**2
    return np.clip(P, 0, 1)


def apply_pdp_to_data(data, omega, fringe, brightness=1.2):
    """Apply Photon-Dark Photon physics to uploaded data"""
    h, w = data.shape
    result = data.copy()
    
    # Create FDM soliton-like enhancement
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    soliton = np.exp(-r**2 / (2 * (0.2 * (50/fringe))**2))
    
    # Create wave pattern
    wave = np.sin(2 * np.pi * fringe * r / 50) * soliton
    
    # Mix
    mixing = omega * 0.5
    result = data * (1 - mixing * 0.5)
    result = result + wave * mixing * 0.5
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    return result, wave, soliton


# ── PLOTTING FUNCTIONS ─────────────────────────────────────────────

def plot_data_with_overlay(data, title, cmap='inferno', figsize=(6, 5)):
    """Plot data with overlay"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_magnetar_field(B_surface, R_ns, inclination):
    """Magnetar field visualization"""
    resolution = 40
    r = np.linspace(R_ns, 5*R_ns, resolution)
    theta = np.linspace(0, np.pi, resolution)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    B_r, B_theta, B_mag = magnetar_dipole_field(B_surface, R_ns, R, Theta, inclination)
    
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)
    U = B_r * np.sin(Theta) + B_theta * np.cos(Theta)
    V = B_r * np.cos(Theta) - B_theta * np.sin(Theta)
    
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-9)
    V_norm = V / (magnitude + 1e-9)
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    skip = 2
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              U_norm[::skip, ::skip], V_norm[::skip, ::skip],
              alpha=0.7, color='blue', width=0.008)
    
    ax.add_patch(Circle((0, 0), R_ns, color='#d62728', alpha=0.9))
    ax.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12)
    
    for angle in np.linspace(0, 2*np.pi, 12):
        t = np.linspace(0, 1, 40)
        r_line = R_ns * (1 + t * 3.5)
        theta_line = angle + 0.3 * np.sin(angle) * t
        x_line = r_line * np.cos(theta_line)
        y_line = r_line * np.sin(theta_line)
        ax.plot(x_line, y_line, 'gray', linewidth=0.8, alpha=0.5)
    
    ax.set_aspect('equal')
    ax.set_xlim(-5*R_ns, 5*R_ns)
    ax.set_ylim(-5*R_ns, 5*R_ns)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(f'Magnetar Field | B = {B_surface:.1e} G')
    
    return fig


def plot_qed_vacuum(B_ratio):
    """QED vacuum polarization plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    
    B_range = np.logspace(-1, min(2.5, np.log10(max(2, B_ratio))), 80)
    x = B_range**2
    delta_n = 4 * alpha_fine/(45 * np.pi) * x
    delta_n_para = 7 * alpha_fine/(45 * np.pi) * x
    
    axes[0].loglog(B_range, delta_n, 'b-', linewidth=2, label='⟂ Polarization')
    axes[0].loglog(B_range, delta_n_para, 'r-', linewidth=2, label='∥ Polarization')
    axes[0].set_xlabel('B / B_critical')
    axes[0].set_ylabel('n - 1')
    axes[0].set_title('Vacuum Refractive Index')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    delta_birefringence = np.abs(delta_n - delta_n_para)
    axes[1].loglog(B_range, delta_birefringence, 'g-', linewidth=2)
    axes[1].set_xlabel('B / B_critical')
    axes[1].set_ylabel('|Δn|')
    axes[1].set_title('Vacuum Birefringence')
    axes[1].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_facecolor('#f8f9fa')
    
    fig.tight_layout()
    return fig


def plot_dark_photon_conversion(B, epsilon, m_dark):
    """Dark photon conversion plot"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    L = np.logspace(-2, 2, 500)
    P = dark_photon_conversion(B, L, epsilon, m_dark)
    
    ax.semilogx(L, P, 'b-', linewidth=2)
    ax.set_xlabel('Propagation Length (km)')
    ax.set_ylabel('Conversion Probability')
    ax.set_title(f'γ ↔ A\' Conversion | ε = {epsilon:.1e}, m\' = {m_dark:.1e} eV')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    return fig


def plot_kerr_geodesic(a_spin):
    """Kerr geodesic plot"""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    r_horizon = 1 + np.sqrt(1 - a_spin**2)
    circle = Circle((0, 0), r_horizon, color='gray', alpha=0.5)
    ax.add_patch(circle)
    ax.text(0, 0, 'BH', color='black', ha='center', va='center', fontsize=10)
    
    if a_spin <= 0.999:
        r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r_photon * np.cos(theta), r_photon * np.sin(theta), 
                'r--', linewidth=2, label='Photon Sphere', alpha=0.7)
    
    for impact in [6, 8, 10]:
        t = np.linspace(0, 50, 300)
        r = 12 * np.exp(-t/35) + r_horizon + 0.5
        phi = (impact/10) * np.sin(t/25)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        ax.plot(x, y, 'b-', linewidth=1.2, alpha=0.6)
    
    ax.set_aspect('equal')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel('x/M')
    ax.set_ylabel('y/M')
    ax.set_title(f'Kerr Spacetime | a/M = {a_spin:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Stellaris QED")
    st.markdown("*Drag & Drop Any Science File*")
    st.markdown("---")
    
    # Drag-and-drop file uploader
    uploaded_file = st.file_uploader(
        "📁 **Drop files here**",
        type=['fits', 'fit', 'fts', 'csv', 'dcm', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'npy'],
        accept_multiple_files=False,
        help="Supports FITS, CSV, DICOM, Images, NumPy arrays"
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
    
    st.caption("Tony Ford Model | v6.0 - Full File Support")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Stellaris QED Explorer")
st.markdown("*Quantum Vacuum Engineering for Extreme Astrophysical Environments*")
st.markdown("---")

# Metrics row
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
        # Display file info
        st.success(f"✅ Loaded: {uploaded_file.name} | Type: {file_data['type'].upper()}")
        
        # Process data
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
        
        # Display results
        st.markdown("### 📊 Processed Data")
        
        col_orig, col_result = st.columns(2)
        with col_orig:
            fig = plot_data_with_overlay(data, "Original Data", 'gray')
            st.pyplot(fig)
            plt.close(fig)
        
        with col_result:
            fig = plot_data_with_overlay(result, f"PDP Entangled (Ω={omega:.2f}, fringe={fringe})", 'inferno')
            st.pyplot(fig)
            plt.close(fig)
        
        # Physics components
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
        
        # Download processed data
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            # Save as PNG
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(result, cmap='inferno')
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            st.download_button("📸 Download as PNG", buf.getvalue(), "processed.png")
        
        with col_d2:
            # Save as NumPy
            buf = io.BytesIO()
            np.save(buf, result)
            st.download_button("📊 Download as NPY", buf.getvalue(), "processed.npy")

else:
    # Show instructions when no file uploaded
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
    
    # Show example file types
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

# Physics Tabs (always available)
st.markdown("---")
st.markdown("### 🔬 Quantum Vacuum Physics")

tab1, tab2, tab3, tab4 = st.tabs(["🌌 Magnetar Field", "⚛️ QED Vacuum", "🕳️ Dark Photons", "🌀 Kerr Geodesics"])

with tab1:
    fig = plot_magnetar_field(B_surface, 10, 0)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig = plot_qed_vacuum(B_ratio)
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig = plot_dark_photon_conversion(B_surface, epsilon, m_dark)
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    fig = plot_kerr_geodesic(a_spin)
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v6.0** | Full File Support | Tony Ford Model")
