"""
Stellaris QED Explorer v7.0 – QCI-Style Overlays
Soliton cores | Wave fringes | RGB overlays | Full physics
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from scipy.constants import c, hbar, e, m_e, alpha
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import convolve2d
import warnings
import time
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Optional imports
try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Stellaris QED Explorer v7.0",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# High contrast dark theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0a0a1a; }
    [data-testid="stSidebar"] { background: #0f0f1f; border-right: 2px solid #00aaff; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: #ffffff !important; }
    .stTitle, h1, h2, h3 { color: #00aaff !important; }
    [data-testid="stMetricValue"] { color: #00aaff !important; }
    [data-testid="stFileUploader"] { background-color: #1a1a2a; border: 2px dashed #00aaff; }
    .stInfo { background-color: #1a2a3a; border-left: 4px solid #00aaff; }
    .stSuccess { background-color: #1a3a2a; border-left: 4px solid #00ffaa; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)
alpha_fine = 1/137.036


# ── CORE PHYSICS FUNCTIONS (From QCI Framework) ─────────────────────────────────────────────

def create_fdm_soliton(size, fringe):
    """
    FDM Soliton Core - ρ(r) ∝ [sin(kr)/(kr)]²
    From QCI AstroEntangle Refiner
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    
    # Soliton scale depends on fringe
    r_s = 0.2 * (50.0 / max(fringe, 1))
    k = np.pi / max(r_s, 0.01)
    kr = k * r
    
    with np.errstate(divide='ignore', invalid='ignore'):
        soliton = np.where(kr > 1e-6, (np.sin(kr) / kr)**2, 1.0)
    
    # Normalize to [0,1]
    soliton = (soliton - soliton.min()) / (soliton.max() - soliton.min() + 1e-9)
    soliton = gaussian_filter(soliton, sigma=2)
    
    return soliton


def create_dark_photon_wave(size, fringe):
    """
    Dark Photon Wave Pattern - λ = h/(m v)
    From QCI AstroEntangle Refiner
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    theta = np.arctan2(y - cy, x - cx)
    k = fringe / 20.0
    
    # Multiple wave modes
    radial = np.sin(k * 2 * np.pi * r * 3)
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    angular = np.sin(k * 3 * theta)
    
    if fringe < 50:
        pattern = radial * 0.6 + spiral * 0.4
    elif fringe < 80:
        pattern = radial * 0.4 + spiral * 0.4 + angular * 0.2
    else:
        pattern = spiral * 0.5 + angular * 0.3 + radial * 0.2
    
    # Normalize to [0,1]
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    
    return pattern


def create_dark_matter_density(image, soliton):
    """
    Dark Matter Density from gradients
    From QCI AstroEntangle Refiner
    """
    smoothed = gaussian_filter(image, sigma=8)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    if gradient.max() > gradient.min():
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    else:
        gradient = np.zeros_like(gradient)
    
    dm = soliton * 0.6 + gradient * 0.4
    return np.clip(dm, 0, 1)


def create_rgb_overlay(image, dark_photon, dm_density, soliton, alpha=0.6):
    """
    Create RGB composite overlay like QCI Refiner
    R: Original image
    G: Dark Photon Field + Soliton
    B: Dark Matter Density + Soliton
    """
    # Normalize inputs
    img_norm = np.clip(image, 0, 1)
    dp_norm = np.clip(dark_photon, 0, 1)
    dm_norm = np.clip(dm_density, 0, 1)
    sol_norm = np.clip(soliton, 0, 1)
    
    # RGB channels
    red = img_norm
    green = img_norm * 0.3 + dp_norm * 0.5 + sol_norm * 0.2
    blue = img_norm * 0.2 + dm_norm * 0.6 + sol_norm * 0.2
    
    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(rgb, 0, 1)


def add_annotations_to_image(img_array, metadata, scale_kpc=100):
    """
    Add physics annotations to image (scale bar, formulas, etc.)
    """
    if len(img_array.shape) == 3:
        img = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    h, w = img_array.shape[:2]
    
    # Scale bar
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    draw.rectangle([20, h-40, 20+scale_bar_px, h-35], fill='white', outline='white')
    draw.text((20+30, h-58), f"{scale_bar_kpc:.0f} kpc", fill='white', font=font_tiny)
    
    # North indicator
    draw.line([w-30, 30, w-30, 60], fill='white', width=2)
    draw.text((w-38, 15), "N", fill='white', font=font_small)
    
    # Physics info box
    info_lines = [
        f"Ω = {metadata['omega']:.2f} | Fringe = {metadata['fringe']}",
        f"Mixing = {metadata['mixing']:.3f} | Entropy = {metadata['entropy']:.3f}",
        f"λ = {scale_bar_kpc / metadata['fringe'] * 8:.1f} kpc"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((15, 15 + i*22), line, fill='cyan', font=font_small, stroke_width=1, stroke_fill='black')
    
    # Formulas
    formulas = [
        r"ρ(r) ∝ [sin(kr)/kr]²",
        r"λ = h/(m v)",
        r"P(γ→A') = (εB/m')² sin²(m'²L/4ω)"
    ]
    
    formula_y = h - 80
    for i, formula in enumerate(formulas):
        draw.text((w - 210, formula_y + i*18), formula, fill='#88ff88', font=font_tiny)
    
    return np.array(img_pil) / 255.0


# ── FILE LOADING ─────────────────────────────────────────────

def load_file(uploaded_file):
    """Universal file loader"""
    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()
    data_bytes = uploaded_file.read()
    
    try:
        if ext in ['fits', 'fit', 'fts'] and HAS_ASTROPY:
            with fits.open(io.BytesIO(data_bytes)) as hdul:
                data = hdul[0].data.astype(np.float32)
                if len(data.shape) > 2:
                    data = data[0] if data.shape[0] < data.shape[1] else data[:, :, 0]
                return {'data': data, 'type': 'FITS', 'name': filename}
        
        elif ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']:
            img = Image.open(io.BytesIO(data_bytes))
            if img.mode != 'L':
                img = img.convert('L')
            data = np.array(img, dtype=np.float32)
            return {'data': data, 'type': 'IMAGE', 'name': filename}
        
        elif ext == 'csv':
            df = pd.read_csv(io.BytesIO(data_bytes))
            data = df.values.astype(np.float32)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            return {'data': data, 'type': 'CSV', 'name': filename}
        
        elif ext == 'npy':
            data = np.load(io.BytesIO(data_bytes))
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            return {'data': data, 'type': 'NUMPY', 'name': filename}
        
        else:
            return None
            
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None


def normalize_data(data):
    """Normalize to [0,1] range"""
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    if data.max() > data.min():
        return (data - data.min()) / (data.max() - data.min())
    return data


# ── MAIN PROCESSING PIPELINE ─────────────────────────────────────────────

def process_image_with_pdp(image, omega, fringe, brightness=1.2, scale_kpc=100):
    """
    Full PDP processing with QCI-style outputs
    """
    h, w = image.shape
    
    # Create physics components
    soliton = create_fdm_soliton((h, w), fringe)
    dark_photon = create_dark_photon_wave((h, w), fringe)
    dm_density = create_dark_matter_density(image, soliton)
    
    # Mixing strength
    mixing = omega * 0.6
    
    # Entangled image
    result = image * (1 - mixing * 0.4)
    result = result + dark_photon * mixing * 0.5
    result = result + dm_density * mixing * 0.3
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    # RGB composite overlay
    rgb_overlay = create_rgb_overlay(result, dark_photon, dm_density, soliton)
    
    # Annotated version
    metadata = {
        'omega': omega,
        'fringe': fringe,
        'mixing': mixing,
        'entropy': -mixing * np.log(mixing + 1e-12),
        'brightness': brightness,
        'scale_kpc': scale_kpc
    }
    
    annotated = add_annotations_to_image(rgb_overlay, metadata, scale_kpc)
    
    return {
        'original': image,
        'entangled': result,
        'soliton': soliton,
        'dark_photon': dark_photon,
        'dark_matter': dm_density,
        'rgb_overlay': rgb_overlay,
        'annotated': annotated,
        'metadata': metadata
    }


# ── PLOTTING FUNCTIONS ─────────────────────────────────────────────

def display_image(img, title, cmap=None, figsize=(5, 5)):
    """Display image with optional colormap"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    if len(img.shape) == 3:
        ax.imshow(np.clip(img, 0, 1))
    else:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_title(title, color='#00aaff', fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    return fig


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Stellaris QED v7.0")
    st.markdown("*QCI-Style Overlays*")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 **Drop file here**",
        type=['fits', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'npy', 'csv'],
        help="FITS | Images | NumPy | CSV"
    )
    
    st.markdown("---")
    st.markdown("### ⚛️ PDP Parameters")
    omega = st.slider("Ω Entanglement", 0.1, 1.0, 0.70, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, 65, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], index=1)
    
    st.markdown("---")
    st.markdown("### 🌌 Magnetar Parameters")
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("Kinetic Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin a/M", 0.0, 0.998, 0.9)
    
    st.markdown("---")
    st.latex(r"\rho_{\text{FDM}} \propto \left[\frac{\sin(kr)}{kr}\right]^2")
    st.latex(r"P_{\gamma\to A'} = \left(\frac{\varepsilon B}{m_{A'}}\right)^2\sin^2\left(\frac{m_{A'}^2 L}{4\omega}\right)")
    
    st.caption("Tony Ford Model | v7.0 - QCI Style")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Stellaris QED Explorer")
st.markdown("*Photon-Dark Photon Entanglement with FDM Soliton Overlays*")
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


# ── PROCESS UPLOADED FILE ─────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner(f"📂 Loading {uploaded_file.name}..."):
        file_data = load_file(uploaded_file)
    
    if file_data is not None:
        st.success(f"✅ Loaded: {file_data['name']} | Type: {file_data['type']} | Shape: {file_data['data'].shape}")
        
        # Process data
        data = normalize_data(file_data['data'])
        
        # Resize if too large
        MAX_SIZE = 500
        if data.shape[0] > MAX_SIZE or data.shape[1] > MAX_SIZE:
            from skimage.transform import resize
            data = resize(data, (MAX_SIZE, MAX_SIZE), preserve_range=True)
            data = normalize_data(data)
        
        # Apply PDP processing
        with st.spinner("⚛️ Applying PDP physics with soliton overlays..."):
            results = process_image_with_pdp(data, omega, fringe, brightness, scale_kpc)
        
        # ── MAIN COMPARISON (ANNOTATED) ─────────────────────────────────────────────
        st.markdown("### 📊 Annotated Comparison")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0a0a1a')
        
        ax1.imshow(results['original'], cmap='gray')
        ax1.set_title("Before: Original Image", color='white', fontsize=12)
        ax1.axis('off')
        
        ax2.imshow(results['annotated'])
        ax2.set_title("After: Photon-Dark-Photon Entangled\nFDM Overlays", color='white', fontsize=12)
        ax2.axis('off')
        
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # ── PHYSICS COMPONENTS ─────────────────────────────────────────────
        st.markdown("### ⚛️ FDM Physics Components")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            fig = display_image(results['soliton'], "FDM Soliton Core", 'hot', (4, 4))
            st.pyplot(fig)
            plt.close(fig)
            st.caption(r"$\rho(r) \propto [\sin(kr)/(kr)]^2$")
        
        with col_b:
            fig = display_image(results['dark_photon'], "Dark Photon Field", 'plasma', (4, 4))
            st.pyplot(fig)
            plt.close(fig)
            st.caption(r"$\lambda = h/(m v)$ interference pattern")
        
        with col_c:
            fig = display_image(results['dark_matter'], "Dark Matter Density", 'viridis', (4, 4))
            st.pyplot(fig)
            plt.close(fig)
            st.caption(r"From $\nabla^2\Phi = 4\pi G\rho$")
        
        # ── RGB OVERLAY ─────────────────────────────────────────────
        st.markdown("### 🎨 RGB Composite Overlay")
        st.caption("Red: Image | Green: Dark Photon + Soliton | Blue: Dark Matter + Soliton")
        
        fig = display_image(results['rgb_overlay'], "PDP Entangled RGB Composite", None, (8, 8))
        st.pyplot(fig)
        plt.close(fig)
        
        # ── METRICS ─────────────────────────────────────────────
        st.markdown("### 📈 Physics Metrics")
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        with col_m1:
            st.metric("Soliton Peak", f"{results['soliton'].max():.3f}")
        with col_m2:
            st.metric("Fringe Contrast", f"{results['dark_photon'].std():.3f}")
        with col_m3:
            st.metric("Mixing Angle", f"{results['metadata']['mixing']:.3f}")
        with col_m4:
            st.metric("Entanglement Entropy", f"{results['metadata']['entropy']:.3f}")
        with col_m5:
            gain = results['entangled'].std() / (results['original'].std() + 1e-9)
            st.metric("Contrast Gain", f"{gain:.2f}x")
        
        # ── DOWNLOAD ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        def array_to_png(arr, cmap=None):
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
            if len(arr.shape) == 3:
                ax.imshow(np.clip(arr, 0, 1))
            else:
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            plt.close(fig)
            return buf.getvalue()
        
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        
        with col_d1:
            st.download_button("📸 Annotated Comparison", array_to_png(results['annotated']), "annotated_comparison.png")
        with col_d2:
            st.download_button("🌌 Entangled Image", array_to_png(results['entangled'], 'inferno'), "entangled.png")
        with col_d3:
            st.download_button("⭐ Soliton Core", array_to_png(results['soliton'], 'hot'), "soliton.png")
        with col_d4:
            st.download_button("🌊 Fringe Pattern", array_to_png(results['dark_photon'], 'plasma'), "fringe.png")

else:
    st.info("""
    ## 📁 **Drop an image to see FDM Soliton Overlays**
    
    **What you'll see:**
    - 🎨 **RGB Composite**: Red=Image, Green=Dark Photon, Blue=Dark Matter
    - ⭐ **FDM Soliton Core**: Central [sin(kr)/kr]² profile
    - 🌊 **Dark Photon Field**: Wave interference patterns
    - 📏 **Scale Bar**: Physical scale in kpc
    - 📐 **Physics Formulas**: Key equations overlaid
    
    **Try with:**
    - Crab Nebula, Bullet Cluster, Abell 1689, or any astronomical image
    """)

# ── PHYSICS TABS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Quantum Vacuum Physics")

tab1, tab2, tab3 = st.tabs(["🌌 Magnetar Field", "🕳️ Dark Photons", "🌀 Kerr Geodesics"])

with tab1:
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0a0a1a')
    r = np.linspace(1.2, 5, 40)
    theta = np.linspace(0, 2*np.pi, 40)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    B_val = B_surface / (R**3)
    B_norm = np.log10(B_val + 1e-9)
    B_norm = (B_norm - B_norm.min()) / (B_norm.max() - B_norm.min() + 1e-9)
    sc = ax.scatter(X, Y, c=B_norm, cmap='plasma', s=3, alpha=0.7)
    ax.add_patch(Circle((0, 0), 1, color='#ff4444', alpha=0.9))
    ax.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_title(f'Magnetar Field | B = {B_surface:.1e} G', color='#00aaff')
    ax.axis('off')
    plt.colorbar(sc, ax=ax, fraction=0.046, label='log₁₀|B|')
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0a0a1a')
    L = np.logspace(-2, 2, 500)
    if m_dark <= 0:
        P = (epsilon * B_surface / 1e15)**2 * np.ones_like(L)
    else:
        hbar_ev_s = 6.582e-16
        c_km_s = 3e5
        conv_len = 4 * 1e18 * hbar_ev_s * c_km_s / (m_dark**2)
        P = (epsilon * B_surface / 1e15)**2 * np.sin(np.pi * L / conv_len)**2
    P = np.clip(P, 0, 1)
    ax.semilogx(L, P, '#00aaff', linewidth=2.5)
    ax.axhline(y=(epsilon * B_surface / 1e15)**2, color='#ff8888', linestyle='--', 
               label=f'Max P = {(epsilon * B_surface / 1e15)**2:.2e}')
    ax.set_xlabel('Length (km)', color='white')
    ax.set_ylabel('P(γ→A\')', color='white')
    ax.set_title('Dark Photon Conversion', color='#00aaff')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(colors='white')
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0a0a1a')
    r_horizon = 1 + np.sqrt(1 - a_spin**2)
    ax.add_patch(Circle((0, 0), r_horizon, color='#555555', alpha=0.7))
    ax.text(0, 0, 'BH', color='white', ha='center', va='center', fontsize=12)
    if a_spin <= 0.999:
        r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r_photon * np.cos(theta), r_photon * np.sin(theta), 
                '#ff8888', linewidth=2, linestyle='--', label='Photon Sphere')
    for impact in [6, 8, 10]:
        t = np.linspace(0, 50, 400)
        r = 12 * np.exp(-t/35) + r_horizon + 0.5
        phi = (impact/10) * np.sin(t/25)
        ax.plot(r * np.cos(phi), r * np.sin(phi), '#88ff88', linewidth=1.5, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_title(f'Kerr Spacetime | a/M = {a_spin:.3f}', color='#00aaff')
    ax.legend()
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v7.0** | QCI-Style Overlays | Soliton Cores | Wave Fringes | Tony Ford Model")
