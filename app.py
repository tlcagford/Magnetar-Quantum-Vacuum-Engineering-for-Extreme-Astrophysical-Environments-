"""
Stellaris QED Explorer v8.0 – QCI-Style Side-by-Side with Overlays
Full integration: Magnetar Field + Dark Photons + FDM Solitons + Overlays
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.constants import c, hbar, e, m_e, alpha
from scipy.ndimage import gaussian_filter, sobel
from PIL import Image, ImageDraw, ImageFont
import warnings
import time
import pandas as pd

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
    page_title="Stellaris QED Explorer v8.0",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Professional dark theme with light accents
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0a0a1a; }
    [data-testid="stSidebar"] { background: #0f0f1f; border-right: 2px solid #00aaff; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: #ffffff !important; }
    .stTitle, h1, h2, h3 { color: #00aaff !important; }
    [data-testid="stMetricValue"] { color: #00aaff !important; }
    [data-testid="stFileUploader"] { background-color: #1a1a2a; border: 2px dashed #00aaff; border-radius: 10px; }
    .stInfo { background-color: #1a2a3a; border-left: 4px solid #00aaff; }
    .stSuccess { background-color: #1a3a2a; border-left: 4px solid #00ffaa; }
    .stWarning { background-color: #3a2a1a; border-left: 4px solid #ffaa00; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)
alpha_fine = 1/137.036


# ── QCI-STYLE ANNOTATION FUNCTION ─────────────────────────────────────────────

def add_qci_annotations(image_array, metadata, scale_kpc=100):
    """
    Add annotations to image like QCI AstroEntangle Refiner
    Includes: scale bar, north indicator, physics info, formulas
    """
    # Convert to PIL
    if len(image_array.shape) == 3:
        img = (image_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    h, w = image_array.shape[:2]
    
    # ── SCALE BAR ─────────────────────────────────────────────
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    bar_y = h - 40
    bar_x_start = 20
    bar_x_end = bar_x_start + scale_bar_px
    
    draw.rectangle([bar_x_start, bar_y, bar_x_end, bar_y + 5], fill='white')
    draw.text((bar_x_start + 30, bar_y - 18), f"{scale_bar_kpc:.0f} kpc", 
              fill='white', font=font_small, stroke_width=1, stroke_fill='black')
    
    # ── NORTH INDICATOR ─────────────────────────────────────────────
    north_x = w - 30
    draw.line([north_x, 30, north_x, 60], fill='white', width=2)
    draw.text((north_x - 8, 15), "N", fill='white', font=font_medium)
    
    # ── PHYSICS INFO BOX (LEFT TOP) ─────────────────────────────────────────────
    info_lines = [
        f"Ω = {metadata['omega']:.2f} | Fringe = {metadata['fringe']}",
        f"Mixing = {metadata['mixing']:.3f} | Entropy = {metadata['entropy']:.3f}",
        f"λ_FDM = {scale_bar_kpc / metadata['fringe'] * 8:.1f} kpc",
        f"PDP Active: {metadata['active_modules']}"
    ]
    
    # Background box
    box_width = 240
    box_height = len(info_lines) * 22 + 10
    draw.rectangle([10, 10, 10 + box_width, 10 + box_height], 
                   fill=(0, 0, 0, 180), outline='white')
    
    for i, line in enumerate(info_lines):
        draw.text((15, 15 + i * 22), line, fill='cyan', font=font_small)
    
    # ── FORMULA BOX (RIGHT BOTTOM) ─────────────────────────────────────────────
    formulas = [
        r"ρ(r) ∝ [sin(kr)/kr]²",
        r"λ = h/(m v)",
        r"P(γ→A') = (εB/m')² sin²(m'²L/4ω)",
        r"S = -Tr(ρ log ρ)"
    ]
    
    formula_y_start = h - 100
    formula_width = 220
    draw.rectangle([w - formula_width - 10, formula_y_start - 5, 
                    w - 10, formula_y_start + len(formulas) * 18 + 5],
                   fill=(0, 0, 0, 160), outline='#88ff88')
    
    for i, formula in enumerate(formulas):
        draw.text((w - formula_width - 5, formula_y_start + i * 18), 
                  formula, fill='#88ff88', font=font_small)
    
    return np.array(img_pil) / 255.0


def create_side_by_side_comparison(original, processed, metadata, scale_kpc=100):
    """
    Create side-by-side comparison with annotations like QCI app
    """
    # Annotate both images
    original_annotated = add_qci_annotations(original, metadata, scale_kpc)
    
    # Processed gets additional physics overlay
    processed_with_overlay = add_qci_annotations(processed, metadata, scale_kpc)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0a0a1a')
    
    # Original
    ax1.imshow(original_annotated)
    ax1.set_title("Before: Standard View\n(Public HST/JWST Data)", 
                  color='white', fontsize=12, pad=10)
    ax1.axis('off')
    
    # Processed
    ax2.imshow(processed_with_overlay)
    ax2.set_title("After: Photon-Dark-Photon Entangled\nFDM Overlays (Stellaris Model)", 
                  color='white', fontsize=12, pad=10)
    ax2.axis('off')
    
    fig.tight_layout()
    return fig


# ── PHYSICS COMPONENTS ─────────────────────────────────────────────

def create_fdm_soliton(size, fringe):
    """FDM Soliton Core - ρ(r) ∝ [sin(kr)/(kr)]²"""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    
    r_s = 0.2 * (50.0 / max(fringe, 1))
    k = np.pi / max(r_s, 0.01)
    kr = k * r
    
    with np.errstate(divide='ignore', invalid='ignore'):
        soliton = np.where(kr > 1e-6, (np.sin(kr) / kr)**2, 1.0)
    
    soliton = (soliton - soliton.min()) / (soliton.max() - soliton.min() + 1e-9)
    soliton = gaussian_filter(soliton, sigma=2)
    
    return soliton


def create_dark_photon_wave(size, fringe):
    """Dark Photon Wave Pattern - λ = h/(m v)"""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    theta = np.arctan2(y - cy, x - cx)
    k = fringe / 20.0
    
    radial = np.sin(k * 2 * np.pi * r * 3)
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    angular = np.sin(k * 3 * theta)
    
    if fringe < 50:
        pattern = radial * 0.6 + spiral * 0.4
    elif fringe < 80:
        pattern = radial * 0.4 + spiral * 0.4 + angular * 0.2
    else:
        pattern = spiral * 0.5 + angular * 0.3 + radial * 0.2
    
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    return pattern


def create_dark_matter_density(image, soliton):
    """Dark Matter Density from gradients"""
    smoothed = gaussian_filter(image, sigma=8)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    if gradient.max() > gradient.min():
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    else:
        gradient = np.zeros_like(gradient)
    
    return np.clip(soliton * 0.6 + gradient * 0.4, 0, 1)


def create_rgb_overlay(image, dark_photon, dm_density, soliton):
    """RGB Composite: R=Image, G=Dark Photon+Soliton, B=Dark Matter+Soliton"""
    img_norm = np.clip(image, 0, 1)
    dp_norm = np.clip(dark_photon, 0, 1)
    dm_norm = np.clip(dm_density, 0, 1)
    sol_norm = np.clip(soliton, 0, 1)
    
    red = img_norm
    green = img_norm * 0.3 + dp_norm * 0.5 + sol_norm * 0.2
    blue = img_norm * 0.2 + dm_norm * 0.6 + sol_norm * 0.2
    
    return np.clip(np.stack([red, green, blue], axis=-1), 0, 1)


def process_image(image, omega, fringe, brightness=1.2):
    """Full PDP processing pipeline"""
    h, w = image.shape
    
    # Create components
    soliton = create_fdm_soliton((h, w), fringe)
    dark_photon = create_dark_photon_wave((h, w), fringe)
    dm_density = create_dark_matter_density(image, soliton)
    
    # Mixing
    mixing = omega * 0.6
    
    # Entangled result
    result = image * (1 - mixing * 0.4)
    result = result + dark_photon * mixing * 0.5
    result = result + dm_density * mixing * 0.3
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    # RGB overlay
    rgb = create_rgb_overlay(result, dark_photon, dm_density, soliton)
    
    # Entropy
    entropy = -mixing * np.log(mixing + 1e-12)
    
    metadata = {
        'omega': omega,
        'fringe': fringe,
        'mixing': mixing,
        'entropy': entropy,
        'brightness': brightness,
        'active_modules': 'FDM + Dark Photon + DM'
    }
    
    return {
        'original': image,
        'entangled': result,
        'soliton': soliton,
        'dark_photon': dark_photon,
        'dark_matter': dm_density,
        'rgb_overlay': rgb,
        'metadata': metadata
    }


# ── FILE LOADING ─────────────────────────────────────────────

def load_file(uploaded_file):
    """Load image file"""
    ext = uploaded_file.name.split(".")[-1].lower()
    data_bytes = uploaded_file.read()
    
    try:
        if ext in ['fits', 'fit', 'fts'] and HAS_ASTROPY:
            with fits.open(io.BytesIO(data_bytes)) as hdul:
                data = hdul[0].data.astype(np.float32)
                if len(data.shape) > 2:
                    data = data[0] if data.shape[0] < data.shape[1] else data[:, :, 0]
                return {'data': data, 'type': 'FITS'}
        
        elif ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp']:
            img = Image.open(io.BytesIO(data_bytes)).convert('L')
            return {'data': np.array(img, dtype=np.float32), 'type': 'IMAGE'}
        
        else:
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def normalize(data):
    """Normalize to [0,1]"""
    data = np.nan_to_num(data, nan=0.0)
    if data.max() > data.min():
        return (data - data.min()) / (data.max() - data.min())
    return data


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Stellaris QED v8.0")
    st.markdown("*QCI-Style Side-by-Side*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload Image", type=['fits', 'png', 'jpg', 'jpeg', 'tif', 'tiff'])
    
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
    
    st.caption("Tony Ford Model | v8.0 - QCI Style")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Stellaris QED Explorer")
st.markdown("*Photon-Dark-Photon Entangled FDM with Soliton Overlays*")
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


# ── PROCESS UPLOADED IMAGE ─────────────────────────────────────────────
if uploaded is not None:
    with st.spinner("Loading image..."):
        file_data = load_file(uploaded)
    
    if file_data is not None:
        st.success(f"✅ Loaded: {uploaded.name} | Type: {file_data['type']}")
        
        # Process image
        data = normalize(file_data['data'])
        
        # Resize if needed
        MAX_SIZE = 500
        if data.shape[0] > MAX_SIZE or data.shape[1] > MAX_SIZE:
            from skimage.transform import resize
            data = resize(data, (MAX_SIZE, MAX_SIZE), preserve_range=True)
            data = normalize(data)
        
        # Apply PDP
        with st.spinner("Applying PDP physics..."):
            results = process_image(data, omega, fringe, brightness)
        
        # ── SIDE-BY-SIDE COMPARISON (QCI STYLE) ─────────────────────────────────────────────
        st.markdown("### 📊 Before vs After")
        
        comparison_fig = create_side_by_side_comparison(
            results['original'], 
            results['rgb_overlay'], 
            results['metadata'], 
            scale_kpc
        )
        st.pyplot(comparison_fig)
        plt.close(comparison_fig)
        
        # ── PHYSICS COMPONENTS ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚛️ FDM Physics Components")
        
        col_a, col_b, col_c = st.columns(3)
        
        def show_component(img, title, cmap):
            fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, color='#00aaff')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        with col_a:
            show_component(results['soliton'], "FDM Soliton Core", 'hot')
            st.caption(r"$\rho(r) \propto [\sin(kr)/(kr)]^2$")
        
        with col_b:
            show_component(results['dark_photon'], "Dark Photon Field", 'plasma')
            st.caption(r"$\lambda = h/(m v)$ interference")
        
        with col_c:
            show_component(results['dark_matter'], "Dark Matter Density", 'viridis')
            st.caption(r"From $\nabla^2\Phi = 4\pi G\rho$")
        
        # ── METRICS ─────────────────────────────────────────────
        st.markdown("---")
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
        
        def save_fig(img, cmap=None):
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
            if len(img.shape) == 3:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            plt.close(fig)
            return buf.getvalue()
        
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        
        with col_d1:
            # Save comparison figure
            buf = io.BytesIO()
            comparison_fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            st.download_button("📸 Comparison", buf.getvalue(), "comparison.png")
        with col_d2:
            st.download_button("🌌 PDP Entangled", save_fig(results['entangled'], 'inferno'), "entangled.png")
        with col_d3:
            st.download_button("⭐ Soliton Core", save_fig(results['soliton'], 'hot'), "soliton.png")
        with col_d4:
            st.download_button("🌊 Fringe Pattern", save_fig(results['dark_photon'], 'plasma'), "fringe.png")

else:
    st.info("""
    ## 📁 **Upload an image to see FDM Soliton Overlays**
    
    **What you'll see:**
    - 📊 **Side-by-Side Comparison**: Before/After with annotations
    - 📏 **Scale Bar**: Physical scale in kpc
    - 🧭 **North Indicator**: Orientation marker
    - 📐 **Physics Info Box**: Ω, fringe, mixing angle, entropy
    - 📝 **Formula Overlays**: Key equations on the image
    - ⚛️ **FDM Soliton Core**: [sin(kr)/kr]² profile
    - 🌊 **Dark Photon Field**: Wave interference patterns
    - 🌌 **Dark Matter Density**: Substructure map
    
    **Try with:** Crab Nebula, Bullet Cluster, Abell 1689, or any astronomical image
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
st.markdown("⚡ **Stellaris QED Explorer v8.0** | QCI-Style Side-by-Side | FDM Soliton Overlays | Tony Ford Model")
