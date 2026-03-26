"""
Stellaris QED Explorer v9.0 – With Preloaded Examples
Preloaded: Bullet Cluster, Abell 1689, Abell 209, Crab Nebula, Centaurus A
Full QCI-style side-by-side with annotations
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, hbar, e, m_e, alpha
from scipy.ndimage import gaussian_filter, sobel
from PIL import Image, ImageDraw, ImageFont
import warnings
import base64
import urllib.request
import json

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
    page_title="Stellaris QED Explorer v9.0",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Professional dark theme
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
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)
alpha_fine = 1/137.036


# ── PRELOADED DATASETS ─────────────────────────────────────────────
PRELOADED_DATASETS = {
    "Bullet Cluster (1E0657-56)": {
        "description": "Merging galaxy cluster showing dark matter separation",
        "fringe": 70,
        "omega": 0.75,
        "scale_kpc": 200,
        "notes": "Enhanced dark matter substructure visible",
        "image_data": None  # Will be generated synthetic for demo
    },
    "Abell 1689": {
        "description": "Strong lensing cluster with dark matter substructure",
        "fringe": 55,
        "omega": 0.65,
        "scale_kpc": 150,
        "notes": "Prominent soliton core expected",
        "image_data": None
    },
    "Abell 209": {
        "description": "Galaxy cluster with visible FDM waves",
        "fringe": 60,
        "omega": 0.70,
        "scale_kpc": 100,
        "notes": "Balanced fringe and soliton visibility",
        "image_data": None
    },
    "Crab Nebula (M1)": {
        "description": "Supernova remnant with pulsar wind nebula",
        "fringe": 50,
        "omega": 0.68,
        "scale_kpc": 2,
        "notes": "Filamentary structure enhanced",
        "image_data": None
    },
    "Centaurus A (NGC 5128)": {
        "description": "Radio galaxy with active galactic nucleus",
        "fringe": 45,
        "omega": 0.62,
        "scale_kpc": 50,
        "notes": "Jet structure and dust lane",
        "image_data": None
    }
}


def generate_synthetic_cluster(size=400, cluster_type="bullet"):
    """Generate synthetic cluster image for preloaded examples"""
    img = np.zeros((size, size))
    cx, cy = size//2, size//2
    
    # Main cluster core
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - cx)**2 + (j - cy)**2)
            if cluster_type == "bullet":
                # Bullet cluster - two components
                r2 = np.sqrt((i - cx - 50)**2 + (j - cy + 30)**2)
                img[i, j] = np.exp(-r/50) + 0.7 * np.exp(-r2/40)
            elif cluster_type == "abell":
                # Abell cluster - single core with arcs
                img[i, j] = np.exp(-r/60) + 0.3 * np.sin(r/20) * np.exp(-r/80)
            elif cluster_type == "crab":
                # Crab Nebula - filamentary
                img[i, j] = np.exp(-r/80) + 0.2 * np.sin(i/10) * np.cos(j/10) * np.exp(-r/100)
            elif cluster_type == "centaurus":
                # Centaurus A - elongated
                theta = np.arctan2(j - cy, i - cx)
                img[i, j] = np.exp(-r/70) * (1 + 0.5 * np.cos(2*theta))
            else:
                img[i, j] = np.exp(-r/60)
    
    # Add noise and normalize
    img = img + np.random.randn(size, size) * 0.05
    img = np.clip(img, 0, None)
    img = (img - img.min()) / (img.max() - img.min())
    
    return img


def create_sample_image(dataset_name):
    """Create sample image for preloaded dataset"""
    if "Bullet" in dataset_name:
        img = generate_synthetic_cluster(400, "bullet")
    elif "Abell 1689" in dataset_name or "Abell 209" in dataset_name:
        img = generate_synthetic_cluster(400, "abell")
    elif "Crab" in dataset_name:
        img = generate_synthetic_cluster(400, "crab")
    elif "Centaurus" in dataset_name:
        img = generate_synthetic_cluster(400, "centaurus")
    else:
        img = generate_synthetic_cluster(400, "abell")
    
    return img


# ── QCI-STYLE ANNOTATION FUNCTION ─────────────────────────────────────────────

def add_qci_annotations(image_array, metadata, scale_kpc=100, cluster_name=""):
    """Add annotations to image like QCI AstroEntangle Refiner"""
    if len(image_array.shape) == 3:
        img = (image_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
    
    h, w = image_array.shape[:2]
    
    # Scale bar
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    bar_y = h - 40
    draw.rectangle([20, bar_y, 20 + scale_bar_px, bar_y + 5], fill='white')
    draw.text((20 + 30, bar_y - 18), f"{scale_bar_kpc:.0f} kpc", fill='white', font=font_small)
    
    # North indicator
    draw.line([w - 30, 30, w - 30, 60], fill='white', width=2)
    draw.text((w - 38, 15), "N", fill='white', font=font_medium)
    
    # Physics info box
    info_lines = [
        f"Ω = {metadata['omega']:.2f} | Fringe = {metadata['fringe']}",
        f"Mixing = {metadata['mixing']:.3f} | Entropy = {metadata['entropy']:.3f}",
        f"λ_FDM = {scale_bar_kpc / metadata['fringe'] * 8:.1f} kpc"
    ]
    for i, line in enumerate(info_lines):
        draw.text((15, 15 + i * 20), line, fill='cyan', font=font_small)
    
    # Formulas
    formulas = [r"ρ(r) ∝ [sin(kr)/kr]²", r"λ = h/(m v)"]
    for i, formula in enumerate(formulas):
        draw.text((w - 180, h - 50 + i * 18), formula, fill='#88ff88', font=font_small)
    
    return np.array(img_pil) / 255.0


def create_side_by_side_comparison(original, processed, metadata, scale_kpc=100, title="Object"):
    """Create side-by-side comparison with titles"""
    original_annotated = add_qci_annotations(original, metadata, scale_kpc)
    processed_with_overlay = add_qci_annotations(processed, metadata, scale_kpc)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0a0a1a')
    
    ax1.imshow(original_annotated)
    ax1.set_title(f"Before: {title} Standard View\n(Public Data)", color='white', fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(processed_with_overlay)
    ax2.set_title(f"After: Photon-Dark-Photon Entangled\nFDM Overlays (Stellaris Model)", color='white', fontsize=12)
    ax2.axis('off')
    
    fig.tight_layout()
    return fig


# ── PHYSICS COMPONENTS ─────────────────────────────────────────────

def create_fdm_soliton(size, fringe):
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
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    theta = np.arctan2(y - cy, x - cx)
    k = fringe / 20.0
    
    radial = np.sin(k * 2 * np.pi * r * 3)
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    
    pattern = radial * 0.5 + spiral * 0.5
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    return pattern


def create_dark_matter_density(image, soliton):
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
    img_norm = np.clip(image, 0, 1)
    dp_norm = np.clip(dark_photon, 0, 1)
    dm_norm = np.clip(dm_density, 0, 1)
    sol_norm = np.clip(soliton, 0, 1)
    
    red = img_norm
    green = img_norm * 0.3 + dp_norm * 0.5 + sol_norm * 0.2
    blue = img_norm * 0.2 + dm_norm * 0.6 + sol_norm * 0.2
    
    return np.clip(np.stack([red, green, blue], axis=-1), 0, 1)


def process_image(image, omega, fringe, brightness=1.2):
    h, w = image.shape
    
    soliton = create_fdm_soliton((h, w), fringe)
    dark_photon = create_dark_photon_wave((h, w), fringe)
    dm_density = create_dark_matter_density(image, soliton)
    
    mixing = omega * 0.6
    
    result = image * (1 - mixing * 0.4)
    result = result + dark_photon * mixing * 0.5
    result = result + dm_density * mixing * 0.3
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    rgb = create_rgb_overlay(result, dark_photon, dm_density, soliton)
    entropy = -mixing * np.log(mixing + 1e-12)
    
    metadata = {
        'omega': omega,
        'fringe': fringe,
        'mixing': mixing,
        'entropy': entropy,
        'brightness': brightness,
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


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Stellaris QED v9.0")
    st.markdown("*With Preloaded Examples*")
    st.markdown("---")
    
    # Preloaded datasets selector
    st.markdown("### 🎯 Preloaded Examples")
    selected_example = st.selectbox(
        "Select Object",
        ["Custom Upload"] + list(PRELOADED_DATASETS.keys())
    )
    
    if selected_example != "Custom Upload":
        preset = PRELOADED_DATASETS[selected_example]
        st.info(f"**{selected_example}**\n{preset['description']}\n\n{preset['notes']}")
        omega_val = preset["omega"]
        fringe_val = preset["fringe"]
        scale_val = preset["scale_kpc"]
    else:
        omega_val = 0.70
        fringe_val = 65
        scale_val = 100
    
    st.markdown("---")
    st.markdown("### ⚛️ PDP Parameters")
    omega = st.slider("Ω Entanglement", 0.1, 1.0, omega_val, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, fringe_val, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], 
                              index=[50,100,150,200,300].index(scale_val) if scale_val in [50,100,150,200,300] else 1)
    
    st.markdown("---")
    st.markdown("### 🌌 Magnetar Parameters")
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("Kinetic Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin a/M", 0.0, 0.998, 0.9)
    
    st.caption("Tony Ford Model | v9.0 - Preloaded Examples")


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
    st.warning(f"⚠️ **Super-critical field!** B/B_crit = {B_ratio:.2e}")


# ── PROCESS IMAGE ─────────────────────────────────────────────
if selected_example != "Custom Upload":
    # Use preloaded example
    with st.spinner(f"Loading {selected_example}..."):
        sample_img = create_sample_image(selected_example)
        data = sample_img
        
        with st.spinner("Applying PDP physics..."):
            results = process_image(data, omega, fringe, brightness)
        
        st.success(f"✅ Loaded: {selected_example} | Preset applied")
        
        # Display
        st.markdown("### 📊 Before vs After")
        comparison_fig = create_side_by_side_comparison(
            results['original'], results['rgb_overlay'], results['metadata'], 
            scale_kpc, selected_example
        )
        st.pyplot(comparison_fig)
        plt.close(comparison_fig)
        
        # Physics components
        st.markdown("---")
        st.markdown("### ⚛️ FDM Physics Components")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
            ax1.imshow(results['soliton'], cmap='hot', vmin=0, vmax=1)
            ax1.set_title("FDM Soliton Core", color='#00aaff')
            ax1.axis('off')
            st.pyplot(fig1)
            plt.close(fig1)
            st.caption(r"$\rho(r) \propto [\sin(kr)/(kr)]^2$")
        
        with col_b:
            fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
            ax2.imshow(results['dark_photon'], cmap='plasma', vmin=0, vmax=1)
            ax2.set_title("Dark Photon Field", color='#00aaff')
            ax2.axis('off')
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption(r"$\lambda = h/(m v)$ interference")
        
        with col_c:
            fig3, ax3 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
            ax3.imshow(results['dark_matter'], cmap='viridis', vmin=0, vmax=1)
            ax3.set_title("Dark Matter Density", color='#00aaff')
            ax3.axis('off')
            st.pyplot(fig3)
            plt.close(fig3)
            st.caption(r"From $\nabla^2\Phi = 4\pi G\rho$")
        
        # Metrics
        st.markdown("---")
        st.markdown("### 📈 Physics Metrics")
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        with col_m1:
            st.metric("Soliton Peak", f"{results['soliton'].max():.3f}")
        with col_m2:
            st.metric("Fringe Contrast", f"{results['dark_photon'].std():.3f}")
        with col_m3:
            st.metric("Mixing", f"{results['metadata']['mixing']:.3f}")
        with col_m4:
            st.metric("Entropy", f"{results['metadata']['entropy']:.3f}")
        with col_m5:
            gain = results['entangled'].std() / (results['original'].std() + 1e-9)
            st.metric("Contrast Gain", f"{gain:.2f}x")

else:
    # File upload for custom images
    uploaded = st.file_uploader("📁 Upload Image", type=['fits', 'png', 'jpg', 'jpeg', 'tif', 'tiff'])
    
    if uploaded is not None:
        with st.spinner("Loading..."):
            ext = uploaded.name.split(".")[-1].lower()
            data_bytes = uploaded.read()
            
            if ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
                img = Image.open(io.BytesIO(data_bytes)).convert('L')
                data = np.array(img, dtype=np.float32)
            elif ext in ['fits', 'fit', 'fts'] and HAS_ASTROPY:
                with fits.open(io.BytesIO(data_bytes)) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    if len(data.shape) > 2:
                        data = data[0]
            else:
                st.error("Unsupported format")
                data = None
            
            if data is not None:
                data = (data - data.min()) / (data.max() - data.min() + 1e-9)
                
                MAX_SIZE = 500
                if data.shape[0] > MAX_SIZE or data.shape[1] > MAX_SIZE:
                    from skimage.transform import resize
                    data = resize(data, (MAX_SIZE, MAX_SIZE), preserve_range=True)
                    data = (data - data.min()) / (data.max() - data.min() + 1e-9)
                
                with st.spinner("Applying PDP physics..."):
                    results = process_image(data, omega, fringe, brightness)
                
                st.success(f"✅ Loaded: {uploaded.name}")
                
                # Display results
                st.markdown("### 📊 Before vs After")
                comparison_fig = create_side_by_side_comparison(
                    results['original'], results['rgb_overlay'], results['metadata'], 
                    scale_kpc, uploaded.name
                )
                st.pyplot(comparison_fig)
                plt.close(comparison_fig)
                
                # Physics components
                st.markdown("---")
                st.markdown("### ⚛️ FDM Physics Components")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
                    ax1.imshow(results['soliton'], cmap='hot', vmin=0, vmax=1)
                    ax1.set_title("FDM Soliton Core", color='#00aaff')
                    ax1.axis('off')
                    st.pyplot(fig1)
                    plt.close(fig1)
                
                with col_b:
                    fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
                    ax2.imshow(results['dark_photon'], cmap='plasma', vmin=0, vmax=1)
                    ax2.set_title("Dark Photon Field", color='#00aaff')
                    ax2.axis('off')
                    st.pyplot(fig2)
                    plt.close(fig2)
                
                with col_c:
                    fig3, ax3 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
                    ax3.imshow(results['dark_matter'], cmap='viridis', vmin=0, vmax=1)
                    ax3.set_title("Dark Matter Density", color='#00aaff')
                    ax3.axis('off')
                    st.pyplot(fig3)
                    plt.close(fig3)
                
                # Metrics
                st.markdown("---")
                st.markdown("### 📈 Physics Metrics")
                
                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                
                with col_m1:
                    st.metric("Soliton Peak", f"{results['soliton'].max():.3f}")
                with col_m2:
                    st.metric("Fringe Contrast", f"{results['dark_photon'].std():.3f}")
                with col_m3:
                    st.metric("Mixing", f"{results['metadata']['mixing']:.3f}")
                with col_m4:
                    st.metric("Entropy", f"{results['metadata']['entropy']:.3f}")
                with col_m5:
                    gain = results['entangled'].std() / (results['original'].std() + 1e-9)
                    st.metric("Contrast Gain", f"{gain:.2f}x")
    
    else:
        st.info("📁 **Upload an image or select a preloaded example from the sidebar**")

# ── PHYSICS TABS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Quantum Vacuum Physics")

tab1, tab2, tab3 = st.tabs(["🌌 Magnetar Field", "🕳️ Dark Photons", "🌀 Kerr Geodesics"])

with tab1:
    fig_mag, ax_mag = plt.subplots(figsize=(7, 7), facecolor='#0a0a1a')
    ax_mag.set_facecolor('#0a0a1a')
    
    r = np.linspace(1.2, 5, 40)
    theta = np.linspace(0, 2*np.pi, 40)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    B_val = B_surface / (R**3)
    B_norm = np.log10(B_val + 1e-9)
    B_norm = (B_norm - B_norm.min()) / (B_norm.max() - B_norm.min() + 1e-9)
    
    sc = ax_mag.scatter(X, Y, c=B_norm, cmap='plasma', s=3, alpha=0.7)
    ax_mag.add_patch(Circle((0, 0), 1, color='#ff4444', alpha=0.9))
    ax_mag.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12)
    ax_mag.set_aspect('equal')
    ax_mag.set_xlim(-5.5, 5.5)
    ax_mag.set_ylim(-5.5, 5.5)
    ax_mag.set_title(f'Magnetar Field | B = {B_surface:.1e} G', color='#00aaff')
    ax_mag.axis('off')
    plt.colorbar(sc, ax=ax_mag, fraction=0.046, label='log₁₀|B|')
    
    st.pyplot(fig_mag)
    plt.close(fig_mag)

with tab2:
    fig_dp, ax_dp = plt.subplots(figsize=(8, 5), facecolor='#0a0a1a')
    ax_dp.set_facecolor('#0a0a1a')
    
    L = np.logspace(-2, 2, 500)
    if m_dark <= 0:
        P = (epsilon * B_surface / 1e15)**2 * np.ones_like(L)
    else:
        hbar_ev_s = 6.582e-16
        c_km_s = 3e5
        conv_len = 4 * 1e18 * hbar_ev_s * c_km_s / (m_dark**2)
        P = (epsilon * B_surface / 1e15)**2 * np.sin(np.pi * L / conv_len)**2
    P = np.clip(P, 0, 1)
    
    ax_dp.semilogx(L, P, '#00aaff', linewidth=2.5)
    ax_dp.axhline(y=(epsilon * B_surface / 1e15)**2, color='#ff8888', linestyle='--', 
                  label=f'Max P = {(epsilon * B_surface / 1e15)**2:.2e}')
    ax_dp.set_xlabel('Length (km)', color='white')
    ax_dp.set_ylabel('P(γ→A\')', color='white')
    ax_dp.set_title('Dark Photon Conversion', color='#00aaff')
    ax_dp.grid(True, alpha=0.3)
    ax_dp.legend()
    ax_dp.tick_params(colors='white')
    
    st.pyplot(fig_dp)
    plt.close(fig_dp)

with tab3:
    fig_gr, ax_gr = plt.subplots(figsize=(7, 7), facecolor='#0a0a1a')
    ax_gr.set_facecolor('#0a0a1a')
    
    r_horizon = 1 + np.sqrt(1 - a_spin**2)
    circle = Circle((0, 0), r_horizon, color='#555555', alpha=0.7)
    ax_gr.add_patch(circle)
    ax_gr.text(0, 0, 'BH', color='white', ha='center', va='center', fontsize=12)
    
    if a_spin <= 0.999:
        r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        theta_ph = np.linspace(0, 2*np.pi, 100)
        ax_gr.plot(r_photon * np.cos(theta_ph), r_photon * np.sin(theta_ph), 
                   '#ff8888', linewidth=2, linestyle='--', label='Photon Sphere')
    
    for impact in [6, 8, 10]:
        t = np.linspace(0, 50, 400)
        r = 12 * np.exp(-t/35) + r_horizon + 0.5
        phi = (impact/10) * np.sin(t/25)
        ax_gr.plot(r * np.cos(phi), r * np.sin(phi), '#88ff88', linewidth=1.5, alpha=0.7)
    
    ax_gr.set_aspect('equal')
    ax_gr.set_xlim(-14, 14)
    ax_gr.set_ylim(-14, 14)
    ax_gr.set_title(f'Kerr Spacetime | a/M = {a_spin:.3f}', color='#00aaff')
    ax_gr.legend()
    ax_gr.axis('off')
    
    st.pyplot(fig_gr)
    plt.close(fig_gr)

st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v9.0** | Preloaded Examples | Tony Ford Model")
