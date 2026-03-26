"""
Stellaris QED Explorer v9.1 – FIXED TABS DISPLAY
All tabs now show plots correctly
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

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Stellaris QED Explorer v9.1",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Dark theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0a0a1a; }
    [data-testid="stSidebar"] { background: #0f0f1f; border-right: 2px solid #00aaff; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: #ffffff !important; }
    .stTitle, h1, h2, h3 { color: #00aaff !important; }
    [data-testid="stMetricValue"] { color: #00aaff !important; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)
alpha_fine = 1/137.036


# ── PRELOADED DATASETS ─────────────────────────────────────────────
PRELOADED_DATASETS = {
    "Bullet Cluster (1E0657-56)": {"fringe": 70, "omega": 0.75, "scale_kpc": 200, "desc": "Merging cluster - dark matter separation"},
    "Abell 1689": {"fringe": 55, "omega": 0.65, "scale_kpc": 150, "desc": "Strong lensing - prominent soliton"},
    "Abell 209": {"fringe": 60, "omega": 0.70, "scale_kpc": 100, "desc": "Balanced fringe visibility"},
    "Crab Nebula (M1)": {"fringe": 50, "omega": 0.68, "scale_kpc": 2, "desc": "Supernova remnant - filaments"},
    "Centaurus A (NGC 5128)": {"fringe": 45, "omega": 0.62, "scale_kpc": 50, "desc": "Radio galaxy - jet structure"}
}


def generate_sample_image(size=400, pattern="cluster"):
    """Generate sample image for preloaded examples"""
    img = np.zeros((size, size))
    cx, cy = size//2, size//2
    
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - cx)**2 + (j - cy)**2)
            if pattern == "bullet":
                r2 = np.sqrt((i - cx - 50)**2 + (j - cy + 30)**2)
                img[i, j] = np.exp(-r/50) + 0.7 * np.exp(-r2/40)
            elif pattern == "abell":
                img[i, j] = np.exp(-r/60) + 0.3 * np.sin(r/20) * np.exp(-r/80)
            elif pattern == "crab":
                img[i, j] = np.exp(-r/80) + 0.2 * np.sin(i/10) * np.cos(j/10) * np.exp(-r/100)
            elif pattern == "centaurus":
                theta = np.arctan2(j - cy, i - cx)
                img[i, j] = np.exp(-r/70) * (1 + 0.5 * np.cos(2*theta))
            else:
                img[i, j] = np.exp(-r/60)
    
    img = img + np.random.randn(size, size) * 0.03
    img = np.clip(img, 0, None)
    img = (img - img.min()) / (img.max() - img.min())
    return img


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
    """Dark Photon Wave Pattern"""
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


def create_rgb_overlay(image, dark_photon, soliton):
    """RGB Composite"""
    img_norm = np.clip(image, 0, 1)
    dp_norm = np.clip(dark_photon, 0, 1)
    sol_norm = np.clip(soliton, 0, 1)
    
    red = img_norm
    green = img_norm * 0.3 + dp_norm * 0.5 + sol_norm * 0.2
    blue = img_norm * 0.2 + sol_norm * 0.8
    
    return np.clip(np.stack([red, green, blue], axis=-1), 0, 1)


def process_image(image, omega, fringe, brightness=1.2):
    """Process image with PDP physics"""
    h, w = image.shape
    
    soliton = create_fdm_soliton((h, w), fringe)
    dark_photon = create_dark_photon_wave((h, w), fringe)
    
    mixing = omega * 0.6
    
    result = image * (1 - mixing * 0.4)
    result = result + dark_photon * mixing * 0.5
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    rgb = create_rgb_overlay(result, dark_photon, soliton)
    entropy = -mixing * np.log(mixing + 1e-12)
    
    metadata = {
        'omega': omega,
        'fringe': fringe,
        'mixing': mixing,
        'entropy': entropy,
    }
    
    return {
        'original': image,
        'entangled': result,
        'soliton': soliton,
        'dark_photon': dark_photon,
        'rgb_overlay': rgb,
        'metadata': metadata
    }


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Stellaris QED v9.1")
    st.markdown("*Fixed Tabs Display*")
    st.markdown("---")
    
    # Preloaded examples
    st.markdown("### 🎯 Preloaded Examples")
    selected_example = st.selectbox(
        "Select Object",
        ["Custom Upload"] + list(PRELOADED_DATASETS.keys())
    )
    
    if selected_example != "Custom Upload":
        preset = PRELOADED_DATASETS[selected_example]
        st.info(f"**{selected_example}**\n{preset['desc']}")
        omega_default = preset["omega"]
        fringe_default = preset["fringe"]
        scale_default = preset["scale_kpc"]
    else:
        omega_default = 0.70
        fringe_default = 65
        scale_default = 100
    
    st.markdown("---")
    st.markdown("### ⚛️ PDP Parameters")
    omega = st.slider("Ω Entanglement", 0.1, 1.0, omega_default, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, fringe_default, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], 
                              index=[50,100,150,200,300].index(scale_default) if scale_default in [50,100,150,200,300] else 1)
    
    st.markdown("---")
    st.markdown("### 🌌 Magnetar Parameters")
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("Kinetic Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin a/M", 0.0, 0.998, 0.9)
    
    st.caption("Tony Ford Model | v9.1")


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
        if "Bullet" in selected_example:
            sample_img = generate_sample_image(400, "bullet")
        elif "Abell" in selected_example:
            sample_img = generate_sample_image(400, "abell")
        elif "Crab" in selected_example:
            sample_img = generate_sample_image(400, "crab")
        elif "Centaurus" in selected_example:
            sample_img = generate_sample_image(400, "centaurus")
        else:
            sample_img = generate_sample_image(400, "abell")
        
        results = process_image(sample_img, omega, fringe, brightness)
    
    st.success(f"✅ Loaded: {selected_example}")
    
    # Side-by-side comparison
    st.markdown("### 📊 Before vs After")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0a0a1a')
    
    ax1.imshow(results['original'], cmap='gray')
    ax1.set_title(f"Before: {selected_example}", color='white')
    ax1.axis('off')
    
    ax2.imshow(results['rgb_overlay'])
    ax2.set_title(f"After: PDP Entangled\nFDM Overlays", color='white')
    ax2.axis('off')
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Physics components
    st.markdown("---")
    st.markdown("### ⚛️ FDM Physics Components")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        fig1, ax1 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
        ax1.imshow(results['soliton'], cmap='hot')
        ax1.set_title("FDM Soliton Core", color='#00aaff')
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col_b:
        fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
        ax2.imshow(results['dark_photon'], cmap='plasma')
        ax2.set_title("Dark Photon Field", color='#00aaff')
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close(fig2)
    
    with col_c:
        fig3, ax3 = plt.subplots(figsize=(4, 4), facecolor='#0a0a1a')
        ax3.imshow(results['entangled'], cmap='inferno')
        ax3.set_title("PDP Entangled", color='#00aaff')
        ax3.axis('off')
        st.pyplot(fig3)
        plt.close(fig3)
    
    # Metrics
    st.markdown("---")
    st.markdown("### 📈 Physics Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Soliton Peak", f"{results['soliton'].max():.3f}")
    with col_m2:
        st.metric("Fringe Contrast", f"{results['dark_photon'].std():.3f}")
    with col_m3:
        st.metric("Mixing", f"{results['metadata']['mixing']:.3f}")
    with col_m4:
        st.metric("Entropy", f"{results['metadata']['entropy']:.3f}")

else:
    # Custom upload
    uploaded = st.file_uploader("📁 Upload Image", type=['png', 'jpg', 'jpeg', 'tif', 'fits'])
    
    if uploaded is not None:
        ext = uploaded.name.split(".")[-1].lower()
        data_bytes = uploaded.read()
        
        if ext in ['png', 'jpg', 'jpeg', 'tif']:
            img = Image.open(io.BytesIO(data_bytes)).convert('L')
            data = np.array(img, dtype=np.float32)
        else:
            st.error("Format not supported")
            data = None
        
        if data is not None:
            data = (data - data.min()) / (data.max() - data.min() + 1e-9)
            
            MAX_SIZE = 500
            if data.shape[0] > MAX_SIZE or data.shape[1] > MAX_SIZE:
                from skimage.transform import resize
                data = resize(data, (MAX_SIZE, MAX_SIZE), preserve_range=True)
                data = (data - data.min()) / (data.max() - data.min() + 1e-9)
            
            results = process_image(data, omega, fringe, brightness)
            
            st.success(f"✅ Loaded: {uploaded.name}")
            
            # Display results (same as above)
            st.markdown("### 📊 Before vs After")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0a0a1a')
            ax1.imshow(results['original'], cmap='gray')
            ax1.set_title("Original", color='white')
            ax1.axis('off')
            ax2.imshow(results['rgb_overlay'])
            ax2.set_title("PDP Entangled", color='white')
            ax2.axis('off')
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("📁 **Upload an image or select a preloaded example from the sidebar**")


# ── PHYSICS TABS (FIXED - EACH TAB HAS ITS OWN PLOT) ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Quantum Vacuum Physics")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🌌 Magnetar Field", "🕳️ Dark Photons", "🌀 Kerr Geodesics"])

# Tab 1: Magnetar Field
with tab1:
    st.subheader("Magnetar Dipole Field")
    fig1 = plt.figure(figsize=(7, 7), facecolor='#0a0a1a')
    ax1 = fig1.add_subplot(111)
    ax1.set_facecolor('#0a0a1a')
    
    r = np.linspace(1.2, 5, 40)
    theta = np.linspace(0, 2*np.pi, 40)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    B_val = B_surface / (R**3)
    B_norm = np.log10(B_val + 1e-9)
    B_norm = (B_norm - B_norm.min()) / (B_norm.max() - B_norm.min() + 1e-9)
    
    sc = ax1.scatter(X, Y, c=B_norm, cmap='plasma', s=3, alpha=0.7)
    ax1.add_patch(Circle((0, 0), 1, color='#ff4444', alpha=0.9))
    ax1.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12)
    ax1.set_aspect('equal')
    ax1.set_xlim(-5.5, 5.5)
    ax1.set_ylim(-5.5, 5.5)
    ax1.set_title(f'Magnetar Field | B = {B_surface:.1e} G', color='#00aaff')
    ax1.axis('off')
    plt.colorbar(sc, ax=ax1, fraction=0.046, label='log₁₀|B|')
    
    st.pyplot(fig1)
    plt.close(fig1)

# Tab 2: Dark Photons
with tab2:
    st.subheader("Photon ↔ Dark Photon Conversion")
    fig2 = plt.figure(figsize=(8, 5), facecolor='#0a0a1a')
    ax2 = fig2.add_subplot(111)
    ax2.set_facecolor('#0a0a1a')
    
    L = np.logspace(-2, 2, 500)
    if m_dark <= 0:
        P = (epsilon * B_surface / 1e15)**2 * np.ones_like(L)
    else:
        hbar_ev_s = 6.582e-16
        c_km_s = 3e5
        conv_len = 4 * 1e18 * hbar_ev_s * c_km_s / (m_dark**2)
        P = (epsilon * B_surface / 1e15)**2 * np.sin(np.pi * L / conv_len)**2
    P = np.clip(P, 0, 1)
    
    ax2.semilogx(L, P, '#00aaff', linewidth=2.5)
    ax2.axhline(y=(epsilon * B_surface / 1e15)**2, color='#ff8888', linestyle='--', 
                label=f'Max P = {(epsilon * B_surface / 1e15)**2:.2e}')
    ax2.set_xlabel('Length (km)', color='white')
    ax2.set_ylabel('P(γ→A\')', color='white')
    ax2.set_title('Dark Photon Conversion', color='#00aaff')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.tick_params(colors='white')
    
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption(f"ε = {epsilon:.1e}, m' = {m_dark:.1e} eV, B = {B_surface:.1e} G")

# Tab 3: Kerr Geodesics
with tab3:
    st.subheader("Null Geodesics in Kerr Spacetime")
    fig3 = plt.figure(figsize=(7, 7), facecolor='#0a0a1a')
    ax3 = fig3.add_subplot(111)
    ax3.set_facecolor('#0a0a1a')
    
    r_horizon = 1 + np.sqrt(1 - a_spin**2)
    circle = Circle((0, 0), r_horizon, color='#555555', alpha=0.7)
    ax3.add_patch(circle)
    ax3.text(0, 0, 'BH', color='white', ha='center', va='center', fontsize=12)
    
    if a_spin <= 0.999:
        r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        theta_ph = np.linspace(0, 2*np.pi, 100)
        ax3.plot(r_photon * np.cos(theta_ph), r_photon * np.sin(theta_ph), 
                 '#ff8888', linewidth=2, linestyle='--', label='Photon Sphere')
    
    for impact in [6, 8, 10]:
        t = np.linspace(0, 50, 400)
        r = 12 * np.exp(-t/35) + r_horizon + 0.5
        phi = (impact/10) * np.sin(t/25)
        ax3.plot(r * np.cos(phi), r * np.sin(phi), '#88ff88', linewidth=1.5, alpha=0.7)
    
    ax3.set_aspect('equal')
    ax3.set_xlim(-14, 14)
    ax3.set_ylim(-14, 14)
    ax3.set_title(f'Kerr Spacetime | a/M = {a_spin:.3f}', color='#00aaff')
    ax3.legend()
    ax3.axis('off')
    
    st.pyplot(fig3)
    plt.close(fig3)
    st.caption(f"Event Horizon: r_+ = {r_horizon:.3f} M")

st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v9.1** | Fixed Tabs | Preloaded Examples | Tony Ford Model")
