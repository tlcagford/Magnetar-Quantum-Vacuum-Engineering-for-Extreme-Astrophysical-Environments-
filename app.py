"""
Magnetar QED Explorer v2.1 – Fixed NameError
Annotated side-by-side with all required functions
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
    page_title="Magnetar QED Explorer v2.1",
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
    .stDownloadButton button { background-color: #00aaff; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)
alpha_fine = 1/137.036


# ── PRELOADED DATASETS ─────────────────────────────────────────────
PRELOADED = {
    "🌌 Bullet Cluster": {"fringe": 70, "omega": 0.75, "pattern": "bullet", "desc": "Merging cluster - dark matter separation"},
    "🔭 Abell 1689": {"fringe": 55, "omega": 0.65, "pattern": "abell", "desc": "Strong lensing - prominent soliton"},
    "✨ Abell 209": {"fringe": 60, "omega": 0.70, "pattern": "abell", "desc": "Balanced fringe visibility"},
    "🦀 Crab Nebula": {"fringe": 50, "omega": 0.68, "pattern": "crab", "desc": "Supernova remnant - filaments"},
    "📡 Centaurus A": {"fringe": 45, "omega": 0.62, "pattern": "centaurus", "desc": "Radio galaxy - jet structure"}
}


def generate_sample(size=400, pattern="abell"):
    """Generate synthetic image"""
    img = np.zeros((size, size))
    cx, cy = size//2, size//2
    
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - cx)**2 + (j - cy)**2)
            if pattern == "bullet":
                r2 = np.sqrt((i - cx - 50)**2 + (j - cy + 30)**2)
                img[i, j] = np.exp(-r/50) + 0.7 * np.exp(-r2/40)
            elif pattern == "abell":
                img[i, j] = np.exp(-r/60) + 0.2 * np.sin(r/25) * np.exp(-r/80)
            elif pattern == "crab":
                img[i, j] = np.exp(-r/80) + 0.15 * np.sin(i/15) * np.cos(j/15) * np.exp(-r/100)
            elif pattern == "centaurus":
                theta = np.arctan2(j - cy, i - cx)
                img[i, j] = np.exp(-r/70) * (1 + 0.4 * np.cos(2*theta))
            else:
                img[i, j] = np.exp(-r/60)
    
    img = img + np.random.randn(size, size) * 0.02
    img = np.clip(img, 0, None)
    img = (img - img.min()) / (img.max() - img.min())
    return img


def create_soliton(size, fringe):
    """FDM Soliton Core - [sin(kr)/kr]²"""
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


def create_wave(size, fringe):
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


def process_image(image, omega, fringe, brightness=1.2):
    """Full PDP processing"""
    h, w = image.shape
    
    soliton = create_soliton((h, w), fringe)
    wave = create_wave((h, w), fringe)
    
    mixing = omega * 0.6
    
    result = image * (1 - mixing * 0.4)
    result = result + wave * mixing * 0.5
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    rgb = np.stack([
        result,
        result * 0.3 + wave * 0.5 + soliton * 0.2,
        result * 0.2 + soliton * 0.8
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    entropy = -mixing * np.log(mixing + 1e-12)
    
    return {
        'original': image,
        'entangled': result,
        'soliton': soliton,
        'wave': wave,
        'rgb': rgb,
        'mixing': mixing,
        'entropy': entropy,
        'metadata': {
            'omega': omega,
            'fringe': fringe,
            'mixing': mixing,
            'entropy': entropy,
            'brightness': brightness,
        }
    }


def array_to_pil(arr):
    """Convert numpy array to PIL Image"""
    return Image.fromarray((arr * 255).astype(np.uint8))


def add_annotations(image_array, metadata, scale_kpc=100, title_prefix="Before"):
    """Add QCI-style annotations"""
    if len(image_array.shape) == 3:
        img = (image_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    h, w = image_array.shape[:2]
    
    # Scale bar
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    bar_y = h - 40
    draw.rectangle([20, bar_y, 20 + scale_bar_px, bar_y + 5], fill='white')
    draw.text((20 + 30, bar_y - 18), f"{scale_bar_kpc:.0f} kpc", fill='white', font=font_tiny)
    
    # North indicator
    draw.line([w - 30, 30, w - 30, 60], fill='white', width=2)
    draw.text((w - 38, 15), "N", fill='white', font=font_small)
    
    # Physics info box
    info_lines = [
        f"Ω = {metadata['omega']:.2f} | Fringe = {metadata['fringe']}",
        f"Mixing = {metadata['mixing']:.3f} | Entropy = {metadata['entropy']:.3f}",
        f"λ_FDM = {scale_bar_kpc / metadata['fringe'] * 8:.1f} kpc"
    ]
    
    box_width = 240
    box_height = len(info_lines) * 22 + 10
    draw.rectangle([10, 10, 10 + box_width, 10 + box_height], fill=(0, 0, 0, 180), outline='white')
    
    for i, line in enumerate(info_lines):
        draw.text((15, 15 + i * 22), line, fill='cyan', font=font_tiny)
    
    # Formulas
    formulas = [
        r"ρ(r) ∝ [sin(kr)/kr]²",
        r"λ = h/(m v)",
        r"P(γ→A') = (εB/m')² sin²(m'²L/4ω)"
    ]
    
    formula_y_start = h - 80
    for i, formula in enumerate(formulas):
        draw.text((w - 210, formula_y_start + i * 18), formula, fill='#88ff88', font=font_tiny)
    
    # Title overlay
    if title_prefix == "Before":
        title_text = "Before: Standard View\n(Public HST/JWST Data)"
    else:
        title_text = "After: Photon-Dark-Photon Entangled\nFDM Overlays (Tony Ford Model)"
    
    # Title background
    title_bbox = draw.textbbox((0, 0), title_text, font=font_small)
    title_width = title_bbox[2] - title_bbox[0]
    draw.rectangle([w//2 - title_width//2 - 10, 10, w//2 + title_width//2 + 10, 50], 
                   fill=(0, 0, 0, 180), outline='white')
    draw.text((w//2 - title_width//2, 15), title_text, fill='white', font=font_small, align='center')
    
    return np.array(img_pil) / 255.0


def create_annotated_side_by_side(original, processed, metadata, scale_kpc=100):
    """Create side-by-side with annotations"""
    original_annotated = add_annotations(original, metadata, scale_kpc, "Before")
    processed_annotated = add_annotations(processed, metadata, scale_kpc, "After")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0a0a1a')
    
    ax1.imshow(original_annotated)
    ax1.axis('off')
    
    ax2.imshow(processed_annotated)
    ax2.axis('off')
    
    fig.tight_layout()
    return fig


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Magnetar QED")
    st.markdown("*Quantum Vacuum Physics*")
    st.markdown("---")
    
    data_source = st.radio("📁 Data", ["🌌 Preloaded", "📤 Upload"])
    
    if data_source == "🌌 Preloaded":
        selected = st.selectbox("Select Object", list(PRELOADED.keys()))
        preset = PRELOADED[selected]
        st.info(preset['desc'])
        use_preload = True
        omega_default = preset["omega"]
        fringe_default = preset["fringe"]
    else:
        use_preload = False
        omega_default = 0.70
        fringe_default = 65
        uploaded = st.file_uploader("Drop Image", type=['png', 'jpg', 'jpeg'])
    
    st.markdown("---")
    st.markdown("### ⚛️ Parameters")
    omega = st.slider("Ω Entanglement", 0.1, 1.0, omega_default, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, fringe_default, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], index=1)
    
    st.markdown("---")
    st.markdown("### 🌌 Magnetar")
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("ε Mixing", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin", 0.0, 0.998, 0.9)
    
    st.caption("Tony Ford | Magnetar QED v2.1")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Magnetar QED Explorer")
st.markdown("*Quantum Vacuum Engineering for Extreme Astrophysics*")
st.markdown("---")

# Metrics
B_ratio = B_surface / B_crit
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("B / B_crit", f"{B_ratio:.2e}")
with col2:
    st.metric("Max γ→A'", f"{(epsilon * B_surface / 1e15)**2:.2e}")
with col3:
    st.metric("Dark Photon Mass", f"{m_dark:.1e} eV")
with col4:
    st.metric("Ω Entanglement", f"{omega:.2f}")

if B_ratio > 1:
    st.warning(f"⚠️ Super-critical field! B/B_crit = {B_ratio:.2e} | QED dominates.")


# ── PROCESS ─────────────────────────────────────────────
results = None

if use_preload:
    with st.spinner(f"Loading {selected}..."):
        pattern = PRELOADED[selected]["pattern"]
        img_data = generate_sample(400, pattern)
        results = process_image(img_data, omega, fringe, brightness)
        st.success(f"✅ Loaded: {selected}")

elif data_source == "📤 Upload" and uploaded is not None:
    with st.spinner(f"Processing..."):
        img = Image.open(uploaded).convert('L')
        img_data = np.array(img, dtype=np.float32) / 255.0
        if img_data.shape[0] > 500:
            from skimage.transform import resize
            img_data = resize(img_data, (500, 500), preserve_range=True)
        results = process_image(img_data, omega, fringe, brightness)
        st.success(f"✅ Loaded: {uploaded.name}")

else:
    st.info("📁 **Select a preloaded object or upload an image**")


# ── DISPLAY ANNOTATED SIDE-BY-SIDE ─────────────────────────────────────────────
if results is not None:
    # Update metadata
    results['metadata'].update({
        'omega': omega,
        'fringe': fringe,
        'mixing': results['mixing'],
        'entropy': results['entropy'],
        'brightness': brightness,
    })
    
    # Annotated comparison
    st.markdown("### 📊 Annotated Comparison")
    comparison_fig = create_annotated_side_by_side(
        results['original'],
        results['rgb'],
        results['metadata'],
        scale_kpc
    )
    st.pyplot(comparison_fig)
    plt.close(comparison_fig)
    
    # Quantum components
    st.markdown("---")
    st.markdown("### ⚛️ Quantum Components")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.image(array_to_pil(results['soliton']), caption="FDM Soliton Core", use_container_width=True)
        st.caption(f"Peak: {results['soliton'].max():.3f}")
    with col_b:
        st.image(array_to_pil(results['wave']), caption="Dark Photon Field", use_container_width=True)
        st.caption(f"Contrast: {results['wave'].std():.3f}")
    with col_c:
        st.image(array_to_pil(results['entangled']), caption="PDP Entangled", use_container_width=True)
        st.caption(f"Mixing: {results['mixing']:.3f} | Entropy: {results['entropy']:.3f}")
    
    # Download buttons
    st.markdown("---")
    st.markdown("### 💾 Download Results")
    
    def save_fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        buf.seek(0)
        return buf.getvalue()
    
    def save_array_png(arr, cmap=None):
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
        st.download_button("📸 Annotated Comparison", save_fig_to_bytes(comparison_fig), "annotated_comparison.png", use_container_width=True)
    with col_d2:
        st.download_button("⭐ Soliton Core", save_array_png(results['soliton'], 'hot'), "soliton.png", use_container_width=True)
    with col_d3:
        st.download_button("🌊 Dark Photon", save_array_png(results['wave'], 'plasma'), "dark_photon.png", use_container_width=True)
    with col_d4:
        st.download_button("⚡ Entangled", save_array_png(results['entangled'], 'inferno'), "entangled.png", use_container_width=True)


# ── MAGNETAR PHYSICS TABS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Magnetar Physics")

tab1, tab2, tab3 = st.tabs(["🌌 Magnetic Field", "🕳️ Dark Photons", "🌀 Kerr Spacetime"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(8, 7), facecolor='#0a0a1a')
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
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight', facecolor='black')
    buf1.seek(0)
    st.download_button("📥 Download Field Plot", buf1, "magnetar_field.png", use_container_width=True)
    plt.close(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0a0a1a')
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
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight', facecolor='black')
    buf2.seek(0)
    st.download_button("📥 Download Conversion Plot", buf2, "dark_photon_conversion.png", use_container_width=True)
    plt.close(fig2)
    st.caption(f"ε = {epsilon:.1e} | m' = {m_dark:.1e} eV | B = {B_surface:.1e} G")

with tab3:
    fig3, ax3 = plt.subplots(figsize=(8, 7), facecolor='#0a0a1a')
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
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', bbox_inches='tight', facecolor='black')
    buf3.seek(0)
    st.download_button("📥 Download Geodesic Plot", buf3, "kerr_geodesic.png", use_container_width=True)
    plt.close(fig3)
    st.caption(f"Event Horizon: r_+ = {r_horizon:.3f} M")

st.markdown("---")
st.markdown("⚡ **Magnetar QED Explorer v2.1** | Annotated Side-by-Side | Tony Ford Model")
