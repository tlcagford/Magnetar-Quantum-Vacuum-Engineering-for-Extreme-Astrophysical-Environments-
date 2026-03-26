"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS) v1.0
Complete Integration of All Four Projects:
1. QCI AstroEntangle Refiner - FDM Solitons & Image Processing
2. Stellaris QED Explorer - Magnetar Physics & Dark Photons
3. Primordial Entanglement - Quantum Mixing & Conversion
4. QCIS - Quantum-Corrected Cosmology & Power Spectra

Author: Tony E. Ford
License: Dual License (Academic/Commercial)
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.constants import c, hbar, e, m_e, alpha, pi
from scipy.ndimage import gaussian_filter, sobel, zoom
from scipy.special import jv, erf
from scipy.integrate import odeint
from scipy.fft import fft2, fftshift
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from PIL import Image, ImageDraw, ImageFont
import warnings
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List
import pandas as pd

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="QCAUS - Unified Quantum Cosmology Suite",
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

# Professional dark-light hybrid theme
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { 
    background: linear-gradient(135deg, #0a0a2a 0%, #1a1a3a 100%);
}
[data-testid="stSidebar"] { 
    background: linear-gradient(135deg, #1a1a3a 0%, #0a0a2a 100%);
    border-right: 2px solid #00ffff;
}
.stTitle, h1, h2, h3 { 
    color: #00ffff;
    font-family: 'Courier New', monospace;
}
[data-testid="stMetricValue"] { color: #00ffff; }
[data-testid="stMetricLabel"] { color: #ffffff; }
.stInfo { background-color: #1e3a5f; border-left: 3px solid #00ffff; }
.stWarning { background-color: #3a2a1e; border-left: 3px solid #ffaa00; }
.stSuccess { background-color: #1e3a2a; border-left: 3px solid #00ffaa; }
.stProgress > div > div > div > div { background-color: #00ffff; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS (All Projects) ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)  # 4.4e13 G
alpha_fine = 1/137.036
lambda_compton = hbar / (m_e * c)
m_eV = m_e * c**2 / 1.602e-19
G_newton = 6.67430e-11
M_sun = 1.989e30
kpc_to_m = 3.086e19


# ── DATA CLASSES ─────────────────────────────────────────────
@dataclass
class QCIOutput:
    """QCI AstroEntangle Refiner Output"""
    entangled_image: np.ndarray
    soliton_core: np.ndarray
    dark_photon_field: np.ndarray
    dark_matter_density: np.ndarray
    rgb_composite: np.ndarray
    mixing_angle: float
    entanglement_entropy: float
    power_spectrum: Tuple[np.ndarray, np.ndarray]
    processing_time: float
    metadata: Dict

@dataclass
class StellarisOutput:
    """Stellaris QED Explorer Output"""
    B_field: np.ndarray
    vacuum_n: np.ndarray
    dark_photon_P: np.ndarray
    pair_rate: float
    conversion_length: float
    geodesic_data: Dict
    metadata: Dict

@dataclass
class PrimordialOutput:
    """Primordial Entanglement Output"""
    mixing_evolution: np.ndarray
    entropy_evolution: np.ndarray
    conversion_matrix: np.ndarray
    final_mixing: float
    final_entropy: float
    metadata: Dict

@dataclass
class QCISOutput:
    """QCIS Framework Output"""
    power_spectrum: np.ndarray
    transfer_function: np.ndarray
    quantum_correction: np.ndarray
    tensor_spectrum: np.ndarray
    metadata: Dict


# ── QCI ASTROENTANGLE REFINER FUNCTIONS ─────────────────────────────────────────────

def qci_normalize(arr):
    """Safe normalization"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    vmin, vmax = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if vmax - vmin < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)


def qci_psf_correct(data, amount=0.6):
    """PSF correction"""
    try:
        kernel = Gaussian2DKernel(x_stddev=1.8)
        psf = kernel.array / kernel.array.sum()
        blurred = convolve(data, psf, boundary='wrap')
        result = np.clip(data + amount * (data - blurred), 0, 1)
        return np.nan_to_num(result)
    except:
        return data


def qci_schrodinger_poisson_soliton(size, fringe):
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
    return gaussian_filter(soliton, sigma=2)


def qci_dark_photon_field(size, fringe, scale_kpc=100):
    """Dark Photon Interference - λ = h/(m v)"""
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


def qci_power_spectrum(field, k_bins=50):
    """Compute power spectrum P(k)"""
    fft_field = fft2(field)
    power = np.abs(fft_field)**2
    power_shifted = fftshift(power)
    
    h, w = field.shape
    cy, cx = h//2, w//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    k_max = min(cx, cy)
    if k_max <= 0:
        return np.array([0]), np.array([0])
    
    k_edges = np.linspace(0, k_max, k_bins + 1)
    k_centers = (k_edges[:-1] + k_edges[1:]) / 2
    
    power_spectrum = []
    for i in range(k_bins):
        mask = (r >= k_edges[i]) & (r < k_edges[i+1])
        if np.any(mask):
            power_spectrum.append(np.mean(power_shifted[mask]))
        else:
            power_spectrum.append(0)
    
    return k_centers, np.array(power_spectrum)


def qci_apply(image, omega, fringe, brightness=1.2, scale_kpc=100):
    """Apply QCI AstroEntangle Refiner physics"""
    start_time = time.time()
    h, w = image.shape
    
    soliton = qci_schrodinger_poisson_soliton((h, w), fringe)
    dark_photon = qci_dark_photon_field((h, w), fringe, scale_kpc)
    
    # Dark matter density from gradients
    smoothed = gaussian_filter(image, sigma=8)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    dm_density = np.sqrt(grad_x**2 + grad_y**2)
    dm_density = (dm_density - dm_density.min()) / (dm_density.max() - dm_density.min() + 1e-9)
    dm_density = soliton * 0.6 + dm_density * 0.4
    
    mixing = omega * 0.5
    result = image * (1 - mixing * 0.3)
    result = result + dark_photon * mixing * 0.5
    result = result + dm_density * mixing * 0.3
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    rgb = np.stack([
        result,
        result * 0.4 + dark_photon * 0.6,
        result * 0.3 + dm_density * 0.7
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    rho = np.array([[1-mixing, mixing], [mixing, mixing]])
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log(eigenvalues)) if len(eigenvalues) > 0 else 0
    
    k_centers, power_spec = qci_power_spectrum(result)
    
    processing_time = time.time() - start_time
    
    metadata = {
        "omega": float(omega),
        "fringe": int(fringe),
        "brightness": float(brightness),
        "scale_kpc": int(scale_kpc),
        "mixing_angle": float(mixing),
        "entanglement_entropy": float(entropy),
        "processing_time": processing_time
    }
    
    return QCIOutput(
        entangled_image=result,
        soliton_core=soliton,
        dark_photon_field=dark_photon,
        dark_matter_density=dm_density,
        rgb_composite=rgb,
        mixing_angle=mixing,
        entanglement_entropy=entropy,
        power_spectrum=(k_centers, power_spec),
        processing_time=processing_time,
        metadata=metadata
    )


# ── STELLARIS QED FUNCTIONS ─────────────────────────────────────────────

def stellaris_dipole_field(B_surface, R_ns, r, theta, inclination=0):
    """Magnetar dipole field"""
    B0 = B_surface * (R_ns / (r + 1e-9))**3
    theta_rad = np.radians(theta + inclination)
    B_r = 2 * B0 * np.cos(theta_rad)
    B_theta = B0 * np.sin(theta_rad)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    return B_r, B_theta, B_mag


def stellaris_dark_photon_conversion(B, L, epsilon, m_dark, omega=1e18):
    """Dark photon conversion probability"""
    if m_dark <= 0:
        return (epsilon * B / 1e15)**2 * np.ones_like(L)
    hbar_ev_s = 6.582e-16
    c_km_s = 3e5
    conversion_length = 4 * omega * hbar_ev_s * c_km_s / (m_dark**2)
    P = (epsilon * B / 1e15)**2 * np.sin(np.pi * L / conversion_length)**2
    return np.clip(P, 0, 1), conversion_length


def stellaris_schwinger_pair(B_field):
    """Schwinger pair production rate"""
    B_norm = B_field / B_crit
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.exp(-np.pi / (B_norm + 1e-9))
    return np.clip(rate, 0, 1)


def stellaris_euler_heisenberg_n(B_over_Bc, polarization='perp'):
    """Euler-Heisenberg vacuum refractive index"""
    x = B_over_Bc**2
    if polarization == 'perp':
        return 1 + 4 * alpha_fine/(45 * np.pi) * x
    else:
        return 1 + 7 * alpha_fine/(45 * np.pi) * x


def stellaris_apply(B_surface, epsilon, m_dark, R_ns=10, inclination=0):
    """Apply Stellaris QED physics"""
    start_time = time.time()
    
    # Grid for field calculation
    r_grid = np.linspace(R_ns, 5*R_ns, 100)
    theta_grid = np.linspace(0, np.pi, 100)
    R, Theta = np.meshgrid(r_grid, theta_grid, indexing='ij')
    
    B_r, B_theta, B_mag = stellaris_dipole_field(B_surface, R_ns, R, Theta, inclination)
    
    # Dark photon conversion
    L_sample = np.logspace(-2, 2, 100)
    P_conv, conv_len = stellaris_dark_photon_conversion(B_surface, L_sample, epsilon, m_dark)
    
    # Schwinger rate
    pair_rate = stellaris_schwinger_pair(B_surface)
    
    # QED vacuum
    B_ratio = B_surface / B_crit
    n_perp = stellaris_euler_heisenberg_n(B_ratio, 'perp')
    n_para = stellaris_euler_heisenberg_n(B_ratio, 'para')
    
    processing_time = time.time() - start_time
    
    metadata = {
        "B_surface": float(B_surface),
        "epsilon": float(epsilon),
        "m_dark": float(m_dark),
        "R_ns": float(R_ns),
        "inclination": float(inclination),
        "B_ratio": float(B_ratio),
        "pair_rate": float(pair_rate),
        "conversion_length_km": float(conv_len),
        "n_perp_minus_1": float(n_perp - 1),
        "n_para_minus_1": float(n_para - 1),
        "processing_time": processing_time
    }
    
    return StellarisOutput(
        B_field=B_mag,
        vacuum_n=np.array([n_perp, n_para]),
        dark_photon_P=P_conv,
        pair_rate=pair_rate,
        conversion_length=conv_len,
        geodesic_data={"r_horizon": 2, "r_photon": 3},
        metadata=metadata
    )


# ── PRIMORDIAL ENTANGLEMENT FUNCTIONS ─────────────────────────────────────────────

def primordial_von_neumann_evolution(omega, m_dark, H=70, t_max=1.0):
    """Solve von Neumann equation for photon-dark photon system"""
    epsilon = omega * 0.1
    t = np.linspace(0, t_max, 200)
    
    # Full von Neumann solution
    def rho_deriv(rho_flat, t, H, epsilon, m_dark):
        rho = rho_flat.reshape(2, 2)
        mixing = epsilon * np.exp(-H * t)
        drho_dt = np.zeros_like(rho, dtype=complex)
        drho_dt[0,0] = 2 * mixing * np.imag(rho[0,1])
        drho_dt[1,1] = -2 * mixing * np.imag(rho[0,1])
        drho_dt[0,1] = -1j * (1.0 - m_dark) * rho[0,1] - 1j * mixing * (rho[0,0] - rho[1,1])
        drho_dt[1,0] = np.conj(drho_dt[0,1])
        return drho_dt.flatten()
    
    rho0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    
    try:
        sol = odeint(rho_deriv, rho0, t, args=(H, epsilon, m_dark))
        mixing_evolution = np.abs(sol[:, 1])
        entropy_evolution = []
        for s in sol:
            rho_mat = s.reshape(2, 2)
            eigvals = np.linalg.eigvalsh(rho_mat)
            eigvals = eigvals[eigvals > 1e-12]
            entropy = -np.sum(eigvals * np.log(eigvals)) if len(eigvals) > 0 else 0
            entropy_evolution.append(entropy)
        entropy_evolution = np.array(entropy_evolution)
    except:
        # Analytic approximation
        mixing_evolution = epsilon * (1 - np.exp(-H * t))
        entropy_evolution = -mixing_evolution * np.log(mixing_evolution + 1e-12)
    
    final_mixing = mixing_evolution[-1]
    final_entropy = entropy_evolution[-1]
    
    # Conversion matrix
    energy_range = np.logspace(0, 4, 50)
    L_range = np.logspace(-2, 2, 50)
    conversion_matrix = np.zeros((len(energy_range), len(L_range)))
    
    for i, E in enumerate(energy_range):
        omega_val = E / (hbar * 1.602e-19)
        for j, length in enumerate(L_range):
            if m_dark > 0:
                conv_len = 4 * omega_val * 6.582e-16 * 3e5 / (m_dark**2)
                P = (epsilon * 1e15 / 1e15)**2 * np.sin(np.pi * length / conv_len)**2
            else:
                P = (epsilon)**2
            conversion_matrix[i, j] = np.clip(P, 0, 1)
    
    metadata = {
        "omega": float(omega),
        "m_dark": float(m_dark),
        "epsilon": float(epsilon),
        "final_mixing": float(final_mixing),
        "final_entropy": float(final_entropy)
    }
    
    return PrimordialOutput(
        mixing_evolution=mixing_evolution,
        entropy_evolution=entropy_evolution,
        conversion_matrix=conversion_matrix,
        final_mixing=final_mixing,
        final_entropy=final_entropy,
        metadata=metadata
    )


# ── QCIS FRAMEWORK FUNCTIONS ─────────────────────────────────────────────

def qcis_power_spectrum(k, A_s=2.1e-9, n_s=0.965, f_nl=1.0, r=0.01):
    """Quantum-corrected power spectrum"""
    k0 = 0.05  # Mpc⁻¹ pivot scale
    P_lcdm = A_s * (k / k0)**(n_s - 1)
    quantum_correction = 1 + f_nl * (k / k0)**0.8
    P_quantum = P_lcdm * quantum_correction
    tensor_spectrum = r * P_lcdm
    return P_lcdm, P_quantum, tensor_spectrum, quantum_correction


def qcis_transfer_function(k, omega_m=0.3, omega_b=0.05, h=0.7):
    """Matter transfer function with quantum corrections"""
    q = k / (omega_m * h**2)
    T_EH = np.log(1 + 2.34*q) / (2.34*q) * (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
    return T_EH


def qcis_apply(omega):
    """Apply QCIS framework"""
    start_time = time.time()
    
    k = np.logspace(-3, 0, 100)
    P_lcdm, P_quantum, P_tensor, q_corr = qcis_power_spectrum(k, f_nl=omega)
    T_k = qcis_transfer_function(k)
    
    processing_time = time.time() - start_time
    
    metadata = {
        "omega": float(omega),
        "processing_time": processing_time
    }
    
    return QCISOutput(
        power_spectrum=(k, P_quantum),
        transfer_function=(k, T_k),
        quantum_correction=(k, q_corr),
        tensor_spectrum=(k, P_tensor),
        metadata=metadata
    )


# ── UNIFIED VISUALIZATION FUNCTIONS ─────────────────────────────────────────────

def fig_to_pil(fig):
    """Convert matplotlib figure to PIL for guaranteed display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=100)
    buf.seek(0)
    return Image.open(buf)


def display_fig(fig, caption=None):
    """Display figure using PIL"""
    st.image(fig_to_pil(fig), caption=caption, use_container_width=True)
    plt.close(fig)


def unified_dashboard_visualization(qci_result, stellaris_result, primordial_result, qcis_result):
    """Create unified dashboard with all four projects"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='#0a0a2a')
    
    # 1. QCI: Soliton Core
    ax1 = axes[0, 0]
    im1 = ax1.imshow(qci_result.soliton_core, cmap='hot', extent=[-5, 5, -5, 5])
    ax1.set_title('FDM Soliton Core\nQCI AstroEntangle Refiner', color='white', fontsize=10)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, label='Density')
    
    # 2. Stellaris: Dark Photon Conversion
    ax2 = axes[0, 1]
    L = np.logspace(-2, 2, 100)
    P, _ = stellaris_dark_photon_conversion(stellaris_result.metadata['B_surface'], L, 
                                             stellaris_result.metadata['epsilon'], 
                                             stellaris_result.metadata['m_dark'])
    ax2.semilogx(L, P, 'b-', linewidth=2)
    ax2.set_xlabel('Length (km)', color='white')
    ax2.set_ylabel('P(γ→A\')', color='white')
    ax2.set_title('Dark Photon Conversion\nStellaris QED', color='white')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')
    ax2.set_ylim(0, 1)
    
    # 3. Primordial: von Neumann Evolution
    ax3 = axes[1, 0]
    t = np.linspace(0, 1, len(primordial_result.mixing_evolution))
    ax3.plot(t, primordial_result.mixing_evolution, 'r-', linewidth=2, label='Mixing')
    ax3.plot(t, primordial_result.entropy_evolution, 'b--', linewidth=2, label='Entropy')
    ax3.set_xlabel('Scale Factor', color='white')
    ax3.set_ylabel('Amplitude', color='white')
    ax3.set_title('Von Neumann Evolution\nPrimordial Entanglement', color='white')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.tick_params(colors='white')
    
    # 4. QCIS: Power Spectrum
    ax4 = axes[1, 1]
    k, P_lcdm = qcis_result.power_spectrum
    k, P_quantum = qcis_result.power_spectrum
    ax4.loglog(k, P_lcdm, 'b-', linewidth=2, label='ΛCDM')
    ax4.loglog(k, P_quantum, 'r-', linewidth=2, label='Quantum-corrected')
    ax4.fill_between(k, P_lcdm, P_quantum, alpha=0.3, color='red')
    ax4.set_xlabel('k (Mpc⁻¹)', color='white')
    ax4.set_ylabel('P(k)', color='white')
    ax4.set_title('Quantum-Corrected Power Spectrum\nQCIS Framework', color='white')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.tick_params(colors='white')
    
    fig.tight_layout()
    return fig


# ── ANNOTATION FUNCTION ─────────────────────────────────────────────

def add_annotations(image_array, metadata, scale_kpc=100):
    """Add physics annotations to image"""
    if len(image_array.shape) == 3:
        img = (image_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_small = ImageFont.load_default()
    
    h, w = image_array.shape[:2]
    
    # Scale bar
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    draw.rectangle([20, h-40, 20+scale_bar_px, h-35], fill='white')
    draw.text((20+30, h-58), f"{scale_bar_kpc:.0f} kpc", fill='white', font=font_small)
    
    # Info box
    info_lines = [
        f"Ω={metadata['omega']:.2f} | Fringe={metadata['fringe']}",
        f"S_entropy={metadata['entanglement_entropy']:.3f}",
        f"Mixing={metadata['mixing_angle']:.3f}"
    ]
    
    for i, line in enumerate(info_lines):
        draw.text((15, 15 + i*18), line, fill='cyan', font=font_small)
    
    return np.array(img_pil) / 255.0


# ── EXPORT FUNCTIONS ─────────────────────────────────────────────

def export_all_results(qci_result, stellaris_result, primordial_result, qcis_result):
    """Export all results as zip-compatible dictionary"""
    exports = {}
    
    # QCI images
    for name, arr in [
        ('entangled', qci_result.entangled_image),
        ('soliton_core', qci_result.soliton_core),
        ('dark_photon', qci_result.dark_photon_field),
        ('dark_matter', qci_result.dark_matter_density),
        ('rgb_composite', qci_result.rgb_composite)
    ]:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(arr, cmap='inferno', vmin=0, vmax=1)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        exports[f'qci_{name}.png'] = buf.getvalue()
    
    # Metadata
    exports['qci_metadata.json'] = json.dumps(qci_result.metadata, indent=2).encode()
    exports['stellaris_metadata.json'] = json.dumps(stellaris_result.metadata, indent=2).encode()
    exports['primordial_metadata.json'] = json.dumps(primordial_result.metadata, indent=2).encode()
    exports['qcis_metadata.json'] = json.dumps(qcis_result.metadata, indent=2).encode()
    
    return exports


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCAUS")
    st.markdown("*Quantum Cosmology & Astrophysics Unified Suite*")
    st.markdown("---")
    
    # Project selector
    st.markdown("### 📁 Active Modules")
    modules = st.multiselect(
        "Enable Projects",
        ["QCI AstroEntangle Refiner", "Stellaris QED", "Primordial Entanglement", "QCIS Framework"],
        default=["QCI AstroEntangle Refiner", "Stellaris QED", "Primordial Entanglement", "QCIS Framework"]
    )
    
    st.markdown("---")
    
    # Unified parameters
    st.markdown("### ⚛️ Unified Parameters")
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, 0.70, 0.05, 
                       help="Coupling strength across all projects")
    fringe = st.slider("Fringe Scale", 20, 120, 65, 5, 
                        help="FDM wavelength parameter")
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    
    st.markdown("---")
    st.markdown("### 🌌 Astrophysical Parameters")
    
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("Kinetic Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200], index=1)
    
    st.markdown("---")
    st.markdown("### 📚 Unified References")
    st.latex(r"\rho_{\text{FDM}} \propto \left[\frac{\sin(kr)}{kr}\right]^2")
    st.latex(r"P_{\gamma\to A'} = \left(\frac{\varepsilon B}{m_{A'}}\right)^2\sin^2\left(\frac{m_{A'}^2 L}{4\omega}\right)")
    st.latex(r"i\partial_t\rho = [H_{\text{eff}}, \rho]")
    st.latex(r"P(k) = P_{\Lambda\text{CDM}}(k) \times \left(1 + f_{\text{NL}}\left(\frac{k}{k_0}\right)^{n_q}\right)")
    
    st.caption("Tony Ford Model | QCAUS v1.0 | Unified Physics Suite")


# ── MAIN APP ─────────────────────────────────────────────
st.title("🔭 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("*Complete Integration: QCI AstroEntangle Refiner + Stellaris QED + Primordial Entanglement + QCIS*")
st.markdown("---")

# Metrics
B_ratio = B_surface / B_crit
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("B / B_crit", f"{B_ratio:.2e}", 
              delta="Supercritical" if B_ratio > 1 else "Subcritical")
with col2:
    max_P = (epsilon * B_surface / 1e15)**2
    st.metric("Max γ→A' P", f"{max_P:.2e}")
with col3:
    st.metric("Dark Photon Mass", f"{m_dark:.1e} eV")
with col4:
    st.metric("Ω Entanglement", f"{omega:.2f}")

if B_ratio > 1:
    st.warning(f"⚠️ **Super-critical field!** B/B_crit = {B_ratio:.2e} | Quantum electrodynamic effects dominate across all modules.")

# Run all physics engines
with st.spinner("Running all physics engines..."):
    # QCI - with synthetic image if no upload
    img_size = 300
    synthetic_image = np.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            synthetic_image[i, j] = np.sin(i/30) * np.cos(j/30) + 0.5 * np.exp(-((i-150)**2 + (j-150)**2)/5000)
    synthetic_image = (synthetic_image - synthetic_image.min()) / (synthetic_image.max() - synthetic_image.min())
    
    qci_result = qci_apply(synthetic_image, omega, fringe, brightness, scale_kpc)
    stellaris_result = stellaris_apply(B_surface, epsilon, m_dark)
    primordial_result = primordial_von_neumann_evolution(omega, m_dark)
    qcis_result = qcis_apply(omega)

# Unified Dashboard
st.markdown("### 📊 Unified Dashboard")
unified_fig = unified_dashboard_visualization(qci_result, stellaris_result, primordial_result, qcis_result)
st.pyplot(unified_fig)
plt.close(unified_fig)

# Individual Project Tabs
tabs = st.tabs(["🔬 QCI Refiner", "⚡ Stellaris QED", "🕳️ Primordial Entanglement", "🌌 QCIS Framework", "📈 Cross-Correlation"])

# Tab 1: QCI AstroEntangle Refiner
with tabs[0]:
    st.header("QCI AstroEntangle Refiner")
    st.markdown("*FDM Soliton Physics & Image Processing*")
    
    uploaded = st.file_uploader("Upload FITS/Image", type=["fits", "png", "jpg", "jpeg"], key="qci_upload")
    
    if uploaded is not None:
        ext = uploaded.name.split(".")[-1].lower()
        data_bytes = uploaded.read()
        try:
            if ext == "fits":
                with fits.open(io.BytesIO(data_bytes)) as h:
                    img = h[0].data.astype(np.float32)
                    if len(img.shape) > 2:
                        img = img[0]
            else:
                img = Image.open(io.BytesIO(data_bytes)).convert("L")
                img = np.array(img, dtype=np.float32)
            
            img = qci_normalize(img)
            MAX_SIZE = 400
            if img.shape[0] > MAX_SIZE or img.shape[1] > MAX_SIZE:
                from skimage.transform import resize
                img = resize(img, (MAX_SIZE, MAX_SIZE), preserve_range=True)
            
            qci_img_result = qci_apply(img, omega, fringe, brightness, scale_kpc)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img, caption="Original", use_container_width=True)
            with col2:
                annotated = add_annotations(qci_img_result.entangled_image, qci_img_result.metadata, scale_kpc)
                st.image(annotated, caption="PDP Entangled", use_container_width=True)
            with col3:
                st.image(qci_img_result.soliton_core, caption="FDM Soliton Core", use_container_width=True)
            
            st.metric("Entanglement Entropy", f"{qci_img_result.entanglement_entropy:.4f}")
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload an image to see FDM soliton effects on real data")
        st.image(qci_result.soliton_core, caption="Synthetic Soliton Core (Demo)", use_container_width=True)

# Tab 2: Stellaris QED
with tabs[1]:
    st.header("Stellaris QED Explorer")
    st.markdown("*Magnetar Physics & Quantum Vacuum*")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Schwinger Pair Rate", f"{stellaris_result.pair_rate:.2e}")
        st.metric("Conversion Length", f"{stellaris_result.conversion_length:.2e} km")
        st.metric("n_⟂ - 1", f"{stellaris_result.metadata['n_perp_minus_1']:.2e}")
    
    with col2:
        L_plot = np.logspace(-2, 2, 100)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0a0a2a')
        ax.semilogx(L_plot, stellaris_result.dark_photon_P, 'b-', linewidth=2)
        ax.set_xlabel('Length (km)', color='white')
        ax.set_ylabel('P(γ→A\')', color='white')
        ax.set_title('Dark Photon Conversion', color='white')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close(fig)
    
    # Field visualization
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0a2a')
    ax.set_facecolor('#1a1a3a')
    r = np.linspace(10, 50, 40)
    theta = np.linspace(0, 2*np.pi, 40)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    ax.plot(X, Y, 'b-', alpha=0.2, linewidth=0.5)
    ax.add_patch(Circle((0, 0), 10, color='#d62728', alpha=0.8))
    ax.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_title('Magnetar Field Lines', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)
    plt.close(fig)

# Tab 3: Primordial Entanglement
with tabs[2]:
    st.header("Primordial Photon-DarkPhoton Entanglement")
    st.markdown("*Von Neumann Evolution & Quantum Mixing*")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0a0a2a')
        t = np.linspace(0, 1, len(primordial_result.mixing_evolution))
        ax.plot(t, primordial_result.mixing_evolution, 'r-', linewidth=2)
        ax.set_xlabel('Scale Factor', color='white')
        ax.set_ylabel('Mixing Amplitude', color='white')
        ax.set_title('Mixing Evolution', color='white')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close(fig)
        st.metric("Final Mixing", f"{primordial_result.final_mixing:.4f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0a0a2a')
        ax.plot(t, primordial_result.entropy_evolution, 'b-', linewidth=2)
        ax.set_xlabel('Scale Factor', color='white')
        ax.set_ylabel('Entanglement Entropy', color='white')
        ax.set_title('Entropy Evolution', color='white')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close(fig)
        st.metric("Final Entropy", f"{primordial_result.final_entropy:.4f}")
    
    st.latex(r"\mathcal{L}_{\text{mix}} = \frac{\varepsilon}{2} F_{\mu\nu} F'^{\mu\nu}")
    st.latex(r"i\partial_t\rho = [H_{\text{eff}}, \rho]")

# Tab 4: QCIS Framework
with tabs[3]:
    st.header("Quantum Cosmology Integration Suite")
    st.markdown("*Quantum-Corrected Power Spectra & Transfer Functions*")
    
    k, P_quantum = qcis_result.power_spectrum
    k, P_lcdm = qcis_result.power_spectrum
    k, P_tensor = qcis_result.tensor_spectrum
    k, T_k = qcis_result.transfer_function
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0a0a2a')
        ax.loglog(k, P_lcdm, 'b-', linewidth=2, label='ΛCDM')
        ax.loglog(k, P_quantum, 'r-', linewidth=2, label='Quantum-corrected')
        ax.fill_between(k, P_lcdm, P_quantum, alpha=0.3, color='red')
        ax.set_xlabel('k (Mpc⁻¹)', color='white')
        ax.set_ylabel('P(k)', color='white')
        ax.set_title('Matter Power Spectrum', color='white')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0a0a2a')
        ax.loglog(k, P_tensor, 'g-', linewidth=2, label='Tensor Modes')
        ax.set_xlabel('k (Mpc⁻¹)', color='white')
        ax.set_ylabel('P_t(k)', color='white')
        ax.set_title('Tensor Power Spectrum', color='white')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close(fig)
    
    st.metric("Tensor-to-Scalar Ratio", "r < 0.036 (Planck 2018)", delta="Quantum corrections constrain r")

# Tab 5: Cross-Correlation
with tabs[4]:
    st.header("Cross-Correlation Analysis")
    st.markdown("*Connecting all four frameworks*")
    
    col1, col2 = st.columns(2)
    with col1:
        # Mixing vs QCIS correlation
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0a0a2a')
        ax.plot(primordial_result.mixing_evolution, qcis_result.quantum_correction[1][:len(primordial_result.mixing_evolution)], 
                'ro', markersize=3, alpha=0.5)
        ax.set_xlabel('Primordial Mixing', color='white')
        ax.set_ylabel('QCIS Quantum Correction', color='white')
        ax.set_title('Mixing vs Quantum Corrections', color='white')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.info("""
        **Cross-Correlation Insights**
        
        - **Primordial Mixing ↔ QCIS**: Entanglement entropy correlates with quantum corrections to power spectrum
        - **Dark Photons ↔ FDM**: Conversion probability related to soliton core size
        - **Magnetar Field ↔ All**: B field strength scales all quantum effects
        
        **Unified Physics Picture**:
        Quantum vacuum fluctuations → Mixing evolution → Dark photon production → Modified structure formation
        """)
    
    # Unified equation
    st.markdown("---")
    st.latex(r"\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{QED}} + \frac{\varepsilon}{2}F_{\mu\nu}F'^{\mu\nu} + \mathcal{L}_{\text{FDM}} + \mathcal{L}_{\text{QCIS}}}")

# Export
st.markdown("---")
if st.button("📦 Export All Results (ZIP)", use_container_width=True):
    with st.spinner("Preparing export..."):
        exports = export_all_results(qci_result, stellaris_result, primordial_result, qcis_result)
        
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, data in exports.items():
                zip_file.writestr(filename, data)
        
        st.download_button(
            label="Download ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"qcaus_results_{time.strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("🔭 **QCAUS v1.0** | Unified Quantum Cosmology & Astrophysics Suite | Tony Ford Model")
st.markdown("*Four Projects, One Framework: QCI + Stellaris + Primordial + QCIS*")
