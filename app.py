"""
Stellaris QED Explorer v6.0 – Complete with Optical Conversion Chart
Full integration: Photon-Dark Photon Conversion, Magnetar QED, QCIS
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.constants import c, hbar, e, m_e, alpha, pi
from scipy.special import jv
from PIL import Image
import warnings
import time
import json

warnings.filterwarnings('ignore')

# ── PAGE CONFIG – LIGHT THEME ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Stellaris QED Explorer v6.0",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e0e4e8; }
.stTitle, h1, h2, h3 { color: #1e3a5f; }
[data-testid="stMetricValue"] { color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)  # 4.4e13 G
alpha_fine = 1/137.036


# ── PHOTON-DARK PHOTON CONVERSION OPTICAL CHART (FROM YOUR PROJECT) ─────────────────────────────────────────────

def photon_dark_photon_conversion_matrix(B_field, epsilon, m_dark, omega_range, L):
    """
    Compute full conversion matrix for photon-dark photon system
    From: Primordial Photon-DarkPhoton Entanglement framework
    """
    hbar_ev_s = 6.582e-16
    c_km_s = 3e5
    
    # Conversion probability
    P_conversion = np.zeros((len(omega_range), len(L)))
    
    for i, omega in enumerate(omega_range):
        for j, length in enumerate(L):
            if m_dark > 0:
                conversion_length = 4 * omega * hbar_ev_s * c_km_s / (m_dark**2)
                P = (epsilon * B_field / 1e15)**2 * np.sin(np.pi * length / conversion_length)**2
            else:
                P = (epsilon * B_field / 1e15)**2
            P_conversion[i, j] = np.clip(P, 0, 1)
    
    return P_conversion


def plot_optical_conversion_chart(B_field, epsilon, m_dark):
    """
    Optical chart showing photon-dark photon conversion probability
    as function of photon energy and propagation length
    """
    # Energy range (eV) – optical to X-ray
    energy_eV = np.logspace(0, 4, 100)  # 1 eV to 10 keV
    omega = energy_eV / hbar  # angular frequency
    
    # Length range (km)
    L = np.logspace(-2, 2, 100)  # 0.01 km to 100 km
    
    # Compute conversion matrix
    P_matrix = photon_dark_photon_conversion_matrix(B_field, epsilon, m_dark, omega, L)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    
    # 2D colormap
    im = ax1.pcolormesh(L, energy_eV, P_matrix, 
                        shading='nearest', cmap='plasma', norm='log')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Propagation Length (km)', fontsize=11)
    ax1.set_ylabel('Photon Energy (eV)', fontsize=11)
    ax1.set_title(f'γ ↔ A\' Conversion Probability\nB = {B_field:.1e} G, ε = {epsilon:.1e}, m_A\' = {m_dark:.1e} eV', 
                  fontsize=12)
    plt.colorbar(im, ax=ax1, label='Conversion Probability')
    
    # Overlay optical range
    ax1.axhspan(1.8, 3.1, alpha=0.3, color='green', label='Visible Range')
    ax1.axhspan(0.1, 10, alpha=0.2, color='yellow', label='Optical/X-ray')
    ax1.legend()
    
    # Maximum conversion probability vs energy
    P_max_vs_energy = np.max(P_matrix, axis=1)
    ax2.semilogx(energy_eV, P_max_vs_energy, 'b-', linewidth=2)
    ax2.axvspan(1.8, 3.1, alpha=0.3, color='green', label='Visible Range')
    ax2.set_xlabel('Photon Energy (eV)', fontsize=11)
    ax2.set_ylabel('Max Conversion Probability', fontsize=11)
    ax2.set_title('Peak Conversion Probability by Photon Energy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.tight_layout()
    return fig


def plot_dark_photon_spectrum(B_field, epsilon, m_dark):
    """
    Dark photon spectrum produced from conversion
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Energy range
    energy_eV = np.logspace(0, 4, 500)
    omega = energy_eV / hbar
    
    # Length for maximum conversion
    hbar_ev_s = 6.582e-16
    c_km_s = 3e5
    if m_dark > 0:
        L_opt = 4 * 1e18 * hbar_ev_s * c_km_s / (m_dark**2) / 2
    else:
        L_opt = 10
    
    P = np.zeros_like(energy_eV)
    for i, omega_val in enumerate(omega):
        if m_dark > 0:
            conversion_length = 4 * omega_val * hbar_ev_s * c_km_s / (m_dark**2)
            P[i] = (epsilon * B_field / 1e15)**2 * np.sin(np.pi * L_opt / conversion_length)**2
        else:
            P[i] = (epsilon * B_field / 1e15)**2
        P[i] = np.clip(P[i], 0, 1)
    
    # Dark photon flux (arbitrary normalization)
    dark_photon_flux = P * (energy_eV)**(-2)  # Simple power law spectrum
    
    ax.loglog(energy_eV, dark_photon_flux, 'r-', linewidth=2, label='Dark Photon Flux')
    ax.loglog(energy_eV, P, 'b--', linewidth=1.5, alpha=0.7, label='Conversion Probability')
    
    # Mark optical range
    ax.axvspan(1.8, 3.1, alpha=0.3, color='green', label='Visible Range')
    
    ax.set_xlabel('Photon Energy (eV)', fontsize=11)
    ax.set_ylabel('Relative Flux / Probability', fontsize=11)
    ax.set_title(f'Dark Photon Spectrum from Conversion\nB = {B_field:.1e} G, ε = {epsilon:.1e}, m_A\' = {m_dark:.1e} eV',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


# ── MAGNETAR PHYSICS FUNCTIONS ─────────────────────────────────────────────

def magnetar_dipole_field(B_surface, R_ns, r, theta, inclination=0):
    """Magnetar dipole field"""
    B0 = B_surface * (R_ns / (r + 1e-9))**3
    theta_rad = np.radians(theta + inclination)
    B_r = 2 * B0 * np.cos(theta_rad)
    B_theta = B0 * np.sin(theta_rad)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    return B_r, B_theta, B_mag


def euler_heisenberg_n(B_over_Bc):
    """Euler-Heisenberg refractive index"""
    x = B_over_Bc**2
    return 4 * alpha_fine/(45 * np.pi) * x


def dark_photon_conversion(B, L, epsilon, m_dark, omega=1e18):
    """Dark photon conversion probability"""
    if m_dark <= 0:
        return (epsilon * B / 1e15)**2 * np.ones_like(L)
    hbar_ev_s = 6.582e-16
    c_km_s = 3e5
    conversion_length = 4 * omega * hbar_ev_s * c_km_s / (m_dark**2)
    P = (epsilon * B / 1e15)**2 * np.sin(np.pi * L / conversion_length)**2
    return np.clip(P, 0, 1)


def schwinger_pair_production(B_field):
    """Schwinger pair production rate"""
    B_norm = B_field / B_crit
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.exp(-np.pi / (B_norm + 1e-9))
    return np.clip(rate, 0, 1)


# ── PLOTTING FUNCTIONS ─────────────────────────────────────────────

def plot_magnetar_field(B_surface, R_ns, inclination):
    """Magnetar field using quiver"""
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
    logB = np.log10(B_mag + 1e-9)
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    skip = 2
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              U_norm[::skip, ::skip], V_norm[::skip, ::skip],
              logB[::skip, ::skip], cmap='plasma', scale=25, width=0.008, alpha=0.8)
    
    ax.add_patch(Circle((0, 0), R_ns, color='#d62728', alpha=0.9))
    ax.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    
    for angle in np.linspace(0, 2*np.pi, 12):
        t = np.linspace(0, 1, 50)
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
    ax.set_title(f'Magnetar Dipole Field | B_surface = {B_surface:.1e} G | Inclination = {inclination}°')
    plt.colorbar(ax.collections[0], ax=ax, label='log₁₀(|B|) [G]')
    
    return fig


def plot_qed_vacuum(B_ratio):
    """QED vacuum polarization plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    
    B_range = np.logspace(-1, min(3, np.log10(max(2, B_ratio*2))), 100)
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
    
    for impact in [6, 8, 10, 12]:
        t = np.linspace(0, 50, 500)
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
    st.title("⚡ Stellaris QED v6.0")
    st.markdown("*Complete with Optical Conversion Chart*")
    st.markdown("---")
    
    st.markdown("### 🌌 Magnetar Parameters")
    B_surface = st.slider("Surface B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    R_ns = st.slider("Neutron Star Radius (km)", 8, 15, 10)
    inclination = st.slider("Magnetic Inclination (°)", 0, 90, 0)
    
    st.markdown("---")
    st.markdown("### 🕳️ Dark Sector Parameters")
    epsilon = st.slider("Kinetic Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin a/M", 0.0, 0.998, 0.9)
    
    st.markdown("---")
    st.markdown("### 📚 Physics References")
    st.latex(r"B_{\text{crit}} = \frac{m_e^2 c^2}{e\hbar} = 4.4\times10^{13}\text{ G}")
    st.latex(r"n = 1 + \frac{\alpha}{45\pi}\left(\frac{B}{B_c}\right)^2")
    st.latex(r"P_{\gamma\to A'} = \left(\frac{\varepsilon B}{m_{A'}}\right)^2\sin^2\left(\frac{m_{A'}^2 L}{4\omega}\right)")
    st.latex(r"\mathcal{L}_{\text{mix}} = \frac{\varepsilon}{2} F_{\mu\nu} F'^{\mu\nu}")
    
    st.caption("Tony Ford Model | Primordial Photon-DarkPhoton Entanglement | v6.0")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Stellaris QED Explorer")
st.markdown("*Complete Integration: Photon-Dark Photon Conversion + Magnetar QED + QCIS*")
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
    st.metric("Kerr Spin", f"{a_spin:.3f}")

if B_ratio > 1:
    st.warning(f"⚠️ **Super-critical field!** B/B_crit = {B_ratio:.2e} | Schwinger pair production and vacuum birefringence are significant.")

# Tabs – NEW: Optical Conversion Chart as first tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔆 Optical Conversion Chart",
    "🌌 Magnetar Field", 
    "⚛️ QED Vacuum", 
    "🕳️ Dark Photons", 
    "🌀 Kerr Geodesics",
    "🔬 QCIS Integration"
])

# Tab 1: Optical Conversion Chart (NEW - from your project)
with tab1:
    st.header("Photon ↔ Dark Photon Conversion")
    st.markdown("*From Primordial Photon-DarkPhoton Entanglement Framework*")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.info(f"""
        **Conversion Parameters**
        - Magnetic Field: B = {B_surface:.1e} G
        - Kinetic Mixing: ε = {epsilon:.1e}
        - Dark Photon Mass: m_A' = {m_dark:.1e} eV
        - Critical Field: B_crit = {B_crit:.1e} G
        
        **Optical Range** (shaded green)
        - Visible light: 1.8 - 3.1 eV (400-700 nm)
        - X-ray: > 100 eV
        - Gamma-ray: > 100 keV
        
        **Physics**: γ ↔ A' oscillations in external B field
        """)
        
        # Show optical range conversion
        optical_energy = 2.5  # eV (visible)
        hbar_ev_s = 6.582e-16
        c_km_s = 3e5
        omega_opt = optical_energy / hbar_ev_s
        
        if m_dark > 0:
            conversion_length = 4 * omega_opt * hbar_ev_s * c_km_s / (m_dark**2)
            P_opt = (epsilon * B_surface / 1e15)**2 * np.sin(np.pi * 10 / conversion_length)**2
        else:
            P_opt = (epsilon * B_surface / 1e15)**2
        
        st.metric("Visible Light Conversion", f"{P_opt:.2e}", 
                  delta="Detectable" if P_opt > 1e-6 else "Below threshold")
    
    with col_b:
        with st.spinner("Generating optical conversion chart..."):
            fig = plot_optical_conversion_chart(B_surface, epsilon, m_dark)
            st.pyplot(fig)
            plt.close(fig)
    
    # Second row: Dark photon spectrum
    st.markdown("---")
    st.subheader("Dark Photon Spectrum")
    
    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.markdown("""
        **Dark Photon Production**
        - Conversion produces dark photons with same energy as parent photons
        - Spectrum follows source photon distribution × conversion probability
        - Peaks at energies where conversion is coherent
        - Can be detected via reconversion to photons in B field
        """)
    
    with col_d:
        with st.spinner("Computing dark photon spectrum..."):
            fig = plot_dark_photon_spectrum(B_surface, epsilon, m_dark)
            st.pyplot(fig)
            plt.close(fig)

# Tab 2: Magnetar Field
with tab2:
    st.header("Magnetar Magnetic Field")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        pair_rate = schwinger_pair_production(B_surface)
        st.info(f"""
        **Magnetar Parameters**
        - Surface Field: {B_surface:.1e} G
        - Critical Field: {B_crit:.1e} G
        - B/B_crit = {B_ratio:.2e}
        - Radius: {R_ns} km
        - Inclination: {inclination}°
        """)
        st.metric("Schwinger Pair Rate", f"{pair_rate:.2e}", 
                  delta="Active" if pair_rate > 0.01 else "Suppressed")
    
    with col_b:
        with st.spinner("Rendering magnetar field..."):
            fig = plot_magnetar_field(B_surface, R_ns, inclination)
            st.pyplot(fig)
            plt.close(fig)

# Tab 3: QED Vacuum
with tab3:
    st.header("Euler-Heisenberg QED Vacuum")
    
    col_a, col_b = st.columns(2)
    with col_a:
        delta_n = 4 * alpha_fine/(45 * np.pi) * (B_ratio**2)
        st.info(f"""
        **Current Field Effects**
        - B/B_crit = {B_ratio:.2e}
        - Refractive index shift: Δn = {delta_n:.2e}
        - Vacuum birefringence: Δn_⟂ - Δn_∥ = {3 * alpha_fine/(45 * np.pi) * (B_ratio**2):.2e}
        """)
    
    with col_b:
        with st.spinner("Rendering QED vacuum..."):
            fig = plot_qed_vacuum(B_ratio)
            st.pyplot(fig)
            plt.close(fig)

# Tab 4: Dark Photons
with tab4:
    st.header("Dark Photon Conversion Details")
    
    col_a, col_b = st.columns(2)
    with col_a:
        max_P = (epsilon * B_surface / 1e15)**2
        st.info(f"""
        **Conversion Parameters**
        - Kinetic mixing ε = {epsilon:.1e}
        - Dark photon mass m' = {m_dark:.1e} eV
        - Maximum conversion probability: {max_P:.2e}
        
        **Coherence Length**
        - L_coherent = 4πω/m'²
        - For optical light: L_coherent ≈ {4 * np.pi * 2.5 / (m_dark**2 + 1e-20):.2e} km
        """)
    
    with col_b:
        with st.spinner("Rendering dark photon conversion..."):
            L_sample = np.logspace(-2, 2, 1000)
            P_sample = dark_photon_conversion(B_surface, L_sample, epsilon, m_dark)
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
            ax.semilogx(L_sample, P_sample, 'b-', linewidth=2)
            ax.set_xlabel('Propagation Length (km)')
            ax.set_ylabel('Conversion Probability')
            ax.set_title(f'γ ↔ A\' Conversion | m_A\' = {m_dark:.1e} eV')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            plt.close(fig)

# Tab 5: Kerr Geodesics
with tab5:
    st.header("Null Geodesics in Kerr Spacetime")
    
    col_a, col_b = st.columns(2)
    with col_a:
        r_horizon = 1 + np.sqrt(1 - a_spin**2)
        if a_spin <= 0.999:
            r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        else:
            r_photon = 3
        st.info(f"""
        **Kerr Parameters**
        - Spin a/M = {a_spin:.3f}
        - Event horizon: r_+ = {r_horizon:.3f} M
        - Photon sphere: r_ph = {r_photon:.3f} M
        """)
    
    with col_b:
        with st.spinner("Rendering Kerr geodesics..."):
            fig = plot_kerr_geodesic(a_spin)
            st.pyplot(fig)
            plt.close(fig)

# Tab 6: QCIS Integration
with tab6:
    st.header("Quantum Cosmology Integration Suite")
    st.markdown("*Quantum-corrected power spectra and entanglement entropy*")
    
    # FDM Soliton
    r = np.linspace(0, 5, 100)
    soliton = np.sin(np.pi * r) / (np.pi * r + 1e-9)
    soliton = soliton**2
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.plot(r, soliton, 'b-', linewidth=2)
        ax.set_xlabel('r / r_s')
        ax.set_ylabel('ρ(r) / ρ₀')
        ax.set_title('FDM Soliton Profile\n[sin(kr)/(kr)]²')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        plt.close(fig)
    
    with col_b:
        # Power spectrum
        k = np.logspace(-3, 0, 100)
        A_s = 2.1e-9
        n_s = 0.965
        k0 = 0.05
        P_lcdm = A_s * (k / k0)**(n_s - 1)
        quantum_corr = 1 + 1.0 * (k / k0)**0.8
        
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.loglog(k, P_lcdm / quantum_corr, 'b-', linewidth=2, label='ΛCDM')
        ax.loglog(k, P_lcdm, 'r-', linewidth=2, label='Quantum-corrected')
        ax.fill_between(k, P_lcdm / quantum_corr, P_lcdm, alpha=0.3, color='red')
        ax.set_xlabel('k (Mpc⁻¹)')
        ax.set_ylabel('P(k)')
        ax.set_title('Quantum-Corrected Power Spectrum')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        plt.close(fig)

st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v6.0** | Complete with Optical Conversion Chart | Tony Ford Model")
