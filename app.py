"""
Stellaris QED Explorer v2.1 – Fixed Grid & Large Field Handling
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, hbar, e, m_e, alpha
from scipy.integrate import odeint
import warnings
import time
import json
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Stellaris QED Explorer v2.1",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0a2a; }
[data-testid="stSidebar"] { background: #1a1a3a; }
.stTitle, h1, h2, h3 { color: #00ffff; }
[data-testid="stMetricValue"] { color: #00ffff; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)  # 4.4e13 G
alpha_fine = 1/137.036


# ── CORRECTED FIELD PLOTTING FUNCTION ─────────────────────────────────────────────

def plot_magnetar_field(B_surface, R_ns, inclination, resolution=50):
    """Fixed streamplot with proper grid dimensions"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create grid with consistent dimensions
    r = np.linspace(R_ns, 5*R_ns, resolution)
    theta = np.linspace(0, np.pi, resolution)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    # Calculate field components
    inclination_rad = np.radians(inclination)
    B0 = B_surface * (R_ns / (R + 1e-9))**3
    
    B_r = 2 * B0 * np.cos(Theta + inclination_rad)
    B_theta = B0 * np.sin(Theta + inclination_rad)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    
    # Convert to Cartesian coordinates
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)
    
    # Convert field components to Cartesian
    U = B_r * np.sin(Theta) + B_theta * np.cos(Theta)
    V = B_r * np.cos(Theta) - B_theta * np.sin(Theta)
    
    # Clip extreme values for visualization
    B_mag_clipped = np.clip(B_mag, 0, B_surface)
    
    # Streamplot with error handling
    try:
        ax.streamplot(X, Y, U, V, color=np.log10(B_mag_clipped + 1e-9), 
                     cmap='plasma', density=1.2, linewidth=1)
    except Exception as e:
        # Fallback: quiver plot
        skip = max(1, resolution // 20)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U[::skip, ::skip], V[::skip, ::skip],
                  np.log10(B_mag_clipped[::skip, ::skip] + 1e-9),
                  cmap='plasma', scale=30, width=0.005)
    
    # Neutron star
    ax.add_patch(Circle((0, 0), R_ns, color='red', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.set_xlim(-5*R_ns, 5*R_ns)
    ax.set_ylim(-5*R_ns, 5*R_ns)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    
    # Format B_surface for display
    B_display = f"{B_surface:.1e}" if B_surface >= 1e13 else f"{B_surface:.0f}"
    ax.set_title(f'Magnetar Dipole Field | B_surface = {B_display} G | Inclination = {inclination}°')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma')
    sm.set_array(np.log10(B_mag_clipped.flatten() + 1e-9))
    plt.colorbar(sm, ax=ax, label='log10(|B|) [G]')
    
    return fig


def plot_qed_vacuum(B_range):
    """Euler-Heisenberg vacuum polarization plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Safe handling of large B values
    B_range_safe = np.clip(B_range, 0.01, 1000)
    x = B_range_safe**2
    delta_n_perp = 4 * alpha_fine/(45 * np.pi) * x
    delta_n_para = 7 * alpha_fine/(45 * np.pi) * x
    
    axes[0].loglog(B_range_safe, delta_n_perp, 'b-', linewidth=2, label='⟂ Polarization')
    axes[0].loglog(B_range_safe, delta_n_para, 'r-', linewidth=2, label='∥ Polarization')
    axes[0].set_xlabel('B / B_critical')
    axes[0].set_ylabel('n - 1')
    axes[0].set_title('Vacuum Refractive Index')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    delta_n = np.abs(delta_n_perp - delta_n_para)
    axes[1].loglog(B_range_safe, delta_n, 'g-', linewidth=2)
    axes[1].set_xlabel('B / B_critical')
    axes[1].set_ylabel('|Δn|')
    axes[1].set_title('Vacuum Birefringence')
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_dark_photon_conversion(B, epsilon, m_dark):
    """Dark photon conversion probability"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    L = np.logspace(-2, 2, 1000)
    
    if m_dark > 0:
        omega = 1e18  # Reference frequency
        conversion_length = 4 * omega * 1.0546e-27 / (m_dark**2 * 1.602e-19 * 3e8)
        P = (epsilon * B / 1e15)**2 * np.sin(np.pi * L / conversion_length)**2
    else:
        P = (epsilon * B / 1e15)**2 * np.ones_like(L)
    
    P = np.clip(P, 0, 1)
    
    ax.semilogx(L, P, 'b-', linewidth=2)
    ax.set_xlabel('Propagation Length (km)')
    ax.set_ylabel('Conversion Probability')
    ax.set_title(f'γ ↔ A\' Conversion\nB = {B:.1e} G, ε = {epsilon:.1e}, m_A\' = {m_dark:.1e} eV')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    return fig


def plot_schwinger_rate(B_field, B_crit_val):
    """Schwinger pair production rate"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    B_norm = B_field / B_crit_val
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.exp(-np.pi / (B_norm + 1e-9))
    rate = np.clip(rate, 0, 1)
    
    ax.semilogy(B_norm, rate, 'r-', linewidth=2)
    ax.axvline(x=1, color='k', linestyle='--', alpha=0.5, label='B_critical')
    ax.set_xlabel('B / B_critical')
    ax.set_ylabel('Pair Production Rate (arb. units)')
    ax.set_title('Schwinger Pair Production')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, max(5, B_norm.max()))
    
    return fig


def plot_kerr_geodesic(a_spin):
    """Kerr geodesic visualization"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Event horizon
    r_horizon = 1 + np.sqrt(1 - a_spin**2)
    circle = Circle((0, 0), r_horizon, color='black', alpha=0.8)
    ax.add_patch(circle)
    
    # Photon sphere
    if a_spin <= 1:
        r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r_photon * np.cos(theta), r_photon * np.sin(theta), 
                'r--', linewidth=2, label='Photon Sphere')
    
    # Sample geodesic
    t = np.linspace(0, 50, 500)
    r = 10 * np.exp(-t/30) + r_horizon
    theta = np.pi/2 + 0.5 * np.sin(t/10)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, 'cyan', linewidth=2, label='Photon Trajectory')
    
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
    st.title("⚡ Stellaris QED Explorer")
    st.markdown("*Quantum Vacuum Engineering*")
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
    st.latex(r"\Gamma \propto \exp\left(-\frac{\pi E_{\text{crit}}}{E}\right)")
    
    st.caption("Tony Ford Model | Stellaris QED v2.1")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Stellaris QED Explorer")
st.markdown("*Quantum Vacuum Engineering for Extreme Astrophysical Environments*")
st.markdown("---")

# Display metrics
B_ratio = B_surface / B_crit
st.markdown("### 📊 Live Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("B / B_crit", f"{B_ratio:.2e}")
with col2:
    st.metric("Dark Photon Coupling", f"{epsilon:.1e}")
with col3:
    st.metric("Dark Photon Mass", f"{m_dark:.1e} eV")
with col4:
    st.metric("Kerr Spin", f"{a_spin:.3f}")

# Warning for super-critical field
if B_ratio > 1:
    st.warning(f"⚠️ **Super-critical field!** B/B_crit = {B_ratio:.2e} | QED effects dominate. Pair production and vacuum birefringence are significant.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌌 Magnetar Field", "⚛️ QED Vacuum", "🕳️ Dark Photons", "🌀 Kerr Geodesics"
])

with tab1:
    st.header("Magnetar Magnetic Field")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.info(f"""
        **Magnetar Parameters**
        - Surface Field: {B_surface:.1e} G
        - Critical Field: {B_crit:.1e} G
        - B/B_crit = {B_ratio:.2e}
        - Radius: {R_ns} km
        - Inclination: {inclination}°
        
        **Field Physics**
        - Dipole: B ∝ 1/r³
        - Force-free electrodynamics
        - Quantum vacuum polarization
        """)
        
        if B_ratio > 1:
            st.warning("⚠️ Super-critical regime: QED effects dominate!")
    
    with col_b:
        with st.spinner("Computing field lines..."):
            fig = plot_magnetar_field(B_surface, R_ns, inclination)
            st.pyplot(fig)
            plt.close(fig)

with tab2:
    st.header("Euler-Heisenberg QED Vacuum")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.latex(r"\mathcal{L}_{\text{EH}} = \frac{2\alpha^2}{45m_e^4}\left[(\mathbf{E}^2 - \mathbf{B}^2)^2 + 7(\mathbf{E}\cdot\mathbf{B})^2\right]")
        st.info(f"""
        **Current Field Strength**
        - B/B_crit = {B_ratio:.2e}
        - Refractive index shift: Δn ≈ {4 * alpha_fine/(45 * np.pi) * (B_ratio**2):.2e}
        - Birefringence: Δn_⟂ - Δn_∥ ≈ {3 * alpha_fine/(45 * np.pi) * (B_ratio**2):.2e}
        """)
    
    with col_b:
        B_range = np.logspace(-1, min(3, np.log10(max(2, B_ratio*2))), 100)
        fig = plot_qed_vacuum(B_range)
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    st.header("Photon ↔ Dark Photon Conversion")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.latex(r"\mathcal{L}_{\text{mix}} = \frac{\varepsilon}{2} F_{\mu\nu} F'^{\mu\nu}")
        st.info(f"""
        **Conversion Parameters**
        - Kinetic mixing ε = {epsilon:.1e}
        - Dark photon mass m' = {m_dark:.1e} eV
        - Magnetic field B = {B_surface:.1e} G
        - Characteristic conversion length: ~{(4 * 1e18 * 1e-27) / (m_dark**2 * 1.6e-19 * 3e8):.2e} km
        """)
    
    with col_b:
        fig = plot_dark_photon_conversion(B_surface, epsilon, m_dark)
        st.pyplot(fig)
        plt.close(fig)
    
    # Maximum conversion probability
    max_P = (epsilon * B_surface / 1e15)**2
    st.metric("Maximum Conversion Probability", f"{max_P:.2e}")

with tab4:
    st.header("Null Geodesics in Kerr Spacetime")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.latex(r"ds^2 = -\left(1-\frac{2Mr}{\Sigma}\right)dt^2 - \frac{4aMr\sin^2\theta}{\Sigma}dtd\phi + \frac{\Sigma}{\Delta}dr^2 + \Sigma d\theta^2 + \left(r^2 + a^2 + \frac{2a^2Mr\sin^2\theta}{\Sigma}\right)\sin^2\theta d\phi^2")
        st.info(f"""
        **Kerr Parameters**
        - Spin a/M = {a_spin:.3f}
        - Event horizon: r_+ = 1 + √(1-a²) = {1 + np.sqrt(1 - a_spin**2):.3f} M
        - Photon sphere: r_ph = {2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin)))):.3f} M
        """)
    
    with col_b:
        fig = plot_kerr_geodesic(a_spin)
        st.pyplot(fig)
        plt.close(fig)

# Footer
st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v2.1** | Fixed Streamplot | Large Field Handling | Tony Ford Model")
