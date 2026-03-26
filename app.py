"""
Stellaris QED Explorer v2.2 – Guaranteed Display
Multiple fallback methods for Streamlit Cloud compatibility
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, hbar, e, m_e, alpha
import warnings
import time
import json
from dataclasses import dataclass
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Stellaris QED Explorer v2.2",
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


# ── RELIABLE PLOTTING FUNCTION (MULTIPLE FALLBACKS) ─────────────────────────────────────────────

def plot_magnetar_field_robust(B_surface, R_ns, inclination):
    """
    Plot magnetar field with multiple fallback methods for guaranteed display
    """
    resolution = 60
    
    # Create grid
    r = np.linspace(R_ns, 5*R_ns, resolution)
    theta = np.linspace(0, np.pi, resolution)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    # Calculate field
    inclination_rad = np.radians(inclination)
    B0 = B_surface * (R_ns / (R + 1e-9))**3
    
    B_r = 2 * B0 * np.cos(Theta + inclination_rad)
    B_theta = B0 * np.sin(Theta + inclination_rad)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    
    # Convert to Cartesian
    X = R * np.sin(Theta)
    Y = R * np.cos(Theta)
    U = B_r * np.sin(Theta) + B_theta * np.cos(Theta)
    V = B_r * np.cos(Theta) - B_theta * np.sin(Theta)
    
    # Normalize vectors for visualization
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-9)
    V_norm = V / (magnitude + 1e-9)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Method 1: Try streamplot
    try:
        # Color by log of field strength
        colors = np.log10(B_mag + 1e-9)
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-9)
        
        ax.streamplot(X, Y, U_norm, V_norm, 
                     color=colors, 
                     cmap='plasma', 
                     density=1.2,
                     linewidth=1.5,
                     arrowsize=1)
    except Exception as e1:
        # Method 2: Fallback to quiver with fewer points
        try:
            skip = 4
            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                     U_norm[::skip, ::skip], V_norm[::skip, ::skip],
                     np.log10(B_mag[::skip, ::skip] + 1e-9),
                     cmap='plasma', 
                     scale=50, 
                     width=0.008,
                     alpha=0.8)
        except Exception as e2:
            # Method 3: Simple contour plot of field strength
            contour = ax.contourf(X, Y, np.log10(B_mag + 1e-9), 
                                  levels=20, cmap='plasma', alpha=0.8)
            plt.colorbar(contour, ax=ax, label='log10(|B|) [G]')
    
    # Add neutron star
    ax.add_patch(Circle((0, 0), R_ns, color='red', alpha=0.8))
    
    # Add field lines manually (simplified)
    for angle in np.linspace(0, 2*np.pi, 16):
        start_r = R_ns * 1.1
        start_x = start_r * np.cos(angle)
        start_y = start_r * np.sin(angle)
        
        # Simple dipole field line approximation
        t = np.linspace(0, 1, 50)
        r_line = R_ns * (1 + t * 4)
        theta_line = angle + 0.5 * np.sin(angle) * t
        x_line = r_line * np.cos(theta_line)
        y_line = r_line * np.sin(theta_line)
        ax.plot(x_line, y_line, 'w-', linewidth=0.5, alpha=0.3)
    
    ax.set_aspect('equal')
    ax.set_xlim(-5*R_ns, 5*R_ns)
    ax.set_ylim(-5*R_ns, 5*R_ns)
    ax.set_xlabel('x (km)', color='white')
    ax.set_ylabel('y (km)', color='white')
    ax.tick_params(colors='white')
    
    B_display = f"{B_surface:.1e}" if B_surface >= 1e13 else f"{B_surface:.0f}"
    ax.set_title(f'Magnetar Dipole Field | B_surface = {B_display} G | Inclination = {inclination}°', 
                 color='white', fontsize=12)
    
    # Add colorbar if using contour
    if 'contour' in locals():
        pass
    else:
        sm = plt.cm.ScalarMappable(cmap='plasma')
        sm.set_array(np.log10(B_mag.flatten() + 1e-9))
        cbar = plt.colorbar(sm, ax=ax, label='log10(|B|) [G]')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
    
    fig.patch.set_facecolor('#0a0a2a')
    ax.set_facecolor('#0a0a2a')
    
    return fig


def plot_qed_vacuum_robust(B_ratio):
    """Reliable QED vacuum plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    B_range = np.logspace(-1, min(3, np.log10(max(2, B_ratio*2))), 100)
    B_range_safe = np.clip(B_range, 0.01, 1e6)
    x = B_range_safe**2
    
    delta_n_perp = 4 * alpha_fine/(45 * np.pi) * x
    delta_n_para = 7 * alpha_fine/(45 * np.pi) * x
    
    axes[0].loglog(B_range_safe, delta_n_perp, 'b-', linewidth=2, label='⟂ Polarization')
    axes[0].loglog(B_range_safe, delta_n_para, 'r-', linewidth=2, label='∥ Polarization')
    axes[0].set_xlabel('B / B_critical', color='white')
    axes[0].set_ylabel('n - 1', color='white')
    axes[0].set_title('Vacuum Refractive Index', color='white')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].tick_params(colors='white')
    
    delta_n = np.abs(delta_n_perp - delta_n_para)
    axes[1].loglog(B_range_safe, delta_n, 'g-', linewidth=2)
    axes[1].set_xlabel('B / B_critical', color='white')
    axes[1].set_ylabel('|Δn|', color='white')
    axes[1].set_title('Vacuum Birefringence', color='white')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(colors='white')
    
    for ax in axes:
        ax.set_facecolor('#0a0a2a')
    
    fig.patch.set_facecolor('#0a0a2a')
    fig.tight_layout()
    
    return fig


def plot_dark_photon_conversion_robust(B, epsilon, m_dark):
    """Reliable dark photon conversion plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    L = np.logspace(-2, 2, 1000)
    
    if m_dark > 1e-11:
        # Use physical constants
        hbar_ev_s = 6.582e-16  # eV·s
        c_cm = 3e10  # cm/s
        omega = 1e18  # Hz
        conversion_length = 4 * omega * hbar_ev_s * c_cm / (m_dark**2)
        P = (epsilon * B / 1e15)**2 * np.sin(np.pi * L / conversion_length)**2
    else:
        P = (epsilon * B / 1e15)**2 * np.ones_like(L)
    
    P = np.clip(P, 0, 1)
    
    ax.semilogx(L, P, 'b-', linewidth=2)
    ax.set_xlabel('Propagation Length (km)', color='white')
    ax.set_ylabel('Conversion Probability', color='white')
    ax.set_title(f'γ ↔ A\' Conversion\nB = {B:.1e} G, ε = {epsilon:.1e}, m_A\' = {m_dark:.1e} eV', 
                 color='white', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0a2a')
    fig.patch.set_facecolor('#0a0a2a')
    
    return fig


def plot_schwinger_rate_robust(B_ratio):
    """Reliable Schwinger rate plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    B_norm = np.logspace(-1, np.log10(max(2, B_ratio*2)), 100)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.exp(-np.pi / (B_norm + 1e-9))
    rate = np.clip(rate, 0, 1)
    
    ax.semilogy(B_norm, rate, 'r-', linewidth=2)
    ax.axvline(x=1, color='w', linestyle='--', alpha=0.5, label='B_critical')
    ax.set_xlabel('B / B_critical', color='white')
    ax.set_ylabel('Pair Production Rate (arb. units)', color='white')
    ax.set_title('Schwinger Pair Production', color='white')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0a2a')
    fig.patch.set_facecolor('#0a0a2a')
    
    return fig


def plot_kerr_geodesic_robust(a_spin):
    """Reliable Kerr geodesic plot"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Event horizon
    r_horizon = 1 + np.sqrt(1 - a_spin**2)
    circle = Circle((0, 0), r_horizon, color='gray', alpha=0.6)
    ax.add_patch(circle)
    ax.text(0, 0, 'BH', color='white', ha='center', va='center', fontsize=8)
    
    # Photon sphere
    if a_spin <= 0.999:
        r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r_photon * np.cos(theta), r_photon * np.sin(theta), 
                'r--', linewidth=2, label='Photon Sphere', alpha=0.7)
    
    # Sample geodesic
    t = np.linspace(0, 50, 500)
    r = 10 * np.exp(-t/40) + r_horizon + 1
    phi = np.pi/4 + 0.3 * np.sin(t/15)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    ax.plot(x, y, 'cyan', linewidth=2, label='Photon Trajectory')
    
    ax.set_aspect('equal')
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_xlabel('x/M', color='white')
    ax.set_ylabel('y/M', color='white')
    ax.set_title(f'Kerr Spacetime | a/M = {a_spin:.3f}', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(colors='white')
    ax.set_facecolor('#0a0a2a')
    fig.patch.set_facecolor('#0a0a2a')
    
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
    
    st.caption("Tony Ford Model | Stellaris QED v2.2")


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
    st.metric("Dark Photon Coupling", f"{epsilon:.1e}")
with col3:
    st.metric("Dark Photon Mass", f"{m_dark:.1e} eV")
with col4:
    st.metric("Kerr Spin", f"{a_spin:.3f}")

if B_ratio > 1:
    st.warning(f"⚠️ **Super-critical field!** B/B_crit = {B_ratio:.2e} | QED effects dominate.")

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
        """)
        if B_ratio > 1:
            st.warning("⚠️ Super-critical regime: QED effects dominate!")
    
    with col_b:
        with st.spinner("Rendering magnetar field..."):
            fig = plot_magnetar_field_robust(B_surface, R_ns, inclination)
            st.pyplot(fig)
            plt.close(fig)

with tab2:
    st.header("Euler-Heisenberg QED Vacuum")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.latex(r"\mathcal{L}_{\text{EH}} = \frac{2\alpha^2}{45m_e^4}\left[(\mathbf{E}^2 - \mathbf{B}^2)^2 + 7(\mathbf{E}\cdot\mathbf{B})^2\right]")
        delta_n = 4 * alpha_fine/(45 * np.pi) * (B_ratio**2)
        st.info(f"""
        **Current Field Strength**
        - B/B_crit = {B_ratio:.2e}
        - Refractive index shift: Δn ≈ {delta_n:.2e}
        """)
    
    with col_b:
        with st.spinner("Rendering QED vacuum..."):
            fig = plot_qed_vacuum_robust(B_ratio)
            st.pyplot(fig)
            plt.close(fig)

with tab3:
    st.header("Photon ↔ Dark Photon Conversion")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.latex(r"\mathcal{L}_{\text{mix}} = \frac{\varepsilon}{2} F_{\mu\nu} F'^{\mu\nu}")
        max_P = (epsilon * B_surface / 1e15)**2
        st.info(f"""
        **Conversion Parameters**
        - Kinetic mixing ε = {epsilon:.1e}
        - Dark photon mass m' = {m_dark:.1e} eV
        - Maximum conversion probability: {max_P:.2e}
        """)
    
    with col_b:
        with st.spinner("Rendering dark photon conversion..."):
            fig = plot_dark_photon_conversion_robust(B_surface, epsilon, m_dark)
            st.pyplot(fig)
            plt.close(fig)

with tab4:
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
            fig = plot_kerr_geodesic_robust(a_spin)
            st.pyplot(fig)
            plt.close(fig)

st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v2.2** | Guaranteed Display | Tony Ford Model")
