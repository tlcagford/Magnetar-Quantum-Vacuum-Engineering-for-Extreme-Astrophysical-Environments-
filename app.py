"""
Stellaris QED Explorer v5.2 – Final Working Version
No system dependencies required
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, hbar, e, m_e, alpha
import warnings
import time

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Stellaris QED Explorer",
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


def schwinger_pair_production(B_field):
    """Schwinger pair production rate"""
    B_norm = B_field / B_crit
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.exp(-np.pi / (B_norm + 1e-9))
    return np.clip(rate, 0, 1)


# ── RELIABLE PLOTTING ─────────────────────────────────────────────

def plot_magnetar_field(B_surface, R_ns, inclination):
    """Magnetar field visualization"""
    resolution = 35
    
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
    ax.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    
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
    ax.set_title(f'Magnetar Dipole Field | B = {B_surface:.1e} G | Inclination = {inclination}°')
    
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
    ax.set_title(f'γ ↔ A\' Conversion | B = {B:.1e} G, ε = {epsilon:.1e}')
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
    st.title("⚡ Stellaris QED Explorer")
    st.markdown("*Quantum Vacuum Engineering*")
    st.markdown("---")
    
    B_surface = st.slider("Surface B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    R_ns = st.slider("Neutron Star Radius (km)", 8, 15, 10)
    inclination = st.slider("Magnetic Inclination (°)", 0, 90, 0)
    epsilon = st.slider("Kinetic Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin a/M", 0.0, 0.998, 0.9)
    
    st.markdown("---")
    st.latex(r"B_{\text{crit}} = 4.4\times10^{13}\text{ G}")
    st.latex(r"P_{\gamma\to A'} = \left(\frac{\varepsilon B}{m_{A'}}\right)^2\sin^2\left(\frac{m_{A'}^2 L}{4\omega}\right)")
    
    st.caption("Tony Ford Model | v5.2")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Stellaris QED Explorer")
st.markdown("*Quantum Vacuum Engineering for Extreme Astrophysical Environments*")
st.markdown("---")

B_ratio = B_surface / B_crit
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("B / B_crit", f"{B_ratio:.2e}")
with col2:
    st.metric("Max γ→A' P", f"{(epsilon * B_surface / 1e15)**2:.2e}")
with col3:
    st.metric("Dark Photon Mass", f"{m_dark:.1e} eV")
with col4:
    st.metric("Kerr Spin", f"{a_spin:.3f}")

if B_ratio > 1:
    st.warning(f"⚠️ Super-critical field! B/B_crit = {B_ratio:.2e}")

tab1, tab2, tab3, tab4 = st.tabs(["🌌 Magnetar Field", "⚛️ QED Vacuum", "🕳️ Dark Photons", "🌀 Kerr Geodesics"])

with tab1:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.info(f"""
        **Magnetar Parameters**
        - B_surface = {B_surface:.1e} G
        - B/B_crit = {B_ratio:.2e}
        - Radius = {R_ns} km
        - Inclination = {inclination}°
        """)
    with col_b:
        fig = plot_magnetar_field(B_surface, R_ns, inclination)
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    fig = plot_qed_vacuum(B_ratio)
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"""
        **Conversion Parameters**
        - ε = {epsilon:.1e}
        - m_A' = {m_dark:.1e} eV
        - B = {B_surface:.1e} G
        """)
    with col_b:
        fig = plot_dark_photon_conversion(B_surface, epsilon, m_dark)
        st.pyplot(fig)
        plt.close(fig)

with tab4:
    col_a, col_b = st.columns(2)
    with col_a:
        r_horizon = 1 + np.sqrt(1 - a_spin**2)
        st.info(f"""
        **Kerr Parameters**
        - a/M = {a_spin:.3f}
        - r_horizon = {r_horizon:.3f} M
        """)
    with col_b:
        fig = plot_kerr_geodesic(a_spin)
        st.pyplot(fig)
        plt.close(fig)

st.markdown("---")
st.markdown("⚡ **Stellaris QED Explorer v5.2** | No System Dependencies | Tony Ford Model")
