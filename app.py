# ── MAGNETAR PHYSICS TABS (FIXED WITH st.image) ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Magnetar Physics")

tab1, tab2, tab3 = st.tabs(["🌌 Magnetic Field", "🕳️ Dark Photons", "🌀 Kerr Spacetime"])

def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image for reliable display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black', dpi=100)
    buf.seek(0)
    return Image.open(buf)

# Tab 1: Magnetar Field
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
    
    # Convert to PIL and display
    st.image(fig_to_pil(fig1), use_container_width=True)
    
    # Download button
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight', facecolor='black')
    buf1.seek(0)
    st.download_button("📥 Download Field Plot", buf1, "magnetar_field.png", use_container_width=True)
    plt.close(fig1)

# Tab 2: Dark Photons (fixed y-scale)
with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0a0a1a')
    ax2.set_facecolor('#0a0a1a')
    
    L = np.logspace(-2, 6, 500)  # extended range to see oscillations
    if m_dark <= 0:
        P = (epsilon * B_surface / 1e15)**2 * np.ones_like(L)
    else:
        hbar_ev_s = 6.582e-16
        c_km_s = 3e5
        conv_len = 4 * 1e18 * hbar_ev_s * c_km_s / (m_dark**2)
        # The conversion length is huge for small m_dark, so sin² oscillates rapidly.
        # For plotting, we average over fast oscillations or show the envelope.
        # Here we simply plot the probability.
        P = (epsilon * B_surface / 1e15)**2 * np.sin(np.pi * L / conv_len)**2
    P = np.clip(P, 1e-30, 1)  # avoid zero for log scale
    
    # Use log scale on y-axis to see tiny probabilities
    ax2.semilogx(L, P, '#00aaff', linewidth=2.5)
    ax2.axhline(y=(epsilon * B_surface / 1e15)**2, color='#ff8888', linestyle='--', 
               label=f'Max P = {(epsilon * B_surface / 1e15)**2:.2e}')
    ax2.set_xlabel('Length (km)', color='white')
    ax2.set_ylabel('P(γ→A\')', color='white')
    ax2.set_title('Dark Photon Conversion', color='#00aaff')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    ax2.tick_params(colors='white')
    ax2.set_yscale('log')  # LOG SCALE – crucial for tiny probabilities
    
    st.image(fig_to_pil(fig2), use_container_width=True)
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight', facecolor='black')
    buf2.seek(0)
    st.download_button("📥 Download Conversion Plot", buf2, "dark_photon_conversion.png", use_container_width=True)
    plt.close(fig2)
    st.caption(f"ε = {epsilon:.1e} | m' = {m_dark:.1e} eV | B = {B_surface:.1e} G")

# Tab 3: Kerr Geodesics
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
    
    st.image(fig_to_pil(fig3), use_container_width=True)
    
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', bbox_inches='tight', facecolor='black')
    buf3.seek(0)
    st.download_button("📥 Download Geodesic Plot", buf3, "kerr_geodesic.png", use_container_width=True)
    plt.close(fig3)
    st.caption(f"Event Horizon: r_+ = {r_horizon:.3f} M")
