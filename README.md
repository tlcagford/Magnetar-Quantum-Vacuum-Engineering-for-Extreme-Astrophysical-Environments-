[![License: Dual License](https://img.shields.io/badge/license-Dual--License-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red.svg)](https://streamlit.io)
[![arXiv](https://img.shields.io/badge/arXiv-2503.12345-b31b1b.svg)](https://arxiv.org)


# Quantum Cosmology & Astrophysics Unified Suite (QCAUS)

A collection of four interconnected open‑source projects that explore the quantum nature of the universe – from the early cosmos to extreme astrophysical environments.
## 📸 Live Demo

The application is deployed on Streamlit Cloud:

Live App test now: https://w7xsyujq7jzxrftum8xzcj.streamlit.app/

- **QCI AstroEntangle Refiner** – FDM soliton physics & image processing  
- **Magnetar QED Explorer** – Magnetar fields, dark photons & vacuum QED  
- **Primordial Photon–DarkPhoton Entanglement** – Von Neumann evolution in an expanding universe  
- **QCIS (Quantum Cosmology Integration Suite)** – Quantum‑corrected cosmological perturbations

---

## 🔭 Overview

These four projects form a complete computational framework for **quantum‑inspired astrophysics**. Together they enable:

- **Image analysis** of galaxy clusters (Abell, Bullet, etc.) using Fuzzy Dark Matter (FDM) soliton overlays.
- **Simulation of magnetar magnetospheres** with strong‑field QED, dark photon production, and Kerr geodesics.
- **First‑principles modeling** of photon–dark photon entanglement in the expanding universe.
- **Quantum‑corrected power spectra** and transfer functions for cosmological parameter inference.

All tools are designed to be accessible, interactive, and fully open for academic research.

---

## 📦 The Four Projects

### 1. [QCI AstroEntangle Refiner](https://github.com/tlcagford/QCI_AstroEntangle_Refiner)
*FDM soliton physics & image processing*

- Upload FITS or standard images of galaxy clusters.
- Apply Photon‑Dark‑Photon (PDP) entanglement to produce overlays showing FDM soliton cores, dark photon wave patterns, and dark matter density.
- Annotated side‑by‑side comparison with scale bar, north indicator, and physics formulas.
- Export results as PNG and metadata JSON.

### 2. [Magnetar QED Explorer](https://github.com/tlcagford/Magnetar-Quantum-Vacuum-Engineering-for-Extreme-Astrophysical-Environments-)
*Magnetar fields, dark photons & vacuum QED*

- Interactive simulation of magnetar dipole fields, Euler‑Heisenberg vacuum polarisation, and dark photon conversion.
- Visualise Kerr spacetime null geodesics.
- Drag‑and‑drop image upload or preloaded examples (Bullet Cluster, Crab Nebula, etc.) to see PDP‑enhanced overlays.
- All plots downloadable.

### 3. [Primordial Photon–DarkPhoton Entanglement](https://github.com/tlcagford/Primordial-Photon-DarkPhoton-Entanglement)
*Von Neumann evolution in an expanding universe*

- Solves the von Neumann equation \(i\partial_t\rho = [H_{\text{eff}},\rho]\) for a coupled photon‑dark photon system.
- Computes entanglement entropy evolution and mixing probabilities.
- Provides full theoretical framework and numerical validation scripts.

### 4. [QCIS – Quantum Cosmology Integration Suite](https://github.com/tlcagford/Quantum-Cosmology-Integration-Suite-QCIS-)
*Quantum‑corrected cosmological perturbations*

- Implements quantum‑corrected Mukhanov‑Sasaki equations.
- Calculates matter and tensor power spectra with backreaction from quantum fields.
- Validates against Planck 2018 data and provides Bayesian evidence tools.

---

## 🚀 Features at a Glance

| Project | Key Physics | Interactive | Input | Output |
|---------|-------------|-------------|-------|--------|
| **QCI AstroEntangle Refiner** | FDM soliton, PDP mixing | ✅ sliders | FITS / images | Annotated PNG, metadata |
| **Magnetar QED Explorer** | Dipole field, dark photon conversion, Kerr | ✅ sliders, preloads | FITS / images | Plots, PNG exports |
| **Primordial Entanglement** | von Neumann evolution, mixing | ✅ (via notebooks) | parameters | Evolution plots, matrices |
| **QCIS** | Quantum‑corrected power spectra | ✅ (via scripts) | cosmological parameters | Power spectra, transfer functions |

---

## 📥 Installation

Each project can be installed separately, but they share a common set of dependencies.  
To use the interactive apps, install the required packages and launch Streamlit:

```bash
# Clone the desired repository
git clone https://github.com/tlcagford/QCI_AstroEntangle_Refiner.git
cd QCI_AstroEntangle_Refiner

# Install dependencies (common for all apps)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The `requirements.txt` for each app typically includes:

```
streamlit>=1.28.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
astropy>=5.3.0
scikit-image>=0.21.0
Pillow>=10.0.0
pandas>=2.0.0
```

For the **Primordial Entanglement** and **QCIS** libraries, you may need to run their Jupyter notebooks or Python scripts directly (no Streamlit UI).

---

## 📚 Physics References

Each project is grounded in rigorous theoretical frameworks:

- **FDM Soliton** – derived from the Schrödinger–Poisson system:
  \[
  \rho(r) \propto \left[\frac{\sin(kr)}{kr}\right]^2
  \]
- **Dark Photon Conversion** – kinetic mixing Lagrangian:
  \[
  \mathcal{L}_{\text{mix}} = \frac{\varepsilon}{2} F_{\mu\nu} F'^{\mu\nu}
  \]
- **Von Neumann Evolution** – density matrix evolution in an expanding universe:
  \[
  i\partial_t\rho = [H_{\text{eff}},\rho]
  \]
- **Quantum‑Corrected Power Spectrum**:
  \[
  P(k) = P_{\Lambda\text{CDM}}(k) \times \left(1 + f_{\text{NL}}\left(\frac{k}{k_0}\right)^{n_q}\right)
  \]

Full derivations and references are provided in the individual repositories.

---

## 🧪 Quick Start Examples

### QCI AstroEntangle Refiner
1. Launch the app: `streamlit run app.py`
2. Select a preloaded example (Bullet Cluster, Abell 1689, …) or upload your own FITS/image.
3. Adjust Ω (entanglement strength) and fringe scale.
4. Download the annotated comparison and individual physics components.

### Magnetar QED Explorer
1. Launch the app.
2. Use the preloaded examples or drag‑and‑drop an image.
3. Explore the three magnetar physics tabs (Magnetic Field, Dark Photons, Kerr Spacetime).
4. Export plots and processed images.

### Primordial Entanglement (theoretical)
```python
from primordial_entanglement import solve_von_neumann
mixing, entropy, t = solve_von_neumann(omega=0.7, m_dark=1e-9)
```

### QCIS (cosmological spectra)
```python
from qcis import qcis_power_spectrum
k, P_quantum = qcis_power_spectrum(omega_m=0.3, omega_b=0.05, h=0.7, f_nl=1.0)
```

---

## 📄 Citation

If you use any of these projects in your research, please cite the respective repository and the author’s work:

```bibtex
@software{Ford2025QCI,
  author = {Ford, Tony E.},
  title = {QCI AstroEntangle Refiner},
  year = {2025},
  url = {https://github.com/tlcagford/QCI_AstroEntangle_Refiner}
}

@software{Ford2025Magnetar,
  author = {Ford, Tony E.},
  title = {Magnetar QED Explorer},
  year = {2025},
  url = {https://github.com/tlcagford/Magnetar-Quantum-Vacuum-Engineering-for-Extreme-Astrophysical-Environments-}
}

@software{Ford2025Primordial,
  author = {Ford, Tony E.},
  title = {Primordial Photon–DarkPhoton Entanglement},
  year = {2025},
  url = {https://github.com/tlcagford/Primordial-Photon-DarkPhoton-Entanglement}
}

@software{Ford2025QCIS,
  author = {Ford, Tony E.},
  title = {QCIS – Quantum Cosmology Integration Suite},
  year = {2025},
  url = {https://github.com/tlcagford/Quantum-Cosmology-Integration-Suite-QCIS-}
}
```

---

## 📜 License

All four projects are released under a **Dual License**:

- **Academic / Non‑Commercial Use:** Free for research, education, and personal projects.
- **Commercial Use:** Requires a separate license. Please contact the author for details.

See the `LICENSE` file in each repository for full terms.

---

## 📧 Contact

**Tony E. Ford**  
Independent Researcher / Astrophysics & Quantum Systems  
Email: tlcagford@gmail.com  
GitHub: [@tlcagford](https://github.com/tlcagford)

---

## 🙏 Acknowledgments

- NASA/ESA Hubble Space Telescope & JWST for public FITS data.
- The FDM, QED, and cosmology communities for foundational research.
- Streamlit, NumPy, SciPy, Matplotlib, and all other open‑source libraries that made these tools possible.

---

*“Exploring the quantum nature of the universe – from the first moments to the most extreme objects.”*
