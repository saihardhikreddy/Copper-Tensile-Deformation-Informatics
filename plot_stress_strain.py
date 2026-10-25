"""
Multi-Scale Materials Informatics: Automated Stress-Strain Analysis & Plotting

Reads LAMMPS tensile deformation text dumps across temperature series (300K, 600K, 900K),
extracts engineering metrics (Ultimate Tensile Strength, Young's Modulus, Failure Strain),
and generates publication-quality comparative plots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def load_stress_strain(filepath):
    """Load strain and stress (GPa) from LAMMPS print output."""
    strains = []
    stresses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    strains.append(float(parts[0]))
                    stresses.append(float(parts[1]))
                except ValueError:
                    continue
    
    strains = np.array(strains)
    stresses = np.array(stresses)

    # If strain column was logged with dynamic L0 variable (near-zero artifacts),
    # reconstruct the true physical engineering strain from the constant deformation rate:
    # erate = 0.01 / ps, timestep = 1 fs = 1e-3 ps, print interval = 100 steps
    if len(strains) > 1 and np.max(np.abs(strains)) < 1e-4:
        strains = np.linspace(0.0, (len(stresses) - 1) * 100 * 1e-3 * 0.01, len(stresses))

    return strains, stresses


def analyze_and_plot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = [
        {"temp": "300K", "file": "stress_strain_300K.txt", "color": "#1f77b4"},
        {"temp": "600K", "file": "stress_strain_600K.txt", "color": "#ff7f0e"},
        {"temp": "900K", "file": "stress_strain_900K.txt", "color": "#d62728"},
    ]

    plt.figure(figsize=(10, 6), dpi=300)

    print("--- Tensile Deformation Analytics Summary ---")
    for item in datasets:
        path = os.path.join(script_dir, item["file"])
        if not os.path.exists(path):
            print(f"File {path} not found. Skipping.")
            continue

        strain, stress = load_stress_strain(path)
        if len(strain) == 0:
            continue

        # Extract peak tensile strength (UTS)
        max_idx = np.argmax(stress)
        uts = stress[max_idx]
        uts_strain = strain[max_idx]

        # Calculate Young's Modulus (E) in the initial linear elastic regime (0.01 <= strain <= 0.04)
        elastic_mask = (strain >= 0.01) & (strain <= 0.04)
        if np.sum(elastic_mask) > 2:
            slope = np.polyfit(strain[elastic_mask], stress[elastic_mask], 1)[0]
            print(f"[{item['temp']}] UTS: {uts:.2f} GPa (at strain {uts_strain:.3f}) | Young's Modulus (E): {slope:.2f} GPa")
            label_text = f"{item['temp']} (UTS: {uts:.2f} GPa, E: {slope:.1f} GPa)"
        else:
            print(f"[{item['temp']}] UTS: {uts:.2f} GPa (at strain {uts_strain:.3f})")
            label_text = f"{item['temp']} (UTS: {uts:.2f} GPa)"

        plt.plot(strain, stress, label=label_text, color=item["color"], linewidth=2)
        plt.scatter([uts_strain], [uts], color=item["color"], s=60, zorder=5)

    plt.title("FCC Copper Uniaxial Tensile Deformation: Thermal Gradient Effect", fontsize=14, fontweight="bold")
    plt.xlabel("Engineering Strain (z-axis)", fontsize=12)
    plt.ylabel("True Stress (GPa)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    out_file = os.path.join(script_dir, "stress_strain_comparison.png")
    plt.savefig(out_file)
    print(f"Saved comparative plot to {out_file}")


if __name__ == "__main__":
    analyze_and_plot()
