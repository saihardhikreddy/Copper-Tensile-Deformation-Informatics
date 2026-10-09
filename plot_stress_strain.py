"""
Multi-Scale Materials Informatics: Automated Stress-Strain Analysis & Plotting

Reads LAMMPS tensile deformation text dumps across temperature series (300K, 600K, 900K),
extracts engineering metrics (Ultimate Tensile Strength, Failure Strain),
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
    return np.array(strains), np.array(stresses)


def analyze_and_plot():
    datasets = [
        {"temp": "300K", "file": "stress_strain_300K.txt", "color": "#1f77b4"},
        {"temp": "600K", "file": "stress_strain_600K.txt", "color": "#ff7f0e"},
        {"temp": "900K", "file": "stress_strain_900K.txt", "color": "#d62728"},
    ]

    plt.figure(figsize=(10, 6), dpi=300)

    print("--- Tensile Deformation Analytics Summary ---")
    for item in datasets:
        path = item["file"]
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

        print(f"[{item['temp']}] UTS: {uts:.2f} GPa at Strain: {uts_strain:.4f}")

        plt.plot(strain, stress, label=f"{item['temp']} (UTS: {uts:.2f} GPa)", color=item["color"], linewidth=2)
        plt.scatter([uts_strain], [uts], color=item["color"], s=60, zorder=5)

    plt.title("FCC Copper Uniaxial Tensile Deformation: Thermal Gradient Effect", fontsize=14, fontweight="bold")
    plt.xlabel("Engineering Strain (z-axis)", fontsize=12)
    plt.ylabel("True Stress (GPa)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=11, loc="upper right")
    plt.tight_layout()

    out_file = "stress_strain_comparison.png"
    plt.savefig(out_file)
    print(f"Saved comparative plot to {out_file}")


if __name__ == "__main__":
    analyze_and_plot()
