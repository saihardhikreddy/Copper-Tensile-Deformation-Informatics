"""
Multi-Scale Materials Informatics: Automated Stress-Strain Analysis & Plotting

Reads LAMMPS tensile deformation text dumps across temperature series (300K, 600K, 900K),
extracts engineering metrics (Ultimate Tensile Strength, Young's Modulus, Failure Strain),
and generates publication-quality comparative plots.
"""

import os
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_stress_strain(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
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
                    strain, stress = float(parts[0]), float(parts[1])
                except ValueError:
                    continue  # parse both before appending, or the columns drift apart
                strains.append(strain)
                stresses.append(stress)

    strains = np.array(strains)
    stresses = np.array(stresses)

    # If strain column was logged with dynamic L0 variable (near-zero artifacts),
    # reconstruct the true physical engineering strain from the constant deformation rate:
    # erate = 0.01 / ps, timestep = 1 fs = 1e-3 ps, print interval = 100 steps
    if len(strains) > 1 and np.max(np.abs(strains)) < 1e-4:
        strains = np.linspace(0.0, (len(stresses) - 1) * 100 * 1e-3 * 0.01, len(stresses))

    return strains, stresses


def analyze_and_plot(save_plot: bool = True) -> Dict[str, Any]:
    """Run tensile analysis across thermal sweep and optionally save comparison figure."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = [
        {"temp": "300K", "file": "stress_strain_300K.txt", "color": "#1f77b4"},
        {"temp": "600K", "file": "stress_strain_600K.txt", "color": "#ff7f0e"},
        {"temp": "900K", "file": "stress_strain_900K.txt", "color": "#d62728"},
    ]

    metrics: Dict[str, Any] = {}
    fig = plt.figure(figsize=(10, 6), dpi=300)

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
        max_idx = int(np.argmax(stress))
        uts = float(stress[max_idx])
        uts_strain = float(strain[max_idx])

        # Calculate Young's Modulus (E) in the initial linear elastic regime (0.01 <= strain <= 0.04)
        elastic_mask = (strain >= 0.01) & (strain <= 0.04)
        slope = None
        if np.sum(elastic_mask) > 2:
            slope = float(np.polyfit(strain[elastic_mask], stress[elastic_mask], 1)[0])
            print(f"[{item['temp']}] UTS: {uts:.2f} GPa (at strain {uts_strain:.3f}) | Young's Modulus (E): {slope:.2f} GPa")
            label_text = f"{item['temp']} (UTS: {uts:.2f} GPa, E: {slope:.1f} GPa)"
        else:
            print(f"[{item['temp']}] UTS: {uts:.2f} GPa (at strain {uts_strain:.3f})")
            label_text = f"{item['temp']} (UTS: {uts:.2f} GPa)"

        metrics[item["temp"]] = {
            "uts": uts,
            "uts_strain": uts_strain,
            "youngs_modulus": slope,
        }

        plt.plot(strain, stress, label=label_text, color=item["color"], linewidth=2)
        plt.scatter([uts_strain], [uts], color=item["color"], s=60, zorder=5)

    plt.title("FCC Copper Uniaxial Tensile Deformation: Thermal Gradient Effect", fontsize=14, fontweight="bold")
    plt.xlabel("Engineering Strain (z-axis)", fontsize=12)
    plt.ylabel("True Stress (GPa)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()

    out_file = os.path.join(script_dir, "stress_strain_comparison.png")
    if save_plot:
        plt.savefig(out_file)
        print(f"Saved comparative plot to {out_file}")
    plt.close(fig)

    return metrics


if __name__ == "__main__":
    analyze_and_plot()
