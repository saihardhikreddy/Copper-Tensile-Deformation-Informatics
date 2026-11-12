"""
Multi-Scale Materials Informatics: Automated Stress-Strain Analysis & Plotting

Reads LAMMPS tensile deformation text dumps across temperature series (300K, 600K, 900K),
extracts engineering metrics (Ultimate Tensile Strength and the strain it occurs at, Young's
Modulus, toughness as the area under the curve), and generates publication-quality comparative plots.

    python plot_stress_strain.py              # print the summary and save the plot
    python plot_stress_strain.py --markdown   # also print the metrics as a Markdown table

Every stress_strain_<T>K.txt next to the script is included, so a new run such as
`lmp -in in.cu_tensile.txt -var T 450` is picked up automatically.
"""

import glob
import os
import re
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
                if not (np.isfinite(strain) and np.isfinite(stress)):
                    # LAMMPS prints "nan"/"-nan" once a run blows up; float() accepts them, and a
                    # single NaN made argmax report it as the UTS and turned E and toughness to NaN.
                    continue
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


COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def toughness(strain: np.ndarray, stress: np.ndarray) -> float:
    """Energy absorbed per unit volume: the area under the stress-strain curve (trapezoid rule).

    Stress in GPa times dimensionless strain gives GPa, which is the same as GJ/m^3. It covers the
    whole recorded run (0.25 strain for the bundled ones), so compare runs stretched equally far.
    """
    if len(strain) < 2:
        return 0.0
    return float(np.sum((stress[1:] + stress[:-1]) * np.diff(strain)) / 2.0)


def find_datasets(data_dir: str) -> list:
    """Every stress_strain_<T>K.txt in data_dir, coolest first.

    The LAMMPS script names its output after the temperature (-var T 450 writes
    stress_strain_450K.txt), so any extra run shows up here without editing the code.
    """
    found = []
    for path in glob.glob(os.path.join(data_dir, "stress_strain_*K.txt")):
        match = re.fullmatch(r"stress_strain_(\d+(?:\.\d+)?)K\.txt", os.path.basename(path))
        if match:
            found.append((float(match.group(1)), match.group(1) + "K", os.path.basename(path)))
    found.sort()
    return [{"temp": temp, "file": name, "color": COLORS[i % len(COLORS)]}
            for i, (_, temp, name) in enumerate(found)]


def analyze_and_plot(save_plot: bool = True, data_dir: str = None) -> Dict[str, Any]:
    """Run tensile analysis across thermal sweep and optionally save comparison figure."""
    script_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
    datasets = find_datasets(script_dir)
    if not datasets:
        print(f"No stress_strain_<T>K.txt files found in {script_dir}.")

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
            "toughness": toughness(strain, stress),
        }

        plt.plot(strain, stress, label=label_text, color=item["color"], linewidth=2)
        plt.scatter([uts_strain], [uts], color=item["color"], s=60, zorder=5)

    if not metrics:
        # Nothing loaded (no files, or only empty ones): an empty, unlabelled plot written over
        # the last good stress_strain_comparison.png helps nobody, so keep the old one.
        plt.close(fig)
        if save_plot:
            print("No usable runs, so the comparison plot was not written.")
        return metrics

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


def metrics_table(metrics: Dict[str, Any]) -> str:
    """The metrics as a Markdown table, so a README can quote exactly what the data gives."""
    rows = ["| Temperature | Young's Modulus E (GPa) | UTS (GPa) | Strain at UTS | Toughness (GJ/m³) |",
            "| :--- | :--- | :--- | :--- | :--- |"]
    for temp, m in metrics.items():
        e = "n/a" if m["youngs_modulus"] is None else f"{m['youngs_modulus']:.2f}"
        rows.append(f"| {temp} | {e} | {m['uts']:.2f} | {m['uts_strain']:.3f} | {m['toughness']:.3f} |")
    return "\n".join(rows)


def main(argv=None) -> Dict[str, Any]:
    """Command line: python plot_stress_strain.py [--data-dir DIR] [--no-plot] [--markdown]."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze and plot LAMMPS stress-strain runs.")
    parser.add_argument("--data-dir", help="folder with the stress_strain_<T>K.txt files "
                                           "(default: this script's folder)")
    parser.add_argument("--no-plot", action="store_true", help="print the metrics only, don't write the PNG")
    parser.add_argument("--markdown", action="store_true", help="also print the metrics as a Markdown table")
    args = parser.parse_args(argv)  # a mistyped flag is an error now, not silently ignored
    if args.data_dir and not os.path.isdir(args.data_dir):
        parser.error(f"--data-dir {args.data_dir!r} is not a folder")

    results = analyze_and_plot(save_plot=not args.no_plot, data_dir=args.data_dir)
    if args.markdown:
        print()
        print(metrics_table(results))
    return results


if __name__ == "__main__":
    main()
