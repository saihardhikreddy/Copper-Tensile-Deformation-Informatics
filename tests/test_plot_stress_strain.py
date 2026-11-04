"""Tests for the LAMMPS stress-strain loader and the thermal-sweep analysis."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_stress_strain import analyze_and_plot, load_stress_strain  # noqa: E402


def write(tmp_path, text):
    path = tmp_path / "ss.txt"
    path.write_text(text)
    return str(path)


def test_skips_comments_blank_and_malformed_lines(tmp_path):
    path = write(tmp_path, "# header\n\n0.00 0.0\n0.01 nan-ish\n0.02 1.5\nonly_one\n0.03 2.0\n")
    strain, stress = load_stress_strain(path)
    np.testing.assert_allclose(strain, [0.0, 0.02, 0.03])
    np.testing.assert_allclose(stress, [0.0, 1.5, 2.0])


def test_rebuilds_strain_when_logged_column_is_zero(tmp_path):
    # LAMMPS printed ~1e-16 noise instead of strain; the loader rebuilds it from the
    # deformation rate (0.01/ps, 1 fs steps, one row every 100 steps -> 0.001 per row).
    path = write(tmp_path, "0 0.05\n-5e-16 0.10\n1e-15 0.20\n3e-16 0.30\n")
    strain, stress = load_stress_strain(path)
    np.testing.assert_allclose(strain, [0.0, 0.001, 0.002, 0.003])
    np.testing.assert_allclose(stress, [0.05, 0.10, 0.20, 0.30])


def test_real_strain_column_is_kept(tmp_path):
    strain, _ = load_stress_strain(write(tmp_path, "0.0 0\n0.5 1\n"))
    np.testing.assert_allclose(strain, [0.0, 0.5])


def test_analysis_of_bundled_runs():
    metrics = analyze_and_plot(save_plot=False)
    assert set(metrics) == {"300K", "600K", "900K"}
    # Copper gets weaker and softer as it heats up.
    assert metrics["300K"]["uts"] > metrics["600K"]["uts"] > metrics["900K"]["uts"]
    moduli = [metrics[t]["youngs_modulus"] for t in ("300K", "600K", "900K")]
    assert moduli == sorted(moduli, reverse=True)
    assert metrics["300K"]["uts"] == pytest.approx(9.46, abs=0.01)
