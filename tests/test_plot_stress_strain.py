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


def test_metrics_table_lists_each_temperature():
    from plot_stress_strain import metrics_table

    table = metrics_table({"300K": {"uts": 9.456, "uts_strain": 0.121, "youngs_modulus": 74.97, "toughness": 0.7994},
                           "900K": {"uts": 6.371, "uts_strain": 0.228, "youngs_modulus": None, "toughness": 0.8044}})
    lines = table.splitlines()
    assert lines[2] == "| 300K | 74.97 | 9.46 | 0.121 | 0.799 |"
    assert lines[3] == "| 900K | n/a | 6.37 | 0.228 | 0.804 |"


def test_extra_temperature_runs_are_picked_up(tmp_path):
    from plot_stress_strain import find_datasets

    curve = "".join(f"{i * 0.005} {i * 0.5}\n" for i in range(12))
    for name in ("stress_strain_900K.txt", "stress_strain_450K.txt", "stress_strain_77K.txt",
                 "stress_strain_notes.txt"):
        (tmp_path / name).write_text(curve)
    assert [d["temp"] for d in find_datasets(str(tmp_path))] == ["77K", "450K", "900K"]
    metrics = analyze_and_plot(save_plot=False, data_dir=str(tmp_path))
    assert list(metrics) == ["77K", "450K", "900K"]
    assert metrics["450K"]["uts"] == pytest.approx(5.5)


def test_toughness_is_the_area_under_the_curve():
    from plot_stress_strain import metrics_table, toughness

    # A triangle: up to 10 GPa at 0.1 strain and back to 0 at 0.2 -> area 1.0 GJ/m^3.
    strain = np.array([0.0, 0.1, 0.2])
    stress = np.array([0.0, 10.0, 0.0])
    assert toughness(strain, stress) == pytest.approx(1.0)
    assert toughness(strain[:1], stress[:1]) == 0.0
    metrics = analyze_and_plot(save_plot=False)
    # The bundled 300 K run: 0.80 GJ/m^3 over its 0.25 strain.
    assert metrics["300K"]["toughness"] == pytest.approx(0.799, abs=0.001)
    assert "Toughness" in metrics_table(metrics)


def test_command_line(tmp_path, capsys):
    from plot_stress_strain import main

    curve = "".join(f"{i * 0.005} {i * 0.5}\n" for i in range(12))
    (tmp_path / "stress_strain_450K.txt").write_text(curve)
    metrics = main(["--data-dir", str(tmp_path), "--no-plot", "--markdown"])
    assert list(metrics) == ["450K"]
    assert not (tmp_path / "stress_strain_comparison.png").exists()
    assert "| 450K |" in capsys.readouterr().out
    for argv in (["--markdwon"], ["--data-dir", str(tmp_path / "missing")]):
        with pytest.raises(SystemExit):
            main(argv)


def test_nan_rows_from_a_blown_up_run_are_dropped(tmp_path):
    # LAMMPS prints nan/-nan/inf once a run goes unstable; float() parses them all.
    path = write(tmp_path, "0.00 0.0\n0.01 1.0\n0.02 nan\n0.03 -nan\n0.04 inf\nnan 2.0\n0.05 1.5\n")
    strain, stress = load_stress_strain(path)
    np.testing.assert_allclose(strain, [0.0, 0.01, 0.05])
    np.testing.assert_allclose(stress, [0.0, 1.0, 1.5])


def test_metrics_stay_finite_with_nan_rows(tmp_path):
    rows = [f"{e:.3f} {100 * e:.4f}" for e in np.arange(0, 0.06, 0.001)]
    rows.insert(20, "0.0195 nan")
    (tmp_path / "stress_strain_300K.txt").write_text("\n".join(rows) + "\n")
    m = analyze_and_plot(save_plot=False, data_dir=str(tmp_path))["300K"]
    assert m["uts"] == pytest.approx(5.9)
    assert m["youngs_modulus"] == pytest.approx(100.0)
    assert np.isfinite(m["toughness"])
