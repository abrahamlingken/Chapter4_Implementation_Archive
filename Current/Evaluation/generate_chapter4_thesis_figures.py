#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate thesis-ready Chapter 4 figures for 3D steady-state RTE."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams
from scipy.special import roots_legendre

SCRIPT_PATH = Path(__file__).resolve()
EVAL_DIR = SCRIPT_PATH.parent
CHAPTER_ROOT = EVAL_DIR.parent
REPO_ROOT = CHAPTER_ROOT.parent

for extra_path in (
    CHAPTER_ROOT,
    CHAPTER_ROOT / "Core",
    CHAPTER_ROOT / "Models",
    CHAPTER_ROOT / "EquationModels",
    REPO_ROOT,
    REPO_ROOT.parent,
    REPO_ROOT.parent / "Core",
    REPO_ROOT.parent / "Models",
    REPO_ROOT.parent / "EquationModels",
    REPO_ROOT.parent / "Chapter4_3D_SteadyState",
    REPO_ROOT.parent / "Chapter4_3D_SteadyState" / "Core",
    REPO_ROOT.parent / "Chapter4_3D_SteadyState" / "Models",
    REPO_ROOT.parent / "Chapter4_3D_SteadyState" / "EquationModels",
):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from EquationModels.RadTrans3D_Complex import RadTrans3D_Physics  # noqa: E402

CASE_METADATA = {
    "A": {
        "name": "Case 3D-A",
        "folder": "Results_3D_CaseA",
        "kappa": 5.0,
        "sigma_s": 0.0,
        "g": 0.0,
        "description": "Pure absorption",
    },
    "B": {
        "name": "Case 3D-B",
        "folder": "Results_3D_CaseB",
        "kappa": 0.5,
        "sigma_s": 4.5,
        "g": 0.0,
        "description": "Isotropic scattering",
    },
    "C": {
        "name": "Case 3D-C",
        "folder": "Results_3D_CaseC",
        "kappa": 0.5,
        "sigma_s": 4.5,
        "g": 0.8,
        "description": "Strong forward scattering",
    },
}

LINE_COLORS = {
    "pinn": "#1F77B4",
    "ref": "#D62728",
    "error": "#4D4D4D",
    "annotation": "#6E6E6E",
}

CENTER = np.array([0.5, 0.5, 0.5], dtype=np.float64)
SOURCE_RADIUS = 0.2
MAIN_CMAP = "jet"
ERROR_CMAP = "Reds"
THESIS_FONT_SIZE = 10.5
THESIS_SMALL_FONT_SIZE = 9.0


class MissingArtifactError(RuntimeError):
    """Raised when a required model or benchmark file is not found."""


def set_plot_style(font_size: float = THESIS_FONT_SIZE) -> None:
    """Apply thesis-oriented plotting style with uniform per-figure font sizing."""
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "SimSun",
                "STSong",
                "Songti SC",
                "DejaVu Serif",
            ],
            "font.sans-serif": ["SimSun", "Times New Roman", "DejaVu Sans"],
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )


def add_panel_label(ax, label: str) -> None:
    """Add a panel label at the upper-left corner."""
    label_fontsize = float(rcParams.get("font.size", THESIS_FONT_SIZE))
    if hasattr(ax, "text2D"):
        ax.text2D(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=label_fontsize,
            fontweight="bold",
        )
        return
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=label_fontsize,
        fontweight="bold",
    )


def plot_centerline(
    ax,
    x: np.ndarray,
    g_pinn: np.ndarray,
    g_ref: np.ndarray,
    ref_label: str,
    pinn_label: str = "PINN Prediction",
    error: np.ndarray | None = None,
    legend_loc: str = "best",
) -> None:
    """Plot a thesis-style centerline comparison."""
    ax.plot(
        x,
        g_pinn,
        color=LINE_COLORS["pinn"],
        linestyle="-",
        linewidth=2.2,
        label=pinn_label,
    )
    ax.plot(
        x,
        g_ref,
        color=LINE_COLORS["ref"],
        linestyle="--",
        linewidth=2.2,
        label=ref_label,
    )
    if error is not None:
        ax.plot(
            x,
            error,
            color=LINE_COLORS["error"],
            linestyle=":",
            linewidth=1.8,
            label="Absolute Error",
        )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$G(x, 0.5, 0.5)$")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True)
    ax.legend(frameon=False, loc=legend_loc)


def draw_plane_panel(
    ax,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field: np.ndarray,
    levels: np.ndarray,
    cmap: str,
    title: str,
    panel_label: str,
):
    """Render a single filled-contour plane panel with shared styling."""
    image = ax.contourf(x_grid, y_grid, field, levels=levels, cmap=cmap, extend="both")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    ax.set_title(title)
    add_panel_label(ax, panel_label)
    return image


def plot_plane_distribution(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    fields: list[np.ndarray],
    titles: list[str],
    panel_labels: list[str],
    stem: str,
    outdir: Path,
    field_roles: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Plot one or more mid-plane contour panels with consistent thesis styling."""
    if field_roles is None:
        field_roles = ["main"] * len(fields)

    set_plot_style(THESIS_SMALL_FONT_SIZE)
    if figsize is None:
        figsize = (5.8 * len(fields), 4.6)
    fig, axes = plt.subplots(1, len(fields), figsize=figsize, constrained_layout=True)
    axes_arr = np.atleast_1d(axes)

    main_fields = [field for field, role in zip(fields, field_roles) if role == "main"]
    err_fields = [field for field, role in zip(fields, field_roles) if role == "error"]
    main_levels = np.linspace(
        min(float(np.min(field)) for field in main_fields),
        max(float(np.max(field)) for field in main_fields),
        120,
    )
    err_levels = (
        np.linspace(0.0, max(float(np.max(field)) for field in err_fields), 120)
        if err_fields
        else None
    )

    main_axes = []
    err_axes = []
    main_im = None
    err_im = None
    for ax, field, title, panel_label, role in zip(
        axes_arr, fields, titles, panel_labels, field_roles
    ):
        if role == "error":
            err_im = draw_plane_panel(
                ax,
                x_grid,
                y_grid,
                field,
                err_levels,
                ERROR_CMAP,
                title,
                panel_label,
            )
            err_axes.append(ax)
        else:
            main_im = draw_plane_panel(
                ax,
                x_grid,
                y_grid,
                field,
                main_levels,
                MAIN_CMAP,
                title,
                panel_label,
            )
            main_axes.append(ax)

    if main_im is not None:
        cbar_main = fig.colorbar(main_im, ax=main_axes, fraction=0.035, pad=0.02)
        cbar_main.set_label(r"Incident Radiation $G$")
    if err_im is not None:
        cbar_err = fig.colorbar(err_im, ax=err_axes, fraction=0.05, pad=0.02)
        cbar_err.set_label(r"$|G_{\mathrm{PINN}} - G_{\mathrm{ref}}|$")

    outputs = save_figure(fig, stem, outdir)
    metrics = {
        "main_min": float(min(np.min(field) for field in main_fields)),
        "main_max": float(max(np.max(field) for field in main_fields)),
    }
    if err_fields:
        metrics["max_abs_error"] = float(max(np.max(field) for field in err_fields))
    return outputs, metrics


def plot_3d_visualization(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    field: np.ndarray,
    stem: str,
    outdir: Path,
    percentile: float = 70.0,
    max_points: int = 18000,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Render a 3D scatter visualization of the strongest incident-radiation regions."""
    set_plot_style(THESIS_SMALL_FONT_SIZE)
    x_flat = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_grid, dtype=np.float64).reshape(-1)
    z_flat = np.asarray(z_grid, dtype=np.float64).reshape(-1)
    g_flat = np.asarray(field, dtype=np.float64).reshape(-1)

    positive_values = g_flat[g_flat > 0.0]
    threshold = (
        float(np.percentile(positive_values, percentile))
        if positive_values.size
        else float(np.max(g_flat))
    )
    mask = g_flat >= threshold
    selected_indices = np.flatnonzero(mask)
    if selected_indices.size > max_points:
        step = max(1, selected_indices.size // max_points)
        selected_indices = selected_indices[::step]

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    scatter = ax.scatter(
        x_flat[selected_indices],
        y_flat[selected_indices],
        z_flat[selected_indices],
        c=g_flat[selected_indices],
        cmap=MAIN_CMAP,
        s=10,
        alpha=0.55,
        linewidths=0.0,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_title("PINN Prediction")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=24, azim=-52)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.08)
    cbar.set_label(r"Incident Radiation $G$")

    outputs = save_figure(fig, stem, outdir)
    metrics = {
        "threshold_percentile": percentile,
        "threshold_value": threshold,
        "points_rendered": int(selected_indices.size),
    }
    return outputs, metrics


def save_figure(
    fig: plt.Figure, stem: str, outdir: Path, save_pdf: bool = True
) -> dict[str, str]:
    """Save a figure to PNG and optionally PDF."""
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{stem}.png"
    try:
        fig.tight_layout()
    except RuntimeError:
        pass
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    outputs: dict[str, str] = {"png": str(png_path)}
    if save_pdf:
        pdf_path = outdir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs["pdf"] = str(pdf_path)
    plt.close(fig)
    return outputs


def save_npz(outdir: Path, name: str, **arrays: Any) -> str:
    """Persist figure data as npz for reproducibility."""
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / name
    np.savez(npz_path, **arrays)
    return str(npz_path)


def resolve_outdir(outdir: str | Path) -> Path:
    """Resolve the output directory relative to the repository root."""
    outdir_path = Path(outdir)
    if not outdir_path.is_absolute():
        outdir_path = REPO_ROOT / outdir_path
    outdir_path.mkdir(parents=True, exist_ok=True)
    return outdir_path


def cell_centered_coords(nx: int) -> np.ndarray:
    """Return cell-centered coordinates for an nx-grid on [0, 1]."""
    return np.linspace(0.5 / nx, 1.0 - 0.5 / nx, nx)


def direction_to_angles(direction: tuple[float, float, float]) -> tuple[float, float]:
    """Convert Cartesian direction vector to (theta, phi)."""
    sx, sy, sz = direction
    vec = np.array([sx, sy, sz], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Direction vector cannot be zero.")
    vec = vec / norm
    theta = float(np.arccos(np.clip(vec[2], -1.0, 1.0)))
    phi = float(np.mod(np.arctan2(vec[1], vec[0]), 2.0 * np.pi))
    return theta, phi


def find_existing_path(candidates: list[Path], label: str) -> Path:
    """Return the first existing candidate path or raise a friendly error."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise MissingArtifactError(
        f"{label} not found. Checked:\n{joined}\n"
        "Please confirm the required model or FMC benchmark has been generated."
    )


def case_a_archive_candidates(filename: str) -> list[Path]:
    """Candidate validated Case A archive data files bundled with this archive."""
    return [
        REPO_ROOT / "Artifacts" / "Figures_3D_Validation" / filename,
        CHAPTER_ROOT / "Artifacts" / "Figures_3D_Validation" / filename,
    ]


def load_case_a_centerline_archive() -> dict[str, Any]:
    """Load validated Case A centerline data from the archive."""
    path = find_existing_path(
        case_a_archive_candidates("G_CaseA_Centerline_Data.npz"),
        "Validated Case A centerline archive",
    )
    data = np.load(path)
    return {
        "path": str(path),
        "x": np.asarray(data["x"], dtype=np.float64),
        "G_ref": np.asarray(data["G_exact"], dtype=np.float64),
        "G_pinn": np.asarray(data["G_pinn"], dtype=np.float64),
        "abs_err": np.asarray(data["error"], dtype=np.float64),
        "rel_err": np.asarray(data["rel_error"], dtype=np.float64),
    }


def load_case_a_plane_archive() -> dict[str, Any]:
    """Load validated Case A mid-plane data from the archive."""
    path = find_existing_path(
        case_a_archive_candidates("G_CaseA_2D_Data.npz"),
        "Validated Case A mid-plane archive",
    )
    data = np.load(path)
    return {
        "path": str(path),
        "X": np.asarray(data["X"], dtype=np.float64),
        "Y": np.asarray(data["Y"], dtype=np.float64),
        "G_ref": np.asarray(data["G_exact"], dtype=np.float64),
        "G_pinn": np.asarray(data["G_pinn"], dtype=np.float64),
        "G_err": np.asarray(data["Error"], dtype=np.float64),
    }


def model_candidates(case_key: str) -> list[Path]:
    """Candidate model locations for a case."""
    folder = CASE_METADATA[case_key]["folder"]
    return [
        REPO_ROOT.parent / folder / "model.pkl",
        REPO_ROOT / folder / "model.pkl",
        CHAPTER_ROOT / folder / "model.pkl",
        CHAPTER_ROOT / "Results" / folder / "model.pkl",
    ]


def history_candidates(case_key: str) -> list[Path]:
    """Candidate training-history locations for a case."""
    folder = CASE_METADATA[case_key]["folder"]
    return [
        REPO_ROOT.parent / folder / "training_history.json",
        REPO_ROOT / folder / "training_history.json",
        CHAPTER_ROOT / folder / "training_history.json",
        CHAPTER_ROOT / "Results" / folder / "training_history.json",
    ]


def fmc_candidates(case_key: str) -> list[Path]:
    """Candidate FMC benchmark file locations for cases B and C."""
    if case_key not in {"B", "C"}:
        raise ValueError("FMC candidates are only defined for cases B and C.")
    candidates: list[Path] = []
    rqmc_candidates = sorted(
        (REPO_ROOT / "Artifacts" / "MC3D_Raw_Benchmarks").glob(
            f"FMC_G_3D_Case{case_key}_FIXED_RQMCAgg*.npy"
        )
    )
    candidates.extend(rqmc_candidates)
    stems = [
        f"FMC_G_3D_Case{case_key}_FIXED_HighStats.npy",
        f"FMC_G_3D_Case{case_key}_FIXED_UltraStats.npy",
        f"FMC_G_3D_Case{case_key}_FIXED.npy",
        f"FMC_G_3D_Case{case_key}.npy",
    ]
    for stem in stems:
        candidates.extend(
            [
                REPO_ROOT / "Artifacts" / "MC3D_Raw_Benchmarks" / stem,
                REPO_ROOT.parent / "Solvers" / "MC" / "MC3D_Results" / stem,
                REPO_ROOT / "Solvers" / "MC" / "MC3D_Results" / stem,
                CHAPTER_ROOT / "Solvers" / "MC" / "MC3D_Results" / stem,
                CHAPTER_ROOT / "MC3D_Results" / stem,
                REPO_ROOT.parent / stem,
                REPO_ROOT / stem,
            ]
        )
    return candidates


def validate_fmc_metadata_path(fmc_path: Path, case_key: str) -> None:
    """Reject stale or post-processed FMC files when metadata is available."""
    meta_path = fmc_path.with_name(fmc_path.stem + "_meta.npz")
    if not meta_path.exists():
        return

    meta = np.load(meta_path)
    expected = CASE_METADATA[case_key]

    for key in ("kappa", "sigma_s", "g"):
        if key in meta.files and not np.isclose(float(meta[key]), expected[key]):
            raise MissingArtifactError(
                f"FMC metadata mismatch for case {case_key}: "
                f"{key}={float(meta[key])} but expected {expected[key]}.\n"
                f"File: {fmc_path}\nMetadata: {meta_path}\n"
                "Please regenerate the FMC benchmark with the current Chapter 4 parameters."
            )

    if "postprocessed" in meta.files and bool(meta["postprocessed"]):
        raise MissingArtifactError(
            f"Post-processed FMC benchmark rejected for case {case_key}:\n"
            f"  data: {fmc_path}\n  meta: {meta_path}\n"
            "Please use a raw FMC tally for validation."
        )

    if "symmetry_enforced" in meta.files and bool(meta["symmetry_enforced"]):
        raise MissingArtifactError(
            f"Symmetry-enforced FMC benchmark rejected for case {case_key}:\n"
            f"  data: {fmc_path}\n  meta: {meta_path}\n"
            "Please regenerate without symmetry post-processing."
        )

    if "gaussian_sigma" in meta.files and float(meta["gaussian_sigma"]) > 0.0:
        raise MissingArtifactError(
            f"Gaussian-smoothed FMC benchmark rejected for case {case_key}:\n"
            f"  data: {fmc_path}\n  meta: {meta_path}\n"
            "Please regenerate without Gaussian diffusion."
        )


def source_term(
    x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray
) -> np.ndarray:
    """Spherical source term used in the Chapter 4 benchmark."""
    r = np.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2 + (z - CENTER[2]) ** 2)
    return np.maximum(0.0, 1.0 - 5.0 * r)


def compute_exact_intensity_single(
    x: float,
    y: float,
    z: float,
    theta: float,
    phi: float,
    kappa: float = 5.0,
    num_points: int = 100,
) -> float:
    """Compute a high-accuracy reference intensity for the pure-absorption case."""
    pos = np.array([x, y, z], dtype=np.float64)
    s_vec = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=np.float64,
    )

    t_candidates: list[float] = []
    for idx in range(3):
        if abs(s_vec[idx]) <= 1e-14:
            continue
        t_to_0 = pos[idx] / s_vec[idx]
        t_to_1 = (pos[idx] - 1.0) / s_vec[idx]
        if t_to_0 > 1e-14:
            t_candidates.append(float(t_to_0))
        if t_to_1 > 1e-14:
            t_candidates.append(float(t_to_1))

    if not t_candidates:
        return 0.0

    path_length = min(t_candidates)
    if path_length <= 1e-14:
        return 0.0

    xi, weights = roots_legendre(num_points)
    l_values = 0.5 * path_length * (xi + 1.0)
    quad_weights = 0.5 * path_length * weights

    integrand = np.zeros_like(l_values)
    for index, distance in enumerate(l_values):
        curr_pos = pos - distance * s_vec
        s_val = source_term(curr_pos[0], curr_pos[1], curr_pos[2])
        integrand[index] = s_val * np.exp(-kappa * distance)

    return float(max(0.0, np.sum(integrand * quad_weights)))


class ExactGSolver:
    """High-accuracy quadrature for G in the pure-absorption case."""

    def __init__(self, n_theta: int = 16, n_phi: int = 32):
        self.n_theta = n_theta
        self.n_phi = n_phi
        mu, w_mu = roots_legendre(n_theta)
        self.theta_q = np.arccos(-mu)
        self.w_theta = w_mu
        self.phi_q = (
            np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False) + np.pi / n_phi
        )
        self.w_phi = np.full(n_phi, 2.0 * np.pi / n_phi)
        self.theta_grid, self.phi_grid = np.meshgrid(
            self.theta_q, self.phi_q, indexing="ij"
        )
        self.weights = self.w_theta.reshape(-1, 1) * self.w_phi.reshape(1, -1)

    def compute_G(self, x: float, y: float, z: float) -> float:
        """Compute G at a single point."""
        values = np.zeros((self.n_theta, self.n_phi), dtype=np.float64)
        for i in range(self.n_theta):
            for j in range(self.n_phi):
                values[i, j] = compute_exact_intensity_single(
                    x, y, z, self.theta_grid[i, j], self.phi_grid[i, j], num_points=100
                )
        return float(np.sum(values * self.weights))

    def compute_G_field(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Compute G on a collection of points."""
        flat_x = np.asarray(x).reshape(-1)
        flat_y = np.asarray(y).reshape(-1)
        flat_z = np.asarray(z).reshape(-1)
        field = np.empty_like(flat_x, dtype=np.float64)
        for index in range(flat_x.size):
            field[index] = self.compute_G(
                float(flat_x[index]), float(flat_y[index]), float(flat_z[index])
            )
        return field.reshape(np.shape(x))


def load_model(model_path: Path, device: torch.device):
    """Load a serialized PINN model."""
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


def build_engine(case_key: str, device: torch.device) -> RadTrans3D_Physics:
    """Instantiate the Chapter 4 physics engine for a given case."""
    meta = CASE_METADATA[case_key]
    return RadTrans3D_Physics(
        kappa_val=meta["kappa"],
        sigma_s_val=meta["sigma_s"],
        g_val=meta["g"],
        n_theta=8,
        n_phi=16,
        dev=device,
    )


def infer_G_points(
    engine: RadTrans3D_Physics,
    model,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    """Infer G over a list of spatial points in manageable batches."""
    x_flat = np.asarray(x, dtype=np.float32).reshape(-1)
    y_flat = np.asarray(y, dtype=np.float32).reshape(-1)
    z_flat = np.asarray(z, dtype=np.float32).reshape(-1)
    results = np.empty_like(x_flat, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, x_flat.size, batch_size):
            end = min(start + batch_size, x_flat.size)
            x_t = torch.tensor(
                x_flat[start:end], dtype=torch.float32, device=engine.dev
            )
            y_t = torch.tensor(
                y_flat[start:end], dtype=torch.float32, device=engine.dev
            )
            z_t = torch.tensor(
                z_flat[start:end], dtype=torch.float32, device=engine.dev
            )
            g_t = engine.compute_incident_radiation(x_t, y_t, z_t, model)
            results[start:end] = g_t.detach().cpu().numpy().reshape(-1)

    return results.reshape(np.shape(x))


def infer_intensity_points(
    model, coords: np.ndarray, device: torch.device, batch_size: int = 4096
) -> np.ndarray:
    """Infer intensity for explicit (x, y, z, theta, phi) points."""
    coords = np.asarray(coords, dtype=np.float32)
    output = np.empty(coords.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            end = min(start + batch_size, coords.shape[0])
            batch = torch.tensor(coords[start:end], dtype=torch.float32, device=device)
            output[start:end] = model(batch).detach().cpu().numpy().reshape(-1)
    return output


def process_mc_benchmark(npy_file: Path, nx: int = 50) -> np.ndarray:
    """Extract a raw FMC centerline without hidden smoothing."""
    g_field = np.load(npy_file)
    iy1, iy2 = nx // 2 - 1, nx // 2
    iz1, iz2 = nx // 2 - 1, nx // 2
    return np.mean(g_field[:, iy1 : iy2 + 1, iz1 : iz2 + 1], axis=(1, 2))


def reflection_average_centerline(line: np.ndarray) -> np.ndarray:
    """Apply unbiased reflection averaging about x = 0.5 for symmetric cases."""
    line = np.asarray(line, dtype=np.float64)
    return 0.5 * (line + line[::-1])


def align_node_to_cell_line(values: np.ndarray, nx_target: int) -> np.ndarray:
    """Map node-centered PINN line data onto FMC cell-centered coordinates."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    x_pinn = np.linspace(0.0, 1.0, len(values))
    x_target = cell_centered_coords(nx_target)
    return np.interp(x_target, x_pinn, values)


def centered_slab_bounds(n: int, slab_planes: int) -> tuple[int, int]:
    """Return [start, stop) bounds for a centered slab on an even or odd grid."""
    slab_planes = max(2, int(slab_planes))
    center_left = n // 2 - 1
    center_right = n // 2
    extra = max(0, slab_planes - 2)
    left_extra = extra // 2
    right_extra = extra - left_extra
    start = max(0, center_left - left_extra)
    stop = min(n, center_right + right_extra + 1)
    return start, stop


def extract_midplane_average(field: np.ndarray, axis: int = 2) -> np.ndarray:
    """Average the two central planes to hit the physical mid-plane for even grids."""
    n = field.shape[axis]
    idx1, idx2 = n // 2 - 1, n // 2
    if axis == 0:
        return np.mean(field[idx1 : idx2 + 1, :, :], axis=0)
    if axis == 1:
        return np.mean(field[:, idx1 : idx2 + 1, :], axis=1)
    return np.mean(field[:, :, idx1 : idx2 + 1], axis=2)


def extract_centered_slab_average(
    field: np.ndarray, axis: int = 2, slab_planes: int = 2
) -> np.ndarray:
    """Average a thin centered slab to reduce Monte Carlo plane noise."""
    start, stop = centered_slab_bounds(field.shape[axis], slab_planes)
    if axis == 0:
        return np.mean(field[start:stop, :, :], axis=0)
    if axis == 1:
        return np.mean(field[:, start:stop, :], axis=1)
    return np.mean(field[:, :, start:stop], axis=2)


def reflection_average_plane(field: np.ndarray) -> np.ndarray:
    """Average reflection/permutation-equivalent views for symmetric xy-planes."""
    field = np.asarray(field, dtype=np.float64)
    transforms = np.stack(
        [
            field,
            field[::-1, :],
            field[:, ::-1],
            field[::-1, ::-1],
            field.T,
            field.T[::-1, :],
            field.T[:, ::-1],
            field.T[::-1, ::-1],
        ],
        axis=0,
    )
    return np.mean(transforms, axis=0)


def write_structured_points_vtk(
    path: Path, field: np.ndarray, scalar_name: str = "G"
) -> str:
    """Write a uniform-grid scalar field to a legacy ASCII VTK file."""
    field = np.asarray(field, dtype=np.float64)
    nx, ny, nz = field.shape
    spacing = 1.0 / nx
    origin = 0.5 / nx
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("Chapter 4 scalar field\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_POINTS\n")
        handle.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        handle.write(f"ORIGIN {origin:.8f} {origin:.8f} {origin:.8f}\n")
        handle.write(f"SPACING {spacing:.8f} {spacing:.8f} {spacing:.8f}\n")
        handle.write(f"POINT_DATA {nx * ny * nz}\n")
        handle.write(f"SCALARS {scalar_name} float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for value in field.ravel(order="F"):
            handle.write(f"{float(value):.8e}\n")
    return str(path)


def maybe_import_pyvista():
    """Try importing pyvista; return None if unavailable."""
    try:
        import pyvista as pv

        return pv
    except Exception:
        return None


def render_isosurface_matplotlib(
    ax,
    field: np.ndarray,
    levels: list[float],
    colors: list[str],
    alphas: list[float],
    view: tuple[float, float],
) -> None:
    """Render one or more isosurfaces with marching cubes on a Matplotlib 3D axis."""
    try:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes
    except Exception as exc:
        raise RuntimeError(
            "Isosurface rendering requires mpl_toolkits.mplot3d and scikit-image."
        ) from exc

    nx = field.shape[0]
    spacing = (1.0 / nx, 1.0 / nx, 1.0 / nx)
    origin = 0.5 / nx
    for level, color, alpha in zip(levels, colors, alphas):
        verts, faces, _, _ = marching_cubes(field, level=level, spacing=spacing)
        verts = verts + origin
        mesh = Poly3DCollection(
            verts[faces], alpha=alpha, facecolor=color, edgecolor="none"
        )
        ax.add_collection3d(mesh)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=view[0], azim=view[1])
    ax.grid(False)


def load_training_history(case_key: str) -> tuple[dict[str, Any], list[str]]:
    """Load training history for a case if available."""
    tried = [str(path) for path in history_candidates(case_key)]
    for candidate in history_candidates(case_key):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle), tried
    return {}, tried


def get_resources(
    case_key: str, need_fmc: bool = False, need_model: bool = True
) -> dict[str, Any]:
    """Resolve case resources with friendly error messages."""
    resources = {"case": case_key, "meta": CASE_METADATA[case_key]}
    if need_model:
        resources["model_path"] = find_existing_path(
            model_candidates(case_key), f"Model for case {case_key}"
        )
    resources["training_history"], resources["history_checked"] = load_training_history(
        case_key
    )
    if need_fmc:
        fmc_path = find_existing_path(
            fmc_candidates(case_key), f"FMC benchmark for case {case_key}"
        )
        validate_fmc_metadata_path(fmc_path, case_key)
        resources["fmc_path"] = fmc_path
    return resources


def build_summary_row(
    name: str,
    status: str,
    elapsed: float,
    outputs: dict[str, str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Create a uniform summary record."""
    return {
        "figure": name,
        "status": status,
        "elapsed_s": round(elapsed, 2),
        "outputs": outputs,
        "metrics": metrics,
    }


def figure_fig4_1(
    outdir: Path, resources_a: dict[str, Any], nx: int, device: torch.device
) -> dict[str, Any]:
    """Generate Fig4-1: centerline G for the pure-absorption case."""
    archive = load_case_a_centerline_archive()
    x = archive["x"]
    g_ref = archive["G_ref"]
    g_pinn = archive["G_pinn"]
    abs_err = archive["abs_err"]
    rel_err = archive["rel_err"]

    set_plot_style(THESIS_FONT_SIZE)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    plot_centerline(
        ax,
        x,
        g_pinn,
        g_ref,
        ref_label="Numerical Reference",
        pinn_label="PINN Prediction",
        legend_loc="upper right",
    )
    outputs = save_figure(fig, "Fig4-1_G_centerline_absorption", outdir)

    data_path = save_npz(
        outdir,
        "Fig4-1_G_centerline_absorption_data.npz",
        x=x,
        G_pinn=g_pinn,
        G_ref=g_ref,
        abs_err=abs_err,
        rel_err=rel_err,
        archive_source=np.array(archive["path"]),
        kappa=np.array(resources_a["meta"]["kappa"]),
        case_name=np.array(resources_a["meta"]["name"]),
    )
    outputs["data"] = data_path
    return {
        "outputs": outputs,
        "metrics": {
            "max_abs_error": float(abs_err.max()),
            "mean_abs_error": float(abs_err.mean()),
            "max_rel_error": float(rel_err.max()),
            "data": data_path,
        },
    }


def figure_fig4_2(
    outdir: Path, resources_a: dict[str, Any], nx: int, device: torch.device
) -> dict[str, Any]:
    """Generate Fig4-2: mid-plane G(x, y, 0.5) for the pure-absorption case."""
    archive = load_case_a_plane_archive()
    outputs, metrics = plot_plane_distribution(
        archive["X"],
        archive["Y"],
        [archive["G_ref"], archive["G_pinn"]],
        ["Numerical Reference", "PINN Prediction"],
        ["(a)", "(b)"],
        "Fig4-2_G_plane_absorption",
        outdir,
        field_roles=["main", "main"],
        figsize=(10.8, 4.6),
    )
    data_path = save_npz(
        outdir,
        "Fig4-2_G_plane_absorption_data.npz",
        X=archive["X"],
        Y=archive["Y"],
        G_ref=archive["G_ref"],
        G_pinn=archive["G_pinn"],
        G_err=archive["G_err"],
        archive_source=np.array(archive["path"]),
        kappa=np.array(resources_a["meta"]["kappa"]),
        case_name=np.array(resources_a["meta"]["name"]),
    )
    outputs["data"] = data_path
    metrics["max_abs_error"] = float(np.max(archive["G_err"]))
    metrics["data"] = data_path
    return {"outputs": outputs, "metrics": metrics}


def figure_fig4_3(
    outdir: Path, resources_b: dict[str, Any], nx: int, device: torch.device
) -> dict[str, Any]:
    """Generate Fig4-3: isotropic-scattering centerline G compared with FMC."""
    x = cell_centered_coords(nx)
    y = np.full_like(x, 0.5)
    z = np.full_like(x, 0.5)

    engine_b = build_engine("B", device)
    model_b = load_model(resources_b["model_path"], device)
    g_pinn = infer_G_points(engine_b, model_b, x, y, z).reshape(-1)
    g_fmc = reflection_average_centerline(
        process_mc_benchmark(resources_b["fmc_path"], nx=nx)
    )
    g_pinn_aligned = align_node_to_cell_line(g_pinn, nx)
    abs_err = np.abs(g_pinn_aligned - g_fmc)

    set_plot_style(THESIS_FONT_SIZE)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    plot_centerline(
        ax,
        x,
        g_pinn_aligned,
        g_fmc,
        ref_label="FMC Reference",
        pinn_label="PINN Prediction",
        legend_loc="upper right",
    )
    outputs = save_figure(fig, "Fig4-3_G_centerline_isotropic", outdir)

    data_path = save_npz(
        outdir,
        "Fig4-3_G_centerline_isotropic_data.npz",
        x=x,
        G_pinn=g_pinn_aligned,
        G_ref=g_fmc,
        abs_err=abs_err,
        fmc_source=np.array(str(resources_b["fmc_path"])),
        fmc_variance_reduction=np.array("reflection_average_centerline"),
        case_name=np.array(resources_b["meta"]["name"]),
    )
    outputs["data"] = data_path
    return {
        "outputs": outputs,
        "metrics": {
            "l2_rel_percent": float(
                np.linalg.norm(g_pinn_aligned - g_fmc) / np.linalg.norm(g_fmc) * 100.0
            ),
            "max_abs_error": float(abs_err.max()),
            "data": data_path,
        },
    }


def figure_fig4_4(
    outdir: Path, resources_b: dict[str, Any], nx: int, device: torch.device
) -> dict[str, Any]:
    """Generate Fig4-4: isotropic-scattering mid-plane FMC / PINN / error."""
    slab_planes = 6
    fmc_field = np.load(resources_b["fmc_path"])
    g_fmc = reflection_average_plane(
        extract_centered_slab_average(fmc_field, axis=2, slab_planes=slab_planes)
    )

    coords = cell_centered_coords(nx)
    x_grid, y_grid = np.meshgrid(coords, coords, indexing="xy")
    z_start, z_stop = centered_slab_bounds(nx, slab_planes)
    z_values = coords[z_start:z_stop]

    engine_b = build_engine("B", device)
    model_b = load_model(resources_b["model_path"], device)
    pinn_planes = []
    for z_value in z_values:
        z_grid = np.full_like(x_grid, z_value)
        pinn_planes.append(
            infer_G_points(engine_b, model_b, x_grid, y_grid, z_grid, batch_size=256)
        )
    g_pinn = np.mean(np.stack(pinn_planes, axis=0), axis=0)
    g_err = np.abs(g_pinn - g_fmc)

    outputs, metrics = plot_plane_distribution(
        x_grid,
        y_grid,
        [g_fmc, g_pinn, g_err],
        [
            f"FMC Reference ({slab_planes}-layer avg.)",
            f"PINN Prediction ({slab_planes}-layer avg.)",
            "Absolute Error",
        ],
        ["(a)", "(b)", "(c)"],
        "Fig4-4_G_plane_isotropic",
        outdir,
        field_roles=["main", "main", "error"],
        figsize=(13.4, 4.6),
    )
    data_path = save_npz(
        outdir,
        "Fig4-4_G_plane_isotropic_data.npz",
        X=x_grid,
        Y=y_grid,
        G_fmc=g_fmc,
        G_pinn=g_pinn,
        G_err=g_err,
        z_values=z_values,
        slab_planes=np.array(slab_planes),
        fmc_variance_reduction=np.array("reflection_average_plane"),
    )
    outputs["data"] = data_path
    metrics["l2_rel_percent"] = float(
        np.linalg.norm(g_pinn - g_fmc) / np.linalg.norm(g_fmc) * 100.0
    )
    metrics["slab_planes"] = slab_planes
    metrics["data"] = data_path
    return {"outputs": outputs, "metrics": metrics}


def figure_fig4_5(
    outdir: Path, resources_c: dict[str, Any], nx: int, device: torch.device
) -> dict[str, Any]:
    """Generate Fig4-5: forward-scattering centerline G compared with FMC."""
    x = cell_centered_coords(nx)
    y = np.full_like(x, 0.5)
    z = np.full_like(x, 0.5)

    engine_c = build_engine("C", device)
    model_c = load_model(resources_c["model_path"], device)
    g_pinn = infer_G_points(engine_c, model_c, x, y, z).reshape(-1)
    g_fmc = reflection_average_centerline(
        process_mc_benchmark(resources_c["fmc_path"], nx=nx)
    )
    g_pinn_aligned = align_node_to_cell_line(g_pinn, nx)
    abs_err = np.abs(g_pinn_aligned - g_fmc)

    set_plot_style(THESIS_FONT_SIZE)
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    plot_centerline(
        ax,
        x,
        g_pinn_aligned,
        g_fmc,
        ref_label="FMC Reference",
        pinn_label="PINN Prediction",
        legend_loc="upper right",
    )
    ax.axvspan(
        CENTER[0] + SOURCE_RADIUS,
        1.0,
        color=LINE_COLORS["annotation"],
        alpha=0.10,
        zorder=0,
    )
    ax.text(
        0.82,
        0.92 * max(float(np.max(g_pinn_aligned)), float(np.max(g_fmc))),
        "penetration region",
        color=LINE_COLORS["annotation"],
        ha="center",
        va="center",
    )
    outputs = save_figure(fig, "Fig4-5_G_centerline_forward", outdir)

    data_path = save_npz(
        outdir,
        "Fig4-5_G_centerline_forward_data.npz",
        x=x,
        G_pinn=g_pinn_aligned,
        G_ref=g_fmc,
        abs_err=abs_err,
        fmc_source=np.array(str(resources_c["fmc_path"])),
        fmc_variance_reduction=np.array("reflection_average_centerline"),
        case_name=np.array(resources_c["meta"]["name"]),
    )
    outputs["data"] = data_path
    return {
        "outputs": outputs,
        "metrics": {
            "l2_rel_percent": float(
                np.linalg.norm(g_pinn_aligned - g_fmc) / np.linalg.norm(g_fmc) * 100.0
            ),
            "tail_mean_ratio": float(
                np.mean(g_pinn_aligned[x >= CENTER[0] + SOURCE_RADIUS])
                / np.maximum(np.mean(g_fmc[x >= CENTER[0] + SOURCE_RADIUS]), 1e-12)
            ),
            "data": data_path,
        },
    }


def figure_fig4_6(
    outdir: Path, resources_c: dict[str, Any], nx: int, device: torch.device
) -> dict[str, Any]:
    """Generate Fig4-6: forward-scattering mid-plane contour comparison."""
    slab_planes = 6
    g_fmc = reflection_average_plane(
        extract_centered_slab_average(
            np.load(resources_c["fmc_path"]), axis=2, slab_planes=slab_planes
        )
    )
    coords = cell_centered_coords(nx)
    x_grid, y_grid = np.meshgrid(coords, coords, indexing="xy")
    z_start, z_stop = centered_slab_bounds(nx, slab_planes)
    z_values = coords[z_start:z_stop]

    engine_c = build_engine("C", device)
    model_c = load_model(resources_c["model_path"], device)
    pinn_planes = []
    for z_value in z_values:
        z_grid = np.full_like(x_grid, z_value)
        pinn_planes.append(
            infer_G_points(engine_c, model_c, x_grid, y_grid, z_grid, batch_size=256)
        )
    g_pinn = np.mean(np.stack(pinn_planes, axis=0), axis=0)

    outputs, metrics = plot_plane_distribution(
        x_grid,
        y_grid,
        [g_fmc, g_pinn],
        [
            f"FMC Reference ({slab_planes}-layer avg.)",
            f"PINN Prediction ({slab_planes}-layer avg.)",
        ],
        ["(a)", "(b)"],
        "Fig4-6_G_plane_forward",
        outdir,
        field_roles=["main", "main"],
        figsize=(10.8, 4.6),
    )
    data_path = save_npz(
        outdir,
        "Fig4-6_G_plane_forward_data.npz",
        X=x_grid,
        Y=y_grid,
        G_fmc=g_fmc,
        G_pinn=g_pinn,
        G_err=np.abs(g_pinn - g_fmc),
        z_values=z_values,
        slab_planes=np.array(slab_planes),
        fmc_variance_reduction=np.array("reflection_average_plane"),
    )
    outputs["data"] = data_path
    metrics["l2_rel_percent"] = float(
        np.linalg.norm(g_pinn - g_fmc) / np.linalg.norm(g_fmc) * 100.0
    )
    metrics["slab_planes"] = slab_planes
    metrics["data"] = data_path
    return {"outputs": outputs, "metrics": metrics}


def figure_fig4_7(
    outdir: Path, resources_c: dict[str, Any], nx_3d: int, device: torch.device
) -> dict[str, Any]:
    """Generate Fig4-7: 3D visualization of the forward-scattering PINN field."""
    coords = cell_centered_coords(nx_3d)
    x_grid, y_grid, z_grid = np.meshgrid(coords, coords, coords, indexing="ij")
    engine_c = build_engine("C", device)
    model_c = load_model(resources_c["model_path"], device)
    g_field = infer_G_points(
        engine_c, model_c, x_grid, y_grid, z_grid, batch_size=2048
    )

    outputs, metrics = plot_3d_visualization(
        x_grid,
        y_grid,
        z_grid,
        g_field,
        "Fig4-7_G_3D_visualization",
        outdir,
        percentile=70.0,
        max_points=18000,
    )
    vtk_path = write_structured_points_vtk(outdir / "Fig4-7_G_3D_visualization.vtk", g_field)
    data_path = save_npz(
        outdir,
        "Fig4-7_G_3D_visualization_data.npz",
        X=x_grid,
        Y=y_grid,
        Z=z_grid,
        G_pinn=g_field,
        nx_3d=np.array(nx_3d),
        case_name=np.array(resources_c["meta"]["name"]),
    )
    outputs["vtk"] = vtk_path
    outputs["data"] = data_path
    metrics["data"] = data_path
    return {"outputs": outputs, "metrics": metrics}


def create_figure_manifest(outdir: Path, entries: list[dict[str, Any]]) -> str:
    """Write the figure manifest JSON."""
    manifest_path = outdir / "figure_manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chapter": "Chapter 4",
        "output_directory": str(outdir),
        "figures": entries,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return str(manifest_path)


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    """Print a compact generation summary table."""
    print("\n" + "=" * 108)
    print(f"{'Figure':<34}{'Status':<12}{'Time(s)':<10}{'Key Metric':<22}Output")
    print("=" * 108)
    for row in rows:
        metric_text = "-"
        if row["metrics"]:
            first_key = next(iter(row["metrics"]))
            metric_text = f"{first_key}={row['metrics'][first_key]}"
            if len(metric_text) > 20:
                metric_text = metric_text[:19] + "…"
        output_text = (
            next(iter(row["outputs"].values()), "-") if row["outputs"] else "-"
        )
        print(
            f"{row['figure']:<34}{row['status']:<12}{row['elapsed_s']:<10}{metric_text:<22}{output_text}"
        )
    print("=" * 108)


def figure_plan(case: str) -> list[str]:
    """Resolve which figures to generate."""
    if case == "A":
        return ["Fig4-1", "Fig4-2"]
    if case == "B":
        return ["Fig4-3", "Fig4-4"]
    if case == "C":
        return ["Fig4-5", "Fig4-6", "Fig4-7"]
    return ["Fig4-1", "Fig4-2", "Fig4-3", "Fig4-4", "Fig4-5", "Fig4-6", "Fig4-7"]


def run() -> int:
    """Program entry point."""
    parser = argparse.ArgumentParser(description="Generate Chapter 4 thesis figures.")
    parser.add_argument("--case", choices=["A", "B", "C", "ALL"], default="ALL")
    parser.add_argument("--outdir", default="figures/chapter4")
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--nx3d", type=int, default=36)
    args = parser.parse_args()

    outdir = resolve_outdir(args.outdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_plot_style()

    print("=" * 84)
    print("Chapter 4 Thesis Figure Generator")
    print("=" * 84)
    print(f"Chapter root : {CHAPTER_ROOT}")
    print(f"Repo root    : {REPO_ROOT}")
    print(f"Output dir   : {outdir}")
    print(f"Device       : {device}")
    print(f"Grid nx      : {args.nx}")
    print(f"3D grid nx   : {args.nx3d}")
    print("=" * 84)

    summary_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    resources_cache: dict[str, dict[str, Any]] = {}

    def get_cached(case_key: str, need_fmc: bool = False, need_model: bool = True) -> dict[str, Any]:
        cached = resources_cache.get(case_key)
        if (
            cached is None
            or (need_model and "model_path" not in cached)
            or (need_fmc and "fmc_path" not in cached)
        ):
            cached = get_resources(case_key, need_fmc=need_fmc, need_model=need_model)
            resources_cache[case_key] = cached
        return cached

    generation_targets = figure_plan(args.case)

    for figure_name in generation_targets:
        start = time.perf_counter()
        outputs: dict[str, str] = {}
        metrics: dict[str, Any] = {}
        status = "SUCCESS"
        error_text = ""
        try:
            if figure_name == "Fig4-1":
                result = figure_fig4_1(
                    outdir, get_cached("A", need_model=False), args.nx, device
                )
            elif figure_name == "Fig4-2":
                result = figure_fig4_2(
                    outdir, get_cached("A", need_model=False), args.nx, device
                )
            elif figure_name == "Fig4-3":
                result = figure_fig4_3(
                    outdir,
                    get_cached("B", need_fmc=True),
                    args.nx,
                    device,
                )
            elif figure_name == "Fig4-4":
                result = figure_fig4_4(
                    outdir, get_cached("B", need_fmc=True), args.nx, device
                )
            elif figure_name == "Fig4-5":
                result = figure_fig4_5(
                    outdir, get_cached("C", need_fmc=True), args.nx, device
                )
            elif figure_name == "Fig4-6":
                result = figure_fig4_6(
                    outdir,
                    get_cached("C", need_fmc=True),
                    args.nx,
                    device,
                )
            elif figure_name == "Fig4-7":
                result = figure_fig4_7(
                    outdir, get_cached("C", need_fmc=True), args.nx3d, device
                )
            else:
                raise ValueError(f"Unknown figure target: {figure_name}")
            outputs = result["outputs"]
            metrics = result["metrics"]
        except MissingArtifactError as exc:
            status = "MISSING"
            error_text = str(exc)
            print(f"\n[{figure_name}] Missing artifact:\n{error_text}\n")
        except Exception as exc:
            status = "FAILED"
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"\n[{figure_name}] Unexpected failure:\n{error_text}\n")
            print(traceback.format_exc())

        elapsed = time.perf_counter() - start
        summary = build_summary_row(figure_name, status, elapsed, outputs, metrics)
        summary_rows.append(summary)

        if figure_name in {"Fig4-1", "Fig4-2"}:
            dependency_cases = ("A",)
        elif figure_name in {"Fig4-3", "Fig4-4"}:
            dependency_cases = ("B",)
        elif figure_name in {"Fig4-5", "Fig4-6", "Fig4-7"}:
            dependency_cases = ("C",)
        else:
            dependency_cases = tuple()

        manifest_entries.append(
            {
                "figure": figure_name,
                "status": status,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(elapsed, 2),
                "outputs": outputs,
                "metrics": metrics,
                "error": error_text,
                "case_parameters": {
                    key: {
                        "kappa": CASE_METADATA[key]["kappa"],
                        "sigma_s": CASE_METADATA[key]["sigma_s"],
                        "g": CASE_METADATA[key]["g"],
                        "name": CASE_METADATA[key]["name"],
                    }
                    for key in dependency_cases
                },
                "grid": {
                    "nx": args.nx,
                    "nx_3d": args.nx3d,
                    "coordinates": "cell-centered",
                },
                "dependencies": {
                    case_key: {
                        "model_candidates": [
                            str(path) for path in model_candidates(case_key)
                        ],
                        "history_candidates": [
                            str(path) for path in history_candidates(case_key)
                        ],
                        "fmc_candidates": (
                            [str(path) for path in fmc_candidates(case_key)]
                            if case_key in {"B", "C"}
                            else []
                        ),
                    }
                    for case_key in dependency_cases
                },
            }
        )

    manifest_path = create_figure_manifest(outdir, manifest_entries)
    print_summary_table(summary_rows)
    print(f"\nManifest written to: {manifest_path}")

    if any(row["status"] in {"FAILED", "MISSING"} for row in summary_rows):
        print(
            "One or more figures were not generated. Please inspect the summary and missing-file hints above."
        )
        return 1

    print("All requested Chapter 4 figures were generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
