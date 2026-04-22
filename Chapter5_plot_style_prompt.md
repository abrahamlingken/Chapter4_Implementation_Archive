# Chapter 5 Plot Style Prompt

Use the following prompt to instruct another Codex thread to generate Chapter 5 figures while preserving the plotting language established in Chapter 4.

---

You are generating publication-quality figures for Chapter 5 of a Master's thesis. All Chapter 5 figures must strictly inherit the visual language already established and actually used in Chapter 4, so that the figures across chapters look like they belong to the same thesis, the same plotting system, and the same author.

Do not invent a new visual style unless explicitly requested. Reuse the existing Chapter 4 plotting conventions, code organization, figure export rules, and scientific presentation logic as much as possible.

## 1. Core Objective

- Generate thesis-ready figures for Chapter 5.
- Keep style, typography, color logic, legend conventions, and layout fully consistent with Chapter 4.
- Make all outputs suitable for direct inclusion in LaTeX.
- Save all Chapter 5 figure outputs to `figures/chapter5/`.
- Export every figure as:
  - `PNG` at `600 dpi`
  - `PDF` as a vector figure
- Save with:
  - `bbox_inches='tight'`
  - white background
  - tight, publication-safe layout
- When useful, also save reproducibility data as `*.npz`.

## 2. Typography and Thesis Formatting

Strictly follow these font rules:

- Use thesis-style serif fonts.
- Chinese text priority:
  - `SimSun`
  - `STSong`
  - `Songti SC`
- English text priority:
  - `Times New Roman`
- Keep the figure language professional and academic.
- Physical quantities, variables, and mathematical symbols must use italic mathematical notation.

Recommended matplotlib configuration:

```python
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimSun", "STSong", "Songti SC", "DejaVu Serif"],
    "font.sans-serif": ["SimSun", "Times New Roman", "DejaVu Sans"],
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "mathtext.default": "it",
})
```

Font-size rules:

- Standard single-panel figures: `10.5 pt` (close to Chinese 5-point thesis text)
- Multi-panel figures / dense figures / 3D figures: `9 pt` (small 5-point)
- Within the same figure, all text sizes must remain uniform.
- Axis labels, titles, legends, tick labels, and colorbar labels should feel visually matched.

## 3. Color and Line Conventions

### 3.1 Line plots

Use exactly this color logic:

- PINN / model prediction:
  - color: `#1F77B4`
  - linestyle: solid `-`
  - linewidth: about `2.2`
- Reference / benchmark / numerical / FMC:
  - color: `#D62728`
  - linestyle: dashed `--`
  - linewidth: about `2.2`
- Error curves:
  - color: `#4D4D4D`
  - linestyle: dotted `:`
  - linewidth: about `1.8`

Do not casually replace this with green, orange, purple, or pastel schemes.

### 3.2 Contour / scalar-field plots

- Main field colormap: `jet`
- Error-field colormap: `Reds`

Keep this consistent across Chapter 5 unless explicitly instructed otherwise.

## 4. Legends

- Use borderless legends:
  - `frameon=False`
- Preferred locations:
  - line plots: `upper right` or `best`
- Legends must not cover peaks, sharp transitions, or key comparison regions.
- Legend labels should stay concise and aligned with Chapter 4 wording, for example:
  - `PINN Prediction`
  - `FMC Reference`
  - `Numerical Reference`
  - `Absolute Error`

If Chapter 5 uses a different model name, preserve the same tone, for example:

- `Model Prediction`
- `Reference Solution`
- `Absolute Error`

## 5. Axes and Grid Styling

### 5.1 Axes

Use mathematical notation for labels, for example:

- `r"$x$"`
- `r"$y$"`
- `r"$z$"`
- `r"$G(x, 0.5, 0.5)$"`

Use the following styling:

```python
plt.rcParams.update({
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.unicode_minus": False,
})
```

### 5.2 Grid lines

- Line plots should normally include a grid.
- Grid style:
  - dashed `--`
  - alpha about `0.35`
- Contour plots and 3D plots normally do not need grids.

## 6. Multi-panel Layout and Subfigure Labels

- Multi-panel figures must use subfigure tags:
  - `(a)`
  - `(b)`
  - `(c)`
- Put the panel label at the upper-left corner inside each subplot.
- Use bold styling for panel labels.
- Keep panel labels aligned across the chapter.
- Subtitles should remain short and professional, for example:
  - `Numerical Reference`
  - `PINN Prediction`
  - `FMC Reference`
  - `Absolute Error`

When comparing multiple scalar fields:

- Use the same contour levels for all main-field panels.
- Use a separate contour range for the error panel.
- Do not let each main panel auto-scale independently.

For 2D spatial plots:

- Always enforce equal aspect ratio:
  - `ax.set_aspect("equal")`

## 7. Colorbar Rules

All contour plots must include clearly labeled colorbars.

- Main field colorbar label example:
  - `Incident Radiation $G$`
- Error colorbar label example:
  - `r"$|G_{\mathrm{pred}} - G_{\mathrm{ref}}|$"`

Colorbar label fonts and tick fonts must match the rest of the figure.

If a figure contains two main-field panels and one error panel:

- The main-field panels should share one common colorbar.
- The error panel should use its own colorbar.

## 8. 3D Visualization Style

If Chapter 5 requires 3D visualization, follow the Chapter 4 approach:

- Prefer a clean high-value-region scatter visualization rather than visually noisy volume rendering.
- Plot only the stronger-value region, for example by thresholding with a percentile.
- Use:
  - colormap: `jet`
  - moderate transparency, e.g. `alpha≈0.55`
  - point size around `s≈10`
  - `linewidths=0`

3D axes should satisfy:

- labeled `x`, `y`, `z`
- domain ranges consistent with the physical problem
- usually `[0, 1]` for normalized cubic domains
- equal box proportions:
  - `set_box_aspect((1, 1, 1))`
- stable view angle, similar to Chapter 4:
  - `elev≈24`
  - `azim≈-52`

Include a colorbar with the correct physical label.

## 9. Scientific Plotting Discipline

These figures are for a thesis, so visual consistency is not enough; comparison fairness also matters.

- Reference and prediction plots must be directly comparable.
- Use the same:
  - grid
  - slice location
  - physical quantity definition
  - contour scale for comparable main fields
- If the physical mid-plane lies between grid layers, it is acceptable to use:
  - centered two-layer averaging
  - thin slab averaging
- But this must be:
  - physically justified
  - clearly encoded in the code
  - saved in metadata when relevant

Do not apply undeclared smoothing to benchmark/reference data simply to make figures prettier.

If variance-reduction or denoising strategies are used in a scientifically valid way, such as:

- multi-run averaging
- slab averaging
- symmetry averaging
- RQMC aggregation

then preserve this information in:

- code comments
- saved metadata
- `*.npz` outputs when useful

## 10. Function Organization

Structure the plotting code using reusable functions aligned with the Chapter 4 pattern, such as:

- `set_plot_style()`
- `plot_centerline()`
- `plot_plane_distribution()`
- `plot_comparison()`
- `plot_3d_visualization()`
- `save_figure()`
- `save_npz()`

Requirements:

- Centralize shared style logic.
- Do not duplicate plotting style settings in every figure.
- Each figure function should be clean and single-purpose.
- Return or record:
  - output paths
  - key error metrics when relevant
  - data file paths when saved

## 11. File Naming Rules

Use a consistent naming pattern:

- `Fig5-1_xxx`
- `Fig5-2_xxx`
- `Fig5-3_xxx`

For each figure, save:

- `Fig5-x_xxx.png`
- `Fig5-x_xxx.pdf`
- `Fig5-x_xxx_data.npz`

If needed for 3D post-processing, also export:

- `Fig5-x_xxx.vtk`

## 12. Language and Figure Titles

- Keep in-figure text concise and professional.
- Use English inside the figure unless otherwise required.
- Let the thesis body or LaTeX caption carry the longer Chinese explanation.
- Avoid long sentences inside plots.
- Keep titles short, such as:
  - `PINN Prediction`
  - `FMC Reference`
  - `Absolute Error`

## 13. Execution Instructions

Before creating new plotting code, first search the local codebase for the Chapter 4 plotting implementation and reuse its style and helper functions where possible.

Priority reference:

- `Current/Evaluation/generate_chapter4_thesis_figures.py`

Do not create a visually different plotting system for Chapter 5. Change only the figure content, not the design language.

The final Chapter 5 figures should make a reader immediately feel that Chapter 4 and Chapter 5 belong to one coherent thesis.

---

Recommended handoff note:

"Please follow this prompt exactly, and search the existing Chapter 4 plotting code before making any new figure logic."
