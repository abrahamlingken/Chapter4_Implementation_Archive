# Archive Independence Notes

## Goal

Make `Chapter4_Implementation_Archive/` runnable as a standalone archive project, without relying on code or result paths from the parent repository, and ensure it can rebuild the Chapter 4 thesis figures under `figures/chapter4`.

## Code Files Added

Added `Archive_Compat_Code/` as the archive-local compatibility layer:

- `Archive_Compat_Code/__init__.py`
- `Archive_Compat_Code/ImportFile.py`
- `Archive_Compat_Code/ModelClassTorch2.py`

### Why these files were necessary

- `ImportFile.py` was missing from the archive, while the legacy model stack depended on it in the parent project.
- Old serialized models in `model.pkl` refer to the module name `ModelClassTorch2`, so the archive needed a local model definition compatible with that legacy pickle reference.
- The parent `Core/ImportFile.py` and `Core/ModelClassTorch2.py` brought in broader legacy dependencies that are not present inside this archive. For archive independence, the copied versions were reduced to the minimal subset needed for:
  - loading archived `model.pkl`
  - instantiating the `Pinns` network
  - applying legacy Xavier initialization during local retraining

### Compatibility strategy

- `Archive_Compat_Code/ModelClassTorch2.py` holds the standalone implementation.
- `Current/Models/ModelClassTorch2.py` was converted into a shim that re-exports `Pinns`, `Swish`, `activation`, and `init_xavier`.
- The shim forces `Pinns.__module__` and `Swish.__module__` back to `ModelClassTorch2`, so legacy `model.pkl` files still load under the expected module name.

## Result Files Copied

Copied the minimum required training outputs into the archive root:

- `Results_3D_CaseA/model.pkl`
- `Results_3D_CaseA/training_history.json`
- `Results_3D_CaseA/case_config.json`
- `Results_3D_CaseB/model.pkl`
- `Results_3D_CaseB/training_history.json`
- `Results_3D_CaseB/case_config.json`
- `Results_3D_CaseC/model.pkl`
- `Results_3D_CaseC/training_history.json`
- `Results_3D_CaseC/case_config.json`

### Why these result files were necessary

- `generate_chapter4_thesis_figures.py` needs local `model.pkl` files for Cases B and C inference.
- The archive now keeps the same case-result folder names that the figure-generation script expects.
- `training_history.json` and `case_config.json` were copied as the minimum associated metadata requested for each archived case.
- `Artifacts/MC3D_Raw_Benchmarks/` and `Artifacts/Figures_3D_Validation/` were already present in the archive and were not duplicated.

## Scripts Repaired

Patched these scripts so they only resolve archive-local paths:

- `Current/Evaluation/generate_chapter4_thesis_figures.py`
- `Current/Training/train_3d_multicase.py`
- `Current/Evaluation/validate_3d_pure_absorption.py`
- `Legacy_Figures3D/plot_3d_paper_figures.py`

### What changed

- Removed parent-project path fallbacks from runtime `sys.path` setup.
- Standardized local path bootstrapping around:
  - archive root
  - `Archive_Compat_Code/`
  - `Current/`
  - `Current/Models/`
  - `Current/EquationModels/`
- Updated result/output locations so they resolve inside `Chapter4_Implementation_Archive/` regardless of whether a script is launched from the archive root or from the script's own directory.
- Updated `validate_3d_pure_absorption.py` to write into `Artifacts/Figures_3D_Validation/`, which matches the archive data location used by the Chapter 4 figure pipeline.

## Independence Verification

### Main standalone verification

Executed from the archive root:

```bash
python Current/Evaluation/generate_chapter4_thesis_figures.py --case ALL --outdir figures/chapter4_rebuild
```

Result:

- Success
- No parent-directory code paths were required
- All requested figures were generated:
  - `Fig4-1` through `Fig4-7`
- Output written to:
  - `figures/chapter4_rebuild/`

Generated artifacts include:

- PNG figures
- PDF figures
- figure data `.npz`
- `Fig4-7_G_3D_visualization.vtk`
- `figure_manifest.json`

### Additional validation performed

- Archive-local `torch.load(...)` test passed for:
  - `Results_3D_CaseA/model.pkl`
  - `Results_3D_CaseB/model.pkl`
  - `Results_3D_CaseC/model.pkl`
- `python -m py_compile` passed for all added and modified compatibility/entry scripts.
- Runtime import checks passed for:
  - `Current/Training/train_3d_multicase.py`
  - `Current/Evaluation/validate_3d_pure_absorption.py`
  - `Legacy_Figures3D/plot_3d_paper_figures.py`

## Final Status

The archive is now able to independently regenerate the Chapter 4 thesis figures using only files inside `Chapter4_Implementation_Archive/`.

### Remaining blockers

- None for the Chapter 4 figure-regeneration target that was validated here.
- Note: the copied `ImportFile.py` and `ModelClassTorch2.py` are intentionally minimal archive-compat versions, not a full restoration of the entire parent legacy training stack.
