# 第四章代码实现归档

本目录是硕士论文第四章“3D 稳态辐射传输方程 RTE 的 PINN 求解与验证”的独立归档版本。当前归档已经补齐兼容代码、最小训练结果和路径修复，能够脱离父目录工程，直接在本目录内重建 Chapter 4 的论文图。

## 当前状态

- 已完成独立归档
- 可在本目录内独立运行 Chapter 4 主出图脚本
- 兼容旧 `model.pkl` 的最小代码层已归档到 `Archive_Compat_Code/`
- 最小必要训练结果已归档到 `Results_3D_CaseA/`、`Results_3D_CaseB/`、`Results_3D_CaseC/`

独立重建命令：

```bash
python Current/Evaluation/generate_chapter4_thesis_figures.py --case ALL --outdir figures/chapter4_rebuild
```

独立归档修复说明见：

- `ARCHIVE_INDEPENDENCE_NOTES.md`
- `FIGURE_LINEAGE_AUDIT.md`
- `FIGURE_LINEAGE_TABLE.csv`

## 目录结构

```text
Chapter4_Implementation_Archive/
├── README.md
├── ARCHITECTURE.md
├── ARCHIVE_INDEPENDENCE_NOTES.md
├── FIGURE_LINEAGE_AUDIT.md
├── FIGURE_LINEAGE_TABLE.csv
├── Archive_Compat_Code/
│   ├── __init__.py
│   ├── ImportFile.py
│   └── ModelClassTorch2.py
├── Artifacts/
│   ├── Figures_3D/
│   ├── Figures_3D_Validation/
│   └── MC3D_Raw_Benchmarks/
├── Current/
│   ├── README.md
│   ├── CODE_LINEAGE.md
│   ├── Training/
│   │   └── train_3d_multicase.py
│   ├── EquationModels/
│   │   └── RadTrans3D_Complex.py
│   ├── Models/
│   │   └── ModelClassTorch2.py
│   ├── Evaluation/
│   │   ├── generate_chapter4_thesis_figures.py
│   │   ├── plot_3d_paper_figures.py
│   │   ├── validate_3d_pure_absorption.py
│   │   └── validate_pinn_vs_fmc_fixed.py
│   └── Solvers/
│       └── MC/
│           ├── FMC_3D_Solver_Fixed.py
│           └── FMC_3D_Solver_Ultra.py
├── Legacy_Figures3D/
│   └── plot_3d_paper_figures.py
├── Results_3D_CaseA/
├── Results_3D_CaseB/
├── Results_3D_CaseC/
└── figures/
    ├── chapter4/
    ├── chapter4_rebuild/
    └── chapter4_verify_check/
```

## 关键目录说明

- `Archive_Compat_Code/`
  - 归档内兼容层，用来替代父目录旧 `Core/` 结构中的最小必要代码。
  - 重点解决旧 `model.pkl` 对 `ModelClassTorch2` / `ImportFile` 的依赖。
- `Current/`
  - 当前第四章主线实现，包含训练、物理引擎、评估和论文图生成脚本。
- `Legacy_Figures3D/`
  - 保存历史 `Figures_3D/` 结果对应的旧版绘图脚本。
- `Results_3D_CaseA/`、`Results_3D_CaseB/`、`Results_3D_CaseC/`
  - 归档内最小必要训练输出，包含 `model.pkl`、`training_history.json` 和 `case_config.json`。
- `Artifacts/`
  - 保存已归档的历史中间结果和基准数据。
  - `Artifacts/Figures_3D_Validation/` 为 Case A 高精验证数据。
  - `Artifacts/MC3D_Raw_Benchmarks/` 为 Case B/C 的 FMC 基准数据。
- `figures/`
  - 保存论文图输出。
  - `figures/chapter4/` 是归档内已有的论文图集合。
  - `figures/chapter4_rebuild/`、`figures/chapter4_verify_check/` 是独立归档验证过程中重建得到的输出。

## 关键用途

- 训练入口：
  - `Current/Training/train_3d_multicase.py`
- 物理核心：
  - `Current/EquationModels/RadTrans3D_Complex.py`
- 模型兼容入口：
  - `Current/Models/ModelClassTorch2.py`
- 兼容实现主体：
  - `Archive_Compat_Code/ModelClassTorch2.py`
- Case A 高精验证：
  - `Current/Evaluation/validate_3d_pure_absorption.py`
- Case B/C 与 FMC 对比：
  - `Current/Evaluation/validate_pinn_vs_fmc_fixed.py`
- Chapter 4 论文图总控：
  - `Current/Evaluation/generate_chapter4_thesis_figures.py`
- 历史 `Figures_3D/` 溯源脚本：
  - `Legacy_Figures3D/plot_3d_paper_figures.py`

## 推荐使用方式

### 1. 重建 Chapter 4 论文图

```bash
python Current/Evaluation/generate_chapter4_thesis_figures.py --case ALL --outdir figures/chapter4_rebuild
```

### 2. 生成纯吸收验证图

```bash
python Current/Evaluation/validate_3d_pure_absorption.py
```

### 3. 重新训练 Case A/B/C

```bash
python Current/Training/train_3d_multicase.py --case A
python Current/Training/train_3d_multicase.py --case B
python Current/Training/train_3d_multicase.py --case C
```

## 重要说明

- 本归档已经满足“独立重建 Chapter 4 论文图”的目标。
- 当前兼容层是为归档复现服务的最小实现，不是父目录完整旧工程的全量迁移。
- 如果后续继续扩展归档范围，应优先保持现有目录内自洽，不再引回父目录路径依赖。
