# 第四章代码实现归档

本目录用于集中整理硕士论文第四章“3D 稳态辐射传输方程 RTE 的 PINN 求解与验证”相关代码，便于后续查阅、复现实验、追溯图像来源与单独归档。

## 目录结构

```text
Chapter4_Implementation_Archive/
├── README.md
├── ARCHITECTURE.md
├── Artifacts/
│   ├── Figures_3D/
│   └── Figures_3D_Validation/
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
└── Legacy_Figures3D/
    └── plot_3d_paper_figures.py
```

## 文件整理原则

- `Artifacts/`：保存已生成的历史结果文件，便于直接查阅与打包。
- `Current/`：保存当前第四章主线实现，覆盖训练、物理引擎、PINN 模型、验证脚本、论文图总控脚本与 FMC 求解器。
- `Legacy_Figures3D/`：单独保存旧版 `plot_3d_paper_figures.py`，因为根目录 `Figures_3D/` 这一套历史结果实际由这份脚本生成，而不是当前 `Chapter4_3D_SteadyState/Evaluation/plot_3d_paper_figures.py` 生成。

## 关键用途

- 训练入口：
  - `Current/Training/train_3d_multicase.py`
- 物理核心：
  - `Current/EquationModels/RadTrans3D_Complex.py`
- PINN 网络定义：
  - `Current/Models/ModelClassTorch2.py`
- Case A 高精验证：
  - `Current/Evaluation/validate_3d_pure_absorption.py`
- Case B/C 与 FMC 对比：
  - `Current/Evaluation/validate_pinn_vs_fmc_fixed.py`
- 第四章论文配图总控：
  - `Current/Evaluation/generate_chapter4_thesis_figures.py`
- 历史 `Figures_3D/` 溯源脚本：
  - `Legacy_Figures3D/plot_3d_paper_figures.py`

## 对应结果目录关系

- `Artifacts/Figures_3D/`
  - 从根目录 `Figures_3D/` 复制而来
- `Artifacts/Figures_3D_Validation/`
  - 从根目录 `Figures_3D_Validation/` 复制而来
- 根目录 `Figures_3D/`
  - 对应 `Legacy_Figures3D/plot_3d_paper_figures.py`
- 根目录 `Figures_3D_Validation/`
  - 对应 `Current/Evaluation/validate_3d_pure_absorption.py`
- `Chapter4_3D_SteadyState/Figures_Thesis/Chapter4/`
  - 对应 `Current/Evaluation/generate_chapter4_thesis_figures.py`

## 建议阅读顺序

1. 先看 `ARCHITECTURE.md`，理解模块层次与数据流。
2. 再看 `Current/Training/train_3d_multicase.py`，明确模型与结果文件如何生成。
3. 再看 `Current/EquationModels/RadTrans3D_Complex.py`，理解 `G(x,y,z)` 的计算方式。
4. 最后按需求查看评估脚本：
   - 纯吸收验证：`validate_3d_pure_absorption.py`
   - FMC 对比：`validate_pinn_vs_fmc_fixed.py`
   - 论文出图：`generate_chapter4_thesis_figures.py`
