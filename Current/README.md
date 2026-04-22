# 第四章当前主线实现

`Current/` 保存 Chapter 4 归档项目的主线代码实现。和早期工程版本不同，这里已经移除了对父目录 `Core/`、`Chapter4_3D_SteadyState/` 等结构的运行时依赖，所有脚本都以归档根目录为基准解析本地路径。

## 当前结构

```text
Current/
├── README.md
├── CODE_LINEAGE.md
├── Training/
│   └── train_3d_multicase.py
├── EquationModels/
│   └── RadTrans3D_Complex.py
├── Models/
│   └── ModelClassTorch2.py
├── Evaluation/
│   ├── generate_chapter4_thesis_figures.py
│   ├── plot_3d_paper_figures.py
│   ├── validate_3d_pure_absorption.py
│   └── validate_pinn_vs_fmc_fixed.py
└── Solvers/
    └── MC/
        ├── FMC_3D_Solver_Fixed.py
        └── FMC_3D_Solver_Ultra.py
```

## 模块职责

- `Training/train_3d_multicase.py`
  - 训练 Case A/B/C 的 3D PINN 模型
  - 输出写到归档根目录：
    - `Results_3D_CaseA/`
    - `Results_3D_CaseB/`
    - `Results_3D_CaseC/`
- `EquationModels/RadTrans3D_Complex.py`
  - Chapter 4 的 3D RTE 物理引擎
  - 提供 `RadTrans3D_Physics`
- `Models/ModelClassTorch2.py`
  - 当前是兼容 shim
  - 对外继续暴露 `ModelClassTorch2.Pinns`
  - 实际实现来自归档根目录的 `Archive_Compat_Code/ModelClassTorch2.py`
- `Evaluation/generate_chapter4_thesis_figures.py`
  - 论文图总控脚本
  - 可独立生成 `Fig4-1` 到 `Fig4-7`
- `Evaluation/validate_3d_pure_absorption.py`
  - 纯吸收 Case A 高精验证
  - 输出到 `Artifacts/Figures_3D_Validation/`
- `Evaluation/validate_pinn_vs_fmc_fixed.py`
  - Case B/C 的 PINN 与 FMC 对比验证
- `Evaluation/plot_3d_paper_figures.py`
  - 当前目录下保留的评估/绘图脚本
  - 历史 `Figures_3D/` 真正对应的旧版脚本见归档根目录 `Legacy_Figures3D/plot_3d_paper_figures.py`
- `Solvers/MC/`
  - FMC 参考求解器代码

## 运行依赖关系

当前主线代码依赖的归档内目录是：

- `../Archive_Compat_Code/`
- `../Results_3D_CaseA/`
- `../Results_3D_CaseB/`
- `../Results_3D_CaseC/`
- `../Artifacts/Figures_3D_Validation/`
- `../Artifacts/MC3D_Raw_Benchmarks/`

不再要求：

- 父目录 `Core/`
- 父目录 `Models/`
- 父目录 `EquationModels/`
- 父目录 `Chapter4_3D_SteadyState/`

## 推荐命令

从归档根目录执行：

### 1. 重建 Chapter 4 论文图

```bash
python Current/Evaluation/generate_chapter4_thesis_figures.py --case ALL --outdir figures/chapter4_rebuild
```

### 2. 重做纯吸收高精验证

```bash
python Current/Evaluation/validate_3d_pure_absorption.py
```

### 3. 训练 3D 三个案例

```bash
python Current/Training/train_3d_multicase.py --case A
python Current/Training/train_3d_multicase.py --case B
python Current/Training/train_3d_multicase.py --case C
```

## 输出关系

- `generate_chapter4_thesis_figures.py`
  - 默认输出到 `figures/chapter4/`
  - 可通过 `--outdir` 指向 `figures/chapter4_rebuild/` 等目录
- `validate_3d_pure_absorption.py`
  - 输出到 `Artifacts/Figures_3D_Validation/`
- `train_3d_multicase.py`
  - 输出到归档根目录 `Results_3D_Case{A,B,C}/`

## 说明

- `Current/` 内没有 `Core/` 目录，这不是缺失，而是因为旧 `Core/` 职责已经被归档根目录的 `Archive_Compat_Code/` 最小兼容层替代。
- 如果你看到旧 `model.pkl` 仍能正常加载，依赖的正是 `Current/Models/ModelClassTorch2.py` 这个 shim 和 `Archive_Compat_Code/` 中的兼容实现。
