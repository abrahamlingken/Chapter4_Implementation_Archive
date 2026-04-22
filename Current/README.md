# 第四章：三维稳态辐射传输 (3D Steady-State Radiative Transfer)

## 目录结构

```
Chapter4_3D_SteadyState/
├── README.md                          # 本文件
├── CODE_LINEAGE.md                    # 代码溯源文档
├── Evaluation/                        # 评估与可视化脚本
│   ├── plot_3d_paper_figures.py       # 生成 Figures_3D/ 的论文图表
│   ├── validate_3d_pure_absorption.py # 3D纯吸收案例验证 → Figures_3D_Validation/
│   ├── validate_pinn_vs_fmc_fixed.py  # PINN vs FMC验证（坐标修正版）
│   └── validate_pinn_vs_fmc.py        # PINN vs FMC验证
├── Training/                          # PINN训练脚本
│   └── train_3d_multicase.py          # 3D案例训练 (Case A/B/C)
├── Solvers/                           # 基准求解器
│   └── MC/
│       ├── FMC_3D_Solver_Fixed.py     # 200M光子标准版
│       └── FMC_3D_Solver_Ultra.py     # 1B光子超精版
├── Core/                              # 核心工具模块
│   └── ImportFile.py                  # 统一导入模块
├── Models/                            # PINN核心类定义
│   ├── ModelClassTorch2.py            # Pinns神经网络类
│   ├── DatasetTorch2.py               # 数据集处理类
│   └── ObjectClass.py                 # 几何对象类
├── EquationModels/                    # 物理方程定义
│   ├── RadTrans3D_Complex.py          # 3D RTE物理引擎 (OOP版本)
│   └── RadTrans3D_paper.py            # 3D RTE论文版本
├── Figures/                           # 运行时生成图表目录
└── Results/                           # 训练结果存放目录 (运行时生成)
    ├── Results_3D_CaseA/              # Case A结果
    ├── Results_3D_CaseB/              # Case B结果
    └── Results_3D_CaseC/              # Case C结果
```

## 案例配置

### 3D案例 (Chapter 4)
| 案例 | kappa | sigma_s | g | 描述 | 输出文件夹 |
|------|-------|---------|---|------|-----------|
| 3D-A | 5.0 | 0.0 | 0.0 | 纯吸收介质 | Results_3D_CaseA/ |
| 3D-B | 0.5 | 4.5 | 0.0 | 各向同性散射 | Results_3D_CaseB/ |
| 3D-C | 0.5 | 4.5 | 0.8 | 前向散射 | Results_3D_CaseC/ |

## 使用流程

### 1. 训练PINN模型

```bash
# 训练3D案例 (Case A/B/C)
python Training/train_3d_multicase.py --case A
python Training/train_3d_multicase.py --case B
python Training/train_3d_multicase.py --case C
```

### 2. 生成FMC基准解

```bash
# 运行前向蒙特卡洛求解器
python Solvers/MC/FMC_3D_Solver_Fixed.py  # 标准版 (200M光子)
python Solvers/MC/FMC_3D_Solver_Ultra.py   # 超精版 (1B光子)
```

### 3. 生成可视化图表

```bash
# 生成 Figures_3D/ 目录的论文图表
python Evaluation/plot_3d_paper_figures.py

# 生成 Figures_3D_Validation/ 目录的验证图表
python Evaluation/validate_3d_pure_absorption.py

# PINN vs FMC验证
python Evaluation/validate_pinn_vs_fmc_fixed.py
```

## 输出文件说明

### Figures_3D/ 目录 (由 plot_3d_paper_figures.py 生成)
| 文件 | 说明 |
|------|------|
| G_along_centerline.png/pdf | 中心线G(x)分布对比图 |
| G_centerline_data.npz | 中心线数据文件 |
| G_center_slice.png/pdf | 中心截面热图 |
| G_center_slice_data.npz | 截面数据文件 |
| G_3D_Case_3D-A_Pure_Absorption.vts | Case A VTK体数据 |
| G_3D_Case_3D-B_Isotropic.vts | Case B VTK体数据 |
| G_3D_Case_3D-C_Forward_Scattering.vts | Case C VTK体数据 |
| physical_assessment_report.json | 物理评估报告 |

### Figures_3D_Validation/ 目录 (由 validate_3d_pure_absorption.py 生成)
| 文件 | 说明 |
|------|------|
| G_CaseA_Centerline_HighPrecision.png/pdf | 高精度中心线对比图 |
| G_CaseA_Centerline_Data.npz | 中心线数据 |
| G_CaseA_2D_HighPrecision.png/pdf | 高精度2D热图 |
| G_CaseA_2D_Data.npz | 2D数据 |

## 代码依赖关系

### Figures_3D 生成流程
```
plot_3d_paper_figures.py
    ├── Models/ModelClassTorch2.py (Pinns类)
    ├── Models/DatasetTorch2.py
    ├── Models/ObjectClass.py
    ├── Core/ImportFile.py
    ├── EquationModels/RadTrans3D_Complex.py (物理引擎)
    └── Results_3D_Case{A,B,C}/model.pkl
```

### Figures_3D_Validation 生成流程
```
validate_3d_pure_absorption.py
    ├── Models/ModelClassTorch2.py (Pinns类)
    ├── Core/ImportFile.py
    ├── EquationModels/RadTrans3D_Complex.py (物理引擎)
    ├── Results_3D_CaseA/model.pkl
    └── Beer-Lambert解析解 (内置)
```

### PINN vs FMC 验证流程
```
validate_pinn_vs_fmc_fixed.py
    ├── Results_3D_Case{B,C}/model.pkl
    └── Solvers/MC/FMC_G_3D_Case{B,C}_FIXED_HighStats.npy
```
