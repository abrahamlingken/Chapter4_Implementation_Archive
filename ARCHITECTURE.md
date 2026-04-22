# 第四章代码架构说明

本文档说明第四章 3D 稳态辐射传输 PINN 实现的代码层次、模块职责、数据流以及历史结果与当前代码的关系。

## 1. 总体分层

第四章代码可以分为 5 层：

1. 训练层
   - 负责定义 Case A/B/C 的参数、构造采样点、训练 PINN、保存模型与训练历史。
   - 代表文件：`Current/Training/train_3d_multicase.py`

2. 物理层
   - 负责 3D RTE 的物理残差、边界条件、角度积分与入射辐射 `G(x,y,z)` 计算。
   - 代表文件：`Current/EquationModels/RadTrans3D_Complex.py`

3. 模型层
   - 负责 PINN 网络结构定义与初始化。
   - 代表文件：`Current/Models/ModelClassTorch2.py`

4. 基准解层
   - 负责 Monte Carlo 参考场生成，主要用于 Case B / Case C 对比验证。
   - 代表文件：
     - `Current/Solvers/MC/FMC_3D_Solver_Fixed.py`
     - `Current/Solvers/MC/FMC_3D_Solver_Ultra.py`

5. 评估与绘图层
   - 负责纯吸收验证、PINN vs FMC 对比、论文图生成与历史图像脚本。
   - 代表文件：
     - `Current/Evaluation/validate_3d_pure_absorption.py`
     - `Current/Evaluation/validate_pinn_vs_fmc_fixed.py`
     - `Current/Evaluation/generate_chapter4_thesis_figures.py`
     - `Legacy_Figures3D/plot_3d_paper_figures.py`

## 2. 模块职责

### 2.1 训练层

`train_3d_multicase.py` 的职责：

- 定义 3 个算例：
  - Case 3D-A：`kappa=5.0, sigma_s=0.0, g=0.0`
  - Case 3D-B：`kappa=0.5, sigma_s=4.5, g=0.0`
  - Case 3D-C：`kappa=0.5, sigma_s=4.5, g=0.8`
- 调用物理引擎生成：
  - 内部配点 `generate_collocation_points`
  - 边界配点 `generate_boundary_points`
- 训练 PINN
- 保存结果：
  - `Results_3D_CaseA/model.pkl`
  - `Results_3D_CaseB/model.pkl`
  - `Results_3D_CaseC/model.pkl`
  - `training_history.json`
  - `case_config.json`

### 2.2 物理层

`RadTrans3D_Complex.py` 是第四章的核心物理引擎，主要职责：

- 定义吸收、散射与 HG 相函数；
- 构造方向积分的 Gauss-Legendre + 均匀方位角离散；
- 计算 PDE 残差 `compute_res(...)`；
- 应用边界条件 `apply_bc(...)`；
- 生成入射辐射：
  - `compute_incident_radiation(x, y, z, model)`

这里的 `compute_incident_radiation(...)` 是几乎所有后处理脚本的核心入口。

### 2.3 模型层

`ModelClassTorch2.py` 中的 `Pinns` 类负责：

- 定义输入维数为 5 的网络：
  - `(x, y, z, theta, phi)`
- 输出辐射强度 `I`
- 提供 Xavier 初始化等辅助功能

### 2.4 基准解层

`FMC_3D_Solver_Fixed.py` 与 `FMC_3D_Solver_Ultra.py` 的职责：

- 用前向 Monte Carlo 计算 3D 标量场 `G`
- 输出 `.npy` 主结果与 `.npz` 元数据
- 典型输出：
  - `FMC_G_3D_CaseB_FIXED_HighStats.npy`
  - `FMC_G_3D_CaseC_FIXED_HighStats.npy`
  - `FMC_G_3D_CaseB_FIXED_UltraStats.npy`
  - `FMC_G_3D_CaseC_FIXED_UltraStats.npy`

## 3. 当前评估与绘图链路

### 3.1 `validate_3d_pure_absorption.py`

用途：

- 专门验证 Case A 纯吸收情形；
- 通过脚本内高精积分公式构造参考解；
- 与 PINN 预测进行中心线与中心截面比较；
- 输出到 `Figures_3D_Validation/`

内部数据来源：

- 参考解：
  - `source_term(...)`
  - `compute_exact_intensity_single(...)`
  - `ExactGSolver`
- PINN 解：
  - `Results_3D_CaseA/model.pkl`
  - `RadTrans3D_Complex.compute_incident_radiation(...)`

### 3.2 `validate_pinn_vs_fmc_fixed.py`

用途：

- 对 Case B / Case C 做 PINN vs FMC 验证；
- 重点处理 FMC 的 cell-centered 网格与 PINN 的 node-centered 预测之间的对齐问题。

关键物理修正：

- 中心线提取时，对 FMC 的 4 个中心 cell 求平均；
- 使用 `np.interp` 把 PINN 曲线对齐到 FMC 的 cell-centered 坐标。

### 3.3 `generate_chapter4_thesis_figures.py`

用途：

- 统一生成第四章论文正式图；
- 自动探测模型与 FMC 文件；
- 统一输出到 `Chapter4_3D_SteadyState/Figures_Thesis/Chapter4/`

它整合了 3 类来源：

- Case A 高精参考积分
- Case B/C 的 FMC 基准场
- 当前训练得到的 PINN 模型

## 4. 历史 `Figures_3D/` 链路

这里要特别说明：

- 根目录 `Figures_3D/` 这套结果，不是由当前 `Current/Evaluation/plot_3d_paper_figures.py` 生成的；
- 它实际来自 `Legacy_Figures3D/plot_3d_paper_figures.py`

这份旧版脚本的特点：

- 读取 `Results_3D_CaseA/B/C/model.pkl`
- 动态推理 `G(x,y,z)`
- 生成：
  - `G_along_centerline.png/pdf`
  - `G_centerline_data.npz`
  - `G_center_slice.png/pdf`
  - `G_center_slice_data.npz`
  - `G_3D_*.vts`

因此，`Legacy_Figures3D` 主要是为了保留历史图像的真实来源。

## 5. 数据流

可以把第四章的数据流理解成：

`Case 参数 -> 训练脚本 -> model.pkl -> 物理引擎 compute_incident_radiation -> 评估/绘图脚本 -> 图像与 npz`

而 Case B / C 还多一条基准支路：

`Case 参数 -> FMC 求解器 -> FMC_G_3D_*.npy -> 验证/论文图脚本 -> 对比图`

## 6. 结果目录关系

### 根目录历史结果

- `Figures_3D/`
  - 历史论文图
  - 来源：`Legacy_Figures3D/plot_3d_paper_figures.py`

- `Figures_3D_Validation/`
  - Case A 高精验证图
  - 来源：`Current/Evaluation/validate_3d_pure_absorption.py`

### 归档后的结果副本

- `Artifacts/Figures_3D/`
  - 对根目录 `Figures_3D/` 的归档副本
- `Artifacts/Figures_3D_Validation/`
  - 对根目录 `Figures_3D_Validation/` 的归档副本

### 当前正式论文图

- `Chapter4_3D_SteadyState/Figures_Thesis/Chapter4/`
  - 第四章最终论文图
  - 来源：`Current/Evaluation/generate_chapter4_thesis_figures.py`

## 7. 推荐使用方式

如果目的是理解第四章实现：

1. 先看训练脚本 `train_3d_multicase.py`
2. 再看物理引擎 `RadTrans3D_Complex.py`
3. 再看高精验证脚本 `validate_3d_pure_absorption.py`
4. 再看 FMC 对齐验证脚本 `validate_pinn_vs_fmc_fixed.py`
5. 最后看总控脚本 `generate_chapter4_thesis_figures.py`

如果目的是追溯旧图：

1. 先看 `Legacy_Figures3D/plot_3d_paper_figures.py`
2. 再对照根目录 `Figures_3D/` 中的文件名
