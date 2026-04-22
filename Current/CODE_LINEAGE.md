# 第四章代码溯源清单 (Code Lineage for Chapter 4)

## 一、Figures_3D 溯源

### 第0层：直接生成图表的代码
```
Evaluation/plot_3d_paper_figures.py (原始版本在 backup_20260403_183516/)
    ├── 输出目录: Figures_3D/
    └── 生成文件:
        ├── G_along_centerline.png/pdf          (中心线G(x)分布)
        ├── G_centerline_data.npz               (中心线数据)
        ├── G_center_slice.png/pdf              (中心截面热图)
        ├── G_center_slice_data.npz             (截面数据)
        ├── G_3D_Case_3D-A_Pure_Absorption.vts  (Case A VTK)
        ├── G_3D_Case_3D-B_Isotropic.vts        (Case B VTK)
        ├── G_3D_Case_3D-C_Forward_Scattering.vts (Case C VTK)
        └── physical_assessment_report.json     (物理评估报告)
```

**关键代码位置：**
```python
# Line 103
output_dir = 'Figures_3D'

# Line 152-154: 中心线图
plt.savefig(os.path.join(output_dir, 'G_along_centerline.png'), dpi=600)

# Line 168-169: 中心线数据
np.savez(os.path.join(output_dir, 'G_centerline_data.npz'), **centerline_data)

# Line 209-211: 截面热图
plt.savefig(os.path.join(output_dir, 'G_center_slice.png'), dpi=600)

# Line 229-230: 截面数据
np.savez(os.path.join(output_dir, 'G_center_slice_data.npz'), **slice_data)

# Line 260-263: VTK导出
vtk_path = os.path.join(output_dir, f'G_3D_{case_short}')
gridToVTK(vtk_path, X_vtk, Y_vtk, Z_vtk, pointData={"G": G_vtk})
```

### 第1层：直接依赖
| 依赖项 | 路径 | 说明 |
|--------|------|------|
| Case A模型 | Results_3D_CaseA/model.pkl | 纯吸收介质 |
| Case B模型 | Results_3D_CaseB/model.pkl | 各向同性散射 |
| Case C模型 | Results_3D_CaseC/model.pkl | 前向散射 |

**代码位置 (Line 56-64):**
```python
model_path = os.path.join(case_folder, 'model.pkl')
model = torch.load(model_path, map_location=engine.dev, weights_only=False)
```

### 第2层：训练代码
```
Training/train_3d_multicase.py
    ├── 生成: Results_3D_CaseA/model.pkl (kappa=5.0, sigma_s=0.0, g=0.0)
    ├── 生成: Results_3D_CaseB/model.pkl (kappa=0.5, sigma_s=4.5, g=0.0)
    └── 生成: Results_3D_CaseC/model.pkl (kappa=0.5, sigma_s=4.5, g=0.8)
    
配置 (Line 59, 67, 75):
CASE_CONFIGS = {
    'A': {'folder': 'Results_3D_CaseA', ...},
    'B': {'folder': 'Results_3D_CaseB', ...},
    'C': {'folder': 'Results_3D_CaseC', ...}
}
```

---

## 二、Figures_3D_Validation 溯源

### 第0层：直接生成图表的代码
```
Evaluation/validate_3d_pure_absorption.py
    ├── 输出目录: Figures_3D_Validation/
    └── 生成文件:
        ├── G_CaseA_Centerline_HighPrecision.png/pdf
        ├── G_CaseA_Centerline_Data.npz
        ├── G_CaseA_2D_HighPrecision.png/pdf
        └── G_CaseA_2D_Data.npz
```

**关键代码位置 (Line 195):**
```python
output_dir = 'Figures_3D_Validation'
```

**图表生成 (Line 250-266, 335-352):**
```python
plt.savefig(os.path.join(output_dir, 'G_CaseA_Centerline_HighPrecision.png'), dpi=600)
np.savez(os.path.join(output_dir, 'G_CaseA_Centerline_Data.npz'), **validation_1d)
plt.savefig(os.path.join(output_dir, 'G_CaseA_2D_HighPrecision.png'), dpi=600, bbox_inches='tight')
np.savez(os.path.join(output_dir, 'G_CaseA_2D_Data.npz'), **validation_2d)
```

### 第1层：直接依赖
| 依赖项 | 路径 | 说明 |
|--------|------|------|
| Case A模型 | Results_3D_CaseA/model.pkl | 纯吸收案例 |
| 解析解 | Beer-Lambert (内置) | exact_solution_pure_absorption() |

**代码位置 (Line 180, 364):**
```python
def plot_G_comparison(model_path="Results_3D_CaseA/model.pkl"):
```

### 第2层：解析解实现
```python
# validate_3d_pure_absorption.py 内置 (Line 67-89)
def exact_solution_pure_absorption(kappa, x, mu, epsilon=1e-10):
    """Beer-Lambert 精确解析解"""
    X, Mu = np.meshgrid(x, mu)
    u_exact = np.zeros_like(X)
    positive_mask = Mu > epsilon
    u_exact[positive_mask] = np.exp(-kappa * X[positive_mask] / Mu[positive_mask])
    return u_exact
```

---

## 三、PINN vs FMC 验证溯源

### 第0层：验证代码
```
Evaluation/validate_pinn_vs_fmc_fixed.py
    ├── 输出: Validation_PINN_vs_FMC.png/pdf
    └── 输入依赖:
        ├── Results_3D_CaseB/model.pkl
        ├── Results_3D_CaseC/model.pkl
        └── Solvers/MC/FMC_G_3D_Case{B,C}_FIXED_HighStats.npy

Evaluation/validate_pinn_vs_fmc.py
    └── 类似结构 (旧版本)
```

### 第1层：FMC基准数据
```
Solvers/MC/FMC_3D_Solver_Fixed.py
    ├── 生成: FMC_G_3D_CaseB_FIXED_HighStats.npy
    └── 生成: FMC_G_3D_CaseC_FIXED_HighStats.npy

Solvers/MC/FMC_3D_Solver_Ultra.py
    ├── 生成: FMC_G_3D_CaseB_FIXED_UltraStats.npy (1B光子)
    └── 生成: FMC_G_3D_CaseC_FIXED_UltraStats.npy (1B光子)
```

---

## 四、完整依赖关系图

```
Chapter4_3D_SteadyState/
│
├── Evaluation/                    # 评估脚本 (第0层)
│   ├── plot_3d_paper_figures.py ───────────┐
│   ├── validate_3d_pure_absorption.py ─────┼──→ 生成 Figures_3D/ 和 Figures_3D_Validation/
│   └── validate_pinn_vs_fmc_fixed.py ──────┘
│           │
│           ├── 依赖 Models/ModelClassTorch2.py (Pinns类)
│           ├── 依赖 Models/DatasetTorch2.py
│           ├── 依赖 Models/ObjectClass.py
│           ├── 依赖 Core/ImportFile.py
│           └── 依赖 EquationModels/RadTrans3D_Complex.py (物理引擎)
│
├── Training/                      # 训练代码 (第2层)
│   └── train_3d_multicase.py
│       └── 生成 Results_3D_Case{A,B,C}/model.pkl
│
├── Solvers/MC/                    # FMC求解器 (第2层)
│   ├── FMC_3D_Solver_Fixed.py
│   │   └── 生成 FMC_G_3D_Case{B,C}_FIXED_HighStats.npy
│   └── FMC_3D_Solver_Ultra.py
│       └── 生成 FMC_G_3D_Case{B,C}_FIXED_UltraStats.npy
│
├── Core/                          # 核心模块 (第3层)
│   └── ImportFile.py
│
├── Models/                        # PINN类定义 (第3层)
│   ├── ModelClassTorch2.py        # ← 最关键: Pinns类
│   ├── DatasetTorch2.py
│   └── ObjectClass.py
│
└── EquationModels/                # 物理方程 (第3层)
    ├── RadTrans3D_Complex.py      # OOP版本物理引擎
    └── RadTrans3D_paper.py        # 论文版本
```

---

## 五、3D案例配置汇总

| 案例 | kappa | sigma_s | g | 介质类型 | 训练脚本 | 输出文件夹 |
|------|-------|---------|---|---------|----------|-----------|
| 3D-A | 5.0 | 0.0 | 0.0 | 纯吸收 | train_3d_multicase.py | Results_3D_CaseA/ |
| 3D-B | 0.5 | 4.5 | 0.0 | 各向同性散射 | train_3d_multicase.py | Results_3D_CaseB/ |
| 3D-C | 0.5 | 4.5 | 0.8 | 前向散射 | train_3d_multicase.py | Results_3D_CaseC/ |

---

## 六、关键物理参数

### RadTrans3D_Complex.py (OOP物理引擎)
- `compute_incident_radiation()`: 计算G(x,y,z)
- 使用Gauss-Legendre求积法在方向角上积分
- 支持各向异性散射 (Henyey-Greenstein相函数)

### 坐标系统
- **PINN**: Node-centered (边界包含: linspace(0, 1, n))
- **FMC**: Cell-centered (linspace(0.5/nx, 1-0.5/nx, nx))
- **坐标对齐**: validate_pinn_vs_fmc_fixed.py 中使用 np.interp 进行显式映射

---

## 七、一键运行命令

```bash
# 步骤1: 训练3D模型
cd Chapter4_3D_SteadyState
python Training/train_3d_multicase.py --case A
python Training/train_3d_multicase.py --case B
python Training/train_3d_multicase.py --case C

# 步骤2: 生成FMC基准解 (可选，用于验证)
python Solvers/MC/FMC_3D_Solver_Fixed.py

# 步骤3: 生成论文图表
python Evaluation/plot_3d_paper_figures.py          # → Figures_3D/
python Evaluation/validate_3d_pure_absorption.py    # → Figures_3D_Validation/
python Evaluation/validate_pinn_vs_fmc_fixed.py     # → Validation_PINN_vs_FMC.png
```
