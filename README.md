# RePAIR — Robotic Pose Estimation & Grasping of Archaeological Fragments

Durham University Level 4 Dissertation — Hybrid perception-to-control pipeline for 6-DoF pose estimation and risk-averse grasping of irregular, textureless archaeological fragments from the [RePAIR dataset](https://www.repairproject.eu/).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   PERCEPTION PIPELINE                        │
│                                                              │
│  Module 1: Deep Learning Perception                          │
│    voxel_downsample_normals.py   Voxel grid + PCA normals    │
│    uncertainty/geotransformer.py GeoTransformer + MC Dropout │
│                                                              │
│  Module 2: Classical Refinement (SE(3) Registration)         │
│    registration/fpfh_features.py      FPFH descriptors       │
│    registration/teaser_registration.py TEASER++ + TLS cost   │
│    registration/weighted_svd.py        Kabsch algorithm      │
│    registration/se3_utils.py           SE(3) utilities       │
│                                                              │
│  Module 3: Uncertainty & Grasping                            │
│    uncertainty/mc_inference.py    Welford MC accumulator      │
│    uncertainty/pose_covariance.py SE(3) Lie algebra Σ        │
│    uncertainty/variance_cloud.py  Epistemic variance PCD     │
│    scripts/force_closure.py       GWS + Force-Closure LP     │
│    scripts/cvar_grasp_validator.py CVaR α=0.05 filter        │
│                                                              │
│  Module 4: ROS2 Simulation                                   │
│    repair_simulation/hand_eye.py      AX=XB Tsai-Lenz        │
│    repair_simulation/grasp_executor.py MoveIt2 top-down grasp│
└──────────────────────────────────────────────────────────────┘
```

## Hardware & Software

| Component | Specification |
|---|---|
| **Robot** | Wlkata Mirobot 6-DoF desktop arm |
| **Sensor** | Intel RealSense D405 Sub-Millimeter Depth Camera |
| **Frameworks** | ROS2 Jazzy, MoveIt2, PyTorch, Open3D |
| **Languages** | Python 3.10+, modern C++ |

---

## Mathematical Foundations

### Module 1 — Perception

**Voxel Downsampling:** Each voxel cell is collapsed to its centroid $\mathbf{p}_{\text{voxel}} = \frac{1}{N_v}\sum_{i} \mathbf{p}_i$, preserving uniform point density.

**PCA Normal Estimation:** For each point's $k=30$ nearest neighbours, the local covariance matrix $C = \frac{1}{k}\sum(\mathbf{p}_j - \boldsymbol{\mu})(\mathbf{p}_j - \boldsymbol{\mu})^\top$ is decomposed. The eigenvector of the smallest eigenvalue is the surface normal.

**FPFH Descriptors:** 33-dimensional geometric features computed via angular triplets $(\alpha, \phi, \theta)$ between the query point normal and each neighbour normal, robust to photometric feature collapse.

**GeoTransformer:** Multi-head geometric attention with distance-based bias $\exp(-\|\mathbf{p}_i - \mathbf{p}_j\|^2 / 2\sigma^2)$, plus MC Dropout bottleneck for epistemic uncertainty quantification.

### Module 2 — TEASER++ Registration

**TLS Objective (vs L2 ICP):**

$$L_2\text{ (ICP):} \quad \min_{T\in SE(3)} \sum_i e_i^2 \quad \text{— outliers dominate quadratically}$$

$$\text{TLS (TEASER++):} \quad \min_{T\in SE(3)} \sum_i \min(e_i^2, c^2) \quad \text{— residuals beyond } c \text{ clamped}$$

The gradient of the TLS cost is zero for residuals $> c$, so subsurface scattering outliers contribute **no force** to the optimisation. Combined with Maximum Clique Inlier Selection (pairwise distance consistency: $d_{ij} = \big|\|\mathbf{q}_i - \mathbf{q}_j\| - \|\mathbf{p}_i - \mathbf{p}_j\|\big| \leq 2\varepsilon$) and Graduated Non-Convexity (GNC) annealing, TEASER++ is certifiably robust to ≥99% outliers.

**SE(3) Certification:** The output includes a suboptimality bound from SDP relaxation — a mathematical guarantee on distance from the global TLS optimum.

### Module 3 — Grasp Wrench Space & CVaR

**Force-Closure LP:**

$$\max \; \varepsilon \quad \text{s.t.} \quad W\boldsymbol{\alpha} = \mathbf{0},\; \mathbf{1}^\top\boldsymbol{\alpha} = 1,\; \alpha_j \geq \varepsilon$$

If $\varepsilon > 0 \implies$ Force-Closure. The wrench matrix $W = [W_1 \mid W_2] \in \mathbb{R}^{6 \times 2m}$ has $2m = 16$ columns (8 friction cone generators per contact, $\mu=0.5$ for plaster).

**CVaR$_5$% Filter:**

$$\text{CVaR}_{5\%} = \frac{1}{K}\sum_{k=1}^K \varepsilon_{(k)},\quad K = \lceil 0.05 \cdot N \rceil$$

Accepts a grasp only if Force-Closure holds in **all** of the worst 5% of geometric realizations (sampled from the MC Dropout epistemic variance cloud). A grasp scoring $\varepsilon=0.1$ in 95% of cases but $\varepsilon=0$ in 5% is correctly rejected with $\text{CVaR}_{5\%}=0$.

### Module 4 — Hand-Eye Calibration

**AX=XB (Tsai-Lenz):**

$$\text{skew}(\mathbf{r}_a + \mathbf{r}_b) \cdot \mathbf{P}'_x = \mathbf{r}_b - \mathbf{r}_a \quad \text{→ least squares for } R_X$$

$$(R_A - I) \cdot \mathbf{t}_X = R_X\mathbf{t}_B - \mathbf{t}_A \quad \text{→ least squares for } \mathbf{t}_X$$

---

## Project Structure

```
diss-code/
├── AGENTS.md                       # Project specification & rules
├── README.md                       # This file
├── .gitignore
├── voxel_downsample_normals.py     # Standalone: voxel + PCA normals
│
├── registration/                   # SE(3) Registration Sub-Package
│   ├── __init__.py                 # Public API exports
│   ├── fpfh_features.py            # FPFH computation & matching
│   ├── teaser_registration.py      # TEASER++ with TLS + SE(3) cert
│   ├── weighted_svd.py             # Kabsch algorithm (PyTorch)
│   └── se3_utils.py                # SE(3) utilities (PyTorch)
│
├── uncertainty/                    # Epistemic Uncertainty Sub-Package
│   ├── __init__.py                 # Public API exports
│   ├── geotransformer.py           # GeoTransformer + MC Dropout
│   ├── mc_inference.py             # Welford's online MC accumulator
│   ├── pose_covariance.py          # SE(3) Lie algebra 6×6 Σ
│   └── variance_cloud.py           # PCD I/O + colormap
│
├── scripts/                        # Standalone CLI Tools
│   ├── compute_fpfh.py             # FPFH PCA-RGB visualisation
│   ├── teaser_register.py          # End-to-end TEASER++ registration
│   ├── fpfh_parameter_sweep.py     # Grid search for FPFH params
│   ├── force_closure.py            # Two-finger GWS + FC analysis
│   ├── cvar_grasp_validator.py     # CVaR multi-candidate ranking
│   ├── mc_dropout_variance.py      # MC inference → variance cloud
│   ├── mc_pose_covariance.py       # MC loop → pose Σ
│   └── sample_candidates.json      # Sample grasp candidates
│
└── repair_simulation/              # ROS2 MoveIt2 Package
    ├── CMakeLists.txt              # colcon build rules
    ├── package.xml                 # ROS2 dependencies
    ├── setup.py                    # Python entry_points
    ├── resource/repair_simulation  # Package marker
    └── repair_simulation/
        ├── __init__.py
        ├── hand_eye.py             # AX=XB Tsai-Lenz calibration
        └── grasp_executor.py       # MoveIt2 top-down grasp node
```

---

## Quick Start

### 1. Voxel Downsampling + Normal Estimation

```bash
python voxel_downsample_normals.py fragment.ply --voxel-size 0.005 --knn 30 --viz
```

### 2. FPFH Descriptor Computation

```bash
python scripts/compute_fpfh.py fragment.ply --voxel-size 0.005 --fpfh-radius 0.025 --stats --viz
```

### 3. TEASER++ Global Registration

```bash
python scripts/teaser_register.py src.ply tgt.ply \
    --voxel-size 0.005 --c-threshold 0.01 --output result.ply --viz
```

### 4. FPFH Parameter Sweep

```bash
python scripts/fpfh_parameter_sweep.py src.ply tgt.ply --quick --output sweep.csv
```

### 5. MC Dropout Variance Cloud

```bash
python scripts/mc_dropout_variance.py fragment.pcd \
    --model checkpoints/geotransformer_best.pt --num-passes 50 \
    --dropout-rate 0.2 --output variance_cloud.pcd --viz
```

### 6. Force-Closure Analysis

```bash
python scripts/force_closure.py fragment.stl \
    --contact1 "0.023 -0.015 0.041" --contact2 "-0.019 0.021 -0.038" \
    --mu 0.5 --quality --viz
```

### 7. CVaR Grasp Validation

```bash
python scripts/cvar_grasp_validator.py variance_cloud.pcd \
    --candidates scripts/sample_candidates.json \
    --mu 0.5 --num-realizations 100 --top-k 5 \
    --output accepted_grasps.json --viz
```

### 8. ROS2 Simulation (Build & Run)

```bash
# Build
colcon build --symlink-install --packages-select repair_simulation
source install/setup.bash

# Run with CVaR output
ros2 run repair_simulation grasp_executor \
    --ros-args -p grasp_file:=accepted_grasps.json

# Run with direct pose
ros2 run repair_simulation grasp_executor \
    --ros-args -p target_x:=0.35 -p target_y:=0.0 -p target_z:=0.12 \
    -p roll:=0.0 -p pitch:=3.14 -p yaw:=0.0
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Linear algebra |
| `torch` | ≥2.0 | PyTorch, neural networks |
| `open3d` | ≥0.17 | Point cloud I/O, visualisation |
| `scipy` | ≥1.10 | Linear programming (HiGHS) |
| `trimesh` | ≥3.21 | Mesh loading, surface normals |
| `teaserpp-python` | (optional) | TEASER++ Python bindings |
| `rclpy` | Jazzy | ROS2 Python client |
| `moveit-py` | Jazzy | MoveIt2 Python API |

Install base dependencies:
```bash
pip install numpy torch open3d scipy trimesh
```

---

## Phase 1 Progress Summary

- **15 commits** on `main` branch
- **28 files** across 4 packages + standalone scripts
- **~8,000 lines** of strictly-typed Python with inline mathematical derivations

| Module | Status | Key Feature |
|---|---|---|
| **1. Perception** | ✓ Complete | GeoTransformer + MC Dropout + FPFH + PCA normals |
| **2. Registration** | ✓ Complete | TEASER++ with TLS, SE(3) certification, Maximum Clique |
| **3. Grasping** | ✓ Complete | GWS, Force-Closure LP, CVaR filter, epistemic Σ |
| **4. Simulation** | ✓ Complete | ROS2 MoveIt2 package, hand-eye calibration, top-down grasp |

---

## License

MIT — Rory Hlustik, Durham University 2026.
