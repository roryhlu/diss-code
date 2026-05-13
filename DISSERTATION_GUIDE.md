# RePAIR Disseration Guide — 6-DoF Pose Estimation &amp; Risk-Averse Grasping

---

## 1. Project Overview

This is a Level 4 (Masters-equivalent) dissertation project at Durham University. The goal is to build and evaluate a **hybrid perception-to-control pipeline** that can:

1. Estimate the **6-DoF pose** (3 translation + 3 rotation) of irregular, textureless archaeological pottery fragments from the RePAIR dataset
2. Use that pose estimate to **plan and execute robust robotic grasps** on a Wlkata Mirobot 6-DoF desktop arm
3. **Quantify uncertainty** in both the perception and grasping stages to ensure the robot does not destroy fragile fragments

The pipeline runs on real hardware (Intel RealSense D405 depth camera, Wlkata Mirobot arm, ROS2 Jazzy, MoveIt2) but all modules work offline with point-cloud data for evaluation.

---

## 2. Why This Matters

Archaeological fragments recovered from dig sites need to be sorted, catalogued, and eventually reassembled. These are **fragile, priceless artefacts** — a single bad grasp destroys irreplaceable history. The fragments are also:

- **Textureless**: subsurface scattering in plaster/ceramic goes in random directions, so RGB cameras produce unreliable color values — standard photometric features (SIFT, ORB) collapse
- **Irregular shapes**: no symmetry to exploit, no CAD model to match against
- **Cluttered**: fragments lie in heaps on tabletops; the robot must pick one without disturbing others

The solution combines **deep learning** for geometric feature extraction, **certifiably robust optimisation** for pose registration, and **risk-averse planning** via uncertainty quantification.

---

## 3. Hardware &amp; Software Stack

| Component | Specification |
|---|---|
| Robot | Wlkata Mirobot 6-DoF desktop arm |
| Sensor | Intel RealSense D405 Sub-Millimeter Depth Camera |
| OS | Ubuntu 24.04 |
| Middleware | ROS2 Jazzy, MoveIt2 |
| Deep Learning | PyTorch ≥2.0 |
| Point Cloud | Open3D ≥0.17 |
| Optimisation | SciPy ≥1.10 (HiGHS LP solver) |
| Mesh | Trimesh ≥3.21 |
| Registration | TEASER++ (optional Python bindings) |
| Languages | Python 3.10+, modern C++ |

---

## 4. Pipeline Architecture

```
                    DEPTH CAMERA / PLY FILE
                           │
                   ┌───────▼────────┐
                   │  Voxel Grid +  │  Module 1a: Preprocessing
                   │  PCA Normals   │
                   └───────┬────────┘
                           │
              ┌────────────┼────────────┐
              │                         │
     ┌────────▼────────┐     ┌─────────▼─────────┐
     │   FPFH Features │     │  GeoTransformer +  │  Module 1b: Feature Extraction
     │  (33-D Geometry)│     │   MC Dropout       │
     └────────┬────────┘     └─────────┬─────────┘
              │                         │
              └──────────┬──────────────┘
                         │
              ┌──────────▼───────────┐
              │  TEASER++ Global     │  Module 2: SE(3) Registration
              │  Registration (TLS)  │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  MC Dropout Pose     │  Module 3a: Uncertainty
              │  Covariance Σ (6×6)  │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  Force-Closure LP +  │  Module 3b: Grasp Synthesis
              │  CVaR Filter (α=0.05)│
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  MoveIt2 Grasp       │  Module 4: Execution
              │  Execution (ROS2)    │
              └──────────────────────┘
```

---

## 5. Mathematical Foundations (Detailed)

### 5.1 Voxel Downsampling

Given a raw point cloud with potentially millions of noisy points, we partition 3D space into a regular grid of cubes with edge length `v_size` (typically 5 mm). All points falling inside voxel `v` collapse to their centroid:

$\mathbf{p}_{\mathrm{voxel}} = \frac{1}{N_v}\sum_{i=1}^{N_v} \mathbf{p}_i$

This enforces **uniform point density**, which is critical because surface normals and descriptors compute neighbourhood statistics that break down when sampling is uneven.

### 5.2 PCA Normal Estimation

For each remaining point, we query its `k=30` nearest neighbours via KD-Tree. From the neighbourhood $\{\mathbf{p}_1, \dots, \mathbf{p}_k\}$, we construct the local 3×3 covariance matrix:

$C = \frac{1}{k}\sum_{j=1}^k (\mathbf{p}_j - \boldsymbol{\mu})(\mathbf{p}_j - \boldsymbol{\mu})^\top, \quad \boldsymbol{\mu} = \frac{1}{k}\sum_j \mathbf{p}_j$

We eigendecompose $C = V \Lambda V^\top$ where eigenvalues are sorted $\lambda_1 \geq \lambda_2 \geq \lambda_3$. The eigenvector $\mathbf{v}_3$ corresponding to the smallest eigenvalue $\lambda_3$ is the **surface normal** — it points in the direction of least variance, which is perpendicular to the local tangent plane.

The ratio $\lambda_3 / (\lambda_1 + \lambda_2 + \lambda_3)$ gives a quality metric: near 0 means a clear planar surface; near 1/3 means isotropic noise (poor normal).

### 5.3 Fast Point Feature Histograms (FPFH)

FPFH descriptors are 33-dimensional geometric features designed to be **robust to photometric collapse**. They are computed from angular relationships between normals.

For any two points $\mathbf{p}_i, \mathbf{p}_j$ with normals $\mathbf{n}_i, \mathbf{n}_j$, define a local Darboux frame:

$\mathbf{u} = \mathbf{n}_i$
$\mathbf{v} = (\mathbf{p}_j - \mathbf{p}_i) \times \mathbf{u}$
$\mathbf{w} = \mathbf{u} \times \mathbf{v}$

The three angular features are:

$\alpha = \mathbf{v} \cdot \mathbf{n}_j$
$\phi = \mathbf{u} \cdot (\mathbf{p}_j - \mathbf{p}_i) / \|\mathbf{p}_j - \mathbf{p}_i\|$
$\theta = \arctan(\mathbf{w} \cdot \mathbf{n}_j, \mathbf{u} \cdot \mathbf{n}_j)$

A **Simplified Point Feature Histogram (SPFH)** is a 33-bin histogram (11 bins for α + 11 for φ + 11 for θ). The **FPFH** averages SPFH over neighbours:

$\text{FPFH}(\mathbf{p}_i) = \text{SPFH}(\mathbf{p}_i) + \frac{1}{k}\sum_{j=1}^k \frac{1}{\omega_j} \text{SPFH}(\mathbf{p}_j)$

where $\omega_j$ is the distance between points. This creates a smooth, discriminative, 33-D descriptor per point.

### 5.4 TEASER++ Registration — Why TLS Beats L2

Standard registration (ICP) minimises:

$L_2: \quad \min_{T \in SE(3)} \sum_i e_i^2$

where $e_i = \|\mathbf{q}_i - T(\mathbf{p}_i)\|$ is the residual. The problem: **one outlier** with a large residual contributes $e_i^2$, which can be enormous. The optimiser warps the entire solution to accommodate that one bad correspondence.

TEASER++ uses a **Truncated Least Squares (TLS)** cost:

$\text{TLS}: \quad \min_{T \in SE(3)} \sum_i \min(e_i^2, c^2)$

Residuals beyond the truncation threshold $c$ (recommended 10 mm) are **clamped** to a constant $c^2$. The gradient of $\min(e_i^2, c^2)$ is zero for $e_i > c$, meaning **outliers exert zero force** on the optimisation.

This makes TEASER++ **certifiably robust to ≥99% outliers** — the worst contamination $\epsilon$ is bounded by the Maximum Clique phase.

**TEASER++ works in 4 stages:**

1. **Maximum Clique Inlier Selection** — Build a consistency graph between pairs (i,j) of putative correspondences. An edge exists iff $|\|\mathbf{q}_i - \mathbf{q}_j\| - \|\mathbf{p}_i - \mathbf{p}_j\|| \leq 2\varepsilon$ (rigid-body distance invariance). Extract the maximum clique: this is a high-confidence inlier set.

2. **GNC-TLS Rotation** — Graduate Non-Convexity: start with a convex approximation, then gradually anneal to the full non-convex TLS. Solved with minimal moments or SDP relaxation.

3. **Adaptive Voting Translation** — Each surviving correspondence votes $\mathbf{t}_i = \mathbf{q}_i - R\mathbf{p}_i$. The maximum-consensus location is the translation estimate.

4. **SDP Certificate** — A semidefinite programming relaxation provides a **provable suboptimality bound** — a mathematical guarantee on how far the solution is from the global TLS optimum. This is the "certificate" reported in the output.

### 5.5 Grasp Wrench Space &amp; Force-Closure

A robot grasp is **force-closed** if it can resist any external wrench (force + torque). For a two-finger parallel-jaw gripper with Coulomb friction coefficient $\mu$:

Each contact supports a **friction cone** — the set of all force vectors inside a cone of half-angle $\alpha = \arctan(\mu)$. We discretise this cone with $m=8$ generators:

$\mathbf{f}_k = \cos(\alpha)\hat{\mathbf{n}} + \sin(\alpha)(\cos\theta_k \mathbf{u} + \sin\theta_k \mathbf{v})$
$\theta_k = 2\pi k / m, \quad k = 1,\dots,m$

where $\hat{\mathbf{n}}$ is the inward surface normal and $\mathbf{u}, \mathbf{v}$ span the tangent plane.

Each generator produces a **wrench** (6-D: 3 force + 3 torque):

$\mathbf{w}_k = \begin{bmatrix} \mathbf{f}_k \\ \mathbf{c} \times \mathbf{f}_k \end{bmatrix} \in \mathbb{R}^6$

The **Grasp Wrench Space (GWS)** is the convex hull of all wrench vectors from all contacts. The grasp is force-closed iff the origin is **strictly inside** the GWS.

**Force-Closure Linear Program:**

$\max \; \varepsilon \quad \text{s.t.} \quad W\boldsymbol{\alpha} = \mathbf{0},\; \mathbf{1}^\top\boldsymbol{\alpha} = 1,\; \alpha_j \geq \varepsilon$

If the optimal $\varepsilon > 0$, the origin is in the interior of the convex hull → **force-closure**. The value of $\varepsilon$ is the **grasp quality metric** — higher is better (more margin against wrenches).

### 5.6 Monte Carlo Dropout &amp; Epistemic Uncertainty

**Why MC Dropout?** The fragment shapes are reconstructed from noisy depth data. Hidden, occluded, or subsurface-scattered geometry is inherently uncertain. Standard deep learning models output a single "best guess" — they don't tell you **how confident** to be.

With MC Dropout, we keep dropout active at **inference time**. Running the model $T$ times with different random dropout masks produces $T$ different outputs $\{\hat{\mathbf{y}}_t\}_{t=1}^T$. The **epistemic variance** (model uncertainty) at each point $\mathbf{p}_i$ is:

$\sigma^2_i = \frac{1}{T-1}\sum_{t=1}^T \|\hat{\mathbf{y}}_{t,i} - \boldsymbol{\mu}_i\|^2$

where $\boldsymbol{\mu}_i = \frac{1}{T}\sum_t \hat{\mathbf{y}}_{t,i}$ is the mean prediction. High variance indicates the model struggles with that geometry — likely occluded or complex surfaces.

**Why Welford's Algorithm?** Storing all $T$ outputs for $N$ points requires $O(T \times N)$ memory. Welford's online accumulator computes the mean and variance incrementally in $O(N)$ memory regardless of $T$.

### 5.7 Pose Covariance in the SE(3) Lie Algebra

The pose estimates from each MC pass are $4 \times 4$ rigid-body transformations $T_t \in SE(3)$. We cannot compute covariance directly in $SE(3)$ because it is not a vector space. Instead, we map to the **Lie algebra** $\mathfrak{se}(3)$:

1. **Log map**: $\boldsymbol{\xi}_t = \log(T_t) \in \mathbb{R}^6$, the twist vector $[\mathbf{v}; \boldsymbol{\omega}]$ (3 linear + 3 angular velocity)
2. **Mean**: $\bar{\boldsymbol{\xi}} = \frac{1}{T}\sum_t \boldsymbol{\xi}_t$
3. **Covariance**: $\Sigma = \frac{1}{T-1}\sum_t (\boldsymbol{\xi}_t - \bar{\boldsymbol{\xi}})(\boldsymbol{\xi}_t - \bar{\boldsymbol{\xi}})^\top \in \mathbb{R}^{6 \times 6}$

The 6×6 covariance decomposes:
- $\Sigma_{tt} = \Sigma[0:3, 0:3]$ — translation covariance (m²)
- $\Sigma_{rr} = \Sigma[3:6, 3:6]$ — rotation covariance (rad²)
- $\Sigma_{tr} = \Sigma[0:3, 3:6]$ — cross-covariance

To visualise spatial uncertainty, the 6×6 pose covariance is projected onto each point via the Jacobian of the SE(3) action:

$J_k = [I_3 \mid -[\mathbf{p}_k]_\times] \in \mathbb{R}^{3 \times 6}$

$\Sigma_{\mathbf{p}_k} = J_k \Sigma J_k^\top \in \mathbb{R}^{3 \times 3}$ — the per-point 3×3 spatial covariance

The scalar variance per point is $\sigma^2_k = \text{trace}(\Sigma_{\mathbf{p}_k})$. High values → blue; medium → white; low → red in the variance cloud visualisation.

### 5.8 CVaR — Conditional Value-at-Risk Grasp Filter

Given $N=100$ geometric realisations of the fragment (each perturbed by MC dropout variance), we test force-closure on each. Let $\varepsilon_{(1)} \leq \varepsilon_{(2)} \leq \dots \leq \varepsilon_{(N)}$ be the sorted grasp quality scores (ascending = worst first).

The Conditional Value-at-Risk at level $\alpha = 0.05$ is:

$\mathrm{CVaR}_{0.05} = \frac{1}{K}\sum_{k=1}^K \varepsilon_{(k)}, \quad K = \lceil 0.05 \cdot N \rceil$

This is the **average quality over the worst 5% of geometric realisations**. A grasp that scores $\varepsilon = 0.1$ in 95% of cases but $\varepsilon = 0$ (no force-closure) in 5% of cases has $\mathrm{CVaR}_{0.05} = 0$ and is **rejected**.

This is strictly safer than **expected-value planning** (which would see $\mathbb{E}[\varepsilon] \approx 0.095$ and accept the grasp, risking fragment destruction 5% of the time).

### 5.9 Hand-Eye Calibration (AX=XB, Tsai-Lenz)

The camera sees the fragment, but the robot needs coordinates in its own frame. The unknown rigid-body transform $X \in SE(3)$ between camera and robot must be calibrated.

Given $M \geq 2$ motion pairs $(A_i, B_i)$ where $A_i$ is the robot end-effector motion and $B_i$ is the corresponding camera-observed calibration-pattern motion, we solve:

$A_i X = X B_i \quad \Rightarrow \quad AX = XB$

**Tsai-Lenz Decomposition:**

1. **Rotation** — From axis-angle $\mathbf{r}_a = \log(R_A), \mathbf{r}_b = \log(R_B)$:
   $\text{skew}(\mathbf{r}_a + \mathbf{r}_b) \cdot \mathbf{P}'_x = \mathbf{r}_b - \mathbf{r}_a$
   Solved via least squares; recovers $R_X$ from $\mathbf{P}'_x = 2\sin(\theta/2)\hat{\mathbf{n}}$.

2. **Translation** — With $R_X$ known:
   $(R_A - I) \cdot \mathbf{t}_X = R_X\mathbf{t}_B - \mathbf{t}_A$
   Solved via least squares.

---

## 6. Project Structure &amp; File Reference

### Top-Level Files

| File | Purpose |
|---|---|
| `AGENTS.md` | AI assistant configuration — project specification, rules, hardware, tech stack |
| `voxel_downsample_normals.py` | Standalone Module 1a: voxel downsampling + PCA normal estimation |
| `DISSERTATION_GUIDE.md` | This document — comprehensive project guide |

### `registration/` — SE(3) Registration Sub-Package

| File | Purpose | Key Functions |
|---|---|---|
| `__init__.py` | Public API exports (12 symbols) | Re-exports `register_teaser`, `weighted_svd_se3`, etc. |
| `fpfh_features.py` | 33-D FPFH descriptors + matching | `compute_fpfh()`, `match_features()`, `extract_correspondence_clouds()` |
| `teaser_registration.py` | TEASER++ with TLS + SE(3) certificate | `register_teaser()`, `register_scene_to_cad()`, `SE3Result`, `TeaserParams` |
| `weighted_svd.py` | Weighted Kabsch algorithm (PyTorch) | `weighted_svd_se3()` — batched SE(3) via SVD |
| `se3_utils.py` | SE(3) utilities (PyTorch) | `transform_points()`, `extract_rt()`, `compose()`, `inverse_transform()` |

### `uncertainty/` — Epistemic Uncertainty Sub-Package

| File | Purpose | Key Functions |
|---|---|---|
| `__init__.py` | Public API exports | Re-exports all uncertainty functions |
| `geotransformer.py` | GeoTransformer neural network + MC Dropout bottleneck | `GeoTransformer`, `GeometricAttention`, `MCDropoutBottleneck`, `FeatureDecoder` |
| `mc_inference.py` | Welford online MC accumulator | `run_mc_passes()` — memory-efficient MC Dropout inference |
| `pose_covariance.py` | SE(3) Lie algebra 6×6 covariance | `compute_pose_covariance()`, `project_spatial_variance()`, `se3_log()`, `se3_exp()` |
| `variance_cloud.py` | Variance cloud I/O + colormap | `compute_variance_cloud()`, `save_variance_cloud()`, `visualise_variance()` |

### `scripts/` — Standalone CLI Tools

| File | Purpose | CLI Command |
|---|---|---|
| `compute_fpfh.py` | FPFH descriptor computation + PCA-RGB visualisation | `python scripts/compute_fpfh.py fragment.ply --viz` |
| `teaser_register.py` | End-to-end TEASER++ registration pipeline | `python scripts/teaser_register.py src.ply tgt.ply --viz` |
| `fpfh_parameter_sweep.py` | Grid search for optimal FPFH parameters | `python scripts/fpfh_parameter_sweep.py src.ply tgt.ply --output sweep.csv` |
| `force_closure.py` | Two-finger GWS + Force-Closure LP analysis | `python scripts/force_closure.py fragment.stl --contact1 "x y z" --contact2 "x y z" --viz` |
| `cvar_grasp_validator.py` | CVaR multi-candidate grasp validation + ranking | `python scripts/cvar_grasp_validator.py variance_cloud.pcd --candidates sample_candidates.json --output accepted.json` |
| `mc_dropout_variance.py` | MC inference → epistemic variance cloud | `python scripts/mc_dropout_variance.py fragment.pcd --model checkpoint.pt --num-passes 50` |
| `mc_pose_covariance.py` | MC loop → 6×6 pose covariance Σ | `python scripts/mc_pose_covariance.py src.ply tgt.ply --model checkpoint.pt --num-passes 50` |
| `sample_candidates.json` | Sample grasp candidate contact-pair coordinates | Input for `cvar_grasp_validator.py` |

### `repair_simulation/` — ROS2 MoveIt2 Package

| File | Purpose |
|---|---|
| `package.xml` | ROS2 package manifest (format 3), dependencies |
| `CMakeLists.txt` | `colcon build` configuration |
| `setup.py` | Python entry_points registration |
| `resource/repair_simulation` | Package marker file |
| `repair_simulation/__init__.py` | Package init, exports `GraspExecutor`, `HandEyeCalibration` |
| `repair_simulation/hand_eye.py` | AX=XB Tsai-Lenz hand-eye calibration solver |
| `repair_simulation/grasp_executor.py` | MoveIt2 grasp execution node (plan + execute trajectory) |

---

## 7. Quick Start — Step-by-Step Timeline

This is the recommended order for running the pipeline. Each step builds on the previous one.

### Step 1: Preprocessing (5 minutes)

Voxel-downsample your raw fragment point cloud and estimate PCA surface normals:

```bash
python voxel_downsample_normals.py fragment.ply --voxel-size 0.005 --knn 30 --viz
```

**What this does:** Reads a PLY/PCD file → partitions into 5 mm voxels → collapses each voxel to its centroid → computes normals via PCA on 30 nearest neighbours → saves `fragment_ds.ply` → opens Open3D visualisation (normals as purple lines).

**Expected result:** A visually cleaner, uniformly-spaced point cloud with well-oriented normal vectors (purple lines pointing outward).

---

### Step 2: Compute FPFH Descriptors (2 minutes)

```bash
python scripts/compute_fpfh.py fragment.ply --voxel-size 0.005 --fpfh-radius 0.025 --stats --viz
```

**What this does:** Computes 33-D geometric descriptors per point → PCA-projects to 3 RGB channels → prints descriptor statistics → renders side-by-side (original geometry vs. FPFH-coloured cloud).

**Expected result:** A coloured point cloud where regions of similar geometry share similar colours. Overhangs, ridges, and flat surfaces should have distinct colour clusters. The descriptor should have non-zero entropy (>2.0) — indicating discriminative features.

---

### Step 3: TEASER++ Registration (3 minutes)

To register a source fragment to a target scene or CAD model:

```bash
python scripts/teaser_register.py src_fragment.ply target_scene.ply \
    --voxel-size 0.005 --c-threshold 0.01 --output result.ply --viz
```

**What this does:**
1. Voxel-downsamples both clouds
2. Computes FPFH descriptors on both
3. Matches descriptors with mutual nearest-neighbour + Lowe ratio test
4. Runs TEASER++ (or RANSAC fallback if TEASER++ bindings not installed)
5. Applies resulting SE(3) transform to source
6. Reports rotation angle, translation norm, and SDP certification
7. Visualises: source (orange), target (blue), aligned (green)

**Expected result:**
- Rotation angle < 5° (good alignment) or < 20° (approximate alignment)
- Translation norm matches expected camera-to-object distance
- SDP certificate close to 0 (tight optimality bound)
- Aligned green cloud overlapping the blue target cloud
- If the certificate is "N/A (RANSAC)" — install TEASER++ Python bindings for certified results

**Understanding the certificate:** The SDP certificate is an upper bound on $\text{cost}(T_{\text{found}}) - \text{cost}(T_{\text{opt}})$. A value of 0.001 means the found transformation is within $10^{-3}$ of the globally optimal TLS solution — provably.

---

### Step 4: FPFH Parameter Optimisation (Optional, 10-30 minutes)

If registration quality is poor, sweep FPFH parameters to find the best configuration for your fragment type:

```bash
python scripts/fpfh_parameter_sweep.py src.ply tgt.ply \
    --quick --output sweep_results.csv
```

**What this does:** Grid-searches `normal_radius` (0.005–0.05), `fpfh_radius` (0.01–0.1), and `ratio_threshold` (0.7–0.99). For each combination, runs full registration, computes inlier rate, registration error, and a composite score.

**Expected result:** A CSV file with all parameter combinations ranked by quality score. Identify the combination that maximises inlier rate while minimising registration error.

---

### Step 5: MC Dropout Variance Cloud (10 minutes, requires GPU recommended)

Generate an epistemic uncertainty map over the fragment's surface:

```bash
python scripts/mc_dropout_variance.py fragment.pcd \
    --model checkpoints/geotransformer_best.pt \
    --num-passes 50 --dropout-rate 0.2 \
    --output variance_cloud.pcd --viz
```

**What this does:**
1. Loads the GeoTransformer model from checkpoint
2. Runs $T = 50$ stochastic forward passes with MC Dropout active
3. Uses Welford's algorithm to compute mean and variance per point
4. Maps variance to colour (blue = certain, red = uncertain)
5. Saves coloured variance cloud PCD
6. Prints variance statistics (mean, median, p95, p99)

**Expected result:**
- A coloured point cloud showing which surface regions have high epistemic uncertainty
- Typically: edges, thin features, and occluded regions are red (high variance)
- Flat, well-sampled surfaces are blue (low variance)
- This variance cloud feeds into Step 7 for risk-averse grasping

**Without a trained model:** The script works with randomly initialised weights. Results will be less meaningful but the pipeline functions.

---

### Step 6: MC Pose Covariance (10 minutes)

Quantify the 6×6 pose uncertainty from the MC Dropout loop:

```bash
python scripts/mc_pose_covariance.py src.ply tgt.ply \
    --model checkpoints/geotransformer_best.pt \
    --num-passes 50 --voxel-size 0.005 \
    --output covariance_cloud.pcd --viz
```

**What this does:**
1. Runs the full pose inference pipeline (GeoTransformer features → TEASER++ registration) in an MC loop
2. Collects all pose estimates $\{T_t\}_{t=1}^{50}$
3. Maps to $\mathfrak{se}(3)$ Lie algebra, computes 6×6 covariance $\Sigma$
4. Projects spatial uncertainty onto each point
5. Saves: coloured covariance cloud PCD, aligned mean pose PLY, $\Sigma$ as `.npy`
6. Prints covariance report: RMS translation uncertainty (mm), RMS rotation uncertainty (degrees), eigenvalues of $\Sigma$, per-point variance statistics

**Expected result:**
- RMS translation uncertainty < 5 mm (confident) or > 20 mm (explore why)
- RMS rotation uncertainty < 3° (confident)
- Covariance matrix saved as `.npy` for downstream uncertainty-aware planning

---

### Step 7: Force-Closure Analysis (1 minute)

Test whether a candidate grasp pair is force-closed:

```bash
python scripts/force_closure.py fragment.stl \
    --contact1 "0.023 -0.015 0.041" \
    --contact2 "-0.019 0.021 -0.038" \
    --mu 0.5 --quality --viz
```

**What this does:**
1. Loads mesh from STL file
2. Finds nearest vertices to specified contact points
3. Extracts surface normals at those points
4. Builds polyhedral friction cones (8 generators each, $\mu = 0.5$ for plaster)
5. Constructs combined 6×16 wrench matrix $W$
6. Solves the force-closure LP → produces $\varepsilon$ (quality score) and boolean FC
7. Checks analytical antipodal condition as sanity check
8. Visualises the mesh with contact points and normal directions

**Expected result:**
- `Force-Closure: Yes` with $\varepsilon > 0$ → the grasp is force-closed
- $\varepsilon \approx 0.01$–0.1 is typical for good grasps
- $\varepsilon \approx 0.001$ is marginal (close to losing FC)
- The LP solver uses HiGHS (SciPy ≥1.10); if unavailable, falls back to antipodal heuristic

---

### Step 8: CVaR Grasp Validation (5 minutes)

Validate candidate grasps against geometric uncertainty from the variance cloud:

```bash
python scripts/cvar_grasp_validator.py variance_cloud.pcd \
    --candidates scripts/sample_candidates.json \
    --mu 0.5 --num-realizations 100 --top-k 5 \
    --output accepted_grasps.json --viz
```

**What this does:**
1. Loads the variance cloud from Step 5 (mean + per-point variance)
2. Loads candidate grasp pairs from JSON
3. **Baseline check:** Tests force-closure on the reference (mean) geometry
4. **Uncertainty sampling:** Generates $N = 100$ geometric realisations by perturbing each point: $\mathbf{p}_i^{(k)} \sim \mathcal{N}(\boldsymbol{\mu}_i, \sigma^2_i I_3)$
5. For each realisation: re-estimates normals, rebuilds friction cones, tests force-closure
6. Computes $\mathrm{CVaR}_{0.05}$ = average $\varepsilon$ over the worst 5% (5 realisations)
7. **Accepts** a grasp only if BOTH: baseline FC passes AND $\mathrm{CVaR}_{0.05} > 10^{-12}$
8. Ranks accepted grasps by CVaR value (higher = more robust)
9. Saves accepted grasps to JSON for the ROS2 executor
10. Visualises: variance cloud with green spheres (accepted) / red spheres (rejected) at contact points

**Expected result:**
- 1–3 grasps accepted (most fail either baseline or CVaR)
- Accepted grasps sorted by CVaR quality
- Green contact spheres on blue (low-uncertainty) regions of the fragment
- Red contact spheres on red (high-uncertainty) regions — these are correctly rejected
- The JSON output feeds directly into Step 9

---

### Step 9: ROS2 Simulation — Build (2 minutes)

```bash
colcon build --symlink-install --packages-select repair_simulation
source install/setup.bash
```

**What this does:** Compiles the ROS2 package, symlinks Python files (no rebuild needed on edits).

**Expected result:** Build summary with no errors. The `repair_simulation` package is now available to `ros2 run`.

---

### Step 10: ROS2 Simulation — Execute Grasp

```bash
# Option A: Execute top-ranked CVaR grasp from JSON
ros2 run repair_simulation grasp_executor \
    --ros-args -p grasp_file:=accepted_grasps.json

# Option B: Execute a manually specified pose
ros2 run repair_simulation grasp_executor \
    --ros-args -p target_x:=0.35 -p target_y:=0.0 -p target_z:=0.12 \
    -p roll:=0.0 -p pitch:=3.14 -p yaw:=0.0
```

**What this does (Option A):**
1. Reads `accepted_grasps.json` from Step 8
2. Takes the top-ranked (highest CVaR) grasp
3. Computes grasp centre (midpoint of the two contacts) and grasp direction (vector between contacts)
4. Builds an SE(3) grasp pose: tool Z = downward, tool X aligned with inter-contact direction
5. Transforms through hand-eye calibration to robot frame
6. Executes a **4-stage trajectory**:
   - **Stage 1:** Open gripper
   - **Stage 2:** Joint-space plan to pre-grasp (5 cm above target)
   - **Stage 3:** Cartesian descent to grasp pose (waypoints at 5 mm steps)
   - **Stage 4:** Close gripper, Cartesian lift to retreat

**Expected result:**
- Trajectory planning succeeds with feasibility ≥ 95%
- Robot arm moves smoothly through stages
- Gripper closes on the fragment at the validated contact points
- Arm lifts the fragment to the retreat position
- In simulation (MoveIt2 RViz), the planned trajectory is visualised

**What this does (Option B):**
- Direct pose execution bypassing the CVaR pipeline
- Useful for testing calibration and trajectory execution independently

---

## 8. Interpreting Results

### Registration Quality Table

| Metric | Excellent | Good | Fair | Poor |
|---|---|---|---|---|
| Rotation error | < 1° | 1–5° | 5–15° | > 15° |
| Translation error | < 2 mm | 2–10 mm | 10–30 mm | > 30 mm |
| SDP certificate | < 0.001 | 0.001–0.01 | 0.01–0.1 | > 0.1 or N/A |
| Inlier rate | > 80% | 50–80% | 20–50% | < 20% |

### Grasp Quality Interpretation

| $\varepsilon$ (FC quality) | Meaning |
|---|---|
| > 0.05 | Very robust — large margin against external wrenches |
| 0.01–0.05 | Good — adequate for tabletop manipulation |
| 0.001–0.01 | Marginal — near boundary of friction cone |
| 0 | No force-closure — grasp will slip |

### Variance Cloud Colours

| Colour | Meaning |
|---|---|
| Blue | Low epistemic variance — model confident |
| White | Moderate uncertainty |
| Red | High epistemic variance — model uncertain (edges, thin features, occlusions) |

### CVaR Behaviour

- A grasp accepted with $\mathrm{CVaR}_{0.05} = 0.03$ is robust: even in the worst 5% of geometric realisations, the average quality is 0.03
- A grasp rejected with $\mathrm{CVaR}_{0.05} = 0$ means at least one of the worst 5 realisations had zero force-closure — the grasp is unsafe
- **CVaR > expected value** is the normal behaviour for conservative planning

---

## 9. Installation &amp; Dependencies

### Base Python Dependencies

```bash
pip install numpy torch open3d scipy trimesh
```

### TEASER++ (Optional but Recommended)

TEASER++ provides the TLS cost function and SDP certificate. Install via:

```bash
pip install teaserpp-python
```

Without it, the pipeline falls back to Open3D RANSAC (L2 cost, no outlier rejection, no certificate).

### ROS2 (For Simulation Only)

```bash
# Ubuntu 24.04
sudo apt install ros-jazzy-ros-base
sudo apt install ros-jazzy-moveit
pip install rclpy moveit-py
```

### Checking Your Installation

```bash
python -c "import numpy, torch, open3d, scipy, trimesh; print('All core deps OK')"
python -c "import teaserpp_python" 2>/dev/null && echo "TEASER++ OK" || echo "TEASER++ not installed (RANSAC fallback)"
```

---

## 10. Evaluation Metrics

### ADD-S (Average Distance of Symmetric-Defined)

Used when the object has continuous symmetries (e.g., a bowl is rotationally symmetric about its axis):

$\text{ADD-S}(\hat{T}, T_{gt}) = \frac{1}{N}\sum_i \min_{\mathbf{p}_j \in \mathcal{S}} \|\hat{T}(\mathbf{p}_i) - T_{gt}(\mathbf{p}_j)\|$

where $\mathcal{S}$ is the set of points on the model. For each transformed model point, find the closest ground-truth model point, averange distances. Robust to symmetries because matching is per-point, not per-correspondence.

### Chamfer Distance

For unregistered evaluation:

$d_{\text{Chamfer}}(P, Q) = \frac{1}{|P|}\sum_{p \in P} \min_{q \in Q} \|p - q\| + \frac{1}{|Q|}\sum_{q \in Q} \min_{p \in P} \|q - p\|$

Lower is better. Measures bidirectional geometric similarity. Works without known correspondences.

---

## 11. Troubleshooting Guide

| Symptom | Likely Cause | Solution |
|---|---|---|
| Registration fails completely (source far from target) | Descriptors not discriminative | Increase `--fpfh-radius` to 0.035, run `fpfh_parameter_sweep.py` |
| TEASER++ certificate is "N/A (RANSAC)" | `teaserpp-python` not installed | `pip install teaserpp-python` |
| Rotation error > 30° | Insufficient inlier ratio | Decrease `--ratio-threshold` to 0.7, increase `--c-threshold` to 0.05 |
| All grasps rejected by CVaR | Variance cloud has very high uncertainty | Check model checkpoint, increase `--num-passes` to 100, verify dropout rate |
| Force-closure LP fails | SciPy < 1.10 (no HiGHS) | `pip install scipy>=1.10` or use `--no-lp` flag for antipodal heuristic |
| Variance cloud is all one colour | Dropout rate too low or model in eval mode | Set `--dropout-rate 0.2`, ensure MC dropout is active |
| ROS2 build fails | Missing dependencies | `rosdep install --from-paths . --ignore-src -y` |
| `ModuleNotFoundError: No module named 'registration'` | Package not on PYTHONPATH | Run from project root, or `pip install -e .` if setup.py exists |
| Open3D visualisation window doesn't show | Headless environment | Use `--no-viz` flag |

---

## 12. Quick Reference — All CLI Commands

```bash
# Module 1: Perception
python voxel_downsample_normals.py fragment.ply --voxel-size 0.005 --knn 30 --viz
python scripts/compute_fpfh.py fragment.ply --voxel-size 0.005 --fpfh-radius 0.025 --stats --viz

# Module 2: Registration
python scripts/teaser_register.py src.ply tgt.ply --voxel-size 0.005 --c-threshold 0.01 --viz
python scripts/fpfh_parameter_sweep.py src.ply tgt.ply --quick --output sweep.csv

# Module 3: Uncertainty + Grasping
python scripts/mc_dropout_variance.py fragment.pcd --model checkpoint.pt --num-passes 50 --output var.pcd --viz
python scripts/mc_pose_covariance.py src.ply tgt.ply --model checkpoint.pt --num-passes 50 --output cov.pcd --viz
python scripts/force_closure.py fragment.stl --contact1 "x1 y1 z1" --contact2 "x2 y2 z2" --mu 0.5 --viz
python scripts/cvar_grasp_validator.py var.pcd --candidates sample_candidates.json --mu 0.5 --num-realizations 100 --output accepted.json --viz

# Module 4: ROS2 Simulation
colcon build --symlink-install --packages-select repair_simulation
source install/setup.bash
ros2 run repair_simulation grasp_executor --ros-args -p grasp_file:=accepted_grasps.json
```

---

## 13. Phase 1 Summary

- **22 commits** on `main` branch
- **28 files** across 4 packages + standalone scripts
- **~8,000 lines** of strictly-typed Python with inline mathematical derivations

| Module | Status | Key Feature |
|---|---|---|
| 1. Perception | Complete | GeoTransformer + MC Dropout + FPFH + PCA normals |
| 2. Registration | Complete | TEASER++ with TLS, SE(3) certification, Maximum Clique inlier selection |
| 3. Grasping | Complete | GWS, Force-Closure LP, CVaR α=0.05 filter, epistemic Σ in se(3) |
| 4. Simulation | Complete | ROS2 MoveIt2 package, hand-eye calibration, 4-stage top-down grasp |

---

*Rory Hlustik — Durham University Level 4 Dissertation, 2026*
