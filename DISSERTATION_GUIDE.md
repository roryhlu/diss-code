# RePAIR Dissertation Guide
## 6-DoF Pose Estimation &amp; Risk-Averse Robotic Grasping of Archaeological Fragments

---

**Author:** Rory Hlustik — Durham University Level 4 Dissertation, 2026

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Background Concepts — The Basics](#2-background-concepts--the-basics)
3. [Why Is This Hard? The Three Challenges](#3-why-is-this-hard-the-three-challenges)
4. [The Pipeline — Big Picture](#4-the-pipeline--big-picture)
5. [Mathematical Foundations — Explained Step by Step](#5-mathematical-foundations--explained-step-by-step)
6. [How Each File Works — Deep Dive](#6-how-each-file-works--deep-dive)
7. [Step-by-Step Walkthrough — Running the Pipeline](#7-step-by-step-walkthrough--running-the-pipeline)
8. [Interpreting Your Results](#8-interpreting-your-results)
9. [Installation &amp; Dependencies](#9-installation--dependencies)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Troubleshooting Guide](#11-troubleshooting-guide)
12. [Glossary of Terms](#12-glossary-of-terms)
13. [Quick Reference — All Commands](#13-quick-reference--all-commands)
14. [Phase 1 Summary](#14-phase-1-summary)

---

## 1. What Is This Project?

### The Goal in One Sentence

Build a system that lets a robot **see** irregular, broken pottery fragments, **figure out** where they are and how they're oriented in 3D space, and **pick them up** without breaking them — even when the robot isn't sure about the exact shape.

### The Real-World Context

Archaeological excavations produce vast numbers of pottery fragments. These fragments need to be sorted, catalogued, and eventually reassembled by archaeologists. Each fragment is:

- **Thousands of years old** — irreplaceable
- **Delicate** — a single bad robotic grasp destroys history
- **Unique** — no two fragments are identical

Currently, this sorting is done by hand. It is slow, labour-intensive, and every human handling risks damage. If a robot could safely pick and sort these fragments, archaeologists could focus on interpretation and reconstruction.

### The Technical Goal

This is a **perception-to-control pipeline**. It takes raw sensor data (a 3D scan of a tabletop covered in fragments) and produces a robot motion that picks up one specific fragment. Along the way, it must:

1. **Perceive** — understand the 3D shape of each fragment
2. **Register** — figure out where that shape is and how it's oriented
3. **Quantify uncertainty** — admit when it doesn't know the exact shape
4. **Plan safely** — pick grasp points that work even if the shape is slightly different from what was measured
5. **Execute** — command the robot arm to move

### The Hardware

| Component | What It Is |
|---|---|
| **Wlkata Mirobot** | A 6-joint desktop robot arm, about the size of a human forearm. Six joints give it the full range of motion needed to approach objects from any direction. |
| **Intel RealSense D405** | A depth camera that sits above the workspace. It projects infrared light and measures how long it takes to bounce back, building a 3D picture of the scene accurate to less than a millimetre. |
| **Desktop Computer** | Runs Ubuntu Linux with ROS2 (Robot Operating System) for communication between components, PyTorch for neural networks, and MoveIt2 for motion planning. |

---

## 2. Background Concepts — The Basics

This section explains every concept from scratch. If you're comfortable with geometry and robotics, skip to Section 3.

### 2.1 What Is a Point Cloud?

A point cloud is simply a **list of 3D coordinates**. Each point is an $(x, y, z)$ position in space. Think of it as a dusting of dots that outlines the surfaces of objects in a scene.

```
        ·  ··  ·
      ·   ····   ·
     ·  ·······  ·
      ·   ····   ·
        ·  ··  ·
```

A depth camera produces point clouds by measuring the distance to every pixel it sees. A fragment scan might contain anywhere from 10,000 to several million points. These are stored in files with extensions like `.ply` or `.pcd`.

Each point can also carry extra data: a **normal vector** (which way the surface faces at that point), a **colour** (red, green, blue), or a **feature descriptor** (a numerical "signature" that describes the local shape).

### 2.2 What Is a Normal Vector?

A normal is an arrow perpendicular to the surface at a given point. It tells you which way the surface is "facing."

```
        ↑ ↑ ↑ ↑
    ————————————————   ← surface
```

Normals are crucial because:
- Descriptors like FPFH use angles between normals to describe shape
- Force-closure analysis uses normals to compute friction
- Rendering uses normals for shading

### 2.3 What Is a Pose? What Is 6-DoF?

A **pose** describes where something is and how it's oriented in 3D space. It has **6 degrees of freedom (6-DoF)**:

- **3 translation components:** Where is it? $(x, y, z)$ position
- **3 rotation components:** How is it tilted? (roll, pitch, yaw or equivalent)

For example, a coffee mug on a table has a pose: it sits at coordinates (0.3, 0.2, 0.05) metres from the robot base, and it's upright (no tilt).

When we say "6-DoF pose estimation," we mean: look at a 3D scan of a fragment and determine exactly where it is and how it's rotated.

### 2.4 What Is SE(3)?

$SE(3)$ stands for the **Special Euclidean group in 3 dimensions**. It is the set of all possible rigid-body transformations (rotations + translations) in 3D space.

**Why does this matter?** When you pick up a rigid object and place it somewhere else, you don't stretch, squeeze, or deform it. You just rotate and translate it. This is exactly what $SE(3)$ describes.

A transformation $T \in SE(3)$ is stored as a $4 \times 4$ matrix:

$$T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

- The top-left $3 \times 3$ block $R$ is a **rotation matrix** — it encodes how the object is tilted. $R$ must satisfy $R^\top R = I$ and $\det(R) = 1$.
- The top-right $3 \times 1$ column $\mathbf{t}$ is the **translation vector** — where the object's origin moves to.
- The bottom row $[0, 0, 0, 1]$ is a convention that makes the matrix algebra work.

To transform a point $\mathbf{p} = (p_x, p_y, p_z)$ by $T$, you append a 1 to make it a 4D vector and multiply:

$$\begin{bmatrix} \mathbf{p}' \\ 1 \end{bmatrix} = T \begin{bmatrix} \mathbf{p} \\ 1 \end{bmatrix} = \begin{bmatrix} R\mathbf{p} + \mathbf{t} \\ 1 \end{bmatrix}$$

So the point gets rotated by $R$, then shifted by $\mathbf{t}$.

### 2.5 What Is Registration?

**Registration** means: given two point clouds of the same object (or overlapping scene), find the $SE(3)$ transformation that aligns them.

```
   Cloud A                Cloud B            A aligned to B
   ·  ·                  ··   ·               ·· ··
     ···      + ?     ··   ··       =       ······
    ·····             ·· ··  ·              ·······
```

Think of it like this: you have two photographs of the same building from different angles. Registration finds the "camera movement" between them so you can overlay them perfectly.

In our pipeline, registration finds the pose of a fragment in a cluttered scene — like finding one specific puzzle piece on a table full of pieces.

### 2.6 What Is Force-Closure?

A robot gripper has two fingers that close on an object. **Force-closure** means the fingers can resist **any** external force or torque that tries to dislodge the object.

Think of holding a pencil between your thumb and forefinger:
- If you pinch near the middle with fingers directly opposite each other, you have force-closure — you can resist pushes in any direction.
- If you pinch near the tip with fingers barely touching, you don't have force-closure — the pencil will pivot and slip out.

Mathematically, force-closure is about the **friction cone** at each contact point. The friction cone is the set of all force directions that won't cause slipping. Two friction cones together form the **Grasp Wrench Space** — if this space contains the origin, the grasp is force-closed.

### 2.7 What Is Uncertainty?

Every measurement has error. The depth camera has noise (about 0.1 mm for the D405). The neural network's predictions aren't perfect. Some parts of the fragment might be hidden or poorly scanned.

**Epistemic uncertainty** is uncertainty about the model itself — "I don't know the true shape here because my training data didn't cover this kind of geometry." This is what MC Dropout quantifies.

**Aleatoric uncertainty** is uncertainty from the sensor — "the depth reading keeps jumping around by 0.1 mm no matter what." This is inherent and cannot be reduced by more training.

Our pipeline focuses on epistemic uncertainty because it directly affects grasp safety: if the model is unsure about a surface's shape, we should avoid grasping there.

### 2.8 What Is ROS2 and MoveIt2?

**ROS2** (Robot Operating System 2) is a framework for building robot software. It lets different programs (called "nodes") communicate by passing messages. For example:
- A camera node publishes point cloud messages
- A perception node subscribes to those messages, does processing, and publishes pose estimates
- A motion planning node subscribes to poses and publishes joint trajectories

**MoveIt2** is a motion planning library built on ROS2. Given a target pose for the robot's gripper, MoveIt2 computes a sequence of joint angles that moves the arm there without hitting anything. It handles collision checking, inverse kinematics (figuring out joint angles from a desired gripper position), and trajectory smoothing.

---

## 3. Why Is This Hard? The Three Challenges

### Challenge 1: Textureless Surfaces

Most computer vision systems use **colour patterns** to recognise objects. Think of how your phone's face unlock works: it looks for the unique pattern of your facial features.

Archaeological pottery fragments are **uniform beige/grey**. Subsurface scattering — light penetrating the plaster surface and bouncing around randomly before emerging — destroys any subtle colour variations. Standard feature detectors like SIFT and ORB, which rely on colour gradients, produce useless output.

**Our solution:** Use only geometric features — the 3D shape itself. FPFH descriptors use angles between surface normals, not colours. The GeoTransformer learns geometric patterns from point positions alone.

### Challenge 2: Irregular Shapes

Industrial robots typically handle known objects: the same gear, the same bottle, the same box. You can pre-program the exact grasp points because you know exactly what the object looks like.

Every archaeological fragment is **unique**. There is no CAD model to match against. The shape is irregular, with broken edges, varying thickness, and no symmetry to exploit.

**Our solution:** Registration against a reference scan of the same fragment. We scan the fragment once (the "source" or "model"), then register that scan to the cluttered scene. FPFH features and TEASER++ handle the matching even when the fragment is rotated and partially occluded.

### Challenge 3: Fragility Means Zero Tolerance for Failure

If a robot drops a metal gear in a factory, you pick it up and try again. If a robot drops a 2,000-year-old pottery fragment, it's gone forever.

A grasp that works for the measured shape might fail if the true shape is slightly different. The fragment might be thinner than measured (subsurface scattering shortens depth readings), or a hidden crack might change the surface normal.

**Our solution:** Risk-averse planning via uncertainty quantification. MC Dropout tells us where the model is uncertain. The CVaR filter ensures that grasps work even in the worst 5% of possible geometric variations. We reject grasps that look good on average but fail on rare, dangerous deviations.

---

## 4. The Pipeline — Big Picture

The pipeline is a linear chain of processing stages. Each stage's output feeds into the next.

```
                         INPUT
                     Depth Camera / PLY File
                           │
                           │  Raw point cloud: millions of points, uneven density,
                           │  no normals, noisy
                           │
              ┌────────────▼────────────┐
              │   MODULE 1: PERCEPTION  │
              │                        │
              │  1a. Voxel Downsampling │  Reduces points, uniform spacing
              │  1b. PCA Normals        │  Computes surface orientation per point
              │  1c. FPFH Features      │  33-D geometric descriptors
              │  1d. GeoTransformer     │  Learned features + MC Dropout
              └────────────┬────────────┘
                           │
                           │  Clean, feature-rich point cloud with descriptors
                           │
              ┌────────────▼──────────────┐
              │  MODULE 2: REGISTRATION   │
              │                           │
              │  TEASER++ Global          │  Finds SE(3) transform aligning
              │  Registration (TLS cost)  │  fragment to scene
              │                           │
              │  Output: 4×4 pose matrix  │
              │  + SDP optimality cert    │
              └────────────┬──────────────┘
                           │
                           │  (x, y, z, roll, pitch, yaw) of the fragment
                           │
              ┌────────────▼──────────────┐
              │  MODULE 3: UNCERTAINTY &  │
              │           GRASPING        │
              │                           │
              │  3a. MC Dropout variance  │  How certain are we about each point?
              │  3b. Pose covariance Σ    │  6×6 uncertainty in the pose
              │  3c. Force-Closure LP     │  Is this grasp physically stable?
              │  3d. CVaR filter (α=0.05) │  Does it work under shape variations?
              └────────────┬──────────────┘
                           │
                           │  Ranked list of safe grasp candidate pairs
                           │
              ┌────────────▼──────────────┐
              │   MODULE 4: EXECUTION     │
              │                           │
              │  Hand-Eye Calibration     │  Camera frame → Robot frame
              │  MoveIt2 Motion Planning  │  Compute collision-free trajectory
              │  Trajectory Execution     │  Send commands to robot arm
              └───────────────────────────┘
```

### The Flow of Data

1. **PLY/PCD file** → point cloud $(N, 3)$
2. **After voxel downsampling** → reduced point cloud $(M, 3)$ with normals $(M, 3)$
3. **After FPFH** → each point has a 33-D descriptor $(M, 33)$
4. **After feature matching** → correspondence pairs $(K, 2)$ linking source and target
5. **After TEASER++** → $4 \times 4$ pose matrix $T$
6. **After MC Dropout** → per-point variance $(M,)$ + $6 \times 6$ pose covariance
7. **After CVaR** → accepted grasp candidates JSON
8. **After MoveIt2** → robot joint trajectory

### What You Need to Run Each Stage

| Stage | Input | Output | Time |
|---|---|---|---|
| Voxel + Normals | Raw PLY/PCD | Clean PLY with normals | ~30 seconds |
| FPFH | Clean PLY | PCD with 33-D descriptors | ~1 minute |
| Registration | Source + Target PLY | 4×4 pose matrix | ~2 minutes |
| Parameter Sweep | Source + Target PLY | CSV results | 10–30 minutes |
| MC Variance | Fragment PCD | Variance cloud PCD | ~10 minutes |
| Pose Covariance | Source + Target PLY | 6×6 Σ + cloud PCD | ~10 minutes |
| Force-Closure | Fragment STL | FC boolean + quality ε | ~30 seconds |
| CVaR Validator | Variance PCD + JSON | Accepted grasps JSON | ~5 minutes |
| ROS2 Build | Source code | Compiled package | ~2 minutes |
| ROS2 Execute | Grasp JSON | Robot trajectory | ~5 seconds |

---

## 5. Mathematical Foundations — Explained Step by Step

Every formula is accompanied by an intuitive explanation. You should be able to understand **why** we do each operation, not just **what** the formula says.

### 5.1 Voxel Downsampling — Cleaning Up the Point Cloud

**The problem:** A raw depth camera scan can produce hundreds of thousands of points. Many are redundant (multiple points on the same flat surface). Some are spurious (reflections, sensor noise). Processing all of them is slow and the uneven density messes up neighbourhood-based calculations.

**The idea:** Divide 3D space into a regular grid of tiny cubes (voxels). Each voxel is typically $5 \text{ mm} \times 5 \text{ mm} \times 5 \text{ mm}$. Replace all points inside each voxel with their average position:

$$\mathbf{p}_{\mathrm{voxel}} = \frac{1}{N_v}\sum_{i=1}^{N_v} \mathbf{p}_i$$

**What this means in plain English:** "For every 5mm cube, take all the scan points that fell inside it, compute their centre of mass, and keep only that one point."

**Why this works:**
- **Uniform density:** After downsampling, every part of the surface gets roughly one point per voxel face. The PCA normal estimation in Step 1b uses a fixed $k=30$ neighbours — if some regions have 100× more points than others, the effective search radius changes and normals become unreliable.
- **Noise reduction:** Averaging within each voxel suppresses random sensor noise (the errors tend to cancel out).
- **Speed:** A cloud of 500,000 points might reduce to 20,000 points — much faster for all downstream computation.

**Analogy:** Think of a point cloud as a photograph with too many pixels. Downsampling is like reducing the resolution — you lose some detail, but the overall shape remains clear and the file is much smaller.

### 5.2 PCA Normal Estimation — Finding Surface Orientation

**What is a surface normal?** At any point on a surface, the normal is the direction perpendicular to the surface. On a flat wall, all normals point straight out. On a curved bowl, normals fan out radially.

**The problem:** The scanner gives us $(x, y, z)$ positions but not which way the surface faces. We need normals because:
- FPFH features compare angles between normals
- Force-closure needs normals to build friction cones
- Visualisation uses normals for shading

**The idea:** Look at a small neighbourhood around each point. If the surface is locally flat, the points lie on a plane. The direction perpendicular to that plane is the normal.

**Step by step:**

1. Pick a point $\mathbf{p}_i$. Find its $k = 30$ nearest neighbours in 3D space (using a KD-Tree for fast search).

2. Compute the mean of these neighbours:
   $$\boldsymbol{\mu} = \frac{1}{k}\sum_{j=1}^k \mathbf{p}_j$$

3. Build the $3 \times 3$ covariance matrix that captures how the points are spread:
   $$C = \frac{1}{k}\sum_{j=1}^k (\mathbf{p}_j - \boldsymbol{\mu})(\mathbf{p}_j - \boldsymbol{\mu})^\top$$

   Each term $(\mathbf{p}_j - \boldsymbol{\mu})(\mathbf{p}_j - \boldsymbol{\mu})^\top$ is a $3 \times 3$ outer product. $C$ describes the 3D spread: its eigenvectors point along the principal axes of the point cluster, and its eigenvalues tell you the variance along each axis.

4. Eigendecompose $C$:
   $$C = V \Lambda V^\top = \begin{bmatrix} | & | & | \\ \mathbf{v}_1 & \mathbf{v}_2 & \mathbf{v}_3 \\ | & | & | \end{bmatrix} \begin{bmatrix} \lambda_1 & 0 & 0 \\ 0 & \lambda_2 & 0 \\ 0 & 0 & \lambda_3 \end{bmatrix} \begin{bmatrix} - & \mathbf{v}_1^\top & - \\ - & \mathbf{v}_2^\top & - \\ - & \mathbf{v}_3^\top & - \end{bmatrix}$$

   Sort eigenvalues so $\lambda_1 \geq \lambda_2 \geq \lambda_3$.

5. $\mathbf{v}_3$ (the eigenvector of the smallest eigenvalue) is the surface normal. $\lambda_3$ measures variance **along** the normal — for a flat surface, this should be near zero.

**Intuition:** Imagine a scatter of points on a tabletop. The points spread widely along the table (high variance, $\lambda_1$ and $\lambda_2$) and barely at all vertically (low variance, $\lambda_3$). The vertical direction is the surface normal.

**Quality check:** The ratio $\lambda_3 / (\lambda_1 + \lambda_2 + \lambda_3)$ tells you how "planar" the neighbourhood is:
- Near 0 → very flat, normal is reliable
- Near 1/3 → isotropic scatter, normal is ambiguous (e.g., on a sharp edge)

**Analogy:** Hold a pencil against a table. The pencil's direction is the normal. The table's surface is the plane spanned by the other two directions.

### 5.3 FPFH Descriptors — Geometric Signatures

**What is a feature descriptor?** A descriptor is a fixed-length vector of numbers that describes the local shape around a point. If two points have similar descriptors, they're likely on similar geometric features (e.g., both on a curved rim, or both on a flat surface).

**Why FPFH?** FPFH (Fast Point Feature Histograms) are designed to work on **textureless** surfaces. They use only geometric information — the relative positions and orientations of nearby points. They are computed from **angular triplets** that are invariant to rotation and translation of the whole object.

**The Darboux Frame:**

For a pair of points $\mathbf{p}_i$ (the "query") and $\mathbf{p}_j$ (a "neighbour"), with normals $\mathbf{n}_i$ and $\mathbf{n}_j$:

1. Define three orthogonal axes anchored at $\mathbf{p}_i$:
   $$\mathbf{u} = \mathbf{n}_i \quad \text{(the normal at the query point)}$$
   $$\mathbf{v} = (\mathbf{p}_j - \mathbf{p}_i) \times \mathbf{u} \quad \text{(perpendicular to both the line and normal)}$$
   $$\mathbf{w} = \mathbf{u} \times \mathbf{v} \quad \text{(completes the right-handed frame)}$$

2. Compute three angles that capture the relative geometry:
   $$\alpha = \mathbf{v} \cdot \mathbf{n}_j \quad \text{(how much the neighbour's normal tilts in the v-direction)}$$
   $$\phi = \frac{\mathbf{u} \cdot (\mathbf{p}_j - \mathbf{p}_i)}{\|\mathbf{p}_j - \mathbf{p}_i\|} \quad \text{(how much the line tilts relative to the normal)}$$
   $$\theta = \arctan(\mathbf{w} \cdot \mathbf{n}_j,\; \mathbf{u} \cdot \mathbf{n}_j) \quad \text{(the twist angle of the neighbour's normal)}$$

**Building the histogram:**

- Divide each angle's range into $11$ bins. This gives a $3 \times 11 = 33$ bin histogram called the **Simplified Point Feature Histogram (SPFH)**.
- The **FPFH** is a smoothed version that averages SPFH over neighbours:
  $$\text{FPFH}(\mathbf{p}_i) = \text{SPFH}(\mathbf{p}_i) + \frac{1}{k}\sum_{j=1}^k \frac{1}{\omega_j} \text{SPFH}(\mathbf{p}_j)$$

  where $\omega_j$ is the distance between $\mathbf{p}_i$ and $\mathbf{p}_j$. Far-away neighbours contribute less.

**Why it works for textureless fragments:** The angles $\alpha, \phi, \theta$ capture the relative shape — whether the local surface is flat, convex, concave, edge-like, etc. This is purely geometric, so it doesn't matter that the plaster has no colour variation.

**Analogy:** If you close your eyes and run your finger over a surface, you can tell if it's flat, curved, or has an edge. FPFH is the mathematical version of that tactile sense — it "feels" the local shape and converts it to numbers.

### 5.4 TEASER++ Registration — Why We Reject Standard ICP

**What is registration?** Given a point cloud of a fragment (the "source") and a point cloud of a cluttered scene (the "target"), find the $SE(3)$ transformation that moves the source to align with where the fragment sits in the scene.

**The standard approach (ICP — Iterative Closest Point):**

ICP works by alternating two steps:
1. For each source point, find the closest target point (the "correspondence")
2. Minimise the sum of squared distances between corresponding pairs

The cost function is:
$$\min_{T \in SE(3)} \sum_{i=1}^N \| \mathbf{q}_i - T(\mathbf{p}_i) \|^2$$

This is called the **L2 cost** (squared error). It works beautifully when all correspondences are correct. But it fails catastrophically when there are **outliers** — wrong correspondences.

**The problem with L2 and outliers:**

Imagine 100 correct correspondences with errors of ~1 mm each. The L2 cost from these is $100 \times 1^2 = 100$.

Now a single outlier — a wrong match where the points are 100 mm apart. Its L2 cost is $1 \times 100^2 = 10,000$. This one bad match contributes **100 times more** to the total cost than all 100 correct matches combined.

The optimiser will twist and warp the entire alignment just to reduce this one enormous error. This is like letting one person shouting the wrong answer override 100 people quietly giving the right answer.

**The TLS solution (TEASER++):**

Truncated Least Squares changes the cost function:
$$\min_{T \in SE(3)} \sum_{i=1}^N \min(e_i^2,\; c^2)$$

where $e_i = \|\mathbf{q}_i - T(\mathbf{p}_i)\|$ and $c$ is the **truncation threshold** (recommended 10 mm).

For any correspondence with error larger than $c$, the cost is **clamped** to a constant $c^2$. The gradient of $\min(e_i^2, c^2)$ is **zero** for $e_i > c$, so that correspondence contributes **no force** to the optimisation. It is simply ignored.

**Why $c = 10$ mm?** The RealSense D405 has sub-millimetre accuracy, and our voxel size is 5 mm. A correct correspondence should be within a few millimetres. Anything beyond 10 mm is almost certainly a wrong match. Rather than letting those wrong matches dominate, we cap their influence.

**TEASER++ operates in four stages:**

#### Stage 1: Maximum Clique Inlier Selection

**What is a clique?** In graph theory, a clique is a set of nodes where every pair is connected by an edge.

**Building the consistency graph:**
- Each node is a putative correspondence $(i)$ between source point $\mathbf{p}_i$ and target point $\mathbf{q}_i$
- Two correspondences $(i)$ and $(j)$ are connected by an edge if they agree on the distance between the points:
  $$d_{ij} = \big| \|\mathbf{q}_i - \mathbf{q}_j\| - \|\mathbf{p}_i - \mathbf{p}_j\| \big| \leq 2\varepsilon$$

  This uses the fundamental property of rigid-body transformations: distances between points don't change when you rotate and translate them. If $d_{ij} \leq 2\varepsilon$, then correspondences $i$ and $j$ are mutually consistent.

- Extract the **maximum clique** — the largest set of pairwise-consistent correspondences. These are the high-confidence inliers.

**Intuition:** "If you tell me point A corresponds to point X, and point B corresponds to point Y, then the distance between A and B should (roughly) equal the distance between X and Y. If it doesn't, at least one of your claims is wrong."

#### Stage 2: GNC-TLS Rotation Estimation

**GNC (Graduated Non-Convexity):** The TLS cost $\min(e^2, c^2)$ is non-convex (it has a flat region), which makes optimisation hard. GNC starts by solving a convex approximation (easy), then gradually deforms it toward the true TLS cost. At each step, the previous solution "warms up" the next.

Think of it like slowly lowering the temperature — you start with a smooth, easy landscape and gradually introduce the sharp features until you're optimising the actual cost function.

#### Stage 3: Adaptive Voting Translation

Once the rotation $R$ is known, each correspondence votes for the translation:
$$\mathbf{t}_i = \mathbf{q}_i - R\mathbf{p}_i$$

All correct correspondences should vote for roughly the same translation. The maximum-consensus location is the translation estimate.

**Analogy:** After a concert, everyone in the audience points toward the exit. Most point in roughly the same direction (the actual exit). A few confused people point elsewhere. The translation estimate is where most people agree the exit is.

#### Stage 4: SDP Certification

**SDP (Semidefinite Programming)** provides a **provable upper bound** on how far the found solution is from the true global optimum. This is reported as a "certificate" number.

If the certificate is 0.001, that means $\text{cost}(T_{\text{found}}) - \text{cost}(T_{\text{optimal}}) \leq 0.001$. The solution is **provably within 0.001 of the global TLS optimum**.

This is the mathematical equivalent of a guarantee. Most registration algorithms (ICP, RANSAC) give you an answer with no information about how good it is. TEASER++ gives you a certificate.

**Analogy:** An ICP solution is like "I think the treasure is buried here." A TEASER++ solution with a certificate of 0.001 is like "I can prove mathematically that the treasure is within 1 mm of this spot."

### 5.5 Force-Closure — Can the Gripper Hold the Fragment?

**The concept:** A grasp is force-closed if the robot fingers can resist **any** external force or torque applied to the object. This is the minimum requirement for a successful grasp.

**Friction cones:**

At each contact point, a finger can push in many directions. The set of all force directions that don't cause slipping forms a **friction cone**:

```
           ↑  normal (push straight in)
          /|\
         / | \     ← friction cone, half-angle α
        /  |  \
       /   |   \
    ——|———+———|——  surface
```

The half-angle $\alpha$ is determined by the coefficient of friction $\mu$:
$$\alpha = \arctan(\mu)$$

For plaster on rubber/felt gripper pads, $\mu \approx 0.5$, so $\alpha \approx 26.6°$. A higher friction coefficient means a wider cone — more forgiving grasps.

**Discretising the cone:**

A continuous cone is hard to work with mathematically, so we approximate it with $m = 8$ force vectors (generators):

$$\mathbf{f}_k = \cos(\alpha)\hat{\mathbf{n}} + \sin(\alpha)\bigl(\cos\theta_k \,\mathbf{u} + \sin\theta_k \,\mathbf{v}\bigr)$$
$$\theta_k = \frac{2\pi k}{m}, \quad k = 1, 2, \dots, m$$

where $\hat{\mathbf{n}}$ is the inward surface normal and $\mathbf{u}, \mathbf{v}$ span the tangent plane.

```
          f₀
         /|\
      f₇ | f₁
        \|/
    f₆ — + — f₂    ← 8 generators around the normal
        /|\
      f₅ | f₃
         \|/
          f₄
```

**From forces to wrenches:**

A **wrench** is a 6-dimensional vector combining force (3D) and torque (3D):

$$\mathbf{w}_k = \begin{bmatrix} \mathbf{f}_k \\ \mathbf{c} \times \mathbf{f}_k \end{bmatrix} \in \mathbb{R}^6$$

- Top 3 components: the force direction
- Bottom 3 components: the torque = contact position $\mathbf{c}$ cross the force (torque = force × lever arm)

**The Grasp Wrench Space (GWS):**

The GWS is the convex hull of all wrench vectors from all contacts. For two contacts with 8 generators each, we have a $6 \times 16$ wrench matrix:
$$W = \bigl[ \mathbf{w}_{1,1} \mid \mathbf{w}_{1,2} \mid \dots \mid \mathbf{w}_{1,8} \mid \mathbf{w}_{2,1} \mid \dots \mid \mathbf{w}_{2,8} \bigr]$$

**The Force-Closure test:**

The grasp is force-closed if the **origin is strictly inside** the convex hull of the wrench columns. This means: there exists a combination of contact forces (all inside the friction cones) that can resist any external wrench.

**The Linear Program:**

$$\begin{aligned} \max \; & \varepsilon \\ \text{subject to} \quad & W\boldsymbol{\alpha} = \mathbf{0} \\ & \mathbf{1}^\top\boldsymbol{\alpha} = 1 \\ & \alpha_j \geq \varepsilon \end{aligned}$$

**What each line means:**

1. $W\boldsymbol{\alpha} = \mathbf{0}$ — the weighted sum of wrenches equals zero (equilibrium — the object doesn't move)
2. $\mathbf{1}^\top\boldsymbol{\alpha} = 1$ — the $\alpha$ weights sum to 1 (normalisation)
3. $\alpha_j \geq \varepsilon$ — every generator gets at least $\varepsilon$ weight (positive contribution from all directions)

If the optimal $\varepsilon > 0$, the origin is in the **interior** of the GWS → force-closure. The value of $\varepsilon$ is the **grasp quality** — how much margin there is before losing force-closure.

| $\varepsilon$ | Interpretation |
|---|---|
| > 0.05 | Very robust — large safety margin |
| 0.01–0.05 | Good — adequate for careful handling |
| 0.001–0.01 | Marginal — close to losing grip |
| ≤ 0 | No force-closure — object will slip |

**The antipodal condition (intuitive check):**

For a two-finger grasp to have any chance of force-closure, the contact points should be roughly opposite each other (antipodal). Specifically, the line connecting the contacts should fall inside both friction cones:

$$\hat{\mathbf{d}} \cdot \mathbf{n}_1 \geq \cos\alpha \quad \text{AND} \quad -\hat{\mathbf{d}} \cdot \mathbf{n}_2 \geq \cos\alpha$$

where $\hat{\mathbf{d}}$ is the unit vector from contact 1 to contact 2.

**Analogy:** Hold a pen between your thumb and forefinger. They should be on opposite sides. If both fingers are on the same side, the pen will roll away. The angle between your fingers' normals should be close to 180°.

### 5.6 Monte Carlo Dropout — Quantifying Model Uncertainty

**What is dropout?** During training, dropout randomly deactivates a fraction of neurons in each layer. This prevents overfitting — the network can't rely on any single neuron because it might be "asleep" during any given forward pass.

**What is MC Dropout?** Normally, dropout is turned **off** during inference (test time). MC Dropout keeps it **on**. Running the same input through the network $T$ times produces $T$ slightly different outputs because different neurons are deactivated each time.

```
Forward pass 1:  dropout mask 011010... → output y₁
Forward pass 2:  dropout mask 101100... → output y₂
Forward pass 3:  dropout mask 110001... → output y₃
...
Forward pass T:  dropout mask 010111... → output y_T
```

**Why this measures uncertainty:** If the model is confident about a particular point, the $T$ predictions will be tightly clustered (low variance). If the model is uncertain (maybe the geometry is unusual or poorly represented in the training data), the predictions will spread out (high variance).

**The variance per point:**

After $T$ forward passes producing outputs $\{\hat{\mathbf{y}}_{t,i}\}_{t=1}^T$ for point $i$:

$$\sigma^2_i = \frac{1}{T-1}\sum_{t=1}^T \|\hat{\mathbf{y}}_{t,i} - \boldsymbol{\mu}_i\|^2, \quad \boldsymbol{\mu}_i = \frac{1}{T}\sum_{t=1}^T \hat{\mathbf{y}}_{t,i}$$

This is the standard sample variance (Bessel-corrected with $T-1$).

**Why Welford's online algorithm?**

Storing all $T$ outputs for $N$ points requires $\mathcal{O}(T \times N)$ memory. With $T = 50$ and $N = 20,000$, that's 1 million vectors. Welford's algorithm computes mean and variance **incrementally**, requiring only $\mathcal{O}(N)$ memory regardless of $T$:

```
For each forward pass t:
  For each point i:
    count += 1
    delta = y_{t,i} - mean_i
    mean_i += delta / count
    delta2 = y_{t,i} - mean_i    (using updated mean)
    M2_i += delta * delta2

After all passes:
  variance_i = M2_i / (T - 1)
```

**Analogy:** Imagine you ask $T = 50$ experts to estimate the shape of a fragment. For well-scanned flat regions, all 50 give almost the same answer → low variance, high confidence. For hidden or thin regions, the experts disagree → high variance, low confidence. The variance heatmap shows you where the experts argue.

### 5.7 Pose Covariance in SE(3) Lie Algebra

**The problem:** Each MC Dropout pass produces a $4 \times 4$ pose matrix $T_t \in SE(3)$. We collect $T$ such matrices. How do we compute the variance of these poses?

We can't just compute the variance entry-by-entry on the $4 \times 4$ matrices — that would violate the constraints of $SE(3)$ (the rotation part must stay orthonormal with determinant 1). We need to work in a space where poses can be properly added and scaled like vectors.

**The Lie algebra $\mathfrak{se}(3)$:**

$SE(3)$ is a **Lie group** — a smooth manifold where each point is a valid rigid-body transformation. The tangent space at the identity is the **Lie algebra** $\mathfrak{se}(3)$, which IS a vector space (isomorphic to $\mathbb{R}^6$).

A point in $\mathfrak{se}(3)$ is called a **twist** or **screw**:
$$\boldsymbol{\xi} = \begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^6$$

- $\mathbf{v} \in \mathbb{R}^3$ — linear velocity (relates to translation)
- $\boldsymbol{\omega} \in \mathbb{R}^3$ — angular velocity (relates to rotation)

**The log map:** $SE(3) \to \mathfrak{se}(3)$

Given a pose $T$, the log map extracts the twist:
$$\boldsymbol{\xi} = \log(T)$$

This is like "unwinding" the transformation. Specifically:
- $\boldsymbol{\omega}$ comes from the axis-angle representation of the rotation $R$: $R = \exp([\boldsymbol{\omega}]_\times)$
- $\mathbf{v}$ comes from the translation, corrected by the left Jacobian of $SO(3)$

**The exp map:** $\mathfrak{se}(3) \to SE(3)$

The reverse operation reconstructs the pose from the twist:
$$T = \exp(\boldsymbol{\xi})$$

**Computing the covariance:**

1. Map each pose to the Lie algebra: $\boldsymbol{\xi}_t = \log(T_t)$
2. Compute the mean in the Lie algebra (which is a vector space — this is allowed!): $\bar{\boldsymbol{\xi}} = \frac{1}{T}\sum_t \boldsymbol{\xi}_t$
3. Compute the $6 \times 6$ covariance: $\Sigma = \frac{1}{T-1}\sum_t (\boldsymbol{\xi}_t - \bar{\boldsymbol{\xi}})(\boldsymbol{\xi}_t - \bar{\boldsymbol{\xi}})^\top$

**Interpreting the 6×6 covariance:**

$$\Sigma = \begin{bmatrix} \Sigma_{tt} & \Sigma_{tr} \\ \Sigma_{tr}^\top & \Sigma_{rr} \end{bmatrix}$$

- **$\Sigma_{tt}$ (top-left 3×3):** Translation uncertainty in $\text{m}^2$. Large values mean the model disagrees on where the fragment is.
- **$\Sigma_{rr}$ (bottom-right 3×3):** Rotation uncertainty in $\text{rad}^2$. Large values mean the model disagrees on how the fragment is oriented.
- **$\Sigma_{tr}$ (top-right 3×3):** Cross-covariance. Non-zero values mean translation and rotation errors are correlated (e.g., a rotation error at the far end of the fragment creates a large translation error there).

**Projecting to per-point spatial uncertainty:**

The pose uncertainty affects different points differently. A rotation error of 1° has almost no effect on a point near the rotation centre, but could shift a point 10 cm away by nearly 2 mm.

For each point $\mathbf{p}_k$, the spatial variance is:
$$J_k = [I_3 \mid -[\mathbf{p}_k]_\times] \in \mathbb{R}^{3 \times 6}$$
$$\Sigma_{\mathbf{p}_k} = J_k \Sigma J_k^\top \in \mathbb{R}^{3 \times 3}$$
$$\sigma^2_k = \text{trace}(\Sigma_{\mathbf{p}_k})$$

$J_k$ is the **Jacobian** of the SE(3) action at point $\mathbf{p}_k$ — it linearises how a small change in pose affects the position of that point. The $[\mathbf{p}_k]_\times$ term captures the lever-arm effect: farther points move more when the pose rotates.

**Analogy:** If a door hinge (rotation) has 1° of play, the handle (far from the hinge) wobbles visibly, but the hinge itself barely moves. The Jacobian accounts for this distance effect.

### 5.8 CVaR — Making Safe Decisions Under Uncertainty

**The problem:** MC Dropout gives us 100 possible geometric realisations of the fragment. We want to pick a grasp that works across **all of them**, not just on average.

**Why the average is dangerous:**

Consider a grasp candidate that scores:
- $\varepsilon = 0.1$ (good) on 95 out of 100 realisations
- $\varepsilon = 0$ (no force-closure) on 5 out of 100 realisations

The **expected value** (average) is:
$$\mathbb{E}[\varepsilon] = \frac{95 \times 0.1 + 5 \times 0}{100} = 0.095$$

This looks great! A planner using expected value would accept this grasp.

But in 5% of cases, the grasp **fails completely** — the fragment slips and possibly breaks. Would you trust a bridge that collapses 5% of the time?

**Conditional Value-at-Risk (CVaR):**

CVaR looks only at the **worst outcomes**. Sort all quality scores from worst to best:
$$\varepsilon_{(1)} \leq \varepsilon_{(2)} \leq \dots \leq \varepsilon_{(N)}$$

Take the mean of the worst $\alpha N$ scores (where $\alpha = 0.05$, meaning the worst 5%):
$$\mathrm{CVaR}_{0.05} = \frac{1}{K}\sum_{k=1}^K \varepsilon_{(k)}, \quad K = \lceil 0.05 \cdot N \rceil$$

For our example with $N = 100$ and $\alpha = 0.05$:
$$\mathrm{CVaR}_{0.05} = \frac{1}{5}\sum_{k=1}^5 \varepsilon_{(k)} = \frac{5 \times 0}{5} = 0$$

**CVaR correctly rejects this grasp.** Expected value would have accepted it.

**When does CVaR accept a grasp?** Only when ALL of the worst 5% of realisations still have $\varepsilon > 0$. In other words: "Is there a single likely geometric variation under which this grasp fails?" If yes → rejected.

**Analogy:** You're choosing a route to drive. The highway has a 95% chance of taking 30 minutes and a 5% chance of being closed (2-hour detour). The back road reliably takes 35 minutes. Expected value says highway (0.95 × 30 + 0.05 × 120 = 30 + 6 = 36 min). CVaR (worst-5% focus) says: the back road's worst 5% is 35, the highway's worst 5% includes the 120-minute closure → take the back road.

### 5.9 Hand-Eye Calibration — Translating Between Camera and Robot

**The problem:** The camera sees the fragment and says "it's at position $\mathbf{p}_{\text{cam}}$." But the robot needs coordinates $\mathbf{p}_{\text{robot}}$ in its own frame. There is an unknown fixed transform $X$ between the camera frame and the robot base frame:

$$\mathbf{p}_{\text{robot}} = X \cdot \mathbf{p}_{\text{cam}}$$

**The calibration procedure:**

We move the robot arm to several known positions while holding a calibration pattern (like a checkerboard) visible to the camera. At each position $i$, we record:

- $A_i$: the robot's end-effector motion from position $i$ to position $i+1$ (known from joint encoders)
- $B_i$: the calibration pattern's apparent motion as seen by the camera (computed from camera images)

These satisfy:
$$A_i X = X B_i \quad \Rightarrow \quad AX = XB$$

This is the classic **hand-eye calibration** equation ("hand" = robot, "eye" = camera).

**Solving AX=XB (Tsai-Lenz method):**

The equation $AX = XB$ couples rotation and translation. Tsai-Lenz decomposes it:

**Step 1 — Solve for rotation $R_X$:**

From the axis-angle representation $\mathbf{r}_a = \log(R_A)$, $\mathbf{r}_b = \log(R_B)$:
$$\text{skew}(\mathbf{r}_a + \mathbf{r}_b) \cdot \mathbf{P}'_x = \mathbf{r}_b - \mathbf{r}_a$$

where $\mathbf{P}'_x = 2\sin(\theta/2)\hat{\mathbf{n}}$ is the modified Rodrigues vector for $R_X$. This is a linear system solved by least squares across all motion pairs.

**Step 2 — Solve for translation $\mathbf{t}_X$:**

With $R_X$ known:
$$(R_A - I) \cdot \mathbf{t}_X = R_X\mathbf{t}_B - \mathbf{t}_A$$

Again a linear system solved by least squares.

**Analogy:** If you're looking through a window (the camera) at a room (the robot workspace), hand-eye calibration answers: "If I see something at that spot through the window, where is it actually in the room?" You figure this out by moving a known object around and observing how its apparent position changes.

---

## 6. How Each File Works — Deep Dive

This section explains every file in the project: what it does, how it does it, and why it's structured the way it is.

### 6.1 Top-Level Files

---

#### `voxel_downsample_normals.py` — Point Cloud Preprocessing

**Purpose:** The entry point of Module 1. Takes a raw point cloud file and produces a clean, uniformly-sampled cloud with PCA surface normals. Every downstream stage depends on this preprocessing.

**Functions (in execution order):**

1. **`load_point_cloud(file_path)`** → `o3d.geometry.PointCloud`
   - Reads any format Open3D supports: PLY, PCD, XYZ, OBJ, STL
   - Uses `o3d.io.read_point_cloud()` with auto-format detection
   - Returns Open3D PointCloud object (an optimised C++ structure holding positions, optionally normals and colours)

2. **`downsample_voxel_grid(pcd, voxel_size)`** → `o3d.geometry.PointCloud`
   - Calls `pcd.voxel_down_sample(voxel_size)` — a highly optimised Open3D C++ function
   - Internally: builds a hash map from voxel indices to point lists, then averages positions within each occupied voxel
   - Voxel size 0.005 means 5 mm cubes — chosen to match the D405's resolution while keeping point count manageable
   - Output typically has 10,000–30,000 points (down from 100,000–500,000)

3. **`estimate_normals_pca_knn(pcd, k_neighbors=30)`** → `o3d.geometry.PointCloud`
   - Builds a KD-Tree for fast neighbour queries ($\mathcal{O}(k \log N)$ per point vs $\mathcal{O}(N^2)$ brute force)
   - For each point, queries $k = 30$ nearest neighbours
   - Computes centroid $\boldsymbol{\mu}$
   - Builds $3 \times 3$ covariance matrix $C$ (using `np.cov` on the neighbour positions)
   - Eigendecomposes $C$ via `np.linalg.eigh` (for symmetric matrices — faster than `eig`)
   - Takes eigenvector of smallest eigenvalue as normal
   - Stores normal in `pcd.normals` (Open3D attribute)

4. **`save_point_cloud(pcd, output_path)`** — writes PLY file
5. **`visualize(pcd)`** — opens interactive Open3D window (rotate/zoom/pan)
6. **`main()`** — CLI entry point using argparse:
   - `--voxel-size` (default 0.005)
   - `--knn` (default 30)
   - `--output` (auto-generates filename if not specified)
   - `--viz` flag for visualisation

**Why this file is standalone (not in a package):** It's the simplest entry point, usable as a one-liner. No imports from the rest of the project. Good for quick testing of new fragment scans.

---

#### `AGENTS.md` — AI Assistant Configuration

**Purpose:** Defines the project rules and context for AI coding assistants. Not code — it's a specification document that tells the AI what this project is, what stack it uses, what conventions to follow, and how to verify correctness.

**Contents:**
- Project role (Level 4 Durham dissertation)
- Core objective (hybrid perception-to-control pipeline)
- Hardware/software stack specification
- Technical architecture with four modules
- Workflow rules (explain math before code, ensure colcon build compatibility)

---

### 6.2 `registration/` — SE(3) Registration Sub-Package

---

#### `registration/__init__.py` — Public API

Exports 12 symbols that form the public interface of the registration package. Users (and other modules) should import from `registration`, not from individual files:
- `register_teaser` — main registration function
- `register_scene_to_cad` — convenience wrapper for scene→CAD
- `SE3Result` — dataclass for registration output
- `TeaserParams` — dataclass for solver parameters
- `validate_se3` — check if a matrix is a valid SE(3) element
- `weighted_svd_se3` — Kabsch algorithm
- `compute_fpfh`, `match_features` — from fpfh_features
- `transform_points`, `extract_rt`, `compose`, `inverse_transform` — from se3_utils

---

#### `registration/fpfh_features.py` — FPFH Descriptors & Matching

**Purpose:** The geometric feature backbone of registration. Computes 33-dimensional FPFH descriptors and matches them between two point clouds.

**Functions:**

1. **`compute_fpfh(pcd, normal_radius=0.01, normal_k=30, fpfh_radius=0.025)`** → Feature(N, 33)
   - First estimates normals (if not already present) using Open3D's hybrid KD-tree normal estimation with `normal_radius` search radius
   - If normals exist, skips re-estimation
   - Calls `o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)` — Open3D's optimised FPFH implementation
   - Search parameter `fpfh_radius = 0.025` (25 mm) — each point looks at neighbours within 2.5 cm
   - Returns `(N, 33)` numpy array of float64 descriptors

2. **`match_features(fpfh_src, fpfh_tgt, mutual_filter=True, ratio_threshold=0.9, max_correspondences=5000)`** → correspondence set
   - For each source descriptor, finds the 2 nearest neighbours in the target descriptor space (33-D Euclidean distance)
   - **Lowe's ratio test:** Keeps match only if `distance_to_1st / distance_to_2nd < ratio_threshold`. This discards ambiguous matches where the best and second-best are similarly close.
   - **Mutual filter (optional):** Also matches target→source. Keeps only pairs where each is the other's best match. This eliminates one-way matches.
   - Limits to `max_correspondences` (default 5000) to keep downstream computation manageable

3. **`extract_correspondence_clouds(pcd_src, pcd_tgt, correspondences)`** → `(np.array(M,3), np.array(M,3))`
   - Converts correspondence indices to actual 3D point coordinates
   - Returns matched source points and matched target points as numpy arrays

4. **`_build_correspondence_set(source_features, target_features, ratio_threshold, mutual_filter)`**
   - Internal: brute-force KNN using `np.linalg.norm` with broadcasting
   - Computes all-pairs distances: `d[i,j] = ||src[i] - tgt[j]||` — efficient with vectorised numpy operations
   - For each source point, finds smallest and second-smallest distances using `np.argpartition` (partial sort, faster than full sort)

**Design decisions:**
- Uses Open3D's FPFH computation (C++ backend, much faster than pure numpy/python)
- Uses numpy for matching (more flexible than Open3D's built-in matcher, supports ratio test and mutual filtering)
- Descriptors are float64 for precision in nearest-neighbour search

---

#### `registration/teaser_registration.py` — TEASER++ Registration

**Purpose:** The core of Module 2. Implements the full TEASER++ pipeline: FPFH → match → TEASER++ solve → SE(3) validation.

**Key Classes:**

- **`SE3Result`** (dataclass):
  - `T`: $4 \times 4$ SE(3) transformation matrix (numpy array)
  - `certificate`: float, the SDP suboptimality bound (None if RANSAC fallback)
  - `runtime_sec`: float, wall-clock time
  - `num_correspondences`: int, how many correspondences were used
  - `converged`: bool, did the solver converge
  - `rotation_angle_deg`: float, magnitude of the rotation component
  - `translation_norm`: float, magnitude of the translation component
  - Properties: `R` (extracts $3 \times 3$ rotation), `t` (extracts $3 \times 1$ translation)

- **`TeaserParams`** (dataclass):
  - `c_threshold`: TLS truncation threshold (0.01 = 10 mm)
  - `noise_bound`: sensor noise level (0.001 = 1 mm for D405)
  - `rotation_gnc_factor`: GNC annealing factor (1.4)
  - `rotation_estimation_algorithm`: choice of solver
  - `normal_radius`, `fpfh_radius`: FPFH search radii
  - `ratio_threshold`, `mutual_filter`, `max_correspondences`: matching params
  - `ransac_iterations`, `ransac_confidence`: RANSAC fallback params
  - `se3_validation_tol`: tolerance for SE(3) validity check

**Key Functions:**

1. **`validate_se3(T, tol=1e-5)`** → bool
   - Shape check: must be $(4, 4)$
   - Bottom row check: must be $[0, 0, 0, 1]$
   - Orthonormality: $\|R^\top R - I\|_\infty < tol$
   - Determinant: $|\det(R) - 1| < tol$
   - All checks passed → valid SE(3) matrix

2. **`register_teaser(pcd_src, pcd_tgt, params)`** → SE3Result
   - Voxel-downsamples both clouds (if voxel_size > 0)
   - Computes FPFH on both → matches descriptors
   - Extracts correspondence point clouds
   - Routes to `_solve_teaser_core()` if `teaserpp_python` is installed, else `_solve_ransac_fallback()`
   - Validates output with `validate_se3()`

3. **`_solve_teaser_core(src_pts, tgt_pts, params)`** → SE3Result
   - Configures TEASER++ C++ solver via Python bindings
   - Sets noise bound, c threshold, rotation estimation parameters
   - Calls `solver.solve(src_pts, tgt_pts)`
   - Extracts `solution.rotation`, `solution.translation`, `solution.certificate`
   - Times execution

4. **`_solve_ransac_fallback(src_pts, tgt_pts, params)`** → SE3Result
   - Uses Open3D's `registration_ransac_based_on_feature_matching`
   - **No TLS cost** — uses standard L2 (squared error)
   - **No SDP certificate** — returns certificate=None
   - **No guaranteed outlier rejection** — RANSAC is heuristic, not certifiably robust
   - But works without TEASER++ installed, and is still decent for well-matched features

5. **`register_scene_to_cad(pcd_scene, pcd_cad, voxel_size=0.005, ...)`** → SE3Result
   - Convenience wrapper tuned for Scene→CAD registration
   - Tighter `c_threshold = 0.005` (CAD has no sensor noise)
   - Larger `fpfh_radius = 0.035` (scene is noisier, need wider search)
   - Presets optimised for the RePAIR dataset

**Design decisions:**
- TEASER++ is optional — the code works without it (RANSAC fallback), but results are less robust
- All parameters are gathered in `TeaserParams` for clean configuration and parameter sweeps
- SE(3) validation prevents returning invalid transformations

---

#### `registration/weighted_svd.py` — Weighted Kabsch Algorithm

**Purpose:** Solves the Weighted Orthogonal Procrustes Problem: find the best $SE(3)$ transformation aligning two point sets with per-point weights. Used as a building block when correspondences are known and reliable.

**Function: `weighted_svd_se3(src, tgt, weights, normalize_weights=False, allow_grad=False)`** → Tensor (4,4) or (B,4,4)

**Algorithm (Kabsch with weights):**

1. Normalise weights: $w_i' = w_i / \sum w_i$
2. Weighted centroids: $\bar{\mathbf{p}} = \sum w_i' \mathbf{p}_i$, $\bar{\mathbf{q}} = \sum w_i' \mathbf{q}_i$
3. Centre both point sets: $\tilde{\mathbf{p}}_i = \mathbf{p}_i - \bar{\mathbf{p}}$, $\tilde{\mathbf{q}}_i = \mathbf{q}_i - \bar{\mathbf{q}}$
4. Weighted cross-covariance: $H = \sum_i w_i' \tilde{\mathbf{q}}_i \tilde{\mathbf{p}}_i^\top$
5. SVD: $H = U \text{diag}(S) V^\top$
6. Rotation: $R = V \text{diag}(1, 1, \det(V U^\top)) U^\top$
   - The determinant correction ensures $\det(R) = 1$ (proper rotation, no reflection)
7. Translation: $\mathbf{t} = \bar{\mathbf{q}} - R\bar{\mathbf{p}}$
8. Assemble $4 \times 4$ SE(3) matrix

**Why PyTorch (not numpy)?** Primarily for batched operation and potential gradient propagation. If `allow_grad=True`, the SVD supports backpropagation — useful if this is used inside a neural network training loop.

**Why "Kabsch"?** Named after Wolfgang Kabsch (1976). The solution to the orthogonal Procrustes problem via SVD of the cross-covariance matrix.

---

#### `registration/se3_utils.py` — SE(3) Utility Functions

**Purpose:** Low-level SE(3) algebra needed across the codebase. All functions support batched tensors for efficiency.

**Functions:**

1. **`transform_points(T, points)`** → transformed points
   - $T$: $(4, 4)$ or $(B, 4, 4)$
   - `points`: $(N, 3)$ or $(B, N, 3)$
   - Implements: $\mathbf{p}' = R\mathbf{p} + \mathbf{t}$
   - Uses `torch.bmm` (batched matrix multiply) for efficiency

2. **`extract_rt(T)`** → `(R, t)`
   - Extracts $R$ (top-left $3 \times 3$) and $\mathbf{t}$ (top-right $3 \times 1$)

3. **`compose(T_ab, T_bc)`** → $T_{ac} = T_{ab} @ T_{bc}$
   - Standard SE(3) composition via matrix multiplication

4. **`inverse_transform(T)`** → $T^{-1}$
   - Uses the closed-form SE(3) inverse (no matrix inversion):
     $$T^{-1} = \begin{bmatrix} R^\top & -R^\top\mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$$
   - Much faster and numerically stable than `torch.inverse(T)`

---

### 6.3 `uncertainty/` — Epistemic Uncertainty Sub-Package

---

#### `uncertainty/geotransformer.py` — GeoTransformer Neural Network

**Purpose:** A neural network that processes point clouds and outputs per-point features. The architecture combines geometric attention (distance-aware) with Monte Carlo Dropout for uncertainty quantification. This is the learned feature extractor — it produces descriptors that can be used instead of (or alongside) FPFH.

**Architecture Walk-through:**

```
Input: (N, 6)  ←  (x, y, z, nx, ny, nz) per point
  │
  ▼
VoxelFeatureEncoder
  - Linear(6→64) → LayerNorm → ReLU
  - Linear(64→128) → LayerNorm → ReLU
  - Linear(128→128) → LayerNorm
  Output: (N, 128)
  │
  ▼
+ SinusoidalPositionEncoding  ←  adds positional signal
  │
  ▼
GeometricTransformer (4 blocks)
  Each block:
    GeometricAttention(distance-biased)
    MLP with GELU
    Pre-norm residual connections
  Output: (N, 128)
  │
  ▼
MCDropoutBottleneck  ←  STAYS ACTIVE during inference
  - Dropout(p=0.2)
  - Linear(128→64) → ReLU → Dropout(0.2)
  - Linear(64→128)
  Output: (N, 128)
  │
  ▼
FeatureDecoder
  - Linear(128→64) → LayerNorm → ReLU
  - Linear(64→32) → LayerNorm → ReLU
  - Linear(32→3)
  Output: (N, 3)  ←  3D coordinates
```

**Key Components:**

1. **`GeometricAttention(embed_dim=128, num_heads=4, dropout=0.1)`**
   - Standard multi-head attention augmented with geometric distance bias:
   - Query, Key, Value projections (Linear layers)
   - Attention scores: $\text{score}_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_k}} + \gamma \cdot \exp\left(-\frac{\|\mathbf{p}_i - \mathbf{p}_j\|^2}{2\sigma^2}\right)$
   - $\gamma$ and $\sigma$ are **learnable parameters** — the network learns how much to penalise distant points
   - Closer points get more attention weight; distant points are downweighted

2. **`MCDropoutBottleneck(dropout_rate=0.2)`**
   - The critical component for uncertainty quantification
   - Overrides `train()`/`eval()` behaviour: when `_mc_mode = True`, dropout stays on even in eval mode
   - Each forward pass deactivates a different random set of neurons → stochastic outputs
   - `enable_mc()` / `disable_mc()` control this behaviour

3. **`SinusoidalPositionEncoding(d_model=128)`**
   - Standard Transformer positional encoding, but applied to 3D coordinates
   - PE(pos, 2i) = sin(pos / 1000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos / 1000^(2i/d_model))
   - Applied independently to x, y, z and concatenated
   - Gives the network awareness of absolute position in 3D space

**Why Geometric Attention?** Standard attention treats all points equally — a point 1 cm away gets the same "attention budget" as a point 1 m away. Geometric attention biases toward nearby points, which makes physical sense for surface features.

---

#### `uncertainty/mc_inference.py` — Welford MC Accumulator

**Purpose:** Runs the GeoTransformer $T$ times with MC Dropout enabled and computes per-point mean and variance using Welford's online algorithm.

**Functions:**

1. **`run_mc_passes(model, point_cloud, T=50, batch_size=4096, device="cpu")`** → `(mean, variance)`
   - Enables MC mode on the model (`model.enable_mc()`)
   - Loops $T$ times:
     - Forward pass with MC Dropout active
     - Updates Welford accumulators online
   - Disables MC mode
   - Returns: `mean` $(N, 3)$ — the average predicted position per point, `variance` $(N,)$ — the scalar epistemic variance per point

2. **`run_mc_passes_batched(model, point_cloud, T=50, ...)`** → `(mean, variance)`
   - Alternative that stores all $T$ outputs for debugging
   - Memory-hungry ($\mathcal{O}(T \times N)$) but allows computing other statistics (percentiles, full distributions)
   - Useful for small $T$ or when you need per-realisation data

**Why Welford?** $T = 50$ passes on $N = 20,000$ points storing float32 positions would require $50 \times 20000 \times 3 \times 4 = 12$ MB — manageable. But with larger point clouds ($N = 100,000$) and $T = 100$, that's 120 MB. And for downstream processing, you might not want to store all intermediates. Welford guarantees constant memory.

**What's returned:** `mean` is the model's best guess of the point's position. `variance` is the scalar epistemic uncertainty — how much the model's predictions vary for that point across dropout masks.

---

#### `uncertainty/pose_covariance.py` — Pose Covariance in SE(3) Lie Algebra

**Purpose:** Given $T$ pose estimates from an MC Dropout loop, computes the $6 \times 6$ covariance matrix in $\mathfrak{se}(3)$ and projects spatial uncertainty onto the point cloud.

**Functions:**

1. **`se3_log(T)`** → $\boldsymbol{\xi} \in \mathbb{R}^6$
   - $\boldsymbol{\omega}$: axis-angle from rotation matrix via Rodrigues formula
     - $\theta = \arccos((\text{tr}(R)-1)/2)$
     - if $\theta \approx 0$: $\boldsymbol{\omega} = \mathbf{0}$ (avoid division by zero)
     - else: $\boldsymbol{\omega} = \frac{\theta}{2\sin\theta}(R - R^\top)^\vee$ (vee = extract vector from skew-symmetric matrix)
   - $\mathbf{v}$: translation corrected by left Jacobian inverse
     - $\mathbf{v} = J_{\text{left}}^{-1}(\boldsymbol{\omega}) \cdot \mathbf{t}$

2. **`se3_exp(xi)`** → $T \in SE(3)$
   - Inverse of log map
   - $R = \exp([\boldsymbol{\omega}]_\times)$ via Rodrigues
   - $\mathbf{t} = J_{\text{left}}(\boldsymbol{\omega}) \cdot \mathbf{v}$

3. **`_so3_left_jacobian(omega)`** → $J \in \mathbb{R}^{3 \times 3}$
   - $J = I + \frac{1-\cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times + \frac{\theta-\sin\theta}{\theta^3}[\boldsymbol{\omega}]_\times^2$
   - Handles small-angle case ($\theta \to 0$: $J \to I$)

4. **`_so3_left_jacobian_inverse(omega)`** → $J^{-1} \in \mathbb{R}^{3 \times 3}$
   - $J^{-1} = I - \frac{1}{2}[\boldsymbol{\omega}]_\times + \left(\frac{1}{\theta^2} - \frac{1+\cos\theta}{2\theta\sin\theta}\right)[\boldsymbol{\omega}]_\times^2$

5. **`compute_pose_covariance(poses)`** → `(Sigma(6,6), T_mean(4,4))`
   - Maps all poses to $\mathfrak{se}(3)$ via log
   - Computes mean twist $\bar{\boldsymbol{\xi}}$
   - Computes sample covariance $\Sigma$
   - Maps mean twist back to SE(3): $T_{\text{mean}} = \exp(\bar{\boldsymbol{\xi}})$

6. **`project_spatial_variance(Sigma, points)`** → `(N,)`
   - Projects 6×6 pose covariance to per-point scalar spatial variance via Jacobian

7. **`pose_covariance_statistics(Sigma)`** → dict
   - RMS translation uncertainty: $\sqrt{\frac{1}{3}\text{tr}(\Sigma_{tt})}$ in mm
   - RMS rotation uncertainty: $\sqrt{\frac{1}{3}\text{tr}(\Sigma_{rr})}$ in degrees
   - Trace of full Σ

8. **`print_covariance_report(Sigma, T_mean, spatial_var)`**
   - Prints eigenvalues, RMS uncertainties, mean pose, per-point variance percentiles

---

#### `uncertainty/variance_cloud.py` — Variance Cloud I/O

**Purpose:** Converts the output of MC Dropout inference (mean positions + per-point variance) into a coloured point cloud file (PCD or PLY) and provides visualisation.

**Functions:**

1. **`compute_variance_cloud(mean, variance)`** → `o3d.geometry.PointCloud`
   - Positions = mean (the model's best point estimates)
   - Colours = `variance_to_rgb(variance)` (blue-white-red colormap)
   - Normals.x channel stores the raw scalar variance values (for chaining to downstream tools)

2. **`variance_to_rgb(variance, clip_percentile=99)`** → `(N, 3)` RGB
   - Clips at 99th percentile (so one outlier point doesn't wash out the colour scale)
   - Normalises to [0, 1]
   - Low variance → Blue (cold, confident)
   - Medium variance → White (transition)
   - High variance → Red (hot, uncertain)

3. **`save_variance_cloud(mean, variance, output_path)`**
   - Computes cloud, writes PCD/PLY file

4. **`visualise_variance(mean, variance, title, clip_percentile)`**
   - Renders in interactive Open3D window with title

5. **`print_variance_statistics(variance)`** → dict
   - mean, median, std, min, max, p5, p25, p75, p95, p99

---

### 6.4 `scripts/` — Standalone CLI Tools

---

#### `scripts/compute_fpfh.py` — FPFH Descriptor Visualisation

**Purpose:** Computes FPFH descriptors and **colour-codes the point cloud** by projecting the 33-D descriptors down to 3-D RGB via PCA. Points with similar local geometry get similar colours.

**Function: `fpfh_to_rgb(fpfh)`** → `(N, 3)`
1. Mean-centre the $(N, 33)$ descriptor matrix
2. Build covariance $C = F_{\text{tilde}}^\top F_{\text{tilde}} / (N-1)$
3. Eigendecompose, take top 3 eigenvectors (capturing the most descriptor variation)
4. Project all descriptors onto these 3 directions → $(N, 3)$
5. Normalise to [0, 1] range → RGB

**Why PCA→RGB?** 33 dimensions can't be directly visualised. PCA finds the 3 directions that capture the most variance in the descriptor space, so points that look similar in the coloured cloud have similar FPFH descriptors — meaning similar geometry.

**CLI:** `python scripts/compute_fpfh.py fragment.ply --voxel-size 0.005 --fpfh-radius 0.025 --stats --viz`

---

#### `scripts/teaser_register.py` — End-to-End Registration

**Purpose:** The main registration script. Loads two point clouds, runs the full pipeline (downsample → FPFH → match → TEASER++), applies the transform, measures error, and visualises.

**Workflow:**
1. Load source and target clouds
2. Voxel-downsample both
3. Call `register_teaser()` from the registration package
4. Transform source by the estimated pose: `src_transformed = src.transform(T)`
5. Compute RMS distance between transformed source and target (Chamfer-like error)
6. Report rotation angle, translation norm, certificate
7. Visualise with source in orange, target in blue, aligned in green

**CLI:** `python scripts/teaser_register.py src.ply tgt.ply --voxel-size 0.005 --c-threshold 0.01 --output result.ply --viz`

---

#### `scripts/fpfh_parameter_sweep.py` — Parameter Grid Search

**Purpose:** Systematically tests FPFH parameter combinations to find the best settings for a specific fragment type.

**Parameter ranges:**
- `normal_radius`: 0.005, 0.01, 0.02, 0.03, 0.05 (m)
- `fpfh_radius`: 0.01, 0.015, 0.02, 0.025, 0.035, 0.05, 0.1 (m)
- `ratio_threshold`: 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99

**Quick mode** tests fewer combinations (3 × 3 × 3 = 27 instead of 5 × 7 × 7 = 245).

**Composite score formula:**
$$\text{score} = 0.3 \times \text{cert\_score} + 0.3 \times \text{inlier\_score} + 0.1 \times \text{time\_score} + 0.3 \times \text{rot\_score}$$

Each sub-score is normalised across all parameter combinations.

**Output:** CSV file with all combinations ranked by score. Use the top-ranked combination for production registration.

**CLI:** `python scripts/fpfh_parameter_sweep.py src.ply tgt.ply --quick --output sweep.csv`

---

#### `scripts/force_closure.py` — Force-Closure Analysis

**Purpose:** Given a mesh and two contact points, determines if a two-finger grasp is force-closed and computes the quality score $\varepsilon$.

**Key Classes:**

- **`Contact`:** Stores position, inward normal, friction cone generators $(m, 3)$, and wrench matrix $(6, m)$
- **`GraspResult`:** Stores contacts, antipodal scores, force_closure boolean, epsilon quality, LP status message

**Functions (in execution order):**

1. **`find_nearest_vertex(mesh, point)`** → `(index, position)` — uses mesh's KD-Tree to find closest vertex
2. **`get_vertex_normal(mesh, vertex_idx)`** → $\mathbf{n}$ — area-weighted vertex normal from triangular mesh
3. **`orthonormal_basis(n)`** → $(\mathbf{u}, \mathbf{v})$ — builds two orthogonal vectors perpendicular to the normal using a Householder-like construction
4. **`friction_cone_generators(normal, mu, m=8)`** → $(m, 3)$ — discretises the Coulomb friction cone
5. **`build_contact_wrench(position, generators)`** → $(6, m)$ — converts forces to wrenches via cross product
6. **`check_antipodal(c1, n1, c2, n2, mu)`** → `(bool, score1, score2)` — analytical two-finger check
7. **`test_force_closure_lp(W)`** → `(bool, epsilon, status, msg)` — solves the LP via SciPy HiGHS
8. **`analyse_grasp(mesh, c1, c2, mu, m_generators)`** → GraspResult — full pipeline

**CLI:** `python scripts/force_closure.py fragment.stl --contact1 "x y z" --contact2 "x y z" --mu 0.5 --quality --viz`

---

#### `scripts/cvar_grasp_validator.py` — CVaR Grasp Validation

**Purpose:** The most sophisticated script. Validates multiple grasp candidates against geometric uncertainty from the variance cloud, using CVaR to filter unsafe grasps.

**Key Classes:**

- **`GraspCandidate`:** Contact pair with optional normals
- **`CandidateResult`:** Per-candidate result — baseline epsilon, CVaR epsilon, number of failed realisations, accepted boolean

**Functions:**

1. **`load_variance_cloud(file_path)`** → `(mean, variance)`
   - Reads PCD/PLY file
   - Extracts mean from positions (xyz)
   - Extracts variance from normals.x channel (where `variance_cloud.py` stores it), or from the red colour channel intensity, or falls back to uniform small variance

2. **`estimate_normals_from_cloud(points)`** → `(N, 3)` — Open3D KD-tree normal estimation

3. **`sample_realizations(mean, variance, N=100, seed)`** → `(N, M, 3)`
   - For each of $N$ realisations: perturbs every point by isotropic Gaussian noise: $\mathbf{p}_i^{(k)} \sim \mathcal{N}(\boldsymbol{\mu}_i, \sigma^2_i I_3)$
   - Uses seeded RNG for reproducibility

4. **`evaluate_realization(points, normals, c1, c2, mu, m_generators)`** → `(fc_bool, eps)`
   - Rebuilds friction cones on the perturbed geometry
   - Tests force-closure via LP

5. **`compute_cvar(epsilon_values, alpha=0.05)`** → `(cvar_value, sorted_eps, k)`
   - Sorts epsilons ascending (worst first)
   - Averages the $K = \lceil \alpha N \rceil$ smallest

6. **`validate_grasps(candidates, mean_cloud, base_normals, variance, mu, m_generators, num_realizations=100, cvar_alpha=0.05)`** → list of CandidateResult
   - For each candidate:
     - Baseline force-closure on reference (mean) geometry
     - If baseline fails → rejected immediately
     - Generate $N$ realisations, test each
     - Compute CVaR, check against threshold
     - If CVaR > 0 → accepted
   - Rank accepted grasps by CVaR value (descending — higher is safer)

7. **`visualise_variance_with_contacts(...)`** — coloured variance cloud overlaid with green (accepted) or red (rejected) contact spheres

**CLI:** `python scripts/cvar_grasp_validator.py variance_cloud.pcd --candidates sample_candidates.json --mu 0.5 --num-realizations 100 --top-k 5 --output accepted_grasps.json --viz`

---

#### `scripts/mc_dropout_variance.py` — Variance Cloud Generation

**Purpose:** Generates the epistemic variance cloud that feeds into the CVaR validator. This is the bridge between the GeoTransformer model and the grasp validation pipeline.

**Workflow:**
1. Load point cloud (position + normals) → $(N, 6)$ tensor
2. Load GeoTransformer from checkpoint (or random init)
3. Call `run_mc_passes(model, cloud, T=50)` → `(mean, variance)`
4. Print variance statistics
5. Save coloured variance cloud PCD
6. Visualise if requested

**CLI:** `python scripts/mc_dropout_variance.py fragment.pcd --model checkpoint.pt --num-passes 50 --dropout-rate 0.2 --output var.pcd --viz`

---

#### `scripts/mc_pose_covariance.py` — Pose Covariance Estimation

**Purpose:** Runs the complete pose inference pipeline in an MC loop to estimate the $6 \times 6$ pose covariance.

**Workflow:**
1. Load source and target point clouds
2. Optionally load GeoTransformer model
3. For each MC pass (x50):
   - Extract GeoTransformer features → match → TEASER++ → collect pose $T_t$
4. Compute $\Sigma$ (6×6) from all poses
5. Project spatial uncertainty onto source points
6. Save: covariance cloud PCD, aligned mean pose PLY, $\Sigma$ as `.npy`
7. Print covariance report
8. Visualise

**CLI:** `python scripts/mc_pose_covariance.py src.ply tgt.ply --model checkpoint.pt --num-passes 50 --voxel-size 0.005 --output cov.pcd --viz`

---

#### `scripts/sample_candidates.json` — Sample Grasp Candidates

**Purpose:** Provides example grasp candidate data for testing the CVaR validator. Format:
```json
[
  {
    "contact1": [0.023, -0.015, 0.041],
    "contact2": [-0.019, 0.021, -0.038]
  },
  ...
]
```

Each entry is a pair of 3D coordinates specifying where the two gripper fingers should make contact.

---

### 6.5 `repair_simulation/` — ROS2 MoveIt2 Package

---

#### `repair_simulation/repair_simulation/hand_eye.py` — AX=XB Calibration

**Purpose:** Implements the Tsai-Lenz hand-eye calibration algorithm. Supports both eye-to-hand (camera fixed in world — our setup) and eye-in-hand (camera on end-effector).

**Key Class: `HandEyeCalibration`**:
- Constructor takes `mode` ("eye_to_hand" or "eye_in_hand") and optional `X_init` (default: identity with heuristic offsets)
- `calibrate(motions_A, motions_B)` → $X$ — solves $AX = XB$ from motion pairs
- `transform_pose(T_camera)` → $T_{\text{robot}} = X \cdot T_{\text{camera}}$ (eye-to-hand)
- `transform_pose_with_ee(T_camera, T_ee)` → $T_{\text{robot}} = T_{\text{ee}} \cdot X \cdot T_{\text{camera}}$ (eye-in-hand)
- `set_calibration(X)` — manually set $X$

**Internal helpers:** `_so3_log(R)` → axis-angle, `_so3_exp(omega)` → rotation matrix (both standalone numpy implementations).

---

#### `repair_simulation/repair_simulation/grasp_executor.py` — MoveIt2 Grasp Execution

**Purpose:** ROS2 node that takes CVaR-validated grasps and executes them on the robot (or in simulation). This is the final stage of the pipeline.

**Key Class: `GraspExecutor(Node)`:**

**Trajectory Stages:**

1. **Stage 1 — Open gripper:** Sends command to open the parallel-jaw gripper to a width wider than the fragment
2. **Stage 2 — Pre-grasp approach:** Joint-space plan to a position 5 cm directly above the target grasp pose. Joint-space planning is fast and robust — no need for straight-line constraint here.
3. **Stage 3 — Cartesian descent:** Linearly interpolates from pre-grasp down to the grasp pose in 5 mm steps, checking each waypoint for feasibility. At least 95% of waypoints must be reachable. This stage is the most critical — it ensures the gripper approaches straight down without drifting sideways (which could knock over neighbouring fragments).
4. **Stage 4 — Close gripper + lift:** Closes the gripper on the fragment, then Cartesian lift back to retreat height.

**Configuration:**
- Planning group: `mirobot_arm`
- Gripper joints: `gripper_left_joint`, `gripper_right_joint`
- EE link: `tool0`
- Base frame: `world`
- Approach distance: 0.05 m
- Retreat: 0.05 m
- Cartesian step size: 0.005 m

**Key Functions:**

- **`execute_from_file(grasp_file, rank=0)`** — reads JSON, takes the `rank`-th grasp, builds pose, executes
- **`execute_pose(x, y, z, roll, pitch, yaw)`** — direct pose execution (bypasses CVaR pipeline)
- **`_build_grasp_pose(centre, direction)`** — builds SE(3) pose: tool Z = world $-Z$ (downward), tool X aligned with inter-contact direction, tool Y = Z × X. Transforms through hand-eye calibration.
- **`_plan_and_move_to_pose(pose, label)`** — joint-space planning + execution via MoveIt2
- **`_execute_cartesian_path(waypoints, label)`** — Cartesian path planning + execution

---

#### Supporting ROS2 Files

| File | Purpose |
|---|---|
| `package.xml` | ROS2 package manifest — declares dependencies on `rclpy`, `moveit_core`, `moveit_ros_planning_interface`, `moveit_ros_move_group`, `geometry_msgs`, `tf2_*`, `control_msgs`, `trajectory_msgs` |
| `CMakeLists.txt` | colcon build configuration — uses `ament_cmake` and `ament_cmake_python`, installs Python package and executable |
| `setup.py` | Python entry points — registers `grasp_executor = repair_simulation.grasp_executor:main` |
| `resource/repair_simulation` | Package marker — empty file that marks the package root for ament |

---

## 7. Step-by-Step Walkthrough — Running the Pipeline

This section is a hands-on guide. Follow these steps in order. Each step lists the command, explains what happens, and tells you what to expect.

---

### Prerequisites Check

Before starting, verify your installation:

```bash
python -c "import numpy, torch, open3d, scipy, trimesh; print('Core deps: OK')"
python -c "import teaserpp_python" 2>/dev/null && echo "TEASER++: OK" || echo "TEASER++: not installed (RANSAC fallback will be used)"
```

---

### Step 1 — Preprocessing (30 seconds)

Clean up your raw fragment scan: downsampling for uniform density, PCA normal estimation for surface orientation.

```bash
python voxel_downsample_normals.py fragment.ply --voxel-size 0.005 --knn 30 --viz
```

**Behind the scenes:**
1. `load_point_cloud("fragment.ply")` — reads the file
2. `downsample_voxel_grid(pcd, 0.005)` — partitions space into 5mm cubes, averages points within each
3. `estimate_normals_pca_knn(pcd, 30)` — for each point, finds 30 nearest neighbours, builds covariance, eigendecomposes, takes smallest-eigenvalue eigenvector as normal
4. `save_point_cloud(...)` — writes `fragment_ds.ply`
5. `visualize(pcd)` — opens interactive 3D view

**What you'll see:** A cleaner point cloud with purple lines (normals) sticking out from each point. The normals should point consistently outward. If normals on a flat surface point in random directions, try increasing `--knn` to 50.

**Output file:** `fragment_ds.ply` — feeds into Step 2.

---

### Step 2 — FPFH Descriptors (1 minute)

Compute 33-D geometric descriptors and colour-code the cloud for inspection.

```bash
python scripts/compute_fpfh.py fragment.ply --voxel-size 0.005 --fpfh-radius 0.025 --stats --viz
```

**Behind the scenes:**
1. Loads and voxel-downsamples the cloud
2. `compute_fpfh(pcd, fpfh_radius=0.025)` — for each point: SPFH → weighted neighbour average → FPFH
3. `fpfh_to_rgb(fpfh)` — PCA projects 33-D → 3-D RGB
4. Prints: shape, mean, std, sparsity, entropy of descriptors
5. Renders side-by-side: original geometry (left), FPFH-coloured (right)

**What you'll see:** A coloured point cloud. Flat surfaces should be one colour, curved rims another, sharp edges a third. If the whole cloud is one uniform colour, the `fpfh_radius` might be too small — descriptors aren't capturing enough neighbourhood to distinguish features.

**Key metric:** Entropy > 2.0 → descriptors are discriminative. Entropy < 1.5 → descriptors are too uniform, increase `--fpfh-radius`.

---

### Step 3 — TEASER++ Registration (2–3 minutes)

Register your fragment to a scene scan. This is the core pose estimation.

```bash
python scripts/teaser_register.py src_fragment.ply target_scene.ply \
    --voxel-size 0.005 --c-threshold 0.01 --output aligned.ply --viz
```

**Behind the scenes:**
1. Loads both clouds, voxel-downsamples
2. FPFH on both → matching with mutual filter + Lowe ratio test (0.9)
3. If TEASER++ available: GNC-TLS with $c = 0.01$ (10 mm truncation), SDP certificate
4. If not: Open3D RANSAC with L2 cost (no certificate)
5. Applies $T$ to source → aligned cloud
6. Reports: rotation angle, translation norm, certificate, runtime, inlier count

**What you'll see (good result):**
- Rotation angle < 5°
- Certificate close to 0 (e.g., 0.001)
- Three overlapping clouds in visualisation: orange (source), blue (target), green (aligned)
- Green cloud should closely overlay blue target

**What you'll see (poor result):**
- Rotation > 30° or certificate is N/A
- Aligned green cloud far from blue target
- → Run Step 4 to find better FPFH parameters

**Output files:** `aligned.ply` (the transformed source cloud, now registered to the target's coordinate frame).

---

### Step 4 — FPFH Parameter Sweep (Optional, 10–30 minutes)

If registration quality from Step 3 is poor, sweep parameters:

```bash
python scripts/fpfh_parameter_sweep.py src.ply tgt.ply \
    --quick --output sweep_results.csv
```

**What happens:**
- Grid-searches `normal_radius`, `fpfh_radius`, `ratio_threshold`
- For each combination: runs full registration, computes inlier rate, error, composite score
- Writes ranked CSV

**How to use the results:**
1. Open `sweep_results.csv` in Excel or any spreadsheet
2. Sort by `score` descending
3. Use the top-ranked combination's parameters for Step 3:
   ```bash
   python scripts/teaser_register.py src.ply tgt.ply \
       --fpfh-radius <best_radius> --ratio-threshold <best_ratio> --c-threshold 0.01 --viz
   ```

---

### Step 5 — MC Dropout Variance Cloud (5–10 minutes)

Generate the epistemic uncertainty map that will guide safe grasping.

```bash
python scripts/mc_dropout_variance.py fragment.pcd \
    --model checkpoints/geotransformer_best.pt \
    --num-passes 50 --dropout-rate 0.2 \
    --output variance_cloud.pcd --viz
```

**Behind the scenes:**
1. Loads GeoTransformer from checkpoint
2. Loads fragment point cloud with normals → (N, 6) tensor
3. Enables MC Dropout mode (dropout stays on during inference)
4. Runs 50 stochastic forward passes
5. Welford's algorithm computes per-point mean and variance incrementally
6. Variance → RGB colormap (blue=low, red=high)
7. Saves PCD with positions=mean, colours=variance_rgb, normals.x=variance_values

**What you'll see:**
- A point cloud coloured by uncertainty
- **Blue regions:** Flat, well-sampled surfaces. Model is confident. Safe to grasp.
- **Red regions:** Edges, thin parts, complex curves. Model disagrees with itself. Avoid grasping here.
- Variance statistics printed to console

**Output file:** `variance_cloud.pcd` — feeds into Step 8.

**Note:** If you don't have a trained model, the script works with random weights. The variance patterns will be less informative but the pipeline still functions for testing.

---

### Step 6 — Pose Covariance (5–10 minutes)

Quantify the uncertainty in the full 6-DoF pose estimate.

```bash
python scripts/mc_pose_covariance.py src.ply tgt.ply \
    --model checkpoints/geotransformer_best.pt \
    --num-passes 50 --voxel-size 0.005 \
    --output cov_cloud.pcd --viz
```

**Behind the scenes:**
1. For each of 50 MC passes:
   - GeoTransformer forward pass → features
   - Feature matching → correspondences
   - TEASER++ → pose $T_t$
2. All poses mapped to se(3) via log map
3. 6×6 covariance $\Sigma$ computed
4. Spatial uncertainty projected onto each point via Jacobian
5. Saves: covariance cloud PCD, aligned mean pose PLY, $\Sigma$ as `.npy`

**What you'll see:**
- Covariance report printed to console
- RMS translation uncertainty: < 5 mm (good), > 20 mm (poor)
- RMS rotation uncertainty: < 3° (good), > 10° (poor)
- Coloured cloud showing spatial uncertainty distribution
- `.npy` file with the full 6×6 $\Sigma$ matrix

**Output files:** `cov_cloud.pcd`, `aligned_mean.ply`, `pose_covariance.npy`.

---

### Step 7 — Force-Closure Analysis (30 seconds)

Test whether a specific grasp pair is physically stable.

```bash
python scripts/force_closure.py fragment.stl \
    --contact1 "0.023 -0.015 0.041" \
    --contact2 "-0.019 0.021 -0.038" \
    --mu 0.5 --quality --viz
```

**Behind the scenes:**
1. Loads mesh from STL
2. Finds nearest vertices to contact coordinates
3. Gets surface normals at those vertices
4. Builds 8 friction cone generators per contact
5. Constructs 6×16 wrench matrix
6. Solves force-closure LP → $\varepsilon$, boolean FC
7. Checks antipodal condition
8. Renders mesh with contact points

**What you'll see:**
- `Force-Closure: Yes` / `Force-Closure: No`
- $\varepsilon$ quality score
- Mesh visualisation with contact spheres and normal arrows

**Good facts to check:**
- Normals should point inward (into the object), not outward
- Contacts should be on opposite sides (antipodal)
- $\varepsilon > 0.01$ is a solid grasp

---

### Step 8 — CVaR Grasp Validation (5 minutes)

Validate all grasp candidates against geometric uncertainty.

```bash
python scripts/cvar_grasp_validator.py variance_cloud.pcd \
    --candidates scripts/sample_candidates.json \
    --mu 0.5 --num-realizations 100 --top-k 5 \
    --output accepted_grasps.json --viz
```

**Behind the scenes:**
1. Loads variance cloud (mean + per-point variance)
2. Loads grasp candidates from JSON
3. Baseline FC check on reference geometry
4. Generates 100 geometric realisations (Monte Carlo sampling with per-point variance)
5. For each realisation: re-estimate normals, rebuild friction cones, test FC
6. Compute $\mathrm{CVaR}_{0.05}$ = average over worst 5 realisations
7. Accept iff baseline FC AND CVaR > 0
8. Rank accepted by CVaR value
9. Save accepted grasps to JSON
10. Visualise: variance cloud with green/red contact spheres

**What you'll see:**
- How many candidates were accepted (typically 1–3)
- CVaR scores for each
- Which realisations caused failure for rejected grasps
- Green spheres on blue (low-variance) regions → good
- Red spheres on red (high-variance) regions → correctly rejected

**Output file:** `accepted_grasps.json` — feeds into Step 10.

---

### Step 9 — Build ROS2 Package (2 minutes)

```bash
colcon build --symlink-install --packages-select repair_simulation
source install/setup.bash
```

**Behind the scenes:**
- `colcon build` compiles the CMake-based ROS2 package
- `--symlink-install` symlinks Python files (no rebuild on edits)
- `--packages-select repair_simulation` builds only our package (faster)
- `source install/setup.bash` adds the package to your ROS2 environment

**Verify:** `ros2 pkg list | grep repair_simulation` should show the package.

---

### Step 10 — Execute Grasp (seconds)

```bash
# Option A: Execute top-ranked CVaR grasp
ros2 run repair_simulation grasp_executor \
    --ros-args -p grasp_file:=accepted_grasps.json

# Option B: Execute manually specified pose
ros2 run repair_simulation grasp_executor \
    --ros-args -p target_x:=0.35 -p target_y:=0.0 -p target_z:=0.12 \
    -p roll:=0.0 -p pitch:=3.14 -p yaw:=0.0
```

**Behind the scenes (Option A):**
1. Reads `accepted_grasps.json`
2. Takes the top-ranked grasp (highest CVaR)
3. Computes grasp centre and direction from contact points
4. Builds SE(3) grasp pose
5. Transforms from camera frame to robot frame (hand-eye calibration)
6. **Stage 1:** Opens gripper
7. **Stage 2:** Joint-space plan to pre-grasp position (5 cm above target)
8. **Stage 3:** Cartesian descent through waypoints to grasp pose
9. **Stage 4:** Closes gripper, lifts to retreat

**What you'll see:**
- MoveIt2 trajectory visualisation in RViz (if GUI available)
- Console output logging each stage
- Final result: fragment at retreat position

---

## 8. Interpreting Your Results

### Registration Quality

| Metric | Excellent | Good | Fair | Poor | Action |
|---|---|---|---|---|---|
| Rotation error | < 1° | 1–5° | 5–15° | > 15° | If poor: sweep FPFH params |
| Translation error | < 2 mm | 2–10 mm | 10–30 mm | > 30 mm | If poor: check target cloud quality |
| SDP certificate | < 0.001 | 0.001–0.01 | 0.01–0.1 | > 0.1 or N/A | Install TEASER++ if N/A |
| Inlier rate | > 80% | 50–80% | 20–50% | < 20% | Lower ratio_threshold |

### Grasp Quality

| $\varepsilon$ (FC quality) | Verdict | Meaning |
|---|---|---|
| > 0.05 | Excellent | Large margin — fragment is very secure |
| 0.01–0.05 | Good | Adequate for tabletop manipulation |
| 0.001–0.01 | Marginal | Near boundary — test with CVaR |
| 0 | No FC | Grasp will slip — reject candidate |

### Variance Cloud Colours

| Colour | Variance | Interpretation | Grasp here? |
|---|---|---|---|
| Deep blue | Very low | Model is confident; flat, well-sampled surface | Yes — safest choice |
| Light blue / cyan | Low | Minor uncertainty | Yes — likely safe |
| White | Moderate | Some disagreement between MC passes | Caution |
| Orange | High | Significant disagreement | Risky |
| Red | Very high | Model is uncertain (edges, thin features, occlusions) | Avoid |

### CVaR Behaviour

- **Accepted, $\mathrm{CVaR}_{0.05} = 0.05$:** Grasp works under all variation, large margin. Top choice.
- **Accepted, $\mathrm{CVaR}_{0.05} = 0.001$:** Works but barely. Consider different grasp.
- **Rejected, $\mathrm{CVaR}_{0.05} = 0$:** At least one geometric realisation had zero force-closure. Unsafe.
- **Rejected, baseline FC fails:** The grasp doesn't even work on the reference geometry. Try a different contact pair.

### Typical Good Configuration

For a typical RePAIR plaster fragment (~10 cm diameter, 1 cm thick, scanned at 30 cm distance with D405):

- Voxel size: 0.005 m
- FPFH radius: 0.025 m (for source), 0.035 m (for scene)
- Normal radius: 0.01 m
- Ratio threshold: 0.9
- c_threshold: 0.01 m
- μ (friction): 0.5
- MC passes: 50
- CVaR α: 0.05
- CVaR realisations: 100

---

## 9. Installation &amp; Dependencies

### Python Core (Required)

```bash
pip install numpy torch open3d scipy trimesh
```

These five packages run everything except the ROS2 simulation.

| Package | Version | Used For |
|---|---|---|
| `numpy` | ≥1.24 | Linear algebra, SVD, matrix operations |
| `torch` | ≥2.0 | PyTorch — GeoTransformer, batched SVD |
| `open3d` | ≥0.17 | Point cloud I/O, FPFH, KD-tree, visualisation |
| `scipy` | ≥1.10 | Linear programming (HiGHS solver) |
| `trimesh` | ≥3.21 | Mesh loading, vertex normals |

### TEASER++ (Recommended)

```bash
pip install teaserpp-python
```

Without it: pipeline falls back to Open3D RANSAC (L2 cost, no SDP certificate, no guaranteed outlier rejection). Registration still works but is less robust.

### ROS2 (For Simulation Only)

```bash
# Ubuntu 24.04
sudo apt install ros-jazzy-ros-base ros-jazzy-moveit
pip install rclpy moveit-py
```

### Verify Installation

```bash
python -c "import numpy, torch, open3d, scipy, trimesh; print('Core: OK')"
python -c "import teaserpp_python" 2>/dev/null && echo "TEASER++: OK" || echo "TEASER++: MISSING"
python -c "import rclpy, moveit_py" 2>/dev/null && echo "ROS2: OK" || echo "ROS2: MISSING (simulation only)"
```

---

## 10. Evaluation Metrics

### ADD-S (Average Distance of Symmetric-Defined)

Used when the object has continuous symmetries (e.g., a bowl is rotationally symmetric about its axis — there's no single "correct" orientation):

$$\text{ADD-S}(\hat{T}, T_{gt}) = \frac{1}{N}\sum_{i=1}^N \min_{\mathbf{p}_j \in \mathcal{S}} \|\hat{T}(\mathbf{p}_i) - T_{gt}(\mathbf{p}_j)\|$$

For each point transformed by the estimated pose, find the closest point on the ground-truth model. Average those distances. Robust to symmetries because the matching is per-point, not fixed-correspondence.

### Chamfer Distance

For evaluating registration when you don't have point-to-point correspondences:

$$d_{\text{Chamfer}}(P, Q) = \frac{1}{|P|}\sum_{\mathbf{p} \in P} \min_{\mathbf{q} \in Q} \|\mathbf{p} - \mathbf{q}\| + \frac{1}{|Q|}\sum_{\mathbf{q} \in Q} \min_{\mathbf{p} \in P} \|\mathbf{q} - \mathbf{p}\|$$

Measures bidirectional geometric similarity. The first term is the average forward distance (each source point to its nearest target); the second is the average backward distance. Both together ensure the clouds truly overlap, not just that one is contained in the other. Lower is better.

### RMS Pose Error

$$\text{RMS}_{\text{trans}} = \|\mathbf{t}_{\text{est}} - \mathbf{t}_{\text{gt}}\|_2$$
$$\text{RMS}_{\text{rot}} = \arccos\left(\frac{\text{tr}(R_{\text{est}}^\top R_{\text{gt}}) - 1}{2}\right)$$

Only usable when you have ground-truth pose (e.g., from a calibrated motion capture system or known CAD alignment).

---

## 11. Troubleshooting Guide

### Registration Problems

| Symptom | Likely Cause | Fix |
|---|---|---|
| Source cloud far from target after registration | FPFH descriptors not discriminative | Increase `--fpfh-radius` to 0.035, run parameter sweep |
| Registration takes > 5 minutes | Too many correspondences | Reduce `--max-correspondences` to 1000 |
| TEASER++ certificate = "N/A (RANSAC)" | `teaserpp-python` not installed | `pip install teaserpp-python` |
| Rotation error > 30° | Insufficient inlier ratio | Decrease `--ratio-threshold` to 0.7, increase `--c-threshold` to 0.05 |
| Large SDP certificate (> 0.1) | Many outliers, solver struggled | Clean the scene cloud, check for occlusions |

### Grasping Problems

| Symptom | Likely Cause | Fix |
|---|---|---|
| All grasps rejected by CVaR | Variance cloud has high uncertainty everywhere | Check model checkpoint, increase `--num-passes` to 100 |
| Force-closure LP fails | SciPy < 1.10 (no HiGHS solver) | `pip install scipy>=1.10` |
| Normal vectors point wrong direction | Mesh normals not oriented | Use `--flip-normals` flag or preprocess mesh |
| CVaR always returns 0 | Variance too high, all realisations fail | Reduce dropout rate to 0.1, increase num-passes |
| `sample_candidates.json` has wrong coordinates | Contact points outside the mesh | Adjust coordinates to lie on actual fragment surface |

### Build &amp; Runtime Problems

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'registration'` | Package not on PYTHONPATH | Run from project root directory |
| Open3D visualisation doesn't appear | Headless / SSH environment | Use `--no-viz` flag |
| ROS2 build fails | Missing dependencies | `rosdep install --from-paths . --ignore-src -y` |
| `colcon build` can't find package | Wrong directory | Run from workspace root (containing `repair_simulation/`) |
| `ros2 run` can't find executable | Didn't source setup.bash | `source install/setup.bash` |
| Variance cloud is all one colour | Dropout rate too low or MC mode not active | Set `--dropout-rate 0.2`, verify `enable_mc()` called |
| GPU out of memory | Point cloud too large | Reduce batch size, voxel-downsample more aggressively |

### Interpreting Errors

| Error Message | Meaning | What To Do |
|---|---|---|
| `LinAlgError: SVD did not converge` | Degenerate point configuration (all points collinear) | Downsample more, or the cloud is too flat |
| `ValueError: zero-size array` | No correspondences found | Decrease ratio_threshold, increase fpfh_radius |
| `RuntimeError: CUDA out of memory` | Point cloud too large for GPU | Use `--device cpu`, or reduce voxel size further |
| `AssertionError: not valid SE(3)` | Registration produced invalid output | Check c_threshold, try RANSAC fallback |

---

## 12. Glossary of Terms

| Term | Definition |
|---|---|
| **6-DoF** | Six degrees of freedom — 3 translations (x, y, z) + 3 rotations (roll, pitch, yaw). Fully describes a rigid body's position and orientation in 3D space. |
| **ADD-S** | Average Distance of Symmetric-Defined — evaluation metric robust to object symmetries. |
| **Aleatoric uncertainty** | Uncertainty from inherent randomness (sensor noise). Cannot be reduced by more training. |
| **Antipodal** | Two points on opposite sides of an object. A necessary condition for force-closure. |
| **Cartesian path** | A straight-line path in 3D space (as opposed to joint space where each joint moves independently). |
| **Chamfer distance** | Bidirectional distance between two point clouds — forward + backward nearest-neighbour distances. |
| **Correspondence** | A pairing between a point in the source cloud and a point in the target cloud. |
| **Coulomb friction** | Friction model where maximum static friction force = μ × normal force. The friction cone has half-angle arctan(μ). |
| **CVaR (Conditional Value-at-Risk)** | The average over the worst α×100% of outcomes. Used to filter risky grasps. |
| **Darboux frame** | A local coordinate frame defined by a point's normal and the vector to a neighbour. |
| **Eigendecomposition** | Factorisation of a matrix into eigenvectors (directions) and eigenvalues (magnitudes). |
| **Epistemic uncertainty** | Uncertainty from lack of knowledge — the model doesn't know because it hasn't seen similar data. Reducible with more training. |
| **Force-closure** | A grasp that can resist any external wrench (force + torque). The minimum requirement for a stable grasp. |
| **FPFH** | Fast Point Feature Histogram — a 33-dimensional geometric descriptor robust to textureless surfaces. |
| **Friction cone** | The set of all force directions at a contact that don't cause slipping. Discretised with 8 generators. |
| **GNC (Graduated Non-Convexity)** | Optimisation technique that starts with a convex problem and gradually deforms it to the non-convex target. |
| **GWS (Grasp Wrench Space)** | The convex hull of all wrench vectors from all contact points. |
| **ICP (Iterative Closest Point)** | Classic registration algorithm that alternates between finding nearest-neighbour correspondences and minimising L2 error. Not used here due to outlier sensitivity. |
| **Inlier** | A correct correspondence. |
| **Jacobian** | Matrix of partial derivatives. In SE(3) context: how a small change in pose affects point positions. |
| **Kabsch algorithm** | Closed-form solution for optimal rotation + translation given known correspondences. Uses SVD. |
| **KD-Tree** | Spatial data structure for fast nearest-neighbour queries. |
| **Lie algebra (se(3))** | The tangent space of SE(3) at the identity. A vector space where poses can be added and averaged. |
| **Lie group (SE(3))** | The set of all rigid-body transformations in 3D. A smooth, curved manifold — not a vector space. |
| **Log map** | Mapping from SE(3) to se(3). "Unwinds" a transformation into a twist vector. |
| **Lowe's ratio test** | Filters correspondences where the best match is not significantly better than the second-best. |
| **Maximum Clique** | The largest set of mutually consistent correspondences in a graph. Used by TEASER++ for inlier selection. |
| **MC Dropout** | Keeping dropout active at inference time. Each forward pass samples a different sub-network, enabling uncertainty estimation. |
| **MoveIt2** | ROS2 motion planning framework. Handles inverse kinematics, collision checking, and trajectory generation. |
| **Normal (surface)** | Vector perpendicular to the surface at a point. |
| **Outlier** | A wrong correspondence — two points paired together that aren't actually the same physical point. |
| **PCA (Principal Component Analysis)** | Dimensionality reduction technique. Here used for normal estimation (direction of least variance). |
| **PCD / PLY** | Point cloud file formats. PCD = Point Cloud Data (PCL native), PLY = Polygon File Format (Stanford). |
| **RANSAC** | RANdom SAmple Consensus — heuristic algorithm that repeatedly samples random subsets to find inliers. |
| **Registration** | Finding the SE(3) transformation that aligns one point cloud to another. |
| **Rodrigues formula** | Converts between axis-angle representation and rotation matrix. |
| **ROS2** | Robot Operating System 2 — middleware for building robot applications. |
| **SE(3)** | Special Euclidean group in 3D — the set of all rigid-body transformations. |
| **SDP (Semidefinite Programming)** | Convex optimisation over positive semidefinite matrices. Used to provide provable optimality bounds. |
| **SPFH** | Simplified Point Feature Histogram — the raw 33-bin angular histogram before neighbour averaging. |
| **SVD (Singular Value Decomposition)** | Factorisation $A = U \Sigma V^\top$. Used in Kabsch algorithm and PCA. |
| **TEASER++** | Truncated least squares Estimation And SEmidefinite Relaxation — certifiably robust registration algorithm. |
| **TLS (Truncated Least Squares)** | Cost function that caps residuals beyond a threshold, making the optimisation robust to outliers. |
| **Tsai-Lenz** | Two-stage hand-eye calibration method — solves rotation then translation. |
| **Twist / Screw** | An element of se(3): a 6D vector (3 linear + 3 angular velocity) that encodes an incremental rigid-body motion. |
| **Voxel** | Volume element — a small 3D cube in a regular grid partition of space. |
| **Welford's algorithm** | Online algorithm for computing mean and variance in a single pass with constant memory. |
| **Wrench** | A 6D vector combining force (3D) and torque (3D). |
| **$\mathfrak{se}(3)$** | The Lie algebra of SE(3) — the tangent space at the identity, isomorphic to ℝ⁶. |
| **$\varepsilon$ (epsilon)** | Grasp quality score from the force-closure LP. Higher = more robust. |
| **$\mu$ (mu)** | Coefficient of friction. μ = 0.5 for plaster on rubber/gripper pads. |
| **$\Sigma$ (Sigma)** | Covariance matrix. 6×6 for pose uncertainty in se(3). |
| **$\sigma^2$ (sigma squared)** | Variance — per-point epistemic uncertainty from MC Dropout. |

---

## 13. Quick Reference — All Commands

```bash
# ── Module 1: Perception ──────────────────────────────────────

# Step 1: Preprocessing (voxel + normals)
python voxel_downsample_normals.py fragment.ply --voxel-size 0.005 --knn 30 --viz

# Step 2: FPFH descriptor computation + visualisation
python scripts/compute_fpfh.py fragment.ply --voxel-size 0.005 --fpfh-radius 0.025 --stats --viz

# ── Module 2: Registration ────────────────────────────────────

# Step 3: TEASER++ global registration
python scripts/teaser_register.py src.ply tgt.ply --voxel-size 0.005 --c-threshold 0.01 --output result.ply --viz

# Step 4: FPFH parameter sweep (optimisation)
python scripts/fpfh_parameter_sweep.py src.ply tgt.ply --quick --output sweep_results.csv

# ── Module 3: Uncertainty + Grasping ──────────────────────────

# Step 5: MC Dropout variance cloud
python scripts/mc_dropout_variance.py fragment.pcd --model checkpoint.pt --num-passes 50 --dropout-rate 0.2 --output var.pcd --viz

# Step 6: MC pose covariance (6×6 Σ)
python scripts/mc_pose_covariance.py src.ply tgt.ply --model checkpoint.pt --num-passes 50 --voxel-size 0.005 --output cov.pcd --viz

# Step 7: Force-closure analysis
python scripts/force_closure.py fragment.stl --contact1 "x1 y1 z1" --contact2 "x2 y2 z2" --mu 0.5 --quality --viz

# Step 8: CVaR grasp validation
python scripts/cvar_grasp_validator.py var.pcd --candidates scripts/sample_candidates.json --mu 0.5 --num-realizations 100 --top-k 5 --output accepted.json --viz

# ── Module 4: ROS2 Simulation ─────────────────────────────────

# Step 9: Build
colcon build --symlink-install --packages-select repair_simulation
source install/setup.bash

# Step 10: Execute
ros2 run repair_simulation grasp_executor --ros-args -p grasp_file:=accepted_grasps.json
# OR: direct pose
ros2 run repair_simulation grasp_executor --ros-args -p target_x:=0.35 -p target_y:=0.0 -p target_z:=0.12 -p roll:=0.0 -p pitch:=3.14 -p yaw:=0.0
```

---

## 14. Phase 1 Summary

| Metric | Count |
|---|---|
| Commits on `main` | 22 |
| Files in repository | 28 |
| Lines of Python code | ~8,000 |
| Packages / packages | 4 |
| Standalone scripts | 7 |
| ROS2 package | 1 |

| Module | Status | Key Features |
|---|---|---|
| **1. Perception** | Complete | Voxel downsampling, PCA normals, FPFH (33-D), GeoTransformer with geometric attention + MC Dropout |
| **2. Registration** | Complete | TEASER++ with TLS cost, Maximum Clique inlier selection, GNC-TLS rotation, SDP optimality certificate, RANSAC fallback |
| **3. Grasping** | Complete | GWS construction, Force-Closure LP (HiGHS), CVaR α=0.05 filter, epistemic variance cloud, SE(3) pose covariance Σ |
| **4. Simulation** | Complete | ROS2 MoveIt2 package, AX=XB hand-eye calibration, 4-stage trajectory execution (approach + descent + grasp + lift) |

---

*Rory Hlustik — Durham University Level 4 Dissertation, 2026*

*For questions, feedback, or issues: refer to the project repository or contact the author.*
