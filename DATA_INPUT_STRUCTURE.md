# Input/Output Data Structures: Diffusion-CCSP & Flow Models

## Overview

Both the **Diffusion-CCSP** and **Flow Matching** models operate on **constraint graphs** representing Compositional Constraint Satisfaction Problems (CCSPs). The input is NOT a latent space — it's a **structured graph** with node features (geometry + pose) and edge features (constraint types), processed through learned embeddings.

---

## 1. Raw Input Format (From Simulation/JSON)

### Source Data Structure
The raw data comes from simulation environments (PyBullet) and is stored as JSON:

```python
{
    'container': {
        'tray_dim': [w0, l0],           # tray width, length
        'tray_pose': [x0, y0, z0]       # tray position
    },
    'placements': [
        {
            'name': 'obj_1',
            'extent': [w, l, h],        # object dimensions
            'place_pose': [[x, y, z], quat],  # target position
            'grasp_pose': ...,          # for robot tasks
            'grasp_id': ...,            # grasp configuration
            'scale': ...                # size scaling
        },
        ...
    ],
    'supports': [(i, j), ...],          # object i supported by object j
    'constraints': [                    # qualitative constraints
        ('left-of', i, j),
        ('close-to', i, j),
        ...
    ]
}
```

---

## 2. Transformed Input: PyTorch Geometric Graph

### Data Transform Pipeline (`networks/data_transforms.py`)

The raw JSON is converted to a **PyTorch Geometric `Data` object** via `pre_transform()`:

```python
Data(
    x = [n_nodes, n_features],           # Node features (geometry + pose)
    edge_index = [2, n_edges],           # Constraint graph edges
    edge_attr = [n_edges],               # Constraint type indices (0, 1, 2, ...)
    mask = [n_nodes],                    # 1 = clamped (fixed), 0 = free (to be solved)
    conditioned_variables = [n_clamped], # Indices of fixed nodes (e.g., tray)
    shuffled = [n_nodes],                # Permutation for data augmentation
    x_extract = [n_nodes],               # Graph ID (which graph this node belongs to)
    edge_extract = [n_edges],            # Graph ID for edges
    world_dims = (w_tray, l_tray),       # Normalization constants
    original_x = ...,                    # Raw features (before normalization)
    original_y = ...                     # Solution labels
)
```

---

## 3. Node Feature Structure (x tensor)

### Feature Layout Per Object

**Each object** (node in the graph) has a concatenated feature vector:

```
x[i] = [geom_features] + [pose_features]
```

The exact dimensions depend on the **input mode**:

### A. Qualitative Domain (2D boxes)
```python
# Total: 6 features per node
geom = [w/w0, l/l0]                    # 2 features: normalized width, length
pose = [x, y, cs, sn]                  # 4 features: position + orientation (cos/sin)

# Example: [0.15, 0.10, 1.2, 0.8, 0.707, 0.707]
#           w     l     x     y    cosθ   sinθ
```

**Normalization:**
- `w, l` normalized by tray dimensions: `w / w_tray`, `l / l_tray`
- `x, y` normalized to `[0, 2]` range: `x / (w_tray / 2)` (tray center at `[1, 1]`)
- Orientation encoded as `[cos(θ), sin(θ)]` for continuity

### B. Puzzle Domain (2D triangles)
```python
# Total: 7 features per node
geom = [l, x3, y3]                     # 3 features: side length + vertex offset
pose = [x1, y1, r1]                    # 3 features: centroid + rotation (normalized)

# Or with sin/cos encoding: 8 features
geom = [w, l, 0]                       # 3 features (padded)
pose = [x, y, cs, sn]                  # 4 features
```

### C. Stability Domain (3D boxes)
```python
# Total: 6 features per node
geom = [w/w0, l/l0]                    # 2 features: normalized dimensions
pose = [x, y, sn, cs]                  # 4 features: 2D projection + orientation

# Note: z-coordinate ignored in flat stability task
```

### D. Robot Domain (3D packing with grasps)
```python
# Total: 21+ features per node
geom = [w/w0, l/l0, h/h0,              # 3: normalized object dimensions
        w0, l0, h0, x0, y0,            # 5: tray dimensions and position
        mobility_id, scale]            # 2: object metadata

pose = [x, y, z, sn, cs]               # 5: 3D position + orientation

grasp = [grasp_side_onehot,            # 6: grasp direction (x+, x-, y+, y-, z+, id)
         grasp_id,                     # 1: grasp configuration
         pick_pose]                    # 6+7: pick pose features
```

### E. CNN Image Domain (for GeomEncoderImage)
```python
# Instead of geometry features, use 64×64 images
geom_image = [64, 64]                  # Binary mask of object shape
pose = [x, y, cs, sn]                  # 4 features as usual
```

---

## 4. Constraint Graph Structure

### Edge Index and Attributes

```python
edge_index = [
    [i1, j1],  # Edge 1: objects i1 and j1
    [i2, j2],  # Edge 2: objects i2 and j2
    ...
]

edge_attr = [
    c1,  # Constraint type index for edge 1 (e.g., 0 = 'in', 2 = 'cfree')
    c2,  # Constraint type index for edge 2
    ...
]
```

### Constraint Type Mappings

**Qualitative Domain (13 types):**
```python
qualitative_constraints = [
    'in',          # 0: Object in tray
    'center-in',   # 1: Object near tray center
    'left-in',     # 2: Object in left region
    'right-in',    # 3: Object in right region
    'top-in',      # 4: Object in top region
    'bottom-in',   # 5: Object in bottom region
    'cfree',       # 6: Collision-free
    'left-of',     # 7: Left of another object
    'top-of',      # 8: Top of another object
    'close-to',    # 9: Close to another object
    'away-from',   # 10: Far from another object
    'h-aligned',   # 11: Horizontally aligned
    'v-aligned',   # 12: Vertically aligned
]
```

**Puzzle Domain (2 types):**
```python
puzzle_constraints = ['in', 'cfree']
```

**Stability Domain (3 types):**
```python
stability_constraints = ['within', 'supportedby', 'cfree']
```

**Robot Domain (2 + 13 types):**
```python
robot_constraints = ['gin', 'gfree']  # Grasp in, grasp free
robot_qualitative_constraints = robot_constraints + qualitative_constraints
```

---

## 5. Model Input Processing

### A. Diffusion Model (ConstraintDiffuser)

**Forward Signature:**
```python
def forward(self, poses_in, batch, t, **kwargs):
    """
    Args:
        poses_in: [n_nodes, pose_dim]    # Noisy poses to denoise
        batch:    DataBatch               # PyG batch with x, edge_index, edge_attr
        t:        [1]                     # Diffusion timestep (0-999)
    
    Returns:
        all_poses_out: [n_nodes, pose_dim]  # Denoised poses (or gradients if EBM)
    """
```

**Processing Pipeline:**

```
1. Extract geometry from batch.x
   └─ geoms_in = x[:, :geom_dim]  # e.g., [n_nodes, 2] for qualitative

2. Encode geometry
   └─ geoms_emb = geom_encoder(geoms_in)  # [n_nodes, hidden_dim]
      # geom_encoder: Linear(2, 128) → SiLU → Linear(128, 256) → SiLU

3. Encode noisy poses
   └─ poses_emb = pose_encoder(poses_in)  # [n_nodes, hidden_dim]
      # pose_encoder: Linear(4, 128) → SiLU → Linear(128, 256) → SiLU

4. Encode timestep
   └─ time_emb = time_mlp(t)  # [1, hidden_dim]
      # time_mlp: SinusoidalPosEmb → Linear → Mish → Linear

5. For each constraint type (loop over edge types):
   └─ For constraint type i:
      a. Find edges of this type:
         edges = torch.where(edge_attr == i)
      
      b. Get constraint inputs:
         input_dict = {
             'geoms_emb': geoms_emb[edges],      # [n_edges, 2, hidden]
             'poses_emb': poses_emb[edges],      # [n_edges, 2, hidden]
             'time_embedding': time_emb,          # [n_edges, hidden]
             'args': edges                        # [n_edges, 2] node indices
         }
      
      c. Concatenate for MLP:
         inputs = cat([geom_i, geom_j, pose_i, pose_j, time])
         # Shape: [n_edges, 5 * hidden_dim]
      
      d. Pass through constraint MLP:
         outputs = mlps[i](inputs)  # [n_edges, 2 * hidden_dim]
         outs_1, outs_2 = split(outputs)  # Each: [n_edges, hidden_dim]
      
      e. Decode to pose updates:
         outputs = pose_decoder(cat([outs_1, outs_2]))
         # Shape: [n_edges, 2, pose_dim]

6. Aggregate via scatter-add:
   └─ all_poses_out = scatter_add(outputs, node_indices)
   └─ all_counts = count how many constraints touch each node
   └─ if normalize: all_poses_out /= sqrt(all_counts)

7. Mask clamped nodes:
   └─ all_poses_out[mask] = batch.x[mask, pose_dims]

8. Return result
```

**EBM Mode (Energy-Based Model):**
```python
if tag == 'EBM':
    # Compute energy instead of poses
    energy = sum(||outputs - poses_in||² for each constraint)
    gradients = autograd.grad(energy, poses_in)
    return gradients, energy
```

---

### B. Flow Matching Model (FlowMatchingCCSP)

**Forward Signature:**
```python
def forward(self, x_t, batch, t):
    """
    Args:
        x_t:    [n_nodes, pose_dim]    # Current poses at time t
        batch:  DataBatch              # PyG batch
        t:      float ∈ [0, 1]         # Continuous time
    
    Returns:
        v:      [n_nodes, pose_dim]    # Predicted velocity
    """
```

**Processing Pipeline (similar to diffusion but simpler):**

```
1. Encode geometry (same as diffusion)
   └─ geoms_emb = geom_encoder(x[:, :geom_dim])  # [n_nodes, hidden]

2. Encode current poses
   └─ pose_emb = pose_encoder(x_t)  # [n_nodes, hidden]

3. Encode timestep
   └─ t_tensor = int(t * 999)
   └─ time_emb = time_mlp(t_tensor)  # [1, hidden]

4. For each constraint type:
   └─ inputs = cat([geom_i, geom_j, pose_i, pose_j, time_emb])
   └─ outputs = constraint_mlps[i](inputs)
   └─ v_i, v_j = pose_decoder(split(outputs))

5. Aggregate (3 strategies):
   
   a) Sum (scatter-add + sqrt-norm):
      all_v = scatter_add(v, nodes) / sqrt(count)
   
   b) Attention (softmax-weighted):
      # Each edge predicts velocity + urgency logit
      logits = pose_decoder(...)[:, pose_dim:]  # extra dim
      weights = softmax(logits per node)
      all_v = sum(weights * v)
   
   c) Maxpool (winner-take-all):
      # For each node+dimension, pick max |v| across edges
      all_v = max_by_abs(v across incoming edges)

6. Mask clamped nodes
   └─ all_v[mask] = 0

7. Return velocity
```

**Key Difference from Diffusion:**
- Flow predicts **velocity** directly (no noise prediction)
- Flow uses continuous time `t ∈ [0, 1]` (discrete `t ∈ [0, 999]` for diffusion)
- Flow output is used in ODE: `dx/dt = v(x, t)`

---

## 6. Training Data Flow

### Training Loop (Diffusion)

```python
# 1. Load batch from dataset
batch = next(train_loader)  # DataBatch

# 2. Extract clean poses from batch.x
pose_begin = dims[-1][1]  # e.g., 2 (after geom features)
pose_end = dims[-1][2]    # e.g., 6
x_clean = batch.x[:, pose_begin:pose_end]  # [n_nodes, 4]

# 3. Sample timestep and noise
t = randint(0, num_timesteps)  # e.g., tensor([523])
noise = torch.randn_like(x_clean)
noise[mask] = 0  # Don't add noise to clamped nodes

# 4. Create noisy poses
x_noisy = sqrt(alpha_t) * x_clean + sqrt(1 - alpha_t) * noise

# 5. Forward pass (predict noise)
predicted_noise = denoise_fn(x_noisy, batch, t)

# 6. Compute loss (MSE on free nodes only)
free = ~mask
loss = mse_loss(predicted_noise[free], noise[free])

# 7. Backprop
loss.backward()
optimizer.step()
```

### Training Loop (Flow Matching)

```python
# 1. Load batch
batch = next(train_loader)

# 2. Extract clean poses
x_1 = batch.x[:, pose_begin:pose_end].clone()  # target poses

# 3. Sample random noise and time
x_0 = torch.randn_like(x_1)  # source noise
t = rand(1).item()  # e.g., 0.37

# 4. Create interpolated poses (OT-CFM)
x_t = (1 - t) * x_0 + t * x_1
x_t[mask] = x_1[mask]  # Clamped nodes stay at clean poses

# 5. Compute velocity target
u_t = x_1 - x_0  # OT conditional velocity

# 6. Forward pass (predict velocity)
v_pred = flow_model(x_t, batch, t)

# 7. Compute loss (MSE on free nodes)
free = ~mask
loss = mse_loss(v_pred[free], u_t[free])

# 8. Backprop
loss.backward()
optimizer.step()
```

---

## 7. Inference/Sampling

### Diffusion Sampling (Reverse Process)

```python
# 1. Start from noise
x_t = torch.randn(n_nodes, pose_dim) * 0.5
x_t[mask] = x_clean[mask]  # Clamp fixed nodes

# 2. Reverse diffusion loop
for t in reversed(range(num_timesteps)):  # 999 → 0
    # Predict noise
    noise_pred = denoise_fn(x_t, batch, t=tensor([t]))
    
    # Compute posterior mean/variance
    x_0_pred = predict_start_from_noise(x_t, t, noise_pred)
    x_t_mean, x_t_var = q_posterior(x_0_pred, x_t, t)
    
    # Sample next state
    if t > 0:
        noise = torch.randn_like(x_t)
        x_t = x_t_mean + sqrt(x_t_var) * noise
    else:
        x_t = x_t_mean  # No noise at t=0
    
    # Clamp fixed nodes
    x_t[mask] = x_clean[mask]
    
    # EBM correction (if enabled)
    if EBM and t % ebm_per_steps == 0:
        gradients, energy = denoise_fn(x_t, batch, t, tag='EBM')
        # ULA step: x = x - lr * gradient + noise
        x_t = x_t - lr * gradients + sqrt(2*lr) * noise

# 3. Return final poses
return x_t  # [n_nodes, pose_dim]
```

### Flow Matching Sampling (ODE Integration)

```python
# 1. Start from noise
x_t = torch.randn(n_nodes, pose_dim)
x_t[mask] = x_clean[mask]

# 2. Euler ODE integration
n_steps = 20  # or more
dt = 1.0 / n_steps

for step in range(n_steps):
    t = step * dt  # 0.0, 0.05, 0.10, ..., 0.95
    
    # Predict velocity
    v = flow_model(x_t, batch, t)
    
    # Euler step
    x_t = x_t + v * dt
    
    # Clamp fixed nodes
    x_t[mask] = x_clean[mask]
    
    # Optional: tray clamping
    x_t = clamp_to_tray(x_t, geoms, mask)

# 3. Return final poses
return x_t
```

---

## 8. Output Format

### Model Outputs

Both models output **pose tensors** in the same format:

```python
output = [n_nodes, pose_dim]  # e.g., [8, 4] for 8 objects in qualitative domain

# For qualitative:
# output[i] = [x, y, cos(θ), sin(θ)]  for object i

# To convert back to world coordinates:
x_world = output[i, 0] * (w_tray / 2)
y_world = output[i, 1] * (l_tray / 2)
theta = atan2(output[i, 3], output[i, 2])  # from cos/sin
```

### Evaluation Output

```python
# Check constraints on output poses
all_satisfied, per_constraint = check_constraints(poses, batch, constraint_types)

per_constraint = {
    edge_idx: {
        'type': 'cfree',
        'h_val': 0.15,          # Barrier value (positive = satisfied)
        'satisfied': True
    },
    ...
}

# Success metrics
trial_success_rate = n_successful_trials / n_total_trials
scene_top1 = n_scenes_solved_on_first_try / n_total_scenes
scene_topk = n_scenes_solved_in_k_tries / n_total_scenes
```

---

## 9. Data Normalization Constants

### Qualitative Domain
```python
# Tray dimensions (example)
w_tray = 2.0  # world units
l_tray = 2.0

# Normalization
w_norm = w_obj / w_tray           # [0, 1]
l_norm = l_obj / l_tray           # [0, 1]
x_norm = x_obj / (w_tray / 2)     # [0, 2], tray center at 1.0
y_norm = y_obj / (l_tray / 2)     # [0, 2]

# Denormalization
x_obj = x_norm * (w_tray / 2)
y_obj = y_norm * (l_tray / 2)
```

### Robot Domain
```python
# Additional normalization for robot features
geom_norm = {
    'w': w_obj / w_tray,
    'l': l_obj / l_tray,
    'h': h_obj / h_tray,
    'x0': x_tray / world_scale,
    'y0': y_tray / world_scale,
}

pose_norm = {
    'x': (x_obj - x_tray) / (w_tray / 2),
    'y': (y_obj - y_tray) / (l_tray / 2),
    'z': z_obj / h_tray,
}
```

---

## 10. Dimension Summary Table

| Domain | Geom Dim | Pose Dim | Total/Node | Constraint Types |
|--------|----------|----------|------------|------------------|
| **Qualitative** | 2 | 4 | 6 | 13 |
| **Puzzle** | 3 | 3-4 | 6-7 | 2 |
| **Stability** | 2 | 4 | 6 | 3 |
| **Robot** | 10 | 5+ | 15+ | 15 |
| **Robot+Qualitative** | 10 | 5+ | 15+ | 15 |

**Note:** The `dims` tuple encodes this structure:
```python
# For qualitative:
dims = ((2, 0, 2),      # Geometry: 2 features, indices [0:2]
        (4, 2, 6))      # Pose: 4 features, indices [2:6]

# For robot:
dims = ((8, 0, 8),      # Geometry: 8 features, indices [0:8]
        (5, 10, 15),    # Pose: 5 features, indices [10:15]
        (5, 16, 21))    # Extra pose features (e.g., pick pose)
```

---

## 11. Key Architectural Insights

### 1. **No Latent Space Bottleneck**
Unlike VAEs or autoencoders, these models do **NOT** compress inputs into a latent bottleneck. Instead:
- Geometry is encoded to `hidden_dim` (e.g., 256)
- Pose is encoded to `hidden_dim`
- These remain at `hidden_dim` throughout processing
- Only the final decoder maps back to `pose_dim` (e.g., 4)

### 2. **Per-Constraint Processing**
Each constraint type has its **own dedicated MLP**:
- Allows specialization per constraint type
- Enables compositional generalization to unseen constraint combinations
- Aggregation via scatter-add enables variable-sized graphs

### 3. **Equivariance to Node Ordering**
The scatter-add aggregation ensures:
- Output is invariant to node reordering
- Works with variable numbers of objects
- Generalizes to unseen graph structures

### 4. **Time Conditioning**
- **Diffusion:** Discrete timesteps `t ∈ [0, 999]` via sinusoidal embedding
- **Flow:** Continuous time `t ∈ [0, 1]` via same embedding (scaled to [0, 999])

### 5. **Mask Mechanism**
The `mask` tensor distinguishes:
- `mask[i] = 1`: Clamped node (e.g., tray), excluded from loss, fixed during sampling
- `mask[i] = 0`: Free node, included in loss, updated during sampling

---

## 12. Example: Complete Forward Pass

### Qualitative Domain, 3 Objects (1 tray + 2 boxes)

```python
# Input JSON → Graph
data = {
    'container': {'tray_dim': [10.0, 10.0]},
    'placements': [
        {'extent': [1.5, 1.5, 0.5], 'place_pose': [[3.0, 4.0, 0], quat(0)]},
        {'extent': [1.0, 2.0, 0.5], 'place_pose': [[7.0, 6.0, 0], quat(π/4)]}
    ],
    'constraints': [('cfree', 1, 2)]
}

# After pre_transform
batch = Data(
    x = [
        [0, 1.0, 1.0, 0, 0, 0, 0],        # Tray (geom only, mask=1)
        [1, 0.15, 0.15, 0.6, 0.8, 1.0, 0], # Box 1 (mask=0)
        [1, 0.10, 0.20, 1.4, 1.2, 0.707, 0.707]  # Box 2 (mask=0)
    ],
    edge_index = [[0, 1], [0, 2], [1, 2]],  # in, in, cfree
    edge_attr = [0, 0, 6],                   # constraint type indices
    mask = [1, 0, 0]                         # tray is fixed
)

# During training (diffusion)
t = 500
x_clean = batch.x[:, 2:6]  # [[0,0,0,0], [0.6,0.8,1,0], [1.4,1.2,0.707,0.707]]
x_noisy = q_sample(x_clean, mask, t)  # Add noise to free nodes
predicted_noise = denoise_fn(x_noisy, batch, t=500)
loss = mse_loss(predicted_noise[1:], x_noisy[1:])  # Only free nodes

# During inference
x_t = torch.randn(3, 4)  # Start from noise
x_t[0] = batch.x[0, 2:6]  # Clamp tray pose

for t in reversed(range(1000)):
    noise_pred = denoise_fn(x_t, batch, t)
    x_t = p_sample(x_t, noise_pred, t)
    x_t[0] = batch.x[0, 2:6]  # Keep tray fixed

# Output poses in normalized coords
poses = x_t  # [[0,0,0,0], [x1,y1,cs1,sn1], [x2,y2,cs2,sn2]]

# Denormalize
x_world = poses[:, 0] * 5.0  # * (w_tray / 2)
y_world = poses[:, 1] * 5.0  # * (l_tray / 2)
theta = torch.atan2(poses[:, 3], poses[:, 2])
```

---

## Summary

**Input:** Structured constraint graph with normalized geometry + pose features  
**Not:** Latent codes, embeddings from pretrained models, or learned representations  

**Processing:** Per-constraint MLPs with learned encodings, aggregated via scatter operations  
**Output:** Pose tensors in normalized coordinates, requiring denormalization for world use  

**Key:** The compositional structure comes from the **constraint graph topology**, not from latent space manipulations.
