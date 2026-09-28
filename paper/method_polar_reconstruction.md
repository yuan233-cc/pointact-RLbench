# Method: Polarization-Aware PointACT with Interaction-Surface Reconstruction

## 3.1 Problem Formulation and Overview

Given a task instruction \(l\), the current robot state \(s_t\), and a front-view observation, our policy predicts an absolute end-effector action

\[
a_t = [p_t, r_t, g_t],
\]

where \(p_t\in\mathbb{R}^3\) is the end-effector position, \(r_t\in\mathbb{R}^3\) is its Euler-angle orientation, and \(g_t\in\{0,1\}\) denotes the gripper state. Our method augments PointACT with two complementary mechanisms. First, polarization cues are attached directly to geometrically aligned 3D points, allowing the policy to reason jointly about geometry, appearance, and material-dependent optical properties. Second, a training-only interaction-surface reconstruction branch regularizes the point encoder toward the manipulated object and its task-relevant counterpart. The reconstruction branch shares the Point Transformer encoder with the action policy but is removed during inference, and therefore introduces no test-time supervision or decoding cost.

The complete data flow is

\[
\text{RGB--D + polarization}
\rightarrow \text{filled 9-D point cloud}
\rightarrow \text{language-conditioned PointACT encoder}
\rightarrow
\begin{cases}
\text{action classification},\\
\text{interaction-surface reconstruction (training only).}
\end{cases}
\]

## 3.2 Polarization-Aligned 3D Representation

To alleviate missing geometry in the observed point cloud, we use a graphics-based depth-filling method to complete point-cloud holes before policy inference.

### Polar point features

For every retained 3D point \(i\), we form the feature

\[
x_i = [X_i,Y_i,Z_i,R_i,G_i,B_i,\rho_i,\cos(2\phi_i),\sin(2\phi_i)]\in\mathbb{R}^{9},
\]

where \(\rho_i\in[0,1]\) is the degree of linear polarization (DoLP) and \(\phi_i\) is the angle of linear polarization (AoLP) sampled at the image pixel to which the point currently projects. We use the doubled-angle encoding because AoLP is axial: \(\phi\) and \(\phi+\pi\) describe the same polarization orientation. The pair \((\cos 2\phi,\sin 2\phi)\) removes this angular discontinuity. Invalid AoLP values are represented by zeros in both angular channels.

The point geometry is obtained by unprojecting the RLBench/CoppeliaSim depth image with the archived per-frame camera calibration. We crop points to the robot workspace and voxelize them at \(1.2\) cm. RGB and polarization are sampled from the same projected pixel after geometric corruption, preserving cross-modal alignment even when a point has been displaced. At training time, we randomly retain up to \(4{,}096\) points, center the cloud by its mean coordinate, and apply the same translation to the robot state, action position, and reconstruction targets. RGB is augmented and mapped to \([-1,1]\); the polarization channels retain their physical ranges.

### Corruption-aware depth filling

To reduce holes caused by simulated missing geometry, we apply a deterministic preprocessing step before policy inference. Let \(D_s\) be the sparse depth map formed by z-buffering the corrupted point cloud. Candidate pixels are restricted to

\[
\mathcal{H}=\mathcal{P}_{\mathrm{clean}}\setminus
\mathcal{P}_{\mathrm{retained}},
\]

namely pixels represented before corruption but absent afterward. This restriction prevents ordinary empty pixels introduced by voxel downsampling from being mistaken for missing geometry. We invert depth and apply depth-dependent cross-kernel dilation, morphological closing, median filtering, iterative local propagation, and bilateral smoothing. A new 3D point is created only for a pixel in \(\mathcal H\) for which a valid depth can be estimated. Its RGB and polarization features are sampled at that same pixel. We refer to the resulting \(N\times9\) observation as the **filled9 point cloud**. This stage is an offline/image-space geometric preprocessing procedure, rather than the learned reconstruction branch introduced below.

## 3.3 Language-Conditioned Point-Action Encoder

We encode the task instruction using a frozen Qwen2.5-VL-3B backbone and linearly project its token embeddings to 512 dimensions. In the current configuration, RGB is consumed through the point features rather than through the VLM image tower. The robot state is mapped by a category-specific MLP to a 64-dimensional token. Because the policy predicts one action at a time, we initialize one learned action token and prepend the state token to the action-token sequence.

The 9-D point features are processed by a five-stage Concerto Point Transformer V3 encoder. Its channel widths are \([64,128,256,512,768]\), its block depths are \([3,3,3,12,3]\), and its attention-head counts are \([4,8,16,32,48]\). Consecutive stages use stride-2 grid pooling with a \(1\) cm grid size. Points are serialized using Z-order and Hilbert-order variants, and self-attention is evaluated within patches of at most \(1{,}024\) points.

Action tokens are inserted into every serialized point patch. Consequently, local self-attention jointly updates the point features and the action representation; action features obtained from multiple patches are averaged before the next block. At every encoder block, both point tokens and action tokens also cross-attend to the projected language tokens. This produces a bottleneck point representation \(F=\{f_i\}\) and a task-, state-, and geometry-conditioned action embedding \(q_t\).

## 3.4 Discrete Action Prediction

We use a mixture-of-experts-style classification head to predict translation, rotation, and gripper state. For translation, each encoded point coordinate \(c_i\) acts as a spatial anchor. Along each Cartesian axis \(d\in\{x,y,z\}\), we construct \(B_p=100\) candidates

\[
\hat p_{i,d,b}=c_{i,d}+\left(b-\frac{B_p}{2}\right)\Delta_p,
\qquad \Delta_p=0.01\ \mathrm{m},
\]

and score them with an MLP applied to the concatenated point and action features, \([f_i;q_t]\). The ground-truth distribution places equal probability on candidates lying within one bin width of the demonstrated coordinate. At inference, the maximum-probability point--offset pair is selected independently for each axis.

Orientation and gripper state are predicted directly from \(q_t\). Each Euler angle is discretized into \(72\) bins at \(5^\circ\) resolution, while gripper openness is modeled by a binary logit. The action loss is

\[
\mathcal{L}_{\mathrm{act}}
=\mathcal{L}_{\mathrm{pos}}
+\mathcal{L}_{\mathrm{rot}}
+\mathcal{L}_{\mathrm{grip}},
\]

where the first two terms are cross-entropy losses and the last term is binary cross entropy.

## 3.5 Training-Only Interaction-Surface Reconstruction

### Visible interaction targets

The auxiliary target is not the full scene. For each task, we define an **interaction surface** consisting of the manipulated object and its task-relevant related object(s), such as a bottle and rack or a phone and base. Visible ground-truth points are extracted from the simulator depth and object-ID buffers, voxelized at \(5\) mm, and subsampled to at most \(M=512\) points. Half of the sampling budget is reserved for the manipulated object whenever possible, and the remainder is distributed across related objects.

In addition to the target set \(Y=\{y_j\}_{j=1}^{M'}\), we construct a binary label \(m_i\) for every input point. A point is positive if its projected pixel belongs to an interaction object and its projected depth agrees with the simulator depth within \(5\) cm. These labels supervise localization of the relevant support within the observed cloud.

### Auxiliary U-Net decoder

The action policy consumes the encoder bottleneck directly. During training only, we copy the encoder feature hierarchy and pass it through a four-stage Point Transformer decoder with skip connections and channel widths \([384,256,128,128]\) from coarse to fine. Copying the hierarchy is necessary because unpooling mutates its point containers; it ensures that auxiliary decoding cannot alter the action-path activations.

For each full-resolution decoded feature \(z_i\), a shared MLP predicts a target logit \(\ell_i\) and a bounded coordinate residual:

\[
h_i=\operatorname{GELU}(W_hz_i+b_h),\qquad
\ell_i=W_mh_i+b_m,
\]

\[
\tilde c_i=c_i+0.1\tanh(W_\delta h_i+b_\delta).
\]

Thus, the branch selects and refines points on the existing observation support, with a maximum correction of \(10\) cm per coordinate. It should therefore be interpreted as interaction-surface reconstruction rather than unconstrained scene completion.

We supervise target selection with class-balanced binary cross entropy,

\[
\mathcal{L}_{\mathrm{mask}}
=\operatorname{BCEWithLogits}(\ell,m;w_+),\qquad
w_+=\operatorname{clip}\!\left(\frac{N_-}{\max(N_+,1)},1,20\right).
\]

For geometric supervision, we select the \(K=\min(512,N)\) points with the largest predicted target logits and compute a symmetric squared Chamfer loss against the visible target set:

\[
\mathcal{L}_{\mathrm{geo}}
=\frac{1}{K}\sum_{i\in\operatorname{TopK}(\ell)}
\min_j\|\tilde c_i-y_j\|_2^2
+\frac{1}{M'}\sum_j
\min_{i\in\operatorname{TopK}(\ell)}\|y_j-\tilde c_i\|_2^2.
\]

The reconstruction objective is

\[
\mathcal{L}_{\mathrm{rec}}
=\mathcal{L}_{\mathrm{geo}}
+\lambda_m\mathcal{L}_{\mathrm{mask}},
\qquad \lambda_m=0.1.
\]

Although the discrete top-\(K\) operation does not pass gradients from the Chamfer term to the selection logits, the mask loss directly trains these logits, while the Chamfer term trains the selected point features and residual regressor.

## 3.6 Joint Training and Inference

The final training objective is

\[
\mathcal{L}
=\mathcal{L}_{\mathrm{act}}
+\lambda_r\mathcal{L}_{\mathrm{rec}},
\qquad \lambda_r=2.5.
\]

We initialize the point encoder from Concerto, freeze the Qwen language/vision backbone and multimodal merger, and optimize the PointACT policy and auxiliary branch with AdamW. The current model uses a batch size of \(128\), a learning rate of \(10^{-4}\), weight decay \(10^{-3}\), and a cosine schedule.

At inference, only the language encoder, state encoder, Point Transformer encoder, and action classification head are evaluated. Neither simulator masks nor target points are required, and the auxiliary decoder and reconstruction head are not executed. The policy therefore has the same inference architecture as the encoder-only action model while retaining the task-focused geometric representation learned from auxiliary supervision.

## Implementation-Verified Configuration

This description corresponds to pointact-rlbench-polar9-v2-recon-step17728:

| Component | Setting |
|---|---|
| Policy | VLAEncDec3DWithActionClassificationModel |
| Point input | 9-D XYZ, RGB, DoLP, cos(2AoLP), sin(2AoLP) |
| Dense polar/material conditioner | Disabled |
| Action chunk | 1 |
| Point cross-attention to language | Enabled |
| Encoder channels | 64, 128, 256, 512, 768 |
| Encoder depths | 3, 3, 3, 12, 3 |
| Auxiliary decoder channels | 384, 256, 128, 128 (coarse to fine) |
| Reconstruction candidates | 512 |
| \(\lambda_r\) / \(\lambda_m\) | 2.5 / 0.1 |
| Training data | 1,000 episodes, 5,051 keyframes, 10 RLBench tasks |
