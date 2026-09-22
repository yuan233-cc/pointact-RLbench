# Training-only interaction surface reconstruction

This experiment keeps the original 9-channel filled point cloud and action
classifier. During training, PTv3's existing U-Net decoder runs on a copy of
the encoder's point containers. A pointwise head predicts target membership
and a bounded XYZ correction. Only the encoder features are shared with the
action path; the reconstruction output is never used by the action head.
`sample_actions()` does not execute the auxiliary decoder.

The visible interaction labels are stored in `interaction_visible_gt`, an LMDB keyed by
`episode_index-frame_index`. For each same-state native-rendered frame, the
exporter reads world-coordinate `point_cloud`, camera depth, and instance
`object_mask`. Instance handles come from the frame snapshot's mesh names.
`interaction_target_meshes.json` lists the manipulated object and task-related
objects. Their visible masks are merged into one binary foreground mask. The
original manipulated-only labels remain in `target_visible_gt` and are not
overwritten.

The mapping is fixed by task, not inferred from the wording of a particular
prompt. For example, `phone_on_base` includes the handset and its base,
`sweep_to_dustpan` includes the broom, dirt, and dustpan, and `water_plants`
includes the watering can, plant, and pot. The dataset's
`meta/tasks.jsonl` stores alternative English instructions separated by
`<br>`; the training loader randomly chooses one alternative each time it
reads a sample. `interaction_mask_audit_10tasks.png` overlays one representative
native-rendered manipulated/related mask pair per task for visual review.

The interaction geometry is reduced with a 5 mm voxel grid and capped at 512
points. Half of the budget is reserved for the manipulated object when visible;
the rest is shared fairly across related object groups. Unused capacity is
reassigned, so a large fridge body does not erase the small door from the
sampled GT. The frame task includes the hanger but omits the placement table:
the native-rendered table has no depth-consistent corresponding input points
in sampled filled9 frames. The prediction top-K cap is 512 for this mode.
The per-input-point target mask is a **label**, computed using each filled
point's current projection and rendered target depth. Neither the clean target
points nor the target mask enter the auxiliary decoder as inputs. The decoder
selects up to 512 points using its own predicted target logits. The training
loss is `action_loss + target_reconstruction_weight * (symmetric_chamfer +
target_mask_loss_weight * weighted_mask_BCE)`. This experiment wrapper uses
`target_reconstruction_weight=2.5` and `target_mask_loss_weight=0.1`, so its
default total is `L_action + 2.5 * (L_chamfer + 0.1 * L_mask)`. With the
observed raw reconstruction loss near 0.186 and a converged action loss near
5, this makes the auxiliary contribution about 9.3% of the action loss. The outer
weight can be overridden with `TARGET_RECONSTRUCTION_WEIGHT`. Frames with no
visible target contribute mask loss but skip Chamfer loss.

Run from the repository root:

```bash
../.conda/envs/pointact/bin/python experiments/10_rlbench/export_visible_target_gt.py \
  --task-map experiments/10_rlbench/interaction_target_meshes.json \
  --output robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps_polar_incomplete9_v1/interaction_visible_gt
bash experiments/10_rlbench/train_10task_polar_filled9_target_reconstruction.sh
```

The new sidecar is stored alongside the original point clouds in the local
ten-task dataset (5051 frames); it is not a new LeRobot dataset.
The experiment wrapper defaults to batch size 8 because the training decoder
adds activations; `PER_DEVICE_BATCH_SIZE` overrides it. The original filled9
script keeps reconstruction and direct point-to-VLM cross-attention disabled by
default. The reconstruction wrapper enables direct point cross-attention in
PTv3's encoder and training-only decoder. Each point cross-attention block
reads the same final VLM hidden-state sequence as context; it does not pair
individual VLM layers with PTv3 layers. The data config retains
`video_key_ids_for_vlm: []`, so that context contains task text but no image
tokens, while PTv3 receives XYZ, RGB, and polar features from the point cloud.
