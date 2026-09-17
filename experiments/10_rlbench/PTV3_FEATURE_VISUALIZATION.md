# PTV3 feature visualization for RLBench

This add-on visualizes the point features used by the PointACT action head. It
does not modify the model, processor, inference service, or RLBench client.
Instead, a separate server registers a PyTorch forward hook on
`model.ptv3_model` and a second hook on `model.action_head`.

The current RLBench checkpoint has `ptv3_enc_mode=true`. Its captured output is
therefore the final encoder representation: 768-dimensional features on the
deepest, spatially downsampled point set.

## Quick run

From the repository root, on the same GPU machine and display setup used for
the normal RLBench evaluation:

```bash
bash experiments/10_rlbench/eval_low_success_ptv3_viz.sh
```

Defaults run one episode each for:

- `sweep_to_dustpan+0` (previous success rate: 40%)
- `take_frame_off_hanger+0` (previous success rate: 50%)
- `water_plants+0` (previous success rate: 30%)

By default, every third policy request is captured across the three tasks, and
the model server is loaded only once. Useful overrides include:

```bash
NUM_EPISODES=2 CAPTURE_EVERY=2 MAX_CAPTURES=40 OUTPUT_DIR=/path/to/output \
  bash experiments/10_rlbench/eval_low_success_ptv3_viz.sh
```

## Outputs

- `captures/capture_*.npz`: centered coordinates, float16 PTV3 features,
  action-token features, action-head position probability, predicted position,
  offsets, and input point/RGB arrays.
- `captures/manifest.jsonl`: instruction and shape metadata.
- `rendered/*.png`: input RGB, dense global-PCA feature color, dense action-head
  position probability, and the actual sparse final encoder points.
- `rendered/captures_capture_*.ply`: full input cloud colored by the nearest
  final PTV3 feature's global-PCA color.
- `rendered/*_action_relevance.ply`: full input cloud colored by action-head
  position probability.
- `rendered/*_sparse.ply`: the unmodified final PTV3 encoder points.
- `rendered/ptv3_features.mp4`: chronological visualization frames.
- `rendered/summary.json`: PCA variance and an index of all artifacts.
- `geometry_analysis/pooling/*_pooling_partition.png`: exact final-anchor
  partition obtained by composing all four `pooling_inverse` mappings.
- `geometry_analysis/pooling/*_ancestry.png`: stage0 through stage4 ancestry
  for the final anchor with maximum action-position probability.
- `geometry_analysis/retrieval/query_*.png`: cross-task cosine retrieval of
  final 768-D features. Highlighted input regions are the exact pooling
  descendants of each retrieved anchor.
- `geometry_analysis/summary.json`: machine-readable pooling and retrieval
  results, including similarities and descendant counts.

The renderer fits one PCA basis across every supplied task and frame. Colors
are consequently comparable across images, unlike fitting an independent PCA
for every point cloud. Because this checkpoint is encoder-only and reduces the
cloud to a few dozen final points, display colors are propagated to each input
point from its nearest final encoder point. The sparse PLY and lower-right plot
make that distinction explicit.

The geometry-analysis renderer is stricter than the dense display renderer:
it never uses nearest-neighbor propagation for receptive fields. It composes
the mappings recorded by the four real `GridPoolingWithAction` modules. For
retrieval, features are L2-normalized and ranked by cosine similarity; matches
are restricted to different task instructions, with at most one best anchor
per captured frame.

## Render existing captures again

```bash
python experiments/10_rlbench/render_ptv3_features.py \
  --args.inputs PTV3_feature_visualizations_low_success/captures \
  --args.output-dir PTV3_feature_visualizations_low_success/rendered
```

Geometry analysis requires captures made by the updated server:

```bash
python experiments/10_rlbench/render_ptv3_geometry_analysis.py \
  --args.inputs PTV3_feature_visualizations_low_success/captures \
  --args.output-dir PTV3_feature_visualizations_low_success/geometry_analysis
```
