# Frozen-teacher geometry cache and throughput tuning

The observation-defined PointACT operation workspace, NaN holes, camera
transforms, native polar preprocessing and the two losses are unchanged.
`cache_workspace_geometry.py` reads every calibrated frame once, computes only
frozen eval teacher normals/features in BF16, and stores FP32 output arrays.
It never reads clean depth or GT normals. No adapter/student feature is cached.

The uncompressed column cache is complete only when `manifest.json` exists.
Its checkpoint SHA256, preprocessing code signatures, backbone, frame keys and
operation-space bounds are verified by the loader. Each split is normally
loaded into host RAM once; workers inherit the immutable arrays and perform
only point sampling/collation. Point/pixel associations and NaN values survive.
All available workspace points are cached before the usual random <=4096
sampling, so cache creation does not freeze a subsample of a larger point set.
The frozen teacher can stay on CPU during student training.

`benchmark_workspace_geometry.py` completes real forward/backward/optimizer
steps for increasing physical batches. Choose the highest measured samples/s
with at least 8% CUDA reserved-memory headroom, not merely the largest batch.
Keep each probe output unique. Teacher caches are invalid when inputs,
checkpoint, geometric preprocessing or teacher code change, and must not be
used with online polar/RGB augmentation or teacher fine-tuning.

`monitor_geometry_utilization.py` collects at least 300 seconds of active
training after warm-up. It excludes initialization, cache building, validation
and checkpoint saves using the trainer's atomic phase marker. Data waiting
during training is included. Below 50% core or 15% memory occupancy triggers
normal supervisor termination and dedicated-job release; no automatic restart.
Fresh training uses measured full duration plus 40 minutes, capped at 24 hours.
