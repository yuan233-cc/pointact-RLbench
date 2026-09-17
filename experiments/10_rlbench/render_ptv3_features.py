"""Render captured PointACT PTV3 features with a shared PCA color basis."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import tyro

from ptv3_feature_viz import discover_captures, fit_global_pca, render_capture


@dataclasses.dataclass
class Args:
    inputs: list[str]
    output_dir: str = "ptv3_feature_visualizations"
    max_pca_samples: int = 50000
    seed: int = 7
    dpi: int = 150
    make_video: bool = True
    video_fps: float = 2.0


def main(args: Args) -> None:
    captures = discover_captures(args.inputs)
    if not captures:
        raise FileNotFoundError(f"No capture_*.npz files found in: {args.inputs}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pca, low, high = fit_global_pca(captures, args.max_pca_samples, args.seed)
    records = [render_capture(path, output_dir, pca, low, high, args.dpi) for path in captures]

    video_path = None
    if args.make_video:
        try:
            import imageio.v2 as imageio

            video_path = output_dir / "ptv3_features.mp4"
            with imageio.get_writer(video_path, fps=args.video_fps, codec="libx264") as writer:
                for record in records:
                    writer.append_data(imageio.imread(record["png"]))
        except (ImportError, RuntimeError, ValueError) as exc:
            print(f"Video creation skipped: {exc}")
            video_path = None

    summary = {
        "captures": len(records),
        "global_pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "global_pca_color_percentile_low": low.tolist(),
        "global_pca_color_percentile_high": high.tolist(),
        "video": str(video_path) if video_path else None,
        "frames": records,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
