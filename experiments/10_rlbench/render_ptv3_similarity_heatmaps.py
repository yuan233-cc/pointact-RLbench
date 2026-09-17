"""Render intra-scene cosine-similarity heatmaps for final PTV3 points."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro

from ptv3_feature_viz import discover_captures, set_axes_equal
from render_ptv3_geometry_analysis import (
    CaptureData,
    load_capture,
    normalized_features,
    select_action_anchor,
)


@dataclasses.dataclass
class Args:
    inputs: list[str]
    output_dir: str = "ptv3_similarity_heatmaps"
    dpi: int = 180


def configure_axis(ax, xyz: np.ndarray) -> None:
    ax.view_init(elev=35, azim=35)
    ax.set_proj_type("ortho")
    set_axes_equal(ax, xyz)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def render_similarity_heatmap(
    capture: CaptureData,
    query_anchor: int,
    output_dir: Path,
    dpi: int,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    features = normalized_features(capture)
    similarities = features @ features[query_anchor]
    propagated = similarities[capture.input_to_final]
    descendant_counts = np.bincount(
        capture.input_to_final, minlength=len(capture.final_coordinates)
    )

    # Use the actual range so structure remains visible even when all learned
    # features have positive cosine similarity. Exact values remain on the bar.
    color_min = min(float(similarities.min()), 0.0)
    norm = Normalize(vmin=color_min, vmax=1.0)
    cmap = plt.get_cmap("turbo")

    fig = plt.figure(figsize=(18, 6.6))

    rgb_ax = fig.add_subplot(1, 3, 1, projection="3d")
    xyz = capture.input_coordinates
    rgb_ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=capture.input_rgb, s=2, linewidths=0,
    )
    rgb_ax.scatter(
        *capture.final_coordinates[query_anchor], marker="*", s=220,
        c="white", edgecolors="black", linewidths=1.2,
    )
    configure_axis(rgb_ax, xyz)
    rgb_ax.set_title("Input RGB context\nwhite star = query anchor")

    final_ax = fig.add_subplot(1, 3, 2, projection="3d")
    final_xyz = capture.final_coordinates
    final_ax.scatter(
        final_xyz[:, 0], final_xyz[:, 1], final_xyz[:, 2],
        c=similarities, cmap=cmap, norm=norm, s=65,
        edgecolors="black", linewidths=0.25,
    )
    final_ax.scatter(
        *final_xyz[query_anchor], marker="*", s=260,
        c="white", edgecolors="black", linewidths=1.4,
    )
    configure_axis(final_ax, xyz)
    final_ax.set_title(
        f"All {len(final_xyz)} final PTV3 points\n"
        "color = 768-D cosine similarity"
    )

    input_ax = fig.add_subplot(1, 3, 3, projection="3d")
    scatter = input_ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=propagated, cmap=cmap, norm=norm, s=3, linewidths=0,
    )
    input_ax.scatter(
        *final_xyz[query_anchor], marker="*", s=220,
        c="white", edgecolors="black", linewidths=1.2,
    )
    configure_axis(input_ax, xyz)
    input_ax.set_title(
        "Similarity propagated to input points\n"
        "using exact input_to_final pooling map"
    )

    colorbar = fig.colorbar(
        scatter,
        ax=[final_ax, input_ax],
        orientation="horizontal",
        fraction=0.055,
        pad=0.10,
        aspect=45,
    )
    colorbar.set_label("Cosine similarity to query final feature")

    ranking = np.argsort(similarities)[::-1]
    top_text = " | ".join(
        f"a{int(anchor)}={float(similarities[anchor]):.3f}"
        for anchor in ranking[:8]
    )
    fig.suptitle(
        f"Intra-scene final-feature similarity | {capture.instruction}\n"
        f"{capture.path.stem}, query anchor {query_anchor}, "
        f"{int(descendant_counts[query_anchor])} input descendants\n"
        f"Top anchors: {top_text}",
        fontsize=12,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.015, right=0.985, bottom=0.15, top=0.78, wspace=0.02
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / (
        f"{capture.path.stem}_query_anchor_{query_anchor:03d}_similarity_heatmap.png"
    )
    values_path = output_dir / (
        f"{capture.path.stem}_query_anchor_{query_anchor:03d}_similarities.json"
    )
    fig.savefig(image_path, dpi=dpi)
    plt.close(fig)

    values = [
        {
            "anchor": int(anchor),
            "cosine_similarity": float(similarities[anchor]),
            "input_descendants": int(descendant_counts[anchor]),
        }
        for anchor in ranking
    ]
    with values_path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)

    return {
        "capture": str(capture.path),
        "instruction": capture.instruction,
        "query_anchor": query_anchor,
        "query_input_descendants": int(descendant_counts[query_anchor]),
        "num_final_points": len(final_xyz),
        "similarity_min": float(similarities.min()),
        "similarity_median": float(np.median(similarities)),
        "image": str(image_path),
        "values": str(values_path),
    }


def main(args: Args) -> None:
    capture_paths = discover_captures(args.inputs)
    if not capture_paths:
        raise FileNotFoundError(f"No capture_*.npz files found in: {args.inputs}")

    output_dir = Path(args.output_dir)
    records = []
    for path in capture_paths:
        capture = load_capture(path)
        query_anchor = select_action_anchor(capture)
        records.append(
            render_similarity_heatmap(capture, query_anchor, output_dir, args.dpi)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": (
                    "cosine similarity from one final 768-D PTV3 feature to all "
                    "final features in the same capture"
                ),
                "captures": records,
            },
            handle,
            indent=2,
        )
    print(json.dumps({"rendered": len(records), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
