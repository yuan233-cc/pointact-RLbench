"""Render exact PTV3 pooling ancestry and cross-scene feature retrieval."""

from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tyro

from ptv3_feature_viz import (
    discover_captures,
    load_manifest_record,
    normalize_input_rgb,
    set_axes_equal,
    write_ply,
)


@dataclasses.dataclass
class Args:
    inputs: list[str]
    output_dir: str = "ptv3_geometry_analysis"
    top_k: int = 5
    dpi: int = 150


@dataclasses.dataclass
class CaptureData:
    path: Path
    instruction: str
    arrays: dict[str, np.ndarray]

    @property
    def num_stages(self) -> int:
        return int(self.arrays["num_pooling_stages"])

    @property
    def final_features(self) -> np.ndarray:
        return self.arrays["features"].astype(np.float32)

    @property
    def final_coordinates(self) -> np.ndarray:
        return self.arrays["coordinates"].astype(np.float32)

    @property
    def input_coordinates(self) -> np.ndarray:
        return self.arrays["input_coordinates"].astype(np.float32)

    @property
    def input_rgb(self) -> np.ndarray:
        return normalize_input_rgb(self.arrays["input_rgb"])

    @property
    def input_to_final(self) -> np.ndarray:
        return self.arrays["input_to_final"].astype(np.int64)


def load_capture(path: Path) -> CaptureData:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    required = {
        "input_coordinates",
        "input_rgb",
        "features",
        "coordinates",
        "input_to_final",
        "num_pooling_stages",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(
            f"{path} lacks exact pooling data {missing}; recapture it with the updated server"
        )
    record = load_manifest_record(path)
    return CaptureData(path, record.get("instruction", "unknown"), arrays)


def stage_coordinates(capture: CaptureData, stage: int) -> np.ndarray:
    if stage == capture.num_stages:
        return capture.final_coordinates
    return capture.arrays[f"stage{stage}_coordinates"].astype(np.float32)


def stage_to_final(capture: CaptureData, stage: int) -> np.ndarray:
    coordinates = stage_coordinates(capture, stage)
    assignment = np.arange(len(coordinates), dtype=np.int64)
    for input_stage in range(stage, capture.num_stages):
        key = f"pooling_inverse_stage{input_stage}_to_stage{input_stage + 1}"
        assignment = capture.arrays[key].astype(np.int64)[assignment]
    return assignment


def plot_highlighted_region(
    ax,
    all_xyz: np.ndarray,
    selected: np.ndarray,
    anchor: np.ndarray,
    title: str,
    selected_rgb: np.ndarray | None = None,
    highlight_color: str = "#ff6b35",
    focus: bool = False,
) -> None:
    ax.scatter(
        all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2],
        c="#a8adb4", s=1, alpha=0.08, linewidths=0,
    )
    selected_xyz = all_xyz[selected]
    color = selected_rgb[selected] if selected_rgb is not None else highlight_color
    ax.scatter(
        selected_xyz[:, 0], selected_xyz[:, 1], selected_xyz[:, 2],
        c=color, s=8, alpha=0.95, linewidths=0,
    )
    ax.scatter(
        anchor[0], anchor[1], anchor[2], marker="*", s=150,
        c="white", edgecolors="black", linewidths=1.1,
    )
    ax.view_init(elev=35, azim=35)
    ax.set_proj_type("ortho")
    if focus:
        center = (selected_xyz.min(axis=0) + selected_xyz.max(axis=0)) / 2
        radius = max(float(np.ptp(selected_xyz, axis=0).max()) * 0.8, 0.045)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        set_axes_equal(ax, all_xyz)
    ax.set_title(title, fontsize=10, pad=8)


def select_action_anchor(capture: CaptureData, min_descendants: int = 20) -> int:
    """Choose an action-relevant anchor with enough points to expose shape."""
    probability = capture.arrays.get("action_point_probability")
    if probability is None:
        probability = np.ones(len(capture.final_coordinates), dtype=np.float32)
    counts = np.bincount(
        capture.input_to_final, minlength=len(capture.final_coordinates)
    )
    eligible = counts >= min_descendants
    if not eligible.any():
        return int(np.argmax(probability))
    scores = np.where(eligible, probability, -np.inf)
    return int(np.argmax(scores))


def render_pooling_partition(capture: CaptureData, output_dir: Path, dpi: int) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = capture.input_to_final
    num_anchors = len(capture.final_coordinates)
    rng = np.random.default_rng(7)
    hues = (np.arange(num_anchors) * 0.61803398875) % 1.0
    rng.shuffle(hues)
    anchor_colors = plt.get_cmap("hsv")(hues)[:, :3]
    point_colors = anchor_colors[labels]

    fig = plt.figure(figsize=(13, 6))
    for index, (elev, azim, title) in enumerate(
        [(35, 35, "Oblique view"), (90, -90, "Top view")], start=1
    ):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        xyz = capture.input_coordinates
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=point_colors, s=2, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        set_axes_equal(ax, xyz)
        if elev == 90:
            ax.set_zticks([])
            ax.set_zlabel("")
        ax.set_title(title)
    fig.suptitle(
        f"Exact final-anchor pooling partition | {capture.instruction}\n"
        f"{len(capture.input_coordinates)} input points → {num_anchors} final anchors",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.86, wspace=0.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = capture.path.stem
    png_path = output_dir / f"{stem}_pooling_partition.png"
    ply_path = output_dir / f"{stem}_pooling_partition.ply"
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    write_ply(ply_path, capture.input_coordinates, point_colors)
    return {"png": str(png_path), "ply": str(ply_path)}


def render_multiscale_receptive_field(
    capture: CaptureData, final_anchor: int, output_dir: Path, dpi: int
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_stages = capture.num_stages
    final_xyz = capture.final_coordinates[final_anchor]
    fig = plt.figure(figsize=(16, 10))
    descendant_counts = []

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    xyz = capture.input_coordinates
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=capture.input_rgb, s=2, linewidths=0)
    ax.scatter(
        final_xyz[0], final_xyz[1], final_xyz[2], marker="*", s=150,
        c="white", edgecolors="black", linewidths=1.1,
    )
    ax.view_init(elev=35, azim=35)
    ax.set_proj_type("ortho")
    set_axes_equal(ax, xyz)
    ax.set_title("Input RGB context")

    for subplot_index, stage in enumerate(range(num_stages + 1), start=2):
        coords = stage_coordinates(capture, stage)
        assignment = stage_to_final(capture, stage)
        selected = assignment == final_anchor
        descendant_counts.append(int(selected.sum()))
        plot_highlighted_region(
            fig.add_subplot(2, 3, subplot_index, projection="3d"),
            coords,
            selected,
            final_xyz,
            f"stage{stage}: {selected.sum()} contributing points",
            selected_rgb=capture.input_rgb if stage == 0 else None,
        )

    probability = capture.arrays.get("action_point_probability")
    probability_text = ""
    if probability is not None:
        probability_text = f" | action probability={float(probability[final_anchor]):.5f}"
    fig.suptitle(
        f"Exact pooling ancestry for final anchor {final_anchor}{probability_text}\n"
        f"{capture.instruction}",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90, wspace=0.02, hspace=0.18)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{capture.path.stem}_anchor_{final_anchor:03d}_ancestry.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return {
        "png": str(path),
        "final_anchor": final_anchor,
        "descendant_counts_stage0_to_final": descendant_counts,
    }


def normalized_features(capture: CaptureData) -> np.ndarray:
    features = capture.final_features
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def retrieve_cross_task(
    query_capture: CaptureData,
    query_anchor: int,
    captures: list[CaptureData],
    top_k: int,
) -> list[dict]:
    query = normalized_features(query_capture)[query_anchor]
    candidates = []
    for capture_index, capture in enumerate(captures):
        if capture.instruction == query_capture.instruction:
            continue
        similarity = normalized_features(capture) @ query
        best_anchor = int(np.argmax(similarity))
        candidates.append(
            {
                "capture_index": capture_index,
                "anchor": best_anchor,
                "similarity": float(similarity[best_anchor]),
            }
        )
    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    return candidates[:top_k]


def render_retrieval(
    query_capture: CaptureData,
    query_anchor: int,
    results: list[dict],
    captures: list[CaptureData],
    output_dir: Path,
    dpi: int,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(query_capture, query_anchor, None)] + [
        (captures[result["capture_index"]], result["anchor"], result["similarity"])
        for result in results
    ]
    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    fig = plt.figure(figsize=(16, 5.2 * rows))
    rendered_results = []
    for panel_index, (capture, anchor, similarity) in enumerate(panels, start=1):
        selected = capture.input_to_final == anchor
        title_prefix = "QUERY" if similarity is None else f"cosine={similarity:.4f}"
        title = (
            f"{title_prefix}\n{capture.instruction}\n"
            f"{capture.path.stem}, anchor {anchor}, {selected.sum()} input points"
        )
        plot_highlighted_region(
            fig.add_subplot(rows, columns, panel_index, projection="3d"),
            capture.input_coordinates,
            selected,
            capture.final_coordinates[anchor],
            title,
            selected_rgb=None,
            highlight_color="#00bcd4" if similarity is None else "#ff9800",
            focus=True,
        )
        if similarity is not None:
            rendered_results.append(
                {
                    "capture": str(capture.path),
                    "instruction": capture.instruction,
                    "anchor": anchor,
                    "cosine_similarity": similarity,
                    "input_descendants": int(selected.sum()),
                }
            )
    for panel_index in range(len(panels) + 1, rows * columns + 1):
        ax = fig.add_subplot(rows, columns, panel_index)
        ax.axis("off")
    fig.suptitle(
        "Cross-task retrieval of final 768-D PTV3 features\n"
        "Highlighted points are exact pooling descendants; ranking uses cosine similarity",
        fontsize=13,
        y=0.985,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.83, wspace=0.02, hspace=0.28)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_slug = query_capture.instruction.replace(" ", "_")
    path = output_dir / f"query_{query_slug}_{query_capture.path.stem}_anchor_{query_anchor:03d}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return {
        "png": str(path),
        "query_capture": str(query_capture.path),
        "query_instruction": query_capture.instruction,
        "query_anchor": query_anchor,
        "results": rendered_results,
    }


def main(args: Args) -> None:
    capture_paths = discover_captures(args.inputs)
    if not capture_paths:
        raise FileNotFoundError(f"No capture_*.npz files found in: {args.inputs}")
    captures = [load_capture(path) for path in capture_paths]
    output_dir = Path(args.output_dir)
    pooling_dir = output_dir / "pooling"
    retrieval_dir = output_dir / "retrieval"

    pooling_records = []
    for capture in captures:
        anchor = select_action_anchor(capture)
        pooling_records.append(
            {
                "capture": str(capture.path),
                "instruction": capture.instruction,
                "partition": render_pooling_partition(capture, pooling_dir, args.dpi),
                "ancestry": render_multiscale_receptive_field(
                    capture, anchor, pooling_dir, args.dpi
                ),
            }
        )

    # Use the earliest captured observation for one query per task.
    first_by_instruction: OrderedDict[str, CaptureData] = OrderedDict()
    for capture in captures:
        first_by_instruction.setdefault(capture.instruction, capture)
    retrieval_records = []
    for capture in first_by_instruction.values():
        query_anchor = select_action_anchor(capture)
        results = retrieve_cross_task(capture, query_anchor, captures, args.top_k)
        retrieval_records.append(
            render_retrieval(
                capture, query_anchor, results, captures, retrieval_dir, args.dpi
            )
        )

    summary = {
        "captures": len(captures),
        "method_1": "exact pooling ancestry composed from four pooling_inverse maps",
        "method_2": "cross-task cosine retrieval of L2-normalized final 768-D features",
        "pooling": pooling_records,
        "retrieval": retrieval_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "captures": len(captures),
                "pooling_visualizations": len(pooling_records),
                "retrieval_queries": len(retrieval_records),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)
