"""Analyze final PTV3 feature similarity between the three reach-target balls."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro

from ptv3_feature_viz import set_axes_equal
from render_ptv3_geometry_analysis import load_capture, normalized_features


@dataclasses.dataclass
class Args:
    capture: str
    output_dir: str = "reach_target_ball_analysis"
    dpi: int = 180


def kmeans_three(xyz: np.ndarray) -> np.ndarray:
    """Small deterministic k-means used only for the three colored spheres."""
    centers = [xyz[np.argmin(xyz[:, 0])]]
    for _ in range(2):
        distance = np.stack(
            [np.linalg.norm(xyz - center, axis=1) for center in centers], axis=1
        )
        centers.append(xyz[np.argmax(distance.min(axis=1))])
    centers = np.asarray(centers)
    labels = np.zeros(len(xyz), dtype=np.int64)
    for _ in range(30):
        distance = np.linalg.norm(xyz[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distance, axis=1)
        next_centers = np.stack(
            [xyz[next_labels == index].mean(axis=0) for index in range(3)]
        )
        if np.array_equal(next_labels, labels):
            break
        labels, centers = next_labels, next_centers
    return labels


def rankdata(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)).astype(np.float64)


def configure_3d(ax, xyz: np.ndarray, top: bool = False) -> None:
    ax.view_init(elev=90 if top else 35, azim=-90 if top else 35)
    ax.set_proj_type("ortho")
    set_axes_equal(ax, xyz)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("" if top else "z")
    if top:
        ax.set_zticks([])


def main(args: Args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    capture = load_capture(Path(args.capture))
    xyz = capture.input_coordinates
    rgb = capture.input_rgb

    # The task objects are the only strongly saturated regions in this scene.
    saturation = rgb.max(axis=1) - rgb.min(axis=1)
    colored_indices = np.flatnonzero(saturation > 0.25)
    if len(colored_indices) < 30:
        raise ValueError("Could not find the three colored reach_target spheres")
    colored_labels = kmeans_three(xyz[colored_indices])

    objects = []
    for cluster in range(3):
        indices = colored_indices[colored_labels == cluster]
        mean_rgb = rgb[indices].mean(axis=0)
        anchor_counts = np.bincount(
            capture.input_to_final[indices],
            minlength=len(capture.final_coordinates),
        )
        objects.append(
            {
                "indices": indices,
                "centroid": xyz[indices].mean(axis=0),
                "mean_rgb": mean_rgb,
                "representative_anchor": int(np.argmax(anchor_counts)),
                "anchor_counts": anchor_counts,
            }
        )

    # Variation 0 asks for the red target. Identify it from RGB, not location.
    target_index = int(
        np.argmax([obj["mean_rgb"][0] - obj["mean_rgb"][1] for obj in objects])
    )
    target = objects[target_index]
    distractors = [obj for index, obj in enumerate(objects) if index != target_index]
    distractors.sort(key=lambda obj: float(obj["centroid"][0]))
    ordered_objects = [target] + distractors
    labels = ["target red ball", "distractor ball A", "distractor ball B"]

    query_anchor = int(target["representative_anchor"])
    features = normalized_features(capture)
    similarities = features @ features[query_anchor]
    propagated = similarities[capture.input_to_final]
    distances = np.linalg.norm(
        capture.final_coordinates - capture.final_coordinates[query_anchor], axis=1
    )
    not_query = np.arange(len(similarities)) != query_anchor
    pearson = float(np.corrcoef(distances[not_query], similarities[not_query])[0, 1])
    spearman = float(
        np.corrcoef(
            rankdata(distances[not_query]), rankdata(similarities[not_query])
        )[0, 1]
    )

    color_min = min(float(similarities.min()), 0.0)
    norm = Normalize(vmin=color_min, vmax=1.0)
    cmap = plt.get_cmap("turbo")
    object_colors = ["#e53935", "#00a86b", "#00695c"]

    fig = plt.figure(figsize=(17, 12))
    rgb_ax = fig.add_subplot(2, 2, 1, projection="3d")
    rgb_ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=3, linewidths=0)
    configure_3d(rgb_ax, xyz)
    rgb_ax.set_title("Input RGB: detected three colored balls")

    heat_axes = [
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
    ]
    scatter = None
    for heat_ax, top, title in [
        (heat_axes[0], False, "Feature-similarity heatmap: oblique view"),
        (heat_axes[1], True, "Feature-similarity heatmap: top view"),
    ]:
        scatter = heat_ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=propagated, cmap=cmap, norm=norm, s=4, linewidths=0,
        )
        configure_3d(heat_ax, xyz, top=top)
        heat_ax.set_title(title)

    object_records = []
    for label, obj, object_color in zip(labels, ordered_objects, object_colors):
        anchor = int(obj["representative_anchor"])
        similarity = float(similarities[anchor])
        center = obj["centroid"]
        for ax in [rgb_ax, *heat_axes]:
            ax.scatter(
                *center, marker="o", s=220, facecolors="none",
                edgecolors=object_color, linewidths=2.2,
            )
            ax.text(
                center[0], center[1], center[2] + 0.035,
                f"{label}\na{anchor}, cos={similarity:.3f}",
                color=object_color, fontsize=9,
            )
        touching = np.flatnonzero(obj["anchor_counts"])
        object_records.append(
            {
                "label": label,
                "representative_anchor": anchor,
                "cosine_similarity": similarity,
                "centroid": center.tolist(),
                "mean_rgb": obj["mean_rgb"].tolist(),
                "input_points": int(len(obj["indices"])),
                "touching_final_anchors": [
                    {
                        "anchor": int(item),
                        "ball_points": int(obj["anchor_counts"][item]),
                        "cosine_similarity": float(similarities[item]),
                    }
                    for item in touching
                ],
            }
        )

    query_xyz = capture.final_coordinates[query_anchor]
    for ax in heat_axes:
        ax.scatter(
            *query_xyz, marker="*", s=270, c="white",
            edgecolors="black", linewidths=1.4,
        )

    distance_ax = fig.add_subplot(2, 2, 4)
    distance_ax.scatter(
        distances[not_query], similarities[not_query],
        c="#757575", s=45, alpha=0.8, label="other final anchors",
    )
    for label, obj, object_color in zip(labels, ordered_objects, object_colors):
        anchor = int(obj["representative_anchor"])
        distance_ax.scatter(
            distances[anchor], similarities[anchor],
            c=object_color, edgecolors="black", s=120, zorder=3,
        )
        distance_ax.annotate(
            f"{label}\na{anchor}",
            (distances[anchor], similarities[anchor]),
            xytext=(6, 7), textcoords="offset points", fontsize=8,
        )
    distance_ax.set_xlabel("Euclidean distance from query final anchor")
    distance_ax.set_ylabel("768-D feature cosine similarity")
    distance_ax.set_title(
        "Does similarity only follow position?\n"
        f"excluding query: Pearson r={pearson:.3f}, Spearman ρ={spearman:.3f}"
    )
    distance_ax.grid(alpha=0.25)

    assert scatter is not None
    colorbar_axis = fig.add_axes([0.08, 0.035, 0.40, 0.018])
    colorbar = fig.colorbar(scatter, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Cosine similarity to target-ball final feature")

    fig.suptitle(
        f"RLBench reach_target: one target ball → whole-scene PTV3 heatmap\n"
        f"{capture.instruction} | {capture.path.stem} | query final anchor {query_anchor}",
        fontsize=14,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.03, right=0.98, bottom=0.10, top=0.91, wspace=0.09, hspace=0.18
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{capture.path.stem}_three_ball_feature_heatmap.png"
    summary_path = output_dir / f"{capture.path.stem}_three_ball_feature_analysis.json"
    fig.savefig(image_path, dpi=args.dpi)
    plt.close(fig)

    summary = {
        "capture": str(capture.path),
        "instruction": capture.instruction,
        "query_anchor": query_anchor,
        "num_final_points": len(similarities),
        "distance_similarity_correlation_excluding_query": {
            "pearson": pearson,
            "spearman": spearman,
        },
        "objects": object_records,
        "image": str(image_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
