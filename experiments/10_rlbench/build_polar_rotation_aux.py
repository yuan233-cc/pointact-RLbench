#!/usr/bin/env python3
"""Build row-aligned surface metadata for online filled9 polar rotation.

The existing filled9 point clouds are not modified.  For every point row this
script records the Coppelia surface pixel, surface position and world normal,
the native-polar material id, and validity masks.  Camera extrinsics are stored
per frame so a training-time augmentation can recompute view-dependent polar
features after rotating XYZ and the associated reference normal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT.parent
RLBENCH_ROOT = PROJECT_ROOT / "rlbench_custom_render/RLBench"
for path in (RLBENCH_ROOT / "tools",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from episode_snapshot_archive import EpisodeSnapshotArchive  # noqa: E402


msgpack_numpy.patch()

TASKS = (
    "close_box", "close_laptop_lid", "toilet_seat_down", "sweep_to_dustpan",
    "close_fridge", "phone_on_base", "take_umbrella_out_of_umbrella_stand",
    "take_frame_off_hanger", "stack_wine", "water_plants",
)
DATASET = REPO_ROOT / (
    "robot_data/rlbench/lerobot_point_lmdb/"
    "hybridvla_10tasks_train_keysteps_polar_rlbench9_v2"
)
RAW = RLBENCH_ROOT / "output/ten_tasks_polar_train_20260921"
AUX_NAME = "polar_rotation_aux_filled9"


def camera_pointcloud(depth: np.ndarray, camera: dict) -> np.ndarray:
    """Match the RLBench depth unprojection used to build the v2 dataset."""
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32))
    pixel_depth = np.stack((u * depth, v * depth, depth), axis=-1)
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    rotation = to_world[:3, :3]
    center = to_world[:3, 3:4]
    world_to_camera = np.concatenate((rotation.T, -rotation.T @ center), axis=1)
    projection = intrinsics @ world_to_camera
    inverse = np.linalg.inv(np.vstack((projection, [0., 0., 0., 1.])))[:3]
    homogeneous = np.concatenate(
        (pixel_depth, np.ones((height, width, 1), dtype=np.float32)), axis=-1)
    return (homogeneous.reshape(-1, 4) @ inverse.T).reshape(height, width, 3).astype(np.float32)


def project_world(xyz: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    to_world = np.asarray(camera["to_world"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    camera_xyz = (np.asarray(xyz, dtype=np.float64) - to_world[:3, 3]) @ to_world[:3, :3]
    depth = camera_xyz[:, 2]
    image = camera_xyz @ intrinsics.T
    uv = image[:, :2] / np.where(np.abs(image[:, 2:3]) > 1e-12,
                                 image[:, 2:3], np.nan)
    return uv, depth


def _axis_tangent(points: np.ndarray, depth: np.ndarray, object_mask: np.ndarray,
                  axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a same-object image tangent with central/one-sided differences."""
    valid = np.isfinite(depth) & (depth > .01) & np.isfinite(points).all(axis=-1)
    threshold = np.maximum(.015, .03 * depth)
    prev_points = np.roll(points, 1, axis=axis)
    next_points = np.roll(points, -1, axis=axis)
    prev_depth = np.roll(depth, 1, axis=axis)
    next_depth = np.roll(depth, -1, axis=axis)
    prev_mask = np.roll(object_mask, 1, axis=axis)
    next_mask = np.roll(object_mask, -1, axis=axis)
    prev_valid = (valid & np.roll(valid, 1, axis=axis)
                  & (prev_mask == object_mask)
                  & (np.abs(prev_depth - depth) <= threshold))
    next_valid = (valid & np.roll(valid, -1, axis=axis)
                  & (next_mask == object_mask)
                  & (np.abs(next_depth - depth) <= threshold))
    edge_first = [slice(None)] * 2
    edge_last = [slice(None)] * 2
    edge_first[axis] = 0
    edge_last[axis] = -1
    prev_valid[tuple(edge_first)] = False
    next_valid[tuple(edge_last)] = False
    tangent = np.zeros_like(points, dtype=np.float32)
    both = prev_valid & next_valid
    only_next = ~prev_valid & next_valid
    only_prev = prev_valid & ~next_valid
    tangent[both] = next_points[both] - prev_points[both]
    tangent[only_next] = next_points[only_next] - points[only_next]
    tangent[only_prev] = points[only_prev] - prev_points[only_prev]
    return tangent, prev_valid | next_valid


def dense_surface_geometry(depth: np.ndarray, object_mask: np.ndarray,
                           camera: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = camera_pointcloud(depth, camera)
    tangent_u, valid_u = _axis_tangent(points, depth, object_mask, axis=1)
    tangent_v, valid_v = _axis_tangent(points, depth, object_mask, axis=0)
    normals = np.cross(tangent_u, tangent_v)
    norm = np.linalg.norm(normals, axis=-1)
    valid = valid_u & valid_v & np.isfinite(norm) & (norm > 1e-10)
    normals[valid] /= norm[valid, None]
    normals[~valid] = 0
    camera_center = np.asarray(camera["to_world"], dtype=np.float32)[:3, 3]
    toward_camera = np.sum(normals * (camera_center - points), axis=-1)
    normals[toward_camera < 0] *= -1
    return points.astype(np.float32), normals.astype(np.float32), valid


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def default_material(mesh: dict) -> dict:
    """Mirror native_renderer._default_material without initializing CUDA."""
    name = mesh["name"].lower()
    color = _jsonable(np.asarray(mesh["base_color"], dtype=np.float32))
    if "finger" in name or "gripper" in name:
        return {"type": "roughconductor", "material": "Al", "alpha": .18}
    if any(word in name for word in ("wall", "roof", "floor")):
        return {"type": "diffuse", "reflectance": color}
    eta, alpha = 1.5, .2
    if name in ("target", "distractor0", "distractor1"):
        eta, alpha = 1.49, .08
    elif "panda" in name:
        alpha = .14
    elif "table" in name or "workspace" in name:
        eta, alpha = 1.48, .28
    return {"type": "pplastic", "eta": eta, "alpha": alpha,
            "diffuse_reflectance": color}


class MaterialRegistry:
    def __init__(self):
        self._canonical_to_id: dict[str, int] = {}
        self.records: list[dict] = []

    def id_for(self, task: str, mesh: dict, overrides: dict) -> int:
        source = "override" if mesh["name"] in overrides else (
            "bsdf" if mesh.get("bsdf") is not None else "synthetic_preset")
        spec = overrides.get(mesh["name"], mesh.get("bsdf"))
        spec = default_material(mesh) if spec is None else _jsonable(spec)
        canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        if canonical not in self._canonical_to_id:
            material_id = len(self.records)
            self._canonical_to_id[canonical] = material_id
            self.records.append({"material_id": material_id, "spec": spec,
                                 "examples": [{"task": task, "mesh": mesh["name"],
                                               "source": source}]})
        else:
            material_id = self._canonical_to_id[canonical]
            examples = self.records[material_id]["examples"]
            example = {"task": task, "mesh": mesh["name"], "source": source}
            if example not in examples and len(examples) < 8:
                examples.append(example)
        return material_id


def transform_vertices(mesh: dict) -> np.ndarray:
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    to_world = np.asarray(mesh["to_world"], dtype=np.float32)
    return vertices @ to_world[:3, :3].T + to_world[:3, 3]


def material_ids_for_points(surface_xyz: np.ndarray, handles: np.ndarray,
                            meshes: list[dict], task: str, overrides: dict,
                            registry: MaterialRegistry) -> tuple[np.ndarray, np.ndarray]:
    mesh_groups: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for mesh in meshes:
        if mesh["name"] == "workspace":
            continue
        material_id = registry.id_for(task, mesh, overrides)
        mesh_groups[int(mesh["handle"])].append((material_id, transform_vertices(mesh)))

    result = np.full(len(handles), -1, dtype=np.int32)
    valid = np.zeros(len(handles), dtype=bool)
    for handle in np.unique(handles):
        rows = np.flatnonzero(handles == handle)
        candidates = mesh_groups.get(int(handle), [])
        if not candidates:
            continue
        material_ids = {item[0] for item in candidates}
        if len(material_ids) == 1:
            result[rows] = next(iter(material_ids))
            valid[rows] = True
            continue
        vertices = np.concatenate([item[1] for item in candidates], axis=0)
        vertex_materials = np.concatenate([
            np.full(len(item[1]), item[0], dtype=np.int32) for item in candidates])
        finite = np.isfinite(surface_xyz[rows]).all(axis=1)
        if not finite.any():
            continue
        distance, nearest = cKDTree(vertices).query(surface_xyz[rows[finite]], k=1)
        selected_rows = rows[finite]
        result[selected_rows] = vertex_materials[np.asarray(nearest)]
        # Large flat triangles can have distant vertices, so distance is diagnostic,
        # not a validity cutoff. The Coppelia handle still identifies the object.
        valid[selected_rows] = np.isfinite(distance)
    return result, valid


def unpack(txn: lmdb.Transaction, key: bytes) -> np.ndarray:
    value = txn.get(key)
    if value is None:
        raise KeyError(key.decode("ascii"))
    return np.asarray(msgpack.unpackb(bytes(value)))


def episode_specs(raw: Path, episodes_per_task: int):
    return [(task_index * episodes_per_task + episode_index, task,
             raw / task / f"episode_{episode_index:06d}")
            for task_index, task in enumerate(TASKS)
            for episode_index in range(episodes_per_task)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--limit-episodes", type=int)
    args = parser.parse_args()
    output = args.output or (args.dataset / AUX_NAME)
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"remove or resume existing staging directory: {staging}")
    staging.mkdir(parents=True)

    info = json.loads((args.dataset / "meta/info.json").read_text())
    specs = episode_specs(args.raw, args.episodes_per_task)
    if args.start_episode:
        specs = specs[args.start_episode:]
    if args.limit_episodes is not None:
        specs = specs[:args.limit_episodes]
    expected_frames = (None if args.limit_episodes is not None or args.start_episode
                       else int(info["total_frames"]))

    filled_env = lmdb.open(str(args.dataset / "points_frontview_polar_filled9"),
                           readonly=True, lock=False, readahead=False, max_readers=2)
    incomplete_env = lmdb.open(str(args.dataset / "points_frontview_polar_incomplete9"),
                               readonly=True, lock=False, readahead=False, max_readers=2)
    pixel_env = lmdb.open(str(args.dataset / "point_pixel_indices"),
                          readonly=True, lock=False, readahead=False, max_readers=2)
    output_env = lmdb.open(str(staging), map_size=2 * 1024**3, subdir=True, sync=True)
    registry = MaterialRegistry()
    totals = defaultdict(int)
    episode_records = []
    try:
        filled_txn = filled_env.begin(buffers=True)
        incomplete_txn = incomplete_env.begin(buffers=True)
        pixel_txn = pixel_env.begin(buffers=True)
        for position, (global_episode, task, raw_episode) in enumerate(specs, 1):
            if not raw_episode.is_dir():
                raise FileNotFoundError(raw_episode)
            overrides = json.loads((raw_episode / "materials.json").read_text())
            archive = EpisodeSnapshotArchive(raw_episode / "snapshots")
            frame_files = sorted((raw_episode / "frames").glob("[0-9]" * 6 + ".npz"))
            frame_records = []
            write_txn = output_env.begin(write=True)
            try:
                for frame_index, frame_file in enumerate(frame_files):
                    key = f"{global_episode}-{frame_index}".encode("ascii")
                    filled = unpack(filled_txn, key)
                    incomplete = unpack(incomplete_txn, key)
                    stored_pixels = unpack(pixel_txn, key).astype(np.int32, copy=False)
                    if filled.ndim != 2 or filled.shape[1] != 9:
                        raise ValueError(f"{key!r}: expected filled Nx9, got {filled.shape}")
                    if len(incomplete) > len(filled) or len(stored_pixels) != len(incomplete):
                        raise ValueError(f"{key!r}: invalid filled/incomplete/pixel lengths")
                    if not np.array_equal(filled[:len(incomplete)], incomplete):
                        raise ValueError(f"{key!r}: filled prefix is not the incomplete cloud")

                    meshes, _, cameras, _ = archive.load_frame(frame_index)
                    camera = cameras["front"]
                    with np.load(frame_file) as source:
                        depth = np.asarray(source["coppelia_depth_m"], dtype=np.float32)
                        object_mask = np.asarray(source["coppelia_object_mask"], dtype=np.int32)
                    height, width = depth.shape
                    uv, camera_depth = project_world(filled[:, :3], camera)
                    finite = np.isfinite(uv).all(axis=1) & np.isfinite(camera_depth)
                    xy = np.floor(np.where(finite[:, None], uv, -1.) + 1e-4).astype(np.int64)
                    inside = (finite & (camera_depth > .01) & (xy[:, 0] >= 0)
                              & (xy[:, 0] < width) & (xy[:, 1] >= 0)
                              & (xy[:, 1] < height))
                    pixels = np.full(len(filled), -1, dtype=np.int32)
                    pixels[inside] = (xy[inside, 1] * width + xy[inside, 0]).astype(np.int32)
                    mismatches = int(np.count_nonzero(pixels[:len(incomplete)] != stored_pixels))
                    if mismatches:
                        raise ValueError(f"{key!r}: {mismatches} stored pixel projection mismatches")

                    dense_xyz, dense_normals, dense_normal_valid = dense_surface_geometry(
                        depth, object_mask, camera)
                    flat_xyz = dense_xyz.reshape(-1, 3)
                    flat_normals = dense_normals.reshape(-1, 3)
                    flat_normal_valid = dense_normal_valid.reshape(-1)
                    flat_handles = object_mask.reshape(-1)
                    surface_xyz = np.zeros((len(filled), 3), dtype=np.float32)
                    normal_world = np.zeros((len(filled), 3), dtype=np.float32)
                    handles = np.full(len(filled), -1, dtype=np.int32)
                    normal_valid = np.zeros(len(filled), dtype=bool)
                    surface_valid = inside.copy()
                    selected = pixels[inside]
                    surface_xyz[inside] = flat_xyz[selected]
                    normal_world[inside] = flat_normals[selected]
                    handles[inside] = flat_handles[selected]
                    surface_valid[inside] &= (np.isfinite(surface_xyz[inside]).all(axis=1)
                                              & (depth.reshape(-1)[selected] > .01))
                    normal_valid[inside] = flat_normal_valid[selected]
                    material_ids, material_valid = material_ids_for_points(
                        surface_xyz, handles, meshes, task, overrides, registry)
                    material_valid &= inside
                    valid = surface_valid & normal_valid & material_valid

                    camera_to_world = np.asarray(camera["to_world"], dtype=np.float32)
                    payload = {
                        "pixel_indices": pixels,
                        "surface_xyz_world": surface_xyz,
                        "normal_world": normal_world.astype(np.float16),
                        "material_id": material_ids,
                        "surface_valid_mask": surface_valid.astype(np.uint8),
                        "normal_valid_mask": normal_valid.astype(np.uint8),
                        "material_valid_mask": material_valid.astype(np.uint8),
                        "valid_mask": valid.astype(np.uint8),
                        "camera_to_world": camera_to_world,
                        "incomplete_count": np.asarray(len(incomplete), dtype=np.int32),
                    }
                    write_txn.put(key, msgpack.packb(payload))
                    record = {
                        "episode_index": global_episode, "frame_index": frame_index,
                        "rows": len(filled), "incomplete_rows": len(incomplete),
                        "filled_rows": len(filled) - len(incomplete),
                        "surface_valid": int(surface_valid.sum()),
                        "normal_valid": int(normal_valid.sum()),
                        "material_valid": int(material_valid.sum()),
                        "fully_valid": int(valid.sum()),
                        "filled_fully_valid": int(valid[len(incomplete):].sum()),
                    }
                    frame_records.append(record)
                    for name in ("rows", "incomplete_rows", "filled_rows", "surface_valid",
                                 "normal_valid", "material_valid", "fully_valid",
                                 "filled_fully_valid"):
                        totals[name] += record[name]
                    totals["frames"] += 1
                write_txn.commit()
            except Exception:
                write_txn.abort()
                raise
            episode_records.extend(frame_records)
            print(f"built {position}/{len(specs)} episode {global_episode} {task}: "
                  f"{len(frame_records)} frames", flush=True)
    finally:
        filled_env.close()
        incomplete_env.close()
        pixel_env.close()
        output_env.close()

    if expected_frames is not None and totals["frames"] != expected_frames:
        raise ValueError(f"expected {expected_frames} frames, got {totals['frames']}")
    check_env = lmdb.open(str(staging), readonly=True, lock=False, readahead=False)
    try:
        entries = int(check_env.stat()["entries"])
    finally:
        check_env.close()
    if entries != totals["frames"]:
        raise ValueError(f"LMDB entry count {entries} != processed frames {totals['frames']}")

    material_table = {"schema_version": 1, "materials": registry.records}
    material_text = json.dumps(material_table, indent=2) + "\n"
    material_path = args.dataset / "meta/polar_rotation_materials.json"
    stats_path = args.dataset / "meta/polar_rotation_aux.json"
    frame_stats_path = args.dataset / "polar_rotation_aux_frame_stats.jsonl"
    summary = {
        "complete": args.limit_episodes is None,
        "aux_dirname": output.name,
        "source_point_dirname": "points_frontview_polar_filled9",
        "normal_source": "same-object finite differences on archived Coppelia depth",
        "material_source": "archived Coppelia object mask + native renderer mesh BSDF",
        "normal_orientation": "world frame, flipped toward archived front camera",
        "normal_storage_dtype": "float16",
        "surface_xyz_storage_dtype": "float32",
        "rows_are_filled9_aligned": True,
        "entries": entries,
        "totals": dict(totals),
        "valid_fraction": float(totals["fully_valid"] / max(totals["rows"], 1)),
        "filled_valid_fraction": float(
            totals["filled_fully_valid"] / max(totals["filled_rows"], 1)),
        "material_count": len(registry.records),
        "material_table_sha256": hashlib.sha256(material_text.encode()).hexdigest(),
    }
    if args.limit_episodes is None:
        material_path.write_text(material_text)
        stats_path.write_text(json.dumps(summary, indent=2) + "\n")
        frame_stats_path.write_text("".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in episode_records))
    else:
        (staging / "materials.json").write_text(material_text)
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    staging.rename(output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
