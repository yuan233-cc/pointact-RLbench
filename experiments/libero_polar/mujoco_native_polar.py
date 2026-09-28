"""Adapter from a live robosuite MuJoCo scene to the RLBench native renderer.

The official LIBERO package is never patched.  Visual geoms are read through
MuJoCo model/data arrays and converted to the renderer-neutral mesh dictionaries
accepted by ``rlbench.native_renderer.NativePolarizationRenderer``.
"""

from __future__ import annotations

import fnmatch
import sys
import types
from pathlib import Path

import numpy as np


def _clean(vertices, faces):
    """Remove invalid/degenerate triangles before the strict CUDA BVH build."""
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    in_range = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    candidate = faces[in_range]
    triangle = vertices[candidate]
    area2 = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0],
                                    triangle[:, 2] - triangle[:, 0]), axis=1)
    good = np.isfinite(triangle).all(axis=(1, 2)) & (area2 > 1e-10)
    candidate = candidate[good]
    if not len(candidate):
        raise ValueError("Geometry has no finite nondegenerate triangles")
    used, remap = np.unique(candidate.reshape(-1), return_inverse=True)
    return vertices[used], remap.reshape(-1, 3).astype(np.uint32)


def _box(size):
    x, y, z = np.asarray(size, dtype=np.float32)
    vertices = np.asarray([(a*x, b*y, c*z) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=np.float32)
    faces = np.asarray([(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
                        (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)], dtype=np.uint32)
    return vertices, faces


def _uv_surface(radius_xy, half_z, rings=12, segments=24, capsule=False):
    vertices = []
    for ring in range(rings + 1):
        theta = np.pi * ring / rings
        radial = np.sin(theta)
        z = np.cos(theta)
        if capsule:
            z = z * radius_xy[0] + (half_z if z >= 0 else -half_z)
            rx, ry = radius_xy
        else:
            z *= half_z
            rx, ry = radius_xy
        for segment in range(segments):
            phi = 2 * np.pi * segment / segments
            vertices.append((rx * radial * np.cos(phi), ry * radial * np.sin(phi), z))
    faces = []
    for ring in range(rings):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            c, d = a + segments, b + segments
            faces.extend(((a, c, d), (a, d, b)))
    return np.asarray(vertices, np.float32), np.asarray(faces, np.uint32)


def _cylinder(radius, half_length, segments=32):
    vertices = [(0, 0, -half_length), (0, 0, half_length)]
    for z in (-half_length, half_length):
        vertices.extend((radius*np.cos(2*np.pi*i/segments), radius*np.sin(2*np.pi*i/segments), z) for i in range(segments))
    faces = []
    lower, upper = 2, 2 + segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(((0, lower+j, lower+i), (1, upper+i, upper+j),
                      (lower+i, lower+j, upper+j), (lower+i, upper+j, upper+i)))
    return np.asarray(vertices, np.float32), np.asarray(faces, np.uint32)


class MujocoSceneAdapter:
    """Extract cached local meshes plus current rigid transforms from MjSim."""

    def __init__(self, sim):
        self.sim = sim
        self._geometry = {}

    def _geometry_for(self, geom_id):
        model = self.sim.model
        geom_type = int(model.geom_type[geom_id])
        size = np.asarray(model.geom_size[geom_id], dtype=np.float32)
        data_id = int(model.geom_dataid[geom_id])
        key = (geom_type, tuple(size), data_id)
        if key in self._geometry:
            return self._geometry[key]
        if geom_type == 0:  # plane; MuJoCo uses zero size for an infinite plane
            extent = np.where(size[:2] > 0, size[:2], 5.0)
            value = _box((extent[0], extent[1], 0.002))
        elif geom_type == 2:  # sphere
            value = _uv_surface((size[0], size[0]), size[0])
        elif geom_type == 3:  # capsule
            value = _uv_surface((size[0], size[0]), size[1], capsule=True)
        elif geom_type == 4:  # ellipsoid
            value = _uv_surface((size[0], size[1]), size[2])
        elif geom_type == 5:  # cylinder
            value = _cylinder(size[0], size[1])
        elif geom_type == 6:  # box
            value = _box(size)
        elif geom_type == 7:  # mesh
            if data_id < 0:
                raise ValueError(f"Mesh geom {geom_id} has no mesh data")
            va, vn = int(model.mesh_vertadr[data_id]), int(model.mesh_vertnum[data_id])
            fa, fn = int(model.mesh_faceadr[data_id]), int(model.mesh_facenum[data_id])
            vertices = np.asarray(model.mesh_vert[va:va+vn], dtype=np.float32).copy()
            faces = np.asarray(model.mesh_face[fa:fa+fn], dtype=np.int64).copy()
            # MuJoCo versions expose either mesh-local or global indices.
            if len(faces) and faces.max() >= vn:
                faces -= va
            value = vertices, faces.astype(np.uint32)
        else:
            raise NotImplementedError(f"MuJoCo geom type {geom_type} is not supported")
        value = _clean(*value)
        self._geometry[key] = value
        return value

    def meshes(self):
        model, data = self.sim.model, self.sim.data
        result = []
        for geom_id in range(model.ngeom):
            # robosuite convention: group 0 collision, group 1 visual.
            if int(model.geom_group[geom_id]) != 1:
                continue
            name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
            vertices, faces = self._geometry_for(geom_id)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
            pose[:3, 3] = np.asarray(data.geom_xpos[geom_id])
            material_id = int(model.geom_matid[geom_id])
            rgba = (np.asarray(model.mat_rgba[material_id]) if material_id >= 0
                    else np.asarray(model.geom_rgba[geom_id]))
            if rgba[3] <= 1e-4:
                continue
            result.append({
                "key": str(geom_id), "handle": geom_id, "name": name,
                "vertices": vertices, "faces": faces, "to_world": pose,
                "base_color": np.clip(rgba[:3], 0, 1).astype(np.float32),
                "geometry_revision": hash((int(model.geom_type[geom_id]), int(model.geom_dataid[geom_id]), tuple(model.geom_size[geom_id]))) & 0x7fffffff,
            })
        return result


class MujocoNativePolarRenderer:
    """Physical CUDA Mueller rendering of a live MuJoCo scene."""

    def __init__(self, sim, renderer_repo: Path, *, spp=512, max_depth=8, seed=7, device=0,
                 material_overrides=None):
        renderer_repo = Path(renderer_repo).resolve()
        if not (renderer_repo / "rlbench/native_renderer.py").is_file():
            raise FileNotFoundError(f"Not an RLBench native-renderer checkout: {renderer_repo}")
        if str(renderer_repo) not in sys.path:
            sys.path.insert(0, str(renderer_repo))
        # Loading the renderer does not require RLBench/PyRep.  A namespace
        # package prevents rlbench/__init__.py from importing the simulator and
        # its unrelated dependencies into the LIBERO environment.
        if "rlbench" not in sys.modules:
            package = types.ModuleType("rlbench")
            package.__path__ = [str(renderer_repo / "rlbench")]
            package.__package__ = "rlbench"
            sys.modules["rlbench"] = package
        from rlbench.native_config import NativePolarizationConfig
        from rlbench.native_renderer import NativePolarizationRenderer
        self.sim = sim
        self.adapter = MujocoSceneAdapter(sim)
        material_overrides = dict(material_overrides or {})
        geom_names = [sim.model.geom_id2name(i) or f"geom_{i}"
                      for i in range(sim.model.ngeom)
                      if int(sim.model.geom_group[i]) == 1]
        resolved = {}
        unmatched = []
        for pattern, specification in material_overrides.items():
            if pattern.startswith("_"):
                continue
            matches = [name for name in geom_names if fnmatch.fnmatchcase(name, pattern)]
            if not matches:
                unmatched.append(pattern)
            for name in matches:
                resolved[name] = specification
        if unmatched:
            raise ValueError("Material patterns match no MuJoCo geom: " + ", ".join(unmatched))
        self.material_patterns = sorted(k for k in material_overrides if not k.startswith("_"))
        self.material_override_count = len(resolved)
        self.renderer = NativePolarizationRenderer(NativePolarizationConfig(
            spp=spp, max_depth=max_depth, seed=seed, device=device,
            lighting="reference", geometry_source="native",
            material_overrides=resolved,
        ))

    def set_sim(self, sim):
        """Rebind after LIBERO hard-reset replaces its underlying MjSim."""
        if sim is not self.sim:
            self.sim = sim
            self.adapter = MujocoSceneAdapter(sim)

    def render(self, camera_name, width, height, intrinsics, camera_to_world, *, seed):
        model = self.sim.model
        near = float(model.vis.map.znear * model.stat.extent)
        far = float(model.vis.map.zfar * model.stat.extent)
        result = self.renderer.render(
            self.adapter.meshes(), [],
            {"width": width, "height": height, "intrinsics": intrinsics,
             "to_world": camera_to_world, "near_clip": near, "far_clip": far},
            seed=seed, geometry=False,
        )
        angle = np.asarray(result["AoLP"], dtype=np.float32)
        valid = np.asarray(result["AoLP_valid_mask"], dtype=bool)
        result["cos2AoLP"] = np.where(valid, np.cos(2*angle), 0).astype(np.float32)
        result["sin2AoLP"] = np.where(valid, np.sin(2*angle), 0).astype(np.float32)
        result["metadata"]["scene_adapter"] = "robosuite-mujoco-visual-geoms-v1"
        result["metadata"]["camera_name"] = camera_name
        result["metadata"]["geometry_outputs"] = "disabled; LIBERO MuJoCo depth is authoritative"
        result["metadata"]["explicit_material_overrides"] = self.material_override_count
        return result

    def close(self):
        self.renderer.close()
