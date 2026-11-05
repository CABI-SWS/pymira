"""Watertight meshes for radius-bearing spatial graphs.

Each input centreline segment is represented by a linearly tapered polyball:
the envelope of radius-bearing balls swept along the unchanged straight input
segment. The mesher extracts the boundary of their implicit union without
moving or smoothing any supplied centreline point or radius.
"""

from __future__ import annotations

import warnings

import numpy as np


def _graph_segments(graph, edge_filter=None, min_radius=0.0, radius_scale=1.0):
    points = np.asarray(graph.get_data("EdgePointCoordinates"), dtype=float)
    counts = np.asarray(graph.get_data("NumEdgePoints"), dtype=np.int64).reshape(-1)
    radius_field = graph.get_radius_field()
    radii = (np.ones(points.shape[0], dtype=float) if radius_field is None else
             np.asarray(radius_field["data"], dtype=float).reshape(-1))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("EdgePointCoordinates must have shape (n, 3)")
    if counts.size != graph.nedge or np.any(counts < 2):
        raise ValueError("Each graph edge must have at least two points")
    if int(np.sum(counts)) != points.shape[0] or radii.size != points.shape[0]:
        raise ValueError("Graph point, radius and NumEdgePoints arrays are inconsistent")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(radii)):
        raise ValueError("Mesh input coordinates and radii must be finite")
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be positive")

    if edge_filter is None:
        edge_filter = np.ones(counts.size, dtype=bool)
    else:
        edge_filter = np.asarray(edge_filter, dtype=bool).reshape(-1)
        if edge_filter.size != counts.size:
            raise ValueError("edge_filter must contain one value per graph edge")

    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    starts_out, ends_out, rstarts_out, rends_out = [], [], [], []
    for edge_index, (start, count) in enumerate(zip(starts, counts)):
        if not edge_filter[edge_index]:
            continue
        edge_points = points[start:start + count]
        edge_radii = radii[start:start + count] * radius_scale
        for point_index in range(count - 1):
            p0, p1 = edge_points[point_index:point_index + 2]
            r0, r1 = edge_radii[point_index:point_index + 2]
            if max(r0, r1) < min_radius:
                continue
            if r0 <= 0.0 or r1 <= 0.0:
                raise ValueError("All exported vessel radii must be positive")
            if np.linalg.norm(p1 - p0) <= np.finfo(float).eps:
                continue
            starts_out.append(p0)
            ends_out.append(p1)
            rstarts_out.append(r0)
            rends_out.append(r1)
    if not starts_out:
        return (np.empty((0, 3)), np.empty((0, 3)),
                np.empty(0), np.empty(0))
    return (np.asarray(starts_out), np.asarray(ends_out),
            np.asarray(rstarts_out), np.asarray(rends_out))


def _grid_spec(p0, p1, r0, r1, resolution, voxel_size, max_voxels,
               extra_padding=0.0):
    min_radius = float(np.min(np.minimum(r0, r1)))
    max_radius = float(np.max(np.maximum(r0, r1)))
    if voxel_size is None:
        voxel_size = min_radius / max(3.0, float(resolution) / 3.0)
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be a finite positive number")
    if max_voxels < 8:
        raise ValueError("max_voxels must be at least 8")
    padding = max_radius + extra_padding + 2.0 * voxel_size
    lower = np.minimum(np.min(p0, axis=0), np.min(p1, axis=0)) - padding
    upper = np.maximum(np.max(p0, axis=0), np.max(p1, axis=0)) + padding
    shape = np.maximum(3, np.ceil((upper - lower) / voxel_size).astype(np.int64) + 1)
    requested = int(np.prod(shape, dtype=np.int64))
    if requested > max_voxels:
        voxel_size *= (requested / float(max_voxels)) ** (1.0 / 3.0) * 1.001
        shape = np.maximum(3, np.ceil((upper - lower) / voxel_size).astype(np.int64) + 1)
        warnings.warn(
            "Implicit mesh grid exceeded max_voxels; voxel_size was increased "
            f"to {voxel_size:g}. Increase max_voxels or filter small vessels "
            "if their radii must be resolved.", RuntimeWarning, stacklevel=3)
    return lower, tuple(int(value) for value in shape), voxel_size


def _polyball_segment_field(x, y, z, p0, p1, r0, r1):
    """Implicit field for a radius-bearing ball swept along one segment.

    Axial clamping retains the exact input endpoints while overlapping ball
    envelopes eliminate planar ribs at angular samples and branch nodes.
    """
    direction = p1 - p0
    length_squared = float(np.dot(direction, direction))
    dx, dy, dz = x - p0[0], y - p0[1], z - p0[2]
    fraction = np.clip(
        (dx * direction[0] + dy * direction[1] + dz * direction[2]) /
        length_squared, 0.0, 1.0)
    cx = p0[0] + fraction * direction[0]
    cy = p0[1] + fraction * direction[1]
    cz = p0[2] + fraction * direction[2]
    radius = r0 + fraction * (r1 - r0)
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius


def _update_union(field, origin, spacing, p0, p1, r0, r1,
                  union_mode="hard", blend_radius=0.0,
                  slab_voxels=2_000_000):
    influence = max(r0, r1) + blend_radius + 2.0 * spacing
    lower, upper = np.minimum(p0, p1) - influence, np.maximum(p0, p1) + influence
    first = np.maximum(0, np.floor((lower - origin) / spacing).astype(int))
    last = np.minimum(np.asarray(field.shape) - 1,
                      np.ceil((upper - origin) / spacing).astype(int))
    if np.any(last < first):
        return
    nx, ny = int(last[0] - first[0] + 1), int(last[1] - first[1] + 1)
    z_chunk = max(1, int(slab_voxels // max(1, nx * ny)))
    xs = origin[0] + np.arange(first[0], last[0] + 1) * spacing
    ys = origin[1] + np.arange(first[1], last[1] + 1) * spacing
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    for z_first in range(first[2], last[2] + 1, z_chunk):
        z_last = min(last[2] + 1, z_first + z_chunk)
        zs = origin[2] + np.arange(z_first, z_last) * spacing
        values = _polyball_segment_field(
            x_grid[:, :, None], y_grid[:, :, None], zs[None, None, :],
            p0, p1, r0, r1)
        target = field[first[0]:last[0] + 1, first[1]:last[1] + 1,
                       z_first:z_last]
        if union_mode == "hard":
            np.minimum(target, values, out=target)
        else:
            difference = np.abs(target - values)
            h = np.maximum(blend_radius - difference, 0.0) / blend_radius
            target[...] = np.minimum(target, values) - 0.25 * blend_radius * h * h


def hard_union_mesh(graph, edge_filter=None, min_radius=0.0,
                    radius_scale=1.0, resolution=10, voxel_size=None,
                    max_voxels=32_000_000, union_mode="hard",
                    blend_radius=None):
    """Create a watertight union mesh from an unmodified spatial graph."""
    from skimage.measure import marching_cubes
    import open3d as o3d

    aliases = {"hard": "hard", "hard_union": "hard",
               "smooth": "smooth", "smooth_union": "smooth"}
    if union_mode not in aliases:
        raise ValueError("union_mode must be 'hard' or 'smooth'")
    union_mode = aliases[union_mode]
    p0, p1, r0, r1 = _graph_segments(
        graph, edge_filter=edge_filter, min_radius=min_radius,
        radius_scale=radius_scale)
    if p0.shape[0] == 0:
        return None
    if blend_radius is None:
        blend_radius = 0.25 * float(np.min(np.minimum(r0, r1)))
    blend_radius = float(blend_radius)
    if union_mode == "hard":
        blend_radius = 0.0
    elif not np.isfinite(blend_radius) or blend_radius <= 0.0:
        raise ValueError("blend_radius must be positive for a smooth union")

    origin, shape, spacing = _grid_spec(
        p0, p1, r0, r1, resolution, voxel_size, int(max_voxels),
        extra_padding=blend_radius)
    field = np.full(shape, np.float32(np.inf), dtype=np.float32)
    for start, end, start_radius, end_radius in zip(p0, p1, r0, r1):
        _update_union(field, origin, spacing, start, end,
                      float(start_radius), float(end_radius),
                      union_mode=union_mode, blend_radius=blend_radius)
    finite = np.isfinite(field)
    if not np.any(finite) or np.min(field[finite]) > 0.0:
        raise ValueError("voxel_size is too large to resolve any included vessel")
    field[~finite] = float(np.max(r0) + blend_radius + 2.0 * spacing)
    vertices, faces, _, _ = marching_cubes(
        field, level=0.0, spacing=(spacing,) * 3, allow_degenerate=False)
    vertices += origin
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


def mesh_is_watertight(mesh):
    """Return whether every undirected mesh edge has exactly two faces."""
    if mesh is None:
        return False
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if triangles.size == 0:
        return False
    edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]],
                               triangles[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))
