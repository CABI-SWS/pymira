"""Fast topology-preserving compiled hard-union exporter."""

from itertools import chain
from pathlib import Path
import tempfile
import time

import numpy as np

from pymira.compiled_field_kernel import make_field_kernel
from pymira.compiled_tube_mesh import _FACE_DTYPE, _write_binary_ply
from pymira.tube_mesh import _graph_segments
from pymira.tiled_tube_mesh_core import _tile_index


def _append_tile_mesh(vertices, faces, cell_start, cell_shape, spacing,
                      origin, boundary_vertices, vertex_stream, face_stream,
                      vertex_count, face_count):
    """Vectorise interior IDs; hash only vertices shared across tile faces."""
    local_grid = vertices / spacing
    global_grid = local_grid + cell_start
    world_vertices = (vertices + origin + cell_start * spacing).astype("<f4")
    boundary_mask = np.any(
        np.isclose(local_grid, 0.0, atol=1e-5) |
        np.isclose(local_grid, cell_shape[None, :], atol=1e-5), axis=1)
    new_mask = np.ones(len(vertices), dtype=bool)
    local_to_global = np.empty(len(vertices), dtype=np.int64)
    new_boundary = []
    for local_index in np.flatnonzero(boundary_mask):
        key = tuple(np.rint(
            global_grid[local_index] * 100000.0).astype(np.int64))
        existing = boundary_vertices.get(key)
        if existing is None:
            new_boundary.append((key, int(local_index)))
        else:
            new_mask[local_index] = False
            local_to_global[local_index] = existing

    new_indices = np.flatnonzero(new_mask)
    next_count = vertex_count + len(new_indices)
    if next_count > np.iinfo(np.int32).max:
        raise OverflowError("PLY vertex count exceeds signed int32")
    local_to_global[new_indices] = np.arange(
        vertex_count, next_count, dtype=np.int64)
    for key, local_index in new_boundary:
        boundary_vertices[key] = int(local_to_global[local_index])
    if len(new_indices):
        world_vertices[new_indices].tofile(vertex_stream)

    global_faces = local_to_global[faces]
    valid = ((global_faces[:, 0] != global_faces[:, 1]) &
             (global_faces[:, 1] != global_faces[:, 2]) &
             (global_faces[:, 2] != global_faces[:, 0]))
    global_faces = global_faces[valid].astype("<i4", copy=False)
    records = np.empty(len(global_faces), dtype=_FACE_DTYPE)
    records["count"] = 3
    records["indices"] = global_faces
    records.tofile(face_stream)
    return next_count, face_count + len(global_faces)


def compiled_hard_union_to_ply(
        graph, ofile, edge_filter=None, min_radius=0.0, radius_scale=1.0,
        resolution=10, voxel_size=None, tile_cells=64,
        max_tile_references=50_000_000, batch_tiles=64, verbose=True):
    """Stream a hard polyball union with compiled fields and Lewiner MC."""
    from skimage.measure import marching_cubes

    if not ofile:
        raise ValueError("The compiled exporter requires an output filename")
    tile_cells, batch_tiles = int(tile_cells), int(batch_tiles)
    if tile_cells < 4 or batch_tiles < 1:
        raise ValueError("tile_cells must be >= 4 and batch_tiles must be >= 1")
    p0, p1, r0, r1 = _graph_segments(
        graph, edge_filter=edge_filter, min_radius=min_radius,
        radius_scale=radius_scale)
    if len(p0) == 0:
        return None
    p0 = np.ascontiguousarray(p0, dtype=np.float64)
    p1 = np.ascontiguousarray(p1, dtype=np.float64)
    r0 = np.ascontiguousarray(r0, dtype=np.float64)
    r1 = np.ascontiguousarray(r1, dtype=np.float64)
    minimum_radius = float(np.min(np.minimum(r0, r1)))
    maximum_radius = float(np.max(np.maximum(r0, r1)))
    if voxel_size is None:
        voxel_size = minimum_radius / max(3.0, float(resolution) / 3.0)
    spacing = float(voxel_size)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("voxel_size must be a finite positive number")
    padding = maximum_radius + 2.0 * spacing
    origin = np.ascontiguousarray(
        np.minimum(np.min(p0, axis=0), np.min(p1, axis=0)) - padding,
        dtype=np.float64)
    upper = np.maximum(np.max(p0, axis=0), np.max(p1, axis=0)) + padding
    grid_shape = np.ascontiguousarray(
        np.maximum(3, np.ceil((upper - origin) / spacing).astype(np.int64) + 1),
        dtype=np.int64)

    index_started = time.perf_counter()
    tiles, reference_count = _tile_index(
        p0, p1, r0, r1, origin, tuple(grid_shape), spacing, tile_cells,
        0.0, int(max_tile_references))
    ordered_tiles = sorted(tiles.items(), key=lambda item: len(item[1]))
    if verbose:
        print(f"Compiled sparse union: {len(p0):,} segments, "
              f"{len(ordered_tiles):,} active tiles, "
              f"{reference_count:,} tile references; index "
              f"{time.perf_counter() - index_started:.1f} s", flush=True)

    kernel = make_field_kernel()
    kernel(np.zeros((1, 3), dtype=np.int64), np.zeros(2, dtype=np.int64),
           np.zeros(0, dtype=np.int64), p0, p1, r0, r1, origin, grid_shape,
           spacing, tile_cells)
    output = Path(ofile)
    output.parent.mkdir(parents=True, exist_ok=True)
    boundary_vertices = {}
    vertex_count = face_count = surface_tiles = 0
    report_every = max(batch_tiles, min(10_000, len(ordered_tiles) // 100))
    next_report = report_every
    mesh_started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="pymira_compiled_mesh_") as temp_dir:
        vertex_path = Path(temp_dir) / "vertices.bin"
        face_path = Path(temp_dir) / "faces.bin"
        with open(vertex_path, "wb") as vertex_stream, \
                open(face_path, "wb") as face_stream:
            for batch_start in range(0, len(ordered_tiles), batch_tiles):
                batch = ordered_tiles[batch_start:batch_start + batch_tiles]
                tile_coordinates = np.asarray(
                    [item[0] for item in batch], dtype=np.int64)
                reference_offsets = np.zeros(len(batch) + 1, dtype=np.int64)
                for index, item in enumerate(batch):
                    reference_offsets[index + 1] = (
                        reference_offsets[index] + len(item[1]))
                reference_indices = np.fromiter(
                    chain.from_iterable(item[1] for item in batch),
                    dtype=np.int64, count=int(reference_offsets[-1]))
                fields, cell_shapes = kernel(
                    tile_coordinates, reference_offsets, reference_indices,
                    p0, p1, r0, r1, origin, grid_shape, spacing, tile_cells)

                for local_tile in range(len(batch)):
                    cell_shape = cell_shapes[local_tile]
                    if np.any(cell_shape <= 0):
                        continue
                    field = fields[
                        local_tile, :cell_shape[0] + 1,
                        :cell_shape[1] + 1, :cell_shape[2] + 1]
                    finite = np.isfinite(field)
                    if not np.any(finite):
                        continue
                    field[~finite] = maximum_radius + 2.0 * spacing
                    if np.min(field) > 0.0 or np.max(field) < 0.0:
                        continue
                    vertices, faces, _, _ = marching_cubes(
                        field, level=0.0, spacing=(spacing,) * 3,
                        allow_degenerate=False)
                    surface_tiles += 1
                    cell_start = tile_coordinates[local_tile] * tile_cells
                    vertex_count, face_count = _append_tile_mesh(
                        vertices, faces, cell_start, cell_shape, spacing,
                        origin, boundary_vertices, vertex_stream, face_stream,
                        vertex_count, face_count)

                visited = batch_start + len(batch)
                if verbose and visited >= next_report:
                    elapsed = time.perf_counter() - mesh_started
                    print(f"Compiled sparse union: visited {visited:,}/"
                          f"{len(ordered_tiles):,} active tiles "
                          f"({surface_tiles:,} surface tiles, "
                          f"{visited / max(elapsed, 1e-9):,.0f} tiles/s)",
                          flush=True)
                    next_report += report_every
        _write_binary_ply(output, vertex_path, face_path, vertex_count, face_count)

    result = {
        "vertices": vertex_count, "triangles": face_count,
        "active_tiles": len(ordered_tiles), "surface_tiles": surface_tiles,
        "tile_references": reference_count, "voxel_size": spacing,
        "tile_cells": tile_cells, "batch_tiles": batch_tiles,
        "ofile": str(output),
    }
    if verbose:
        print(f"Mesh written to {output} ({vertex_count:,} vertices, "
              f"{face_count:,} triangles)", flush=True)
    return result
