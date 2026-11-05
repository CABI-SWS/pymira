"""Sparse tiled meshing dispatch; compiled hard union is the default."""

from pymira.compiled_tube_mesh_fast import compiled_hard_union_to_ply
from pymira import tiled_tube_mesh_core as _legacy


def tiled_hard_union_to_ply(
        graph, ofile, edge_filter=None, min_radius=0.0, radius_scale=1.0,
        resolution=10, voxel_size=None, tile_cells=64,
        union_mode="hard", blend_radius=None,
        max_tile_references=50_000_000, tile_workers=None,
        batch_tiles=64, verbose=True):
    aliases = {"hard": "hard", "hard_union": "hard",
               "smooth": "smooth", "smooth_union": "smooth"}
    if union_mode not in aliases:
        raise ValueError("union_mode must be 'hard' or 'smooth'")
    if aliases[union_mode] == "hard":
        return compiled_hard_union_to_ply(
            graph, ofile, edge_filter=edge_filter, min_radius=min_radius,
            radius_scale=radius_scale, resolution=resolution,
            voxel_size=voxel_size, tile_cells=tile_cells,
            max_tile_references=max_tile_references,
            batch_tiles=batch_tiles, verbose=verbose)
    return _legacy.tiled_hard_union_to_ply(
        graph, ofile, edge_filter=edge_filter, min_radius=min_radius,
        radius_scale=radius_scale, resolution=resolution,
        voxel_size=voxel_size, tile_cells=tile_cells,
        union_mode="smooth", blend_radius=blend_radius,
        max_tile_references=max_tile_references,
        tile_workers=tile_workers, verbose=verbose)


_tile_index = _legacy._tile_index
_boundary_key = _legacy._boundary_key
_write_binary_ply = _legacy._write_binary_ply
_segment_intersects_expanded_tile = _legacy._segment_intersects_expanded_tile

__all__ = ["tiled_hard_union_to_ply", "compiled_hard_union_to_ply"]
