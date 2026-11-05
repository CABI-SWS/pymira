import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from pymira.tiled_tube_mesh import tiled_hard_union_to_ply
from pymira.tube_mesh import mesh_is_watertight


class DummyGraph:
    def __init__(self, points, counts, radii):
        self.nedge = len(counts)
        self._points = np.asarray(points, dtype=float)
        self._counts = np.asarray(counts, dtype=int)
        self._radii = np.asarray(radii, dtype=float)

    def get_data(self, name):
        return {"EdgePointCoordinates": self._points,
                "NumEdgePoints": self._counts}[name]

    def get_radius_field(self):
        return {"data": self._radii}


class TiledHardUnionTests(unittest.TestCase):
    def test_tile_boundaries_are_welded_at_bifurcation(self):
        graph = DummyGraph(
            [[0, 0, 0], [-3, 0, 0],
             [0, 0, 0], [2.5, 1.5, 0],
             [0, 0, 0], [2.5, -1.5, 0]],
            [2, 2, 2],
            [0.8, 1.0, 0.8, 0.55, 0.8, 0.5])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tiled.ply"
            result = tiled_hard_union_to_ply(
                graph, output, voxel_size=0.15, tile_cells=8,
                max_tile_references=100000, verbose=False)
            mesh = o3d.io.read_triangle_mesh(str(output))
        self.assertGreater(result["active_tiles"], 1)
        self.assertEqual(result["vertices"], len(mesh.vertices))
        self.assertEqual(result["triangles"], len(mesh.triangles))
        self.assertTrue(mesh_is_watertight(mesh))
        self.assertTrue(mesh.is_edge_manifold())
        self.assertTrue(mesh.is_vertex_manifold())

    def test_long_tube_crosses_many_tiles_without_seams(self):
        graph = DummyGraph(
            [[-5, 0, 0], [5, 0, 0]], [2], [0.5, 0.5])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "long_tube.ply"
            result = tiled_hard_union_to_ply(
                graph, output, voxel_size=0.1, tile_cells=6,
                max_tile_references=100000, verbose=False)
            mesh = o3d.io.read_triangle_mesh(str(output))
        self.assertGreater(result["active_tiles"], 10)
        self.assertTrue(mesh_is_watertight(mesh))


if __name__ == "__main__":
    unittest.main()
