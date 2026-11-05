import importlib.util
import unittest

import numpy as np

from pymira.tube_mesh import hard_union_mesh, mesh_is_watertight


HAS_MESH_DEPS = (importlib.util.find_spec("open3d") is not None and
                 importlib.util.find_spec("skimage") is not None)


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


@unittest.skipUnless(HAS_MESH_DEPS, "mesh dependencies are not installed")
class HardUnionMeshTests(unittest.TestCase):
    def test_bifurcation_is_watertight_and_manifold(self):
        graph = DummyGraph(
            [[0, 0, 0], [-3, 0, 0],
             [0, 0, 0], [2.5, 1.5, 0],
             [0, 0, 0], [2.5, -1.5, 0]],
            [2, 2, 2],
            [0.8, 1.0, 0.8, 0.55, 0.8, 0.5])
        mesh = hard_union_mesh(
            graph, voxel_size=0.15, max_voxels=2_000_000)
        self.assertTrue(mesh_is_watertight(mesh))
        self.assertTrue(mesh.is_edge_manifold())
        self.assertTrue(mesh.is_vertex_manifold())

    def test_hard_union_is_the_default(self):
        graph = DummyGraph([[0, 0, 0], [2, 0, 0]], [2], [0.5, 0.5])
        default_mesh = hard_union_mesh(
            graph, voxel_size=0.15, max_voxels=500_000)
        explicit_mesh = hard_union_mesh(
            graph, voxel_size=0.15, max_voxels=500_000,
            union_mode="hard")
        np.testing.assert_allclose(
            np.asarray(default_mesh.vertices),
            np.asarray(explicit_mesh.vertices))
        np.testing.assert_array_equal(
            np.asarray(default_mesh.triangles),
            np.asarray(explicit_mesh.triangles))


if __name__ == "__main__":
    unittest.main()
