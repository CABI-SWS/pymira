"""Numba kernel for batched hard-union field evaluation and polygonisation."""

import numpy as np


CORNER_X = np.asarray([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)
CORNER_Y = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int32)
CORNER_Z = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
EDGE_START = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
                        dtype=np.int32)
EDGE_END = np.asarray([1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7],
                      dtype=np.int32)


def classic_triangle_table():
    from skimage.measure import _marching_cubes_lewiner as marching
    from skimage.measure import _marching_cubes_lewiner_luts as lookup
    return np.ascontiguousarray(marching._to_array(lookup.CASESCLASSIC),
                                dtype=np.int8)


def make_compiled_kernel():
    """Create lazily so importing PyMIRA does not require Numba."""
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def mesh_batch(tile_coordinates, reference_offsets, reference_indices,
                   p0, p1, r0, r1, origin, grid_shape, spacing, tile_cells,
                   triangle_table):
        tile_count = tile_coordinates.shape[0]
        max_vertices = 3 * tile_cells * (tile_cells + 1) * (tile_cells + 1)
        max_faces = 5 * tile_cells * tile_cells * tile_cells
        vertex_counts = np.zeros(tile_count, dtype=np.int32)
        face_counts = np.zeros(tile_count, dtype=np.int32)
        vertex_edges = np.empty((tile_count, max_vertices, 4), dtype=np.int32)
        vertex_positions = np.empty(
            (tile_count, max_vertices, 3), dtype=np.float32)
        faces = np.empty((tile_count, max_faces, 3), dtype=np.int32)

        for tile_index in prange(tile_count):
            start_x = tile_coordinates[tile_index, 0] * tile_cells
            start_y = tile_coordinates[tile_index, 1] * tile_cells
            start_z = tile_coordinates[tile_index, 2] * tile_cells
            cells_x = min(tile_cells, grid_shape[0] - 1 - start_x)
            cells_y = min(tile_cells, grid_shape[1] - 1 - start_y)
            cells_z = min(tile_cells, grid_shape[2] - 1 - start_z)
            if cells_x <= 0 or cells_y <= 0 or cells_z <= 0:
                continue

            field = np.full(
                (tile_cells + 1, tile_cells + 1, tile_cells + 1),
                np.float32(np.inf), dtype=np.float32)
            tile_origin_x = origin[0] + start_x * spacing
            tile_origin_y = origin[1] + start_y * spacing
            tile_origin_z = origin[2] + start_z * spacing

            for reference_position in range(
                    reference_offsets[tile_index],
                    reference_offsets[tile_index + 1]):
                segment_index = reference_indices[reference_position]
                ax, ay, az = p0[segment_index]
                bx, by, bz = p1[segment_index]
                radius_a = r0[segment_index]
                radius_b = r1[segment_index]
                influence = max(radius_a, radius_b) + 2.0 * spacing
                first_x = max(0, int(np.floor(
                    (min(ax, bx) - influence - tile_origin_x) / spacing)))
                first_y = max(0, int(np.floor(
                    (min(ay, by) - influence - tile_origin_y) / spacing)))
                first_z = max(0, int(np.floor(
                    (min(az, bz) - influence - tile_origin_z) / spacing)))
                last_x = min(cells_x, int(np.ceil(
                    (max(ax, bx) + influence - tile_origin_x) / spacing)))
                last_y = min(cells_y, int(np.ceil(
                    (max(ay, by) + influence - tile_origin_y) / spacing)))
                last_z = min(cells_z, int(np.ceil(
                    (max(az, bz) + influence - tile_origin_z) / spacing)))
                if last_x < first_x or last_y < first_y or last_z < first_z:
                    continue
                dx, dy, dz = bx - ax, by - ay, bz - az
                length_squared = dx * dx + dy * dy + dz * dz
                radius_delta = radius_b - radius_a
                for local_x in range(first_x, last_x + 1):
                    world_x = tile_origin_x + local_x * spacing
                    offset_x = world_x - ax
                    for local_y in range(first_y, last_y + 1):
                        world_y = tile_origin_y + local_y * spacing
                        offset_y = world_y - ay
                        for local_z in range(first_z, last_z + 1):
                            world_z = tile_origin_z + local_z * spacing
                            offset_z = world_z - az
                            fraction = (offset_x * dx + offset_y * dy +
                                        offset_z * dz) / length_squared
                            fraction = min(1.0, max(0.0, fraction))
                            centre_x = ax + fraction * dx
                            centre_y = ay + fraction * dy
                            centre_z = az + fraction * dz
                            radius = radius_a + fraction * radius_delta
                            ex = world_x - centre_x
                            ey = world_y - centre_y
                            ez = world_z - centre_z
                            value = np.sqrt(ex * ex + ey * ey + ez * ez) - radius
                            if value < field[local_x, local_y, local_z]:
                                field[local_x, local_y, local_z] = value

            x_ids = np.full(
                (tile_cells, tile_cells + 1, tile_cells + 1), -1,
                dtype=np.int32)
            y_ids = np.full(
                (tile_cells + 1, tile_cells, tile_cells + 1), -1,
                dtype=np.int32)
            z_ids = np.full(
                (tile_cells + 1, tile_cells + 1, tile_cells), -1,
                dtype=np.int32)
            number_vertices = 0
            number_faces = 0

            for local_x in range(cells_x):
                for local_y in range(cells_y):
                    for local_z in range(cells_z):
                        case = 0
                        for corner in range(8):
                            if field[local_x + CORNER_X[corner],
                                     local_y + CORNER_Y[corner],
                                     local_z + CORNER_Z[corner]] < 0.0:
                                case |= 1 << corner
                        if case == 0 or case == 255:
                            continue
                        table_position = 0
                        while (table_position < 16 and
                               triangle_table[case, table_position] >= 0):
                            for triangle_corner in range(3):
                                edge = int(triangle_table[
                                    case, table_position + triangle_corner])
                                corner_a = EDGE_START[edge]
                                corner_b = EDGE_END[edge]
                                ax_local = local_x + CORNER_X[corner_a]
                                ay_local = local_y + CORNER_Y[corner_a]
                                az_local = local_z + CORNER_Z[corner_a]
                                bx_local = local_x + CORNER_X[corner_b]
                                by_local = local_y + CORNER_Y[corner_b]
                                bz_local = local_z + CORNER_Z[corner_b]
                                lower_x = min(ax_local, bx_local)
                                lower_y = min(ay_local, by_local)
                                lower_z = min(az_local, bz_local)
                                if ax_local != bx_local:
                                    axis = 0
                                    vertex_id = x_ids[lower_x, lower_y, lower_z]
                                elif ay_local != by_local:
                                    axis = 1
                                    vertex_id = y_ids[lower_x, lower_y, lower_z]
                                else:
                                    axis = 2
                                    vertex_id = z_ids[lower_x, lower_y, lower_z]
                                if vertex_id < 0:
                                    vertex_id = number_vertices
                                    number_vertices += 1
                                    if axis == 0:
                                        x_ids[lower_x, lower_y, lower_z] = vertex_id
                                    elif axis == 1:
                                        y_ids[lower_x, lower_y, lower_z] = vertex_id
                                    else:
                                        z_ids[lower_x, lower_y, lower_z] = vertex_id
                                    value_a = field[ax_local, ay_local, az_local]
                                    value_b = field[bx_local, by_local, bz_local]
                                    denominator = value_a - value_b
                                    fraction = 0.5
                                    if abs(denominator) > 1e-20:
                                        fraction = value_a / denominator
                                    fraction = min(1.0, max(0.0, fraction))
                                    vertex_edges[tile_index, vertex_id, 0] = lower_x
                                    vertex_edges[tile_index, vertex_id, 1] = lower_y
                                    vertex_edges[tile_index, vertex_id, 2] = lower_z
                                    vertex_edges[tile_index, vertex_id, 3] = axis
                                    vertex_positions[tile_index, vertex_id, 0] = (
                                        tile_origin_x + (ax_local + fraction *
                                        (bx_local - ax_local)) * spacing)
                                    vertex_positions[tile_index, vertex_id, 1] = (
                                        tile_origin_y + (ay_local + fraction *
                                        (by_local - ay_local)) * spacing)
                                    vertex_positions[tile_index, vertex_id, 2] = (
                                        tile_origin_z + (az_local + fraction *
                                        (bz_local - az_local)) * spacing)
                                faces[tile_index, number_faces,
                                      triangle_corner] = vertex_id
                            number_faces += 1
                            table_position += 3
            vertex_counts[tile_index] = number_vertices
            face_counts[tile_index] = number_faces
        return (vertex_counts, face_counts, vertex_edges,
                vertex_positions, faces)

    return mesh_batch
