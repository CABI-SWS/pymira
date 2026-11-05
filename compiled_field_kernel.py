"""Parallel Numba kernel for batches of hard-union scalar fields."""

import numpy as np


def make_field_kernel():
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def evaluate_fields(tile_coordinates, reference_offsets, reference_indices,
                        p0, p1, r0, r1, origin, grid_shape, spacing,
                        tile_cells):
        tile_count = tile_coordinates.shape[0]
        fields = np.full(
            (tile_count, tile_cells + 1, tile_cells + 1, tile_cells + 1),
            np.float32(np.inf), dtype=np.float32)
        cell_shapes = np.empty((tile_count, 3), dtype=np.int32)

        for tile_index in prange(tile_count):
            start_x = tile_coordinates[tile_index, 0] * tile_cells
            start_y = tile_coordinates[tile_index, 1] * tile_cells
            start_z = tile_coordinates[tile_index, 2] * tile_cells
            cells_x = min(tile_cells, grid_shape[0] - 1 - start_x)
            cells_y = min(tile_cells, grid_shape[1] - 1 - start_y)
            cells_z = min(tile_cells, grid_shape[2] - 1 - start_z)
            cell_shapes[tile_index, 0] = cells_x
            cell_shapes[tile_index, 1] = cells_y
            cell_shapes[tile_index, 2] = cells_z
            if cells_x <= 0 or cells_y <= 0 or cells_z <= 0:
                continue

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
                            if value < fields[
                                    tile_index, local_x, local_y, local_z]:
                                fields[tile_index, local_x, local_y, local_z] = value
        return fields, cell_shapes

    return evaluate_fields
