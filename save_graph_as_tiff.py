from pymira.spatialgraph import SpatialGraph
import argparse
import numpy as np
import tifffile

# WORK IN PROGRESS!!!!
def save_graph_as_tiff(args):
    print("This function is a work in progress. It may not work as expected.")
    am_file = args.graph_file
    target_voxel_size = args.voxel_size
    shape = args.shape
    out_tiff = args.output_file

    g = SpatialGraph()
    g.read(am_file)

    # Scale graph coordinates from voxel indices into physical units using the supplied voxel size.
    if target_voxel_size is not None:
        scale_matrix = np.eye(4, dtype=float)
        scale_matrix[0, 0] = 1/target_voxel_size[0]
        scale_matrix[1, 1] = 1/target_voxel_size[1]
        scale_matrix[2, 2] = 1/target_voxel_size[2]
        g.scale_graph(tr=scale_matrix, radius_index=0)

    pts = g.get_data('EdgePointCoordinates')
    vtype = g.get_data('VesselType')
    radius_name = g.get_radius_field_name()
    radii = g.get_data(radius_name) if radius_name is not None else None
    if pts is None or vtype is None:
        raise ValueError('EdgePointCoordinates or VesselType missing in graph')


    # Output image shape is the fixed canvas size; graph coordinates are not distorted or recentered.
    if shape is not None:
        nx, ny, nz = shape
    else:
        # If no shape is given, build a minimal canvas that fits the scaled graph.
        max_x = int(np.ceil(pts[:,0].max()))
        max_y = int(np.ceil(pts[:,1].max()))
        max_z = int(np.ceil(pts[:,2].max()))
        nx = max(max_x + 1, 1)
        ny = max(max_y + 1, 1)
        nz = max(max_z + 1, 1)

    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError('Computed output volume shape is invalid: {}'.format((nx, ny, nz)))

    vol = np.zeros((nz, ny, nx), dtype=np.uint8)

    def world_to_index(pt):
        ix = int(np.round(pt[0]))
        iy = int(np.round(pt[1]))
        iz = int(np.round(pt[2]))
        return iz, iy, ix

    def mark_voxel_radius(center_pt, radius, vt):
        if radius is None or radius <= 0:
            iz,iy,ix = world_to_index(center_pt)
            vol[iz,iy,ix] = max(vol[iz,iy,ix], vt)
            return

        ix_c, iy_c, iz_c = world_to_index(center_pt)
        
        ix0 = max(0, int(np.round(ix_c - radius)))
        ix1 = min(nx - 1, int(np.round(ix_c + radius)))
        iy0 = max(0, int(np.round(iy_c - radius)))
        iy1 = min(ny - 1, int(np.round(iy_c + radius)))
        iz0 = max(0, int(np.round(iz_c - radius)))
        iz1 = min(nz - 1, int(np.round(iz_c + radius)))

        for iz in range(iz0, iz1 + 1):
            dz = (iz - iz_c) 
            dz2 = dz * dz
            for iy in range(iy0, iy1 + 1):
                dy = (iy - iy_c) 
                dy2 = dy * dy
                for ix in range(ix0, ix1 + 1):
                    dx = (ix - ix_c) 
                    if dx*dx + dy2 + dz2 <= radius * radius:
                        vol[iz, iy, ix] = max(vol[iz, iy, ix], vt)

    def draw_voxel_line(p1, p2, r1, r2, vt):
        i1 = np.array(world_to_index(p1), dtype=int)
        i2 = np.array(world_to_index(p2), dtype=int)
        delta = i2 - i1
        steps = int(np.max(np.abs(delta)))
        if steps == 0:
            mark_voxel_radius(p1, r1, vt)
            return
        for s in range(steps + 1):
            t = s / steps
            pos = p1 + (p2 - p1) * t
            radius = float(r1 + (r2 - r1) * t)
            mark_voxel_radius(pos, radius, vt)

    # Fill volume using edge-point connectivity
    nEdgePoints = g.get_data('NumEdgePoints')  # array: points per edge
    if nEdgePoints is None or radii is None:
        for p, vt in zip(pts, vtype):
            iz,iy,ix = world_to_index(p)
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                vol[iz,iy,ix] = max(vol[iz,iy,ix], int(vt) + 1)
    else:
        cum = np.cumsum(nEdgePoints)
        starts = np.concatenate(([0], cum[:-1]))
        for s,n in zip(starts, nEdgePoints):
            edge_pts = pts[s:s+n]
            edge_vt = vtype[s:s+n]
            edge_rads = radii[s:s+n]
            if edge_pts.shape[0] == 0:
                continue
            for i in range(edge_pts.shape[0] - 1):
                draw_voxel_line(edge_pts[i], edge_pts[i+1], float(edge_rads[i]), float(edge_rads[i+1]), int(edge_vt[i]) + 1)
            mark_voxel_radius(edge_pts[-1], float(edge_rads[-1]), int(edge_vt[-1]) + 1)

    # Write 3D TIFF (z,y,x)
    tifffile.imwrite(out_tiff, vol, photometric='minisblack')

# Example usage:
# save_graph_as_tiff('my_graph.am', 'out_stack.tiff', shape=(512,512,128))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert spatial graph to TIFF using fixed output shape and voxel scaling.")
    parser.add_argument("--graph_file", type=str, default=None,
                        help="Spatial graph file ending in .am.")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[1,1,1],
                        help="Voxel size in um per pixel along x y z. Graph coordinates in um are converted to pixel units.")
    parser.add_argument("--shape", type=int, nargs=3, default=None,
                        help="Shape of the output volume (nx ny nz). The graph is placed into this fixed canvas without rescaling to fill it.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output TIFF file.")
    args = parser.parse_args()
    save_graph_as_tiff(args)