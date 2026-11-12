"""Fast viewer for very large Amira spatial graphs (millions of edges).

Why it is fast
  * Only the fields needed for display are parsed (edge points, point counts, radii, labels).
  * The parsed arrays are cached next to the graph as `<file>.fastview.npz`; later opens skip the text parse.
  * Vessels are drawn as one GPU line set (no per-segment Python objects). Vessels above a radius
    threshold are optionally drawn as real tubes, built in one vectorised pass.
  * A segment budget keeps the view interactive: all large vessels are kept, the smallest are randomly thinned.

Usage
  python -m pymira.fast_view graph.am
  python -m pymira.fast_view graph.am --color radius --tubes-above 8 --max-segments 4000000
  python -m pymira.fast_view graph.am --vessel-type 0 --min-radius 3
  python -m pymira.fast_view graph.am --info
  python -m pymira.fast_view graph.am --screenshot out.png      (renders, saves, exits)
"""
import argparse
import os
import sys
import time

import numpy as np

RADIUS_NAMES = ('Radii', 'radii', 'Radius', 'radius', 'thickness')
OPTIONAL = ('VesselType', 'Plexus', 'Organ')
CACHE_VERSION = 1


def _cache_path(path):
    return path + '.fastview.npz'


def load_graph(path, use_cache=True, verbose=True):
    """Return dict(points[N,3] f4, npts[E] i4, radius[N] f4, and optional point labels)."""
    t0 = time.time()
    cache = _cache_path(path)
    st = os.stat(path)
    key = np.array([CACHE_VERSION, st.st_size, int(st.st_mtime)], dtype=np.int64)
    if use_cache and os.path.exists(cache):
        try:
            z = np.load(cache)
            if np.array_equal(z['key'], key):
                d = {k: z[k] for k in z.files if k != 'key'}
                if verbose:
                    print(f'loaded cache {os.path.basename(cache)} in {time.time()-t0:.1f}s')
                return d
        except Exception as e:
            print('cache unreadable, re-reading file:', e)

    from pymira import amiramesh
    am = amiramesh.AmiraMesh()
    am.read(path, quiet=True, only=('EdgePointCoordinates', 'NumEdgePoints') + RADIUS_NAMES + OPTIONAL)
    fields = {f['name']: f['data'] for f in am.fields}
    if verbose:
        print(f'parsed {os.path.basename(path)} in {time.time()-t0:.1f}s ({", ".join(sorted(fields))})')
    d = {'points': np.asarray(fields['EdgePointCoordinates'], dtype=np.float32).reshape(-1, 3),
         'npts': np.asarray(fields['NumEdgePoints']).reshape(-1).astype(np.int32)}
    rname = next((n for n in RADIUS_NAMES if n in fields), None)
    d['radius'] = (np.asarray(fields[rname], dtype=np.float32).reshape(-1) if rname
                   else np.ones(d['points'].shape[0], np.float32))
    for n in OPTIONAL:
        if n in fields:
            d[n] = np.asarray(fields[n]).reshape(-1).astype(np.int32 if n != 'VesselType' else np.int8)
    if use_cache:
        try:
            np.savez(cache, key=key, **d)
        except Exception as e:
            print('could not write cache:', e)
    return d


def segments(d):
    """Vectorised segment list: (index of first point, index of second point) for every consecutive pair inside an edge."""
    npts = d['npts'].astype(np.int64)
    n = int(npts.sum())
    ends = np.cumsum(npts)
    keep = np.ones(max(n - 1, 0), dtype=bool)
    last = ends[:-1] - 1                      # last point of every edge but the final one
    keep[last[(last >= 0) & (last < n - 1)]] = False
    i0 = np.nonzero(keep)[0]
    return i0, i0 + 1


def select(d, i0, i1, min_radius=0., vessel_type=None, plexus=None, max_segments=None, seed=0, protect_radius=None):
    """Filter segments and thin to a budget. Returns the kept segment mask."""
    r = np.maximum(d['radius'][i0], d['radius'][i1])
    m = r >= min_radius
    if vessel_type is not None:
        if 'VesselType' not in d:
            print('no VesselType field; ignoring --vessel-type')
        else:
            m &= d['VesselType'][i0] == vessel_type
    if plexus is not None:
        if 'Plexus' not in d:
            print('no Plexus field; ignoring --plexus')
        else:
            m &= d['Plexus'][i0] == plexus
    if max_segments is not None and m.sum() > max_segments:
        idx = np.nonzero(m)[0]
        order = np.argsort(-r[idx], kind='stable')          # largest first
        n_keep_big = int(max_segments * 0.25) if protect_radius is None else int((r[idx] >= protect_radius).sum())
        n_keep_big = min(n_keep_big, int(max_segments * 0.75))
        big = idx[order[:n_keep_big]]
        rest = idx[order[n_keep_big:]]
        rng = np.random.default_rng(seed)
        rest = rng.choice(rest, size=max_segments - n_keep_big, replace=False)
        m = np.zeros_like(m)
        m[big] = True
        m[rest] = True
        print(f'thinned to {m.sum():,} segments: kept the {n_keep_big:,} largest, sampled the rest '
              f'({100.*(max_segments-n_keep_big)/max(len(idx)-n_keep_big,1):.1f}% of the small vessels)')
    return m


def colours(d, i0, mode):
    n = i0.size
    if mode == 'type' and 'VesselType' in d:
        vt = d['VesselType'][i0]
        c = np.full((n, 3), 0.6, dtype=np.float32)
        c[vt == 0] = (1., 0.15, 0.15)
        c[vt == 1] = (0.2, 0.35, 1.)
        c[vt == 2] = (0.65, 0.65, 0.65)
        return c
    import matplotlib
    cm = matplotlib.colormaps['turbo']
    if mode in ('plexus', 'organ'):
        key = 'Plexus' if mode == 'plexus' else 'Organ'
        if key in d:
            v = d[key][i0].astype(np.float32)
            lo, hi = float(v.min()), float(v.max())
            return cm((v - lo) / max(hi - lo, 1e-9))[:, :3].astype(np.float32)
    r = d['radius'][i0]
    lr = np.log10(np.maximum(r, 1e-3))
    lo, hi = np.percentile(lr, 1), np.percentile(lr, 99.5)
    return cm(np.clip((lr - lo) / max(hi - lo, 1e-9), 0, 1))[:, :3].astype(np.float32)


def tube_mesh(p0, p1, radius, col, sides=8):
    """One triangle mesh with a tube per segment, built without Python loops."""
    import open3d as o3d
    s = p0.shape[0]
    d = p1 - p0
    ln = np.linalg.norm(d, axis=1)
    ok = ln > 0
    p0, p1, d, ln, radius, col = p0[ok], p1[ok], d[ok], ln[ok], radius[ok], col[ok]
    s = p0.shape[0]
    d = d / ln[:, None]
    helper = np.where(np.abs(d[:, [2]]) < 0.9, np.array([[0, 0, 1.]]), np.array([[1., 0, 0]]))
    u = np.cross(d, helper)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = (np.cos(th)[None, :, None] * u[:, None, :] + np.sin(th)[None, :, None] * v[:, None, :]) * radius[:, None, None]
    a = (p0[:, None, :] + ring).reshape(-1, 3)
    b = (p1[:, None, :] + ring).reshape(-1, 3)
    verts = np.concatenate([a, b]).astype(np.float64)
    base = (np.arange(s) * sides)[:, None]
    k = np.arange(sides)[None, :]
    k1 = (k + 1) % sides
    a0, a1 = base + k, base + k1
    b0, b1 = a0 + s * sides, a1 + s * sides
    tris = np.concatenate([np.stack([a0, a1, b0], -1).reshape(-1, 3), np.stack([a1, b1, b0], -1).reshape(-1, 3)])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris.astype(np.int32)))
    vc = np.repeat(col, sides, axis=0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.concatenate([vc, vc]).astype(np.float64))
    mesh.compute_vertex_normals()
    return mesh


def build_geometry(d, args):
    import open3d as o3d
    t0 = time.time()
    i0, i1 = segments(d)
    m = select(d, i0, i1, min_radius=args.min_radius, vessel_type=args.vessel_type, plexus=args.plexus,
               max_segments=args.max_segments, protect_radius=args.tubes_above)
    i0, i1 = i0[m], i1[m]
    pts = d['points']
    col = colours(d, i0, args.color)
    geoms = []
    r = np.maximum(d['radius'][i0], d['radius'][i1])

    line_sel = np.ones(i0.size, bool)
    if args.tubes_above is not None:
        big = np.nonzero(r >= args.tubes_above)[0]
        if big.size > args.max_tubes:
            big = big[np.argsort(-r[big])[:args.max_tubes]]
        if big.size:
            geoms.append(tube_mesh(pts[i0[big]].astype(np.float64), pts[i1[big]].astype(np.float64), r[big], col[big]))
            line_sel[big] = False
            print(f'{big.size:,} segments drawn as tubes (radius >= {args.tubes_above})')
    # line set over only the points that are used
    li0, li1 = i0[line_sel], i1[line_sel]
    if li0.size:
        used, inv = np.unique(np.concatenate([li0, li1]), return_inverse=True)
        lines = inv.reshape(2, -1).T.astype(np.int32)
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts[used].astype(np.float64)),
                                  o3d.utility.Vector2iVector(lines))
        ls.colors = o3d.utility.Vector3dVector(col[line_sel].astype(np.float64))
        geoms.append(ls)
    print(f'{i0.size:,} segments ready in {time.time()-t0:.1f}s')
    return geoms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('file', nargs='+', help='graph .am file (extra words are joined, for paths with spaces)')
    ap.add_argument('--color', choices=('type', 'radius', 'plexus', 'organ'), default=None)
    ap.add_argument('--min-radius', type=float, default=0.)
    ap.add_argument('--vessel-type', type=int, default=None, help='0 artery, 1 vein, 2 other')
    ap.add_argument('--plexus', type=int, default=None)
    ap.add_argument('--max-segments', type=int, default=8000000, help='thin the smallest vessels beyond this (0 = no limit)')
    ap.add_argument('--tubes-above', type=float, default=None, help='draw segments with radius >= this as tubes')
    ap.add_argument('--max-tubes', type=int, default=400000)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--info', action='store_true', help='print statistics and exit')
    ap.add_argument('--screenshot', default=None, help='save an image and exit')
    ap.add_argument('--bg', default='0,0,0')
    ap.add_argument('--size', default='1800,1100')
    args = ap.parse_args(argv)
    path = ' '.join(args.file)
    if args.max_segments == 0:
        args.max_segments = None

    d = load_graph(path, use_cache=not args.no_cache)
    if args.color is None:
        args.color = 'type' if 'VesselType' in d and len(np.unique(d['VesselType'][:1000000])) > 1 else 'radius'
    p, r = d['points'], d['radius']
    print(f'{d["npts"].size:,} edges, {p.shape[0]:,} points; radius pcts (1/50/99/100) '
          f'{np.percentile(r[::max(1,r.size//2000000)],[1,50,99]).round(1)} {r.max():.1f}')
    print('extent min', p.min(0).round(0), 'max', p.max(0).round(0))
    if args.info:
        return 0

    import open3d as o3d
    geoms = build_geometry(d, args)
    vis = o3d.visualization.Visualizer()
    w, h = (int(x) for x in args.size.split(','))
    vis.create_window('pymira fast view', width=w, height=h, visible=args.screenshot is None)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.array([float(x) for x in args.bg.split(',')])
        opt.mesh_show_back_face = True
    vis.get_view_control().set_zoom(0.7)
    if args.screenshot:
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(args.screenshot, do_render=True)
        vis.destroy_window()
        print('saved', args.screenshot)
    else:
        vis.run()
        vis.destroy_window()
    return 0


if __name__ == '__main__':
    sys.exit(main())
