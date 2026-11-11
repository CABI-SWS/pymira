"""GVars.insert_node_in_edge edits storage in place; it must leave exactly the state the reference implementation leaves."""
import copy

import numpy as np

from pymira import spatialgraph


def _graph(rng, nedge=12, float32_points=True):
    """A tree of edges with several points each (so splits fall in the middle), EdgeLabel/NodeLabel and extra scalars."""
    nodes, conn, pts, nep = [np.array([0., 0., 0.])], [], [], []
    for e in range(nedge):
        parent = int(rng.integers(0, len(nodes)))
        direction = rng.normal(size=3)
        n = int(rng.integers(2, 7))
        steps = np.cumsum(rng.uniform(5., 30., (n - 1, 3)) * np.sign(direction), axis=0)
        p = np.vstack([nodes[parent], nodes[parent] + steps])
        nodes.append(p[-1])
        conn.append([parent, len(nodes) - 1])
        pts.append(p)
        nep.append(n)
    nodes, conn, pts = np.array(nodes), np.array(conn), np.vstack(pts)
    npoint = pts.shape[0]
    g = spatialgraph.SpatialGraph(initialise=True, scalars=['Radii', 'VesselType', 'EdgeLabel', 'Plexus', 'Shyness'], node_scalars=['NodeLabel', 'NodeActive'])
    g.set_definition_size('VERTEX', nodes.shape[0]); g.set_definition_size('EDGE', len(conn)); g.set_definition_size('POINT', npoint)
    g.set_data(nodes, name='VertexCoordinates'); g.set_data(conn, name='EdgeConnectivity')
    g.set_data(pts, name='EdgePointCoordinates'); g.set_data(np.array(nep), name='NumEdgePoints')
    g.set_data(rng.uniform(1., 50., npoint), name='Radii')
    g.set_data(rng.integers(0, 2, npoint), name='VesselType')
    g.set_data(np.repeat(np.arange(len(conn)), nep), name='EdgeLabel')
    g.set_data(rng.integers(0, 3, npoint), name='Plexus')
    g.set_data(rng.integers(0, 3, npoint), name='Shyness')
    g.set_graph_sizes(labels=True)
    g.set_data(rng.integers(0, 2, nodes.shape[0]).astype(bool), name='NodeActive')
    return g


def _state(gv):
    out = {'ptrs': (gv.node_ptr, gv.edge_ptr, gv.edgepnt_ptr)}
    nn, ne, npnt = gv._prefix_counts()
    out['nodecoords'] = gv.nodecoords[:nn]
    out['edgeconn'] = gv.edgeconn[:ne]
    out['nedgepoints'] = gv.nedgepoints[:ne]
    out['edgepoints'] = gv.edgepoints[:npnt]
    for i, s in enumerate(gv.scalar_values):
        out[f'scalar{i}'] = s[:npnt]
    for i, s in enumerate(gv.node_scalar_values):
        out[f'node_scalar{i}'] = s[:nn]
    out['dtypes'] = (gv.nodecoords.dtype, gv.edgepoints.dtype, gv.edgeconn.dtype) + tuple(s.dtype for s in gv.scalar_values)
    out['shapes'] = (gv.nodecoords.shape, gv.edgeconn.shape, gv.edgepoints.shape)
    out['masks'] = (np.count_nonzero(gv.nodecoords_allocated), np.count_nonzero(gv.edgeconn_allocated), np.count_nonzero(gv.edgepoints_allocated))
    return out


def _same(a, b, label):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), f'{label}: {k}'
        else:
            assert a[k] == b[k], f'{label}: {k}: {a[k]} vs {b[k]}'


def _run(seed, n_all, n_ops, use_newpoint, interleave=True):
    rng = np.random.default_rng(seed)
    g = _graph(rng)
    g_ref, g_new = copy.deepcopy(g), copy.deepcopy(g)
    ref, new = spatialgraph.GVars(g_ref, n_all=n_all), spatialgraph.GVars(g_new, n_all=n_all)
    for op in range(n_ops):
        ne = ref.edge_ptr
        edge = int(rng.integers(0, ne))
        counts = ref.nedgepoints[:ne]
        x0 = int(np.sum(counts[:edge]))
        pts = ref.edgepoints[x0:x0 + int(counts[edge])]
        if use_newpoint:
            seg = int(rng.integers(0, pts.shape[0] - 1))
            f = rng.uniform(0.05, 0.95)
            newpoint = (pts[seg] + f * (pts[seg + 1] - pts[seg])).astype(np.float64) + rng.normal(scale=0.3, size=3)
            kw = dict(newpoint=newpoint, fr=None)
        else:
            kw = dict(fr=float(rng.uniform(0.1, 0.9)))
        r_ref = ref._insert_node_in_edge_reference(edge, 1, **kw)
        r_new = new.insert_node_in_edge(edge, 1, **kw)
        for a, b in zip(r_ref, r_new):
            if a is None:
                assert b is None
            else:
                a, b = np.asarray(a), np.asarray(b)
                assert a.dtype == b.dtype and np.array_equal(a, b), f'return values, op {op}'
        _same(_state(ref), _state(new), f'seed {seed} op {op}')
        if interleave and op % 7 == 0:        # interleave appending a plain edge as the CCO loop does
            for gv in (ref, new):
                gv.add_node(np.array([1., 2., 3.]), new_scalar_vals=[gv.graph.unique_node_label() if s['name'] == 'NodeLabel' else True for s in gv.node_scalars])
                gv.add_edge(0, gv.node_ptr - 1, [1., 0, 1000 + op, 0, 0], points=np.array([gv.nodecoords[0], gv.nodecoords[gv.node_ptr - 1]]))
            _same(_state(ref), _state(new), f'seed {seed} after add_edge {op}')
    # set_in_graph must hand the graph identical data
    ga, gb = ref.set_in_graph(), new.set_in_graph()
    for f in ga.fieldNames:
        da, db = ga.get_data(f), gb.get_data(f)
        assert da.dtype == db.dtype and np.array_equal(da, db), f
    assert (ga.nnode, ga.nedge) == (gb.nnode, gb.nedge)


def test_insert_node_in_edge_matches_reference_with_fraction():
    for seed in range(4):
        _run(seed, n_all=3, n_ops=60, use_newpoint=False)


def test_insert_node_in_edge_matches_reference_with_newpoint():
    for seed in range(4):
        _run(seed + 10, n_all=500, n_ops=60, use_newpoint=True)


def test_insert_node_in_edge_matches_reference_with_tiny_preallocation():
    _run(99, n_all=1, n_ops=80, use_newpoint=True, interleave=False)
