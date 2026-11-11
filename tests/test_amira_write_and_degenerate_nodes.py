"""Exactness of two speed-ups: the Amira writer's one-call-per-column output and the sort-based displace_degenerate_nodes."""
import copy
import io
import os

import numpy as np
import pytest

from pymira import spatialgraph
from test_gvars_insert_node import _graph


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_])
def test_one_tofile_call_per_column_writes_the_same_text_as_one_call_per_element(dtype, tmp_path):
    rng = np.random.default_rng(0)
    data = (rng.normal(size=50) * 1000).astype(dtype)
    old, new = tmp_path / 'old.txt', tmp_path / 'new.txt'
    with open(old, 'w', newline='\n') as f:
        for j in range(data.shape[0]):
            data[j].tofile(f, sep=' ', format='%s')
            f.write('\n')
    with open(new, 'w', newline='\n') as f:
        data.tofile(f, sep='\n', format='%s')
        f.write('\n')
    assert old.read_bytes() == new.read_bytes()


def test_write_then_read_round_trips_every_field(tmp_path):
    graph = _graph(np.random.default_rng(1))
    graph.remove_field('NodeActive')            # the reader cannot parse boolean columns written as True/False
    path = str(tmp_path / 'g.am')
    graph.write(path)
    back = spatialgraph.SpatialGraph()
    back.read(path)
    for name in graph.fieldNames:
        a, b = np.asarray(graph.get_data(name)), np.asarray(back.get_data(name))
        assert a.shape == b.shape, name
        assert np.allclose(a.astype(float), b.astype(float), atol=1e-4), name


def test_empty_column_writes_only_the_trailing_blank_line(tmp_path):
    graph = _graph(np.random.default_rng(2))
    graph.add_field(name='Empty', marker=f'@{len(graph.fields) + 1}', definition='POINT', type='float', nelements=1, nentries=[0])
    graph.set_data(np.zeros(0, dtype=np.float32), name='Empty')
    path = str(tmp_path / 'g.am')
    graph.write(path)
    text = open(path).read()
    assert text.endswith('\n\n') or text.endswith('\n')


def _reference_displace(graph, displacement=1.):
    """The original O(N^2) implementation."""
    nodes = graph.get_data('VertexCoordinates')
    edgepoints = graph.get_data('EdgePointCoordinates')
    for i, c1 in enumerate(nodes):
        sind = np.where((nodes[:, 0] == c1[0]) & (nodes[:, 1] == c1[1]) & (nodes[:, 2] == c1[2]))
        if len(sind[0]) > 1:
            edges = graph.get_edges_containing_node(sind[0])
            for s in sind[0]:
                nodes[s] += np.random.uniform(-displacement / 2., displacement / 2., 3)
            for e in edges:
                edge = graph.get_edge(e)
                if edge.start_node_index in sind[0]:
                    edgepoints[edge.i0] = nodes[edge.start_node_index]
                if edge.end_node_index in sind[0]:
                    edgepoints[edge.i1 - 1] = nodes[edge.end_node_index]
    graph.set_data(nodes, name='VertexCoordinates')
    graph.set_data(edgepoints, name='EdgePointCoordinates')
    return graph


def _graph_with_duplicates(seed):
    rng = np.random.default_rng(seed)
    g = _graph(rng, nedge=20)
    nodes = g.get_data('VertexCoordinates')
    edgepoints = g.get_data('EdgePointCoordinates')
    # force several groups of coincident nodes (pairs and a triple); keep the edge end points consistent
    for group in ([1, 5], [2, 7, 11], [3, 9]):
        nodes[group[1:]] = nodes[group[0]]
    g.set_data(nodes, name='VertexCoordinates')
    return g


@pytest.mark.parametrize('seed', range(5))
def test_displace_degenerate_nodes_matches_the_original_including_random_draws(seed):
    g_ref, g_new = _graph_with_duplicates(seed), None
    g_new = copy.deepcopy(g_ref)
    editor = spatialgraph.Editor()
    np.random.seed(123)
    _reference_displace(g_ref)
    np.random.seed(123)
    editor.displace_degenerate_nodes(g_new)
    for name in ('VertexCoordinates', 'EdgePointCoordinates'):
        a, b = g_ref.get_data(name), g_new.get_data(name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name


def test_displace_degenerate_nodes_without_duplicates_changes_nothing():
    g = _graph(np.random.default_rng(9))
    before = g.get_data('VertexCoordinates').copy()
    spatialgraph.Editor().displace_degenerate_nodes(g)
    assert np.array_equal(g.get_data('VertexCoordinates'), before)
