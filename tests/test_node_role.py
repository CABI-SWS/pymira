"""Flow sources stored on the nodes (NodeRole): resolver, validation, edit rules and the radius heuristic's failure cases."""
import copy
import os
import tempfile

import numpy as np
import pytest

from pymira import spatialgraph
from pymira.combine_graphs import combine_graphs
from pymira.spatialgraph import ROLE_INLET, ROLE_OUTLET, delete_vertices


def av_graph(root_radius=60., stub_radius=10.):
    """artery 0-1-2 (root radius at node 0) with a blind stub 1-6, capillary 2-3, vein 3-4-5 (outlet node 5)."""
    nodes = np.array([[0., 0., 0.], [100., 0., 0.], [200., 0., 0.], [300., 0., 0.], [400., 0., 0.], [500., 0., 0.], [100., 100., 0.]])
    conn = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 6]])
    vt = np.array([0, 0, 2, 1, 1, 0])
    rad = np.array([root_radius, 40., 4., 40., root_radius, stub_radius])
    pts = np.array([nodes[c] for c in conn]).reshape(-1, 3)
    g = spatialgraph.SpatialGraph(initialise=True, scalars=['Radii', 'VesselType'], node_scalars=['NodeLabel'])
    g.set_definition_size('VERTEX', len(nodes))
    g.set_definition_size('EDGE', len(conn))
    g.set_definition_size('POINT', pts.shape[0])
    g.set_data(nodes, name='VertexCoordinates')
    g.set_data(conn, name='EdgeConnectivity')
    g.set_data(np.full(len(conn), 2, dtype=int), name='NumEdgePoints')
    g.set_data(pts, name='EdgePointCoordinates')
    g.set_data(np.repeat(rad, 2), name='Radii')
    g.set_data(np.repeat(vt, 2), name='VesselType')
    g.set_graph_sizes(labels=True)
    return g


def test_without_an_annotation_the_heuristic_is_used_and_sources_report_it():
    g = av_graph()
    assert [int(x) for x in g.identify_inlet_outlet()] == [0, 5]
    assert not g.has_source_annotation() and g.get_sources() == {'inlets': [], 'outlets': []}
    assert g.identify_sources() == {'inlets': [0], 'outlets': [5]}
    assert g.validate_sources()[0][0] == 'info'


def test_the_annotation_wins_over_a_heuristic_that_a_wide_blind_stub_or_a_second_root_edge_would_fool():
    wide_stub = av_graph(stub_radius=200.)
    assert int(wide_stub.identify_inlet_outlet()[0]) == 6               # the heuristic follows the widest free end
    wide_stub.set_sources(inlets=[0], outlets=[5])
    assert [int(x) for x in wide_stub.identify_inlet_outlet()] == [0, 5]
    assert any('heuristic would pick inlet 6' in m for s, m in wide_stub.validate_sources())   # ... and the audit says so


def test_ignore_filters_annotated_sources_and_missing_sides_come_back_as_none():
    g = av_graph().set_sources(inlets=[0], outlets=[5])
    assert g.identify_inlet_outlet(ignore=[0]) == (None, 5)
    assert g.identify_inlet_outlet(ignore=[5]) == (0, None)


def test_set_sources_replaces_previous_roles_and_rejects_bad_indices():
    g = av_graph().set_sources(inlets=[0], outlets=[5])
    g.set_sources(inlets=[6], outlets=[5])
    assert g.get_sources() == {'inlets': [6], 'outlets': [5]}
    g.set_sources(inlets=[0], clear=False)
    assert g.get_sources() == {'inlets': [0, 6], 'outlets': [5]}
    with pytest.raises(ValueError):
        g.set_sources(inlets=[99])


def test_validation_flags_wrong_vessel_type_isolated_nodes_and_unreachable_outlets():
    g = av_graph().set_sources(inlets=[5], outlets=[0])               # swapped: inlet on a vein, outlet on an artery
    errors = [m for s, m in g.validate_sources() if s == 'error']
    assert any('inlet node 5' in m for m in errors) and any('outlet node 0' in m for m in errors)
    ok = av_graph().set_sources(inlets=[0], outlets=[5])
    assert not [m for s, m in ok.validate_sources() if s in ('error', 'warning')]
    inner = av_graph().set_sources(inlets=[0], outlets=[1])           # outlet on the artery side
    assert any(s == 'error' for s, m in inner.validate_sources())
    missing = av_graph().set_sources(inlets=[0])
    assert any('no venous outlet' in m for s, m in missing.validate_sources())


def test_a_source_that_gains_a_second_edge_is_still_found_when_annotated_but_flagged():
    g = av_graph().set_sources(inlets=[0], outlets=[5])
    gv = spatialgraph.GVars(g)
    gv.add_node(np.array([0., 80., 0.]))
    new = gv.node_ptr - 1
    gv.add_edge(0, new, [30., 0], points=np.vstack([gv.nodecoords[0], gv.nodecoords[new]]))
    g = gv.set_in_graph()
    assert [int(x) for x in g.identify_inlet_outlet()] == [0, 5]
    assert any('node 0 has 2 edges' in m for s, m in g.validate_sources())


@pytest.mark.parametrize('inlet_is_start', [True, False])
@pytest.mark.parametrize('method', ['insert_node_in_edge', '_insert_node_in_edge_reference'])
def test_a_node_created_by_splitting_the_inlet_edge_does_not_inherit_the_role(inlet_is_start, method):
    g = av_graph().set_sources(inlets=[0], outlets=[5])
    conn = np.asarray(g.get_data('EdgeConnectivity')).copy()
    if inlet_is_start != (conn[0, 0] == 0):
        pts = np.asarray(g.get_data('EdgePointCoordinates')).copy()
        pts[0:2] = pts[0:2][::-1]
        g.set_data(pts, name='EdgePointCoordinates')
        conn[0] = conn[0][::-1]
        g.set_data(conn, name='EdgeConnectivity')
    gv = spatialgraph.GVars(g)
    getattr(gv, method)(0, 1, fr=0.5)
    out = gv.set_in_graph()
    assert out.get_sources() == {'inlets': [0], 'outlets': [5]}


def test_added_nodes_start_without_a_role():
    g = av_graph().set_sources(inlets=[0], outlets=[5])
    gv = spatialgraph.GVars(g)
    gv.add_node(np.array([9., 9., 9.]))
    assert gv.set_in_graph().get_sources() == {'inlets': [0], 'outlets': [5]}


def test_roles_follow_their_nodes_through_deletions_and_removals():
    g = av_graph().set_sources(inlets=[0], outlets=[5])
    keep = np.ones(g.nnode, dtype=bool)
    keep[[6]] = False
    out = delete_vertices(copy.deepcopy(g), keep)
    out = g if out is None else out
    assert out.get_sources() == {'inlets': [0], 'outlets': [5]}
    g2 = av_graph(); g2.set_sources(inlets=[2 + 4], outlets=[5])            # roles on nodes that move when node 3 goes
    gv = spatialgraph.GVars(g2)
    gv.remove_nodes(np.array([3]))
    assert gv.set_in_graph().get_sources()['outlets'] == [4]                  # old node 5 is node 4 now


def test_combination_keeps_both_organs_sources_and_pads_a_field_only_one_graph_has():
    a, b = av_graph().set_sources(inlets=[0], outlets=[5]), av_graph().set_sources(inlets=[0], outlets=[5])
    c = combine_graphs(a, b)
    assert c.get_sources() == {'inlets': [0, 7], 'outlets': [5, 12]}
    only_first = combine_graphs(av_graph().set_sources(inlets=[0], outlets=[5]), av_graph())
    assert len(only_first.get_data('NodeRole')) == only_first.nnode == 14 and only_first.get_sources() == {'inlets': [0], 'outlets': [5]}
    only_second = combine_graphs(av_graph(), av_graph().set_sources(inlets=[0], outlets=[5]))
    assert only_second.get_sources() == {'inlets': [7], 'outlets': [12]}


def test_annotation_survives_a_file_round_trip_and_the_migration_reports_margins():
    g = av_graph()
    report = g.annotate_sources_from_heuristic(min_margin=2.0)
    assert report['changed'] and report['inlet'] == 0 and report['outlet'] == 5
    assert report['margins']['artery'] == pytest.approx(60. / 10.) and not report['ambiguous']
    assert g.annotate_sources_from_heuristic()['changed'] is False        # existing annotation kept
    ambiguous = av_graph(stub_radius=55.).annotate_sources_from_heuristic(min_margin=2.0)
    assert ambiguous['ambiguous']
    path = os.path.join(tempfile.mkdtemp(), 'g.am')
    g.write(path)
    back = spatialgraph.SpatialGraph()
    back.read(path)
    assert back.get_sources() == {'inlets': [0], 'outlets': [5]} and back.has_source_annotation()


def test_no_reachability_warning_before_a_capillary_bed_exists():
    g = av_graph()
    vt = np.asarray(g.get_data('VesselType')).copy()
    vt[vt == 2] = 1                                          # no capillary edges left: artery and vein trees are disjoint
    g.set_data(vt, name='VesselType')
    g.set_sources(inlets=[0], outlets=[5])
    assert not [m for s, m in g.validate_sources() if 'not connected' in m]
