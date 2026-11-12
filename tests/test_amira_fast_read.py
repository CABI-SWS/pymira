import numpy as np

from pymira import spatialgraph
from pymira.fast_view import segments

HEADER = """# AmiraMesh 3D ASCII 2.0


define VERTEX 3
define EDGE 2
define POINT 5

Parameters {
    ContentType "HxSpatialGraph"
}

VERTEX { float[3] VertexCoordinates } @1
EDGE { int[2] EdgeConnectivity } @2
EDGE { int NumEdgePoints } @3
POINT { float[3] EdgePointCoordinates } @4
POINT { float Radii } @5
POINT { bool Perfused } @6

# Data section follows
"""

BODY = """@1
0 0 0
1 0 0
2 0 0

@2
0 1
1 2

@3
2
3

@4
0 0 0
1 0 0
1 0 0
1.5 1 0
2 0 0

@5
1
2
3
4
5

@6
True
False
True
True
False
"""


def _write(tmp_path, text):
    f = tmp_path / "g.am"
    f.write_text(text)
    return str(f)


def test_ascii_graph_is_read_exactly(tmp_path):
    g = spatialgraph.SpatialGraph()
    g.read(_write(tmp_path, HEADER + BODY))
    assert g.get_data("VertexCoordinates").shape == (3, 3)
    assert g.get_data("EdgeConnectivity").tolist() == [[0, 1], [1, 2]]
    assert g.get_data("NumEdgePoints").tolist() == [2, 3]
    assert np.allclose(g.get_data("EdgePointCoordinates")[3], [1.5, 1, 0])
    assert g.get_data("Radii").tolist() == [1, 2, 3, 4, 5]
    assert g.get_data("Perfused").tolist() == [True, False, True, True, False]   # True/False text fields are parsed


def test_truncated_file_skips_the_missing_fields_instead_of_reading_garbage(tmp_path, capsys):
    text = HEADER + BODY[:BODY.index("@5")]          # the copy stopped before the Radii data
    g = spatialgraph.SpatialGraph()
    g.read(_write(tmp_path, text))
    assert "truncated" in capsys.readouterr().out
    assert g.get_field("Radii") is None
    assert g.get_data("EdgeConnectivity").tolist() == [[0, 1], [1, 2]]


def test_only_reads_the_requested_fields(tmp_path):
    from pymira import amiramesh
    am = amiramesh.AmiraMesh()
    am.read(_write(tmp_path, HEADER + BODY), quiet=True, only=("EdgePointCoordinates", "NumEdgePoints"))
    assert sorted(f["name"] for f in am.fields) == ["EdgePointCoordinates", "NumEdgePoints"]


def test_fast_view_segments_stay_inside_edges():
    d = {"npts": np.array([2, 3, 1, 2], dtype=np.int32)}      # 8 points
    i0, i1 = segments(d)
    # edge point ranges: [0,1], [2,3,4], [5], [6,7]
    assert i0.tolist() == [0, 2, 3, 6]
    assert (i1 - i0 == 1).all()
