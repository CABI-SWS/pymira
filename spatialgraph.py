# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 11:49:52 2016

@author: simon

Amira SpatialGraph loader and writer

"""

from pymira import amiramesh
import numpy as np
arr = np.asarray
#np.seterr(invalid='raise')
import os
from tqdm import tqdm, trange # progress bar
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy

def update_array_index(vals,inds,keep):
    # Updates/offets indices for an array (vals) to exclude values in a flag array (keep)
    # inds: array indices for vals.
    
    # Vertex coords (mx3), connections (nx2), vertex indices to keep (boolean, m)
    
    # Index vertices to be deleted
    del_inds = np.where(keep==False)[0]
    # Total number of vertices, prior to deletion
    npoints = vals.shape[0]
    # Indices of all vertices, prior to deletion
    old_inds = np.linspace(0,npoints-1,npoints,dtype='int')
    # Lookup for vertices, post deletion (-1 corresponds to a deletion)
    new_inds_lookup = np.zeros(npoints,dtype='int')-1
    new_inds_lookup[~np.in1d(old_inds,del_inds)] = np.linspace(0,npoints-del_inds.shape[0]-1,npoints-del_inds.shape[0])
    # Create a new index array using updated index lookup table
    if type(inds) is not list and inds.dtype!=object:
        new_inds = new_inds_lookup[inds] 
        # Remove -1 values that reference deleted nodes
        new_inds = new_inds[(new_inds[:,0]>=0) & (new_inds[:,1]>=0)]
    else: # Nested list
        new_inds = []
        #valid = np.ones(len(inds),dtype='bool')
        for i in inds:
            nxt = new_inds_lookup[i]
            if np.all(nxt>=0):
                new_inds.append(nxt)
    
    return vals[keep],new_inds,new_inds_lookup
    
def delete_vertices(graph,keep_nodes,return_lookup=False,return_keep_edge=False): # #verts,edges,keep_nodes):

    """
    Efficiently delete vertices as flagged by a boolean array (keep_nodes) and update the indexing of an
    edge (index) array that potentially references those vertices
    """
    
    nodecoords,edgeconn,edgepoints,nedgepoints = graph.get_standard_fields()
    
    # Find which edges must be deleted
    del_node_inds = np.where(keep_nodes==False)[0]
    del_edges = [np.where((edgeconn[:,0]==i) | (edgeconn[:,1]==i))[0] for i in del_node_inds]
    # Remove empties (i.e. where the node doesn't appear in any edges)
    del_edges = [x for x in del_edges if len(x)>0]
    # Flatten
    del_edges = [item for sublist in del_edges for item in sublist]
    # Convert to numpy
    del_edges = arr(del_edges)
    # List all edge indices
    inds = np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int')
    # Define which edge each edgepoint belongs to
    edge_inds = np.repeat(inds,nedgepoints)
    # Create a mask of points to keep for edgepoint variables
    keep_edgepoints = ~np.in1d(edge_inds,del_edges)
    # Apply mask to edgepoint array
    edgepoints = edgepoints[keep_edgepoints]
    # Apply mask to scalars
    scalars = graph.get_scalars()
    for scalar in scalars:
        graph.set_data(scalar['data'][keep_edgepoints],name=scalar['name'])
    
    # Create a mask for removing edge connections and apply to the nedgepoint array
    keep_edges = np.ones(edgeconn.shape[0],dtype='bool')
    if len(del_edges)>0:
        keep_edges[del_edges] = False
        nedgepoints = nedgepoints[keep_edges]
    
    # Remove nodes and update indices
    nodecoords, edgeconn, edge_lookup = update_array_index(nodecoords,edgeconn,keep_nodes)
    
    node_scalars = graph.get_node_scalars()
    for i,sc in enumerate(node_scalars):
        graph.set_data(node_scalars[i]['data'][keep_nodes],name=sc['name'])
    
    # Update VERTEX definition
    vertex_def = graph.get_definition('VERTEX')
    vertex_def['size'] = [nodecoords.shape[0]]
    # Update EDGE definition
    edge_def = graph.get_definition('EDGE')
    edge_def['size'] = [edgeconn.shape[0]]
    # Update POINT definition
    edgepoint_def = graph.get_definition('POINT')
    edgepoint_def['size'] = [edgepoints.shape[0]]
    
    graph.set_data(nodecoords,name='VertexCoordinates')
    graph.set_data(edgeconn,name='EdgeConnectivity')
    graph.set_data(nedgepoints,name='NumEdgePoints')
    graph.set_data(edgepoints,name='EdgePointCoordinates')
            
    graph.set_graph_sizes()

    if return_lookup:
        return graph, edge_lookup
    elif return_keep_edge:
        return graph, keep_edges
    else:
        return graph
        
def split_artery_vein(graph,gfile=None,capillaries=False):

    # Arterial graph
    if gfile is None:
        agraph = graph.copy()
    else:
        agraph = spatialgraph.SpatialGraph()
        agraph.read(gfile)
    seg_cat = agraph.get_data('VesselType')  
    epi = agraph.edgepoint_edge_indices()
    inds = np.where(seg_cat!=0)
    edges_to_delete = np.unique(epi[inds])
    from pymira.spatialgraph import GVars
    gv = GVars(agraph)
    gv.remove_edges(edges_to_delete)
    gv.set_in_graph()
    
    # Arterial graph
    if gfile is None:
        vgraph = graph.copy()
    else:
        vgraph = spatialgraph.SpatialGraph()
        vgraph.read(gfile)
    seg_cat = vgraph.get_data('VesselType')  
    epi = vgraph.edgepoint_edge_indices()
    inds = np.where(seg_cat!=1)
    edges_to_delete = np.unique(epi[inds])
    #ed.delete_edges(vgraph,edges_to_delete)
    gv = GVars(vgraph)
    gv.remove_edges(edges_to_delete)
    gv.set_in_graph()
    
    ed = Editor()
    agraph = ed.remove_disconnected_nodes(agraph)
    vgraph = ed.remove_disconnected_nodes(vgraph)
    
    # Capillaries
    if capillaries:
        if gfile is None:
            cgraph = graph.copy()
        else:
            cgraph = spatialgraph.SpatialGraph()
            cgraph.read(gfile)
        seg_cat = cgraph.get_data('VesselType')  
        epi = cgraph.edgepoint_edge_indices()
        inds = np.where(seg_cat!=2)
        edges_to_delete = np.unique(epi[inds])
        #ed.delete_edges(vgraph,edges_to_delete)
        gv = GVars(cgraph)
        gv.remove_edges(edges_to_delete)
        gv.set_in_graph()
        
        cgraph = ed.remove_disconnected_nodes(cgraph)
    
        return agraph,vgraph,cgraph
    else:
        return agraph,vgraph

class SpatialGraph(amiramesh.AmiraMesh):

    """
    Spatial graph class
    """
    
    def __init__(self,header_from=None,initialise=False,scalars=[],node_scalars=[],path=None):
        amiramesh.AmiraMesh.__init__(self)
        
        self.nodeList = None
        self.edgeList = None
        self.edgeListPtr = 0
        self.path = path
        
        self.edge_label_counter = 0
        self.node_label_counter = 0
        self.point_label_counter = 0
        
        if header_from is not None:
            import copy
            self.parameters = copy.deepcopy(header_from.parameters)
            self.definitions = copy.deepcopy(header_from.definitions)
            self.header = copy.deepcopy(header_from.header)
            self.fieldNames = copy.deepcopy(header_from.fieldNames)
            
        if initialise:
            self.initialise(scalars=scalars,node_scalars=node_scalars)        
            
    def __repr__(self):
        """
        Print to cli for debugging
        """
        print('GRAPH')
        #print(('Fields: {}'.format(self.fieldNames)))
        for i,d in enumerate(self.definitions):
            print(f"Definition {i}: {d['name']}, size: {d['size']}")
        for i,f in enumerate(self.fields):
            print( f"Field {i}: {f['name']}, type: {f['type']}, shape: {f['shape']}") #, data: {f['data']}")
        return ''
            
    def print(self):
        self.__repr__()
        
    def unique_node_label(self,nl=None):
        if nl is None:
            nl = self.get_data('NodeLabel')
        self.node_label_counter += 1
        nxt = self.node_label_counter
        if nl is not None:
            if nxt in nl:
                nxt = np.max(nl) + 1
                self.node_label_counter = nxt
        return nxt
        
    def unique_edge_label(self,el=None):
        if el is None:
            el = self.get_data('EdgeLabel')
        self.edge_label_counter += 1
        nxt = self.edge_label_counter
        if el is not None:
            if nxt in el:
                nxt = np.max(el) + 1
                self.edge_label_counter = nxt
        return nxt
        
    def unique_point_label(self):
        self.point_label_counter += 1
        return self.point_label_counter
        
    def copy(self):
        graph_copy = SpatialGraph()
        
        import copy
        graph_copy.parameters = copy.deepcopy(self.parameters)
        graph_copy.definitions = copy.deepcopy(self.definitions)
        graph_copy.header = copy.deepcopy(self.header)
        graph_copy.fieldNames = copy.deepcopy(self.fieldNames)
        
        graph_copy.fields = []
        for i,f in enumerate(self.fields):
            fcopy = f.copy()
            fcopy['data'] = f['data'].copy()
            graph_copy.fields.append(fcopy)
            
        graph_copy.set_graph_sizes()
        
        graph_copy.fileType = self.fileType
        graph_copy.filename = self.filename
            
        return graph_copy
            
    def initialise(self,scalars=[],node_scalars=[]):
    
        """
        Set default fields 
        """
    
        self.fileType = '3D ASCII 2.0'
        self.filename = ''
        
        self.add_definition('VERTEX',[0])
        self.add_definition('EDGE',[0])
        self.add_definition('POINT',[0])
        
        self.add_parameter('ContentType','HxSpatialGraph')

        self.add_field(name='VertexCoordinates',marker='@1',
                              definition='VERTEX',type='float',
                              nelements=3,nentries=[0],data=None)
        self.add_field(name='EdgeConnectivity',marker='@2',
                              definition='EDGE',type='int',
                              nelements=2,nentries=[0],data=None)
        self.add_field(name='NumEdgePoints',marker='@3',
                              definition='EDGE',type='int',
                              nelements=1,nentries=[0],data=None)
        self.add_field(name='EdgePointCoordinates',marker='@4',
                              definition='POINT',type='float',
                              nelements=3,nentries=[0],data=None)
                              
        offset = len(self.fields) + 1
        
        if len(scalars)>0:
            if type(scalars) is not list:
                scalars = [scalars]
            for i,sc in enumerate(scalars):

                if sc=='EdgeLabel':
                    self.add_field(name='EdgeLabel',marker=f'@{len(self.fields)+1}',
                                                  definition='POINT',type='int',
                                                  nelements=1,nentries=[0])  
                else:
                    self.add_field(name=sc,marker='@{}'.format(offset),
                                      definition='POINT',type='float',
                                      nelements=1,nentries=[0])
                offset = len(self.fields) + 1
                                  
        if len(node_scalars)>0:
            if type(node_scalars) is not list:
                node_scalars = [node_scalars]
            for i,sc in enumerate(node_scalars):
                if sc=='NodeLabel':
                    self.add_field(name='NodeLabel',marker=f'@{len(self.fields)+1}',
                                                  definition='VERTEX',type='int',
                                                  nelements=1,nentries=[0])   
                else:
                    self.add_field(name=sc,marker='@{}'.format(i+offset),
                                      definition='VERTEX',type='float',
                                      nelements=1,nentries=[0])
                offset = len(self.fields) + 1
                              
        self.fieldNames = [x['name'] for x in self.fields]
        
    def read(self,*args,**kwargs):
        """
        Read spatial graph from .am Amira (or JSON) file
        """
        if args[0].endswith('.json'):
            self.read_json(args[0])
        else:
            if not amiramesh.AmiraMesh.read(self,*args,**kwargs):
                return False
            if "HxSpatialGraph" not in self.get_parameter_value("ContentType"):
                print('Warning: File is not an Amira SpatialGraph!')
                pass

        self.set_graph_sizes()
        
        node_labels = self.get_data('NodeLabel')
        if node_labels is None:
            self.node_label_counter = self.nnode
        else:
            self.node_label_counter = np.max(node_labels) + 1
            
        edge_labels = self.get_data('EdgeLabel')
        if edge_labels is None:
            self.edge_label_counter = self.nedge
        else:
            self.edge_label_counter = np.max(edge_labels) + 1
            
        point_labels = self.get_data('PointLabel')
        if point_labels is None:
            self.point_label_counter = self.nedgepoint
        else:
            self.point_label_counter = np.max(point_labels) + 1
                
        return True
            
    def remove_edges(self,edge_inds_to_remove):
        gv = GVars(self)
        gv.remove_edges(edge_inds_to_remove)
        graph = gv.set_in_graph()
   
    def remove_loopsOLD(self):  
        duplicate_edges = self.get_duplicated_edges()
        edgeconn = self.get_data('EdgeConnectivity')
        sind = np.where(duplicate_edges>0)
        rads = self.point_scalars_to_edge_scalars(name='thickness')
        
        rem_edge = []
        for ei in np.unique(duplicate_edges):
            if ei>0:
                stmp = np.where(duplicate_edges==ei)
                edges = []
                for s in stmp[0]:
                    edges.extend([self.get_edge(s)])
                # Self-connection - remove
                if len(edges)==1:
                    rem_edge.extend([edges[0].index])
                else:
                    edgeRads = rads[stmp[0]]
                    lengths = arr([np.sum(np.linalg.norm(e.coordinates[:-1]-e.coordinates[1:])) for e in edges])
                    vols = np.pi*np.power(edgeRads,2)*lengths
                    mx = np.argmax(vols)
                    inds = np.linspace(0,len(vols)-1,len(vols),dtype='int')
                    rem_edge.extend(stmp[0][inds[inds!=mx]])

        rem_edge = arr(rem_edge)
        if rem_edge.shape[0]==0:
            print('No loops identified')
            return
        self.remove_edges(rem_edge)

    def unpack_edgepoints(self, edge_points, edge_point_counts, scalars=[]):
        """Turn flattened EdgePointCoordinates + NumEdgePoints into a list of per-edge arrays."""
        counts = arr(edge_point_counts, dtype=int).flatten()
        pts = arr(edge_points, dtype=float).reshape(-1, 3)
        out = []
        scalars_out = []
        i = 0
        for c in counts:
            out.append(pts[i:i+c].copy())
            pScalars = []
            for x in scalars:
                pScalars.append(x[i:i+c].copy())
            scalars_out.append(pScalars)
            i += c
        return out, scalars_out

    def repack_edgepoints(self, per_edge_points):
        """Flatten a list of per-edge arrays back into EdgePointCoordinates + NumEdgePoints."""
        counts = [0 if p is None else int(len(p)) for p in per_edge_points]
        if sum(counts) == 0:
            flat = np.empty((0, 3), dtype=float)
        else:
            flat = np.vstack([p if p is not None and len(p) else np.empty((0, 3), float) for p in per_edge_points])
        return flat.astype(float), np.asarray(counts, dtype=int)
    
    def repack_edge_data(self,
                     per_coords,
                     per_scal,
                     *,
                     return_scalar_matrix: bool = False):
        """
        Flatten per-edge coordinates and per-edge per-channel scalars.

        Inputs
        ------
        per_coords : list of arrays, each (Ni, 3)
        per_scal   : list over edges, each item is a list over channels,
                    per_scal[e][c] is a 1D array of length Ni

        Parameters
        ----------
        return_scalar_matrix : if True, also returns scalars as a single (P, K) matrix.
                            If False, returns a list of K vectors of length P, one per channel.

        Returns
        -------
        edge_points        : (P, 3) float
        edge_point_counts  : (M,)   int
        scalars_out        : list[np.ndarray] with length K, each shape (P,) if return_scalar_matrix is False
                            or a single (P, K) float array if return_scalar_matrix is True
        """
        import numpy as np

        # Counts and points
        counts = [0 if pc is None else int(len(pc)) for pc in per_coords]
        M = len(per_coords)
        P = sum(counts)

        if P == 0:
            edge_points = np.empty((0, 3), dtype=float)
            edge_point_counts = np.asarray(counts, dtype=int)
            if per_scal is None:
                return edge_points, edge_point_counts, [] if not return_scalar_matrix else np.empty((0, 0), float)
            # Infer K even with empty edges
            K = len(per_scal[0]) # self._num_channels(per_scal)
            if return_scalar_matrix:
                return edge_points, edge_point_counts, np.empty((0, K), float)
            else:
                return edge_points, edge_point_counts, [np.empty((0,), float) for _ in range(K)]

        # Stack coordinates
        edge_points = np.vstack([pc if pc is not None and len(pc) else np.empty((0, 3), float) for pc in per_coords]).astype(float)
        edge_point_counts = np.asarray(counts, dtype=int)

        # Scalars
        if per_scal is None or len(per_scal) == 0:
            # No scalars provided, return empty vectors
            if return_scalar_matrix:
                return edge_points, edge_point_counts, np.empty((P, 0), float)
            else:
                return edge_points, edge_point_counts, []

        # Number of channels K
        K = len(per_scal[0]) #self._num_channels(per_scal)

        # Prepare concatenated vectors per channel, padding or truncating to match Ni if needed
        chan_vectors = [list() for _ in range(K)]
        for e_idx, pc in enumerate(per_coords):
            Ni = int(len(pc)) if pc is not None else 0
            # Ensure channel list exists for this edge
            edge_chans = per_scal[e_idx] if e_idx < len(per_scal) and isinstance(per_scal[e_idx], list) else [np.array([], float) for _ in range(K)]
            if len(edge_chans) < K:
                # pad missing channels with empty arrays
                edge_chans = edge_chans + [np.array([], float) for _ in range(K - len(edge_chans))]

            for c in range(K):
                ch = np.asarray(edge_chans[c], dtype=float).reshape(-1)
                if Ni == 0:
                    seg = np.empty((0,), float)
                else:
                    if ch.size == Ni:
                        seg = ch
                    elif ch.size > Ni:
                        # truncate surplus samples
                        seg = ch[:Ni]
                    else:
                        # pad with NaN to reach Ni
                        pad = np.full((Ni - ch.size,), np.nan, dtype=float)
                        seg = np.concatenate([ch, pad], axis=0)
                chan_vectors[c].append(seg)

        # Concatenate per channel across edges
        chan_concat = [np.concatenate(parts, axis=0) if len(parts) else np.empty((0,), float)
                    for parts in chan_vectors]

        if return_scalar_matrix:
            # Stack into (P, K)
            if K == 0:
                scal_mat = np.empty((P, 0), float)
            else:
                scal_mat = np.column_stack(chan_concat)
            return edge_points, edge_point_counts, scal_mat
        else:
            return edge_points, edge_point_counts, chan_concat

    def _build_adj_by_edge(self, n_nodes, edges):
        adj = [[] for _ in range(n_nodes)]
        for ei, (u, v) in enumerate(edges):
            adj[u].append(ei)
            adj[v].append(ei)
        return adj
    
    def _edge_other(self, edges, eidx, node):
        u, v = edges[eidx]
        return v if u == node else u
    
    def _normalize(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _unique_undirected(self, edges):
        e_sorted = np.sort(edges, axis=1)
        view = e_sorted.view([('', e_sorted.dtype)]*2)
        _, idx = np.unique(view, return_index=True)
        return edges[np.sort(idx)]

    def _remove_self_loops(self, edges):
        keep = edges[:, 0] != edges[:, 1]
        return edges[keep]
    
    def _closest_is_endpoint(self, poly, endpoint, atol):
        """Return which end of poly matches endpoint within tolerance [is_start, is_end]."""
        if poly is None or len(poly) == 0:
            return False, False
        is_start = np.allclose(poly[0], endpoint, atol=atol, rtol=0)
        is_end = np.allclose(poly[-1], endpoint, atol=atol, rtol=0)
        return is_start, is_end

    def _ensure_polyline(self, poly, start_pos, end_pos):
        """Ensure an edge has a minimally valid polyline from start to end if it had none."""
        if poly is None or len(poly) == 0:
            return np.vstack([start_pos, end_pos])
        return poly
    
    def _ensure_scalars(self, scal=None, K=0, s_start=None, s_end=None):
        """
        If scalars missing, create a 2 row array, using s_start, s_end if provided,
        else fill with NaN.
        """
        if scal is None or len(scal) == 0:
            s0 = s_start if s_start is not None else np.full((1, K), np.nan)
            s1 = s_end   if s_end   is not None else np.full((1, K), np.nan)
            return np.vstack([s0, s1])
        if scal.ndim == 1:
            return scal.reshape(-1, 1)
        return scal

    def _interp_scalar(self, a, b, t):
        return (1.0 - t) * a + t * b

    def _make_endpoint_scalar(self, scal, which, policy, fill_value=None):
        """
        Create a scalar row for an inserted endpoint.

        which: 'start' or 'end'
        policy: 'copy', 'interp', 'constant', 'nan'
        """
        K = scal.shape[1]
        if policy == "copy":
            return scal[0:1] if which == "start" else scal[-1:]
        if policy == "interp":
            if len(scal) >= 2:
                return _interp_scalar(scal[0:1], scal[1:2], 0.0) if which == "start" else _interp_scalar(scal[-2:-1], scal[-1:], 1.0)
            return scal[0:1]
        if policy == "constant":
            val = np.full((1, K), np.nan if fill_value is None else float(fill_value))
            return val
        # default nan
        return np.full((1, K), np.nan)

    def _adjust_polyline_endpoint(self, poly, scal, old_pos, new_pos, other_end_pos, atol=None, endpoint_atol=0.,
            move_start=False,
            scalar_policy='copy',
            scalar_fill_value=None,
            known_start_scalar=None,
            known_end_scalar=None):
        """
        Move the polyline endpoint that used to sit at old_pos to new_pos,
        keeping interior samples intact. If no endpoint sample matches old_pos,
        insert new_pos at the corresponding end to remain consistent.
        """

        poly = self._ensure_polyline(poly, old_pos, other_end_pos)
        K = scal[0].shape[1] if scal is not None and scal[0].ndim == 2 else (known_start_scalar.shape[1] if known_start_scalar is not None else 1)
        scal = [self._ensure_scalars(sc, K, known_start_scalar, known_end_scalar) for sc in scal]

        is_start_match = np.allclose(poly[0], old_pos, atol=endpoint_atol, rtol=0)
        is_end_match   = np.allclose(poly[-1], old_pos, atol=endpoint_atol, rtol=0)

        if move_start:
            if is_start_match:
                poly[0] = new_pos
            else:
                poly = np.vstack([new_pos, poly])
                for i,sc in scal:
                    srow = self._make_endpoint_scalar(sc, "start", scalar_policy, scalar_fill_value[i])
                    scal = np.vstack([srow, sc])
        else:
            if is_end_match:
                poly[-1] = new_pos
                # keep last scalar as is
            else:
                poly = np.vstack([poly, new_pos])
                for i,sc in scal:
                    srow = self._make_endpoint_scalar(sc, "end", scalar_policy, scalar_fill_value[i])
                    scal = np.vstack([sc, srow])

        return poly, scal
    
    def _endpoint_index_for_node(self, edge_coords, node_pos, atol):
        """
        Return 0 if the start matches node_pos, -1 if the end matches, else None.
        """
        if edge_coords is None or len(edge_coords) == 0:
            return None
        if np.allclose(edge_coords[0], node_pos, atol=atol, rtol=0):
            return 0
        if np.allclose(edge_coords[-1], node_pos, atol=atol, rtol=0):
            return -1
        return None

    def _endpoint_scalar_values(self, per_scal_edge, idx):
        """
        Extract scalar values at endpoint index idx for all channels of one edge.
        per_scal_edge is list of C arrays with shape (Ni,), idx is 0 or -1.
        Returns a list of floats of length C, with np.nan where unavailable.
        """
        out = []
        for ch in per_scal_edge:
            if ch is None or ch.size == 0:
                out.append(np.nan)
            else:
                out.append(float(ch[idx]))
        return out

    def _aggregate_channelwise(self, rows, mode="median", lengths=None):
        """
        rows is a list of lists, shape approx [num_edges_in_chunk][C].
        Returns a list of C aggregated values.
        mode in {"median","mean","length_weighted","min","max"}.
        """
        if not rows:
            return []
        A = np.array(rows, dtype=float)  # shape (E,C)
        if mode == "median":
            return np.nanmedian(A, axis=0).tolist()
        if mode == "mean":
            return np.nanmean(A, axis=0).tolist()
        if mode == "min":
            return np.nanmin(A, axis=0).tolist()
        if mode == "max":
            return np.nanmax(A, axis=0).tolist()
        if mode == "length_weighted":
            if lengths is None:
                w = np.ones((A.shape[0], 1), float)
            else:
                w = np.asarray(lengths, float).reshape(-1, 1)
            W = np.where(np.isfinite(A), w, 0.0)
            num = np.nansum(A * W, axis=0)
            den = np.nansum(W, axis=0)
            return (num / np.clip(den, 1e-12, None)).tolist()
        return np.nanmedian(A, axis=0).tolist()  

    def _ensure_per_scal_len(self, per_scal, target_len, num_channels):
        """Pad per_scal to target_len with empty channel arrays per edge."""
        if per_scal is None:
            per_scal = []
        while len(per_scal) < target_len:
            per_scal.append([np.array([], dtype=float) for _ in range(num_channels)])
        return per_scal  

    def split_high_degree_nodes(self, max_degree=3, displacement=1e-3, endpoint_atol=None,scalar_endpoint_policy='copy',scalar_fill_value=None):
        """
        Split nodes with degree > max_degree by adding displaced helper nodes,
        reassign incident edges to helpers, and keep edge polylines consistent.

        Inputs
        ------
        nodes  : (N,3)
        edges  : (M,2)
        edge_points : (P,3) flattened EdgePointCoordinates
        edge_point_counts : (M,) NumEdgePoints
        max_degree : maximum degree allowed per node
        displacement : helper node offset in world units
        endpoint_atol : tolerance for matching a polyline endpoint to a node position.
                        Default is 10 * displacement, good for small offsets.

        Returns
        -------
        new_nodes, new_edges, new_edge_points, new_edge_point_counts
        """
        nodes = self.get_data('VertexCoordinates')
        edges = self.get_data('EdgeConnectivity')
        edge_points = self.get_data('EdgePointCoordinates')
        edge_point_counts = self.get_data('NumEdgePoints')

        # Clean obvious issues before we touch polylines
        #edges = _remove_self_loops(edges)
        #edges = _unique_undirected(edges)

        scalar_fields = self.get_scalars()
        scalars = [x['data'] for x in self.get_scalars()]
        per_coords,per_scal = self.unpack_edgepoints(edge_points, edge_point_counts, scalars=scalars)

        if endpoint_atol is None:
            endpoint_atol = max(1e-9, 10.0 * float(displacement))

        new_nodes = nodes.tolist()
        edges_np = edges.copy()

        # Build adjacency by edge indices
        def build_adj_by_edges(n_nodes, E):
            adj = [[] for _ in range(n_nodes)]
            for ei, (u, v) in enumerate(E):
                if u < n_nodes:
                    adj[u].append(ei)
                if v < n_nodes:
                    adj[v].append(ei)
            return adj

        changed = True
        while changed:
            changed = False
            adj = build_adj_by_edges(len(new_nodes), edges_np)
            deg = [len(a) for a in adj]
            for n_idx, d in enumerate(deg):
                if d <= max_degree:
                    continue

                inc = adj[n_idx]
                # Directions for ordering
                dirs = []
                for eidx in inc:
                    u, v = edges_np[eidx]
                    other = v if u == n_idx else u
                    vec = np.asarray(new_nodes[other]) - np.asarray(new_nodes[n_idx])
                    nrm = np.linalg.norm(vec)
                    dirs.append(vec / nrm if nrm > 0 else np.array([1.0, 0.0, 0.0]))
                dirs = np.array(dirs)
                # Build local frame for angular order
                ref = dirs[0]
                tmp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(tmp, ref)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0])
                ux = self._normalize(np.cross(ref, tmp))
                vy = self._normalize(np.cross(ref, ux))
                ang = np.arctan2(dirs @ vy, dirs @ ux)
                order = np.argsort(ang)
                inc = [inc[i] for i in order]
                dirs = dirs[order]

                # Partition incident edges
                q = (len(inc) + max_degree - 1) // max_degree  # number of chunks that would be made
                keep_on_root = max(1, max_degree - (q - 1))     # leave room for connector edges

                # Build chunks so that root keeps only keep_on_root edges
                chunks = []
                if keep_on_root >= len(inc):
                    chunks = [inc]  # nothing to split
                else:
                    chunks.append(inc[:keep_on_root])           # edges that remain on original
                    rest = inc[keep_on_root:]
                    for i in range(0, len(rest), max_degree):
                        chunks.append(rest[i:i+max_degree])     # helper chunks

                base = np.asarray(new_nodes[n_idx])

                for c_idx, chunk in enumerate(chunks):
                    if c_idx == 0:
                        continue
                    d_slice = dirs[c_idx*max_degree:(c_idx+1)*max_degree]
                    mean_dir = self._normalize(d_slice.mean(axis=0)) if len(d_slice) else np.array([1.0, 0.0, 0.0])

                    helper_pos = base + displacement * mean_dir
                    helper_idx = len(new_nodes)
                    new_nodes.append(helper_pos.tolist())

                    # Reassign endpoints and adjust per-edge data
                    for eidx in chunk:
                        a, b = edges_np[eidx]
                        if a == n_idx:
                            edges_np[eidx, 0] = helper_idx
                            per_coords[eidx], per_scal[eidx] = self._adjust_polyline_endpoint(
                                per_coords[eidx], per_scal[eidx],
                                old_pos=base, new_pos=helper_pos, other_end_pos=np.asarray(new_nodes[b]),
                                endpoint_atol=endpoint_atol, move_start=True,
                                scalar_policy=scalar_endpoint_policy, scalar_fill_value=scalar_fill_value
                            )
                        elif b == n_idx:
                            edges_np[eidx, 1] = helper_idx
                            per_coords[eidx], per_scal[eidx] = self._adjust_polyline_endpoint(
                                per_coords[eidx], per_scal[eidx],
                                old_pos=base, new_pos=helper_pos, other_end_pos=np.asarray(new_nodes[a]),
                                endpoint_atol=endpoint_atol, move_start=False,
                                scalar_policy=scalar_endpoint_policy, scalar_fill_value=scalar_fill_value
                            )

                    # Add connector edge A -> H with two points
                    edges_np = np.vstack([edges_np, np.array([[n_idx, helper_idx]], dtype=int)])
                    per_coords.append(np.vstack([base, helper_pos]))

                    # Determine channel count and ensure per_scal length
                    C = len(per_scal[0])
                    per_scal = self._ensure_per_scal_len(per_scal, len(per_coords) - 1, C)

                    # Collect endpoint scalar rows from the edges moved to this helper
                    endpoint_rows = []
                    lengths_for_weights = []
                    for eidx in chunk:
                        # which endpoint of this edge lies on the original node?
                        ei = self._endpoint_index_for_node(per_coords[eidx], base, endpoint_atol)
                        if ei is None:
                            # fallback, pick closer endpoint
                            if len(per_coords[eidx]) >= 2:
                                d0 = np.linalg.norm(per_coords[eidx][0] - base)
                                d1 = np.linalg.norm(per_coords[eidx][-1] - base)
                                ei = 0 if d0 <= d1 else -1
                            else:
                                ei = 0
                        endpoint_rows.append(self._endpoint_scalar_values(per_scal[eidx], ei))

                        # optional weight by current edge length
                        if len(per_coords[eidx]) >= 2:
                            lengths_for_weights.append(float(np.linalg.norm(per_coords[eidx][-1] - per_coords[eidx][0])))
                        else:
                            lengths_for_weights.append(1.0)

                    # Choose how to aggregate values into a representative per-channel vector
                    rep_mode = "median"  # try "mean", "length_weighted", "min", "max" if preferred
                    rep_vals = self._aggregate_channelwise(endpoint_rows, mode=rep_mode, lengths=lengths_for_weights)
                    # Ensure we have C values
                    if len(rep_vals) < C:
                        rep_vals = rep_vals + [np.nan] * (C - len(rep_vals))

                    # Build per-channel arrays for the new connector edge.
                    # For a connector, it is reasonable to duplicate the node value at both endpoints.
                    new_edge_scalars = []
                    for c in range(C):
                        v = rep_vals[c]
                        new_edge_scalars.append(np.array([v, v], dtype=float))

                    # Append scalars for the new edge
                    per_scal.append(new_edge_scalars)
                    changed = True        

                #breakpoint()
                #gc = self.get_node_count()
                #print(n_idx,np.mean(deg),changed)
                #breakpoint()
                if changed:
                    break  # rebuild degrees after each split batch

        # Final dedupe to be safe, keep polylines aligned
        #edges_np, keep_idx2 = self._unique_undirected(edges_np)
        #per_coords = [per_coords[i] for i in keep_idx2]
        #per_scal   = None if per_scal is None else [per_scal[i] for i in keep_idx2]

        new_pts, new_cnt, new_scal = self.repack_edge_data(per_coords, per_scal)
        nodes = arr(new_nodes)
        self.set_data(nodes,name='VertexCoordinates')
        self.set_data(edges_np,name='EdgeConnectivity')
        self.set_data(new_pts,name='EdgePointCoordinates')
        self.set_data(new_cnt,name='NumEdgePoints')
        for i,sc in enumerate(new_scal): self.set_data(sc,name=scalar_fields[i]['name'])
        self.set_definition_size('VERTEX',nodes.shape[0])
        self.set_definition_size('EDGE',edges_np.shape[0])
        self.set_definition_size('POINT',new_pts.shape[0])
                
    def read_json(self,filename):
    
        import json
        
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
            
        req = ['VertexCoordinates','EdgeConnectivity','NumEdgePoints','EdgePointCoordinates']
        if not np.all([x in list(data.keys()) for x in req]):
            print('Invalid JSON file format!')
            return
            
        self.initialise()
            
        for k,v in zip(data.keys(),data.values()):
            if k in req:
                vals = arr(v)
                self.set_data(vals,name=k)
                if k=='VertexCoordinates':
                    self.set_definition_size('VERTEX',vals.shape[0])
                elif k=='EdgeConnectivity':                    
                    self.set_definition_size('EDGE',vals.shape[0])
                elif k=='EdgePointCoordinates':                    
                    self.set_definition_size('POINT',vals.shape[0])   
            else:
                # Assume for now that all additional fields are point scalars...
                self.add_field(name=k,marker=self.generate_next_marker(),
                                  definition='POINT',type='float',
                                  nelements=1,nentries=[0])
                self.set_data(arr(v),name=k)
        
    def export_mesh(self,vessel_type=None,radius_scale=1,min_radius=0,ofile='',resolution=10):
        if vessel_type is not None:
            vtypeEdge = self.point_scalars_to_edge_scalars(name='VesselType')
            tp = self.plot_graph(show=False,block=False,min_radius=min_radius,edge_filter=vtypeEdge==vessel_type,cyl_res=resolution,radius_scale=radius_scale,radius_based_resolution=False)
        else:
            tp = self.plot_graph(show=False,block=False,min_radius=min_radius,cyl_res=resolution,radius_scale=radius_scale,radius_based_resolution=False)

        gmesh = tp.cylinders_combined
        import open3d as o3d
        gmesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(ofile,gmesh)
        tp.destroy_window()
        print(f'Mesh written to {ofile}')
        
    def change_field_name(self,original_name,new_name):
    
        try:
            ind = self.fieldNames.index(original_name)
        except:
            return
            
        self.fields[ind]['name'] = new_name
        self.fieldNames[ind] = new_name
        
    def set_graph_sizes(self,labels=False):
        """
        Ensure consistency between data size fields and the data itself
        """
        try:
            self.nnode = self.get_definition('VERTEX')['size'][0]
        except:
            pass
        try:
            self.nedge = self.get_definition('EDGE')['size'][0]
        except:
            pass
        try:
            self.nedgepoint = self.get_definition('POINT')['size'][0]
        except:
            pass
            
        if labels:
            sc = [x['name'] for x in self.fields]
            if 'EdgeLabel' in sc:
                nedgepoint = self.get_data('NumEdgePoints')
                edgeLabel = np.repeat(np.linspace(0,self.nedge-1,self.nedge,dtype='int'),nedgepoint)
                self.set_data(edgeLabel,name='EdgeLabel')   
            if 'NodeLabel' in sc:
                nodeLabel = np.linspace(0,self.nnode-1,self.nnode,dtype='int')
                self.set_data(nodeLabel,name='NodeLabel')    
            
    def get_standard_fields(self):
        """
        Convenience method for retrieving fields that are always present
        """

        res = []
        nodecoords = self.get_data('VertexCoordinates')
        edgeconn = self.get_data('EdgeConnectivity')
        edgepoints = self.get_data('EdgePointCoordinates')
        nedgepoints = self.get_data('NumEdgePoints')
        
        return nodecoords,edgeconn,edgepoints,nedgepoints
        
    def rescale_coordinates(self,xscale,yscale,zscale,ofile=None):
        """
        Scale spatial coordinates by a fixed factor
        """
        nodeCoords = self.get_data('VertexCoordinates')
        edgeCoords = self.get_data('EdgePointCoordinates')
        
        for i,n in enumerate(nodeCoords):
            nodeCoords[i] = [n[0]*xscale,n[1]*yscale,n[2]*zscale]
        for i,n in enumerate(edgeCoords):
            edgeCoords[i] = [n[0]*xscale,n[1]*yscale,n[2]*zscale]
        
        if ofile is not None:
            self.write(ofile)
            
    def rescale_radius(self,rscale,ofile=None):
        """
        Scale radii by a fixed factor
        """
        radf = self.get_radius_field()
        radii = radf['data']
        #radii = self.get_data('Radii')
        mnR = np.min(radii)
        for i,r in enumerate(radii):
            radii[i] = r * rscale / mnR
            
        if ofile is not None:
            self.write(ofile)
    
    def reset_data(self):
        """
        Set all data to None
        """
        for x in self.fields:
            x['data'] = None
        for x in self.definitions:
            x['size'] = [0]
        for x in self.fields:
            x['shape'] = [0,x['nelements']]
            x['nentries'] = [0]
            
    def add_node(self,node=None,index=0,coordinates=[0.,0.,0.]):
        """
        Append a node onto the VertexCoordinates field
        """
        nodeCoords = self.get_field('VertexCoordinates')['data']
        if node is not None:
            coordinates = node.coords
            index = node.index
        if nodeCoords is not None:
            newData = np.vstack([nodeCoords, np.asarray(coordinates)])
            self.set_definition_size('VERTEX',newData.shape)
        else:
            newData = np.asarray(coordinates)
            self.set_definition_size('VERTEX',[1,newData.shape])
        self.set_data(newData,name='VertexCoordinates')
    
    def add_node_connection(self,startNode,endNode,edge):
        """
        Add a new edge into the graph
        """
        edgeConn = self.get_field('EdgeConnectivity')['data']
        nedgepoints = self.get_field('NumEdgePoints')['data']
        edgeCoords = self.get_field('EdgePointCoordinates')['data']
        
        # Add connection
        if edgeConn is not None:
            newData = np.squeeze(np.vstack([edgeConn, np.asarray([startNode.index,endNode.index])]))
            self.set_definition_size('EDGE',[1,newData.shape[0]])
        else:
            newData = np.asarray([[startNode.index,endNode.index]])
            self.set_definition_size('EDGE',newData.shape[0])
        self.set_data(newData,name='EdgeConnectivity')
        
        # Add number of edge points
        npoints = edge.coordinates.shape[0]
        if nedgepoints is not None:
            try:
                newData = np.append(nedgepoints,npoints) #np.squeeze(np.vstack([np.squeeze(nedgepoints), np.asarray(npoints)]))
            except Exception as exep:
                print(exep)
                import pdb
                pdb.set_trace()
        else:
            newData = np.asarray([npoints])
        self.set_data(newData,name='NumEdgePoints')
        
        # Add edge point coordinates
        if edgeCoords is not None:
            newData = np.squeeze(np.vstack([np.squeeze(edgeCoords), np.asarray(edge.coordinates)]))
        else:
            newData = np.asarray([edge.coordinates])
        self.set_definition_size('POINTS',newData.shape[0])
        self.set_data(newData,name='EdgePointCoordinates')

    def number_of_node_connections(self,file=None):
    
       """
       DEPRECATED: Use get_node_count method
       Returns the number of edge connections for each node
       """

       #Identify terminal nodes
       conn = self.fields[1]['data']
       nConn = np.asarray([len(np.where((conn[:,0]==i) | (conn[:,1]==i))[0]) for i in range(self.nnode)])
       return nConn
               
    def clone(self):
        """
        Create a deep copy of the graph object
        """
        import copy
        return copy.deepcopy(self)
       
    # NODE LIST: Converts the flat data structure into a list of node class objects, with connectivity data included
           
    def node_list(self,path=None):
        
        # Try and load from pickle file
        nodeList = self.load_node_list(path=path)
        if nodeList is not None:
            self.nodeList = nodeList
            return self.nodeList
        
        # Convert graph to a list of node (and edge) objects
        nodeCoords = self.get_field('VertexCoordinates')['data']
        nnode = nodeCoords.shape[0]
        self.nodeList = []
        
        self.nodeList = [None] * nnode
        import time
        for nodeIndex in trange(nnode):
            #t0 = time.time()
            self.nodeList[nodeIndex] = Node(graph=self,index=nodeIndex)
            #if nodeIndex%1000==0:
            #    print(time.time()-t0)
            
        if path is not None:
            self.write_node_list(path=path)
            
        return self.nodeList
           
    def write_node_list(self,path=None):
        
        if path is not None:
            self.path = path
            import dill as pickle
            ofile = os.path.join(path,'nodeList.dill')
            with open(ofile,'wb') as fo:
                pickle.dump(self.nodeList,fo)
            
    def load_node_list(self,path=None):
        
        if path is not None:
            self.path = path
            try:
                nfile = os.path.join(path,'nodeList.dill')
                if os.path.isfile(nfile):
                    print(('Loading node list from file: {}'.format(nfile)))
                    import dill as pickle
                    with open(nfile,'rb') as fo:
                        nodeList = pickle.load(fo)
                    return nodeList
            except Exception as e:
                print(e)
        return None
        
    def edges_from_node_list(self,nodeList):
        
        #import pdb
        #pdb.set_trace()
        nedges = self.nedge
        edges = [None]*nedges
        indices = []
        pbar = tqdm(total=len(nodeList))
        for n in nodeList:
            pbar.update(1)
            for e in n.edges:
                if edges[e.index] is None:
                    edges[e.index] = e
                #if e.index not in indices: # only unique edges
                #    edges[e.index] = e
                    #indices.append(e.index)
        pbar.close()
        if None in edges:
            print('Warning, edge(s) missing from edge list')
                
        return edges

    def node_list_to_graph(self,nodeList):
        
        nodeCoords = np.asarray([n.coords for n in nodeList])
        nnode = nodeCoords.shape[0]
        
        edges = self.edges_from_node_list(nodeList)

        edgeConn = np.asarray([[x.start_node_index,x.end_node_index] for x in edges if x is not None])
        edgeCoords = np.concatenate([x.coordinates for x in edges if x is not None])
        nedgepoint = np.array([x.npoints for x in edges if x is not None])
        
        scalarNames = edges[0].scalarNames
        scalarData = [x.scalars for x in edges if x is not None]        
        scalars = []
        nscalar = len(scalarNames)
        for i in range(nscalar): 
            scalars.append(np.concatenate([s[i] for s in scalarData]))
        
        nodeScalarNames = nodeList[0].scalarNames
        nodeScalarData = np.asarray([x.scalars for x in nodeList])
        nnodescalar = len(nodeScalarNames)
        nodeScalars = np.zeros([nnodescalar,nnode])
        for i in range(nnodescalar):
            #nodeScalars.append(np.concatenate([s[i] for s in nodeScalarData]))
            nodeScalars[i,:] = nodeScalarData[i::nnodescalar][0]
        
        #import spatialgraph
        graph = SpatialGraph(initialise=True,scalars=scalarNames,node_scalars=nodeScalarNames)
        
        graph.set_definition_size('VERTEX',nodeCoords.shape[0])
        graph.set_definition_size('EDGE',edgeConn.shape[0])
        graph.set_definition_size('POINT',edgeCoords.shape[0])

        graph.set_data(nodeCoords,name='VertexCoordinates')
        graph.set_data(edgeConn,name='EdgeConnectivity')
        graph.set_data(nedgepoint,name='NumEdgePoints')
        graph.set_data(edgeCoords,name='EdgePointCoordinates')
        for i,s in enumerate(scalars):
            graph.set_data(s,name=scalarNames[i])
        for i,s in enumerate(nodeScalars):
            graph.set_data(s,name=nodeScalarNames[i])
        
        return graph        
        
    # Spatial methods
        
    def node_spatial_extent(self):
        
        """
        Calculates the rectangular boundary box containing all nodes in the graph
        Returns [[x_min,x_max], [y_min,y_max],[z_min,z_max]]
        """
        nodecoords = self.get_data('VertexCoordinates')
        rx = [np.min(nodecoords[:,0]),np.max(nodecoords[:,0])]
        ry = [np.min(nodecoords[:,1]),np.max(nodecoords[:,1])]
        rz = [np.min(nodecoords[:,2]),np.max(nodecoords[:,2])]
        return [rx,ry,rz]
        
    def edge_spatial_extent(self):
    
        """
        Calculates the rectangular boundary box containing all edgepoints in the graph
        Returns [[x_min,x_max], [y_min,y_max],[z_min,z_max]]
        """
        
        coords = self.get_data('EdgePointCoordinates')
        rx = [np.min(coords[:,0]),np.max(coords[:,0])]
        ry = [np.min(coords[:,1]),np.max(coords[:,1])]
        rz = [np.min(coords[:,2]),np.max(coords[:,2])]
        return [rx,ry,rz]
        
    def edge_point_index(self):
        
        coords = self.get_data('EdgePointCoordinates')
        nedgepoint = self.get_data('NumEdgePoints')
        npoint = coords.shape[0]
        edgeInd = np.zeros(npoint,dtype='int') - 1

        cntr = 0
        curN = nedgepoint[0]
        j = 0
        for i in range(npoint):
            edgeInd[i] = j
            cntr += 1
            if cntr>=curN:
                cntr = 0
                j += 1
                if j<nedgepoint.shape[0]:
                    curN = nedgepoint[j]
                elif i!=npoint-1:
                    import pdb
                    pdb.set_trace()
                
        return edgeInd
        
    def constrain_nodes(self,xrange=[None,None],yrange=[None,None],zrange=[None,None],no_copy=True,keep_stradling_edges=False):
    
        """
        Delete all nodes outside a rectangular region
        """
        
        assert len(xrange)==2
        assert len(yrange)==2
        assert len(zrange)==2

        if not no_copy:        
            graph = self.clone()
        else:
            graph = self

        nodeCoords = graph.get_data('VertexCoordinates')
        nnode = len(nodeCoords)
        
        # Spatial extent of nodes
        r = self.node_spatial_extent()

        # Locate nodes outside of ranges
        if xrange[1] is None:
            xrange[1] = r[0][1]
        if yrange[1] is None:
            yrange[1] = r[1][1]
        if zrange[1] is None:
            zrange[1] = r[2][1]
        xrange = [np.max([r[0][0],xrange[0]]),np.min([r[0][1],xrange[1]])]
        yrange = [np.max([r[1][0],yrange[0]]),np.min([r[1][1],yrange[1]])]
        zrange = [np.max([r[2][0],zrange[0]]),np.min([r[2][1],zrange[1]])]
        
        # Mark which nodes to keep / delete
        keepNode = np.ones(nnode,dtype='bool')
        for i in range(nnode):
            x,y,z = nodeCoords[i,:]
            if x<xrange[0] or x>xrange[1] or y<yrange[0] or y>yrange[1] or z<zrange[0] or z>zrange[1]:
                keepNode[i] = False
                
        # Keep edges that straddle the boundary
        if keep_stradling_edges:
            keepNodeEdit = keepNode.copy()
            ec = self.get_data('EdgeConnectivity')
            ch = 0
            for i,kn in enumerate(keepNode):
                if kn==True:
                    conns = np.empty([0])
                    inds0 = np.where(ec[:,0]==i)
                    if len(inds0[0])>0:
                        conns = np.concatenate([conns,ec[inds0[0],1]])
                    inds1 = np.where(ec[:,1]==i)
                    if len(inds1[0])>0:
                        conns = np.concatenate([conns,ec[inds1[0],0]])
                    conns = conns.astype('int')
                    if np.any(keepNode[conns]==False):
                        keepNodeEdit[conns] = True
                        ch += 1
            if ch>0:
                keepNode = keepNodeEdit
                
        nodes_to_delete = np.where(keepNode==False)
        nodes_to_keep = np.where(keepNode==True)
        if len(nodes_to_keep[0])==0:
            print('No nodes left!')
            return
        
        editor = Editor()
        return editor.delete_nodes(self,nodes_to_delete[0])
        
    def crop(self,*args,**kwargs):
        """
        Rectangular cropping of the graph spatial extent
        Just a wrapper for constrain_nodes
        """
        return self.constrain_nodes(*args,**kwargs)
        
    def remove_field(self,fieldName):
        """
        Remove a data field from the graph
        """
        f = [(i,f) for (i,f) in enumerate(self.fields) if f['name']==fieldName]
        if len(f)==0 or f[0][1] is None:
            print(('Could not locate requested field: {}'.format(fieldName)))
            return
        _  = self.fields.pop(f[0][0])
        _  = self.fieldNames.pop(f[0][0])
        
    def get_node(self,index):
        """
        Create a node class instance for the node index supplied
        """
        return Node(graph=self,index=index)
        
    def get_edge(self,index):
        """
        Create an edge class instance for the edge index supplied
        """
        return Edge(graph=self,index=index)
        
    def edge_index_from_point(self,pointIndex):
        """
        Given the index of an edge point, returns the edge index that it is part of
        """
        nEdgePoint = self.get_data('NumEdgePoints')
        nEdgePointCum = np.cumsum(nEdgePoint)
        wh = np.where(nEdgePointCum<=pointIndex)
        try:
            if wh[0].shape==(0,):
                return 0
            else:
                return np.max(wh)
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
        
    def edgepoint_edge_indices(self):
        """
        Creates an array relating edgepoints to the index of the edge that they're from
        """
        edgeconn = self.get_data('EdgeConnectivity')
        nedge = edgeconn.shape[0]
        nEdgePoint = self.get_data('NumEdgePoints')
        conn_inds = np.linspace(0,nedge-1,nedge,dtype='int')
        return np.repeat(conn_inds,nEdgePoint)
        
    def get_edges_containing_node(self,node_inds,mode='or'):
        """
        Return which edges contain supplied node indices
        """
        edgeconn = self.get_data('EdgeConnectivity')
        if mode=='or':
            return np.where(np.in1d(edgeconn[:,0],node_inds) | np.in1d(edgeconn[:,1],node_inds))[0]
        elif mode=='and':
            return np.where(np.in1d(edgeconn[:,0],node_inds) & np.in1d(edgeconn[:,1],node_inds))[0]
        
    def get_scalars(self):
        """
        Return scalar edge fields
        """
        return [f for f in self.fields if f['definition'].lower() in ['point',''] and len(f['shape'])==1 and f['name']!='EdgePointCoordinates']
        #return [f for f in self.fields if f['shape'][0]==self.nedgepoint and len(f['shape'])==1 and f['name']!='EdgePointCoordinates']
        
    def get_node_scalars(self):
        """
        Return scalar edge fields
        """
        return [f for f in self.fields if f is not None and f['definition'].lower()=='vertex' and len(f['shape'])==1 and f['name']!='VertexCoordinates']
        #return [f for f in self.fields if f['shape'][0]==self.nnode and len(f['shape'])==1 and f['name']!='VertexCoordinates']
        
    def get_radius_field(self):
        """
        Edge radius is given several names ('thickness' by Amira, fo example!)
        This helper function looks through several common options and returns the first that matches (all converted to lower case)
        NOTE: Diameter is also in the lookup list!
        """
        names = ['radius','radii','diameter','diameters','thickness']
        for name in names:
            match = [self.fields[i] for i,field in enumerate(self.fieldNames) if field.lower()==name.lower()]
            if len(match)>0:
                return match[0]
        return None
        
    def get_radius_field_name(self):
        f = self.get_radius_field()
        if f is None:
            return None
        else: 
            return f['name']
            
    def get_radius_data(self):
        f = self.get_radius_field()
        if f is None:
            return None
        else: 
            return f['data']
        
    def edgepoint_indices(self,edgeIndex):
        """
        For a given edge index, return the start and end indices corresponding to edgepoints and scalars
        """
        nedgepoints = self.get_data('NumEdgePoints')
        edgeCoords = self.get_data('EdgePointCoordinates')
        nedge = len(nedgepoints)
        
        assert edgeIndex>=0
        assert edgeIndex<nedge
        #chase
        npoints = nedgepoints[edgeIndex]
        start_index = np.sum(nedgepoints[:edgeIndex])
        end_index = start_index + npoints
        
        return [start_index,end_index]
        
    def check_for_degenerate_edges(self):

        edgeconn = self.get_data('EdgeConnectivity')
        un,cn = np.unique(edgeconn,axis=0,return_counts=True)
        if np.any(cn>1):
            return True
        else:
            return False
            
    def sanity_check(self,deep=False):
        """
        Check that all fields have the correct size, plus other checks and tests
        """ 
        self.set_graph_sizes()
        err = ''
        
        for d in self.definitions:
            defName = d['name']
            defSize = d['size'][0]
            fields = [f for f in self.fields if f['definition']==defName]
            for f in fields:
                if f['nentries'][0]!=defSize:
                    err = f'{f["name"]} field size does not match {defName} definition size!'
                    print(err)
                if f['shape'][0]!=defSize:
                    err = f'{f["name"]} shape size does not match {defName} definition size!'               
                    print(err)
                if not all(x==y for x,y in zip(f['data'].shape,f['shape'])):
                    err = f'{f["name"]} data shape does not match shape field!'
                    print(err)             

        if deep:
            self.edgeList = None
            for nodeInd in range(self.nnode):
                node = self.get_node(nodeInd)
                for i,e in enumerate(node.edges):
                    if not node.edge_indices_rev[i]:
                        if not all(x.astype('float32')==y.astype('float32') for x,y in zip(e.start_node_coords,node.coords)):
                            err = f'Node coordinates ({node.index}) do not match start of edge ({e.index}) coordinates: {e.start_node_coords} {node.coords}'
                            #print(('Node coordinates ({}) do not match start of edge ({}) coordinates: {} {}'.format(node.index,e.index,e.start_node_coords,node.coords)))
                        if not all(x.astype('float32')==y.astype('float32') for x,y in zip(e.coordinates[0,:],e.start_node_coords)):
                            err = f'Edge start point does not match edge/node start ({e.index}) coordinates'
                            #print(('Edge start point does not match edge/node start ({}) coordinates'.format(e.index)))
                        if not all(x.astype('float32')==y.astype('float32') for x,y in zip(e.coordinates[-1,:],e.end_node_coords)):
                            err = f'Edge end point does not match edge/node end ({e.index}) coordinates'
                            #print(('Edge end point does not match edge/node end ({}) coordinates'.format(e.index)))
                    else:
                        if not all(x.astype('float32')==y.astype('float32') for x,y in zip(e.end_node_coords,node.coords)):
                            err = f'Node coordinates ({node.index}) do not match end of edge ({e.index}) coordinates'
                            print(err)
                            #print(('Node coordinates ({}) do not match end of edge ({}) coordinates'.format(node.index,e.index)))
                        if not all(x.astype('float32')==y.astype('float32') for x,y in zip(e.coordinates[0,:],e.start_node_coords)):
                            err = f'Edge end point does not match edge start (REVERSE) ({e.index}) coordinates'
                            print(err)
                            #print(('Edge end point does not match edge start (REVERSE) ({}) coordinates'.format(e.index)))
                        if not all(x.astype('float32')==y.astype('float32') for x,y in zip(e.coordinates[-1,:],e.end_node_coords)):
                            err = f'Edge start point does not match edge end (REVERSE) (edge {e.index}) coordinates'
                            print(err)
                            #print(('Edge start point does not match edge end (REVERSE) ({}) coordinates'.format(e.index)))        

        # Check for nans
        nodes = self.get_data('VertexCoordinates') 
        if np.any(np.isfinite(nodes)==False):
            err = 'Non-finite node values present'
            print(err)
        points = self.get_data('EdgePointCoordinates')   
        if np.any(np.isfinite(points)==False):
            err = 'Non-finite edgepoint values present'
            print(err)        

        if err!='':
            return False
        else:
            return True

        
    def nodes_connected_to(self,nodes,path=None):
        """
        DEPRECATED(?)
        """
        import pymira.front as frontPKG
        
        nodeCoords = graph.get_data('VertexCoordinates')
        nnodes = len(nodeCoords)
        if self.nodeList is None:
            nodeList = self.node_list(path=path)
        else:
            nodeList = self.nodeList
        
        connected = np.zeros(nnodes,dtype='bool')
        for strtNode in nodes:
            front = frontPKG.Front([strtNode])
            endloop = False
            curNode = strtNode
            while endloop is False:
                if front.front_size>0 and endloop is False:
                    for curNode in front.get_current_front():
                        next_nodes = [cn for cn in curNode.connecting_node if connected[cn] is False]
                        connected[nxtNodes] = True
                        front.step_front(next_nodes)
                else:
                    endloop = True
                    
        return connected
        
    def get_all_connections_to_node(self,nodeInds,maxIter=10000):
    
        """
        Find all edges that are connected to a node
        """
    
        nodecoords = self.get_data('VertexCoordinates')
        edgeconn = self.get_data('EdgeConnectivity')
    
        nodeStore, edgeStore, conn_order = [], [], []
        edges = self.get_edges_containing_node(nodeInds)
        if len(edges)>0:
            edgeStore.extend(edges.flatten().tolist())
            
            count = 0
            while True:
                next_nodes = edgeconn[edges].flatten()
                edges = self.get_edges_containing_node(next_nodes)
                # Take out edges already in store
                edges = edges[~np.in1d(edges,edgeStore)]
                # If there are new edges, add them in, otherwise break
                if len(edges)>0:
                    edgeStore.extend(edges.flatten().tolist())
                    nodeStore.extend(next_nodes.flatten().tolist())
                    conn_order.extend([count]*next_nodes.shape[0])
                    count += 1
                else:
                    break
                if count>maxIter:
                    print(f'Warning, GET_ALL_CONNECTIONS_TO_NODE: Max. iteration count reached!')
                    break
        return arr(nodeStore), arr(edgeStore)
        
    def replace_nodes(self,nodes):
    
        if nodes.shape[0]!=self.nnode:
            print('Spatialgraph.replace_nodes: Supplied node array is the wrong shape!')
            return
        
        nep = self.get_data('NumEdgePoints')
        edgeconn = self.get_data('EdgeConnectivity')
        if np.all(nep)==2:
            edges = nodes[edgeconn.flatten()]
            self.set_data(edges,name='EdgePointCoordinates')
            self.set_data(nodes,name='VertexCoordinates')
        else:
            print('Error,Spatialgraph.replace_nodes: Not implemented!')
            return
        
    def connected_nodes(self,index, return_edges=True):
        # Return all nodes connected to a supplied node index, 
        # along with the edge indices they are connected by
        vertCoords = self.get_data('VertexCoordinates')
        edgeconn = self.get_data('EdgeConnectivity')
        
        conn_edges = self.get_edges_containing_node(index)
        end_nodes = edgeconn[conn_edges]
        # Remove the current (source) node from the end node list
        end_nodes = arr([e[e!=index] for e in end_nodes]).flatten()

        if return_edges:
            return end_nodes, conn_edges
        else: 
            return end_nodes
           
        # Old, slower version...           
        s0 = np.where(edgeConn[:,0]==index)
        ns0 = len(s0[0])
        s1 = np.where(edgeConn[:,1]==index)
        ns1 = len(s1[0])
            
        nconn = ns0 + ns1
        try:
            edge_inds = np.concatenate((s0[0],s1[0]))
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
            
        connecting_node = np.zeros(nconn,dtype='int')
        connecting_node[0:ns0] = edgeConn[s0[0],1]
        connecting_node[ns0:ns0+ns1] = edgeConn[s1[0],0]
        return connecting_node, edge_inds
        
    def get_node_to_node_lengths(self):
        """
        Calculate the distance between connected nodes (not following edges)
        """
        vertexCoordinates = self.get_data('VertexCoordinates')
        edgeConnectivity = self.get_data('EdgeConnectivity') 
        lengths = np.linalg.norm(vertexCoordinates[edgeConnectivity[:,1]]-vertexCoordinates[edgeConnectivity[:,0]],axis=1)
        return lengths
        
    def get_edge_lengths(self,node=False):
    
        edgeconn = self.get_data('EdgeConnectivity') 
        nedgepoints = self.get_data('NumEdgePoints')
        edgeCoords = self.get_data('EdgePointCoordinates')
        nodes = self.get_data('VertexCoordinates')
        
        lengths = np.zeros(self.nedge)
        
        if node==False:
            for i in range(self.nedge):
                x0 = np.sum(nedgepoints[:i])
                npts = nedgepoints[i]
                pts = edgeCoords[x0:x0+npts]
                lengths[i] = np.sum(np.linalg.norm(pts[:-1]-pts[1:],axis=1))
        else:
            edge_nodes = nodes[edgeconn]
            lengths = np.linalg.norm(edge_nodes[:,1]-edge_nodes[:,0],axis=1)
            
        return lengths
        
    def get_node_count(self,edge_node_lookup=None,restore=False,tmpfile=None,graph_params=None):

        nodecoords = self.get_data('VertexCoordinates')
        edgeconn = self.get_data('EdgeConnectivity')
        
        # Which edge each node appears in
        if edge_node_lookup is not None:
            node_count = arr([len(edge_node_lookup[i]) for i in range(nodecoords.shape[0])])
        else:
            unq,count = np.unique(edgeconn,return_counts=True)
            all_nodes = np.linspace(0,nodecoords.shape[0]-1,nodecoords.shape[0],dtype='int')
            node_count = np.zeros(nodecoords.shape[0],dtype='int') 
            node_count[np.in1d(all_nodes,unq)] = count
        return node_count
        
    def identify_inlet_outlet(self,tmpfile=None,restore=False,ignore=None):

        nodecoords = self.get_data('VertexCoordinates')
        edgeconn = self.get_data('EdgeConnectivity')
        edgepoints = self.get_data('EdgePointCoordinates')
        nedgepoints = self.get_data('NumEdgePoints')
        #radius = self.get_data(self.get_radius_field_name())
        edge_radius = self.point_scalars_to_edge_scalars(name=self.get_radius_field_name(),func=np.max)
        category = self.get_data('VesselType')
        edge_category = self.point_scalars_to_edge_scalars(name='VesselType')
        
        if category is None:
            category = np.zeros(edgepoints.shape[0],dtype='int')
            edge_category = np.zeros(self.nedge,dtype='int')

        #inds = np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int')
        #edge_inds = np.repeat(inds,nedgepoints)
        #first_edgepoint_inds = np.concatenate([[0],np.cumsum(nedgepoints)[:-1]])

        #edge_node_lookup = create_edge_node_lookup(nodecoords,edgeconn,tmpfile=tmpfile,restore=restore)
                
        # Calculate node connectivity
        node_count = self.get_node_count() #,edge_node_lookup=edge_node_lookup)
        # Find graph end nodes (1 connection only)
        term_inds = np.where(node_count==1)
        terminal_node = np.zeros(nodecoords.shape[0],dtype='bool')
        terminal_node[term_inds] = True

        # Assign a radius to nodes using the largest radius of each connected edge
        #edge_radius = radius[first_edgepoint_inds]

        # Assign a category to each node using the minimum category of each connected edge (thereby favouring arteries/veins (=0,1) over capillaries (=2))
        #edge_category = category[first_edgepoint_inds]
        
        # Locate arterial input(s)
        mask = np.ones(edgeconn.shape[0])
        mask[(edge_category!=0) | ((node_count[edgeconn[:,0]]!=1) & (node_count[edgeconn[:,1]]!=1))] = np.nan
        if ignore is not None:
            mask[(np.in1d(edgeconn[:,0],ignore)) | (np.in1d(edgeconn[:,1],ignore))] = np.nan
        if np.nansum(mask)==0.:
            a_inlet_node = None
        else:
            a_inlet_edge_ind = np.nanargmax(edge_radius*mask)
            a_inlet_edge_nodes = edgeconn[a_inlet_edge_ind]
            a_inlet_node = a_inlet_edge_nodes[node_count[a_inlet_edge_nodes]==1][0]
        
        # Locate vein output(s)
        mask = np.ones(edgeconn.shape[0])
        mask[(edge_category!=1) | ((node_count[edgeconn[:,0]]!=1) & (node_count[edgeconn[:,1]]!=1))] = np.nan
        if ignore is not None:
            mask[(np.in1d(edgeconn[:,0],ignore)) | (np.in1d(edgeconn[:,1],ignore))] = np.nan
        if np.nansum(mask)==0.:
            v_outlet_node = None
        else:
            v_outlet_edge_ind = np.nanargmax(edge_radius*mask)
            v_outlet_edge_nodes = edgeconn[v_outlet_edge_ind]
            v_outlet_node = v_outlet_edge_nodes[node_count[v_outlet_edge_nodes]==1][0]
        
        return a_inlet_node,v_outlet_node 

    def get_duplicated_edges(self):
        edges = self.get_data('EdgeConnectivity')
        #duplicate_edges = np.zeros(edges.shape[0],dtype='int')
        duplicate_edge_index = np.zeros(edges.shape[0],dtype='int') - 1
        dind = 0
        for i,x in enumerate(edges): 
            s1 = np.where( (edges[:,0]==x[0]) & (edges[:,1]==x[1]) & (duplicate_edge_index==-1) )[0]
            s2 = np.where( (edges[:,1]==x[0]) & (edges[:,0]==x[1]) & (duplicate_edge_index==-1) )[0]
            if len(s1)+len(s2)>1:
                duplicate_edge_index[s1] = dind
                duplicate_edge_index[s2] = dind
                dind += 1
        
        return duplicate_edge_index
        
    def check_artery_vein_classification(self):
        # Check artery/vein classification
        # Assumes two seperate graphs, one for arteries (=0) one for veins (=1)
        gr = self.identify_graphs()
        # Check that there are only two graphs
        ugr = np.unique(gr)
        if not np.all(np.in1d(ugr,[1,2])):
            return -3
        # Check that veins and arteries are in separate graphs
        vtn = self.point_scalars_to_node_scalars(name='VesselType')
        chk_a = np.all(gr[vtn==0]==gr[vtn==0][0])
        chk_v = np.all(gr[vtn==1]==gr[vtn==1][0])
        if not chk_a:
            return -1
        if not chk_v:
            return -2
        return 0
        
    def remove_subsidiary_graphs(self):
        # Removes all but the largest arterial graph and the largest venous graph
        
        vtn = self.point_scalars_to_node_scalars(name='VesselType')
        nvt = np.unique(vtn)
        
        gr,cnts = self.identify_graphs()
        #gr = np.concatenate([gra,grv])
        # Check that there are only two graphs
        ugr,cnt = np.unique(gr,return_counts=True)
        if len(ugr)<=2:
            return
        
        gr_type = arr([vtn[gr==u][0] for u in ugr])
        
        cnt_a = cnt.copy()
        cnt_a[gr_type!=0] = 0
        a_keep = ugr[np.nanargmax(cnt_a)]
        cnt_v = cnt.copy()
        cnt_v[gr_type!=1] = 0
        v_keep = ugr[np.nanargmax(cnt_v)]

        keep_node = np.zeros(self.nnode,dtype='bool')
        keep_node[gr==a_keep] = True
        keep_node[gr==v_keep] = True
        
        _ = delete_vertices(self,keep_node)

    def test_treelike(self, inlet=None, outlet=None, euler=True, ignore_type=False, quiet=False, return_data=False):

        if inlet is None:
            inlet,outlet = self.identify_inlet_outlet()
            
        nodecoords = self.get_data('VertexCoordinates')
        edgeconn = self.get_data('EdgeConnectivity')
        edgepoints = self.get_data('EdgePointCoordinates')
        nedgepoints = self.get_data('NumEdgePoints')
        radius = self.get_data(self.get_radius_field_name())
        
        visited = []
        # Start at either arterial input or venous outlet (if both exist)
        for i,root in enumerate([inlet,outlet]):
            if root is not None:
                # Initialise front from previous iterations
                prev_front = None
                # Intitialise current front
                front = [root]
                # Initialise node that have been visited
                visited.extend(front)
                count = 0
                while True:
                    # Find edges containing nodes in the current front
                    edges = self.get_edges_containing_node(front)
                    all_conn_nodes = edgeconn[edges].flatten()
                    # Store nodes not in front
                    if prev_front is not None:
                        next_front = all_conn_nodes[~np.in1d(all_conn_nodes,front) & ~np.in1d(all_conn_nodes,prev_front)]
                    else:
                        next_front = all_conn_nodes[~np.in1d(all_conn_nodes,front)]
                    if len(next_front)>0:
                        dplicates = np.in1d(next_front,visited)
                        if np.any(dplicates):
                            err = f'Test treelike, revisited: {next_front[dplicates]}, it: {count}'
                            if not quiet:
                                print(err)
                            dnodes = next_front[dplicates]
                            edges = self.get_edges_containing_node(dnodes)
                            if return_data==True:
                                return False, err, [next_front,dplicates]
                            else:
                                return False
                        unq,cnt = np.unique(next_front,return_counts=True)
                        if np.any(cnt)>1:
                            err = f'Test treelike, duplicate paths to node: {unq[cnt>1]}'
                            if not quiet:
                                print(err)
                            #breakpoint()
                            if return_data==True:
                                return False, err, [count]
                            else:
                                return False
                        visited.extend(next_front.tolist())
                        prev_front = front
                        front = next_front
                    else:
                        break
                    count += 1
                    if count>edgeconn.shape[0]*2:
                        err = f'Test treelike: Count limit reached...'
                        if not quiet:
                            print(err)
                        if return_data==True:
                            return False, err, [count]
                        else:
                            return False
        
        # Double check
        all_in = np.in1d(np.arange(nodecoords.shape[0]),visited)
        if not np.all(all_in):
            err = f'Not all nodes visited: {np.arange(nodecoords.shape[0])[~all_in]}'
            if not quiet:
                print(err)
            if return_data==True:
                return False, err, [all_in,np.arange(nodecoords.shape[0])[~all_in]]
            else:
                return False
            
        gc = self.get_node_count()
        vt = self.edge_scalar_to_node_scalar('VesselType')
        edges = self.get_data('EdgeConnectivity')
        
        # Euler: Arterial nodes
        if euler:
            if vt is None:
                if self.nnode!=self.nedge+1:
                    err = f'Euler criterion failed ({self.nnode} nodes, {self.nedge} edges)'
                    if not quiet:
                        print(err)
                    if return_data==True:
                        return False, err, [self.nnode,self.nedge]
                    else:
                        return False        
            if ignore_type:
                n_anode = np.sum((vt==0) | (vt==1))
            else:
                n_anode = np.sum((vt==0))
            if n_anode>0:
                if ignore_type:
                    a_nodes = np.where((vt==0) | (vt==1))
                else:
                    a_nodes = np.where(vt==0)
                a_edges = self.get_edges_containing_node(a_nodes)
                n_aedges = a_edges.shape[0]
                if n_anode!=n_aedges+1:
                    if n_anode>n_aedges+1:
                        err = f'Euler criterion failed (arterial, too many nodes! {n_anode} nodes, {n_aedges} edges)'
                        if not quiet:
                            print(err)
                        if return_data==True:
                            return False, err, [n_anode,n_aedges]
                        else:
                            return False 
                    if n_anode<n_aedges+1:
                        err = f'Euler criterion failed (arterial, too many edges! {n_anode} nodes, {n_aedges} edges)'
                        if not quiet:
                            print(err)
                        if return_data==True:
                            return False, err, [n_anode,n_aedges]
                        else:
                            return False 
                
            # Euler: Venous nodes
            if ignore_type==False:
                n_vnode = np.sum((vt==1))
                if n_vnode>0:
                    v_nodes = np.where(vt==1)
                    v_edges = self.get_edges_containing_node(v_nodes)
                    n_vedges = v_edges.shape[0]
                    if n_vnode!=n_vedges+1:
                        if n_vnode>n_vedges+1:
                            err = f'Euler criterion failed (venous, too many nodes! {n_vnode} nodes, {n_vedges} edges)'
                            if not quiet:
                                print(err)
                            if return_data==True:
                                return False, err, [n_vnode,n_vedges]
                            else:
                                return False 
                        if n_vnode<n_vedges+1:
                            err = f'Euler criterion failed (venous, too many edges! {n_vnode} nodes, {n_vedges} edges)'
                            if not quiet:
                                print(err)
                            if return_data==True:
                                return False, err, [n_vnode,n_vedges]
                            else:
                                return False 
         
        duplicate_edges = self.get_duplicated_edges()
        if np.any(duplicate_edges>0):
            err = f'Duplicated edges!'
            if not quiet:
                print(err)
            if return_data==True:
                return False, err, [duplicate_edges]
            else:
                return False 
                
        selfconnected_edges = (edges[:,0]==edges[:,1])
        if np.any(selfconnected_edges):
            err = f'Self-connected edges!'
            if not quiet:
                print(err)
            if return_data==True:
                return False, err, [selfconnected_edges]
            else:
                return False 
            
        # Test for degeneracy
        res,_ = self.test_node_degeneracy(find_all=False)
        if res: 
            err = 'Degenerate nodes present!'
            if not quiet:
                print(err)
            if return_data==True:
                return False, err, [res]
            else:
                return False
        
        if return_data==True:
            return True, "", []
        else:
            return True
        
    def test_node_degeneracy(self,find_all=False):
        degen_nodes = []
        nodecoords = self.get_data('VertexCoordinates')
        for i,c1 in enumerate(nodecoords):
            sind = np.where((nodecoords[:,0]==c1[0]) & (nodecoords[:,1]==c1[1]) & (nodecoords[:,2]==c1[2]))
            if len(sind[0])>1:
                if find_all==False:
                    return True, sind[0]
                else:
                    degen_nodes.append(sind[0])
        if len(degen_nodes)==0:
            return False,arr(degen_nodes).flatten()
        else:
            return True,arr(degen_nodes).flatten()
            
    def get_degenerate_nodes(self):
    
        """
        Find degenerate nodes
        Return an array with nnode elements, with value equal to the first node with a degenerate coordinate identified (or -1 if not degenerate)
        """
    
        nodecoords = self.get_data('VertexCoordinates')
        degen_nodes = np.zeros(nodecoords.shape[0],dtype='int') - 1
        
        for i,c1 in enumerate(nodecoords):
            if degen_nodes[i]<0:
                sind = np.where((nodecoords[:,0]==c1[0]) & (nodecoords[:,1]==c1[1]) & (nodecoords[:,2]==c1[2]))
                if len(sind[0])>1:
                    degen_nodes[sind[0]] = i
        return degen_nodes
            
    def scale_graph(self,tr=np.identity(4),radius_index=0):
    
        nodes = self.get_data('VertexCoordinates')
        ones = np.ones([nodes.shape[0],1])
        nodesH = np.hstack([nodes,ones])
        edgepoints = self.get_data('EdgePointCoordinates')
        ones = np.ones([edgepoints.shape[0],1])
        edgepointsH = np.hstack([edgepoints,ones])
        rads = self.get_data(self.get_radius_field_name())
        
        nodes = (tr @ nodesH.T).T[:,:3]
        edgepoints = (tr @ edgepointsH.T).T[:,:3]
        
        # TODO - proper treatment of radii based on orientation of vessel relative to transform axes
        # For now, just scale by one of the transform scalars
        rads = np.abs(rads * tr[radius_index,radius_index])
        self.set_data(nodes,name='VertexCoordinates')
        self.set_data(edgepoints,name='EdgePointCoordinates')
        self.set_data(rads,name=self.get_radius_field_name())
                    
    def identify_graphs(self,progBar=False,ignore_node=None,ignore_edge=None,verbose=False,add_scalar=True,add_to_graph=False):

        # NEW VERSION (faster)!
        # Find all connected nodes
        gc = self.get_node_count()
        sends = np.where(gc<=1)
        nodes_visited = []
        node_graph_index = np.zeros(self.nnode,dtype='int') - 1
        #node_graph_contains_root = np.zeros(graph.nnode,dtype='bool')
        graph_index_count = 0
        for send in sends[0]:
            if node_graph_index[send]==-1:
                node_graph_index[send] = graph_index_count
                #node_graph_contains_root[send] = np.any(frozenNode[send])
                edges = self.get_edges_containing_node(send)
                cnodes,cedges = self.get_all_connections_to_node(send)

                if len(cnodes)>0:                            
                    node_graph_index[cnodes] = graph_index_count
                    #node_graph_contains_root[cnodes] = np.any(frozenNode[cnodes])

                graph_index_count += 1

            if np.all(node_graph_index>=0):
                break
                
        unique, counts = np.unique(node_graph_index, return_counts=True)

        if add_to_graph:
            edge_graph_index = np.zeros(self.nedge,dtype='int')
            for unq in unique:
                cedges = self.get_edges_containing_node(np.where(node_graph_index==unq))
                edge_graph_index[cedges] = unq
            nedgepoints = self.get_data('NumEdgePoints')
            point_graph_index = np.repeat(edge_graph_index,nedgepoints)

            if 'GraphIndex' not in self.fieldNames:
                self.add_field(name='GraphIndex',definition='POINT',type='int',data=point_graph_index)
            self.set_data(point_graph_index,name='GraphIndex')
                
        return node_graph_index, counts
        
        
        nodeCoords = self.get_data('VertexCoordinates')
        conn = self.get_data('EdgeConnectivity')
        nnodes = len(nodeCoords)
        nedge = len(conn)
        
        if ignore_node is None:
            ignore_node = np.zeros(self.nnode,dtype='bool')
            ignore_node[:] = False
        if ignore_edge is None:
            ignore_edge = np.zeros(self.nedge,dtype='bool')
            ignore_edge[:] = False
        
        #import pdb
        #pdb.set_trace()
        
        def next_count_value(graphIndex):
            return np.max(graphIndex)+1

        count = -1
        graphIndex = np.zeros(nnodes,dtype='int') - 1
        
        if progBar:
            pbar = tqdm(total=nnodes) # progress bar
        
        for nodeIndex,node in enumerate(nodeCoords):
            if progBar:
                pbar.update(1)
            
            #if graphIndex[nodeIndex] == -1:
            if not ignore_node[nodeIndex]:
                connIndex,edge_inds = self.connected_nodes(nodeIndex)
                connIndex = [connIndex[ei] for ei,edgeInd in enumerate(edge_inds) if not ignore_edge[edgeInd]]
                nconn = len(connIndex)
                # See if connected nodes have been assigned a graph index
                if nconn>0:
                    # Get graph indices for connected nodes
                    connGraphIndex = graphIndex[connIndex]
                    #if not ignore_edge[edge_inds]:
                    if True:
                        # If one or more connected nodes has an index, assign the minimum one to the curret node
                        if not all(connGraphIndex==-1):
                            #mn = np.min(np.append(connGraphIndex[connGraphIndex>=0],count))
                            #unq = np.unique(np.append(connGraphIndex[connGraphIndex>=0],count))
                            mn = np.min(connGraphIndex[connGraphIndex>=0])
                            unq = np.unique(connGraphIndex[connGraphIndex>=0])
                            inds = [i for i,g in enumerate(graphIndex) if g in unq]
                            graphIndex[inds] = mn
                            graphIndex[connIndex] = mn
                            graphIndex[nodeIndex] = mn
                            #print 'Node {} set to {} (from neighbours)'.format(nodeIndex,mn)
                            count = mn
                        else:
                            # No graph indices in vicinity
                            if graphIndex[nodeIndex] == -1:
                                count = next_count_value(graphIndex)
                                graphIndex[connIndex] = count
                                graphIndex[nodeIndex] = count
                                #print 'Node {} set to {} (new index)'.format(nodeIndex,count)
                            else:
                                count = graphIndex[nodeIndex]
                                graphIndex[connIndex] = count
                                #print 'Node {} neighbours set to {}'.format(nodeIndex,count)
                            #graphIndex[nodeIndex] = count
                            #graphIndex[connIndex] = count
                else:
                    # No graph indices in vicinity and no connected nodes
                    count = next_count_value(graphIndex)
                    if graphIndex[nodeIndex] == -1:
                        graphIndex[nodeIndex] = count
        
        if progBar:            
            pbar.close()

        # Make graph indices contiguous        
        unq = np.unique(graphIndex)
        ngraph = len(unq)
        newInd = np.linspace(0,ngraph-1,num=ngraph,dtype='int')
        for i,ind in enumerate(unq):
            graphIndex[graphIndex==ind] = i
            
        graph_size = np.histogram(graphIndex,bins=newInd)[0]
        
        if self.nodeList is None:
            self.nodeList = self.node_list()
            
        edges = self.edges_from_node_list(self.nodeList)
        
        if add_scalar:
            for e in edges:
                indS,indE = graphIndex[e.start_node_index],graphIndex[e.end_node_index]
                if indS!=indE:
                    import pdb
                    pdb.set_trace()
                e.add_scalar('Graph',np.repeat(indS,e.npoints))
            
        return graphIndex, graph_size
        
    def edge_scalar_to_node_scalar(self,name,maxval=False):
    
        if False: #type(name) is str:
            data = self.get_data(name)
            if data is None:
                return None
            conns = self.get_data('EdgeConnectivity')

            if False:
                mask = (conns[:, 0][:, None] == np.arange(self.nnode)) | (conns[:, 1][:, None] == np.arange(self.nnode))
                masked_data = np.where(mask, data[:, None], np.nan)
                node_scalar = np.nanmax(masked_data, axis=0)
                node_scalar = np.where(np.all(np.isnan(masked_data), axis=0), None, node_scalar).astype(data.dtype)
            else:
                node_scalar = arr([None if len(np.where((conns[:,0]==n) | (conns[:,1]==n))[0])==0
                                   else np.nanmax(data[np.where((conns[:,0]==n) | (conns[:,1]==n))])
                                   for n in range(self.nnode)]).astype(data.dtype)
            
            return node_scalar

        scalar_points = self.get_data(name)
        if scalar_points is None:
            return None
    
        verts = self.get_data('VertexCoordinates')
        conns = self.get_data('EdgeConnectivity')
        npoints = self.get_data('NumEdgePoints')
        points = self.get_data('EdgePointCoordinates')
        
        scalar_nodes = np.zeros(verts.shape[0],dtype=scalar_points.dtype)
        eei = self.edgepoint_edge_indices()
    
        for nodeIndex in range(self.nnode):
            edgeIds = np.where((conns[:,0]==nodeIndex) | (conns[:,1]==nodeIndex))
            if len(edgeIds[0])>0:
                vals = []
                for edgeId in edgeIds[0]:
                    npts = int(npoints[edgeId])
                    x0 = int(np.sum(npoints[0:edgeId]))
                    vtype = scalar_points[x0:x0+npts]
                    pts = points[x0:x0+npts,:]
                    node = verts[nodeIndex]
                    
                    if not maxval:
                        if np.all(pts[0,:]==node):
                            scalar_nodes[nodeIndex] = scalar_points[x0]
                        else:
                            scalar_nodes[nodeIndex] = scalar_points[x0+npts-1]
                        break
                    else:
                        vals.append(scalar_points[x0:x0+npts])
                if maxval:
                    scalar_nodes[nodeIndex] = np.max(vals)
                        
        return scalar_nodes
        
    def point_scalars_to_node_scalars(self,mode='max',name=None,func=np.nanmax):
    
        if True: #type(name) is str and mode=='max':
            data = self.get_data(name)
            if data is None:
                return None
            npts = self.get_data('NumEdgePoints')
            epi = np.repeat(np.linspace(0,self.nedge-1,self.nedge,dtype='int'),npts)
            sums = np.bincount(epi, weights=data, minlength=self.nedge)
            counts = np.bincount(epi, minlength=self.nedge)
            means = np.divide(sums, counts, where=counts!=0)
            conns = self.get_data('EdgeConnectivity')

            #if False:
            mask = (conns[:, 0][:, None] == np.arange(self.nnode)) | (conns[:, 1][:, None] == np.arange(self.nnode))
            masked_data = np.where(mask, means[:, None], np.nan)
            if callable(func):
                node_scalar = func(masked_data, axis=0)
                node_scalar = np.where(np.all(np.isnan(masked_data), axis=0), None, node_scalar).astype(data.dtype)
                return node_scalar
            elif isinstance(func, list) and all(callable(f) for f in func):
                res = []
                for f in func:
                    node_scalar = f(masked_data, axis=0)
                    node_scalar = np.where(np.all(np.isnan(masked_data), axis=0), None, node_scalar).astype(data.dtype)
                    res.append(node_scalar)
                return res
            else:
                breakpoint()
            #else:
            #node_scalar2 = arr([None if len(np.where((conns[:,0]==n) | (conns[:,1]==n))[0])==0
            #                       else np.nanmax(means[np.where((conns[:,0]==n) | (conns[:,1]==n))])
            #                       for n in range(self.nnode)]).astype(data.dtype)


        scalars = self.get_scalars()
        if name is not None:
            scalars = [x for x in scalars if x['name']==name]
            if len(scalars)==0:
                return None
    
        nodes = self.get_data('VertexCoordinates')
        conns = self.get_data('EdgeConnectivity')
        npoints = self.get_data('NumEdgePoints')
        points = self.get_data('EdgePointCoordinates')
        
        nsc = len(scalars)
        scalar_nodes = np.zeros([nsc,nodes.shape[0]]) + np.nan
    
        for i,conn in enumerate(conns):
            npts = int(npoints[i])
            x0 = int(np.sum(npoints[0:i]))
            x1 = x0+npts

            for j,scalar in enumerate(scalars):
                    
                data = scalar['data']
                if data is not None:
                    for node in conn:
                        if mode=='max':
                            scalar_nodes[j,node] = np.nanmax([np.max(data[x0:x1]),scalar_nodes[j,node]])
                        elif scalar['type']=='int':
                            scalar_nodes[j,node] = np.nanmin([np.min(data[x0:x1]),scalar_nodes[j,node]])
                        else:
                            scalar_nodes[j,node] = np.nanmin([np.min(data[x0:x1]),scalar_nodes[j,node]])

        scalar_nodes = scalar_nodes.squeeze()
        if not np.all((scalar_nodes==node_scalar) | ((~np.isfinite(scalar_nodes) & (~np.isfinite(node_scalar))))):
            breakpoint()
        return node_scalar
        
    def point_scalars_to_edge_scalars(self,func=np.mean,name=None,data=None):
    
        if True: #type(name) is str and func==np.mean:
            if data is None:
                data = self.get_data(name)
            elif data.shape[0]!=self.nedgepoint:
                return None
            if data is None:
                return None
            npts = self.get_data('NumEdgePoints')
            epi = np.repeat(np.linspace(0,self.nedge-1,self.nedge,dtype='int'),npts)
            #edata = arr([np.mean(data[epi==i]) for i in range(self.nedge)])
            sums = np.bincount(epi, weights=data, minlength=self.nedge)
            counts = np.bincount(epi, minlength=self.nedge)
            means = np.divide(sums, counts, where=counts!=0)
            return means

        scalars = self.get_scalars()
        if name is not None:
            scalars = [x for x in scalars if x['name']==name]
            if len(scalars)==0:
                return None
    
        verts = self.get_data('VertexCoordinates')
        conns = self.get_data('EdgeConnectivity')
        npoints = self.get_data('NumEdgePoints')
        points = self.get_data('EdgePointCoordinates')
        
        nsc = len(scalars)
        scalar_edges = np.zeros([nsc,conns.shape[0]])
    
        for i,conn in enumerate(conns):
            npts = int(npoints[i])
            x0 = int(np.sum(npoints[0:i]))
            x1 = x0+npts

            for j,scalar in enumerate(scalars):
                    
                data = scalar['data']
                if data is not None:
                    if scalar['type']=='float':
                        scalar_edges[j,i] = func(data[x0:x1])
                    elif scalar['type']=='int':
                        scalar_edges[j,i] = data[x0]
                    else:
                        scalar_edges[j,i] = func(data[x0:x1])
        return scalar_edges.squeeze()
 
    def point_scalars_to_segment_scalars(self,func=np.mean,name=None,domain=None):

        scalars = self.get_scalars()
        if name is not None:
            scalars = [x for x in scalars if x['name']==name]
            if len(scalars)==0:
                return None
    
        verts = self.get_data('VertexCoordinates')
        conns = self.get_data('EdgeConnectivity')
        npoints = self.get_data('NumEdgePoints')
        nsegpoints = npoints - 1
        points = self.get_data('EdgePointCoordinates')
        
        nsc = len(scalars)
        nseg = self.nedgepoint - self.nedge
        scalar_segs = np.zeros([nsc,nseg])
        
        ins = None
        if domain is not None:
            segments = self.get_segments(domain=domain)
            ins = (np.all(segments[:,0]>=domain[:,0],axis=1)) & (np.all(segments[:,0]<=domain[:,1],axis=1)) & \
                  (np.all(segments[:,1]>=domain[:,0],axis=1)) & (np.all(segments[:,1]<=domain[:,1],axis=1))
    
        for i,conn in enumerate(conns):
            npts = int(npoints[i])
            x0 = int(np.sum(npoints[0:i]))
            x1 = x0+npts
            s0 = int(np.sum(nsegpoints[0:i]))
            s1 = s0+int(nsegpoints[i])

            for j,scalar in enumerate(scalars):
                    
                data = scalar['data']
                if data is not None:
                    if npts>2:
                        scalar_segs[j,s0:s1] = data[x0:x1-1]
                    else:
                        scalar_segs[j,s0] = data[x0]
        if ins is None:
            return scalar_segs.squeeze()
        else:
            return scalar_segs[ins].squeeze()
 
    def plot_radius(self,*args,**kwargs):
        _ = self.plot_histogram(self.get_radius_field_name(),*args,**kwargs)
 
    def plot_histogram(self,field_name,*args,**kwargs):
    
        data = self.get_data(field_name)
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        n, bins, patches = plt.hist(data,*args,**kwargs)
        #fig.patch.set_alpha(0) # transparent
        plt.xlabel = field_name
        #plt.gca().set_xscale("log")
        plt.show()
        return fig
        
    def plot_pv(self,cylinders=None, vessel_type=None, color=None, edge_color=None, plot=True, grab=False, min_radius=0., \
                         domain_radius=None, domain_centre=arr([0.,0.,0.]),radius_based_resolution=True,cyl_res=10,use_edges=True,\
                         cmap_range=[None,None],bgcolor=[1,1,1],cmap=None,win_width=1920,win_height=1080,radius_scale=1.):
    
        import pyvista as pv
    
        nc = self.get_data('VertexCoordinates')
        points = self.get_data('EdgePointCoordinates')
        npoints = self.get_data('NumEdgePoints')
        conns = self.get_data('EdgeConnectivity')
        radField = self.get_radius_field()
        if radField is None:
            print('Could not locate vessel radius data!')
            radii = np.ones(points.shape[0])
        else:
            radii = radField['data']
        vType = self.get_data('VesselType')
        
        cols = None
        if edge_color is not None:
            cmap_range = arr(cmap_range)
            if cmap_range[0] is None:
                cmap_range[0] = edge_color.min()
            if cmap_range[1] is None:
                cmap_range[1] = edge_color.max()
            if cmap is None or cmap=='':
                from pymira.turbo_colormap import turbo_colormap_data
                cmap_data = turbo_colormap_data
                cols = turbo_colormap_data[(np.clip((edge_color-cmap_range[0]) / (cmap_range[1]-cmap_range[0]),0.,1.)*(turbo_colormap_data.shape[0]-1)).astype('int')]
            else:
                import matplotlib.pyplot as plt
                cmapObj = plt.cm.get_cmap(cmap)
                col_inds = np.clip((edge_color-cmap_range[0]) / (cmap_range[1]-cmap_range[0]),0.,1.)
                cols = cmapObj(col_inds)[:,0:3]

        network = pv.MultiBlock()        

        print('Preparing graph...')
        edge_def = self.get_definition('EDGE')
        #tubes = []
        tubes = np.empty(self.nedgepoint,dtype='object') # [None]*self.graph.nedgepoint
        for i in trange(edge_def['size'][0]):
            i0 = np.sum(npoints[:i])
            i1 = i0+npoints[i]
            coords = points[i0:i1]
            rads = radii[i0:i1]
            if vType is None:
                vt = np.zeros(coords.shape[0],dtype='int')
            else:
                vt = vType[i0:i1]
            
            if vessel_type is None or vessel_type==vt[0]:
                if color is not None:
                    col = color
                elif edge_color is not None:
                    col = cols[i]
                elif vt[0]==0: # artery
                    col = [1.,0.,0.]
                elif vt[1]==1:
                    col = [0.,0.,1.]
                else:
                    col = [0.,1.,0.]
                    
                poly = pv.PolyData()
                poly.points = coords
                the_cell = np.arange(0, len(coords), dtype=np.int_)
                the_cell = np.insert(the_cell, 0, len(coords))
                poly.lines = the_cell
                poly['radius'] = rads
                #tube = poly.tube(radius=rads[0],n_sides=3) # scalars='stuff', 
                
                tube = pv.Spline(coords, coords.shape[0]).tube(radius=rads[0])
                #breakpoint()
                tube['color'] = np.linspace(rads[0],rads[1],tube.n_points)
                #tubes.append(tube)
                tubes[i] = tube
                
                #if i>10000:
                #    break
                
        blocks = pv.MultiBlock(tubes)
        merged = blocks.combine()
        p = pv.Plotter()
        p.add_mesh(merged, smooth_shading=True) # scalars='length', 
        p.show()
        
    def plot(self,**kwargs):
        tp = self.plot_graph(**kwargs)
        return tp
        
    def plot_graph(self, **kwargs):
                         
        """
        Plot the graph using Open3d
        """
        
        from pymira.tubeplot import TubePlot
        
        tp = TubePlot(self, **kwargs)

        return tp 
        
    def smooth_radii(self,window=5,order=3,mode='savgol'):
    
        from scipy.signal import savgol_filter
        
        rad_field_name = self.get_radius_field_name()
        radius = self.get_data(rad_field_name)

        for e in range(self.nedge):
            edge = self.get_edge(e)
            rads = radius[edge.i0:edge.i1]
            x = np.cumsum(np.linalg.norm(edge.coordinates[1:]-edge.coordinates[:-1],axis=1))
            if len(rads)>window:
                if mode=='savgol':
                    radius[edge.i0:edge.i1] = savgol_filter(rads, window, order)
                elif mode=='movingav':
                    box = np.ones(window)/window
                    radius[edge.i0:edge.i1] = np.convolve(rads, box, mode='same')
        
        self.set_data(radius,name=rad_field_name)
        
    def identify_graphsOLD(self):
    
        graphInd = np.zeros(self.nnode,dtype='int')
        curGraph = 1
    
        inlet,outlet = self.identify_inlet_outlet()
        curnodes = []
        if inlet is not None:
            curnodes.append(inlet)
            graphInd[inlet] = curGraph
        if outlet is not None:
            curnodes.append(outlet)
            curGraph += 1
            graphInd[outlet] = curGraph
        if len(curnodes)==0:
            return None

        while True:
            visited_nodes,visited_edges = curnodes.copy(),[]

            while True:
                n_edge_added = 0
                nextnodes = []
                for node in curnodes:
                    connected_nodes,connected_edges = self.connected_nodes(node)
                    connected_nodes = [x for x in connected_nodes if x not in visited_nodes]
                    connected_edges = [x for x in connected_edges if x not in visited_edges]
                    
                    if len(connected_nodes)>0:
                        graphInd[arr(connected_nodes)] = graphInd[node] #curGraph
                        nextnodes.extend(connected_nodes)
                        visited_nodes.extend(connected_nodes)
                    if len(connected_edges)>0:
                        visited_edges.extend(connected_edges)
                        
                        n_edge_added += 1
                if n_edge_added==0:
                    break
                else:
                    curnodes = nextnodes
            
            if np.all(graphInd>0):
                break
            else:
                curGraph += 1
                sind = np.where(graphInd==0)[0]
                curnodes = np.random.choice(sind)
                graphInd[curnodes] = curGraph
                curnodes = [curnodes]
                
        return graphInd
        
    def identify_loops(self, return_paths=False,store_ranks=True):
    
        inlet,outlet = self.identify_inlet_outlet()
        paths = [[inlet]]
        edgepaths = [[-1]]
        visited_nodes,visited_edges = arr(paths.copy()).flatten(),[]
        loops = []
        ranks = np.zeros(self.nedge,dtype='int')

        count = -1
        while True:
            count += 1
            n_edge_added = 0
            nextpaths = copy.deepcopy(paths)
            nextedgepaths = copy.deepcopy(edgepaths)

            for i,path in enumerate(paths):
                node = path[-1]
                connected_nodes,connected_edges = self.connected_nodes(node)
                connected_node_inds = np.where(connected_nodes!=node)[0]
                connected_nodes = connected_nodes[connected_node_inds]
                connected_edges = connected_edges[connected_node_inds]

                # Look for nodes already visited by other paths
                l = arr([[n,e] for n,e in zip(connected_nodes,connected_edges) if n in visited_nodes and e not in visited_edges ])                
                if len(l)>0:
                    # Find where else node occurs
                    for j,path0 in enumerate(paths):
                        if i!=j:
                            mtch = [x for x in path0 if x==l[0,0]]
                            if len(mtch)>0:
                                nodepath1 = arr(paths[i]+[l[0,0]])
                                nodepath2 = arr(paths[j])
                                edgepath1 = arr(edgepaths[i]+[l[0,1]])
                                edgepath2 = arr(edgepaths[j])
                                # Find earliest common node
                                mnlen = np.min([len(nodepath1),len(nodepath2)])
                                eca = [k for k,x in enumerate(range(mnlen)) if nodepath1[k]!=nodepath2[k]][0]
                                n1 = np.where(nodepath1==l[0,0])
                                n2 = np.where(nodepath2==l[0,0])
                                
                                loop = np.hstack([edgepath1[eca:n1[0][0]+1],edgepath2[eca:n2[0][0]+1]])
                                loops.append(loop)
                                
                                #self.plot(fixed_radius=0.5,edge_highlight=loop)
                                #breakpoint()

                connected_node_inds = [k for k,x in enumerate(connected_nodes) if x not in visited_nodes]
                connected_nodes = connected_nodes[connected_node_inds]
                connected_edges = connected_edges[connected_node_inds]
                if len(connected_nodes)>0:
                    #breakpoint()
                    nextpaths[i].extend([connected_nodes[0]])
                    nextedgepaths[i].extend([connected_edges[0]])
                    
                    # Record ranks
                    parent_edge = edgepaths[i][-1]
                    if parent_edge<0:
                        ranks[connected_edges] = 1
                    else:
                        ranks[connected_edges] = ranks[parent_edge] + 1

                    if len(connected_nodes)>1:
                        for j,cn in enumerate(connected_nodes):
                            if j>0:
                                pathC = copy.deepcopy(path)
                                epathC = copy.deepcopy(edgepaths[i])
                                nextpaths.append(pathC[:-1]+[cn])
                                nextedgepaths.append(epathC[:-1]+[connected_edges[j]])
                    visited_nodes = np.concatenate([arr(visited_nodes).flatten(),connected_nodes])
                    visited_edges = np.concatenate([arr(visited_edges).flatten(),connected_edges])
                    
                    n_edge_added += 1
            if n_edge_added==0:
                paths = copy.deepcopy(nextpaths)
                edgepaths = copy.deepcopy(nextedgepaths)
                break
            else:
                paths = copy.deepcopy(nextpaths)
                edgepaths = copy.deepcopy(nextedgepaths)
                
        if store_ranks==True:
            if 'Ranks' in self.fieldNames:
                self.set_data(ranks,name='Ranks')
            else:
                f = self.add_field(name='Ranks',data=ranks,type='int',shape=[ranks.shape[0]])  
                
        if return_paths==True:
            return loops, paths, edgepaths
        else:
            return loops
        
    def remove_loops(self,
                    prefer: str = "short",    # "short" -> MST, "long" -> maximum spanning forest
                    use_polyline_length: bool = True,
                    dedupe_first: bool = True) -> dict:
        """
        Remove cycles from the spatial graph while preserving polylines and per-point scalars.

        prefer:
            "short" keeps the minimum total length per component (minimum spanning forest),
            "long"  keeps the maximum total length per component (maximum spanning forest).

        use_polyline_length:
            If True, edge weight is the polyline arc length if available, else straight-line node distance.
            If False, always use straight-line node distance between edge endpoints.

        dedupe_first:
            If True, remove exact duplicate undirected edges before cycle removal, keeping the best by weight.

        Returns
        -------
        summary dict with keys:
            kept_edges, dropped_edges, num_nodes, num_edges_before, num_edges_after
        """
        # ---- fetch data ----
        nodes = self.get_data('VertexCoordinates')
        edges = self.get_data('EdgeConnectivity')
        edge_points = self.get_data('EdgePointCoordinates')
        edge_point_counts = self.get_data('NumEdgePoints')
        scalar_fields = self.get_scalars()          # [{'name': ..., 'data': ...}, ...]
        scalars = [x['data'] for x in scalar_fields]

        # per_coords: list of (Ni,3)
        # per_scal:   list over edges, each is list over channels; per_scal[e][c] is (Ni,)
        per_coords, per_scal = self.unpack_edgepoints(edge_points, edge_point_counts, scalars=scalars)

        E = np.asarray(edges, dtype=int)
        N = len(nodes)

        # ---- helpers ----
        def poly_len(coords: np.ndarray) -> float:
            if coords is None or len(coords) < 2:
                return 0.0
            dif = np.diff(coords, axis=0)
            return float(np.linalg.norm(dif, axis=1).sum())

        def edge_weight(ei: int) -> float:
            u, v = int(E[ei, 0]), int(E[ei, 1])
            if use_polyline_length and per_coords[ei] is not None and len(per_coords[ei]) >= 2:
                return poly_len(per_coords[ei])
            pu, pv = nodes[u], nodes[v]
            return float(np.linalg.norm(pv - pu))

        # ---- drop self-loops upfront, keep arrays aligned ----
        self_loop_mask = (E[:, 0] == E[:, 1])
        if np.any(self_loop_mask):
            keep = np.where(~self_loop_mask)[0]
            E = E[keep]
            per_coords = [per_coords[i] for i in keep]
            per_scal = [per_scal[i] for i in keep]

        # ---- optional dedupe of parallel edges (undirected) ----
        if dedupe_first and len(E) > 0:
            key = np.sort(E, axis=1)
            # group by pair
            # pick best per pair depending on prefer and weight
            from collections import defaultdict
            buckets = defaultdict(list)
            for i, (a, b) in enumerate(key):
                buckets[(int(a), int(b))].append(i)

            keep_idx = []
            for pair, idxs in buckets.items():
                if len(idxs) == 1:
                    keep_idx.append(idxs[0])
                    continue
                # choose per weight
                ws = [edge_weight(i) for i in idxs]
                choice = int(np.argmin(ws)) if prefer == "short" else int(np.argmax(ws))
                keep_idx.append(idxs[choice])
            keep_idx = np.array(sorted(keep_idx), dtype=int)
            E = E[keep_idx]
            per_coords = [per_coords[i] for i in keep_idx]
            per_scal = [per_scal[i] for i in keep_idx]

        # ---- compute weights ----
        M = len(E)
        if M == 0:
            # write back trivial result
            new_pts, new_cnt, new_scal = self.repack_edge_data(per_coords, per_scal)
            self.set_data(nodes, name='VertexCoordinates')
            self.set_data(E, name='EdgeConnectivity')
            self.set_data(new_pts, name='EdgePointCoordinates')
            self.set_data(new_cnt, name='NumEdgePoints')
            for i, sc in enumerate(new_scal):
                self.set_data(sc, name=scalar_fields[i]['name'])
            self.set_definition_size('VERTEX', nodes.shape[0])
            self.set_definition_size('EDGE', 0)
            self.set_definition_size('POINT', new_pts.shape[0])
            self.set_graph_sizes()
            return dict(kept_edges=[], dropped_edges=[], num_nodes=nodes.shape[0],
                        num_edges_before=0, num_edges_after=0)

        W = np.array([edge_weight(i) for i in range(M)], dtype=float)
        order = np.argsort(W) if prefer == "short" else np.argsort(-W)

        # ---- Kruskal union-find ----
        parent = np.arange(N, dtype=int)
        rank = np.zeros(N, dtype=int)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[rb] < rank[ra]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        keep_edges_idx = []
        drop_edges_idx = []

        for ei in order:
            u, v = int(E[ei, 0]), int(E[ei, 1])
            if union(u, v):
                keep_edges_idx.append(ei)
            else:
                drop_edges_idx.append(ei)

        keep_edges_idx = np.array(sorted(keep_edges_idx), dtype=int)
        drop_edges_idx = np.array(sorted(drop_edges_idx), dtype=int)

        # ---- filter arrays to kept set ----
        E_kept = E[keep_edges_idx]
        per_coords_kept = [per_coords[i] for i in keep_edges_idx]
        per_scal_kept = [per_scal[i] for i in keep_edges_idx]

        # ---- write back to SpatialGraph ----
        new_pts, new_cnt, new_scal = self.repack_edge_data(per_coords_kept, per_scal_kept)

        self.set_data(nodes, name='VertexCoordinates')
        self.set_data(E_kept, name='EdgeConnectivity')
        self.set_data(new_pts, name='EdgePointCoordinates')
        self.set_data(new_cnt, name='NumEdgePoints')
        for i, sc in enumerate(new_scal):
            self.set_data(sc, name=scalar_fields[i]['name'])

        self.set_definition_size('VERTEX', nodes.shape[0])
        self.set_definition_size('EDGE', E_kept.shape[0])
        self.set_definition_size('POINT', new_pts.shape[0])
        self.set_graph_sizes()

        return dict(
            kept_edges=keep_edges_idx.tolist(),
            dropped_edges=drop_edges_idx.tolist(),
            num_nodes=nodes.shape[0],
            num_edges_before=M,
            num_edges_after=E_kept.shape[0],
        )

    def calculate_ranks(self):
        #if not self.test_treelike():
        #    return 0
        
        ranks = np.zeros(self.nedge,dtype='int')

        inlet,outlet = self.identify_inlet_outlet()
        
        for i in range(2):
            if i==0 and inlet is not None:
                curnodes = [inlet]
            elif i==1 and outlet is not None:
                curnodes = [outlet]
            else:
                curnodes = []
            visited_nodes,visited_edges = curnodes.copy(),[]
            
            curRank = 1

            while True:
                n_edge_added = 0
                nextnodes = []
                for node in curnodes:
                    connected_nodes,connected_edges = self.connected_nodes(node)
                    connected_nodes = [x for x in connected_nodes if x not in visited_nodes]
                    connected_edges = [x for x in connected_edges if x not in visited_edges]
                    
                    if len(connected_edges)>0:
                        ranks[arr(connected_edges)] = curRank
                    
                        nextnodes.extend(connected_nodes)
                        visited_nodes.extend(connected_nodes)
                        visited_edges.extend(connected_edges)
                        
                        n_edge_added += 1
                if n_edge_added==0:
                    break
                else:
                    curRank += 1
                    curnodes = nextnodes

        if 'Ranks' in self.fieldNames:
            self.set_data(ranks,name='Ranks')
        else:
            f = self.add_field(name='Ranks',definition='EDGE',data=ranks,type='int',shape=[ranks.shape[0]])  
            
    def get_edges_connected_to_edge(self, edgeInd):
    
        edge = self.get_edge(edgeInd)
        # Edges connected to start node
        es = self.get_edges_containing_node(edge.start_node_index)
        es = es[es!=edgeInd]
        # Edges connected to end node
        ee = self.get_edges_containing_node(edge.end_node_index)
        ee = ee[ee!=edgeInd]
        
        return [es,ee]
        
    def get_nodes_connected_to_node(self, nodeInd):
       
        edgeconn = self.get_data('EdgeConnectivity')
        nodes = edgeconn[(np.in1d(edgeconn[:,0],nodeInd)) | (np.in1d(edgeconn[:,1],nodeInd))]
        nodes = np.unique(nodes)
        nodes = nodes[nodes!=nodeInd]
        return nodes

    def get_subgraph_by_rank(self,edgeInd):
    
        # Get subgraph consisting of edges with a higher rank than the supplied edge
        
        if not self.test_treelike():
            return
            
        edge = self.get_edge(edgeInd)   

        ranks = self.get_data('Ranks')
        if ranks is None:
            self.calculate_ranks()
            ranks = self.get_data('Ranks')
          
        # Starting rank      
        r0 = ranks[edgeInd]
        edgeStore = [edgeInd]
        
        es,ee = self.get_edges_connected_to_edge(edgeInd) 
        es = np.concatenate([es,ee])
        esr = ranks[es]        
        
        # Get edges with a higher rank than starting edge
        es = es[esr>r0]
        edgeStore.extend(es)
        
        curEdges = es
        
        while True:
            # Get all connected edges and find their ranks
            nadded,nextEdges = 0,[]
            for ce in curEdges:
                es,ee = self.get_edges_connected_to_edge(ce) 
                es = np.concatenate([es,ee])
                es = [x for x in es if x not in edgeStore]

                if len(es)>0:                
                    edgeStore.extend(es)
                    nadded += len(es)
                    nextEdges.extend(es)
                
            if nadded==0:
                break
            else: 
                curEdges = nextEdges
        
        return edgeStore
        
    def _translate_edge_coords(self,nodeIndex,e,coords=None,displacement=None):
    
        """
        nodeIndex: index of node to be moved to new coordinates (coords/displacement)
        e: edge object
        coords: new location of nodeIndex (or displacement)
        """
    
        edgepoints = self.get_data('EdgePointCoordinates').copy()
    
        if e.start_node_index==nodeIndex:
            old_coords = e.start_node_coords
        elif e.end_node_index==nodeIndex:
            old_coords = e.end_node_coords
        else:
            return None
            
        new_points = e.coordinates.copy()
        new_coords = old_coords
        
        if e.npoints==2:
            if e.start_node_index==nodeIndex:
                if coords is None and displacement is not None:
                    new_coords = e.start_node_coords + displacement
                elif coords is not None:
                    new_coords = coords
                else:
                    return
                dist1 = np.linalg.norm(e.end_node_coords-new_coords)
                if dist1==0.:
                    return
                edgepoints[e.i0] = new_coords
            elif e.end_node_index==nodeIndex:
                if coords is None and displacement is not None:
                    new_coords = e.end_node_coords + displacement
                elif coords is not None:
                    new_coords = coords
                else:
                    return
                dist1 = np.linalg.norm(e.start_node_coords-new_coords)
                if dist1==0.:
                    return
                edgepoints[e.i0+1] = new_coords
            else:
                breakpoint()
        else:
        
            if e.start_node_index==nodeIndex:
                                    
                if coords is None and displacement is not None:
                    new_coords = e.start_node_coords + displacement
                elif coords is not None:
                    new_coords = coords
                else:
                    return
            
                v0 = e.end_node_coords-old_coords
                v1 = e.end_node_coords-new_coords
                dist0 = np.linalg.norm(e.end_node_coords-e.start_node_coords)
                dist1 = np.linalg.norm(e.end_node_coords-new_coords)
                translated_points = e.coordinates - e.coordinates[-1]
            else:
                if coords is None and displacement is not None:
                    new_coords = e.end_node_coords + displacement
                elif coords is not None:
                    new_coords = coords
                else:
                    return
            
                v0 = e.start_node_coords-old_coords
                v1 = e.start_node_coords-new_coords
                dist0 = np.linalg.norm(e.start_node_coords-e.end_node_coords)
                dist1 = np.linalg.norm(e.start_node_coords-new_coords)
                translated_points = e.coordinates - e.coordinates[0]
            if np.any(np.isfinite(translated_points)==False) or dist0==0. or dist1==0.:
                #breakpoint()
                return
            
            scale_factor = dist1 / dist0
                
            u0 = v0 / dist0
            u1 = v1 / dist1
            rotation_axis = np.cross(u0, u1)
            axis_magnitude = np.linalg.norm(rotation_axis)
            
            if axis_magnitude!=0:
                rotation_axis /= axis_magnitude  # Normalize the rotation axis
                angle = np.arccos(np.clip(np.dot(u0, u1), -1.0, 1.0))  # Angle between the two vectors
                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],[rotation_axis[2], 0, -rotation_axis[0]],[-rotation_axis[1], rotation_axis[0], 0]])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                
                if e.start_node_index==nodeIndex:
                    new_points = scale_factor * np.dot(R,translated_points.transpose()).transpose() + e.end_node_coords
                    new_points[0] = new_coords
                    new_points[-1] = e.end_node_coords
                elif e.end_node_index==nodeIndex:
                    new_points = scale_factor * np.dot(R,translated_points.transpose()).transpose() + e.start_node_coords
                    new_points[-1] = new_coords
                    new_points[0] = e.start_node_coords
                else:
                    breakpoint()
            else:
                # Rotation angle is zero
                if scale_factor!=1.:
                    if e.start_node_index==nodeIndex:
                        new_points = scale_factor * translated_points + e.end_node_coords
                    elif e.end_node_index==nodeIndex:
                        new_points = scale_factor * translated_points + e.start_node_coords
                        
            if np.any(np.isfinite(new_points)==False):
                breakpoint()
            edgepoints[e.i0:e.i1] = new_points
            
        self.set_data(edgepoints,name='EdgePointCoordinates')
        
        return new_coords
     
    def move_node(self,nodeIndex,coords=None,displacement=None):
    
        """
        Move nodeIndex to coords and translate any connected edge
        """
    
        nodes = self.get_data('VertexCoordinates').copy()
        edgepoints = self.get_data('EdgePointCoordinates') #.copy()
        conn_edges = self.get_edges_containing_node(nodeIndex)
        old_coords = nodes[nodeIndex]
        
        for ei in conn_edges:
            e = self.get_edge(ei)

            if np.all(np.all(e.coordinates==e.coordinates[0],axis=0)):
                breakpoint()
            
            new_coords = self._translate_edge_coords(nodeIndex,e,coords=coords,displacement=displacement)
            
            if False:
                plt.ion()
                plt.scatter(e.coordinates[:,0],e.coordinates[:,1],c='b')
                plt.scatter(coords[0],coords[1],c='g')
                if e.start_node_index==nodeIndex:
                    plt.scatter(e.coordinates[0,0],e.coordinates[0,1],c='r')
                elif e.end_node_index==nodeIndex:
                    plt.scatter(e.coordinates[-1,0],e.coordinates[-1,1],c='r')
                plt.scatter(new_points[:,0],new_points[:,1],c='orange')

        nodes[nodeIndex] = new_coords

        self.set_data(nodes,name='VertexCoordinates')
        #self.set_data(edgepoints,name='EdgePointCoordinates')
        
        return conn_edges
        
    def get_segments(self,return_edge=False,return_counts=False,domain=None):
        nedgepoint = self.get_data(name='NumEdgePoints') 
        edgepoints = self.get_data('EdgePointCoordinates')
        epi = self.edge_point_index()
        
        gc = self.get_node_count()
            
        segments = np.hstack([edgepoints[:-1],edgepoints[1:]])
        segments = segments.reshape([segments.shape[0],2,3])
        epi_seg = np.vstack([epi[:-1],epi[1:]]).transpose()
        valid = epi_seg[:,0]==epi_seg[:,1]
        
        if domain is not None:
            ins = (np.all(segments[:,0]>=domain[:,0],axis=1)) & (np.all(segments[:,0]<=domain[:,1],axis=1)) & \
                  (np.all(segments[:,1]>=domain[:,0],axis=1)) & (np.all(segments[:,1]<=domain[:,1],axis=1))
            valid = valid & ins
        
        segments = segments[valid]
        segment_edges = epi_seg[valid,0]
        #segment_points = np.linspace(0,np.sum(nedgepoint)-1,np.sum(nedgepoint),dtype='int')-np.repeat(np.cumsum(nedgepoint)-nedgepoint,nedgepoint)
        #segment_points = segment_points.reshape([int(segment_points.shape[0]/2),2])
        p0 = np.concatenate([np.linspace(0,x-2,x-1,dtype='int') for x in nedgepoint])

        # Which segment points correspond to a node (-1 if none)
        edgeconn = self.get_data('EdgeConnectivity')
        segment_node = np.concatenate([np.concatenate([[arr(edgeconn[i,0])],np.repeat(-1,x-2),[arr(edgeconn[i,1])]]) for i,x in enumerate(nedgepoint)])
        # How many connections for each segment point
        sc = gc[segment_node]
        sc[segment_node<0] = 2
        segment_points = np.vstack([p0,p0+1]).transpose()
        
        if return_edge==False:
            return segments
        elif return_counts:
            return segments,segment_edges,segment_points,segment_node,sc
        else:
            return segments,segment_edges,segment_points
            
    def check_for_nan(self):
        nodes = self.get_data('VertexCoordinates')
        if np.any(~np.isfinite(nodes)):
            inds = np.where(~np.isfinite(nodes))
            return True, 0, inds[0]
        points = self.get_data('EdgePointCoordinates')
        if np.any(~np.isfinite(points)):
            inds = np.where(~np.isfinite(points))
            return True, 1, inds[0]
        return False, -1, None

class Editor(object):     

    def _insert_node_in_edge(self,edge_index,edgepoint_index,nodeCoords,edgeConn,nedgepoints,edgeCoords,scalars=None):
    
        # Returns the new node index and the two new edges (if any are made)
    
        nnode = len(nodeCoords)
        nedge = len(edgeConn)
        nedgepoint = len(edgeCoords)
        
        x0 = int(np.sum(nedgepoints[:int(edge_index)]))
        x1 = x0 + int(nedgepoints[int(edge_index)])
        edge = edgeCoords[x0:x1]
        npoints = edge.shape[0]
        
        xp = int(edgepoint_index)
        new_node_coords = edge[xp]
        
        start_node = edgeConn[edge_index,0]
        end_node = edgeConn[edge_index,1]
        
        if int(edgepoint_index)<npoints-1 and int(edgepoint_index)>0:
            new_edge0 = edge[:xp+1]
            new_edge1 = edge[xp:]
        elif int(edgepoint_index)<0:
            return edge, None, start_node, None, nodeCoords, edgeConn, nedgepoints, edgeCoords, scalars
        elif int(edgepoint_index)==npoints-1:
            print('ERROR: _insert_node_in_edge: Edgepoint index>number of edgepoints!')
            return edge, None, end_node, None, nodeCoords, edgeConn, nedgepoints, edgeCoords, scalars
        else:
            return [None]*9
            
        # Assign the first new edge to the location of the supplied edge
        # Create a new location for the second new edge
        nedgepoints[int(edge_index)] = new_edge0.shape[0]
        nedgepoints = np.concatenate([nedgepoints,[new_edge1.shape[0]]])
        
        # Squeeze in new edges into storage array
        # Grab all edge coordinates prior to edge to be bisected
        if x0>0:
            edgeCoords_0 = edgeCoords[:x0]
        else:
            edgeCoords_0 = []
        # Edge coordinates listed after the bisected edge
        if edgeCoords.shape[0]>x0+npoints:
            edgeCoords_1 = edgeCoords[x1:]
        else:
            edgeCoords_1 = []

        edgeCoords = np.concatenate([x for x in [edgeCoords_0,new_edge0.copy(),edgeCoords_1,new_edge1.copy()] if len(x)>0 and not np.all(x)==-1])
        
        # Amend original connection
        new_node_index = nodeCoords.shape[0]
        edgeConn[edge_index] = [start_node,new_node_index]
        new_conn = np.asarray([new_node_index,end_node])
        edgeConn = np.concatenate([edgeConn,[new_conn]])
        new_edge_index = nedge
        # Add in new node coords
        nodeCoords = np.concatenate([nodeCoords,[new_node_coords]])
        
        # Sort out scalars
        for i,data in enumerate(scalars):
            if x0>0:
                sc_0 = data[:x0]
            else:
                sc_0 = []
            if data.shape[0]>x0+npoints:
                sc_1 = data[x1:]
            else:
                sc_1 = []
            new_sc0 = data[x0:x0+xp+1]
            new_sc1 = data[x0+xp:x1]
            #scalars[i] = np.concatenate([sc_0,new_sc0,sc_1,new_sc1])
            scalars[i] = np.concatenate([x for x in [sc_0,new_sc0.copy(),sc_1,new_sc1.copy()] if len(x)>0 and not np.all(x)==-1])
           
        return new_edge0.copy(), new_edge1.copy(), new_node_index,new_conn,nodeCoords,edgeConn,nedgepoints,edgeCoords,scalars

    def _del_nodes(self,nodes_to_delete,nodeCoords,edgeConn,nedgepoints,edgeCoords,scalars=[]):
    
        nnode = len(nodeCoords)
        nedge = len(edgeConn)
        nedgepoint = len(edgeCoords)
        
        nodes_to_keep = [x for x in range(nnode) if x not in nodes_to_delete]
        nodeCoords_ed = np.asarray([nodeCoords[x] for x in nodes_to_keep])
        
        # Find connected edges
        keepEdge = np.in1d(edgeConn, nodes_to_keep).reshape(edgeConn.shape)
        keepEdge = np.asarray([all(x) for x in keepEdge])
        edges_to_delete = np.where(keepEdge==False)[0]
        edges_to_keep = np.where(keepEdge==True)[0]
        edgeConn_ed = np.asarray([edgeConn[x] for x in edges_to_keep])

        # Offset edge indices to 0
        unqNodeIndices = nodes_to_keep
        nunq = len(unqNodeIndices)
        newInds = np.arange(nunq)            
        edgeConn_ed_ref = np.zeros(edgeConn_ed.shape,dtype='int') - 1
        edgeConn_was = np.zeros(edgeConn_ed.shape,dtype='int') - 1
        # Update edge indices
        for i,u in enumerate(unqNodeIndices):
            sInds = np.where(edgeConn_ed==u)
            newIndex = newInds[i]
            if len(sInds[0])>0:
                edgeConn_ed_ref[sInds[0][:],sInds[1][:]] = newIndex #newInds[i]
                edgeConn_was[sInds[0][:],sInds[1][:]] = u
        edgeConn_ed = edgeConn_ed_ref

        # Modify edgepoint number array
        nedgepoints_ed = np.asarray([nedgepoints[x] for x in edges_to_keep])

        # Mark which edgepoints to keep / delete
        keepEdgePoint = np.zeros(nedgepoint,dtype='bool') + True
        for edgeIndex in edges_to_delete:
            npoints = nedgepoints[edgeIndex]
            strt = np.sum(nedgepoints[0:edgeIndex])
            fin = strt + npoints
            keepEdgePoint[strt:fin] = False

        # Modify edgepoint coordinates
        edgeCoords_ed = edgeCoords[keepEdgePoint==True] #np.asarray([edgeCoords[x] for x in edgepoints_to_keep)
        
        #Check for any other scalar fields
        if nedgepoint!=len(edgeCoords_ed):
            for i,data in enumerate(scalars):
                scalars[i] = data[keepEdgePoint==True]
                
        info = {'edges_deleted':edges_to_delete,'edges_kept':edges_to_keep,'points_kept':keepEdgePoint,'nodes_deleted':nodes_to_delete,'nodes_kept':nodes_to_keep}
        
        return nodeCoords_ed,edgeConn_ed,nedgepoints_ed,edgeCoords_ed,scalars,info
    
    def delete_nodes(self,graph,nodes_to_delete):
        
        nodeCoords = graph.get_data('VertexCoordinates')
        edgeConn = graph.get_data('EdgeConnectivity')
        nedgepoints = graph.get_data('NumEdgePoints')
        edgeCoords = graph.get_data('EdgePointCoordinates')
        
        nnode = len(nodeCoords)
        nedge = len(edgeConn)
        nedgepoint = len(edgeCoords)
        
        # Look for scalars that need updating (must be POINT type)
        scalars, scalar_names = [],[]
        for f in graph.fields:
            if f['definition'].lower()=='point' and len(f['shape'])==1:
                scalars.append(f['data'])
                scalar_names.append(f['name'])
        if len(scalars)==0:
            scalars = None

        nodeCoords_ed,edgeConn_ed,nedgepoints_ed,edgeCoords_ed,scalars,info = self._del_nodes(nodes_to_delete,nodeCoords,edgeConn,nedgepoints,edgeCoords,scalars=scalars)
        
        node_scalars = graph.get_node_scalars()
        node_to_keep = np.ones(nnode,dtype='bool')
        node_to_keep[nodes_to_delete] = False
        for sc in node_scalars:
            graph.set_data(sc['data'][node_to_keep],name=sc['name'])

        # Update VERTEX definition
        vertex_def = graph.get_definition('VERTEX')
        vertex_def['size'] = [nodeCoords_ed.shape[0]]
        # Update EDGE definition
        edge_def = graph.get_definition('EDGE')
        edge_def['size'] = [edgeConn_ed.shape[0]]
        # Update POINT definition
        edgepoint_def = graph.get_definition('POINT')
        edgepoint_def['size'] = [edgeCoords_ed.shape[0]]
        
        graph.set_data(nodeCoords_ed,name='VertexCoordinates')
        graph.set_data(edgeConn_ed,name='EdgeConnectivity')
        graph.set_data(nedgepoints_ed,name='NumEdgePoints')
        graph.set_data(edgeCoords_ed,name='EdgePointCoordinates')
        
        #Check for any other scalar fields
        if nedgepoint!=len(edgeCoords_ed):
            for i,data in enumerate(scalars):
                graph.set_data(data,name=scalar_names[i])
            
        graph.set_graph_sizes()
        return graph
        
    def delete_edges(self,graph,edges_to_delete,remove_disconnected_nodes=True):
        
        nodeCoords = graph.get_data('VertexCoordinates')
        edgeConn = graph.get_data('EdgeConnectivity')
        nedgepoints = graph.get_data('NumEdgePoints')
        edgeCoords = graph.get_data('EdgePointCoordinates')
        
        #nnode = len(nodeCoords)
        nedge = len(edgeConn)
        nedgepoint = len(edgeCoords)
        
        edges_to_keep = np.asarray([x for x in range(nedge) if x not in edges_to_delete])
        edgeConn_ed = np.asarray([edgeConn[x] for x in edges_to_keep])

        # Modify edgepoint number array
        nedgepoints_ed = np.asarray([nedgepoints[x] for x in edges_to_keep])

        # Mark which edgepoints to keep / delete
        keepEdgePoint = np.zeros(nedgepoint,dtype='bool') + True
        for edgeIndex in edges_to_delete:
            npoints = nedgepoints[edgeIndex]
            strt = np.sum(nedgepoints[0:edgeIndex])
            fin = strt + npoints
            keepEdgePoint[strt:fin] = False

        # Modify edgepoint coordinates
        edgeCoords_ed = edgeCoords[keepEdgePoint==True] #np.asarray([edgeCoords[x] for x in edgepoints_to_keep)

        # Update EDGE definition
        edge_def = graph.get_definition('EDGE')
        edge_def['size'] = [len(edges_to_keep)]
        # Update POINT definition
        edgepoint_def = graph.get_definition('POINT')
        edgepoint_def['size'] = [len(edgeCoords_ed)]
        
        #graph.set_data(nodeCoords_ed,name='VertexCoordinates')
        graph.set_data(edgeConn_ed,name='EdgeConnectivity')
        graph.set_data(nedgepoints_ed,name='NumEdgePoints')
        graph.set_data(edgeCoords_ed,name='EdgePointCoordinates')
        
        #Check for any other scalar fields
        scalars = [f for f in graph.fields if f['definition'].lower()=='point' and len(f['shape'])==1]
        print('Updating scalars...')
        for sc in scalars:
            #data_ed = np.delete(sc['data'],edgepoints_to_delete[0],axis=0)
            data = sc['data']
            data_ed = data[keepEdgePoint==True]
            graph.set_data(data_ed,name=sc['name'])
            
        graph.set_graph_sizes()
        
        if remove_disconnected_nodes:
            graph = self.remove_disconnected_nodes(graph)
        
        return graph
        
    def delete_edgepoints(self,graph,edgepoints_to_delete):
        point_to_edge = graph.edgepoint_edge_indices()
        ###
        
    def remove_disconnected_nodes(self,graph):
        nodeCoords = graph.get_data('VertexCoordinates')
        gc = graph.get_node_count()

        zero_conn = np.where(gc==0)
        if len(zero_conn[0])==0:
            return graph
            
        graph = self.delete_nodes(graph,zero_conn[0])
        print(('{} isolated nodes removed'.format(len(zero_conn[0]))))
        return graph
        
    def remove_selfconnected_edges(self,graph):
        nodeCoords = graph.get_data('VertexCoordinates')
        nodeInds = np.arange(0,nodeCoords.shape[0]-1)
        edgeConn = graph.get_data('EdgeConnectivity')
        
        nedge = len(edgeConn)
        selfconn = [i for i,x in enumerate(edgeConn) if x[0]==x[1]]
        if len(selfconn)==0:
            return graph
            
        print('Removing {} self-connected edges...'.format(len(selfconn)))
        self.delete_edges(graph,selfconn,remove_disconnected_nodes=False)
        return graph
        
    def simplify_edges(self,graph,factor=2.,fixed=None,exclude=[]):
        
        nodecoords = graph.get_data('VertexCoordinates')
        edgeconn = graph.get_data('EdgeConnectivity')
        points = graph.get_data('EdgePointCoordinates')
        nedge = graph.get_data('NumEdgePoints')
        scalars = graph.get_scalars()
        nscalars = graph.get_node_scalars()
        
        scalars = graph.get_scalars()
        scalar_data = [x['data'] for x in scalars]
        scalar_type = [str(x.dtype) for x in scalar_data]
        scalar_data_interp = [[] for x in scalars]
        
        points_new = points.copy() * 0.
        nedge_new = np.zeros(graph.nedge,dtype='int')
        
        e_counter = 0
        for i in range(graph.nedge):
            edge = graph.get_edge(i)
            
            if i in exclude:
                nn = edge.npoints
            elif fixed is None:
                nn = np.clip(int(np.ceil(edge.npoints / float(factor))),2,None)
            else:
                nn = fixed
            pts = edge.coordinates
            
            if nn!=edge.npoints:
                from scipy import interpolate
                try:
                    if nn<=4:
                        pcur = np.linspace(pts[0],pts[-1],nn)
                    else:
                        k = 1
                        # Interpolate fails if all values are equal (to zero?)
                        # This most commonly happens in z-direction, for retinas at least, so add noise and remove later
                        if np.all(pts[:,2]==pts[0,2]):
                            z = pts[:,2] + np.random.normal(0.,0.1,pts.shape[0])
                        else:
                            z = pts[:,2]
                        tck, u = interpolate.splprep([pts[:,0], pts[:,1], z],k=k,s=0) #, s=2)
                        u_fine = np.linspace(0,1,nn)
                        pcur = np.zeros([nn,3])
                        pcur[:,0], pcur[:,1], pcur[:,2] = interpolate.splev(u_fine, tck)
                        if np.all(pts[:,2]==pts[0,2]):
                            pcur[:,2] = pts[0,2]
                except Exception as e:
                    breakpoint()

            else:
                pcur = pts
                           
            for j,sd in enumerate(scalar_data):
                sdc = sd[edge.i0:edge.i1]
                if 'float' in scalar_type[j]:
                    scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[-1],nn))
                elif 'int' in scalar_type[j]:
                    if sdc[0]==sdc[-1]:
                        scalar_data_interp[j].extend(np.zeros(nn)+sdc[0])
                    else:
                        scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[-1],nn,dtype='int'))
                elif 'bool' in scalar_type[j]:
                    scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[-1],nn,dtype='bool'))
                else:
                    breakpoint()
                
            points_new[e_counter:e_counter+nn] = pcur
            nedge_new[i] = nn
            e_counter += nn
            
        graph.set_data(points_new[:e_counter], name='EdgePointCoordinates')
        graph.set_data(nedge_new, name='NumEdgePoints')
        
        for j,sd in enumerate(scalar_data_interp):
            graph.set_data(arr(sd),name=scalars[j]['name'])
        
        graph.set_definition_size('POINT',e_counter)   
        graph.set_graph_sizes()     
        
        return graph
        
    def remove_intermediate_nodes(self,graph):
        """
        Remove nodes that have exactly two connections, and add them into the edge data
        """
        
        nodecoords = graph.get_data('VertexCoordinates')
        edgeconn = graph.get_data('EdgeConnectivity')
        points = graph.get_data('EdgePointCoordinates')
        nedge = graph.get_data('NumEdgePoints')
        scalars = graph.get_scalars()
        nscalars = graph.get_node_scalars()
        
        nedges = edgeconn.shape[0]
        nnode = nodecoords.shape[0]
        
        node_count = graph.get_node_count()
        inline_nodes = np.where(node_count==2)
        
        edge_del_flag = np.zeros(nedges,dtype='bool')
        node_del_flag = np.zeros(nnode,dtype='bool')
        
        consol_edge = []
        # Loop through each of the inline nodes (i.e. nodes with exactly 2 connections)
        for i,_node_inds in enumerate(inline_nodes[0]):
            consolodated_edges = []
            start_or_end_node = []
            node_inds = [_node_inds]
            
            # Track which nodes are connected to inline nodes, and aggregate chains of inline nodes        
            count = 0
            while True:
                next_nodes = []
                for node_ind in node_inds:
                    if node_del_flag[node_ind]==False and node_count[node_ind]==2:
                        node_del_flag[node_ind] = True
                        cur_conn = np.where(((edgeconn[:,0]==node_ind) | (edgeconn[:,1]==node_ind)) & (edge_del_flag==False))
                        
                        # Mark the edges as needing to be consolodated 
                        consolodated_edges.append(cur_conn[0])
                        edge_del_flag[cur_conn[0]] = True
                        conn_nodes = np.unique(edgeconn[cur_conn].flatten())
                        conn_nodes = conn_nodes[conn_nodes!=node_ind]
                        conn_count = node_count[conn_nodes]
                    
                        # Look for endpoints or branching points
                        edge_or_branch = np.where((conn_count==1) | (conn_count>2))
                        if len(edge_or_branch[0])>0:
                            start_or_end_node.append(conn_nodes[edge_or_branch[0]])
                            
                        if len(start_or_end_node)>=2:
                            break
                           
                        if len(conn_nodes)>0: 
                            next_nodes.append(conn_nodes)
                        count += 1
                        #print(next_nodes)
                    
                if len(start_or_end_node)>=2:
                    break
                else:
                    # Next iteration prep.
                    if len(next_nodes)==0:
                        break
                    node_inds = np.concatenate(next_nodes)

            # Aggregate identified edges containing inline nodes (will remove loops!)
            if count>0 and len(start_or_end_node)>0:
                consolodated_edges = np.concatenate(consolodated_edges)
                start_or_end_node = np.concatenate(start_or_end_node)
                
                # Merge the edges into one
                cur_conns = edgeconn[consolodated_edges]
                start_node, end_node = start_or_end_node[0],start_or_end_node[1]
                cur_node = start_node
                
                consol_points = []
                consol_scalars = [[] for _ in range(len(scalars))]
                visited = np.zeros(len(consolodated_edges),dtype='bool')
                count1 = 0
                while True:
                    cur_edge_ind = consolodated_edges[np.where(((cur_conns[:,0]==cur_node) | (cur_conns[:,1]==cur_node)) & (visited==False))][0]
                    visited[np.where(consolodated_edges==cur_edge_ind)] = True
                    
                    cur_edge = edgeconn[cur_edge_ind]
                    x0 = np.sum(nedge[:cur_edge_ind])
                    x1 = x0 + nedge[cur_edge_ind]
                    cur_pts = points[x0:x1]
                    
                    if cur_edge[0]==cur_node:
                        # Correct way round
                        consol_points.append(cur_pts)
                        next_node = cur_edge[1]
                        for j,sc in enumerate(scalars):
                            consol_scalars[j].append(sc['data'][x0:x1])
                    else:
                        #breakpoint()
                        consol_points.append(np.flip(cur_pts,axis=0))
                        next_node = cur_edge[0]
                        for j,sc in enumerate(scalars):
                            consol_scalars[j].append(np.flip(sc['data'][x0:x1],axis=0))
                        
                    if next_node==end_node:
                        break
                    else:
                        cur_node = next_node
                    count1 += 1
                        
                consol_points = np.concatenate(consol_points)
                consol_edge.append({'start_node':start_node,'end_node':end_node,'points':consol_points,'scalars':consol_scalars})
                
        # Delete inline edges
        #edgeconn = edgeconn[~edge_del_flag]
        
        # Add new edges to graph
        scalar_names = [x['name'] for x in scalars]
        for edge in consol_edge:
            new_conn = arr([edge['start_node'],edge['end_node']])
            new_pts = edge['points']
            # Create indices defining the first, last and every second index in between
            skip_inds = np.arange(0,len(new_pts),2)
            if new_pts.shape[0]%2==0:
                skip_inds = np.concatenate([skip_inds,[new_pts.shape[0]-1]])
            new_pts = new_pts[skip_inds]
            
            if not np.all(new_pts[0]==nodecoords[new_conn[0]]) or not np.all(new_pts[-1]==nodecoords[new_conn[1]]):
                breakpoint()
            
            new_npts = new_pts.shape[0]
            edgeconn = np.vstack((edgeconn,new_conn))
            nedge = np.concatenate((nedge,[new_npts]))
            points = np.vstack((points,new_pts))

            for j,sc in enumerate(scalars):
                if scalar_names[j]=='EdgeLabel':
                    new_data = np.concatenate(edge['scalars'][j])[skip_inds]
                    new_data[:] = graph.unique_edge_label() #  np.min(new_data)
                else:
                    new_data = np.concatenate(edge['scalars'][j])[skip_inds]
                sc['data'] = np.concatenate((sc['data'],new_data))
        
        graph.set_data(edgeconn,name='EdgeConnectivity')
        graph.set_data(points,name='EdgePointCoordinates')
        graph.set_data(nedge,name='NumEdgePoints')
        graph.set_definition_size('EDGE',edgeconn.shape[0])
        graph.set_definition_size('POINT',points.shape[0])

        # Delete inline nodes and edges connecting them
        node_del_flag[node_count==0] = True
        graph = delete_vertices(graph,~node_del_flag,return_lookup=False)
        
        return graph        
        
    def largest_graph(self, graph):

        graphNodeIndex, graph_size = graph.identify_graphs()
        unq_graph_indices, graph_size = np.unique(graphNodeIndex, return_counts=True)
        largest_graph_index = np.argmax(graph_size)
        node_indices = np.arange(graph.nnode)
        nodes_to_delete = node_indices[graphNodeIndex!=largest_graph_index]
        graph = self.delete_nodes(graph,nodes_to_delete)
        
        return graph
        
    def remove_graphs_smaller_than(self, graph, lim, pfile=None):

        if True: #pfile is None:
            graphNodeIndex, graph_size = graph.identify_graphs(progBar=True)
        else:
            import pickle
            plist = pickle.load(open(pfile,"r"))
            graphNodeIndex, graph_size = plist[0],plist[1]
            
        unq_graph_indices, graph_size = np.unique(graphNodeIndex, return_counts=True)
            
        graph_index_to_delete = np.where(graph_size<lim)
        if len(graph_index_to_delete)==0:
            return graph
            
        nodes_to_delete = []
        for gitd in np.hstack([unq_graph_indices[graph_index_to_delete],-1]):
            inds = np.where(graphNodeIndex==gitd)
            if len(inds)>0:
                nodes_to_delete.extend(inds[0].tolist())
        nodes_to_delete = np.asarray(nodes_to_delete)

        graph = self.delete_nodes(graph,nodes_to_delete)
        graph.set_graph_sizes()
        
        return graph
        
    def filter_graph(self,graph,parameter='Radii',min_value=None,max_value=None,clip_value=None,write=False,ofile='',keep_stubs=False,stub_len=100.,ignore=None,return_edge_inds=False):

        """
        min_value: All edges with radii < this value are deleted
        clip_value: All edges with radii < this value and > min_value are set to clip_value    
        """

        #nodecoords, edgeconn, edgepoints, nedgepoints, radius, category, mlp = get_graph_fields(graph)
        
        nodecoords = graph.get_data('VertexCoordinates')
        edgeconn = graph.get_data('EdgeConnectivity')
        edgepoints = graph.get_data('EdgePointCoordinates')
        nedgepoints = graph.get_data('NumEdgePoints')
        paramVals = graph.get_data(parameter)
        scalars = graph.get_scalars()
        nscalars = graph.get_node_scalars()
        
        nedges = edgeconn.shape[0]
        nnode = nodecoords.shape[0]

        # List all edge indices
        inds = np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int')
        # Define which edge each edgepoint belongs to
        edge_inds = np.repeat(inds,nedgepoints)
        if clip_value is not None:
            clip_edge_inds = np.where((paramVals>=clip_value) & (paramVals<=min_value))
            paramVals[clip_edge_inds] = min_value
        else:
            clip_edge_inds = [[]]
        
        if min_value is not None and max_value is not None:      
            del_edge_inds = np.where((paramVals<min_value) | (paramVals>max_value))        
        elif min_value is None and max_value is not None:
            del_edge_inds = np.where((paramVals>max_value))        
        elif min_value is not None and max_value is None:
            del_edge_inds = np.where((paramVals<min_value))   
        elif min_value is None and max_value is None:
            return graph

        #print(f'{len(clip_edge_inds[0])} edges with {parameter}>{clip_value} and {parameter}<{min_filter_radius} clipped.')
        #print(f'{len(del_edge_inds[0])} edges with {parameter}<{min_filter_radius} clipped.')

        # Find unique edge index references
        keep_edge = np.ones(edgeconn.shape[0],dtype='bool')
        # Convert to segments
        del_inds = np.unique(edge_inds[del_edge_inds])
        keep_edge[del_inds] = False
        
        if ignore is not None:
            keep_edge[ignore] = True
        
        keep_inds = np.where(keep_edge)[0]
        # Define nodes to keep positively (i.e. using the keep_inds rather than del_inds) so that nodes are retained that appear in edges that aren't flagged for deletion
        node_keep_inds = np.unique(edgeconn[keep_inds].flatten())
        keep_node = np.zeros(nodecoords.shape[0],dtype='bool')
        keep_node[node_keep_inds] = True
        
        graph.set_data(nodecoords,name='VertexCoordinates')
        graph.set_data(edgeconn,name='EdgeConnectivity')
        graph.set_data(edgepoints,name='EdgePointCoordinates')
        graph.set_data(nedgepoints,name='NumEdgePoints')
        graph.set_data(paramVals,name=parameter)
        graph.set_definition_size('VERTEX',nodecoords.shape[0])
        graph.set_definition_size('EDGE',edgeconn.shape[0])
        graph.set_definition_size('POINT',edgepoints.shape[0])            
        graph.set_graph_sizes()
        
        graph,keep_edge = delete_vertices(graph,keep_node,return_keep_edge=True)
        
        if write:
            graph.write(ofile)  
            
        if return_edge_inds:
            return graph,keep_edge
        else:
            return graph  
            
    def filter_graph_by_radius(self,graph,min_filter_radius=5.,filter_clip_radius=None,write=False,ofile='',keep_stubs=False,stub_len=100.,ignore=None,return_edge_inds=False):

        """
        min_filter_radius: All edges with radii < this value are deleted
        filter_clip_radius: All edges with radii < this value and > min_filter_radius are set to filter_clip_radius    
        """

        #nodecoords, edgeconn, edgepoints, nedgepoints, radius, category, mlp = get_graph_fields(graph)
        
        nodecoords = graph.get_data('VertexCoordinates')
        edgeconn = graph.get_data('EdgeConnectivity')
        edgepoints = graph.get_data('EdgePointCoordinates')
        nedgepoints = graph.get_data('NumEdgePoints')
        radius = graph.get_data(graph.get_radius_field_name())
        scalars = graph.get_scalars()
        nscalars = graph.get_node_scalars()
        
        nedges = edgeconn.shape[0]
        nnode = nodecoords.shape[0]

        # List all edge indices
        inds = np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int')
        # Define which edge each edgepoint belongs to
        edge_inds = np.repeat(inds,nedgepoints)
        if filter_clip_radius is not None:
            clip_edge_inds = np.where((radius>=filter_clip_radius) & (radius<=min_filter_radius))
            radius[clip_edge_inds] = min_filter_radius
        else:
            clip_edge_inds = [[]]
        del_edge_inds = np.where(radius<min_filter_radius)
        
        edge_stubs = arr([])
        if keep_stubs:
            node_radius = np.zeros(nodecoords.shape[0]) - 1
            rname = graph.get_radius_field_name()
            rind = [i for i,s in enumerate(graph.get_scalars()) if s['name']==rname][0]
            for i in range(graph.nedge):
                edge = graph.get_edge(i)
                rads = np.min(edge.scalars[rind])
                node_radius[edge.start_node_index] = np.max([rads,node_radius[edge.start_node_index]])
                node_radius[edge.end_node_index] = np.max([rads,node_radius[edge.end_node_index]])
            
            is_stub = np.zeros(radius.shape[0],dtype='bool')
            stub_loc = np.zeros(graph.nedge,dtype='int') - 1
            for i in range(graph.nedge):
                edge = graph.get_edge(i)   
                rads = edge.scalars[rind]
                #epointinds = np.linspace(edge.i0,edge.i1,edge.npoints,dtype='int')
                if np.any(rads<min_filter_radius):
                    if node_radius[edge.start_node_index]>min_filter_radius:
                        is_stub[edge.i0] = True
                        stub_loc[i] = 0
                    elif node_radius[edge.end_node_index]>min_filter_radius:
                        is_stub[edge.i0] = True   
                        stub_loc[i] = 1

            del_edge_inds = np.where((radius<min_filter_radius) & (is_stub==False))
            edgepoint_edges = np.repeat(np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int'),nedgepoints)
            edge_stubs = np.unique(edgepoint_edges[is_stub])
            #breakpoint()
            
            edges = edgeconn[edge_stubs]
            stub_loc = stub_loc[edge_stubs]
            # Shorten stubs
            edgepoints_valid = np.ones(edgepoints.shape[0],dtype='bool') 
            for i,edge in enumerate(edges):
                 nodes = nodecoords[edge]
                 edgeObj = graph.get_edge(edge_stubs[i])
                 points = edgeObj.coordinates
                 if stub_loc[i]==0:
                     lengths = np.concatenate([[0.],np.cumsum(np.linalg.norm(points[1:]-points[:-1]))])
                     if np.max(lengths)>=stub_len:
                         x0 = points[0]
                         x1 = points[lengths>=stub_len]
                         clen = stub_len
                     else:
                         x0 = points[0]
                         x1 = points[-1]
                         clen = np.linalg.norm(x1-x0)
                     vn = x1 - x0
                     vn = vn / np.linalg.norm(x1-x0)
                     nodecoords[edge[1]] = x0 + vn*stub_len
                     new_edgepoints = [x0,x1]
                 elif stub_loc[i]==1:
                     points = np.flip(points,axis=0)
                     lengths = np.concatenate([[0.],np.cumsum(np.linalg.norm(points[1:]-points[:-1]))])
                     if np.max(lengths)>=stub_len:
                         x0 = points[0]
                         x1 = points[lengths>=stub_len]
                         clen = stub_len
                     else:
                         x0 = points[0]
                         x1 = points[-1]
                         clen = np.linalg.norm(x1-x0)
                     vn = x1 - x0
                     vn = vn / np.linalg.norm(x1-x0)
                     nodecoords[edge[0]] = x0 + vn*clen
                     new_edgepoints = [x1,x0]
                 else:
                     breakpoint()
                 edgepoints[edgeObj.i0] = new_edgepoints[0]
                 edgepoints[edgeObj.i0+1] = new_edgepoints[1]
                 if edgeObj.npoints>2:
                     edgepoints_valid[edgeObj.i0+1:edgeObj.i0] = False
                 nedgepoints[i] = 2
            #breakpoint()
            edgepoints = edgepoints[edgepoints_valid]

        print(f'{len(clip_edge_inds[0])} edges with radii>{filter_clip_radius} and radii<{min_filter_radius} clipped.')
        print(f'{len(del_edge_inds[0])} edges with radii<{min_filter_radius} clipped.')

        # Find unique edge index references
        keep_edge = np.ones(edgeconn.shape[0],dtype='bool')
        # Convert to segments
        del_inds = np.unique(edge_inds[del_edge_inds])
        keep_edge[del_inds] = False
        
        if ignore is not None:
            keep_edge[ignore] = True
        
        keep_inds = np.where(keep_edge)[0]
        # Define nodes to keep positively (i.e. using the keep_inds rather than del_inds) so that nodes are retained that appear in edges that aren't flagged for deletion
        node_keep_inds = np.unique(edgeconn[keep_inds].flatten())
        keep_node = np.zeros(nodecoords.shape[0],dtype='bool')
        keep_node[node_keep_inds] = True
        
        graph.set_data(nodecoords,name='VertexCoordinates')
        graph.set_data(edgeconn,name='EdgeConnectivity')
        graph.set_data(edgepoints,name='EdgePointCoordinates')
        graph.set_data(nedgepoints,name='NumEdgePoints')
        graph.set_data(radius,name=graph.get_radius_field_name())
        graph.set_definition_size('VERTEX',nodecoords.shape[0])
        graph.set_definition_size('EDGE',edgeconn.shape[0])
        graph.set_definition_size('POINT',edgepoints.shape[0])            
        graph.set_graph_sizes()
        
        graph,keep_edge = delete_vertices(graph,keep_node,return_keep_edge=True)
        
        if write:
            graph.write(ofile)  
            
        if return_edge_inds:
            return graph,keep_edge
        else:
            return graph  

    def interpolate_edges(self,graph,interp_resolution=None,interp_radius_factor=None,ninterp=2,filter=None,noise_sd=0.):
        
        """
        Linear interpolation of edge points, to a fixed minimum resolution
        Filter: bool(m) where m=number of edges in graph. Only edges with filter[i]=True will be interpolated
        """
        
        coords = graph.get_data('VertexCoordinates')
        points = graph.get_data('EdgePointCoordinates')
        npoints = graph.get_data('NumEdgePoints')
        conns = graph.get_data('EdgeConnectivity')
        radii = graph.get_radius_data()
        
        scalars = graph.get_scalars()
        scalar_data = [x['data'] for x in scalars]
        scalar_type = [str(x.dtype) for x in scalar_data]
        scalar_data_interp = [[] for x in scalars]
        scalar_names = [x['name'] for x in scalars]
        
        if filter is None:
            filter = np.ones(conns.shape[0],dtype='bool')
        
        pts_interp,npoints_interp = [],np.zeros(conns.shape[0],dtype='int')-1
        for i,conn in enumerate(conns):
            i0 = np.sum(npoints[:i])
            i1 = i0 + npoints[i]
            pts = points[i0:i1]
            
            if filter[i]==False: # Ignore if filter is False
                pts_interp.extend(points[i0:i1])
                npoints_interp[i] = npoints[i] #.append(2)
                for j,sd in enumerate(scalar_data):
                    scalar_data_interp[j].extend(sd[i0:i1])
            else:
            
                # If the current edge has only 2 points
                if npoints[i]==2:  
                    ninterp = 2 # default   
                    # Find how many additional points to interpolate in (if length>interpolation resolution)
                    if interp_radius_factor is not None and radii is not None:
                        length = np.linalg.norm(pts[1]-pts[0])
                        meanRadius = np.mean(radii[i0:i1])
                        cur_interp_res = interp_radius_factor*meanRadius
                        if length>cur_interp_res:
                            ninterp = np.clip(int(np.ceil(length / cur_interp_res)+1),2,None)
                        #print(f'Ninterp: {ninterp}, npoints: {npoints[i]}, cur_interp_res:{cur_interp_res}')
                    elif interp_resolution is not None:
                        length = np.linalg.norm(pts[1]-pts[0])
                        if length>interp_resolution:
                            ninterp = np.clip(int(np.ceil(length / interp_resolution)+1),2,None)
                        
                    pcur = np.linspace(pts[0],pts[-1],ninterp)
                    if noise_sd>0.:
                        pcur += np.random.normal(0.,noise_sd)
                        pcur[0],pcur[-1] = pts[0],pts[-1]
                    pts_interp.extend(pcur)
                    
                    for j,sd in enumerate(scalar_data):
                        sdc = sd[i0:i1]
                        if 'float' in scalar_type[j]:
                            scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[1],ninterp))
                        elif 'int' in scalar_type[j]:
                            if sdc[0]==sdc[-1]:
                                scalar_data_interp[j].extend(np.zeros(ninterp)+sdc[0])
                            else:
                                #breakpoint()
                                scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[1],ninterp,dtype='int'))
                        elif 'bool' in scalar_type[j]:
                            scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[-1],ninterp,dtype='bool'))                                
                        else:
                            breakpoint()
                    
                    npoints_interp[i] = ninterp
                    
                    if ninterp!=pcur.shape[0]:
                        breakpoint()
                        
                # If the existing edge has more than 2 points
                elif npoints[i]>2:
                    # Spline interpolate curve at required interval
                    i0 = np.sum(npoints[:i])
                    i1 = i0 + npoints[i]
                    pts = points[i0:i1]

                    dists = arr([np.linalg.norm(pts[i]-pts[i-1]) for i,p in enumerate(pts[1:])])
                    length = np.sum(dists)
                    
                    if interp_radius_factor is not None and radii is not None:
                        meanRadius = np.mean(radii[i0:i1])
                        cur_interp_res = interp_radius_factor*meanRadius
                        if length>cur_interp_res:
                            ninterp = np.clip(int(np.ceil(length / cur_interp_res)+1),2,None)
                        else:
                            ninterp = 2
                        #print(f'Ninterp: {ninterp}, npoints: {npoints[i]}, cur_interp_res:{cur_interp_res}')
                    elif length>interp_resolution:
                        ninterp = np.clip(int(np.ceil(length / interp_resolution)+1),2,None)
                    else:
                        ninterp = 2

                    from scipy import interpolate
                    try:
                        if npoints[i]<=4:
                            pcur = np.linspace(pts[0],pts[-1],ninterp)
                            if noise_sd>0.:
                                pcur += np.random.normal(0.,noise_sd)
                                pcur[0],pcur[-1] = pts[0],pts[-1]
                        else:
                            k = 1
                            # Interpolate fails if all values are equal (to zero?)
                            # This most commonly happens in z-direction, for retinas at least, so add noise and remove later
                            if np.all(pts[:,2]==pts[0,2]):
                                z = pts[:,2] + np.random.normal(0.,0.1,pts.shape[0])
                            else:
                                z = pts[:,2]
                            tck, u = interpolate.splprep([pts[:,0], pts[:,1], z],k=k,s=0) #, s=2)
                            u_fine = np.linspace(0,1,ninterp)
                            pcur = np.zeros([ninterp,3])
                            pcur[:,0], pcur[:,1], pcur[:,2] = interpolate.splev(u_fine, tck)
                            if np.all(pts[:,2]==pts[0,2]):
                                pcur[:,2] = pts[0,2]
                    except Exception as e:
                        breakpoint()
                    
                    pcur[0] = pts[0]
                    pcur[-1] = pts[-1]
                    
                    pts_interp.extend(pcur)
                        
                    #for j,sd in enumerate(scalar_data):
                    #    sdc = sd[i0:i1]
                    #    scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[-1],pcur.shape[0]))
                        
                    for j,sd in enumerate(scalar_data):
                        sdc = sd[i0:i1]
                        if scalar_names[j]=='EdgeLabel':
                            scalar_data_interp[j].extend(np.repeat(sdc[0],pcur.shape[0]))
                        else:
                            if 'float' in scalar_type[j]:
                                scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[1],pcur.shape[0]))
                            elif 'int' in scalar_type[j]:
                                #breakpoint()
                                if sdc[0]==sdc[-1]:
                                    scalar_data_interp[j].extend(np.zeros(ninterp)+sdc[0])
                                else:
                                    scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[1],pcur.shape[0],dtype='int'))
                            elif 'bool' in scalar_type[j]:
                                scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[-1],ninterp,dtype='bool')) 
                            else:
                                breakpoint()
                    
                    npoints_interp[i] = ninterp
                    
                    if ninterp!=pcur.shape[0]:
                        breakpoint()

                    if False:
                        import matplotlib.pyplot as plt
                        from mpl_toolkits.mplot3d import Axes3D
                        fig2 = plt.figure(2)
                        ax3d = fig2.add_subplot(111, projection='3d')
                        ax3d.plot(pcur[:,0], pcur[:,1], pcur[:,2], 'b')
                        ax3d.plot(pcur[:,0], pcur[:,1], pcur[:,2], 'b*')
                        ax3d.plot(pts[:,0], pts[:,1], pts[:,2], 'r*')
                        plt.show()
                        breakpoint()
                else:
                    breakpoint()

                # Check nodes match!
                if not np.all(pts_interp[-ninterp]==coords[conn[0]]) or not np.all(pts_interp[-1]==coords[conn[1]]) or \
                   not np.all(pts_interp[-ninterp]==pts[0]) or not np.all(pts_interp[-1]==pts[-1]):
                    #breakpoint()
                    pass

        pts_interp = arr(pts_interp)
        graph.set_data(pts_interp,name='EdgePointCoordinates')
        graph.set_data(npoints_interp,name='NumEdgePoints')
       
        for j,sd in enumerate(scalar_data_interp):
            graph.set_data(arr(sd),name=scalars[j]['name'])
        
        graph.set_definition_size('POINT',pts_interp.shape[0])   
        graph.set_graph_sizes()  
        
        return graph
        
    def insert_nodes_in_edges(self,graph,interp_resolution=None,interp_radius_factor=None,filter=None):

        """
        Given an interpolation resolution, insert nodes into edges with even spacing
        """

        coords = graph.get_data('VertexCoordinates')
        points = graph.get_data('EdgePointCoordinates')
        npoints = graph.get_data('NumEdgePoints')
        conns = graph.get_data('EdgeConnectivity')
        radii = graph.get_radius_data()

        if filter is None:
            filter = np.ones(conns.shape[0],dtype='bool')
                                            
        gvars = GVars(graph)
        g_nedgepoints = gvars.nedgepoints[gvars.edgeconn_allocated]
        g_edgeCoords = gvars.edgepoints[gvars.edgepoints_allocated]
        g_nodeCoords = gvars.nodecoords[gvars.nodecoords_allocated]
        g_edgeConn = gvars.edgeconn[gvars.edgeconn_allocated]
        g_scalars,g_scalar_names = [],[]
        for j,sc in enumerate(gvars.scalar_values):
            g_scalars.append(gvars.scalar_values[j][gvars.edgepoints_allocated]) 
            g_scalar_names.append(gvars.scalars[j]['name']) 
        g_node_scalars,g_node_scalar_names = [],[]
        for j,sc in enumerate(gvars.node_scalar_values):
            g_node_scalars.append(gvars.node_scalar_values[j][gvars.nodecoords_allocated])  
            g_node_scalar_names.append(gvars.node_scalars[j]['name'])

        for i,conn in enumerate(conns):

            if filter[i]==True: # Ignore if filter is False           
            
                i0 = np.sum(npoints[:i])
                i1 = i0 + npoints[i]
                pts = points[i0:i1]
                
                i0p = np.sum(g_nedgepoints[:i])
                i1p = i0p + g_nedgepoints[i]
            
                lengths = np.linalg.norm(pts[1:]-pts[:-1],axis=1) 
                cumulative_lengths = np.insert(np.cumsum(lengths), 0, 0)
                total_length = np.sum(lengths)
                meanRadius = np.mean(radii[i0:i1])
                
                if interp_resolution is None:
                    cur_interp_res = interp_radius_factor*meanRadius
                else:
                    cur_interp_res = interp_resolution

                if total_length>cur_interp_res:
                    ninterp = np.clip(total_length/cur_interp_res,1,None).astype('int')
                    distances = np.linspace(0,total_length,ninterp+2)[1:-1] 

                    segment_index = np.searchsorted(cumulative_lengths, distances, side='right') - 1
                                                 
                    # Compute the local fraction along the current segment
                    segment_start_length = cumulative_lengths[segment_index]
                    segment_end_length = cumulative_lengths[segment_index + 1]
                    local_fractions = (distances - segment_start_length) / (segment_end_length - segment_start_length)
                    
                    # Interpolate between the two points of the segment
                    start_point = pts[segment_index]
                    end_point = pts[segment_index + 1]
                    interpolated_points = ((1-local_fractions)*start_point.transpose()).transpose() + (local_fractions*end_point.transpose()).transpose() 
                         
                    segments = np.linspace(0,pts.shape[0]-1,pts.shape[0],dtype='int')
                    order = np.concatenate([segments,segment_index+local_fractions])
                    types = np.zeros(order.shape[0],dtype='int')
                    types[segments.shape[0]:] = 1
                    pts_interp = np.vstack([pts,interpolated_points])
                    srt = np.argsort(order)
                    order = order[srt]
                    types = types[srt]
                    pts_interp = pts_interp[srt]
                    
                    int_pts = np.where(types==1)
                    int_pts = np.concatenate([[0],int_pts[0],[pts_interp.shape[0]]])
                    new_conns = np.concatenate([[0],np.repeat(int_pts[1:-1],2),[int_pts[-1]]])
                    new_conns = new_conns.reshape([int(new_conns.shape[0]/2),2])
                    new_edges = [[] for k in range(new_conns.shape[0])]
                    for j,cn in enumerate(new_conns): 
                        new_edges[j] = pts_interp[cn[0]:cn[1]+1]
                    # Edge node indices
                    nnodes = g_nodeCoords.shape[0]
                    new_edgeconns = np.concatenate([[conn[0]],np.repeat(np.linspace(nnodes,nnodes+ninterp-1,ninterp,dtype='int'),2),[conn[1]]])
                    new_edgeconns = new_edgeconns.reshape([int(new_edgeconns.shape[0]/2),2])

                    # Grab all edge coordinates prior to edge to be split
                    if i0p>0:
                        edgeCoords_0 = g_edgeCoords[:i0p]
                    else:
                        edgeCoords_0 = []
                    # Edge coordinates listed after the current edge
                    if g_edgeCoords.shape[0]>i1p:
                        edgeCoords_1 = g_edgeCoords[i1p:]
                    else:
                        edgeCoords_1 = []

                    # Combine edges together
                    g_edgeCoords = np.concatenate([x for x in [edgeCoords_0,new_edges[0],edgeCoords_1,np.concatenate(new_edges[1:])] if len(x)>0])
                    g_nedgepoints = np.concatenate([g_nedgepoints,[ned.shape[0] for ned in new_edges[1:]]])
                    g_nedgepoints[i] = new_edges[0].shape[0]
                    g_edgeConn = np.concatenate([g_edgeConn,new_edgeconns[1:]])
                    g_edgeConn[i] = new_edgeconns[0]
                    g_nodeCoords = np.concatenate([g_nodeCoords,pts_interp[types==1]])

                    # Sort out scalars
                    # Make all node scalars equal to the value for the start node
                    try:
                        new_node_scalars = [np.repeat(x[conn[0]],ninterp) for x in g_node_scalars]   
                        if 'NodeLabel' in g_node_scalar_names:
                            new_node_scalars[g_node_scalar_names.index('NodeLabel')] = [graph.unique_node_label() for _ in range(ninterp)]                        
                        for j,data in enumerate(g_node_scalars):
                            g_node_scalars[j] = np.concatenate([data,np.concatenate([new_node_scalars[j]])])
                    except Exception as e:
                        print(e)
                        breakpoint()

                    for j,data in enumerate(g_scalars):
                        if i0p>0:
                            sc_0 = data[:i0p]
                        else:
                            sc_0 = []
                        if data.shape[0]>i1p:
                            sc_1 = data[i1p:]
                        else:
                            sc_1 = []
                        edata = data[i0p:i1p]
                        interp_data = np.zeros(pts_interp.shape[0],dtype=data.dtype)
                        interp_data[types==0] = edata
                        # Assign interpolated points the value at the start of the segment (maybe interpolate in future?)
                        interp_data[types==1] = edata[segment_index]
                        new_sc = [[] for k in range(new_conns.shape[0])]
                        for k,cn in enumerate(new_conns): 
                            if g_scalar_names[j]=='EdgeLabel':
                                new_sc[k] = np.repeat(graph.unique_edge_label(),interp_data[cn[0]:cn[1]+1].shape[0])
                            else:
                                new_sc[k] = interp_data[cn[0]:cn[1]+1]

                        g_scalars[j] = np.concatenate([x for x in [sc_0,new_sc[0],sc_1,np.concatenate(new_sc[1:])] if len(x)>0])

        if len(g_node_scalars)>0 and g_node_scalars[0].shape[0]!=g_nodeCoords.shape[0]:
            breakpoint()
        gvars.set_nodecoords(g_nodeCoords,scalars=g_node_scalars)  
        gvars.set_edgeconn(g_edgeConn,g_nedgepoints)  
        gvars.set_edgepoints(g_edgeCoords,scalars=g_scalars)
        graph = gvars.set_in_graph()

        return graph
        
    def add_noise(self,graph,filter=None,radius_factor=2.):
    
        edges = graph.get_data('EdgeConnectivity')
        edgepoints = graph.get_data('EdgePointCoordinates')
        radius = graph.get_data(graph.get_radius_field_name())
        
        if filter is None:
            filter = np.ones(graph.nedge,dtype='bool')
        for i,e in enumerate(tqdm(edges)):
            if filter[i]:
                edge = graph.get_edge(i)
                if edge.npoints>2:
                    dirs = edge.coordinates[1:]-edge.coordinates[:-1]/(np.linalg.norm(edge.coordinates[1:]-edge.coordinates[:-1]))
                    orth = np.cross(dirs,arr([0.,0.,1]))
                    orth = arr([x / np.linalg.norm(x) for x in orth])
                    orth = np.vstack([arr([0.,0.,0.]),orth])
                    edgepoints[edge.i0+1:edge.i1-1] += orth[1:-1] + np.random.normal(0.,radius[edge.i0+1:edge.i1-1]*radius_factor)
                        
        graph.set_data(edgepoints,name='EdgePointCoordinates') 
        return graph 
        
    def displace_degenerate_nodes(self,graph,displacement=1.):
    
        nodes = graph.get_data('VertexCoordinates')
        edgepoints = graph.get_data('EdgePointCoordinates')
        
        for i,c1 in enumerate(nodes):
            sind = np.where((nodes[:,0]==c1[0]) & (nodes[:,1]==c1[1]) & (nodes[:,2]==c1[2]))
            if len(sind[0])>1:
                #print(f'Degenerate nodes: {sind[0]}')
                edges = graph.get_edges_containing_node(sind[0])
                for s in sind[0]:
                    nodes[s] += np.random.uniform(-displacement/2.,displacement/2.,3)
                for e in edges:
                    edge = graph.get_edge(e)
                    #print(f'Fixing edge {e} (nodes: {edge.start_node_index}, {edge.end_node_index})')
                    if edge.start_node_index in sind[0]:
                        edgepoints[edge.i0] = nodes[edge.start_node_index]
                    if edge.end_node_index in sind[0]:
                        edgepoints[edge.i1-1] = nodes[edge.end_node_index] 
                        
        graph.set_data(nodes,name='VertexCoordinates')
        graph.set_data(edgepoints,name='EdgePointCoordinates') 
        return graph 

class Node(object):
    
    def __init__(self, graph=None, index=0, edge_indices=None, edge_indices_rev=None,
                 connecting_node=None, edges=None, coordinates=None, old_index=None,
                 scalars=None, scalarNames=None ):
                     
        self.index = index
        self.nconn = 0
        self.edge_indices = edge_indices
        self.edge_indices_rev = edge_indices_rev
        self.connecting_node = connecting_node
        self.edges = edges
        self.coords = coordinates
        self.old_index = old_index
        self.scalars = []
        self.scalarNames = []
        
        if graph is not None:
            # Initialise edge list in graph object
            
            if graph.edgeList is None:
                graph.edgeList = arr([None]*graph.nedge)
            edgeInds = np.where(graph.edgeList!=None)[0] # [e for e in graph.edgeList if e is not None]

            vertCoords = graph.get_field('VertexCoordinates')['data']
            if vertCoords is None:
                return
            edgeConn = graph.get_field('EdgeConnectivity')['data']
            
            #s0 = [j for j,x in enumerate(edgeConn) if index in x]
            #s0 = np.where(edgeConn==index)
            #ns0 = len(s0)
            if edgeConn is not None:
                s0 = np.where(edgeConn[:,0]==index)
                ns0 = len(s0[0])
                s1 = np.where(edgeConn[:,1]==index)
                ns1 = len(s1[0])
            else:
                ns0,ns1,s0,s1 = 0,0,[],[]
            
            self.coords = vertCoords[index,:]
            self.nconn = ns0 + ns1
            
            self.edge_indices = edge_indices
            if self.edge_indices is None:
                self.edge_indices = []
            self.edge_indices_rev = []
            self.connecting_node = []
            self.edges = []
    
            if len(s0)>0:
                for e in s0[0]:
                    self.edge_indices.append(e)
                    if e not in edgeInds:  
                        newEdge = Edge(graph=graph,index=e)
                        edgeInds = np.append(edgeInds,e)
                        if graph.edgeList[e] is None:
                            graph.edgeList[e] = newEdge
                    else:
                        newEdge = graph.edgeList[e] #[edge for edge in graph.edgeList if edge is not None and edge.index==e]
                    self.edges.append(newEdge)
                    self.edge_indices_rev.append(False)
                    self.connecting_node.append(edgeConn[e,1])
            if len(s1)>0:
                for e in s1[0]:
                    self.edge_indices.append(e)
                    self.edge_indices_rev.append(True)
                    if e not in edgeInds:                  
                        newEdge = Edge(graph=graph,index=e)
                        edgeInds = np.append(edgeInds,e)
                        if graph.edgeList[e] is None:
                            graph.edgeList[e] = newEdge
                    else:
                        newEdge = graph.edgeList[e] # [edge for edge in graph.edgeList if edge is not None and edge.index==e]
                    self.edges.append(newEdge)
                    self.connecting_node.append(edgeConn[e,0])
                
    def add_edge(self,edge,reverse=False):
        if self.edges is None:
            self.edges = []
            
        current_edge_indices = [e.index for e in self.edges]  
        if edge.index in current_edge_indices:
            return False
            
        self.edges.append(edge)
        if self.edge_indices_rev is None:
            self.edge_indices_rev = []
        self.edge_indices_rev.append(reverse)
        if self.connecting_node is None:
            self.connecting_node = []
        if not reverse:
            self.connecting_node.append(edge.end_node_index)
        else:
            self.connecting_node.append(edge.start_node_index)
        self.nconn += 1
        return True
        
#    def remove_edge(self,edgeIndex):
#        keep_edge_ind = [i for i,e in enumerate(self.edges) if e.index not in edgeIndex]
#        self.edges = [self.edges[i] for i in keep_edge_ind]
#        self.edge_indices = [self.edge_indices[i] for i in keep_edge_ind]
#        self.edge_indices_rev = [self.edge_indices_rev[i] for i in keep_edge_ind]
#        self.connecting_node = [self.connecting_node[i] for i in keep_edge_ind]
#        self.nconn = len(self.edges)
        
    def add_scalar(self,name,values):
        
        if name in self.scalarNames:
            print(('Error: Node scalar field {} already exists!'.format(name)))
            return
            
        if len(self.scalars)==0:
            self.scalars = [values]
            self.scalarNames = [name]
        else:
            self.scalars.append([values])
            self.scalarNames.append(name)
            
    def get_scalar(self,name):
        scalar = [x for i,x in enumerate(self.scalars) if self.scalarNames[i]==name]
        if len(scalar)==0:
            return None
        scalar[0]
            
    def _print(self):
        print(('NODE ({}):'.format(self.index)))
        print(('Coordinate: {}'.format(self.coords)))
        print(('Connected to: {}'.format(self.connecting_node)))
        if len(self.connecting_node)>0:
            edgeInd = [e.index for e in self.edges]
            print(('Connected via edges: {}'.format(edgeInd)))
            
class Edge(object):
    
    def __init__(self, index=0, graph=None, 
                 start_node_index=None, start_node_coords=None,
                 end_node_index=None, end_node_coords=None,
                 npoints=0, coordinates=None, scalars=None,
                 scalarNames=None):
        self.index = index
        self.start_node_index = start_node_index
        self.start_node_coords = start_node_coords # numpy array
        self.end_node_index = end_node_index
        self.end_node_coords = end_node_coords # numpy array
        self.npoints = npoints
        self.coordinates = coordinates # numpy array
        self.complete = False
        self.scalars = scalars
        self.scalarNames = scalarNames
        self.i0,self.i1 = -1,-1
        
        if graph is not None:
            nodeCoords = graph.get_field('VertexCoordinates')['data']
            edgeConn = graph.get_field('EdgeConnectivity')['data']
            nedgepoints = graph.get_field('NumEdgePoints')['data']
            self.coordinates = np.squeeze(self.get_coordinates_from_graph(graph,index))
            self.start_node_index = edgeConn[index,0]
            self.start_node_coords = nodeCoords[self.start_node_index,:]
            self.npoints = nedgepoints[index]
            self.scalars,self.scalarNames = self.get_scalars_from_graph(graph,index)
            stat = self.complete_edge(nodeCoords[edgeConn[index,1],:],edgeConn[index,1])
            
            self.i0 = np.sum(nedgepoints[:index])
            self.i1 = self.i0 + nedgepoints[index]
        
    def get_coordinates_from_graph(self,graph,index):
        nedgepoints = graph.get_field('NumEdgePoints')['data']
        coords = graph.get_field('EdgePointCoordinates')['data']
        if index>0:
            nprev = np.sum(nedgepoints[0:index])
        else:
            nprev = 0
        ncur = nedgepoints[index]
        e_coords = coords[nprev:nprev+ncur,:]
        return e_coords
        
    def get_scalars_from_graph(self,graph,index):
        scalars = graph.get_scalars()
        if len(scalars)==0:
            return None,None
        nedgepoints = graph.get_field('NumEdgePoints')['data']
        nprev = np.sum(nedgepoints[0:index])
        ncur = nedgepoints[index]
        scalarData = []
        scalarNames = []
        for s in scalars:
            scalarData.append(s['data'][nprev:nprev+ncur])
            scalarNames.append(s['name'])
        return scalarData,scalarNames
        
    def add_point(self,coords,is_end=False,end_index=None,scalars=None,remove_last=False):
        coords = np.asarray(coords)
        #assert len(coords)==3
        if len(coords.shape)==2:
            npoints = coords.shape[0]
            p0 = coords[0]
        else:
            npoints = 1
            p0 = coords
        
        if self.coordinates is None:
            self.coordinates = []
            self.scalars = []
            self.scalarNames = []
            if self.start_node_coords is None:
                self.start_node_coords = np.asarray(p0)
            self.coordinates = np.asarray(coords)
            self.npoints = npoints
        else:
            if remove_last:
                if self.npoints>1:
                    self.coordinates = self.coordinates[0:-1,:]
                else:
                    self.coordinates = []
                self.npoints -= 1
            self.coordinates = np.vstack([self.coordinates, np.asarray(coords)])
            self.npoints += npoints
            if scalars is not None:
                if remove_last:
                    if self.npoints==0:
                        self.scalars = []
                    else:
                        self.scalars = [s[0:-1] for s in self.scalars]
                for i,sc in enumerate(scalars):
                    self.scalars[i] = np.append(self.scalars[i],scalars[i])
        if is_end:
            self.complete_edge(np.asarray(coords),end_index)
            
    def complete_edge(self,end_node_coords,end_node_index,quiet=True):
        stat = 0
        self.end_node_coords = np.asarray(end_node_coords)
        self.end_node_index = end_node_index
        self.complete = True
        
        if self.coordinates.ndim<2 or self.coordinates.shape[0]<2:
            if not quiet:
                print(f'Error, too few points in edge {self.index}')
            stat = -3
            return stat
        
        if not all([x.astype('float32')==y.astype('float32') for x,y in zip(self.end_node_coords,self.coordinates[-1,:])]):
            if not quiet:
                print('Warning: End node coordinates do not match last edge coordinate!')
            stat = -1
        if not all([x.astype('float32')==y.astype('float32') for x,y in zip(self.start_node_coords,self.coordinates[0,:])]):
            if not quiet:
                print('Warning: Start node coordinates do not match first edge coordinate!')
            stat = -2
            
        return stat
            
    def at_start_node(self,index):
        if index==self.start_node_index:
            return True
        else:
            return False
            
    def add_scalar(self,name,values,set_if_exists=True):
        
        # TODO: add support for repeated scalars
        
        if len(values)!=self.npoints:
            print('Error: Scalar field has incorrect number of points')
            return
        if name in self.scalarNames:
            if set_if_exists:
                self.set_scalar(name,values)
            else:
                print(('Error: Scalar field {} already exists!'.format(name)))
            return
            
        if len(self.scalars)==0:
            self.scalars = values
            self.scalarNames = [name]
        else:
            self.scalars.append(values)
            self.scalarNames.append(name)
            
    def get_scalar(self,name,reverse=False):
        scalar = [x for i,x in enumerate(self.scalars) if self.scalarNames[i]==name]
        if len(scalar)==0:
            return None
        if reverse:
            return scalar[0][::-1]
        else:
            return scalar[0]
            
    def set_scalar(self,name,values):
        scalarInd = [i for i,x in enumerate(self.scalars) if self.scalarNames[i]==name]
        if len(scalarInd)==0:
            print('Scalar does not exist!')
            return
        oldVals = self.scalars[scalarInd[0]]
        if len(values)!=len(oldVals):
            print('Incorrect number of scalar values!')
            return
        self.scalars[scalarInd[0]] = values            
            
    def _print(self):
        print(('EDGE ({})'.format(self.index)))
        print(('Number of points: {}'.format(self.npoints)))
        print(('Start node (index,coords): {} {}'.format(self.start_node_index,self.start_node_coords)))
        print(('End node (index,coords): {} {}'.format(self.end_node_index,self.end_node_coords)))
        if self.scalarNames is not None:
            print(('Scalar fields: {}'.format(self.scalarNames)))
        if not self.complete:
            print('Incomplete...')
            
# Create a leight-weight object to pass graph variables around with
# Useful for editing!
class GVars(object):
    def __init__(self,graph,n_all=500):
        self.node_ptr = 0
        self.edge_ptr = 0
        self.edgepnt_ptr = 0
        self.graph = graph
        
        # Set batche size to preallocate
        self.n_all = n_all 
        
        self.nodecoords = graph.get_data('VertexCoordinates').astype('float32')
        self.edgeconn = graph.get_data('EdgeConnectivity').astype('int')
        self.edgepoints = graph.get_data('EdgePointCoordinates').astype('float32')
        self.nedgepoints = graph.get_data('NumEdgePoints').astype('int')
        
        self.node_ptr = self.nodecoords.shape[0]
        self.edge_ptr = self.edgeconn.shape[0]
        self.edgepnt_ptr = self.edgepoints.shape[0]
        
        self.nodecoords_allocated = np.ones(self.nodecoords.shape[0],dtype='bool')
        self.edgeconn_allocated = np.ones(self.edgeconn.shape[0],dtype='bool')
        self.edgepoints_allocated = np.ones(self.edgepoints.shape[0],dtype='bool')
        
        self.set_scalars()
        
    def set_scalars(self):
        scalars = self.graph.get_scalars()
        scalar_values = [x['data'].copy() for x in scalars]
        self.scalar_values = scalar_values
        self.scalars = scalars
        radname = self.graph.get_radius_field()['name']
        self.radname = radname
        self.radind = [i for i,x in enumerate(self.scalars) if x['name']==radname][0]
        
        node_scalars = self.graph.get_node_scalars()
        node_scalar_values = [x['data'].copy() for x in node_scalars]
        self.node_scalar_values = node_scalar_values
        self.node_scalars = node_scalars
        
    def set_nodecoords(self,nodecoords,scalars=None,update_pointer=False):
        # Reset all nodes to array argument provided
        if self.nodecoords.shape[0]<nodecoords.shape[0]:
            self.preallocate_nodes(nodecoords.shape[0]-self.nodecoords.shape[0],set_pointer_to_start=False)
        self.nodecoords[:nodecoords.shape[0]] = nodecoords
        self.nodecoords_allocated[:] = False
        self.nodecoords_allocated[:nodecoords.shape[0]] = True        
        self.node_ptr = nodecoords.shape[0]
        for i,sc in enumerate(scalars):
            self.node_scalar_values[i][:nodecoords.shape[0]] = sc
        
    def set_edgeconn(self,edgeconn,nedgepoints,update_pointer=False):
        # Reset all edgeconn and nedgepoints to array argument provided
        if self.edgeconn.shape[0]<edgeconn.shape[0]:
            self.preallocate_edges(edgeconn.shape[0],set_pointer_to_start=False)
        self.edgeconn[:edgeconn.shape[0]] = edgeconn
        self.nedgepoints[:nedgepoints.shape[0]] = nedgepoints
        self.edgeconn_allocated[:] = False
        self.edgeconn_allocated[:edgeconn.shape[0]] = True        
        self.edge_ptr = edgeconn.shape[0]
        
    def set_edgepoints(self,edgepoints,scalars=None,update_pointer=False):
        # Reset all nodes to array argument provided
        if self.edgepoints.shape[0]<edgepoints.shape[0]:
            self.preallocate_edgepoints(edgepoints.shape[0]-self.edgepoints.shape[0],set_pointer_to_start=False)
        self.edgepoints[:edgepoints.shape[0]] = edgepoints
        self.edgepoints_allocated[:] = False
        self.edgepoints_allocated[:edgepoints.shape[0]] = True        
        self.edgepnt_ptr = edgepoints.shape[0]
        for i,sc in enumerate(scalars):
            self.scalar_values[i][:edgepoints.shape[0]] = sc
        
    def add_node(self,node,new_scalar_vals=[]):
        # Assign existing node slot to supplied node coordinate
        if self.node_ptr>=self.nodecoords.shape[0]:
            self.preallocate_nodes(self.n_all,set_pointer_to_start=False)
        self.nodecoords[self.node_ptr] = node
        self.nodecoords_allocated[self.node_ptr] = True
        
        if len(new_scalar_vals)==0:
            node_scalar_names = [x['name'] for x in  self.node_scalars]
            new_scalar_vals = [x['data'][0] for x in self.node_scalars]   
            if 'NodeLabel' in node_scalar_names:
                new_scalar_vals[node_scalar_names.index('NodeLabel')] = self.graph.unique_node_label()                      
        
        for i,sc in enumerate(self.node_scalar_values):
            self.node_scalar_values[i][self.node_ptr] = new_scalar_vals[i]
        self.node_ptr += 1
        if self.node_ptr>=self.nodecoords.shape[0]:
            self.preallocate_nodes(self.n_all,set_pointer_to_start=False)
            
    def append_nodes(self,nodes,update_pointer=False):
        # Create new slots for an array containing multiple node coordinates
        self.nodecoords = np.vstack([self.nodecoords,nodes])
        self.nodecoords_allocated = np.concatenate([self.nodecoords_allocated,np.ones(nodes.shape[0],dtype='bool')])
        if update_pointer:
            self.node_ptr = self.nodecoords.shape[0]
            
    def remove_nodes(self,node_inds_to_remove):
    
        nodecoords = self.nodecoords[self.nodecoords_allocated]
        edgeconn = self.edgeconn[self.edgeconn_allocated]
        nedgepoints = self.nedgepoints[self.edgeconn_allocated]
        edgepoints = self.edgepoints[self.edgepoints_allocated]
                
        nnode = nodecoords.shape[0]
        keep = np.ones(nnode,dtype='bool')
        keep[node_inds_to_remove] = False

        # Remove edges containing nodes
        # Find which edges must be deleted
        del_node_inds = np.where(keep==False)[0]
        del_edges = [np.where((edgeconn[:,0]==i) | (edgeconn[:,1]==i))[0] for i in del_node_inds]
        # Remove empties (i.e. where the node doesn't appear in any edges)
        del_edges = [x for x in del_edges if len(x)>0]
        # Flatten
        del_edges = [item for sublist in del_edges for item in sublist]
        # Convert to numpy
        del_edges = arr(del_edges)
        # List all edge indices
        inds = np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int')
        # Define which edge each edgepoint belongs to
        edge_inds = np.repeat(inds,nedgepoints)
        # Create a mask of points to keep for edgepoint variables
        keep_edgepoints = ~np.in1d(edge_inds,del_edges)
        # Apply mask to edgepoint array
        edgepoints = edgepoints[keep_edgepoints]
        # Apply mask to scalars
        scalars = []
        for i,scalar in enumerate(self.scalar_values):
            scalars.append(scalar[self.edgepoints_allocated][keep_edgepoints])
              
        # Create a mask for removing edge connections and apply to the nedgepoint array
        keep_edges = np.ones(edgeconn.shape[0],dtype='bool')
        if len(del_edges)>0:
            keep_edges[del_edges] = False
            nedgepoints = nedgepoints[keep_edges]
        
        # Remove nodes and update indices
        nodecoords, edgeconn, edge_lookup = update_array_index(nodecoords,edgeconn,keep)
        
        node_scalars = []
        for i,sc in enumerate(self.node_scalar_values):
            sc = sc[self.nodecoords_allocated][keep]
            node_scalars.append(sc)
            
        self.set_nodecoords(nodecoords,scalars=node_scalars)
        self.set_edgeconn(edgeconn,nedgepoints)
        self.set_edgepoints(edgepoints,scalars=scalars)
            
    def add_edgeconn(self,conn,npts=2):
        if self.edge_ptr>=self.edgeconn.shape[0]:
            self.preallocate_edges(self.n_all,set_pointer_to_start=False)
        self.edgeconn[self.edge_ptr] = conn
        self.edgeconn_allocated[self.edge_ptr] = True
        self.nedgepoints[self.edge_ptr] = npts
        self.edge_ptr += 1
        if self.edge_ptr>=self.edgeconn.shape[0]:
            self.preallocate_edges(self.n_all,set_pointer_to_start=False)

    def add_edge(self,start_node_index,end_node_index,new_scalar_vals,points=None):
        new_conn = [start_node_index,end_node_index]
        nodes = self.nodecoords[new_conn]
        if points is None or not np.all(points[0]-self.nodecoords[new_conn[0]]<1e-12) or not np.all(points[-1]-self.nodecoords[new_conn[1]]<1e-12):
            self.add_edgeconn(new_conn)
            self.add_edgepoints(self.nodecoords[new_conn],new_scalar_vals,edgeInd=self.edge_ptr-1)
        else:
            npts = points.shape[0]
            self.add_edgeconn(new_conn,npts=npts)
            self.add_edgepoints(points,new_scalar_vals,edgeInd=self.edge_ptr-1)
            
    def add_edgepoints(self,pnt,new_scalar_vals,edgeInd=-1):
        npts = pnt.shape[0]
        if self.edgepoints.shape[0]-self.edgepnt_ptr<=npts:
            self.preallocate_edgepoints(self.n_all,set_pointer_to_start=False)
        self.edgepoints[self.edgepnt_ptr:self.edgepnt_ptr+npts] = pnt
        self.edgepoints_allocated[self.edgepnt_ptr:self.edgepnt_ptr+npts] = True
        if edgeInd>=0:
            self.nedgepoints[edgeInd] = npts
            
        if len(new_scalar_vals)==0:
            scalar_names = [x['name'] for x in  self.scalars]
            new_scalar_vals = [x['data'][0] for x in self.scalars]   
            if 'EdgeLabel' in scalar_names:
                new_scalar_vals[scalar_names.index('EdgeLabel')] = self.graph.unique_edge_label()                      
            
        for i,sc in enumerate(self.scalar_values):
            dt = self.scalar_values[i].dtype
            self.scalar_values[i][self.edgepnt_ptr:self.edgepnt_ptr+npts] = np.zeros(npts,dtype=dt)+new_scalar_vals[i]
        self.edgepnt_ptr += npts     
        if self.edgepnt_ptr>=self.edgepoints.shape[0]:
            self.preallocate_edgepoints(self.n_all,set_pointer_to_start=False)
        
    def remove_edges(self,edge_inds_to_remove):

        if edge_inds_to_remove.shape[0]==0:
            return
    
        edgeconn = self.edgeconn[self.edgeconn_allocated]
        nedgepoints = self.nedgepoints[self.edgeconn_allocated]
        edgepoints = self.edgepoints[self.edgepoints_allocated]
                
        nedge = edgeconn.shape[0]
        keep = np.ones(edgeconn.shape[0],dtype='bool')
        keep[edge_inds_to_remove] = False
        edgeconn = edgeconn[keep]
                
        # Which edge is each edgepoint from
        edgeInds = np.repeat(np.linspace(0,nedge-1,nedge,dtype='int'),nedgepoints)
        
        # Flag edgepoints from removed edges
        flag = np.in1d(edgeInds,edge_inds_to_remove)
        # Filter edgepoints and scalars
        edgepoints = edgepoints[~flag]
        for i,sc in enumerate(self.scalar_values):
            self.scalar_values[i] = self.scalar_values[i][self.edgepoints_allocated][~flag]
            
        # Filter n edgepoints
        nedgepoints = nedgepoints[keep]
        
        # Set fields
        self.edgeconn = edgeconn
        self.edgepoints = edgepoints
        self.nedgepoints = nedgepoints

        # Set pointers
        self.edge_ptr = self.edgeconn.shape[0]
        self.edgepnt_ptr = self.edgepoints.shape[0]

        # Set pre-allocation
        self.edgeconn_allocated = self.edgeconn_allocated[self.edgeconn_allocated][keep]
        self.edgepoints_allocated = self.edgepoints_allocated[self.edgepoints_allocated][~flag]
        
        
    def plot(self,**kwargs):
        self.set_in_graph()
        self.graph.plot_graph(**kwargs)
    def preallocate_nodes(self,n,set_pointer_to_start=False):
        if set_pointer_to_start:
            self.node_ptr = self.nodecoords.shape[0]
        self.nodecoords = np.vstack([self.nodecoords,np.zeros([n,3],dtype=self.nodecoords.dtype)])
        self.nodecoords_allocated = np.concatenate([self.nodecoords_allocated,np.zeros(n,dtype='bool')])
        for i,sc in enumerate(self.node_scalar_values):
            if sc.dtype in ['bool']:
                self.node_scalar_values[i] = np.concatenate([self.node_scalar_values[i],np.zeros(n,dtype=sc.dtype)])
            else:
                self.node_scalar_values[i] = np.concatenate([self.node_scalar_values[i],np.zeros(n,dtype=sc.dtype)-1])
    def preallocate_edges(self,n,set_pointer_to_start=False):
        if set_pointer_to_start:
            self.edge_ptr = self.edgeconn.shape[0]
        #print(f'Edge preallocation: added {n}')
        self.edgeconn = np.vstack([self.edgeconn,np.zeros([n,2],dtype='int')-1])
        self.nedgepoints = np.concatenate([self.nedgepoints,np.zeros(n,dtype='int')])
        self.edgeconn_allocated = np.concatenate([self.edgeconn_allocated,np.zeros(n,dtype='bool')])
    def preallocate_edgepoints(self,n,set_pointer_to_start=False):
        if set_pointer_to_start:
            self.edgepnt_ptr = self.edgepoints.shape[0]
        self.edgepoints = np.vstack([self.edgepoints,np.zeros([n,3])])
        self.edgepoints_allocated = np.concatenate([self.edgepoints_allocated,np.zeros(n,dtype='bool')])
        for i,sc in enumerate(self.scalar_values):
            self.scalar_values[i] = np.concatenate([self.scalar_values[i],np.zeros(n,dtype=sc.dtype)-1])
    def remove_preallocation(self):
        # Remove all unoccupied slots from each data field
        self.nodecoords = self.nodecoords[self.nodecoords_allocated]
        self.edgeconn = self.edgeconn[self.edgeconn_allocated]
        self.nedgepoints = self.nedgepoints[self.edgeconn_allocated]
        self.edgepoints = self.edgepoints[self.edgepoints_allocated]
        for i,sc in enumerate(self.scalar_values):
            self.scalar_values[i] = self.scalar_values[i][self.edgepoints_allocated]
            
        self.nodecoords_allocated = self.nodecoords_allocated[self.nodecoords_allocated]
        for i,sc in enumerate(self.node_scalar_values):
            self.node_scalar_values[i] = self.node_scalar_values[i][self.nodecoords_allocated]
        self.edgeconn_allocated = self.edgeconn_allocated[self.edgeconn_allocated]
        self.edgepoints_allocated = self.edgepoints_allocated[self.edgepoints_allocated]
        
        self.node_ptr = self.nodecoords.shape[0]-1
        self.edge_ptr = self.edgeconn.shape[0]-1
        self.edgepnt_ptr = self.edgepoints.shape[0]-1
        
    def convert_edgepoints_to_nodes(self,interp_radius_factor=None):

        nedgepoint = self.edgepnt_ptr
        strt = self.node_ptr
        with tqdm(total=nedgepoint) as pbar:
            pbar.update(self.node_ptr)
            while True:
                nep = self.nedgepoints[self.edgeconn_allocated] #graph.get_data('NumEdgePoints')
                sind = np.where(nep>2)
                if len(sind[0])>0:            
                    self.insert_node_in_edge(sind[0][0],1)
                    pbar.update(1)
                else:
                    break
                
    def insert_nodes_in_edges(self,interp_resolution=None,interp_radius_factor=None,filter=None):
        
        points = self.edgepoints[self.edgepoints_allocated]
        if filter is None:
            filter_pts = np.ones(points.shape[0],dtype='bool')
        else:
            # Convert from edge to edgepoint
            filter_pts = np.repeat(filter,npoints)
            
        # Add filter field
        self.graph = self.set_in_graph()
        self.graph.add_field(name='Filter',marker=f'@{len(self.graph.fields)+1}',definition='POINT',type='bool',nelements=1,nentries=[0])  
        self.graph.set_data(filter_pts,name='Filter')
        
        self.set_scalars()
        
        print('Inserting nodes in edges...')

        i = 0
        ninterp = 0
        while True:
            filter_pts = self.scalar_values[-1]
            radii = self.scalar_values[self.radind]

            npoints = self.nedgepoints[self.edgeconn_allocated]
            i0 = np.sum(npoints[:i])
            i1 = i0 + npoints[i]
            pts = self.edgepoints[i0:i1]

            if filter_pts[i0]==True: # Ignore if filter is False           
                lengths = arr([np.linalg.norm(pts[j]-pts[j-1]) for j in range(1,npoints[i])])
                meanRadius = np.mean(radii[i0:i1])
                cur_interp_res = interp_radius_factor*meanRadius
                #print(i,ninterp,self.edge_ptr)
                if np.sum(lengths)>cur_interp_res:
                    stmp = np.where(np.cumsum(lengths)>=cur_interp_res)
                    if len(stmp[0])>0 and npoints[i]>2 and (stmp[0][0]+1)<(npoints[i]-1):
                        _ = self.insert_node_in_edge(i,stmp[0][0]+1)
                        ninterp += 1
                        
            i += 1
            if i>=self.edge_ptr:
                break
                
        self.graph.remove_field('Filter')
        
    def set_edge(self,edge_index,edgepoints,new_scalar_values):
    
        nedgepoints = self.nedgepoints[self.edgeconn_allocated]
        edgeCoords = self.edgepoints[self.edgepoints_allocated]

        npoints = edgepoints.shape[0]
        npoints_cur = nedgepoints[int(edge_index)]
        dif = npoints - npoints_cur
    
        if self.edgepoints.shape[0]-self.edgepnt_ptr<=dif:
            self.preallocate_edgepoints(self.n_all,set_pointer_to_start=False)
        
        x0 = int(np.sum(nedgepoints[:int(edge_index)]))
        x1 = x0 + int(nedgepoints[int(edge_index)])
        if npoints==npoints_cur:
            self.edgepoints[x0:x1] = edgepoints
        elif npoints>npoints_cur:
            
            nedgepoints[int(edge_index)] = npoints
            self.nedgepoints[self.edgeconn_allocated] = nedgepoints
            # Reallocate existing data  
            alloc = self.edgepoints_allocated
            alloc[x1+dif:] = alloc[x1:-dif]
            alloc[x0:x1+dif] = True
            self.edgepoints_allocated = alloc
            edgeCoords = self.edgepoints #[self.edgepoints_allocated]
            edgeCoords[x1+dif:] = edgeCoords[x1:-dif]
            edgeCoords[x0:x1+dif] = edgepoints
            #self.edgepoints[self.edgepoints_allocated] = edgeCoords
            self.edgepoints = edgeCoords
            for i in range(len(self.scalar_values)):
                scalars = self.scalar_values[i] #[self.edgepoints_allocated]
                scalars[x1+dif:] = scalars[x1:-dif]
                scalars[x0:x1+dif] = new_scalar_values[i]
                #self.scalar_values[i][self.edgepoints_allocated] = scalars
                self.scalar_values[i] = scalars
            if len(self.edgepoints)==0 or np.all(self.edgepoints_allocated==False):
                breakpoint()
        elif npoints<npoints_cur:
            #breakpoint()
            nedgepoints[int(edge_index)] = npoints
            self.nedgepoints[self.edgeconn_allocated] = nedgepoints
            # Reallocate existing data  
            alloc = self.edgepoints_allocated
            alloc[x1+dif:dif] = alloc[x1:]
            alloc[dif:] = False
            self.edgepoints_allocated = alloc
            edgeCoords = self.edgepoints#[self.edgepoints_allocated]
            edgeCoords[x1+dif:dif] = edgeCoords[x1:]
            edgeCoords[x0:x1+dif] = edgepoints
            edgeCoords[dif:] = 0.
            self.edgepoints = edgeCoords
            for i in range(len(self.scalar_values)):
                scalars = self.scalar_values[i] #[self.edgepoints_allocated]
                scalars[x1+dif:dif] = scalars[x1:]
                scalars[x0:x1+dif] = new_scalar_values[i]
                #self.scalar_values[i][self.edgepoints_allocated] = scalars
                self.scalar_values[i] = scalars
            if len(self.edgepoints)==0 or np.all(self.edgepoints_allocated==False):
                breakpoint()
        else:
            breakpoint()

    def insert_node_in_edge(self,edge_index,edgepoint_index,new_scalar_values=None,node_location_only=False,fr=0.5,newpoint=None):
    
        # Returns the new node index and the two new edges (if any are made)
        
        nodeCoords = self.nodecoords[self.nodecoords_allocated]
        edgeConn = self.edgeconn[self.edgeconn_allocated]
        nedgepoints = self.nedgepoints[self.edgeconn_allocated]
        edgeCoords = self.edgepoints[self.edgepoints_allocated]
        scalars,scalar_names = [],[]
        for i,sc in enumerate(self.scalar_values):
            scalars.append(self.scalar_values[i][self.edgepoints_allocated])
            scalar_names.append(self.scalars[i]['name'])
        node_scalars,node_scalar_names = [],[]
        for i,sc in enumerate(self.node_scalar_values):
            node_scalars.append(self.node_scalar_values[i][self.nodecoords_allocated])
            node_scalar_names.append(self.node_scalars[i]['name'])
    
        nnode = len(nodeCoords)
        nedge = len(edgeConn)
        nedgepoint = len(edgeCoords)
        
        x0 = int(np.sum(nedgepoints[:int(edge_index)]))
        x1 = x0 + int(nedgepoints[int(edge_index)])
        edge = edgeCoords[x0:x1]
        npoints = edge.shape[0]
        
        start_node = edgeConn[edge_index,0]
        end_node = edgeConn[edge_index,1]  

        # Calculate location of insertion point, and create a new edge point where new node will subsequently be created
        if True: #fr is not None or npoints==2:
            # Calculate distance along the edge
            dists = np.linalg.norm(edge-edge[0],axis=1)
            if dists[-1]==0.:
                if node_location_only:
                    return None
                else:
                    return None, None, None, None
            t = np.cumsum(dists)

            # Calculate insertion point
            new_loc = t[-1]*fr
            s0 = np.where(t<=new_loc)[0][-1]
            s1 = np.where(t>=new_loc)[0][0]
            s0fr = dists[s0]/t[-1]
            s1fr = dists[s1]/t[-1]
            sfr = fr - s0fr

            if newpoint is None:
                newpoint = edge[s0] + (edge[s1] - edge[s0])*sfr
            
            # Option to return locaiton only and bypass creation of new edge and node
            if node_location_only:
                return newpoint
            
            # Insert new point into edge
            new_edge = np.vstack([edge[:s0+1],newpoint,edge[s1:]])
            # Get current scalar values
            edge_scalar_values = [x[x0:x1] for x in scalars]
            # Insert additional vlaue for new node
            new_scalar_values = [np.hstack([x[:s0+1],[x[s0]],x[s1:]]) for x in edge_scalar_values]
            #print(new_scalar_values)
            self.set_edge(edge_index,new_edge,new_scalar_values)
            edgepoint_index = s0 + 1
            edge = new_edge
            npoints += 1
            #breakpoint()
        
        # Reload data
        nedgepoints = self.nedgepoints[self.edgeconn_allocated]
        edgeCoords = self.edgepoints[self.edgepoints_allocated]
        x0 = int(np.sum(nedgepoints[:int(edge_index)]))
        x1 = x0 + int(nedgepoints[int(edge_index)])
        scalars = []
        for i,sc in enumerate(self.scalar_values):
            scalars.append(self.scalar_values[i][self.edgepoints_allocated])
               
        xp = int(edgepoint_index)
        new_node_coords = edge[xp]
        
        if int(edgepoint_index)<npoints-1 and int(edgepoint_index)>0:
            new_edge0 = edge[:xp+1]
            new_edge1 = edge[xp:]
        elif int(edgepoint_index)<=0:
            print('ERROR: GVars.insert_node_in_edge: Edgepoint index<=0!')
            breakpoint()
            return edge, None, start_node, None
        elif int(edgepoint_index)>=npoints-1:
            print('ERROR: GVars.insert_node_in_edge: Edgepoint index>number of edgepoints!')
            breakpoint()
            return edge, None, end_node, None
        else:
            return None, None, None, None
            
        # Assign the first new edge to the location of the supplied edge
        # Create a new location for the second new edge
        nedgepoints[int(edge_index)] = new_edge0.shape[0]
        nedgepoints = np.concatenate([nedgepoints,[new_edge1.shape[0]]])
        
        # Squeeze in new edges into storage array
        # Grab all edge coordinates prior to edge to be bisected
        if x0>0:
            edgeCoords_0 = edgeCoords[:x0]
        else:
            edgeCoords_0 = []
        # Edge coordinates listed after the bisected edge
        if edgeCoords.shape[0]>x0+npoints:
            edgeCoords_1 = edgeCoords[x1:]
        else:
            edgeCoords_1 = []

        edgeCoords = np.concatenate([x for x in [edgeCoords_0,new_edge0.copy(),edgeCoords_1,new_edge1.copy()] if len(x)>0 and not np.all(x)==-1])
        
        # Amend original connection
        new_node_index = nodeCoords.shape[0]
        edgeConn[edge_index] = [start_node,new_node_index]
        new_conn = np.asarray([new_node_index,end_node])
        edgeConn = np.concatenate([edgeConn,[new_conn]])
        new_edge_index = nedge
        # Add in new node coords
        nodeCoords = np.concatenate([nodeCoords,[new_node_coords]])
        
        new_node_scalars = [x[start_node] for x in node_scalars]
        
        # Sort out scalars
        for i,data in enumerate(node_scalars):
            if node_scalar_names[i]=='NodeLabel':
                new_node_scalars[i] = self.graph.unique_node_label()
            node_scalars[i] = np.concatenate([data,[new_node_scalars[i]]])
            
        for i,data in enumerate(scalars):
            if x0>0:
                sc_0 = data[:x0]
            else:
                sc_0 = []
            if data.shape[0]>x0+npoints:
                sc_1 = data[x1:]
            else:
                sc_1 = []
            new_sc0 = data[x0:x0+xp+1]
                        
            if scalar_names[i]=='EdgeLabel': 
                new_lab = self.graph.unique_edge_label()
                new_sc1 = np.repeat(new_lab,data[x0+xp:x1].shape[0])
            else:
                new_sc1 = data[x0+xp:x1]
            scalars[i] = np.concatenate([x for x in [sc_0,new_sc0.copy(),sc_1,new_sc1.copy()] if len(x)>0 and not np.all(x)==-1])
        
        self.set_nodecoords(nodeCoords,scalars=node_scalars)  
        self.set_edgeconn(edgeConn,nedgepoints)  
        self.set_edgepoints(edgeCoords,scalars=scalars)
           
        return new_edge0.copy(), new_edge1.copy(), new_node_index, new_conn
        
    def set_in_graph(self):
        fieldNames = self.graph.fieldNames #['VertexCoordinates','EdgeConnectivity','EdgePointCoordinates','NumEdgePoints','Radii','VesselType','midLinePos']
        fields = self.graph.fields
        scalars = self.graph.get_scalars()
        scalar_names = [x['name'] for x in scalars]
        node_scalars = self.graph.get_node_scalars()
        node_scalar_names = [x['name'] for x in node_scalars]
        
        nodecoords = self.nodecoords[self.nodecoords_allocated]
        edgeconn = self.edgeconn[self.edgeconn_allocated]
        nedgepoints = self.nedgepoints[self.edgeconn_allocated]
        edgepoints = self.edgepoints[self.edgepoints_allocated]
        scalar_values = [[] for i in range(len(self.scalar_values))]
        for i,sc in enumerate(self.scalar_values):
            scalar_values[i] = self.scalar_values[i][self.edgepoints_allocated]
        node_scalar_values = [[] for i in range(len(self.node_scalar_values))]
        for i,sc in enumerate(self.node_scalar_values):
            node_scalar_values[i] = self.node_scalar_values[i][self.nodecoords_allocated]
        
        for i,field in enumerate(fields):
            if field['name']=='VertexCoordinates':
                self.graph.set_data(nodecoords.astype('float32'),name=fieldNames[i])
            elif field['name']=='EdgeConnectivity':
                self.graph.set_data(edgeconn.astype('int'),name=fieldNames[i])
            elif field['name']=='EdgePointCoordinates':
                self.graph.set_data(edgepoints.astype('float32'),name=fieldNames[i])
            elif field['name']=='NumEdgePoints':
                self.graph.set_data(nedgepoints.astype('int'),name=fieldNames[i])
            elif field['name'] in scalar_names:
                data = scalar_values[scalar_names.index(field['name'])]
                self.graph.set_data(data,name=fieldNames[i])
            elif field['name'] in node_scalar_names:
                data = node_scalar_values[node_scalar_names.index(field['name'])]
                self.graph.set_data(data,name=fieldNames[i])

        self.graph.set_definition_size('VERTEX',nodecoords.shape[0])
        self.graph.set_definition_size('EDGE',edgeconn.shape[0])
        self.graph.set_definition_size('POINT',edgepoints.shape[0])  
        self.graph.set_graph_sizes(labels=False)
        self.graph.edgeList = None
        return self.graph
