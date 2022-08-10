# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 11:49:52 2016

@author: simon

Amira SpatialGraph loader and writer

"""

from pymira import amiramesh
import numpy as np
arr = np.asarray
import os
from tqdm import tqdm, trange # progress bar
import open3d as o3d


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
    new_inds = new_inds_lookup[inds] 
    # Remove -1 values that reference deleted nodes
    new_inds = new_inds[(new_inds[:,0]>=0) & (new_inds[:,1]>=0)]
    return vals[keep],new_inds,new_inds_lookup
    
def delete_vertices(graph,keep_nodes,return_lookup=False): # #verts,edges,keep_nodes):

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
    else:
        return graph

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, 0.
    axis_ = np.cross(a, b)
    l = np.linalg.norm(axis_)
    if l>0.:
        axis_ = axis_ / np.linalg.norm(axis_)
        angle = np.arccos(np.dot(a, b))
    else:
        angle = 0.

    return axis_, angle

class SpatialGraph(amiramesh.AmiraMesh):
    
    def __init__(self,header_from=None,initialise=False,scalars=[],node_scalars=[],path=None):
        amiramesh.AmiraMesh.__init__(self)
        
        self.nodeList = None
        self.edgeList = None
        self.path = path
        
        if header_from is not None:
            import copy
            self.parameters = copy.deepcopy(header_from.parameters)
            self.definitions = copy.deepcopy(header_from.definitions)
            self.header = copy.deepcopy(header_from.header)
            self.fieldNames = copy.deepcopy(header_from.fieldNames)
        if initialise:
            self.initialise(scalars=scalars,node_scalars=node_scalars)
            
    def initialise(self,scalars=None,node_scalars=None):
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
                self.add_field(name=sc,marker='@{}'.format(offset),
                                  definition='POINT',type='float',
                                  nelements=1,nentries=[0])
                offset = len(self.fields) + 1
                                  
        if len(node_scalars)>0:
            if type(node_scalars) is not list:
                node_scalars = [node_scalars]
            for i,sc in enumerate(node_scalars):
                self.add_field(name=sc,marker='@{}'.format(i+offset),
                                  definition='VERTEX',type='float',
                                  nelements=1,nentries=[0])
                offset = len(self.fields) + 1
                              
        self.fieldNames = [x['name'] for x in self.fields]
        
    def read(self,*args,**kwargs):
        if not amiramesh.AmiraMesh.read(self,*args,**kwargs):
            return False
        if self.get_parameter_value("ContentType")!="HxSpatialGraph":
            print('Warning: File is not an Amira SpatialGraph!')

        self.set_graph_sizes()
                
        return True
        
    def set_graph_sizes(self):
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
            
    def get_standard_fields(self):

        res = []
        nodecoords = self.get_data('VertexCoordinates')
        edgeconn = self.get_data('EdgeConnectivity')
        edgepoints = self.get_data('EdgePointCoordinates')
        nedgepoints = self.get_data('NumEdgePoints')
        
        return nodecoords,edgeconn,edgepoints,nedgepoints
        
    def rescale_coordinates(self,xscale,yscale,zscale,ofile=None):
        nodeCoords = self.get_data('VertexCoordinates')
        edgeCoords = self.get_data('EdgePointCoordinates')
        
        for i,n in enumerate(nodeCoords):
            nodeCoords[i] = [n[0]*xscale,n[1]*yscale,n[2]*zscale]
        for i,n in enumerate(edgeCoords):
            edgeCoords[i] = [n[0]*xscale,n[1]*yscale,n[2]*zscale]
        
        if ofile is not None:
            self.write(ofile)
            
    def rescale_radius(self,rscale,ofile=None):
        radf = self.get_radius_field()
        radii = radf['data']
        #radii = self.get_data('Radii')
        mnR = np.min(radii)
        for i,r in enumerate(radii):
            radii[i] = r * rscale / mnR
            
        if ofile is not None:
            self.write(ofile)
    
    def reset_data(self):
        for x in self.fields:
            x['data'] = None
        for x in self.definitions:
            x['size'] = [0]
        for x in self.fields:
            x['shape'] = [0,x['nelements']]
            x['nentries'] = [0]
            
    def add_node(self,node=None,index=0,coordinates=[0.,0.,0.]):
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
        
        nodeCoords = None # free memory (necessary?)
    
    def add_node_connection(self,startNode,endNode,edge):
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

       #Identify terminal nodes
       #nodeCoords = self.fields[0]['data']
       #nnode = nodeCoords.shape[0]
       #nConn = np.asarray([0]*nnode)
       conn = self.fields[1]['data']
       nConn = np.asarray([len(np.where((conn[:,0]==i) | (conn[:,1]==i))[0]) for i in range(self.nnode)])
       #nConn = np.asarray([len([j for j,x in enumerate(conn) if i in x]) for i in range(self.nnode)])
       return nConn
       
       #for i in range(nnode):
       #    #ntmp1 = len(np.where(conn[:,0]==i)[0])
       #    #ntmp2 = len(np.where(conn[:,1]==i)[0])
       #    ntmp1 = len([j for j,x in enumerate(conn) if i in x])
       #    nConn[i] = ntmp1 #+ ntmp2
           
       #return nConn
       
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
        
        pbar = tqdm(total=nnode) # progress bar
        self.nodeList = [0] * nnode
        for nodeIndex in range(nnode):
            pbar.update(1)
            self.nodeList[nodeIndex] = Node(graph=self,index=nodeIndex)
        pbar.close()
            
        if path is not None:
            self.write_node_list(path=path)
            
        return self.nodeList
        
    def clone(self):
        import copy
        return copy.deepcopy(self)
        
    def node_spatial_extent(self):
        
        nodecoords = self.get_data('VertexCoordinates')
        rx = [np.min(nodecoords[:,0]),np.max(nodecoords[:,0])]
        ry = [np.min(nodecoords[:,1]),np.max(nodecoords[:,1])]
        rz = [np.min(nodecoords[:,2]),np.max(nodecoords[:,2])]
        return [rx,ry,rz]
        
    def edge_spatial_extent(self):
        
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
        
    def constrain_nodes(self,xrange=[None,None],yrange=[None,None],zrange=[None,None],no_copy=True):
        
        assert len(xrange)==2
        assert len(yrange)==2
        assert len(zrange)==2

        if not no_copy:        
            graph = self.clone()
        else:
            graph = self

        nodeCoords = graph.get_data('VertexCoordinates')
        nnode = len(nodeCoords)
        #nedge = len(edgeConn)
        #nedgepoint = len(edgeCoords)
        
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
        keepNode = np.zeros(nnode,dtype='bool') + True
        for i in range(nnode):
            x,y,z = nodeCoords[i,:]
            if x<xrange[0] or x>xrange[1] or y<yrange[0] or y>yrange[1] or z<zrange[0] or z>zrange[1]:
                keepNode[i] = False
                
        nodes_to_delete = np.where(keepNode==False)
        nodes_to_keep = np.where(keepNode==True)
        if len(nodes_to_keep[0])==0:
            print('No nodes left!')
            return
        
        editor = Editor()
        return editor.delete_nodes(self,nodes_to_delete[0])
        
    def crop(self,*args,**kwargs):
        return self.constrain_nodes(*args,**kwargs)
        
    def remove_field(self,fieldName):
        f = [(i,f) for (i,f) in enumerate(self.fields) if f['name']==fieldName]
        if f[0][1] is None:
            print(('Could not locate requested field: {}'.format(fieldName)))
            return
        _  = self.fields.pop(f[0][0])
        
#    def remove_edges(self,indices):
#        pass
#        for ind in indices:
#            pass
        
    def get_node(self,index):
        return Node(graph=self,index=index)
        
    def get_edge(self,index):
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
        # Get array relating edgepoints to the index of the edge that they're from
        points = self.get_data('EdgePointCoordinates')
        npoints = points.shape[0]
        nEdgePoint = self.get_data('NumEdgePoints')
        edgePointIndex = np.zeros(npoints,dtype='int')
        offset = 0
        edgeCount = 0
        for npCur in nEdgePoint:
            edgePointIndex[offset:offset+npCur] = edgeCount
            edgeCount += 1
            offset += npCur
        return edgePointIndex
        
    def get_scalars(self):
        return [f for f in self.fields if f['definition'].lower()=='point' and len(f['shape'])==1 and f['name']!='EdgePointCoordinates']
        
    def get_radius_field(self):
        names = ['radius','radii','diameter','diameters','thickness']
        for name in names:
            match = [self.fields[i] for i,field in enumerate(self.fieldNames) if field.lower()==name.lower()]
            if len(match)>0:
                return match[0]
        return None
        
    def edgepoint_indices(self,edgeIndex):
        nedgepoints = self.get_data('NumEdgePoints')
        edgeCoords = self.get_data('EdgePointCoordinates')
        nedge = len(nedgepoints)
        
        assert edgeIndex>=0
        assert edgeIndex<nedge
        
        npoints = nedgepoints[edgeIndex]
        if edgeIndex==0:
            start_index = 0
        else:
            start_index = np.sum(nedgepoints[0:edgeIndex])
        end_index = start_index + npoints
        
        return [start_index,end_index]
        
    def _print(self):
        print('GRAPH')
        print(('Fields: {}'.format(self.fieldNames)))
        for f in self.fields:
            print(f)
            
    def sanity_check(self,deep=False):
        
        self.set_graph_sizes()
        
        for d in self.definitions:
            defName = d['name']
            defSize = d['size'][0]
            fields = [f for f in self.fields if f['definition']==defName]
            for f in fields:
                if f['nentries'][0]!=defSize:
                    print(('{} field size does not match {} definition size!'.format(f['name'],defName)))
                if f['shape'][0]!=defSize:
                    print(('{} shape size does not match {} definition size!'.format(f['name'],defName)))
                if not all(x==y for x,y in zip(f['data'].shape,f['shape'])):
                    print(('{} data shape does not match shape field!'.format(f['name'])))

        if deep:
            for nodeInd in range(self.nnode):
                node = self.get_node(nodeInd)
                for i,e in enumerate(node.edges):
                    if not node.edge_indices_rev[i]:
                        if not all(x==y for x,y in zip(e.start_node_coords,node.coords)):
                            print(('Node coordinates ({}) do not match start of edge ({}) coordinates: {} {}'.format(node.index,e.index,e.start_node_coords,node.coords)))
                        if not all(x==y for x,y in zip(e.coordinates[0,:],e.start_node_coords)):
                            print(('Edge start point does not match edge/node start ({}) coordinates'.format(e.index)))
                        if not all(x==y for x,y in zip(e.coordinates[-1,:],e.end_node_coords)):
                            print(('Edge end point does not match edge/node end ({}) coordinates'.format(e.index)))
                    else:
                        if not all(x==y for x,y in zip(e.end_node_coords,node.coords)):
                            print(('Node coordinates ({}) do not match end of edge ({}) coordinates'.format(node.index,e.index)))
                        if not all(x==y for x,y in zip(e.coordinates[0,:],e.start_node_coords)):
                            print(('Edge end point does not match edge start (REVERSE) ({}) coordinates'.format(e.index)))
                        if not all(x==y for x,y in zip(e.coordinates[-1,:],e.end_node_coords)):
                            print(('Edge start point does not match edge end (REVERSE) ({}) coordinates'.format(e.index)))        

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
        
    def nodes_connected_to(self,nodes,path=None):
        
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
        
    def connected_nodes(self,index):
        vertCoords = self.get_data('VertexCoordinates')
        edgeConn = self.get_data('EdgeConnectivity')
            
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
        
    def identify_graphs(self,progBar=False,ignore_node=None,ignore_edge=None,verbose=False,add_scalar=True):
        
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
        
    def edge_scalar_to_node_scalar(self,name):

        scalar_points = self.get_data(name)
        if scalar_points is None:
            return None
    
        verts = self.get_data('VertexCoordinates')
        conns = self.get_data('EdgeConnectivity')
        npoints = self.get_data('NumEdgePoints')
        points = self.get_data('EdgePointCoordinates')
        
        scalar_nodes = np.zeros(verts.shape[0],dtype=scalar_points.dtype)
    
        for nodeIndex in range(self.nnode):
            edgeIds = np.where((conns[:,0]==nodeIndex) | (conns[:,1]==nodeIndex))
            if len(edgeIds[0])>0:
                edgeId = edgeIds[0][0]
                    
                npts = int(npoints[edgeId])
                x0 = int(np.sum(npoints[0:edgeId]))
                vtype = scalar_points[x0:x0+npts]
                pts = points[x0:x0+npts,:]
                node = verts[nodeIndex]
                if np.all(pts[0,:]==node):
                    scalar_nodes[nodeIndex] = scalar_points[0]
                else:
                    scalar_nodes[nodeIndex] = scalar_points[-1]
        return scalar_nodes
 
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
        
    def plot_graph(self, cylinders=None, vessel_type=None, color=None, plot=True, min_radius=0., domain_radius=None, domain_centre=arr([0.,0.,0.]),radius_based_resolution=True,cyl_res=10,use_edges=True):
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
        if use_edges:

            if cylinders is None:
                try:
                    print('Preparing graph...')
                    edge_def = self.get_definition('EDGE')
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
                            elif vt[0]==0: # artery
                                col = [1.,0.,0.]
                            elif vt[1]==1:
                                col = [0.,0.,1.]
                            else:
                                col = [0.,1.,0.]
                            
                            if np.any(rads>=min_radius) and (domain_radius is None or np.any(np.linalg.norm(coords-domain_centre)<=domain_radius)):
                                for j in range(1,coords.shape[0]):
                                    if rads[j]>=min_radius:
                                        x0,x1 = coords[j-1],coords[j]
                                        vec = x1-x0
                                        height = np.linalg.norm(x1-x0)
                                        
                                        if height>0. and (domain_radius is None or (np.linalg.norm(x0-domain_centre<=domain_radius) and np.linalg.norm(x1-domain_centre<=domain_radius))):
                                            vec = vec / height
                                            if rads[j]<20. and radius_based_resolution:
                                                resolution = 4
                                            else:
                                                resolution = cyl_res
                                            cyl = o3d.geometry.TriangleMesh.create_cylinder(height=height,radius=rads[j], resolution=resolution)
                                            translation = x0 + vec*height*0.5
                                            cyl = cyl.translate(translation, relative=False)
                                            axis, angle = align_vector_to_another(np.asarray([0.,0.,1.]), vec)
                                            if angle!=0.:
                                                axis_a = axis * angle
                                                cyl = cyl.rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cyl.get_center()) 
                                            
                                            cyl.paint_uniform_color(col)
                                            if cylinders is not None:
                                                cylinders += cyl
                                            else:
                                                cylinders = cyl
                except Exception as e:
                    #breakpoint()
                    print(e)
  
            #else:
            #    diameters = rads * 2
            #    nbins = 20
            #    r_bins = np.linspace(0.,np.max(diameters),nbins)
            #    for i in range(1,nbins):
            #        inds = np.where((diameters>r_bins[i-1]) & (diameters<=r_bins[i-1]))
            #        if len(inds[0])>0:
            #            line_set = o3d.geometry.LineSet()
            #            line_set.points = o3d.utility.Vector3dVector(points[inds,:])
            #            line_set.lines = o3d.utility.Vector2iVector(conns)
            #    #colors = np.zeros([conns.shape[0],3],dtype='int')
            #    #line_set.colors = o3d.utility.Vector3dVector(colors)
            #    #o3d.visualization.draw_geometries([line_set])
                        
            #breakpoint()
            if plot:
                o3d.visualization.draw_geometries([cylinders],mesh_show_wireframe=False)
                #o3d.visualization.draw_geometries([pcd_a,pcd_v],mesh_show_wireframe=True)
            return cylinders    
        else:
            # Legacy
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            skip = 1
            ax.scatter(nc[::skip,0],nc[::skip,1],nc[::skip,2], c='r', marker='o',s=1)

class Editor(object):

    def _remove_intermediate_nodes(self, nodeCoords,edgeConn,nedgepoints,edgeCoords,scalars=None):
    
        # Returns an edited graph where nodes with exactly two connections are replaced by edgepoints
        # TBC
    
        nnode = len(nodeCoords)
        nedge = len(edgeConn)
        nedgepoint = len(edgeCoords)
        
        for i,node in nodeCoords:
            conns_with_node = [j for j,c in enumerate(edgeConn) if np.any(c==i)]
            if len(conns_with_node)==2:
                pass
            

    def _insert_node_in_edge(self, edge_index,edgepoint_index,nodeCoords,edgeConn,nedgepoints,edgeCoords,scalars=None):
    
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

    def _del_nodes(self,nodes_to_delete,nodeCoords,edgeConn,nedgepoints,edgeCoords,scalars=None):
    
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
        nodeInds = np.arange(0,nodeCoords.shape[0]-1)
        edgeConn = graph.get_data('EdgeConnectivity')
        
        zero_conn = [x for x in nodeInds if x not in edgeConn]
        if len(zero_conn)==0:
            return graph
            
        graph = self.delete_nodes(graph,zero_conn)
        print(('{} isolated nodes removed'.format(len(zero_conn))))
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

    def remove_intermediate_nodes(self,graph,file=None,nodeList=None,path=None):
        
        import pickle
        import os

        print('Generating node list...')
        nodeList = graph.node_list(path=path)
        print('Node list complete.')
        
        nnode = graph.nnode
        nedge = graph.nedge        
        nconn = np.array([node.nconn for node in nodeList])
        new_nodeList = []
        new_edgeList = []
        
        # Initialise list for mapping old to new node indices
        node_index_lookup = np.zeros(nnode,dtype='int') - 1
        edge_index_lookup = np.zeros(nedge,dtype='int') - 1
        # Mark if a node has become an edge point
        node_now_edge = np.zeros(nnode,dtype='int')
        node_converted = np.zeros(nnode,dtype='int')
        node_edges_checked = np.zeros(nnode,dtype='int')
        edge_converted = np.zeros(nedge,dtype='int')
        
        newNodeCount = 0
        newEdgeCount = 0
        
        for cntr,node in enumerate(nodeList):
            
            # Is the current node branching (or terminal)?
            if (node.nconn==1 or node.nconn>2) and node_now_edge[node.index]==0 and node_edges_checked[node.index]==0:
                # If so, make a new node object
                if node_converted[node.index]==0:
                    print(('NODE (START) {} {} {}'.format(newNodeCount,node.index,node.nconn)))
                    newNode = Node(index=newNodeCount,coordinates=node.coords,connecting_node=[],old_index=node.index)
                    # Mark node as having been converted to a new node (rather than an edge)
                    node_converted[node.index] = 1
                    new_nodeList.append(newNode)
                    newNodeIndex = newNodeCount
                    node_index_lookup[node.index] = newNodeIndex
                    newNodeCount += 1
                else:
                    print(('NODE (START, REVISITED) {} {} {}'.format(newNodeCount,node.index,node.nconn)))
                    ind = node_index_lookup[node.index]
                    if ind<0:
                        import pdb
                        pdb.set_trace()
                    newNode = new_nodeList[ind]
                    newNodeIndex = newNode.index
                    
                node_edges_checked[node.index] = 1
                
                edges_complete = np.zeros(node.nconn,dtype='bool') + False
                
                # Loop through each branch
                for node_counter,connecting_node_index in enumerate(node.connecting_node):

                    # Initialise variables                    
                    curNodeIndex = connecting_node_index
                    endNode = None
                    visited = [node.index]
                    visited_edges = []

                    # Compile connecting edges -
                    connecting_edge = [e for x,e in zip(node.connecting_node,node.edges) if x==connecting_node_index]
                        
                    for connEdge in connecting_edge:

                        # Check if edge has already been converted (e.g. the return of a loop)
                        if edge_converted[connEdge.index]==0:
                            # Check whether to reverse coordinates in edge
                            if connEdge.end_node_index==node.index:
                                reverse_edge_indices = True
                                ecoords = connEdge.coordinates
                                ecoords = ecoords[::-1,:]
                                scalars = [s[::-1] for s in connEdge.scalars]
                            elif connEdge.start_node_index==node.index:
                                reverse_edge_indices = False
                                ecoords = connEdge.coordinates
                                scalars = connEdge.scalars
                            else:
                                import pdb
                                pdb.set_trace()
                                
                            # Create edge object to add points to during walk
                            print(('EDGE {}'.format(newEdgeCount)))
                            newEdge = Edge(index=newEdgeCount,start_node_index=newNode.index,
                                               start_node_coords=newNode.coords,
                                               coordinates=ecoords,
                                               npoints=ecoords.shape[0],
                                               scalars=scalars,
                                               scalarNames=connEdge.scalarNames)
                                               
                            new_edgeList.append(newEdge)
                            assert len(new_edgeList)==newEdgeCount+1
                            
                            edge_index_lookup[connEdge.index] = newEdgeCount
                            visited_edges.append(connEdge.index)
                            edge_converted[connEdge.index] = 1
                            
                            newEdgeCount += 1
                        
                            # Start walking - complete when a branching node is encountered
                            endFlag = False
                            
                            while endFlag is False:
                                curNode = nodeList[curNodeIndex]
                                visited.append(curNodeIndex)
                                
                                # If it's an intermediate (connecting) node
                                if curNode.nconn==2:
                                    # Check which connecting nodes have been visited already
                                    next_node_index = [x for x in curNode.connecting_node if x not in visited]
                                    # Get connecting edge (connected to unvisited, unconverted node)
                                    connecting_edge_walk = [e for x,e in zip(curNode.connecting_node,curNode.edges) if x not in visited and edge_converted[e.index]==0 ]
                                    # If no unvisited nodes have been identified...
                                    if len(connecting_edge_walk)==0:
                                        # Look for branching nodes that have been visited (i.e. loops)
                                        connecting_edge_walk = [e for x,e in zip(curNode.connecting_node,curNode.edges) if edge_converted[e.index]==0 ]
                                        if len(connecting_edge_walk)==1:
                                            foundConn = False
                                            # Check both start and end node indices
                                            for i,j in enumerate([connecting_edge_walk[0].start_node_index,connecting_edge_walk[0].end_node_index]):
                                                if nodeList[j].nconn > 2:
                                                    #Loop!
                                                    # Look for a connecting branch point
                                                    next_node_index = [j]
                                                    foundConn = True
                                            # If still nothing found...
                                            if not foundConn:
                                                import pdb
                                                pdb.set_trace()
                                                
                                    # If a connected edge has been found...
                                    if len(connecting_edge_walk)>0:
                                        # Check whether to reverse edge points
                                        if connecting_edge_walk[0].end_node_index==curNode.index:
                                            reverse_edge_indices = True
                                        elif connecting_edge_walk[0].start_node_index==curNode.index:
                                            reverse_edge_indices = False
                                        else:
                                            import pdb
                                            pdb.set_trace()
            
                                        # Add in connecting edge points
                                        # Reverse edge coordinates if necessary
                                        if reverse_edge_indices:
                                            scalars = [s[::-1] for s in connecting_edge_walk[0].scalars]
                                            #scalars = [s[1:-1] for s in scalars]
                                            coords = connecting_edge_walk[0].coordinates
                                            coords = coords[::-1,:]
                                            newEdge.add_point(coords,scalars=scalars,remove_last=True)
                                        else:
                                            scalars = connecting_edge_walk[0].scalars
                                            newEdge.add_point(connecting_edge_walk[0].coordinates,
                                                          scalars=scalars,remove_last=True)
        
                                        # Mark that node is now an edge point
                                        node_now_edge[curNodeIndex] = 1
            
                                        # If we've run out of nodes, then quit;
                                        # Otherwise, walk to the next node
                                        if len(next_node_index)==0:                                
                                            endFlag = True
                                        else:
                                            curNodeIndex = next_node_index[0]
                                            edge_converted[connecting_edge_walk[0].index] = 1
                                            edge_index_lookup[connecting_edge_walk[0].index] = newEdge.index
                                    else: # No connected edges found
                                        print('No connected edges...')
                                        endFlag = True
                                        
                                else: # Branch or terminal point
                                    endFlag = True
                                    end_node_index = curNode.index
                                    # Add in final edge coordinates, if necessary
                                    if not all([x==y for x,y in zip(newEdge.coordinates[-1,:],curNode.coords)]):
                                        # Reverse edge coordinates if necessary
                                        if connEdge.start_node_index!=curNode.index:
                                            scalars = [s[::-1] for s in connEdge.scalars]
                                            coords = connEdge.coordinates
                                            coords = coords[::-1,:]
                                            newEdge.add_point(coords,scalars=scalars,remove_last=True)
                                        else:
                                            scalars = connEdge.scalars
                                            newEdge.add_point(connEdge.coordinates,
                                                          scalars=scalars,remove_last=True)
                                    
                                # Sort out end nodes and edges
                                if endFlag:
                                    # Find end node
                                    if newEdge is None:
                                        import pdb
                                        pdb.set_trace()
                                    # If node has already been visited
                                    if node_converted[curNodeIndex]==1 and node_now_edge[curNodeIndex]==0:
                                        end_node_index_new = int(node_index_lookup[end_node_index])
                                        if end_node_index_new<0:
                                            import pdb
                                            pdb.set_trace()
                                        endNode = new_nodeList[end_node_index_new]
                                        #print('REVISITED NODE {} (END)'.format(endNode.index))
                                    # If node hasn't been converted, and isn't an edge
                                    elif node_now_edge[curNodeIndex]==0:
                                        print(('NODE (END) {} {}'.format(newNodeCount,curNode.index)))
                                        end_node_index_new = newNodeCount
                                        endNode = Node(index=end_node_index_new,coordinates=curNode.coords,connecting_node=[],old_index=curNode.index)
                                        node_converted[curNodeIndex] = 1
                                        new_nodeList.append(endNode) #[newNodeCount] = endNode
                                        node_index_lookup[end_node_index] = newNodeCount
                                        newNodeCount += 1
                                    else:
                                        import pdb
                                        pdb.set_trace()
                                        
                                    try:
                                        stat = newEdge.complete_edge(endNode.coords,end_node_index_new)
                                        if stat!=0:
                                            import pdb
                                            pdb.set_trace()
                                        print(('EDGE COMPLETE: end node {}'.format(endNode.index)))
                                    except Exception as e:
                                        print(e)
                                        import pdb
                                        pdb.set_trace()
                                        
                                    res = newNode.add_edge(newEdge)
                                    if not res:
                                        import pdb
                                        pdb.set_trace()
                                    if endNode.index!=newNode.index:
                                        res = endNode.add_edge(newEdge,reverse=True)
                                        if not res:
                                            import pdb
                                            pdb.set_trace()
                                    
                                    edges_complete[node_counter] = True
                                    
                                    break
                        else: # Edge has already been converted
                            newEdgeIndex = edge_index_lookup[connEdge.index]
                            if newEdgeIndex<0:
                                import pdb
                                pdb.set_trace()
                            newEdge = new_edgeList[newEdgeIndex]
                            if newEdge.start_node_index==newNode.index:
                                res = newNode.add_edge(newEdge)
                                if not res:
                                    print(('Error: Edge {} is already attached to node {}'.format(newEdge.index,newNode.index)))
                            elif newEdge.end_node_index==newNode.index:
                                res = newNode.add_edge(newEdge,reverse=True)
                                if not res:
                                    print(('Error: Edge {} is already attached to node {}'.format(newEdge.index,newNode.index)))
                            else:
                                import pdb
                                pdb.set_trace()
                            edges_complete[node_counter] = True
                        
#                    if edges_complete[node_counter]==False:
#                        import pdb
#                        pdb.set_trace()
                        
#                if newNode.nconn==2:
#                    import pdb
#                    pdb.set_trace()
#                if newNode.nconn!=node.nconn:
#                    import pdb
#                    pdb.set_trace()
#                    #assert endNode is not None
#                if not all(edges_complete):
#                    import pdb
#                    pdb.set_trace()

        #return new_nodeList
        #se = np.where(edge_converted==0)
        #elu = np.where(edge_index_lookup<0)
        #incomplete_edges = [e for e in new_edgeList if e.complete is False]
        #incomp = np.where(edge_converted==0)
        #node2 = [n for n in new_nodeList if n.nconn==2]

        new_nedge = newEdgeCount
        new_nnode = newNodeCount
        
        new_graph = graph.node_list_to_graph(new_nodeList)
        return new_graph
        
    def largest_graph(self, graph):

        graphNodeIndex, graph_size = graph.identify_graphs(progBar=True)
        largest_graph_index = np.argmax(graph_size)
        node_indices = np.arange(graph.nnode)
        nodes_to_delete = node_indices[graphNodeIndex!=largest_graph_index]
        graph = self.delete_nodes(graph,nodes_to_delete)
        
        return graph
        
    def remove_graphs_smaller_than(self, graph, lim, pfile=None):

        if pfile is None:
            graphNodeIndex, graph_size = graph.identify_graphs(progBar=True)
        else:
            import pickle
            plist = pickle.load(open(pfile,"r"))
            graphNodeIndex, graph_size = plist[0],plist[1]
            
        graph_index_to_delete = np.where(graph_size<lim)
        nodes_to_delete = []
        for gitd in graph_index_to_delete[0]:
            inds = np.where(graphNodeIndex==gitd)
            nodes_to_delete.extend(inds[0].tolist())
        nodes_to_delete = np.asarray(nodes_to_delete)
            
        #node_indices = np.arange(graph.nnode)
        #nodes_to_delete = node_indices[graphNodeIndex!=largest_graph_index]

        #breakpoint()
        graph = self.delete_nodes(graph,nodes_to_delete)
        graph.set_graph_sizes()
        
        return graph
        
        
    def interpolate_edges(self,graph,iterp_resolution=5.,filter=None):
        
        """
        Linear interpolation of edge points, to a fixed minimum resolution
        """
        
        coords = graph.get_data('VertexCoordinates')
        points = graph.get_data('EdgePointCoordinates')
        npoints = graph.get_data('NumEdgePoints')
        conns = graph.get_data('EdgeConnectivity')
        
        scalars = graph.get_scalars()
        scalar_data = [x['data'] for x in scalars]
        scalar_data_interp = [[] for x in scalars]
        
        if filter is None:
            filter = np.ones(conns.shape[0],dtype='bool')
        
        pts_interp,npoints_interp = [],[]
        for i,conn in enumerate(conns):
            if npoints[i]==2:
                if not filter[i]:
                    pts_interp.extend(points[i0:i1])
                    npoints_interp.append(2)
                    for j,sd in enumerate(scalar_data):
                        scalar_data_interp[j].extend(sd[i0:i1])
                else:
                    i0 = np.sum(npoints[:i])
                    i1 = i0 + npoints[i]
                    pts = points[i0:i1]
                        
                    length = np.linalg.norm(pts[1]-pts[0])
                    if length>iterp_resolution:
                        ninterp = np.clip(int(np.ceil(length / iterp_resolution)+1),2,None)
                    else:
                        ninterp = 2
                        
                    pts_interp.extend(np.linspace(pts[0],pts[1],ninterp))
                    
                    for j,sd in enumerate(scalar_data):
                        sdc = sd[i0:i1]
                        scalar_data_interp[j].extend(np.linspace(sdc[0],sdc[1],ninterp))
                    
                    npoints_interp.append(ninterp)
            else:
                breakpoint()

        pts_interp = arr(pts_interp)
        npoints_interp = arr(npoints_interp)
        graph.set_data(pts_interp,name='EdgePointCoordinates')
        graph.set_data(npoints_interp,name='NumEdgePoints')
       
        for j,sd in enumerate(scalar_data_interp):
            graph.set_data(arr(sd),name=scalars[j]['name'])
        
        graph.set_definition_size('POINT',pts_interp.shape[0])        

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
                graph.edgeList = []
            edgeInds = [e.index for e in graph.edgeList]
                
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
                        #import pdb
                        #pdb.set_trace()
                        newEdge = Edge(graph=graph,index=e)
                        edgeInds.append(e)
                        graph.edgeList.append(newEdge)
                    else:
                        newEdge = [edge for edge in graph.edgeList if edge.index==e]
                        if len(newEdge)==1:
                            newEdge = newEdge[0]
                        else:
                            import pdb
                            pdb.set_trace()
                    self.edges.append(newEdge)
                    self.edge_indices_rev.append(False)
                    self.connecting_node.append(edgeConn[e,1])
            if len(s1)>0:
                for e in s1[0]:
                    self.edge_indices.append(e)
                    self.edge_indices_rev.append(True)
                    if e not in edgeInds:                  
                        newEdge = Edge(graph=graph,index=e)
                        edgeInds.append(e)
                        graph.edgeList.append(newEdge)
                    else:
                        newEdge = [edge for edge in graph.edgeList if edge.index==e]
                        if len(newEdge)==1:
                            newEdge = newEdge[0]
                        else:
                            import pdb
                            pdb.set_trace()
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
        
        if graph is not None:
            nodeCoords = graph.get_field('VertexCoordinates')['data']
            edgeConn = graph.get_field('EdgeConnectivity')['data']
            nedgepoints = graph.get_field('NumEdgePoints')['data']
            self.coordinates = np.squeeze(self.get_coordinates_from_graph(graph,index))
            self.start_node_index = edgeConn[index,0]
            self.start_node_coords = nodeCoords[self.start_node_index,:]
            #self.end_node_index = edgeConn[index,1]
            #self.end_node_coords = nodeCoords[self.end_node_index,:]
            self.npoints = nedgepoints[index]
            #self.coordinates = self.get_coordinates_from_graph(graph,index)
            self.scalars,self.scalarNames = self.get_scalars_from_graph(graph,index)
            self.complete_edge(nodeCoords[edgeConn[index,1],:],edgeConn[index,1])
            
        #if self.coordinates is not None:
        #    assert self.npoints==len(self.coordinates[:,0])
        #    # Make sure start coordinates match
        #    assert all([x==y for x,y in zip(self.coordinates[0,:],self.start_node_coords)])
        #    # Make sure end coordinates match
        #    assert all([x==y for x,y in zip(self.coordinates[-1,:],self.end_node_coords)])
        
    def get_coordinates_from_graph(self,graph,index):
        nedgepoints = graph.get_field('NumEdgePoints')['data']
        coords = graph.get_field('EdgePointCoordinates')['data']
        nprev = np.sum(nedgepoints[0:index])
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
            
    def complete_edge(self,end_node_coords,end_node_index):
        stat = 0
        self.end_node_coords = np.asarray(end_node_coords)
        self.end_node_index = end_node_index
        self.complete = True
        
        if not all([x==y for x,y in zip(self.end_node_coords,self.coordinates[-1,:])]):
            print('Warning: End node coordinates do not match last edge coordiate!')
            stat = -1
        if not all([x==y for x,y in zip(self.start_node_coords,self.coordinates[0,:])]):
            print('Warning: Start node coordinates do not match first edge coordiate!')
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
