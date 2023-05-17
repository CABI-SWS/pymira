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
#import matplotlib as mpl
#from matplotlib import pyplot as plt

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
    else:
        return graph

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
            
    def initialise(self,scalars=None,node_scalars=None):
    
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
        """
        Read spatial graph from .am Amira file
        """
        if not amiramesh.AmiraMesh.read(self,*args,**kwargs):
            return False
        if self.get_parameter_value("ContentType")!="HxSpatialGraph":
            print('Warning: File is not an Amira SpatialGraph!')

        self.set_graph_sizes()
                
        return True
        
    def set_graph_sizes(self):
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
        
        nodeCoords = None # free memory (necessary?)
    
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
        
    def constrain_nodes(self,xrange=[None,None],yrange=[None,None],zrange=[None,None],no_copy=True):
    
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
        return [f for f in self.fields if f['definition'].lower()=='point' and len(f['shape'])==1 and f['name']!='EdgePointCoordinates']
        
    def get_node_scalars(self):
        """
        Return scalar edge fields
        """
        return [f for f in self.fields if f is not None and f['definition'].lower()=='vertex' and len(f['shape'])==1 and f['name']!='VertexCoordinates']
        
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
        chase
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
            for nodeInd in trange(self.nnode):
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
                            err = f'Edge start point does not match edge end (REVERSE) ({e.index}) coordinates'
                            print(err)
                            #print(('Edge start point does not match edge end (REVERSE) ({}) coordinates'.format(e.index)))        

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
        
    def get_edge_lengths(self):
    
        edgeconn = self.get_data('EdgeConnectivity') 
        nedgepoints = self.get_data('NumEdgePoints')
        edgeCoords = self.get_data('EdgePointCoordinates')
        
        lengths = np.zeros(self.nedge)
        for i in trange(self.nedge):
            x0 = np.sum(nedgepoints[:i])
            npts = nedgepoints[i]
            pts = edgeCoords[x0:x0+npts]
            lengths[i] = np.sum(np.linalg.norm(pts[:-1]-pts[1:],axis=1))
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
        radius = self.get_data(self.get_radius_field_name())
        category = self.get_data('VesselType')
        
        if category is None:
            category = np.zeros(edgepoints.shape[0],dtype='bool')

        inds = np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int')
        edge_inds = np.repeat(inds,nedgepoints)
        first_edgepoint_inds = np.concatenate([[0],np.cumsum(nedgepoints)[:-1]])

        #edge_node_lookup = create_edge_node_lookup(nodecoords,edgeconn,tmpfile=tmpfile,restore=restore)
                
        # Calculate node connectivity
        node_count = self.get_node_count() #,edge_node_lookup=edge_node_lookup)
        # Find graph end nodes (1 connection only)
        term_inds = np.where(node_count==1)
        terminal_node = np.zeros(nodecoords.shape[0],dtype='bool')
        terminal_node[term_inds] = True

        # Assign a radius to nodes using the largest radius of each connected edge
        edge_radius = radius[first_edgepoint_inds]

        # Assign a category to each node using the minimum category of each connected edge (thereby favouring arteries/veins (=0,1) over capillaries (=2))
        edge_category = category[first_edgepoint_inds]
        
        # Locate arterial input(s)
        mask = np.ones(edgeconn.shape[0])
        mask[(edge_category!=0) | ((node_count[edgeconn[:,0]]!=1) & (node_count[edgeconn[:,0]]!=1))] = np.nan
        if ignore is not None:
            mask[(np.in1d(edgeconn[:,0],ignore)) | (np.in1d(edgeconn[:,1],ignore))] = np.nan
        if np.nansum(mask)==0.:
            a_inlet_node = None
        else:
            a_inlet_edge_ind = np.nanargmax(edge_radius*mask)
            a_inlet_edge = edgeconn[a_inlet_edge_ind]
            a_inlet_node = a_inlet_edge[node_count[a_inlet_edge]==1][0]
        
        # Locate vein output(s)
        mask = np.ones(edgeconn.shape[0])
        mask[(edge_category!=1) | ((node_count[edgeconn[:,0]]!=1) & (node_count[edgeconn[:,0]]!=1))] = np.nan
        if ignore is not None:
            mask[(np.in1d(edgeconn[:,0],ignore)) | (np.in1d(edgeconn[:,1],ignore))] = np.nan
        if np.nansum(mask)==0.:
            v_outlet_node = None
        else:
            v_outlet_edge_ind = np.nanargmax(edge_radius*mask)
            v_outlet_edge = edgeconn[v_outlet_edge_ind]
            v_outlet_node = v_outlet_edge[node_count[v_outlet_edge]==1][0]
        
        return a_inlet_node,v_outlet_node  

    def test_treelike(self, inlet=None, outlet=None, euler=True, ignore_type=False):

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
                            print(f'Test treelike, revisited: {next_front[dplicates]}, it: {count}')
                            dnodes = next_front[dplicates]
                            edges = self.get_edges_containing_node(dnodes)
                            return False
                        unq,cnt = np.unique(next_front,return_counts=True)
                        if np.any(cnt)>1:
                            print(f'Test treelike, duplicate paths to node: {unq[cnt>1]}')
                            #breakpoint()
                            return False
                        visited.extend(next_front.tolist())
                        prev_front = front
                        front = next_front
                    else:
                        break
                    count += 1
                    if count>edgeconn.shape[0]*2:
                        print(f'Test treelike: Count limit reached...')
                        return False
        
        # Double check
        all_in = np.in1d(np.arange(nodecoords.shape[0]),visited)
        if not np.all(all_in):
            print(f'Not all nodes visited: {np.arange(nodecoords.shape[0])[~all_in]}')
            return False
            
        gc = self.get_node_count()
        vt = self.edge_scalar_to_node_scalar('VesselType')
        edges = self.get_data('EdgeConnectivity')
        
        # Euler: Arterial nodes
        if euler:
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
                        print(f'Euler criterion failed (arterial, too many nodes! {n_anode} nodes, {n_aedges} edges)')
                    if n_anode<n_aedges+1:
                        print(f'Euler criterion failed (arterial, too many edges! {n_anode} nodes, {n_aedges} edges)')
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
                            print(f'Euler criterion failed (venous, too many nodes! {n_vnode} nodes, {n_vedges} edges)')
                        if n_vnode<n_vedges+1:
                            print(f'Euler criterion failed (venous, too many edges! {n_vnode} nodes, {n_vedges} edges)')
                        return False
            
        duplicate_edges = np.zeros(edges.shape[0],dtype='int')
        for i,x in enumerate(edges): 
            duplicate_edges[i] = len(np.where( (edges[:,0]==x[0]) & (edges[:,1]==x[1]))[0]) + len(np.where( (edges[:,1]==x[0]) & (edges[:,0]==x[1]))[0])
        if np.any(duplicate_edges>1):
            print(f'Duplicated edges!')
            return False
                
        selfconnected_edges = (edges[:,0]==edges[:,1])
        if np.any(selfconnected_edges):
            print(f'Self-connected edges!')
            return False
            
        # Test for degeneracy
        res,_ = self.test_node_degeneracy(find_all=False)
        if res: 
            print('Degenerate nodes present!')
            return False
        
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
                    
    def identify_graphs(self,progBar=False,ignore_node=None,ignore_edge=None,verbose=False,add_scalar=True):

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

        scalar_points = self.get_data(name)
        if scalar_points is None:
            return None
    
        verts = self.get_data('VertexCoordinates')
        conns = self.get_data('EdgeConnectivity')
        npoints = self.get_data('NumEdgePoints')
        points = self.get_data('EdgePointCoordinates')
        
        scalar_nodes = np.zeros(verts.shape[0],dtype=scalar_points.dtype)
        eei = self.edgepoint_edge_indices()
    
        for nodeIndex in trange(self.nnode):
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
        
    def point_scalars_to_node_scalars(self,mode='max',name=None):

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
    
        for i,conn in enumerate(tqdm(conns)):
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
        return scalar_nodes.squeeze()
        
    def point_scalars_to_edge_scalars(self,func=np.mean,name=None):

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
    
        for i,conn in enumerate(tqdm(conns)):
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
        return scalar_edges.squeeze()
 
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
        for i,_node_inds in enumerate(tqdm(inline_nodes[0])):
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
        for edge in tqdm(consol_edge):
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

    def remove_intermediate_nodesOLD(self,graph,file=None,nodeList=None,path=None):
        
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
        
    def filter_graph_by_radius(self,graph,min_filter_radius=5.,filter_clip_radius=None,write=False,ofile='',keep_stubs=False,stub_len=100.):

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
            for i in trange(graph.nedge):
                edge = graph.get_edge(i)
                rads = np.min(edge.scalars[rind])
                node_radius[edge.start_node_index] = np.max([rads,node_radius[edge.start_node_index]])
                node_radius[edge.end_node_index] = np.max([rads,node_radius[edge.end_node_index]])
            
            is_stub = np.zeros(radius.shape[0],dtype='bool')
            stub_loc = np.zeros(graph.nedge,dtype='int') - 1
            for i in trange(graph.nedge):
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
            for i,edge in enumerate(tqdm(edges)):
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
        graph.set_definition_size('VERTEX',nodecoords[0].shape[0])
        graph.set_definition_size('EDGE',edgeconn[1].shape[0])
        graph.set_definition_size('POINT',edgepoints[4].shape[0])            
        graph.set_graph_sizes()
        
        graph = delete_vertices(graph,keep_node)
        
        if write:
            graph.write(ofile)  
            
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
        
        if filter is None:
            filter = np.ones(conns.shape[0],dtype='bool')
        
        pts_interp,npoints_interp = [],np.zeros(conns.shape[0],dtype='int')-1
        for i,conn in enumerate(tqdm(conns)):
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
                    ninterp = 2    
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
                    breakpoint()

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
        """
        
        coords = graph.get_data('VertexCoordinates')
        points = graph.get_data('EdgePointCoordinates')
        npoints = graph.get_data('NumEdgePoints')
        conns = graph.get_data('EdgeConnectivity')
        radii = graph.get_radius_data()

        if filter is None:
            filter_pts = np.ones(points.shape[0],dtype='bool')
        else:
            # Convert from edge to edgepoint
            filter_pts = np.repeat(filter,npoints)
            
        # Add filter field
        graph.add_field(name='Filter',marker=f'@{len(graph.fields)+1}',definition='POINT',type='bool',nelements=1,nentries=[0])  
        graph.set_data(filter_pts,name='Filter')
        
        print('Inserting nodes in edges...')

        while True:
            coords = graph.get_data('VertexCoordinates')
            points = graph.get_data('EdgePointCoordinates')
            npoints = graph.get_data('NumEdgePoints')
            conns = graph.get_data('EdgeConnectivity')
            filter_pts = graph.get_data('Filter')
            radii = graph.get_radius_data()
        
            change = False
            
            for i,conn in enumerate(tqdm(conns)):
                i0 = np.sum(npoints[:i])
                i1 = i0 + npoints[i]
                pts = points[i0:i1]

                if filter_pts[i0]==True: # Ignore if filter is False           
                    lengths = arr([np.linalg.norm(pts[j]-pts[j-1]) for j in range(1,npoints[i])])
                    meanRadius = np.mean(radii[i0:i1])
                    cur_interp_res = interp_radius_factor*meanRadius
                    print(i,np.sum(lengths),cur_interp_res)
                    if np.sum(lengths)>cur_interp_res:
                        stmp = np.where(np.cumsum(lengths)>=cur_interp_res)
                        if len(stmp[0])>0 and npoints[i]>2 and (stmp[0][0]+1)<(npoints[i]-1):
                            # and stmp[0][0]>0 and stmp[0][-1]<npoints[i]-1:
                            gvars = GVars(graph)
                            _ = gvars.insert_node_in_edge(i,stmp[0][0]+1)
                            graph = gvars.set_in_graph()
                            change = True
                            print('Change!')
                            break
            if not change:
                break
                
        graph.remove_field('Filter')
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
        
        for i,c1 in enumerate(tqdm(nodes)):
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
        if points is None or not np.all(points[0]==self.nodecoords[new_conn[0]]) or not np.all(points[-1]==self.nodecoords[new_conn[1]]):
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
        for i,sc in enumerate(self.scalar_values):
            dt = self.scalar_values[i].dtype
            self.scalar_values[i][self.edgepnt_ptr:self.edgepnt_ptr+npts] = np.zeros(npts,dtype=dt)+new_scalar_vals[i]
        self.edgepnt_ptr += npts     
        if self.edgepnt_ptr>=self.edgepoints.shape[0]:
            self.preallocate_edgepoints(self.n_all,set_pointer_to_start=False)
        
    def remove_edges(self,edge_inds_to_remove):
    
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
        
    def insert_node_in_edge(self,edge_index,edgepoint_index):
    
        # Returns the new node index and the two new edges (if any are made)
        
        nodeCoords = self.nodecoords[self.nodecoords_allocated]
        edgeConn = self.edgeconn[self.edgeconn_allocated]
        nedgepoints = self.nedgepoints[self.edgeconn_allocated]
        edgeCoords = self.edgepoints[self.edgepoints_allocated]
        scalars = []
        for i,sc in enumerate(self.scalar_values):
            scalars.append(self.scalar_values[i][self.edgepoints_allocated])
        node_scalars = []
        for i,sc in enumerate(self.node_scalar_values):
            node_scalars.append(self.node_scalar_values[i][self.nodecoords_allocated])
    
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
            new_sc1 = data[x0+xp:x1]
            scalars[i] = np.concatenate([x for x in [sc_0,new_sc0.copy(),sc_1,new_sc1.copy()] if len(x)>0 and not np.all(x)==-1])
        
        #breakpoint()
        self.set_nodecoords(nodeCoords,scalars=node_scalars)  
        self.set_edgeconn(edgeConn,nedgepoints)  
        self.set_edgepoints(edgeCoords,scalars=scalars)
        
        #self.nodecoords[:nodeCoords.shape[0]] = nodeCoords
        #self.edgeconn[:edgeConn.shape[0]] = edgeConn
        #self.nedgepoints[:nedgepoints.shape[0]] = nedgepoints
        #self.edgepoints[:edgeCoords.shape[0]] = edgeCoords
        
        #self.nodecoords_allocated[:nodeCoords.shape[0]] = True #np.linspace(0,nodeCoords.shape[0]-1,nodeCoords.shape[0],dtype='int')
        #self.edgeconn_allocated[:edgeConn.shape[0]] = True #np.linspace(0,edgeConn.shape[0]-1,edgeConn.shape[0],dtype='int')
        #self.edgepoints_allocated[:edgeCoords.shape[0]] = True #np.linspace(0,edgeCoords.shape[0]-1,edgeCoords.shape[0],dtype='int')
        #self.node_ptr = nodeCoords.shape[0]
        #self.edge_ptr = edgeConn.shape[0]
        #self.edgepnt_ptr = edgeCoords.shape[0]

        #for i,sc in enumerate(self.scalar_values):
        #    self.scalar_values[i][:scalars[i].shape[0]] = scalars[i]
        #for i,sc in enumerate(self.node_scalar_values):
        #    self.node_scalar_values[i][:node_scalars[i].shape[0]] = node_scalars[i]
           
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
        self.graph.set_graph_sizes()
        self.graph.edgeList = None
        return self.graph
