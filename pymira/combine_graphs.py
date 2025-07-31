# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:37:47 2017

@author: simon

Merge two spatial grphs
Required for converting Paul Sweeney's files

"""

import pymira.spatialgraph as sp
import os
import numpy as np
join = os.path.join
from tqdm import tqdm, trange

def merge_graphs(graph1,graph2):

    # Add fields from graph2 that aren't in graph1

    dif1  = list(set(graph1.fieldNames) - set(graph2.fieldNames))
    dif2  = list(set(graph2.fieldNames) - set(graph1.fieldNames))
    
    for fName in dif2:
        f = graph2.get_field(fName)
        marker = graph1.generate_next_marker()
        f['marker'] = marker
        #print(('Adding {} {}...'.format(marker,fName)))
        graph1.fields.append(f)
        
def combine_graphs(graph1,graph2):

    # Combine fields common to both graphs
    
    req_fields = ['VertexCoordinates', 'EdgePointCoordinates', 'EdgeConnectivity', 'NumEdgePoints']

    # Common fields
    fields = list(set(graph1.fieldNames).intersection(graph2.fieldNames))
    
    graphComb = graph1.copy()
    
    # Get data sizes
    for fName in req_fields:
        f1 = graph1.get_field(fName)
        f2 = graph2.get_field(fName)
        if fName=='VertexCoordinates':
            nnode1 = f1['data'].shape[0]
            nnode2 = f2['data'].shape[0]
        elif fName=='EdgePointCoordinates':
            npoints1 = f1['data'].shape[0]
            npoints2 = f2['data'].shape[0]
        elif fName=='EdgeConnectivity':
            nconn1 = f1['data'].shape[0]
            nconn2 = f2['data'].shape[0]
    
    # Combine fields
    for fName in fields:
        f1 = graph1.get_field(fName)
        f2 = graph2.get_field(fName)
        
        if fName=='EdgeConnectivity':
            # Offset all values by the number of nodes in graph 1
            data = np.concatenate([f1['data'],f2['data']+nnode1])
        else:
            data = np.concatenate([f1['data'],f2['data']])
        
        #print('Combining {}'.format(fName))
        graphComb.set_data(data,name=fName)
    
    #print(nnode1,nnode2)
    graphComb.set_definition_size('VERTEX',nnode1+nnode2)
    graphComb.nnode = nnode1+nnode2
    graphComb.set_definition_size('EDGE',nconn1+nconn2)
    graphComb.nedge = nconn1+nconn2
    graphComb.set_definition_size('POINT',npoints1+npoints2)
    graphComb.nedgepoints = npoints1+npoints2
    graphComb.set_graph_sizes()

    if 'NodeLabel' in graphComb.fieldNames:
        newNodeLabel = np.linspace(0,graphComb.nnode-1,graphComb.nnode,dtype='int')
        graphComb.set_data(newNodeLabel,name='NodeLabel')
    if 'EdgeLabel' in graphComb.fieldNames:
        newEdgeLabel = np.linspace(0,graphComb.nedge-1,graphComb.nedge,dtype='int')
        npoints = graphComb.get_data('NumEdgePoints')
        graphComb.set_data(np.repeat(newEdgeLabel,npoints),name='EdgeLabel')
    
    return graphComb
    
def combine_cco(path,mFiles,ofile):
    
    # Flag inlet / outlet nodes that shouldn't be connected
    ignore_node = np.zeros(len(mFiles),dtype='int')
    
    for i,f in enumerate(tqdm(mFiles)):
    
        if i==0:
            ignore_node[i] = 0
        else:
            ignore_node[i] = graph.nnode
 
        graph_to_add = sp.SpatialGraph()
        #print('Merging {}'.format(f))
        #graph_to_add.read(join(path,f))
        graph_to_add.read(f)
        
        marker = graph_to_add.generate_next_marker()
        if 'artery' in f:
            vesselType = np.zeros(graph_to_add.nedgepoint)
        elif 'vein' in f:
            vesselType = np.zeros(graph_to_add.nedgepoint) + 1
        if 'upper' in f:
            midLinePos = np.zeros(graph_to_add.nedgepoint)
        elif 'lower' in f:
            midLinePos = np.zeros(graph_to_add.nedgepoint) + 1
        else:
            midLinePos = np.zeros(graph_to_add.nedgepoint)
        marker = graph_to_add.generate_next_marker()
        graph_to_add.add_field(name='VesselType',marker=marker,definition='POINT',type='float',nelements=1,data=vesselType)
        marker = graph_to_add.generate_next_marker()
        graph_to_add.add_field(name='midLinePos',marker=marker,definition='POINT',type='float',nelements=1,data=midLinePos)
        
        if i>0:
            graph = combine_graphs(graph,graph_to_add)
        else:
            graph = graph_to_add

    graph.sanity_check()
    
    if ofile is not None:
        print('Combined graph written to {}'.format(join(path,ofile)))
        graph.write(join(path,ofile))
    
    return graph

