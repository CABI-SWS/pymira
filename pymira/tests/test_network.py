# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:26:51 2017

@author: simon
"""

import numpy as np
from pymira import spatialgraph

nodes = [[0.,0.,0.],
         [100.,0.,0.],
         [200.,0.,0.],
         [200.,100.,0.],
         [200.,100.,100.]]
nodes = np.asarray(nodes,dtype='float')

edgeConn = [[0,1],
            [1,2],
            [2,3],
            [2,4]]
edgeConn = np.asarray(edgeConn,dtype='int')
            
edgePoints = [nodes[0],
              [10.,0.,0.],
              [30.,0.,0.],
              [50.,0.,0.],
              [75.,0.,0.],
              nodes[1],
              nodes[1],
              [110.,0.,0.],
              [130.,0.,0.],
              [150.,0.,0.],
              [175.,0.,0.],
              nodes[2],
              nodes[2],
              [200.,20.,0.],
              [200.,40.,0.],
              [200.,60.,0.],
              [200.,80.,0.],
              nodes[3],
              nodes[2],
              [200.,100.,20.],
              [200.,100.,40.],
              [200.,100.,60.],
              [200.,100.,80.],
              nodes[4] 
              ]
              
edgePoints = np.asarray(edgePoints,dtype='float')

f = 5.
edgePoints *= f
nodes *= f

nedgePoints = edgePoints.shape[0]

pressure = np.linspace(60.,30.,6*3)
pressure = np.append(pressure,np.linspace(pressure[-6],pressure[-1],6))
#print pressure
              
radii = np.zeros(nedgePoints) + 50.
velocity = np.zeros(nedgePoints) + 5.
flow = np.zeros(nedgePoints)
              
nedgepoints = [6,
               6,
               6,
               6]
nedgepoints = np.asarray(nedgepoints,dtype='int')               
               
graph = spatialgraph.SpatialGraph(initialise=True,scalars=['Radii','Velocity','Pressure','Flow'])
graph.set_definition_size('VERTEX',nodes.shape[0])
graph.set_definition_size('EDGE',edgeConn.shape[0])
graph.set_definition_size('POINT',edgePoints.shape[0])
graph.set_data(nodes,name='VertexCoordinates')
graph.set_data(edgeConn,name='EdgeConnectivity')
graph.set_data(nedgepoints,name='NumEdgePoints')
graph.set_data(edgePoints,name='EdgePointCoordinates')
graph.set_data(radii,name='Radii')
graph.set_data(velocity,name='Velocity')
graph.set_data(pressure,name='Pressure')
graph.set_data(flow,name='Flow')
ofile = 'C:\\Anaconda2\\Lib\\site-packages\\pymira\\test_graph.am'
graph.write(ofile)