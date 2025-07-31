# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 07:18:10 2017

@author: simon
"""

import numpy as np
import spatialgraph

nturn = 10
nnode = 5
nedge = nnode
nconnpoints = int(nturn/np.float(nnode))
#nturn = nedge*nconnpoints
npoint = nturn + nnode

centre = [10.,10.,10.]
radius = 5.
thickness = 1.
flowValue = 0.5

points = np.zeros([npoint,3],dtype='float')
nodes = np.zeros([nnode,3],dtype='float')
connectivity = np.zeros([nedge,2],dtype='int')
nedgepoints = np.zeros([nedge],dtype='int')
edgepoints = np.zeros([npoint,3],dtype='float')
radii = np.zeros([npoint],dtype='float') + thickness
flow = np.zeros([npoint],dtype='float') + flowValue

angular_spacing = np.deg2rad(360. / np.float(nturn))

nodeCount = 0
pointCount = 0
curAngle = 0
turnCount = 0
print npoint,nconnpoints
while turnCount<nturn:
#for i in range(0,npoint,nconnpoints):
    print 'NODE',nodeCount,np.rad2deg(curAngle)
    print 'EDGE',pointCount,np.rad2deg(curAngle)
    # Start point
    edgepoints[pointCount,:] = [radius*np.cos(curAngle)+centre[0],
                   radius*np.sin(curAngle)+centre[1],
                   0.]
    pointCount += 1

    nodes[nodeCount,:] = edgepoints[pointCount-1,:]
    if nodeCount<nnode-1:
        connectivity[nodeCount,:] = [nodeCount,nodeCount+1]
    else:
        connectivity[nodeCount,:] = [nodeCount,0]
    nedgepoints[nodeCount] = nconnpoints + 1
    nodeCount += 1
    
    for j in range(pointCount+1,pointCount+nconnpoints+1):
        curAngle += angular_spacing
        turnCount += 1
        print 'EDGE',pointCount,np.rad2deg(curAngle)
        #print 'TURNS:',turnCount
        edgepoints[pointCount,:] = [radius*np.cos(curAngle)+centre[0],
                   radius*np.sin(curAngle)+centre[1],
                   0.]
        pointCount += 1
                
midpoint = int(nnode / 2.)
connectivity = np.append(connectivity,[[0,midpoint]],axis=0)
nedgepoints = np.append(nedgepoints,[2],axis=0)
edgepoints = np.append(edgepoints,[nodes[0,:]],axis=0)
edgepoints = np.append(edgepoints,[nodes[midpoint,:]],axis=0)
radii = np.append(radii,[thickness,thickness],axis=0)
flow = np.append(flow,[flowValue,flowValue],axis=0)

nnode = nodes.shape[0]
nedge = nedgepoints.shape[0]
npoint = edgepoints.shape[0]
    
graph = spatialgraph.SpatialGraph(initialise=True,scalars=['Radii','Flow'])
graph.set_definition_size('VERTEX',nnode)
graph.set_definition_size('EDGE',nedge)
graph.set_definition_size('POINT',npoint)
graph.set_data(nodes,name='VertexCoordinates')
graph.set_data(connectivity,name='EdgeConnectivity')
graph.set_data(nedgepoints,name='NumEdgePoints')
graph.set_data(edgepoints,name='EdgePointCoordinates')
graph.set_data(radii,name='Radii')
graph.set_data(flow,name='Flow')
#graph._print()
ofile = 'C:\\Anaconda2\\Lib\\site-packages\\pymira\\circle.am'
graph.write(ofile)