# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:50:50 2017

@author: simon
"""

from pymira import spatialgraph
import os
from scipy.spatial import ConvexHull
import numpy as np

path = r'C:\Users\simon\Dropbox\170606_Ben Vessel Networks\C1M3\2%'

graph = spatialgraph.SpatialGraph()
graph.read(os.path.join(path,'C1M3_flow_sims_no_selfconn.am'))

extent = graph.node_spatial_extent()

nodecoords = graph.get_data('VertexCoordinates')
#hull = ConvexHull(nodecoords)
#hullcoords = nodecoords[hull.vertices,0], nodecoords[hull.vertices,1], nodecoords[hull.vertices,2]
#
#def in_hull(p, hull):
#    """
#    Test if points in `p` are in `hull`
#
#    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
#    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
#    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
#    will be computed
#    """
#    from scipy.spatial import Delaunay
#    if not isinstance(hull,Delaunay):
#        hull = Delaunay(hull)
#
#    return hull.find_simplex(p)>=0

subvol = [400.,400.,500.] # um
#centre = [800,800,500]
#centre = [500,500,500]
centre = [1500,1000,1500]
#centre = [1400,1100,1000]

verts = [[ centre[0]-subvol[0]/2.,centre[0]+subvol[0]/2.],
         [ centre[1]-subvol[1]/2.,centre[1]+subvol[1]/2.],
         [ centre[2]-subvol[2]/2.,centre[2]+subvol[2]/2.] ]
    
nnode = len(nodecoords)
keepNode = np.zeros(nnode)
node_to_delete = []
for i,nc in enumerate(nodecoords):
    if nc[0]>verts[0][0] and nc[0]<verts[0][1] \
        and nc[1]>verts[1][0] and nc[1]<verts[1][1] \
        and nc[2]>verts[2][0] and nc[2]<verts[2][1]:
        keepNode[i] = 1
    else:
        node_to_delete.append(i)
        
ed = spatialgraph.Editor()
ed.delete_nodes(graph,node_to_delete)

graph = ed.remove_disconnected_nodes(graph)

graph.write(os.path.join(path,'subvol_{}_{}_{}.am'.format(centre[0],centre[1],centre[2])))