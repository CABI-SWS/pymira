# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:42:15 2017

@author: simon
"""

from pymira import spatialgraph
import os

dir_ = r'C:\Users\simon\Dropbox\Ben Vessel Networks'
f = os.path.join(dir_,'C2M3_vessels.am')

print 'Reading graph: {}'.format(f)
graph = spatialgraph.SpatialGraph()
graph.read(f)
print 'Graph read'

editor = spatialgraph.Editor()
graph = editor.largest_graph(graph)

ofile = os.path.join(dir_,'C2M3_vessels_largest_network.am')
graph.write(ofile)
