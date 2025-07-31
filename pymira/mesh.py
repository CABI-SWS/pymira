# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 11:49:52 2016

@author: simon

Amira SpatialGraph loader and writer

"""

from pymira import amiramesh
import numpy as np
import os
from tqdm import tqdm # progress bar

class Mesh(amiramesh.AmiraMesh):
    
    def __init__(self,header_from=None,initialise=True,scalars=[],path=None,data=None,boundingBox=None):
        amiramesh.AmiraMesh.__init__(self)

        self.path = path
        
        if header_from is not None:
            import copy
            self.parameters = copy.deepcopy(header_from.parameters)
            self.definitions = copy.deepcopy(header_from.definitions)
            self.header = copy.deepcopy(header_from.header)
            self.fieldNames = copy.deepcopy(header_from.fieldNames)
        if initialise or data is not None:
            self.initialise(data=data,boundingBox=boundingBox)
            
    def initialise(self,data=None,boundingBox=[0,1,0,1,0,1],dimensions=[1,1,1],data_type='float'):
        self.fileType = '3D ASCII 2.0'
        self.filename = ''
        
        self.add_definition('Lattice',[0])
        
        if boundingBox is None:
            boundingBox = [0,1,0,1,0,1]
        self.add_parameter('BoundingBox',' '.join([str(b) for b in boundingBox]))
        if dimensions is None:
            dimensions=[1,1,1]
        dimStr = 'x'.join([str(d) for d in dimensions])
        self.add_parameter('Content','{} {}, uniform coordinates'.format(dimStr,data_type))
        self.add_parameter('CoordType','uniform')

        self.add_field(name='Lattice',marker='@1',
                              definition='Lattice',type='float',
                              nelements=1,nentries=[0])
                              
        offset = len(self.fields) + 1
        
        if data is not None:
            self.set_lattice_data(data,boundingBox=boundingBox)
        
    def set_lattice_data(self,data,boundingBox=None):
        data = np.asarray(data)
        np_data_type = data.dtype.str
        if np_data_type=='<f8':
            data_type = 'float'
        elif np_data_type=='<i4':
            data_type = 'short'
        else:
            print(('Data type not supported! {}'.format(np.data_type)))
            return
            
        dims = data.shape
        ndims = len(dims)
        
        self.set_data(data,name='Lattice')
        
        # Update parameters
        self.set_content(dims,data_type)
        if boundingBox is not None:
            self.set_bounding_box(boundingBox)
        else:
            boundingBox = [0,dims[0],0,dims[1],0,dims[2]] # Assumes 3D data!
            self.set_bounding_box(boundingBox)
            
        # Update definitions
        self.set_lattice_definition(dims)
        
    def set_content(self,dimensions,data_type):
        content_param = [p for p in self.parameters if p['parameter']=='Content'][0]
        dimStr = 'x'.join([str(d) for d in dimensions])
        content_param['value'] = '{} {}, uniform coordinates'.format(dimStr,data_type)
        
    def set_bounding_box(self,boundingBox):
        bb_param = [p for p in self.parameters if p['parameter']=='BoundingBox'][0]
        #bbStr = ' '.join([str(b) for b in boundingBox])
        bb_param['value'] = boundingBox
        
    def set_lattice_definition(self,dimensions):
        ld = [d for d in self.definitions if d['name']=='Lattice'][0]
        #dimStr = ' '.join([str(d) for d in dimensions])
        dimList = [d for d in dimensions]
        ld['size'] = [dimList]