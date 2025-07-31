# -*- coding: utf-8 -*-
"""
Created on Tue May 02 17:36:01 2017

@author: simon
"""

from pymira import mesh
import numpy as np
import os
import scipy.ndimage

interDir = 'C:\\Users\\simon\\Dropbox\\160113_paul_simulation_results\\LS147T\\1\\interstitium_calcs'
f = os.path.join(interDir,'interstitium_inlet3939.npz')

data = np.load(f)
grid = data['grid']
grid_dims = data['grid_dims']
embedDims = data['embedDims']
        
pixdim = [data['dx'],data['dy'],data['dz'],data['dt']]

boundingBox = data['embedDims'].flatten()
cur = grid[-1,:,:,:]
medwinsize = 5
#cur = scipy.ndimage.filters.median_filter(cur,size=medwinsize)

m = mesh.Mesh(data=cur,boundingBox=boundingBox)
ofile = r'C:\Users\simon\Dropbox\test_mesh.am'
import pdb
pdb.set_trace()
m.write(ofile)

#import nibabel as nib
#ndims = len(cur.shape)
#img = nib.Nifti1Image(cur,affine=np.eye(4))
#hdr = img.header
#hdr['pixdim'][1] = pixdim[0]
#hdr['pixdim'][2] = pixdim[1]
#hdr['pixdim'][3] = pixdim[2]
#
#ofile = r'C:\Users\simon\Dropbox\test_mesh.nii'
#print('Saving to {}'.format(ofile))
#nib.save(img,ofile)