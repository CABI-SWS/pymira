import os
from os import listdir
from os.path import isfile, join
from tqdm import trange, tqdm
import numpy as np
arr = np.asarray
import open3d as o3d
from skimage import draw
import multiprocessing
import nibabel as nib
from PIL import Image
from pymira import spatialgraph
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
import argparse

"""
Python implementation for embedding a spatial graph (from pymira) in a 3D volume
"""

"""
Worker functions---
"""
def rotate_points_about_axis(coords,axis,theta,R=None,centre=arr([0.,0.,0.])):
    """ 
    Rotate 3D coordinates by theta *deg* about an arbitrary axis, centred at origin
    """
    if R is None:     
        # Transform to rotate supplied axis to z-axis
        R1 = np.eye(4)
        axis = axis / norm(axis)
        if np.all(axis==zaxis):
            R1inv = np.eye(4)
        else:
            R1[0:3,0:3] = align_vectors(axis,zaxis)
            R1inv = np.linalg.inv(R1)
        # Rotation about z
        R2 = np.eye(4)
        a = np.deg2rad(theta)
        R2[0,0],R2[0,1],R2[1,0],R2[1,1] = cos(a),-sin(a),sin(a),cos(a)
        # Rotation from z-axis back to supplied axis
        R = R1inv.dot(R2.dot(R1))

    # Homogeneous coordinates
    coords_h = np.zeros([coords.shape[0],4],dtype=coords.dtype)
    coords_h[:,0:3] = coords
    
    if centre is not None:
        # Translate points to origin
        Tr = np.eye(4)
        Tr[:3,3] = -centre
        # Return points to original frame after rotation
        Tinv = np.eye(4)
        Tinv[:3,3] = centre
        for i,coord in enumerate(coords_h):
            #coords_h[i,:] = Tinv.dot(R.dot(Tr.dot(coord)))
            coord[0:3] -= centre
            coord = np.dot(R,coord)
            coord[0:3] += centre
            coords_h[i,:] = coord
        return coords_h[:,0:3]
    else:
        for i,coord in enumerate(coords_h):
            coords_h[i,:] = R.dot(coord)
        return coords_h[:,0:3]

def coord_to_pixel(coords,domain,dims):

    domain_size = domain[:,1] - domain[:,0]
    coord_fr = (coords - domain[:,0]) / domain_size
    inds = np.round(coord_fr * dims).astype('int')
    return inds
    
def embed_mesh(meshfile,domain=None,voxel_size=[20.,20.,20.],dims=None):

    # Step 1: Create a 3D mesh (e.g., a cube)
    meshi = o3d.io.read_triangle_mesh(meshfile)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(meshi)
    
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    
    extent = domain[:,1] - domain[:,0]
    
    if dims is not None:
        voxel_size = [extent[i]/float(dims[i]) for i in range(3)]
        nx,ny,nz = dims[0],dims[1],dims[2]
    else:
        nx = np.clip(int(extent[0]/voxel_size[0]),2,None)
        ny = np.clip(int(extent[1]/voxel_size[1]),2,None)
        nz = np.clip(int(extent[2]/voxel_size[2]),2,None)
    
    x = np.linspace(domain[0,0],domain[0,1],nx)
    y = np.linspace(domain[1,0],domain[1,1],ny)
    z = np.linspace(domain[2,0],domain[2,1],nz)
    xv, yv, zv = np.meshgrid(x,y,z,indexing='xy')
    pnts = np.vstack([xv.flatten(),yv.flatten(),zv.flatten()]).transpose()
    query_point = o3d.core.Tensor(pnts, dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query_point)
    
    x = np.linspace(0,nx-1,nx,dtype='int')
    y = np.linspace(0,ny-1,ny,dtype='int')
    z = np.linspace(0,nz-1,nz,dtype='int')
    xv, yv, zv = np.meshgrid(x,y,z,indexing='xy')
    pntsInd = np.vstack([xv.flatten(),yv.flatten(),zv.flatten()]).transpose()
    
    pntsInd = pntsInd[(occupancy.numpy()>0)]
    
    result = np.zeros([nx,ny,nz],dtype='int')
    result[pntsInd[:,0],pntsInd[:,1],pntsInd[:,2]] = 1
    
    return result
    
class EmbedVessels(object):

    """
    Object class for embedding vessel segments (cylinders) in 3D space
    """

    def __init__(self,ms=256,vessel_grid_size=None,domain=None,resolution=2.5,offset=arr([0.,0.,0.]),store_midline=True,store_midline_diameter=False,store_diameter=False,store_alpha=True):
    
        """
        Class for embedding spatial graphs into 3D arrays
        ms = 3D array size
        resolution = spatial resolution
        offset = centre of domain (um)
        """
        
        self.ms = ms # n pixels
        
        self.dtype = 'int16'
        
        if vessel_grid_size is None:
            if type(ms)==int:
                self.vessel_grid_size = arr([ms,ms,ms])
            else:
                self.vessel_grid_size = arr(ms)
        else:
            self.vessel_grid_size = np.asarray(vessel_grid_size)
            
        # Get domain size and resolution from domain and grid size (and ignore any passed resolution).
        # Otherwise use the resolution as passed
        if domain is not None:
            self.domain = domain
            self.domain_size = (self.domain[:,1] - self.domain[:,0]).squeeze()
            self.resolution = self.domain_size / self.vessel_grid_size
            self.domain_offset = None
        else:
            self.domain_offset = offset
            self.resolution = arr([resolution,resolution,resolution]) # um
            self.domain_size = self.vessel_grid_size * self.resolution # um
        
        if self.domain_offset is not None:
            self.domain = arr([[(-self.domain_size[0]/2) + self.domain_offset[0], (self.domain_size[0]/2) + self.domain_offset[0]],
                               [(-self.domain_size[1]/2) + self.domain_offset[1], (self.domain_size[1]/2) + self.domain_offset[1]],
                               [self.domain_offset[2]-(self.domain_size[2]/2), self.domain_offset[2]+(self.domain_size[2]/2.)]]) # um
        elif self.domain is None:
            self.domain = arr( [ [0,self.domain_size[0]], [0,self.domain_size[1]], [0,self.domain_size[2]] ] )
        #self.domain[:,0] += self.domain_offset
        #self.domain[:,1] += self.domain_offset
        self.vessel_grid = np.zeros(self.vessel_grid_size,dtype='float')
        #self.o2_grid = np.zeros(self.vessel_grid_size,dtype=self.dtype)
        
        if store_midline:
            self.vessel_midline = np.zeros(self.vessel_grid_size,dtype=self.dtype)
        else:
            self.vessel_midline = None
        if store_diameter:
            self.vessel_diameter = np.zeros(self.vessel_grid_size,dtype='float')
        else:
            self.vessel_diameter = None
        if store_midline_diameter:
            self.vessel_midline_diameter = np.zeros(self.vessel_grid_size,self.dtype)
        else:
            self.vessel_midline_diameter = None
        if store_alpha:
            self.alpha = np.zeros(self.vessel_grid_size,dtype=self.dtype)
        else:
            self.alpha = None
        self.surface_grid = None
        self.label_grid = None
        
        #print('Vessel embedding domain: {}'.format(self.domain_size))

    def coord_to_pix(self,coord,extent_um,dims,clip=True,voxel_size=None,y_flip=True):
        """ Convert 3D spatial coordinates into 3D pixel indices """
        if voxel_size is None:
            dims,extent_um = arr(dims),arr(extent_um)
            voxel_size = (extent_um[:,1]-extent_um[:,0])/dims
        pix = (coord-extent_um[:,0])/voxel_size
        pix = pix.astype('int')
        if clip:
            pix = np.clip(pix,0,dims-1)
        
        if y_flip:
            pix[1] = dims[1] - pix[1] - 1

        return pix
            
    def embed_sphere_aniso(self, x0, r, dims=None,fill_val=None,data=None,resolution=[1.,1.,1.],fill_mode='binary'):
    
        if data is not None:
            dims = data.shape
        else:
            if dims is None:
                dims = [100,100,100]
            data = np.zeros(dims,self.dtype)
            
        # Create ellipse
        ellip_base = draw.ellipsoid(r/2.,r/2.,r/2.,spacing=resolution,levelset=False)
        subs = np.where(ellip_base)
        sx = subs[0] + x0[0] - int(np.floor(ellip_base.shape[0]/2))
        sy = subs[1] + x0[1] - int(np.floor(ellip_base.shape[1]/2))
        sz = subs[2] + x0[2] - int(np.floor(ellip_base.shape[2]/2))
        
        sind = np.where((sx>=0) & (sx<dims[0]) & (sy>=0) & (sy<dims[1]) & (sz>=0) & (sz<dims[2]))
        if len(sind[0])==0:
            return data
        sx, sy, sz = sx[sind[0]], sy[sind[0]], sz[sind[0]]
        
        if fill_val is None:
            fill_val = 255
            
        if fill_mode=='binary':
            data[sx,sy,sz] = fill_val
        elif fill_mode=='radius':
            vals = np.zeros(sx.shape[0]) + r
            old_vals = data[sx,sy,sz]
            vals = np.max(np.vstack([vals,old_vals]),axis=0)
            data[sx,sy,sz] = vals
        elif fill_mode=='diameter':
            vals = np.zeros(sx.shape[0]) + r*2
            old_vals = data[sx,sy,sz]
            vals = np.max(np.vstack([vals,old_vals]),axis=0)
            data[sx,sy,sz] = vals
        elif fill_mode=='partial_volume':
            if r>1:
                data[sx,sy,sz] = r
            else:
                data[sx,sy,sz] = r

        return data                    
            
    def embed_sphere(self,x0,r,dims=None,fill_val=None,data=None,fill_mode='binary'):
    
        """
        Embed a sphere in a 3D volume
        x0: centre coordinate (pixels)
        r: radius (pixels)
        """
    
        if data is not None:
            dims = data.shape
        else:
            if dims is None:
                dims = [100,100,100]
            data = np.zeros(dims,self.dtype)
            
        r_upr = np.ceil(r).astype('int')
        r_lwr = np.floor(r).astype('int')
            
        z0 = np.clip(x0[2]-r_upr,0,dims[2])
        z1 = np.clip(x0[2]+r_upr,0,dims[2])
        
        if fill_val is None:
            fill_val = 1.

        for i,zp in enumerate(range(z0,z1)):
            dz = np.abs(x0[2]-zp)

            rz = np.square(r_lwr)-np.square(dz)
            if rz>0.:
                xp, yp = draw.disk(x0[0:2],np.sqrt(rz), shape=dims[0:2])
                
                if fill_mode=='binary':
                    data[xp,yp,zp] = fill_val
                elif fill_mode=='radius':
                    vals = np.zeros(xp.shape[0]) + r
                    old_vals = data[xp,yp,zp]
                    vals = np.max(np.vstack([vals,old_vals]),axis=0)
                    data[xp,yp,zp] = vals
                elif fill_mode=='diameter':
                    vals = np.zeros(xp.shape[0]) + r*2
                    old_vals = data[xp,yp,zp]
                    vals = np.max(np.vstack([vals,old_vals]),axis=0)
                    data[xp,yp,zp] = vals
                elif fill_mode=='partial_volume':
                    if r>1:
                        data[xp,yp,zp] = 1
                    else:
                        data[xp,yp,zp] += r
                    

        return data                     
            
    def embed_cylinder(self,x0i,x1i,r0i,r1i,dims=None,fill_val=None,data=None):
    
        x0 = np.array(x0i,dtype='float')
        x1 = np.array(x1i,dtype='float')
        r0 = np.array(r0i,dtype='float')
        r0_upr = np.ceil(r0i).astype('float')
        r0_lwr = np.floor(r0i).astype('float')
        r1 = np.array(r1i,dtype='float')
        r1_upr = np.ceil(r1i).astype('float')
        r1_lwr = np.floor(r1i).astype('float')
        
        if fill_val is None:
            fill_val = 1. #np.mean([r0,r1]) # Fill value is mean radius

        C = x1 - x0
        length = norm(C)

        X, Y, Z = np.eye(3)
        
        if data is None:
            if dims is None:
                dims = [100,100,100]
            data = np.zeros(dims,dtype=self.dtype)
        else:
            dims = data.shape
        nx, ny, nz = data.shape
        
        # If x0=x1, return empty data
        if np.all(C==0.):
            return data

        orient = np.argmin([geometry.vector_angle(np.asarray(x,'float'), C) for x in np.eye(3)])
        if orient==0:
            ao = arr([1,2,0])
            shape = (ny, nz)
        elif orient==1:
            ao = arr([0,2,1])
            shape = (nx, nz)
        else:
            ao = arr([0,1,2])
            shape = (nx, ny)

        theta = geometry.vector_angle(Z, C)
        alpha = geometry.vector_angle(X, x0 + C)
        
        z0_range = x0[ao[2]]-r0*np.sin(theta),x1[ao[2]]-r1*np.sin(theta)
        z1_range = x0[ao[2]]+r0*np.sin(theta),x1[ao[2]]+r1*np.sin(theta)

        z0 = np.clip(np.ceil(np.min([z0_range+z1_range])).astype('int'),0,dims[ao[2]])
        z1 = np.clip(np.floor(np.max([z0_range+z1_range])).astype('int'),0,dims[ao[2]])
        if z1-z0==0:
            return
        
        r = np.linspace(r0_lwr,r1_lwr,z1-z0)
        r_feather = np.linspace(r0_upr,r1_upr,z1-z0)
        feather_fill = np.linspace(r0-r0_lwr,r1-r1_lwr,z1-z0)
        
        #if C[ao[2]]==0.:
        #    import pdb
        #    pdb.set_trace()

        for i,zp in enumerate(range(z0,z1)):
            lam = - (x0[ao[2]] - zp)/C[ao[2]]
            #print(i,zp,length,lam,x0[ao[2]],C[ao[2]])
            P = x0 + C * lam

            minor_axis = r[i]
            major_axis = r[i] / np.cos(theta)
            minor_axis_feather = r_feather[i]
            major_axis_feather = r_feather[i] / np.cos(theta)
            
            #Test for crossing start cap
            crossing_start_cap = False
            crossing_end_cap = False
            if zp>=z0_range[0] and zp<z1_range[0]:
                crossing_start_cap = True
            if zp>=z0_range[1] and zp<z1_range[1]:
                crossing_end_cap = True
                

            xpf, ypf = draw.ellipse(P[ao[0]], P[ao[1]], major_axis_feather, minor_axis_feather, shape=shape, rotation=alpha)
            inside = []
            for x,y in zip(xpf,ypf):
                if orient==0:
                    inside.append(geometry.point_in_cylinder(x0, x1, r_feather[i], arr([zp,x,y],dtype='float')))
                elif orient==1:
                    inside.append(geometry.point_in_cylinder(x0, x1, r_feather[i], arr([x,zp,y],dtype='float')))
                else:
                    inside.append(geometry.point_in_cylinder(x0, x1, r_feather[i], arr([x,y,zp],dtype='float')))
            inside = arr(inside)
            if len(inside)>0:
                xpf,ypf = xpf[inside],ypf[inside]   

            if len(xpf)>0 and len(ypf)>0:
                if orient==0:
                    data[zp,xpf,ypf] = np.clip(data[zp,xpf,ypf] + feather_fill[i],0,1)
                elif orient==1:
                    data[xpf,zp,ypf] = np.clip(data[xpf,zp,ypf] + feather_fill[i],0,1)           
                else:
                    data[xpf,ypf,zp] = np.clip(data[xpf,ypf,zp] + feather_fill[i],0,1)        
                    
            
            xp, yp = draw.ellipse(P[ao[0]], P[ao[1]], major_axis, minor_axis, shape=shape, rotation=alpha)
            inside = []
            for x,y in zip(xp,yp):
                if orient==0:
                    inside.append(geometry.point_in_cylinder(x0, x1, r[i], arr([zp,x,y],dtype='float')))
                elif orient==1:
                    inside.append(geometry.point_in_cylinder(x0, x1, r[i], arr([x,zp,y],dtype='float')))
                else:
                    inside.append(geometry.point_in_cylinder(x0, x1, r[i], arr([x,y,zp],dtype='float')))
            inside = arr(inside)
            if len(inside)>0:
                xp,yp = xp[inside],yp[inside]   

            if len(xp)>0 and len(yp)>0:
                if orient==0:
                    data[zp,xp,yp] = fill_val
                elif orient==1:
                    data[xp,zp,yp] = fill_val            
                else:
                    data[xp,yp,zp] = fill_val

        return data             

    def embed_vessel_in_grid(self,tree,seg_id,extent=None,dims=None,r=10.,c=np.asarray([0.,0.,0.]),
                       R=None,rot=[0.,0.,0.],l=10.,inside=1,outside=0,verbose=False,no_copy=True,
                       sphere_connections=False):
        
        """ 
        Embed a cylinder inside a 3D array
        extent: bounding box for matrix (um)
        dims: grid dimensions
        r: radius (um)
        c: centre (um)
        l: length (um)
        inside: pixel value for inside vessel
        outside: pixel value for outside vessel
        """
        
        extent = self.domain
        dims = self.vessel_grid_size
        start_coord = tree.node_coords[tree.segment_start_node_id[seg_id]]
        end_coord = tree.node_coords[tree.segment_end_node_id[seg_id]]
        rs,re = tree.node_diameter[tree.segment_start_node_id[seg_id]] / 2.,tree.node_diameter[tree.segment_end_node_id[seg_id]] / 2.
        
        self._embed_vessel(start_coord,end_coord,rs,re,self.vessel_grid,dims=dims,extent=extent,tree=tree,sphere_connections=sphere_connections,seg_id=seg_id)
        
    def _embed_vessel(self,start_coord,end_coord,rs,re,vessel_grid,dims=None,extent=None,voxel_size=None,node_type=[None,None],tree=None,sphere_connections=False,
                           seg_id=None,clip_at_grid_resolution=True,fill_mode='diameter',fill_val=None,graph_embedding=True,clip_vessel_radius=[None,None],ignore_midline=False):
        c = (end_coord + start_coord) / 2.
        
        if dims is None:
            dims = np.asarray(vessel_grid.shape,dtype='int')
        if extent is None:
            extent = self.domain
        if voxel_size is None:
            voxel_size = self.resolution #(extent[:,1]-extent[:,0]) / dims

        # Radius of vessel in pixels
        #dims,extent = arr(dims),arr(extent)
        
        if clip_vessel_radius[0] is not None or clip_vessel_radius[1] is not None:
            rs = np.clip(rs,clip_vessel_radius[0],clip_vessel_radius[1])
        
        r_pix_s_f = rs/voxel_size
        r_pix_s = np.floor(r_pix_s_f).astype('int32')
        r_pix_e_f = re/voxel_size
        r_pix_e = np.floor(r_pix_e_f).astype('int32')
        r_pix_s[r_pix_s<=0] = 1
        r_pix_e[r_pix_e<=0] = 1
        # Length of vessel in pixels
        l = np.linalg.norm(end_coord-start_coord)
        l_pix_f = l*dims/(extent[:,1]-extent[:,0])
        #l_pix_f = np.clip(l_pix_f,1,dims)
        l_pix = np.clip(l_pix_f,1,dims).astype('int32')
        
        start_pix = self.coord_to_pix(start_coord,extent,dims,clip=False)
        end_pix = self.coord_to_pix(end_coord,extent,dims,clip=False)
        # Check vessel is inside grid
        if (np.all(start_pix<0) and np.all(end_pix<0)) or (np.all(start_pix>=dims) and np.all(end_pix>=dims)):
            return
            
        #vessel_grid = self.embed_cylinder(start_pix,end_pix,r_pix_s_f[0],r_pix_e_f[0],data=vessel_grid)
        
        #if np.any(np.isinf(self.vessel_grid)) or np.any(np.isnan(self.vessel_grid)):
        #    import pdb
        #    pdb.set_trace()
        
        # Midline
        start_pix = self.coord_to_pix(start_coord,extent,dims,voxel_size=voxel_size,clip=False)
        end_pix = self.coord_to_pix(end_coord,extent,dims,voxel_size=voxel_size,clip=False)
        n = int(np.max(np.abs(end_pix - start_pix)))*2 #/np.min([r_pix_s_f[0],r_pix_e_f[0]]))
        if n<=0: # No pixels
            vessel_grid = self.embed_sphere_aniso(start_pix,r_pix_s_f[0],data=vessel_grid,fill_val=fill_val,fill_mode=fill_mode,resolution=voxel_size)
            return vessel_grid
            
        xl = np.linspace(start_pix,end_pix,n,dtype='int')
        rl = np.linspace(r_pix_s_f[0],r_pix_e_f[0],n,dtype='float')
        radius = np.linspace(rs,re,n)
        
        for i,coord in enumerate(xl):
            if coord[0]<dims[0] and coord[1]<dims[1] and coord[2]<dims[2] and (clip_at_grid_resolution and rl[i]*2>=1) or not clip_at_grid_resolution:
                vessel_grid = self.embed_sphere_aniso(coord,radius[i],data=vessel_grid,fill_mode=fill_mode,fill_val=fill_val,resolution=voxel_size)
                
        # Add spheres to connect cylinders
        #if sphere_connections:
        #    vessel_grid = self.embed_sphere(end_pix,r_pix_e_f[0],data=vessel_grid,fill_mode=fill_mode)
        #    vessel_grid = self.embed_sphere(start_pix,r_pix_s_f[0],data=vessel_grid,fill_mode=fill_mode)                

        if self.vessel_midline is not None and not ignore_midline: # and np.all(start_pix>0) and np.all(start_pix<dims) and np.all(end_pix>0) and np.all(end_pix<dims):
                
                for i,coord in enumerate(xl):
                    if coord[0]<dims[0] and coord[1]<dims[1] and coord[2]<dims[2] and \
                       np.all(coord>=0) and \
                       ((clip_at_grid_resolution and rl[i]*2>=1) or not clip_at_grid_resolution):
                        self.vessel_midline[coord[0],coord[1],coord[2]] = 1
                        #vessel_grid = self.embed_sphere(coord,rl[i],data=vessel_grid,fill_mode=fill_mode)

                if graph_embedding: # Set pixel value based on node type
                    if ((clip_at_grid_resolution and rl[0]*2>=1) or not clip_at_grid_resolution) and np.all(start_pix<dims) and np.all(start_pix>=0):
                        if tree is not None and seg_id is not None:
                            # Branch nodes?         
                            if tree.node_type[tree.segment_start_node_id[seg_id]]==2:
                                self.vessel_midline[start_pix[0],start_pix[1],start_pix[2]] = 2
                            # Tips?
                            if tree.node_type[tree.segment_start_node_id[seg_id]]==3:
                                self.vessel_midline[start_pix[0],start_pix[1],start_pix[2]] = 3
                        else:
                            if node_type[0] is not None:
                                self.vessel_midline[start_pix[0],start_pix[1],start_pix[2]] = node_type[0]

                    if ((clip_at_grid_resolution and rl[-1]*2>=1) or not clip_at_grid_resolution) and np.all(end_pix<dims) and np.all(end_pix>=0):
                        if tree is not None and seg_id is not None:
                            # Branch nodes?         
                            if tree.node_type[tree.segment_end_node_id[seg_id]]==2:
                                self.vessel_midline[end_pix[0],end_pix[1],end_pix[2]] = 2
                            # Tips?
                            if tree.node_type[tree.segment_end_node_id[seg_id]]==3:
                                self.vessel_midline[end_pix[0],end_pix[1],end_pix[2]] = 3
                        else:
                            if node_type[1] is not None:
                                self.vessel_midline[end_pix[0],end_pix[1],end_pix[2]] = node_type[1]
                            
        if self.vessel_midline_diameter is not None: # Set midline pixel value based on radius (useful to have - can just set all values to 1 for midline learning)
            for i,xli in enumerate(xl):
                if xli[0]<dims[0] and xli[1]<dims[1] and xli[2]<dims[2] and np.all(xli>=0):
                    if clip_at_grid_resolution and rl[i]*2>=1:
                        self.vessel_midline_diameter[xli[0],xli[1],xli[2]] = rl[i]*2
                    elif not clip_at_grid_resolution:   
                        self.vessel_midline_diameter[xli[0],xli[1],xli[2]] = rl[i]*2

        return vessel_grid
            
    def write_vessel_grid(self,dfile,binary=False,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        #dx = self.resolution
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.vessel_grid is None:
            return
        if np.any(np.isinf(self.vessel_grid)):
            import pdb
            pdb.set_trace()

        if nifti:
            if binary:
                #grid_cpy = self.vessel_grid.copy().astype('int16')
                #grid_cpy[grid_cpy>0] = 1
                img = nib.Nifti1Image(self.vessel_grid, tr)
            else:
                img = nib.Nifti1Image(self.vessel_grid ,tr)
            print('Writing vessel grid: resolution: {}um, offset: {}, filename: {}, '.format(self.resolution,self.domain[:,0],dfile))
            nib.save(img,dfile)
        else:
            print('Writing vessel grid: resolution: {}um, offset: {}, filename: {}, '.format(self.resolution,self.domain[:,0],dfile))
            np.savez(dfile, image=self.vessel_grid, tr=tr)
            
    def write_diameter_grid(self,dfile,binary=False,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        #dx = self.resolution
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.vessel_diameter is None:
            return
        if np.any(np.isinf(self.vessel_diameter)):
            import pdb
            pdb.set_trace()

        if nifti:
            if binary:
                #grid_cpy = self.vessel_diameter.copy().astype('int16')
                #grid_cpy[grid_cpy>0] = 1
                img = nib.Nifti1Image(self.vessel_diameter, tr)
            else:
                img = nib.Nifti1Image(self.vessel_diameter ,tr)
            print('Writing diameter grid: resolution: {}um, offset: {}, filename: {}, '.format(self.resolution,self.domain[:,0],dfile))
            nib.save(img,dfile)
        else:
            print('Writing diameter grid: resolution: {}um, offset: {}, filename: {}, '.format(self.resolution,self.domain[:,0],dfile))
            np.savez(dfile, image=self.vessel_diameter, tr=tr)

    def write_midline_grid(self,dfile,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.vessel_midline is None:
            return
            
        if nifti:
            img = nib.Nifti1Image(self.vessel_midline,tr)
            print('Writing midline grid: {}'.format(dfile))
            nib.save(img,dfile)
        else:
            print('Writing midline grid: {}'.format(dfile))
            np.savez(dfile, image=self.vessel_midline, tr=tr)
        
    def write_midline_diameter_grid(self,dfile,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.vessel_midline_diameter is None:
            return
        
        if nifti:    
            img = nib.Nifti1Image(self.vessel_midline_diameter,tr)
            print('Writing midline diameter grid: {}'.format(dfile))
            nib.save(img,dfile)
        else:
            print('Writing midline diameter grid: {}'.format(dfile))
            np.savez(dfile, image=self.vessel_midline_diameter, tr=tr)
            
    def write_surface_grid(self,dfile,nifti=True,tiff=False):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.surface_grid is None:
            return
        
        if nifti:    
            img = nib.Nifti1Image(self.surface_grid,tr)
            print('Writing surface grid: {}'.format(dfile))
            nib.save(img,dfile)
        if tiff:
            from tifffile import imsave
            import tifffile as tiff
            imsave(dfile, self.surface_grid)
        if not nifti and not tiff:
            print('Writing surface grid: {}'.format(dfile))
            np.savez(dfile, image=self.surface_grid, tr=tr)
            
    def write_label_grid(self,dfile,nifti=True):

        tr = np.eye(4)
        dx = [(x[1]-x[0])/float(self.vessel_grid_size[i]) for i,x in enumerate(self.domain)]
        tr[0,0],tr[1,1],tr[2,2] = dx[0],dx[1],dx[2]
        tr[0,3],tr[1,3],tr[2,3] = self.domain[0][0],self.domain[1][0],self.domain[2][0]
        
        if self.label_grid is None:
            return
        
        if nifti:    
            img = nib.Nifti1Image(self.label_grid,tr)
            print('Writing label grid: {}'.format(dfile))
            nib.save(img,dfile)
        else:
            print('Writing label grid: {}'.format(dfile))
            np.savez(dfile, image=self.label_grid, tr=tr)
  
"""
Function to embed a graph (pymira spatialgraph class)
"""
  
def embed_graph(embedding_resolution=[10.,10.,10.],embedding_dim=[512,512,512],domain=None,store_midline_diameter=True,store_diameter=True,
                path=None,filename=None,overwrite_existing=True,graph_embedding=False,d_dir=None,m_dir=None,m2_dir=None,
                file_type='nii',centre_mode='fixed',centre_point=arr([0.,0.,0.]),rep_index=0,ignore_null_vessels=False,
                radial_centre=None,prefix='',write=True,sg=None,clip_vessel_radius=[None,None],iter_data=None,fill_mode='binary',fill_val=None,clip_at_grid_resolution=True,ind=0,
                psf=False,mc=None,subsample_factor=1,ijk=None,quiet=True,radius_scale=1.,lumen_radius_fraction=0.5,lumen=False,val=255
                ):
                
    noise = False

    if iter_data is not None:
        filename = iter_data['filename']
        embedding_resolution = iter_data['res']
        embedding_dim = iter_data['dim']
        domain = iter_data['domain']
        if 'surface_file' in iter_data.keys():
            surface_file = iter_data['surface_file']
        rep_index = iter_data['rep_index']
            
    if file_type=='npz':
        nifti = False
        file_ext = 'npz'
    else:
        nifti = True
        file_ext = 'nii'

    count = 0
    while True:
        if sg is None:
            print('Embedding file: {}, dim:{}, res:{}, path:{}, d_dir:{}'.format(filename,embedding_dim,embedding_resolution,path,d_dir))    
            sg = spatialgraph.SpatialGraph(initialise=True)
            sg.read(join(path,filename))
            print('Graph read')

        node_coords = sg.fields[0]['data']
        edges = sg.fields[1]['data']
        points = sg.fields[3]['data'] 
        radii = sg.fields[4]['data']
        npts = sg.fields[2]['data']
        
        if centre_mode=='random':
            # Pick a node to centre on
            nedge = edges.shape[0]
            nnode = node_coords.shape[0]
            npoints = np.sum(npts)
            nbins = 20
            hist = np.histogram(radii,bins=nbins)
            npch = int(np.floor(npoints/nbins))
            prob = 1. - np.repeat(hist[0]/npoints,npch)
            prob = prob / np.sum(prob)

            centre_point = np.random.choice(np.arange(0,prob.shape[0]),p=prob)
            rng = int(round(embedding_dim/4.))
            embedding_offset = points[centre_point] + np.random.uniform(-rng,rng,3)
            #centre_node = int(np.random.uniform(0,nnode))
            #embedding_offset = node_coords[centre_node]
        else:
            #centre_point = arr([embedding_dim/2.,embedding_dim/2.,embedding_dim/2.])
            domain_centre = (domain[:,1]-domain[:,0])/2.
            embedding_offset = arr([0.,0.,0.]) #centre_point - domain_centre

        if write:
            if d_dir is None:
                d_dir = path.replace('graph','volume')
            if m_dir is None:
                m_dir = path.replace('graph','volume_midline')
            if m2_dir is None:
                m2_dir = path.replace('graph','volume_midline_diameter')            

            file_stub = filename.replace('.am','')
            file_stub = file_stub.replace('.','p')
            dfile = join(d_dir,file_stub+'.{}'.format(file_ext))
            mfile = join(m_dir,file_stub+'.{}'.format(file_ext))
            mfile2 = join(m2_dir,file_stub+'.{}'.format(file_ext))

        embed_vessels = EmbedVessels(domain=domain,ms=embedding_dim,resolution=embedding_resolution,offset=embedding_offset,store_midline_diameter=store_midline_diameter,store_diameter=store_diameter)

        extent = embed_vessels.domain
        dims = embed_vessels.vessel_grid_size
        grid = embed_vessels.vessel_grid
        diameter = embed_vessels.vessel_diameter
        #print('Embedding initialised: {}, {}'.format(filename,grid.shape))
        
        #breakpoint()
        def inside_domain(coord,extent,margin=0.):
            return coord[0]>=(extent[0,0]-margin) and coord[0]<=(extent[0,1]+margin) and coord[1]>=(extent[1,0]-margin) and coord[1]<=(extent[1,1]+margin) and coord[2]>=(extent[2,0]-margin) and coord[2]<=(extent[2,1]+margin) 
           
        def inside_domain_capsule(xs,xe,radius,extent,margin=0.):
            # Define capsule parameters
            C_capsule = np.mean(np.vstack([xs,xe]),axis=0) # capsule_center
            length = np.linalg.norm(xe - xs)
            if length==0:
                return False
            O_capsule = (xe - xs) / length #capsule_orientation
            r_capsule = radius #capsule_radius
            h_capsule = length/2. #capsule_half_length
        
            C_cuboid = np.mean(extent,axis=1) #cuboid_center
            cuboid_dims = extent[:,1] - extent[:,0] #cuboid_length
            # Compute the closest point on the capsule's axis to the cuboid center
            P = C_capsule + np.dot(C_cuboid - C_capsule, O_capsule) * O_capsule

            # Calculate the distances along each axis
            dx = np.abs(P-C_cuboid)

            # Check for overlap
            if dx[0] <= (r_capsule + cuboid_dims[0]) and dx[1] <= (r_capsule + cuboid_dims[1]) and dx[2] <= (r_capsule + cuboid_dims[2]):
                # Capsule and cuboid overlap
                return True
            else:
                return False
                
        # Check any points are in the domain
        if np.any(points.min(axis=0)>domain[:,1]) or np.any(points.max(axis=0)<domain[:,0]):
            print('No vessels in domain - abandoning graph embedding')
            nvessel = 0
            break
            
        embed_count = 0
        gc = sg.get_node_count()
        for ei,edge in enumerate(edges):
            x0 = node_coords[edge[0]]
            x1 = node_coords[edge[1]]
            p0 = np.sum(npts[:ei])
            p1 = p0 + npts[ei]
            pts = points[p0:p1] # um
            rads = radii[p0:p1] * radius_scale
            
            node_type = [gc[edge[0]],gc[edge[1]]]

            for i,pt in enumerate(pts):
                if i>0:
                    xs,xe = pts[i-1],pts[i]
                    pnt_type = [1,1]
                    # Check if inside subvolume
                    #breakpoint()
                    #if inside_domain(xs,extent) and inside_domain(xe,extent):
                    if inside_domain_capsule(xs,xe,np.max([rads[i-1],rads[i]]),extent):
                        if i==1:
                            pnt_type[0] = node_type[0]
                        if i==pts.shape[0]-1:
                            pnt_type[1] = node_type[1]

                        embed_vessels._embed_vessel(xs,xe,rads[i-1],rads[i],grid,node_type=pnt_type,dims=dims,extent=extent,clip_at_grid_resolution=clip_at_grid_resolution,fill_mode='binary',fill_val=fill_val,graph_embedding=graph_embedding,clip_vessel_radius=clip_vessel_radius)
                        if diameter is not None:
                            embed_vessels._embed_vessel(xs,xe,rads[i-1],rads[i],diameter,node_type=pnt_type,dims=dims,extent=extent,clip_at_grid_resolution=clip_at_grid_resolution,fill_mode='diameter',graph_embedding=graph_embedding,clip_vessel_radius=clip_vessel_radius,ignore_midline=True)
                        embed_count += 1

        if embed_count>0 or ignore_null_vessels:
            if embed_count==0 and not quiet:
                print('No vessel verts in domain')
            elif not quiet:
                print(f'{embed_count} vessel pixels embedded')
            break
        else:
            count += 1
            print('Zero pixels embedded. Trying again (it. {} of 10)'.format(count))
            if count>10:
                return None, sg

    # END WHILE

    if write:            
        dfile = dfile.replace(file_ext,'{}_res{}_dim{}_centre{}_n{}_rep{}.{}'.format(prefix,int(embedding_resolution[0]),embedding_dim[0],centre_point,n,rep_index,file_ext))
        mfile = mfile.replace(file_ext,'{}_res{}_dim{}_centre{}_n{}_rep{}.{}'.format(prefix,int(embedding_resolution[0]),embedding_dim[0],centre_point,n,rep_index,file_ext))
        mfile2 = mfile2.replace(file_ext,'{}_res{}_dim{}_centre{}_n{}_rep{}.{}'.format(prefix,int(embedding_resolution[0]),embedding_dim[0],centre_point,n,rep_index,file_ext))

        embed_vessels.write_vessel_grid(dfile,binary=False,nifti=nifti)
        embed_vessels.write_midline_grid(mfile,nifti=nifti)
        embed_vessels.write_midline_diameter_grid(mfile2,nifti=nifti) 

    return embed_vessels, sg
    
"""
Worker function for parallisation
"""
    
def embed_worker(arg):

    print(f'Worker {arg["ind"]} initialising...')

    import time
    t0 = time.time()
    
    embed_v,sg = embed_graph(**arg)
    
    t1 = time.time()
    
    if embed_v is not None and embed_v.vessel_grid is not None:
    
        from scipy.ndimage import gaussian_filter
        from skimage.measure import block_reduce
        
        cur_vess_lrg = embed_v.vessel_grid #.copy()
        inds = np.where(cur_vess_lrg>0)
        npix = len(inds[0])
        cur_vessL = None
        if npix>0:   
            dt = time.time() - t0
            
            tpp = int(npix/dt)
            print(f'Worker {arg["ind"]}: {npix} pixels embedded in {dt:.2f}s ({tpp} pixels per second')
            
            if arg['lumen']==True: # Lumen
                argL = arg.copy()
                argL['radius_scale'] = argL['lumen_radius_fraction']
                embed_L,sgL = embed_graph(**argL)
                cur_vessL = cur_vess_lrgL = embed_L.vessel_grid
            else:
                embed_L = None
            
            subsample_factor = arg['subsample_factor']
            if subsample_factor>1:
            
                if arg['psf']: # point spread function
                    psf_size = 1.2
                    cur_vess_lrg = gaussian_filter(cur_vess_lrg,psf_size)
                    dt = time.time() - t1
                    print(f'Worker {arg["ind"]}: PSF completed in {dt:.2f}s')
                    
                    # Lumen processing
                    if embed_L is not None:
                        cur_vess_lrgL = embed_L.vessel_grid
                        cur_vess_lrgL = gaussian_filter(cur_vess_lrgL,psf_size)
            
                t2 = time.time()
                
                cur_vess = block_reduce(cur_vess_lrg,block_size=(subsample_factor,subsample_factor,subsample_factor),func=np.mean)
                
                # Lumen resample
                if embed_L is not None:
                    cur_vessL = block_reduce(cur_vess_lrgL,block_size=(subsample_factor,subsample_factor,subsample_factor),func=np.mean)
                
                cur_ml = block_reduce(embed_v.vessel_midline,block_size=(subsample_factor,subsample_factor,subsample_factor),func=np.mean)
                cur_mld = block_reduce(embed_v.vessel_midline_diameter,block_size=(subsample_factor,subsample_factor,subsample_factor),func=np.mean)
                
                dt = time.time() - t2
                print(f'Worker {arg["ind"]}: block reduce completed in {dt:.2f}s')
            else:
                cur_vess = cur_vess_lrg
                cur_ml = embed_v.vessel_midline
                cur_mld = embed_v.vessel_midline_diameter
            cv_dims = cur_vess.shape
            mc = arg['mc']
            
            # Lumen subtraction
            cur_alpha = cur_vess.copy() / 255.
            if cur_vessL is not None:
                cur_vess = cur_vess - cur_vessL
            
            cur_vess = cur_vess[mc[0,0]:cv_dims[0]-mc[0,1],mc[1,0]:cv_dims[1]-mc[1,1],mc[2,0]:cv_dims[2]-mc[2,1]]
            cur_ml = cur_ml[mc[0,0]:cv_dims[0]-mc[0,1],mc[1,0]:cv_dims[1]-mc[1,1],mc[2,0]:cv_dims[2]-mc[2,1]]
            cur_mld = cur_mld[mc[0,0]:cv_dims[0]-mc[0,1],mc[1,0]:cv_dims[1]-mc[1,1],mc[2,0]:cv_dims[2]-mc[2,1]]
            cur_alpha = cur_alpha[mc[0,0]:cv_dims[0]-mc[0,1],mc[1,0]:cv_dims[1]-mc[1,1],mc[2,0]:cv_dims[2]-mc[2,1]]
        else:
            dt = time.time() - t0
            print(f'Worker {arg["ind"]}: 0 pixels embedded in {dt:.2f}s')
            cur_vess,cur_ml,cur_mld,cur_alpha = None,None,None,None
    else:
        dt = time.time() - t0
        print(f'Worker {arg["ind"]}: 0 pixels embedded in {dt:.2f}s')
        cur_vess,cur_ml,cur_mld,cur_alpha = None,None,None,None
        
    dt = time.time() - t0
        
    print(f'Worker {arg["ind"]} worker completed in {dt:.2f}s')
                                                    
    return cur_vess,cur_ml,cur_mld,cur_alpha

def embed(graph=None,filename=None,efile=None,g_file=None,s_file=None,v_file=None,c_file=None,l_file=None,d_file=None,ml_file=None,mld_file=None,
          prefix='',mesh_file_format='ply',surface_dir=None,dsize=6,dim=arr([500,500,1500]),offset=arr([0.,0.,0.]),
          output_path=None,domain=None,patch_factor=1,subsample_factor=1,theta=0.,phi=0.,chi=0.,write=True,rotation_centre=arr([0.,0.,0.]),centre=arr([0.,0.,0.]),
          clip_at_grid_resolution=True,graph_embedding=True,parallel=True,psf=True,nproc=4,ijk=None,radius_scale=1.,lumen=False,lumen_radius_fraction=0.5):

    """
    Embed graph in a 3D volume
    offset: embedding offset in um. Mainly used to centre the surface in the volume
    """

    # Transform eye geometry parameters to embedding space
    embed_geometry = {}
    embed_geometry['domain'] = domain
    embed_geometry['rotation_centre'] = rotation_centre

    # Set geometry
    fov = domain[:,1] - domain[:,0]
    embed_domain = domain.copy()
    embed_domain[0,:] += offset[0]
    embed_domain[1,:] += offset[1]
    embed_domain[2,:] += offset[2]
    resolution = fov/dim
    
    # pre-load graph
    if graph is not None:
        sg = graph
    else:
        print(f'Loading graph: {filename}')
        sg = spatialgraph.SpatialGraph(initialise=True)
        sg.read(join(path,filename))

    if theta!=0.:
        coords = sg.get_data('VertexCoordinates')
        points = sg.get_data('EdgePointCoordinates')
        coords = rotate_points_about_axis(coords,arr([0.,0.,1.]),theta,centre=rotation_centre) # centre=arr([centre[0],centre[1],0.]))
        points = rotate_points_about_axis(points,arr([0.,0.,1.]),theta,centre=rotation_centre) #centre=arr([centre[0],centre[1],0.]))
        
        sg.set_data(coords,name='VertexCoordinates')
        sg.set_data(points,name='EdgePointCoordinates')
    if phi!=0.:   
        coords = sg.get_data('VertexCoordinates')
        points = sg.get_data('EdgePointCoordinates')
        coords = rotate_points_about_axis(coords,arr([1.,0.,0.]),phi,centre=rotation_centre)
        points = rotate_points_about_axis(points,arr([1.,0.,0.]),phi,centre=rotation_centre)
        sg.set_data(coords,name='VertexCoordinates')
        sg.set_data(points,name='EdgePointCoordinates')
    if chi!=0.:   
        coords = sg.get_data('VertexCoordinates')
        points = sg.get_data('EdgePointCoordinates')
        coords = rotate_points_about_axis(coords,arr([0.,1.,0.]),chi,centre=rotation_centre)
        points = rotate_points_about_axis(points,arr([0.,1.,0.]),chi,centre=rotation_centre)
        sg.set_data(coords,name='VertexCoordinates')
        sg.set_data(points,name='EdgePointCoordinates')

    # Patch edges in fov dimensions (um)
    pe_x,pe_y,pe_z = np.meshgrid( np.linspace(embed_domain[0,0],embed_domain[0,1],patch_factor+1), 
                                  np.linspace(embed_domain[1,0],embed_domain[1,1],patch_factor+1), 
                                  np.linspace(embed_domain[2,0],embed_domain[2,1],patch_factor+1), indexing='xy' )
    # Patch edges in  pixels
    d_x,d_y,d_z = np.meshgrid(    np.linspace(0,dim[0]-1,patch_factor+1,dtype='int'), 
                                  np.linspace(dim[1]-1,0,patch_factor+1,dtype='int'), # reversed!
                                  np.linspace(0,dim[2]-1,patch_factor+1,dtype='int'), indexing='xy' )                                  
    patch_dim = dim #(dim / patch_factor).astype('int')

    vessels = np.zeros(dim,dtype='float')
    vessel_midline = np.zeros(dim,dtype='int')
    vessel_midline_diameter = np.zeros(dim,dtype='int')
    alpha = np.zeros(dim,dtype='float')
    
    fov = embed_domain[:,1] - embed_domain[:,0]
    resolution = fov/dim
    
    # Margin in fov dimensions (um)
    margins = [0.,0.,0.] #fov * 0.3 / patch_factor
    #margins = arr([np.max([x,30.]) for x in margins])
    # Margin in pixels
    pm = np.ceil(margins / resolution).astype('int')  

    print(f'Embedding resolution: {resolution}, margins: {margins}')   
    
    # Prepare inputs
    inputs = []
    count = 0
    for i in range(1,patch_factor+1):
        for j in range(1,patch_factor+1):
            for k in range(1,patch_factor+1):
                print(i,j,k)

                # Current patch edges (fov units), including margin
                pe_xc = [np.clip(pe_x[i-1,j-1,k-1]-margins[0],embed_domain[0,0],None), np.clip(pe_x[i,j,k]+margins[0],None,embed_domain[0,1]) ]
                pe_yc = [np.clip(pe_y[i-1,j-1,k-1]-margins[1],embed_domain[1,0],None), np.clip(pe_y[i,j,k]+margins[1],None,embed_domain[1,1]) ]
                pe_zc = [np.clip(pe_z[i-1,j-1,k-1]-margins[2],embed_domain[2,0],None), np.clip(pe_z[i,j,k]+margins[2],None,embed_domain[2,1]) ]
                patch_embed_domain = arr( [ [pe_xc[0],pe_xc[1]], [pe_yc[0],pe_yc[1]], [pe_zc[0],pe_zc[1]] ] )
                
                patch_fov = patch_embed_domain[:,1] - patch_embed_domain[:,0]
                #patch_resolution = patch_fov/patch_dim
                
                # Calculate edges of patch (pixels), including pixel margin if not overhanding array edges
                ec = [ [  np.clip(d_x[i-1,j-1,k-1]-pm[0],0,None), np.clip(d_x[i,j,k]+pm[0],None,dim[0]-1) ], 
                       [  np.clip(d_y[i,j,k]-pm[1],0,None), np.clip(d_y[i-1,j-1,k-1]+pm[1],None,dim[1]-1) ],
                       [  np.clip(d_z[i-1,j-1,k-1]-pm[2],0,None), np.clip(d_z[i,j,k]+pm[2],None,dim[2]-1) ] ]
                ec = arr(ec) 
                
                # Calculate current margin
                mc = [ [ d_x[i-1,j-1,k-1] - ec[0,0], ec[0,1] - d_x[i,j,k] ],
                       [ d_y[i,j,k] - ec[1,0], ec[1,1] - d_y[i-1,j-1,k-1] ],
                       [ d_z[i-1,j-1,k-1] - ec[2,0], ec[2,1] - d_z[i,j,k] ] ]
                mc = arr(mc)
                       
                patch_dim = arr([ ec[0,1] - ec[0,0], np.abs(ec[1,1] - ec[1,0]), ec[2,1] - ec[2,0] ]) * subsample_factor
                
                #inputs.append([None,patch_dim,patch_embed_domain,[2.,None],True,centre,False,sg,clip_at_grid_resolution,graph_embedding])
                inputs.append( {'embedding_resolution':None,'embedding_dim':patch_dim,'domain':patch_embed_domain,'clip_vessel_radius':[2.5,None], \
                                 'ignore_null_vessels':True,'radial_centre':centre,'write':False,'sg':sg,'clip_at_grid_resolution':clip_at_grid_resolution,
                                 'lumen':lumen,'lumen_radius_fraction':lumen_radius_fraction,
                                 'graph_embedding':graph_embedding,'ind':count,'psf':psf,'mc':mc,'subsample_factor':subsample_factor,'ijk':[i,j,k]} )
                count += 1

    if parallel:
        import time
        t0 = time.time()
        print(f'Starting {nproc} parallel processes')
        with multiprocessing.Pool(processes=nproc) as pool:
            results = list(tqdm(pool.imap(embed_worker, inputs),total=len(inputs)))

        dt = time.time() - t0
        print(f'Embedding completed in {dt:.2f}s')
    else:
        #breakpoint()
        results = []
        rmax = 10
        for inp in inputs: #[:rmax]:
            v1,v2,v3,v4 = embed_worker(inp)
            results.append([v1,v2,v3,v4])
    
    count = 0
    print('Compiling data...')
    for count,res in enumerate(results):
        i,j,k = inputs[count]['ijk']
        print(i,j,k)
        if results[count][0] is not None:
            cur_vess,cur_ml,cur_mld,cur_alpha = results[count]
            vessels[d_x[i-1,j-1,k-1]:d_x[i,j,k],d_y[i,j,k]:d_y[i-1,j-1,k-1],d_z[i-1,j-1,k-1]:d_z[i,j,k]] = cur_vess
            vessel_midline[d_x[i-1,j-1,k-1]:d_x[i,j,k],d_y[i,j,k]:d_y[i-1,j-1,k-1],d_z[i-1,j-1,k-1]:d_z[i,j,k]] = cur_ml
            vessel_midline_diameter[d_x[i-1,j-1,k-1]:d_x[i,j,k],d_y[i,j,k]:d_y[i-1,j-1,k-1],d_z[i-1,j-1,k-1]:d_z[i,j,k]] = cur_mld
            alpha[d_x[i-1,j-1,k-1]:d_x[i,j,k],d_y[i,j,k]:d_y[i-1,j-1,k-1],d_z[i-1,j-1,k-1]:d_z[i,j,k]] = cur_alpha

    embed_v = EmbedVessels(domain=domain,ms=dim,resolution=resolution)
    embed_v.vessel_grid = vessels
    embed_v.domain = embed_domain
    embed_v.vessel_midline = vessel_midline
    embed_v.vessel_midline_diameter = vessel_midline_diameter
    embed_v.alpha = alpha


    embed_v.vessel_grid = embed_v.vessel_grid.astype('uint8')
    embed_v.vessel_midline = embed_v.vessel_midline.astype('uint8')
    #embed_v.vessel_midline_diameter = embed_v.vessel_midline_diameter .astype('uint8')
    
    if write==True:               
        if output_path is None:
            output_path = surface_dir.replace('surface','embedding')  
        if not os.path.exists(output_path):
            from pathlib import Path
            Path(output_path).mkdir(parents=True, exist_ok=True)
            #os.mkdir(output_path)                 
          
        if s_file is None:              
            s_file = join(output_path,'tiled_surface.nii')
        if v_file is None:
            v_file = join(output_path,'tiled_vessels.nii')
        if d_file is None:
            d_file = join(output_path,'tiled_diameter.nii')
        if c_file is None:
            c_file = join(output_path,'tiled_composite.nii')
        if l_file is None:
            l_file = join(output_path,'tiled_labels.nii')
        if ml_file is None:
            ml_file = join(output_path,'tiled_midline.nii')
        if mld_file is None:
            mld_file = join(output_path,'tiled_midline_diameter.nii')

        embed_v.write_label_grid(l_file)
        embed_v.write_vessel_grid(v_file)
        embed_v.write_midline_grid(ml_file)
        embed_v.write_diameter_grid(d_file)  
    
    # Create blended overlay
    # Blend
    if False:
        surface = embed_v.surface_grid
        vessels = embed_v.vessel_grid
        composite = surface.copy()*0.

        for i in range(vessels.shape[0]):
            Xim = Image.fromarray((vessels[i]).astype('uint8')).convert('RGBA')
            Xim.putalpha(Image.fromarray(((vessels[i])).astype('uint8'),mode='L'))
            Xbg = Image.fromarray(np.uint8(surface[i])).convert('RGBA')
            Xbg.putalpha(255)
            composite[i] = np.asarray(Image.alpha_composite(Xbg, Xim).convert('L'))
            
        composite = composite.astype('uint8')
        embed_v.surface_grid = composite
        
        if write:
            embed_v.write_surface_grid(c_file)
            embed_v.write_surface_grid(c_file.replace('.nii','.tif'),tiff=True,nifti=False)
    
    print('Embed complete')
    return embed_v


def main(graph, domain=None, size=[256,256,256], path=None, 
                patchsize=1,graph_embedding=True,subsample_factor=1,nproc=4,
                lumen=False,lumen_radius_fraction=0.5):
    
    if domain is None:
        domain = arr(graph.edge_spatial_extent())

    dims = arr(size)
    
    fov = domain[:,1] - domain[:,0]
    # Calculate resolution
    res = fov / dims
    
    # Define affine
    tr = np.eye(4)
    tr[0,0], tr[1,1], tr[2,2] = res[0],res[1],res[2]
    tr[0,3] = domain[0,0]
    tr[1,3] = domain[1,0]
    tr[2,3] = domain[2,0]

    # Perform embedding
    vessels = embed(graph=graph,filename=None,dim=dims,offset=arr([0.,0.,0.]),
                    output_path=path,domain=domain,subsample_factor=subsample_factor,patch_factor=patchsize,write=False,
                    rotation_centre=arr([0.,0.,0.]),clip_at_grid_resolution=True,
                    graph_embedding=graph_embedding,psf=True,nproc=nproc,lumen=lumen,lumen_radius_fraction=lumen_radius_fraction)
    vg = vessels.vessel_grid
    
    # Write vessel data
    print(f'Writing subsampled vessels...')
    nifti_image = nib.Nifti1Image(vg, affine=tr)  
    output_file = join(path,"vessel_binary.nii")
    nib.save(nifti_image, output_file)
    print(f'Written {output_file}')
    
if __name__=='__main__':

    # Where to store result (as vessels.nii)
    output_path = r'/INSERT/OUTPUT/PATH'
    # Location of Amira graph file
    filename = r'/FULL/PATH/TO/GRAPH/FILE.am'
    # Spatial extent of embedding ([xmin,xmax],[ymin,ymax],[zmin,zmax])
    domain = np.asarray([[-2000, 2000], [-2000 , 2000], [ -100., 100.]]) 
    # Volume dimensions
    size = [256,256,32]
    # Whether to add a central (dark) lumen to vessels
    lumen = True                       
    # If adding a lumen, what fraction of the vessel radius is taken up by lumen    
    lumen_radius_fraction = 0.5 
    """
    Subsampling of patches, for averaging during reconstruction.
    For accurate binary representation, use subsample_factor = 1
    To simulate partial volume (for example), use subsample_factor>1
    Must be an integer value.
    """
    subsample_factor = 1
    """ 
    Patch division for parallel processing 
    patchsize=1 calculates all data in a single pass; 
    patchsize=4 divides data into 4x4 grid calculated in separate process.
    Other integer values divide the data accordingly.
    Overhead for creating new processes can be high, so best value depends on size of data,
    time and memory available, etc.
    """
    patchsize = 1
    # Number of parallel processes to use
    nproc = 4

    # Load graph
    graph = spatialgraph.SpatialGraph()
    graph.read(filename)    
    
    main(graph,
         path=output_path,
         size=size,
         domain=domain,
         lumen=lumen,
         lumen_radius_fraction=lumen_radius_fraction,
         subsample_factor=subsample_factor,
         patchsize=patchsize,
         nproc=nproc)