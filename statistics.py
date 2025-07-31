# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 09:37:29 2017

@author: simon

Statistical analysis of Amira SpatialGraph file

"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
import nibabel as nib
from tqdm import tqdm, trange # progress bar
import os

from pymira import spatialgraph

class Statistics(object):
    
    def __init__(self, graph, path=None):
        
        self.graph = graph
        self.nodes = None
        self.edges = None
        self.radius_field_name = self.graph.get_radius_field()['name']
                
        self.radii = None
        self.nconn = None
        self.branching_angles = None
        self.node_connections = None
        
        self.edge_intervessel_distance = None
        self.edge_length = None
        self.edge_volume = None
        self.edge_radii = None
        self.edge_euclidean = None      
        self.edge_tortuosity = None                   
        
        #if self.nodes is None:
        #    print('Generating node list...')
        #    self.nodes = self.graph.node_list(path=path)
        #    print('Node list complete')
        #    print('Generating node geometry...')
        #    self.node_geometry(self.nodes)
        #    print('Node geometry complete')
        #if False: #self.edges is None:
        #    print('Generating edge list...')
        #    #self.edges = self.graph.edges_from_node_list(self.nodes)
        #    print('Edge list complete')
            
        print('Generating node geometry...')
        self.node_geometry()
        print('Generating edge geometry...')
        self.edge_geometry()
        print('Edge geometry complete')
        
        
    def coords_in_cube(self,coords,cube):
        
        res = np.zeros(len(coords),dtype='bool')
        for i,c in enumerate(coords):
            if c[0]>=cube[0] and c[0]<=cube[1] and \
               c[1]>=cube[2] and c[1]<=cube[3] and \
               c[2]>=cube[4] and c[2]<=cube[5]:
                res[i] = True
            else:
                res[i] = False
        return res
        
    def summary_image(self,voxel_size=[250,250,250.],parameter='Radii',output_path=''):

        nse = self.graph.edge_spatial_extent()
        dnse = [np.abs(x[1]-x[0]) for x in nse]
        
        nstep = [np.int(np.ceil(dnse[i] / np.float(voxel_size[i]))) for i in range(3)]
        pixel_volume = np.product(voxel_size)
        
        volume = np.zeros(nstep)
        bfrac = np.zeros(nstep)
        radius = np.zeros(nstep)
        length = np.zeros(nstep)
        flow = np.zeros(nstep)
        count = np.zeros(nstep)
        
        x = np.linspace(nse[0][0],nse[0][1],num=nstep[0])
        y = np.linspace(nse[1][0],nse[1][1],num=nstep[1])
        z = np.linspace(nse[2][0],nse[2][1],num=nstep[2])
        
        pbar = tqdm(total=len(self.edges))
                    
        for ei,edge in enumerate(self.edges):
            pbar.update(1)

            radii,lengths,volumes,coords,flows = self.blood_volume([edge],sum_edge=False)
            
            for cInd,coords in enumerate(edge.coordinates):
                xInds = [i1 for i1,xp in enumerate(x) if (xp-coords[0])>=0]
                yInds = [i1 for i1,yp in enumerate(y) if (yp-coords[1])>=0]
                zInds = [i1 for i1,zp in enumerate(z) if (zp-coords[2])>=0]
                try:
                    i = xInds[0]
                    j = yInds[0]
                    k = zInds[0]
                except Exception as e:
                    print(e)
                    import pdb
                    pdb.set_trace()
                    
                #print i,j,k,len(volumes),coords.shape
                
                if cInd<volumes.shape[0]:
                    volume[i,j,k] += volumes[cInd]
                    if flows is not None:
                        flow[i,j,k] += flows[cInd]
                        
                    radius[i,j,k] += radii[cInd]
                    length[i,j,k] += lengths[cInd]
                    count[i,j,k] += 1
                    
        pbar.close()
        
        # Take averages
        radius = radius / count
        radius[~np.isfinite(radius)] = 0.
        length = length / count
        length[~np.isfinite(length)] = 0.
        bfrac = volume / pixel_volume

        img = nib.Nifti1Image(bfrac,affine=np.eye(4))
        ofile = output_path+'blood_volume.nii'
        nib.save(img,ofile)
    
        img = nib.Nifti1Image(radius,affine=np.eye(4))
        ofile = output_path+'radius.nii'
        #ofile = 'C:\\Users\\simon\\Dropbox\\radius.nii'
        nib.save(img,ofile)
        
        img = nib.Nifti1Image(length,affine=np.eye(4))
        #ofile = 'C:\\Users\\simon\\Dropbox\\length.nii'
        ofile = output_path+'length.nii'
        nib.save(img,ofile)
        
        img = nib.Nifti1Image(flow,affine=np.eye(4))
        #ofile = 'C:\\Users\\simon\\Dropbox\\flow.nii'
        ofile = output_path+'flow.nii'
        nib.save(img,ofile)
        
    def edge_geometry(self): #,edges):
        
        #pbar = tqdm(total=len(edges))

        #edgeInds = self.graph.edgepoint_edge_indices()
        #points = self.graph.get_data('EdgePointCoordinates')
        #radii = self.graph.get_data(self.radius_field_name)
        
        graph = self.graph
        nodecoords = graph.get_data('VertexCoordinates')
        edgeconn = graph.get_data('EdgeConnectivity')
        edgepoints = graph.get_data('EdgePointCoordinates')
        nedgepoints = graph.get_data('NumEdgePoints')
        radii = self.graph.get_data(self.radius_field_name)
        
        nedges = edgeconn.shape[0]
        
        edgeInds = np.zeros(edgepoints.shape[0],dtype='int')
        for edge_ind in range(nedges):
            nep = nedgepoints[edge_ind]
            x0 = np.sum(nedgepoints[:edge_ind])
            x1 = x0 + nep
            edgeInds[x0:x1] = edge_ind
 
        self.edge_intervessel_distance = np.zeros(nedges)   
        self.edge_length = np.zeros(nedges)
        self.edge_radii = np.zeros(nedges)
        self.edge_volume = np.zeros(nedges)
        self.edge_euclidean = np.zeros(nedges)      
        self.edge_tortuosity = np.zeros(nedges)   

        for edge_ind in trange(nedges):
            #try:
            if True:
                nep = nedgepoints[edge_ind]
                x0 = np.sum(nedgepoints[:edge_ind])
                x1 = x0 + nep
                pts = edgepoints[x0:x1]
                rads = radii[x0:x1]
            
                #pts = edge.coordinates
                #rads = edge.get_scalar(self.radius_field_name)
                
                # Define search range
                rng = [np.min(pts,axis=0),np.max(pts,axis=0)]
                #if rng[0]<0.:
                #   rng[0] *= 1.2*np.max(rads)
                #else:
                #   rng[0] *= 0.8*np.max(rads)
                #if rng[1]<0.:
                #   rng[1] *= 0.8*np.max(rads)
                #else:
                #   rng[1] *= 1.2*np.max(rads)
                   
                lim = 10. #100. #um
                # Get nearby points
                inds = [(edgepoints[:,0]>rng[0][0]-lim) & (edgepoints[:,0]<rng[1][0]+lim) & (edgepoints[:,1]>rng[0][1]-lim) & (edgepoints[:,1]<rng[1][1]+lim) & (edgepoints[:,2]>rng[0][2]-lim) & (edgepoints[:,2]<rng[1][2]+lim) & (edgeInds!=edge_ind)]
                #inds = []
                if len(inds)>0:
                    curPoints = edgepoints[inds[0]]
                    curRadii = radii[inds[0]]
                else:
                    curPoints,curRadii = [],[]

                dist = np.zeros(pts.shape[0]-1)

                length = np.sum([np.linalg.norm(pts[i]-pts[i-1]) for i,x in enumerate(pts[1:])])
                volume = np.sum([np.linalg.norm(pts[i]-pts[i-1])*np.square(rads[i]) for i,x in enumerate(pts[1:])])
                
                for i in range(pts.shape[0]-1):
                    if len(curPoints)>0:
                        dist[i] = np.min([np.linalg.norm(pts[i]-p)-rads[i]-curRadii[j] for j,p in enumerate(curPoints)])
                      
                #breakpoint()  
                if len(curPoints)>0:
                    self.edge_intervessel_distance[edge_ind] = np.max(dist)
                else:
                    self.edge_intervessel_distance[edge_ind] = -1.
                self.edge_length[edge_ind] = length
                self.edge_volume[edge_ind] = volume
                self.edge_radii[edge_ind] = np.mean(rads)
                self.edge_euclidean[edge_ind] = np.linalg.norm(pts[-1]-pts[0])

                #breakpoint()
                self.edge_tortuosity[edge_ind] = self.edge_euclidean[edge_ind] / length
                
            #except Exception as e:
            #    print('Error, edge {}: {}'.format(edge,e))

            
    def _branching_angle(self,vec1,vec2,acute=False):

        if np.linalg.norm(vec1)*np.linalg.norm(vec2)==0.:
            return 0.
        rad = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        deg = np.rad2deg(rad)
        
        if acute:
            deg = deg % 90.
        
        return deg
        
    def volume(self,coords):
        from scipy.spatial import ConvexHull
        return ConvexHull(coords).volume
            
    def node_geometry(self): #,nodes):
    
        graph = self.graph
        nodecoords = graph.get_data('VertexCoordinates')
        edgeconn = graph.get_data('EdgeConnectivity')
        edgepoints = graph.get_data('EdgePointCoordinates')
        nedgepoints = graph.get_data('NumEdgePoints')
        radii = self.graph.get_data(self.radius_field_name)
        
        nnodes = nodecoords.shape[0]
        
        self.branching_angles = []
        self.node_connections = []

        for node_ind in trange(nnodes):
            sind = np.where((edgeconn[:,0]==node_ind) | (edgeconn[:,1]==node_ind))

            if len(sind[0])>0:
                if len(sind[0])>1:
                    self.node_connections.append(len(sind[0]))
                    
                for edge_ind in sind[0]:
                
                    # Edge direction
                    if edgeconn[edge_ind,0]==node_ind:
                        direction = 1
                    else:
                        direction = -1
                        
                    nep = nedgepoints[edge_ind]
                    x0 = np.sum(nedgepoints[:edge_ind])
                    x1 = x0 + nep
                    pts = edgepoints[x0:x1]
                    
                    if direction==-1:
                        pts = pts[::-1]
                        
                    for edge_ind2 in sind[0]:
                        if edge_ind!=edge_ind2:
                            
                            # Edge direction
                            if edgeconn[edge_ind2,0]==node_ind:
                                direction2 = 1
                            else:
                                direction2 = -1
                                
                            nep2 = nedgepoints[edge_ind2]
                            x02 = np.sum(nedgepoints[:edge_ind2])
                            x12 = x02 + nep2
                            pts2 = edgepoints[x02:x12]
                            
                            if direction2==-1:
                                pts2 = pts2[::-1]

                            veci = pts[0]-pts[1]
                            vecj = pts2[0]-pts2[1]
                            if not all(x==y for x,y in zip(veci,vecj)):
                                self.branching_angles.append(self._branching_angle(veci,vecj))
        
    def histogram(self,v,range=None,xlabel=None,nbins=50,show=False):
    
        plt.clf()
        
        # the histogram of the data
        n, bins, patches = plt.hist(v, nbins, range=range, density=1, facecolor='green', alpha=0.75)
        
        # add a 'best fit' line
        #y = mlab.normpdf( bins, mu, sigma)
        #l = plt.plot(bins, y, 'r--', linewidth=1)
        
        if xlabel is not None:
            plt.xlabel(xlabel)
        #plt.ylabel('Probability')
        #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
        #plt.axis([40, 160, 0, 0.03])
        #plt.grid(True)
        
        if show:
            plt.show()
        
    def boxplot(self,v):
        
        plt.boxplot(v)
        
    def blood_volume(self,edges,sum_edge=True):
        
        nedge = len(edges)
        
        radii = [None] * nedge #np.zeros(self.graph.nedgepoint)
        lengths = [None] * nedge #np.zeros(self.graph.nedgepoint)
        volumes = [None] * nedge #np.zeros(self.graph.nedgepoint)
        coords = [None] * nedge #np.zeros((self.graph.nedgepoint,3))
        if 'Flow' in edges[0].scalarNames:
            flow = [None] * nedge #np.zeros(self.graph.nedgepoint)
        else:
            flow = None
             
        pbar = tqdm(total=len(edges))
        #import pdb
        #pdb.set_trace()
        #pcount = 0
        for ei,edge in enumerate(edges):
            pbar.update(1)
            npoints = edge.npoints
            rad = edge.get_scalar(self.radius_field_name)
            radii[ei] = rad
            lengths[ei] = np.append(edge.length,0)
            curVol = np.pi*np.square(rad[0:-1])*edge.length
            volumes[ei] = curVol
            if flow is not None:
                curFlow = edge.get_scalar('Flow')
                flow[ei] = curFlow
            
            if sum_edge:
                volumes[ei] = np.sum(volumes[ei])
                lengths[ei] = np.sum(lengths[ei])
            coords[ei] = edge.coordinates
            
            #pcount += npoints
        pbar.close()
            
        return radii,lengths,volumes,coords,flow
        
    def do_stats(self,path=None,prefix='',pix_scale=None):
        
        #print('Calculating statistics...')
        #print('Estimating network parameters...')
        #import pdb
        #pdb.set_trace()
        #radii,lengths,volumes,coords,flow = self.blood_volume(self.edges)
        #print('Finished estimating network parameters...')

        print('Calculating stats...')
        nconn = np.asarray(self.node_connections)
        ba = np.asarray(self.branching_angles)
        ba = ba[np.isfinite(ba)]
        #breakpoint()
        ivd = self.edge_intervessel_distance[self.edge_intervessel_distance>0.]
        
        try:
            self.histogram(ba,range=[0,180],nbins=50,xlabel='Vessel branching angle (deg)')
            if path is not None:
                plotfile = os.path.join(path,prefix+'branching_angle_histogram.png')
                plt.savefig(plotfile) #,transparent=True)
        except Exception as e:
            print(e)

        try:
            diameters = self.edge_radii*2
            if pix_scale is not None:
                diameters *= pix_scale
            self.histogram(diameters,range=[0,np.max(diameters)],nbins=30,xlabel='Vessel diameter (um)')
            if path is not None:
                plotfile = os.path.join(path,prefix+'diameter_histogram.png')
                plt.savefig(plotfile) #,transparent=True)
        except Exception as e:
            print(e)
            
        try:
            #lengthsFlat = [item for sublist in lengths for item in sublist]
            lengths = self.edge_length
            if pix_scale is not None:
                lengths *= pix_scale
            self.histogram(lengths,range=[0,np.max(lengths)],nbins=50,xlabel='Vessel length (um)')
            if path is not None:
                plotfile = os.path.join(path,prefix+'vessel_length_histogram.png')
                plt.savefig(plotfile) #,transparent=True)
        except Exception as e:
            print(e)
            
        try:
            volumes = self.edge_volume
            if pix_scale is not None:
                lengths *= np.power(pix_scale,3)
            self.histogram(volumes,range=[0,np.max(volumes)],nbins=50,xlabel='Vessel volume (um3)')
            if path is not None:
                plotfile = os.path.join(path,prefix+'vessel_volume_histogram.png')
                plt.savefig(plotfile) #,transparent=True)
        except Exception as e:
            print(e)
            
        try:                
            self.histogram(self.edge_tortuosity,range=[0,np.max(self.edge_tortuosity)],nbins=50,xlabel='Vessel tortuosity')
            if path is not None:
                plotfile = os.path.join(path,prefix+'vessel_tortuosity_histogram.png')
                plt.savefig(plotfile) #,transparent=True)
        except Exception as e:
            print(e)
            
        try: 
            if pix_scale is not None:
                ivd *= pix_scale       
            self.histogram(ivd,nbins=50,range=[0,np.max(ivd)],xlabel='Intervessel distance (um)') # range=[0,100]
            if path is not None:
                plotfile = os.path.join(path,prefix+'intervessel_distance_histogram.png')
                plt.savefig(plotfile) #,transparent=True)
        except Exception as e:
            print(e)                

        #blood_volume = np.sum(volumes)
        #coords = self.graph.get_data('VertexCoordinates')
        #try:
        #    total_volume = self.volume(coords)
        #except Exception as e:
        #    print(e)
        #    total_volume = -1.
        #blood_fraction = blood_volume / total_volume
        #print(('TUMOUR VOLUME (um3): {}'.format(total_volume)))
        #print(('BLOOD VOLUME (um3): {}'.format(blood_volume)))
        #print(('BLOOD VOLUME FRACTION: {}'.format(blood_fraction)))
        
        if path is not None:
            #with open(os.path.join(path,'volume.txt'),'w') as fo:
            #    fo.write('TUMOUR VOLUME (um3) \t{}\n'.format(total_volume))
            #    fo.write('BLOOD VOLUME (um3) \t{}\n'.format(blood_volume))
            #    fo.write('BLOOD VOLUME FRACTION \t{}\n'.format(blood_fraction))
                
            filename = prefix+'stats.txt'
            with open(os.path.join(path,filename),'w') as fo:
                
                # Write header
                hdr = ['PARAM','Mean','SD','median','min','max']
                hdr = '\t'.join(hdr)+'\n'
                fo.write(hdr)
                
                params = [self.edge_radii,self.edge_length,ivd,nconn,ba,self.edge_volume]
                paramNames = ['Radius (um)','Vessel length (um)','Intervessel distance( um)','Number of connections','Branching angle (deg)','Vessel volume (um3)']
                
                for v,n in zip(params,paramNames):
                    print(n)
                    try:
                        cur = [n,np.mean(v),np.std(v),np.median(v),np.min(v),np.max(v)]
                    except:
                        cur = [n,-1.,-1.,-1.,-1.,-1.]
                    cur = ['{}'.format(c) for c in cur]
                    cur = '\t'.join(cur)+'\n'
                    fo.write(cur)
                    
            with open(os.path.join(path,prefix+'radii.p'),'wb') as fo:
                pickle.dump(self.edge_radii,fo)
            with open(os.path.join(path,prefix+'branching_angle.p'),'wb') as fo:
                pickle.dump(ba,fo)
            with open(os.path.join(path,prefix+'vessel_length.p'),'wb') as fo:
                pickle.dump(self.edge_length,fo)
            with open(os.path.join(path,prefix+'nconn.p'),'wb') as fo:
                pickle.dump(nconn,fo)
            with open(os.path.join(path,prefix+'vessel_volume.p'),'wb') as fo:
                pickle.dump(self.edge_volume,fo)
            with open(os.path.join(path,prefix+'intervessel_distance.p'),'wb') as fo:
                pickle.dump(ivd,fo)

    
