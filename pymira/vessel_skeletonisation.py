import nibabel as nib
import numpy as np
from os.path import join, isfile
from os import listdir
from PIL import Image
from PIL.TiffTags import TAGS
import os
from skimage import io
from scipy.ndimage import zoom
arr = np.asarray
from matplotlib import pyplot as plt
import re
import skimage
import cv2
import scipy
from skimage.filters import meijering, sato, frangi, hessian
from pymira import spatialgraph

"""
3D vessel skeletonisation from binary images
"""

def calculate_vesselness(im):

    print('Calculating vesselness')
    sigmas = np.arange(2,10,1)
    nsigma = len(sigmas)
    
    if False:
        vesselness = np.zeros([nsigma,im.shape[0],im.shape[1],im.shape[2]])
        for i,sigma in enumerate(sigmas):
            print('Sigma={}, {} of {}'.format(sigma,i+1,nsigma))
            vesselness[i] = sato(im,sigmas=[sigma],mode='reflect')
    
        scale = np.argmax(vesselness,axis=0)
        vesselness_ms = np.max(vesselness,axis=0)
        enface_ms = np.max(vesselness_ms,axis=-1)
    else:
        # Downsample
        vesselness = sato(im,sigmas=sigmas,mode='reflect',black_ridges=False)
    
    return vesselness

def check_resolution():
    path_root = r'C:\Users\simon\SWS Dropbox\Simon Walker-Samuel\retinas - GIULIA'
    groups = ['control','diabetic','KO control','KO diabetic']
    #groups = ['KO diabetic']

    for group in groups:
        path = join(path_root,group)
        opath = os.path.join(path,'screengrabs')
        if not os.path.exists(opath):
            os.mkdir(opath)
        gpath = os.path.join(opath,'graph')
        if not os.path.exists(gpath):
            os.mkdir(gpath)

        files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.tif')]

        new_dims = arr([64,512,512])
        new_resolution = [None,1.,1.]
        channelInd = 1
        isotropic = True

        for f in files:

            #try:
            if True:

                data = io.imread(join(path,f))[:,:,:,channelInd]
                #print(data.shape,data[...,0].max(),data[...,1].max(),data[...,2].max())
                img = Image.open(join(path,f))
                
                meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
                descr = meta_dict['ImageDescription'][0].split('\n')
                spacing = float([x for x in descr if 'spacing' in x][0].replace('spacing=',''))
                xres,yres = meta_dict['XResolution'][0][0]/1e6,meta_dict['YResolution'][0][0]/1e6
                #breakpoint()
                #xres,yres = 1.76,1.76
                data_resolution = arr([spacing,xres,yres])
                #breakpoint()
                
                print(f'Data resolution ({group}): {data_resolution}um, size: {data.shape}')

def identify_graphs(graph):

    # Find all connected nodes
    gc = graph.get_node_count()
    sends = np.where(gc<=1)
    nodes_visited = []
    node_graph_index = np.zeros(graph.nnode,dtype='int') - 1
    #node_graph_contains_root = np.zeros(graph.nnode,dtype='bool')
    graph_index_count = 0
    for send in sends[0]:
        if node_graph_index[send]==-1:
            node_graph_index[send] = graph_index_count
            #node_graph_contains_root[send] = np.any(frozenNode[send])
            edges = graph.get_edges_containing_node(send)
            cnodes,cedges = graph.get_all_connections_to_node(send)

            if len(cnodes)>0:                            
                node_graph_index[cnodes] = graph_index_count
                graph_index_count += 1
                #node_graph_contains_root[cnodes] = np.any(frozenNode[cnodes])

        if np.all(node_graph_index>=0):
            break
    return node_graph_index
    
def skeletonize(image,resolution=arr([1.,1.,1.,]),offset=arr([0.,0.,0.])):

    from skimage.morphology import skeletonize as skel
    skeleton = skel(image)

    #plt.imsave(join(opath,f.replace('.tif','_mask.png')),(blur*255).astype('int'))
    #plt.imsave(join(opath,f.replace('.tif','_skeleton.png')),(skeleton*255/np.max(skeleton)).astype('int'))
    #plt.imsave(join(opath,f.replace('.tif','_enface.png')),(enface).astype('int'))
    
    import scipy
    dist = scipy.ndimage.distance_transform_edt(image) #np.abs(blur-1.))
    rad = skeleton * dist
    
    from pymira import midline_to_graph
    # 2D to 3D
    #skeleton3d = np.expand_dims(skeleton,axis=-1)
    skeleton3d = skeleton.copy()
    m2g = midline_to_graph.Midline2Graph(skeleton3d.astype('float'))
    edgepointsInt, edgeconn = m2g.convert() 

    nodes = edgepointsInt.copy().astype('float')   
    edgepoints = nodes[edgeconn].reshape([2*edgeconn.shape[0],3])
    nedgepoints = np.zeros(edgeconn.shape[0],dtype='int') + 2
    
    radii_node = dist[edgepointsInt[:,0],edgepointsInt[:,1],edgepointsInt[:,2]]
    
    radii = radii_node[edgeconn].reshape([2*edgeconn.shape[0]])
    if resolution is not None:
        radii *= resolution[0]
        for i in range(3):
            nodes[:,i] = (nodes[:,i]*resolution[i]) + offset[i]
            edgepoints[:,i] = (edgepoints[:,i]*resolution[i]) + offset[i]
    
    graph = spatialgraph.SpatialGraph(initialise=True,scalars=['Radius'])      
    graph.set_definition_size('VERTEX',nodes.shape[0])
    graph.set_definition_size('EDGE',edgeconn.shape[0])
    graph.set_definition_size('POINT',edgepoints.shape[0])
    graph.set_data(nodes,name='VertexCoordinates')
    graph.set_data(edgeconn,name='EdgeConnectivity')
    graph.set_data(nedgepoints,name='NumEdgePoints')
    graph.set_data(edgepoints,name='EdgePointCoordinates')
    graph.set_data(radii,name='Radius')
    graph.set_graph_sizes()

    #ed = spatialgraph.Editor()
    #graph = ed.remove_intermediate_nodes(graph)
    
    return graph
    
    
    
    edgepoints = edgepoints[:,:2]
    
    # Remove self-connected points
    edgeconns = arr([x for x in edgeconn if x[0]!=x[1]])
    
    # Convert to node & edge format
    unq,count = np.unique(edgeconn,return_counts=True)
    all_nodes = np.linspace(0,edgepoints.shape[0]-1,edgepoints.shape[0],dtype='int')
    node_count = np.zeros(edgepoints.shape[0],dtype='int') 
    node_count[np.in1d(all_nodes,unq)] = count
    count = node_count
        
    nodes_edgeinds = np.where(count!=2)[0]
    nodes = edgepoints[nodes_edgeinds]
    nnode = nodes.shape[0]
    
    edges = []
    edgepoint_inds = []

    edge_visited = np.zeros(edgeconn.shape[0],dtype='bool')
    for i in range(nnode):
        # Find which edges the current node is connected to
        cur_conns0 = np.where( ((edgeconn[:,0]==nodes_edgeinds[i]) | (edgeconn[:,1]==nodes_edgeinds[i])) & (edge_visited==False))
        #cur_conns = cur_conns[0][~np.in1d(cur_conns[0],cur_edges[j])]
        cur_conns0 = cur_conns0[0]
        nconn = len(cur_conns0)
        
        if nconn>0:
            term = np.zeros(nconn,dtype='bool')
            edge_invalid = np.zeros(nconn,dtype='bool')
        
            #edge_visited[cur_conns[0]] = True
            cur_edges = [[x] for x in cur_conns0]
            #cur_edgepoint = [edgepoints[x] for x in cur_edges]
            
            # Initialise the edgepoint lists
            cur_edgepoint_ind = [[nodes_edgeinds[i]] for _ in cur_conns0]
            # Add in the first connection
            for j in range(nconn):
                if edgeconn[cur_conns0[j],0]!=cur_edgepoint_ind[j][-1] and edgeconn[cur_conns0[j],1]==cur_edgepoint_ind[j][-1]:
                    cur_edgepoint_ind[j].append(edgeconn[cur_conns0[j],0])
                elif edgeconn[cur_conns0[j],1]!=cur_edgepoint_ind[j][-1] and edgeconn[cur_conns0[j],0]==cur_edgepoint_ind[j][-1]:
                    cur_edgepoint_ind[j].append(edgeconn[cur_conns0[j],1])
                else:
                    breakpoint()
            
                #cur_edgepoint_ind[j].append( [edgeconn[x,0] if edgeconn[x,0]!=cur_edgepoint_ind[j][-1] else edgeconn[x,1] for x in cur_conns0][0] )
                if count[cur_edgepoint_ind[j][-1]]!=2:
                    term[j] = True
 
            # Walk edges until another node is encountered
            itcount = 0
            while True:
                for j in range(nconn):
                    if term[j]==False:
                        cur_conns = np.where( ((edgeconn[:,0]==cur_edgepoint_ind[j][-1]) | (edgeconn[:,1]==cur_edgepoint_ind[j][-1])) & (edge_visited==False) )
                        cur_conns = cur_conns[0][~np.in1d(cur_conns[0],cur_edges[j])]
                        if len(cur_conns)==1:
                            #breakpoint()
                            if edgeconn[cur_conns,0]!=cur_edgepoint_ind[j][-1] and edgeconn[cur_conns,1]==cur_edgepoint_ind[j][-1]:
                                cur_edgepoint_ind[j].append(edgeconn[cur_conns,0][0])
                            elif edgeconn[cur_conns,1]!=cur_edgepoint_ind[j][-1] and edgeconn[cur_conns,0]==cur_edgepoint_ind[j][-1]:
                                cur_edgepoint_ind[j].append(edgeconn[cur_conns,1][0])
                            else:
                                breakpoint()
                            cur_edges[j].append(cur_conns[0])
                            edge_visited[cur_conns[0]] = True
                            
                            if count[cur_edgepoint_ind[j][-1]]!=2:
                                term[j] = True
                        elif len(cur_conns)>1:
                            #raise
                            breakpoint()
                        else:
                            edge_invalid[j] = True
                            term[j] = True
                            #edge_visited[arr(cur_edges[j])] = False
                            #breakpoint()

                if np.all(term):
                    break
                elif itcount>1000:
                    print(f'Itcount>1000!')
                    breakpoint()
                else:
                    #cur_edgepoint_ind = next_edgepoint_ind
                    pass
                itcount += 1
                
            #breakpoint() 
            for j,e in enumerate(cur_edges): 
                if not edge_invalid[j]:        
                    edges.append(e)   
            for j,e in enumerate(cur_edgepoint_ind): 
                if not edge_invalid[j]:
                    edgepoint_inds.append(e) 

    unvisited_edges = edgeconn[edge_visited==False]
    #fig = plt.figure()
    
    for i,e in enumerate(edgeconn):
        if edge_visited[i]==False:
            ec0 = edgepoints[e[0]]
            ec1 = edgepoints[e[1]]
            #plt.plot([ec0[0],ec1[0]],[ec0[1],ec1[1]],c='r')
            
    conns = np.zeros([len(edgepoint_inds),2],dtype='int')
    for i,e in enumerate(edgepoint_inds):
        ec = edgepoints[e]
        #plt.plot(ec[ :,0],ec[:,1],c='b')
        try:
            conns[i,0] = np.where(nodes_edgeinds==e[0])[0]
            conns[i,1] = np.where(nodes_edgeinds==e[-1])[0]
        except Exception as err:
            print(err)
            breakpoint()

    nedgepoints = arr([len(x) for x in edgepoint_inds])
    edgepoint_inds = [x for y in edgepoint_inds for x in y]
    edgepoints = edgepoints[edgepoint_inds]
    
    #plt.show()

    radii = dist[edgepoints[:,0],edgepoints[:,1]]
    if resolution is not None:
        radii *= resolution[0]
    
    nodes3d = np.zeros([nodes.shape[0],3])
    nodes3d[:,:2] = nodes
    if resolution is not None:
        nodes3d[:,0] *= resolution[0]
        nodes3d[:,1] *= resolution[1]
        #nodes3d[:,2] *= resolution[1]
    edgepoints3d = np.zeros([edgepoints.shape[0],3]) 
    edgepoints3d[:,:2] = edgepoints
    if resolution is not None:
        edgepoints3d[:,0] *= resolution[0]
        edgepoints3d[:,1] *= resolution[1]
        #edgepoints3d[:,2] *= resolution[1]
    
    graph = spatialgraph.SpatialGraph(initialise=True,scalars=['Radius'])      
    graph.set_definition_size('VERTEX',nodes.shape[0])
    graph.set_definition_size('EDGE',conns.shape[0])
    graph.set_definition_size('POINT',edgepoints.shape[0])
    graph.set_data(nodes3d,name='VertexCoordinates')
    graph.set_data(conns,name='EdgeConnectivity')
    graph.set_data(nedgepoints,name='NumEdgePoints')
    graph.set_data(edgepoints3d,name='EdgePointCoordinates')
    graph.set_data(radii,name='Radius')
    graph.set_graph_sizes()

    graphNodeIndex, graph_size = graph.identify_graphs(progBar=True)
    ed = spatialgraph.Editor()
    graph = ed.remove_graphs_smaller_than(graph,20) #identify_graphs(graph)
    graphNodeIndex, graph_size2 = graph.identify_graphs(progBar=True)
    
    for e in range(graph.nedge):
       edge = graph.get_edge(e)
       if edge.npoints>2:
           #radii[edge.i0:edge.i1] = np.mean(radii[edge.i0:edge.i1])
           pass
    graph.set_data(radii,name='Radius')
    #graph.plot(fixed_radius=0.2,cmap_range=[0.,.1],cmap='gray')
    return graph
    
def extract_segment():

    import rasterio
 
    path_root = 'C:\\Users\\simon\\Desktop\\lung-files'
    files = ['Vessels_and_airways.tif']
    ftype = '.tif'
    resolution = arr([200.21,200.076,200.])
    mincoord = arr([75.1016,75.0391,75.])
    
    path = join(path_root,'')
    opath = os.path.join(path,'screengrabs')
    if not os.path.exists(opath):
        os.mkdir(opath)
    gpath = os.path.join(opath,'graph')
    if not os.path.exists(gpath):
        os.mkdir(gpath)

    f = files[0]
    data = io.imread(join(path,f))
    with rasterio.open(join(path,f)) as src:
        tr = src.transform
    
    airways = data==2
    airways = airways.astype('uint8')
    graph = skeletonize(airways,resolution=resolution) #,offset=mincoord)
    
    breakpoint()
    graph.write(join(path,'airways_skel.am'))

def main():

    path_root = r'/mnt/data2/OCT_nii_cube/cycleGAN_seg'
    path_root = r'Z:\OCT_nii_cube\cycleGAN_seg'
    path_root = r'C:\Users\simon\SWS Dropbox\Simon Walker-Samuel\cycleGAN_seg'
    groups = ['']
    ftype = '.png'
    files = ['PDR_case_9_Angio (12mmx12mm)_2-15-2019_14-26-22_OS_sn2159_FlowCube_z_enface_fake.png']
    #files = ['1.2.276.0.75.2.2.44.79497678600052.20180702120449820.210035586_SuperficialAngioEnface__BSCR_P01_fake.png']
    make_binary = False
    
    #path_root = r'C:\Users\simon\Desktop\turing2'
    #groups = ['']
    #ftype = '.png'
    #files = ['PDR_case_9_Angio (12mmx12mm)_2-15-2019_14-26-22_OS_sn2159_FlowCube_z_enface_fake.png']
    #files = ['im_0.05_0.06111111111111111.png']
    #files = ['im_0.04_0.06125.png']
    #make_binary = True
    
    path_root = 'C:\\Users\\simon\\Desktop\\lung-files'
    files = ['Vessels_and_airways.tif']
    ftype = '.tif'
    breakpoint()

    #path_root = 'PATH' 
    #groups = ['']
    #ftype = '.png'
    #files = ['FILE.png']
    #make_binary = True

    for group in groups:
        path = join(path_root,group)
        opath = os.path.join(path,'screengrabs')
        if not os.path.exists(opath):
            os.mkdir(opath)
        gpath = os.path.join(opath,'graph')
        if not os.path.exists(gpath):
            os.mkdir(gpath)

        #files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(ftype)]

        new_dims = arr([64,512,512])
        new_resolution = [None,1.,1.]
        channelInd = 1
        isotropic = True

        for f in files:

            #try:
            if True:

                data = io.imread(join(path,f))[:,:,channelInd]
                data = skimage.transform.resize(data, (500, 500)) * 255
                #breakpoint()
                if make_binary:
                    binary_threshold = 128
                    data[data>=binary_threshold] = 255
                    data[data<binary_threshold] = 0
                
                #print(data.shape,data[...,0].max(),data[...,1].max(),data[...,2].max())
                img = Image.open(join(path,f))
                #breakpoint()
                
                #meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
                #descr = meta_dict['ImageDescription'][0].split('\n')
                #spacing = float([x for x in descr if 'spacing' in x][0].replace('spacing=',''))
                #xres,yres = meta_dict['XResolution'][0][0]/1e6,meta_dict['YResolution'][0][0]/1e6
                #if data.shape[1]==1024:
                xres,yres = 0.568,0.568
                #else:
                #    xres,yres = 1.14,1.14
                data_resolution = arr([xres,yres])
                #breakpoint()
                
                print(f'Data resolution ({group}): {data_resolution}um')
                
                data_dims = arr(data.shape)
                
                if isotropic:
                    new_resolution = arr([1.,1.]) #arr([np.min(data_resolution)]*3)
                
                #print('Creating isotropic resolution...')
                fr = data_resolution/new_resolution
                #data = zoom(data,fr)
                resolution = data_resolution
                
                if False:
                    enface = np.max(data,axis=0)
                    av = np.mean(enface)
                    
                    import scipy
                    bg = scipy.ndimage.gaussian_filter(enface.astype('float'), (200,200))
                    plt.imsave(join(opath,f.replace('.tif','_bg.png')),(bg).astype('int'))
                    
                    #breakpoint()
                    dif = enface - bg
                    dif = (dif - np.min(dif))*255/(np.max(dif)-np.min(dif))
                    plt.imsave(join(opath,f.replace('.tif','_enface_norm.png')),(dif).astype('int'))
                    
                    thr = 40 #- 0.2*av
                    mask = dif.copy()
                    mask[dif<thr] = 0
                    mask[dif>=thr] = 255
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
             
                    blur = scipy.ndimage.gaussian_filter(mask.astype('float'), (3,3))
                    blur[blur>=thr] = 255
                    blur[blur<thr] = 0
                    blur = scipy.ndimage.binary_erosion(blur, structure=np.ones((3,3))).astype(blur.dtype)
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
                else:
                    blur = data>40
                    
                blur = blur / np.max(blur)
                graph = skeletonize(blur)
                
                if True:
                    from vessel_sim.retina_cco.retina_create_surface import create_surface
                    from vessel_sim.retina_cco.retina_project_graph_to_surface import project_to_surface
                    surface_path = join(path_root,'surface')
                    if not os.path.exists(surface_path):
                        os.mkdir(surface_path)
                    ofile = join(surface_path,'retina_surface.ply')
                    plot_file = ofile.replace('.ply','_profile.png')
                    
                    radii = graph.get_data('Radius')
                    nodes = graph.get_data('VertexCoordinates')
                    edgepoints =  graph.get_data('EdgePointCoordinates')
                    pixel_size = [20.,20.,20.]
                    radii *= pixel_size[0]
                    nodes[:,0] *= pixel_size[0]
                    nodes[:,1] *= pixel_size[1]
                    nodes[:,2] *= pixel_size[2]
                    edgepoints[:,0] *= pixel_size[0]
                    edgepoints[:,1] *= pixel_size[1]
                    edgepoints[:,2] *= pixel_size[2]
                    
                    # Swap x-y
                    nodes_rot = nodes.copy()
                    nodes_rot[:,0],nodes_rot[:,1] = nodes[:,1],nodes[:,0]
                    edgepoints_rot = edgepoints.copy()
                    edgepoints_rot[:,0],edgepoints_rot[:,1] = edgepoints[:,1],edgepoints[:,0]
                    
                    graph.set_data(nodes_rot,name='VertexCoordinates')
                    graph.set_data(edgepoints_rot,name='EdgePointCoordinates')
                    graph.set_data(radii,name='Radius')
                    
                    from vessel_sim.retina_cco.retina_lsystem import Eye
                    eye = Eye()
                    centre = edgepoints[np.argmax(radii)]
                    eye.optic_disc_centre = centre
                    eye.macula_centre = eye.macula_centre + centre
                    if True:
                        create_surface(path=None,ofile=ofile,plot=True,plot_file=plot_file,vessel_depth=1000.,add_simplex_noise=False,eye=eye,project=True)
                    
                    vfile = join(surface_path,'retina_surface_vessels.ply')
                    ofile = '' #os.path.join(gpath,f.replace(ftype,'_proj.am'))
                    graphProj = project_to_surface(graph=graph,eye=eye,vfile=vfile,plot=False,interpolate=False,ofile=ofile,write_mesh=False,iterp_resolution=10.,filter=False)

                    mesh = o3d.io.read_triangle_mesh(vfile)
                    
                    eye.domain[2] = [-100.,100.]
                    graphProj.plot(domain=eye.domain,cmap='gray',cmap_range=[-50,50],radius_scale=0.5,additional_meshes=mesh)
                    breakpoint()
                else:  
                    radii = graph.get_data('Radius')
                    nodes = graph.get_data('VertexCoordinates')
                    edgepoints =  graph.get_data('EdgePointCoordinates')
                    centre = edgepoints[np.argmax(radii)]
                    dist = np.linalg.norm(edgepoints[:,:2]-centre[:2],axis=1)
                    edgepoints[:,2] = np.square(dist/40.)
                    dist = np.linalg.norm(nodes[:,:2]-centre[:2],axis=1)
                    nodes[:,2] = np.square(dist/40.)
                    graph.set_data(nodes,name='VertexCoordinates')
                    graph.set_data(edgepoints,name='EdgePointCoordinates')

                ofile = os.path.join(gpath,f.replace(ftype,'.am'))
                graph.write(ofile)
                print(f'Graph saved to {ofile}')
                
                if False:
                    show,block = False,False
                    vis = graph.plot_graph(show=show,block=block)
                    if show==False:
                        vis.screen_grab(join(gpath,f.replace('.tif','_screen_grab.png')))
                        vis.destroy_window()

                if False: #for ch in range(3):
                    tr = np.eye(4)
                    tr[0,0] = new_resolution[0]
                    tr[1,1] = new_resolution[1]
                    #tr[2,2] = new_resolution[2]
                    array_img = nib.Nifti1Image(data, tr)
                    fname = os.path.splitext(files[0])[0]
                    ofile = join(opath,f+'_{}_.nii'.format(channelInd))
                    nib.save(array_img, ofile)
                    print('Tiff2nifti. Saved to {}'.format(ofile))
            #except Exception as e:
            #    print(e)
            
if __name__=='__main__':
    main()
    #check_resolution()
    #extract_segment()
