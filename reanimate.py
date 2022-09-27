from pymira import spatialgraph as sp
import numpy as np
arr = np.asarray

"""
Helper functions for import/export of text files
for REANIMATE
"""

def import_dat(filename,plot=True):

    graph = sp.SpatialGraph(initialise=True,scalars=['Radii','VesselType','Flow','Pressure'])

    with open(filename,'r') as f:
        
        graphname = f.readline()
        bounding_box = arr([float(x) for x in f.readline().split(' ')[0:3]])
        tissue_points = arr([int(x) for x in f.readline().split(' ')[0:3]])
        outer_bound_distance = float(f.readline().split(' ')[0])
        max_seg_length = float(f.readline().split(' ')[0])
        max_seg_per_node = int(f.readline().split(' ')[0])
        
        nseg = int(f.readline().split(' ')[0])
        header1 = f.readline().strip().split(' ')
        ncol = len(header1)
        seg_name = np.zeros(nseg,dtype='int')
        vessType = np.zeros(nseg,dtype='int')
        conns = np.zeros([nseg,2],dtype='int')
        diameter = np.zeros(nseg,dtype='float')
        flow = np.zeros(nseg,dtype='float')
        hd = np.zeros(nseg,dtype='float')
        for i in range(nseg):
            curLine = f.readline().strip().split(' ')
            seg_name[i], vessType[i], conns[i,0], conns[i,1], diameter[i], flow[i], hd[i] = int(curLine[0]), int(curLine[1]), int(curLine[2]), int(curLine[3]), float(curLine[4]), float(curLine[5]), float(curLine[6])
        
        nnode = int(f.readline().split(' ')[0])
        header2 = f.readline().strip().split(' ')
        node_name = np.zeros(nnode,dtype='int')
        nodecoords = np.zeros([nnode,3],dtype='float')
        node_pressure = np.zeros(nnode,dtype='float')
        ncol = len(header2)
        for i in range(nnode):
            curLine = f.readline().strip().split(' ')
            node_name[i], nodecoords[i,0], nodecoords[i,1], nodecoords[i,2] = int(curLine[0]), float(curLine[1]), float(curLine[2]), float(curLine[3])
            if len(curLine)==5:
                node_pressure[i] = float(curLine[4])
        
        nbcnodes = int(f.readline().split(' ')[0])
        header3 = f.readline().strip().split(' ')
        bc_name = np.zeros(nbcnodes,dtype='int')
        bctype = np.zeros(nbcnodes,dtype='int')
        bcprfl = np.zeros(nbcnodes,dtype='float')
        bcHd = np.zeros(nbcnodes,dtype='float')
        ncol = len(header3)
        for i in range(nbcnodes):
            curLine = f.readline().strip().split(' ')
            bc_name[i], bctype[i], bcprfl[i], bcHd[i] = int(curLine[0]), int(curLine[1]), float(curLine[2]), float(curLine[3])
            
    # Zero-index
    conns = conns - 1
    
    # Convert vessel type codes
    vessTypeConv = vessType.copy()
    vessTypeConv[vessType==1] = 0 # artery
    vessTypeConv[vessType==2] = 2 # capillary
    vessTypeConv[vessType==3] = 1 # vein
            
    # Create edgepoints
    edgepoints = []
    point_radii = []
    point_flow = []
    point_vessel_type = []
    point_pressure = []
    nedgepoints = []
    for i,conn in enumerate(conns):
        edgepoints.append(nodecoords[conn[0]])
        edgepoints.append(nodecoords[conn[1]])
        nedgepoints.append(2)
        point_radii.append([diameter[i]/2.,diameter[i]/2.])
        point_flow.append([flow[i],flow[i]])
        point_vessel_type.append([vessTypeConv[i],vessTypeConv[i]])
        ppress = np.mean([node_pressure[conn[0]],node_pressure[conn[1]]])
        point_pressure.append([ppress,ppress])
    edgepoints = arr(edgepoints)
    nedgepoints = arr(nedgepoints).flatten()
    point_radii = arr(point_radii).flatten()
    point_flow = arr(point_flow).flatten()
    point_vessel_type = arr(point_vessel_type).flatten()
    point_pressure = arr(point_pressure).flatten()
    
    npoint = edgepoints.shape[0]
        
    graph.set_definition_size('VERTEX',nnode)
    graph.set_definition_size('EDGE',nseg)
    graph.set_definition_size('POINT',npoint)
    graph.set_data(nodecoords,name='VertexCoordinates')
    graph.set_data(conns,name='EdgeConnectivity')
    graph.set_data(nedgepoints,name='NumEdgePoints')
    graph.set_data(edgepoints,name='EdgePointCoordinates')
    graph.set_data(point_radii,name='Radii')
    graph.set_data(point_vessel_type,name='VesselType')
    graph.set_data(point_flow,name='Flow')
    graph.set_data(point_pressure,name='Pressure')
    
    if plot:
        edge_col = graph.point_scalars_to_edge_scalars(name='Flow')
        edge_col = np.abs(edge_col)
        edge_col[edge_col==0.] = 1e-6
        edge_col = np.log(edge_col)
        cyls = graph.plot_graph(edge_color=edge_col,cmap='jet',bgcolor=[0,0,0],plot=False,grab=True,win_width=1920*6,win_height=1080*6,radius_scale=1.5)
        
    if graph.sanity_check() is not None:
        breakpoint()
    
    return graph
    
def export_dat(graph,ofile,network_name='anon',remove_intermediate=True,nbc=2):

    if remove_intermediate:
        print('Removing intermedate nodes...')
        ed = sp.Editor()
        graph = ed.remove_intermediate_nodes(graph)
    #graph.write(fname.replace('.am','_datprep.am'))

    vertexCoordinates = graph.get_data('VertexCoordinates')
    edgeConnectivity = graph.get_data('EdgeConnectivity')
    nedgePoints = graph.get_data('NumEdgePoints')
    edgePointCoordinates = graph.get_data('EdgePointCoordinates')
    
    rad_field = graph.get_radius_field()
    thickness = graph.point_scalars_to_edge_scalars(name=rad_field['name'])

    vessType = graph.point_scalars_to_edge_scalars(name='VesselType')
    if vessType is None:
        vessType = np.zeros(thickness.shape[0])

    nnod = vertexCoordinates.shape[0]
    nseg = edgeConnectivity.shape[0]
    npoint = edgePointCoordinates.shape[0]
    #nnod = nvertex

    maxEdge = np.max(edgePointCoordinates,axis=0)
    maxVertex = np.max(vertexCoordinates,axis=0)
    mx = np.max(np.vstack([maxEdge,maxVertex]),axis=0)
    minEdge = np.min(edgePointCoordinates,axis=0)
    minVertex = np.min(vertexCoordinates,axis=0)
    mn = np.min(np.vstack([minEdge,minVertex]),axis=0)
    
    # Check vessel lengths
    lengths = np.linalg.norm(vertexCoordinates[edgeConnectivity[:,1]]-vertexCoordinates[edgeConnectivity[:,0]],axis=1)
    if np.any(lengths==0.):
        breakpoint()
    
    alx,aly,alz = mx - mn

    with open(ofile,'w') as handle:
        handle.write(f"{network_name} network derived from Amira Spatial Graph\n")
        handle.write(f"{alx} {aly} {alz} box dimensions in microns - adjust as needed\n")
        handle.write(f"32 32 32 number of tissue points in x,y,z directions - adjust as needed\n")
        handle.write(f"10	outer bound distance - adjust as needed\n")
        handle.write(f"100	max. segment length - adjust as needed\n")
        handle.write(f"30	maximum number of segments per node - adjust as needed\n")
        handle.write(f"{nseg}	total number of segments\n")
        handle.write(f"SegName Type StartNode EndNode Diam   Flow[nl/min]    Hd\n");

        flow = 1.
        diammax = -1.
        for i in range(nseg):
            segnodname1 = edgeConnectivity[i,0] + 1
            segnodname2 = edgeConnectivity[i,1] + 1
            diam = np.max([2.*thickness[i],4.])
            if diam > diammax:
                diammax = diam
                idiammax = i
                
            # Convert vessel type codes
            if vessType[i]==0: # artery
                vt = 1
            elif vessType[i]==2: # capillary
                vt = 2
            elif vessType[i]==1: # vein
                vt = 3

            handle.write(f"{i+1} {vt} {segnodname1} {segnodname2} {diam} {flow} {0.45}\n")
            
        print(f"Max diameter = {diammax}, idx = {idiammax}")

        handle.write(f"{nnod}   number of nodes\n")
        handle.write(f"Name    x       y       z\n")
        for i in range(nnod):
            handle.write(f"{i+1} {vertexCoordinates[i,0]} {vertexCoordinates[i,1]} {vertexCoordinates[i,2]} {1.0}\n")

        if False:
            inod = nnod
            for i in range(nseg):   # nodes from npoint
                x0 = np.sum(nedgePoints[:i])
                x1 = x0 + nedgePoints[i]
                pts = edgePointCoordinates[x0:x1]
                
                for j,p in ennumerate(pts): # Maybe need to limit this to exclude start and end points?
                    handle.write(f"{inod} {p[0]} {p[1]} {p[2]}\n")
                    inod += 1
                        
            if (cnt != nnod):
                print("*** Error: incorrect number of nodes")

        # Boundary conditions
        if nbc>0:
            nodtype = graph.get_node_count()
            endnodes = np.where(nodtype==1)
            nnodbc = len(endnodes[0])
            handle.write(f"{nnodbc}   number of boundary nodes\n")
            handle.write(f"Name    bctyp     bcprfl     bchd\n")
            inod = 0
            consthd = 0.45
            bctype = 0 # 0=pressure, 1=flow, 3=estimated
            
            if nbc==4:
                pressure = [80.,80.,10.,10.]
            elif nbc==2:
                pressure = [80,60.]

            for inod,pr in zip(endnodes[0],pressure):
                handle.write(f"{inod+1} {bctype} {pr} {consthd}\n")
            
    return True
    
def amira2dat(fname,ofile=None,remove_intermediate=False,nbc=2):
    """
    Convert Amira spatial graph format to Paul Sweeney's REANIMATE format
    """
    graph = sp.SpatialGraph()
    graph.read(fname)
    print(f'Converting {fname}')

    if ofile is None:
        ofile = fname.replace('.am','.dat')        

    export_dat(graph,ofile,remove_intermediate=remove_intermediate,nbc=nbc)
    print(f'Written {ofile}')
    
def dat2amira(fname,ofile=None,plot=False):
    """
    Convert Paul Sweeney's REANIMATE format to Amira format
    """
    graph = import_dat(fname,plot=plot)
    
    if ofile is None:
        ofile = fname.replace('.dat','.am')
    
    print(f'Writing to {ofile}')
    graph.write(ofile)
    
if __name__=='__main__':

    conv = 'd2a' #'a2d'
    
    if conv=='a2d': # export
        #fname = '/mnt/data2/retinasim/data/cco_circ_domain/graph/retina_cco_a2v_reanimate_conn.am'
        #fname = '/mnt/data2/retinasim/data/cco_circ_domain/graph/retina_cco_a2v_datprep.am'
        #fname ='/mnt/data2/retinasim/data/cco_circ_domain/graph/test_network.am'
        fname ='/mnt/data2/retinasim/cco/graph/retina_cco_a2v.am'
        
        #ofile = '/mnt/data2/retinasim/data/cco_circ_domain/graph/reanimate/retina_cco_datprep.dat'
        #ofile = '/mnt/data2/retinasim/data/cco_circ_domain/graph/reanimate/test_network.dat'
        ofile = fname.replace('.am','.dat')
        
        amira2dat(fname,ofile=ofile,remove_intermediate=False,nbc=2)
    elif conv=='d2a': # import
        fname ='/mnt/ml/anaconda_envs/vessel_growth_38/lib/python3.8/site-packages/Reanimate/Build_Data/SolvedBloodFlow.txt'
        
        #ofile = '/mnt/data2/retinasim/data/cco_circ_domain/graph/retina_cco_a2v_reanimate.am'
        ofile = '/mnt/data2/retinasim/cco/graph/retina_cco_a2v_reanimate.am'
        
        dat2amira(fname,ofile=ofile,plot=False)

