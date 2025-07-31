import csv
import argparse
import numpy as np
from pymira import spatialgraph

def csv2amira(filepath, ofile='', sanity_check=True,quiet=True):
    start_coords, end_coords, radii = [], [], []
    with open(filepath, 'r') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row)==7:
                try:
                    start_coords.append(np.asarray([float(x) for x in row[0:3]]))
                    end_coords.append(np.asarray([float(x) for x in row[3:6]]))
                    radii.append(float(row[-1]))
                except Exception as e: # Any error, jump ship!
                    print(e)
                    print('Aborting...!')
                    return
            
    start_coords = np.asarray(start_coords)
    end_coords = np.asarray(end_coords)
    radii = np.asarray(radii)
    
    nseg = start_coords.shape[0]
    
    # Infer connectivity from uniqueness of nodes
    all_coords = np.vstack([start_coords,end_coords])
    node = np.zeros(all_coords.shape[0],dtype='int') - 1
    node_coords = []
    node_count = 0
    for i,crd in enumerate(all_coords):
        if node[i]==-1:
            inds = np.where((all_coords[:,0]==crd[0]) & (all_coords[:,1]==crd[1]) & (all_coords[:,2]==crd[2]))
            if len(inds[0])>0:
                node[inds[0]] = node_count
                node_count += 1
                node_coords.append(crd)
                
    node_coords = np.asarray(node_coords)
    start_node_inds = node[:nseg]
    end_node_inds = node[nseg:]
    
    conns = np.zeros([nseg,2],dtype='int')
    points = np.zeros([nseg*2,3],dtype=start_coords.dtype)
    # Radii are defined by edge points rather than segments
    point_radii = np.zeros(nseg*2)
    npoints = np.zeros(nseg,dtype='int') + 2
    for i in range(nseg):
        conns[i] = [start_node_inds[i],end_node_inds[i]]
        points[2*i] = node_coords[start_node_inds[i]]
        points[2*i+1] = node_coords[end_node_inds[i]]
        point_radii[2*i] = radii[i]
        point_radii[2*i+1] = radii[i]

    graph = spatialgraph.SpatialGraph(initialise=True,scalars=['Radii'])
    graph.set_definition_size('VERTEX',node_count)
    graph.set_definition_size('EDGE',nseg)
    graph.set_definition_size('POINT',nseg*2)
    graph.set_data(node_coords,name='VertexCoordinates')
    graph.set_data(conns,name='EdgeConnectivity')
    graph.set_data(npoints,name='NumEdgePoints')
    graph.set_data(points,name='EdgePointCoordinates')
    graph.set_data(point_radii,name='Radii')

    if sanity_check:
        graph.sanity_check(deep=True)

    if ofile=='':
        ofile = filepath+'.am'
    graph.write(ofile)
    
    if quiet==False:
        print('Converted {} to {}'.format(filepath,ofile))
    return ofile
    
def main():
    parser = argparse.ArgumentParser(description="csv2amira argument parser")

    # Add arguments
    parser.add_argument("filename", type=str, help="JSON filepath")
    parser.add_argument("-o","--opath", type=str, default=None, help="AM filepath")
    parser.add_argument("-s","--stl", type=bool, default=True, help="STL")
    
    args = parser.parse_args()
    
    # Access the parsed arguments
    filename = args.filename
    ofile = args.opath
    stl = args.stl
    
    if ofile is None:
        ofile = filename.lower().replace('.csv','.am')
    csv2amira(filename,ofile=ofile)

    if stl is True:
        graph = spatialgraph.SpatialGraph()
        graph.read(ofile)
        graph.export_mesh(ofile=ofile.replace('.am','.ply'),resolution=10)
            
if __name__=='__main__':
    main()

