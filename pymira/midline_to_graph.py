import numpy as np
arr = np.asarray
#from pymira import spatialgraph
from skimage.graph import route_through_array
from matplotlib import pyplot as plt

class Midline2Graph(object):

    def __init__(self, midline_volume=None):
        self.midline = midline_volume
        self.dims = np.asarray(self.midline.shape)
        
    def get_neighbours(self,vol,i,j,k):
        neigh = np.zeros([3,3,3]) + np.nan
        offsets = np.zeros([3,3,3,3],dtype='int')
        
        for ni,ii in enumerate([-1,0,1]):
            for nj,jj in enumerate([-1,0,1]):
                for nk,kk in enumerate([-1,0,1]):
                    offsets[ni,nj,nk] = [ii,jj,kk]
        
                    if i+ii<0:
                        neigh[ni,nj,nk] = np.nan
                    elif i+ii>vol.shape[0]-1:
                        neigh[ni,nj,nk] = np.nan
                    elif j+jj<0:
                        neigh[ni,nj,nk] = np.nan
                    elif j+jj>vol.shape[1]-1:
                        neigh[ni,nj,nk] = np.nan
                    elif k+kk<0:
                        neigh[ni,nj,nk] = np.nan
                    elif k+kk>vol.shape[2]-1:
                        neigh[ni,nj,nk] = np.nan
                    elif ii==0 and jj==0 and kk==0: # middle pixel
                        neigh[ni,nj,nk] = np.nan
                    #elif np.any(arr([ii,jj,kk])==0):
                    else:
                        neigh[ni,nj,nk] = vol[i+ii,j+jj,k+kk]                        
            
        return neigh, offsets
        
    def convert(self):
    
        self.nodes = np.zeros(self.dims,dtype='int')
        
        edgepoints = np.where(self.midline>0)
        connections = []
        nconn = np.zeros(len(edgepoints[0]),dtype='int')

        # Loop through each midline point, connecting it to each filled neighbour
        for ind in range(len(edgepoints[0])):
            i,j,k = edgepoints[0][ind],edgepoints[1][ind],edgepoints[2][ind]
            # Find filled neighbours
            neigh,offsets = self.get_neighbours(self.midline,i,j,k)            
            sneigh = np.where((np.isfinite(neigh)) & (neigh>0))
            # Loop through each neighbour and find which edge point index it refers to
            for ni in range(len(sneigh[0])):
                nii,njj,nkk = offsets[sneigh[0][ni],sneigh[1][ni],sneigh[2][ni]]
                ii,jj,kk = i+nii,j+njj,k+nkk
                #offset = offsets[ii,jj,kk]
                conn_ind = np.where((edgepoints[0]==ii) & (edgepoints[1]==jj) & (edgepoints[2]==kk))
                connections.append([ind,int(conn_ind[0])])
            
        # Delete doubled-up connections
        connections = arr(connections)
        rem_conn = np.zeros(connections.shape[0])
        for i,conn in enumerate(connections):
            if rem_conn[i]==0:
                sind = np.where((connections[:,0]==conn[1]) & (connections[:,1]==conn[0]))
                if len(sind[0])>0:
                    rem_conn[sind] = 1
                    
                sind = np.where((connections[:,0]==conn[0]) & (connections[:,1]==conn[1]))
                if len(sind[0])>1:
                    sind[0] = sind[0][sind[0]!=i]
                    rem_conn[sind[1:]] = 1
                
        # Delete labelled connections
        rem_inds = np.where(rem_conn)
        if len(rem_inds[0])>0:
            connections = np.delete(connections,rem_inds[0],axis=0) 
        
        for ind in range(len(edgepoints[0])):
            sind = np.where((connections[:,0]==ind) | (connections[:,1]==ind))
            nconn[ind] = len(sind[0])
            
        edgepoints = arr(edgepoints).transpose()
        
        #self.plot(edgepoints,connections)
        
        return edgepoints, connections
        
        # Simplify by walking through connections. Create nodes connected by edges
        sind = np.where((nconn>2) | (nconn==1))
        conn_visited = np.zeros(connections.shape[0])
        rem_conn = np.zeros(connections.shape[0])
        nodes = []

        for si in sind[0]:
            ei = np.where((connections[:,0]==si) | (connections[:,1]==si)) # All connections will be bi-directional at this point, so only consider first element
            
            for e in ei[0]:
                
                start_edge_index = e
                start_conn = connections[e,:]
                start_edgepoint = si
                cur_edgepoint = start_conn[start_conn!=si][0]
                
                edgepoint_trace = [start_edgepoint]
                conn_trace = [e]
                conn_visited[e] = 1
                edgepoint_visited = [si] #,cur_edgepoint]
                
                count = 0
                while True:
                    count += 1

                    edgepoint_visited.append(cur_edgepoint)
                    print(edgepoint_visited)
                    
                    # Find next edge
                    fi = [i for i,x in enumerate(connections) if (x[0]==cur_edgepoint or x[1]==cur_edgepoint) and conn_visited[i]==0]

                    next_conn = connections[fi]
                    if next_conn.shape[0]>1:
                        breakpoint()
                        cur_edgepoint = connections[fi[0],:]
                        cur_edgepoint = cur_edgepoint[cur_edgepoint==e]
                        end_edgepoint = cur_edgepoint
                        break
                    else:
                        if len(fi)>0:
                            edgepoint_trace.append(fi[0])
                            conn_trace.append(fi[0])
                            rem_conn[fi[0]] = 1
                            conn_visited[fi[0]] = 1
                            cur_conn = connections[fi[0],:]
                            cur_edgepoint = cur_conn[cur_conn!=cur_edgepoint][0]
                        else: # no edges left - abandon?
                            end_edgepoint = cur_edgepoint
                            #conn_trace.append(fi[0])
                            break
                            
                if len(edgepoint_trace)>0:
                    #nodes.append(edgepoints[])
                    breakpoint()
                
    def plot(self,edgepoints,connections):
    
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(edgepoints[:,0],edgepoints[:,1],edgepoints[:,2])
        
        for conn in connections:
            x = [edgepoints[conn[0],0],edgepoints[conn[1],0]]
            y = [edgepoints[conn[0],1],edgepoints[conn[1],1]]
            z = [edgepoints[conn[0],2],edgepoints[conn[1],2]]
            plt.plot(x,y,z)
        
        plt.show()
                        
