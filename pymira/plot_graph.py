import numpy as np
import os
join = os.path.join

from mayavi import mlab
import matplotlib as mpl
from pymira import spatialgraph
from tqdm import trange

class Graph(object):

    def __init__(self, graph_file):
    
        self.init_size = 20000
        
        self.coords = np.zeros([self.init_size,3]) - 1
        self.conns = np.zeros([self.init_size,2],dtype='int') - 1
        self.nedgepoints = np.zeros(self.init_size,dtype='int') 
        self.edgepoints = np.zeros([self.init_size*100,3]) - 1
        self.radii = np.zeros(self.init_size*100) - 1
        self.depth_order = np.zeros(self.init_size,dtype='int') - 1
        self.vessel_type = np.zeros(self.init_size*100,dtype='int') - 1 # 0=artery, 1=vein, 2=undefined
        self.scalar_fields = []
        
        self.read(graph_file)
        
    def read(self, graph_file):
    
        self.graph = spatialgraph.SpatialGraph()
        self.graph.read(graph_file)
        
        coords = self.graph.get_field('VertexCoordinates')['data']
        conns = self.graph.get_field('EdgeConnectivity')['data']
        nedgepoints = self.graph.get_field('NumEdgePoints')['data']
        edgepoints = self.graph.get_field('EdgePointCoordinates')['data']
        rad_names = ['thickness','radius','radii','Radii','Radius']
        for name in rad_names:
            radii = self.graph.get_field(name)
            if radii is not None:
                radii = radii['data']
                break
        vessel_type = self.graph.get_field('vessel_type')

        self.ncoords = coords.shape[0]
        self.nconns = conns.shape[0]
        self.nedges = edgepoints.shape[0]
        
        self.coords[:self.ncoords] = coords
        self.conns[:self.nconns] = conns
        self.nedgepoints[:self.nconns] = nedgepoints
        self.edgepoints[:self.nedges] = edgepoints
        self.radii[:self.nedges] = radii
        if vessel_type is not None:
            self.vessel_type[:self.nedges] = vessel_type['data']
        else:
            self.vessel_type[:self.nedges] = 2
        
        self.scalar_fields = []
        for field in self.graph.fields:
            if field['definition']=='POINT':
                #new_vals = np.zeros(self.init_size,dtype=field['data'].dtype)
                new_name = field['name'].replace(' ','_')
                setattr(self,new_name,field['data'])
                self.scalar_fields.append(new_name)
            
    def get_coords(self):
        return self.coords[:self.ncoords]
        
    def set_coords(self, coords):
        self.ncoords = coords.shape[0]
        self.coords[:self.ncoords] = coords
        self.coords[self.ncoords:] = -1
        
    def get_conns(self):
        return self.conns[:self.nconns]
        
    def set_conns(self, conns):
        self.nconns = conns.shape[0]
        self.conns[:self.nconns] = conns
        self.conns[self.nconns:] = -1        
        
    def get_nedgepoints(self):
        return self.nedgepoints[:self.nconns]
        
    def set_nedgepoints(self, nedgepoints):
        self.nconns = nedgepoints.shape[0]
        self.nedgepoints[:self.nconns] = nedgepoints
        self.nedgepoints[self.nconns:] = -1      
        
    def get_edgepoints(self):
        return self.edgepoints[:self.nedges]
        
    def set_edgepoints(self, edgepoints):
        self.nedges = edgepoints.shape[0]
        self.edgepoints[:] = -1
        self.edgepoints[:self.nedges] = edgepoints
        #self.edgepoints[self.nedges:] = -1    
        
    def get_radii(self):
        return self.radii[:self.nedges]
        
    def set_radii(self, radii):
        #self.nedges = radii.shape[0]
        self.radii[:self.nedges] = radii
        self.radii[self.nedges:] = -1    
        
    def get_vessel_types(self):
        return self.vessel_type[:self.nedges]
        
    def set_vessel_types(self, vessel_type):
        #self.nedges = radii.shape[0]
        self.vessel_type[:self.nedges] = vessel_type
        self.vessel_type[self.nedges:] = -1           
        
    def delete_node(self, node_index):
        if node_index>=self.ncoords or node_index<0:
            print('Error, DELETE_NODE: Invalid node index')
            return None
            
        # Delete nodes 
        ed = Editor()
        nodes_to_delete = [node_index]
        coords,conns,nedgepoints,edgepoints,scalars,info = ed._del_nodes(nodes_to_delete,self.get_coords(),self.get_conns(),self.get_nedgepoints(),self.get_edgepoints(),scalars=[self.get_radii()])
        self.set_coords(coords)
        self.set_conns(conns)
        self.set_nedgepoints(nedgepoints)
        self.set_edgepoints(edgepoints)
        self.set_radii(scalars[0])
        
        return info
            
    def delete_edge(self, ind):
    
        edgepoints = self.get_edgepoints()
        nedgepoints = self.get_nedgepoints()
    
        npoints = int(nedgepoints[ind])
        
        x0 = int(np.sum(nedgepoints[:ind]))
        x1 = x0 + npoints
        if x0>0:
            edges_pre = edgepoints[:x0]
        else:
            edges_pre = []  
        if x1<edgepoints.shape[0]:
            edges_post = edgepoints[x1:]
        else:
            edges_post = []
        edgepoints = np.concatenate([x for x in [ edges_pre, edges_post ] if len(x)>0])
        self.set_edgepoints(edgepoints)
        self.set_nedgepoints(nedgepoints)
         
            
    def replace_edge(self,ind,path,radius=None):
    
        # Replaces an edge with new points
    
        edgepoints = self.get_edgepoints()
        nedgepoints = self.get_nedgepoints()
    
        npoints = path.shape[0]
        npoints_old = int(nedgepoints[ind])
        nedgepoints[ind] = npoints
        
        x0 = int(np.sum(nedgepoints[:ind]))
        x1_old = x0 + npoints_old
        x1 = x0 + npoints
        if x0>0:
            edges_pre = edgepoints[:x0]
        else:
            edges_pre = []  
        if x1_old<edgepoints.shape[0]:
            edges_post = edgepoints[x1_old:]
        else:
            edges_post = []
        edgepoints = np.concatenate([x for x in [ edges_pre, path, edges_post ] if len(x)>0])
        #print('Replace edge {}'.format(ind))
        #print('Replace edge nedgepoints: {}'.format(nedgepoints))        
        #print('Replace edge edgepoints: {}'.format(edgepoints))
        self.set_edgepoints(edgepoints)
        self.set_nedgepoints(nedgepoints)            

    def get_edges_containing_node(self, ind):
        
        return [self.get_edge(i) for i,x in enumerate(self.conns) if ind in x]
        
    def get_edge(self,i):
        if self.nconns==0:
            return None
        if i>=self.nconns or i<0:
            return None
            
        x0 = int(np.sum(self.nedgepoints[:i]))
        x1 = x0 + int(self.nedgepoints[i])
        radii = self.radii[x0:x1]
        coords = self.edgepoints[x0:x1]
        vessel_type = self.vessel_type[x0]
        
        edge = {'x0':x0,'x1':x1,'coords':coords,'radii':radii,'npoints':x1-x0,'vessel_type':vessel_type,'nodes':self.conns[i]}
        
        if len(self.scalar_fields)>0:
            for name in self.scalar_fields:
                vals = getattr(self,name)
                if vals is not None:
                    edge[name] = vals[x0]
        
        return edge
        
    def export(self):
        root = Tk()
        
        if self.ofile is None:
            init_file = self.ofile_suggestion
        else:
            init_file = self.ofile
        root.filename = filedialog.asksaveasfilename(initialdir=self.opath,initialfile=init_file,title="Select export file",filetypes = (("Amira spatial graph files","*.am"),("all files","*.*")))

        ofile = root.filename
        root.destroy()
        
        if ofile=='': 
            return

        graph = spatialgraph.SpatialGraph(initialise=True,scalars=['Radii','vessel_type'])
        graph.set_definition_size('VERTEX',self.ncoords)
        graph.set_definition_size('EDGE',self.nconns)
        graph.set_definition_size('POINT',self.nedges)
        graph.set_data(self.get_coords(),name='VertexCoordinates')
        graph.set_data(self.get_conns(),name='EdgeConnectivity')
        graph.set_data(self.get_nedgepoints(),name='NumEdgePoints')
        graph.set_data(self.get_edgepoints(),name='EdgePointCoordinates')
        graph.set_data(self.get_radii(),name='Radii')
        graph.set_data(self.get_vessel_types(),name='vessel_type')
        #graph.set_data(self.depth_order,name='VesselType')

        #ofile = 'C:\\Anaconda2\\Lib\\site-packages\\pymira\\test_network.am'
        graph.write(ofile)
        self.ofile = os.path.basename(ofile)
        self.opath = os.path.dirname(ofile)
        print(self.ofile,self.opath)        

class PlotGraph(Graph): 

    def __init__(self,graph_file, show=False, figure=None, radius_scaling=1., plot_nodes=False, line_color_property=None):
    
        super().__init__(graph_file)
        
        self.radius_scaling = radius_scaling
        self.line_color_property = line_color_property
        
        # Create plots and widgets
        self.select_color = 'y'
        self.unselect_color = 'b'
        self.artery_color = 'r'
        self.vein_color = 'b'
        self.undefined_color = 'g'
        
        self.plot_nodes = plot_nodes
        
        self.figure = figure
        
        self.initialise_graph(graph_file)
        self.initialise_plot()
        if show:
            self.show()
        
    def show(self):
        mlab.show()

    def initialise_graph(self,graph_file):

        self.display_elements_3d = {'Image':True,'Segments':True,'Nodes':True}
        
        self.scats3d = []
        self.lines3d = []
        
        self.last_point = None
        self.last_point_index = None
        self.active_object = None
        self.current_vessel_type = 0 # artery
        
        self.current_draw_radius = 2.0

        # Load graph
        self.graph = Graph(graph_file)
           
    def initialise_plot(self):

        self.plot_selected = None

        self.image_3d = None
        self.mlab_wid,self.mlab_hei = 1500,1500
        
        if self.figure is None:
            self.figure = mlab.figure(size=(self.mlab_wid, self.mlab_hei))
        #picker = self.fig3d.on_mouse_pick(self.mlab_picker_callback)

        # Add plots...
        self.lines3d = []
        lim = None
        for i in trange(self.nconns):
            line = self.create_3d_line_plot(i)
            self.lines3d.append(line)
            if lim is not None and i>lim:
                break
        
        if self.plot_nodes:
            for i in range(self.ncoords):
                self.add_3d_scatter_point(i,active=False)   
                if lim is not None and i>lim:
                    break
        
    def get_line_color(self, edgeIndex):
        if self.line_color_property is not None:
            edge = self.get_edge(edgeIndex)
            try:
                x = edge[self.line_color_property]
                norm = mpl.colors.Normalize(vmin=-1, vmax=3)
                cmap = mpl.cm.hot
                m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                rgba = m.to_rgba(x)
                return rgba[0:3]
                #return col
            except Exception as e:
                print(e)
                
        edge = self.get_edge(edgeIndex)
        if edge['vessel_type']==0: # artery
            return self.artery_color
        elif edge['vessel_type']==1: # vein
            return self.vein_color
        else:
            return self.undefined_color
            
    def get_node_color(self, nodeIndex):
    
        edges = [self.get_edge(i) for i,x in enumerate(self.conns) if nodeIndex in x]
        if len(edges)==0:
            return self.unselect_color
            
        vessel_type = edges[0]['vessel_type']
        if vessel_type==0:
            return self.artery_color
        elif vessel_type==1:
            return self.vein_color
        else:
            return self.undefined_color
            
    def get_node_vessel_type(self, nodeIndex):
    
        edges = [self.get_edge(i) for i,x in enumerate(self.conns) if nodeIndex in x]
        if len(edges)==0:
            return -1
            
        vessel_type = edges[0]['vessel_type']
        if vessel_type==0:
            return 0
        elif vessel_type==1:
            return 1
        else:
            return 2
            
    def update_3d_display_visibility(self, element, new_status):

        if element.lower()=='image':
            #self.im_surface_3d.set_visible(new_status)
            #self.image_3d.set_enabled(new_status)
            #self.image_3d.interaction = new_status
            self.display_elements_3d['Image'] = new_status
        elif element.lower()=='segments':
            for line in self.lines3d:
                line.set_visible(new_status)
            self.display_elements_3d['Segments'] = new_status
        elif element.lower()=='nodes':
            for scat in self.scats3d:
                scat.set_visible(new_status)
            self.display_elements_3d['Nodes'] = new_status           
        
    def create_active_text(self, res=None):
        if res is None:
            res = self.active_object
    
        if res is None:
            return 'None'
            
        if res['type']=='node':
            x,y,z = self.coords[res['index']]
            return 'Node {}, z={}'.format(res['index'],z)
        elif res['type']=='edge':
            try:
                rad = res['radii'][res['point_index']]
            except Exception as e:
                print(e)
                rad = -1
            return 'Edge {}, point {}, radius {}'.format(res['index'],res['point_index'],rad)
        else:
            return 'Error'
        
    def update_annot(self, event, res):
    
        x,y = event.xdata, event.ydata
        self.annot.xy = [x,y]
        
        text = self.create_active_text(res)

        self.annot.set_text(text)
        cmap = plt.cm.RdYlGn
        self.annot.get_bbox_patch().set_facecolor(cmap(0))
        self.annot.get_bbox_patch().set_alpha(0.4)
        
    def mouse_over_object(self, event):
    
        cont, res = False, None
        if event.inaxes == self.ax:
            cont = False
            xy = np.asarray([event.xdata, event.ydata])
            for i,line in enumerate(self.lines):
                cont, ind = line.contains(event)
                if cont:                  
                    inds = ind['ind']
                    # Check if hovering over start or end node
                    start_node,end_node = False,False
                    edge = self.get_edge(i)
                    if 0 in inds or edge['npoints']-1 in inds:
                        nodes = self.conns[i]
                        # Get start and end points for edge
                        e0,en = edge['coords'][0], edge['coords'][-1]
                        # See which node the mouse is closest to
                        if np.linalg.norm(e0[0:2]-xy) < np.linalg.norm(en[0:2]-xy):
                            node_index = nodes[0]
                            res = {'type':'node','index':node_index}
                        else:
                            node_index = nodes[1]
                            res = {'type':'node','index':node_index}
                    else:
                        point_index = np.argmin([np.linalg.norm(x[0:2]-xy) for x in edge['coords']])
                        res = {'type':'edge','index':i,'point_index':point_index,'point_coord':edge['coords'][point_index],'radii':edge['radii']}
                    break
                    
        return cont, res

    def hover(self, event):
        vis = self.annot.get_visible()
        
        cont, res = self.mouse_over_object(event)

        if cont:
            self.update_annot(event, res)
            self.annot.set_visible(True)
            self.fig.canvas.draw_idle()
        else:
            if vis:
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()
                
    def control_hover(self, event):
        vis = self.annot.get_visible()
        
        if event.inaxes in self.control_active_axes:
            x,y = event.xdata, event.ydata
            self.annot.xy = [x,y]
            
            text = self.create_active_text(res)

            self.annot.set_text(text)
            cmap = plt.cm.RdYlGn
            self.annot.get_bbox_patch().set_facecolor(cmap(0))
            self.annot.get_bbox_patch().set_alpha(0.4)
            self.annot.set_visible(True)

            if vis:
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()

    def add_coord(self, coord):
        self.coords[self.ncoords,:] = coord
        self.ncoords += 1
        self.depth_order[self.ncoords] = 1
        
        #self.add_scatter_plot(self.ncoords-1)
        if self.plot_nodes:
            self.add_3d_scatter_point(self.ncoords-1)        
        
    def add_line_plot(self,path,at_ind=None):
    
        if at_ind is None:
            col = self.get_line_color(self.nconns-1)
        else:
            col = self.get_line_color(at_ind)
            
        if path is None:
            x,y = [0,0], [0,0]
        else:
            x,y = path[:,0], path[:,1]
        line, = self.ax.plot(x,y,c=col,picker=True)
        line.set_visible(self.display_elements['Segments'])
        
        #print('Adding line: {}'.format())
        
        if at_ind is None:
            self.lines.append(line)
        else:
            try:
                self.lines[at_ind].remove()
            except:
                pass
            self.lines[at_ind] = line
            
    def add_3d_line_plot(self,path,at_ind=None):
    
        if at_ind is None:
            ind = self.nconns-1
        else:
            ind = at_ind
            
        line = self.create_3d_line_plot(ind, path=path)
        
        if at_ind is not None:
            try:
                self.lines3d[at_ind].remove()
            except Exception as e:
                print('Error: {}'.format(e))
                pass
            self.lines3d[at_ind] = line
        else:
            self.lines3d.append(line)  
            
    def create_3d_line_plot(self, ind, path=None):
            
        col = self.get_line_color(ind)
        edge = self.get_edge(ind)
        
        if path is None:
            path = edge['coords']

        x,y,z = path[:,0],path[:,1],path[:,2]
        s = np.zeros(x.shape) + 2
        color_rgb = self.to_rgb(col)
        #tube_rad = 5
        tube_rad = edge['radii'][0] * self.radius_scaling
        if tube_rad<0:
            tube_rad = 2
        line = mlab.plot3d(x,y,z,s,color=color_rgb,tube_radius=tube_rad)
        
        return line 
        
    def add_3d_scatter_point(self,coord_index,active=True):
        coord = self.coords[coord_index]

        if active:
            color_rgb = self.to_rgb(self.select_color)
        else:
            col = self.get_node_color(coord_index)
            color_rgb = self.to_rgb(col)
        size = [10]
        scat = mlab.points3d(coord[0],coord[1],coord[2],size,color=color_rgb)
        scat.glyph.glyph.clamping = False
        self.scats3d.append(scat)
            
    def mlab_picker_callback(self, picker):
        """ Picker callback: this get called when on pick events.
        """
        
        if picker.actor is None:
            return

        scat_actors = [x.actor.actor._vtk_obj for x in self.scats3d]
        line_actors = [x.actor.actor._vtk_obj for x in self.lines3d]

        if picker.actor._vtk_obj in scat_actors:
            ind = scat_actors.index(picker.actor._vtk_obj)
            self.set_active_object({'type':'node','index':ind})
        elif picker.actor._vtk_obj in line_actors:
            ind = line_actors.index(picker.actor._vtk_obj)
            edge = self.get_edge(ind)
            point_index = 0
            self.set_active_object({'type':'edge','index':ind,'point_index':point_index,'point_coord':edge['coords'][point_index]})
        
    def add_path(self, path, node_inds=None, radii=None, vessel_type=2):
        self.edgepoints[self.nedges:self.nedges+path.shape[0]] = path
        if radii is not None:
            self.radii[self.nedges:self.nedges+path.shape[0]] = radii
        self.vessel_type[self.nedges:self.nedges+path.shape[0]] = vessel_type #np.zeros(path.shape[0])+vessel_type
        self.nedgepoints[self.nconns] = path.shape[0]
        self.conns[self.nconns,:] = node_inds
        self.nconns += 1
        self.nedges += path.shape[0]

        self.add_3d_line_plot(path)
        
    def set_active_object(self, res):
    
        prev_active = self.active_object
        self.active_object = res
        
        self.mark_object_selected(res)
        if prev_active is not None:
            self.mark_object_deselected(prev_active)
            
        text = self.create_active_text()
        self.active_box.set_val(text)
        
        print(res['type'])
        if res['type']=='edge':
            edge = self.get_edge(res['index'])
            print(edge)
            txt = "{}".format(edge['radii'][0])
            #txt = 'test'
            self.active_radius_box.set_val(txt)
            self.depth_order_box.set_val("-")
            self.set_vessel_type_box()
        else:
            self.active_radius_box.set_val("-")
            self.depth_order_box.set_val("{}".format(self.depth_order[res['index']]))
            self.set_vessel_type_box()
                
        self.fig.canvas.draw()
        
    def to_rgb(self, color):
        return mpl.colors.to_rgb(color)
        
    def mark_object_deselected(self, res):
        if res['type']=='node':
            col = self.get_node_color(res['index'])
            sc3d = self.scats3d[res['index']]
            sc3d.mlab_source.trait_set(scalars=[10]) #scale_factor=5)
            sc3d.actor.actor.property.color = self.to_rgb(col)
        elif res['type']=='edge':
            col = self.get_line_color(res['index'])
            self.lines3d[res['index']].actor.actor.property.color = self.to_rgb(col)
            
    def mark_object_selected(self, res):
        if res['type']=='node':
            sc3d = self.scats3d[res['index']]
            sc3d.mlab_source.trait_set(scalars=[10]) #,color=(0.,0.,1.))
            sc3d.actor.actor.property.color = self.to_rgb(self.select_color)
            print(sc3d,sc3d.parent,sc3d.parent.parent,sc3d.parent.parent.parent)
            node_type = self.get_node_vessel_type(res['index'])
            if node_type>=0:
                self.vessel_type_radio.set_active(node_type)
        elif res['type']=='edge':   
            self.lines3d[res['index']].actor.actor.property.color = self.to_rgb(self.select_color)
            edge = self.get_edge(res['index'])
            # Change the vessel type to match the active edge
            if edge['vessel_type']>=0:
                self.current_vessel_type = edge['vessel_type']
                self.vessel_type_radio.set_active(np.clip(self.current_vessel_type,0,2))
        
    def deselect_active(self):
    
        if self.active_object is not None:
            self.mark_object_deselected(self.active_object)
            self.active_object = None
            self.active_box.set_val(text)
            
    def get_edges_containing_node(self, ind):
        
        return [self.get_edge(i) for i,x in enumerate(self.conns) if ind in x]
            

        
    def get_coords(self):
        return self.coords[:self.ncoords]
        
    def set_coords(self, coords):
        self.ncoords = coords.shape[0]
        self.coords[:self.ncoords] = coords
        self.coords[self.ncoords:] = -1
        
    def get_conns(self):
        return self.conns[:self.nconns]
        
    def set_conns(self, conns):
        self.nconns = conns.shape[0]
        self.conns[:self.nconns] = conns
        self.conns[self.nconns:] = -1        
        
    def get_nedgepoints(self):
        return self.nedgepoints[:self.nconns]
        
    def set_nedgepoints(self, nedgepoints):
        self.nconns = nedgepoints.shape[0]
        self.nedgepoints[:self.nconns] = nedgepoints
        self.nedgepoints[self.nconns:] = -1      
        
    def get_edgepoints(self):
        return self.edgepoints[:self.nedges]
        
    def set_edgepoints(self, edgepoints):
        self.nedges = edgepoints.shape[0]
        self.edgepoints[:] = -1
        self.edgepoints[:self.nedges] = edgepoints
        #self.edgepoints[self.nedges:] = -1    
        
    def get_radii(self):
        return self.radii[:self.nedges]
        
    def set_radii(self, radii):
        #self.nedges = radii.shape[0]
        self.radii[:self.nedges] = radii
        self.radii[self.nedges:] = -1    
        
    def get_vessel_types(self):
        return self.vessel_type[:self.nedges]
        
    def set_vessel_types(self, vessel_type):
        #self.nedges = radii.shape[0]
        self.vessel_type[:self.nedges] = vessel_type
        self.vessel_type[self.nedges:] = -1           
        
    def delete_node(self, node_index):
        if node_index>=self.ncoords or node_index<0:
            print('Error, DELETE_NODE: Invalid node index')
            return
            
        info = super().delete_node(node_index)
        if info is None:
            return
        
        # Update plots
        if len(info['edges_deleted'])>0:
            #print('Deleting edges: {}'.format(info['edges_deleted']))
            for line_ind in info['edges_deleted']:
                self.lines[line_ind].remove()
                try:
                    self.lines3d[line_ind].remove()
                except Exception as e:
                    print(e)
                    pass
                    
            # Maintain state
            self.lines3d = [x for i,x in enumerate(self.lines3d) if i in info['edges_kept']]
            
        if len(info['nodes_deleted'])>0:
            #print('Deleting nodes: {}'.format(info['nodes_deleted']))
            for node_ind in info['nodes_deleted']:
                self.scats[node_ind].remove()
                try:
                    self.scats3d[node_ind].remove()
                except:
                    pass
            self.scats3d = [x for i,x in enumerate(self.scats3d) if i in info['nodes_kept']]
            
    def delete_edge(self, ind):
    
        super().delete_edge(ind)
        
        # Update plots
        try:
            self.lines3d[ind].remove()
        except Exception as e:
            print(e)
            pass
                    
        # Maintain state
        self.lines = [x for i,x in enumerate(self.lines) if i!=ind]
        self.lines3d = [x for i,x in enumerate(self.lines3d) if i!=ind]
            
    def display_elements_callback(self, event):

        status = self.display_element_check.get_status()
        
        new_status = None
        for i,k in enumerate(self.display_elements.keys()):
            if k.lower()==event.lower():
                new_status = status[i]
                
        if new_status is None:
            print('Error, DISPLAY_ELEMENTS_CALLBACK: Element not identified: {}'.format(event))
            return
        
        self.update_display_visibility(event,new_status)
        
    def display_elements_3d_callback(self, event):

        status = self.display_element_check.get_status()
        
        new_status = None
        for i,k in enumerate(self.display_elements_3d.keys()):
            if k.lower()==event.lower():
                new_status = status[i]
                
        if new_status is None:
            print('Error, DISPLAY_ELEMENTS_3D_CALLBACK: Element not identified: {}'.format(event))
            return
        
        self.update_3d_display_visibility(event,new_status)
        
