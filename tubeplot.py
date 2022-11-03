import numpy as np
arr = np.asarray
import open3d as o3d
import pyvista as pv
from tqdm import tqdm, trange

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, 0.
    axis_ = np.cross(a, b)
    l = np.linalg.norm(axis_)
    if l>0.:
        axis_ = axis_ / np.linalg.norm(axis_)
        angle = np.arccos(np.dot(a, b))
    else:
        angle = 0.

    return axis_, angle

class TubePlot(object):

    def __init__(self,graph, cylinders=None, cylinders_combined=None, color=None, edge_color=None, 
                         min_radius=0.,domain_radius=None,radius_scale=1.,domain_centre=arr([0.,0.,0.]),radius_based_resolution=True,cyl_res=10,edge_filter=None,node_filter=None,
                         cmap_range=[None,None],bgcolor=[0.,0.,0.],cmap=None,win_width=1920,win_height=1080,grab_file=None,
                         edge_highlight=[],node_highlight=[],highlight_color=[1,1,1],scalar_color_name=None,log_color=False,
                         show=True,block=True,engine='open3d'):
        self.vis = None
        self.graph = graph
        self.cylinders = cylinders
        self.cylinders_combined = cylinders_combined
        
        # Minimum vessel radius to plot (scalar)
        self.min_radius = min_radius
        # Size of area to plot
        self.domain_radius = domain_radius
        # Factor to scale vessel radii by
        self.radius_scale = radius_scale
        # Centre of plot domain
        self.domain_centre = domain_centre
        # Cylinder resolution based on radius (boolean)
        self.radius_based_resolution = radius_based_resolution
        # Cylinder resolution (scalar)
        self.cyl_res = cyl_res
        # Edges and nodes to include in plot (np.where result)
        self.edge_filter = edge_filter
        self.node_filter = node_filter
        
        # Backend (open3d or pyvista)
        self.engine = engine
        # Colour range for edges [None,None]
        self.cmap_range = cmap_range
        # Background colour
        self.bgcolor = bgcolor
        # Array of edge colours ([nedge])
        self.edge_color = edge_color
        # Edge colors ([nedge,3])
        self.color = color
        # Colour map name ('gray','jet')
        self.cmap = cmap
        # Window dimensions
        self.win_width = win_width
        self.win_height = win_height
        # Make window visible (boolean)
        self.show = show
        # Blocking behaviour (boolean)
        self.block = block
        
        # Array identifying edges to highlight (np.where result)
        self.edge_highlight = edge_highlight
        # Array identifying nodes to highlight
        self.node_highlight = node_highlight
        # Hightlight colour for above
        self.highlight_color = highlight_color
        # Scalar parameter for edge colours (radius by default)
        self.scalar_color_name = scalar_color_name
        # Whether to log colour scale (boolean)
        self.log_color = log_color
        
        # Create cylinders if they have not been provided
        if self.cylinders is None and self.cylinders_combined is None:
            self.create_plot_cylinders()
     
        # Create plot window
        self.create_plot_window()
            
        # Set colours (only if raw cylinders have been provided)
        if self.cylinders_combined is None: 
            print('Preparing graph (adding color and combining)...')
            if self.scalar_color_name is None:
                if 'VesselType' in graph.fieldNames:
                    self.scalar_color_name = 'VesselType'
                else:
                    radName = graph.get_radius_field()['name']
                    self.scalar_color_name = radName
            self.set_cylinder_colors()
            # Combine cylinders
            self.combine_cylinders()                

        if self.block:
            self._show_plot()

    def set_cylinder_colors(self,edge_color=None,scalar_color_name=None,cmap=None,cmap_range=None,update=True):
    
        if scalar_color_name is not None:
            self.scalar_color_name = scalar_color_name
        if cmap is not None:
            self.cmap = cmap
        if cmap_range is not None:
            self.cmap_range = cmap_range

        nedge = self.graph.nedge
        nedgepoint = self.graph.nedgepoint
        sind = self.cylinder_inds
            
        # Grab scalar data for lookup table, if required
        if edge_color is None:
            scalars = self.graph.get_scalars()
            scalarNames = [x['name'] for x in scalars]
            if self.scalar_color_name in scalarNames:
                self.edge_color = self.graph.get_data(self.scalar_color_name) # self.graph.point_scalars_to_edge_scalars(name=self.scalar_color_name)
            else:
                self.edge_color = np.ones(nedge)
        else:
            self.edge_color = edge_color
                        
        if self.edge_color is None:
            #print(f'Error: no edge color provided!')
            return
                     
        if self.log_color:   
            self.edge_color = np.abs(self.edge_color)
            self.edge_color[self.edge_color==0.] = 1e-6
            self.edge_color = np.log(self.edge_color)
                            
        # Set range
        self.cmap_range = arr(self.cmap_range)
        if self.cmap_range[0] is None:
            self.cmap_range[0] = self.edge_color.min()
        if self.cmap_range[1] is None:
            self.cmap_range[1] = self.edge_color.max()
        if self.cmap_range[0]>=self.cmap_range[1]:
            print('Error: Invalid Cmap range!')
            self.cmap_range[0] = 0.
            self.cmap_range[0] = 1.
        
        # Set colour map (lookup table) 
        if self.scalar_color_name=='VesselType':  
            cols = np.zeros([nedgepoint,3]) 
            s_art = np.where(self.edge_color==0) 
            cols[s_art[0],:] = [1.,0.,0.]
            s_vei = np.where(self.edge_color==1) 
            cols[s_vei[0],:] = [0.,0.,1.]
            s_cap = np.where(self.edge_color==2) 
            cols[s_cap[0],:] = [0.5,0.5,0.5]
            s_oth = np.where((self.edge_color>2) | (self.edge_color<0)) 
            cols[s_oth[0],:] = [1.,1.,1.]
        else:
            import matplotlib.pyplot as plt
            cmapObj = plt.cm.get_cmap(self.cmap)
            col_inds = np.clip((self.edge_color-self.cmap_range[0]) / (self.cmap_range[1]-self.cmap_range[0]),0.,1.)
            cols = cmapObj(col_inds)[:,0:3]

        if len(self.edge_highlight)>0:
            self.edge_highlight = arr(self.edge_highlight)
            cols[self.edge_highlight] = self.highlight_color

        for i in sind[0]:
            cyl = self.cylinders[i]
            if cyl is not None:
                if self.engine=='open3d':
                    cyl.paint_uniform_color(cols[i])
                elif self.engine=='pyvista':
                    cyl['color'] = np.zeros(cyl.n_points) + self.edge_color[i]
            
        self.combine_cylinders()
        
        if update:
            self.update()
        
    def create_plot_cylinders(self):
    
        nc = self.graph.get_data('VertexCoordinates')
        points = self.graph.get_data('EdgePointCoordinates')
        npoints = self.graph.get_data('NumEdgePoints')
        conns = self.graph.get_data('EdgeConnectivity')
        radField = self.graph.get_radius_field()
        if radField is None:
            print('Could not locate vessel radius data!')
            radii = np.ones(points.shape[0])
        else:
            radii = radField['data']
    
        nedge = self.graph.nedge
        if self.edge_filter is None:
            self.edge_filter = np.ones(conns.shape[0],dtype='bool')
        if self.node_filter is None:
            self.node_filter = np.ones(nc.shape[0],dtype='bool')

        self.cylinders = np.empty(self.graph.nedgepoint,dtype='object') # [None]*self.graph.nedgepoint
            
        print('Preparing graph (creating cylinders)...')
        # Create cylinders
        excluded = []
        for i in trange(nedge):
            excl = True
            if self.edge_filter[i] and self.node_filter[conns[i,0]] and self.node_filter[conns[i,1]]:
                i0 = np.sum(npoints[:i])
                i1 = i0+npoints[i]
                coords = points[i0:i1]
                rads = radii[i0:i1]

                if np.any(rads>=self.min_radius) and (self.domain_radius is None or np.any(np.linalg.norm(coords-self.domain_centre)<=self.domain_radius)):
                    for j in range(1,coords.shape[0]):
                        if rads[j]>=self.min_radius:
                            x0,x1 = coords[j-1],coords[j]
                            vec = x1-x0
                            height = np.linalg.norm(x1-x0)
                            
                            if height>0. and np.isfinite(height) and (self.domain_radius is None or (np.linalg.norm(x0-self.domain_centre<=self.domain_radius) and np.linalg.norm(x1-self.domain_centre<=self.domain_radius))):
                                vec = vec / height
                                if rads[j]<20. and self.radius_based_resolution:
                                    resolution = 4
                                else:
                                    resolution = self.cyl_res
                                    
                                if self.radius_scale!=1.:
                                    rad_cur = rads[j] * self.radius_scale
                                else:
                                    rad_cur = rads[j]
                                    
                                if self.engine=='open3d':
                                    cyl = o3d.geometry.TriangleMesh.create_cylinder(height=height,radius=rad_cur, resolution=resolution)
                                    translation = x0 + vec*height*0.5
                                    cyl = cyl.translate(translation, relative=False)
                                    axis, angle = align_vector_to_another(np.asarray([0.,0.,1.]), vec)
                                    if angle!=0.:
                                        axis_a = axis * angle
                                        cyl = cyl.rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cyl.get_center()) 

                                    # Default - paint white
                                    cyl.paint_uniform_color([1.,1.,1.])
                                    
                                    self.cylinders[i0+j] = cyl
                                    
                                elif self.engine=='pyvista':
                                    poly = pv.PolyData()
                                    poly.points = coords
                                    the_cell = np.arange(0, len(coords), dtype=np.int_)
                                    the_cell = np.insert(the_cell, 0, len(coords))
                                    poly.lines = the_cell
                                    poly['radius'] = rads
                                    #tube = poly.tube(radius=rads[0],n_sides=3) # scalars='stuff', 
                                    tube = pv.Spline(coords, coords.shape[0]).tube(radius=rads[0])
                                    #tube['color'] = np.linspace(1,1,tube.n_points)
                                    self.cylinders[i0+j] = tube
                                    
                                excl = False
                
        self.cylinder_inds = np.where(self.cylinders)
        
    def combine_cylinders(self):
    
        if self.engine=='open3d':
            if self.vis is not None:
                self.vis.remove_geometry(self.cylinders_combined)
        
            # Combine (select active cylinder entries)
            sind = self.cylinder_inds
            if len(sind[0])>2:
                # Sum first two - otherwise combined variable becomes first cylinder reference
                self.cylinders_combined = self.cylinders[sind[0][0]] + self.cylinders[sind[0][2]]
                for cyl in self.cylinders[sind[0][2:]]:
                    if cyl is not None:
                        self.cylinders_combined += cyl
            elif len(sind[0])==2:
                self.cylinders_combined = self.cylinders[sind[0][0]] + self.cylinders[sind[0][1]] 
            elif len(sind[0])==1:
                self.cylinders_combined = self.cylinders[sind[0][0]] 
                
            if self.vis is not None:
                self.vis.add_geometry(self.cylinders_combined)
                
        elif self.engine=='pyvista':
            blocks = pv.MultiBlock(self.cylinders[self.cylinder_inds].tolist())
            self.cylinders_combined = blocks.combine()
            self.vis.add_mesh(self.cylinders_combined, smooth_shading=True, scalar_bar_args={'title':self.scalar_color_name}) # scalars='length', 
            self.vis.show()
        
    def create_plot_window(self,bgcolor=None,win_width=None,win_height=None):
    
        if win_width is not None:
            self.win_width = win_width
        if win_height is not None:
            self.win_height = win_height
        if bgcolor is not None:
            self.bgcolor = bgcolor                      
    
        if self.engine=='open3d':
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(width=self.win_width,height=self.win_height,visible=self.show)
        
            if self.cylinders_combined is not None:
                self.vis.add_geometry(self.cylinders_combined)
            
            opt = self.vis.get_render_option()
            opt.background_color = np.asarray(self.bgcolor)
        elif self.engine=='pyvista':
            self.vis = pv.Plotter(window_size=[self.win_width,self.win_height])
            self.vis.set_background(self.bgcolor)
        
    def _show_plot(self):
        if self.vis is not None:
            self.vis.run()
            self.vis.destroy_window()
        
    def update(self):
        if self.engine=='open3d':
            if self.vis is not None:
                self.vis.update_geometry(self.cylinders_combined)
        
    def screen_grab(self,fname):
        if self.engine=='open3d':
            if self.vis is not None:
                self.vis.capture_screen_image(fname,do_render=True)     
        
    def destroy_window(self):
        if self.engine=='open3d':
            if self.vis is not None:
                self.vis.destroy_window()              
