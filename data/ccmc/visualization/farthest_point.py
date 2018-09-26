import numpy as np
from pandas import DataFrame
import pandas as pd

import plotly.graph_objs as go
from plotly.offline import plot
from plotly.graph_objs import Scatter, Layout
from plotly.tools import FigureFactory as FF
from skimage import measure

from collections import namedtuple

import colorlover as cl

from ccmc import _CCMC as ccmc
from scipy.spatial import Delaunay, Voronoi
Voronoi.func_name = 'Voronoi'

import itertools, time
from collections import defaultdict

import operator

import sys, argparse


def main(argv):
	parser = argparse.ArgumentParser(description="Interpolates variables onto grid.")
	parser.add_argument("-v", "--verbose", action="count", default=0, help = 'verbosity of output')
	parser.add_argument("input_file", metavar = 'full/path/to/input_file.cdf', type=str, help="kameleon-compatible file")
	parser.add_argument("-db", "--debug", default = False, help = 'debugging flag')

	# grid options
	grid_options = parser.add_argument_group(title = 'grid options', description = 'interpolation options for a grid of points')
	grid_options.add_argument("-x", "--x-range", default = (-40, 20), type = float, nargs = 2, metavar = ('xmin','xmax'), help = "range of x")
	grid_options.add_argument("-y", "--y-range", default = (-20, 20), type = float, nargs = 2, metavar = ('ymin','ymax'), help = "range of y")
	grid_options.add_argument("-z", "--z-range", default = (-20, 20), type = float, nargs = 2, metavar = ('zmin','zmax'), help = "range of z")
	grid_options.add_argument("-r", "--r-range", default = None, type = float, nargs = 2, metavar = ('rmin', 'rmax'), help = "range of r, default is 0 to inf")
	# grid_options.add_argument("-b", "--box", type = float, nargs = 6, metavar=('xmin','xmax', 'ymin','ymax', 'zmin', 'zmax'), help = "min and max corners of the grid")
	grid_options.add_argument("-res", "--resolution", default = (50, 40, 30), type = int, nargs = '+', metavar=('nx','ny',), help = "resolution of the grid along each axis")
	grid_options.add_argument("-vars", "--variables", default = None, type=str, nargs='+',metavar = ('var1','var2',), help='list of variables to be interpolated')
	grid_options.add_argument("-slices", default = None, type = int, nargs = 3, metavar = ('islice', 'jslice', 'kslice'), help = 'list of slices to take in each dimension')


	# isovalue
	iso_options = parser.add_argument_group(title = 'isosurface options', description = 'options for generating isosurface')
	iso_options.add_argument("-iso_var", "--isosurface_variable", default = None, type = str, help = 'variable to generate isosurface from')
	iso_options.add_argument("-iso_val", "--isosurface_value", default = None, type = float, help = 'isosurface value to generate')
	iso_options.add_argument("-iso_opacity", "--isosurface_opacity", default = 1, type = float, help = 'opacity of the isosurface')

	# topology analysis
	top_options = parser.add_argument_group(title = 'topology options', description = 'options for finding separator surfaces')
	top_options.add_argument("-init_seeds", "--initial_seeds", default = "initial_seeds.csv", help = 'csv file containing initial seed points - should be far from separators')
	top_options.add_argument("-init_flines", "--initial_fieldlines", default = None, help = 'fieldline file containing output from previous run')
	top_options.add_argument("-save_flines", "--save_fieldlines", default = None, help = "output file name to store results")
	top_options.add_argument("-sep_iter", "--separator_iterations", default = None, type = int)
	args = parser.parse_args()

	kameleon = ccmc.Kameleon()

	kameleon.open(args.input_file)

	interpolator = kameleon.createNewInterpolator()

	if args.r_range is None:
		args.r_range = (0, np.inf)
	limits = DataFrame([list(args.x_range), list(args.y_range), list(args.z_range), list(args.r_range)], columns = ['min', 'max'], index = ['x','y','z','r']).T

	traces_dict = defaultdict(list)
	if args.variables is not None:
		grid = get_analysis_grid(args.resolution, args.variables, interpolator, limits)

		slice_traces = plot_grid_slices(args, grid)
		# traces += slice_traces
		traces_dict['slices'] = slice_traces

	if args.isosurface_variable is not None:
		if args.isosurface_variable in args.variables:
			pass
		else:
			grid = get_analysis_grid(args.resolution, [args.isosurface_variable], interpolator, limits)
		iso_traces = get_isosurface(grid, limits, args) 
		# traces += iso_traces
		traces_dict['isosurface'] = iso_traces


	if args.separator_iterations is not None:
		tracer = ccmc.Tracer(kameleon)
		tracer.setMaxIterations(20000)
		tracer.setDn(.2) 

		initial_seeds = pd.read_csv(args.initial_seeds)
		if args.initial_fieldlines is not None:
			initial_fieldlines = pd.read_csv(args.initial_fieldlines)
		else:
			initial_fieldlines = None

		fieldlines, separators, max_seed, voronoi = farthest_point(tracer, initial_seeds, limits, 
			min_arc_length = .5, iterations = args.separator_iterations, fieldlines = initial_fieldlines)
		
		if args.save_fieldlines is not None:
			with open(args.save_fieldlines, 'w') as f:
				f.write(fieldlines.to_csv())

		# intersection = getIntersection(separators.values())
		# intersection_neighbors = list(set.union(*intersection.values()))

		nulls = get_null_points(fieldlines)

		null_edges = plot_null_edges(fieldlines, nulls)
		null_fieldlines = plot_null_fieldlines(fieldlines, nulls, limits, legendgroup = 'nulls')
		# traces += fline_traces

		separator_edges = get_separator_edges(separators)
		closed_separators = {k:separator_edges[k] for k in separator_edges if 'closed' in k}
		separator_traces = plot_separators(voronoi, closed_separators, limits)
		print 'len(separator_traces)', len(separator_traces)	
		traces_dict['separators'] =  null_edges + null_fieldlines + separator_traces

		traces_dict['fieldlines'] = plot_fline_groups(fieldlines, cl.scales['6']['qual']['Set1'], limits)
	traces = []
	for k,v in traces_dict.items():
		traces += v

	fig = go.Figure(data=traces, layout = get_layout(kameleon, limits, traces_dict))
	plot(fig)

def plot_fline_groups(fieldlines, colors, limits):
	traces = []
	for i,t in enumerate(np.unique(fieldlines.topology.values)):
		traces.append(plot_flines(	fieldlines[fieldlines.topology == t], 
									limits, 
									color=colors[i], 
									name = t))
	return traces

def get_layout(kameleon, limits, traces_dict):
	scene=dict(	xaxis=dict(range=list(limits.x.values),autorange=False),
	            yaxis=dict(range=list(limits.y.values),autorange=False),
	            zaxis=dict(range=list(limits.z.values),autorange=False),
	            aspectmode='data',
	           )
	total_traces = 0
	slices_start = None
	isosurface_start = None

	for group_name, traces in traces_dict.items():
		print group_name, len(traces)
		if group_name == 'slices':
			if slices_start is None:
				slices_start = total_traces
		if group_name == 'isosurface':
			if isosurface_start is None:
				isosurface_start = total_traces #get first isosurface
		total_traces += len(traces)


	rename_buttons = dict(
	            x = -0.05,
	            y = .8,
	            yanchor = 'top',
	            buttons = list(
	                [
	                dict(
	                    args=[{'title':'New title'}],
	                    label = 'rename',
	                    method='relayout'
	                    )
	                    
	                ])
	            )

	updatemenus=list()

	if slices_start is not None:
		slice_buttons = dict(
		            x=-0.05,
		            y=1,
		            yanchor='top',
		            buttons = get_hide_slices(total_traces, slices_start),
		        )
		# updatemenus = list([slice_buttons])
		updatemenus += [slice_buttons]

	if isosurface_start is not None:
		updatemenus.append(dict(
					            x=-0.05,
					            y=.8,
					            yanchor='top',
					            buttons = get_hide_isosurfaces(total_traces, isosurface_start),
					        ))

	layout = Layout(
	    title=str(kameleon.getModelName()),
	    orientation='v',
	    updatemenus = updatemenus,
		scene = scene,
	)
	return layout

def get_hide_isosurfaces(total_traces, isosurface_start):
	show_all = [True for i in range(total_traces)]
	hide_isosurface = [True for i in range(total_traces)]; hide_isosurface[isosurface_start] = False
	hide_buttons = [
					dict(args = ['visible', show_all], label='show isosurface', method = 'restyle'),
					dict(args = ['visible', hide_isosurface], label='hide isosurface', method = 'restyle'),
					]
	return hide_buttons

def get_hide_slices(total_traces, slices_start):
	show_all = [True for i in range(total_traces)]
	hide_x = [True for i in range(total_traces)]; hide_x[slices_start    ] = False
	hide_y = [True for i in range(total_traces)]; hide_y[slices_start + 1] = False
	hide_z = [True for i in range(total_traces)]; hide_z[slices_start + 2] = False
	hide_all = [True for i in range(total_traces)]; hide_all[slices_start:slices_start+2] = [False for i in range(3)]

	hide_buttons=[
	        dict(
	            args=['visible', show_all],
	            label='show all slices',
	            method='restyle'
	        ),
	        dict(
	            args=['visible', hide_x],
	            label='hide: x',
	            method='restyle'
	        ),
	        dict(
	            args=['visible', hide_y],
	            label='hide: y',
	            method='restyle'
	        ),
	        dict(
	            args=['visible', hide_z],
	            label='hide: z',
	            method='restyle'
	        ),
	        dict(
	            args=['visible', hide_all],
	            label='hide all slices',
	            method='restyle'
	        ),
	    ]
	return hide_buttons

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print '%s function took %0.3f s' % (f.func_name, (time2-time1))
        return ret
    return wrap


def get_nulls(intersection):
    nulls = []
    for a,b in intersection.items():
        if len(b) == 4:
            nulls.append([a,b])
    return dict(nulls)

def plot_null_edges(fieldlines,nulls, legendgroup= 'nulls'):
	null_traces = []
	edges = np.array([(a,b, None) for a,b in itertools.combinations(range(4),2)]).ravel()

	for null in nulls:
	    vf = fieldlines.iloc[null]
	    x,y,z = [vf[c].values for c in ['x','y','z']]
	    x_,y_,z_ = [np.array([(c_[c0], c_[c1], None) for c0,c1 in itertools.combinations(range(4),2)]).ravel() for c_ in [x,y,z]]
	    null_trace = go.Scatter3d(x=x_,y=y_,z=z_, mode='lines',
	                              line=dict(
	                                color='black',
	                                width=2
	                                ),
	                              showlegend= False,
	                              legendgroup= legendgroup,
	                              name = 'null tets'
	                             )
	    null_traces.append(null_trace)

	null_traces[0]['showlegend'] = True
	return null_traces

def plot_null_fieldlines(fieldlines, nulls, limits, legendgroup= None):
	null_fieldlines = []
	null_neighbors = list(nulls.ravel())
	null_seeds = np.unique(fieldlines.seed_number.iloc[null_neighbors].values)
	null_fieldlines = [plot_flines(fieldlines, limits, seeds = null_seeds, legendgroup = legendgroup, name = 'null fieldlines')]
	# for f in iter_fieldlines(fieldlines, null_seeds):
	# 	null_fieldlines.append(plot_fline(f, f.arc_length, legendgroup = legendgroup, showlegend = True, name = 'null'))
	return null_fieldlines

@timing
def get_null_points(fieldlines):
    delaunay = Delaunay(fieldlines[['x', 'y', 'z']].values)
    dt = fieldlines.topology.values[delaunay.simplices]
    simplices = delaunay.simplices
    for c0,c1 in itertools.combinations(range(4),2):
        indices = np.where(dt[:,c0] != dt[:,c1])
        simplices = simplices[indices]
        dt = dt[indices]
    return simplices


def get_max_circumcenter(sorted_circumradii, limits):
    for circumcenter, circumradii in reversed(sorted_circumradii):
        if in_bounds(np.array(circumcenter), limits):
            return circumcenter, circumradii


def farthest_point(tracer, seeds, limits, min_arc_length = .3, 
					fieldlines = None, circumradii_dict = {}, 
					iterations = 5, focus = lambda x: True):
	Voronoi_ = timing(Voronoi)
	if fieldlines is not None:
		seeds = None
	for i in range(iterations):
		print '\nfarthest point iteration:', i
		if seeds is not None:
		    fieldlines = trace_fieldlines(tracer, seeds, limits, min_arc_length, fieldlines = fieldlines)

		points = fieldlines[['x', 'y', 'z']].values

			
		voronoi = Voronoi_(points, incremental=False, qhull_options="QJ")
		separators = get_separators(fieldlines.topology, voronoi)
		closed_separators = {t: separators[t] for t in separators.keys() if focus(t)}
		sorted_circumradii = get_all_circumradii(separators, voronoi, #circumradii_dict
		                                        )
		max_circumcenter, max_circumradius = get_max_circumcenter(sorted_circumradii, limits)
		print 'circumcenter max:', max_circumcenter, max_circumradius
		seeds = pd.Series(list(max_circumcenter)+[max_circumradius], 
		                     index=['x','y','z', 'circumradius']).to_frame().T
	return fieldlines, separators, seeds, voronoi

def focus_sw_closed(t):
	return (('closed' in t) &('solar_wind' in t))

def vertex_frame(voronoi, edges, limits):
    '''Gets unique vertices from edge list'''
    verts = np.unique(np.array(edges).ravel())

    sep_frame = pd.DataFrame(voronoi.vertices[verts], index = verts, columns = ['x','y','z'])
    sep_frame.index.name = "vor_vertex"

    sep_frame = sep_frame[(sep_frame.index >= 0) & 
              (sep_frame.x < limits.x[1]) & (sep_frame.x > limits.x[0]) &
              (sep_frame.y < limits.y[1]) & (sep_frame.y > limits.y[0]) &
              (sep_frame.z < limits.z[1]) & (sep_frame.z > limits.z[0])
             ]
    return sep_frame

def get_separator_edges(separators):
    separator_edges = {}
    for topologies, ridge_dict in separators.items():
        separator_edges[topologies] = edge_map(ridge_dict)
    return separator_edges

def plot_separators(voronoi, separator_edges, limits, size = 1):
    qual_color_8 = cl.scales['8']['qual']['Dark2']
    separator_traces = []
    for color_index, sep_type in enumerate(separator_edges.keys()): 
        marker = dict(color=qual_color_8[color_index], size=size)
        edges = separator_edges[sep_type].keys()
        
        if len(edges) > 0:
        	edge_trace = plot_edges(voronoi, edges, limits,
									color = qual_color_8[color_index], 
									name = "{0}-{1}".format(*sep_type), 
									size = size)
        	print "len(edge_trace)", len(edge_trace)
        	separator_traces.append(edge_trace)

            # vf = vertex_frame(voronoi,edges,limits)
            # separator_traces.append(go.Scatter3d(x=vf.x, y=vf.y, z=vf.z, mode='markers', 
            #                           marker=marker, name = "{0}-{1}".format(*sep_type)
            #                          )
            # )
          
    return separator_traces

def plot_edges(voronoi, edge_list, limits, legendgroup = None, visible = True, showlegend = True, color = 'black', name = 'edges', size = .5):
    edges = np.array(edge_list)
    e1, e2 = edges[:,0], edges[:,1]

    v1 = [voronoi.vertices[e1][:,i] for i in range(3)]
    v2 = [voronoi.vertices[e2][:,i] for i in range(3)]
    

    inbounds = np.ones(len(e1), dtype = bool)
    for v in (v1,v2):
        for i,c in enumerate(['x','y','z']):
            inbounds = inbounds & (v[i] < limits[c]['max'])
            inbounds = inbounds & (v[i] > limits[c]['min'])
        
    inbounds = np.where(inbounds)[0]
    
    reduced = [np.array([ (a,b,None) for a,b in zip(c1[inbounds],c2[inbounds])]).ravel() for c1, c2 in zip(v1,v2)]
    # x = np.array([ (a,b,None) for a,b in zip(x1,x2)]).ravel()
    # y = np.array([ (a,b,None) for a,b in zip(y1,y2)]).ravel()
    # z = np.array([ (a,b,None) for a,b in zip(z1,z2)]).ravel()
    x, y, z = reduced
    
    trace = go.Scatter3d(x=x,y=y,z=z, mode='lines',
                         line=dict(
	                                color=color,
	                                width=size
	                                ),
                         visible = visible,
	                              showlegend= showlegend,
	                              # legendgroup= legendgroup,
                         name = name
	                             )
    return trace

def in_bounds(c, limits):
    radius = np.sqrt(c.dot(c))
    truth =[limits.x[0] < c[0] < limits.x[1],
            limits.y[0] < c[1] < limits.y[1],
            limits.z[0] < c[2] < limits.z[1],
            limits.r[0] < radius < limits.r[1]]
    return all(truth)
    

@timing
def get_circumradii_2(ridges_dict,voronoi, circumradii = None):
    if circumradii is None:
        circumradii = {}
    for ridge, vertices in ridges_dict.items():
        point = voronoi.points[ridge[0]] # get an adjacent point
        face_vertices = voronoi.vertices[vertices]
        face_circumcenters = np.linalg.norm(face_vertices - point, axis=1)
        circumradii.update({tuple(f):c for f,c in zip(face_vertices,face_circumcenters)})
    return circumradii

@timing
def get_circumradii(delaunay, voronoi, limits, circumradii = None):
    if circumradii is None:
        circumradii = {}
    for v,d_points in delaunay.items():
        point = voronoi.points[list(d_points)[0]] #last is more likely to be not -1?
        circumcenter = tuple(voronoi.vertices[v])
        if circumradii.has_key(circumcenter):
            pass
        elif in_bounds(np.array(circumcenter), limits):
            circumradii[circumcenter] = np.linalg.norm(circumcenter - point)
    return circumradii

@timing
def get_all_circumradii(separators_dict, voronoi):
    circumradii = {}
    for topologies, ridges_dict in separators_dict.items():
        circumradii = get_circumradii_2(ridges_dict, voronoi, circumradii)
    return sorted(circumradii.items(), key=operator.itemgetter(1))

@timing
def edge_map(ridges):
    """ 
        returns an edge_map dictionary:
        - key: unique edges describing vertices of Voronoi graph [v1, v2] where v2 > v1 
        - value: [[i1, i2], [i3, i4], ..] list of ridges indexing into points of the Delaunay graph
    """
    edge_dict = defaultdict(set)
    for ridge, edges in ridges.items():
        for i in range(len(edges)):
            edge = tuple(sorted([edges[i], edges[i-1]])) #creates a sorted edge indexing into vertices
            edge_dict[edge] |= set(ridge)
    return edge_dict 

@timing
def getIntersection(ridges): 
    """Returns an intersection dictionary:
        - key: voronoi edge representing intersection of multiple ridge surfaces
        - value: list of intersecting ridges (pairs of delaunay points forming tetrahedron for two surfaces) 
        """

    intersection = defaultdict(set)
    edge_maps = []
    if len(ridges) ==1:
        for edge,value in edge_map(ridges[0]).items():
            intersection[edge].append(value)
    else: # find edges incident on multiple surfaces. 
        for ridge in ridges:
            edge_maps.append(edge_map(ridge))

        for ridge1_map, ridge2_map in itertools.combinations(edge_maps, 2): #unique combinations of ridges
            for edge2, ridge2 in ridge2_map.items():
                if ridge1_map.has_key(edge2): #found an intersecting edge
                    try:
                        intersection[edge2] |= ridge1_map[edge2]
                        intersection[edge2] |= ridge2
                    except:
                        print edge2, ridge2
                        print ridge1_map[edge2]
                        raise
                        

    return intersection

def iter_separators(series, ridges):
    for ridge in ridges:
        try:
            a, b = series.iloc[list(ridge)]
            if a != b:
                yield tuple(sorted([a,b])), ridge
        except:
            pass

@timing
def get_separators(series, voronoi):
    ridges = np.array(voronoi.ridge_dict.keys())

    t0 = series.iloc[ridges[:,0]].values
    t1 = series.iloc[ridges[:,1]].values
    w = np.where(t0 != t1)
    separator_ridges = ridges[w] # select only separator ridges
    
    t_list = [tuple(sorted(t)) for t in zip(t0[w],t1[w])]
            
    separators = defaultdict(dict)
    for topologies, r0, r1 in zip(t_list, separator_ridges[:,0], separator_ridges[:,1]):
        separators[topologies][(r0,r1)] = voronoi.ridge_dict[(r0,r1)]
#     for topologies, ridge in iter_separators(series, voronoi.ridge_dict.keys()):
#         separators[topologies][ridge] = voronoi.ridge_dict[ridge]
    return separators

@timing
def get_delaunay(ridge_dict):
    delaunay = defaultdict(set)
    for ridge, vertices in ridge_dict.items():
        for v in vertices:
            if v != -1: #only finite regions
                delaunay[v] |= set(ridge)

    return delaunay


def radius(f):
    return np.linalg.norm([f.x, f.y, f.z])

def in_range(point, limits):
    x,y,z = point
    return  (limits.x[0] < x < limits.x[1] ) & \
            (limits.y[0] < y < limits.y[1] ) & \
            (limits.z[0] < z < limits.z[1] )

def get_topology(fieldline, limits):
	boundary = limits.r['min']
	endpoints = fieldline[['x','y','z']].iloc[[0,-1]]
	in_boundary = np.linalg.norm(endpoints,axis=1) < boundary
	if np.all(in_boundary):
	    return 'closed'
	elif in_boundary[0]:
	    return 'south'
	elif in_boundary[1]:
	    return 'north'
	else:
	    return 'solar_wind'
        
def get_fieldline(tracer, seed, limits, columns = ['b', 'x', 'y', 'z', 'arc_length'], dummy = 'bx', min_arc = .1):
    fieldline = tracer.bidirectionalTrace(columns[0], seed.x, seed.y, seed.z)
    fieldline_dict = defaultdict(list)
    if fieldline.size() <= 3:
        return pd.DataFrame([],columns + ['resolution', 'radius', 'max_radius', 'min_radius', 'topology']).T
    else:
	    for i in range(fieldline.size()):
	        pos = fieldline.getPosition(i)
	        p = np.array([pos.component1, pos.component2, pos.component3])
	        if in_range(p, limits):
	            fieldline_dict[columns[0]].append(fieldline.getData(i))
	            fieldline_dict[columns[1]].append(p[0])
	            fieldline_dict[columns[2]].append(p[1])
	            fieldline_dict[columns[3]].append(p[2])
	            fieldline_dict[columns[4]].append(fieldline.getLength(i))
	            res = np.array(tracer.interpolator.interpolate_dc(dummy, *p))[1:]
	            fieldline_dict['resolution'].append(np.sqrt(res.dot(res)))
	    fieldline_frame = pd.DataFrame(fieldline_dict)
	    radii = fieldline_frame.apply(radius, axis =1)
	    try:
	        fieldline_frame['radius'] = radii
	    except:
	    	print fieldline.size()
	        print radii
	        print fieldline_frame
	        print fieldline_dict
	        print 'seed:', seed

	    fieldline_frame['max_radius'] = fieldline_frame.radius.max()
	    fieldline_frame['min_radius'] = fieldline_frame.radius.min()
	    fieldline_frame['topology'] = get_topology(fieldline_frame, limits)
	    return fieldline_frame

def resample_fieldline(fieldline_frame, min_arc_length = .3):
    if len(fieldline_frame) <= 3:
        return fieldline_frame
    u, indices = np.unique(np.floor(fieldline_frame.arc_length/min_arc_length), return_index=True)
    return fieldline_frame.iloc[indices]

def face_to_tuple(faces):
    return [(face['adjacent_cell'], face['vertices']) for face in faces]

@timing
def trace_fieldlines(tracer, seeds, limits, min_arc_length = .2, fieldlines = None):
	if fieldlines is None:
		fieldlines = pd.DataFrame(columns = ['arc_length', 
		                                     'b', 
		                                     'resolution', 
		                                     'x', 'y', 'z', 
		                                     'radius', 
		                                     'max_radius',
		                                     'min_radius',
		                                     'seed_number'])
		fieldlines.seed_number = fieldlines.seed_number.astype(np.int)
		next_seed_number = 0 #will start counting from 0
	else:
		next_seed_number = fieldlines.seed_number.iloc[-1] + 1

	for i in range(len(seeds.index)):
		try:
			resampled = resample_fieldline(get_fieldline(tracer,seeds.iloc[i], limits), min_arc_length)
		except:
			raise ArithmeticError("problem seed", i, seeds.iloc[i])

		resampled['seed_number'] = next_seed_number*np.ones(len(resampled), dtype=np.int)
		next_seed_number += 1
		fieldlines = pd.concat([fieldlines, resampled], axis = 0, ignore_index = True)
	    
	return fieldlines

def iter_fieldlines(flines, seeds = None):
    '''iterates fieldlines based on seed_number'''
    if len(flines) == 0:
        raise StopIteration
    if seeds is None:
        seed_numbers, start_indices = np.unique(flines.seed_number, return_index=True)
        end_indices = np.roll(start_indices - 1, -1)
        for start, end in zip(start_indices, end_indices):
            yield flines[start:end]
    else: #get seeds for incident points
        for seed in seeds:
            yield flines[flines.seed_number == seed]

def plot_fline(fline, colordata = None, colorscale='Viridis', size = 2, name = 'fieldline', legendgroup= None, showlegend = True):
    if colordata is None:
        colordata = fline.z
    trace = go.Scatter3d(
        x=fline.x, y=fline.y, z=fline.z,
        marker=dict(
            size=size,
            color=colordata,
            colorscale=colorscale,
        ),
        line=dict(
            color='#1f77b4',
            width=1
        ),
        name = name + ' ' + str(fline.seed_number.values[0]),
        legendgroup = legendgroup,
        showlegend = showlegend
    )
    return trace

def plot_flines(fieldlines, limits, color = None, seeds = None, colorscale='Viridis', size = 2, name = 'fieldline', legendgroup= None, showlegend = True):
	x, y, z = [], [], []
	for f in iter_fieldlines(fieldlines, seeds):
		inbounds = np.ones(len(f), dtype = bool)
		for c in ['x','y','z']:
			inbounds = inbounds & (f[c] < limits[c]['max'])
			inbounds = inbounds & (f[c] > limits[c]['min'])
		inbounds = np.where(inbounds)[0]
		x += list(f.x.iloc[inbounds]) + [None]
		y += list(f.y.iloc[inbounds]) + [None]
		z += list(f.z.iloc[inbounds]) + [None]
	trace = go.Scatter3d(
		    x=x, y=y, z=z,
		    line=dict(
		        color=color,
		        width=size
		    ),
		    mode='lines',
		    name = name,
		    legendgroup = legendgroup,
		    showlegend = showlegend
		)
	return trace	



def get_isosurface(grid, limits, args): 
	variable = grid[args.isosurface_variable].reshape(*args.resolution) #assume resolution matches grid
	if args.isosurface_value is None:
		vertices, simplices = measure.marching_cubes(variable,variable.mean())
	else:
		vertices, simplices = measure.marching_cubes(variable,args.isosurface_value)

	x,y,z = zip(*vertices)

	ni, nj, nk = args.resolution

	x_ = np.linspace(*limits.x, num = ni)
	y_ = np.linspace(*limits.y, num = nj)
	z_ = np.linspace(*limits.z, num = nk)

	iso_x = np.interp(x, np.arange(ni), x_)
	iso_y = np.interp(y, np.arange(nj), y_)
	iso_z = np.interp(z, np.arange(nk), z_)
	colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
	fig = FF.create_trisurf(x=iso_x,
	                        y=iso_y, 
	                        z=iso_z, 
	                        plot_edges=False,
	                        show_colorbar=False,
	                        colormap=colormap,
	                        simplices=simplices,
	                        title="Isosurface",
	                        )
	trace = fig.data[0]
	trace.opacity = args.isosurface_opacity
	return [trace]



	
def get_analysis_grid(resolution, variables, interpolator, limits):
	# set up analysis grid
	ni, nj, nk = resolution

	x_ = np.linspace(*limits.x, num = ni)
	y_ = np.linspace(*limits.y, num = nj)
	z_ = np.linspace(*limits.z, num = nk)
	
	index = pd.MultiIndex.from_product([range(ni),range(nj),range(nk)], names = ['i','j','k'])

	x, y, z = np.meshgrid(x_,y_, z_, indexing ='ij')

	df = pd.DataFrame(dict(x=x.ravel(), y=y.ravel(), z=z.ravel()), 
	                  columns = ['x', 'y', 'z'], index = index
	                 )

	points = df.T.head().as_matrix()

	

	variable_names = list(variables)
	var_tuple = namedtuple('Variables', variable_names)

	results = var_tuple(*interpolate_variables(interpolator.interpolate, *points, var_tuple = var_tuple))

	variables = DataFrame(results._asdict(), index =index)

	point_data = pd.concat([df,variables], axis=1)

	return point_data

def plot_grid_slices(args, grid):
	shapes = np.array(args.resolution)

	if args.slices is None:
		slices_ = shapes[0]/2, shapes[1]/2, shapes[2]/2
	else:
		slices_ = args.slices

	slice_dict = dict(zip(['i','j','k'], slices_))
	slices = [get_grid_slice(grid, slice_dict[level], level) for level in ['i','j','k']]
	traces = []

	for i,s in enumerate(slices):
	    shape = tuple([n for j,n in enumerate(shapes) if j != i]) # shape = tuple(np.roll(shapes, i)[1:])
	    trace = go.Surface(	x = s.x.reshape(*shape), 
	    					y = s.y.reshape(*shape), 
	    					z = s.z.reshape(*shape), 
	    					showscale = False,
	    					showlegend= True,
	    					legendgroup = str(i)+'slice',
	    					name = str(i) + ' slice'
	    					)
	    trace['surfacecolor'] = s.bx.reshape(*shape)
	    if i > 0: trace['showscale'] = False
	    traces.append(trace)

	return traces


def get_grid_slice(grid, index, level = 'i'):
	return grid.xs(index, level = level)

@np.vectorize
def interpolate_variables(interpolator, c0, c1, c2, var_tuple):
    """returns a named tuple of interpolated variables"""
    results = [interpolator(variable, c0, c1, c2) for variable in var_tuple._fields]
    return var_tuple(*results)

def get_frame_bbx(df):
    minmax_attr = min, max
    coord =  'x', 'y', 'z'
    box = [[attr(df[c]) for attr in minmax_attr] for c in coord]
    return  DataFrame(box, columns = ['min', 'max'], index = coord)

def get_bounding_box(kameleon, normalization = 637100000):
    minmax_attr = 'actual_min', 'actual_max'
    coord =  'x', 'y', 'z'
    box = [[getAttributeValue(kameleon.getVariableAttribute(c,attr))/normalization for attr in minmax_attr] for c in coord]
    return  DataFrame(box, columns = minmax_attr, index = coord)

def getAttributeValue(attribute):
	if attribute.getAttributeType() == ccmc.Attribute.STRING:
		return attribute.getAttributeString()
	elif attribute.getAttributeType() == ccmc.Attribute.FLOAT:
		return attribute.getAttributeFloat()
	elif attribute.getAttributeType() == ccmc.Attribute.INT:
		return attribute.getAttributeInt()

def list_variables(kameleon, verbose = True):
    nvar = kameleon.getNumberOfVariables()
    nglobal = kameleon.getNumberOfGlobalAttributes() 
    #kameleon variable attribute ids come after global ones
    if verbose: print 'number of variables in file:', nvar		
    for i in range(nvar):
        var_name = kameleon.getVariableName(i)
        vis_unit = kameleon.getVisUnit(var_name)
        print var_name, '[', vis_unit ,']'
        if verbose:
            for j in range(kameleon.getNumberOfVariableAttributes()):
                attr_name = kameleon.getVariableAttributeName(nglobal+j)
                attr = kameleon.getVariableAttribute(var_name, attr_name)
                print '\t',attr_name, ':', getAttributeValue(attr)

if __name__ == '__main__':
    main(sys.argv[1:])


