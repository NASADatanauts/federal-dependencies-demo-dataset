import PAMHD
import asdf
import sys, argparse
import numpy as np


def main(argv):
	parser = argparse.ArgumentParser(description="Interpolates variables onto grid.")
	parser.add_argument("-v", "--verbose", action="count", default=0, help = 'verbosity of output')
	parser.add_argument("input_file", metavar = 'full/path/to/input_file.dc', type=str, help="PAMHD raw output file")
	parser.add_argument("-output_file", metavar = 'path/to/output_file.asdf', default = None)
	parser.add_argument("-ginfo","--global-info", action='store_true', help = 'print global attributes')
	parser.add_argument("-db", "--debug", default = False, help = 'debugging flag')
	# include version argument



	args = parser.parse_args()

	if args.verbose:
		print "Using asdf version", asdf.__version__ 


	if args.output_file is None:
		filename = args.input_file + '.asdf'
	else:
		filename = args.output_file

	if args.verbose:
		print "input file:", args.input_file
		print "output file:", filename


	mhd_dict = {}

	sim_params = PAMHD.load(args.input_file, mhd_dict)

	global_attributes = dict(sim_params._asdict())

	global_attributes['model_name'] = 'PAMHD'

	if args.global_info:
		for attr in global_attributes:
			print global_attributes[attr]


	mhd_array = get_mhd_array(mhd_dict)


	
	grid = get_regular_grid(mhd_array)



	# grid['global_attributes'] = global_attributes
	
	mhd_array = grid['mhd_sorted']

	# mhd_array.reshape(grid['resolution'])

	components = ['c0', 'c1', 'c2']

	variables = get_variables(mhd_array, components, grid)
	coordinates = get_coordinates()
	fields = get_fields()

	from collections import OrderedDict #why isn't this working?
	tree_data = OrderedDict()
	tree_data['global_attributes'] = global_attributes
	tree_data['fields'] = fields
	tree_data['coordinates'] = coordinates
	tree_data['variables'] = variables


	write_to(tree_data, components, variables, filename, verbose = args.verbose)

	
def get_coordinates():
	return dict(	magnetosphere = dict(positions=['c0','c1','c2'],
	                                         reference_frame = 'GSM', #?
	                                         representation = 'CARTESIAN',
	                                         ))

def get_fields():
	return dict( mass_density = dict( coordinates = 'magnetosphere',
	                                   variables = ['mass_density'],
	                                   reference_frame = 'GSM',
	                                   representation = 'SCALAR'
	                                  ),
	               p	        = dict(	coordinates='magnetosphere',
											variables = ['px', 'py', 'pz'],
											reference_frame = 'GSM',
											representation = 'CARTESIAN',
											interpolation_type = 'REGULAR_ND_GRID'
											),
	               total_energy_density = dict(coordinates = 'magnetosphere',
	                                          variables = ['total_energy_density'],
	                                          refrence_frame = 'GSM',
	                                          representation = 'SCALAR'),
	          	   B 			= dict( variables = ['bx','by', 'bz'], 
	          								coordinates='magnetosphere',
	          								reference_frame = 'GSM',
	          								representation = 'CARTESIAN',
	          								interpolation_type = 'REGULAR_ND_GRID'
	          								),
	              J			= dict( variables = ['jx','jy', 'jz'], 
	          								coordinates='magnetosphere',
	          								reference_frame = 'GSM',
	          								representation = 'CARTESIAN',
	          								interpolation_type = 'REGULAR_ND_GRID'
	          								),
	              electric_resistivity = dict( coordinates = 'magnetosphere',
	                                   variables = ['electric_resistivity'],
	                                   reference_frame = 'GSM',
	                                   representation = 'SCALAR'
	                                  ),
	             )


def write_to(tree_data,components, variables, filename, verbose = False):
	print 'writing to', filename
	ff = asdf.AsdfFile(tree_data)

	for v in variables:
	    if v not in components:
	        ff.set_array_storage(variables[v]['data'], 'external')
	    else:
	        ff.set_array_storage(variables[v]['data'], 'inline')
	        
	ff.write_to(filename)


def get_regular_grid(mhd_array):
    np.sort(mhd_array, order=['c0','c1','c2'])
    unique_points = np.unique(mhd_array.c0), np.unique(mhd_array.c1), np.unique(mhd_array.c2)

    res_3D = tuple([len(u) for u in unique_points]) # (n0,n1,n2)

    _select = tuple([i for i in range(3) if res_3D[i] > 1]) # e.g. (0,2) if n1 = 1 

    if res_3D[0]*res_3D[1]*res_3D[2] == len(mhd_array):
        resolution = tuple([res_3D[s] for s in _select]) # e.g. (n0,n2) if n1 = 1
        mhd_array.reshape(resolution)
    else:
        raise ArithmeticError('Could not construct regular grid')
        
    return dict(c0 = unique_points[0], c1 = unique_points[1], c2 = unique_points[2], resolution = resolution, mhd_sorted = mhd_array)

def get_mhd_array(mhd_dict):
	mhd_data = []
	for cell_id, mhd_array in mhd_dict.items():
		for mhd in mhd_array:
			mhd_data.append(tuple(list(mhd)+[cell_id]))

	mhd_dtype = np.dtype([
							('c0', float),
							('c1', float), 
							('c2',float),
							('mass_density', float),
							('px',float),
							('py',float),
							('pz',float),
							('total_energy_density',float),
							('bx',float),
							('by',float),
							('bz',float),
							('jx',float),
							('jy',float),
							('jz',float),
							('cell_type',int),
							('mpi_rank',int),
							('electric_resistivity',float),
							('cell_id',int)
							])
	mhd_data = np.array(mhd_data, dtype=mhd_dtype)

	return mhd_data.view(np.recarray)

def get_variables(mhd_array, components, grid):

	variables = {}

	for var_name in mhd_array.dtype.names:
	    if var_name not in components:
	        data = mhd_array[var_name]
	        variables[var_name] = dict(data=data,attributes = dict(actual_min = data.min(), actual_max = data.max()))

	units = dict(c0='m', c1='m',c2='m', 
	             mass_density='kg/m^3', 
	             px='kg-m/s', py = 'kg-m/s', pz = 'kg-m/s', 
	             total_energy_density = 'J/m^3', 
	             bx = 'T', by ='T', bz = 'T',
	             jx = 'A', jy = 'A', jz = 'A',
	             cell_type = '1',
	             mpi_rank = '1',
	             electric_resistivity = 'ohm-m',
	             cell_id = '1')

	# put 1-d grid arrays in variables dict, ignoring expanded original arrays
	
	for c in components:
	    data = grid[c]
	    variables[c] = dict(data=data, attributes = dict(actual_min = data.min(), actual_max = data.max()))

	# set units
	for var_name, unit in units.items():
	    variables[var_name]['attributes']['unit'] = unit

	return variables

def set_positions(self):
	np.sort(self.mhd_data, order=['c0','c1','c2'])
	self.unique_points = np.unique(self.mhd_data.c0), np.unique(self.mhd_data.c1), np.unique(self.mhd_data.c2)
	self.res_3D = tuple([len(u) for u in self.unique_points]) # (n0,n1,n2)

	self._select = tuple([i for i in range(3) if self.res_3D[i] > 1]) # e.g. (0,2) if n1 = 1 
	
	if self.res_3D[0]*self.res_3D[1]*self.res_3D[2] == len(self.mhd_data):
		self.resolution = tuple([self.res_3D[s] for s in self._select]) # e.g. (n0,n2) if n1 = 1
		self.sparse_grid = tuple([self.unique_points[s] for s in self._select])
		self.mhd_data = self.mhd_data.reshape(self.resolution)
	else:
		raise ArithmeticError('Could not construct regular grid')


if __name__ == '__main__':
    main(sys.argv[1:])
