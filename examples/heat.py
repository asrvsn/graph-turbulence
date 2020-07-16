''' Example usages of GraphDiffEq ''' 

import networkx as nx
import numpy as np
from itertools import repeat

from core.gde import *
from rendering import *

''' Heat equation '''
n = 10
G = nx.grid_2d_graph(n, n)


def gde1():
	# Initial conditions
	v0 = np.zeros(len(G.nodes()))
	l0 = np.zeros(len(G.edges()))

	# Difference equations
	L = nx.laplacian_matrix(G)
	dv_dt = lambda t, v, l: -v@L.T
	dl_dt = lambda t, v, l: l0

	# Set up problem
	gde = GraphDiffEq(G, v0, l0, dv_dt, dl_dt, desc='Heat equation with non-uniform Dirichlet boundary conditions')

	# Boundary conditions
	dbound = [(i,0) for i in range(n)] + [(i,n-1) for i in range(n)] + [(0,j) for j in range(n)] + [(n-1,j) for j in range(n)] 
	dvals = [0 for _ in range(n)] + [0.5 for _ in range(n)] + [1. for _ in range(n)] + [0 for _ in range(n)]
	gde.set_vertex_boundary(dirichlet=dict(zip(dbound, dvals)))

	return gde

def gde2():
	# Initial conditions
	v0 = np.ones(len(G.nodes()))
	l0 = np.zeros(len(G.edges()))

	# Difference equations
	L = nx.laplacian_matrix(G)
	dv_dt = lambda t, v, l: -v@L.T
	dl_dt = lambda t, v, l: l0

	# Set up problem
	gde = GraphDiffEq(G, v0, l0, dv_dt, dl_dt, desc='Heat equation with mixed boundary conditions')

	# Boundary conditions
	dbound = [(i,0) for i in range(n)] + [(i,n-1) for i in range(n)] + [(0,j) for j in range(n)] 
	dvals = repeat(1.0)
	nbound = [(n-1,j) for j in range(1,n-1)] 
	nvals = repeat(-0.1)
	gde.set_vertex_boundary(dirichlet=dict(zip(dbound, dvals)), neumann=dict(zip(nbound, nvals)))

	return gde


# Render live
render_bokeh([gde1(), gde2()])