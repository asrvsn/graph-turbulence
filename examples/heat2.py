''' Example usages of GraphDiffEq ''' 

import networkx as nx
import numpy as np
from itertools import repeat
import pdb

from core.gde import *
from rendering import *

''' Heat equation '''
n = 10
G = nx.grid_2d_graph(n, n)

# Boundaries
upper, lower, left, right = [(0,j) for j in range(n)], [(n-1,j) for j in range(n)], [(i,0) for i in range(n)], [(i,n-1) for i in range(n)]

def sys1():
	temp = VertexObservable(G, desc='Temperature')
	temp.set_ode(lambda t: temp.laplacian())
	temp.set_initial(y0=0.0)
	temp.set_boundary(
		dirichlet_values=dict(zip(upper + lower + left + right, [0.]*n + [0.5]*n + [1.]*n + [0.]*n))
	)

	sys = System([temp], desc='Heat equation with non-uniform Dirichlet boundary conditions')
	return sys

def sys2():
	temp = VertexObservable(G, desc='Temperature')
	temp.set_ode(lambda t: temp.laplacian())
	temp.set_initial(y0=1.0)
	temp.set_boundary(
		dirichlet_values=dict(zip(upper + lower + left, repeat(1.0))), 
		neumann_values=dict(zip(right[1:-1], repeat(-1.0)))
	)

	sys = System([temp], desc='Heat equation with mixed boundary conditions')
	return sys

render_live([sys1(), sys2()])