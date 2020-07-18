''' Heat equation on graphs ''' 

import networkx as nx
import numpy as np
from itertools import repeat
import pdb

from core.observables import *
from rendering import *

n = 10
G = nx.grid_2d_graph(n, n)

# Boundaries
upper, lower, left, right = [(0,j) for j in range(n)], [(n-1,j) for j in range(n)], [(i,0) for i in range(n)], [(i,n-1) for i in range(n)]

def sys1():
	alpha = 1.0
	temp = VertexObservable(G, desc='Temperature')
	temp.set_ode(lambda t: alpha*laplacian(temp))
	temp.set_initial(
		y0=lambda _: 0.0
	)
	temp.set_boundary(
		dirichlet_values=dict(zip(upper + lower + left + right, [0.]*n + [0.5]*n + [1.]*n + [0.]*n))
	)

	sys = System([temp], desc=f'Heat equation (alpha={alpha}) with non-uniform Dirichlet boundary conditions')
	return sys

def sys2():
	alpha = 1.0
	temp = VertexObservable(G, desc='Temperature')
	temp.set_ode(lambda t: alpha*laplacian(temp))
	temp.set_initial(
		y0=lambda _: 1.0
	)
	temp.set_boundary(
		dirichlet_values=dict(zip(upper + lower + left, repeat(1.0))), 
		neumann_values=dict(zip(right[1:-1], repeat(-0.1)))
	)

	sys = System([temp], desc=f'Heat equation (alpha={alpha}) with mixed boundary conditions')
	return sys

if __name__ == '__main__':
	render_live([sys1(), sys2()])