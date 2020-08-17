''' Heat equation on graphs ''' 

import networkx as nx
import numpy as np
from itertools import repeat
import pdb

from core.observables import *
from core.fd import fd_diffusion
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
	temp.set_nonphysical(lambda y: (np.abs(y) > 1.0).any())

	sys = System([temp], desc=f'Heat equation (alpha={alpha}) with non-uniform Dirichlet boundary conditions')
	return sys

def sys1_decomp():
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
	return [sys, cycle_basis_view(temp)]

def sys1_finite():
	''' Finite-difference version of sys1 ''' 
	alpha = 1.0
	dx = 1.0
	dirichlet = dict(zip(upper + lower + left + right, [0.]*n + [0.5]*n + [1.]*n + [0.]*n))
	f = fd_diffusion((dx, dx), (n, n), dirichlet_bc=dirichlet, alpha=alpha)

	temp = VertexObservable(G, desc='Temperature')
	temp.set_ode(lambda t: f(t, temp.y))
	temp.set_initial(
		y0=lambda _: 0.0
	)
	temp.set_boundary(dirichlet_values=dirichlet)

	sys = System([temp], desc=f'(Finite-difference) Heat equation (alpha={alpha}) with non-uniform Dirichlet boundary conditions')
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

def sys2_finite():
	alpha = 1.0
	dx = 1.0
	dirichlet = dict(zip(upper + lower + left, repeat(1.0)))
	neumann = dict(zip(right[1:-1], repeat(-0.1)))
	f = fd_diffusion((dx, dx), (n, n), dirichlet_bc=dirichlet, neumann_bc=neumann, alpha=alpha)

	temp = VertexObservable(G, desc='Temperature')
	temp.set_ode(lambda t: f(t, temp.y))
	temp.set_initial(
		y0=lambda _: 1.0
	)
	temp.set_boundary(dirichlet_values=dirichlet, neumann_values=neumann)

	sys = System([temp], desc=f'(Finite-difference) Heat equation (alpha={alpha}) with mixed boundary conditions')
	return sys

def sys3():
	alpha = 1.0
	temp = VertexObservable(G, desc='Temperature')
	temp.set_ode(lambda t: alpha*laplacian(temp))
	temp.set_initial(
		y0=lambda _: 1.0
	)
	temp.set_boundary(
		dirichlet_values=dict(zip(upper + lower + left, [1.0]*n + [1.0]*n + [0.2]*n)), 
		neumann_values=dict(zip(right[1:-1], repeat(0.)))
	)

	sys = System([temp], desc=f'Heat equation (alpha={alpha}) with 3 fixed sides and 1 insulated side')
	return sys

def sys4():
	G = nx.random_geometric_graph(100, 0.125)
	dV = random.choices(list(G.nodes()), k=20) # boundary
	alpha = 1.0
	temp = VertexObservable(G, desc='Temperature', default_weight=0.6)
	temp.set_ode(lambda t: alpha*laplacian(temp))
	temp.set_initial(
		y0=lambda _: 0.
	)
	temp.set_boundary(
		dirichlet_values=dict(zip(dV, repeat(0.7)))
	)
	def layout(g):
		pos = nx.get_node_attributes(g, 'pos')
		pos = {k: (np.array(v) - 0.5)*2 for k, v in pos.items()}
		return pos
	temp.set_render_params(layout_func=layout)

	sys = System([temp], desc=f'Heat equation (alpha={alpha}) on random geometric graph')
	return sys

if __name__ == '__main__':
	render_live([[sys1]])
