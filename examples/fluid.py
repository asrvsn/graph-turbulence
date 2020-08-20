''' Incompressible Navier-Stokes on graphs ''' 

import networkx as nx
import numpy as np
from itertools import repeat
import pdb
import colorcet as cc

from core import *
from rendering import *

n = 10
G = nx.grid_2d_graph(n, n)

def sys1():
	pressure = VertexObservable(G, desc='Pressure')
	velocity = EdgeObservable(G, desc='Velocity')

	velocity.set_ode(lambda t: -advect(velocity, velocity) - grad(pressure))

	pressure.set_initial(y0=lambda _: 0)
	pressure.set_boundary(dirichlet_values={(3,3): 1.0, (7,7): -1.0})
	pressure.set_render_params(lo=-1.0, hi=1.0)

	velocity.set_initial(y0=lambda _: 1.0)
	# velocity.set_boundary(dirichlet_values={((3,3), (3,4)): 1.0})
	velocity.set_render_params(palette=cc.kgy)

	couple(pressure, velocity)

	sys = System([velocity, pressure], desc=f'A test fluid flow')
	return sys

def sys2():
	set_seed(9001) # For reproducibility
	n = 200
	G = nx.random_geometric_graph(n, 0.1) 
	def layout(g):
		pos = nx.get_node_attributes(g, 'pos')
		pos = {k: (np.array(v) - 0.5)*2 for k, v in pos.items()}
		return pos

	pressure = VertexObservable(G, desc='Pressure')
	velocity = EdgeObservable(G, desc='Velocity')

	velocity.set_ode(lambda t: -advect(velocity, velocity) - grad(pressure))

	pressure.set_initial(y0=lambda _: 0)
	pressure.set_boundary(dirichlet_values={np.random.randint(n): 1.0, np.random.randint(n): -1.0})
	pressure.set_render_params(lo=-1.0, hi=1.0)

	velocity.set_initial(y0=lambda _: 1.0)
	velocity.set_render_params(palette=cc.kgy, layout_func=layout)

	couple(pressure, velocity)
	
	sys = System([velocity, pressure], desc=f'A test fluid flow')
	return sys

if __name__ == '__main__':
	render_live([sys1])