''' Swift-Hohenberg pattern formation ''' 

import networkx as nx
import numpy as np
from itertools import repeat
import pdb
import colorcet as cc

from core.observables import *
from rendering import *

n = 20
G = nx.grid_2d_graph(n, n)

def swift_hohenberg(desc: str, a: float, b: float, c: float, gam0: float, gam2: float):
	assert c > 0, 'Unstable'
	ampl = VertexObservable(G, desc='Amplitude', default_weight=0.55)
	ampl.set_ode(lambda t: -a*ampl.y - b*(ampl.y**2) -c*(ampl.y**3) + gam0*laplacian(ampl) - gam2*bilaplacian(ampl))
	ampl.set_initial(
		y0=lambda pos: np.random.uniform(),
	)
	ampl.set_render_params(palette=cc.bgy, lo=-1.5, hi=1.5, n_spring_iters=1500)

	sys = System([ampl], desc=desc)
	return sys

def stripes():
	return swift_hohenberg('Stripes', 0.9, 0, 1, -2, 1)

def spots():
	return swift_hohenberg('Spots', 1-1e-2, -1, 1, -2, 1)

def spirals():
	return swift_hohenberg('Spirals', 0.3, -1, 1, -2, 1)

def spirals_irregular():
	a, b, c, gam0, gam2 = 1-1e-2, -1, 50, -2, 1
	assert c > 0, 'Unstable'
	G = nx.random_geometric_graph(300, 0.1)
	ampl = VertexObservable(G, desc='Amplitude', default_weight=0.5)
	ampl.set_ode(lambda t: -a*ampl.y - b*(ampl.y**2) -c*(ampl.y**3) + gam0*laplacian(ampl) - gam2*bilaplacian(ampl))
	ampl.set_initial(
		y0=lambda pos: np.random.uniform(),
	)
	def layout(g):
		pos = nx.get_node_attributes(g, 'pos')
		pos = {k: (np.array(v) - 0.5)*2 for k, v in pos.items()}
		return pos
	ampl.set_render_params(palette=cc.bgy, lo=-1.3, hi=1.3, layout_func=layout)

	sys = System([ampl], desc='spots on an irregular graph')
	return sys

if __name__ == '__main__':
	render_live([spirals_irregular()])