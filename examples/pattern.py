''' Swift-Hohenberg patterns ''' 

import networkx as nx
import numpy as np
from itertools import repeat
import pdb
import colorcet as cc

from core.observables import *
from rendering import *

n = 20
G = nx.grid_2d_graph(n, n)

# Boundaries
upper, lower, left, right = [(0,j) for j in range(n)], [(n-1,j) for j in range(n)], [(i,0) for i in range(n)], [(i,n-1) for i in range(n)]

def swift_hohenberg(desc: str, a: float, b: float, c: float, gam0: float, gam2: float):
	assert c > 0, 'Unstable'
	ampl = VertexObservable(G, desc='Density', default_weight=0.55)
	ampl.set_ode(lambda t: -a*ampl.y - b*(ampl.y**2) -c*(ampl.y**3) + gam0*laplacian(ampl) - gam2*bilaplacian(ampl))
	ampl.set_initial(
		y0=lambda pos: np.random.uniform(),
	)
	ampl.set_render_params(palette=cc.bgy, lo=-2., hi=2.)

	sys = System([ampl], desc=desc)
	return sys

def stripes():
	return swift_hohenberg('Stripes', 0.9, 0, 1, -2, 1)

def spots():
	return swift_hohenberg('Spots', 1-1e-2, -1, 1, -2, 1)

def spirals():
	return swift_hohenberg('Spirals', 0.3, -1, 1, -2, 1)

if __name__ == '__main__':
	render_live([stripes(), spots(), spirals()])