''' Heat equation on graphs ''' 

import networkx as nx
import numpy as np
from itertools import repeat
import pdb
import colorcet as cc

from core.observables import *
from rendering import *

n = 10
G = nx.grid_2d_graph(n, n)

# Boundaries
upper, lower, left, right = [(0,j) for j in range(n)], [(n-1,j) for j in range(n)], [(i,0) for i in range(n)], [(i,n-1) for i in range(n)]

def sys1():
	c = 1.0
	ampl = VertexObservable(G, desc='Amplitude')
	ampl.set_ode(lambda t: (c**2)*ampl.laplacian(), order=2)
	ampl.set_initial(
		y0=lambda pos: pos[0]*pos[1]*(n-pos[0])*(n-pos[1]) / (n**3),
		y0_1=lambda _: 0.0, 
	)
	ampl.set_boundary(
		dirichlet_values=dict(zip(upper + lower + left + right, repeat(0.)))
	)
	ampl.set_render_params(palette=cc.bgy, lo=-ampl.y.max(), hi=ampl.y.max())

	sys = System([ampl], desc=f'Wave equation (c={c}) with fixed boundary conditions')
	return sys

render_live([sys1()])