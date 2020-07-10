''' Example usages of GraphDiffEq ''' 

import networkx as nx
import numpy as np
from itertools import repeat
import dill as pickle

from core.gde import *
from rendering import *

''' Heat equation '''
n = 10
G = nx.grid_2d_graph(n, n)
# Boundary
dS = [(i,0) for i in range(n)] + [(i,n-1) for i in range(n)] + [(0,j) for j in range(n)] + [(n-1,j) for j in range(n)] 

# Initial conditions
v0 = np.zeros(n**2)
l0 = np.zeros(len(G.edges()))

# Difference equations
L = nx.laplacian_matrix(G)
dv_dt = lambda t, v, l: -v@L.T
# dl_dt = lambda t, v, l: l0
offsets = np.random.normal(size=len(l0))
dl_dt = lambda t, v, l: 4*np.sin(4*t + offsets) # Display some random edge behavior...

# Set up problem
gde = GraphDiffEq(G, v0, l0, dv_dt, dl_dt, desc='Heat equation with Dirichlet boundary conditions')
gde.set_vertex_boundary(dict(zip(dS, repeat(1.0))))

# gde2 = GraphDiffEq.from_json(gde.to_json())
# gde2.step(0.1)

# Render live
# s = pickle.dumps(gde)
render_bokeh([gde])