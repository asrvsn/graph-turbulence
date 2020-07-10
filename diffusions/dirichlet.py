import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
import pdb
import seaborn as sns
from itertools import repeat

from utils import set_seed
from diffusions.lib import *
from rendering import *

set_seed(1001)

""" Regular grid example """
n = 10
G = nx.grid_2d_graph(n,n)

# Boundary
dS = [(i,0) for i in range(n)] + [(i,n-1) for i in range(n)] + [(0,j) for j in range(n)] + [(n-1,j) for j in range(n)]

# Initial condition: no internal heat
u0 = np.zeros(n*n)

# Heated boundary
bc = dict(zip(dS, repeat(1.0)))

dt = 5e-3
T = 3.0
dx = 1.0
alpha = 1.0 # dif. constant

# Exact solution
u_ex = solve_exact(G, u0, dirichlet_bc=bc, alpha=alpha) 
# Time-discrete solution
u_num = solve_numeric(G, u0, dirichlet_bc=bc, dt=dt, T_max=T, alpha=alpha)
# Space/time-discrete solution
u_lat = solve_lattice((dx, dx), (n, n), u0, dirichlet_bc=bc, dt=dt, T_max=T, alpha=alpha)

# Plot 
renderers = [
	GraphDiffEq(G, 'exact', u_ex),
	GraphDiffEq(G, f'time-stepped (dt={dt})', u_num),
	GraphDiffEq(G, f'method of lines (dt={dt}, dx={dx})', u_lat),
]

render_live(renderers)
# plot_rmse(sols, dt, T)
# plt.show()