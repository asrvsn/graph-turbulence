import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
import pdb
import seaborn as sns

from utils import set_seed

set_seed(1001)

""" Regular grid example """
n = 7
G = nx.grid_2d_graph(n,n)
# Boundary
dS = set([(i,0) for i in range(n)] + [(i,n-1) for i in range(n)] + [(0,j) for j in range(n)] + [(n-1,j) for j in range(n)])

# Using standard (unweighted) discrete Laplacian
L = csc_matrix(nx.laplacian_matrix(G))

# Initial condition: no internal heat
u0 = np.zeros(n*n)

# Dirichlet boundary condition
b = 1. # Heated boundary
for (i, node) in enumerate(G.nodes):
	if node in dS:
		L[i,:] = 0
		u0[i] = b

u = lambda t: u0@(expm(-L*t).T)

T = 1.
tspace = np.linspace(0, T, 100)
sol = np.array(list(map(u, tspace))).squeeze()

# pdb.set_trace()

# Draw solution 
plt.figure(figsize=(20,40))
cmap = sns.cubehelix_palette(reverse=True, as_cmap=True)
sns.heatmap(sol.T, cmap=cmap, vmin=0., vmax=1.)
plt.title(f'Heat equation on {n}x{n} regular grid for T={T} seconds')


slices = 5
fig, axs = plt.subplots(1,slices,figsize=(20,40))
pos = nx.spring_layout(G, iterations=1000)
for i in range(slices):
	ax = axs[i]
	t = 0 + i*(T/(slices-1))
	vals = u(t)
	nx.draw(G, pos=pos, cmap=cmap, node_color=vals, ax=ax) #, vmin=0., vmax=1.)
	ax.set_title(f'T={t}')


""" TODO weighted graphs, more interesting graphs... """

plt.tight_layout()
plt.show()