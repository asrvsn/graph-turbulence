import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
import pdb
import seaborn as sns

from utils import set_seed

set_seed(9001)

""" Regular grid example """
n = 5
G = nx.grid_2d_graph(n,n)

# Using standard (unweighted) discrete Laplacian
L = csc_matrix(nx.laplacian_matrix(G))

# Initial condition: single heated node
u0 = np.zeros((n*n,1))
u0[np.random.randint(0,n**2)] = 5.0
u = lambda t: expm(-L*t)@u0

T = 10.
tspace = np.linspace(0, T, 100)
sol = np.array(list(map(u, tspace))).squeeze()

# pdb.set_trace()

slices = 4

# Draw solution 
fig, axs = plt.subplots(1,2+slices)
cmap = sns.cubehelix_palette(dark=1, light=0, as_cmap=True)
sns.heatmap(sol.T, ax=axs[0], cmap=cmap)
axs[0].set_title(f'Rasterized solution')

pos = nx.spring_layout(G, iterations=1000)
for i in range(slices+1):
	ax = axs[i+1]
	t = 0 + i*(T/slices)
	vals = u(t).squeeze()
	nx.draw(G, pos=pos, cmap=cmap, node_color=vals, ax=ax)
	ax.set_title(f'T={t}')

fig.suptitle(f'Heat equation on {n}x{n} regular grid for T={T} seconds')

# pos = nx.spring_layout(G, iterations=1000)
# nx.draw(G,pos=pos, nodecolor='r',edge_color='b')

""" TODO weighted graphs, more interesting graphs... """


plt.show()