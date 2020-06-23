import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

from utils import set_seed

def exact_solution(G: nx.Graph, u0: np.ndarray, dirichlet_bc: dict={}, neumann_bc: dict={}):
	''' Returns exact solution for heat eq. on a graph as Callable
	Args:
		G: graph
		u0: initial condition
		dirichlet_bc: dict mapping boundary nodes to dirichlet conditions 
		neumann_bc: dict mapping boundary nodes to neumann conditions 
	'''
	assert len(dirichlet_bc.keys() & neumann_bc.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'
	assert u0.shape[0] == len(G.nodes), 'Incorrect number of initial conditions'

	# Using standard (unweighted) discrete Laplacian
	L = csc_matrix(nx.laplacian_matrix(G))

	for (i, node) in enumerate(G.nodes):
		if node in dirichlet_bc.keys():
			L[i,:] = 0
			u0[i] = dirichlet_bc[node]
		elif node in neumann_bc.keys():
			raise Exception('Neumann conditions not implemented yet')

	u = lambda t: u0@(expm(-L*t).T)
	return u


def plot_snapshots(G: nx.Graph, u: Callable, T: float, n: int, absolute_colors=True):
	'''
	Args:
		G: graph
		u: solution
		T: time extent
		n: # snapshots
		absolute_colors: whether to set heatmap extents globally (default: True) or based on individual snapshots
	'''
	cmap = sns.cubehelix_palette(reverse=True, as_cmap=True)
	fig, axs = plt.subplots(1, n)
	pos = nx.spring_layout(G, iterations=1000)
	for i in range(n):
		ax = axs[i]
		t = 0 + i*(T/(n-1))
		vals = u(t)
		if absolute_colors:
			vmax = max(u(0)) # Assume no other driving
			nx.draw(G, pos=pos, cmap=cmap, node_color=vals, ax=ax, vmin=0., vmax=vmax)
		else:
			nx.draw(G, pos=pos, cmap=cmap, node_color=vals, ax=ax)
		ax.set_title(f'T={t}')

def plot_rasterized(G: nx.Graph, u: Callable, T: float):
	'''
	Args:
		G: graph
		u: solution
		T: time extent
	'''
	tspace = np.linspace(0, T, 100)
	sol = np.array(list(map(u, tspace))).squeeze()
	plt.figure(figsize=(20,40))
	cmap = sns.cubehelix_palette(reverse=True, as_cmap=True)
	sns.heatmap(sol.T, cmap=cmap, vmin=0., vmax=1.)
	plt.title(f'Heat equation for T={T} seconds')