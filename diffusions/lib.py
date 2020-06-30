import networkx as nx
from networkx.readwrite.json_graph import node_link_data
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
from scipy.integrate import odeint
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import time
from pathlib import Path

from utils import map_1d_to_nd, map_nd_to_1d
from utils.bokeh import serve_and_open
from utils.zmq import pubsub_tx

heat_cmap = sns.cubehelix_palette(reverse=True, as_cmap=True)

def constrained_laplacian(G: nx.Graph, dirichlet_bc: dict={}, neumann_bc: dict={}):
	''' Returns graph Laplacian which is (a) unconstrained or solves (b) Dirichlet problem or (c) Neumann problem ''' 
	assert len(dirichlet_bc.keys() & neumann_bc.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'

	# Using standard (unweighted) discrete Laplacian
	L = csc_matrix(nx.laplacian_matrix(G))

	for (i, node) in enumerate(G.nodes):
		if node in dirichlet_bc.keys():
			L[i,:] = 0
		elif node in neumann_bc.keys():
			raise Exception('Neumann conditions not implemented yet') # TODO

	return L


def match_ic_bc(u0: np.ndarray, keyfunc: Callable, dirichlet_bc: dict={}, neumann_bc: dict={}):
	''' Match initial conditions w/ specified boundary conditions ''' 
	for i in range(len(u0)):
		k = keyfunc(i)
		if k in dirichlet_bc.keys():
			u0[i] = dirichlet_bc[k]
	return u0

def solve_exact(G: nx.Graph, u0: np.ndarray, dirichlet_bc: dict={}, neumann_bc: dict={}, alpha=-1):
	''' Returns exact solution for heat eq. on a graph as Callable
	Args:
		G: graph
		u0: initial condition
		dirichlet_bc: dict mapping boundary nodes to dirichlet conditions 
		neumann_bc: dict mapping boundary nodes to neumann conditions 
	'''
	assert u0.shape[0] == len(G.nodes), 'Incorrect number of initial conditions'
	nodelist = list(G.nodes)
	L = constrained_laplacian(G, dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	u0 = match_ic_bc(u0, lambda i: nodelist[i], dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	u = lambda t: u0@(expm(alpha*L*t).T)
	return u


def solve_numeric(G: nx.Graph, u0: np.ndarray, dirichlet_bc: dict={}, neumann_bc: dict={}, dt: float=1e-3, T_max: float=1.0, alpha=-1):
	''' Returns numerically integrated solution for heat eq. on a graph as Callable (using scipy, 'LSODA')
	Args:
		G: graph
		u0: initial condition
		dirichlet_bc: dict mapping boundary nodes to dirichlet conditions 
		neumann_bc: dict mapping boundary nodes to neumann conditions 
		dt: time discretization (i.e. when calling u(), this is the resolution)
		T_max: end time step to solve out to
	'''
	assert u0.shape[0] == len(G.nodes), 'Incorrect number of initial conditions'
	nodelist = list(G.nodes)
	L = constrained_laplacian(G, dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	f = lambda u, t: alpha*u@L.T
	u0 = match_ic_bc(u0, lambda i: nodelist[i], dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	tspace = np.linspace(0, T, T/dt)
	uspace = odeint(f, u0, tspace)
	u = lambda t: uspace[int(round(t/dt))]
	return u


def solve_lattice(dx: tuple, nx: tuple, u0: np.ndarray, dirichlet_bc: dict={}, neumann_bc: dict={}, dt: float=1e-3, T_max: float=1.0, alpha=-1):
	''' Solve heat eq by finite difference discretization of spatial derivatives (method of lines). 
	Use for comparison with graph solutions. 
	Args:
		dx: tuple of differences (dx1, dx2, ... dxn)
		nx: tuple of # points (mx1, mx2, ... mxn)
		u0: initial condition
		dirichlet_bc: dict mapping boundary nodes to dirichlet conditions 
		neumann_bc: dict mapping boundary nodes to neumann conditions 
		dt: time discretization (i.e. when calling u(), this is the resolution)
		T_max: end time step to solve out to

	TODOs: 
		* neumann condition
		* banded jacobian?
	'''
	assert u0.shape[0] == np.prod(nx), 'Incorrect number of initial conditions'
	assert len(dx) == len(nx), 'Dimensions of steps and extents mismatch'
	assert len(dirichlet_bc.keys() & neumann_bc.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'
	d = len(dx) # Dimension of problem
	def f(u, t): # Finite difference ODE
		dudt = np.empty_like(u)
		for i in range(len(u)):
			coord = map_1d_to_nd(i) # embedding nd coords in 1d for odeint() API & to match other funcs
			# Check boundary conditions
			if coord in dirichlet_bc.keys():
				dudt[i] = 0
			elif coord in neumann_bc.keys():
				raise Exception('Neumann conditions not implemented yet') # TODO
			else:
			# Compute discrete Laplacian
				Lu = 0
				for c_i in range(d):
					below = map_nd_to_1d((*coord[:c_i], coord[c_i]-1, *coord[c_i+1:]))
					above = map_nd_to_1d((*coord[:c_i], coord[c_i]+1, *coord[c_i+1:]))
					Lu += (u[below] - 2*u[i] + u[above]) / (dx[c_i] ** 2)
				dudt[i] = alpha*Lu
		return dudt
	u0 = match_ic_bc(u0, map_1d_to_nd, dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	tspace = np.linspace(0, T, T/dt)
	uspace = odeint(f, u0, tspace)
	u = lambda t: uspace[int(round(t/dt))]
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
	fig, axs = plt.subplots(1, n)
	pos = nx.spring_layout(G, iterations=1000)
	for i in range(n):
		ax = axs[i]
		t = 0 + i*(T/(n-1))
		vals = u(t)
		if absolute_colors:
			vmax = max(u(0)) # Assume no other driving
			nx.draw(G, pos=pos, cmap=heat_cmap, node_color=vals, ax=ax, vmin=0., vmax=vmax)
		else:
			nx.draw(G, pos=pos, cmap=heat_cmap, node_color=vals, ax=ax)
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
	sns.heatmap(sol.T, cmap=heat_cmap)
	plt.title(f'Heat equation for T={T} seconds')


def plot_live(G: nx.Graph, u: Callable, T: float, dt: float=0.1, speed: float=0.2):
	'''Plot live simulation with Bokeh.
	Args:
		G: graph
		u: solution
		T: time extent
		dt: timedelta for frames
		speed: framerate (1x is realtime)
	'''
	path = str(Path(__file__).parent / 'bokeh_app.py')
	proc = serve_and_open(path)
	ctx, tx = pubsub_tx()

	try:
		print('Waiting for server to initialize...')
		time.sleep(2) 
		tx({'tag': 'init', 'graph': node_link_data(G)})

		t = 0.
		while t <= T:
			vals = u(t).tolist()
			tx({'tag': 'data', 't': t, 'data': vals})
			t += dt
			time.sleep(dt / speed)
		print('Finished rendering.')
		while True: time.sleep(1) # Let bokeh continue to handle interactivity while we wait
	finally:
		ctx.destroy()
		proc.terminate()