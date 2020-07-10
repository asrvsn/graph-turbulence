''' Class for specifying, running, and visualizing differential equations on graphs ''' 

import networkx as nx
import numpy as np
from typing import Callable, List
import scipy
import dill as pickle # For pickling lambdas
from networkx.readwrite.json_graph import node_link_data, node_link_graph
from matplotlib.colors import rgb2hex
from bokeh.plotting import figure, from_networkx
from bokeh.models.glyphs import Oval, MultiLine
import ujson
import pdb

from utils import *
from utils.rendering import heat_cmap

class GraphDiffEq: 
	def __init__(self, G: nx.Graph, v0: np.ndarray, l0: np.ndarray, dv_dt: Callable, dl_dt: Callable, t0: float=0., desc: str=None, solver: str='dopri5', **solver_args):
		''' Initialize a graph-domain differential equation with time-varying values on vertices and edges.
		Args:
			G: graph
			v0: initial values for vertices
			l0: initial values for edges
			dv_dt: difference equation for vertices taking args (time, vertex_values, edge_values)
			dl_dt: difference equation for edges taking args (time, vertex_values, edge_values)
			t0: start time
			desc: diff.eq description (will be displayed if rendered)
			solver: scipy.integrate.ode solver designation (default: dopri5 aka Runge-Kutta 4/5)
			**solver_args: additional arguments to be passed to integrator
		'''
		# Consistency checks
		assert len(G) == len(v0) == len(dv_dt(t0, v0, l0)), 'Vertex dimensions do not match'
		assert len(G.edges) == len(l0) == len(dl_dt(t0, v0, l0)), 'Edge dimensions do not match'
		self.G = G

		# Initialize integrator (allow replacement of difference values at runtime)
		def dv_dt_with_replacement(t: float, v: np.ndarray, l: np.ndarray, replace_at: list, replace_with: list):
			return replace(dv_dt(t, v, l), replace_at, replace_with)
		def dl_dt_with_replacement(t: float, l: np.ndarray, v: np.ndarray, replace_at: list, replace_with: list):
			return replace(dl_dt(t, v, l), replace_at, replace_with)

		self.v_ode = scipy.integrate.ode(dv_dt_with_replacement).set_integrator(solver, **solver_args).set_initial_value(v0, t0).set_f_params(l0, [], [])
		self.l_ode = scipy.integrate.ode(dl_dt_with_replacement).set_integrator(solver, **solver_args).set_initial_value(l0, t0).set_f_params(v0, [], [])
		self.vertex_bc = ([], []) # Substitions in vertex difference equation
		self.edge_bc = ([], []) # Substitions in edge difference equation

		# Rendering
		self.plot = None
		self.desc = desc

	@property 
	def t(self):
		''' Current time '''
		return self.v_ode.t

	@property
	def v(self):
		''' Current vertex values '''
		return self.v_ode.y

	@property 
	def l(self):
		''' Current edge values ''' 
		return self.l_ode.y

	def set_vertex_boundary(self, dirichlet: dict={}, neumann: dict={}):
		''' Set boundary conditions on vertices ''' 
		assert len(dirichlet.keys() & neumann.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'
		nodes = list(self.G.nodes)
		d_vals = list(dirichlet.values())
		n_vals = list(neumann.values())
		d_idx = [nodes.index(n) for n in dirichlet.keys()]
		n_idx = [nodes.index(n) for n in neumann.keys()]
		# Modify difference equation with boundary conditions -- zero difference for dirichlet, specified difference for neumann
		replace_at = d_idx + n_idx 
		replace_with = np.zeros(len(d_idx)).tolist() + n_vals 
		self.vertex_bc = (replace_at, replace_with)
		# Reset the current value per Dirichlet conditions
		self.v_ode.set_initial_value(replace(self.v, d_idx, d_vals), self.t)

	def set_edge_boundary(self, dirichlet: dict={}, neumann: dict={}):
		''' Set boundary conditions on edges ''' 
		assert len(dirichlet.keys() & neumann.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'
		edges = list(self.G.edges)
		d_vals = list(dirichlet.values())
		n_vals = list(neumann.values())
		d_idx = [edges.index(n) for n in dirichlet.keys()]
		n_idx = [edges.index(n) for n in neumann.keys()]
		# Modify difference equation with boundary conditions -- zero difference for dirichlet, specified difference for neumann
		replace_at = d_idx + n_idx 
		replace_with = np.zeros(len(d_idx)).tolist() + n_vals 
		self.edge_bc = (replace_at, replace_with)
		# Reset the current value per Dirichlet conditions
		self.l_ode.set_initial_value(replace(self.l, d_idx, d_vals), self.t)

	def step(self, dt: float):
		''' Solve system forward for dt time ''' 
		self.v_ode.set_f_params(self.l, self.vertex_bc[0], self.vertex_bc[1]) # Pass edge values, boundary conditions to vertex function
		self.v_ode.integrate(self.t + dt)
		self.l_ode.set_f_params(self.v, self.edge_bc[0], self.edge_bc[1]) # Pass vertex values, boundary conditions to edge function
		self.l_ode.integrate(self.t + dt)
		if self.plot is not None:
			self._render()

	def create_plot(self): 
		''' Create plot for rendering with Bokeh ''' 
		G = nx.convert_node_labels_to_integers(self.G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
		n_v = len(G)
		n_e = len(G.edges())
		tools = "ypan,ywheel_zoom,ywheel_pan,ybox_zoom,reset"
		layout = nx.spring_layout(G, scale=0.9, center=(0,0), iterations=500, seed=1)
		plot = figure(title=self.desc, x_range=(-1.1,1.1), y_range=(-1.1,1.1), tools=tools, toolbar_location=None)
		plot.axis.visible = None
		renderer = from_networkx(G, layout)
		renderer.node_renderer.data_source.data['color'] = ['#000000']*n_v
		renderer.node_renderer.glyph = Oval(height=0.08, width=0.08, fill_color='color')
		renderer.edge_renderer.data_source.data['alpha'] = [0.5]*n_e
		renderer.edge_renderer.glyph = MultiLine(line_color='#000000', line_alpha='alpha', line_width=5)
		# # TODO: render edge direction
		plot.renderers.append(renderer)
		self.plot = plot
		return plot

	def _render(self):
		''' Draw glyphs based on current graph values ''' 
		colors = [rgb2hex(heat_cmap(x)) for x in self.v]
		alphas = [np.abs(0.8*x)+0.2 for x in self.l]
		self.plot.renderers[0].node_renderer.data_source.data['color'] = colors
		self.plot.renderers[0].edge_renderer.data_source.data['alpha'] = alphas
		# # TODO: render edge direction
