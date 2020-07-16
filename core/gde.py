''' Class for specifying, running, and visualizing differential equations on graphs ''' 

import networkx as nx
import numpy as np
from typing import Callable, List
import scipy
import dill as pickle # For pickling lambdas
from networkx.readwrite.json_graph import node_link_data, node_link_graph
import ujson
import pdb
from matplotlib.colors import rgb2hex
from bokeh.plotting import figure, from_networkx
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker
from bokeh.models.glyphs import Oval, MultiLine
from bokeh.transform import linear_cmap
import colorcet as cc

from utils import *

class GraphDiffEq: 
	def __init__(
			self, G: nx.Graph, v0: np.ndarray, l0: np.ndarray, dv_dt: Callable, dl_dt: Callable, 
			t0: float=0., desc: str=None, node_palette=cc.fire, edge_palette=cc.gray, solver: str='dopri5', **solver_args
		):
		''' Initialize a graph-domain differential equation with time-varying values on vertices and edges.
		Args:
			G: graph
			v0: initial values for vertices
			l0: initial values for edges
			dv_dt: difference equation for vertices taking args (time, vertex_values, edge_values)
			dl_dt: difference equation for edges taking args (time, vertex_values, edge_values)
			t0: start time
			desc: diff.eq description (will be displayed if rendered)
			node_palette: color palette for node values 
			edge_palette: color palette for edge values
			solver: scipy.integrate.ode solver designation (default: 'dopri5' aka Runge-Kutta 4/5)
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
		self.node_palette = node_palette
		self.edge_palette = edge_palette
		self.node_lo, self.node_hi = 0., 1.
		self.edge_lo, self.edge_hi = 0., 1.

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
		# TODO: ensure consistency of edges & nodes here..
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
		# TODO: ensure consistency of edges & nodes here..
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

	def set_vertex_limits(self, lower: float=-float('inf'), upper: float=float('inf'), which: list=[]):
		pass # TODO

	def set_edge_limits(self, lower: float=-float('inf'), upper: float=float('inf'), which: list=[]):
		pass # TODO

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
		tools = 'reset'
		tooltips = [('value', '@nodes')]
		layout = nx.spring_layout(G, scale=0.9, center=(0,0), iterations=500, seed=1)
		plot = figure(title=self.desc, x_range=(-1.1,1.1), y_range=(-1.1,1.1), tools=tools, toolbar_location=None, tooltips=tooltips, aspect_ratio=1.2)
		plot.axis.visible = None
		plot.xgrid.grid_line_color = None
		plot.ygrid.grid_line_color = None
		renderer = from_networkx(G, layout)
		renderer.node_renderer.data_source.data['nodes'] = self.v.tolist()
		renderer.node_renderer.glyph = Oval(height=0.08, width=0.08, fill_color=linear_cmap('nodes', self.node_palette, self.node_lo, self.node_hi))
		renderer.edge_renderer.data_source.data['edges'] = self.l.tolist()
		renderer.edge_renderer.glyph = MultiLine(line_color=linear_cmap('edges', self.edge_palette, self.edge_lo, self.edge_hi), line_width=5)
		# # TODO: render edge direction w/ arrows somehow
		# using: https://discourse.bokeh.org/t/hover-over-tooltips-on-network-edges/2439/7
		plot.renderers.append(renderer)
		node_cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.node_palette, low=self.node_lo, high=self.node_hi), ticker=BasicTicker(), title='Node')
		edge_cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.edge_palette, low=self.edge_lo, high=self.edge_hi), ticker=BasicTicker(), title='Edge')
		plot.add_layout(node_cbar, 'right')
		plot.add_layout(edge_cbar, 'right')
		self.plot = plot
		return plot

	def _render(self):
		''' Draw glyphs based on current graph values ''' 
		self.plot.renderers[0].node_renderer.data_source.data['nodes'] = self.v
		self.plot.renderers[0].edge_renderer.data_source.data['edges'] = self.l
		# # TODO: render edge direction w/ arrows somehow

	def reset(self):
		pass # TODO
