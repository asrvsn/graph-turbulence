''' Class for specifying, running, and visualizing differential equations on graphs ''' 

import networkx as nx
import numpy as np
import scipy.integrate
import dill as pickle # For pickling lambdas
from typing import Callable, List, Tuple, Union, Any, Dict
from networkx.readwrite.json_graph import node_link_data, node_link_graph
import ujson
import pdb
from matplotlib.colors import rgb2hex
from bokeh.plotting import figure, from_networkx
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, HoverTool
from bokeh.models.glyphs import Oval, MultiLine
from bokeh.transform import linear_cmap
import colorcet as cc
from itertools import count
from enum import Enum
from abc import ABC, abstractmethod

from utils import *

''' Basic types ''' 

class Domain(Enum):
	Vertex = 0
	Edge = 1
	# Can extend...
	# Face = 2

Vertex = Any
Edge = Tuple[Vertex, Vertex]
# Face = Tuple[Vertex, ...]
GeoObject = Union[Vertex, Edge]

class Observable(ABC):
	''' A real-valued function defined on a graph 
	Note: abstract base class, not intended to be instantiated.
	''' 

	def __init__(self, G: nx.Graph, desc: str=''):
		self.G = G
		self.init_domain()
		self.n = len(self.domain)
		self.ode = None
		self.y = np.zeros(len(self.domain))
		self.t = 0.
		self.dirichlet_values = dict()
		self.neumann_values = dict()
		self.plot = None
		self.desc = desc

	@abstractmethod
	def init_domain(self):
		pass

	''' ODE ''' 

	def set_ode(self, f: Callable[[float], np.ndarray], order: int=1, solver: str='dopri5', **solver_args):

		n = len(self)

		def g(t: float, y: np.ndarray, fixed_vals: list):
			dydt = np.zeros_like(y)
			dydt[n*(order-1):] = replace(f(t), fixed_vals, [0.]*len(fixed_vals))
			for i in range(order-1):
				dydt[n*i:n*(i+1)] = y[n*(i+1):n*(i+2)]
			return dydt

		self.ode = scipy.integrate.ode(g).set_integrator(solver, **solver_args).set_f_params([])
		self.order = order

	''' Initial & Boundary Conditions ''' 

	def set_initial(self, t0: float=0., y0: Any=0., **args):
		self.t = t0
		self.y = fill_1d_array(y0)
		if self.ode is not None:
			n = len(self)
			y0 = np.concatenate(y0, np.zeros((self.order-1)*n))
			assert len(args) == self.order - 1, f'Only {len(args)+1} initial conditions provided but {self.order} needed'
			for i, arg in enumerate(args.values()):
				y0[(i+1)*n:(i+2)*n] = fill_1d_array(arg)
			self.ode.set_initial_value(y0, self.t)

	def set_boundary(self, dirichlet_values: Dict[GeoObject, float]={}, neumann_values: Dict[GeoObject, float]={}):
		assert self.ode is not None
		intersect = dirichlet_values.keys() & neumann_values.keys()
		assert len(intersect) == 0, f'Dirichlet and Neumann conditions overlap on {intersect}'
		self.dirichlet_values = dirichlet_values
		self.neumann_values = neumann_values
		fixed_idx = [self.domain[k] for k in dirichlet_values.keys()]
		y = replace(self.y, fixed_idx, list(dirichlet_values.values()))
		self.set_initial(t0=self.t, y0=y)
		self.ode.set_f_params(fixed_idx)

	''' Integration ''' 

	def step(self, dt: float):
		if self.ode is not None:
			self.ode.integrate(self.t + dt)

	def measure(self):
		if self.ode is not None:
			self.t = self.ode.t
			self.y = self.ode.y[:len(self)]
		if self.plot is not None:
			self.render()
		return self.y

	''' Operators ''' 

	@abstractmethod
	def partial(self, x1: GeoObject, x2: GeoObject) -> float:
		pass

	@abstractmethod
	def grad(self) -> np.ndarray:
		pass

	@abstractmethod
	def div(self) -> np.ndarray:
		pass

	@abstractmethod
	def laplacian(self) -> np.ndarray: 
		pass

	''' Builtins ''' 

	def __len__(self):
		return self.n

	def __call__(self, x: GeoObject):
		return self.y[self.domain[x]]

	''' Rendering ''' 

	def create_plot(self):
		''' Create plot for rendering with Bokeh ''' 
		if self.plot is None:
			G = nx.convert_node_labels_to_integers(self.G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
			layout = nx.spring_layout(G, scale=0.9, center=(0,0), iterations=500, seed=1)
			plot = figure(x_range=(-1.1,1.1), y_range=(-1.1,1.1), tooltips=[])
			plot.axis.visible = None
			plot.xgrid.grid_line_color = None
			plot.ygrid.grid_line_color = None
			renderer = from_networkx(G, layout)
			plot.renderers.append(renderer)
			self.plot = plot
			return plot

	@abstractmethod
	def render(self):
		''' Render current values to the plot ''' 
		pass

''' Observables on specific geometric objects ''' 

class VertexObservable(Observable):
	def init_domain(self):
		self.domain = dict(zip(self.G.nodes(), count()))

	def weight(self, edge: Edge) -> float:
		return 1.0 # TODO

	def partial(self, edge: Edge) -> float:
		return np.sqrt(self.weight(edge)) * self(edge[1]) - self(edge[0]) 

	def grad(self) -> np.ndarray:
		return np.array([self.partial(edge) for edge in self.G.edges()])

	def div(self) -> np.ndarray:
		raise Exception('implement me')

	def laplacian_at(self, x: Vertex) -> float:
		''' Compute Laplacian that solves Neumann problem using phantom-node method '''
		ret = sum([self.weight((x, n)) * (self(n) - self(x)) for n in self.G.neighbors(x)])
		if x in self.neumann_values:
			ret += self.neumann_values[x] # Assume phantom nodes are connected with weight 1
		return ret

	def laplacian(self) -> np.ndarray:
		return np.array([self.laplacian_at(x) for x in self.G.nodes()])

	def create_plot(self):
		super().create_plot()
		palette = cc.fire
		lo, hi = 0., 1.
		self.plot.renderers[0].node_renderer.data_source.data['vertex_data'] = self.y
		self.plot.renderers[0].node_renderer.glyph = Oval(height=0.08, width=0.08, fill_color=linear_cmap('vertex_data', palette, lo, hi))
		cbar = ColorBar(color_mapper=LinearColorMapper(palette=palette, low=lo, high=hi), ticker=BasicTicker(), title=self.desc)
		self.plot.add_layout(cbar, 'right')
		self.plot.add_tools(HoverTool(tooltips=[(self.desc, '@vertex_data')]))
		return self.plot

	def render(self):
		self.plot.renderers[0].node_renderer.data_source.data['vertex_data'] = self.y


class EdgeObservable(Observable):
	def init_domain(self):
		self.domain = dict(zip(self.G.edges(), count()))

	def weight(self, edge: Edge) -> float:
		return 1.0 # TODO

	def partial(self, edge: Edge) -> float:
		raise Exception('implement me')

	def grad(self) -> np.ndarray:
		raise Exception('implement me')

	def div_at(self, x: Vertex) -> float:
		return sum([np.sqrt(self.weight((x, n))) * (self(n) - self(x)) for n in self.G.neighbors(x)])

	def div(self) -> np.ndarray:
		return np.array([self.div_at(x) for x in self.G.nodes()])

	def create_plot(self):
		super().create_plot()
		palette = cc.gray
		lo, hi = 0., 1.
		self.plot.renderers[0].edge_renderer.data_source.data['edge_data'] = self.y
		self.plot.renderers[0].edge_renderer.glyph = MultiLine(line_color=linear_cmap('edge_data', palette, lo, hi), line_width=5)
		cbar = ColorBar(color_mapper=LinearColorMapper(palette=palette, low=lo, high=hi), ticker=BasicTicker(), title=self.desc)
		self.plot.add_layout(cbar, 'right')
		# self.plot.tooltips.append((self.desc, '@edge_data'))
		return self.plot

	def render(self):
		self.plot.renderers[0].edge_renderer.data_source.data['edge_data'] = self.y
		# TODO: render edge direction using: https://discourse.bokeh.org/t/hover-over-tooltips-on-network-edges/2439/7


''' Multiple observables on the same graph ''' 

class System:
	def __init__(self, observables: List[Observable], desc=''):
		assert len(observables) > 0, 'Pass some observables'
		assert len(observables) == len(set([type(o) for o in observables])), 'Multiple observables on the same domain not currently supported'
		assert all([observables[0].G is o.G for o in observables]), 'All observables must be on the same graph instance'
		self.observables = observables
		self.plot = None
		self.desc = desc

	def create_plot(self):
		for i, p in enumerate(self.observables):
			if i == 0:
				self.plot = p.create_plot()
				self.plot.title.text = self.desc
			else:
				p.plot = self.plot
				p.create_plot()
		return self.plot

	def step(self, dt: float):
		for obs in self.observables:
			obs.step(dt)

	def measure(self):
		ret = []
		for obs in self.observables:
			ret.append(obs.measure())
		return ret

	@property
	def t(self):
		return self.observables[0].t