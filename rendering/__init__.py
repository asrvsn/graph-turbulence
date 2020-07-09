''' Solution rendering for graphs ''' 

import dill as pickle # For pickling lambdas
import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph
from typing import Callable, List
from matplotlib.colors import rgb2hex
import seaborn as sns
from pathlib import Path
import time

from bokeh.plotting import figure, from_networkx
from bokeh.models import Oval

from utils.bokeh import serve_and_open
from utils.zmq import pubsub_tx


heat_cmap = sns.cubehelix_palette(reverse=True, as_cmap=True)

class GraphRenderer: 
	def __init__(self, G: nx.Graph, title: str, vertex_func: Callable, edge_func: Callable=None, t=0.):
		self.G = G
		self.title = title
		self.vertex_func = vertex_func
		self.edge_func = edge_func
		self.t = t
		self.plot = None

	def init_plot(self): 
		G = nx.convert_node_labels_to_integers(self.G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
		n = len(G)
		tools = "ypan,ywheel_zoom,ywheel_pan,ybox_zoom,reset"
		layout = nx.spring_layout(G, scale=0.9, center=(0,0), iterations=100)
		plot = figure(title=self.title, x_range=(-1.1,1.1), y_range=(-1.1,1.1), tools=tools, toolbar_location=None)
		plot.axis.visible = None
		renderer = from_networkx(G, lambda _: layout)
		renderer.node_renderer.glyph = Oval(height=0.1, width=0.1, fill_color='color')
		renderer.node_renderer.data_source.data = dict(
			index=list(range(n)),
			color=['#000000']*n,
		)
		# TODO: handle edge rendering
		plot.renderers.append(renderer)
		self.plot = plot
		return plot

	def step(self, dt: float):
		self.t += dt
		self._render()

	def _render(self):
		if self.plot is None:	
			raise Exception('plot not initialized')
		else:
			u = self.vertex_func(self.t) 
			colors = [rgb2hex(heat_cmap(x)) for x in u]
			# TODO: handle edge functions
			self.plot.renderers[0].node_renderer.data_source.data['color'] = colors

	def to_json(self):
		return {
			'G': node_link_data(self.G),
			'title': self.title,
			'vertex_func': pickle.dumps(self.vertex_func),
			'edge_func': pickle.dumps(self.edge_func),
			't': self.t,
		}

	@staticmethod
	def from_json(self, data: dict):
		return GraphRenderer(
			node_link_graph(data['G']),
			data['title'],
			pickle.loads(data['vertex_func']),
			edge_func=pickle.loads(data['edge_func']),
			t=data['t'],
		)


def render_live(rs: List[GraphRenderer]):
	'''Plot live simulation with Bokeh.
	Args:
		rs: list of graph renderers
	'''
	path = str(Path(__file__).parent / 'bokeh_server.py')
	proc = serve_and_open(path)
	ctx, tx = pubsub_tx()

	try:
		print('Waiting for server to initialize...')
		time.sleep(2) 
		tx({'tag': 'init', 'renderers': [r.to_json() for r in rs]})
		print('Done.')
		while True: 
			time.sleep(1) # Let bokeh continue to handle interactivity while we wait
	finally:
		ctx.destroy()
		proc.terminate()