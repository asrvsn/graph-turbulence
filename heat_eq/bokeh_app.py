import ujson
import zmq
import random
import numpy as np
from functools import partial
from threading import Thread
from tornado import gen
import networkx as nx
from networkx.readwrite.json_graph import node_link_graph

from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import ColumnDataSource, Slider, Select, Button
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models.widgets import Div
from bokeh.plotting import from_networkx

from utils.zmq import pubsub_rx
from heat_eq.lib import heat_cmap

# Save curdoc() to make sure all threads see the same document.
doc = curdoc()

'''
Plot configuration
'''
tools = "ypan,ywheel_zoom,ywheel_pan,ybox_zoom,reset"
height = 1000
width = 1000
config = {
	'vmax': 1.0
}


'''
Layout UI
'''
plot = figure(title='Heat Equation', plot_height=height, plot_width=width, tools=tools)
t1 = Div(text='Time:', style={'font-size':'150%'})
t2 = Div(text='N/A', style={'font-size':'150%'})
root = column(
	row([t1, t2]),
	row([plot])
)
root.sizing_mode = 'stretch_both'
doc.add_root(root)

doc.title = 'Bokeh Server'


'''
Process messages
'''

@gen.coroutine
def update(msg):
	if msg['tag'] == 'init':
		G = node_link_graph(ujson.loads(msg['graph_data']))
		layout_func = lambda g: nx.spring_layout(g, iterations=1000)
		renderer = from_networkx(G, layout_func, scale=2, center=(0,0))
		renderer.node_renderer.glyph = Oval(size=1, fill_color='fill_color')
		renderer.node_renderer.data_source.data = dict(
			index=list(range(len(G))),
			fill_color='#CCCCCC'
		)
		plot.renderers.append(renderer)
		if 'vmax' in msg:
			config['vmax'] = msg['vmax']
	elif msg['tag'] == 'data':
		t2.text = str(msg['t'])
		print(msg['data'])
		plot.renderers[0].node_renderer.data_source.data['fill_color'] = [heat_cmap(v / config['vmax']) for v in msg['data']]
 
def stream_data():
	ctx, rx = pubsub_rx()
	try:
		while True:
			msg = rx()
			doc.add_next_tick_callback(partial(update, msg=msg))
	finally:
		ctx.destroy()

thread = Thread(target=stream_data)
thread.start()
