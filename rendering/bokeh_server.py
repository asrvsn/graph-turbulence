import ujson
import zmq
import random
import numpy as np
from functools import partial
from threading import Thread
from tornado import gen
import networkx as nx
from networkx.readwrite.json_graph import node_link_graph
from matplotlib.colors import rgb2hex

from bokeh.plotting import figure, output_file, show, curdoc, from_networkx
from bokeh.models import ColumnDataSource, Slider, Select, Button, Oval
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models.widgets import Div

from utils.zmq import pubsub_rx
from rendering import *

# Save curdoc() to make sure all threads see the same document.
doc = curdoc()


''' 
Variables
'''

plots = {}
renderers = []
render_callback = None
dt = 1e-3

'''
UI
'''

t0 = Div(text='', style={'font-size':'200%'})
t1 = Div(text='Time:', style={'font-size':'150%'})
t2 = Div(text='N/A', style={'font-size':'150%'})
root = column(
	row([t0]),
	row([t1, t2]),
)
root.sizing_mode = 'stretch_both'
doc.add_root(root)
doc.title = 'Bokeh Server'

'''
Updates
'''

@gen.coroutine
def update():
	for r in renderers:
		r.step(dt)
	t0.text = str(renderers[0].t)

@gen.coroutine
def react(msg):
	# print(msg)
	if msg['tag'] == 'init':
		renderers = [GraphRenderer.from_json(r) for r in msg['renderers']]
		for r in renderers:
			plots[r.title] = r.init_plot()
		root.children.append(row(list(plots.values())))
		render_callback = doc.add_periodic_callback(update, int(dt/1e-3))

def stream_data():
	ctx, rx = pubsub_rx()
	try:
		while True:
			msg = rx()
			doc.add_next_tick_callback(partial(react, msg=msg))
	finally:
		ctx.destroy()

thread = Thread(target=stream_data)
thread.start()
