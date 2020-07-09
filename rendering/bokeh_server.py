''' Interactive plot for rendering time-indexed graph simulations ''' 

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
speed = 0.1 
viz_dt = 50 # update every ms

'''
UI
'''

t1 = Div(text='Time:', style={'font-size':'150%'})
t2 = Div(text='N/A', style={'font-size':'150%'})
pp_button = Button(label='► Play', width=60)
speed_slider = Slider(start=-2.0, end=2.0, value=-1.0, step=0.1, title='Speed', width=300)

'''
Callbacks
'''
def update():
	global renderers, viz_dt
	for r in renderers:
		r.step(viz_dt * 1e-3 * speed)
	t1.text = str(renderers[0].t)

def pp_button_cb():
	global viz_dt, render_callback
	if pp_button.label == '► Play':
		pp_button.label = '❚❚ Pause'
		render_callback = doc.add_periodic_callback(update, viz_dt)
	else:
		pp_button.label = '► Play'
		doc.remove_periodic_callback(render_callback)
pp_button.on_click(pp_button_cb)

def speed_slider_cb(attr, old, new):
	global speed
	speed = 10 ** speed_slider.value
speed_slider.on_change('value', speed_slider_cb)

'''
Layout
'''

root = column(
	row([t1, t2]),
	row([pp_button, speed_slider]),
)
root.sizing_mode = 'stretch_both'
doc.add_root(root)
doc.title = 'Bokeh Server'

'''
Updates
'''

@gen.coroutine
def react(msg):
	global renderers, viz_dt
	# print(msg)
	if msg['tag'] == 'init':
		renderers = [GraphRenderer.from_json(r) for r in msg['renderers']]
		for r in renderers:
			plots[r.title] = r.init_plot()
		root.children.append(row(list(plots.values())))
		# render_callback = doc.add_periodic_callback(update, viz_dt) # update every 50 ms

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
