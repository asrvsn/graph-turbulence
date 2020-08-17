''' Interactive plot for rendering time-indexed graph simulations ''' 

import numpy as np
from functools import partial
from threading import Thread
from tornado import gen
import traceback

from bokeh.plotting import figure, output_file, show, curdoc, from_networkx
from bokeh.models import ColumnDataSource, Slider, Select, Button, Oval
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models.widgets import Div, TextInput

from utils.zmq import *
from rendering import *

# Save curdoc() to make sure all threads see the same document.
doc = curdoc()


''' 
Variables
'''

systems = []
render_callback = None
speed = 0.1 
viz_dt = 50 # update every ms

'''
UI
'''

t1 = Div(text='Time:', style={'font-size':'150%'})
t2 = Div(text='N/A', style={'font-size':'150%'})
reset_button = Button(label='⟲ Reset', width=60)
pp_button = Button(label='► Play', width=60)
speed_slider = Slider(start=-2.0, end=1.0, value=-1.0, step=0.02, title='Speed', width=300)

'''
Callbacks
'''
def update():
	global systems, viz_dt, render_callback
	try:
		for r in systems:
			r.step(viz_dt * 1e-3 * speed)
			r.measure()
		t2.text = str(round(systems[0].t, 3))
	except KeyboardInterrupt:
		raise
	except:
		print('Exception caught, stopping')
		pp_button.label = '► Play'
		doc.remove_periodic_callback(render_callback)
		traceback.print_exc()

def reset_button_cb():
	global systems
	for r in systems:
		r.reset()
		r.measure()
	t2.text = str(round(systems[0].t, 3))
reset_button.on_click(reset_button_cb)

def pp_button_cb():
	global viz_dt, render_callback
	if pp_button.label == '► Play':
		pp_button.label = '❚❚ Pause'
		render_callback = doc.add_periodic_callback(update, viz_dt)
	else:
		pp_button.label = '► Play'
		if render_callback in doc.session_callbacks:
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
	row([reset_button, pp_button, speed_slider]),
)
root.sizing_mode = 'stretch_both'
doc.add_root(root)
doc.title = 'Bokeh Server'

'''
Updates
'''

@gen.coroutine
def react(msg):
	global systems, viz_dt
	# print(msg)
	if msg['tag'] == 'init':
		sys_funcs = wire_unpickle(msg['systems'])
		for i in range(len(sys_funcs)):
			if type(sys_funcs[i]) == list:
				for j in range(len(sys_funcs[i])):
					sys_funcs[i][j] = sys_funcs[i][j]()
			else:
				sys_funcs[i] = sys_funcs[i]()

		col_children = []
		for i in range(len(sys_funcs)):
			row_children = []
			for j in range(len(sys_funcs[i])):
				elem = sys_funcs[i][j]
				if type(elem) == list:
					subcol = []
					for k in range(len(elem)):
						systems.append(elem[k])
						plot = elem[k].create_plot()
						subcol.append(plot)
					row_children.append(gridplot(subcol, ncols=int(np.sqrt(len(subcol))), sizing_mode='scale_both'))
				else:
					systems.append(elem)
					row_children.append(elem.create_plot())
			col_children.append(row(row_children, sizing_mode='scale_both'))
		root.children.append(column(col_children, sizing_mode='scale_both'))

def stream_data():
	ctx, rx = ipc_rx()
	try:
		while True:
			msg = rx()
			doc.add_next_tick_callback(partial(react, msg=msg))
	finally:
		ctx.destroy()

print('Bokeh started')
thread = Thread(target=stream_data)
thread.start()
