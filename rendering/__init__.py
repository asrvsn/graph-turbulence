''' Solution rendering for graphs ''' 

from typing import Callable, List
from pathlib import Path
import time
import dill as pickle

from utils.bokeh import serve_and_open
from utils.zmq import pubsub_tx
from core.observables import System


def render_live(rs: List[System]):
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
		tx({'tag': 'init', 'renderers': [pickle.dumps(r).decode('latin1') for r in rs]})
		print('Done.')
		while True: 
			time.sleep(1) # Let bokeh continue to handle interactivity while we wait
	finally:
		ctx.destroy()
		proc.terminate()