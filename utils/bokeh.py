from tornado.ioloop import IOLoop
from bokeh.command.util import build_single_handler_applications
from bokeh.server.server import Server
from bokeh.util.browser import view
from multiprocessing import Process

bokeh_host = 'localhost'
bokeh_port = 8080

def start_server(filepath: str, host: str, port: int):
	files = [filepath]
	argvs = {}
	urls = []
	for f in files:
		argvs[f]=None
		urls.append(f.split('/')[-1].split('.')[0])
	io_loop = IOLoop.instance()
	apps = build_single_handler_applications(files,argvs)
	kwags = {
		'io_loop':io_loop,
		'generade_session_ids':True,
		'redirect_root':True,
		'use_x_headers':False,
		'secret_key':None,
		'num_procs':1,
		'host':['%s:%d'%(host, port)],
		'sign_sessions':False,
		'develop':False,
		'port':port, 
		'use_index':True
	}
	srv = Server(apps,**kwags)
	io_loop.add_callback(view, 'http://{}:{}'.format(bokeh_host, bokeh_port))
	io_loop.start()

def serve(filepath: str, host: str=bokeh_host, port: int=bokeh_port):
	proc = Process(target=start_server, args=(filepath, host, port))
	proc.start()
	print('Server started')
	return proc