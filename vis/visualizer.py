import os
import socket
import getpass
from bottle import response, route, run

class Visualizer:

    def __init__(self, port):
        self.port = port

        @route('/js/<file:path>')
        def serve_js(file):
            response.content_type = 'application/javascript; charset=latin9'
            with open(os.path.join("static", "js", file)) as file:
                return file.read().encode("utf-8")

        @route('/obj/<file:path>')
        def serve_obj(file):
            response.content_type = 'text/text; charset=latin9'
            with open(os.path.join("static", "obj", file)) as file:
                return file.read().encode("utf-8")

    def show(self, objects):
        raise NotImplementedError

    def start(self):
        print("="*20)
        print("If you are running this on a remote server, do not forget to setup port forwarding:\n")
        hostname = socket.gethostname()
        print(" "*4 + "ssh -L {}:{}:{} {}@{} -N".format(self.port, hostname, self.port, getpass.getuser(), hostname))
        print("\nYou can then view the visualization by visiting http://localhost:{}".format(self.port))
        print("="*20)
        run(host='localhost', port=self.port, debug=True)