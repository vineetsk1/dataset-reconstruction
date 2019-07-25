import os
from bottle import response, route, run

from vis.visualizer import Visualizer

class MeshVisualizer(Visualizer):

    def __init__(self, port):
        super().__init__(port)

        @route('/')
        def serve_index():
            response.content_type = 'text/html; charset=latin9'
            with open(os.path.join("static", "mesh.html")) as file:
                return file.read().encode("utf-8")

    def show(self, obj_path):
        @route('/mesh.obj')
        def serve_mesh():
            response.content_type = 'text/text; charset=latin9'
            with open(obj_path) as file:
                return file.read().encode("utf-8")

if __name__ == '__main__':
    mesh = "static/obj/sample.obj"
    vis = MeshVisualizer(port=8080)
    vis.show(mesh)
    vis.start()