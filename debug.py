import random
from datasets.shapenet import Shapenet
from vis.mesh_visualizer import MeshVisualizer

shapenet = Shapenet("data/ShapeNetCore_v2/")
meshes = shapenet.ids()
random.shuffle(meshes)

mesh = shapenet.path(meshes[0])
vis = MeshVisualizer(port=8080)
vis.show(mesh)
vis.start()