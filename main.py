from tasks.master import Master
from tasks.sampler_task import SamplerTask
from tasks.chamfer_task import ChamferTask
from tasks.icp_task import ICPTask
from tasks.icp_task import CandidateICPTask

import queue

class FakeDataset:

    def load(self, mesh_id):
        import pywavefront
        obj = pywavefront.Wavefront("static/obj/teapot.obj", create_materials=True, collect_faces=True)
        return obj

# cache = "cache"
# dataset = FakeDataset()
# inputs = (dataset, "teapot", dataset, "teapot")
# core_task = ChamferTask(cache, inputs)

# ICP between two meshes
# cache = "cache"
# dataset = FakeDataset()
# inputs = (dataset, "teapot", dataset, "teapot")
# core_task = ICPTask(cache, inputs)

# ICP between mesh and candidate meshes
cache = "cache"
dataset = FakeDataset()
inputs = (dataset, "teapot", [dataset, "teapot", dataset, "teapot", dataset, "teapot"])
core_task = CandidateICPTask(cache, inputs)

master = Master(cache)
master.run_task(core_task)
result = core_task.load()

closest_idx, closest_mesh, transform = result
print("Closest Matching Mesh: {} (index {})".format(closest_mesh, closest_idx))
print("Transformation:")
print(transform)