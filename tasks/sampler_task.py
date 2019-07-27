import os
import sys
import time
import random
import multiprocessing

from tasks.task import Task
from utils import sample

import numpy as np

class SamplerTask(Task):

    def __init__(self, fs, inputs, density=512):
        Task.__init__(self, fs)
        self.inputs = inputs
        self.inputs = (*inputs, density)
        self._parents = self.create_parents()
        self.path = os.path.join(self.fs, self.name() + ".npy")

    def name(self):
        dataset, mesh_id, density = self.inputs
        return "sample_{}".format(mesh_id)

    def parents(self):
        return self._parents

    def create_parents(self):
        return [] # No dependencies for mesh->pc conversion

    def run_worker(self):
        dataset, mesh_id, density = self.inputs
        mesh = dataset.load(mesh_id)
        point_cloud = sample.sample_obj(mesh, density=density)

        # Normalization step.
        # Chamfer distance is not invariant w.r.t translation, rotation, or scale.
        # Translation: Center at origin
        # Scale: Rescale to [-1, 1]
        # Rotation: TODO? For now we assume models are aligned along the same axis.
        point_cloud = point_cloud - np.mean(point_cloud, axis=0)
        scale = np.max(np.abs(point_cloud))
        point_cloud = point_cloud / scale
        return point_cloud

    def run_worker_gpu(self, gpu_num):
        return self.run_worker()

    def load(self):
        return np.load(self.escape_path())

    def save(self, data):
        np.save(self.escape_path(), data)
