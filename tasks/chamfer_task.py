import os
import sys
import time
import random
import multiprocessing

from tasks.task import Task
from tasks.sampler_task import SamplerTask
from utils import chamfer

import torch
import numpy as np

class ChamferTask(Task):

    def __init__(self, fs, inputs):
        Task.__init__(self, fs)
        self.inputs = inputs
        self._parents = self.create_parents()
        self.path = os.path.join(self.fs, self.name())

    def name(self):
        ds_1, mesh_1, ds_2, mesh_2 = self.inputs
        if mesh_2 < mesh_1:
            tmp = mesh_1
            mesh_1 = mesh_2
            mesh_2 = tmp
        return "chamfer_{}_{}".format(mesh_1, mesh_2)

    def parents(self):
        return self._parents

    def create_parents(self):
        ds_1, mesh_1, ds_2, mesh_2 = self.inputs
        return [SamplerTask(self.fs, (ds_1, mesh_1)),
                SamplerTask(self.fs, (ds_2, mesh_2))]

    def run_worker(self):
        pc1, pc2 = [parent.load() for parent in self.parents()]
        pc1 = torch.tensor([pc1]).float()
        pc2 = torch.tensor([pc2]).float()

        cd = chamfer.ChamferDistance()
        d1, d2 = cd(pc1, pc2)
        return "{}".format(0.5*(d1.mean()+d2.mean()))

    def run_worker_gpu(self, gpu_num):
        return self.run_worker()

    def load(self):
        with open(self.escape_path(), "r") as f:
            return f.read()

    def save(self, data):
        with open(self.escape_path(), "w") as f:
            f.write(data)