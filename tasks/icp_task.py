import os
import sys
import time
import random
import multiprocessing

from tasks.task import Task
from tasks.sampler_task import SamplerTask
from tasks.chamfer_task import ChamferTask
from utils import icp

import torch
import numpy as np

class ICPTask(Task):

    def __init__(self, fs, inputs):
        Task.__init__(self, fs)
        self.inputs = inputs
        self._parents = self.create_parents()
        self.path = os.path.join(self.fs, self.name() + ".npy")

    def name(self):
        ds_1, mesh_1, ds_2, mesh_2 = self.inputs
        if mesh_2 < mesh_1:
            tmp = mesh_1
            mesh_1 = mesh_2
            mesh_2 = tmp
        return "icp_{}_{}".format(mesh_1, mesh_2)

    def parents(self):
        return self._parents

    def create_parents(self):
        ds_1, mesh_1, ds_2, mesh_2 = self.inputs
        return [SamplerTask(self.fs, (ds_1, mesh_1)),
                SamplerTask(self.fs, (ds_2, mesh_2))]

    def run_worker(self):
        pc1, pc2 = [parent.load() for parent in self.parents()]
        T, _, _ = icp.icp(pc1, pc2)
        return T

    def run_worker_gpu(self, gpu_num):
        return self.run_worker()

    def load(self):
        return np.load(self.path)

    def save(self, data):
        np.save(self.path, data)

class CandidateICPTask(Task):

    def __init__(self, fs, inputs):
        Task.__init__(self, fs)
        self.inputs = inputs
        self._parents = self.create_parents()
        self.path = os.path.join(self.fs, self.name())

    def name(self):
        ds_1, mesh_1, candidates = self.inputs
        names = []
        for i in range(1, len(candidates), 2):
            names.append(candidates[i])
        names.sort()
        return "icp_{}_cands_{}".format(mesh_1, "_".join(names))

    def parents(self):
        return self._parents

    def create_parents(self):
        ds_1, mesh_1, candidates = self.inputs
        parents = []
        for i in range(0, len(candidates), 2):
            ds_2 = candidates[i]
            mesh_2 = candidates[i+1]
            inputs = (ds_1, mesh_1, ds_2, mesh_2)
            parents.append(ChamferTask(self.fs, inputs))
        return parents

    def run_worker(self):
        ds_1, mesh_1, candidates = self.inputs
        dists = np.asarray([parent.load() for parent in self.parents()])
        min_idx = np.argmin(dists)
        ds_2, mesh_2 = candidates[min_idx*2], candidates[min_idx*2 + 1]

        pc1 = SamplerTask(self.fs, (ds_1, mesh_1)).load()
        pc2 = SamplerTask(self.fs, (ds_2, mesh_2)).load()
        T, _, _ = icp.icp(pc1, pc2)
        return (min_idx, mesh_2, T)

    def run_worker_gpu(self, gpu_num):
        return self.run_worker()

    def load(self):
        with open(self.path, "r") as f:
            path1, path2, path3 = f.read().split("\n")
        with open(path1, "r") as f: min_idx = f.read()
        with open(path2, "r") as f: mesh_name = f.read()
        return min_idx, mesh_name, np.load(path3)

    def save(self, data):
        min_idx, mesh_name, T = data
        path1, path2, path3 = self.path + "_idx", self.path + "_name", self.path + "_mesh.npy"
        with open(path1, "w") as f: f.write(str(min_idx))
        with open(path2, "w") as f: f.write(str(mesh_name))
        np.save(path3, T)
        with open(self.path, "w") as f:
            f.write("{}\n{}\n{}".format(path1, path2, path3))