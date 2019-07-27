import os
import sys

import multiprocessing

# Task represents smallest chunk of work, running on one thread
# one one CPU process, using at most one GPU. A task like
# a SceneReconstruction depends on ICPTasks for each of the models
# in that scene. Each ICPTask depends on (potentially overlapping)
# ChamferTasks to find the nearest model. Each ChamferTask depends
# on (probably overlapping) SamplerTasks to convert meshes to point clouds.

# Tasks should be coordinated by a Master thread/process. Upon
# an initial Task creation, a dependency graph should be created
# listing the parent/children tasks that need to be completed in
# a resource efficient fashion. For now, we'll have communication
# of status and input/output communication occur through this master
# thread (in the future we could expand to multiple masters per core).

# Note that these tasks do not have much I/O and are strictly bound
# by compute resources (CPU/GPU). As such, it only makes sense to make
# as many workers as there are cores. Each worker should be tied to
# one core, and zero or one GPUs.

# TODO: Currently inter-worker communication happens through the filesystem
# (cache). For smaller messages we should instead make this operate through
# shared memory on the main thread or sockets. Having all communication go
# through the filesystem is super inefficient and overkill for small jobs
# (but might be the most flexible solution for larger jobs that will
# quickly use all available memory).

# TODO: Prioritize GPU enabled tasks on GPU workers
class Task:

    def __init__(self, fs):
        self.fs = fs

    def __str__(self):
        return self.name()

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name() == other.name()

    def __hash__(self):
        return hash(self.name())

    def name(self):
        raise NotImplementedError

    def parents(self):
        raise NotImplementedError

    def start(self, gpu_num=None):
        if os.path.exists(self.path):
            return self.load()

        data = self.run_worker_gpu(gpu_num) if gpu_num is not None else self.run_worker()
        self.save(data)

    def escape_path(self):
        arr = self.path.split("/")
        return arr[0] + "/" + "_".join(arr[1:])

    def run_worker(self):
        raise NotImplementedError

    def run_worker_gpu(self, gpu_num):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self, data):
        raise NotImplementedError