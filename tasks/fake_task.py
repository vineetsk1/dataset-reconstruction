import os
import sys
import time
import random
import multiprocessing

from tasks.task import Task

class FakeTask(Task):

    def __init__(self, fs, inputs):
        Task.__init__(self, fs)
        self.inputs = inputs
        self._parents = self.create_parents()
        self.path = os.path.join(self.fs, self.name())

    def name(self):
        layer, uid = self.inputs
        name = "fake_task_{}_{}".format(layer, uid)
        return name

    def parents(self):
        return self._parents

    def create_parents(self):
        layer, uid = self.inputs
        if layer == 0:
            return [FakeTask(self.fs, (1, uid*10 + uuid)) for uuid in range(5, 8)]
        if layer == 1:
            return [FakeTask(self.fs, (2, uuid)) for uuid in range(uid, uid*5+5)]
        if layer == 2:
            return []

    def run_worker(self):
        t = random.random() * 5
        time.sleep(t)
        debug_info = []
        for parent in self.parents():
            debug_info.append(parent.load())
        debug_info.append("Task {} complete in {} seconds.".format(self.name(), t))
        debug_info = "\n".join(debug_info)
        return debug_info

    def run_worker_gpu(self, gpu_num):
        return self.run_worker()

    def load(self):
        with open(self.path, "r") as f:
            return f.read()

    def save(self, data):
        with open(self.path, "w") as f:
            f.write(data)

