import os
import sys

import multiprocessing

class Worker(multiprocessing.Process):

    def __init__(self, master, fs, gpu_num=None):
        multiprocessing.Process.__init__(self)
        self.master = master
        self.fs = fs
        self.gpu_num = gpu_num

    def run(self):
        proc_name = self.name
        while True:
            task = self.master.tasks.get()
            if task is None:
                self.master.tasks.task_done()
                break
            task.start(gpu_num=self.gpu_num)
            print("{} completed".format(task.name()))
            self.master.tasks.task_done()
        return