import os
import sys
import torch
import queue
import multiprocessing

from tasks.worker import Worker
from tasks.task import Task

class Master:

    def __init__(self, fs):
        self.fs = fs
        self.num_workers = multiprocessing.cpu_count()
        self.num_gpus = torch.cuda.device_count()

    """
    Collect a set of tasks scheduled for execution. Each task will only
    appear once and will only appear if it is not already evaluated (in
    the cache represented by self.fs).
    """
    def construct_task_dict(self):

        task_dict = {}
        items = queue.Queue()
        items.put(self.core_task)

        while not items.empty():
            task = items.get()
            if task.name() in task_dict or os.path.exists(os.path.join(self.fs, task.name())):
                continue
            task_dict[task.name()] = task
            for prereq in task.parents():
                items.put(prereq)

        self.task_dict = task_dict

    def construct_dependency_graph(self):
        task_layers = []

        items = queue.Queue()
        items.put([self.core_task])
        while not items.empty():

            candidate_tasks = items.get()

            task_layer = set()
            for task in candidate_tasks:
                if task.name() not in self.task_dict:
                    continue
                task_layer.add(task)
            task_layer = list(task_layer)
            if len(task_layer) != 0:
                task_layers.append(task_layer)

            parent_tasks = []
            for task in task_layer:
                for prereq in task.parents():
                    parent_tasks.append(prereq)
            if len(parent_tasks) != 0:
                items.put(parent_tasks)

        task_layers = task_layers[::-1]
        self.task_layers = task_layers

    def print_dependency_graph(self):
        if len(self.task_layers) != 0:
            print("Dependency Graph")
            print("-"*20)

        for layer in self.task_layers:
            for task in layer:
                print(task)
            print("="*10)

        if len(self.task_layers) != 0:
            print("")

    def construct_workers(self):
        self.workers = []
        for i in range(self.num_workers):
            gpu_num = i if i < self.num_gpus else None
            self.workers.append(Worker(self, self.fs, gpu_num))

    def run_jobs(self):
        if len(self.task_layers) != 0:
            print("Job Status")
            print("-"*20)
        for task_layer in self.task_layers:
            self.tasks = multiprocessing.JoinableQueue()
            self.construct_workers()
            for worker in self.workers:
                worker.start()
            for task in task_layer:
                self.tasks.put(task)
            for _ in self.workers:
                self.tasks.put(None)
            self.tasks.join()
            print("=" * 10)

        if len(self.task_layers) != 0: print("")
        else: print("=" * 10)

    def run_task(self, task):
        self.core_task = task

        self.construct_task_dict()
        self.construct_dependency_graph()
        self.print_dependency_graph()
        self.run_jobs()