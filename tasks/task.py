import os
import sys

import multiprocessing

# Task represents smallest chunk of work, running on one thread
# one one CPU process, using at most one GPU. A task like
# a SceneReconstruction depends on ICPTasks for each of the models
# in that scene. Each ICPTask depends on (potentially overlapping)
# ChamferTasks to find the nearest model. Each ChamferTask depends
# on SamplerTasks to convert meshes to point clouds.

# Tasks should be coordinated by a Master thread/process. Upon
# an initial Task creation, a dependency graph should be created
# listing the parent/children tasks that need to be completed in
# a resource efficient fashion. For now, we'll have communication
# of status and input/output communication occur through this master
# thread (in the future we could expand to multiple masters per core).

class Task:
	
	def __init__(self, inputs):
		self.prereqs = []
		self.children = []

	def run_worker(self):
		raise NotImplementedError

	def run_worker_gpu(self):
		raise NotImplementedError

	def start(self):
		raise NotImplementedError