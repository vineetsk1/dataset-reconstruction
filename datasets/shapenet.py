import os
import json
import pywavefront
from datasets.dataset import Dataset

class Shapenet(Dataset):

    def __init__(self, path):
        self._path = path
        self.load_metadata()

    def load_metadata(self):
        path = os.path.join(self._path, "taxonomy.json")
        data = json.load(open(path, "r"))

        children = set()
        top_level = {}
        for i in range(len(data)):
            item = data[i]
            if item["synsetId"] not in children and os.path.exists(os.path.join(self._path, item["synsetId"])):
                top_level[item["synsetId"]] = {}
                top_level[item["synsetId"]]["names"] = item["name"].split(",")
                top_level[item["synsetId"]]["files"] = []
            for child in item["children"]:
                children.add(child)

        ids = []
        for name in top_level:
            top_level[name]["files"] = []
            for file in os.listdir(os.path.join(self._path, name)):
                if "." not in file:
                    top_level[name]["files"].append(file)
                    ids.append(os.path.join(name, file))

        self.taxonomy = top_level
        self._ids = ids

    def ids(self):
        return self._ids

    def path(self, id):
        return os.path.join(self._path, id, "models", "model_normalized.obj")

    def load(self, id):
        obj = pywavefront.Wavefront(self.path(id), create_materials=True, collect_faces=True)
        return obj