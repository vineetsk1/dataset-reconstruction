import json

path = "./datasets/ShapeNetCore_v2/taxonomy.json"
a = json.load(open(path, "r"))

count = 0
children = set()
top_level = set()

for i in range(len(a)):

    item = a[i]

    if item["synsetId"] not in children:
        top_level.add(item["synsetId"])
        count += int(item["numInstances"])

    for child in item["children"]:
        children.add(child)

print("Number meshes", count)