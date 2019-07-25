class Dataset:

    def __init__(self):
        raise NotImplementedError

    def load_metadata(self):
        raise NotImplementedError

    def ids(self):
        raise NotImplementedError

    def get(self, id):
        raise NotImplementedError