from keras.models import load_model


class BaseModel(object):
    def __init__(self):
        super().__init__()

        self.create_model()

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)

    def create_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
