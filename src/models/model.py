class Model:
    def train(self, X, Y):
        pass

    def predict(self, X):
        pass

class MultiOutputModel(Model):
    def prob_predict(self, X):
        pass