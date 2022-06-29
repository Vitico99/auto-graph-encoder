from model import Model
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(Model):
    def __init__(self):
        self.rfst = RandomForestClassifier()

    def train(self, X, Y):
        self.rfst.fit(X, Y)

    def predict(self, X):
        return self.rfst.predict(X)