from model import Model
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(Model):
    def __init__(self):
        self.rfst = RandomForestClassifier(min_samples_leaf=5, min_samples_split=8, n_estimators=1000)

    def train(self, X, Y):
        self.rfst.fit(X, Y)

    def predict(self, X):
        return self.rfst.predict(X)