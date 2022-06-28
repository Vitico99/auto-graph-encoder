from model import Model
from sklearn.svm import SVC

class SVCModel(Model):
    def __init__(self):
        self.svc = SVC()

    def train(self, X, Y):
        self.svc.fit(X, Y)

    def predict(self, X):
        return self.svc.predict(X)
    