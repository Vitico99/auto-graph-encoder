from model import Model
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(Model):
    def __init__(self):
        self.lreg = LogisticRegression(multi_class='ovr')

    def train(self, X, Y):
        self.lreg.fit(X, Y)

    def predict(self, X):
        return self.lreg.predict(X)