from model import Model
from sklearn.naive_bayes import GaussianNB

class NaiveBayesModel(Model):
    def __init__(self) -> None:
        self.gnb = GaussianNB(var_smoothing=1e-05)
    
    def train(self, X, Y):
        self.gnb.fit(X,Y)
    
    def predict(self, X):
        return self.gnb.predict(X)