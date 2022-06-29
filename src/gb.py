from model import Model
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingModel(Model):
    
    def __init__(self):
        self.gbc = GradientBoostingClassifier()
    
    def train(self, X, Y):
        self.gbc.fit(X, Y)
    
    def predict(self, X):
        return self.gbc.predict(X)   
