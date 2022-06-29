from model import Model
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingModel(Model):
    
    def __init__(self):
        self.gbc = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8)
    
    def train(self, X, Y):
        self.gbc.fit(X, Y)
    
    def predict(self, X):
        return self.gbc.predict(X)   
