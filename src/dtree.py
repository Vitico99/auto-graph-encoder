from sklearn.tree import DecisionTreeClassifier
from model import Model

class DecisionTreeModel(Model):
    def __init__(self):
        self.dtc = DecisionTreeClassifier(criterion='gini', max_depth=8)

    def train(self, X, Y):
        self.dtc.fit(X, Y)
    
    def predict(self, X):
        return self.dtc.predict(X)