from sklearn.neighbors import KNeighborsClassifier
from model import Model

class KNeighborsModel(Model):
    def __init__(self):
        self.knn = KNeighborsClassifier(metric= 'minkowski',n_neighbors=10, leaf_size=20)

    def train(self, X, Y):
        self.knn.fit(X, Y)
    
    def predict(self, X):
        return self.knn.predict(X)