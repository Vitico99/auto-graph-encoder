from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from .model import MultiOutputModel

class DecisionTreeModel(MultiOutputModel):
    def __init__(self):
        self.dtc = DecisionTreeClassifier(criterion='gini', max_depth=8)

    def train(self, X, Y):
        self.dtc.fit(X, Y)
    
    def predict(self, X):
        return self.dtc.predict(X)
    
    def prob_predict(self, X):
        return self.dtc.predict_proba(X)

class GradientBoostingModel(MultiOutputModel):
    
    def __init__(self):
        self.gbc = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8)
    
    def train(self, X, Y):
        self.gbc.fit(X, Y)
    
    def predict(self, X):
        return self.gbc.predict(X)   

    def prob_predict(self, X):
        return self.gbc.predict_proba(X)

class KNeighborsModel(MultiOutputModel):
    def __init__(self):
        self.knn = KNeighborsClassifier(metric= 'minkowski',n_neighbors=10, leaf_size=20)

    def train(self, X, Y):
        self.knn.fit(X, Y)
    
    def predict(self, X):
        return self.knn.predict(X)
    
    def prob_predict(self, X):
        return self.knn.predict_proba(X)

class LogisticRegressionModel(MultiOutputModel):
    def __init__(self):
        self.lreg = LogisticRegression(max_iter=2000, multi_class='ovr')

    def train(self, X, Y):
        self.lreg.fit(X, Y)

    def predict(self, X):
        return self.lreg.predict(X)

    def prob_predict(self, X):
        return self.lreg.predict_proba(X)

class MLPModel(MultiOutputModel):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = MLPClassifier(random_state=13, max_iter=200)

    def train(self, X, Y):
        self.mlp.fit(X,Y)

    def predict(self, X):
        return self.mlp.predict(X)
    
    def prob_predict(self, X):
        return self.mlp.predict_proba(X)


class NaiveBayesModel(MultiOutputModel):
    def __init__(self) -> None:
        self.gnb = GaussianNB(var_smoothing=1e-05)
    
    def train(self, X, Y):
        self.gnb.fit(X,Y)
    
    def predict(self, X):
        return self.gnb.predict(X)
    
    def prob_predict(self, X):
        return self.gnb.predict_proba(X)

class RandomForestModel(MultiOutputModel):
    def __init__(self):
        self.rfst = RandomForestClassifier(min_samples_leaf=5, min_samples_split=8, n_estimators=1000)

    def train(self, X, Y):
        self.rfst.fit(X, Y)

    def predict(self, X):
        return self.rfst.predict(X)

    def prob_predict(self, X):
        return self.rfst.predict_proba(X)

class SVCModel(MultiOutputModel):
    def __init__(self):
        self.svc = SVC(C=15, probability=True)

    def train(self, X, Y):
        self.svc.fit(X, Y)

    def predict(self, X):
        return self.svc.predict(X)
    
    def prob_predict(self, X):
        return self.svc.predict_proba(X)