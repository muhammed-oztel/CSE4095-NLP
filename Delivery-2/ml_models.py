from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from ml_model import MLModel 


class MultinomialNaiveBayesModel(MLModel):
    def __init__(self, X, y, model_name, hyperparam_tuning=False):
        super().__init__(X, y, model_name, hyperparam_tuning)
        self.model = MultinomialNB()


class LogisticRegressionModel(MLModel):
    def __init__(self, X, y, model_name, hyperparam_tuning=False):
        super().__init__(X, y, model_name, hyperparam_tuning)
        self.model = LogisticRegression(max_iter=1000, warm_start=True, n_jobs=-1, C=10, solver='sag')
        self.parameters = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ('newton-cg', 'lbfgs', 'sag', 'saga')}

class RandomForestModel(MLModel):
    def __init__(self, X, y, model_name, hyperparam_tuning=False):
        super().__init__(X, y, model_name, hyperparam_tuning)
        self.model = RandomForestClassifier(n_estimators=2000, n_jobs=-1, warm_start=True, random_state=42)
        self.parameters = {'n_estimators': [500, 1000, 1500, 2000], 'criterion': ('gini', 'entropy')}


class SVMModel(MLModel):
    def __init__(self, X, y, model_name, hyperparam_tuning=False):
        super().__init__(X, y, model_name, hyperparam_tuning)
        self.model = SVC(kernel='linear', C=1, gamma='scale')
        self.parameters = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ('linear', 'poly', 'rbf'),
                          'gamma': ('scale', 'auto')}