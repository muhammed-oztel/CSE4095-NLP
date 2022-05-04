from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import pickle
import os
import json

class MLModel:
    def __init__(self, X, y, model_name, hyperparam_tuning=False):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.hyperparam_tuning = hyperparam_tuning
        self.parameters = None
        self.load_model()
        self.vectorize()
        os.makedirs(f'results/{model_name}', exist_ok=True)

    def vectorize(self):
        if os.path.isfile('vectorizer.h5'):
            with open('vectorizer.h5', 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.X = self.vectorizer.transform(self.X)
            return

        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)        
        filename = 'vectorizer.h5'
        with open(filename, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self):
        if os.path.isfile(f'results/{self.model_name}/{self.model_name}.h5'):
            with open(f'results/{self.model_name}/{self.model_name}.h5', 'rb') as f:
                self.model = pickle.load(f)

    def save_model(self):
        filename = f'results/{self.model_name}/{self.model_name}.h5'
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def train(self):
        self.gs = GridSearchCV(self.model, self.parameters, cv=10, scoring='f1_macro', n_jobs=-1)
        self.gs.fit(self.X, self.y)

        self.model = self.gs.best_estimator_

        with open(f'results/{self.model_name}/best_params.json', 'w') as f:
            json.dump(self.gs.best_params_, f)

        self.save_model()
