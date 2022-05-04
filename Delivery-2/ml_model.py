from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import pickle
import os
import json

class MLModel:
    def __init__(self, X, y, model_name):
        self.X = X
        self.y = y
        self.model_name = model_name
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

    def save_best_params(self):
        filename = f'results/{self.model_name}/{self.model_name}_best_params.json'
        with open(filename, 'w') as f:
            json.dump(self.best_params, f)

    def load_best_params(self):
        filename = f'results/{self.model_name}/{self.model_name}_best_params.json'
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                self.best_params = json.load(f)

    def save_model(self):
        filename = f'results/{self.model_name}/{self.model_name}.h5'
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        filename = f'results/{self.model_name}/{self.model_name}.h5'

        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)

    def save_cv_results(self):
        filename = f'results/{self.model_name}/{self.model_name}_cv_results.json'
        with open(filename, 'w') as f:
            json.dump({'results':self.cv_results.tolist()}, f)

    def train(self):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(self.X, self.y):
            break

        self.gs = GridSearchCV(self.model, self.parameters, cv=zip(train_index, test_index), scoring='f1_macro', n_jobs=-1)
        self.gs.fit(self.X, self.y)
        self.best_params = self.gs.best_params_
        self.save_best_params()
        self.model = self.gs.best_estimator_
        self.save_model()

        self.cv_results = cross_val_score(self.gs.best_estimator_, self.X, self.y, cv=10, scoring='f1_macro', verbose=4)
        self.save_cv_results()

