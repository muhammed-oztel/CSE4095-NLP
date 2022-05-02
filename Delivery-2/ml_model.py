from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
        if os.path.isfile('vectorizer.sav'):
            self.vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
            self.X = self.vectorizer.transform(self.X)
            return

        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)        
        filename = 'vectorizer.sav'
        with open(filename, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self):
        if os.path.isfile(f'results/{self.model_name}/{self.model_name}.sav'):
            with open(f'results/{self.model_name}/{self.model_name}.sav', 'rb') as f:
                self.model = pickle.load(f)

    def save_model(self):
        filename = f'results/{self.model_name}/{self.model_name}.sav'
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def _hyperparameter_tuning(self):
        self.gs = GridSearchCV(self.model, self.parameters, cv=10, scoring='f1_macro')
        self.gs.fit(self.X, self.y)

        self.model = self.gs.best_estimator_

        with open(f'results/{self.model_name}/best_params.json', 'w') as f:
            json.dump(self.gs.best_params_, f)

    def _k_fold_cv(self):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        old_score = 0
        res = {'results': []}
        fold = 1
        for train_index, test_index in skf.split(self.X, self.y):
            print(f'Fold {fold}')
            fold += 1
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            score = f1_score(y_test, y_pred, average='macro')
            res['results'].append(score)
            
            if score > old_score:
                self.self.model = self.model
                old_score = score
        
        with open(f'results/{self.model_name}/results.json', 'w') as f:
            json.dump(res, f, indent=4)

    def train(self):
        if self.hyperparam_tuning:
            self._hyperparameter_tuning()
        else:
            self._k_fold_cv()

        self.save_model()
