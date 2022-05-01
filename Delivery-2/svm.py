from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pickle
import os
import json


class SVMModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.vectorize()
        os.makedirs('results/svm', exist_ok=True)

    def vectorize(self):
        if os.path.isfile('vectorizer.sav'):
            self.vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
            self.X = self.vectorizer.transform(self.X)
            return

        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)        
        filename = 'vectorizer.sav'
        pickle.dump(self.vectorizer, open(filename, 'wb'))

    def train(self, hyperparam_tuning=False):
        self.hyperparam_tuning = hyperparam_tuning
        if hyperparam_tuning:
            parameters = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ('linear', 'poly', 'rbf'),
                          'gamma': ('scale', 'auto')}
            svm = SVC()
            self.model = GridSearchCV(svm, parameters)
            self.model.fit(self.X, self.y)

            with open('results/svm/best_params.json', 'w') as f:
                json.dump(self.model.best_params_, f)

            return

        else:
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            old_score = 0
            res = {'results': []}
            fold = 1
            for train_index, test_index in skf.split(self.X, self.y):
                print(f'Fold {fold}')
                fold += 1
                
                model = SVC(kernel='linear', C=1, gamma='scale')
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred, average='macro')
                res['results'].append(score)
                
                if score > old_score:
                    self.model = model
                    old_score = score
            
            with open('results/svm/results.json', 'w') as f:
                json.dump(res, f, indent=4)
        
        filename = f'results/svm/svm.sav'
        pickle.dump(self.model, open(filename, 'wb'))
