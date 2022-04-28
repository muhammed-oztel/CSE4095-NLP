from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import os


class SVMModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.vectorize()
        os.makedirs('results/svm', exist_ok=True)

    def vectorize(self):
        self.vectorizer = TfidfVectorizer()
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)
        # maybe add StandardScaler

    def train(self, hyperparam_tuning=False):
        self.hyperparam_tuning = hyperparam_tuning
        if hyperparam_tuning:
            parameters = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'),
                          'gamma': ('scale', 'auto')}
            svm = SVC()
            self.model = GridSearchCV(svm, parameters)
            self.model.fit(self.X_train, self.y_train)
            print('best_params:', self.model.best_params_)
        else:
            self.model = SVC(kernel='linear')
            self.model.fit(self.X_train, self.y_train)
        
        filename = f'results/svm/svm_{hyperparam_tuning}.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        
    def confusion_matrix(self, str_labels):
        disp = ConfusionMatrixDisplay.from_estimator(
            self.model,
            self.X_test,
            self.y_test,
            display_labels=str_labels,
            cmap=plt.cm.Blues,
            normalize=None,
        )
        disp.ax_.set_title('Confusion Matrix')

        plt.xticks(rotation = 90, fontsize=7)
        plt.yticks(fontsize=7)
        plt.savefig(f'results/svm/cm_{self.hyperparam_tuning}.png', dpi=300, bbox_inches='tight')

    def classification_report(self, str_labels):
        self.cr = classification_report(self.y_test, self.y_pred, target_names=str_labels)
        print(self.cr)
