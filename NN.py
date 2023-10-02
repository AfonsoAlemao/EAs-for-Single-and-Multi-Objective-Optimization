import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def MLP(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    mlp_gs = MLPClassifier(max_iter=1000, verbose=False)
    parameter_space = {
        'classifier__activation': ['tanh', 'relu'],
        'classifier__solver': ['sgd', 'adam'],
        'classifier__alpha': [0.0001, 0.05],
        'classifier__learning_rate': ['constant', 'adaptive'],
    }

    # Create a pipeline with StandardScaler and MLPClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalization step
        ('classifier', mlp_gs)  # MLPClassifier step
    ])

    clf = GridSearchCV(pipeline, parameter_space, n_jobs=-1, cv=5, verbose=False)
    clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels

    print('Best parameters found:\n', clf.best_params_)

    # print('clf.get_params = ', clf.get_params, 'clf.best_params_ = ', clf.best_params_)
    # print('clf.get_params = ', clf.get_params)
    
    y_true, y_pred = y_test , clf.predict(X_test)

    print('Results on the test set:')
    print(classification_report(y_true, y_pred))
    

import pandas as pd

df = pd.read_csv('ACI23-24_Proj1_SampleData_new.csv', encoding='utf-8')

y_iris = (np.array(df))[:,-1]
X_iris = (np.array(df))[:,:-1]

MLP(X_iris, y_iris)
