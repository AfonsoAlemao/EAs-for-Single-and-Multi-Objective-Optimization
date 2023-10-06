import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def MLP_training(X_train, y_train):
    mlp_gs = MLPClassifier(max_iter=1000, verbose=False)
    parameter_space = {
        'classifier__activation': ['tanh', 'relu'],
        'classifier__solver': ['sgd', 'adam'],
        'classifier__alpha': [0.001, 0.05, 0.01],
        'classifier__learning_rate': ['constant', 'adaptive'],
        'classifier__hidden_layer_sizes': [(8),(9,6),(8,7,6)]
    }

    # Create a pipeline with StandardScaler and MLPClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalization step
        ('classifier', mlp_gs)  # MLPClassifier step
    ])

    clf = GridSearchCV(pipeline, parameter_space, n_jobs=-1, cv=5, verbose=False)
    clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels

    print('Best parameters found:\n', clf.best_params_)
    
    return clf

def MLP_testing(clf): 
    y_true, y_pred = y_test , clf.predict(X_test)

    print('Results on the test set:')
    print(classification_report(y_true, y_pred))
    
    return y_pred
    

import pandas as pd

df = pd.read_csv('ACI23-24_Proj1_SampleData_new.csv', encoding='utf-8')

y = (np.array(df))[:,-1]
X = (np.array(df))[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = MLP_training(X_train, y_train)

ypred = MLP_testing(clf)

df['CLPVariation_pred'] = ypred

df.to_excel('MLP_Results.xlsx', index=False)
df.to_csv('MLP_Results.csv', index=False, encoding='utf-8')
