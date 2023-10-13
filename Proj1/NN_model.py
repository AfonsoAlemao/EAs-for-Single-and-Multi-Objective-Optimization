import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pickle


def MLP_training(X_train, y_train):
    mlp_gs = MLPRegressor(max_iter=1000, verbose=False, early_stopping=True, n_iter_no_change=10) 
    # n_iter_no_change:patience of early stopping
    
    parameter_space = {
        'regressor__activation': ['tanh', 'relu', 'logistic'],
        'regressor__solver': ['sgd', 'adam'],
        'regressor__alpha': [0.001, 0.05, 0.01, 0],
        'regressor__learning_rate': ['constant', 'adaptive'],
        'regressor__learning_rate_init': [0.01, 0.1, 0.05],
        'regressor__hidden_layer_sizes': [(12,12,12),(10,10,10), (8,6,3), (6,4,2), (4,5,4),(4,3,3),(8),(9,6),(8,7,6)],
    }

    # Create a pipeline with StandardScaler and MLPRegressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalization step
        ('regressor', mlp_gs)  # MLPRegressor step
    ])

    model = GridSearchCV(pipeline, parameter_space, n_jobs=-1, cv=5, verbose=False, scoring='neg_mean_squared_error')
    
    model.fit(X_train, y_train) # X is train samples and y is the corresponding labels

    print('Best parameters found:\n', model.best_params_)
    
    return model

def MLP_testing(model, X_test, y_test, datasetType): 
    y_true, y_pred = y_test , model.predict(X_test)

    for index, element in enumerate(y_pred):
        if(element > 1):
            y_pred[index] = 1
        elif(element < -1):
            y_pred[index] = -1
        
    print('Results of NN on the test set from {}:'.format(datasetType))
    print('Mean Squared Error = {}'.format(mean_squared_error(y_true, y_pred)))
    print('Root Mean Squared Error = {}'.format(mean_squared_error(y_true, y_pred, squared=False)))
    print('Mean Absolute Error = {}'.format(mean_absolute_error(y_true, y_pred)))
    
    return y_pred
    
    
df_generated = pd.read_csv('Proj1_TestS_GeneratedData.csv', encoding='utf-8')

y = (np.array(df_generated['CLPVariation']))
X = (np.array(df_generated))[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train2 = X_train
X_test2 = X_test

# Indexes meaning:
# MemoryUsage: 0, ProcessorLoad: 1, InpNetThroughput: 2, OutNetThroughput: 3, OutBandwidth: 4, Latency: 5
# V_MemoryUsage: 6,V_ProcessorLoad: 7,V_InpNetThroughput: 8,V_OutNetThroughput: 9,
# V_OutBandwidth: 10, V_Latency: 11, CLPVariation: 12

X_train = np.delete(X_train, [2,6,7,8,9,10],axis=1)
X_test = np.delete(X_test, [2,6,7,8,9,10],axis=1)

model = MLP_training(X_train, y_train)

ypred = MLP_testing(model, X_test, y_test, 'generated dataset by fuzzy system')

df2 = pd.DataFrame()
df2['MemoryUsage'] = X_test2[:,0]
df2['ProcessorLoad'] = X_test2[:,1]
df2['InpNetThroughput'] = X_test2[:,2]
df2['OutNetThroughput'] = X_test2[:,3]
df2['OutBandwidth'] = X_test2[:,4]
df2['Latency'] = X_test2[:,5]
df2['V_MemoryUsage'] = X_test2[:,6]
df2['V_ProcessorLoad'] = X_test2[:,7]
df2['V_InpNetThroughput'] = X_test2[:,8]
df2['V_OutNetThroughput'] = X_test2[:,9]
df2['V_OutBandwidth'] = X_test2[:,10]
df2['V_Latency'] = X_test2[:,11]
df2['CLPVariation'] = y_test
df2['CLPVariation_pred'] = ypred

# Results in Holdout set from Fuzzy Generated Dataset
# df2.to_excel('MLP_Results_in_Fuzzy_Generated_Dataset.xlsx', index=False)
df2.to_csv('MLP_Results_in_Fuzzy_Generated_Dataset.csv', index=False, encoding='utf-8')


# Save the model to disk
filename = 'NN_model.sav'
pickle.dump(model, open(filename, 'wb')) # Save a model trained on the generated data