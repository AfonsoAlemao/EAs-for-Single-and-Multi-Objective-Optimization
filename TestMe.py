from simpful import *
import pandas as pd
import matplotlib.pyplot as plt

###################### FIS ######################

def CLPVar_prediction(MU, PL, OT, OBW, L):
    FS1.set_variable("MemoryUsage", MU) 
    FS1.set_variable("ProcessorLoad", PL)

    CR_pred = FS1.inference()['Critical']

    FS2.set_variable("OutBandwidth", OBW)
    FS2.set_variable("Critical", CR_pred) 
    FS2.set_variable("OutNetThroughput", OT)
    FS2.set_variable("Latency", L) 

    FinalOut_pred = FS2.inference()['FinalOut']

    FS3.set_variable("FinalOut", FinalOut_pred) 
    FS3.set_variable("Critical", CR_pred) 

    return FS3.inference()['CLP_variation']

FS1 = FuzzySystem()
FS2 = FuzzySystem()
FS3 = FuzzySystem()

### Memory Usage ### 

MU1 = TrapezoidFuzzySet(0, 0, 0.3 ,0.6, term="Low")
MU2 = TriangleFuzzySet(0.45, 0.6, 0.75, term="Med")
MU3 = TrapezoidFuzzySet(0.6, 0.8, 1, 1, term="High")
FS1.add_linguistic_variable("MemoryUsage", LinguisticVariable([MU1, MU2, MU3], universe_of_discourse=[0,1]))

plt.figure(0)
plt.plot([0, 0, 0.3 ,0.6], [0,1,1,0])
plt.plot([0.45, 0.6, 0.75], [0,1,0])
plt.plot([0.6, 0.8, 1, 1], [0,1,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.title('Membership Function Memory Usage')
plt.xlabel('Memory Usage')
plt.ylabel('MF')

### Processor Load ###

PL1 = TrapezoidFuzzySet(0, 0, 0.3 ,0.6, term="Low")
PL2 = TriangleFuzzySet(0.45, 0.6, 0.75, term="Med")
PL3 = TrapezoidFuzzySet(0.6, 0.8, 1, 1, term="High")
FS1.add_linguistic_variable("ProcessorLoad", LinguisticVariable([PL1, PL2, PL3], universe_of_discourse=[0,1]))

plt.figure(1)
plt.plot([0, 0, 0.3 ,0.6], [0,1,1,0])
plt.plot([0.45, 0.6, 0.75], [0,1,0])
plt.plot([0.6, 0.8, 1, 1], [0,1,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.title('Membership Function Processor Load')
plt.xlabel('Processor Load')
plt.ylabel('MF')

### Critical ###

CR1 = TrapezoidFuzzySet(-1, -1, -0.6, -0.5, term="Low")
CR2 = TrapezoidFuzzySet(-0.7, -0.35, 0.35, 0.7, term="Med")
CR3 = TrapezoidFuzzySet(0.5, 0.6, 1, 1, term="High")
FS1.add_linguistic_variable("Critical", LinguisticVariable([CR1, CR2, CR3], universe_of_discourse=[-1,1]))
FS3.add_linguistic_variable("Critical", LinguisticVariable([CR1, CR2, CR3], universe_of_discourse=[-1,1]))

plt.figure(2)
plt.plot([-1, -1, -0.6, -0.5], [0,1,1,0])
plt.plot([-0.7, -0.35, 0.35, 0.7], [0,1,1,0])
plt.plot([0.5, 0.6, 1, 1], [0,1,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([-1, 1])
plt.title('Membership Function Critical')
plt.xlabel('Critical')
plt.ylabel('MF')


### OutBandwidth ###

OB1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
OB2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
OB3 = TriangleFuzzySet(0.5, 1, 1, term="High")
FS2.add_linguistic_variable("OutBandwidth", LinguisticVariable([OB1, OB2, OB3], universe_of_discourse=[0,1]))

plt.figure(3)
plt.plot([0, 0, 0.5], [0,1,0])
plt.plot([0.3, 0.5, 0.7], [0,1,0])
plt.plot([0.5, 1, 1], [0,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.title('Membership Function OutBandwidth')
plt.xlabel('OutBandwidth')
plt.ylabel('MF')


### OutNetThroughput ###

ONT1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
ONT2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
ONT3 = TriangleFuzzySet(0.5, 1, 1, term="High")
FS2.add_linguistic_variable("OutNetThroughput", LinguisticVariable([ONT1, ONT2, ONT3], universe_of_discourse=[0,1]))

plt.figure(4)
plt.plot([0, 0, 0.5], [0,1,0])
plt.plot([0.3, 0.5, 0.7], [0,1,0])
plt.plot([0.5, 1, 1], [0,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.title('Membership Function OutNetThroughput')
plt.xlabel('OutNetThroughput')
plt.ylabel('MF')


### Latency ### 

L1 = TrapezoidFuzzySet(0, 0, 0.3, 0.5, term="Low")
L2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
L3 = TrapezoidFuzzySet(0.5, 0.7, 1, 1, term="High")
FS2.add_linguistic_variable("Latency", LinguisticVariable([L1, L2, L3], universe_of_discourse=[0,1]))

plt.figure(5)
plt.plot([0, 0, 0.3, 0.5], [0,1,1,0])
plt.plot([0.3, 0.5, 0.7], [0,1,0])
plt.plot([0.5, 0.7, 1, 1], [0,1,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.title('Membership Function Latency')
plt.xlabel('Latency')
plt.ylabel('MF')


### FinalOut ###

FO1 = TriangleFuzzySet(-1, -1, 0, term="Low")
FO2 = TriangleFuzzySet(-0.4, 0, 0.4, term="Med")
FO3 = TriangleFuzzySet(0, 1, 1, term="High")
FS2.add_linguistic_variable("FinalOut", LinguisticVariable([FO1, FO2, FO3], universe_of_discourse=[-1,1]))
FS3.add_linguistic_variable("FinalOut", LinguisticVariable([FO1, FO2, FO3], universe_of_discourse=[-1,1]))

plt.figure(6)
plt.plot([-1, -1, 0], [0,1,0])
plt.plot([-0.4, 0, 0.4], [0,1,0])
plt.plot([0, 1, 1], [0,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([-1, 1])
plt.title('Membership Function FinalOut')
plt.xlabel('FinalOut')
plt.ylabel('MF')


### CLP_variation ###

CLP1 = TriangleFuzzySet(-1, -1, 0, term="Negative")
CLP2 = TriangleFuzzySet(-0.35, 0, 0.35, term="Null")
CLP3 = TriangleFuzzySet(0, 1, 1, term="Positive")

FS3.add_linguistic_variable("CLP_variation", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1,1]))

plt.figure(7)
plt.plot([-1, -1, 0], [0,1,0])
plt.plot([-0.35, 0, 0.35], [0,1,0])
plt.plot([0, 1, 1], [0,1,0])
plt.legend(['Negative', 'Null', 'Positive'])
plt.ylim([0, 1.05])
plt.xlim([-1, 1])
plt.title('Membership Function CLP Variation')
plt.xlabel('CLP Variation')
plt.ylabel('MF')
plt.show()

FS1.set_output_function("High_Critical", "max(MemoryUsage, ProcessorLoad)*2-1")
FS1.set_output_function("Regular", "((MemoryUsage + ProcessorLoad) / 2)*2-1")
FS1.set_output_function("Low_Critical", "((MemoryUsage + ProcessorLoad) / 8)*2-1")

FS1.add_rules([
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Low) THEN (Critical IS Low_Critical)",
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Med) THEN (Critical IS Low_Critical)",
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS High) THEN (Critical IS High_Critical)", 
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Low) THEN (Critical IS Low_Critical)",
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Med) THEN (Critical IS Regular)",
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS High) THEN (Critical IS High_Critical)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS Low) THEN (Critical IS High_Critical)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS Med) THEN (Critical IS High_Critical)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS High) THEN (Critical IS High_Critical)",
])

# x * 2 - 1: normalize x values from [0, 1] to [-1, 1]
FS2.set_output_function("HIGH_LAT", "max(min((0.0 * (1 - OutNetThroughput) + 0.05 * (1 - OutBandwidth) + 1 * Latency + 0.1) * 2 - 1, 1), -1)")
FS2.set_output_function("LOW_OBW", "max(min((0.4 * (1 - OutNetThroughput) + 0.9 * (1 - OutBandwidth) + 0.5 * Latency) * 2 - 1, 1), -1)")
FS2.set_output_function("LOW_ONT", "max(min((0.6 * (1 - OutNetThroughput) + 0.2 * (1 - OutBandwidth) + 0.9 * Latency) * 2 - 1, 1), -1)")
FS2.set_output_function("OTHER", "max(min((0.6 * (1 - OutNetThroughput) + 0.3 * (1 - OutBandwidth) + 0.63 * Latency) * 2 - 1, 1), -1)")

FS2.add_rules([
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Low) AND (Latency IS Low) THEN (FinalOut IS LOW_OBW)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Low) AND (Latency IS Med) THEN (FinalOut IS LOW_OBW)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Low) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Med) AND (Latency IS Low) THEN (FinalOut IS OTHER)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Med) AND (Latency IS Med) THEN (FinalOut IS OTHER)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Med) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS High) AND (Latency IS Low) THEN (FinalOut IS OTHER)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS High) AND (Latency IS Med) THEN (FinalOut IS OTHER)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS High) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS Low) THEN (FinalOut IS LOW_OBW)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS Med) THEN (FinalOut IS LOW_OBW)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS Low) THEN (FinalOut IS OTHER)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS Med) THEN (FinalOut IS OTHER)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS Low) THEN (FinalOut IS OTHER)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS Med) THEN (FinalOut IS OTHER)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS Low) AND (Latency IS Low) THEN (FinalOut IS LOW_OBW)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS Low) AND (Latency IS Med) THEN (FinalOut IS LOW_OBW)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS Low) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS Med) AND (Latency IS Low) THEN (FinalOut IS LOW_ONT)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS Med) AND (Latency IS Med) THEN (FinalOut IS LOW_ONT)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS Med) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS High) AND (Latency IS Low) THEN (FinalOut IS LOW_ONT)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS High) AND (Latency IS Med) THEN (FinalOut IS LOW_ONT)",
    "IF (OutNetThroughput IS Low) AND (OutBandwidth IS High) AND (Latency IS High) THEN (FinalOut IS HIGH_LAT)",
])

FS3.set_output_function("VERY_CRITICAL", "max(min(-0.91 * Critical + 0.09 * FinalOut, 1),-1)")
FS3.set_output_function("NOT_CRITICAL", "max(min(-0.15 * Critical + FinalOut * 0.85, 1),-1)")

FS3.add_rules([
    "IF (Critical IS Low) THEN (CLP_variation IS VERY_CRITICAL)",
    "IF (Critical IS Med) AND (FinalOut IS Low) THEN (CLP_variation IS NOT_CRITICAL)",
    "IF (Critical IS Med) AND (FinalOut IS Med) THEN (CLP_variation IS NOT_CRITICAL)",
    "IF (Critical IS Med) AND (FinalOut IS High) THEN (CLP_variation IS NOT_CRITICAL)",
    "IF (Critical IS High) AND (FinalOut IS Low) THEN (CLP_variation IS VERY_CRITICAL)",
    "IF (Critical IS High) AND (FinalOut IS Med) THEN (CLP_variation IS VERY_CRITICAL)",
    "IF (Critical IS High) AND (FinalOut IS High) THEN (CLP_variation IS VERY_CRITICAL)",
])


df = pd.read_csv('Proj1_TestS.csv', encoding='utf-8')
df.to_excel('Proj1_TestS.xlsx', index=False)

CLP_var_pred_FIS = []

for index, row in df.iterrows():
    MemoryUsage = row['MemoryUsage']
    ProcessorLoad = row['ProcessorLoad']
    OutNetThroughput = row['OutNetThroughput'] 
    OutBandwidth = row['OutBandwidth']
    Latency = row['Latency']

    CLP_var_pred_FIS.append(CLPVar_prediction(MemoryUsage, ProcessorLoad, OutNetThroughput, OutBandwidth, Latency))

#################### Generate Data ####################

import random

def generateDataset():
    new_df = pd.DataFrame()

    range_considered = [0.2 + i / 5 for i in range(4)]
    new_MU = []
    new_PL = []
    new_OT = []
    new_OBW = []
    new_Lat = []
    new_CLP_var = []
    new_IT = []

    random.seed(40)
    random_numbers = [random.uniform(-0.2, 0.2) for _ in range(6144)]
    kk = 0
    
    # Generate dataset
    for nMU in range_considered:
        for nPL in range_considered:
            for nOT in range_considered:
                for nOBW in range_considered:
                    for nL in range_considered:
                        MU = nMU + random_numbers[kk]
                        PL = nPL + random_numbers[kk + 1]
                        OT = nOT + random_numbers[kk + 2]
                        OBW = nOBW + random_numbers[kk + 3]
                        L = nL + random_numbers[kk + 4]
                        new_MU.append(MU)
                        new_PL.append(PL)
                        new_OT.append(OT)
                        new_OBW.append(OBW)
                        new_Lat.append(L)
                        new_IT.append((random_numbers[kk + 5] + 0.2) * 2.5)
                        kk += 6
                        
                        new_CLP_var.append(CLPVar_prediction(MU, PL, OT, OBW, L))
                            
    new_df['MemoryUsage'] = new_MU
    new_df['ProcessorLoad'] = new_PL
    new_df['InpNetThroughput'] = new_IT
    new_df['OutNetThroughput'] = new_OT
    new_df['OutBandwidth'] = new_OBW
    new_df['Latency'] = new_Lat
    
    new_VMU = []
    new_VPL = []
    new_VIT = []	
    new_VOT = []
    new_VOBW = []
    new_VL = []
        
    for index, row in new_df.iterrows():
        if index == 0:
            new_VMU.append(0)
            new_VPL.append(0)
            new_VIT.append(0)
            new_VOT.append(0)
            new_VOBW.append(0)
            new_VL.append(0)
        else:
            new_VMU.append(row['MemoryUsage'] - MUprev)
            new_VPL.append(row['ProcessorLoad'] - PLprev)
            new_VIT.append(row['InpNetThroughput'] - ITprev)
            new_VOT.append(row['OutNetThroughput'] - OTprev)
            new_VOBW.append(row['OutBandwidth'] - OBWprev)
            new_VL.append(row['Latency'] - Lprev)
        
        MUprev = row['MemoryUsage']
        PLprev = row['ProcessorLoad']
        ITprev = row['InpNetThroughput']
        OTprev = row['OutNetThroughput']
        OBWprev = row['OutBandwidth']
        Lprev = row['Latency']
            
    new_df['V_MemoryUsage'] = new_VMU
    new_df['V_ProcessorLoad'] = new_VPL	
    new_df['V_InpNetThroughput'] = new_VIT	
    new_df['V_OutNetThroughput'] = new_VOT
    new_df['V_OutBandwidth'] = new_VOBW
    new_df['V_Latency'] = new_VL
    new_df['CLPVariation'] = new_CLP_var

  
    new_df.to_excel('Proj1_TestS_GeneratedData.xlsx', index=False)
    new_df.to_csv('Proj1_TestS_GeneratedData.csv', index=False, encoding='utf-8')
    
generateDataset()


#################### NN ####################

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def MLP_training(X_train, y_train):
    mlp_gs = MLPRegressor(max_iter=1000, verbose=False)
    parameter_space = {
        'regressor__activation': ['tanh', 'relu'],
        'regressor__solver': ['sgd', 'adam'],
        'regressor__alpha': [0.001, 0.05, 0.01],
        'regressor__learning_rate_init': [0.01, 0.1, 0.05],
        'regressor__learning_rate': ['constant', 'adaptive'],
        'regressor__hidden_layer_sizes': [(12,12,12),(10,10,10), (8,6,3), (6,4,2), (4,5,4),(4,3,3),(8),(9,6),(8,7,6)]
    }
    
#   Size of Input layer = 5 > Size of Hidden layer  > Size of Output layer = 1
#   Size of Hidden layer = 2/3 Size of Input layer + Size of Output layer = 4
#   Size of Hidden layer < 2x Size of Input layer = 10

    # Create a pipeline with StandardScaler and MLPRegressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalization step
        ('regressor', mlp_gs)  # MLPRegressor step
    ])

    clf = GridSearchCV(pipeline, parameter_space, n_jobs=-1, cv=5, verbose=False)
    clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels

    print('Best parameters found:\n', clf.best_params_)
    
    return clf

def MLP_testing(clf, X_test, y_test, datasetType): 
    y_true, y_pred = y_test , clf.predict(X_test)

    for index, element in enumerate(y_pred):
        if(element > 1):
            y_pred[index] = 1
        elif(element < -1):
            y_pred[index] = -1
        
    print('Results on the test set from {}:'.format(datasetType))
    print('Mean Squared Error = {}'.format(mean_squared_error(y_true, y_pred)))
    print('Root Mean Squared Error = {}'.format(mean_absolute_error(y_true, y_pred)))
    print('Mean Absolute Error = {}'.format(mean_squared_error(y_true, y_pred, squared=False)))
    
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

clf = MLP_training(X_train, y_train)

ypred = MLP_testing(clf, X_test, y_test, 'generated dataset by fuzzy system')

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
df2.to_excel('MLP_Results_in_Fuzzy_Generated_Dataset.xlsx', index=False)
df2.to_csv('MLP_Results_in_Fuzzy_Generated_Dataset.csv', index=False, encoding='utf-8')

##################### For the initial provided dataset #####################

y_testS = (np.array(df['CLPVariation']))
X_testS = (np.array(df))[:,:-1]

X_testS2 = np.delete(X_testS, [2,6,7,8,9,10],axis=1)

ypredNN = MLP_testing(clf, X_testS2, y_testS, 'initial provided dataset')

df['CLPVariation_FIS'] = CLP_var_pred_FIS
df['CLPVariation_NN'] = ypredNN
# df['erro_CLP'] = abs(df['CLPVariation_pred'] - df['CLPVariation'])

df.to_excel('TestResult.xlsx', index=False)
df.to_csv('TestResult.csv', encoding='utf-8', index=False)