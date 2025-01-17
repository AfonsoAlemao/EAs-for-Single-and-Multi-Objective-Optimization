from simpful import *
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

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


### Processor Load ###

PL1 = TrapezoidFuzzySet(0, 0, 0.3 ,0.6, term="Low")
PL2 = TriangleFuzzySet(0.45, 0.6, 0.75, term="Med")
PL3 = TrapezoidFuzzySet(0.6, 0.8, 1, 1, term="High")
FS1.add_linguistic_variable("ProcessorLoad", LinguisticVariable([PL1, PL2, PL3], universe_of_discourse=[0,1]))


### Critical ###

CR1 = TrapezoidFuzzySet(-1, -1, -0.6, -0.5, term="Low")
CR2 = TrapezoidFuzzySet(-0.7, -0.35, 0.35, 0.7, term="Med")
CR3 = TrapezoidFuzzySet(0.5, 0.6, 1, 1, term="High")
FS1.add_linguistic_variable("Critical", LinguisticVariable([CR1, CR2, CR3], universe_of_discourse=[-1,1]))
FS3.add_linguistic_variable("Critical", LinguisticVariable([CR1, CR2, CR3], universe_of_discourse=[-1,1]))


### OutBandwidth ###

OB1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
OB2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
OB3 = TriangleFuzzySet(0.5, 1, 1, term="High")
FS2.add_linguistic_variable("OutBandwidth", LinguisticVariable([OB1, OB2, OB3], universe_of_discourse=[0,1]))


### OutNetThroughput ###

ONT1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
ONT2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
ONT3 = TriangleFuzzySet(0.5, 1, 1, term="High")
FS2.add_linguistic_variable("OutNetThroughput", LinguisticVariable([ONT1, ONT2, ONT3], universe_of_discourse=[0,1]))


### Latency ### 

L1 = TrapezoidFuzzySet(0, 0, 0.3, 0.5, term="Low")
L2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
L3 = TrapezoidFuzzySet(0.5, 0.7, 1, 1, term="High")
FS2.add_linguistic_variable("Latency", LinguisticVariable([L1, L2, L3], universe_of_discourse=[0,1]))


### FinalOut ###

FO1 = TriangleFuzzySet(-1, -1, 0, term="Low")
FO2 = TriangleFuzzySet(-0.4, 0, 0.4, term="Med")
FO3 = TriangleFuzzySet(0, 1, 1, term="High")
FS2.add_linguistic_variable("FinalOut", LinguisticVariable([FO1, FO2, FO3], universe_of_discourse=[-1,1]))
FS3.add_linguistic_variable("FinalOut", LinguisticVariable([FO1, FO2, FO3], universe_of_discourse=[-1,1]))


### CLP_variation ###

CLP1 = TriangleFuzzySet(-1, -1, 0, term="Negative")
CLP2 = TriangleFuzzySet(-0.35, 0, 0.35, term="Null")
CLP3 = TriangleFuzzySet(0, 1, 1, term="Positive")

FS3.add_linguistic_variable("CLP_variation", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1,1]))


FS1.set_output_function("High_Critical", "max(MemoryUsage, ProcessorLoad) * 2 - 1")
FS1.set_output_function("Regular", "((MemoryUsage + ProcessorLoad) / 2) * 2 - 1")
FS1.set_output_function("Low_Critical", "((MemoryUsage + ProcessorLoad) / 8) * 2 - 1")

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

##################### FIS prediction for the initial provided dataset #####################

df = pd.read_csv('Proj1_TestS.csv', encoding='utf-8')
# df.to_excel('Proj1_TestS.xlsx', index=False)

CLP_var_pred_FIS = []

for index, row in df.iterrows():
    MemoryUsage = row['MemoryUsage']
    ProcessorLoad = row['ProcessorLoad']
    OutNetThroughput = row['OutNetThroughput'] 
    OutBandwidth = row['OutBandwidth']
    Latency = row['Latency']

    CLP_var_pred_FIS.append(CLPVar_prediction(MemoryUsage, ProcessorLoad, OutNetThroughput, OutBandwidth, Latency))
    
print('Results of FIS model in the initial provided dataset:')
print('Mean Squared Error = {}'.format(mean_squared_error(df['CLPVariation'], CLP_var_pred_FIS)))
print('Root Mean Squared Error = {}'.format(mean_squared_error(df['CLPVariation'], CLP_var_pred_FIS, squared=False)))
print('Mean Absolute Error = {}'.format(mean_absolute_error(df['CLPVariation'], CLP_var_pred_FIS)))

#################### NN ####################

import pickle
import numpy as np

 # Neural Network
 
filename = 'NN_model.sav'
model = pickle.load(open(filename, 'rb')) # Load a previously saved model trained on the generated data

##################### NN prediction For the initial provided dataset #####################

y_testS = (np.array(df['CLPVariation']))
X_testS = (np.array(df))[:,:-1]

X_testS2 = np.delete(X_testS, [2,6,7,8,9,10],axis=1)

y_true, ypredNN = y_testS , model.predict(X_testS2)

for index, element in enumerate(ypredNN):
    if(element > 1):
        ypredNN[index] = 1
    elif(element < -1):
        ypredNN[index] = -1
    
print('Results of NN on the test set from initial provided dataset:')
print('Mean Squared Error = {}'.format(mean_squared_error(y_true, ypredNN)))
print('Root Mean Squared Error = {}'.format(mean_squared_error(y_true, ypredNN, squared=False)))
print('Mean Absolute Error = {}'.format(mean_absolute_error(y_true, ypredNN)))
    
df['CLPVariation_FIS'] = CLP_var_pred_FIS
df['CLPVariation_NN'] = ypredNN

# df.to_excel('TestResult.xlsx', index=False)
df.to_csv('TestResult.csv', encoding='utf-8', index=False)