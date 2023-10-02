from simpful import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FS1 = FuzzySystem()
FS2 = FuzzySystem()
FS3 = FuzzySystem()
FS4 = FuzzySystem()

MU1 = TriangleFuzzySet(0, 0, 0.6, term="Low")
MU2 = TriangleFuzzySet(0.45, 0.575, 0.7, term="Med")
MU3 = TriangleFuzzySet(0.7, 1, 1, term="High")
FS1.add_linguistic_variable("MemoryUsage", LinguisticVariable([MU1, MU2, MU3], universe_of_discourse=[0,1]))

PL1 = TriangleFuzzySet(0, 0, 0.6, term="Low")
PL2 = TriangleFuzzySet(0.45, 0.575, 0.7, term="Med")
PL3 = TriangleFuzzySet(0.7, 1, 1, term="High")
FS1.add_linguistic_variable("ProcessorLoad", LinguisticVariable([PL1, PL2, PL3], universe_of_discourse=[0,1]))

CLP1 = TriangleFuzzySet(-1, -1, 0, term="Negative")
CLP2 = TriangleFuzzySet(-0.35, 0, 0.35, term="Null")
CLP3 = TriangleFuzzySet(0, 1, 1, term="Positive")
FS4.add_linguistic_variable("CLP_variation", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1,1]))

T1 = TriangleFuzzySet(-1, -1, 0, term="Negative")
T2 = TriangleFuzzySet(-0.2, 0, 0.2, term="Null")
T3 = TriangleFuzzySet(0, 1, 1, term="Positive")
FS3.add_linguistic_variable("Throughput", LinguisticVariable([T1, T2, T3], universe_of_discourse=[-1,1]))

OB1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
OB2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
OB3 = TriangleFuzzySet(0.5, 1, 1, term="High")
FS2.add_linguistic_variable("OutBandwidth", LinguisticVariable([OB1, OB2, OB3], universe_of_discourse=[0,1]))

L1 = TriangleFuzzySet(0, 0, 0.6, term="Low")
L2 = TriangleFuzzySet(0.4, 0.55, 0.7, term="Med")
L3 = TriangleFuzzySet(0.7, 1, 1, term="High")
FS3.add_linguistic_variable("Latency", LinguisticVariable([L1, L2, L3], universe_of_discourse=[0,1]))

CR1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
CR2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
CR3 = TriangleFuzzySet(0.5, 1, 1, term="High")
FS1.add_linguistic_variable("Critical", LinguisticVariable([CR1, CR2, CR3], universe_of_discourse=[0,1]))
FS2.add_linguistic_variable("Critical", LinguisticVariable([CR1, CR2, CR3], universe_of_discourse=[0,1]))

Boss1 = TriangleFuzzySet(-1, -1, -0.25, term="Negative")
Boss2 = TriangleFuzzySet(-0.75, 0, 0.75, term="Other")
Boss3 = TriangleFuzzySet(0.25, 1, 1, term="NNegative")
FS2.add_linguistic_variable("Boss", LinguisticVariable([Boss1, Boss2, Boss3], universe_of_discourse=[-1,1]))
FS4.add_linguistic_variable("Boss", LinguisticVariable([Boss1, Boss2, Boss3], universe_of_discourse=[-1,1]))

aux1 = TriangleFuzzySet(-1, -1, 0, term="Negative")
aux2 = TriangleFuzzySet(-0.35, 0, 0.35, term="Null")
aux3 = TriangleFuzzySet(0, 1, 1, term="Positive")
FS3.add_linguistic_variable("aux", LinguisticVariable([aux1, aux2, aux3], universe_of_discourse=[-1,1]))
FS4.add_linguistic_variable("aux", LinguisticVariable([aux1, aux2, aux3], universe_of_discourse=[-1,1]))

FS1.add_rules([
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Low) THEN (Critical IS Low)",
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Med) THEN (Critical IS Med)",
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS High) THEN (Critical IS High)",
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Low) THEN (Critical IS Med)",
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Med) THEN (Critical IS Med)",
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS High) THEN (Critical IS High)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS Low) THEN (Critical IS High)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS Med) THEN (Critical IS High)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS High) THEN (Critical IS High)",
])

FS2.add_rules([
    "IF (Critical IS High) AND (OutBandwidth IS Low) THEN (Boss IS NNegative)",
    "IF (Critical IS High) AND (OutBandwidth IS Med) THEN (Boss IS Negative)",
    "IF (Critical IS High) AND (OutBandwidth IS High) THEN (Boss IS Negative)",
    "IF (Critical IS Med) AND (OutBandwidth IS Low) THEN (Boss IS NNegative)",
    "IF (Critical IS Med) AND (OutBandwidth IS Med) THEN (Boss IS Other)",
    "IF (Critical IS Med) AND (OutBandwidth IS High) THEN (Boss IS Other)",
    "IF (Critical IS Low) AND (OutBandwidth IS Low) THEN (Boss IS NNegative)",
    "IF (Critical IS Low) AND (OutBandwidth IS Med) THEN (Boss IS Other)",
    "IF (Critical IS Low) AND (OutBandwidth IS High) THEN (Boss IS Other)",
])

FS3.add_rules([
    "IF (Throughput IS Negative) AND (Latency IS Low) THEN (aux IS Positive)",
    "IF (Throughput IS Negative) AND (Latency IS Med) THEN (aux IS Positive)",
    "IF (Throughput IS Negative) AND (Latency IS High) THEN (aux IS Positive)",
    "IF (Throughput IS Null) AND (Latency IS Low) THEN (aux IS Negative)",
    "IF (Throughput IS Null) AND (Latency IS Med) THEN (aux IS Null)",
    "IF (Throughput IS Null) AND (Latency IS High) THEN (aux IS Positive)",
    "IF (Throughput IS Positive) AND (Latency IS Low) THEN (aux IS Negative)",
    "IF (Throughput IS Positive) AND (Latency IS Med) THEN (aux IS Negative)",
    "IF (Throughput IS Positive) AND (Latency IS High) THEN (aux IS Positive)",
])

FS4.add_rules([
    "IF (Boss IS NNegative) AND (aux IS Negative) THEN (CLP_variation IS Null)",
    "IF (Boss IS NNegative) AND (aux IS Null) THEN (CLP_variation IS Null)",
    "IF (Boss IS NNegative) AND (aux IS Positive) THEN (CLP_variation IS Positive)",
    "IF (Boss IS Negative) AND (aux IS Negative) THEN (CLP_variation IS Negative)",
    "IF (Boss IS Negative) AND (aux IS Null) THEN (CLP_variation IS Negative)",
    "IF (Boss IS Negative) AND (aux IS Positive) THEN (CLP_variation IS Negative)",
    "IF (Boss IS Other) AND (aux IS Negative) THEN (CLP_variation IS Negative)",
    "IF (Boss IS Other) AND (aux IS Null) THEN (CLP_variation IS Null)",
    "IF (Boss IS Other) AND (aux IS Positive) THEN (CLP_variation IS Positive)",
	])

import pandas as pd

df = pd.read_csv('ACI23-24_Proj1_SampleData.csv', encoding='utf-8')

CLP_var_pred = []
Critical_pred = []
Boss_preds = []
aux_preds = []

for index, row in df.iterrows():
    MemoryUsage = row['MemoryUsage']
    ProcessorLoad = row['ProcessorLoad']
    Throughput = row['OutNetThroughput'] - row['InpNetThroughput']
    OutBandwidth = row['OutBandwidth']
    Latency = row['Latency']

    FS1.set_variable("MemoryUsage", MemoryUsage) 
    FS1.set_variable("ProcessorLoad", ProcessorLoad) 

    CR_pred = FS1.inference()['Critical']
    # print(CR_pred)
    Critical_pred.append(CR_pred)

    FS2.set_variable("OutBandwidth", OutBandwidth)
    FS2.set_variable("Critical", CR_pred) 

    Boss_pred = FS2.inference()['Boss']
    # print(Boss_pred)
    Boss_preds.append(Boss_pred)
    
    FS3.set_variable("Throughput", 0.05)
    FS3.set_variable("Latency", Latency) 

    aux_pred = FS3.inference()['aux']
    # print(aux_pred)
    aux_preds.append(aux_pred)

    FS4.set_variable("Boss", Boss_pred) 
    FS4.set_variable("aux", aux_pred) 

    CLP_variation_pred = FS4.inference()['CLP_variation']
    # print(CLP_variation_pred)
    CLP_var_pred.append(CLP_variation_pred)
    
df['CLP_var_pred'] = CLP_var_pred
df['Critical'] = Critical_pred
df['Boss'] = Boss_preds
df['aux'] = aux_preds

df['erro_CLP'] = df['CLP_var_pred'] - df['CLPVariation']

df.to_excel('ACI23-24_Proj1_SampleData.xlsx', index=False)