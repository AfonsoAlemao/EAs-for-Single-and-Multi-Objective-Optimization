from simpful import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FS1 = FuzzySystem()
FS2 = FuzzySystem()
FS3 = FuzzySystem()
FS4 = FuzzySystem()


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

CR1 = TrapezoidFuzzySet(-1, -1, -0.25, 0, term="Low")
CR2 = TriangleFuzzySet(-0.25, 0, 0.25, term="Med")
CR3 = TrapezoidFuzzySet(0, 0.8, 1, 1, term="High")
FS1.add_linguistic_variable("Critical", LinguisticVariable([CR1, CR2, CR3], universe_of_discourse=[-1,1]))
FS4.add_linguistic_variable("Critical", LinguisticVariable([CR1, CR2, CR3], universe_of_discourse=[-1,1]))

### OutBandwidth ###

OB1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
OB2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
OB3 = TriangleFuzzySet(0.5, 1, 1, term="High")
FS2.add_linguistic_variable("OutBandwidth", LinguisticVariable([OB1, OB2, OB3], universe_of_discourse=[0,1]))


### OutNetThroughput ###

ONT1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
ONT2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
ONT3 = TriangleFuzzySet(0.45, 1, 1, term="High")
FS2.add_linguistic_variable("OutNetThroughput", LinguisticVariable([ONT1, ONT2, ONT3], universe_of_discourse=[0,1]))

### Latency ###  ->>>>>> Será melhor usar uma Gaussian MF?

L1 = TrapezoidFuzzySet(0, 0, 0.3, 0.5, term="Low")
L2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
L3 = TriangleFuzzySet(0.6, 1, 1, term="High")
FS2.add_linguistic_variable("Latency", LinguisticVariable([L1, L2, L3], universe_of_discourse=[0,1]))

### FinalOut ###

FO1 = TriangleFuzzySet(-1, -1, 0.2, term="Low")
FO2 = TriangleFuzzySet(-0.2, -0.1, 0.4, term="Med")
FO3 = TriangleFuzzySet(0.4, 1, 1, term="High")
FS3.add_linguistic_variable("FinalOut", LinguisticVariable([FO1, FO2, FO3], universe_of_discourse=[-1,1]))
FS4.add_linguistic_variable("FinalOut", LinguisticVariable([FO1, FO2, FO3], universe_of_discourse=[-1,1]))

### CLP_variation ###

CLP1 = TriangleFuzzySet(-1, -1, 0, term="Negative")
CLP2 = TriangleFuzzySet(-0.35, 0, 0.35, term="Null")
CLP3 = TriangleFuzzySet(0, 1, 1, term="Positive")

FS4.add_linguistic_variable("CLP_variation", LinguisticVariable([CLP1, CLP2, CLP3], universe_of_discourse=[-1,1]))


FS1.add_rules([
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Low) THEN (Critical IS Low)",
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Med) THEN (Critical IS Low)", # was Med
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS High) THEN (Critical IS High)", 
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Low) THEN (Critical IS Low)", # was Med 
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Med) THEN (Critical IS Low)",
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS High) THEN (Critical IS High)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS Low) THEN (Critical IS High)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS Med) THEN (Critical IS High)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS High) THEN (Critical IS High)",
])

FS2.add_rules([
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Low) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Low) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Low) AND (Latency IS High) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Med) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Med) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS Med) AND (Latency IS High) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS High) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS High) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS High) AND (OutBandwidth IS High) AND (Latency IS High) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS High) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS High) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS High) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Low) AND (Latency IS High) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS Med) AND (Latency IS High) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS Low) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS Med) THEN (FinalOut IS High)",
    "IF (OutNetThroughput IS Med) AND (OutBandwidth IS High) AND (Latency IS High) THEN (FinalOut IS High)",
])

FS3.add_rules([
    "IF (Critical IS Low) THEN (CLP_variation IS Positive)",
    "IF (Critical IS Med) AND (FinalOut IS Low) THEN (CLP_variation IS Negative)",
    "IF (Critical IS Med) AND (FinalOut IS Med) THEN (CLP_variation IS Null)",
    "IF (Critical IS Med) AND (FinalOut IS High) THEN (CLP_variation IS Positive)",
    "IF (Critical IS High) AND (FinalOut IS Low) THEN (CLP_variation IS Negative)",
    "IF (Critical IS High) AND (FinalOut IS Med) THEN (CLP_variation IS Negative)",
    "IF (Critical IS High) AND (FinalOut IS High) THEN (CLP_variation IS Negative)",
	])

import pandas as pd

df = pd.read_csv('Proj1_TestS.csv', encoding='utf-8')
df.to_excel('Proj1_TestS.xlsx', index=False)

CLP_var_pred = []
Critical_pred = []
Out_preds = []
FinalOut_preds = []

# Verify errors in test set

for index, row in df.iterrows():
    MemoryUsage = row['MemoryUsage']
    ProcessorLoad = row['ProcessorLoad']
    OutNetThroughput = row['OutNetThroughput'] 
    InpNetThroughput = row['InpNetThroughput']
    OutBandwidth = row['OutBandwidth']
    Latency = row['Latency']

    FS1.set_variable("MemoryUsage", MemoryUsage) 
    FS1.set_variable("ProcessorLoad", ProcessorLoad) 

    CR_pred = FS1.inference()['Critical']
    # print(CR_pred)
    Critical_pred.append(CR_pred)

    FS2.set_variable("OutNetThroughput", OutNetThroughput)
    FS2.set_variable("OutBandwidth", OutBandwidth) 

    Out_pred = FS2.inference()['Out']
    # print(Boss_pred)
    Out_preds.append(Out_pred)
    
    FS3.set_variable("Out", Out_pred)
    FS3.set_variable("Latency", Latency) 

    FinalOut_pred = FS3.inference()['FinalOut']
    # print(aux_pred)
    FinalOut_preds.append(FinalOut_pred)

    FS4.set_variable("FinalOut", FinalOut_pred) 
    FS4.set_variable("Critical", CR_pred) 

    CLP_variation_pred = FS4.inference()['CLP_variation']
    # print(CLP_variation_pred)
    CLP_var_pred.append(CLP_variation_pred)
    
df['CLPVariation_pred'] = CLP_var_pred
df['Critical'] = Critical_pred
df['Out'] = Out_preds
df['FinalOut'] = FinalOut_preds

df['erro_CLP'] = abs(df['CLPVariation_pred'] - df['CLPVariation'])

df = df.drop(columns=['V_MemoryUsage','V_ProcessorLoad','V_InpNetThroughput','V_OutNetThroughput','V_OutBandwidth','V_Latency'])

df.to_excel('TestResult.xlsx', index=False)
df.to_csv('TestResult.csv', encoding='utf-8', index=False)


# def generateDataset():
#     new_df = pd.DataFrame()

#     range_considered = [i / 10 for i in range(11)] 
#     new_MU = []
#     new_PL = []
#     new_OT = []
#     new_IT = []
#     new_OBW = []
#     new_Lat = []
#     new_CLP_var = []

#     # Generate dataset

#     for nMU in range_considered:
#         for nPL in range_considered:
#             for nOT in range_considered:
#                 for nIT in range_considered:
#                     for nOBW in range_considered:
#                         for nL in range_considered:
#                             new_MU.append(nMU)
#                             new_PL.append(nPL)
#                             new_OT.append(nOT)
#                             new_IT.append(nIT)
#                             new_OBW.append(nOBW)
#                             new_Lat.append(nL)
                            
#                             FS1.set_variable("MemoryUsage", nMU) 
#                             FS1.set_variable("ProcessorLoad", nPL) 

#                             CR_pred = FS1.inference()['Critical']

#                             FS2.set_variable("OutBandwidth", nOBW)
#                             FS2.set_variable("Critical", CR_pred) 

#                             Boss_pred = FS2.inference()['Boss']
                            
#                             nT = nOT - nIT
#                             FS3.set_variable("Throughput", nT)
#                             FS3.set_variable("Latency", nL) 

#                             aux_pred = FS3.inference()['aux']

#                             FS4.set_variable("Boss", Boss_pred) 
#                             FS4.set_variable("aux", aux_pred) 

#                             CLP_variation_pred = FS4.inference()['CLP_variation']
#                             new_CLP_var.append(CLP_variation_pred)
                            
#     new_df['MemoryUsage'] = new_MU
#     new_df['ProcessorLoad'] = new_PL
#     new_df['OutNetThroughput'] = new_OT
#     new_df['InpNetThroughput'] = new_IT
#     new_df['OutBandwidth'] = new_OBW
#     new_df['Latency'] = new_Lat
#     new_df['CLP_variation'] = new_CLP_var
  
#     new_df.to_excel('Proj1_TestS_GeneratedData.xlsx', index=False)
#     new_df.to_csv('Proj1_TestS_GeneratedData.csv', index=False, encoding=False)

# generateDataset()