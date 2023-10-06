from simpful import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# plt.show()


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
# plt.show()

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
# plt.show()

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
# plt.show()

### OutNetThroughput ###

ONT1 = TriangleFuzzySet(0, 0, 0.5, term="Low")
ONT2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
ONT3 = TriangleFuzzySet(0.45, 1, 1, term="High")
FS2.add_linguistic_variable("OutNetThroughput", LinguisticVariable([ONT1, ONT2, ONT3], universe_of_discourse=[0,1]))

plt.figure(4)
plt.plot([0, 0, 0.5], [0,1,0])
plt.plot([0.3, 0.5, 0.7], [0,1,0])
plt.plot([0.45, 1, 1], [0,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.title('Membership Function OutNetThroughput')
plt.xlabel('OutNetThroughput')
plt.ylabel('MF')
# plt.show()

### Latency ###  ->>>>>> Será melhor usar uma Gaussian MF?

L1 = TrapezoidFuzzySet(0, 0, 0.3, 0.5, term="Low")
L2 = TriangleFuzzySet(0.3, 0.5, 0.7, term="Med")
L3 = TrapezoidFuzzySet(0.6, 0.8, 1, 1, term="High")
FS2.add_linguistic_variable("Latency", LinguisticVariable([L1, L2, L3], universe_of_discourse=[0,1]))

plt.figure(5)
plt.plot([0, 0, 0.3, 0.5], [0,1,1,0])
plt.plot([0.3, 0.5, 0.7], [0,1,0])
plt.plot([0.6, 0.8, 1, 1], [0,1,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.title('Membership Function Latency')
plt.xlabel('Latency')
plt.ylabel('MF')
# plt.show()

### FinalOut ###

FO1 = TriangleFuzzySet(-1, -1, 0.2, term="Low")
FO2 = TriangleFuzzySet(-0.2, -0.1, 0.4, term="Med")
FO3 = TriangleFuzzySet(0.4, 1, 1, term="High")
FS2.add_linguistic_variable("FinalOut", LinguisticVariable([FO1, FO2, FO3], universe_of_discourse=[-1,1]))
FS3.add_linguistic_variable("FinalOut", LinguisticVariable([FO1, FO2, FO3], universe_of_discourse=[-1,1]))

plt.figure(6)
plt.plot([-1, -1, 0.2], [0,1,0])
plt.plot([-0.2, -0.1, 0.4], [0,1,0])
plt.plot([0.4, 1, 1], [0,1,0])
plt.legend(['Low', 'Med', 'High'])
plt.ylim([0, 1.05])
plt.xlim([-1, 1])
plt.title('Membership Function FinalOut')
plt.xlabel('FinalOut')
plt.ylabel('MF')
# plt.show()

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
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Med) THEN (Critical IS Low_Critical)", # was Med
    "IF (MemoryUsage IS Low) AND (ProcessorLoad IS High) THEN (Critical IS High_Critical)", 
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Low) THEN (Critical IS Low_Critical)", # was Med 
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Med) THEN (Critical IS Regular)",
    "IF (MemoryUsage IS Med) AND (ProcessorLoad IS High) THEN (Critical IS High_Critical)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS Low) THEN (Critical IS High_Critical)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS Med) THEN (Critical IS High_Critical)",
    "IF (MemoryUsage IS High) AND (ProcessorLoad IS High) THEN (Critical IS High_Critical)",
])

FS2.set_output_function("HIGH_LAT", "max(min((0.0*(1-OutNetThroughput) + 0.05*(1-OutBandwidth) + 1*Latency+0.1)*2-1, 1), -1)")
FS2.set_output_function("LOW_OBW", "max(min((0.4*(1-OutNetThroughput) + 0.9*(1-OutBandwidth) + 0.5*Latency)*2-1, 1), -1)")
FS2.set_output_function("LOW_ONT", "max(min((0.6*(1-OutNetThroughput) + 0.2*(1-OutBandwidth) + 0.9*Latency)*2-1, 1), -1)")
FS2.set_output_function("OTHER", "max(min((0.6*(1-OutNetThroughput) + 0.3*(1-OutBandwidth) + 0.63*Latency)*2-1, 1), -1)")

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

FS3.set_output_function("VERY_CRITICAL", "-0.91*Critical + 0.09*FinalOut")
FS3.set_output_function("NOT_CRITICAL", "-0.15*Critical + FinalOut*0.85")

FS3.add_rules([
    "IF (Critical IS Low) THEN (CLP_variation IS VERY_CRITICAL)",
    "IF (Critical IS Med) AND (FinalOut IS Low) THEN (CLP_variation IS NOT_CRITICAL)",
    "IF (Critical IS Med) AND (FinalOut IS Med) THEN (CLP_variation IS NOT_CRITICAL)",
    "IF (Critical IS Med) AND (FinalOut IS High) THEN (CLP_variation IS NOT_CRITICAL)",
    "IF (Critical IS High) AND (FinalOut IS Low) THEN (CLP_variation IS VERY_CRITICAL)",
    "IF (Critical IS High) AND (FinalOut IS Med) THEN (CLP_variation IS VERY_CRITICAL)",
    "IF (Critical IS High) AND (FinalOut IS High) THEN (CLP_variation IS VERY_CRITICAL)",
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
    FS2.set_variable("Latency", Latency) 

    FinalOut_pred = FS2.inference()['FinalOut']
    # print(aux_pred)
    FinalOut_preds.append(FinalOut_pred)

    FS3.set_variable("FinalOut", FinalOut_pred) 
    FS3.set_variable("Critical", CR_pred) 

    CLP_variation_pred = FS3.inference()['CLP_variation']
    # print(CLP_variation_pred)
    CLP_var_pred.append(CLP_variation_pred)
    
df['CLPVariation_pred'] = CLP_var_pred
df['Critical'] = Critical_pred
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
