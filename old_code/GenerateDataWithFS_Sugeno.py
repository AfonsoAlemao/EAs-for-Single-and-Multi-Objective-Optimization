from old_code.Fuzzy_Sugeno import CLPVar_prediction
import pandas as pd
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