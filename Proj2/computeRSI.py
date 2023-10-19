import pandas as pd
import numpy as np

AAL = pd.read_csv('ACI_Project2_2324_Data/AAL.csv', encoding='utf-8', sep=';') 
AAPL = pd.read_csv('ACI_Project2_2324_Data/AAPL.csv', encoding='utf-8', sep=';') 
AMZN = pd.read_csv('ACI_Project2_2324_Data/AMZN.csv', encoding='utf-8', sep=';') 
BAC = pd.read_csv('ACI_Project2_2324_Data/BAC.csv', encoding='utf-8', sep=';') 
F = pd.read_csv('ACI_Project2_2324_Data/F.csv', encoding='utf-8', sep=';') 
GOOG = pd.read_csv('ACI_Project2_2324_Data/GOOG.csv', encoding='utf-8', sep=';') 
IBM = pd.read_csv('ACI_Project2_2324_Data/IBM.csv', encoding='utf-8', sep=';') 
INTC = pd.read_csv('ACI_Project2_2324_Data/INTC.csv', encoding='utf-8', sep=';') 
NVDA = pd.read_csv('ACI_Project2_2324_Data/NVDA.csv', encoding='utf-8', sep=';') 
XOM = pd.read_csv('ACI_Project2_2324_Data/XOM.csv', encoding='utf-8', sep=';') 

Gain = []
Loss = []
Gain_av = []
Loss_av = []
RS = []
RSI = []

print(AAL)

close_prev = -1
for ind, row in AAL.iterrows():
    if ind == 0:
        Gain.append(0)
        Loss.append(0)
        Gain_av.append(0)
        Loss_av.append(0)
        RS.append(0)
        RSI.append(0)
    elif ind > 0:
        balance = row['Close'] - close_prev 
        
        if balance > 0:
            gain = balance
            loss = 0
        else:
            loss = balance
            gain = 0
            
        Gain.append(gain)
        Loss.append(loss)
        
        if ind <= 6:
            Gain_av.append(0)
            Loss_av.append(0)
            RS.append(0)
            RSI.append(0)
        else:
            gain_av = np.mean(Gain[-6:])
            Gain_av.append(gain_av)
            loss_av = np.mean(Loss[-6:])
            Loss_av.append(loss_av)
            rs = gain_av / loss_av
            RS.append(rs)
            rsi = 100 - 100 / (1 + rs)
            RSI.append(rsi)
            
    close_prev = row['Close']
    
AAL['Gain'] = Gain
AAL['Loss'] = Loss
AAL['Gain_av'] = Gain_av
AAL['Loss_av'] = Loss_av
AAL['RS'] = RS
AAL['RSI'] = RSI

AAL.to_csv("ACI_Project2_2324_Data/AAL.csv", 
            index = None,
            header=True, encoding='utf-8')