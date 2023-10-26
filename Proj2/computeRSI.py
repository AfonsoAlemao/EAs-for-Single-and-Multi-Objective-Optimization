# Calculation of the RSI for:
#   7-day period
#   14-day period
#   21-day period

import pandas as pd
import numpy as np

AAL = pd.read_csv('ACI_Project2_2324_Data/AAL.csv', encoding='utf-8', sep= ';') 
AAPL = pd.read_csv('ACI_Project2_2324_Data/AAPL.csv', encoding='utf-8', sep=';') 
AMZN = pd.read_csv('ACI_Project2_2324_Data/AMZN.csv', encoding='utf-8', sep=';') 
BAC = pd.read_csv('ACI_Project2_2324_Data/BAC.csv', encoding='utf-8', sep=';') 
F = pd.read_csv('ACI_Project2_2324_Data/F.csv', encoding='utf-8', sep=';') 
GOOG = pd.read_csv('ACI_Project2_2324_Data/GOOG.csv', encoding='utf-8', sep=';') 
IBM = pd.read_csv('ACI_Project2_2324_Data/IBM.csv', encoding='utf-8', sep=';') 
INTC = pd.read_csv('ACI_Project2_2324_Data/INTC.csv', encoding='utf-8', sep=';') 
NVDA = pd.read_csv('ACI_Project2_2324_Data/NVDA.csv', encoding='utf-8', sep=';') 
XOM = pd.read_csv('ACI_Project2_2324_Data/XOM.csv', encoding='utf-8', sep=';') 

csvs = [AAL, AAPL, AMZN, BAC, F, GOOG, IBM, INTC, NVDA, XOM]
csvs_names = ['AAL', 'AAPL', 'AMZN', 'BAC', 'F', 'GOOG', 'IBM', 'INTC', 'NVDA', 'XOM']

for i, csv in enumerate(csvs):
    Gain = []
    Loss = []
    Gain_av_7days = []
    Loss_av_7days = []
    RS_7days = []
    RSI_7days = []
    Gain_av_14days = []
    Loss_av_14days = []
    RS_14days = []
    RSI_14days = []
    Gain_av_21days = []
    Loss_av_21days = []
    RS_21days = []
    RSI_21days = []
    close_prev = -1
    for ind, row in csv.iterrows():
        if ind == 0:
            Gain.append(None)
            Loss.append(None)
            Gain_av_7days.append(None)
            Loss_av_7days.append(None)
            RS_7days.append(None)
            RSI_7days.append(None)
            Gain_av_14days.append(None)
            Loss_av_14days.append(None)
            RS_14days.append(None)
            RSI_14days.append(None)
            Gain_av_21days.append(None)
            Loss_av_21days.append(None)
            RS_21days.append(None)
            RSI_21days.append(None)
        elif ind > 0:
            balance = row['Close'] - close_prev 
            
            if balance > 0:
                gain = balance
                loss = 0
            else:
                loss = abs(balance)
                gain = 0
                
            Gain.append(gain)
            Loss.append(loss)
            
            if ind <= 6:
                Gain_av_7days.append(None)
                Loss_av_7days.append(None)
                RS_7days.append(None)
                RSI_7days.append(None)
            else:
                gain_av_7days = np.mean(Gain[-7:])
                Gain_av_7days.append(gain_av_7days)
                loss_av_7days = np.mean(Loss[-7:])
                Loss_av_7days.append(loss_av_7days)
                if(loss_av_7days == 0):
                    RS_7days.append(None)
                    RSI_7days.append(100)
                else:
                    rs_7days = gain_av_7days / loss_av_7days
                    RS_7days.append(rs_7days)
                    rsi_7days = 100 - 100 / (1 + rs_7days)
                    RSI_7days.append(rsi_7days)
            
            if ind <= 13:
                Gain_av_14days.append(None)
                Loss_av_14days.append(None)
                RS_14days.append(None)
                RSI_14days.append(None)
            else:
                gain_av_14days = np.mean(Gain[-14:])
                Gain_av_14days.append(gain_av_14days)
                loss_av_14days = np.mean(Loss[-14:])
                Loss_av_14days.append(loss_av_14days)
                if(loss_av_14days == 0):
                    RS_14days.append(None)
                    RSI_14days.append(100)
                else:
                    rs_14days = gain_av_14days / loss_av_14days
                    RS_14days.append(rs_14days)
                    rsi_14days = 100 - 100 / (1 + rs_14days)
                    RSI_14days.append(rsi_14days)
                    
            if ind <= 20:
                Gain_av_21days.append(None)
                Loss_av_21days.append(None)
                RS_21days.append(None)
                RSI_21days.append(None)
            else:
                gain_av_21days = np.mean(Gain[-21:])
                Gain_av_21days.append(gain_av_21days)
                loss_av_21days = np.mean(Loss[-21:])
                Loss_av_21days.append(loss_av_21days)
                if(loss_av_21days == 0):
                    RS_21days.append(None)
                    RSI_21days.append(100)
                else:
                    rs_21days = gain_av_21days / loss_av_21days
                    RS_21days.append(rs_21days)
                    rsi_21days = 100 - 100 / (1 + rs_21days)
                    RSI_21days.append(rsi_21days)
                
        close_prev = row['Close']
        
    csv['Gain'] = Gain
    csv['Loss'] = Loss
    csv['Gain_av_7days'] = Gain_av_7days
    csv['Loss_av_7days'] = Loss_av_7days
    csv['RS_7days'] = RS_7days
    csv['RSI_7days'] = RSI_7days
    csv['Gain_av_14days'] = Gain_av_14days
    csv['Loss_av_14days'] = Loss_av_14days
    csv['RS_14days'] = RS_14days
    csv['RSI_14days'] = RSI_14days
    csv['Gain_av_21days'] = Gain_av_21days
    csv['Loss_av_21days'] = Loss_av_21days
    csv['RS_21days'] = RS_21days
    csv['RSI_21days'] = RSI_21days

    csv.to_csv('ACI_Project2_2324_Data/' + csvs_names[i] + '.csv', 
                index = None,
                header=True, encoding='utf-8', sep=';')
    