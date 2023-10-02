import pandas as pd

df = pd.read_csv('ACI23-24_Proj1_SampleData.csv', encoding='utf-8')
df.to_excel('ACI23-24_Proj1_SampleData.xlsx', index=False)