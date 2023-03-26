import pandas as pd
import pyarrow as pr
import fastparquet as fpq

with open(r"data/0kg_L.csv", 'r') as fp:
    for count, line in enumerate(fp):
        pass
#print('Total Lines', count + 1)

Okg_L = pd.read_csv('data/0kg_L.csv', skiprows= 21, low_memory=False, nrows=19000, header=2)
Okg_R = pd.read_csv('data/0kg_R.csv', skiprows= 21, low_memory=False, nrows=19000, header=2)
Okg_W = pd.read_csv('data/0kg_W.csv', skiprows= 21, low_memory=False, nrows=19000, header=2)
Ikg_L = pd.read_csv('data/1kgL_L.csv', skiprows= 21, low_memory=False, nrows=19000, header=2)
Ikg_R = pd.read_csv('data/1kgL_R.csv', skiprows= 21, low_memory=False, nrows=19000, header=2)
Ikg_W = pd.read_csv('data/1kgL_W.csv', skiprows= 21, low_memory=False, nrows=19000, header=2)

Okg_L.columns = ['czas','ch1']
Okg_R.columns = ['czas','ch1']
Okg_W.columns = ['czas','ch1']
Ikg_L.columns = ['czas','ch1']
Ikg_R.columns = ['czas','ch1']
Ikg_W.columns = ['czas','ch1']

test = Okg_L.to_parquet()
print(Okg_L)
