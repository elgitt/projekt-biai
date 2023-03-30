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

#wektory z ch1
Okg_L_vec = Okg_L['ch1']
Okg_R_vec = Okg_R['ch1']
Okg_W_vec = Okg_W['ch1']
Ikg_W_vec = Ikg_W['ch1']
Ikg_L_vec = Ikg_L['ch1']
Ikg_R_vec = Ikg_R['ch1']

#decymacja sygnału
Okg_L_decimated = Okg_L_vec[::100]
Okg_W_decimated = Okg_W_vec[::100]
Okg_R_decimated = Okg_R_vec[::100]
Ikg_W_decimated = Ikg_W_vec[::100]
Ikg_L_decimated = Ikg_L_vec[::100]
Ikg_R_decimated = Ikg_R_vec[::100]

test = Okg_L.to_parquet()
print(Okg_L_decimated)

