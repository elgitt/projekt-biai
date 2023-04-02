import csv
import pickle
import os
from itertools import islice
import pandas as pd
import pyarrow.parquet as pq
import numpy as np


def convert_csv_to_bin(input_file: str, output_file: str):
    """Nie używać"""
    with open('data/' + input_file + '.csv', 'r') as csvfile:
        reader = csv.reader(islice(csvfile, 22, None))
        data = [row for row in reader]

    filepath = os.path.join(os.getcwd(), 'data/' + output_file + '.bin')

    with open(filepath, 'wb') as bf:
        pickle.dump(data, bf)
        bf.close()


def save_csv_as_parquet(filepath: str):
    with open('data/' + filepath + '.csv') as csvfile:
        reader = csv.reader(islice(csvfile, 22, None))
        data = [row for row in reader]

    df = pd.DataFrame(data, columns=['Czas', 'CH1'])
    df['Czas'] = pd.to_numeric(df['Czas'], errors='coerce')
    df['Czas'] = df['Czas'].fillna((df['Czas'].shift() + df['Czas'].shift(-1)) / 2)
    df.to_parquet('data/test.parquet', compression='snappy', engine='pyarrow', row_group_size=10000)
    return df


def read_from_parquet(filepath):
    table = pq.read_table('data/' + filepath + '.parquet')
    df = table.to_pandas()
    return df


def median_method(df, window_size):
    rolling_mean = df['Czas'].rolling(window_size).mean()
    df['Sr'] = rolling_mean
    return df


def change_to_vector(df):
    xv1 = df['Sr'].to_numpy()
    yv1 = df['CH1'].to_numpy()

    nan_indices = np.isnan(xv1)
    xv1 = np.delete(xv1, np.where(nan_indices))
    yv1 = np.delete(yv1, np.where(nan_indices))

    return xv1, yv1


def return_median_mx(file_name):
    test = save_csv_as_parquet(file_name)
    m5 = median_method(test, 5)
    m9 = median_method(test, 9)
    m11 = median_method(test, 11)
    m15 = median_method(test, 15)
    m21 = median_method(test, 21)
    return {'m5': m5, "m9": m9, "m11": m11, "m15": m15, "m21": m21, "parquet": test}


def decimate_method(df, decimation_step):
    df_decimated = df.iloc[::decimation_step, :]
    return df_decimated
