import csv
import pickle
import os
from itertools import islice
import pandas as pd
import pyarrow.parquet as pq


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
    x = df['Sr'].to_numpy()
    y = df['CH1'].to_numpy()
    return x, y


def x(file_name):
    test = save_csv_as_parquet(file_name)
    m5 = median_method(test, 5)
    m9 = median_method(test, 9)
    m11 = median_method(test, 11)
    m15 = median_method(test, 15)
    m21 = median_method(test, 21)
    return {'m5': m5, "m9": m9}
