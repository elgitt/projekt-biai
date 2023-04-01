import csv
import pickle
import os
from itertools import islice
import pandas as pd


def convert_csv_to_bin(input_file: str, output_file: str):
    """Nie używać"""
    with open('data/' + input_file + '.csv', 'r') as csvfile:
        reader = csv.reader(islice(csvfile, 22, None))
        data = [row for row in reader]

    filepath = os.path.join(os.getcwd(), 'data/' + output_file + '.bin')

    with open(filepath, 'wb') as bf:
        pickle.dump(data, bf)
        bf.close()


def refactor_binary_to_data_frame(filepath: str):
    with open('data/' + filepath + '.csv') as csvfile:
        reader = csv.reader(islice(csvfile, 22, None))
        data = [row for row in reader]

    df = pd.DataFrame(data, columns=['Czas', 'CH1'])
    window_size = 3
    rolling_mean = df['Czas'].rolling(window_size).mean()

    # utwórz nową kolumnę w DataFrame z wynikiem filtrowania z przesuwnym oknem
    df['Sr'] = rolling_mean
    print(df)
    df.to_parquet('data/test.parquet', compression='snappy', engine='pyarrow', row_group_size=10000)
    return df


def change_to_vector(df):
    x = df['Czas'].to_numpy()
    y = df['CH1'].to_numpy()
    return x, y
