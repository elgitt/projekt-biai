import csv
import pickle
import os
from itertools import islice
import pandas as pd


def convert_csv_to_bin(input_file: str, output_file: str):
    with open('data/' + input_file + '.csv', 'r') as csvfile:
        reader = csv.reader(islice(csvfile, 22, None))
        data = [row for row in reader]

    filepath = os.path.join(os.getcwd(), 'data/' + output_file + '.bin')

    with open(filepath, 'wb') as bf:
        pickle.dump(data, bf)
        bf.close()


def refactor_binary_to_data_frame(filepath: str):
    with open('data/' + filepath + '.bin', 'rb') as bf:
        data = pickle.load(bf)
    df = pd.DataFrame(data, columns=['x', 'y'])
    return df


def change_to_vector(df):
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    return x, y