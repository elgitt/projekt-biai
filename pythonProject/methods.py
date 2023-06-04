import csv
import pickle
import os
from itertools import islice
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def extract_excel_sheet(sheet_name: str, output_file: str):
    temp_file = 'measurements/temp.csv'
    excel_file = pd.read_excel('measurements/Synteza.xlsx', sheet_name=sheet_name)
    excel_file.to_csv(temp_file, index=False)
    df = pd.read_csv(temp_file)
    df = df[20:]
    df = df.iloc[:, :3]
    df.columns = ['TIME', 'CH1', 'CH2']
    df.to_csv('measurements/' + output_file + '.csv', index=False)
    os.remove(temp_file)


# old code -------------------------------------------------------------------------------------------------------------
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
    df[' Czas'] = df['Czas'].fillna((df['Czas'].shift() + df['Czas'].shift(-1)) / 2)
    df.to_parquet('data/' + filepath + '.parquet', compression='snappy', engine='pyarrow', row_group_size=10000)
    return df


def read_from_parquet(filepath):
    table = pq.read_table('data/' + filepath + '.parquet')
    df = table.to_pandas()
    return df


def median_method(df, window_size):
    rolling_median = df['CH1'].rolling(window_size).median()
    df['CH1_filtered'] = rolling_median
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


def time_division(df):
    time_step = 20
    df_divided = df['time_bin'] = (df['Czas'] / time_step).astype(int)
    return df_divided


def preprocess_data(df, decimation_step, filter_window_size, time_step):
    # wykonanie decymacji z zadanym krokiem
    df_decimated = df.iloc[::decimation_step, :]

    # wykonanie filtracji medianowej z oknem przesuwnym o zadanej wielkości
    df_filtered = df_decimated.copy()
    df_filtered['CH1_filtered'] = df_filtered['CH1'].rolling(window=filter_window_size).median()

    # podzielenie przebiegu na odcinki czasu o zadanej długości
    df_time_binned = df_filtered.copy()
    df_time_binned['time_bin'] = (df_time_binned['Czas'] / time_step).astype(int)

    # grupowanie wartości na podstawie time_bin i obliczanie średniej, z pominięciem wartości NaN
    df_time_binned = df_time_binned.groupby('time_bin').mean()
    df_time_binned = df_time_binned.fillna(df_time_binned.mean())

    return df_time_binned


def regression_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['Czas'], df['CH1_filtered'], test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train.values.reshape(-1, 1), y_train)
    y_pred = model.predict(X_test.values.reshape(-1, 1))
    for i in range(len(y_pred)):
        print(f'Predicted: {y_pred[i]}, Actual: {y_test.values[i]}')


def regression_with_preprocessed_data(data):
    df_preprocessed = preprocess_data(data, decimation_step=10, filter_window_size=20, time_step=2)
    # print(df_preprocessed)
    regression_model(df_preprocessed)

    coefficients = np.polyfit(df_preprocessed["Czas"], df_preprocessed["CH1_filtered"], 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    regression_line = slope * df_preprocessed["Czas"] + intercept

    return {'df_preprocessed': df_preprocessed, 'regression_line': regression_line}


def generate_graph(dataL, dataR, dataW):
    graphL = regression_with_preprocessed_data(dataL)
    graphR = regression_with_preprocessed_data(dataR)
    graphW = regression_with_preprocessed_data(dataW)

    df_preprocessedL = graphL['df_preprocessed']
    regression_lineL = graphL['regression_line']
    df_preprocessedR = graphR['df_preprocessed']
    regression_lineR = graphR['regression_line']
    df_preprocessedW = graphW['df_preprocessed']
    regression_lineW = graphW['regression_line']

    plt.figure("Graph")
    plt.scatter(df_preprocessedL["Czas"], df_preprocessedL["CH1_filtered"], label='L', color='purple')
    plt.plot(df_preprocessedL["Czas"], regression_lineL, color='purple', label='L reg.')
    plt.scatter(df_preprocessedR["Czas"], df_preprocessedR["CH1_filtered"], label='R', color='blue')
    plt.plot(df_preprocessedR["Czas"], regression_lineR, color='blue', label='R reg.')
    plt.scatter(df_preprocessedW["Czas"], df_preprocessedW["CH1_filtered"], label='W', color='green')
    plt.plot(df_preprocessedW["Czas"], regression_lineW, color='green', label='W reg.')

    plt.title("Graph showing the dependence of Time on CH1")
    plt.xlabel("Time")
    plt.ylabel("CH1")
    plt.legend()
    plt.show()
# old code -------------------------------------------------------------------------------------------------------------
