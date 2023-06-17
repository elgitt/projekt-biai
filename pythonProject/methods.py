import csv
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew
from scipy import signal


def extract_excel_sheet(sheet_name, output_file):
    temp_file = 'measurements/temp.csv'
    excel_file = pd.read_excel('measurements/Synteza.xlsx', sheet_name=sheet_name)
    excel_file.to_csv(temp_file, index=False)
    df = pd.read_csv(temp_file)
    df = df[20:]
    df = df.iloc[:, :3]
    df.columns = ['TIME', 'CH1', 'CH2']
    df.to_csv('measurements/' + output_file + '.csv', index=False)
    os.remove(temp_file)


def filter_data(input_file, output_file, cutoff_frequency=100, sampling_frequency=1000, order=4):
    data = pd.read_csv(input_file)
    time_values = data['TIME']
    current_values = data['CH1']
    voltage_values = data['CH2']

    normalized_cutoff = cutoff_frequency / (0.5 * sampling_frequency)

    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)[0], \
        signal.butter(order, normalized_cutoff, btype='low', analog=False)[1]

    filtered_currents = signal.filtfilt(b, a, current_values)
    filtered_voltages = signal.filtfilt(b, a, voltage_values)

    filtered_data = pd.DataFrame({'TIME': time_values, 'CH1': filtered_currents, 'CH2': filtered_voltages})
    filtered_data.to_csv(output_file, index=False)


def extract_period_data(file_path, num_periods):
    time_values = []
    current_values = []
    voltage_values = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            time = float(row['TIME'])
            current = float(row['CH1'])
            voltage = float(row['CH2'])
            time_values.append(time)
            current_values.append(current)
            voltage_values.append(voltage)

    sign_changes = np.diff(np.sign(current_values))

    period_start = None
    period_end = None
    periods = []

    for i in range(len(sign_changes)):
        if sign_changes[i] != 0:
            if period_start is None:
                period_start = i
            elif period_end is None:
                period_end = i
                periods.append((period_start, period_end))
                period_start = i
                period_end = None

    if period_start is not None and period_end is None:
        periods.append((period_start, len(sign_changes)))

    period_data = []

    for period in periods[:num_periods]:
        start_index, end_index = period
        period_currents = current_values[start_index:end_index]
        period_times = time_values[start_index:end_index]
        period_voltages = voltage_values[start_index:end_index]

        period_mean = np.mean(period_currents)
        period_std = np.std(period_currents)
        period_duration = period_times[-1] - period_times[0]
        period_median = np.median(period_currents)
        period_max = np.max(period_currents)
        period_min = np.min(period_currents)
        period_dynamics = period_max - period_min
        period_top_values = np.sort(period_currents)[-3:]  # Trzy największe wartości
        period_top1 = period_top_values[-1] if len(period_top_values) >= 1 else np.nan  # Największa wartość
        period_top2 = period_top_values[-2] if len(period_top_values) >= 2 else np.nan  # Druga największa wartość
        period_top3 = period_top_values[-3] if len(period_top_values) >= 3 else np.nan  # Trzecia największa wartość
        period_rms = np.sqrt(np.mean(np.square(period_currents)))
        period_peak = np.max(np.abs(period_currents))
        period_avg_power = np.mean(np.square(period_currents))
        period_skewness = skew(np.array(period_currents))
        period_kurtosis = stats.kurtosis(period_currents)
        period_active_power = np.mean(period_currents * np.array(period_voltages))
        period_reactive_power = np.mean(period_currents * np.imag(signal.hilbert(period_voltages)))

        period_data.append([
            period_mean,  # Średnia
            period_std,  # Odchylenie standardowe
            period_duration,  # Okres czasu
            period_median,  # Mediana
            period_max,  # Wartość maksymalna
            period_min,  # Wartość minimalna
            period_dynamics,  # Dynamika zmian
            period_top1,  # Największa wartość
            period_top2,  # Druga największa wartość
            period_top3,  # Trzecia największa wartość
            period_rms,  # RMS
            period_peak,  # Wartość szczytowa
            period_avg_power,  # Średnia moc
            period_skewness,  # Skośność
            period_kurtosis,  # Kurtoza
            period_active_power,  # Moc czynna
            period_reactive_power  # Moc bierna
        ])

    period_data = np.array(period_data)

    # Uzupełnianie wartości NaN medianą
    for i in range(period_data.shape[1]):
        column = period_data[:, i]
        column_median = np.nanmedian(column)
        column[np.isnan(column)] = column_median

    return period_data



def get_vectors():
    data = ['measurements/Tek0000.csv', 'measurements/Tek0001.csv',
            'measurements/Tek0002.csv', 'measurements/Tek0003.csv',
            'measurements/Tek0004.csv', 'measurements/Tek0005.csv']

    datafrom0 = []
    datafrom1 = []
    datafrom2 = []
    datafrom3 = []
    datafrom4 = []
    datafrom5 = []

    period_data0 = extract_period_data(data[0], 100)
    datafrom0.append(period_data0)
    period_data1 = extract_period_data(data[1], 100)
    datafrom1.append(period_data1)
    period_data2 = extract_period_data(data[2], 100)
    datafrom2.append(period_data2)
    period_data3 = extract_period_data(data[3], 100)
    datafrom3.append(period_data3)
    period_data4 = extract_period_data(data[4], 100)
    datafrom4.append(period_data4)
    period_data5 = extract_period_data(data[5], 100)
    datafrom5.append(period_data5)

    # for data in dataFrom0:
    #     for i, period in enumerate(data):
    #         print(f"Period {i+1}:")
    #         for key, value in period.items():
    #             print(f"{key}: {value}")
    #         print("--------")

    return {"vec0": period_data0, "vec1": period_data1, "vec2": period_data2,
            "vec3": period_data3, "vec4": period_data4, "vec5": period_data5}
