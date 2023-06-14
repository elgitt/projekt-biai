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
        period_top_three = np.sort(period_currents)[-3:]
        period_rms = np.sqrt(np.mean(np.square(period_currents)))
        period_peak = np.max(np.abs(period_currents))
        period_avg_power = np.mean(np.square(period_currents))
        period_skewness = skew(np.array(period_currents))
        period_kurtosis = stats.kurtosis(period_currents)
        period_fft = np.fft.fft(period_currents)
        period_active_power = np.mean(period_currents * np.array(period_voltages))
        period_reactive_power = np.mean(period_currents * np.imag(signal.hilbert(period_voltages)))

        if period_duration > 0:
            period_frequencies = np.fft.fftfreq(len(period_currents), period_duration / len(period_currents))
        else:
            period_frequencies = np.zeros(len(period_currents))

        period_amplitudes = np.abs(period_fft)
        period_harmonics_rms = np.sqrt(np.sum(np.square(period_amplitudes[1:])))
        period_dom_freq = period_frequencies[np.argmax(period_amplitudes)] if period_duration > 0 else 0.0

        period_data.append([
            period_mean,  # Średnia
            period_std,  # Odchylenie standardowe
            period_duration,  # Okres czasu
            period_median,  # Mediana
            period_max,  # Wartość maksymalna
            period_min,  # Wartość minimalna
            period_dynamics,  # Dynamika zmian
            period_top_three,  # Trzy największe wartości
            period_rms,  # Wartość skuteczna
            period_peak,  # Wartość szczytowa
            period_dom_freq,  # Częstotliwość dominująca
            period_avg_power,  # Średnia moc
            period_skewness,  # Skośność
            period_kurtosis,  # Kurtoza
            period_harmonics_rms,  # Wartość skuteczna harmoniczna
            period_active_power,  # Moc czynna
            period_reactive_power,  # Moc bierna
            period_frequencies,  # Częstotliwości
            period_amplitudes  # Amplitudy
        ])

    return period_data


def get_vectors():
    data = ['filtered_data/Tek0000.csv', 'filtered_data/Tek0001.csv',
            'filtered_data/Tek0002.csv', 'filtered_data/Tek0003.csv',
            'measurements/Tek0004.csv', 'filtered_data/Tek0005.csv']

    datafrom0 = []
    datafrom1 = []
    datafrom2 = []
    datafrom3 = []
    datafrom4 = []
    datafrom5 = []

    period_data0 = extract_period_data(data[0], 5)
    datafrom0.append(period_data0)
    period_data1 = extract_period_data(data[1], 5)
    datafrom1.append(period_data1)
    period_data2 = extract_period_data(data[2], 5)
    datafrom2.append(period_data2)
    period_data3 = extract_period_data(data[3], 5)
    datafrom3.append(period_data3)
    period_data4 = extract_period_data(data[4], 5)
    datafrom4.append(period_data4)
    period_data5 = extract_period_data(data[5], 5)
    datafrom5.append(period_data5)

    # for data in dataFrom0:
    #     for i, period in enumerate(data):
    #         print(f"Period {i+1}:")
    #         for key, value in period.items():
    #             print(f"{key}: {value}")
    #         print("--------")

    return {"vec0": period_data0, "vec1": period_data1, "vec2": period_data2,
            "vec3": period_data3, "vec4": period_data4, "vec5": period_data5}
