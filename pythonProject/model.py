import pandas as pd
import methods as m
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def model_MLP(dataL, dataR, dataW):
    dfa = pd.concat([dataW, dataL, dataR])
    data = m.regression_with_preprocessed_data(dfa)
    # Usunięcie wierszy zawierających brakujące wartości
    # df.dropna(inplace=True)
    # Podział danych na cechy (X) i etykiety (y)
    df = data['df_preprocessed']
    X = df['Czas'].values.reshape(-1, 1)
    y = df['CH1_filtered'].values

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standaryzacja danych
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inicjalizacja modelu MLP
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42)

    # Trenowanie modelu
    model.fit(X_train_scaled, y_train)

    # Predykcja na danych testowych
    y_pred = model.predict(X_test_scaled)
    formatted_y_pred = ["{:.3f}".format(pred) for pred in y_pred]
    print("Predicted CH1: ", formatted_y_pred)

    # Obliczenie średniego błędu kwadratowego
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error: ", mse)
