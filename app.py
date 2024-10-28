import matplotlib
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

matplotlib.use('Agg')

# Configura il logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Permette il frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    logging.info("API online request received.")
    return {"message": "API online!"}

class PredictionRequest(BaseModel):
    market_cap_rank: float
    total_volume: float
    price_change_percentage_24h: float
    market_cap: float
    circulating_supply: float
    total_supply: float

# Funzione per salvare la previsione in un file CSV
def save_prediction(data, predicted_price):
    with open("predictions.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now(),
            data['market_cap_rank'],
            data['total_volume'],
            data['price_change_percentage_24h'],
            data['market_cap'],
            data['circulating_supply'],
            data['total_supply'],
            predicted_price
        ])
    logging.info(f"Prediction saved with data: {data} and predicted price: {predicted_price}")

@app.post("/predict")
def predict(request: PredictionRequest):
    model = joblib.load("random_forest_model.pkl")

    input_data = [[
        request.market_cap_rank,
        request.total_volume,
        request.price_change_percentage_24h,
        request.market_cap,
        request.circulating_supply,
        request.total_supply
    ]]
    prediction = model.predict(input_data)[0]
    save_prediction(request.dict(), prediction)

    logging.info(f"Prediction request: {request.dict()}, Predicted price: {prediction}")
    return {"predicted_price": prediction}

# Funzione per ottenere dati da CoinGecko
def get_coin_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 10,
        "page": 1,
        "sparkline": False
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        logging.error(f"Errore durante il recupero dei dati da CoinGecko: {response.status_code}")
        return None

# Funzione per processare i dati e selezionare le funzionalità utili
def preprocess_data(df):
    df = df[['market_cap_rank', 'current_price', 'total_volume', 'price_change_percentage_24h',
             'market_cap', 'circulating_supply', 'total_supply']].copy()
    df.dropna(inplace=True)
    return df

# Funzione per normalizzare i dati
def normalize_data(df):
    scaler = StandardScaler()
    features = ['market_cap_rank', 'total_volume', 'price_change_percentage_24h', 'market_cap',
                'circulating_supply', 'total_supply']
    df[features] = scaler.fit_transform(df[features])
    return df

# Funzione per creare e valutare il modello di Machine Learning
def train_models(df):
    X = df[['market_cap_rank', 'total_volume', 'price_change_percentage_24h', 'market_cap',
            'circulating_supply', 'total_supply']]
    y = df['current_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor()
    rf_param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2]
    }
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    rf_y_pred = best_rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_y_pred)
    logging.info(f"Random Forest Migliori Parametri: {rf_grid_search.best_params_}")
    logging.info(f"Random Forest MSE: {rf_mse}")

    joblib.dump(best_rf_model, "random_forest_model.pkl")
    logging.info("Modello Random Forest salvato come 'random_forest_model.pkl'")

    gb_model = GradientBoostingRegressor()
    gb_param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    gb_grid_search = GridSearchCV(gb_model, gb_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    gb_grid_search.fit(X_train, y_train)
    best_gb_model = gb_grid_search.best_estimator_
    gb_y_pred = best_gb_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_y_pred)
    logging.info(f"Gradient Boosting Migliori Parametri: {gb_grid_search.best_params_}")
    logging.info(f"Gradient Boosting MSE: {gb_mse}")

    return X_test, y_test, rf_y_pred, gb_y_pred

# Funzione per visualizzare i risultati
def plot_results(X_test, y_test, rf_y_pred, gb_y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(X_test['market_cap_rank'], y_test, color='blue', label='True Values')
    plt.plot(X_test['market_cap_rank'], rf_y_pred, color='red', label='Random Forest Predictions')
    plt.plot(X_test['market_cap_rank'], gb_y_pred, color='green', label='Gradient Boosting Predictions')
    plt.title("Market Cap Rank vs. Current Price")
    plt.xlabel("Market Cap Rank")
    plt.ylabel("Current Price")
    plt.legend()
    plt.savefig('coingecko_predictions.png')
    logging.info("Grafico dei risultati salvato come 'coingecko_predictions.png'")

def main():
    logging.info("Starting data collection and model training.")
    df = get_coin_data()
    if df is not None and not df.empty:
        df = preprocess_data(df)
        df = normalize_data(df)
        X_test, y_test, rf_y_pred, gb_y_pred = train_models(df)
        plot_results(X_test, y_test, rf_y_pred, gb_y_pred)
        logging.info("Data processing and model training completed successfully.")
    else:
        logging.warning("Nessun dato disponibile per il grafico.")

if __name__ == "__main__":
    main()
