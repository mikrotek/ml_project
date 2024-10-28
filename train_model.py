import joblib
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

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
        df = pd.DataFrame(data)
        return df
    else:
        print("Errore durante il recupero dei dati:", response.status_code)
        return None

# Funzione per preprocessare i dati
def preprocess_data(df):
    df = df[['market_cap_rank', 'current_price', 'total_volume', 'price_change_percentage_24h',
             'market_cap', 'circulating_supply', 'total_supply']].copy()
    df.dropna(inplace=True)
    return df

# Carica e preprocessa i dati
df = get_coin_data()
if df is not None and not df.empty:
    df = preprocess_data(df)
    X = df[['market_cap_rank', 'total_volume', 'price_change_percentage_24h', 'market_cap',
            'circulating_supply', 'total_supply']]
    y = df['current_price']

    # Definisci il modello e i parametri
    rf_model = RandomForestRegressor()
    rf_param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2]
    }

    # Esegui GridSearchCV per trovare i migliori parametri
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid_search.fit(X, y)

    # Estrai e stampa i migliori parametri e MSE
    best_rf_model = rf_grid_search.best_estimator_
    rf_mse = mean_squared_error(y, best_rf_model.predict(X))
    print(f"Random Forest Migliori Parametri: {rf_grid_search.best_params_}")
    print(f"Random Forest MSE: {rf_mse}")

    # Salva il modello addestrato
    joblib.dump(best_rf_model, "random_forest_model.pkl")
    print("Modello salvato come 'random_forest_model.pkl'")
else:
    print("Nessun dato disponibile per addestrare il modello.")
