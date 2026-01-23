import pandas as pd
import joblib
import os
import numpy as np

class Predictor:
    def __init__(self, model_path):
        self.artifacts = joblib.load(model_path)
        self.model = self.artifacts['model']
        self.prep = self.artifacts['preprocessor']
        self.features = self.artifacts['features']

    def predict_recursive_wide(self, current_state_df, horizon=4):
        """
        current_state_df: contient une seule ligne par agency/sku (le dernier mois réel).
        """
        print(f"Inférence récursive ({horizon} mois)")
        
        # Initialisation nos dfs avec les infos de base
        context_df = current_state_df[['date', 'agency', 'sku', 'volume']].copy()
        all_forecast = context_df[['date','agency','sku','volume']].copy()

        for m in range(1, horizon + 1):
            # Calcul de la date cible
            last_date = context_df['date'].max()
            target_date = last_date + pd.DateOffset(months=1)
            
            # Création du df pour le mois M (on duplique la structure agency/sku)
            next_month = current_state_df[['agency', 'sku']].copy()
            next_month['date'] = target_date
            
            # "photo" complète du mois qui correspond à l'horizon M+h
            full_df = context_df[['agency', 'sku', 'volume']].merge(next_month, on=['agency','sku'], how='inner')

            # Preprocessing (calcul des lags, is_winter, etc.)
            processed_df,_ = self.prep.prepare_data(full_df)

            # Prédiction
            X = processed_df[self.features]
            preds = self.model.predict(X)
            
            # Mise à jour des résultats
            pred_column = f'predict_m{m}'
            next_month['volume'] = np.maximum(0, preds)
            next_month[f'predict_m{m}'] = next_month['volume'] # Colonne spécifique
            
            # Prédictions du M+h ajoutées au df final
            all_forecast = all_forecast.merge(next_month[['agency', 'sku', f'predict_m{m}']], on=['agency', 'sku'], how='left')
            
            # Actualisation du contexte avec l'infos du m+h 
            context_df = next_month[['date', 'agency', 'sku', 'volume']].copy()

        # Renommage des colonnes pour plus de clarté
        all_forecast=all_forecast.rename(
            columns={'volume': 'last_volume', 'date': 'reference_date'}
        )

        return all_forecast

def run_recursive_inference(df_inference, output_csv, model_path="models/carrefour_model.pkl"):
    
    df_inference['date'] = pd.to_datetime(df_inference['date'])

    predictor = Predictor(model_path)

    # Obtention du forecast récursif
    df_forecast = predictor.predict_recursive_wide(df_inference, horizon=4)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_forecast.to_csv(output_csv, index=False)
    print(f"Prévisions exportées : {output_csv}")