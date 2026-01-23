import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelTrainer:
    def __init__(self):
        """
        Initialise le trainer avec des hyperparamètres optimisés pour le retail.
        """
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'num_leaves': 31,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.75,
            'verbose': -1,
            'seed': 42
        }
        self.model = None

    def train(self, df_train, df_val, features_col, target='volume'):
        """
        Entraînement du modèle.
        """
        # Préparation des matrices LightGBM
        X_train, y_train = df_train[features_col], df_train[target]
        X_val, y_val = df_val[features_col], df_val[target]

        train_data   = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Entraînement avec Early Stopping
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # Évaluation des performances        
        print("\n---PERFORMANCES SUR LE TRAIN ---")
        preds = self.model.predict(X_train)
        mae = mean_absolute_error(y_train, preds)
        rmse = np.sqrt(mean_squared_error(y_train, preds)) 

        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")

        print("PERFORMANCES SUR VALIDATION")
        preds = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print("--------------------------------------\n")

    def save_model(self, prep, features_cols, path):
        """
        Sauvegarde l'artefact complet (Modèle + Preprocessor + Featuress).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        artifacts = {
            'model': self.model,
            'preprocessor': prep,
            'features': features_cols
        }

        joblib.dump(artifacts, path)
        print(f"Artefact sauvegardé avec succès dans : {path}")