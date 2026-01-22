import sys
from src.preprocessing import DataPreprocessor
import pandas as pd
import os
import json

DATA_TRAIN_PATH = "data/train_set.csv"
DATA_VAL_PATH = "data/val_set.csv"
MODEL_PATH = "models/carrefour_model.pkl"

def run_preprocessing():
    print("--- DÉMARRAGE DU PREPROCESSING ---")
    df= pd.read_csv('data/ds_assortiment_dataset.csv')
    prep = DataPreprocessor()    
    df, features_cols = prep.prepare_data(df)
    print("Succès : Data chargée, transformée, et loadée")

    print("--- SPLIT DES JEUX DE DONNEES TRAIN, VALIDATION, INFERENCE ---")
    train_df = df[df['date'] <= '2016-11-01']
    train_df.to_csv("data/train_set.csv", index=False)

    val_df = df[(df['date'] >= '2016-12-01') & (df['date'] <= '2017-11-01')]
    val_df.to_csv("data/val_set.csv", index=False)

    inference_df = df[(df['date'] >= '2017-12-01')&(df['date'] <= '2017-12-31')]
    inference_df.to_csv("data/inference_set.csv", index=False)

    # Sauvegarde dans un fichier JSON
    with open("models/features_cols.json", "w") as f:json.dump(features_cols, f)

    print("Succès : sauvegardé")

def run_training():
    print("--- ENTRAÎNEMENT ---")
    
    # 1. Chargement des données
    df_train = pd.read_csv(DATA_TRAIN_PATH)
    df_val = pd.read_csv(DATA_VAL_PATH)
    # 4. Entraînement
    trainer = ModelTrainer()
    features = prep.features_cols
    trainer.train(train_df, val_df, features)

    # 5. Sauvegarde
    trainer.save_model(prep, MODEL_PATH)

    # 6. Export pour audit
    train_df.to_csv("data/train_set.csv", index=False)
    val_df.to_csv("data/validation_set.csv", index=False)
    print("✅ Pipeline terminé avec succès.")  
      
if __name__ == "__main__":
        run_preprocessing()