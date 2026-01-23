import sys
from src.preprocessing import DataPreprocessor
from src.training import ModelTrainer
from src.inference import run_recursive_inference
import pandas as pd
import os
import json

DATA_TRAIN_PATH = "data/train_set.csv"
DATA_VAL_PATH = "data/val_set.csv"
MODEL_PATH = "models/model.pkl"
INFERENCE_PATH = "data/inference_set.csv"
PREDICTIONS_PATH = "data/final_predictions.csv"

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

    # 2. Load training features 
    with open("models/features_cols.json", "r") as f:
        features_cols = json.load(f)

    # 3. Preprocessing
    print("On charge le preprocessor pour l'artifact")
    prep = DataPreprocessor()

    # 4. Entraînement
    trainer = ModelTrainer()
    trainer.train(df_train, df_val, features_cols, target='volume')

    # 5. Sauvegarde
    trainer.save_model(prep, features_cols, MODEL_PATH)


def run_inference():
    print("INFERENCE RÉCURSIVE")
    
    # Dernier état connu des couples agency/sku
    df_inference = pd.read_csv(INFERENCE_PATH)
        
    # Appel de la fonction récursive du forecasting (à 4 mois)
    run_recursive_inference(
        df_inference=df_inference,
        output_csv=PREDICTIONS_PATH,
        model_path=MODEL_PATH
    )

    print(f"Prévisions M+1 à M+4 générées dans {PREDICTIONS_PATH}")

      
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "training":
        run_training()
    elif len(sys.argv) > 1 and sys.argv[1] == "inference":
        run_inference()
    else:
        run_preprocessing()