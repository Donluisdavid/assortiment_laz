import sys
from src.preprocessing import DataPreprocessor
import pandas as pd


def run_train():
    print("--- DÉMARRAGE DU PREPROCESSING ---")
    df = pd.read_csv('data/ds_assortiment_dataset.csv')
    prep = DataPreprocessor()    
    prep.prepare_data(df)
    print("Succès : Data chargée, transformée, et loadée")

    print("--- SPLIT DES JEUX DE DONNEES TRAIN, VALIDATION, INFERENCE ---")
    train_df = df[df['date'] <= '2016-11-01']
    train_df.to_csv("data/train_set.csv", index=False)

    val_df = df[(df['date'] >= '2016-12-01') & (df['date'] <= '2017-11-01')]
    val_df.to_csv("data/val_set.csv", index=False)

    inference_df = df[(df['date'] >= '2017-12-01')&(df['date'] <= '2017-12-31')]
    inference_df.to_csv("data/inference_set.csv", index=False)
    print("Succès : sauvegardé")

if __name__ == "__main__":
        run_train()