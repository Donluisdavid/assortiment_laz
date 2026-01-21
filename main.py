import sys
from src.preprocessing import DataPreprocessor
import pandas as pd

def run_train():
    print("--- DÉMARRAGE DE L'ENTRAÎNEMENT ---")
    df = pd.read_csv('data/ds_assortiment_dataset.csv')
    prep = DataPreprocessor()    
    prep.prepare_data(df)
    print("Succès : Data chargée, transformée, et loadée")
    print(df.head())

if __name__ == "__main__":
        run_train()