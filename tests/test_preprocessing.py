import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor

def test_prepare_data_logic():
    """
    Vérifie la validité de la transformation des données :
    - Extraction du mois et de l'année
    - Calcul des saisons (Hiver/Été)
    - Création des colonnes de Lags
    """
    # 1. ARRANGE : On crée un petit jeu de données de test
    data = {
        'date': ['2023-01-01', '2023-02-01', '2023-07-01'],
        'agency': ['A', 'A', 'A'],
        'sku': ['S1', 'S1', 'S1'],
        'volume': [100, 200, 300]
    }
    df_test = pd.DataFrame(data)
    prep = DataPreprocessor()
    
    # 2. ACT : On applique le preprocessing
    result = prep.prepare_data(df_test)
    
    # 3. ASSERT : On vérifie les résultats
    # Vérification des colonnes temporelles
    assert result['month'].iloc[0] == 1
    assert result['year'].iloc[0] == 2023
    
    # Vérification des saisons (Janvier est en hiver, Juillet en été)
    assert result['is_winter'].iloc[0] == 1
    assert result['is_summer'].iloc[2] == 1
    assert result['is_winter'].iloc[2] == 0
    
    # Vérification du Lag 1 (Le lag de févier doit être le volume de janvier)
    # Dans ton code, le premier élément prend sa propre valeur car lag_1 est NaN
    assert result['volume_lag_1'].iloc[1] == 100 
    
    # Vérification de la liste des features
    assert 'is_winter' in prep.features_cols
    assert 'volume_lag_12' in prep.features_cols

def test_feature_columns_consistency():
    """Vérifie que la liste des features n'est pas vide après processing."""
    df_test = pd.DataFrame({
        'date': ['2023-01-01'], 'agency': ['A'], 'sku': ['S1'], 'volume': [10]
    })
    prep = DataPreprocessor()
    _ = prep.prepare_data(df_test)
    
    assert len(prep.features_cols) > 0