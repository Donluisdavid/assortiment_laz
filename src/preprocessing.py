import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

class DataPreprocessor:

    def prepare_data(self, 
                     df, 
                     ):
        """
        Prépare les données : Features temporelles, saisonalité, et Lags.
        """
        df = df.copy()
        
        # 1. Gestion des dates
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        #2. Seasonalité
        df['is_winter']=np.where(np.isin(df['month'], [12,1,2]),1,0)
        df['is_springtime']=np.where(np.isin(df['month'], [3,4,5]),1,0)
        df['is_summer']=np.where(np.isin(df['month'], [6,7,8]),1,0)
        df['is_automn']=np.where(np.isin(df['month'], [9,10,11]),1,0)

        # 3. Les Lags 
        # On trie par date, magasin, et item, pour que le shift fonctionne
        df = df.sort_values(['agency', 'sku', 'date'])
        
        # Volume du mois précédent
        df['volume_lag_1'] = df.groupby(['agency', 'sku'])['volume'].shift(1)
        df.loc[df['volume_lag_1'].isna(),'volume_lag_1']=df.loc[df['volume_lag_1'].isna(),'volume']
        
        # Volume de l'année précédente (saisonnalité)
        df['volume_lag_12'] = df.groupby(['agency', 'sku'])['volume'].shift(12)
        df.loc[df['volume_lag_12'].isna(),'volume_lag_12']=df.loc[df['volume_lag_12'].isna(),'volume']

        # On définit les colonnes que le modèle va utiliser
        features_cols = [
                 'month', 'is_winter', 'is_springtime', 'is_summer', 'is_automn',  'year',
                 'volume_lag_1', 'volume_lag_12', 
            ]
  
        return df, features_cols