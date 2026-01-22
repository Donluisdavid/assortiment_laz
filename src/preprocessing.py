import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

class DataPreprocessor:
    def __init__(self):
        # On crée un dictionnaire pour stocker un encodeur par colonne
        # self.encoders = {}
        self.features_cols = []
        # self.cat_features = ['agency', 'sku']

    def prepare_data(self, 
                     df, 
                    #  is_training=True
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

        # # Pct du changement du prix actuel en fonction du prix actuel du mois précedent
        # df['price_actual_lag_1'] = df.groupby(['agency', 'sku'])['price_actual'].shift(1)
        # df.loc[df['price_actual_lag_1'].isna(),'price_actual_lag_1']=df.loc[df['price_actual_lag_1'].isna(),'price_actual']
        # df['pct_price_actual_lag_1'] = np.where(df['price_actual_lag_1']>0,(df['price_actual']-df['price_actual_lag_1'])/df['price_actual_lag_1'],0)

        # # Pct du changement du prix regular en fonction du prix regular de l'année précédente (inflation à 1 an)
        # df['price_regular_lag_12'] = df.groupby(['agency', 'sku'])['price_regular'].shift(1)
        # df.loc[df['price_regular_lag_12'].isna(),'price_regular_lag_12']=df.loc[df['price_regular_lag_12'].isna(),'price_regular']
        # df['pct_price_regular_lag_12'] = np.where(df['price_regular_lag_12']>0,(df['price_regular']-df['price_regular_lag_12'])/df['price_regular_lag_12'],0)

        # On définit les colonnes que le modèle va utiliser
        self.features_cols = [
                #'agency', 'sku', 
                 'month', 'is_winter', 'is_springtime', 'is_summer', 'is_automn',  'year',
                #  'easter_day','good_friday','new_year','christmas','labor_day','independence_day','revolution_day_memorial', 
                #  'regional_games','fifa_u_17_world_cup','football_gold_cup','beer_capital','music_fest',
                #  'avg_max_temp', 
                 'volume_lag_1', 'volume_lag_12', 
                #  'pct_price_regular_lag_12', 'pct_price_actual_lag_1',
                #  'discount_in_percent'
            ]
  
        return df
    
    # def data_train_val_inference(self, df): 
    #     """Prépare les données pour l'entraînement, la validation, et l'inférence"""
    #     df = df.copy()
    #     # On sépare les features et la target
    #     X = df[self.features_cols]
    #     y = df['volume']
    #     return X, y


    # def save_preprocessor(self, path):
    #     """Sauvegarde les encodeurs pour l'inférence"""
    #     joblib.dump(self, path)