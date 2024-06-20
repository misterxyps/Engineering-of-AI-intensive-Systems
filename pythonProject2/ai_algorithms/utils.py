import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


class Dataprepper:
    def __init__(self, df: pd.DataFrame, 
                 num_feature_indices: list, 
                 cat_feature_indices: list,
                 ):
        
        self.N, self.D = df.shape
        self.column_names = list(df.columns)
        self.num_features = num_feature_indices
        self.cat_features = cat_feature_indices
        self.missing_df = df.copy()
        self.encoding_object = dict()  # needed for KNN inverse transform on cat features in f1
        self.missing_indices = np.argwhere(df.isna().to_numpy())
        
    def fill_missing(self, df):
        for col_idx, col_name in enumerate(self.column_names):
            if col_idx in self.cat_features:
                mode_val = df[col_name].mode()[0]
                df.fillna({col_name: mode_val}, inplace=True)
            else:
                mean_val = df[col_name].mean()
                df.fillna({col_name: mean_val}, inplace=True)
        return df

    def encode_data(self, dataframe, for_knn=False):
        if dataframe.isna().any().any():
            filled = self.fill_missing(dataframe)
        filled = dataframe.to_numpy()
        encoded_lst = list()

        for col_idx in range(self.D):
            col_arr = filled[:, col_idx].reshape(-1, 1)
            fitted_enc = None

            if col_idx in self.cat_features:
                if for_knn:
                    fitted_enc = OrdinalEncoder().fit(col_arr)
                    enc_arr = fitted_enc.transform(col_arr)
                else:
                    enc_arr = col_arr  # missforest algorithm includes own reverse mapping
            else:
                enc_arr = StandardScaler().fit_transform(col_arr)

            self.encoding_object[col_idx] = fitted_enc
            encoded_lst.append(enc_arr)

        enc_data = np.block(encoded_lst)
        df_enc = pd.DataFrame(enc_data, columns=self.column_names)

        return df_enc

    def get_enc_df_for_discr_model(self, for_knn):
        self.encoded_df = self.encode_data(self.missing_df, for_knn)
        df_to_impute = self.encoded_df.copy()
        for miss in self.missing_indices:
            df_to_impute.iloc[miss[0], miss[1]] = float("nan")

        return df_to_impute
    

class Evaluator:
    def __init__(self, imputed_data, df_mean_mode, mode, target_index):
        self.mode = mode.lower()
        self.target_idx = target_index
        self.baseline_model = RandomForestClassifier() if self.mode == "classification" else RandomForestRegressor()
        self.imputed_data = imputed_data if isinstance(imputed_data, np.ndarray) else imputed_data.to_numpy()
        self.df_mean_mode = df_mean_mode.to_numpy()
        
    def calculate_baseline_score(self):
        scoring = "accuracy" if self.mode == "classification" else "neg_root_mean_squared_error"
        assert self.target_idx == (self.df_mean_mode.shape[1] - 1)
        X, y = self.df_mean_mode[:, :self.target_idx], self.df_mean_mode[:, self.target_idx].reshape(-1, 1)
        scores = cross_val_score(self.baseline_model, X, y, cv=5, scoring=scoring)
        score_avg = np.mean(scores)
        return score_avg
    
    def calculate_downstream_score(self):
        scoring = "accuracy" if self.mode == "classification" else "neg_root_mean_squared_error"
        assert self.target_idx == (self.df_mean_mode.shape[1] - 1)
        X, y = self.imputed_data[:, :self.target_idx], self.imputed_data[:, self.target_idx].reshape(-1, 1)
        scores = cross_val_score(self.baseline_model, X, y, cv=5, scoring=scoring)
        score_avg = np.mean(scores)
        return score_avg
