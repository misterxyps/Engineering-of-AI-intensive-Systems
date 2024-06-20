import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

class Dataprepper:
    def __init__(self, original):
        self.original_df = original
        self.N, self.D = original.shape
        self.column_names = list(original.columns)
        self.num_features = list(range(0, self.D - 1))
        self.cat_features = [self.D - 1]
        self.target_col = [self.D - 1]
        self.missing_df = None
        self.missing_indices = None
        self.encoding_object = dict()  # needed for KNN inverse transform on cat features in f1

    def fill_missing(self, df):
        for col_idx, col_name in enumerate(self.column_names):
            if col_idx in self.cat_features:
                mode_val = df[col_name].mode()[0]
                df[col_name].fillna(mode_val, inplace=True)
            else:
                mean_val = df[col_name].mean()
                df[col_name].fillna(mean_val, inplace=True)
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

    def ordinal_encoding(self, orig_arr, pred_arr):
        fitted_label_enc = OrdinalEncoder().fit(orig_arr.reshape(-1, 1))
        orig_enc = fitted_label_enc.transform(pred_arr.reshape(-1, 1))
        pred_enc = fitted_label_enc.transform(pred_arr.reshape(-1, 1))
        return orig_enc, pred_enc

    def get_enc_df_for_discr_model(self, for_knn):
        missing_df = self.encode_data(self.missing_df, for_knn)

        for miss in self.missing_indices:
            missing_df.iloc[miss[0], miss[1]] = float("nan")

        return missing_df

    def instantiate_df_miss(self, df_missing):
        self.missing_df = df_missing
        self.missing_indices = np.argwhere(df_missing.isna().to_numpy())

    def check_missing_idcs_in_numerical(self):
        missing_col_idcs = [index[1] for index in self.missing_indices]
        return bool(set(missing_col_idcs) & set(self.num_features))

    def check_missing_idcs_in_categorical(self):
        missing_col_idcs = [index[1] for index in self.missing_indices]
        return bool(set(missing_col_idcs) & set(self.cat_features))


class Evaluator(Dataprepper):
    def __init__(self, original, train_size, mode):
        super().__init__(original)
        self.train_size = train_size
        self.mode = mode.lower()
        self.baseline_model = RandomForestClassifier() if self.mode == "classification" else RandomForestRegressor()
        self.test_y_orig = None

        self.enc_original = self.encode_data(original).to_numpy()

        self.target_idx = np.array(self.target_col)  # evaluation of multi-target possible

        self.test_num_features, self.test_cat_features, self.test_target = self.create_test_num_cat_features()
        self.train_X_orig, self.train_y_orig, self.test_X_orig, self.test_y_orig = self.create_train_test_data()
        self.baseline_score = self.get_baseline_score()

    def get_hypertuned_model(self, model):

        RANDOM_FOREST_CLASS_HYPERS = {'max_features': ['sqrt', 'log2'],
                      'n_estimators': [10, 100, 200],
                      'max_features': [1, 3, 5, 7],
                      'criterion': ['gini', 'entropy']}

        RANDOM_FOREST_REG_HYPERS = {
            'criterion': ['squared_error', 'friedman_mse'],
            'n_estimators': [10, 100, 200],
            'max_features': ['sqrt', 'log2'],
        }

        grid_space = RANDOM_FOREST_CLASS_HYPERS if self.mode == "classification" else RANDOM_FOREST_REG_HYPERS

        #skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
        scoring = "accuracy" if self.mode == "classification" else "neg_root_mean_squared_error"
        grid = GridSearchCV(model, param_grid=grid_space, scoring=scoring)

        model_grid = grid.fit(self.train_X_orig, self.train_y_orig)

        return model_grid.best_estimator_, model_grid

    def create_train_test_data(self):
        train_X_orig = self.enc_original[:self.train_size, :self.target_col[0]]
        train_y_orig = self.enc_original[:self.train_size, self.target_col[0]]

        test_X_orig = self.enc_original[self.train_size:, :self.target_col[0]]
        test_y_orig = self.enc_original[self.train_size:, self.target_col[0]]
        return train_X_orig, train_y_orig, test_X_orig, test_y_orig

    def create_test_num_cat_features(self):
        test_cat_features, test_num_features = None, None
        if self.num_features:
            test_num_features = self.enc_original[self.train_size:, np.array(self.num_features)]
        if self.cat_features:
            test_cat_features = self.enc_original[self.train_size:, np.array(self.cat_features)]
        test_target = self.enc_original[self.train_size:, self.target_idx]
        return test_num_features, test_cat_features, test_target

    def get_baseline_score(self):
        self.baseline_model, model_grid = self.get_hypertuned_model(self.baseline_model)
        print("baseline_model.best_params_", model_grid.best_params_)

        fitted_model = self.baseline_model.fit(self.train_X_orig, self.train_y_orig)
        y_pred = fitted_model.predict(self.test_X_orig)

        if self.mode == "classification":
            test_y_orig_enc, y_pred_enc = self.ordinal_encoding(self.test_y_orig, y_pred)
            roc_score = roc_auc_score(test_y_orig_enc, y_pred_enc)
            acc_score = accuracy_score(self.test_y_orig, y_pred)
            print("baseline roc, acc score on complete original test set:", roc_score, acc_score)
            return acc_score, roc_score
        else:
            score = self.get_rmse_score(y_pred, "regression")
            print("RMSE baseline score:", score)
            return score

    def calculate_downstream_impact_score(self, imputed_data):
        # only take features of imputation algorithm output
        test_X_pred = imputed_data[self.train_size:, :self.target_col[0]]

        y_pred = self.baseline_model.predict(test_X_pred)
        #         y_pred = list(map(lambda x: 1 if x == "B" else 0, y_pred))

        if self.mode == "classification":
            test_y_orig_enc, y_pred_enc = self.ordinal_encoding(self.test_y_orig, y_pred)
            roc_score = roc_auc_score(test_y_orig_enc, y_pred_enc)
            acc_score = accuracy_score(self.test_y_orig, y_pred)
            return acc_score, roc_score
        else:
            score2 = None
            score = self.get_rmse_score(y_pred, "regression-prediction")
            return score, score2


    def calculate_rmse_f1_score(self, pred_data, eval_mode: str):
        rmse, f1 = None, None

        if self.check_missing_idcs_in_numerical():
            rmse = self.get_rmse_score(pred_data, eval_mode)

        if self.check_missing_idcs_in_categorical():
            f1 = self.get_f1_score(pred_data)

        return rmse, f1

    def get_rmse_score(self, pred_data, eval_mode):

        if eval_mode == "imputation":
            pred_num_features = pred_data[self.train_size:, np.array(self.num_features)]
            # RMSE: since we have to use scikit-version- we have to add np.sqrt()
            rmse = np.sqrt(mean_squared_error(self.test_num_features, pred_num_features))
        else:
            # assert pred_data.shape == self.test_target.shape  # only necessary in multi-target task
            # pred_data here are only the predictions of regression task?
            rmse = np.sqrt(mean_squared_error(self.test_target, pred_data))

        return rmse

    def get_f1_score(self, pred_data):
        f1_result = None

        f1_macro = list()

        for cat_col_idx in self.cat_features:
            # fitted_enc = OrdinalEncoder().fit(self.train_y_orig.reshape(-1, 1))
            pred_col_arr = (pred_data[self.train_size:, cat_col_idx]).reshape(-1, 1)
            # test_enc, pred_col_arr = fitted_enc.transform(self.test_y_orig.reshape(-1, 1)), fitted_enc.transform(pred_col_arr)

            target_labels = np.unique(pred_col_arr)
            # BINARY or Weighted F1 Score
            if len(target_labels) == 2:
                f1 = f1_score(self.test_y_orig, pred_col_arr, average='binary', pos_label=target_labels[0])
            else:
                f1 = f1_score(self.test_cat_features, pred_col_arr, average='weighted')

            f1_macro.append(f1)

        f1_result = np.mean(f1_macro)

        return f1_result

    def inverse_transform_categorical(self, pred_arr):
        rounded_cat_features = np.round(pred_arr[:, np.array(self.cat_features)])
        pred_arr[:, np.array(self.cat_features)] = rounded_cat_features

        for col_idx in self.cat_features:
            fitted_enc = self.encoding_object[col_idx]
            pred_col_arr = pred_arr[:, col_idx].reshape(-1, 1)
            inv_col_arr = (fitted_enc.inverse_transform(pred_col_arr)).reshape(1, -1)
            pred_arr = pred_arr.astype('O')
            pred_arr[:, col_idx] = inv_col_arr

        return pred_arr