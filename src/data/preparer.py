import numpy as np
import pandas as pd
from scripts.constants import PREDICTION_COLS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreparer:
    def __init__(self):
        self.prediction_cols = PREDICTION_COLS
        pass

    def prepare_for_linear_regression(self, df: pd.DataFrame, target_col: str):
        try:
            subset_df = df[self.prediction_cols].copy()
            subset_df = subset_df[subset_df[target_col] > 0]

            # Inspect Target Column Skewness
            claim_skewness_value = subset_df[target_col].skew()  # 3.8463379940832976
            print(f"Skewness of {target_col}: {claim_skewness_value}")

            # Log transform target col (Total Claim) for now since skew is >1
            target_col_log = target_col + "_log"
            subset_df[target_col_log] = np.log(subset_df[target_col])
            print(f"Transformed {target_col} to Log: {target_col_log}")

            # Separate Categorical and Numeric columns
            feature_cols = [
                c for c in subset_df.columns if c not in [target_col, target_col_log]
            ]
            numeric_cols = (
                subset_df[feature_cols]
                .select_dtypes(include=["int64", "float64"])
                .columns.tolist()
            )
            categorical_cols = (
                subset_df[feature_cols]
                .select_dtypes(include=["object", "category"])
                .columns.tolist()
            )

            preprocessor_lr = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    (
                        "cat",
                        OneHotEncoder(drop="first", handle_unknown="ignore"),
                        categorical_cols,
                    ),
                ]
            )

            y = subset_df[target_col_log]
            X = subset_df.drop(columns=[target_col, target_col_log])

            return X, y, preprocessor_lr, subset_df
        except Exception as e:
            print(
                f"Something went wrong during data preparation for linear regression: {e}"
            )
            raise e
