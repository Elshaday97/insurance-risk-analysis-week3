import numpy as np
import pandas as pd
from scripts.constants import PREDICTION_COLS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreparer:
    def __init__(self):
        self.prediction_cols = PREDICTION_COLS
        pass

    def _get_subset_df(self, df: pd.DataFrame, target_col: str):
        subset_df = df[self.prediction_cols].copy()
        subset_df = subset_df[subset_df[target_col] > 0]

        return subset_df

    def prepare_for_linear_regression(self, df: pd.DataFrame, target_col: str):
        try:
            subset_df = self._get_subset_df(df=df, target_col=target_col)
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

    def prepare_for_tree_model(self, df: pd.DataFrame, target_col: str):
        subset_df = self._get_subset_df(df=df, target_col=target_col)
        y = subset_df[target_col]

        feature_cols = [c for c in subset_df.columns if c not in [target_col]]
        categorical_cols = (
            subset_df[feature_cols]
            .select_dtypes(include=["object", "category"])
            .columns.tolist()
        )

        preprocessor = ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_cols,
                )
            ],
            remainder="passthrough",
        )

        X = subset_df[feature_cols]
        return X, y, preprocessor

    def prepare_for_classification(self, df: pd.DataFrame, target_col: str):
        """
        Prepares data for binary classification (Claim vs No Claim).
        """
        subset_df = df[self.prediction_cols].copy()

        target_bin = "Claimed"
        subset_df[target_bin] = (subset_df[target_col] > 0).astype(int)

        # 3. Separate Features and Target
        # Drop the original continuous target and the new binary target from features
        X = subset_df.drop(columns=[target_col, target_bin])
        y = subset_df[target_bin]

        # 4. Define Preprocessor (Identify Numeric/Categorical)
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_cols,
                ),
            ],
            remainder="passthrough",
        )

        return X, y, preprocessor
