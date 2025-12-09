# src/modeling.py
import pandas as pd
import numpy as np
from scripts.constants import MODEL_TYPES
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class ClaimClassifier:

    def __init__(self, model_type=MODEL_TYPES.XGBOOST.value, random_state=42):
        self.random_state = random_state
        self.model = None
        self.pipeline = None
        self.model_type = model_type

    def _get_classifier(self):
        if self.model_type == MODEL_TYPES.XGBOOST.value:
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                objective="binary:logistic",
            )
        elif self.model_type == MODEL_TYPES.RANDOM_FOREST.value:
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == MODEL_TYPES.LINEAR_REGRESSION.value:
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train(self, X_train, y_train, preprocessor):
        """Initializes and trains the Classifier pipeline."""
        model = self._get_classifier()

        self.pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", model)]
        )

        print(f"Training {self.model_type} Classifier...")
        self.pipeline.fit(X_train, y_train)
        print("Training Complete.")

    def evaluate(self, X_test, y_test):
        """Predicts and prints evaluation metrics."""
        if self.pipeline is None:
            raise Exception("Model not trained yet. Call train() first.")

        y_pred = self.pipeline.predict(X_test)

        # Extract probabilities safely
        final = self.pipeline.named_steps["classifier"]

        if hasattr(final, "predict_proba"):
            y_proba = final.predict_proba(
                self.pipeline.named_steps["preprocessor"].transform(X_test)
            )[:, 1]

        elif hasattr(final, "decision_function"):
            y_proba = final.decision_function(
                self.pipeline.named_steps["preprocessor"].transform(X_test)
            )

        else:
            raise Exception(
                f"Model {self.model_type} does not support probability estimation."
            )

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }

        print("\nModel Performance:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return metrics

    def explain_model(self, X_test, sample_size=100):
        """Generates a SHAP summary plot."""
        if not self.pipeline:
            raise Exception("Model not trained yet.")

        # 1. Extract transformed features
        preprocessor = self.pipeline.named_steps["preprocessor"]
        model = self.pipeline.named_steps["classifier"]

        X_transformed = preprocessor.transform(X_test)

        try:
            cat_feature_names = preprocessor.named_transformers_[
                "cat"
            ].get_feature_names_out()
            num_feature_names = preprocessor.named_transformers_[
                "num"
            ].get_feature_names_out()
            feature_names = list(num_feature_names) + list(cat_feature_names)
        except AttributeError:
            feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]

        # 3. Calculate SHAP values (Use a sample for speed)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        if X_transformed.shape[0] > sample_size:
            idx = np.random.choice(X_transformed.shape[0], sample_size, replace=False)
            X_sample = X_transformed[idx]
        else:
            X_sample = X_transformed

        print("\nCalculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[1], X_sample, feature_names=feature_names, show=False
        )
        plt.title("SHAP Feature Importance (Claim Probability)")
        plt.tight_layout()
        plt.show()


class ClaimSeverityRegressor:

    def __init__(self, model_type=MODEL_TYPES.XGBOOST.value, random_state=42):
        """
        Args:
            model_type (str): 'xgboost', 'random_forest', or 'linear'
            random_state (int): Seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = None
        self.is_log_transformed = False  # specific for Linear Regression path

    def _get_regressor(self):
        if self.model_type == MODEL_TYPES.XGBOOST.value:
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                objective="reg:squarederror",
                n_jobs=-1,
                base_score=0.5,
            )
        elif self.model_type == MODEL_TYPES.RANDOM_FOREST.value:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == MODEL_TYPES.LINEAR_REGRESSION.value:
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train(self, X_train, y_train, preprocessor, log_transformed=False):
        """
        Trains the regression pipeline.

        Args:
            log_transformed (bool): Set True if y_train is log-scaled (common for Linear Regression).
        """
        self.is_log_transformed = log_transformed
        model = self._get_regressor()

        self.pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])

        print(f"Training {self.model_type} Regressor...")
        self.pipeline.fit(X_train, y_train)
        print("Training Complete.")

    def evaluate(self, X_test, y_test):
        """
        Predicts and prints RMSE, MAE, R2.
        Automatically reverts log-transformation for metrics if needed.
        """
        if not self.pipeline:
            raise Exception("Model not trained yet.")

        # Predict
        y_pred = self.pipeline.predict(X_test)

        # Handle Log Transformation Reversal for Evaluation
        if self.is_log_transformed:
            y_pred = np.exp(y_pred)
            y_true = np.exp(y_test)
        else:
            y_true = y_test

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

        print(f"\nModel Performance ({self.model_type}):")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return metrics, y_pred

    def explain_model(self, X_test, sample_size=100):
        """Generates SHAP plots for the regressor."""
        if self.model_type == MODEL_TYPES.LINEAR_REGRESSION.value:
            print(
                "SHAP explanation skipped for Linear Model (Focus on Coefficients instead)."
            )
            return

        # 1. Prepare Data
        preprocessor = self.pipeline.named_steps["preprocessor"]
        model = self.pipeline.named_steps["regressor"]
        X_transformed = preprocessor.transform(X_test)

        # 2. Get Feature Names
        try:
            cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out()
            feature_names = list(cat_names) + [
                f"Num_Feature_{i}"
                for i in range(X_transformed.shape[1] - len(cat_names))
            ]
        except:
            feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]

        # 3. Calculate SHAP
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        if X_transformed.shape[0] > sample_size:
            idx = np.random.choice(X_transformed.shape[0], sample_size, replace=False)
            X_sample = X_transformed[idx]
        else:
            X_sample = X_transformed

        print("\nCalculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

        # 4. Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_names, show=False
        )
        plt.title(f"SHAP Feature Importance ({self.model_type})")
        plt.tight_layout()
        plt.show()
