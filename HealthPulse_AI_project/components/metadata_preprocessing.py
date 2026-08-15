import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from HealthPulse_AI_project.exception.exception import (
    HealthPulseException
)

from HealthPulse_AI_project.logging.logger import logging


class MetaDataPreprocessing:

    def __init__(
        self,
        output_path: str = "project/data/processed/metadata"
    ):

        self.output_path = output_path

        os.makedirs(
            self.output_path,
            exist_ok=True
        )

        self.transformer_path = os.path.join(
            self.output_path,
            "metadata_preprocessor.pkl"
        )

        # Preserve imputed + BMI DataFrames
        # for downstream components if required.

        self.train_imputed_df = None
        self.validation_imputed_df = None
        self.test_imputed_df = None

    # ==========================================
    # Create BMI
    # ==========================================

    def _create_bmi(
        self,
        df: pd.DataFrame,
        name: str
    ):

        result_df = df.copy()

        if "height" not in result_df.columns:
            raise ValueError(
                f"{name}: height column is missing"
            )

        if "weight" not in result_df.columns:
            raise ValueError(
                f"{name}: weight column is missing"
            )

        # Height is stored in centimeters.

        height_m = (
            result_df["height"] / 100.0
        )

        # BMI = weight / height²

        result_df["BMI"] = (
            result_df["weight"]
            / np.square(height_m)
        )

        # Handle invalid mathematical values.

        result_df["BMI"] = (
            result_df["BMI"]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        if result_df["BMI"].isna().any():

            raise ValueError(
                f"{name}: BMI contains missing "
                "values after calculation"
            )

        if not np.isfinite(
            result_df["BMI"]
        ).all():

            raise ValueError(
                f"{name}: BMI contains "
                "infinite values"
            )

        return result_df

    # ==========================================
    # Validate dataframe
    # ==========================================

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        name: str
    ):

        if df is None or df.empty:

            raise ValueError(
                f"{name} dataframe is empty"
            )

        required_columns = [
            "age",
            "sex",
            "height",
            "weight",
            "site",
            "nurse",
            "heart_axis",
            "device",
            "second_opinion"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{name} dataframe is missing "
                f"columns: {missing_columns}"
            )

    # ==========================================
    # Main preprocessing
    # ==========================================

    def initiate_metadata_preprocessing(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):

        try:

            logging.info(
                "Starting metadata preprocessing"
            )

            # ==========================================
            # 1. Validate input
            # ==========================================

            self._validate_dataframe(
                train_df,
                "Train"
            )

            self._validate_dataframe(
                validation_df,
                "Validation"
            )

            self._validate_dataframe(
                test_df,
                "Test"
            )

            # ==========================================
            # 2. Select metadata columns
            # ==========================================

            metadata_columns = [
                "age",
                "sex",
                "height",
                "weight",
                "site",
                "nurse",
                "heart_axis",
                "device",
                "second_opinion"
            ]

            x_train = train_df[
                metadata_columns
            ].copy()

            x_validation = validation_df[
                metadata_columns
            ].copy()

            x_test = test_df[
                metadata_columns
            ].copy()

            # ==========================================
            # 3. Define feature types
            # ==========================================

            numerical_features = [
                "age",
                "height",
                "weight",
                "BMI"
            ]

            categorical_features = [
                "site",
                "nurse",
                "heart_axis",
                "device"
            ]

            binary_features = [
                "sex",
                "second_opinion"
            ]

            # ==========================================
            # 4. Numerical pipeline
            # ==========================================

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # ==========================================
            # 5. Categorical pipeline
            # ==========================================

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            # ==========================================
            # 6. Binary pipeline
            # ==========================================

            binary_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    )
                ]
            )

            # ==========================================
            # 7. We need BMI BEFORE fitting transformer
            #
            # To ensure height/weight imputation
            # comes from TRAIN only, calculate BMI
            # after obtaining TRAIN-derived medians.
            # ==========================================

            # Fit temporary imputer ONLY on training
            # height and weight.

            height_weight_imputer = SimpleImputer(
                strategy="median"
            )

            train_hw = height_weight_imputer.fit_transform(
                x_train[
                    ["height", "weight"]
                ]
            )

            validation_hw = (
                height_weight_imputer.transform(
                    x_validation[
                        ["height", "weight"]
                    ]
                )
            )

            test_hw = (
                height_weight_imputer.transform(
                    x_test[
                        ["height", "weight"]
                    ]
                )
            )

            # Put imputed height/weight back.

            x_train[
                ["height", "weight"]
            ] = train_hw

            x_validation[
                ["height", "weight"]
            ] = validation_hw

            x_test[
                ["height", "weight"]
            ] = test_hw

            logging.info(
                "Height/weight imputation fitted "
                "using TRAIN data only"
            )

            # ==========================================
            # 8. Create BMI
            # ==========================================

            x_train = self._create_bmi(
                x_train,
                "Train"
            )

            x_validation = self._create_bmi(
                x_validation,
                "Validation"
            )

            x_test = self._create_bmi(
                x_test,
                "Test"
            )

            # ==========================================
            # 9. Save imputed + BMI DataFrames
            # ==========================================

            self.train_imputed_df = (
                train_df.copy()
            )

            self.validation_imputed_df = (
                validation_df.copy()
            )

            self.test_imputed_df = (
                test_df.copy()
            )

            self.train_imputed_df[
                ["height", "weight"]
            ] = train_hw

            self.validation_imputed_df[
                ["height", "weight"]
            ] = validation_hw

            self.test_imputed_df[
                ["height", "weight"]
            ] = test_hw

            self.train_imputed_df["BMI"] = (
                x_train["BMI"].values
            )

            self.validation_imputed_df["BMI"] = (
                x_validation["BMI"].values
            )

            self.test_imputed_df["BMI"] = (
                x_test["BMI"].values
            )

            # ==========================================
            # 10. Combined transformer
            # ==========================================

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical",
                        numerical_pipeline,
                        numerical_features
                    ),
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_features
                    ),
                    (
                        "binary",
                        binary_pipeline,
                        binary_features
                    )
                ]
            )

            # ==========================================
            # 11. FIT TRAIN ONLY
            # ==========================================

            logging.info(
                "Fitting metadata transformer "
                "using TRAIN data only"
            )

            x_train_processed = (
                preprocessor.fit_transform(
                    x_train
                )
            )

            # ==========================================
            # 12. Transform validation
            # ==========================================

            x_validation_processed = (
                preprocessor.transform(
                    x_validation
                )
            )

            # ==========================================
            # 13. Transform test
            # ==========================================

            x_test_processed = (
                preprocessor.transform(
                    x_test
                )
            )

            # ==========================================
            # 14. Validate processed arrays
            # ==========================================

            if not np.isfinite(
                x_train_processed
            ).all():

                raise ValueError(
                    "Train metadata contains "
                    "NaN or Inf"
                )

            if not np.isfinite(
                x_validation_processed
            ).all():

                raise ValueError(
                    "Validation metadata contains "
                    "NaN or Inf"
                )

            if not np.isfinite(
                x_test_processed
            ).all():

                raise ValueError(
                    "Test metadata contains "
                    "NaN or Inf"
                )

            # ==========================================
            # 15. Save transformer
            # ==========================================

            joblib.dump(
                preprocessor,
                self.transformer_path
            )

            logging.info(
                f"Metadata preprocessor saved to: "
                f"{self.transformer_path}"
            )

            # ==========================================
            # 16. Feature names
            # ==========================================

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            logging.info(
                f"Final metadata feature count: "
                f"{len(feature_names)}"
            )

            logging.info(
                "Metadata preprocessing completed "
                "successfully"
            )

            return (
                x_train_processed,
                x_validation_processed,
                x_test_processed,
                feature_names
            )

        except Exception as e:

            logging.exception(
                "Error occurred during "
                "metadata preprocessing"
            )

            raise HealthPulseException(
                e,
                sys
            )