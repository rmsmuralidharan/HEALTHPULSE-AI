import sys

import numpy as np
import pandas as pd

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging


class MetadataFeatureEngineering:

    def __init__(self):
        pass

    # ==========================================
    # 1. Validate input dataframe
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
            "patient_id",
            "height",
            "weight",
            "Target"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{name} dataframe is missing "
                f"required columns: "
                f"{missing_columns}"
            )

        # Target must remain binary

        if not df["Target"].isin([0, 1]).all():

            raise ValueError(
                f"{name} contains invalid Target values"
            )

    # ==========================================
    # 2. Validate height and weight
    # ==========================================

    def _validate_height_weight(
        self,
        df: pd.DataFrame,
        name: str
    ):

        missing_height = int(
            df["height"].isna().sum()
        )

        missing_weight = int(
            df["weight"].isna().sum()
        )

        logging.info(
            f"{name} missing height values: "
            f"{missing_height}"
        )

        logging.info(
            f"{name} missing weight values: "
            f"{missing_weight}"
        )

        # Height and weight must already be
        # imputed by MetadataPreprocessing.

        if missing_height > 0:

            raise ValueError(
                f"{name} contains missing height "
                "values. Height must be imputed "
                "before feature engineering."
            )

        if missing_weight > 0:

            raise ValueError(
                f"{name} contains missing weight "
                "values. Weight must be imputed "
                "before feature engineering."
            )

    # ==========================================
    # 3. Create BMI
    # ==========================================

    def _create_bmi(
        self,
        df: pd.DataFrame
    ):

        feature_df = df.copy()

        # PTB-XL height is stored in centimeters.
        # Convert centimeters to meters.

        height_m = (
            feature_df["height"] / 100.0
        )

        # BMI = weight / height²

        feature_df["BMI"] = (
            feature_df["weight"]
            / np.square(height_m)
        )

        # Replace infinite values with NaN
        # so validation can detect them.

        feature_df["BMI"] = (
            feature_df["BMI"]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        return feature_df

    # ==========================================
    # 4. Validate BMI
    # ==========================================

    def _validate_bmi(
        self,
        df: pd.DataFrame,
        name: str
    ):

        if "BMI" not in df.columns:

            raise ValueError(
                f"{name}: BMI feature was not created"
            )

        missing_bmi = int(
            df["BMI"].isna().sum()
        )

        infinite_bmi = int(
            np.isinf(
                df["BMI"].dropna()
            ).sum()
        )

        logging.info(
            f"{name} missing BMI values: "
            f"{missing_bmi}"
        )

        logging.info(
            f"{name} infinite BMI values: "
            f"{infinite_bmi}"
        )

        if missing_bmi > 0:

            raise ValueError(
                f"{name} contains missing BMI values"
            )

        if infinite_bmi > 0:

            raise ValueError(
                f"{name} contains infinite BMI values"
            )

    # ==========================================
    # 5. Validate row count
    # ==========================================

    def _validate_row_count(
        self,
        original_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        name: str
    ):

        if len(original_df) != len(feature_df):

            raise ValueError(
                f"{name} row count changed "
                "during feature engineering"
            )

    # ==========================================
    # 6. Main feature engineering
    # ==========================================

    def initiate_feature_engineering(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):

        try:

            logging.info(
                "Starting metadata feature engineering"
            )

            # ==========================================
            # Validate input
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
            # Validate imputation
            # ==========================================

            self._validate_height_weight(
                train_df,
                "Train"
            )

            self._validate_height_weight(
                validation_df,
                "Validation"
            )

            self._validate_height_weight(
                test_df,
                "Test"
            )

            # ==========================================
            # Create BMI
            # ==========================================

            train_features = self._create_bmi(
                train_df
            )

            validation_features = (
                self._create_bmi(
                    validation_df
                )
            )

            test_features = self._create_bmi(
                test_df
            )

            # ==========================================
            # Validate BMI
            # ==========================================

            self._validate_bmi(
                train_features,
                "Train"
            )

            self._validate_bmi(
                validation_features,
                "Validation"
            )

            self._validate_bmi(
                test_features,
                "Test"
            )

            # ==========================================
            # Validate row counts
            # ==========================================

            self._validate_row_count(
                train_df,
                train_features,
                "Train"
            )

            self._validate_row_count(
                validation_df,
                validation_features,
                "Validation"
            )

            self._validate_row_count(
                test_df,
                test_features,
                "Test"
            )

            # ==========================================
            # Log results
            # ==========================================

            logging.info(
                f"Train feature shape: "
                f"{train_features.shape}"
            )

            logging.info(
                f"Validation feature shape: "
                f"{validation_features.shape}"
            )

            logging.info(
                f"Test feature shape: "
                f"{test_features.shape}"
            )

            logging.info(
                "BMI feature created successfully"
            )

            logging.info(
                "Metadata feature engineering "
                "completed successfully"
            )

            return (
                train_features,
                validation_features,
                test_features
            )

        except Exception as e:

            logging.exception(
                "Error occurred during "
                "metadata feature engineering"
            )

            raise HealthPulseException(
                e,
                sys
            )