import sys

import numpy as np
import pandas as pd

from HealthPulse_AI_project.exception.exception import (
    HealthPulseException
)

from HealthPulse_AI_project.logging.logger import logging


class ModelPreparation:

    def __init__(self):
        pass

    # ==========================================
    # Validate input dataframe
    # ==========================================

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        name: str
    ):

        if df is None:
            raise ValueError(
                f"{name} dataframe is None"
            )

        if df.empty:
            raise ValueError(
                f"{name} dataframe is empty"
            )

        required_columns = [
            "ecg_id",
            "Target"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{name} missing columns: "
                f"{missing_columns}"
            )

    # ==========================================
    # Validate target
    # ==========================================

    def _validate_target(
        self,
        y,
        name: str
    ):

        y = np.asarray(y).reshape(-1)

        if not np.isin(
            y,
            [0, 1]
        ).all():

            raise ValueError(
                f"{name} target contains "
                "invalid values"
            )

        if len(np.unique(y)) < 2:

            raise ValueError(
                f"{name} target contains "
                "only one class"
            )

        return y.astype(np.int32)

    # ==========================================
    # Prepare one split
    # ==========================================

    def _prepare_split(
        self,
        df: pd.DataFrame,
        name: str
    ):

        self._validate_dataframe(
            df,
            name
        )

        # ------------------------------------------
        # Separate target
        # ------------------------------------------

        y = df["Target"].to_numpy()

        # ------------------------------------------
        # Remove identifier and target
        # ------------------------------------------

        X = df.drop(
            columns=[
                "ecg_id",
                "Target"
            ]
        ).copy()

        # ------------------------------------------
        # Validate feature count
        # ------------------------------------------

        if X.shape[1] != 172:

            raise ValueError(
                f"{name} expected 172 features "
                f"after removing ecg_id and Target, "
                f"got {X.shape[1]}"
            )

        # ------------------------------------------
        # Convert features to float32
        # ------------------------------------------

        X = X.to_numpy(
            dtype=np.float32
        )

        # ------------------------------------------
        # Validate NaN / Inf
        # ------------------------------------------

        if not np.isfinite(X).all():

            raise ValueError(
                f"{name} features contain "
                "NaN or infinite values"
            )

        # ------------------------------------------
        # Validate target
        # ------------------------------------------

        y = self._validate_target(
            y,
            name
        )

        # ------------------------------------------
        # Validate row alignment
        # ------------------------------------------

        if X.shape[0] != len(y):

            raise ValueError(
                f"{name} X and y row counts "
                "do not match"
            )

        logging.info(
            f"{name} X shape: {X.shape}"
        )

        logging.info(
            f"{name} y shape: {y.shape}"
        )

        return X, y

    # ==========================================
    # Log class distribution
    # ==========================================

    def _log_class_distribution(
        self,
        y,
        name: str
    ):

        class_counts = np.bincount(
            y,
            minlength=2
        )

        non_mi = int(
            class_counts[0]
        )

        mi = int(
            class_counts[1]
        )

        total = len(y)

        logging.info(
            f"{name} records: {total}"
        )

        logging.info(
            f"{name} Non-MI: "
            f"{non_mi} "
            f"({non_mi / total * 100:.2f}%)"
        )

        logging.info(
            f"{name} MI: "
            f"{mi} "
            f"({mi / total * 100:.2f}%)"
        )

    # ==========================================
    # Main method
    # ==========================================

    def initiate_model_preparation(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):

        try:

            logging.info(
                "Starting model preparation"
            )

            # ==========================================
            # TRAIN
            # ==========================================

            (
                X_train,
                y_train
            ) = self._prepare_split(
                train_df,
                "Train"
            )

            # ==========================================
            # VALIDATION
            # ==========================================

            (
                X_validation,
                y_validation
            ) = self._prepare_split(
                validation_df,
                "Validation"
            )

            # ==========================================
            # TEST
            # ==========================================

            (
                X_test,
                y_test
            ) = self._prepare_split(
                test_df,
                "Test"
            )

            # ==========================================
            # Log distributions
            # ==========================================

            self._log_class_distribution(
                y_train,
                "Train"
            )

            self._log_class_distribution(
                y_validation,
                "Validation"
            )

            self._log_class_distribution(
                y_test,
                "Test"
            )

            # ==========================================
            # Final validation
            # ==========================================

            if X_train.shape[1] != X_validation.shape[1]:
                raise ValueError(
                    "Train and validation feature "
                    "counts do not match"
                )

            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(
                    "Train and test feature "
                    "counts do not match"
                )

            logging.info(
                "Model preparation completed "
                "successfully"
            )

            return (
                X_train,
                y_train,
                X_validation,
                y_validation,
                X_test,
                y_test
            )

        except Exception as e:

            logging.exception(
                "Error occurred during "
                "model preparation"
            )

            raise HealthPulseException(
                e,
                sys
            )