import sys

import numpy as np

from HealthPulse_AI_project.exception.exception import (
    HealthPulseException
)

from HealthPulse_AI_project.logging.logger import logging


class ModelPreparation:

    def __init__(self):
        pass

    # ==========================================
    # Validate feature array
    # ==========================================

    def _validate_features(
        self,
        X,
        name
    ):

        if X is None:

            raise ValueError(
                f"{name} features are None"
            )

        if not isinstance(
            X,
            np.ndarray
        ):

            raise ValueError(
                f"{name} features must be "
                "a NumPy array"
            )

        if X.ndim != 2:

            raise ValueError(
                f"{name} features must be "
                "2-dimensional"
            )

        if X.shape[1] != 172:

            raise ValueError(
                f"{name} expected 172 features, "
                f"got {X.shape[1]}"
            )

        if not np.isfinite(X).all():

            raise ValueError(
                f"{name} contains NaN or Inf"
            )

    # ==========================================
    # Validate target
    # ==========================================

    def _validate_target(
        self,
        y,
        expected_records,
        name
    ):

        if y is None:

            raise ValueError(
                f"{name} target is None"
            )

        y = np.asarray(
            y
        ).reshape(-1)

        if len(y) != expected_records:

            raise ValueError(
                f"{name} target count "
                f"({len(y)}) does not match "
                f"feature count "
                f"({expected_records})"
            )

        if not np.isin(
            y,
            [0, 1]
        ).all():

            raise ValueError(
                f"{name} target contains "
                "values other than 0 and 1"
            )

        if len(np.unique(y)) < 2:

            raise ValueError(
                f"{name} contains only one class"
            )

        return y.astype(
            np.int32
        )

    # ==========================================
    # Prepare one split
    # ==========================================

    def _prepare_split(
        self,
        X,
        y,
        name
    ):

        # --------------------------------------
        # Validate features
        # --------------------------------------

        self._validate_features(
            X,
            name
        )

        # --------------------------------------
        # Validate target
        # --------------------------------------

        y = self._validate_target(
            y,
            X.shape[0],
            name
        )

        # --------------------------------------
        # Convert features to float32
        # --------------------------------------

        X = X.astype(
            np.float32
        )

        # --------------------------------------
        # Final alignment check
        # --------------------------------------

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
        name
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
        train_features,
        train_target,
        validation_features,
        validation_target,
        test_features,
        test_target
    ):

        try:

            logging.info(
                "Starting model preparation"
            )

            # ==================================
            # TRAIN
            # ==================================

            (
                X_train,
                y_train
            ) = self._prepare_split(
                train_features,
                train_target,
                "Train"
            )

            # ==================================
            # VALIDATION
            # ==================================

            (
                X_validation,
                y_validation
            ) = self._prepare_split(
                validation_features,
                validation_target,
                "Validation"
            )

            # ==================================
            # TEST
            # ==================================

            (
                X_test,
                y_test
            ) = self._prepare_split(
                test_features,
                test_target,
                "Test"
            )

            # ==================================
            # Class distribution
            # ==================================

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

            # ==================================
            # Feature consistency
            # ==================================

            if (
                X_train.shape[1]
                != X_validation.shape[1]
            ):

                raise ValueError(
                    "Train and validation "
                    "feature counts do not match"
                )

            if (
                X_train.shape[1]
                != X_test.shape[1]
            ):

                raise ValueError(
                    "Train and test "
                    "feature counts do not match"
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