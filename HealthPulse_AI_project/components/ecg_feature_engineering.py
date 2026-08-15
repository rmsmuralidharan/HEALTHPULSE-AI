import sys

import numpy as np

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging


class ECGFeatureEngineering:

    def __init__(self):

        self.feature_names = []

        # 12 ECG leads
        # 7 features per lead

        for lead in range(1, 13):

            self.feature_names.extend([
                f"lead_{lead}_mean",
                f"lead_{lead}_std",
                f"lead_{lead}_min",
                f"lead_{lead}_max",
                f"lead_{lead}_range",
                f"lead_{lead}_rms",
                f"lead_{lead}_energy"
            ])

    # ==========================================
    # 1. Validate ECG array
    # ==========================================

    def _validate_ecg_array(
        self,
        X,
        name: str
    ):

        if X is None:

            raise ValueError(
                f"{name} ECG data is None"
            )

        if not isinstance(X, np.ndarray):

            raise ValueError(
                f"{name} ECG data must be "
                "a NumPy array"
            )

        if X.ndim != 3:

            raise ValueError(
                f"{name} ECG data must have "
                f"3 dimensions. Got {X.ndim}"
            )

        if X.shape[1:] != (1000, 12):

            raise ValueError(
                f"{name} ECG shape is invalid. "
                f"Expected (*, 1000, 12), "
                f"got {X.shape}"
            )

        if not np.isfinite(X).all():

            raise ValueError(
                f"{name} ECG contains "
                "NaN or infinite values"
            )

    # ==========================================
    # 2. Extract features from one ECG
    # ==========================================

    def _extract_single_ecg_features(
        self,
        signal
    ):

        features = []

        # signal shape:
        # (1000, 12)

        for lead in range(12):

            lead_signal = signal[:, lead]

            mean = np.mean(
                lead_signal
            )

            std = np.std(
                lead_signal
            )

            minimum = np.min(
                lead_signal
            )

            maximum = np.max(
                lead_signal
            )

            signal_range = (
                maximum - minimum
            )

            rms = np.sqrt(
                np.mean(
                    np.square(
                        lead_signal
                    )
                )
            )

            energy = np.sum(
                np.square(
                    lead_signal
                )
            )

            features.extend([
                mean,
                std,
                minimum,
                maximum,
                signal_range,
                rms,
                energy
            ])

        return features

    # ==========================================
    # 3. Process one split
    # ==========================================

    def _process_split(
        self,
        X,
        name: str
    ):

        feature_rows = []

        logging.info(
            f"Starting ECG feature extraction "
            f"for {name}"
        )

        for index, signal in enumerate(X):

            features = (
                self._extract_single_ecg_features(
                    signal
                )
            )

            feature_rows.append(
                features
            )

            if (index + 1) % 1000 == 0:

                logging.info(
                    f"{name}: processed "
                    f"{index + 1} ECG signals"
                )

        if not feature_rows:

            raise ValueError(
                f"No ECG features generated "
                f"for {name}"
            )

        feature_array = np.asarray(
            feature_rows,
            dtype=np.float32
        )

        return feature_array

    # ==========================================
    # 4. Validate generated features
    # ==========================================

    def _validate_features(
        self,
        features,
        name: str
    ):

        if features.ndim != 2:

            raise ValueError(
                f"{name} ECG features must "
                "be 2-dimensional"
            )

        # 12 leads × 7 features

        expected_features = 84

        if features.shape[1] != expected_features:

            raise ValueError(
                f"{name} expected "
                f"{expected_features} ECG features, "
                f"got {features.shape[1]}"
            )

        if not np.isfinite(
            features
        ).all():

            raise ValueError(
                f"{name} ECG features contain "
                "NaN or infinite values"
            )

    # ==========================================
    # 5. Main method
    # ==========================================

    def initiate_ecg_feature_engineering(
        self,
        X_train,
        X_validation,
        X_test
    ):

        try:

            logging.info(
                "Starting ECG feature engineering"
            )

            # ==========================================
            # Validate input ECGs
            # ==========================================

            self._validate_ecg_array(
                X_train,
                "Train"
            )

            self._validate_ecg_array(
                X_validation,
                "Validation"
            )

            self._validate_ecg_array(
                X_test,
                "Test"
            )

            # ==========================================
            # Extract engineered features
            # ==========================================

            train_features = (
                self._process_split(
                    X_train,
                    "Train"
                )
            )

            validation_features = (
                self._process_split(
                    X_validation,
                    "Validation"
                )
            )

            test_features = (
                self._process_split(
                    X_test,
                    "Test"
                )
            )

            # ==========================================
            # Validate features
            # ==========================================

            self._validate_features(
                train_features,
                "Train"
            )

            self._validate_features(
                validation_features,
                "Validation"
            )

            self._validate_features(
                test_features,
                "Test"
            )

            # ==========================================
            # Validate row counts
            # ==========================================

            if (
                train_features.shape[0]
                != X_train.shape[0]
            ):

                raise ValueError(
                    "Train ECG feature row count "
                    "does not match input"
                )

            if (
                validation_features.shape[0]
                != X_validation.shape[0]
            ):

                raise ValueError(
                    "Validation ECG feature row count "
                    "does not match input"
                )

            if (
                test_features.shape[0]
                != X_test.shape[0]
            ):

                raise ValueError(
                    "Test ECG feature row count "
                    "does not match input"
                )

            # ==========================================
            # Logging
            # ==========================================

            logging.info(
                f"Train ECG feature shape: "
                f"{train_features.shape}"
            )

            logging.info(
                f"Validation ECG feature shape: "
                f"{validation_features.shape}"
            )

            logging.info(
                f"Test ECG feature shape: "
                f"{test_features.shape}"
            )

            logging.info(
                f"Total ECG engineered features: "
                f"{len(self.feature_names)}"
            )

            logging.info(
                "ECG feature engineering "
                "completed successfully"
            )

            # IMPORTANT:
            # Return engineered features AND
            # preserve original ECG arrays.

            return (
                train_features,
                validation_features,
                test_features,
                self.feature_names,
                X_train,
                X_validation,
                X_test
            )

        except Exception as e:

            logging.exception(
                "Error occurred during "
                "ECG feature engineering"
            )

            raise HealthPulseException(
                e,
                sys
            )