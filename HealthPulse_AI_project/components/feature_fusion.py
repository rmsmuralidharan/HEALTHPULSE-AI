import sys

import numpy as np

from HealthPulse_AI_project.exception.exception import (
    HealthPulseException
)

from HealthPulse_AI_project.logging.logger import logging


class FeatureFusion:

    def __init__(self):
        pass

    # ==========================================
    # 1. Validate feature array
    # ==========================================

    def _validate_feature_array(
        self,
        features,
        expected_features,
        name
    ):

        if features is None:

            raise ValueError(
                f"{name} features are None"
            )

        if not isinstance(
            features,
            np.ndarray
        ):

            raise ValueError(
                f"{name} features must be "
                "a NumPy array"
            )

        if features.ndim != 2:

            raise ValueError(
                f"{name} features must be "
                "2-dimensional"
            )

        if features.shape[1] != expected_features:

            raise ValueError(
                f"{name} expected "
                f"{expected_features} features, "
                f"got {features.shape[1]}"
            )

        if not np.isfinite(
            features
        ).all():

            raise ValueError(
                f"{name} contains "
                "NaN or infinite values"
            )

    # ==========================================
    # 2. Validate target
    # ==========================================

    def _validate_target(
        self,
        target,
        expected_records,
        name
    ):

        if target is None:

            raise ValueError(
                f"{name} target is None"
            )

        target = np.asarray(
            target
        ).reshape(-1)

        if len(target) != expected_records:

            raise ValueError(
                f"{name} target count "
                f"does not match feature count"
            )

        if not np.isin(
            target,
            [0, 1]
        ).all():

            raise ValueError(
                f"{name} target contains "
                "values other than 0 and 1"
            )

        return target

    # ==========================================
    # 3. Validate ECG IDs
    # ==========================================

    def _validate_ecg_ids(
        self,
        ecg_ids,
        expected_records,
        name
    ):

        if ecg_ids is None:

            raise ValueError(
                f"{name} ECG IDs are None"
            )

        ecg_ids = np.asarray(
            ecg_ids
        ).reshape(-1)

        if len(ecg_ids) != expected_records:

            raise ValueError(
                f"{name} ECG ID count "
                "does not match feature count"
            )

        if len(np.unique(ecg_ids)) != len(ecg_ids):

            raise ValueError(
                f"{name} contains duplicate "
                "ECG IDs"
            )

        return ecg_ids

    # ==========================================
    # 4. Fuse one split
    # ==========================================

    def _fuse_split(
        self,
        ecg_features,
        metadata_features,
        target,
        ecg_ids,
        name
    ):

        # ------------------------------------------
        # Validate ECG
        # ------------------------------------------

        self._validate_feature_array(
            ecg_features,
            84,
            f"{name} ECG"
        )

        # ------------------------------------------
        # Validate metadata
        # ------------------------------------------

        self._validate_feature_array(
            metadata_features,
            88,
            f"{name} Metadata"
        )

        # ------------------------------------------
        # Validate row count
        # ------------------------------------------

        if (
            ecg_features.shape[0]
            != metadata_features.shape[0]
        ):

            raise ValueError(
                f"{name} ECG and metadata "
                "record counts do not match"
            )

        record_count = (
            ecg_features.shape[0]
        )

        # ------------------------------------------
        # Validate target
        # ------------------------------------------

        target = self._validate_target(
            target,
            record_count,
            name
        )

        # ------------------------------------------
        # Validate ECG IDs
        # ------------------------------------------

        ecg_ids = self._validate_ecg_ids(
            ecg_ids,
            record_count,
            name
        )

        # ------------------------------------------
        # Concatenate ECG + metadata
        # ------------------------------------------

        fused_features = np.concatenate(
            [
                ecg_features,
                metadata_features
            ],
            axis=1
        ).astype(
            np.float32
        )

        # ------------------------------------------
        # Final validation
        # ------------------------------------------

        if fused_features.shape[1] != 172:

            raise ValueError(
                f"{name} expected 172 fused "
                f"features, got "
                f"{fused_features.shape[1]}"
            )

        if not np.isfinite(
            fused_features
        ).all():

            raise ValueError(
                f"{name} fused features "
                "contain NaN or Inf"
            )

        logging.info(
            f"{name} fused shape: "
            f"{fused_features.shape}"
        )

        return (
            fused_features,
            target,
            ecg_ids
        )

    # ==========================================
    # 5. Main fusion method
    # ==========================================

    def initiate_feature_fusion(
        self,
        train_ecg_features,
        validation_ecg_features,
        test_ecg_features,
        train_metadata_features,
        validation_metadata_features,
        test_metadata_features,
        y_train,
        y_validation,
        y_test,
        train_ecg_ids,
        validation_ecg_ids,
        test_ecg_ids
    ):

        try:

            logging.info(
                "Starting feature fusion"
            )

            # ==========================================
            # TRAIN
            # ==========================================

            (
                train_fused,
                train_target,
                train_ids
            ) = self._fuse_split(
                train_ecg_features,
                train_metadata_features,
                y_train,
                train_ecg_ids,
                "Train"
            )

            # ==========================================
            # VALIDATION
            # ==========================================

            (
                validation_fused,
                validation_target,
                validation_ids
            ) = self._fuse_split(
                validation_ecg_features,
                validation_metadata_features,
                y_validation,
                validation_ecg_ids,
                "Validation"
            )

            # ==========================================
            # TEST
            # ==========================================

            (
                test_fused,
                test_target,
                test_ids
            ) = self._fuse_split(
                test_ecg_features,
                test_metadata_features,
                y_test,
                test_ecg_ids,
                "Test"
            )

            # ==========================================
            # Final logging
            # ==========================================

            logging.info(
                "Feature fusion completed successfully"
            )

            return (
                train_fused,
                validation_fused,
                test_fused,
                train_target,
                validation_target,
                test_target,
                train_ids,
                validation_ids,
                test_ids
            )

        except Exception as e:

            logging.exception(
                "Error occurred during feature fusion"
            )

            raise HealthPulseException(
                e,
                sys
            )