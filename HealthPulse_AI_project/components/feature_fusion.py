import sys

import numpy as np
import pandas as pd

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging


class FeatureFusion:

    def __init__(self):
        pass

    # ==========================================
    # 1. Validate ECG features
    # ==========================================

    def _validate_ecg_features(
        self,
        ecg_features,
        name: str
    ):

        if ecg_features is None:

            raise ValueError(
                f"{name} ECG features are None"
            )

        if not isinstance(
            ecg_features,
            np.ndarray
        ):

            raise ValueError(
                f"{name} ECG features must "
                "be a NumPy array"
            )

        if ecg_features.ndim != 2:

            raise ValueError(
                f"{name} ECG features must "
                "be 2-dimensional"
            )

        if ecg_features.shape[1] != 84:

            raise ValueError(
                f"{name} must contain "
                f"84 ECG features. "
                f"Got {ecg_features.shape[1]}"
            )

        if not np.isfinite(
            ecg_features
        ).all():

            raise ValueError(
                f"{name} ECG features contain "
                "NaN or infinite values"
            )

    # ==========================================
    # 2. Validate metadata
    # ==========================================

    def _validate_metadata(
        self,
        metadata_df: pd.DataFrame,
        name: str
    ):

        if metadata_df is None:

            raise ValueError(
                f"{name} metadata is None"
            )

        if metadata_df.empty:

            raise ValueError(
                f"{name} metadata is empty"
            )

        required_columns = [
            "ecg_id",
            "Target"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in metadata_df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{name} metadata is missing "
                f"columns: {missing_columns}"
            )

        if metadata_df["ecg_id"].duplicated().any():

            raise ValueError(
                f"{name} metadata contains "
                "duplicate ecg_id values"
            )

        if not metadata_df["Target"].isin(
            [0, 1]
        ).all():

            raise ValueError(
                f"{name} metadata contains "
                "invalid Target values"
            )

    # ==========================================
    # 3. Validate original split
    # ==========================================

    def _validate_original_split(
        self,
        split_df: pd.DataFrame,
        name: str
    ):

        if split_df is None:

            raise ValueError(
                f"{name} split dataframe is None"
            )

        if split_df.empty:

            raise ValueError(
                f"{name} split dataframe is empty"
            )

        required_columns = [
            "ecg_id",
            "Target"
        ]

        for column in required_columns:

            if column not in split_df.columns:

                raise ValueError(
                    f"{name} split is missing "
                    f"column: {column}"
                )

        if split_df["ecg_id"].duplicated().any():

            raise ValueError(
                f"{name} split contains "
                "duplicate ecg_id values"
            )

    # ==========================================
    # 4. Validate ECG alignment
    # ==========================================

    def _validate_ecg_alignment(
        self,
        ecg_features,
        split_df: pd.DataFrame,
        name: str
    ):

        if len(ecg_features) != len(split_df):

            raise ValueError(
                f"{name} ECG feature count "
                f"({len(ecg_features)}) does not "
                f"match split records "
                f"({len(split_df)})"
            )

    # ==========================================
    # 5. Create ECG feature DataFrame
    # ==========================================

    def _create_ecg_dataframe(
        self,
        ecg_features,
        feature_names,
        split_df
    ):

        if len(feature_names) != 84:

            raise ValueError(
                "ECG feature name count must "
                "be 84"
            )

        ecg_feature_df = pd.DataFrame(
            ecg_features,
            columns=feature_names
        )

        # Use the original record order to
        # attach the ECG identifier.

        ecg_feature_df.insert(
            0,
            "ecg_id",
            split_df["ecg_id"].values
        )

        return ecg_feature_df

    # ==========================================
    # 6. Fuse one split
    # ==========================================

    def _fuse_split(
        self,
        ecg_features,
        metadata_df,
        split_df,
        feature_names,
        name: str
    ):

        # ------------------------------------------
        # Validate
        # ------------------------------------------

        self._validate_ecg_features(
            ecg_features,
            name
        )

        self._validate_metadata(
            metadata_df,
            name
        )

        self._validate_original_split(
            split_df,
            name
        )

        self._validate_ecg_alignment(
            ecg_features,
            split_df,
            name
        )

        # ------------------------------------------
        # Create ECG feature DataFrame
        # ------------------------------------------

        ecg_feature_df = (
            self._create_ecg_dataframe(
                ecg_features,
                feature_names,
                split_df
            )
        )

        # ------------------------------------------
        # Keep only metadata features
        #
        # Target is kept separately.
        # ecg_id is used only for alignment.
        # ------------------------------------------

        metadata_features = metadata_df.drop(
            columns=["Target"],
            errors="ignore"
        ).copy()

        # ------------------------------------------
        # Merge using ecg_id
        # ------------------------------------------

        fused_df = pd.merge(
            ecg_feature_df,
            metadata_features,
            on="ecg_id",
            how="inner",
            validate="one_to_one"
        )

        # ------------------------------------------
        # Add target from original split
        # ------------------------------------------

        target_df = split_df[
            ["ecg_id", "Target"]
        ].copy()

        fused_df = pd.merge(
            fused_df,
            target_df,
            on="ecg_id",
            how="inner",
            validate="one_to_one"
        )

        # ------------------------------------------
        # Validate record count
        # ------------------------------------------

        if len(fused_df) != len(split_df):

            raise ValueError(
                f"{name} fusion caused record loss. "
                f"Expected {len(split_df)}, "
                f"got {len(fused_df)}"
            )

        # ------------------------------------------
        # Validate target
        # ------------------------------------------

        if not fused_df["Target"].isin(
            [0, 1]
        ).all():

            raise ValueError(
                f"{name} contains invalid "
                "target values after fusion"
            )

        # ------------------------------------------
        # Validate NaN / Inf
        # ------------------------------------------

        numeric_columns = (
            fused_df
            .select_dtypes(
                include=[np.number]
            )
            .columns
        )

        if not np.isfinite(
            fused_df[numeric_columns]
            .to_numpy()
        ).all():

            raise ValueError(
                f"{name} fused features contain "
                "NaN or infinite values"
            )

        # ------------------------------------------
        # Sort by ecg_id for deterministic output
        # ------------------------------------------

        fused_df = fused_df.sort_values(
            "ecg_id"
        ).reset_index(
            drop=True
        )

        return fused_df

    # ==========================================
    # 7. Main method
    # ==========================================

    def initiate_feature_fusion(
        self,
        train_ecg_features,
        validation_ecg_features,
        test_ecg_features,
        ecg_feature_names,
        train_metadata_df,
        validation_metadata_df,
        test_metadata_df,
        train_df,
        validation_df,
        test_df
    ):

        try:

            logging.info(
                "Starting feature fusion"
            )

            # ==========================================
            # Fuse TRAIN
            # ==========================================

            train_fused = self._fuse_split(
                train_ecg_features,
                train_metadata_df,
                train_df,
                ecg_feature_names,
                "Train"
            )

            # ==========================================
            # Fuse VALIDATION
            # ==========================================

            validation_fused = self._fuse_split(
                validation_ecg_features,
                validation_metadata_df,
                validation_df,
                ecg_feature_names,
                "Validation"
            )

            # ==========================================
            # Fuse TEST
            # ==========================================

            test_fused = self._fuse_split(
                test_ecg_features,
                test_metadata_df,
                test_df,
                ecg_feature_names,
                "Test"
            )

            # ==========================================
            # Logging
            # ==========================================

            logging.info(
                f"Train fused shape: "
                f"{train_fused.shape}"
            )

            logging.info(
                f"Validation fused shape: "
                f"{validation_fused.shape}"
            )

            logging.info(
                f"Test fused shape: "
                f"{test_fused.shape}"
            )

            logging.info(
                "Feature fusion completed "
                "successfully"
            )

            return (
                train_fused,
                validation_fused,
                test_fused
            )

        except Exception as e:

            logging.exception(
                "Error occurred during "
                "feature fusion"
            )

            raise HealthPulseException(
                e,
                sys
            )