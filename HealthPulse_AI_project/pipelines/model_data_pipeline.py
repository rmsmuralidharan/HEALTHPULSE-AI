import os
import sys
from dataclasses import dataclass

import numpy as np

from HealthPulse_AI_project.components.data_ingestion import (
    DataIngestion
)

from HealthPulse_AI_project.components.target_creation import (
    TargetCreation
)

from HealthPulse_AI_project.components.data_splitting import (
    DataSplitting
)

from HealthPulse_AI_project.components.data_preprocessing import (
    DataPreprocessing
)

from HealthPulse_AI_project.components.metadata_preprocessing import (
    MetaDataPreprocessing
)

from HealthPulse_AI_project.components.ecg_feature_engineering import (
    ECGFeatureEngineering
)

from HealthPulse_AI_project.components.feature_fusion import (
    FeatureFusion
)

from HealthPulse_AI_project.components.model_preparation import (
    ModelPreparation
)

from HealthPulse_AI_project.exception.exception import (
    HealthPulseException
)

from HealthPulse_AI_project.logging.logger import logging


# ==========================================================
# DATA CONTAINER
# ==========================================================

@dataclass
class ModelData:

    # ------------------------------------------------------
    # Raw ECG
    # ------------------------------------------------------

    X_train_ecg: np.ndarray
    X_validation_ecg: np.ndarray
    X_test_ecg: np.ndarray

    # ------------------------------------------------------
    # ECG engineered features
    # ------------------------------------------------------

    X_train_ecg_features: np.ndarray
    X_validation_ecg_features: np.ndarray
    X_test_ecg_features: np.ndarray

    ecg_feature_names: np.ndarray

    # ------------------------------------------------------
    # Metadata processed features
    # ------------------------------------------------------

    X_train_metadata: np.ndarray
    X_validation_metadata: np.ndarray
    X_test_metadata: np.ndarray

    metadata_feature_names: np.ndarray

    # ------------------------------------------------------
    # Fused features
    # ------------------------------------------------------

    X_train_fused: np.ndarray
    X_validation_fused: np.ndarray
    X_test_fused: np.ndarray

    fused_feature_names: np.ndarray

    # ------------------------------------------------------
    # Model-ready data
    # ------------------------------------------------------

    X_train: np.ndarray
    y_train: np.ndarray

    X_validation: np.ndarray
    y_validation: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray

    # ------------------------------------------------------
    # ECG identifiers
    # ------------------------------------------------------

    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray


# ==========================================================
# MODEL DATA PIPELINE
# ==========================================================

class ModelDataPipeline:

    def __init__(
        self,
        data_path=os.path.join(
            "data",
            "raw",
            "ptbxl"
        ),
        random_state=42
    ):

        self.data_path = data_path
        self.random_state = random_state

    # ======================================================
    # 1. DATA INGESTION
    # ======================================================

    def _run_data_ingestion(self):

        logging.info(
            "Pipeline step 1: Data ingestion"
        )

        ingestion = DataIngestion(
            data_path=self.data_path
        )

        (
            database_df,
            scp_df
        ) = ingestion.initiate_data_ingestion()

        return (
            database_df,
            scp_df
        )

    # ======================================================
    # 2. TARGET CREATION
    # ======================================================

    def _run_target_creation(
        self,
        database_df,
        scp_df
    ):

        logging.info(
            "Pipeline step 2: Target creation"
        )

        target_creator = TargetCreation()

        target_df = (
            target_creator
            .initiate_target_creation(
                database_df,
                scp_df
            )
        )

        return target_df

    # ======================================================
    # 3. PATIENT-LEVEL SPLITTING
    # ======================================================

    def _run_data_splitting(
        self,
        target_df
    ):

        logging.info(
            "Pipeline step 3: Patient-level splitting"
        )

        splitter = DataSplitting(
            train_size=0.70,
            validation_size=0.15,
            test_size=0.15,
            random_state=self.random_state
        )

        (
            train_df,
            validation_df,
            test_df
        ) = splitter.initiate_data_splitting(
            target_df
        )

        return (
            train_df,
            validation_df,
            test_df
        )

    # ======================================================
    # 4. ECG PREPROCESSING
    # ======================================================

    def _run_ecg_preprocessing(
        self,
        train_df,
        validation_df,
        test_df
    ):

        logging.info(
            "Pipeline step 4: ECG preprocessing"
        )

        preprocessing = DataPreprocessing(
            data_path=self.data_path
        )

        (
            X_train_ecg,
            y_train,
            X_validation_ecg,
            y_validation,
            X_test_ecg,
            y_test
        ) = (
            preprocessing
            .initiate_data_preprocessing(
                train_df,
                validation_df,
                test_df
            )
        )

        return (
            X_train_ecg,
            y_train,
            X_validation_ecg,
            y_validation,
            X_test_ecg,
            y_test
        )

    # ======================================================
    # 5. ECG FEATURE ENGINEERING
    # ======================================================

    def _run_ecg_feature_engineering(
        self,
        X_train_ecg,
        X_validation_ecg,
        X_test_ecg
    ):

        logging.info(
            "Pipeline step 5: ECG feature engineering"
        )

        ecg_fe = ECGFeatureEngineering()

        (
            train_ecg_features,
            validation_ecg_features,
            test_ecg_features,
            ecg_feature_names,
            _,
            _,
            _
        ) = (
            ecg_fe
            .initiate_ecg_feature_engineering(
                X_train_ecg,
                X_validation_ecg,
                X_test_ecg
            )
        )

        return (
            train_ecg_features,
            validation_ecg_features,
            test_ecg_features,
            ecg_feature_names
        )

    # ======================================================
    # 6. METADATA PREPROCESSING
    # ======================================================

    def _run_metadata_preprocessing(
        self,
        train_df,
        validation_df,
        test_df
    ):

        logging.info(
            "Pipeline step 6: Metadata preprocessing"
        )

        metadata = MetaDataPreprocessing()

        (
            X_train_metadata,
            X_validation_metadata,
            X_test_metadata,
            metadata_feature_names
        ) = (
            metadata
            .initiate_metadata_preprocessing(
                train_df,
                validation_df,
                test_df
            )
        )

        return (
            X_train_metadata,
            X_validation_metadata,
            X_test_metadata,
            metadata_feature_names
        )

    # ======================================================
    # 7. FEATURE FUSION
    # ======================================================

    def _run_feature_fusion(
        self,
        train_ecg_features,
        validation_ecg_features,
        test_ecg_features,
        X_train_metadata,
        X_validation_metadata,
        X_test_metadata,
        y_train,
        y_validation,
        y_test,
        train_ids,
        validation_ids,
        test_ids
    ):

        logging.info(
            "Pipeline step 7: Feature fusion"
        )

        fusion = FeatureFusion()

        (
            train_fused,
            validation_fused,
            test_fused,
            train_target,
            validation_target,
            test_target,
            train_ids,
            validation_ids,
            test_ids
        ) = fusion.initiate_feature_fusion(

            train_ecg_features,
            validation_ecg_features,
            test_ecg_features,

            X_train_metadata,
            X_validation_metadata,
            X_test_metadata,

            y_train,
            y_validation,
            y_test,

            train_ids,
            validation_ids,
            test_ids
        )

        # --------------------------------------------------
        # Feature names
        # --------------------------------------------------

        fused_feature_names = np.concatenate(
            [
                np.asarray(
                    self._to_feature_names(
                        train_ecg_features,
                        "ecg"
                    )
                ),

                np.asarray(
                    self._to_feature_names(
                        X_train_metadata,
                        "metadata"
                    )
                )
            ]
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
            test_ids,
            fused_feature_names
        )

    # ======================================================
    # 8. MODEL PREPARATION
    # ======================================================

    def _run_model_preparation(
        self,
        train_fused,
        train_target,
        validation_fused,
        validation_target,
        test_fused,
        test_target
    ):

        logging.info(
            "Pipeline step 8: Model preparation"
        )

        preparation = ModelPreparation()

        (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test
        ) = (
            preparation
            .initiate_model_preparation(

                train_fused,
                train_target,

                validation_fused,
                validation_target,

                test_fused,
                test_target
            )
        )

        return (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test
        )

    # ======================================================
    # FEATURE NAME HELPER
    # ======================================================

    def _to_feature_names(
        self,
        features,
        prefix
    ):

        if hasattr(
            features,
            "columns"
        ):

            return list(
                features.columns
            )

        return [
            f"{prefix}_{index}"
            for index in range(
                features.shape[1]
            )
        ]

    # ======================================================
    # MAIN PIPELINE
    # ======================================================

    def initiate_model_data_pipeline(
        self
    ):

        try:

            logging.info(
                "=" * 60
            )

            logging.info(
                "STARTING MODEL DATA PIPELINE"
            )

            logging.info(
                "=" * 60
            )

            # ==================================================
            # STEP 1
            # ==================================================

            (
                database_df,
                scp_df
            ) = self._run_data_ingestion()

            # ==================================================
            # STEP 2
            # ==================================================

            target_df = (
                self._run_target_creation(
                    database_df,
                    scp_df
                )
            )

            # ==================================================
            # STEP 3
            # ==================================================

            (
                train_df,
                validation_df,
                test_df
            ) = self._run_data_splitting(
                target_df
            )

            # ==================================================
            # STEP 4
            # ==================================================

            (
                X_train_ecg,
                y_train,
                X_validation_ecg,
                y_validation,
                X_test_ecg,
                y_test
            ) = self._run_ecg_preprocessing(

                train_df,
                validation_df,
                test_df
            )

            # ==================================================
            # STEP 5
            # ==================================================

            (
                train_ecg_features,
                validation_ecg_features,
                test_ecg_features,
                ecg_feature_names
            ) = (
                self._run_ecg_feature_engineering(

                    X_train_ecg,
                    X_validation_ecg,
                    X_test_ecg
                )
            )

            # ==================================================
            # STEP 6
            # ==================================================

            (
                X_train_metadata,
                X_validation_metadata,
                X_test_metadata,
                metadata_feature_names
            ) = (
                self._run_metadata_preprocessing(

                    train_df,
                    validation_df,
                    test_df
                )
            )

            # ==================================================
            # STEP 7
            # ==================================================

            (
                train_fused,
                validation_fused,
                test_fused,
                train_target,
                validation_target,
                test_target,
                train_ids,
                validation_ids,
                test_ids,
                fused_feature_names
            ) = self._run_feature_fusion(

                train_ecg_features,
                validation_ecg_features,
                test_ecg_features,

                X_train_metadata,
                X_validation_metadata,
                X_test_metadata,

                y_train,
                y_validation,
                y_test,

                train_df[
                    "ecg_id"
                ].to_numpy(),

                validation_df[
                    "ecg_id"
                ].to_numpy(),

                test_df[
                    "ecg_id"
                ].to_numpy()
            )

            # ==================================================
            # STEP 8
            # ==================================================

            (
                X_train,
                y_train,
                X_validation,
                y_validation,
                X_test,
                y_test
            ) = self._run_model_preparation(

                train_fused,
                train_target,

                validation_fused,
                validation_target,

                test_fused,
                test_target
            )

            # ==================================================
            # FINAL VALIDATION
            # ==================================================

            self._validate_final_data(

                X_train,
                y_train,

                X_validation,
                y_validation,

                X_test,
                y_test,

                X_train_ecg,
                X_validation_ecg,
                X_test_ecg
            )

            logging.info(
                "=" * 60
            )

            logging.info(
                "MODEL DATA PIPELINE COMPLETED"
            )

            logging.info(
                "=" * 60
            )

            # ==================================================
            # RETURN EVERYTHING
            # ==================================================

            return ModelData(

                # Raw ECG
                X_train_ecg=X_train_ecg,
                X_validation_ecg=X_validation_ecg,
                X_test_ecg=X_test_ecg,

                # ECG features
                X_train_ecg_features=train_ecg_features,
                X_validation_ecg_features=validation_ecg_features,
                X_test_ecg_features=test_ecg_features,

                ecg_feature_names=np.asarray(
                    ecg_feature_names
                ),

                # Metadata
                X_train_metadata=X_train_metadata,
                X_validation_metadata=X_validation_metadata,
                X_test_metadata=X_test_metadata,

                metadata_feature_names=np.asarray(
                    metadata_feature_names
                ),

                # Fusion
                X_train_fused=train_fused,
                X_validation_fused=validation_fused,
                X_test_fused=test_fused,

                fused_feature_names=np.asarray(
                    fused_feature_names
                ),

                # Model-ready
                X_train=X_train,
                y_train=y_train,

                X_validation=X_validation,
                y_validation=y_validation,

                X_test=X_test,
                y_test=y_test,

                # IDs
                train_ids=train_ids,
                validation_ids=validation_ids,
                test_ids=test_ids
            )

        except Exception as e:

            logging.exception(
                "Error occurred during "
                "model data pipeline"
            )

            raise HealthPulseException(
                e,
                sys
            )

    # ======================================================
    # FINAL VALIDATION
    # ======================================================

    def _validate_final_data(
        self,

        X_train,
        y_train,

        X_validation,
        y_validation,

        X_test,
        y_test,

        X_train_ecg,
        X_validation_ecg,
        X_test_ecg
    ):

        # --------------------------------------------------
        # Fused/model features
        # --------------------------------------------------

        for (
            X,
            y,
            name
        ) in [

            (
                X_train,
                y_train,
                "Train"
            ),

            (
                X_validation,
                y_validation,
                "Validation"
            ),

            (
                X_test,
                y_test,
                "Test"
            )
        ]:

            if X.ndim != 2:

                raise ValueError(
                    f"{name} features "
                    "must be 2D"
                )

            if X.shape[1] != 172:

                raise ValueError(
                    f"{name} expected "
                    "172 features"
                )

            if X.shape[0] != len(y):

                raise ValueError(
                    f"{name} X/y "
                    "row mismatch"
                )

            if not np.isfinite(X).all():

                raise ValueError(
                    f"{name} contains "
                    "NaN or Inf"
                )

        # --------------------------------------------------
        # Raw ECG
        # --------------------------------------------------

        for (
            X,
            name
        ) in [

            (
                X_train_ecg,
                "Train ECG"
            ),

            (
                X_validation_ecg,
                "Validation ECG"
            ),

            (
                X_test_ecg,
                "Test ECG"
            )
        ]:

            if X.ndim != 3:

                raise ValueError(
                    f"{name} must be 3D"
                )

            if X.shape[1:] != (
                1000,
                12
            ):

                raise ValueError(
                    f"{name} expected "
                    "(N, 1000, 12)"
                )

        logging.info(
            "Final pipeline validation passed"
        )