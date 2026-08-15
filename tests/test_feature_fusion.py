import os
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


if __name__ == "__main__":

    # ==========================================
    # 1. PTB-XL path
    # ==========================================

    data_path = os.path.join(
        "data",
        "raw",
        "ptbxl"
    )

    # ==========================================
    # 2. Data ingestion
    # ==========================================

    ingestion = DataIngestion(
        data_path=data_path
    )

    database_df, scp_df = (
        ingestion.initiate_data_ingestion()
    )

    # ==========================================
    # 3. Target creation
    # ==========================================

    target_creator = TargetCreation()

    target_df = (
        target_creator.initiate_target_creation(
            database_df,
            scp_df
        )
    )

    # ==========================================
    # 4. Patient-level splitting
    # ==========================================

    splitter = DataSplitting(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_state=42
    )

    (
        train_df,
        validation_df,
        test_df
    ) = splitter.initiate_data_splitting(
        target_df
    )

    # ==========================================
    # 5. ECG preprocessing
    # ==========================================

    preprocessing = DataPreprocessing(
        data_path=data_path
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = preprocessing.initiate_data_preprocessing(
        train_df,
        validation_df,
        test_df
    )

    # ==========================================
    # 6. ECG Feature Engineering
    # ==========================================

    ecg_fe = ECGFeatureEngineering()

    (
        train_ecg_features,
        validation_ecg_features,
        test_ecg_features,
        ecg_feature_names,
        _,
        _,
        _
    ) = ecg_fe.initiate_ecg_feature_engineering(
        X_train,
        X_validation,
        X_test
    )

    # ==========================================
    # 7. Metadata preprocessing
    # ==========================================

    metadata_preprocessing = (
        MetaDataPreprocessing()
    )

    (
        X_train_metadata,
        X_validation_metadata,
        X_test_metadata,
        metadata_feature_names
    ) = (
        metadata_preprocessing
        .initiate_metadata_preprocessing(
            train_df,
            validation_df,
            test_df
        )
    )

    # ==========================================
    # 8. ECG IDs
    # ==========================================

    train_ecg_ids = (
        train_df["ecg_id"].to_numpy()
    )

    validation_ecg_ids = (
        validation_df["ecg_id"].to_numpy()
    )

    test_ecg_ids = (
        test_df["ecg_id"].to_numpy()
    )

    # ==========================================
    # 9. Feature Fusion
    # ==========================================

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

        train_ecg_ids,
        validation_ecg_ids,
        test_ecg_ids
    )

    # ==========================================
    # 10. Results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("FEATURE FUSION RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"ECG features      : "
        f"{train_ecg_features.shape}"
    )

    print(
        f"Metadata features : "
        f"{X_train_metadata.shape}"
    )

    print(
        f"Fused features    : "
        f"{train_fused.shape}"
    )

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"ECG features      : "
        f"{validation_ecg_features.shape}"
    )

    print(
        f"Metadata features : "
        f"{X_validation_metadata.shape}"
    )

    print(
        f"Fused features    : "
        f"{validation_fused.shape}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"ECG features      : "
        f"{test_ecg_features.shape}"
    )

    print(
        f"Metadata features : "
        f"{X_test_metadata.shape}"
    )

    print(
        f"Fused features    : "
        f"{test_fused.shape}"
    )

    # ==========================================
    # 11. Target distribution
    # ==========================================

    print("\nTARGET DISTRIBUTION")
    print("-" * 60)

    print(
        "TRAIN:"
    )

    print(
        np.unique(
            train_target,
            return_counts=True
        )
    )

    print(
        "VALIDATION:"
    )

    print(
        np.unique(
            validation_target,
            return_counts=True
        )
    )

    print(
        "TEST:"
    )

    print(
        np.unique(
            test_target,
            return_counts=True
        )
    )

    # ==========================================
    # 12. Validation checks
    # ==========================================

    shape_validation = (
        train_fused.shape
        == (15270, 172)
        and
        validation_fused.shape
        == (3251, 172)
        and
        test_fused.shape
        == (3278, 172)
    )

    feature_count_validation = (
        train_ecg_features.shape[1] == 84
        and
        X_train_metadata.shape[1] == 88
        and
        train_fused.shape[1] == 172
    )

    row_count_validation = (
        train_fused.shape[0]
        == len(y_train)
        == len(train_ids)

        and
        validation_fused.shape[0]
        == len(y_validation)
        == len(validation_ids)

        and
        test_fused.shape[0]
        == len(y_test)
        == len(test_ids)
    )

    finite_validation = (
        np.isfinite(
            train_fused
        ).all()
        and
        np.isfinite(
            validation_fused
        ).all()
        and
        np.isfinite(
            test_fused
        ).all()
    )

    target_validation = (
        np.isin(
            train_target,
            [0, 1]
        ).all()
        and
        np.isin(
            validation_target,
            [0, 1]
        ).all()
        and
        np.isin(
            test_target,
            [0, 1]
        ).all()
    )

    id_validation = (
        np.array_equal(
            np.sort(train_ids),
            np.sort(
                train_df["ecg_id"].to_numpy()
            )
        )
        and
        np.array_equal(
            np.sort(validation_ids),
            np.sort(
                validation_df["ecg_id"].to_numpy()
            )
        )
        and
        np.array_equal(
            np.sort(test_ids),
            np.sort(
                test_df["ecg_id"].to_numpy()
            )
        )
    )

    feature_name_validation = (
        len(ecg_feature_names) == 84
        and
        len(metadata_feature_names) == 88
    )

    # ==========================================
    # 13. Final validation
    # ==========================================

    overall_validation = (
        shape_validation
        and
        feature_count_validation
        and
        row_count_validation
        and
        finite_validation
        and
        target_validation
        and
        id_validation
        and
        feature_name_validation
    )

    # ==========================================
    # 14. Validation report
    # ==========================================

    print("\n")
    print("=" * 60)
    print("FEATURE FUSION VALIDATION")
    print("=" * 60)

    print(
        f"Shape validation        : "
        f"{'PASS' if shape_validation else 'FAIL'}"
    )

    print(
        f"Feature count validation: "
        f"{'PASS' if feature_count_validation else 'FAIL'}"
    )

    print(
        f"Row count validation    : "
        f"{'PASS' if row_count_validation else 'FAIL'}"
    )

    print(
        f"NaN / Inf validation    : "
        f"{'PASS' if finite_validation else 'FAIL'}"
    )

    print(
        f"Target validation       : "
        f"{'PASS' if target_validation else 'FAIL'}"
    )

    print(
        f"ECG ID validation       : "
        f"{'PASS' if id_validation else 'FAIL'}"
    )

    print(
        f"Feature name validation : "
        f"{'PASS' if feature_name_validation else 'FAIL'}"
    )

    print(
        f"\nFeature Fusion validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)