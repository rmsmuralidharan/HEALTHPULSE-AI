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

from HealthPulse_AI_project.components.metadata_feature_engineering import (
    MetadataFeatureEngineering
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
    # 6. ECG feature engineering
    # ==========================================

    ecg_feature_engineering = (
        ECGFeatureEngineering()
    )

    (
        train_ecg_features,
        validation_ecg_features,
        test_ecg_features,
        ecg_feature_names,
        _,
        _,
        _
    ) = (
        ecg_feature_engineering
        .initiate_ecg_feature_engineering(
            X_train,
            X_validation,
            X_test
        )
    )

    # ==========================================
    # 7. Metadata preprocessing
    # ==========================================

    metadata_preprocessing = (
        MetaDataPreprocessing()
    )

    (
        _,
        _,
        _,
        _
    ) = metadata_preprocessing.initiate_metadata_preprocessing(
        train_df,
        validation_df,
        test_df
    )

    # Get imputed DataFrames

    train_imputed_df = (
        metadata_preprocessing.train_imputed_df
    )

    validation_imputed_df = (
        metadata_preprocessing.validation_imputed_df
    )

    test_imputed_df = (
        metadata_preprocessing.test_imputed_df
    )

    # ==========================================
    # 8. Metadata feature engineering
    # ==========================================

    metadata_feature_engineering = (
        MetadataFeatureEngineering()
    )

    (
        train_metadata_features,
        validation_metadata_features,
        test_metadata_features
    ) = (
        metadata_feature_engineering
        .initiate_feature_engineering(
            train_imputed_df,
            validation_imputed_df,
            test_imputed_df
        )
    )

    # ==========================================
    # 9. Feature Fusion
    # ==========================================

    fusion = FeatureFusion()

    (
        train_fused,
        validation_fused,
        test_fused
    ) = fusion.initiate_feature_fusion(
        train_ecg_features,
        validation_ecg_features,
        test_ecg_features,
        ecg_feature_names,
        train_metadata_features,
        validation_metadata_features,
        test_metadata_features,
        train_df,
        validation_df,
        test_df
    )

    # ==========================================
    # 10. Display results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("FEATURE FUSION RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"Shape : {train_fused.shape}"
    )

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"Shape : {validation_fused.shape}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"Shape : {test_fused.shape}"
    )

    # ==========================================
    # 11. Display columns
    # ==========================================

    print("\nFEATURE INFORMATION")
    print("-" * 60)

    print(
        f"Total columns : "
        f"{len(train_fused.columns)}"
    )

    print(
        f"ECG features  : "
        f"{len(ecg_feature_names)}"
    )

    print("\nFirst 20 columns:")

    print(
        train_fused.columns[:20].tolist()
    )

    # ==========================================
    # 12. Target distribution
    # ==========================================

    print("\nTARGET DISTRIBUTION")
    print("-" * 60)

    print(
        train_fused["Target"]
        .value_counts()
        .sort_index()
    )

    # ==========================================
    # 13. Validation checks
    # ==========================================

    # ------------------------------------------
    # Shape validation
    # ------------------------------------------

    shape_validation = (
        len(train_fused) == len(train_df)
        and
        len(validation_fused)
        == len(validation_df)
        and
        len(test_fused)
        == len(test_df)
    )

    # ------------------------------------------
    # ECG feature count
    # ------------------------------------------

    ecg_feature_validation = all(
        feature in train_fused.columns
        for feature in ecg_feature_names
    )

    # ------------------------------------------
    # BMI validation
    # ------------------------------------------

    bmi_validation = (
        "BMI" in train_fused.columns
        and
        "BMI" in validation_fused.columns
        and
        "BMI" in test_fused.columns
    )

    # ------------------------------------------
    # Target validation
    # ------------------------------------------

    target_validation = (
        "Target" in train_fused.columns
        and
        "Target" in validation_fused.columns
        and
        "Target" in test_fused.columns
        and
        train_fused["Target"].isin([0, 1]).all()
        and
        validation_fused["Target"].isin([0, 1]).all()
        and
        test_fused["Target"].isin([0, 1]).all()
    )

    # ------------------------------------------
    # ECG ID validation
    # ------------------------------------------

    ecg_id_validation = (
        "ecg_id" in train_fused.columns
        and
        "ecg_id" in validation_fused.columns
        and
        "ecg_id" in test_fused.columns
        and
        train_fused["ecg_id"].is_unique
        and
        validation_fused["ecg_id"].is_unique
        and
        test_fused["ecg_id"].is_unique
    )

    # ------------------------------------------
    # No NaN / Inf
    # ------------------------------------------

    numeric_train = (
        train_fused
        .select_dtypes(include=[np.number])
    )

    numeric_validation = (
        validation_fused
        .select_dtypes(include=[np.number])
    )

    numeric_test = (
        test_fused
        .select_dtypes(include=[np.number])
    )

    finite_validation = (
        np.isfinite(
            numeric_train.to_numpy()
        ).all()
        and
        np.isfinite(
            numeric_validation.to_numpy()
        ).all()
        and
        np.isfinite(
            numeric_test.to_numpy()
        ).all()
    )

    # ------------------------------------------
    # No target leakage through scp_codes
    # ------------------------------------------

    leakage_validation = (
        "scp_codes"
        not in train_fused.columns
        and
        "scp_codes"
        not in validation_fused.columns
        and
        "scp_codes"
        not in test_fused.columns
    )

    # ------------------------------------------
    # Row order / record preservation
    # ------------------------------------------

    record_validation = (
        set(train_fused["ecg_id"])
        == set(train_df["ecg_id"])
        and
        set(validation_fused["ecg_id"])
        == set(validation_df["ecg_id"])
        and
        set(test_fused["ecg_id"])
        == set(test_df["ecg_id"])
    )

    # ==========================================
    # 14. Final validation
    # ==========================================

    overall_validation = (
        shape_validation
        and
        ecg_feature_validation
        and
        bmi_validation
        and
        target_validation
        and
        ecg_id_validation
        and
        finite_validation
        and
        leakage_validation
        and
        record_validation
    )

    # ==========================================
    # 15. Display validation
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
        f"ECG feature validation  : "
        f"{'PASS' if ecg_feature_validation else 'FAIL'}"
    )

    print(
        f"BMI validation          : "
        f"{'PASS' if bmi_validation else 'FAIL'}"
    )

    print(
        f"Target validation       : "
        f"{'PASS' if target_validation else 'FAIL'}"
    )

    print(
        f"ECG ID validation       : "
        f"{'PASS' if ecg_id_validation else 'FAIL'}"
    )

    print(
        f"NaN / Inf validation    : "
        f"{'PASS' if finite_validation else 'FAIL'}"
    )

    print(
        f"Leakage validation      : "
        f"{'PASS' if leakage_validation else 'FAIL'}"
    )

    print(
        f"Record preservation     : "
        f"{'PASS' if record_validation else 'FAIL'}"
    )

    print(
        f"\nFeature Fusion validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)