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

from HealthPulse_AI_project.components.metadata_preprocessing import (
    MetaDataPreprocessing
)

from HealthPulse_AI_project.components.metadata_feature_engineering import (
    MetadataFeatureEngineering
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
    # 5. Metadata preprocessing
    # ==========================================

    metadata_preprocessing = (
        MetaDataPreprocessing()
    )

    (
        X_train_processed,
        X_validation_processed,
        X_test_processed,
        feature_names
    ) = metadata_preprocessing.initiate_metadata_preprocessing(
        train_df,
        validation_df,
        test_df
    )

    # ==========================================
    # 6. Get imputed DataFrames
    # ==========================================

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
    # 7. Validate imputed DataFrames
    # ==========================================

    if train_imputed_df is None:

        raise ValueError(
            "Train imputed dataframe is None"
        )

    if validation_imputed_df is None:

        raise ValueError(
            "Validation imputed dataframe is None"
        )

    if test_imputed_df is None:

        raise ValueError(
            "Test imputed dataframe is None"
        )

    # ==========================================
    # 8. Feature engineering
    # ==========================================

    feature_engineering = (
        MetadataFeatureEngineering()
    )

    (
        train_features,
        validation_features,
        test_features
    ) = feature_engineering.initiate_feature_engineering(
        train_imputed_df,
        validation_imputed_df,
        test_imputed_df
    )

    # ==========================================
    # 9. Display results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("METADATA FEATURE ENGINEERING RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"Shape       : "
        f"{train_features.shape}"
    )

    print(
        f"BMI missing : "
        f"{train_features['BMI'].isna().sum()}"
    )

    print(
        f"BMI mean    : "
        f"{train_features['BMI'].mean():.2f}"
    )

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"Shape       : "
        f"{validation_features.shape}"
    )

    print(
        f"BMI missing : "
        f"{validation_features['BMI'].isna().sum()}"
    )

    print(
        f"BMI mean    : "
        f"{validation_features['BMI'].mean():.2f}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"Shape       : "
        f"{test_features.shape}"
    )

    print(
        f"BMI missing : "
        f"{test_features['BMI'].isna().sum()}"
    )

    print(
        f"BMI mean    : "
        f"{test_features['BMI'].mean():.2f}"
    )

    # ==========================================
    # 10. BMI sample
    # ==========================================

    print("\nBMI SAMPLE")
    print("-" * 60)

    print(
        train_features[
            [
                "height",
                "weight",
                "BMI"
            ]
        ].head(10)
    )

    # ==========================================
    # 11. Validation checks
    # ==========================================

    bmi_exists = (
        "BMI" in train_features.columns
        and
        "BMI" in validation_features.columns
        and
        "BMI" in test_features.columns
    )

    bmi_no_missing = (
        train_features["BMI"].notna().all()
        and
        validation_features["BMI"].notna().all()
        and
        test_features["BMI"].notna().all()
    )

    bmi_no_infinity = (
        np.isfinite(
            train_features["BMI"]
        ).all()
        and
        np.isfinite(
            validation_features["BMI"]
        ).all()
        and
        np.isfinite(
            test_features["BMI"]
        ).all()
    )

    target_preserved = (
        train_features["Target"].isin([0, 1]).all()
        and
        validation_features["Target"].isin([0, 1]).all()
        and
        test_features["Target"].isin([0, 1]).all()
    )

    row_count_validation = (
        len(train_features)
        == len(train_df)
        and
        len(validation_features)
        == len(validation_df)
        and
        len(test_features)
        == len(test_df)
    )

    # ==========================================
    # 12. Height / Weight validation
    # ==========================================

    height_weight_validation = (
        train_imputed_df[
            ["height", "weight"]
        ].notna().all().all()
        and
        validation_imputed_df[
            ["height", "weight"]
        ].notna().all().all()
        and
        test_imputed_df[
            ["height", "weight"]
        ].notna().all().all()
    )

    # ==========================================
    # 13. Final validation
    # ==========================================

    overall_validation = (
        bmi_exists
        and
        bmi_no_missing
        and
        bmi_no_infinity
        and
        target_preserved
        and
        row_count_validation
        and
        height_weight_validation
    )

    print("\n")
    print("=" * 60)
    print("METADATA FEATURE ENGINEERING VALIDATION")
    print("=" * 60)

    print(
        f"BMI feature validation : "
        f"{'PASS' if bmi_exists else 'FAIL'}"
    )

    print(
        f"BMI missing check      : "
        f"{'PASS' if bmi_no_missing else 'FAIL'}"
    )

    print(
        f"BMI infinity check     : "
        f"{'PASS' if bmi_no_infinity else 'FAIL'}"
    )

    print(
        f"Target preservation    : "
        f"{'PASS' if target_preserved else 'FAIL'}"
    )

    print(
        f"Row count validation   : "
        f"{'PASS' if row_count_validation else 'FAIL'}"
    )

    print(
        f"Height/Weight check    : "
        f"{'PASS' if height_weight_validation else 'FAIL'}"
    )

    print(
        f"\nMetadata FE validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)