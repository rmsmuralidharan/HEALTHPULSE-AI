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

    preprocessing = MetaDataPreprocessing()

    (
        X_train_processed,
        X_validation_processed,
        X_test_processed,
        feature_names
    ) = preprocessing.initiate_metadata_preprocessing(
        train_df,
        validation_df,
        test_df
    )

    # ==========================================
    # 6. Get imputed DataFrames
    # ==========================================

    train_imputed_df = (
        preprocessing.train_imputed_df
    )

    validation_imputed_df = (
        preprocessing.validation_imputed_df
    )

    test_imputed_df = (
        preprocessing.test_imputed_df
    )

    # ==========================================
    # 7. Display results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("METADATA PREPROCESSING RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"Processed shape : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Patients        : "
        f"{train_imputed_df['patient_id'].nunique()}"
    )

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"Processed shape : "
        f"{X_validation_processed.shape}"
    )

    print(
        f"Patients        : "
        f"{validation_imputed_df['patient_id'].nunique()}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"Processed shape : "
        f"{X_test_processed.shape}"
    )

    print(
        f"Patients        : "
        f"{test_imputed_df['patient_id'].nunique()}"
    )

    # ==========================================
    # 8. Feature information
    # ==========================================

    print("\nFEATURES")
    print("-" * 60)

    print(
        f"Number of features: "
        f"{len(feature_names)}"
    )

    print(feature_names)

    # ==========================================
    # 9. Imputation validation
    # ==========================================

    metadata_columns = [
        "age",
        "sex",
        "height",
        "weight",
        "site",
        "nurse",
        "heart_axis",
        "device",
        "second_opinion"
    ]

    train_missing = (
        train_imputed_df[
            metadata_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    validation_missing = (
        validation_imputed_df[
            metadata_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    test_missing = (
        test_imputed_df[
            metadata_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    imputation_validation = (
        train_missing == 0
        and
        validation_missing == 0
        and
        test_missing == 0
    )

    # ==========================================
    # 10. Shape validation
    # ==========================================

    shape_validation = (
        X_train_processed.shape[0]
        == len(train_df)
        and
        X_validation_processed.shape[0]
        == len(validation_df)
        and
        X_test_processed.shape[0]
        == len(test_df)
    )

    # ==========================================
    # 11. Feature count validation
    # ==========================================

    feature_count_validation = (
        X_train_processed.shape[1]
        == len(feature_names)
        and
        X_validation_processed.shape[1]
        == len(feature_names)
        and
        X_test_processed.shape[1]
        == len(feature_names)
    )

    # ==========================================
    # 12. NaN / Inf validation
    # ==========================================

    finite_validation = (
        np.isfinite(
            X_train_processed
        ).all()
        and
        np.isfinite(
            X_validation_processed
        ).all()
        and
        np.isfinite(
            X_test_processed
        ).all()
    )

    # ==========================================
    # 13. Transformer validation
    # ==========================================

    transformer_exists = os.path.exists(
        preprocessing.transformer_path
    )

    # ==========================================
    # 14. Final validation
    # ==========================================

    overall_validation = (
        shape_validation
        and
        feature_count_validation
        and
        finite_validation
        and
        imputation_validation
        and
        transformer_exists
    )

    print("\n")
    print("=" * 60)
    print("METADATA PREPROCESSING VALIDATION")
    print("=" * 60)

    print(
        f"Shape validation       : "
        f"{'PASS' if shape_validation else 'FAIL'}"
    )

    print(
        f"Feature count check    : "
        f"{'PASS' if feature_count_validation else 'FAIL'}"
    )

    print(
        f"Missing-value check    : "
        f"{'PASS' if imputation_validation else 'FAIL'}"
    )

    print(
        f"NaN / Inf check        : "
        f"{'PASS' if finite_validation else 'FAIL'}"
    )

    print(
        f"Transformer saved      : "
        f"{'PASS' if transformer_exists else 'FAIL'}"
    )

    print(
        f"\nMetadata preprocessing validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)