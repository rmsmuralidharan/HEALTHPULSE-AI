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
    # 1. Dataset path
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
    # 4. Patient-level split
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
        X_train_metadata,
        X_validation_metadata,
        X_test_metadata,
        feature_names
    ) = preprocessing.initiate_metadata_preprocessing(
        train_df,
        validation_df,
        test_df
    )

    # ==========================================
    # 6. Display results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("METADATA PREPROCESSING RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"Shape : "
        f"{X_train_metadata.shape}"
    )

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"Shape : "
        f"{X_validation_metadata.shape}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"Shape : "
        f"{X_test_metadata.shape}"
    )

    # ==========================================
    # 7. Feature names
    # ==========================================

    print("\nFEATURES")
    print("-" * 60)

    print(
        f"Number of features: "
        f"{len(feature_names)}"
    )

    print(
        feature_names
    )

    # ==========================================
    # 8. Shape validation
    # ==========================================

    shape_validation = (
        X_train_metadata.ndim == 2
        and
        X_validation_metadata.ndim == 2
        and
        X_test_metadata.ndim == 2
        and
        X_train_metadata.shape[0]
        == len(train_df)
        and
        X_validation_metadata.shape[0]
        == len(validation_df)
        and
        X_test_metadata.shape[0]
        == len(test_df)
    )

    # ==========================================
    # 9. Missing-value validation
    # ==========================================

    missing_validation = (
        not np.isnan(
            X_train_metadata
        ).any()
        and
        not np.isnan(
            X_validation_metadata
        ).any()
        and
        not np.isnan(
            X_test_metadata
        ).any()
    )

    # ==========================================
    # 10. Scaling validation
    # ==========================================

    numerical_feature_count = 3

    train_numerical = X_train_metadata[
        :, :numerical_feature_count
    ]

    train_mean = train_numerical.mean(
        axis=0
    )

    train_std = train_numerical.std(
        axis=0
    )

    scaling_validation = (
        np.allclose(
            train_mean,
            0,
            atol=1e-2
        )
        and
        np.allclose(
            train_std,
            1,
            atol=1e-2
        )
    )

    print("\nSCALING CHECK")
    print("-" * 60)

    print(
        f"Numerical mean: {train_mean}"
    )

    print(
        f"Numerical std : {train_std}"
    )
    # ==========================================
    # 11. Transformer validation
    # ==========================================

    transformer_exists = os.path.exists(
        preprocessing.transformer_path
    )

    # ==========================================
    # 12. Final validation
    # ==========================================

    print("\n")
    print("=" * 60)
    print("METADATA PREPROCESSING VALIDATION")
    print("=" * 60)

    print(
        f"Shape validation       : "
        f"{'PASS' if shape_validation else 'FAIL'}"
    )

    print(
        f"Missing-value check    : "
        f"{'PASS' if missing_validation else 'FAIL'}"
    )

    print(
        f"Train scaling check    : "
        f"{'PASS' if scaling_validation else 'FAIL'}"
    )

    print(
        f"Transformer saved      : "
        f"{'PASS' if transformer_exists else 'FAIL'}"
    )

    overall_validation = (
        shape_validation
        and
        missing_validation
        and
        scaling_validation
        and
        transformer_exists
    )

    print(
        f"\nMetadata preprocessing "
        f"validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)