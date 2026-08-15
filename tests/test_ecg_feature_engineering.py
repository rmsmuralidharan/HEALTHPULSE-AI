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

from HealthPulse_AI_project.components.ecg_feature_engineering import (
    ECGFeatureEngineering
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
    # 2. Get the already prepared split DataFrames
    # ==========================================

    ingestion = DataIngestion(
        data_path=data_path
    )

    database_df, scp_df = (
        ingestion.initiate_data_ingestion()
    )

    target_creator = TargetCreation()

    target_df = (
        target_creator.initiate_target_creation(
            database_df,
            scp_df
        )
    )

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
    # 3. Use COMPLETED ECG preprocessing
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
    # 4. ECG Feature Engineering
    # ==========================================

    feature_engineering = (
        ECGFeatureEngineering()
    )

    (
        train_features,
        validation_features,
        test_features,
        feature_names,
        original_train,
        original_validation,
        original_test
    ) = feature_engineering.initiate_ecg_feature_engineering(
        X_train,
        X_validation,
        X_test
    )

    # ==========================================
    # 5. Results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("ECG FEATURE ENGINEERING RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"ECG input shape : {X_train.shape}"
    )

    print(
        f"Feature shape   : {train_features.shape}"
    )

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"ECG input shape : {X_validation.shape}"
    )

    print(
        f"Feature shape   : "
        f"{validation_features.shape}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"ECG input shape : {X_test.shape}"
    )

    print(
        f"Feature shape   : {test_features.shape}"
    )

    # ==========================================
    # 6. Feature information
    # ==========================================

    print("\nFEATURE INFORMATION")
    print("-" * 60)

    print(
        f"Number of features: "
        f"{len(feature_names)}"
    )

    print(
        "\nFirst 20 feature names:"
    )

    print(
        feature_names[:20]
    )

    # ==========================================
    # 7. Feature sample
    # ==========================================

    print("\nFEATURE SAMPLE")
    print("-" * 60)

    print(
        train_features[:5]
    )

    # ==========================================
    # 8. Validation checks
    # ==========================================

    feature_count_validation = (
        train_features.shape[1] == 84
        and
        validation_features.shape[1] == 84
        and
        test_features.shape[1] == 84
    )

    row_count_validation = (
        train_features.shape[0]
        == X_train.shape[0]
        and
        validation_features.shape[0]
        == X_validation.shape[0]
        and
        test_features.shape[0]
        == X_test.shape[0]
    )

    finite_validation = (
        np.isfinite(
            train_features
        ).all()
        and
        np.isfinite(
            validation_features
        ).all()
        and
        np.isfinite(
            test_features
        ).all()
    )

    feature_name_validation = (
        len(feature_names) == 84
    )

    # ==========================================
    # 9. Original ECG preservation
    # ==========================================

    original_shape_validation = (
        original_train.shape == X_train.shape
        and
        original_validation.shape
        == X_validation.shape
        and
        original_test.shape
        == X_test.shape
    )

    original_data_validation = (
        np.array_equal(
            original_train,
            X_train
        )
        and
        np.array_equal(
            original_validation,
            X_validation
        )
        and
        np.array_equal(
            original_test,
            X_test
        )
    )

    # ==========================================
    # 10. Target alignment
    # ==========================================

    target_validation = (
        len(y_train)
        == len(train_features)
        and
        len(y_validation)
        == len(validation_features)
        and
        len(y_test)
        == len(test_features)
    )

    # ==========================================
    # 11. Final validation
    # ==========================================

    overall_validation = (
        feature_count_validation
        and
        row_count_validation
        and
        finite_validation
        and
        feature_name_validation
        and
        original_shape_validation
        and
        original_data_validation
        and
        target_validation
    )

    print("\n")
    print("=" * 60)
    print("ECG FEATURE ENGINEERING VALIDATION")
    print("=" * 60)

    print(
        f"Feature count validation : "
        f"{'PASS' if feature_count_validation else 'FAIL'}"
    )

    print(
        f"Row count validation     : "
        f"{'PASS' if row_count_validation else 'FAIL'}"
    )

    print(
        f"NaN / Inf validation     : "
        f"{'PASS' if finite_validation else 'FAIL'}"
    )

    print(
        f"Feature name validation  : "
        f"{'PASS' if feature_name_validation else 'FAIL'}"
    )

    print(
        f"Original ECG shape       : "
        f"{'PASS' if original_shape_validation else 'FAIL'}"
    )

    print(
        f"Original ECG preserved   : "
        f"{'PASS' if original_data_validation else 'FAIL'}"
    )

    print(
        f"Target alignment         : "
        f"{'PASS' if target_validation else 'FAIL'}"
    )

    print(
        f"\nECG FE validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)