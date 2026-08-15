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


if __name__ == "__main__":

    # ==========================================
    # 1. PTB-XL dataset path
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
    # 6. Display results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("ECG DATA PREPROCESSING RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"X_train shape : {X_train.shape}"
    )

    print(
        f"y_train shape : {y_train.shape}"
    )

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"X_validation shape : "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation shape : "
        f"{y_validation.shape}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"X_test shape : {X_test.shape}"
    )

    print(
        f"y_test shape : {y_test.shape}"
    )

    # ==========================================
    # 7. Target validation
    # ==========================================

    print("\nTARGET VALIDATION")
    print("-" * 60)

    print(
        f"Train targets      : "
        f"{np.unique(y_train)}"
    )

    print(
        f"Validation targets : "
        f"{np.unique(y_validation)}"
    )

    print(
        f"Test targets       : "
        f"{np.unique(y_test)}"
    )

    target_validation = (
        np.isin(
            y_train,
            [0, 1]
        ).all()
        and
        np.isin(
            y_validation,
            [0, 1]
        ).all()
        and
        np.isin(
            y_test,
            [0, 1]
        ).all()
    )

    # ==========================================
    # 8. Normalization validation
    # ==========================================

    print("\nNORMALIZATION CHECK")
    print("-" * 60)

    train_mean = X_train.mean(
        axis=(0, 1)
    )

    train_std = X_train.std(
        axis=(0, 1)
    )

    print(
        f"Train normalized mean: "
        f"{train_mean}"
    )

    print(
        f"Train normalized std : "
        f"{train_std}"
    )

    normalization_validation = (
        np.allclose(
            train_mean,
            0,
            atol=1e-2
        )
        and
        np.allclose(
            train_std,
            1,
            atol=3e-2
        )
    )

    # ==========================================
    # 9. Shape validation
    # ==========================================

    shape_validation = (
        X_train.ndim == 3
        and
        X_validation.ndim == 3
        and
        X_test.ndim == 3
        and
        X_train.shape[1:] == (1000, 12)
        and
        X_validation.shape[1:] == (1000, 12)
        and
        X_test.shape[1:] == (1000, 12)
        and
        X_train.shape[0] == len(y_train)
        and
        X_validation.shape[0] == len(y_validation)
        and
        X_test.shape[0] == len(y_test)
    )

    # ==========================================
    # 10. Final validation
    # ==========================================

    normalization_file_exists = os.path.exists(
        preprocessing.normalization_path
    )

    print("\n")
    print("=" * 60)
    print("PREPROCESSING VALIDATION")
    print("=" * 60)

    print(
        f"Waveform shape validation : "
        f"{'PASS' if shape_validation else 'FAIL'}"
    )

    print(
        f"Target validation         : "
        f"{'PASS' if target_validation else 'FAIL'}"
    )

    print(
        f"Normalization validation  : "
        f"{'PASS' if normalization_validation else 'FAIL'}"
    )

    print(
        f"Normalization parameters   : "
        f"{'PASS' if normalization_file_exists else 'FAIL'}"
    )

    overall_validation = (
        shape_validation
        and target_validation
        and normalization_validation
        and normalization_file_exists
    )

    print(
        f"\nPreprocessing validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)