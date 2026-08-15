import numpy as np
import pandas as pd

from HealthPulse_AI_project.components.model_preparation import (
    ModelPreparation
)


if __name__ == "__main__":

    # ==========================================
    # Create realistic fused-data test frames
    # ==========================================

    train_records = 15270
    validation_records = 3251
    test_records = 3278

    number_of_features = 172

    rng = np.random.default_rng(42)

    # ------------------------------------------
    # Feature names
    # ------------------------------------------

    feature_names = [
        f"feature_{i}"
        for i in range(number_of_features)
    ]

    # ------------------------------------------
    # Create train
    # ------------------------------------------

    train_features = rng.normal(
        size=(
            train_records,
            number_of_features
        )
    ).astype(np.float32)

    train_df = pd.DataFrame(
        train_features,
        columns=feature_names
    )

    train_df.insert(
        0,
        "ecg_id",
        np.arange(
            1,
            train_records + 1
        )
    )

    train_target = np.array(
        [0] * 11418 +
        [1] * 3852
    )

    train_df["Target"] = train_target

    # ------------------------------------------
    # Create validation
    # ------------------------------------------

    validation_features = rng.normal(
        size=(
            validation_records,
            number_of_features
        )
    ).astype(np.float32)

    validation_df = pd.DataFrame(
        validation_features,
        columns=feature_names
    )

    validation_df.insert(
        0,
        "ecg_id",
        np.arange(
            21799,
            21799 + validation_records
        )
    )

    validation_target = np.array(
        [0] * 2441 +
        [1] * 810
    )

    validation_df["Target"] = validation_target

    # ------------------------------------------
    # Create test
    # ------------------------------------------

    test_features = rng.normal(
        size=(
            test_records,
            number_of_features
        )
    ).astype(np.float32)

    test_df = pd.DataFrame(
        test_features,
        columns=feature_names
    )

    test_df.insert(
        0,
        "ecg_id",
        np.arange(
            25050,
            25050 + test_records
        )
    )

    test_target = np.array(
        [0] * 2471 +
        [1] * 807
    )

    test_df["Target"] = test_target

    # ==========================================
    # Model preparation
    # ==========================================

    preparation = ModelPreparation()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = preparation.initiate_model_preparation(
        train_df,
        validation_df,
        test_df
    )

    # ==========================================
    # Results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("MODEL PREPARATION RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"X_train shape : "
        f"{X_train.shape}"
    )

    print(
        f"y_train shape : "
        f"{y_train.shape}"
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
        f"X_test shape : "
        f"{X_test.shape}"
    )

    print(
        f"y_test shape : "
        f"{y_test.shape}"
    )

    # ==========================================
    # Validation checks
    # ==========================================

    shape_validation = (
        X_train.shape == (15270, 172)
        and
        X_validation.shape == (3251, 172)
        and
        X_test.shape == (3278, 172)
    )

    target_shape_validation = (
        y_train.shape == (15270,)
        and
        y_validation.shape == (3251,)
        and
        y_test.shape == (3278,)
    )

    dtype_validation = (
        X_train.dtype == np.float32
        and
        X_validation.dtype == np.float32
        and
        X_test.dtype == np.float32
        and
        y_train.dtype == np.int32
        and
        y_validation.dtype == np.int32
        and
        y_test.dtype == np.int32
    )

    finite_validation = (
        np.isfinite(X_train).all()
        and
        np.isfinite(X_validation).all()
        and
        np.isfinite(X_test).all()
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

    class_validation = (
        len(np.unique(y_train)) == 2
        and
        len(np.unique(y_validation)) == 2
        and
        len(np.unique(y_test)) == 2
    )

    # ==========================================
    # Verify ecg_id was removed
    # ==========================================

    identifier_removed = (
        X_train.shape[1] == 172
    )

    # ==========================================
    # Final validation
    # ==========================================

    overall_validation = (
        shape_validation
        and
        target_shape_validation
        and
        dtype_validation
        and
        finite_validation
        and
        target_validation
        and
        class_validation
        and
        identifier_removed
    )

    # ==========================================
    # Validation report
    # ==========================================

    print("\n")
    print("=" * 60)
    print("MODEL PREPARATION VALIDATION")
    print("=" * 60)

    print(
        f"Feature shape validation : "
        f"{'PASS' if shape_validation else 'FAIL'}"
    )

    print(
        f"Target shape validation  : "
        f"{'PASS' if target_shape_validation else 'FAIL'}"
    )

    print(
        f"Dtype validation         : "
        f"{'PASS' if dtype_validation else 'FAIL'}"
    )

    print(
        f"NaN / Inf validation     : "
        f"{'PASS' if finite_validation else 'FAIL'}"
    )

    print(
        f"Target validation        : "
        f"{'PASS' if target_validation else 'FAIL'}"
    )

    print(
        f"Class validation         : "
        f"{'PASS' if class_validation else 'FAIL'}"
    )

    print(
        f"Identifier removal       : "
        f"{'PASS' if identifier_removed else 'FAIL'}"
    )

    print(
        f"\nModel preparation "
        f"validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)