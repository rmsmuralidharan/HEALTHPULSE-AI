import os
import sys

import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from HealthPulse_AI_project.pipelines.model_data_pipeline import (
    ModelDataPipeline
)


if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("MODEL DATA PIPELINE RESULTS")
    print("=" * 60)

    pipeline = ModelDataPipeline(
        data_path=os.path.join(
            "data",
            "raw",
            "ptbxl"
        ),
        random_state=42
    )

    data = (
        pipeline
        .initiate_model_data_pipeline()
    )

    # ======================================================
    # RAW ECG
    # ======================================================

    print("\nRAW ECG")
    print("-" * 60)

    print(
        f"Train      : "
        f"{data.X_train_ecg.shape}"
    )

    print(
        f"Validation : "
        f"{data.X_validation_ecg.shape}"
    )

    print(
        f"Test       : "
        f"{data.X_test_ecg.shape}"
    )

    # ======================================================
    # ECG FEATURES
    # ======================================================

    print("\nECG FEATURES")
    print("-" * 60)

    print(
        f"Train      : "
        f"{data.X_train_ecg_features.shape}"
    )

    print(
        f"Validation : "
        f"{data.X_validation_ecg_features.shape}"
    )

    print(
        f"Test       : "
        f"{data.X_test_ecg_features.shape}"
    )

    print(
        f"Feature count : "
        f"{len(data.ecg_feature_names)}"
    )

    # ======================================================
    # METADATA
    # ======================================================

    print("\nMETADATA")
    print("-" * 60)

    print(
        f"Train      : "
        f"{data.X_train_metadata.shape}"
    )

    print(
        f"Validation : "
        f"{data.X_validation_metadata.shape}"
    )

    print(
        f"Test       : "
        f"{data.X_test_metadata.shape}"
    )

    print(
        f"Feature count : "
        f"{len(data.metadata_feature_names)}"
    )

    # ======================================================
    # FUSED FEATURES
    # ======================================================

    print("\nFUSED FEATURES")
    print("-" * 60)

    print(
        f"Train      : "
        f"{data.X_train_fused.shape}"
    )

    print(
        f"Validation : "
        f"{data.X_validation_fused.shape}"
    )

    print(
        f"Test       : "
        f"{data.X_test_fused.shape}"
    )

    print(
        f"Feature count : "
        f"{len(data.fused_feature_names)}"
    )

    # ======================================================
    # MODEL DATA
    # ======================================================

    print("\nMODEL DATA")
    print("-" * 60)

    print(
        f"X_train        : "
        f"{data.X_train.shape}"
    )

    print(
        f"y_train        : "
        f"{data.y_train.shape}"
    )

    print(
        f"X_validation   : "
        f"{data.X_validation.shape}"
    )

    print(
        f"y_validation   : "
        f"{data.y_validation.shape}"
    )

    print(
        f"X_test         : "
        f"{data.X_test.shape}"
    )

    print(
        f"y_test         : "
        f"{data.y_test.shape}"
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    raw_ecg_check = (
        data.X_train_ecg.ndim == 3
        and
        data.X_train_ecg.shape[1:] == (
            1000,
            12
        )
    )

    ecg_feature_check = (
        data.X_train_ecg_features.shape[1] == 84
    )

    metadata_check = (
        data.X_train_metadata.shape[1] == 88
    )

    fusion_check = (
        data.X_train_fused.shape[1] == 172
    )

    model_check = (
        data.X_train.shape[1] == 172
    )

    alignment_check = (
        len(data.X_train) == len(data.y_train)
        and
        len(data.X_validation) == len(
            data.y_validation
        )
        and
        len(data.X_test) == len(data.y_test)
    )

    nan_check = (
        np.isfinite(
            data.X_train
        ).all()

        and

        np.isfinite(
            data.X_validation
        ).all()

        and

        np.isfinite(
            data.X_test
        ).all()
    )

    target_check = (
        set(
            np.unique(
                data.y_train
            )
        ).issubset(
            {0, 1}
        )
    )

    overall_check = (
        raw_ecg_check
        and
        ecg_feature_check
        and
        metadata_check
        and
        fusion_check
        and
        model_check
        and
        alignment_check
        and
        nan_check
        and
        target_check
    )

    print("\n")
    print("=" * 60)
    print("MODEL DATA PIPELINE VALIDATION")
    print("=" * 60)

    print(
        f"Raw ECG shape       : "
        f"{'PASS' if raw_ecg_check else 'FAIL'}"
    )

    print(
        f"ECG features        : "
        f"{'PASS' if ecg_feature_check else 'FAIL'}"
    )

    print(
        f"Metadata features   : "
        f"{'PASS' if metadata_check else 'FAIL'}"
    )

    print(
        f"Feature fusion      : "
        f"{'PASS' if fusion_check else 'FAIL'}"
    )

    print(
        f"Model features      : "
        f"{'PASS' if model_check else 'FAIL'}"
    )

    print(
        f"Data alignment      : "
        f"{'PASS' if alignment_check else 'FAIL'}"
    )

    print(
        f"NaN / Inf check     : "
        f"{'PASS' if nan_check else 'FAIL'}"
    )

    print(
        f"Target validation   : "
        f"{'PASS' if target_check else 'FAIL'}"
    )

    print(
        f"\nModel Data Pipeline "
        f"validation: "
        f"{'PASS' if overall_check else 'FAIL'}"
    )

    print("=" * 60)