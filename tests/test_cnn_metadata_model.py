import os
import sys

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

from HealthPulse_AI_project.models.cnn_metadata_model import (
    CNNMetadataModel
)


if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CNN + METADATA MODEL")
    print("=" * 60)

    # ======================================================
    # DATA PIPELINE
    # ======================================================

    pipeline = ModelDataPipeline()

    data = (
        pipeline
        .initiate_model_data_pipeline()
    )

    # ======================================================
    # MODEL
    # ======================================================

    model = CNNMetadataModel()

    (
        trained_model,
        history,
        validation_metrics,
        test_metrics
    ) = model.initiate_cnn_metadata_training(

        data.X_train_ecg,
        data.X_train_metadata,
        data.y_train,

        data.X_validation_ecg,
        data.X_validation_metadata,
        data.y_validation,

        data.X_test_ecg,
        data.X_test_metadata,
        data.y_test,

        epochs=30,
        batch_size=64
    )

    # ======================================================
    # RESULTS
    # ======================================================

    print("\n")
    print("=" * 60)
    print("CNN + METADATA RESULTS")
    print("=" * 60)

    print("\nMODEL")
    print("-" * 60)

    print(
        f"ECG input shape       : "
        f"{data.X_train_ecg.shape[1:]}"
    )

    print(
        f"Metadata input shape  : "
        f"{data.X_train_metadata.shape[1:]}"
    )

    print(
        f"Training records      : "
        f"{data.X_train_ecg.shape[0]}"
    )

    print(
        f"Parameters            : "
        f"{trained_model.count_params():,}"
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"Accuracy  : "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"MI Recall : "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{validation_metrics['pr_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        validation_metrics[
            "confusion_matrix"
        ]
    )

    # ======================================================
    # TEST
    # ======================================================

    print("\nTEST")
    print("-" * 60)

    print(
        f"Accuracy  : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"MI Recall : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        test_metrics[
            "confusion_matrix"
        ]
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    print("\n")
    print("=" * 60)
    print("CNN + METADATA MODEL VALIDATION")
    print("=" * 60)

    input_shape_check = (
        data.X_train_ecg.shape[1:]
        == (1000, 12)
    )

    metadata_shape_check = (
        data.X_train_metadata.shape[1]
        == 88
    )

    row_alignment_check = (
        data.X_train_ecg.shape[0]
        == data.X_train_metadata.shape[0]
        == len(data.y_train)
    )

    model_check = (
        trained_model is not None
    )

    metric_check = (
        0 <= test_metrics["f1"] <= 1
        and
        0 <= test_metrics["roc_auc"] <= 1
        and
        0 <= test_metrics["pr_auc"] <= 1
    )

    model_saved_check = (
        os.path.exists(
            model.model_path
        )
    )

    overall_check = (
        input_shape_check
        and
        metadata_shape_check
        and
        row_alignment_check
        and
        model_check
        and
        metric_check
        and
        model_saved_check
    )

    print(
        f"ECG shape validation      : "
        f"{'PASS' if input_shape_check else 'FAIL'}"
    )

    print(
        f"Metadata shape validation : "
        f"{'PASS' if metadata_shape_check else 'FAIL'}"
    )

    print(
        f"Row alignment             : "
        f"{'PASS' if row_alignment_check else 'FAIL'}"
    )

    print(
        f"Model creation            : "
        f"{'PASS' if model_check else 'FAIL'}"
    )

    print(
        f"Metric validation         : "
        f"{'PASS' if metric_check else 'FAIL'}"
    )

    print(
        f"Model saved               : "
        f"{'PASS' if model_saved_check else 'FAIL'}"
    )

    print(
        f"\nCNN + Metadata validation: "
        f"{'PASS' if overall_check else 'FAIL'}"
    )

    print("=" * 60)