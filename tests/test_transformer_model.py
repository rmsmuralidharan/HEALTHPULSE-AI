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

from HealthPulse_AI_project.models.transformer_model import (
    TransformerModel
)


if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("ECG TRANSFORMER MODEL")
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

    transformer = TransformerModel()

    (
        model,
        history,
        validation_metrics,
        test_metrics
    ) = transformer.initiate_transformer_training(

        data.X_train_ecg,
        data.y_train,

        data.X_validation_ecg,
        data.y_validation,

        data.X_test_ecg,
        data.y_test,

        epochs=30,
        batch_size=64
    )

    # ======================================================
    # RESULTS
    # ======================================================

    print("\n")
    print("=" * 60)
    print("TRANSFORMER RESULTS")
    print("=" * 60)

    print("\nMODEL")
    print("-" * 60)

    print(
        f"Input shape      : "
        f"{data.X_train_ecg.shape[1:]}"
    )

    print(
        f"Training records : "
        f"{data.X_train_ecg.shape[0]}"
    )

    print(
        f"Parameters       : "
        f"{model.count_params():,}"
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
    print("TRANSFORMER MODEL VALIDATION")
    print("=" * 60)

    input_shape_check = (
        data.X_train_ecg.shape[1:]
        == (1000, 12)
    )

    row_check = (
        data.X_train_ecg.shape[0]
        == len(data.y_train)
    )

    model_check = (
        model is not None
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
            transformer.model_path
        )
    )

    overall_check = (
        input_shape_check
        and
        row_check
        and
        model_check
        and
        metric_check
        and
        model_saved_check
    )

    print(
        f"Input shape validation : "
        f"{'PASS' if input_shape_check else 'FAIL'}"
    )

    print(
        f"Row alignment          : "
        f"{'PASS' if row_check else 'FAIL'}"
    )

    print(
        f"Model creation         : "
        f"{'PASS' if model_check else 'FAIL'}"
    )

    print(
        f"Metric validation      : "
        f"{'PASS' if metric_check else 'FAIL'}"
    )

    print(
        f"Model saved            : "
        f"{'PASS' if model_saved_check else 'FAIL'}"
    )

    print(
        f"\nTransformer validation: "
        f"{'PASS' if overall_check else 'FAIL'}"
    )

    print("=" * 60)