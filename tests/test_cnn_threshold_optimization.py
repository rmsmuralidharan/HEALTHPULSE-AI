import os
import sys
import numpy as np
import tensorflow as tf

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

from HealthPulse_AI_project.models.cnn_threshold_optimization import (
    ThresholdOptimizer
)


if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CNN THRESHOLD OPTIMIZATION")
    print("=" * 60)

    # ======================================================
    # 1. LOAD MODEL DATA
    # ======================================================

    pipeline = ModelDataPipeline()

    data = (
        pipeline
        .initiate_model_data_pipeline()
    )

    # ======================================================
    # 2. LOAD TRAINED CNN
    # ======================================================

    model_path = (
        "project/data/models/cnn_final.keras"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"CNN model not found: "
            f"{model_path}"
        )

    cnn = tf.keras.models.load_model(
        model_path
    )

    # ======================================================
    # 3. VALIDATION PROBABILITIES
    # ======================================================

    validation_probabilities = (
        cnn.predict(
            data.X_validation_ecg,
            verbose=1
        ).reshape(-1)
    )

    # ======================================================
    # 4. TEST PROBABILITIES
    # ======================================================

    test_probabilities = (
        cnn.predict(
            data.X_test_ecg,
            verbose=1
        ).reshape(-1)
    )

    # ======================================================
    # 5. OPTIMIZER
    # ======================================================

    optimizer = ThresholdOptimizer()

    # ======================================================
    # 6. OPTIMIZE USING VALIDATION ONLY
    # ======================================================

    (
        best_threshold,
        validation_best,
        search_results
    ) = optimizer.optimize_threshold(

        data.y_validation,

        validation_probabilities
    )

    # ======================================================
    # 7. FINAL TEST
    # ======================================================

    test_metrics = (
        optimizer.evaluate_test(

            data.y_test,

            test_probabilities
        )
    )

    # ======================================================
    # RESULTS
    # ======================================================

    print("\n")
    print("=" * 60)
    print("CNN THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 60)

    print("\nVALIDATION - BEST THRESHOLD")
    print("-" * 60)

    print(
        f"Threshold    : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Accuracy     : "
        f"{validation_best['accuracy']:.4f}"
    )

    print(
        f"Precision    : "
        f"{validation_best['precision']:.4f}"
    )

    print(
        f"MI Recall    : "
        f"{validation_best['recall']:.4f}"
    )

    print(
        f"F1 Score     : "
        f"{validation_best['f1']:.4f}"
    )

    print(
        f"Specificity  : "
        f"{validation_best['specificity']:.4f}"
    )

    print(
        f"Balanced Acc : "
        f"{validation_best['balanced_accuracy']:.4f}"
    )

    print("\nTEST - FINAL EVALUATION")
    print("-" * 60)

    print(
        f"Threshold    : "
        f"{test_metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy     : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision    : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"MI Recall    : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score     : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Specificity  : "
        f"{test_metrics['specificity']:.4f}"
    )

    print(
        f"Balanced Acc : "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print("\nTEST CONFUSION MATRIX")
    print("-" * 60)

    print(
        f"TN : {test_metrics['tn']}"
    )

    print(
        f"FP : {test_metrics['fp']}"
    )

    print(
        f"FN : {test_metrics['fn']}"
    )

    print(
        f"TP : {test_metrics['tp']}"
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    print("\n")
    print("=" * 60)
    print("CNN THRESHOLD OPTIMIZATION VALIDATION")
    print("=" * 60)

    threshold_check = (
        0.10 <= best_threshold <= 0.90
    )

    search_check = (
        len(search_results) > 0
    )

    metric_check = all(
        0 <= validation_best[key] <= 1
        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy"
        ]
    )

    threshold_saved_check = (
        os.path.exists(
            optimizer.threshold_path
        )
    )

    overall_check = (
        threshold_check
        and search_check
        and metric_check
        and threshold_saved_check
    )

    print(
        f"Threshold validation : "
        f"{'PASS' if threshold_check else 'FAIL'}"
    )

    print(
        f"Search validation    : "
        f"{'PASS' if search_check else 'FAIL'}"
    )

    print(
        f"Metric validation    : "
        f"{'PASS' if metric_check else 'FAIL'}"
    )

    print(
        f"Threshold saved      : "
        f"{'PASS' if threshold_saved_check else 'FAIL'}"
    )

    print(
        f"\nCNN threshold optimization validation: "
        f"{'PASS' if overall_check else 'FAIL'}"
    )

    print("=" * 60)