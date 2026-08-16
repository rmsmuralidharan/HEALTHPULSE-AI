import os
import sys
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

from HealthPulse_AI_project.components.model_evaluation import (
    ModelEvaluator
)


if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("FINAL CNN MODEL EVALUATION")
    print("=" * 60)

    # ======================================================
    # 1. DATA
    # ======================================================

    pipeline = ModelDataPipeline()

    data = (
        pipeline
        .initiate_model_data_pipeline()
    )

    # ======================================================
    # 2. LOAD FINAL CNN
    # ======================================================

    model_path = (
        "project/data/models/cnn_final.keras"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Final CNN model not found: "
            f"{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    # ======================================================
    # 3. TEST PROBABILITIES
    # ======================================================

    test_probabilities = (
        model.predict(
            data.X_test_ecg,
            verbose=1
        ).reshape(-1)
    )

    # ======================================================
    # 4. EVALUATOR
    # ======================================================

    evaluator = ModelEvaluator()

    results = evaluator.evaluate(

        y_true=data.y_test,

        probabilities=test_probabilities,

        threshold=0.40,

        model_name="1D CNN"
    )

    # ======================================================
    # 5. SAVE
    # ======================================================

    evaluator.save_results(
        results
    )

    # ======================================================
    # 6. DISPLAY
    # ======================================================

    print("\n")
    print("=" * 60)
    print("FINAL CNN RESULTS")
    print("=" * 60)

    print("\nMODEL")
    print("-" * 60)

    print(
        f"Model       : "
        f"{results['model']}"
    )

    print(
        f"Threshold   : "
        f"{results['threshold']:.2f}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"Accuracy        : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Precision       : "
        f"{results['precision']:.4f}"
    )

    print(
        f"MI Recall       : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1 Score        : "
        f"{results['f1']:.4f}"
    )

    print(
        f"Specificity     : "
        f"{results['specificity']:.4f}"
    )

    print(
        f"Balanced Acc.   : "
        f"{results['balanced_accuracy']:.4f}"
    )

    print(
        f"ROC-AUC         : "
        f"{results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC          : "
        f"{results['pr_auc']:.4f}"
    )

    print("\nCONFUSION MATRIX")
    print("-" * 60)

    print(
        f"TN : {results['tn']}"
    )

    print(
        f"FP : {results['fp']}"
    )

    print(
        f"FN : {results['fn']}"
    )

    print(
        f"TP : {results['tp']}"
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    print("\n")
    print("=" * 60)
    print("FINAL MODEL EVALUATION VALIDATION")
    print("=" * 60)

    metric_check = all(
        0 <= results[key] <= 1
        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc"
        ]
    )

    threshold_check = (
        results["threshold"] == 0.40
    )

    confusion_check = (
        results["tn"]
        + results["fp"]
        + results["fn"]
        + results["tp"]
        == len(data.y_test)
    )

    saved_check = os.path.exists(
        evaluator.results_path
    )

    overall_check = (
        metric_check
        and threshold_check
        and confusion_check
        and saved_check
    )

    print(
        f"Metric validation    : "
        f"{'PASS' if metric_check else 'FAIL'}"
    )

    print(
        f"Threshold validation : "
        f"{'PASS' if threshold_check else 'FAIL'}"
    )

    print(
        f"Confusion validation : "
        f"{'PASS' if confusion_check else 'FAIL'}"
    )

    print(
        f"Results saved        : "
        f"{'PASS' if saved_check else 'FAIL'}"
    )

    print(
        f"\nFinal evaluation validation: "
        f"{'PASS' if overall_check else 'FAIL'}"
    )

    print("=" * 60)