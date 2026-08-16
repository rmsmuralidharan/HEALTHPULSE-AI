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

from HealthPulse_AI_project.components.prediction import (
    PredictionPipeline
)


if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("HEALTHPULSE-AI PREDICTION PIPELINE")
    print("=" * 60)

    # ======================================================
    # 1. LOAD DATA
    # ======================================================

    pipeline = ModelDataPipeline()

    data = (
        pipeline
        .initiate_model_data_pipeline()
    )

    # ======================================================
    # 2. CREATE PREDICTION PIPELINE
    # ======================================================

    predictor = PredictionPipeline()

    # ======================================================
    # 3. TAKE ONE TEST ECG
    # ======================================================

    sample_ecg = (
        data.X_test_ecg[0]
    )

    actual_class = int(
        data.y_test[0]
    )

    # ======================================================
    # 4. PREDICT
    # ======================================================

    result = predictor.predict(
        sample_ecg
    )

    # ======================================================
    # 5. DISPLAY
    # ======================================================

    print("\n")
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"ECG shape      : "
        f"{sample_ecg.shape}"
    )

    print(
        f"Actual class   : "
        f"{actual_class}"
    )

    print(
        f"Prediction     : "
        f"{result['prediction']}"
    )

    print(
        f"Predicted class: "
        f"{result['class']}"
    )

    print(
        f"Probability    : "
        f"{result['probability']:.4f}"
    )

    print(
        f"Threshold      : "
        f"{result['threshold']:.2f}"
    )

    # ======================================================
    # VALIDATION
    # ======================================================

    print("\n")
    print("=" * 60)
    print("PREDICTION PIPELINE VALIDATION")
    print("=" * 60)

    shape_check = (
        sample_ecg.shape
        == (1000, 12)
    )

    probability_check = (
        0.0 <=
        result["probability"]
        <= 1.0
    )

    threshold_check = np.isclose(
        result["threshold"],
        0.40,
        atol=1e-6
    )

    class_check = (
        result["class"]
        in [0, 1]
    )

    label_check = (
        result["prediction"]
        in ["MI", "Non-MI"]
    )

    overall_check = (
        shape_check
        and probability_check
        and threshold_check
        and class_check
        and label_check
    )

    print(
        f"Input shape validation : "
        f"{'PASS' if shape_check else 'FAIL'}"
    )

    print(
        f"Probability validation : "
        f"{'PASS' if probability_check else 'FAIL'}"
    )

    print(
        f"Threshold validation   : "
        f"{'PASS' if threshold_check else 'FAIL'}"
    )

    print(
        f"Class validation       : "
        f"{'PASS' if class_check else 'FAIL'}"
    )

    print(
        f"Label validation       : "
        f"{'PASS' if label_check else 'FAIL'}"
    )

    print(
        f"\nPrediction pipeline validation: "
        f"{'PASS' if overall_check else 'FAIL'}"
    )

    print("=" * 60)