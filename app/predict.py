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

from HealthPulse_AI_project.components.data_preprocessing import (
    DataPreprocessing
)

from HealthPulse_AI_project.components.prediction import (
    PredictionPipeline
)


# ==========================================================
# CONFIGURATION
# ==========================================================

ECG_DATA_PATH = (
    "data/raw/ptbxl"
)

NORMALIZATION_PATH = (
    "artifacts/preprocessing/"
    "normalization_params.npz"
)


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_ecg_record(
    filename_lr: str
):

    # ------------------------------------------------------
    # ECG preprocessing
    # ------------------------------------------------------

    preprocessing = DataPreprocessing(
        data_path=ECG_DATA_PATH,
        normalization_path=NORMALIZATION_PATH
    )

    processed_ecg = (
        preprocessing
        .preprocess_single_ecg(
            filename_lr
        )
    )

    # ------------------------------------------------------
    # CNN prediction
    # ------------------------------------------------------

    predictor = PredictionPipeline()

    result = predictor.predict(
        processed_ecg
    )

    return result


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HEALTHPULSE-AI ECG PREDICTION")
    print("=" * 60)

    # ------------------------------------------------------
    # Replace this with an actual PTB-XL record
    # ------------------------------------------------------

    filename_lr = (
        "records100/00000/00001_lr"
    )

    print(
        f"\nECG record: {filename_lr}"
    )

    result = predict_ecg_record(
        filename_lr
    )

    print("\n")
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Prediction  : "
        f"{result['prediction']}"
    )

    print(
        f"Class       : "
        f"{result['class']}"
    )

    print(
        f"Probability : "
        f"{result['probability']:.4f}"
    )

    print(
        f"Threshold   : "
        f"{result['threshold']:.2f}"
    )

    print("=" * 60)