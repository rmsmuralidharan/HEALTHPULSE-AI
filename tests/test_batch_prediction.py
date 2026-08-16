import os
import sys
import numpy as np
import pandas as pd

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

ECG_DATA_PATH = "data/raw/ptbxl"

NORMALIZATION_PATH = (
    "artifacts/preprocessing/"
    "normalization_params.npz"
)

# Number of ECGs to test
NUMBER_OF_SAMPLES = 10


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("HEALTHPULSE-AI BATCH PREDICTION TEST")
    print("=" * 60)

    # ======================================================
    # 1. LOAD DATASET METADATA
    # ======================================================

    database_path = os.path.join(
        ECG_DATA_PATH,
        "ptbxl_database.csv"
    )

    if not os.path.exists(database_path):

        raise FileNotFoundError(
            f"PTB-XL database not found: "
            f"{database_path}"
        )

    df = pd.read_csv(
        database_path
    )

    if "filename_lr" not in df.columns:

        raise ValueError(
            "filename_lr column not found"
        )

    # ------------------------------------------------------
    # Select records that actually exist
    # ------------------------------------------------------

    valid_records = []

    for filename in df["filename_lr"]:

        record_path = os.path.join(
            ECG_DATA_PATH,
            filename
        )

        if (
            os.path.exists(
                record_path + ".hea"
            )
            and
            os.path.exists(
                record_path + ".dat"
            )
        ):

            valid_records.append(
                filename
            )

        if len(valid_records) >= NUMBER_OF_SAMPLES:
            break

    if not valid_records:

        raise ValueError(
            "No valid ECG records found"
        )

    # ======================================================
    # 2. CREATE COMPONENTS
    # ======================================================

    preprocessing = DataPreprocessing(
        data_path=ECG_DATA_PATH,
        normalization_path=NORMALIZATION_PATH
    )

    predictor = PredictionPipeline()

    # ======================================================
    # 3. PREDICT MULTIPLE ECGs
    # ======================================================

    results = []

    print(
        f"\nTesting {len(valid_records)} ECG records..."
    )

    for index, filename in enumerate(
        valid_records,
        start=1
    ):

        print(
            f"\nProcessing "
            f"{index}/{len(valid_records)}: "
            f"{filename}"
        )

        # ----------------------------------------------
        # Preprocess
        # ----------------------------------------------

        processed_ecg = (
            preprocessing
            .preprocess_single_ecg(
                filename
            )
        )

        # ----------------------------------------------
        # Prediction
        # ----------------------------------------------

        prediction = predictor.predict(
            processed_ecg
        )

        results.append({

            "filename": filename,

            "prediction": prediction[
                "prediction"
            ],

            "class": prediction[
                "class"
            ],

            "probability": prediction[
                "probability"
            ],

            "threshold": prediction[
                "threshold"
            ]
        })

    # ======================================================
    # 4. RESULTS
    # ======================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 60)
    print("BATCH PREDICTION RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ======================================================
    # 5. SUMMARY
    # ======================================================

    mi_count = int(
        (
            results_df["class"] == 1
        ).sum()
    )

    non_mi_count = int(
        (
            results_df["class"] == 0
        ).sum()
    )

    print("\n")
    print("=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    print(
        f"Total ECGs : "
        f"{len(results_df)}"
    )

    print(
        f"MI         : "
        f"{mi_count}"
    )

    print(
        f"Non-MI     : "
        f"{non_mi_count}"
    )

    print(
        f"Threshold  : "
        f"{results_df['threshold'].iloc[0]:.2f}"
    )

    # ======================================================
    # 6. VALIDATION
    # ======================================================

    shape_check = (
        len(results_df)
        == len(valid_records)
    )

    probability_check = (
        results_df["probability"]
        .between(0, 1)
        .all()
    )

    class_check = (
        results_df["class"]
        .isin([0, 1])
        .all()
    )

    threshold_check = (
        np.isclose(
            results_df["threshold"],
            0.40,
            atol=1e-6
        ).all()
    )

    prediction_check = (
        results_df["prediction"]
        .isin(
            ["MI", "Non-MI"]
        )
        .all()
    )

    overall_check = (
        shape_check
        and probability_check
        and class_check
        and threshold_check
        and prediction_check
    )

    print("\n")
    print("=" * 60)
    print("BATCH PREDICTION VALIDATION")
    print("=" * 60)

    print(
        f"Record count validation : "
        f"{'PASS' if shape_check else 'FAIL'}"
    )

    print(
        f"Probability validation  : "
        f"{'PASS' if probability_check else 'FAIL'}"
    )

    print(
        f"Class validation        : "
        f"{'PASS' if class_check else 'FAIL'}"
    )

    print(
        f"Threshold validation    : "
        f"{'PASS' if threshold_check else 'FAIL'}"
    )

    print(
        f"Prediction validation   : "
        f"{'PASS' if prediction_check else 'FAIL'}"
    )

    print(
        f"\nBatch prediction validation: "
        f"{'PASS' if overall_check else 'FAIL'}"
    )

    print("=" * 60)