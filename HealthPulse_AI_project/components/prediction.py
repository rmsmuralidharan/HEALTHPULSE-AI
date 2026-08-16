import os
import sys
import numpy as np
import tensorflow as tf
import joblib

from HealthPulse_AI_project.exception.exception import (
    HealthPulseException
)

from HealthPulse_AI_project.logging.logger import logging


class PredictionPipeline:

    def __init__(
        self,
        model_path="project/data/models/cnn_final.keras",
        threshold_path="project/data/models/cnn_threshold.pkl"
    ):

        self.model_path = model_path
        self.threshold_path = threshold_path

        self.model = None
        self.threshold = None

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def _load_model(self):

        if not os.path.exists(
            self.model_path
        ):

            raise FileNotFoundError(
                f"CNN model not found: "
                f"{self.model_path}"
            )

        self.model = tf.keras.models.load_model(
            self.model_path
        )

        logging.info(
            f"CNN model loaded from: "
            f"{self.model_path}"
        )

    # ======================================================
    # LOAD THRESHOLD
    # ======================================================

    def _load_threshold(self):

        if not os.path.exists(
            self.threshold_path
        ):

            raise FileNotFoundError(
                f"Threshold file not found: "
                f"{self.threshold_path}"
            )

        self.threshold = float(
            joblib.load(
                self.threshold_path
            )
        )

        logging.info(
            f"CNN threshold loaded: "
            f"{self.threshold:.2f}"
        )

    # ======================================================
    # VALIDATE ECG
    # ======================================================

    def _validate_ecg(
        self,
        ecg
    ):

        if ecg is None:

            raise ValueError(
                "ECG input is None"
            )

        ecg = np.asarray(
            ecg,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Single ECG
        # --------------------------------------------------

        if ecg.ndim == 2:

            if ecg.shape != (
                1000,
                12
            ):

                raise ValueError(
                    "Single ECG must have "
                    "shape (1000, 12)"
                )

            ecg = np.expand_dims(
                ecg,
                axis=0
            )

        # --------------------------------------------------
        # Batch ECG
        # --------------------------------------------------

        elif ecg.ndim == 3:

            if ecg.shape[1:] != (
                1000,
                12
            ):

                raise ValueError(
                    "ECG batch must have "
                    "shape (N, 1000, 12)"
                )

        else:

            raise ValueError(
                "ECG must be either "
                "(1000, 12) or "
                "(N, 1000, 12)"
            )

        # --------------------------------------------------
        # NaN / Inf
        # --------------------------------------------------

        if not np.isfinite(
            ecg
        ).all():

            raise ValueError(
                "ECG contains NaN or "
                "infinite values"
            )

        return ecg

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        ecg
    ):

        try:

            logging.info(
                "Starting ECG prediction"
            )

            # --------------------------------------------------
            # Load artifacts
            # --------------------------------------------------

            self._load_model()

            self._load_threshold()

            # --------------------------------------------------
            # Validate input
            # --------------------------------------------------

            ecg = self._validate_ecg(
                ecg
            )

            # --------------------------------------------------
            # Model prediction
            # --------------------------------------------------

            probabilities = (
                self.model.predict(
                    ecg,
                    verbose=0
                ).reshape(-1)
            )

            predictions = (
                probabilities >=
                self.threshold
            ).astype(
                np.int32
            )

            results = []

            for probability, prediction in zip(
                probabilities,
                predictions
            ):

                if prediction == 1:

                    label = "MI"

                else:

                    label = "Non-MI"

                results.append({

                    "prediction": label,

                    "class": int(
                        prediction
                    ),

                    "probability": float(
                        probability
                    ),

                    "threshold": float(
                        self.threshold
                    )
                })

            logging.info(
                "ECG prediction completed"
            )

            # --------------------------------------------------
            # Single ECG → single result
            # --------------------------------------------------

            if len(results) == 1:

                return results[0]

            # --------------------------------------------------
            # Batch → list
            # --------------------------------------------------

            return results

        except Exception as e:

            logging.exception(
                "Error during ECG prediction"
            )

            raise HealthPulseException(
                e,
                sys
            )