import os
import sys
import json
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score
)

from HealthPulse_AI_project.exception.exception import (
    HealthPulseException
)

from HealthPulse_AI_project.logging.logger import logging


class ModelEvaluator:

    def __init__(
        self,
        output_path="project/data/evaluation"
    ):

        self.output_path = output_path

        os.makedirs(
            self.output_path,
            exist_ok=True
        )

        self.results_path = os.path.join(
            self.output_path,
            "final_model_results.json"
        )

    # ======================================================
    # VALIDATE INPUT
    # ======================================================

    def _validate_input(
        self,
        y_true,
        probabilities,
        name="Evaluation"
    ):

        if y_true is None:
            raise ValueError(
                f"{name} target is None"
            )

        if probabilities is None:
            raise ValueError(
                f"{name} probabilities are None"
            )

        y_true = np.asarray(
            y_true
        ).reshape(-1)

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32
        ).reshape(-1)

        if len(y_true) != len(
            probabilities
        ):

            raise ValueError(
                f"{name} target and probability "
                "lengths do not match"
            )

        if not np.isin(
            y_true,
            [0, 1]
        ).all():

            raise ValueError(
                f"{name} target must contain "
                "only 0 and 1"
            )

        if not np.isfinite(
            probabilities
        ).all():

            raise ValueError(
                f"{name} probabilities contain "
                "NaN or Inf"
            )

        if (
            (probabilities < 0).any()
            or
            (probabilities > 1).any()
        ):

            raise ValueError(
                f"{name} probabilities must "
                "be between 0 and 1"
            )

        return (
            y_true,
            probabilities
        )

    # ======================================================
    # EVALUATE MODEL
    # ======================================================

    def evaluate(
        self,
        y_true,
        probabilities,
        threshold=0.40,
        model_name="Model"
    ):

        try:

            (
                y_true,
                probabilities
            ) = self._validate_input(
                y_true,
                probabilities,
                model_name
            )

            predictions = (
                probabilities >= threshold
            ).astype(np.int32)

            # --------------------------------------------------
            # Confusion matrix
            # --------------------------------------------------

            tn, fp, fn, tp = (
                confusion_matrix(
                    y_true,
                    predictions,
                    labels=[0, 1]
                ).ravel()
            )

            # --------------------------------------------------
            # Threshold-dependent metrics
            # --------------------------------------------------

            accuracy = accuracy_score(
                y_true,
                predictions
            )

            precision = precision_score(
                y_true,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                predictions,
                zero_division=0
            )

            specificity = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else 0.0
            )

            balanced_accuracy = (
                balanced_accuracy_score(
                    y_true,
                    predictions
                )
            )

            # --------------------------------------------------
            # Threshold-independent metrics
            # --------------------------------------------------

            roc_auc = roc_auc_score(
                y_true,
                probabilities
            )

            pr_auc = average_precision_score(
                y_true,
                probabilities
            )

            results = {

                "model": model_name,

                "threshold": float(
                    threshold
                ),

                "accuracy": float(
                    accuracy
                ),

                "precision": float(
                    precision
                ),

                "recall": float(
                    recall
                ),

                "f1": float(
                    f1
                ),

                "specificity": float(
                    specificity
                ),

                "balanced_accuracy": float(
                    balanced_accuracy
                ),

                "roc_auc": float(
                    roc_auc
                ),

                "pr_auc": float(
                    pr_auc
                ),

                "tn": int(tn),

                "fp": int(fp),

                "fn": int(fn),

                "tp": int(tp)
            }

            return results

        except Exception as e:

            logging.exception(
                f"Error evaluating {model_name}"
            )

            raise HealthPulseException(
                e,
                sys
            )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    def save_results(
        self,
        results
    ):

        try:

            with open(
                self.results_path,
                "w"
            ) as file:

                json.dump(
                    results,
                    file,
                    indent=4
                )

            logging.info(
                f"Evaluation results saved to: "
                f"{self.results_path}"
            )

        except Exception as e:

            logging.exception(
                "Error saving evaluation results"
            )

            raise HealthPulseException(
                e,
                sys
            )