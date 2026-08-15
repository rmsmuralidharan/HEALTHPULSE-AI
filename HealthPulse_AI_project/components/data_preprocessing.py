import os
import sys

import numpy as np
import pandas as pd
import wfdb

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging

class DataPreprocessing:
    def __init__(
            self,
            data_path: str,
            normalization_path: str = (
                'artifacts/preprocessing/'
                'normalization_params.npz'
            )
    ):
        self.data_path =data_path
        self.normalization_path = normalization_path
        self.expected_samples = 1000
        self.expected_leads = 12

    def _validate_dataframe(
            self,
            df: pd.DataFrame,
            name: str
    ):
        if df is None or df.empty:
            raise ValueError(
                f"{name} dataframe is empty"
            )

        required_columns = [
            'filename_lr',
            'Target'
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{name} dataframe is missing"
                f"reuired columns: {missing_columns}"
            )

        if df['filename_lr'].isna().any():
            raise ValueError(
                f"{name} contains missing "
                'filename_lr values'
            )

        if not df['Target'].isin([0,1]).all():
            raise ValueError(
                f"{name} contains invalid Target values"
            )

        # 2. Validate ECG files

    def _validate_ecg_files(
            self,
            filename_lr:str
    ):
        if not isinstance(filename_lr, str):
            raise(
                'filename_lr must be a string'
            )

        record_path = os.path.join(
            self.data_path,
            filename_lr
        )

        hea_path = record_path + ".hea"
        dat_path = record_path + ".dat"

        if not os.path.exists(hea_path):
            raise FileNotFoundError(
                f"Missing ECG Header files: {hea_path}"
            )

        if not os.path.exists(dat_path):
            raise FileNotFoundError(
                f"Missing ECG signal file: {dat_path}"
            )

        return record_path

    # 3. Load ECG waveform

    def _load_ecg_signal(
            self,
            filename_lr: str
    ):
        record_path = self._validate_ecg_files(
            filename_lr=filename_lr
        )

        signal, _ = wfdb.rdsamp(
            record_path
        )

        signal = np.asarray(
            signal,
            dtype=np.float32
        )

        expected_shape = (
            self.expected_samples,
            self.expected_leads
        )

        if signal.shape != expected_shape:
            raise ValueError(
                f"unexpected ECG shape for "
                f"{filename_lr}"
                f"Expected {expected_shape, }"
                f"got {signal.shape}"
            )

        return signal

    # 4. Calculate normalization parameters
    #    TRAIN DATA ONLY

    def _calculate_normalization_parameters(
            self, 
            train_df: pd.DataFrame
    ):
        logging.info(
            "Calculating normalization parameters "
            "using TRAIN ECG signals only"
        )

        lead_sum = np.zeros(
            self.expected_leads,
            dtype=np.float64
        )

        lead_squared_sum = np.zeros(
            self.expected_leads,
            dtype=np.float64
        )

        total_samples = 0

        for index, filename in enumerate(
            train_df['filename_lr']
        ):
            signal = self._load_ecg_signal(
                filename
            )

            lead_sum += signal.sum(
                axis=0,
                dtype=np.float64
            )

            lead_squared_sum += (
                np.square(signal)
                .sum(
                    axis = 0,
                    dtype = np.float64
                )
            )

            total_samples += signal.shape[0]

            if total_samples <= 0:
                raise ValueError(
                    "No ECG samples found in "
                    "training data"
                )

        ## per-lead mean
        mean = (
            lead_sum / total_samples
        )

        # variance
        variance = (
            lead_squared_sum / total_samples
        ) - np.square(mean)

        ## protect against tiny neagtive
        ## floating-point values

        variance = np.maximum(
            variance,
            0
        )

        std = np.sqrt(
            variance
        )

        # avoid division by zero
        std[std == 0] = 1.0

        mean = mean.astype(
            np.float32
        )
        std = std.astype(np.float32)
        

        logging.info(
            f"TRAIN normalization mean: {mean}"
        )

        logging.info(
            f"TRAIN normalization std: {std}"
        )

        return mean, std


    # 5. Save normalization parameters

    def _save_normalization_parameters(
            self,
            mean: np.ndarray,
            std : np.ndarray
    ):
        directory = os.path.dirname(
            self.normalization_path
        )        

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        np.savez(
            self.normalization_path,
            mean = mean,
            std = std
        )

        logging.info(
            "Normalization parameters saved to: "
            f"{self.normalization_path}"
        )

    ## 6. Normalize ECG
    def _normalize_signal(
            self,
            signal: np.ndarray,
            mean: np.ndarray,
            std: np.ndarray,
    ):

        normalized_signal = (
            signal - mean
        ) / std

        return normalized_signal.astype(
            np.float32
        )

    # process splt
    def _process_split(
            self,
            split_df: str,
            mean: np.ndarray,
            std: np.ndarray,
            name: str
    ):

        signals = []
        targets = []

        logging.info(
            f"Processing {name} ECG signals"
        )

        for index, row in split_df.iterrows():
            signal = self._load_ecg_signal(
                row['filename_lr']
            )

            normalized_signal = (
                self._normalize_signal(
                    signal,
                    mean,
                    std
                )
            )

            signals.append(
                normalized_signal
            )

            targets.append(
                int(row['Target'])
            )

            if len(signals) % 1000 == 0:
                logging.info(
                    f"{name}: processed "
                    f"{len(signals)} ECG signals"
                )

        if not signals:
            raise ValueError(
            f"No ECG signals Processed "
            f" for {name}"
        )

        signal_array = np.stack(
            signals
        ).astype(
            np.float32
        )

        target_array = np.asarray(
            targets,
            dtype=np.float32
        )

        logging.info(
            f"{name} signal shape: "
            f"{signal_array.shape}"
        )

        logging.info(
            f"{name} target shape: "
            f"{target_array.shape}"
        )

        return (
            signal_array,
            target_array
        )
        

    # 8. Main preprocessing
    def initiate_data_preprocessing(
            self,
            train_df: pd.DataFrame,
            validation_df: pd.DataFrame,
            test_df: pd.DataFrame
    ):
        try:
            logging.info(
                "Starting ECG data preprocessing"
            )

            # validate all three data sets

            self._validate_dataframe(
                train_df,
                'Train'
            )

            self._validate_dataframe(
                validation_df,
                'Validation'
            )

            self._validate_dataframe(
                test_df,
                "Test"
            )

            logging.info(
                'Input dataframe validation completed'
            )

            # Calculate normalization parameters
            # TRAIN ONLY

            mean, std = (
                self._calculate_normalization_parameters(
                    train_df=train_df
                )
            )

            # Save normalization parameters

            self._save_normalization_parameters(
                mean=mean,
                std=std
            )

            # Apply SAME parameters to all splits

            x_train, y_train = (
                self._process_split(
                    train_df,
                    mean=mean,
                    std=std,
                    name='Train'
                )
            )

            x_validation, y_validation = (
                self._process_split(
                    validation_df,
                    mean=mean,
                    std=std,
                    name = "Validation"
                )
            )

            x_test, y_test = (
                self._process_split(
                    test_df,
                    mean=mean,
                    std=std,
                    name = 'Test'
                )
            )

            logging.info(
                "ECG data preprocessing "
                "completed successfully"
            )

            return (
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                y_test
            )

        except Exception as e:
            logging.exception(
                "Error ocurred during "
                "ECG data preprocessing"
            )

            raise HealthPulseException(
                e,
                sys
            )



                               
       



