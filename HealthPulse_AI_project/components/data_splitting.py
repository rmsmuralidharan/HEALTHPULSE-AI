import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging

class DataSplitting:
    def __init__(
            self,
            train_size: float = 0.70,
            validation_size: float = 0.15,
            test_size: float = 0.15,
            random_state: int = 42
    ):
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def log_split_distribution(
            self,
            name: str,
            split_df: pd.DataFrame
    ):
        total = len(split_df)

        if total == 0:
            logging.warning(
                f"{name} dataset is empty"
            )

            return
        target_distribution = (
            split_df['Target']
            .value_counts()
            .sort_index()
        )

        mi_count = int(
            target_distribution.get(1,0)
        )

        non_mi_count = int(
            target_distribution.get(0,0)
        )

        mi_percentage = (
            mi_count / total
        ) * 100

        non_mi_percentage = (
            non_mi_count / total
        ) * 100

        logging.info(
            f"{name} records: {total}"
        )

        logging.info(
            f"{name} patients: "
            f"{split_df['patient_id'].nunique()}"
        )

        logging.info(
            f"{name} MI: "
            f"{mi_count} "
            f"({mi_percentage:.2f}%)"
        )

        logging.info(
            f"{name} Non-MI: "
            f"{non_mi_count} "
            f"({non_mi_percentage:.2f}%)"
        )

    def initiate_data_splitting(
            self,
            df: pd.DataFrame
    ):
        try:
            logging.info(
                "starting patient-level splitting"
            )

            # 1. Validate input

            if df.empty:
                raise ValueError(
                    'Input dataframe is empty'
                )

            required_columns = [
                'patient_id',
                'Target'
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    'Missing required columns '
                    f"{missing_columns}"
                )

            if not abs(
                self.train_size
                + self.validation_size
                + self.test_size - 1.0
            ) < 1e-9:

                raise ValueError(
                    "Train, Validation and test sizes "
                    "must sum to 1.0"
                )

            # 2. Get unique patients

            unique_patients = (
                df['patient_id']
                .dropna()
                .unique()
            )

            logging.info(
                f"Total unique patients: "
                f"{len(unique_patients)}"
            )

            # 3. Train / temporary patient split

            train_patients, temp_patients = (
                train_test_split(
                    unique_patients,
                    test_size=(
                        self.validation_size + self.test_size
                    ),
                    random_state=self.random_state
                )
            )

            # 4. Validation / test patient split

            validation_ratio = (
                self.validation_size
                /
                (
                    self.validation_size + self.test_size
                )
            )

            validation_patients, test_patients = (
                train_test_split(
                    temp_patients,
                    test_size=(
                        1 - validation_ratio
                    ),
                    random_state=self.random_state
                )
            )

            # 5. Create datasets

            train_df = df[
                df['patient_id'].isin(
                    train_patients
                )
            ].copy()

            validation_df = df[
                df['patient_id'].isin(
                    validation_patients
                )
            ].copy()

            test_df = df[
                df['patient_id'].isin(
                    test_patients
                )
            ].copy()

            # 6. Validate record coverage

            total_split_records = (
                len(train_df) + len(validation_df) + len(test_df)
            )

            if total_split_records != len(df):

                raise ValueError(
                    'some records were lost during '
                    'patient-level splitting'
                )

            # 7. Validate patient overlap

            train_patient_set = set(
                train_df['patient_id']
            )

            validation_patient_set = set(
                validation_df['patient_id']
            )

            test_patient_set = set(
                test_df['patient_id']
            )

            train_validation_overlap = (
                train_patient_set
                & 
                validation_patient_set
            )

            train_test_overlap = (
                train_patient_set
                &
                test_patient_set
            )

            validation_test_overlap = (
                validation_patient_set
                &
                test_patient_set
            )

            if(
                train_validation_overlap
                or train_test_overlap
                or validation_test_overlap
            ):
                raise ValueError(
                    'patient overlap detected '
                    'between dataset splits'
                )

            # 8. Log split distribution

            self.log_split_distribution(
                'Train',
                train_df
            )

            self.log_split_distribution(
                'Validation',
                validation_df
            )

            self.log_split_distribution(
                'Test',
                test_df
            )

            # 9. Log shapes

            logging.info(
                f"Train shape: "
                f"{train_df.shape}"
            )

            logging.info(
                f"Validation shape: "
                f"{validation_df.shape}"
            )

            logging.info(
                f"Test shape: "
                f"{test_df.shape}"
            )

            logging.info(
                "Patient-level data splitting "
                "completed successfully"
            )

            return(
                train_df,
                validation_df,
                test_df
            )            
        
        except Exception as e:
            logging.exception(
                'Error occurred during '
                'patient-level data aplitting'
            )

            raise HealthPulseException(e, sys)
