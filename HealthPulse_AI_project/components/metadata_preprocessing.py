import os
import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from HealthPulse_AI_project.exception.exception import HealthPulseException
from HealthPulse_AI_project.logging.logger import logging

class MetaDataPreprocessing:

    def __init__(
            self,
            output_path: str = "project/data/processed/metadata"
    ):
        self.output_path = output_path

        os.makedirs(
            self.output_path,
            exist_ok=True
        )

        self.transformer_path = os.path.join(
            self.output_path,
            "metadata_preprocessor.pkl"
        )

    def initiate_metadata_preprocessing(
            self,
            train_df: pd.DataFrame,
            validation_df: pd.DataFrame,
            test_df: pd.DataFrame
    ):
        try:
            logging.info(
                "Starting metadata preprocessing"
            )

            # 1. Validate input
            if train_df.empty:
                raise ValueError(
                    "training dataframe is empty"
                )

            if validation_df.empty:
                raise ValueError(
                    "validation data frame is empty"
                )

            if test_df.empty:
                raise ValueError(
                    "test dataframe is empty"
                )

            required_columns = [
                "age",
                "sex",
                "height",
                "weight",
                "site",
                "nurse",
                "heart_axis",
                "device",
                "second_opinion"
            ]

            for column in required_columns:
                if column not in train_df.columns:
                    raise ValueError(
                        f"Missing metadata column: {column}"
                    )

            # 2. Select metadata features
            metadata_columns = [
                "age",
                "sex",
                "height",
                "weight",
                "site",
                "nurse",
                "heart_axis",
                "device",
                "second_opinion"
            ]

            x_train = train_df[
                metadata_columns
            ].copy()

            x_validation = validation_df[
                metadata_columns
            ].copy()

            x_test = test_df[
                metadata_columns
            ].copy()

            # 3. Define feature types

            numerical_features = [
                'age',
                'height',
                'weight'
            ]

            categorical_features = [
                'site',
                'nurse',
                'heart_axis',
                'device'
            ]

            binary_features = [
                'sex',
                'second_opinion'
            ]

            # 4. Numerical preprocessing

            numerical_pipeline = Pipeline(
                steps = [
                    (
                        'imputer',
                        SimpleImputer(
                            strategy='median'
                        )
                    ),
                    (
                        'scaler',
                        StandardScaler()
                    )
                ]
            )

            # 5. Categorical preprocessing
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(
                            strategy='most_frequent'
                        )
                    ),
                    (
                        'encoder',
                        OneHotEncoder(
                            handle_unknown='ignore',
                            sparse_output=False
                        )
                    )
                ]
            )

            # 6. Binary preprocessing

            binary_pipeline = Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(
                            strategy='most_frequent'
                        )
                    )
                ]
            )

            # 7. Combined transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'numerical',
                        numerical_pipeline,
                        numerical_features
                    ),
                    (
                        'categorical',
                        categorical_pipeline,
                        categorical_features
                    ),
                    (
                        'binary',
                        binary_pipeline,
                        binary_features
                    )
                ]
            )

            # 8. Fit ONLY on training data

            logging.info(
                "Fitting metadata transformations "
                "using TRAIN data only"
            )

            x_train_processed = (
                preprocessor.fit_transform(
                    x_train
                )
            )

            # 9. Transform validation/test

            x_validation_processed = (
                preprocessor.transform(
                    x_validation
                )
            )

            x_test_processed = (
                preprocessor.transform(
                    x_test
                )
            )

            # 10. Save fitted transformer

            joblib.dump(
                preprocessor,
                self.transformer_path
            )

            logging.info(
                f"metadata preprocessor saved to: "
                f"{self.transformer_path}"
            )

            # 11. Get feature names

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )


            logging.info(
                f"Final metadata feature count: "
                f"{len(feature_names)}"
            )

            logging.info(
                "Metadata preprocessing completed "
                "successfully"
            )

            return (
                x_train_processed,
                x_validation_processed,
                x_test_processed,
                feature_names
            )
        except Exception as e:
            logging.exception(
                "Error ocurred during metadata preprocessing"
            )

            raise HealthPulseException(e, sys)

    