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

        # These will contain the imputed
        # DataFrames after preprocessing.
        self.train_imputed_df = None
        self.validation_imputed_df = None
        self.test_imputed_df = None

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

            # ==========================================
            # 1. Validate input
            # ==========================================

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

                if column not in validation_df.columns:

                    raise ValueError(
                        f"Missing metadata column "
                        f"in validation data: {column}"
                    )

                if column not in test_df.columns:

                    raise ValueError(
                        f"Missing metadata column "
                        f"in test data: {column}"
                    )

            # ==========================================
            # 2. Select metadata features
            # ==========================================

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

            # ==========================================
            # 3. Define feature types
            # ==========================================

            numerical_features = [
                "age",
                "height",
                "weight"
            ]

            categorical_features = [
                "site",
                "nurse",
                "heart_axis",
                "device"
            ]

            binary_features = [
                "sex",
                "second_opinion"
            ]

            # ==========================================
            # 4. Numerical preprocessing
            # ==========================================

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # ==========================================
            # 5. Categorical preprocessing
            # ==========================================

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            # ==========================================
            # 6. Binary preprocessing
            # ==========================================

            binary_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    )
                ]
            )

            # ==========================================
            # 7. Combined transformer
            # ==========================================

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical",
                        numerical_pipeline,
                        numerical_features
                    ),
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_features
                    ),
                    (
                        "binary",
                        binary_pipeline,
                        binary_features
                    )
                ]
            )

            # ==========================================
            # 8. Fit ONLY on training data
            # ==========================================

            logging.info(
                "Fitting metadata transformations "
                "using TRAIN data only"
            )

            x_train_processed = (
                preprocessor.fit_transform(
                    x_train
                )
            )

            # ==========================================
            # 9. Transform validation/test
            # ==========================================

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

            # ==========================================
            # 10. Extract TRAIN-fitted imputers
            # ==========================================

            numerical_fitted_pipeline = (
                preprocessor
                .named_transformers_["numerical"]
            )

            categorical_fitted_pipeline = (
                preprocessor
                .named_transformers_["categorical"]
            )

            binary_fitted_pipeline = (
                preprocessor
                .named_transformers_["binary"]
            )

            numerical_imputer = (
                numerical_fitted_pipeline
                .named_steps["imputer"]
            )

            categorical_imputer = (
                categorical_fitted_pipeline
                .named_steps["imputer"]
            )

            binary_imputer = (
                binary_fitted_pipeline
                .named_steps["imputer"]
            )

            # ==========================================
            # 11. Create imputed DataFrames
            #
            # IMPORTANT:
            # All imputers were fitted on TRAIN.
            # Validation/Test are only transformed.
            # ==========================================

            train_imputed_df = train_df.copy()

            validation_imputed_df = (
                validation_df.copy()
            )

            test_imputed_df = test_df.copy()

            # ------------------------------------------
            # Numerical
            # ------------------------------------------

            train_imputed_df[
                numerical_features
            ] = numerical_imputer.transform(
                train_df[numerical_features]
            )

            validation_imputed_df[
                numerical_features
            ] = numerical_imputer.transform(
                validation_df[numerical_features]
            )

            test_imputed_df[
                numerical_features
            ] = numerical_imputer.transform(
                test_df[numerical_features]
            )

            # ------------------------------------------
            # Categorical
            # ------------------------------------------

            train_imputed_df[
                categorical_features
            ] = categorical_imputer.transform(
                train_df[categorical_features]
            )

            validation_imputed_df[
                categorical_features
            ] = categorical_imputer.transform(
                validation_df[categorical_features]
            )

            test_imputed_df[
                categorical_features
            ] = categorical_imputer.transform(
                test_df[categorical_features]
            )

            # ------------------------------------------
            # Binary
            # ------------------------------------------

            train_imputed_df[
                binary_features
            ] = binary_imputer.transform(
                train_df[binary_features]
            )

            validation_imputed_df[
                binary_features
            ] = binary_imputer.transform(
                validation_df[binary_features]
            )

            test_imputed_df[
                binary_features
            ] = binary_imputer.transform(
                test_df[binary_features]
            )

            # ==========================================
            # 12. Store imputed DataFrames
            # ==========================================

            self.train_imputed_df = (
                train_imputed_df
            )

            self.validation_imputed_df = (
                validation_imputed_df
            )

            self.test_imputed_df = (
                test_imputed_df
            )

            logging.info(
                "Imputed metadata DataFrames "
                "created successfully"
            )

            # ==========================================
            # 13. Validate imputation
            # ==========================================

            for name, df in [
                ("Train", train_imputed_df),
                ("Validation", validation_imputed_df),
                ("Test", test_imputed_df)
            ]:

                missing_count = (
                    df[metadata_columns]
                    .isna()
                    .sum()
                    .sum()
                )

                logging.info(
                    f"{name} metadata missing values "
                    f"after imputation: "
                    f"{missing_count}"
                )

                if missing_count > 0:

                    raise ValueError(
                        f"{name} still contains "
                        "missing metadata values "
                        "after imputation"
                    )

            # ==========================================
            # 14. Save fitted transformer
            # ==========================================

            joblib.dump(
                preprocessor,
                self.transformer_path
            )

            logging.info(
                f"metadata preprocessor saved to: "
                f"{self.transformer_path}"
            )

            # ==========================================
            # 15. Get feature names
            # ==========================================

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            logging.info(
                f"Final metadata feature count: "
                f"{len(feature_names)}"
            )

            # ==========================================
            # 16. Logging
            # ==========================================

            logging.info(
                f"Train metadata shape: "
                f"{x_train_processed.shape}"
            )

            logging.info(
                f"Validation metadata shape: "
                f"{x_validation_processed.shape}"
            )

            logging.info(
                f"Test metadata shape: "
                f"{x_test_processed.shape}"
            )

            logging.info(
                "Metadata preprocessing completed "
                "successfully"
            )

            # ==========================================
            # 17. Keep existing return structure
            # ==========================================

            return (
                x_train_processed,
                x_validation_processed,
                x_test_processed,
                feature_names
            )

        except Exception as e:

            logging.exception(
                "Error occurred during "
                "metadata preprocessing"
            )

            raise HealthPulseException(
                e,
                sys
            )