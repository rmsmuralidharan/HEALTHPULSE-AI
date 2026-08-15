import os
import numpy as np

from HealthPulse_AI_project.components.data_ingestion import (
    DataIngestion
)

from HealthPulse_AI_project.components.target_creation import (
    TargetCreation
)

from HealthPulse_AI_project.components.data_splitting import (
    DataSplitting
)

from HealthPulse_AI_project.components.metadata_preprocessing import (
    MetaDataPreprocessing
)


if __name__ == "__main__":

    # ==========================================
    # 1. PTB-XL path
    # ==========================================

    data_path = os.path.join(
        "data",
        "raw",
        "ptbxl"
    )

    # ==========================================
    # 2. Data ingestion
    # ==========================================

    ingestion = DataIngestion(
        data_path=data_path
    )

    database_df, scp_df = (
        ingestion.initiate_data_ingestion()
    )

    # ==========================================
    # 3. Target creation
    # ==========================================

    target_creator = TargetCreation()

    target_df = (
        target_creator.initiate_target_creation(
            database_df,
            scp_df
        )
    )

    # ==========================================
    # 4. Patient-level splitting
    # ==========================================

    splitter = DataSplitting(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        random_state=42
    )

    (
        train_df,
        validation_df,
        test_df
    ) = splitter.initiate_data_splitting(
        target_df
    )

    # ==========================================
    # 5. Metadata preprocessing
    # ==========================================

    preprocessing = MetaDataPreprocessing()

    (
        X_train,
        X_validation,
        X_test,
        feature_names
    ) = preprocessing.initiate_metadata_preprocessing(
        train_df,
        validation_df,
        test_df
    )

    # ==========================================
    # 6. Results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("METADATA PREPROCESSING RESULTS")
    print("=" * 60)

    print("\nTRAIN")
    print("-" * 60)

    print(
        f"Processed shape : {X_train.shape}"
    )

    print(
        f"Patients        : "
        f"{train_df['patient_id'].nunique()}"
    )

    print("\nVALIDATION")
    print("-" * 60)

    print(
        f"Processed shape : "
        f"{X_validation.shape}"
    )

    print(
        f"Patients        : "
        f"{validation_df['patient_id'].nunique()}"
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"Processed shape : "
        f"{X_test.shape}"
    )

    print(
        f"Patients        : "
        f"{test_df['patient_id'].nunique()}"
    )

    # ==========================================
    # 7. Feature information
    # ==========================================

    print("\nFEATURES")
    print("-" * 60)

    print(
        f"Number of features: "
        f"{len(feature_names)}"
    )

    print(feature_names)

    # ==========================================
    # 8. Check BMI in feature names
    # ==========================================

    bmi_features = [
        feature
        for feature in feature_names
        if "BMI" in feature
    ]

    print("\nBMI FEATURES")
    print("-" * 60)

    print(bmi_features)

    # ==========================================
    # 9. Validation checks
    # ==========================================

    # ------------------------------------------
    # Shape validation
    # ------------------------------------------

    shape_validation = (
        X_train.shape[0] == len(train_df)
        and
        X_validation.shape[0]
        == len(validation_df)
        and
        X_test.shape[0]
        == len(test_df)
    )

    # ==========================================
    # Feature count validation
    # ==========================================

    feature_count_validation = (
        X_train.shape[1]
        == len(feature_names)
        and
        X_validation.shape[1]
        == len(feature_names)
        and
        X_test.shape[1]
        == len(feature_names)
    )

    # ==========================================
    # BMI feature validation
    # ==========================================

    bmi_validation = (
        len(bmi_features) == 1
    )

    # ==========================================
    # Missing-value validation
    # ==========================================

    missing_value_validation = (
        not np.isnan(X_train).any()
        and
        not np.isnan(X_validation).any()
        and
        not np.isnan(X_test).any()
    )

    # ==========================================
    # NaN / Inf validation
    # ==========================================

    finite_validation = (
        np.isfinite(X_train).all()
        and
        np.isfinite(X_validation).all()
        and
        np.isfinite(X_test).all()
    )

    # ==========================================
    # Transformer validation
    # ==========================================

    transformer_saved = os.path.exists(
        preprocessing.transformer_path
    )

    # ==========================================
    # Imputed dataframe validation
    # ==========================================

    imputed_df_validation = (
        preprocessing.train_imputed_df is not None
        and
        preprocessing.validation_imputed_df is not None
        and
        preprocessing.test_imputed_df is not None
    )

    # ==========================================
    # BMI dataframe validation
    # ==========================================

    bmi_dataframe_validation = (
        "BMI"
        in preprocessing.train_imputed_df.columns
        and
        "BMI"
        in preprocessing.validation_imputed_df.columns
        and
        "BMI"
        in preprocessing.test_imputed_df.columns
    )

    # ==========================================
    # Final validation
    # ==========================================

    overall_validation = (
        shape_validation
        and
        feature_count_validation
        and
        bmi_validation
        and
        missing_value_validation
        and
        finite_validation
        and
        transformer_saved
        and
        imputed_df_validation
        and
        bmi_dataframe_validation
    )

    # ==========================================
    # 10. Validation report
    # ==========================================

    print("\n")
    print("=" * 60)
    print("METADATA PREPROCESSING VALIDATION")
    print("=" * 60)

    print(
        f"Shape validation       : "
        f"{'PASS' if shape_validation else 'FAIL'}"
    )

    print(
        f"Feature count check    : "
        f"{'PASS' if feature_count_validation else 'FAIL'}"
    )

    print(
        f"BMI feature check     : "
        f"{'PASS' if bmi_validation else 'FAIL'}"
    )

    print(
        f"Missing-value check    : "
        f"{'PASS' if missing_value_validation else 'FAIL'}"
    )

    print(
        f"NaN / Inf check        : "
        f"{'PASS' if finite_validation else 'FAIL'}"
    )

    print(
        f"Transformer saved      : "
        f"{'PASS' if transformer_saved else 'FAIL'}"
    )

    print(
        f"Imputed DataFrames     : "
        f"{'PASS' if imputed_df_validation else 'FAIL'}"
    )

    print(
        f"BMI DataFrame check    : "
        f"{'PASS' if bmi_dataframe_validation else 'FAIL'}"
    )

    print(
        f"\nMetadata preprocessing "
        f"validation: "
        f"{'PASS' if overall_validation else 'FAIL'}"
    )

    print("=" * 60)