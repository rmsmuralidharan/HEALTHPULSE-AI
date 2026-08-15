import os
import sys

# Make project root available
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from HealthPulse_AI_project.components.data_ingestion import (
    DataIngestion
)

from HealthPulse_AI_project.components.target_creation import (
    TargetCreation
)

from HealthPulse_AI_project.components.data_splitting import (
    DataSplitting
)

from HealthPulse_AI_project.components.data_preprocessing import (
    DataPreprocessing
)

from HealthPulse_AI_project.components.metadata_preprocessing import (
    MetaDataPreprocessing
)

from HealthPulse_AI_project.components.ecg_feature_engineering import (
    ECGFeatureEngineering
)

from HealthPulse_AI_project.components.feature_fusion import (
    FeatureFusion
)

from HealthPulse_AI_project.components.model_preparation import (
    ModelPreparation
)

from HealthPulse_AI_project.models.xgboost_model import (
    XGBoostModel
)


if __name__ == "__main__":

    # ==========================================
    # 1. Data path
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
    # 4. Patient-level split
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
    # 5. ECG preprocessing
    # ==========================================

    ecg_preprocessing = DataPreprocessing(
        data_path=data_path
    )

    (
        X_train_ecg,
        y_train,
        X_validation_ecg,
        y_validation,
        X_test_ecg,
        y_test
    ) = ecg_preprocessing.initiate_data_preprocessing(
        train_df,
        validation_df,
        test_df
    )

    # ==========================================
    # 6. ECG Feature Engineering
    # ==========================================

    ecg_fe = ECGFeatureEngineering()

    (
        train_ecg_features,
        validation_ecg_features,
        test_ecg_features,
        ecg_feature_names,
        _,
        _,
        _
    ) = ecg_fe.initiate_ecg_feature_engineering(
        X_train_ecg,
        X_validation_ecg,
        X_test_ecg
    )

    # ==========================================
    # 7. Metadata preprocessing
    # ==========================================

    metadata_preprocessing = (
        MetaDataPreprocessing()
    )

    (
        X_train_metadata,
        X_validation_metadata,
        X_test_metadata,
        metadata_feature_names
    ) = (
        metadata_preprocessing
        .initiate_metadata_preprocessing(
            train_df,
            validation_df,
            test_df
        )
    )

    # ==========================================
    # 8. Feature Fusion
    # ==========================================

    fusion = FeatureFusion()

    (
        train_fused,
        validation_fused,
        test_fused,
        train_target,
        validation_target,
        test_target,
        train_ids,
        validation_ids,
        test_ids
    ) = fusion.initiate_feature_fusion(

        train_ecg_features,
        validation_ecg_features,
        test_ecg_features,

        X_train_metadata,
        X_validation_metadata,
        X_test_metadata,

        y_train,
        y_validation,
        y_test,

        train_df["ecg_id"].to_numpy(),
        validation_df["ecg_id"].to_numpy(),
        test_df["ecg_id"].to_numpy()
    )

    # ==========================================
    # 9. Model preparation
    # ==========================================

    model_preparation = ModelPreparation()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = model_preparation.initiate_model_preparation(
        train_fused,
        train_target,

        validation_fused,
        validation_target,

        test_fused,
        test_target
        )

    # ==========================================
    # 10. XGBoost
    # ==========================================

    xgb_model = XGBoostModel(scale_pos_weight=2.96)

    (
        model,
        validation_metrics,
        test_metrics
    ) = xgb_model.initiate_xgboost_training(

        X_train,
        y_train,

        X_validation,
        y_validation,

        X_test,
        y_test
    )

    # ==========================================
    # 11. Results
    # ==========================================

    print("\n")
    print("=" * 60)
    print("XGBOOST WEIGHTED RESULTS")
    print("=" * 60)

    print("\nMODEL")
    print("-" * 60)

    print(
        f"Number of features : "
        f"{X_train.shape[1]}"
    )

    print(
        f"Training records   : "
        f"{X_train.shape[0]}"
    )

    print("\nXGBOOST WEIGHTED MODEL VALIDATION")
    print("-" * 60)

    print(
        f"Accuracy  : "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"MI Recall : "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{validation_metrics['pr_auc']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        validation_metrics[
            "confusion_matrix"
        ]
    )

    print("\nTEST")
    print("-" * 60)

    print(
        f"Accuracy  : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"MI Recall : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(
        test_metrics[
            "confusion_matrix"
        ]
    )

    # ==========================================
    # 12. Validation
    # ==========================================

    model_validation = (
        model is not None
        and
        X_train.shape[1] == 172
        and
        validation_metrics["roc_auc"] >= 0
        and
        validation_metrics["roc_auc"] <= 1
        and
        test_metrics["roc_auc"] >= 0
        and
        test_metrics["roc_auc"] <= 1
        and
        os.path.exists(
            xgb_model.model_path
        )
    )

    print("\n")
    print("=" * 60)
    print("XGBOOST MODEL VALIDATION")
    print("=" * 60)

    print(
        f"Model creation       : "
        f"{'PASS' if model is not None else 'FAIL'}"
    )

    print(
        f"Feature count        : "
        f"{'PASS' if X_train.shape[1] == 172 else 'FAIL'}"
    )

    print(
        f"Validation metrics   : "
        f"{'PASS' if 0 <= validation_metrics['roc_auc'] <= 1 else 'FAIL'}"
    )

    print(
        f"Test metrics        : "
        f"{'PASS' if 0 <= test_metrics['roc_auc'] <= 1 else 'FAIL'}"
    )

    print(
        f"Model saved          : "
        f"{'PASS' if os.path.exists(xgb_model.model_path) else 'FAIL'}"
    )

    print(
        f"\nXGBoost baseline validation: "
        f"{'PASS' if model_validation else 'FAIL'}"
    )

    print("=" * 60)