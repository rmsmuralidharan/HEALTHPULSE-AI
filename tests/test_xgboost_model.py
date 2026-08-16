from HealthPulse_AI_project.pipelines.model_data_pipeline import ModelDataPipeline

from HealthPulse_AI_project.models.xgboost_model import XGBoostModel

pipeline = ModelDataPipeline()

data = pipeline.initiate_model_data_pipeline()

xgb = XGBoostModel()

results = xgb.initiate_xgboost_training(
    data.X_train,
    data.y_train,
    data.X_validation,
    data.y_validation,
    data.X_test,
    data.y_test
)