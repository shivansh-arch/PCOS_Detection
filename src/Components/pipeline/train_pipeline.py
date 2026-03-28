import src.logger  # initialize logging
import logging

from src.Components.data_transformation import DataTransformation
from src.Components.model_trainerr import ModelTrainer

from src.exception import CustomException
import sys


class TrainPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("🚀 Training pipeline started")

            # 🔹 Step 1: Data Transformation
            data_transformation = DataTransformation()
            X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation()

            logging.info("✅ Data transformation completed")

            # 🔹 Step 2: Model Training
            model_trainer = ModelTrainer()
            model = model_trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )

            logging.info("✅ Model training completed")

            logging.info("🎯 Training pipeline finished successfully")

            return model

        except Exception as e:
            logging.error(f"❌ Error in training pipeline: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()