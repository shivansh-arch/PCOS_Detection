import logging
import numpy as np
import sys
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException


class ModelTrainer:

    def __init__(self):
       self.model = RandomForestClassifier(
    n_estimators=100,        # ↓ from 200
    max_depth=20,           # limit tree depth
    min_samples_split=5,    # prevent overgrowth
    random_state=42,
    n_jobs=-1               # use CPU efficiently
)

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting model training")

            # ✅ Train model
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            
            os.makedirs("artifacts", exist_ok=True)

            with open("artifacts/model.pkl", "wb") as f:
                pickle.dump(self.model, f)

            logging.info("Model saved successfully")

            # 🔥 SAVE FEATURE NAMES (ADD THIS HERE)
            with open("artifacts/features.pkl", "wb") as f:
                pickle.dump(X_train.columns.tolist(), f)

            logging.info("Feature names saved successfully")

            # 🔥 Train accuracy
            y_train_pred = self.model.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)

            # 🔥 Test accuracy
            y_test_pred = self.model.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)

            logging.info(f"Train Accuracy: {train_acc:.4f}")
            logging.info(f"Test Accuracy: {test_acc:.4f}")

            # 🔥 Detailed evaluation
            logging.info("Classification Report:\n" + classification_report(y_test, y_test_pred))

            # 🔥 Top-3 prediction logic
            logging.info("Generating top-3 predictions")

            probs = self.model.predict_proba(X_test)

            # Get indices of top 3 probabilities
            top_3_indices = np.argsort(probs, axis=1)[:, -3:]

            # Map indices to disease names
            classes = self.model.classes_

            top_3_predictions = []
            for row in top_3_indices:
                diseases = [classes[i] for i in row[::-1]]  # reverse for highest first
                top_3_predictions.append(diseases)

            logging.info(f"Sample Top-3 Predictions: {top_3_predictions[:5]}")

            logging.info("Model training and evaluation completed")

            return self.model

        except Exception as e:
            raise CustomException(e, sys)
        