import pickle
import numpy as np
import pandas as pd
import logging


# 🔥 Basic mapping (you can expand later)
SYMPTOM_MAP = {
    "high fever": "fever",
    "feverish": "fever",
    "body pain": "muscle_pain",
    "cold": "chills",
    "head pain": "headache"
}


class PredictPipeline:

    def __init__(self):
        # Load model
        with open("artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load feature names
        with open("artifacts/features.pkl", "rb") as f:
            self.features = pickle.load(f)

        logging.info("Model and features loaded")

    def _preprocess(self, symptoms_list):

        input_vector = [0] * len(self.features)

        for symptom in symptoms_list:
            symptom = symptom.strip().lower().replace(" ", "_")

            # 🔥 fuzzy matching
            for feature in self.features:
                if symptom in feature or feature in symptom:
                    idx = self.features.index(feature)
                    input_vector[idx] = 1

        return input_vector
    def predict(self, symptoms_list):
        try:
            input_vector = self._preprocess(symptoms_list)
            input_df = pd.DataFrame([input_vector], columns=self.features)
            probs = self.model.predict_proba(input_df)
            

            top_3_indices = np.argsort(probs, axis=1)[:, -3:]
            classes = self.model.classes_

            results = []
            for idx in top_3_indices[0][::-1]:
                disease = classes[idx]
                probability = probs[0][idx]
                results.append({
                    "disease": disease,
                    "probability": round(float(probability), 3)
                })

            return results

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise e