import pandas as pd
import logging
from sklearn.model_selection import train_test_split

class DataTransformation:

    def __init__(self):
        self.data_path = "data/Final_Augmented_dataset_Diseases_and_Symptoms.csv"

    def initiate_data_transformation(self):
        logging.info("Starting data transformation")

        df = pd.read_csv(self.data_path)

        # ✅ FIX target column (verify once)
        target_col = "diseases"

        # ✅ FIRST separate y
        y = df[target_col]

        # ✅ Remove rare classes BEFORE creating X
        value_counts = y.value_counts()
        valid_classes = value_counts[value_counts >= 2].index

        df = df[df[target_col].isin(valid_classes)]

        # ✅ Now create X and y again (IMPORTANT)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        logging.info(f"Filtered dataset shape: {df.shape}")

        # ✅ Now stratify works safely
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logging.info("Transformation done")

        return X_train, X_test, y_train, y_test