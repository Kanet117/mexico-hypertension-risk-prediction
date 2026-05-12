import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utilities.logger import get_logger

logger = get_logger('train')

def train_models(data_path="data/processed/train.csv", model_dir="models/"):
    """
    Trains multiple models and saves them to the models directory.
    """
    logger.info("Starting model training phase...")
    try:
        # Load processed data
        df_train = pd.read_csv(data_path)
        
        target = 'riesgo_hipertension'
        if target not in df_train.columns:
            logger.error(f"Target column '{target}' not found in training data.")
            return False

        X_train = df_train.drop(columns=[target])
        y_train = df_train[target]
        
        # Calculate class weights to handle imbalance
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1
        
        models = {
            'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }

        os.makedirs(model_dir, exist_ok=True)

        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            # Save the model
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
            
        logger.info("Model training completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

if __name__ == "__main__":
    train_models()
