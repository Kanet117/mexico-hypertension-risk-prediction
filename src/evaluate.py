import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, confusion_matrix
from utilities.logger import get_logger

logger = get_logger('evaluate')

def evaluate_models(test_data_path="data/processed/test.csv", model_dir="models/", output_dir="output/"):
    """
    Evaluates trained models on the test dataset and generates metrics and plots.
    """
    logger.info("Starting model evaluation phase...")
    try:
        df_test = pd.read_csv(test_data_path)
        
        target = 'riesgo_hipertension'
        if target not in df_test.columns:
            logger.error(f"Target column '{target}' not found in test data.")
            return False

        X_test = df_test.drop(columns=[target])
        y_test = df_test[target]
        
        models = {}
        for file in os.listdir(model_dir):
            if file.endswith(".pkl"):
                model_name = file.replace(".pkl", "")
                models[model_name] = joblib.load(os.path.join(model_dir, file))
        
        if not models:
            logger.error("No models found in the models directory.")
            return False

        eval_results = []
        
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred # Fallback if no predict_proba
            
            # Metrics
            roc_auc = roc_auc_score(y_test, y_prob)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            eval_results.append({
                "Model": model_name,
                "ROC-AUC": roc_auc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Accuracy": accuracy
            })
            
            # ROC Curve plotting
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(output_dir, f"plots/confusion_matrix_{model_name}.png"))
            plt.close()
            
        # Finish ROC Plot
        plt.figure(1) # Go back to the ROC curve figure
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "plots/roc_curves.png"))
        plt.close()
        
        # Save evaluation results
        eval_df = pd.DataFrame(eval_results)
        
        with open(os.path.join(output_dir, "model_evaluation.txt"), "w") as f:
            f.write("Model Evaluation Results\n")
            f.write("="*30 + "\n\n")
            f.write(eval_df.to_string(index=False))
            
            best_model = eval_df.loc[eval_df['ROC-AUC'].idxmax()]
            f.write("\n\n" + "="*30 + "\n")
            f.write(f"Best Model (by ROC-AUC): {best_model['Model']}\n")
            f.write(f"ROC-AUC: {best_model['ROC-AUC']:.4f}\n")
            f.write(f"Recall: {best_model['Recall']:.4f}\n")
            f.write(f"F1-Score: {best_model['F1-Score']:.4f}\n")
            
        logger.info("Model evaluation completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return False

if __name__ == "__main__":
    evaluate_models()