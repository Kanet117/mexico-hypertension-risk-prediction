import os
from src.data_prep import load_and_clean_data
from src.train import train_models
from src.evaluate import evaluate_models
from utilities.logger import get_logger

# Inicializamos el logger maestro
logger = get_logger("Main_Orchestrator")

def main():
    logger.info("==========================================")
    logger.info("INICIANDO PIPELINE DE ANÁLISIS MÉDICO MIT")
    logger.info("==========================================")

    data_path = os.path.join("data", "raw", "Hipertension_Arterial_Mexico.csv")
    try:
        # 1. Data Preparation
        logger.info("--- Fase 1: Preparación de Datos ---")
        X_train, X_test, y_train, y_test = load_and_clean_data(data_path)
        logger.info(f"ÉXITO: Se procesaron los datos. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # 2. Model Training
        logger.info("\n--- Fase 2: Entrenamiento de Modelos ---")
        train_success = train_models(data_path="data/processed/train.csv", model_dir="models/")
        if train_success:
            logger.info("ÉXITO: Modelos entrenados y guardados correctamente.")
        else:
            logger.error("Fallo en el entrenamiento de modelos.")
            return
            
        # 3. Model Evaluation
        logger.info("\n--- Fase 3: Evaluación de Modelos ---")
        eval_success = evaluate_models(test_data_path="data/processed/test.csv", model_dir="models/", output_dir="output/")
        if eval_success:
            logger.info("ÉXITO: Modelos evaluados y resultados generados.")
        else:
            logger.error("Fallo en la evaluación de modelos.")
            return

    except Exception as e:
        logger.critical(f"Falla catastrófica en el pipeline: {e}", exc_info=True)

    logger.info("==========================================")
    logger.info("PIPELINE FINALIZADO EXITOSAMENTE.")
    logger.info("==========================================")

if __name__ == "__main__":
    main()
