from src.data_prep import load_and_clean_data, scale_features
from utilities.logger import get_logger
import os

# Inicializamos el logger maestro
logger = get_logger("Main_Orchestrator")

def main():
    logger.info("==========================================")
    logger.info("INICIANDO PIPELINE DE ANÁLISIS MÉDICO MIT")
    logger.info("==========================================")

    data_path = os.path.join("data", "../data/raw/Hipertension_Arterial_Mexico.csv")
    try:
        X_raw, y = load_and_clean_data(data_path)
        X_scaled = scale_features(X_raw)
        logger.info("Verificando consistencia de los datos...")
        if X_scaled.shape[0] == y.shape[0]:
            logger.info(f"ÉXITO: Se procesaron {X_scaled.shape[0]} registros médicos.")
            logger.info(f"Características detectadas: {list(X_scaled.columns)}")
        else:
            logger.warning("ALERTA: Desajuste entre características y etiquetas.")

        print("\n--- Vista previa de los datos procesados (X_scaled) ---")
        print(X_scaled.head())
        print("------------------------------------------------------\n")
    except Exception as e:
        logger.critical(f"Falla catastrófica en el pipeline: {e}", exc_info=True)

    logger.info("PIPELINE FINALIZADO EXITOSAMENTE.")

if __name__ == "__main__":
    main()