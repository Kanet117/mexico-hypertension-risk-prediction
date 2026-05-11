import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utilities.logger import get_logger

logger = get_logger()

def load_and_clean_data(filepath: str = "../data/raw/Hipertension_Arterial_Mexico.csv"):
    """
    Separa los datasets, separa las variables y trata los valores nulos
    """
    logger.info(f"Cargando datos desde: {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Datos cargados exitosamente.\nForma:\n{df.shape}")
    except:
        logger.error(f"No se encontro el archivo en la ruta: {filepath}")
        raise
    
    target_col = 'riesgo_hipertension'

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    logger.info("Rellenando valores nulos usando la mediana.")

    imputer = SimpleImputer(strategy="median")
 
    X_cleaned = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_cleaned, y

def scale_features(X_cleaned):
    """
    Estandariza las características para que todos tengan la misma escala.
    """
    logger.info("Escalando las caracteristicas numericas ...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_cleaned), columns=X_cleaned.columns)

    return X_scaled


