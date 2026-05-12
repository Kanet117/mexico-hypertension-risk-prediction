import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from utilities.logger import get_logger

logger = get_logger()

def perform_eda(df: pd.DataFrame, output_dir: str = "output"):
    """
    Realiza Análisis Exploratorio de Datos (EDA).
    Guarda resumen estadístico y gráficos.
    """
    logger.info("Iniciando Exploratory Data Analysis (EDA)...")
    
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Resumen estadístico
    logger.info("Generando resumen estadístico...")
    stats_summary = df.describe(include='all').to_string()
    missing_summary = df.isnull().sum().to_string()
    
    with open(f"{output_dir}/eda_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Resumen Estadístico ===\n")
        f.write(stats_summary)
        f.write("\n\n=== Valores Nulos ===\n")
        f.write(missing_summary)
    
    target_col = 'riesgo_hipertension'
    if target_col in df.columns:
        # Distribución de la variable objetivo
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_col, data=df)
        plt.title('Distribución de Riesgo de Hipertensión')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/target_distribution.png")
        plt.close()

    # Correlaciones (solo numéricas)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/correlation_matrix.png")
        plt.close()
        
    logger.info(f"EDA completado. Resultados guardados en {output_dir}/")

def handle_outliers(df: pd.DataFrame, columns: list, lower_percentile: float = 0.01, upper_percentile: float = 0.99):
    """
    Aplica Winsorization a las columnas numéricas.
    """
    logger.info("Aplicando Winsorization a valores atípicos (outliers)...")
    df_out = df.copy()
    for col in columns:
        lower_limit = df_out[col].quantile(lower_percentile)
        upper_limit = df_out[col].quantile(upper_percentile)
        df_out[col] = np.clip(df_out[col], lower_limit, upper_limit)
    return df_out

def process_data(df: pd.DataFrame):
    """
    Procesa los datos según la estrategia definida.
    """
    logger.info("Iniciando procesamiento de datos...")
    
    df = df.copy()
    
    # Eliminar ID si existe
    if 'FOLIO_I' in df.columns:
        df = df.drop('FOLIO_I', axis=1)
        
    target_col = 'riesgo_hipertension'
    
    if target_col not in df.columns:
        logger.error(f"La columna objetivo '{target_col}' no se encuentra en el dataset.")
        raise ValueError("Dataset sin variable objetivo")
        
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Identificar variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # 'sexo' es usualmente categórica nominal aunque sea numérica
    if 'sexo' in X.columns and 'sexo' not in categorical_cols:
        categorical_cols.append('sexo')
        
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    # 1. Outliers
    X = handle_outliers(X, numeric_cols)
    
    # 2. Imputación
    logger.info("Imputando valores nulos...")
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
        
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        
    # 3. Codificación
    logger.info("Codificando variables categóricas...")
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
    return X, y

def load_and_clean_data(filepath: str = "data/raw/Hipertension_Arterial_Mexico.csv"):
    logger.info(f"Cargando datos desde: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Datos cargados. Forma original: {df.shape}")
    except Exception as e:
        logger.error(f"Error al cargar el archivo: {e}")
        raise
        
    perform_eda(df)
    
    X_processed, y = process_data(df)
    
    # Split
    logger.info("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    # Escalar
    logger.info("Escalando características...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Guardar procesados
    os.makedirs("data/processed", exist_ok=True)
    logger.info("Guardando datasets procesados en data/processed/...")
    
    train_df = pd.concat([X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_scaled.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    logger.info(f"Procesamiento finalizado. Forma Train: {train_df.shape}, Forma Test: {test_df.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    load_and_clean_data()
