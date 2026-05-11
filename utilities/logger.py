import logging
import sys

def get_logger(logger_name="Hypertension_Project"):
    """
    Configura y devuelve un logger profesional nivel producción.
    """
    # 1. Crear el logger
    logger = logging.getLogger(logger_name)
    
    # 2. Evitar que se dupliquen los logs si llamamos a la función varias veces
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.DEBUG) # Nivel base: Captura todo de DEBUG para arriba

    # 3. Crear el formato MIT (Tiempo - Nivel - Archivo:Línea - Mensaje)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 4. Handler 1: Mostrar en la Consola (Pantalla)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # En consola solo vemos lo importante
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 5. Handler 2: Guardar en un Archivo .log (Disco duro)
    file_handler = logging.FileHandler('project_execution.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # En el archivo guardamos TODO para auditoría
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger