import pandas as pd


def verificar_columnas(ruta_archivo):
    try:
        with open(ruta_archivo, 'r') as f:
            df = pd.read_csv(ruta_archivo)
            columnas_requeridas = ["ID", "Nombre", "Edad", "Salario", "Horas_Trabajadas", "Ausencias", "Genero"]
            columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
            if columnas_faltantes:
                print(f"Faltan las siguientes columnas: {', '.join(columnas_faltantes)}")
                return False
            else:
                print("Todas las columnas requeridas están presentes.")
                return True
    except FileNotFoundError:
        print(f"El archivo {ruta_archivo} no se encuentra.")
        return False