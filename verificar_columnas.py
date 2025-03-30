import pandas as pd

# verifica que el archivo CSV tenga todas las columnas requeridas
def verificar_columnas(file_obj):
    try:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, nrows=1)
        required = ["ID", "Nombre", "Edad", "Salario", "Horas_Trabajadas", "Ausencias", "Genero"]
        return all(col in df.columns for col in required)
    except Exception as e:
        print(f"Error en verificación: {e}")
        return False