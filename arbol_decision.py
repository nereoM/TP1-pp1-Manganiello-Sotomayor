import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def leer_y_generar_riesgo(datos_csv):
    df = pd.read_csv(datos_csv)

    # calculamos una probabilidad para calcular el riesgo
    prob_riesgo = (
        0.4 * (df["Horas_Trabajadas"] / df["Horas_Trabajadas"].max()) + 
        0.4 * (df["Ausencias"] / df["Ausencias"].max()) +
        0.1 * (df["Edad"] / df["Edad"].max()) +
        0.1 * (df["Salario"] / df["Salario"].max())
    )

    # si la probabilidad calculada antes es mayor a 0.6, es "alto riesgo"

    df["Riesgo"] = np.where(prob_riesgo > 0.6, 1, 0)

    df_datos = df[["ID", "Horas_Trabajadas", "Ausencias", "Edad", "Salario"]].copy()
    df_riesgos = df[["ID", "Nombre", "Riesgo"]].copy()

    return df_datos, df_riesgos, df

# funcion principal encargada de entrenar el modelo y escalar los datos, utilizando arboles de decision
def main_a(ruta_archivo):

    df_datos, df_riesgos, df = leer_y_generar_riesgo(ruta_archivo)
    print("despues de leer y generar riesgo")
    encoder = OneHotEncoder()
    genero_encoded = encoder.fit_transform(df[["Genero"]])

    encoded_df = pd.DataFrame(genero_encoded.toarray(), columns=encoder.get_feature_names_out())

    df_original = df_datos.copy()

    df_datos = pd.concat([df_datos, encoded_df], axis=1)

    x = df_datos[["Horas_Trabajadas", "Ausencias", "Genero_Femenino", "Genero_Masculino"]]
    y = df_riesgos["Riesgo"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # escalamos los datos de entrada
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(df_riesgos["Riesgo"].value_counts(normalize=True))

    df_datos = pd.concat([df_datos, encoded_df], axis=1)
    
    print(df_datos.head())
    print(df_riesgos.head())

    # entrenamos el modelo de arbol de decision
    # max_depth=2 para evitar sobreajuste
    arbol = DecisionTreeClassifier(max_depth=2, random_state=0)
    arbol.fit(x_train, y_train)

    x_full = scaler.transform(x)

    return arbol, x_train, x_test, y_train, y_test, x_full, df_original, scaler


# funcion encargada de hacer la prediccion individual del empleado que pasamos por parametro
def predecir_riesgo_individual(modelo, scaler, datos_usuario):
    if scaler is None:
        raise ValueError("Error: `scaler` no fue inicializado correctamente en `main_a()`.")

    genero_femenino = 1 if datos_usuario['genero'] == 'Femenino' else 0
    genero_masculino = 1 - genero_femenino

    # creamos un DataFrame con los datos del usuario
    datos_prediccion = pd.DataFrame([[
        datos_usuario['horas_trabajadas'],
        datos_usuario['ausencias'],
        genero_femenino,
        genero_masculino
    ]], columns=["Horas_Trabajadas", "Ausencias", "Genero_Femenino", "Genero_Masculino"])

    print("Datos antes de escalar:", datos_prediccion)
    # escalamos los datos de entrada
    datos_escalados = scaler.transform(datos_prediccion)
    # hacemos la probabilidad de riesgo con el modelo entrenado
    probabilidad = modelo.predict_proba(datos_escalados)[0][1]
    riesgo = 'Alto' if probabilidad > 0.6 else 'Bajo'

    print(f"Predicción realizada: {riesgo} (Probabilidad: {probabilidad:.2f})")
    return riesgo, probabilidad

