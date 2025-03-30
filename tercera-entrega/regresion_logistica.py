import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


def leer_y_generar_riesgo(datos_csv):
    df = pd.read_csv(datos_csv)

    # calculamos la probabilidad de riesgo
    # usando una combinación de horas trabajadas, ausencias, edad y salario
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


def main_r(ruta_archivo):

    df_datos, df_riesgos, df = leer_y_generar_riesgo(ruta_archivo)

    # utilizamos one hot encoding para convertir la columna de género en variables dummy
    encoder = OneHotEncoder()
    genero_encoded = encoder.fit_transform(df[["Genero"]])

    encoded_df = pd.DataFrame(genero_encoded.toarray(), columns=encoder.get_feature_names_out())

    df_original = df_datos.copy()

    df_datos = pd.concat([df_datos, encoded_df], axis=1)

    # convertimos las columnas de los dataframe a variables dummy
    x = df_datos[["Horas_Trabajadas", "Ausencias", "Genero_Femenino", "Genero_Masculino"]]
    y = df_riesgos["Riesgo"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # normalizamos los datos
    # usamos StandardScaler para normalizar las caracteristicas
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(df_riesgos["Riesgo"].value_counts(normalize=True))

    df_datos = pd.concat([df_datos, encoded_df], axis=1)
    
    print(df_datos.head())
    print(df_riesgos.head())

    # entrenamos el modelo de regresion logistica
    # usando las variables de entrenamiento
    log_reg = LogisticRegression(random_state=0)
    log_reg.fit(x_train, y_train)

    # generamos una variable full que contiene todas las variables
    # para predecir el riesgo del todos los elementos del csv original
    x_full = scaler.transform(x)

    return log_reg, x_train, x_test, y_test, x_full, df_original
