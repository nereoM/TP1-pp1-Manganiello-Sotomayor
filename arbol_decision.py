import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns


def leer_y_generar_riesgo(datos_csv):
    df = pd.read_csv(datos_csv)

    # Normalizar las variables y calcular riesgo
    prob_riesgo = (
        0.2 * (df["Horas_Trabajadas"] / df["Horas_Trabajadas"].max()) + 
        0.5 * (df["Ausencias"] / df["Ausencias"].max()) +
        0.2 * (df["Edad"] / df["Edad"].max()) +
        0.1 * (df["Salario"] / df["Salario"].max())
    )

    df["Riesgo"] = np.where(prob_riesgo > 0.5, 1, 0)

    df_datos = df[["ID", "Horas_Trabajadas", "Ausencias", "Edad", "Salario"]].copy()
    df_riesgos = df[["ID", "Nombre", "Riesgo"]].copy()

    return df_datos, df_riesgos, df


datos_csv = r'C:\Users\Nereo\Desktop\TP1-pp1\datos\empleados.csv'

df_datos, df_riesgos, df = leer_y_generar_riesgo(datos_csv)

#print(df_datos)
#print(df_riesgos)
encoder = OneHotEncoder()
genero_encoded = encoder.fit_transform(df[["Genero"]])  # Suponiendo que la columna se llama "Genero"

encoded_df = pd.DataFrame(genero_encoded.toarray(), columns=encoder.get_feature_names_out())

df_datos = pd.concat([df_datos, encoded_df], axis=1)

x = df_datos[["Horas_Trabajadas", "Ausencias", "Genero_Femenino", "Genero_Masculino"]]
y = df_riesgos["Riesgo"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(df_riesgos["Riesgo"].value_counts(normalize=True))

df_datos = pd.concat([df_datos, encoded_df], axis=1)

arbol = DecisionTreeClassifier(max_depth=2, random_state=0)
arbol.fit(x_train, y_train)

y_pred_arbol = arbol.predict(x_test)

print('Reales:     ', y_test.values.tolist())
print('Prediccion: ', y_pred_arbol.tolist())

print('Precision: ', precision_score(y_test, y_pred_arbol))
print('Memoria: ', recall_score(y_test, y_pred_arbol))
print('F1_score: ', f1_score(y_test, y_pred_arbol))
print(classification_report(y_test, y_pred_arbol))

"""
cm = confusion_matrix(y_test, y_pred_arbol)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Bajo Riesgo", "Alto Riesgo"], yticklabels=["Bajo Riesgo", "Alto Riesgo"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()"
"""