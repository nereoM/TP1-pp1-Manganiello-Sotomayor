import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import csv


def leer_y_generar_riesgo(datos_csv):
    df = pd.read_csv(datos_csv)
    prob_riesgo = (
        0.3 * (df["Horas_Trabajadas"] / df["Horas_Trabajadas"].max()) + 
        0.7 * (df["Ausencias"] / df["Ausencias"].max())
    )

    df["Riesgo"] = np.where(prob_riesgo > 0.5, 1, 0)

    df_datos = df[["ID", "Horas_Trabajadas", "Ausencias"]].copy()
    df_riesgos = df[["ID", "Nombre", "Riesgo"]].copy()

    print(df_riesgos["Riesgo"].value_counts())

    return df_datos, df_riesgos


datos_csv = r'C:\Users\Nereo\Desktop\TP1-pp1\datos\empleados.csv'

df_datos, df_riesgos = leer_y_generar_riesgo(datos_csv)

#print(df_datos)
#print(df_riesgos)

x = df_datos[["Horas_Trabajadas", "Ausencias"]]
y = df_riesgos["Riesgo"]

print(df_riesgos["Riesgo"].value_counts(normalize=True))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

log_reg = LogisticRegression(random_state=0)
log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

print('Reales: ', y_test.values.tolist())
print('Prediccion: ', y_pred.tolist())

# Metricas para ver la precision

print('Precision: ', precision_score(y_test, y_pred)) # Porcentaje de acierto en cada vez que compra, 
# es decir, que tan preciso es el modelo al predecir casos positivos.

print('Memoria: ', recall_score(y_test, y_pred)) # Porcentaje de casos positivos identificados correctamente.
print('F1_score: ', f1_score(y_test, y_pred)) # Variable que combina la precision y la memoria.

print(classification_report(y_test, y_pred))

