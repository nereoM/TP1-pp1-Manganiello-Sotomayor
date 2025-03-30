import pandas as pd

# funcion para transformar los riesgos de un dataframe
# recibe un dataframe con una columna 'Riesgo' que contiene 1 o 0
def transformar_riesgos(df):
    df['Riesgo'] = df['Riesgo'].map({1: 'alto riesgo', 0: 'bajo riesgo'})
    return df