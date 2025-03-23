import pandas as pd

def transformar_riesgos(df):
    df['Riesgo'] = df['Riesgo'].map({1: 'alto riesgo', 0: 'bajo riesgo'})
    return df