import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import os
import seaborn as sns
import pandas as pd
import numpy as np

# funcion encargada de generar la imagen de la matriz de confusión y guardarla en el directorio correspondiente
def guardar_matriz_confusion(y_pred, x_test, y_test, clases=None, nombre_archivo='matriz_riesgos.png'):

    IMG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'imagenes')

    os.makedirs(IMG_FOLDER, exist_ok=True)

    matriz = confusion_matrix(y_test, y_pred)

    if clases is None:
        clases = ['Bajo Riesgo', 'Alto Riesgo']

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clases, 
                yticklabels=clases)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicción')

    ruta_completa = os.path.join(IMG_FOLDER, nombre_archivo)
    plt.savefig(ruta_completa, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Matriz de confusión guardada en '{ruta_completa}'")
    
    return matriz

# funcion encargada de generar la imagen de la curva ROC y guardarla en el directorio correspondiente
def guardar_curva_roc(y_pred, y_test, nombre_archivo='curva_roc.png'):
    IMG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'imagenes')
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    ruta_completa = os.path.join(IMG_FOLDER, nombre_archivo)
    plt.savefig(ruta_completa, bbox_inches='tight')
    plt.close()

# funcion encargada de generar la imagen de la curva de aprendizaje y guardarla en el directorio correspondiente
# esta funcion genera una curva de aprendizaje para un modelo dado, utilizando los datos de entrenamiento y prueba proporcionados.
def guardar_curva_aprendizaje(y_pred, x_train, x_test, y_train, y_test):
    IMG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'imagenes')

    X = pd.concat([
        pd.DataFrame(x_train) if not isinstance(x_train, pd.DataFrame) else x_train,
        pd.DataFrame(x_test) if not isinstance(x_test, pd.DataFrame) else x_test
    ])
    
    y = pd.concat([
        pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train,
        pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test
    ])

    n_samples = len(X)
    cv = min(5, max(2, n_samples // 2))

    train_sizes, train_scores, test_scores = learning_curve(
    estimator=y_pred,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Entrenamiento')
    plt.plot(train_sizes, test_mean, color='green', marker='s', label='Validación')
    plt.title('Curva de Aprendizaje - Árbol de Decisión')
    plt.xlabel('Cantidad de Ejemplos de Entrenamiento')
    plt.ylabel('Precisión')
    plt.grid(True)
    plt.legend(loc='lower right')
    ruta_completa = os.path.join(IMG_FOLDER, 'curva_aprendizaje.png')
    plt.savefig(ruta_completa)
    plt.close()