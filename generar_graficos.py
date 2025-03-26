import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import seaborn as sns

def guardar_matriz_confusion(y_pred, x_test, y_test, clases=None, nombre_archivo='matriz_riesgos.png'):

    os.makedirs('imagenes', exist_ok=True)

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

    ruta_completa = os.path.join('imagenes', nombre_archivo)
    plt.savefig(ruta_completa, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Matriz de confusión guardada en '{ruta_completa}'")
    
    return matriz