import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
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

    ruta_completa = os.path.join('TP1-pp1/imagenes', nombre_archivo)
    plt.savefig(ruta_completa, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Matriz de confusión guardada en '{ruta_completa}'")
    
    return matriz

def guardar_curva_roc(y_pred, y_test, nombre_archivo='curva_roc.png'):
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
    ruta_completa = os.path.join('TP1-pp1/imagenes', nombre_archivo)
    plt.savefig(ruta_completa, bbox_inches='tight')
    plt.close()