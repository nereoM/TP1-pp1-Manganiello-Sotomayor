from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from arbol_decision import main
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

app = Flask(__name__)

# Utilizamos esta variable para almacenar el modelo y poder predecir cuando queramos.
modelo = None
df_datos = None
df_riesgos = None
x_train = None
x_test = None
y_test = None


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/entrenar', methods=['POST'])
def entrenar_modelo():
    global modelo, df_datos, df_riesgos, x_train, x_test, y_test
    
    try:
        modelo, x_train, x_test, y_test = main()
        return jsonify({"mensaje": "Modelo entrenado correctamente"})
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}"}), 500
    
@app.route('/predecir', methods=['POST'])
def predecir_modelo():
    global modelo, df_datos, df_riesgos, x_train, x_test, y_test
    
    if modelo is None:
        return jsonify({"error": "Primero entrena el modelo"}), 400
 
    prediccion = modelo.predict(x_test).tolist()
    precision = precision_score(y_test, prediccion)
    memoria = recall_score(y_test, prediccion)
    f1 = f1_score(y_test, prediccion)

    return jsonify({
            "predicciones": prediccion,
            "precision": precision,
            "memoria": memoria,
            "f1": f1
        })

if __name__ == '__main__':
    app.run(debug=True)
    
    
