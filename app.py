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
    # Verificar si la preferencia de modo oscuro está guardada en la cookie
    modo_oscuro = request.cookies.get('modo_oscuro', 'false')  # 'false' es el valor predeterminado
    return render_template("index.html", modo_oscuro=modo_oscuro)

@app.route('/entrenar', methods=['POST'])
def entrenar_modelo():
    global modelo, df_datos, df_riesgos, x_train, x_test, y_test
    
    try:
        # Asumimos que 'main' es una función que entrena el modelo y devuelve los resultados
        modelo, x_train, x_test, y_test = main()
        return jsonify({"mensaje": "Modelo entrenado correctamente"})
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}"}), 500
    
@app.route('/predecir', methods=['POST'])
def predecir_modelo():
    global modelo, df_datos, df_riesgos, x_train, x_test, y_test
    
    if modelo is None:
        return jsonify({"error": "Primero entrena el modelo"}), 400
    
    # Realizamos la predicción con el modelo entrenado
    prediccion = modelo.predict(x_test).tolist()
    precision = precision_score(y_test, prediccion)
    memoria = recall_score(y_test, prediccion)
    f1 = f1_score(y_test, prediccion)

    # Devolvemos las métricas y las predicciones al cliente
    return jsonify({
        "predicciones": prediccion,
        "precision": precision,
        "memoria": memoria,
        "f1": f1
    })

# Ruta para cambiar el modo oscuro
@app.route('/cambiar_modo', methods=['POST'])
def cambiar_modo():
    # Obtener el estado del modo oscuro desde el cliente
    modo_oscuro = request.json.get('modo_oscuro')
    
    # Guardar el estado en una cookie
    resp = jsonify({'mensaje': 'Modo cambiado'})
    resp.set_cookie('modo_oscuro', modo_oscuro, max_age=60*60*24*30)  # Establecer cookie por 30 días
    return resp

if __name__ == '__main__':
    app.run(debug=True)
