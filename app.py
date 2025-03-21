from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from arbol_decision import main
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import os

app = Flask(__name__)

modelo = None
df_datos = None
df_riesgos = None
x_train = None
x_test = None
y_test = None

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    modo_oscuro = request.cookies.get('modo_oscuro', 'false')
    return render_template("index.html", modo_oscuro=modo_oscuro)

@app.route('/subir_archivo', methods=['POST'])
def subir_archivo():
    if 'file' not in request.files:
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "El archivo no tiene nombre"}), 400

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    print(f"File saved at: {filepath}")

    return jsonify({"mensaje": "Archivo subido correctamente", "filepath": filepath})

@app.route('/entrenar', methods=['POST'])
def entrenar_modelo():
    global modelo, df_datos, df_riesgos, x_train, x_test, y_test
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({"error": "No se recibió el filepath"}), 400

        if not os.path.exists(filepath):
            return jsonify({"error": "El archivo no existe"}), 400

        modelo, x_train, x_test, y_test = main(filepath)

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

@app.route('/cambiar_modo', methods=['POST'])
def cambiar_modo():
    modo_oscuro = request.json.get('modo_oscuro')
    
    resp = jsonify({'mensaje': 'Modo cambiado'})
    resp.set_cookie('modo_oscuro', modo_oscuro, max_age=60*60*24*30)
    return resp

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)