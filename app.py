from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from arbol_decision import main
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import os
import pandas as pd
from transformar_riesgos import transformar_riesgos

app = Flask(__name__)

modelo = None
df_datos = None
df_riesgos = None
x_train = None
x_test = None
y_test = None
x_full = None
df = None
df_datos = None
df_unido = None

output_dir = os.path.join(os.getcwd(), 'downloads')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
    global modelo, df_datos, df_riesgos, x_train, x_test, y_test, x_full
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({"error": "No se recibió el filepath"}), 400

        if not os.path.exists(filepath):
            return jsonify({"error": "El archivo no existe"}), 400

        modelo, x_train, x_test, y_test, x_full, df_datos = main(filepath)

        return jsonify({"mensaje": "Modelo entrenado correctamente"})
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}"}), 500

    
@app.route('/predecir', methods=['POST'])
def predecir_modelo():
    global modelo, df_datos, df_riesgos, x_train, x_test, y_test, x_full, df, df_unido
    
    if modelo is None:
        return jsonify({"error": "Primero entrena el modelo"}), 400
    
    if df is None:
        df = pd.DataFrame()
    try:
        prediccion = modelo.predict(x_test)
        print(f"Predicción realizada: {prediccion}")
    except Exception as e:
        print(f"Error al predecir: {e}")
        return jsonify({"error": f"Error al predecir: {str(e)}"}), 500
    
    if "Riesgo" not in df.columns:
        df["Riesgo"] = None

    df["Riesgo"] = modelo.predict(x_full)

    df_unido = pd.concat([df_datos, df], axis=1)

    print(df_unido.head())

    #df_final = transformar_riesgo(df_unido)

    precision = precision_score(y_test, prediccion, average='binary')
    memoria = recall_score(y_test, prediccion, average='binary')
    f1 = f1_score(y_test, prediccion, average='binary')

    return jsonify({
        "predicciones": prediccion.tolist(),
        "precision": precision,
        "memoria": memoria,
        "f1": f1,
        "dataframe": df_unido.to_dict(orient='records')
    })

@app.route('/generar_csv', methods=['POST'])
def generar_csv():

    global df_unido
    df_transformed = transformar_riesgos(df_unido)

    ruta_csv = os.path.join(output_dir, "empleados_con_riesgo.csv")

    if df_transformed.empty:
        return jsonify({"error": "No hay datos para generar el CSV"}), 400

    try:
        df_transformed.to_csv(ruta_csv, index=False)
        return jsonify({"mensaje": "CSV generado con predicciones", "csv_path": ruta_csv})
    except Exception as e:
        return jsonify({"error": f"Error al guardar el archivo: {str(e)}"}), 500


@app.route('/descargar_csv')
def descargar_csv():

    ruta_csv = os.path.join(output_dir, "empleados_con_riesgo.csv")

    if not os.path.exists(ruta_csv):
        return jsonify({"error": "El archivo CSV no existe"}), 404

    return send_file(ruta_csv, as_attachment=True, download_name="empleados_con_riesgo.csv")

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