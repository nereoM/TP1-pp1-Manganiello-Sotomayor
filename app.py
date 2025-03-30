from flask import Flask, render_template, request, jsonify, send_file, session
import pickle
import numpy as np
from arbol_decision import main_a
from regresion_logistica import main_r
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import pandas as pd
from transformar_riesgos import transformar_riesgos
from werkzeug.utils import secure_filename
import json
import csv
import hashlib
import os
import time
from werkzeug.utils import secure_filename
import shutil
from generar_csv import generar_csv_empleados
from verificar_columnas import verificar_columnas

# necesario para el ejecutable
try:
    from sklearn.utils._weight_vector import WeightVector
except ImportError:
    pass


app = Flask(__name__)

app.secret_key = 'tu_clave_secreta'

# Variables globales, las usamos para guardar informacion que queremos utilizar en mas de una funcion
modelo_a = None
modelo_r = None
df_datos = None
df_riesgos = None
x_train_a = None
x_test_a = None
y_test_a = None
x_full_a = None
x_train_r = None
x_test_r = None
y_test_r = None 
x_full_r = None
df  = None
df_unido = None

# creacion de la carpeta uploads
output_dir = os.path.join(os.getcwd(), 'downloads')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# funciones utilizadas para rutas

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index')
def index():
    usuario = session.get('usuario', None)
    return render_template('index.html', usuario=usuario)


@app.route("/regresion")
def regresion():
    return render_template("index_regresion.html")


# funcion encargada de verificar que existe el archivo y contiene las columnas requeridas

@app.route('/subir_archivo', methods=['POST'])
def subir_archivo():
    if 'file' not in request.files:
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "El archivo no tiene nombre"}), 400
    
    filename = secure_filename(file.filename)

    try:

        from io import BytesIO
        file_bytes = BytesIO(file.read())
  
        file_bytes.seek(0)
        if not verificar_columnas(file_bytes):
            return jsonify({"error": "Columnas requeridas faltantes"}), 400
        
        file_bytes.seek(0)
        filename = secure_filename(file.filename)
        ruta_final = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        for intento in range(3):
            try:
                with open(ruta_final, 'wb') as f:
                    f.write(file_bytes.getbuffer())
                break
            except PermissionError:
                time.sleep(0.5)
        else:
            raise PermissionError("No se pudo guardar el archivo")
        
        return jsonify({"mensaje": "Archivo subido correctamente", "filepath": ruta_final})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/generar_csv_pruebas', methods=['POST'])
def generar_csv_pruebas():
    try:
        n = request.json.get('n', 50)
        if n < 50:
            return jsonify({"error": "El número de empleados debe ser mayor a 50"}), 400
        generar_csv_empleados(n)
        return jsonify({"mensaje": f"Archivo CSV de prueba generado en {output_dir}"})
    except Exception as e:
        return jsonify({"error": f"Error al generar el CSV de prueba: {str(e)}"}), 500

# funcion para entrenar el modelo utilizando arboles de decision

@app.route('/entrenar_arbol', methods=['POST'])
def entrenar_modelo_arbol():
    global modelo_a, df_datos, df_riesgos, x_train_a, x_test_a, y_test_a, x_full_a
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({"error": "No se recibió el filepath"}), 400
        
        # borra el archivo empleados_con_riesgo
        archivo = os.path.join(output_dir, "empleados_con_riesgo.csv")
        if os.path.exists(archivo):
            if os.path.isfile(archivo):
                os.remove(archivo)
                print("Archivo borrado.")
            else:
                print("La ruta no es un archivo.")
        else:
            print("El archivo no existe.")

        if not os.path.exists(filepath):
            return jsonify({"error": "El archivo no existe"}), 400
        print("antes de llamar a main_a")
        # llamamos a la funcion main de arbol_decision, donde se encuentra la logica principal del entrenamiento del modelo
        modelo_a, x_train_a, x_test_a, y_test_a, x_full_a, df_datos = main_a(filepath)

        return jsonify({"mensaje": "Modelo entrenado correctamente"})
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}"}), 500

    
# funcion para predecir el modelo, utilizando el conjunto de entrenamiento y el completo
    
@app.route('/predecir_arbol', methods=['POST'])
def predecir_modelo_arbol():
    global modelo_a, df_datos, df_riesgos, x_train_a, x_test_a, y_test_a, x_full_a, df, df_unido
    
    if modelo_a is None:
        return jsonify({"error": "Primero entrena el modelo"}), 400
    
    if df is None:
        df = pd.DataFrame()
    try:
        df = pd.DataFrame()
        prediccion = modelo_a.predict(x_test_a)
        print(f"Predicción realizada: {prediccion}")
    except Exception as e:
        print(f"Error al predecir: {e}")
        return jsonify({"error": f"Error al predecir: {str(e)}"}), 500
    
    if "Riesgo" not in df.columns:
        df["Riesgo"] = None

    df["Riesgo"] = modelo_a.predict(x_full_a)

    df_unido = pd.concat([df_datos, df], axis=1)

    print(df_unido.head())

    # metricas para medir el rendimiento del modelo
    precision = precision_score(y_test_a, prediccion, average='binary')
    memoria = recall_score(y_test_a, prediccion, average='binary')
    f1 = f1_score(y_test_a, prediccion, average='binary')

    reiniciar_variables()

    return jsonify({
        "precision": precision,
        "memoria": memoria,
        "f1": f1,
        "dataframe": df_unido.to_dict(orient='records')
    })


# funcion para entrenar el modelo utilizando regresion logistica

@app.route('/entrenar_regresion', methods=['POST'])
def entrenar_modelo_regresion():
    global modelo_r, df_datos, df_riesgos, x_train_r, x_test_r, y_test_r, x_full_r
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({"error": "No se recibió el filepath"}), 400
        
        # borra el archivo empleados_con_riesgo
        archivo = os.path.join(output_dir, "empleados_con_riesgo.csv")
        if os.path.exists(archivo):
            if os.path.isfile(archivo):
                os.remove(archivo)
                print("Archivo borrado.")
            else:
                print("La ruta no es un archivo.")
        else:
            print("El archivo no existe.")

        if not os.path.exists(filepath):
            return jsonify({"error": "El archivo no existe"}), 400

        # llamamos a la funcion main de regresion_logistica, donde se encuentra la logica principal del entrenamiento del modelo

        modelo_r, x_train_r, x_test_r, y_test_r, x_full_r, df_datos = main_r(filepath)

        return jsonify({"mensaje": "Modelo entrenado correctamente"})
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}"}), 500

    
# funcion para predecir el modelo, utilizando el conjunto de entrenamiento y el completo
    
@app.route('/predecir_regresion', methods=['POST'])
def predecir_modelo_regresion():
    global modelo_r, df_datos, df_riesgos, x_train_r, x_test_r, y_test_r, x_full_r, df, df_unido
    
    if modelo_r is None:
        return jsonify({"error": "Primero entrena el modelo"}), 400
    
    if df is None:
        df = pd.DataFrame()
    try:
        df = pd.DataFrame()
        prediccion = modelo_r.predict(x_test_r)
        print(f"Predicción realizada: {prediccion}")
    except Exception as e:
        print(f"Error al predecir: {e}")
        return jsonify({"error": f"Error al predecir: {str(e)}"}), 500
    
    if "Riesgo" not in df.columns:
        df["Riesgo"] = None

    df["Riesgo"] = modelo_r.predict(x_full_r)

    df_unido = pd.concat([df_datos, df], axis=1)

    print(df_unido.head())

    # metricas para medir la eficiencia del modelo

    precision = precision_score(y_test_r, prediccion, average='binary')
    memoria = recall_score(y_test_r, prediccion, average='binary')
    f1 = f1_score(y_test_r, prediccion, average='binary')

    reiniciar_variables()

    return jsonify({
        "precision": precision,
        "memoria": memoria,
        "f1": f1,
        "dataframe": df_unido.to_dict(orient='records')
    })


# funcion para reiniciar las variables globales, para evitar que se acumulen datos de diferentes modelos
def reiniciar_variables():
    global modelo_a, modelo_r, x_train_a, x_test_a, y_test_a, x_full_a
    global x_train_r, x_test_r, y_test_r, x_full_r
    modelo_a = None
    modelo_r = None
    x_train_a = None
    x_test_a = None
    y_test_a = None
    x_full_a = None
    x_train_r = None
    x_test_r = None
    y_test_r = None 
    x_full_r = None

# muestra los resultados de los empleados con riesgo alto, utilizando el archivo generado por la funcion generar_csv
@app.route('/resultados')
def resultados():
    empleados_riesgo = []
    ruta_csv = os.path.join(output_dir, "empleados_con_riesgo.csv")

    if not os.path.exists(ruta_csv):
        return jsonify({"error": "El archivo empleados_con_riesgo.csv no existe. Asegúrese de generar el CSV primero."}), 404

    try:
        with open(ruta_csv, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                print("Claves del CSV:", row.keys())

                if row.get('Riesgo', '').lower() == 'alto riesgo':
                    empleado = {
                        'id': row.get('ID', ''),
                        'horas_trabajadas': row.get('Horas_Trabajadas', ''),
                        'ausencias': row.get('Ausencias', ''),
                        'edad': row.get('Edad', ''),
                        'salario': row.get('Salario', ''),
                        'riesgo': row.get('Riesgo', ''),
                        'avatar_hash': hashlib.md5(row.get('ID', '').encode('utf-8')).hexdigest()
                    }
                    empleados_riesgo.append(empleado)
    except Exception as e:
        return jsonify({"error": f"Error al leer el archivo CSV: {str(e)}"}), 500

    print("Empleados con riesgo alto:", empleados_riesgo)

    return render_template('resultados.html', empleados_riesgo=empleados_riesgo)


# generamos el csv con las predicciones de riesgos

@app.route('/generar_csv', methods=['POST'])
def generar_csv():

    global df_unido
    print("Antes de transformar:", df_unido.head())
    df_transformed = transformar_riesgos(df_unido)
    print("Después de transformar:", df_transformed.head())

    ruta_csv = os.path.join(output_dir, "empleados_con_riesgo.csv")

    if df_transformed.empty:
        return jsonify({"error": "No hay datos para generar el CSV"}), 400

    try:
        df_transformed.to_csv(ruta_csv, index=False)
        return jsonify({"mensaje": "CSV generado con predicciones", "csv_path": ruta_csv})
    except Exception as e:
        return jsonify({"error": f"Error al guardar el archivo: {str(e)}"}), 500


# envia la ruta para descargar el csv con las predicciones
@app.route('/descargar_csv')
def descargar_csv():

    ruta_csv = os.path.join(output_dir, "empleados_con_riesgo.csv")

    if not os.path.exists(ruta_csv):
        return jsonify({"error": "El archivo CSV no existe"}), 404

    return send_file(ruta_csv, as_attachment=True, download_name="empleados_con_riesgo.csv")

# cambia entre el modo oscuro y claro del fondo

@app.route('/cambiar_modo', methods=['POST'])
def cambiar_modo():
    modo_oscuro = request.json.get('modo_oscuro')
    
    resp = jsonify({'mensaje': 'Modo cambiado'})
    resp.set_cookie('modo_oscuro', modo_oscuro, max_age=60*60*24*30)
    return resp

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5000, debug=False)
    #app.run(host='0.0.0.0', port=8080)

  
