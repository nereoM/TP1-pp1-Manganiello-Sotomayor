from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, session
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from arbol_decision import main
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import os
import pandas as pd
from transformar_riesgos import transformar_riesgos
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import csv
import hashlib

app = Flask(__name__)

app.secret_key = 'tu_clave_secreta'


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

def leer_historial():
    try:
        with open('historial.json', 'r') as archivo:
            contenido = archivo.read().strip()
            if not contenido:
                return []
            return json.loads(contenido)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []



def guardar_historial(historial):
    try:
        with open('historial.json', 'w') as archivo:
            json.dump(historial, archivo, indent=4)
        print("Historial guardado correctamente")
    except Exception as e:
        print(f"Error al guardar el historial: {e}")


def agregar_entrada(usuario, archivo=None):
    historial = leer_historial()
    nueva_entrada = {
        "usuario": usuario,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if archivo:
        nueva_entrada["archivo"] = archivo
    
    historial.append(nueva_entrada)
    guardar_historial(historial)



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['usuario'] = request.form['nombre_usuario']
        print(f"Usuario ingresado: {session['usuario']}")
        agregar_entrada(session['usuario'])
    usuario = session.get('usuario', None)
    return render_template('inicio.html', usuario=usuario)





@app.route('/logout')
def logout():
    session.pop('usuario', None)
    return redirect(url_for('home'))




@app.route('/index')
def index():
    usuario = session.get('usuario', None)
    return render_template("index.html", usuario=usuario)




@app.route('/historial', methods=['GET'])
def mostrar_historial():
    historial = leer_historial()
    return render_template('historial.html', historial=historial)


@app.route('/predecir_individual', methods=['GET', 'POST'])
def predecir_individual():
    global modelo, x_train, x_test, y_test, x_full, df_datos

    if modelo is None:
        FILEPATH = 'C:/Users/nazar/TP1-pp1/datos/empleados.csv'

        if not os.path.exists(FILEPATH):
            return jsonify({"error": f"El archivo {FILEPATH} no existe"}), 500
        
        try:
            modelo, x_train, x_test, y_test, x_full, df_datos = main(FILEPATH)
        except Exception as e:
            return jsonify({"error": f"No se pudo entrenar el modelo: {str(e)}"}), 500

    resultado = None  # Inicializamos el resultado fuera de la lógica del POST

    if request.method == 'POST':
        horas_trabajadas = request.form.get('horas_trabajadas')
        ausencias = request.form.get('ausencias')
        edad = request.form.get('edad')
        salario = request.form.get('salario')
        genero = request.form.get('genero')

        if not all([horas_trabajadas, ausencias, edad, salario, genero]):
            return jsonify({"error": "Datos incompletos"}), 400

        try:
            horas_trabajadas = float(horas_trabajadas)
            ausencias = int(ausencias)
            edad = int(edad)
            salario = float(salario)
        except ValueError:
            return jsonify({"error": "Los datos deben ser numéricos"}), 400

        if genero not in ["Femenino", "Masculino"]:
            return jsonify({"error": "Género inválido"}), 400

        genero_femenino = 1 if genero == "Femenino" else 0
        genero_masculino = 1 if genero == "Masculino" else 0

        df_individual = pd.DataFrame([[horas_trabajadas, ausencias, genero_femenino, genero_masculino]], 
                                     columns=["Horas_Trabajadas", "Ausencias", "Genero_Femenino", "Genero_Masculino"])

        if not hasattr(predecir_individual, "scaler"):
            scaler = StandardScaler()
            df_individual_normalizado = scaler.fit_transform(df_individual)
            predecir_individual.scaler = scaler
        else:
            scaler = predecir_individual.scaler
            df_individual_normalizado = scaler.transform(df_individual)

        prediccion = modelo.predict(df_individual_normalizado)
        resultado = "Alto" if prediccion[0] == 1 else "Bajo"

    return render_template('predecir_individual.html', resultado=resultado)


@app.route('/subir_archivo', methods=['POST'])
def subir_archivo():
    if 'file' not in request.files:
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "El archivo no tiene nombre"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        print(f"File saved at: {filepath}")
    except Exception as e:
        return jsonify({"error": f"No se pudo guardar el archivo: {str(e)}"}), 500

    usuario = session.get('usuario', None)
    if usuario:
        try:
            agregar_entrada(usuario, filename)
            print(f"Entrada agregada al historial: {usuario} subió el archivo {filename}")
        except Exception as e:
            return jsonify({"error": f"No se pudo agregar al historial: {str(e)}"}), 500

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
                        'horas_trabajadas': row.get('Horas Trabajadas', ''),
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

  
