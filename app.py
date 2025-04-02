from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, session, send_from_directory
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from arbol_decision import main_a
from regresion_logistica import main_r
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import os
import pandas as pd
from transformar_riesgos import transformar_riesgos
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import csv
import hashlib
from generar_graficos import guardar_matriz_confusion, guardar_curva_roc, guardar_curva_aprendizaje
from generar_csv import generar_csv_empleados
from verificar_columnas import verificar_columnas
import os
import time
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)

app.secret_key = 'tu_clave_secreta'

# Variables globales, las usamos para guardar informacion que queremos utilizar en mas de una funcion
modelo_a = None
modelo_r = None
df_datos = None
df_riesgos = None
x_train_a = None
x_test_a = None
y_train_a = None
y_test_a = None
x_full_a = None
x_train_r = None
x_test_r = None
y_train_r = None
y_test_r = None 
x_full_r = None
df  = None
df_unido = None
columnas_entrenamiento = None


# creacion de la carpeta uploads
output_dir = os.path.join(os.getcwd(), 'downloads')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# lee el historial de personas que subieron archivos
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

# funciones utilizadas para rutas

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
    return render_template('index.html', usuario=usuario)


@app.route("/regresion")
def regresion():
    return render_template("index_regresion.html")

@app.route('/graficos')
def mostrar_graficos():
    return render_template('graficos.html', 
        imagen_matriz="matriz_riesgos.png",
        imagen_curva="curva_roc.png",
        imagen_curva_aprendizaje="curva_aprendizaje.png"
    )


@app.route('/historial', methods=['GET'])
def mostrar_historial():
    historial = leer_historial()
    return render_template('historial.html', historial=historial)



def entrenar_modelo_desde_csv():
    global modelo_a, df_datos, x_train_a, x_test_a, y_train_a, y_test_a, x_full_a, scaler, columnas_entrenamiento, df_50
    filepath = generar_csv_empleados(50)  # Genera el CSV automáticamente
    
    if not filepath:
        return {"error": "No se pudo generar el archivo CSV"}
    
    modelo_a, x_train_a, x_test_a, y_train_a, y_test_a, x_full_a, df_datos = main_a(filepath)
    columnas_a_eliminar = [col for col in ['ID', 'Riesgo'] if col in df_datos.columns]
    columnas_entrenamiento = df_datos.drop(columns=columnas_a_eliminar).columns.tolist()
    scaler = StandardScaler()
    x_train_a = pd.DataFrame(x_train_a, columns=columnas_entrenamiento)  # Asegurar nombres de columnas
    x_train_a_scaled = scaler.fit_transform(x_train_a)  # Entrenar scaler con nombres
    df_50 = df_datos.copy()
    return {"mensaje": "Modelo entrenado correctamente con el CSV generado"}

@app.route('/entrenar_individual', methods=['POST'])
def entrenar_modelo_individual():
    resultado = entrenar_modelo_desde_csv()
    if "error" in resultado:
        return jsonify(resultado), 400
    return jsonify(resultado)

@app.route('/predecir_individual', methods=['GET', 'POST'])
def predecir_individual():
    global modelo_a, scaler, columnas_entrenamiento, df_50

    if request.method == 'GET':
        return render_template('predecir_individual.html')
    
    if modelo_a is None or scaler is None:
        resultado = entrenar_modelo_desde_csv()
        if "error" in resultado:
            return jsonify(resultado), 400

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

    datos_ingresados = {
        "Horas_Trabajadas": horas_trabajadas,
        "Ausencias": ausencias,
        "Edad": edad,
        "Salario": salario,
        "Genero_Femenino": genero_femenino,
        "Genero_Masculino": genero_masculino
    }

    df_individual = pd.DataFrame([datos_ingresados])
    df_individual = df_individual.reindex(columns=columnas_entrenamiento, fill_value=0)  # Asegurar orden y columnas
    
    print("Datos ingresados antes de la predicción:", df_individual)

    try:
        df_individual_normalizado = scaler.transform(df_individual)
        print("Datos normalizados:", df_individual_normalizado)
        prediccion = modelo_a.predict(df_individual_normalizado)
        print("Predicción del modelo:", prediccion)
        prediccion_redondeada = int(prediccion[0])  # Evitar redondeo incorrecto
        resultado = "Alto" if prediccion_redondeada == 1 else "Bajo"
    except Exception as e:
        return jsonify({"error": f"Error en la predicción: {str(e)}"}), 500

    return render_template('predecir_individual.html', resultado=resultado, datos=datos_ingresados, empleados=df_50.to_dict(orient='records'))


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
        input_dir = os.path.join(os.getcwd(), 'uploads')
        return jsonify({"mensaje": f"Archivo CSV de prueba generado en {input_dir}"})
    except Exception as e:
        return jsonify({"error": f"Error al generar el CSV de prueba: {str(e)}"}), 500


# funcion para entrenar el modelo utilizando arboles de decision

@app.route('/entrenar_arbol', methods=['POST'])
def entrenar_modelo_arbol():
    global modelo_a, df_datos, df_riesgos, x_train_a, x_test_a, y_train_a, y_test_a, x_full_a
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({"error": "No se recibió el filepath"}), 400

        if not os.path.exists(filepath):
            return jsonify({"error": "El archivo no existe"}), 400
        print("antes de llamar a main_a")
        # llamamos a la funcion main de arbol_decision, donde se encuentra la logica principal del entrenamiento del modelo
        modelo_a, x_train_a, x_test_a, y_train_a, y_test_a, x_full_a, df_datos = main_a(filepath)

        return jsonify({"mensaje": "Modelo entrenado correctamente"})
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}"}), 500

    
# funcion para predecir el modelo, utilizando el conjunto de entrenamiento y el completo
    
@app.route('/predecir_arbol', methods=['POST'])
def predecir_modelo_arbol():
    global modelo_a, df_datos, df_riesgos, x_train_a, x_test_a, y_train_a, y_test_a, x_full_a, df, df_unido
    
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
    
    guardar_matriz_confusion(prediccion, x_test_a, y_test_a, 
                         clases=['Bajo Riesgo', 'Alto Riesgo'],
                         nombre_archivo='matriz_riesgos.png')
    
    guardar_curva_roc(prediccion, y_test_a, nombre_archivo='curva_roc.png')

    guardar_curva_aprendizaje(modelo_a, x_train_a, x_test_a, y_train=y_train_a, y_test=y_test_a)

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
    global modelo_r, df_datos, df_riesgos, x_train_r, x_test_r, y_train_r, y_test_r, x_full_r
    try:
        data = request.get_json()
        filepath = data.get('filepath')

        if not filepath:
            return jsonify({"error": "No se recibió el filepath"}), 400

        if not os.path.exists(filepath):
            return jsonify({"error": "El archivo no existe"}), 400

        # llamamos a la funcion main de regresion_logistica, donde se encuentra la logica principal del entrenamiento del modelo

        modelo_r, x_train_r, x_test_r, y_train_r, y_test_r, x_full_r, df_datos = main_r(filepath)

        return jsonify({"mensaje": "Modelo entrenado correctamente"})
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}"}), 500

    
# funcion para predecir el modelo, utilizando el conjunto de entrenamiento y el completo
    
@app.route('/predecir_regresion', methods=['POST'])
def predecir_modelo_regresion():
    global modelo_r, df_datos, df_riesgos, x_train_r, x_test_r, y_train_r, y_test_r, x_full_r, df, df_unido
    
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
    
    guardar_matriz_confusion(prediccion, x_test_r, y_test_r, 
                         clases=['Bajo Riesgo', 'Alto Riesgo'],
                         nombre_archivo='matriz_riesgos.png')
    
    guardar_curva_roc(prediccion, y_test_r, nombre_archivo='curva_roc.png')

    guardar_curva_aprendizaje(modelo_r, x_train_r, x_test_r, y_train=y_train_r, y_test=y_test_r)

    reiniciar_variables()

    return jsonify({
        "precision": precision,
        "memoria": memoria,
        "f1": f1,
        "dataframe": df_unido.to_dict(orient='records')
    })


# funcion para reiniciar las variables globales para evitar que se acumulen datos de diferentes modelos
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


# lee del csv generado todos los empleados que sean de alto riesgo
# para luego guardarlos y enviarlos al front

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


# envia la ruta para descargar el csv con las predicciones

@app.route('/descargar_csv')
def descargar_csv():

    ruta_csv = os.path.join(output_dir, "empleados_con_riesgo.csv")

    if not os.path.exists(ruta_csv):
        return jsonify({"error": "El archivo CSV no existe"}), 404

    return send_file(ruta_csv, as_attachment=True, download_name="empleados_con_riesgo.csv")

@app.route('/limpiar_historial', methods=['POST'])
def limpiar_historial():
    try:
        with open('historial.json', 'w') as archivo:
            json.dump([], archivo, indent=4)
        return jsonify({"mensaje": "Historial limpiado correctamente"}), 200
    except Exception as e:
        return jsonify({"error": f"Error al limpiar el historial: {str(e)}"}), 500


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
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)

  
