import csv
import random
import os
import time
import tempfile
import shutil

# funcion para generar un archivo CSV con datos de empleados 
def generar_csv_empleados(n):
    try:
        data = []
        for i in range(1, n + 1):
            nombre = f"Empleado_{i}"
            edad = random.randint(18, 60)
            salario = random.randint(30000, 120000)
            horas_trabajadas = random.randint(30, 70)
            ausencias = random.randint(0, 10)
            genero = random.choice(["Masculino", "Femenino"])
            data.append([i, nombre, edad, salario, horas_trabajadas, ausencias, genero])

        uploads_dir = os.path.abspath("uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        max_intentos = 5
        for intento in range(max_intentos):
            try:
                # utilizamos with para evitar que el archivo quede abierto
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    delete=False, 
                    suffix='.csv', 
                    dir=uploads_dir, 
                    newline=''
                ) as tmp:
                    writer = csv.writer(tmp)
                    writer.writerow(["ID", "Nombre", "Edad", "Salario", "Horas_Trabajadas", "Ausencias", "Genero"])
                    writer.writerows(data)
                    temp_path = tmp.name

                nombre_final = f"empleados.csv"
                ruta_final = os.path.join(uploads_dir, nombre_final)
                shutil.move(temp_path, ruta_final)
                return ruta_final
                
            except PermissionError:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                time.sleep(0.5 * (intento + 1))
        raise PermissionError("No se pudo generar el archivo después de varios intentos")
    
    except Exception as e:
        print(f"Error crítico: {e}")
        return None