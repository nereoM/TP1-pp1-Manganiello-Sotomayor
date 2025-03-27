import pandas as pd
import random
import os

n = 400
data = []

for i in range(1, n + 1):
    nombre = f"Empleado_{i}"
    edad = random.randint(18, 60)
    salario = random.randint(30000, 120000)
    horas_trabajadas = random.randint(30, 70)
    ausencias = random.randint(0, 10)
    genero = random.choice(["Masculino", "Femenino"])
    
    data.append([i, nombre, edad, salario, horas_trabajadas, ausencias, genero])

df = pd.DataFrame(data, columns=["ID", "Nombre", "Edad", "Salario", "Horas_Trabajadas", "Ausencias", "Genero"])

ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_archivo = os.path.join(ruta_script, "uploads", "empleados.csv")
df.to_csv(ruta_archivo, index=False)

print(f"Archivo CSV generado: {ruta_archivo}")