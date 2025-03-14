import pandas as pd
import random

# Generar datos aleatorios para 50 personas
n = 400
data = []

for i in range(1, n + 1):
    nombre = f"Empleado_{i}"
    edad = random.randint(18, 60)  # Edad aleatoria entre 18 y 60
    salario = random.randint(30000, 120000)  # Salario aleatorio entre 30,000 y 120,000
    horas_trabajadas = random.randint(30, 70)  # Horas trabajadas aleatorias entre 30 y 70
    ausencias = random.randint(0, 10)  # Ausencias aleatorias entre 0 y 10
    genero = random.choice(["Masculino", "Femenino"])  # Género aleatorio
    
    data.append([i, nombre, edad, salario, horas_trabajadas, ausencias, genero])

# Crear DataFrame
df = pd.DataFrame(data, columns=["ID", "Nombre", "Edad", "Salario", "Horas_Trabajadas", "Ausencias", "Genero"])

# Guardar en un archivo CSV
csv_file = r'datos\empleados.csv'
df.to_csv(csv_file, index=False)

print(f"Archivo CSV generado: {csv_file}")