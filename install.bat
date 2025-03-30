@echo off
:: --------------------------------------------------
:: INSTALADOR AUTOMÁTICO PARA APLICACIÓN FLASK
:: --------------------------------------------------
:: Requiere Windows 10/11 con conexión a Internet

echo Verificando Python 3.12...
python --version 2>nul | find "3.12" >nul
if %errorlevel% neq 0 (
    echo Instalando Python 3.12...
    winget install --silent --accept-package-agreements Python.Python.3.12
    setx PATH "%PATH%;%LOCALAPPDATA%\Programs\Python\Python312"
)

echo Creando entorno virtual...
python -m venv venv
call venv\Scripts\activate

echo Instalando dependencias exactas...
pip install --no-warn-script-location flask==2.3.2 scikit-learn==1.5.2 scipy==1.11.4 pandas==2.1.4 numpy==1.26.0

echo Iniciando la aplicación...
start "" "http://localhost:5000"
python app.py

pause

if errorlevel 1 (
    echo Instalando desde requirements.txt...
    pip install -r requirements.txt
    python app.py
)