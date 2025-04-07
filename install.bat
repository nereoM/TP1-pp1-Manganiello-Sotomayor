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
    echo Por favor cierra y reabre el terminal para actualizar PATH
    pause
    exit
)

echo Creando entorno virtual...
python -m venv venv
call venv\Scripts\activate

echo Instalando TODAS las dependencias desde requirements.txt...
pip install --no-warn-script-location -r requirements.txt

echo Iniciando la aplicación...
start "" "http://localhost:5000"
python app.py

pause