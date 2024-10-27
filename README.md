# Trabajo Fin de Máster - Miguel Ángel Benítez Alguacil

Este repositorio contiene mi Trabajo Fin de Máster para el Máster Universitario Oficial en Ciencia de Datos e Ingeniería de Computadores de la Universidad de Granada.

## Instalación

Para instalar el proyecto, sigue estos pasos:

1. Crea un entorno virtual:
   ```bash
   python -m venv venv

2. Activar el entorno virtual:
    ```bash
    venv\Scripts\activate       # Windows
    source venv/bin/activate    # Linux / macOS

3. Instalar dependencias del proyecto:
    ```bash
    pip install -r requirements.txt

4. (Opcional) Configurar pre-commit:
    ```bash
    pre-commit install

## Ejecutar el proyecto
Para iniciar el servidor de desarrollo, utiliza el siguiente comando:

    uvicorn app.main:app --reload

Esto iniciará la API en modo de recarga automática, lo que significa que cualquier cambio en el código se reflejará instantáneamente sin necesidad de reiniciar el servidor manualmente.

## Documentación
Una vez que el servidor esté en funcionamiento, puedes acceder a la documentación interactiva de la API a través de:

- Swagger UI: http://127.0.0.1:8000/docs
