# Usar una imagen base de Python 3.10.12
FROM python:3.10.12-slim

# Evitar generar archivos .pyc y usar stdout para logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema (incluyendo Bash)
RUN apt-get update && apt-get install -y \
    build-essential \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear un usuario no root con Bash
RUN useradd -m -s /bin/bash appuser

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos e instalar dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Jupyter y agregar el kernel para este usuario
RUN pip install --no-cache-dir jupyter ipykernel \
    && python -m ipykernel install --user --name=python3 --display-name "Python 3 (appuser)"

# Cambiar la propiedad de los archivos al usuario no root
RUN chown -R appuser:appuser /app

# Cambiar al usuario no root
USER appuser

# Exponer el puerto de Jupyter Notebook
EXPOSE 8888

# Definir el comando de inicio del contenedor
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
