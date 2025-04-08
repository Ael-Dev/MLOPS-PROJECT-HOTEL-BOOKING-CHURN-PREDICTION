import os
from pathlib import Path

# Definir la estructura de carpetas
project_structure = {
    "artifacts": ["models", "datasets", "logs"],
    "config": ["__init__.py", "settings.py", "paths.py"],
    "notebook": ["exploration.ipynb", "training.ipynb"],
    "src": {
        "data": ["__init__.py", "preprocessing.py"],
        "models": ["__init__.py", "train.py"],
        "evaluation": ["__init__.py", "metrics.py"]
    },
    "static": ["css", "images"],
    "templates": ["base.html"],
    "utils": ["__init__.py", "helpers.py", "logger.py"]
}

# Función para crear carpetas y archivos
def create_project_structure(base_path, structure):
    for folder, contents in structure.items():
        folder_path = Path(base_path) / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(contents, dict):
            create_project_structure(folder_path, contents)
        elif isinstance(contents, list):
            for item in contents:
                item_path = folder_path / item
                if not item_path.exists():
                    if "." in item:  # Es un archivo
                        item_path.touch()
                    else:  # Es una carpeta
                        item_path.mkdir(parents=True, exist_ok=True)

# Ruta base del proyecto
base_path = "mlops_project" # si quiero en el directorio actual cambiar por . punto

# Crear la estructura del proyecto
create_project_structure(base_path, project_structure)

print(f"Estructura del proyecto {base_path}  creada")
