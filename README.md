# LM Conversacional Basico

Proyecto de fine-tuning básico de un modelo de lenguaje conversacional usando Hugging Face Transformers. El ejemplo parte de `distilgpt2` y del dataset público `daily_dialog` para entrenar un modelo causal pequeño orientado a respuestas conversacionales simples.

## Objetivo

Mostrar un flujo mínimo y reproducible para:

- Cargar un modelo base de lenguaje.
- Preparar diálogos como texto de entrenamiento.
- Tokenizar ejemplos con longitud fija.
- Entrenar con `Trainer`.
- Guardar el modelo resultante.
- Probar una generación de texto opcional.

## Estructura

```text
.
├── LM_Conversacional_basico_FIX.ipynb
├── lm_conversacional_basico.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalación

Se recomienda usar un entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Linux o macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Entrenamiento básico:

```bash
python lm_conversacional_basico.py
```

Entrenamiento con menos muestras para una prueba rápida:

```bash
python lm_conversacional_basico.py --train-samples 100 --epochs 1
```

Entrenamiento y prueba de generación:

```bash
python lm_conversacional_basico.py --prompt "Hello, how are you?"
```

## Parámetros principales

| Parámetro | Descripción | Valor por defecto |
| --- | --- | --- |
| `--model-name` | Modelo causal base | `distilgpt2` |
| `--dataset-name` | Dataset de entrenamiento | `daily_dialog` |
| `--train-samples` | Número máximo de diálogos | `1000` |
| `--max-length` | Longitud máxima de tokens | `128` |
| `--epochs` | Épocas de entrenamiento | `1` |
| `--batch-size` | Batch por dispositivo | `2` |
| `--output-dir` | Carpeta de salida del modelo | `trained_model` |

## Notas

El modelo entrenado y los checkpoints no se versionan en Git porque pueden ocupar mucho espacio. Para conservar un resultado, guarda la carpeta indicada en `--output-dir` fuera del repositorio o súbela a un registro de modelos.

## Seguridad

El proyecto no necesita tokens para ejecutarse con los recursos públicos usados por defecto. No guardes credenciales en notebooks, scripts ni salidas de ejecución.
