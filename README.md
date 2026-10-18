# phishing-detector

Mecanismo inteligente para detectar, pronosticar y mitigar ataques de phishing en correo electrónico, con Naive Bayes + TF-IDF y un agente orquestador que traduce la probabilidad en una alerta para el usuario. Proyecto académico (CRISP-DM).

Estado: **diseño completo; implementación en curso.** Hecho: cargador de datos, preprocesamiento + TF-IDF, `CorreoElectronico`, exploración (B.2, B.3) y `ModeloNaiveBayes` con evaluación (B.4, B.5). Pendiente: `AgenteOrquestador`, `ModuloAlerta` e interfaz Streamlit (los métodos marcados con `NotImplementedError`).

## Estructura

```
data/raw/          CSV descargados (no se suben a git)
data/processed/    datos limpios generados por el código
models/            modelo entrenado (se regenera)
notebooks/         exploración de datos (CRISP-DM B.2 y B.3)
src/phishing_detector/   código del mecanismo
app/               interfaz Streamlit
tests/             pruebas
docs/              diseño y diagramas
```

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Descarga el dataset (ver `docs/DISENO.md`, sección 3) y guárdalo en `data/raw/`.

## Uso

```bash
python -m pytest -q tests                      # pruebas (usan datos sintéticos)
python scripts/explorar_datos.py               # B.2: resumen, figuras en docs/figuras/
python scripts/explorar_datos.py data/raw/phishing_email.csv
python scripts/entrenar.py                     # B.4/B.5: entrena, evalúa, guarda modelo y figuras
python scripts/entrenar.py --calibrar          # con calibración sigmoide de probabilidades
python scripts/entrenar.py --externo data/raw/<otro_dataset>.csv   # validación externa
```

## Documentación

- `docs/DISENO.md`: decisiones, arquitectura, niveles de riesgo, plan de evaluación y cronograma.
