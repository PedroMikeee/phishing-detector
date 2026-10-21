# phishing-detector

Mecanismo inteligente para detectar, pronosticar y mitigar ataques de phishing en correo electrónico, con Naive Bayes + TF-IDF y un agente orquestador que traduce la probabilidad en una alerta para el usuario. Proyecto académico (CRISP-DM).

## Estado (19 oct 2026)

| Parte | Estado |
|---|---|
| Carga y limpieza de datos, preprocesamiento + TF-IDF (B.2, B.3) | Hecho |
| `ModeloNaiveBayes`, evaluación, entrenamiento (B.4, B.5) | Hecho |
| Validación externa con el dataset de Zenodo | Hecho |
| Detección de idioma y reglas de riesgo: enlaces, remitente, urgencia, datos sensibles (D7, D8) | Hecho; falta medirlo con datos reales |
| `AgenteOrquestador` y `ModuloAlerta` (B.6) | Hecho |
| Actualización del modelo con ejemplos nuevos (D9) | Hecho (el vocabulario queda fijo) |
| Interfaz Streamlit (B.6) | Pendiente |

Trazabilidad con el anteproyecto y decisiones pendientes: `docs/DISENO.md`, secciones 2 y 11.

## Estructura

```
data/raw/                CSV descargados (no se suben a git)
data/processed/          salidas intermedias generadas por el código (no se suben)
models/                  modelo entrenado (se regenera con scripts/entrenar.py)
src/phishing_detector/   código del mecanismo (una clase por archivo, ver docs/DISENO.md §4)
scripts/                 exploración (B.2), entrenamiento y evaluación (B.4, B.5), análisis y actualización
tests/                   pruebas con datos sintéticos
docs/DISENO.md           diseño, decisiones, resultados y limitaciones
docs/figuras/            figuras generadas por los scripts
docs/resultados/         registros y métricas de cada corrida
```

La carpeta `app/` (interfaz Streamlit) se creará en la fase de despliegue.

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

Los CSV no se suben al repositorio; se descargan una vez y se guardan en `data/raw/`.

| Archivo | Uso | Fuente |
|---|---|---|
| `phishing_email.csv` | Entrenamiento y evaluación (82.486 filas; 82.077 tras limpiar) | Kaggle, "Phishing Email Dataset" |
| `Phishing_validation_emails.csv` | Validación externa (2.000 filas, solo 100 textos únicos) | <https://zenodo.org/records/13474746> (CC BY 4.0) |

Detalles, hallazgos y límites de cada dataset: `docs/DISENO.md`, sección 3. Importante: la clase "maliciosa" de `phishing_email.csv` mezcla spam, estafas y texto sin sentido; no es phishing puro.

## Uso

```bash
python -m pytest -q tests                      # pruebas (usan datos sintéticos)
python scripts/explorar_datos.py data/raw/phishing_email.csv   # B.2: resumen y figuras
python scripts/entrenar.py                     # B.4/B.5: entrena, evalúa, guarda modelo y figuras
python scripts/entrenar.py --calibrar          # con calibración sigmoide de probabilidades
python scripts/entrenar.py --externo data/raw/Phishing_validation_emails.csv   # validación externa
python scripts/probar_ejemplos.py              # modelo solo vs. mecanismo en correos escritos a mano (ilustrativo)
python scripts/evaluar_mecanismo.py            # mide si las reglas y el aviso de idioma mejoran al modelo solo
python scripts/analizar_correo.py --texto "..."  # alerta completa de un correo (o un archivo, o por stdin)
python scripts/actualizar_modelo.py data/raw/correos_nuevos.csv   # incorpora ejemplos etiquetados (D9)
```

## Resultados iniciales

| | Prueba 20 % del dataset | Validación externa (n = 100) |
|---|---|---|
| Precisión | 98,9 % | 52,5 % |
| Sensibilidad | 94,2 % | 91,3 % |
| AUC-ROC | 0,997 | 0,919 |
| R² de la probabilidad | 0,891 | 0,255 |

El modelo separa muy bien el corpus donde se entrenó, pero fuera de él alarma de más con correos legítimos: los términos que más pesan son rasgos de las fuentes (Enron, listas de correo, spam de relojes), no señales de phishing. Detalle y pruebas manuales: `docs/DISENO.md`, sección 8.

## Limitaciones conocidas

- Entrenado solo con inglés. En otro idioma el mecanismo no usa la probabilidad del modelo y lo avisa; solo quedan las reglas.
- El modelo es de bolsa de palabras: no ve remitente ni enlaces (el dataset vino sin URLs) ni entiende contexto. Esa parte la cubren las reglas, que son heurísticas sin entrenar.
- La actualización con ejemplos nuevos no amplía el vocabulario: las palabras nuevas se ignoran hasta reentrenar.
- Los datasets públicos son antiguos y no representan el phishing actual.
