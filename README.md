# Predicción de duración de autocallables

Modelo que predice `avg_duration_months`, la duración media real de un autocallable
hasta su cancelación anticipada o vencimiento, a partir del histórico de solicitudes
de cotización (RFQs), mercado y referencia de subyacentes.

## Estructura del repositorio

```
.
├── data/
│   ├── rfqs.csv                    
│   ├── daily_volatility.csv
│   ├── underlyings_reference.csv
│   └── processed_features.parquet   # generado por preproceso.ipynb
├── artifacts/
│   ├── model.pkl                    # modelo entrenado (XGBoost)
│   └── model_columns.pkl            # columnas exactas vistas en entrenamiento
├── notebooks/
│   ├── preproceso.ipynb             # integración de las 3 tablas + feature engineering
│   ├── entrenamiento.ipynb          # entrenamiento, evaluación e interpretabilidad (SHAP)
├── api.py                           # API de inferencia (FastAPI)
├── requirements.txt
└── README.md
```


## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Ejecutar el entrenamiento a partir de los CSV de origen

1. Abre y ejecuta `preproceso.ipynb` de principio a fin. Integra las tres tablas
   (explode de la cesta de subyacentes, cruce con `underlyings_reference.csv`,
   cruce temporal con `daily_volatility.csv` vía `merge_asof`, agregación a nivel
   de RFQ) y genera `data/processed_features.parquet`.
2. Abre y ejecuta `entrenamiento.ipynb` de principio a fin. Este notebook:
   - Separa entrenamiento/evaluación **por fecha** (últimas RFQs por `requested_date`
     como test, un 20%), no de forma aleatoria — evita fugas de información temporal,
     ya que en producción el modelo predice sobre operaciones futuras.
   - Entrena un `XGBRegressor`, elegido por su capacidad de capturar interacciones
     no lineales entre variables de forma eficiente sobre datos tabulares mixtos.
   - Evalúa con **MAE** como métrica principal (interpretable directamente en meses
     de error) y RMSE como referencia adicional.
   - Calcula importancia de variables con SHAP para interpretabilidad.

## 2. Guardar el artefacto del modelo resultante

Al final de `entrenamiento.ipynb` se guardan dos ficheros en `artifacts/`.  

`model_columns.pkl` guarda el orden y nombre exacto de las columnas (incluidas las
dummies de one-hot) que vio el modelo en entrenamiento. Es necesario porque el
one-hot encoding se genera dinámicamente con `pd.get_dummies`, y una única RFQ
nueva no genera las mismas columnas que todo el dataset de entrenamiento — la API
usa este fichero para alinear cada petición al esquema correcto antes de predecir.

## 3. Levantar la API de inferencia en local

```bash
uvicorn api:app --reload
```

La API queda escuchando en `http://127.0.0.1:8000`.

- `GET /health` — comprobación de estado.
- `POST /predict` — predice `avg_duration_months` para una RFQ.

La API espera una RFQ ya enriquecida con los agregados de cesta (los mismos que
calcula `preproceso.ipynb` al agrupar por `rfq_id`: `num_underlyings`,
`vol_63d_min/max/mean`, `base_vol_min/max/mean`, `sector_diversity`), no los tres
CSV en bruto — recalcular esa agregación en cada petición añade complejidad
innecesaria.

Ejemplo de petición:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_type": "autocallable_basket",
    "basket_type": "worst_of",
    "observation_frequency": "quarterly",
    "counterparty": "CPTY_01",
    "trader_id": "TRD_01",
    "autocall_barrier_pct": 1.0,
    "protection_barrier_pct": 0.7,
    "no_call_period_months": 6,
    "quoted_implied_vol": 0.22,
    "notional_credits": 1000000,
    "start_date": "2024-01-15",
    "end_date": "2027-01-15",
    "num_underlyings": 2,
    "vol_63d_min": 0.18,
    "vol_63d_max": 0.25,
    "vol_63d_mean": 0.215,
    "base_vol_min": 0.19,
    "base_vol_max": 0.24,
    "base_vol_mean": 0.215,
    "sector_diversity": 2
  }'
```

## Resultados


| Modelo   | MAE (meses) | RMSE         |
|----------|-------------|--------------|
| XGBoost  | 4.215409    | 5.722944     |

## Por qué XGBoost

XGBoost es un modelo de **gradient boosting**: un conjunto de árboles de decisión
construidos de forma secuencial, donde cada árbol nuevo corrige los errores
(residuos) que ha dejado el árbol anterior, optimizando una función de pérdida
mediante descenso de gradiente. Funciona especialmente bien en problemas de
regresión sobre datos tabulares como este porque:

- Captura de forma nativa relaciones **no lineales y con umbrales** (por ejemplo,
  el efecto de una barrera de autocall no es un efecto lineal continuo, es más
  bien un "o la supera o no la supera") sin necesidad de transformar variables a mano.
- No requiere escalar ni normalizar las variables numéricas.
- Maneja bien una mezcla de variables numéricas y categóricas (aquí, codificadas
  vía one-hot) e interacciones entre ellas.
- Es robusto y eficiente incluso con un volumen de datos moderado, sin necesitar
  el ajuste tan fino de hiperparámetros que sí requieren, por ejemplo, las redes
  neuronales.

Su principal desventaja frente a otros métodos de árboles es que, al construir
los árboles secuencialmente ajustando los residuos del anterior, es **más sensible
al ruido y a valores atípicos** en la variable objetivo que modelos que promedian
árboles independientes entre sí (como Random Forest) — por eso se controla su
complejidad con un `learning_rate` bajo (0.03), `max_depth` moderado (6) y
`subsample`/`colsample_bytree` por debajo de 1, para no sobreajustar. Como
cualquier modelo basado en árboles, tampoco extrapola bien fuera del rango de
valores vistos en entrenamiento (ver "Limitaciones del modelo").

## Métrica de evaluación y separación train/test

**MAE (Mean Absolute Error) como métrica principal**, porque usa las mismas
unidades que la variable objetivo (meses), lo que la hace directamente
interpretable: "el modelo se equivoca de media X meses". Además, al no elevar
al cuadrado los errores, es menos sensible a valores atípicos puntuales que el
RMSE — algo relevante aquí porque `avg_duration_months` se calcula mediante
simulación, y esa simulación en sí misma introduce algo de ruido en el target.

**RMSE como métrica complementaria**: al penalizar más los errores grandes,
sirve para detectar si el modelo comete fallos puntuales muy grandes que el
MAE, al promediar en valor absoluto, podría estar disimulando.

**Separación 80/20 por fecha, no aleatoria**: el modelo, en producción, se
usaría para predecir la duración de RFQs *futuras* a partir de lo aprendido
con RFQs *pasadas*. Un split aleatorio mezclaría información del futuro dentro
del entrenamiento (fuga temporal) y daría una métrica de evaluación
artificialmente optimista, que no reflejaría el rendimiento real del modelo en
producción. Por eso se ordena por `requested_date` y se reserva el último 20%
cronológico como test.

**Por qué se descartan al principio las filas sin valor en el target**: 
`avg_duration_months` solo existe para las RFQs con `executed = True` — no hay
forma de entrenar un modelo supervisado de regresión sin la etiqueta real, así
que las RFQs no ejecutadas se excluyen del entrenamiento (esto introduce el
sesgo de selección que se menciona más abajo, en limitaciones).

## Análisis de resultados (SHAP)

Según SHAP, las variables con más peso son:

- **`nominal_maturity_months`**: marca el techo temporal del contrato — un producto
  a un año nunca puede durar cinco, así que acota directamente la variable objetivo.
- **`autocall_barrier_pct`**: cuanto más alta la barrera, más difícil que la cesta
  la supere, y por tanto más tiempo tiende a vivir el producto antes de cancelarse
  (o llega a vencimiento).

Ambas relaciones tienen sentido económico directo con la mecánica de un autocallable
descrita en el enunciado: son, respectivamente, el techo estructural del contrato
y la condición que determina si se cancela antes de tiempo.

En el otro extremo, las variables con menos peso *(completar con las 5 que salgan
de `shap_importance.nsmallest(5, 'mean_abs_shap')` en `entrenamiento.ipynb`)* suelen
ser las dummies de identificadores con poca señal económica real — categorías
concretas de `counterparty` o `trader_id`, que identifican *quién* pidió la
cotización más que *cómo* se comporta el producto, y por tanto tiene sentido que
aporten poco frente a variables que describen la mecánica del propio contrato.

## Limitaciones del modelo

- Solo se entrena con RFQs **ejecutadas** (`executed = True`), que no son necesariamente
  una muestra representativa de todas las solicitudes recibidas — sesgo de selección.
- Al ser un modelo basado en árboles, **no extrapola**: ante un contrato con vencimiento
  o volatilidad fuera del rango visto en entrenamiento, la predicción se queda
  estancada en el valor de la última hoja del árbol.
- El modelo usa métricas resumen de volatilidad (media/min/max) en lugar de simular
  la evolución día a día de los precios y su correlación real dentro de la cesta —
  no tiene acceso a la correlación entre subyacentes, solo a volatilidades
  individuales, lo cual es una limitación relevante para un producto *worst-of*.
- Por todo esto, el modelo puede degradarse ante shocks de mercado o cambios de
  régimen no representados en el histórico de entrenamiento.

