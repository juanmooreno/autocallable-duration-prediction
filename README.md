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
├── code/
│   ├── preproceso.ipynb             # integración de las 3 tablas + feature engineering
│   ├── entrenamiento.ipynb          # entrenamiento, evaluación e interpretabilidad (SHAP)
│   └── api.py                       # API de inferencia (FastAPI)
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

## Interpretación y sentido de negocio

Según SHAP, las variables con más peso son:

- **`nominal_maturity_months`**: marca el techo temporal del contrato — un producto
  a un año nunca puede durar cinco, así que acota directamente la variable objetivo.
- **`autocall_barrier_pct`**: cuanto más alta la barrera, más difícil que la cesta
  la supere, y por tanto más tiempo tiende a vivir el producto antes de cancelarse
  (o llega a vencimiento).

Ambas relaciones tienen sentido económico directo con la mecánica de un autocallable
descrita en el enunciado.

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
