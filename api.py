"""
API de inferencia para el modelo de duración media de autocallables.

Uso:
    uvicorn api:app --reload

La API espera una RFQ ya "enriquecido" con los agregados de cesta
(num_underlyings, vol_63d_min/max/mean, base_vol_min/max/mean,
sector_diversity), tal y como los produce preproceso.ipynb al agrupar
por rfq_id. Sobre esa fila se aplica el mismo feature engineering final
que en entrenamiento.ipynb (fechas -> nominal_maturity_months, spreads,
one-hot) antes de predecir con el modelo ya entrenado.
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date

MODEL_PATH = "artifacts/model.pkl"
COLUMNS_PATH = "artifacts/model_columns.pkl"

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

app = FastAPI(title="autocallable-duration-api")

CATEGORICAL_COLS = [
    "product_type",
    "basket_type",
    "observation_frequency",
    "counterparty",
    "trader_id",
]


class RFQEnriched(BaseModel):
    product_type: str
    basket_type: str
    observation_frequency: str
    counterparty: str
    trader_id: str
    autocall_barrier_pct: float
    protection_barrier_pct: float
    no_call_period_months: int
    quoted_implied_vol: float
    notional_credits: float
    start_date: date
    end_date: date
    num_underlyings: int
    vol_63d_min: float
    vol_63d_max: float
    vol_63d_mean: float
    base_vol_min: float
    base_vol_max: float
    base_vol_mean: float
    sector_diversity: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: RFQEnriched):
    row = pd.DataFrame([payload.dict()])

    row["start_date"] = pd.to_datetime(row["start_date"])
    row["end_date"] = pd.to_datetime(row["end_date"])

    # Mismo feature engineering que en entrenamiento.ipynb / preproceso.ipynb
    row["nominal_maturity_months"] = (
        ((row["end_date"] - row["start_date"]).dt.days / 30.4375)
        .clip(lower=0)
        .apply(lambda x: int(x) if float(x).is_integer() else int(x) + 1)
    )
    row["barrier_spread"] = row["autocall_barrier_pct"] - row["protection_barrier_pct"]
    row["vol_spread"] = row["quoted_implied_vol"] - row["vol_63d_mean"]
    row["no_call_ratio"] = row["no_call_period_months"] / row["nominal_maturity_months"].replace(0, pd.NA)

    row_encoded = pd.get_dummies(row, columns=CATEGORICAL_COLS, drop_first=False, dtype="int8")

    # Alineamos con las columnas vistas en entrenamiento:
    # - una categoria no vista en training simplemente se pierde (no aporta señal nueva)
    # - una columna dummy vista en training pero ausente en esta fila queda a 0
    row_aligned = row_encoded.reindex(columns=model_columns, fill_value=0)

    try:
        prediction = model.predict(row_aligned)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {exc}")

    return {"avg_duration_months": float(prediction)}
