"""Gera o JSON estático consumido pelo frontend."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from time_prevision.application.ports import ForecastGateway
from time_prevision.domain.weather import classify_weather

def _numero(valor, casas=1):
    if valor is None or pd.isna(valor):
        return None
    return round(float(valor), casas)


def _dia(row):
    if row.get("modo_previsao") == "tendencia_climatica":
        condicao, emoji = "Tendência climática", ""
    else:
        condicao, emoji = classify_weather(row.get("precipitacao_mm", 0), row.get("chance_chuva", 0),
                                           row.get("limiar_chuva", 50))
    return {
        "date": pd.Timestamp(row["data"]).strftime("%Y-%m-%d"),
        "temperature": _numero(row["temperatura"]),
        "temperatureMin": _numero(row["temperatura_minima"]),
        "temperatureMax": _numero(row["temperatura_maxima"]),
        "humidity": _numero(row["umidade"], 0),
        "precipitation": _numero(row["precipitacao_mm"], 2),
        "rainChance": _numero(row["chance_chuva"], 0),
        "climateRainFrequency": _numero(row.get("frequencia_climatica_chuva"), 0),
        "rainThreshold": _numero(row.get("limiar_chuva", 50), 0),
        "windSpeed": _numero(row["vento_velocidade_ms"]),
        "pressure": _numero(row["pressao_hpa"], 0),
        "cloudiness": _numero(row["nebulosidade"], 0),
        "thermalAmplitude": _numero(row["amplitude_termica"]),
        "approximateDewPoint": _numero(row["ponto_orvalho_aprox"]),
        "condition": condicao,
        "icon": emoji,
        "confidence": row.get("confianca", "desconhecida"),
        "forecastMode": row.get("modo_previsao", "operacional"),
        "nwpAvailable": bool(row.get("nwp_disponivel", False)),
    }


def _hora(row):
    return {
        "time": pd.Timestamp(row["data"]).strftime("%H:%M"),
        "temperature": _numero(row["temperatura"]),
        "humidity": _numero(row["umidade"], 0),
        "precipitation": _numero(row["precipitacao_mm"], 2),
        "rainChance": _numero(row["chance_chuva"], 0),
        "windSpeed": _numero(row["vento_velocidade_ms"]),
        "pressure": _numero(row["pressao_hpa"], 0),
        "condition": row["tipo_precipitacao"],
        "icon": row["emoji_clima"],
    }


def exportar(gateway: ForecastGateway, output, dias):
    ok, mensagem = gateway.verify()
    if not ok:
        raise RuntimeError(mensagem)

    previsao = gateway.predict(dias)
    ultima_data, atraso = gateway.data_status()
    horario = gateway.hourly(previsao.iloc[0])
    metricas = gateway.metrics()

    payload = {
        "location": {"name": "Porto Alegre", "state": "RS", "country": "Brasil"},
        "timezone": "America/Sao_Paulo",
        "generatedAt": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(timespec="seconds"),
        "sourceLastDate": ultima_data.strftime("%Y-%m-%d") if ultima_data is not None else None,
        "sourceDelayDays": int(atraso) if atraso is not None else None,
        "model": {
            "type": "LightGBM + HistGradientBoosting ensemble",
            "strategy": "direct multi-horizon",
            "horizons": [1, 2, 3, 5, 7, 10, 14, 21, 30],
            "outputs": [
                "temperature", "temperatureMin", "temperatureMax", "humidity",
                "precipitation", "windSpeed", "pressure", "cloudiness"
            ],
            "metrics": metricas,
        },
        "daily": [_dia(row) for _, row in previsao.iterrows()],
        "hourly": [_hora(row) for _, row in horario.iterrows()],
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Previsão exportada para {output}")


