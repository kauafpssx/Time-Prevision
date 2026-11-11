"""
Integração com modelo numérico de previsão (NWP) via Open-Meteo — blend de
GFS/ECMWF/ICON, a mesma classe de dado que meteorologista usa de verdade.

Diferente do modelo puramente autoregressivo (que só enxerga padrão local
passado), o NWP resolve física de atmosfera numa grade global e "vê" frentes/
sistemas chegando antes deles afetarem a estação local. Usado aqui como
FEATURE (não como substituto): o LightGBM aprende a corrigir o viés do NWP
para Porto Alegre especificamente (técnica clássica chamada MOS — Model
Output Statistics).

Histórico: Open-Meteo mantém um arquivo de "historical forecast" (o que o
modelo numérico dizia para aquele dia, reprocessado com o modelo atual) —
troca por um dump reprocessado, não é 100% o que teria sido emitido na época,
mas é a aproximação prática disponível de graça. Cobre só os últimos anos
(a API não tem histórico completo desde 1977), então a maioria das ~15800
linhas do treino fica com NaN aqui — o LightGBM lida bem com isso
nativamente (feature informativa só quando presente).
"""

import sqlite3

import pandas as pd
import requests

from time_prevision.infrastructure.paths import DATABASE_PATH

LAT, LON = -30.0331, -51.2300  # Porto Alegre (estação Meteostat 83967)

HIST_FORECAST_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "windspeed_10m_max", "surface_pressure_mean",
    "relative_humidity_2m_mean", "cloud_cover_mean",
]

RENAME = {
    "temperature_2m_max": "nwp_tmax",
    "temperature_2m_min": "nwp_tmin",
    "temperature_2m_mean": "nwp_tmed",
    "precipitation_sum": "nwp_prcp",
    "windspeed_10m_max": "nwp_vento_kmh",
    "surface_pressure_mean": "nwp_pressao",
    "relative_humidity_2m_mean": "nwp_umidade",
    "cloud_cover_mean": "nwp_nebulosidade_pct",
}

NWP_FEATS = ["nwp_tmax", "nwp_tmin", "nwp_tmed", "nwp_prcp", "nwp_vento_ms", "nwp_pressao", "nwp_umidade",
             "nwp_nebulosidade"]

TABELA = "nwp_diario"


def _pos_processar(df):
    df = df.rename(columns=RENAME)
    df["nwp_vento_ms"] = df["nwp_vento_kmh"] / 3.6
    # Open-Meteo dá nebulosidade em % (0-100); nossa escala é oktas (0-8),
    # mesma unidade que a estação Meteostat usa pra 'nebulosidade'.
    df["nwp_nebulosidade"] = df["nwp_nebulosidade_pct"] / 12.5
    df = df.drop(columns=["nwp_vento_kmh", "nwp_nebulosidade_pct"], errors="ignore")
    df["data"] = pd.to_datetime(df["time"]).dt.normalize()
    return df.drop(columns=["time"], errors="ignore")


def atualizar_historico(db_path=DATABASE_PATH, dias=3200):
    """
    Busca o histórico NWP (últimos N dias) e grava/atualiza a tabela no banco.
    A cobertura real da API começa por volta de 2017 (~3200 dias) — usar só
    ~3 anos deixava a maioria dos folds da validação cruzada (que cobre
    1977-hoje) sem essa feature, diluindo o ganho medido a quase zero mesmo
    quando ela ajuda de verdade nos anos recentes.
    """
    fim = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    inicio = fim - pd.Timedelta(days=dias - 1)
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": inicio.strftime("%Y-%m-%d"),
        "end_date": fim.strftime("%Y-%m-%d"),
        "daily": ",".join(DAILY_VARS),
        "timezone": "America/Sao_Paulo",
    }
    r = requests.get(HIST_FORECAST_API, params=params, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"])
    df = _pos_processar(df)

    conn = sqlite3.connect(db_path)
    df[["data"] + NWP_FEATS].to_sql(TABELA, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABELA}_data ON {TABELA}(data)")
    conn.commit()
    conn.close()
    return len(df)


def carregar_historico(db_path=DATABASE_PATH):
    """Carrega a tabela NWP já cacheada no banco (vazio se nunca foi buscada)."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {TABELA}", conn)
        df["data"] = pd.to_datetime(df["data"])
    except Exception:
        df = pd.DataFrame(columns=["data"] + NWP_FEATS)
    conn.close()
    return df


def buscar_previsao_futura(dias):
    """
    Previsão NWP ao vivo para os próximos `dias`. A API gratuita cobre até
    16 dias; horizontes além disso voltam vazios (features NaN, tratado
    nativamente pelo LightGBM — sem essa 'visão externa' extra nesses dias).
    """
    dias_api = min(dias, 16)
    params = {
        "latitude": LAT, "longitude": LON,
        "daily": ",".join(DAILY_VARS),
        "timezone": "America/Sao_Paulo",
        "forecast_days": dias_api,
    }
    try:
        r = requests.get(FORECAST_API, params=params, timeout=20)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["daily"])
        return _pos_processar(df)
    except Exception:
        return pd.DataFrame(columns=["data"] + NWP_FEATS)
