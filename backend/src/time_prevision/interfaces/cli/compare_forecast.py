"""
Compara a previsão do modelo local contra a previsão oficial do Open-Meteo
(mesma fonte que agrega GFS/ECMWF/ICON — usada por meteorologistas) para
Porto Alegre. Roda sem API key (endpoint público e gratuito).

Uso:
    python comparar_previsao.py            # compara os próximos 7 dias
    python comparar_previsao.py --dias 3
    python comparar_previsao.py --historico 14  # valida contra observado (archive-api)
"""

import argparse
import sys

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from time_prevision.infrastructure.ml.prediction_service import carregar_modelo, prever_rapido, verificar_arquivos

LAT, LON = -30.0331, -51.2300  # Porto Alegre (estação Meteostat 83967)

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _buscar_previsao_open_meteo(dias):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "windspeed_10m_max", "surface_pressure_mean",
            "relative_humidity_2m_mean",
        ]),
        "timezone": "America/Sao_Paulo",
        "forecast_days": dias,
    }
    r = requests.get(OPEN_METEO_FORECAST, params=params, timeout=20)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(d)
    df["data"] = pd.to_datetime(df["time"])
    return df.rename(columns={
        "temperature_2m_max": "om_tmax",
        "temperature_2m_min": "om_tmin",
        "temperature_2m_mean": "om_tmed",
        "precipitation_sum": "om_prcp",
        "windspeed_10m_max": "om_vento_kmh",
        "surface_pressure_mean": "om_pressao",
        "relative_humidity_2m_mean": "om_umidade",
    })


def _buscar_historico_open_meteo(dias):
    """Observado (não previsto) dos últimos N dias, p/ validar contra a realidade"""
    fim = pd.Timestamp.now(tz="America/Sao_Paulo").normalize() - pd.Timedelta(days=1)
    inicio = fim - pd.Timedelta(days=dias - 1)
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": inicio.strftime("%Y-%m-%d"),
        "end_date": fim.strftime("%Y-%m-%d"),
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "windspeed_10m_max",
        ]),
        "timezone": "America/Sao_Paulo",
    }
    r = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=20)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(d)
    df["data"] = pd.to_datetime(df["time"])
    return df.rename(columns={
        "temperature_2m_max": "om_tmax",
        "temperature_2m_min": "om_tmin",
        "temperature_2m_mean": "om_tmed",
        "precipitation_sum": "om_prcp",
        "windspeed_10m_max": "om_vento_kmh",
    })


def comparar(dias):
    ok, msg = verificar_arquivos()
    if not ok:
        print(f"❌ {msg}")
        return

    modelo, features, _ = carregar_modelo()
    print(f"📡 Gerando previsão do modelo local p/ {dias} dias...")
    prev = prever_rapido(modelo, features, dias=dias)
    prev["data"] = pd.to_datetime(prev["data"]).dt.normalize()

    print("🌐 Buscando previsão oficial Open-Meteo (GFS/ECMWF/ICON blend)...")
    om = _buscar_previsao_open_meteo(dias)
    om["data"] = om["data"].dt.tz_localize(None)

    df = prev.merge(om, on="data", how="inner")
    if df.empty:
        print("❌ Sem sobreposição de datas entre modelo e Open-Meteo")
        return

    df["om_vento_ms"] = df["om_vento_kmh"] / 3.6
    df["diff_tmax"] = df["temperatura_maxima"] - df["om_tmax"]
    df["diff_tmin"] = df["temperatura_minima"] - df["om_tmin"]
    df["diff_prcp"] = df["precipitacao_mm"] - df["om_prcp"]
    df["diff_vento"] = df["vento_velocidade_ms"] - df["om_vento_ms"]

    print("\n" + "=" * 100)
    print(f"{'Data':<12}{'TMax(mod/om)':<16}{'TMin(mod/om)':<16}{'Chuva(mod/om)':<16}{'Vento(mod/om)':<16}")
    print("-" * 100)
    for _, r in df.iterrows():
        print(
            f"{r['data'].date()!s:<12}"
            f"{r['temperatura_maxima']:.1f}/{r['om_tmax']:.1f}°C".ljust(16)
            + f"{r['temperatura_minima']:.1f}/{r['om_tmin']:.1f}°C".ljust(16)
            + f"{r['precipitacao_mm']:.1f}/{r['om_prcp']:.1f}mm".ljust(16)
            + f"{r['vento_velocidade_ms']:.1f}/{r['om_vento_ms']:.1f}m/s".ljust(16)
        )
    print("=" * 100)
    print(f"MAE vs Open-Meteo -> TMax: {df['diff_tmax'].abs().mean():.2f}°C  "
          f"TMin: {df['diff_tmin'].abs().mean():.2f}°C  "
          f"Chuva: {df['diff_prcp'].abs().mean():.2f}mm  "
          f"Vento: {df['diff_vento'].abs().mean():.2f}m/s")
    print("(diferença contra a previsão oficial — não é erro absoluto contra o real,")
    print(" já que Open-Meteo também é uma previsão, não o observado)")


def validar_historico(dias):
    """Roda o modelo 'no passado' reconstruindo o histórico até N dias atrás
    e compara contra o observado real (archive-api) — teste honesto de acerto."""
    ok, msg = verificar_arquivos()
    if not ok:
        print(f"❌ {msg}")
        return

    modelo, features, _ = carregar_modelo()
    print(f"📡 Gerando previsão do modelo local p/ {dias} dias...")
    prev = prever_rapido(modelo, features, dias=dias)
    prev["data"] = pd.to_datetime(prev["data"]).dt.normalize()

    print("🌐 Buscando observado real (Open-Meteo archive)...")
    obs = _buscar_historico_open_meteo(dias)
    obs["data"] = obs["data"].dt.tz_localize(None)

    df = prev.merge(obs, on="data", how="inner")
    if df.empty:
        print("⚠️ Sem sobreposição — só funciona pros últimos dias já ocorridos, "
              "não pra previsão futura (Open-Meteo archive só tem dado passado).")
        return

    df["om_vento_ms"] = df["om_vento_kmh"] / 3.6
    print("\n" + "=" * 100)
    print(f"{'Data':<12}{'TMax(mod/real)':<18}{'TMin(mod/real)':<18}{'Chuva(mod/real)':<18}")
    print("-" * 100)
    for _, r in df.iterrows():
        print(
            f"{r['data'].date()!s:<12}"
            f"{r['temperatura_maxima']:.1f}/{r['om_tmax']:.1f}°C".ljust(18)
            + f"{r['temperatura_minima']:.1f}/{r['om_tmin']:.1f}°C".ljust(18)
            + f"{r['precipitacao_mm']:.1f}/{r['om_prcp']:.1f}mm".ljust(18)
        )
    mae_tmax = (df["temperatura_maxima"] - df["om_tmax"]).abs().mean()
    mae_tmin = (df["temperatura_minima"] - df["om_tmin"]).abs().mean()
    mae_prcp = (df["precipitacao_mm"] - df["om_prcp"]).abs().mean()
    print("=" * 100)
    print(f"MAE vs OBSERVADO REAL -> TMax: {mae_tmax:.2f}°C  TMin: {mae_tmin:.2f}°C  Chuva: {mae_prcp:.2f}mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara previsão do modelo vs Open-Meteo")
    parser.add_argument("--dias", type=int, default=7, help="Dias de previsão a comparar")
    parser.add_argument("--historico", type=int, default=0,
                         help="Em vez de comparar previsões, valida contra observado real dos últimos N dias")
    args = parser.parse_args()

    if args.historico > 0:
        validar_historico(args.historico)
    else:
        comparar(args.dias)
