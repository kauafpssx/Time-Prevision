"""
Módulo compartilhado de FEATURE ENGINEERING
Usado tanto por treinar_modelo.py quanto por previsao_service.py
para garantir que o treinamento e a previsão usem exatamente as MESMAS features.
"""

import numpy as np
import pandas as pd

from time_prevision.infrastructure.providers.open_meteo import NWP_FEATS

# Variáveis previstas
VARS_NUMERICAS = [
    "temperatura",
    "temperatura_minima",
    "temperatura_maxima",
    "umidade",
    "precipitacao_mm",
    "vento_velocidade_ms",
    "pressao_hpa",
    "nebulosidade",
]

# Horizontes com modelo DIRETO (prevê t+h a partir do estado de hoje, sem
# recursão) — dia solicitado usa o horizonte treinado mais próximo. Evita o
# acúmulo de erro do esquema recursivo antigo (prever em cima da própria
# previsão) em janelas longas (7-30 dias).
HORIZONTES_MODELO = [1, 2, 3, 5, 7, 10, 14, 21, 30]

# Lags e janelas móveis
LAGS = [1, 2, 3, 7, 14, 28]
ROLL_WINDOWS = [3, 7, 14, 28]
ROLL_MINMAX = [7]

# Features cíclicas de calendário
CAL_FEATS = ["doy_sin", "doy_cos", "doy_sin2", "doy_cos2", "dow_sin", "dow_cos", "mes", "ano",
             "radiacao_solar_teorica"]

LATITUDE_RAD = np.radians(-30.0331)  # Porto Alegre (estação Meteostat 83967)


def _radiacao_solar_teorica(doy):
    """
    Radiação solar extraterrestre teórica (MJ/m²/dia), fórmula FAO-56 Penman-
    Monteith. Proxy determinístico de 'quanto sol dá pra bater' nesse dia do
    ano nessa latitude — puramente astronômico (só depende de data + local),
    então é conhecido com certeza até pra dias futuros, sem risco de
    vazamento. Ajuda nebulosidade/temperatura a captar o ciclo de insolação
    sem depender só do padrão autoregressivo.
    """
    Gsc = 0.0820  # MJ/m²/min, constante solar
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(LATITUDE_RAD) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(LATITUDE_RAD) * np.sin(delta) + np.cos(LATITUDE_RAD) * np.cos(delta) * np.sin(ws)
    )


def _calendario(df):
    """Features cíclicas de calendário (dia do ano, dia da semana, mês)"""
    df = df.copy()
    doy = df["data"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # 2º harmônico: capta transições de meio de estação (ex: outono/inverno)
    # que um único seno/cosseno anual não representa bem.
    df["doy_sin2"] = np.sin(4 * np.pi * doy / 365.25)
    df["doy_cos2"] = np.cos(4 * np.pi * doy / 365.25)
    dow = df["data"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["mes"] = df["data"].dt.month
    df["ano"] = df["data"].dt.year
    df["radiacao_solar_teorica"] = _radiacao_solar_teorica(doy)
    return df


def _features_meteorologicas(df):
    """
    Features derivadas SEM vazamento de dados:
    usam os valores do dia ANTERIOR (shift 1), nunca do próprio dia previsto.
    """
    df = df.copy()
    df["amplitude_termica"] = (df["temperatura_maxima"] - df["temperatura_minima"]).shift(1)
    df["ponto_orvalho_aprox"] = (df["temperatura"] - ((100 - df["umidade"]) / 5)).shift(1)
    df["temp_umidade_inter"] = (df["temperatura"] * df["umidade"] / 100).shift(1)
    df["press_umid_inter"] = (df["pressao_hpa"] * df["umidade"] / 100).shift(1)
    # Tendência de pressão: queda rápida = frente/chuva chegando, subida rápida
    # = estabilização. Sinal clássico de meteorologia que faltava (só existia
    # lag/diff bruto de pressão, sem "velocidade" da variação recente).
    pressao_ontem = df["pressao_hpa"].shift(1)
    df["tendencia_pressao_24h"] = pressao_ontem - df["pressao_hpa"].shift(2)
    df["tendencia_pressao_72h"] = pressao_ontem - df["pressao_hpa"].shift(4)
    return df


def _features_nwp(df, nwp_df=None):
    """
    Mescla a previsão do modelo numérico (GFS/ECMWF/ICON via Open-Meteo) como
    feature adicional — NÃO é lag do passado, é a previsão externa PARA o
    próprio dia sendo previsto (legítimo: essa previsão existe e está
    disponível antes do dia acontecer, tanto no treino quanto em produção).
    Sem vazamento porque é gerada por um sistema físico independente do
    histórico local, não deriva do valor observado que estamos tentando prever.
    Colunas ausentes (fora da janela de cobertura do NWP) ficam NaN — o
    LightGBM lida nativamente com valores faltantes.
    """
    df = df.copy()
    if nwp_df is None or nwp_df.empty:
        for c in NWP_FEATS:
            df[c] = np.nan
        return df
    return df.merge(nwp_df[["data"] + NWP_FEATS], on="data", how="left")


def _features_temporais(df, vars_=None, nwp_df=None):
    """
    Cria todas as features temporais (lags, médias/desvios móveis, mín/máx,
    diffs do dia anterior). Todas as features usam APENAS valores passados
    (exceto o bloco NWP, que é previsão externa legítima — ver _features_nwp).
    """
    if vars_ is None:
        vars_ = VARS_NUMERICAS

    df = _calendario(df)
    df = _features_meteorologicas(df)
    df = _features_nwp(df, nwp_df)

    cols = {}
    for v in vars_:
        if v not in df.columns:
            continue
        for l in LAGS:
            cols[f"{v}_lag{l}"] = df[v].shift(l)
        for w in ROLL_WINDOWS:
            cols[f"{v}_media_{w}"] = df[v].shift(1).rolling(w, min_periods=1).mean()
            cols[f"{v}_desvio_{w}"] = df[v].shift(1).rolling(w, min_periods=1).std()
        for w in ROLL_MINMAX:
            cols[f"{v}_min_{w}"] = df[v].shift(1).rolling(w, min_periods=1).min()
            cols[f"{v}_max_{w}"] = df[v].shift(1).rolling(w, min_periods=1).max()
        # Diffs do dia ANTERIOR (shift 1) para não vazar o valor do próprio dia
        cols[f"{v}_diff1"] = df[v].diff(1).shift(1)
        cols[f"{v}_diff7"] = df[v].diff(7).shift(1)

    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def _lista_features(vars_=None):
    """Gera a lista de nomes de features (ordem consistente)"""
    if vars_ is None:
        vars_ = VARS_NUMERICAS

    feats = list(CAL_FEATS)
    feats.extend(["amplitude_termica", "ponto_orvalho_aprox", "temp_umidade_inter", "press_umid_inter",
                  "tendencia_pressao_24h", "tendencia_pressao_72h"])
    feats.extend(NWP_FEATS)

    for v in vars_:
        for l in LAGS:
            feats.append(f"{v}_lag{l}")
        for w in ROLL_WINDOWS:
            feats.append(f"{v}_media_{w}")
            feats.append(f"{v}_desvio_{w}")
        for w in ROLL_MINMAX:
            feats.append(f"{v}_min_{w}")
            feats.append(f"{v}_max_{w}")
        feats.append(f"{v}_diff1")
        feats.append(f"{v}_diff7")

    return feats
