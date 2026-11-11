"""
Serviço de previsão - lógica de modelo/predição, sem dependência de GUI.
Carrega modelo pré-treinado e gera previsões diárias/horárias.
"""

import numpy as np
import pandas as pd
import pickle
import sqlite3
from time_prevision.domain.weather import DRY_DAY_THRESHOLD, classify_weather
from time_prevision.infrastructure.ml.features import VARS_NUMERICAS, _features_temporais, HORIZONTES_MODELO
from time_prevision.infrastructure.paths import DATABASE_PATH, MODEL_DIR
from time_prevision.infrastructure.providers.open_meteo import NWP_FEATS, buscar_previsao_futura

# Caminhos
DB_PATH = DATABASE_PATH
MODELO_PATH = MODEL_DIR / "modelo_treinado.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"
METRICAS_PATH = MODEL_DIR / "metricas.pkl"

# Dias da semana em PT-BR (nomes completos)
DIAS_SEMANA = {
    "Monday": "Segunda-feira",
    "Tuesday": "Terça-feira",
    "Wednesday": "Quarta-feira",
    "Thursday": "Quinta-feira",
    "Friday": "Sexta-feira",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}


def nome_dia_completo(data):
    """Dia da semana por extenso em PT-BR"""
    return DIAS_SEMANA.get(data.strftime("%A"), data.strftime("%A"))


def nome_dia_curto(data):
    """Dia da semana abreviado em PT-BR (Seg, Ter, Qua...)"""
    return nome_dia_completo(data)[:3]


# Limites físicos aplicados às previsões
LIMITES = {
    "temperatura": (-30.0, 50.0),
    "temperatura_minima": (-40.0, 40.0),
    "temperatura_maxima": (-20.0, 55.0),
    "umidade": (0.0, 100.0),
    "precipitacao_mm": (0.0, 500.0),
    "vento_velocidade_ms": (0.0, 60.0),
    "pressao_hpa": (950.0, 1050.0),
    "nebulosidade": (0.0, 8.0),
}


def verificar_arquivos():
    """Verifica se modelo e features existem"""
    if not MODELO_PATH.exists():
        return False, "Modelo não encontrado em 'data/model/'. Execute o módulo de treinamento."
    if not FEATURES_PATH.exists():
        return False, "Features não encontradas em 'data/model/'. Execute o módulo de treinamento."
    if not DB_PATH.exists():
        return False, "Banco de dados não encontrado em 'data/'. Execute o módulo de atualização."
    return True, "OK"


def status_dados():
    """
    Última data disponível no banco e dias de atraso em relação a hoje.
    A estação (Meteostat) publica com atraso (dias/semanas) — o dia 1 da
    previsão é sempre 'último dia do banco + 1', não necessariamente hoje.
    """
    if not DB_PATH.exists():
        return None, None
    conn = sqlite3.connect(DB_PATH)
    ultima = conn.execute("SELECT MAX(data) FROM daily").fetchone()[0]
    conn.close()
    if not ultima:
        return None, None
    ultima_data = pd.to_datetime(ultima)
    atraso = (pd.Timestamp.now().normalize() - ultima_data).days
    return ultima_data, atraso


def carregar_modelo():
    """Carrega modelo pré-treinado da pasta modelo/"""
    with open(MODELO_PATH, "rb") as f:
        modelo = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    try:
        with open(METRICAS_PATH, "rb") as f:
            metricas = pickle.load(f)
    except Exception:
        metricas = None
    return modelo, features, metricas


def _load_db():
    """Carrega dados do banco"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM daily ORDER BY data", conn)
    conn.close()
    df["data"] = pd.to_datetime(df["data"])
    return df


def _clamp(valor, limite):
    """Limita um valor a um intervalo físico (lo, hi)"""
    if pd.isna(valor):
        return valor
    lo, hi = limite
    return min(max(valor, lo), hi)


def _resolver_horizonte(modelo, h):
    """
    modelo[var] agora é {horizonte: modelo_direto} (multi-horizonte, sem
    recursão). Resolve pro horizonte treinado mais próximo do solicitado e
    devolve no formato antigo {var: modelo_direto} que _prever_ensamble espera
    — mantém essa função sem precisar saber de horizonte.
    """
    resolvido = {}
    for var, m_var in modelo.items():
        if isinstance(m_var, dict) and m_var and all(isinstance(k, int) for k in m_var.keys()):
            h_prox = min(m_var.keys(), key=lambda k: abs(k - h))
            resolvido[var] = m_var[h_prox]
        else:
            resolvido[var] = m_var  # formato antigo (sem horizonte) - compatibilidade
    return resolvido


def _prever_ensamble(modelo, entrada):
    """
    Prediz as 8 variáveis com limites físicos.
    Retorna (valores, chance_chuva) — chance real do classificador (0-100) ou None.
    Formato novo: dict {variavel: {'lgbm': m, 'histgb': m}} -> média do ensemble.
    Chuva em 2 estágios: {'clf': classificador, 'reg': quantidade} -> proba * quantidade.
    Formato antigo: modelo único com .predict -> compatibilidade.
    """
    chance = None
    if isinstance(modelo, dict):
        valores = []
        for var in VARS_NUMERICAS:
            m = modelo.get(var)
            if m is None:
                valores.append(float("nan"))
                continue
            if isinstance(m, dict) and "clf" in m:
                # Precipitação em 2 estágios: chance de chuva x quantidade.
                # proba*quant sozinho nunca chega a 0 (a regressão de
                # quantidade só treina em dias que choveram, então tem piso
                # > 0mm mesmo com proba baixa) - isso contradizia dias
                # classificados "Sem Chuva" mostrando mm > 0. Corta o piso
                # no mesmo limiar de classificar_clima e reescala o resto
                # continuamente até proba=1 -> mm = quant.
                proba = float(m["clf"].predict_proba(entrada)[:, 1][0])
                quant = float(m["reg"].predict(entrada)[0])
                if var == "precipitacao_mm":
                    chance = int(round(proba * 100))
                    limiar = LIMIAR_SEM_CHUVA_PCT / 100.0
                    fator = max(0.0, (proba - limiar) / (1.0 - limiar))
                    valores.append(float(fator * quant))
                else:
                    valores.append(float(proba * quant))
            elif isinstance(m, dict):
                preds = [mm.predict(entrada)[0] for mm in m.values()]
                valores.append(float(np.mean(preds, axis=0)))
            else:
                valores.append(float(m.predict(entrada)[0]))
    else:
        valores = list(modelo.predict(entrada)[0])

    valores = [_clamp(v, LIMITES.get(var, (-1e9, 1e9)))
               for v, var in zip(valores, VARS_NUMERICAS)]
    return valores, chance


def _vizinhos_horizonte(i):
    """
    Os dois horizontes treinados que cercam o dia solicitado (ex.: i=17 ->
    (14, 21)). Se i já é um horizonte treinado, ou está fora da faixa da
    escada, os dois vizinhos coincidem (sem interpolação).
    """
    h_baixo = max([h for h in HORIZONTES_MODELO if h <= i], default=HORIZONTES_MODELO[0])
    h_alto = min([h for h in HORIZONTES_MODELO if h >= i], default=HORIZONTES_MODELO[-1])
    return h_baixo, h_alto


def _prever_um_horizonte(modelo, entrada, h):
    modelo_h = _resolver_horizonte(modelo, h)
    return _prever_ensamble(modelo_h, entrada)


def prever_rapido(modelo, features, dias=7):
    """
    Gera previsão usando modelos DIRETOS por horizonte (t+1, t+2, t+3, t+5,
    t+7, t+10, t+14, t+21, t+30), prevendo direto do estado real de hoje.
    Sem recursão: elimina o acúmulo de erro do esquema antigo (prever em
    cima da própria previsão), que é o motivo clássico de previsão de 7+
    dias degradar rápido.

    Dias fora da escada (ex. 17 dias, entre os horizontes treinados 14 e 21)
    interpolam linearmente entre os dois modelos vizinhos — sem isso, vários
    dias seguidos usariam exatamente o mesmo modelo (+ mesmo NWP, já sem
    cobertura além de ~16 dias) e sairiam com valores idênticos dia após dia.
    """
    df = _load_db()
    # Features SEM NWP ainda — mesclado abaixo por dia-alvo específico.
    base = _features_temporais(df, VARS_NUMERICAS, nwp_df=None)
    linha_hoje = base.iloc[-1]
    ultimo_dia = base["data"].iloc[-1]

    # Previsão NWP (GFS/ECMWF/ICON) ao vivo p/ os próximos dias — buscada uma
    # vez só. Cobre até 16 dias (limite da API gratuita); horizontes além
    # disso ficam sem essa feature (NaN, tratado nativamente pelo LightGBM).
    nwp_futuro = buscar_previsao_futura(dias)

    previsoes = []

    for i in range(1, dias + 1):
        data_alvo = (ultimo_dia + pd.Timedelta(days=i)).normalize()

        # O estado de "hoje" (lags/médias móveis) é o mesmo para qualquer
        # horizonte — só o NWP do dia-alvo muda entre as linhas.
        entrada = pd.DataFrame([linha_hoje]).reset_index(drop=True)[features]
        linha_nwp = nwp_futuro[nwp_futuro["data"] == data_alvo]
        for c in NWP_FEATS:
            if c in entrada.columns:
                entrada[c] = float(linha_nwp.iloc[0][c]) if len(linha_nwp) else np.nan

        h_baixo, h_alto = _vizinhos_horizonte(i)
        if h_baixo == h_alto:
            valores, chance = _prever_um_horizonte(modelo, entrada, h_baixo)
        else:
            v_baixo, c_baixo = _prever_um_horizonte(modelo, entrada, h_baixo)
            v_alto, c_alto = _prever_um_horizonte(modelo, entrada, h_alto)
            w = (i - h_baixo) / (h_alto - h_baixo)
            valores = [vb + (va - vb) * w for vb, va in zip(v_baixo, v_alto)]
            if c_baixo is not None and c_alto is not None:
                chance = c_baixo + (c_alto - c_baixo) * w
            else:
                chance = c_baixo if c_baixo is not None else c_alto

        nova = {"data": data_alvo}
        for v, val in zip(VARS_NUMERICAS, valores):
            nova[v] = float(val)
        nova["chance_chuva"] = chance if chance is not None else chance_chuva(nova["precipitacao_mm"])

        nova["amplitude_termica"] = nova["temperatura_maxima"] - nova["temperatura_minima"]
        nova["ponto_orvalho_aprox"] = nova["temperatura"] - ((100 - nova["umidade"]) / 5)

        previsoes.append(nova)

    return pd.DataFrame(previsoes)


LIMIAR_SEM_CHUVA_PCT = DRY_DAY_THRESHOLD


def classificar_clima(row):
    """
    Classifica clima previsto. Usa chance_chuva (%) - saída calibrada do
    classificador (AUC ~0.73) - para decidir Sem Chuva x chuva, em vez do mm
    estimado (proba * quantidade): essa média raramente chega perto de zero
    porque a regressão de quantidade só treina em dias que choveram, então
    usar só o mm faria a UI quase nunca mostrar um dia seco.
    """
    prcp = row.get("precipitacao_mm", 0) or 0
    chance = row.get("chance_chuva")
    if chance is None:
        chance = chance_chuva(prcp)

    return classify_weather(prcp, chance)


def chance_chuva(prcp_mm):
    """Converte precipitação prevista (mm) em chance de chuva em %"""
    if pd.isna(prcp_mm) or prcp_mm <= 0.01:
        return 0
    # Curva exponencial: ~63% em 1.5mm, ~86% em 3mm, ~98% em 6mm
    return int(round(100 * (1 - np.exp(-prcp_mm / 1.5))))


def gerar_previsao_horaria(linha_diaria):
    """
    Deriva a previsão das 24 horas do DIA ATUAL a partir da previsão diária do modelo.
    Usa curva diurna de temperatura (mínima ~03h, máxima ~15h) e distribui as
    demais variáveis ao longo do dia.
    """
    data = linha_diaria["data"]
    tmin = float(linha_diaria["temperatura_minima"])
    tmax = float(linha_diaria["temperatura_maxima"])
    tmed = float(linha_diaria["temperatura"])
    umid = float(linha_diaria["umidade"])
    prcp = float(linha_diaria["precipitacao_mm"])
    vento = float(linha_diaria["vento_velocidade_ms"])
    press = float(linha_diaria["pressao_hpa"])

    horas = np.arange(24)

    # Temperatura: curva diurna exata entre mínima prevista (03h) e máxima (15h)
    amp = (tmax - tmin) / 2.0
    centro = (tmax + tmin) / 2.0
    temp_h = centro + amp * np.cos(2 * np.pi * (horas - 15) / 24)

    # Umidade: inversamente proporcional à temperatura
    hum_h = umid + 3.0 * (tmed - temp_h)
    hum_h = np.clip(hum_h, 20, 100)

    # Precipitação: total diário concentrado à tarde/noite
    if pd.notna(prcp) and prcp > 0:
        pesos = 0.15 + np.exp(-((horas - 17) ** 2) / (2 * 4.0 ** 2))
        pesos[horas < 6] = 0.03
        prcp_h = prcp * pesos / pesos.sum()
        prcp_h = np.round(prcp_h, 2)
    else:
        prcp_h = np.zeros(24)

    # Vento: mais forte à tarde
    vento_h = np.maximum(vento * (1 + 0.2 * np.cos(2 * np.pi * (horas - 15) / 24)), 0)

    # Pressão: leve variação ao longo do dia
    press_h = press + 0.8 * np.sin(2 * np.pi * (horas - 9) / 24)

    df = pd.DataFrame({
        # pd.date_range evita DeprecationWarning do numpy 2.x (timedelta genérico)
        "data": pd.date_range(start=data, periods=24, freq="1h"),
        "temperatura": np.round(temp_h, 1),
        "umidade": np.round(hum_h, 0),
        "precipitacao_mm": prcp_h,
        "vento_velocidade_ms": np.round(vento_h, 1),
        "pressao_hpa": np.round(press_h, 0),
    })

    # Chance horária derivada da chance diária do classificador, seguindo a
    # curva de concentração de chuva ao longo do dia. Normaliza pelo PICO
    # (não pela soma): probabilidade não é aditiva como mm, então a hora de
    # pico deve ficar perto da chance_dia inteira, não de uma fração diluída
    # por 24 horas (o que fazia a aba horária mostrar "Sem Chuva" o dia
    # inteiro mesmo quando a aba diária dizia "Garoa"/"Chuva").
    chance_dia = float(linha_diaria.get("chance_chuva", 0) or 0)
    pico = float(prcp_h.max())
    if pico > 0:
        df["chance_chuva"] = np.clip(np.round(chance_dia * prcp_h / pico), 0, 100).astype(int)
    else:
        df["chance_chuva"] = 0

    classificacoes = df.apply(classificar_clima, axis=1, result_type="expand")
    df["tipo_precipitacao"] = classificacoes[0]
    df["emoji_clima"] = classificacoes[1]

    return df
