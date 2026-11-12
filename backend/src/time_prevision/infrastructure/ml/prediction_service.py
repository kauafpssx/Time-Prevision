"""
Serviço de previsão - lógica de modelo/predição, sem dependência de GUI.
Carrega modelo pré-treinado e gera previsões diárias/horárias.
"""

import numpy as np
import pandas as pd
import pickle
import sqlite3
from time_prevision.domain.weather import DRY_DAY_THRESHOLD, MEANINGFUL_RAIN_MM, classify_weather
from time_prevision.infrastructure.ml.features import VARS_NUMERICAS, _features_temporais, HORIZONTES_MODELO
from time_prevision.infrastructure.paths import DATABASE_PATH, MODEL_DIR
from time_prevision.infrastructure.providers.open_meteo import (
    NWP_FEATS, buscar_previsao_futura, buscar_previsao_horaria,
)

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
    limiar_chuva = float(DRY_DAY_THRESHOLD)
    if isinstance(modelo, dict):
        valores = []
        for var in VARS_NUMERICAS:
            m = modelo.get(var)
            if m is None:
                valores.append(float("nan"))
                continue
            if isinstance(m, dict) and "clf" in m:
                # Precipitação em 2 estágios: probabilidade calibrada vezes
                # a quantidade condicional caso chova. O resultado é o valor
                # esperado, portanto não deve ser zerado por um limiar de
                # classificação: 47% e 0 mm era um contrato contraditório.
                proba = float(m["clf"].predict_proba(entrada)[:, 1][0])
                if "calibrator" in m:
                    proba = float(m["calibrator"].predict([proba])[0])
                limiar_chuva = float(m.get("threshold", DRY_DAY_THRESHOLD / 100.0) * 100)
                quant = float(m["reg"].predict(entrada)[0])
                if var == "precipitacao_mm":
                    chance = int(round(proba * 100))
                    valores.append(float(proba * max(0.0, quant)))
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
    return valores, chance, limiar_chuva


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
            valores, chance, limiar_chuva = _prever_um_horizonte(modelo, entrada, h_baixo)
        else:
            v_baixo, c_baixo, t_baixo = _prever_um_horizonte(modelo, entrada, h_baixo)
            v_alto, c_alto, t_alto = _prever_um_horizonte(modelo, entrada, h_alto)
            w = (i - h_baixo) / (h_alto - h_baixo)
            valores = [vb + (va - vb) * w for vb, va in zip(v_baixo, v_alto)]
            limiar_chuva = t_baixo + (t_alto - t_baixo) * w
            if c_baixo is not None and c_alto is not None:
                chance = c_baixo + (c_alto - c_baixo) * w
            else:
                chance = c_baixo if c_baixo is not None else c_alto

        nova = {"data": data_alvo}
        for v, val in zip(VARS_NUMERICAS, valores):
            nova[v] = float(val)
        nova["chance_chuva"] = chance if chance is not None else chance_chuva(nova["precipitacao_mm"])
        nova["fonte_probabilidade_chuva"] = "modelo_calibrado"
        if len(linha_nwp) and "nwp_prob_chuva" in linha_nwp:
            prob_nwp = linha_nwp.iloc[0]["nwp_prob_chuva"]
            if pd.notna(prob_nwp):
                chance_modelo = float(nova["chance_chuva"])
                # O NWP ao vivo descreve o sistema meteorológico atual; o
                # modelo local atua como correção estatística, não o contrário.
                prcp_nwp = float(linha_nwp.iloc[0].get("nwp_prcp", 0) or 0)
                # A POP do provedor inclui qualquer traço de precipitação.
                # Convertemos para probabilidade de chuva relevante (>=1 mm)
                # para manter o mesmo evento usado no treinamento.
                prob_nwp_relevante = float(prob_nwp) * min(1.0, prcp_nwp / MEANINGFUL_RAIN_MM)
                chance_blend = 0.20 * chance_modelo + 0.80 * prob_nwp_relevante
                if chance_modelo > 0:
                    nova["precipitacao_mm"] *= chance_blend / chance_modelo
                nova["chance_chuva"] = float(round(np.clip(chance_blend, 0, 100)))
                nova["fonte_probabilidade_chuva"] = "blend_modelo_nwp"
        nova["limiar_chuva"] = limiar_chuva
        nova["nwp_disponivel"] = bool(len(linha_nwp))
        if i <= 5:
            nova["confianca"] = "alta"
            nova["modo_previsao"] = "operacional"
        elif i <= 10:
            nova["confianca"] = "media"
            nova["modo_previsao"] = "operacional"
        elif i <= 16:
            nova["confianca"] = "baixa"
            nova["modo_previsao"] = "operacional"
        else:
            historico_mes = df.loc[(df["data"].dt.month == data_alvo.month) &
                                   df["precipitacao_mm"].notna(), "precipitacao_mm"]
            if len(historico_mes):
                nova["frequencia_climatica_chuva"] = float(round((historico_mes >= 0.1).mean() * 100))
            # Depois do limite NWP não existe previsão diária confiável. Não
            # inventamos uma sequência de dias secos/chuvosos: expomos apenas
            # a frequência mensal como tendência separada.
            nova["chance_chuva"] = np.nan
            nova["precipitacao_mm"] = np.nan
            nova["fonte_probabilidade_chuva"] = "climatologia_mensal"
            nova["confianca"] = "muito_baixa"
            nova["modo_previsao"] = "tendencia_climatica"

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
    limiar = row.get("limiar_chuva", LIMIAR_SEM_CHUVA_PCT)
    if chance is None:
        chance = chance_chuva(prcp)

    return classify_weather(prcp, chance, limiar)


def chance_chuva(prcp_mm):
    """Converte precipitação prevista (mm) em chance de chuva em %"""
    if pd.isna(prcp_mm) or prcp_mm <= 0.01:
        return 0
    # Curva exponencial: ~63% em 1.5mm, ~86% em 3mm, ~98% em 6mm
    return int(round(100 * (1 - np.exp(-prcp_mm / 1.5))))


def _condicao_codigo_nwp(codigo):
    codigo = int(codigo)
    if codigo == 0:
        return "Céu Limpo"
    if codigo in (1, 2):
        return "Parcialmente Nublado"
    if codigo == 3:
        return "Nublado"
    if codigo in (45, 48):
        return "Neblina"
    if 51 <= codigo <= 57:
        return "Garoa"
    if codigo in (61, 63, 80, 81):
        return "Chuva Leve"
    if codigo in (65, 67, 82):
        return "Chuva Moderada"
    if codigo >= 95:
        return "Tempestade"
    return "Nublado"


def gerar_previsao_horaria(linha_diaria):
    """Usa exclusivamente a previsão NWP horária real; não inventa uma curva."""
    data = linha_diaria["data"]
    nwp_horario = buscar_previsao_horaria(data)
    nwp_horario["tipo_precipitacao"] = nwp_horario["codigo_tempo"].map(_condicao_codigo_nwp)
    nwp_horario["emoji_clima"] = ""
    return nwp_horario
