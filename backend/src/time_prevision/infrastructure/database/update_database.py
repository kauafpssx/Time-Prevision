from meteostat import daily, hourly, config
from meteostat.enumerations import Provider
from datetime import datetime
import sqlite3
import sys
import pandas as pd
from time_prevision.infrastructure.paths import DATABASE_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

config.block_large_requests = False
# O provider padrão "hourly" é um dump em lote da Meteostat que só é
# republicado a cada ~10 dias -> sem isso, os dias mais recentes nunca
# aparecem no banco. METAR (relatório de aeródromo) é quase tempo real e
# cobre o gap; HOURLY completa o histórico mais antigo que o METAR não tem.
config.aviationweather_user_agent = "time-prevision-app (contato: kauaff3@gmail.com)"
PROVIDERS_HORARIO = [Provider.METAR, Provider.HOURLY]

PORTO = "83967"
DB_PATH = DATABASE_PATH

inicio = datetime(1977, 1, 1)
fim = datetime.now()

RENAME = {
    "temp": "temperatura",
    "tmin": "temperatura_minima",
    "tmax": "temperatura_maxima",
    "rhum": "umidade",
    "prcp": "precipitacao_mm",
    "snwd": "neve_mm",
    "wdir": "direcao_vento",
    "wspd": "vento_velocidade_ms",
    "wpgt": "vento_rajada_ms",
    "pres": "pressao_hpa",
    "tsun": "insolacao_min",
    "cldc": "nebulosidade",
    "coco": "codigo_tempo",
}

# Soma que respeita ausência de dado: um grupo 100% NaN vira NaN (não 0.0).
# pandas .sum() por padrão retorna 0.0 para grupos totalmente NaN (skipna=True),
# o que faz "sem dado" virar silenciosamente "confirmado sem chuva/insolação".
def _soma_sem_dado_e_nan(s):
    return s.sum(min_count=1)


# Colunas agregadas do HORÁRIO para o DIÁRIO.
# O horário tem cobertura muito maior que o diário; com isso TODOS os dias entram.
COLUNAS_AGREGADAS = {
    "temperatura": ("temp", "mean"),
    "temperatura_minima": ("temp", "min"),
    "temperatura_maxima": ("temp", "max"),
    "umidade": ("rhum", "mean"),
    "precipitacao_mm": ("prcp", _soma_sem_dado_e_nan),
    "neve_mm": ("snwd", "mean"),
    "vento_velocidade_ms": ("wspd", "mean"),
    "vento_rajada_ms": ("wpgt", "max"),
    "pressao_hpa": ("pres", "mean"),
    "insolacao_min": ("tsun", _soma_sem_dado_e_nan),
    "nebulosidade": ("cldc", "mean"),
}


def classificar_clima(row):
    """Classifica o tipo de clima baseado nas variáveis meteorológicas"""
    prcp = row.get('precipitacao_mm', 0) or 0
    vento = row.get('vento_velocidade_ms', 0) or 0
    press = row.get('pressao_hpa', 1013) or 1013
    neb = row.get('nebulosidade', 0) or 0

    if prcp == 0:
        tipo_chuva = 'Sem Chuva'
    elif prcp < 2.5:
        tipo_chuva = 'Garoa'
    elif prcp < 10:
        tipo_chuva = 'Chuva Leve'
    elif prcp < 50:
        tipo_chuva = 'Chuva Moderada'
    else:
        tipo_chuva = 'Chuva Forte'

    if prcp > 20 and vento > 10 and press < 1005:
        tipo_chuva = 'Tempestade'
    elif prcp > 10 and vento > 15:
        tipo_chuva = 'Temporal'

    if neb < 2:
        ceu = 'Céu Limpo'
    elif neb < 5:
        ceu = 'Parcialmente Nublado'
    elif neb < 7:
        ceu = 'Nublado'
    else:
        ceu = 'Muito Nublado'

    if vento < 3.4:
        intensidade_vento = 'Calmo'
    elif vento < 5.5:
        intensidade_vento = 'Brisa Leve'
    elif vento < 8.0:
        intensidade_vento = 'Brisa Moderada'
    elif vento < 10.8:
        intensidade_vento = 'Brisa Forte'
    elif vento < 13.9:
        intensidade_vento = 'Vento Moderado'
    else:
        intensidade_vento = 'Vento Forte'

    return tipo_chuva, ceu, intensidade_vento


print("Carregando dados meteorológicos...")
conn = sqlite3.connect(DB_PATH)

df_daily = daily(PORTO, inicio, fim).fetch().reset_index()
df_hourly = hourly(PORTO, inicio, fim, providers=PROVIDERS_HORARIO).fetch().reset_index()

df_daily["data"] = pd.to_datetime(df_daily["time"]).dt.normalize()
df_hourly["data"] = pd.to_datetime(df_hourly["time"]).dt.normalize()

# ==========================================================================
# DIÁRIO COMPLETO: agrega o horário e preenche TODOS os dias disponíveis.
# Prioriza os valores oficiais do diário e usa o horário para preencher lacunas.
# ==========================================================================
print("Agregando horário -> diário (todos os dias)...")
contagens = df_hourly.groupby("data").size()
agg = df_hourly.groupby("data").agg(
    **{k: (v[0], v[1]) for k, v in COLUNAS_AGREGADAS.items()}
).reset_index()

df_daily = df_daily.rename(columns=RENAME)
df_daily = df_daily.drop(columns=["time"], errors="ignore")
df_daily = df_daily.merge(agg, on="data", how="outer", suffixes=("", "_hora"))

for col in COLUNAS_AGREGADAS:
    hora = f"{col}_hora"
    if hora in df_daily.columns:
        # astype(float64) evita erro de cast estrito quando a coluna nativa vem
        # como inteiro anulável (ex.: nebulosidade) e o fallback horário é float
        primaria = pd.to_numeric(df_daily[col], errors="coerce").astype("float64")
        fallback = pd.to_numeric(df_daily[hora], errors="coerce").astype("float64")
        df_daily[col] = primaria.fillna(fallback)
        df_daily = df_daily.drop(columns=[hora])

df_daily = df_daily.sort_values("data").reset_index(drop=True)

# Remove o último dia se estiver incompleto (< 24 horas): é o dia ATUAL em
# andamento. Assim a 1ª previsão volta a ser HOJE (sem perder dias históricos).
ultimo = df_daily["data"].max()
if int(contagens.get(ultimo, 0)) < 24:
    df_daily = df_daily[df_daily["data"] != ultimo]
    print(f"⚠️  Removido dia atual incompleto: {ultimo.date()} ({int(contagens.get(ultimo, 0))} horas)")

df_hourly = df_hourly.rename(columns=RENAME).drop(columns=["time"], errors="ignore")

print(f"✓ Diário com {len(df_daily)} dias (de 1977 a {df_daily['data'].max().year})")

# Classificações e features derivadas para TODOS os dias
print("Classificando condições climáticas...")
classificacoes = df_daily.apply(classificar_clima, axis=1, result_type='expand')
df_daily['tipo_precipitacao'] = classificacoes[0]
df_daily['condicao_ceu'] = classificacoes[1]
df_daily['intensidade_vento'] = classificacoes[2]

print("Calculando features meteorológicas derivadas...")
df_daily['amplitude_termica'] = df_daily['temperatura_maxima'] - df_daily['temperatura_minima']
df_daily['ponto_orvalho_aprox'] = df_daily['temperatura'] - ((100 - df_daily['umidade']) / 5)
df_daily['indice_calor'] = df_daily['temperatura'] + (0.33 * df_daily['umidade'] / 100) * df_daily['temperatura'] - 0.70 * (df_daily['vento_velocidade_ms'] * 3.6) - 4.00

# Salvar no banco com índices otimizados
print("Salvando dados no banco...")
df_daily.to_sql("daily", conn, if_exists="replace", index=False)
df_hourly.to_sql("hourly", conn, if_exists="replace", index=False)

print("Criando índices...")
conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_data ON daily(data)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_prec ON daily(precipitacao_mm)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_temp ON daily(temperatura)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_tipo ON daily(tipo_precipitacao)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_hourly_data ON hourly(data)")

print("Criando tabela de estatísticas agregadas...")
stats_query = """
CREATE TABLE IF NOT EXISTS estatisticas_mensais AS
SELECT 
    strftime('%Y-%m', data) as ano_mes,
    AVG(temperatura) as temp_media,
    AVG(temperatura_minima) as temp_min_media,
    AVG(temperatura_maxima) as temp_max_media,
    AVG(umidade) as umidade_media,
    SUM(precipitacao_mm) as precipitacao_total,
    AVG(vento_velocidade_ms) as vento_medio,
    AVG(pressao_hpa) as pressao_media,
    COUNT(*) as dias_com_dados
FROM daily
GROUP BY ano_mes
"""
conn.execute("DROP TABLE IF EXISTS estatisticas_mensais")
conn.execute(stats_query)
conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_mes ON estatisticas_mensais(ano_mes)")

conn.commit()
conn.close()

print(f"\n✓ Banco de dados criado com sucesso!")
print(f"✓ Total de registros diários: {len(df_daily)} (usando todos os dias do horário)")
print(f"✓ Total de registros horários: {len(df_hourly)}")
print(f"✓ Dias sem dados (NaN nas 8 variáveis): {int(df_daily[['temperatura','umidade','precipitacao_mm','vento_velocidade_ms','pressao_hpa','nebulosidade']].isna().all(axis=1).sum())}")
