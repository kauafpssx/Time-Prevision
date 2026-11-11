"""
Script CLI para treinar e salvar modelo de ML
- Um modelo por variável (cada uma treina com TODOS os dias válidos dela)
- Ensemble LightGBM + HistGradientBoosting (leve, eficiente em memória)
- Precipitação em 2 ESTÁGIOS: classificador (chove? -> chance em %) + regressão
  da quantidade apenas nos dias chuvosos (evita "chuva todos os dias" artificial)
- Features SEM vazamento de dados (módulo features.py)
- Validação cruzada temporal (TimeSeriesSplit) para métricas honestas
- Busca de hiperparâmetros por variável com Optuna (budget de tempo controlado)
- Modelo final de produção é refeito com TODO o histórico disponível
  (a validação cruzada usa só dados de teste temporal; o modelo salvo não
  descarta os dados recentes, ao contrário do split único usado antes)
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime

# Console do Windows (cp1252) não decodifica emoji dos logs quando a saída é
# redirecionada (pipe/arquivo) em vez de terminal interativo -> crash silencioso
# no meio do treino. Força UTF-8 sempre, com fallback seguro.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, fbeta_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

from time_prevision.domain.weather import MEANINGFUL_RAIN_MM
from time_prevision.infrastructure.ml.features import VARS_NUMERICAS, _features_temporais, _lista_features, HORIZONTES_MODELO
from time_prevision.infrastructure.paths import DATABASE_PATH, MODEL_DIR
from time_prevision.infrastructure.providers import open_meteo
from time_prevision.infrastructure.providers.open_meteo import NWP_FEATS

optuna.logging.set_verbosity(optuna.logging.WARNING)

DB_PATH = DATABASE_PATH
MODELO_DIR = MODEL_DIR
MODELO_PATH = MODEL_DIR / "modelo_treinado.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"
METRICAS_PATH = MODEL_DIR / "metricas.pkl"
HYPERPARAMS_PATH = MODEL_DIR / "hyperparams.json"

# Limiar de "dia com chuva" (mm)
LIMIAR_CHUVA = MEANINGFUL_RAIN_MM

# --- Segurança de hardware / tempo (notebook, sem GPU útil para este dataset) ---
N_JOBS_SEGURO = max(1, int(os.getenv("TIME_PREVISION_N_JOBS", max(1, (os.cpu_count() or 4) - 1))))
N_SPLITS_CV = 5                 # janelas expansivas de validação temporal
OPTUNA_TRIALS_POR_ETAPA = 25    # teto de trials por busca
OPTUNA_TIMEOUT_POR_ETAPA = 90   # segundos, o que vier primeiro cancela a busca
CAUDA_EARLY_STOP_DIAS = 60      # cauda recente usada só p/ decidir nº de árvores

# HistGradientBoosting do sklearn (histogramas = leve e rápido).
# Não entra na busca Optuna (já tem early stopping nativo e é o "peso leve"
# do ensemble); é o LightGBM, mais sensível a hiperparâmetros, quem mais
# se beneficia do tuning.
CONFIG_HGB = dict(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.5,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.1,
)

BASE_LGBM = dict(random_state=42, n_jobs=N_JOBS_SEGURO, verbosity=-1)

# --- Elapsed time / banner (usa 'gum' se disponível, cai pra texto puro senão) ---
_INICIO_SCRIPT = time.time()
_GUM = shutil.which("gum")


def _elapsed():
    """Tempo decorrido desde o início do script, formatado mm:ss"""
    s = int(time.time() - _INICIO_SCRIPT)
    return f"{s // 60:02d}:{s % 60:02d}"


def log(msg):
    """Log com timestamp + tempo decorrido"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp} | ⏱ {_elapsed()}] {msg}")


def banner(titulo, subtitulo=""):
    """Banner de destaque - usa 'gum style' se instalado, senão texto simples"""
    if _GUM:
        texto = titulo if not subtitulo else f"{titulo}\n{subtitulo}"
        try:
            subprocess.run(
                [_GUM, "style", "--border", "rounded", "--padding", "0 2",
                 "--border-foreground", "212", "--foreground", "212", "--bold", texto],
                check=False,
            )
            return
        except Exception:
            pass
    print("=" * 80)
    print(titulo)
    if subtitulo:
        print(subtitulo)
    print("=" * 80)


def _carregar_hiperparams_cache():
    """Cache de hiperparâmetros já otimizados (evita re-rodar Optuna a cada dia novo)"""
    if not os.path.exists(HYPERPARAMS_PATH):
        return {}
    with open(HYPERPARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load(db_path=DB_PATH):
    """Carrega dados do banco"""
    log("📂 Carregando dados do banco...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM daily ORDER BY data", conn)
    conn.close()
    df["data"] = pd.to_datetime(df["data"])
    log(f"✓ Carregados {len(df)} registros de {df['data'].min().year} a {df['data'].max().year}")
    return df


def _espaco_busca(trial, extra=None):
    """Espaço de hiperparâmetros comum a regressores/classificadores LightGBM"""
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    )
    if extra:
        params.update(extra)
    return params


def _fit_com_early_stopping(modelo, X_tr, y_tr, X_va, y_va):
    """Ajusta um LightGBM com early stopping (se houver validação suficiente)"""
    if len(X_va) >= 20:
        modelo.fit(
            X_tr, y_tr,
            eval_X=X_va, eval_y=y_va,
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
    else:
        modelo.fit(X_tr, y_tr)
    return modelo


def _cv_mae_regressor(params, X, y, n_splits=N_SPLITS_CV):
    """MAE médio via TimeSeriesSplit - usado como objetivo do Optuna"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    erros = []
    for idx_tr, idx_te in tscv.split(X):
        corte_va = int(len(idx_tr) * 0.9)
        idx_tr_fit, idx_va = idx_tr[:corte_va], idx_tr[corte_va:]
        modelo = LGBMRegressor(**BASE_LGBM, n_estimators=1000, **params)
        _fit_com_early_stopping(modelo, X.iloc[idx_tr_fit], y.iloc[idx_tr_fit],
                                 X.iloc[idx_va], y.iloc[idx_va])
        pred = modelo.predict(X.iloc[idx_te])
        erros.append(mean_absolute_error(y.iloc[idx_te], pred))
    return float(np.mean(erros))


def _cv_auc_classificador(params, X, y_bin, n_splits=N_SPLITS_CV):
    """AUC médio (a maximizar) via TimeSeriesSplit"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for idx_tr, idx_te in tscv.split(X):
        corte_va = int(len(idx_tr) * 0.9)
        idx_tr_fit, idx_va = idx_tr[:corte_va], idx_tr[corte_va:]
        modelo = LGBMClassifier(**BASE_LGBM, n_estimators=1000, **params)
        _fit_com_early_stopping(modelo, X.iloc[idx_tr_fit], y_bin.iloc[idx_tr_fit],
                                 X.iloc[idx_va], y_bin.iloc[idx_va])
        if y_bin.iloc[idx_te].nunique() < 2:
            continue  # fold sem os dois lados da classe - AUC indefinido
        proba = modelo.predict_proba(X.iloc[idx_te])[:, 1]
        aucs.append(roc_auc_score(y_bin.iloc[idx_te], proba))
    return float(np.mean(aucs)) if aucs else 0.5


def _buscar_hiperparametros(nome_etapa, objetivo, direcao):
    """Roda um estudo Optuna com budget de tempo/trials controlado"""
    log(f"   🔍 Otimizando '{nome_etapa}' (até {OPTUNA_TRIALS_POR_ETAPA} trials "
        f"ou {OPTUNA_TIMEOUT_POR_ETAPA}s)...")
    inicio = time.time()
    study = optuna.create_study(direction=direcao,
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objetivo, n_trials=OPTUNA_TRIALS_POR_ETAPA,
                   timeout=OPTUNA_TIMEOUT_POR_ETAPA, n_jobs=1, show_progress_bar=False)
    duracao = time.time() - inicio
    log(f"   ✓ {len(study.trials)} trials em {duracao:.0f}s | melhor valor: {study.best_value:.4f}")
    return study.best_params


def _fit_producao_lgbm(cls, base_params, params, X, y, cauda_dias=CAUDA_EARLY_STOP_DIAS):
    """
    Refit final usando TODO o histórico. Primeiro descobre o nº ideal de
    árvores com early stopping numa cauda recente (sem usá-la p/ treinar),
    depois refaz o fit incluindo a cauda inteira, sem descartar dados.
    """
    n = len(X)
    if n > cauda_dias + 50:
        corte = n - cauda_dias
        X_tr, y_tr = X.iloc[:corte], y.iloc[:corte]
        X_va, y_va = X.iloc[corte:], y.iloc[corte:]
        sonda = cls(**base_params, n_estimators=2000, **params)
        _fit_com_early_stopping(sonda, X_tr, y_tr, X_va, y_va)
        n_estimators_final = max(getattr(sonda, "best_iteration_", None) or 200, 50)
    else:
        n_estimators_final = 500

    final = cls(**base_params, n_estimators=n_estimators_final, **params)
    final.fit(X, y)
    return final


def _treinar_variavel_padrao(var, X, y, cache=None):
    """Tuning + CV honesto + refit de produção p/ uma variável contínua"""
    if cache and var in cache:
        log(f"   ⚡ Usando hiperparâmetros em cache p/ '{var}' (sem busca Optuna)")
        melhores_params = cache[var]
    else:
        def objetivo(trial):
            return _cv_mae_regressor(_espaco_busca(trial), X, y)

        melhores_params = _buscar_hiperparametros(var, objetivo, "minimize")

    # Métricas honestas de CV com os melhores hiperparâmetros
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    maes, rmses, r2s, maes_pers = [], [], [], []
    for idx_tr, idx_te in tscv.split(X):
        corte_va = int(len(idx_tr) * 0.9)
        idx_tr_fit, idx_va = idx_tr[:corte_va], idx_tr[corte_va:]
        modelo = LGBMRegressor(**BASE_LGBM, n_estimators=1000, **melhores_params)
        _fit_com_early_stopping(modelo, X.iloc[idx_tr_fit], y.iloc[idx_tr_fit],
                                 X.iloc[idx_va], y.iloc[idx_va])
        real = y.iloc[idx_te].values
        pred = modelo.predict(X.iloc[idx_te])
        pers = np.roll(real, 1)
        pers[0] = np.nan
        maes.append(mean_absolute_error(real, pred))
        rmses.append(float(np.sqrt(np.mean((real - pred) ** 2))))
        r2s.append(max(r2_score(real, pred), 0))
        maes_pers.append(float(np.nanmean(np.abs(real - pers))))

    metricas = {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "mae_persistencia": float(np.mean(maes_pers)),
        "r2": float(np.mean(r2s)),
    }

    # Refit de produção com TODO o histórico
    lgbm_final = _fit_producao_lgbm(LGBMRegressor, BASE_LGBM, melhores_params, X, y)
    hgb_final = HistGradientBoostingRegressor(**CONFIG_HGB)
    hgb_final.fit(X, y)

    pred_ens = (lgbm_final.predict(X) + hgb_final.predict(X)) / 2.0
    metricas["mae_lgbm"] = float(mean_absolute_error(y, lgbm_final.predict(X)))
    metricas["mae_hgb"] = float(mean_absolute_error(y, hgb_final.predict(X)))

    return {"lgbm": lgbm_final, "histgb": hgb_final}, metricas, melhores_params


def _treinar_precipitacao(X, y, cache=None):
    """Precipitação em 2 estágios: classificador (chove?) + regressão (quantidade)"""
    y = y.clip(lower=0.0)
    chuva = (y >= LIMIAR_CHUVA).astype(int)
    mask_chuva = chuva == 1
    cache_precip = (cache or {}).get("precipitacao_mm")

    if cache_precip:
        log("   ⚡ Usando hiperparâmetros em cache p/ 'precipitacao_mm' (sem busca Optuna)")
        melhores_clf = cache_precip["clf"]
        melhores_reg_final = cache_precip["reg"]
    else:
        def objetivo_clf(trial):
            return _cv_auc_classificador(_espaco_busca(trial), X, chuva)

        melhores_clf = _buscar_hiperparametros("precipitacao_clf", objetivo_clf, "maximize")

        def objetivo_reg(trial):
            params = _espaco_busca(trial, extra={"objective": "tweedie", "tweedie_variance_power": 1.2})
            return _cv_mae_regressor(params, X[mask_chuva], y[mask_chuva])

        melhores_reg = _buscar_hiperparametros("precipitacao_reg", objetivo_reg, "minimize")
        melhores_reg_final = {**melhores_reg, "objective": "tweedie", "tweedie_variance_power": 1.2}

    # Métricas honestas de CV (pipeline completo: proba * quantidade)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    maes, rmses, r2s, maes_pers, accs, aucs = [], [], [], [], [], []
    oof_probas, oof_reais = [], []
    for idx_tr, idx_te in tscv.split(X):
        corte_va = int(len(idx_tr) * 0.9)
        idx_tr_fit, idx_va = idx_tr[:corte_va], idx_tr[corte_va:]

        clf = LGBMClassifier(**BASE_LGBM, n_estimators=1000, **melhores_clf)
        _fit_com_early_stopping(clf, X.iloc[idx_tr_fit], chuva.iloc[idx_tr_fit],
                                 X.iloc[idx_va], chuva.iloc[idx_va])

        mask_tr_chuva = chuva.iloc[idx_tr_fit] == 1
        reg = LGBMRegressor(**BASE_LGBM, n_estimators=1000, **melhores_reg_final)
        if mask_tr_chuva.sum() >= 20:
            reg.fit(X.iloc[idx_tr_fit][mask_tr_chuva], y.iloc[idx_tr_fit][mask_tr_chuva])
        else:
            reg.fit(X.iloc[idx_tr_fit], y.iloc[idx_tr_fit])

        real = y.iloc[idx_te].values
        chuva_real = chuva.iloc[idx_te].values
        proba = clf.predict_proba(X.iloc[idx_te])[:, 1]
        oof_probas.extend(proba.tolist())
        oof_reais.extend(chuva_real.tolist())
        quant = reg.predict(X.iloc[idx_te])
        pred = proba * quant

        pers = np.roll(real, 1)
        pers[0] = np.nan
        maes.append(mean_absolute_error(real, pred))
        rmses.append(float(np.sqrt(np.mean((real - pred) ** 2))))
        r2s.append(max(r2_score(real, pred), 0))
        maes_pers.append(float(np.nanmean(np.abs(real - pers))))
        accs.append(float(accuracy_score(chuva_real, clf.predict(X.iloc[idx_te]))))
        if len(np.unique(chuva_real)) >= 2:
            aucs.append(float(roc_auc_score(chuva_real, proba)))

    oof_probas = np.asarray(oof_probas, dtype=float)
    oof_reais = np.asarray(oof_reais, dtype=int)
    # Seleção de limiar e métrica em cauda temporal separada. Antes o mesmo
    # OOF era usado para ajustar e avaliar a isotônica, deixando o Brier e o
    # limiar otimistas. O calibrador final ainda usa todo OOF após a avaliação.
    corte_cal = max(30, int(len(oof_probas) * 0.75))
    calibrador_avaliacao = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    calibrador_avaliacao.fit(oof_probas[:corte_cal], oof_reais[:corte_cal])
    probas_avaliacao = calibrador_avaliacao.predict(oof_probas[corte_cal:])
    reais_avaliacao = oof_reais[corte_cal:]
    candidatos = np.arange(0.30, 0.71, 0.01)
    # F0.5 favorece precisão sobre recall: para a condição exibida é melhor
    # omitir uma garoa incerta do que anunciar chuva por muitos dias seguidos.
    # A probabilidade contínua continua disponível para quem quiser a incerteza.
    scores = [fbeta_score(reais_avaliacao, probas_avaliacao >= t, beta=0.5, zero_division=0)
              for t in candidatos]
    limiar_otimo = float(candidatos[int(np.argmax(scores))])
    classes_calibradas = probas_avaliacao >= limiar_otimo

    calibrador = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    calibrador.fit(oof_probas, oof_reais)

    metricas = {
        "evento_chuva_mm": float(LIMIAR_CHUVA),
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "mae_persistencia": float(np.mean(maes_pers)),
        "r2": float(np.mean(r2s)),
        "acuracia_chuva": float(np.mean(accs)),
        "auc_chuva": float(np.mean(aucs)) if aucs else float("nan"),
        "brier_bruto": float(brier_score_loss(reais_avaliacao, oof_probas[corte_cal:])),
        "brier_calibrado": float(brier_score_loss(reais_avaliacao, probas_avaliacao)),
        "limiar_chuva": limiar_otimo,
        "balanced_accuracy_chuva": float(balanced_accuracy_score(reais_avaliacao, classes_calibradas)),
        "precision_chuva": float(precision_score(reais_avaliacao, classes_calibradas, zero_division=0)),
        "recall_chuva": float(recall_score(reais_avaliacao, classes_calibradas, zero_division=0)),
        "f1_chuva": float(f1_score(reais_avaliacao, classes_calibradas, zero_division=0)),
        "f0_5_chuva": float(fbeta_score(reais_avaliacao, classes_calibradas, beta=0.5, zero_division=0)),
    }

    # Refit de produção com TODO o histórico
    clf_final = _fit_producao_lgbm(LGBMClassifier, BASE_LGBM, melhores_clf, X, chuva)
    mask_final = chuva == 1
    reg_final = LGBMRegressor(**BASE_LGBM, n_estimators=500, **melhores_reg_final)
    reg_final.fit(X[mask_final], y[mask_final])

    hiperparams = {"clf": melhores_clf, "reg": melhores_reg_final}
    return {"clf": clf_final, "reg": reg_final, "calibrator": calibrador,
            "threshold": limiar_otimo}, metricas, hiperparams


def treinar_e_avaliar(retune=False):
    """Treina um modelo por variável (com tuning + CV temporal), avalia e salva"""
    os.makedirs(MODELO_DIR, exist_ok=True)
    banner("🤖 TREINAMENTO DO MODELO DE PREVISÃO METEOROLÓGICA")
    log(f"📁 Arquivos de modelo serão salvos em: {MODELO_DIR}/")
    log("🧠 Estratégia: tuning Optuna + validação cruzada temporal + ensemble LightGBM/HistGB")
    log(f"⚙️  n_jobs={N_JOBS_SEGURO} (reservando núcleos p/ o sistema), "
        f"budget ≈{OPTUNA_TIMEOUT_POR_ETAPA}s por busca")

    cache = {} if retune else _carregar_hiperparams_cache()
    if cache:
        log(f"⚡ Cache de hiperparâmetros encontrado ({len(cache)} variáveis) — "
            f"pulando busca Optuna, só refit com dado atualizado")
    elif retune:
        log("🔁 --retune: ignorando cache, buscando hiperparâmetros do zero")

    inicio_total = time.time()

    df = _load()

    log("🌐 Atualizando cache de previsão NWP (GFS/ECMWF/ICON) para blending...")
    try:
        n_nwp = open_meteo.atualizar_historico()
        log(f"   ✓ {n_nwp} dias de NWP histórico cacheados")
    except Exception as e:
        log(f"   ⚠️ Falha ao buscar NWP ({e}) — segue sem essa feature (fica NaN)")
    nwp_hist = open_meteo.carregar_historico()

    # Features SEM NWP ainda (mesclado por horizonte abaixo — o NWP de cada
    # linha precisa ser o do dia-ALVO daquele horizonte, não do dia da linha).
    base = _features_temporais(df, VARS_NUMERICAS, nwp_df=None)
    feats = [f for f in _lista_features(VARS_NUMERICAS) if f in base.columns]
    log(f"📊 Total de features: {len(feats)}")
    log(f"🪜 Horizontes diretos treinados: {HORIZONTES_MODELO} "
        f"(dia solicitado usa o horizonte treinado mais próximo, sem recursão)")

    datas = base["data"]

    modelos = {}      # modelos[var][h] = ...
    metricas = {}      # metricas[var][h] = ...
    hiperparams = {}   # hiperparams[var] = ... (mesmo hiperparâmetro reaproveitado em todos os horizontes)

    for h in HORIZONTES_MODELO:
        log(f"\n{'=' * 60}")
        log(f"🪜 HORIZONTE t+{h} dia(s)")
        log(f"{'=' * 60}")

        # As features de uma linha (lag1..lag28) usam só dados até a véspera
        # da PRÓPRIA data da linha (lag1 = shift(1) = ontem em relação à
        # linha) — nunca o valor da própria linha. Por isso o alvo "1 dia à
        # frente" já é o valor NA PRÓPRIA data da linha (deslocamento 0), não
        # +1: a "véspera" já é o gap de 1 dia. deslocamento = h-1 generaliza
        # isso pros demais horizontes. Sem esse -1 os modelos treinavam pra
        # h+1 de verdade (defasados), o que piorou t+1 pra pior que o
        # baseline de persistência.
        deslocamento = h - 1
        y_base = {v: base[v].shift(-deslocamento) for v in VARS_NUMERICAS}

        # NWP mesclado deslocando a data: a linha passa a carregar o NWP
        # válido para a mesma data-alvo do deslocamento acima — mesmo
        # alinhamento usado na hora de prever de verdade (buscar_previsao_futura).
        if not nwp_hist.empty:
            nwp_deslocado = nwp_hist.copy()
            nwp_deslocado["data"] = nwp_deslocado["data"] - pd.Timedelta(days=deslocamento)
            base_h = base.drop(columns=NWP_FEATS, errors="ignore").merge(
                nwp_deslocado[["data"] + NWP_FEATS], on="data", how="left"
            )
        else:
            base_h = base

        X_full_h = base_h[feats].astype(float)

        for var in VARS_NUMERICAS:
            try:
                log(f"\n🎯 Variável: {var} (t+{h})")
                y_full = y_base[var].astype(float)
                mascara = y_full.notna()
                X = X_full_h[mascara].reset_index(drop=True)
                y = y_full[mascara].reset_index(drop=True)
                d = datas[mascara].reset_index(drop=True)
                log(f"   {len(X)} amostras válidas ({d.min().date()} a {d.max().date()})")

                if len(X) < N_SPLITS_CV * 30:
                    log(f"   ⚠️ dados insuficientes para {N_SPLITS_CV} folds, pulando")
                    continue

                if var == "precipitacao_mm":
                    modelo_var, metrica_var, hp_var = _treinar_precipitacao(X, y, cache=cache)
                else:
                    modelo_var, metrica_var, hp_var = _treinar_variavel_padrao(var, X, y, cache=cache)

                modelos.setdefault(var, {})[h] = modelo_var
                metricas.setdefault(var, {})[h] = metrica_var
                if h == 1:
                    hiperparams[var] = hp_var  # cache de hiperparâmetro só precisa 1 cópia por variável

                log(f"   ✓ MAE {metrica_var['mae']:.2f} | RMSE {metrica_var['rmse']:.2f} | "
                    f"baseline {metrica_var['mae_persistencia']:.2f} | R² {metrica_var['r2']:.2f}")

            except Exception as e:
                log(f"   ❌ Falha ao treinar '{var}' (t+{h}): {e}")
                import traceback
                traceback.print_exc()
                continue

    if not modelos:
        raise RuntimeError("Nenhuma variável foi treinada com sucesso")

    faltantes = {
        var: [h for h in HORIZONTES_MODELO if h not in modelos.get(var, {})]
        for var in VARS_NUMERICAS
    }
    faltantes = {var: horizontes for var, horizontes in faltantes.items() if horizontes}
    if faltantes:
        raise RuntimeError(f"Treinamento incompleto; variáveis/horizontes ausentes: {faltantes}")

    # Tabela de resultados (horizonte t+1, referência rápida — ver metricas.pkl p/ todos)
    banner("📈 RESULTADOS (horizonte t+1)", "Validação cruzada temporal (TimeSeriesSplit) - métricas honestas")
    print(f"{'Variável':<22}{'MAE':>8}{'RMSE':>8}{'Baseline':>10}{'R²':>7}{'Ganho%':>8}  {'Extra':>20}")
    print("-" * 90)
    for v, por_h in metricas.items():
        m = por_h.get(1)
        if not m:
            continue
        ganho = (1 - m["mae"] / m["mae_persistencia"]) * 100 if m["mae_persistencia"] else 0.0
        extra = ""
        if "acuracia_chuva" in m:
            extra = f"Acuracia: {m['acuracia_chuva']*100:.0f}%  AUC: {m['auc_chuva']:.2f}"
        print(
            f"{v:<22}{m['mae']:>8.2f}{m['rmse']:>8.2f}{m['mae_persistencia']:>10.2f}"
            f"{m['r2']:>7.2f}{ganho:>8.1f}  {extra:>20}"
        )

    log("\n📉 Ganho %% sobre baseline por horizonte (temperatura / precipitação):")
    for v in ("temperatura", "precipitacao_mm"):
        linha = "   " + v.ljust(18)
        for h in HORIZONTES_MODELO:
            m = metricas.get(v, {}).get(h)
            if m and m.get("mae_persistencia"):
                ganho = (1 - m["mae"] / m["mae_persistencia"]) * 100
                linha += f"t+{h}:{ganho:+.0f}% "
        log(linha)

    # Salvar
    log("\n💾 Salvando modelo, features, métricas e hiperparâmetros...")
    with open(MODELO_PATH, "wb") as f:
        import pickle
        pickle.dump(modelos, f)
    log(f"✓ Modelo salvo em: {MODELO_PATH} ({_tamanho_arquivo(MODELO_PATH)})")

    with open(FEATURES_PATH, "wb") as f:
        import pickle
        pickle.dump(feats, f)
    log(f"✓ Features salvas em: {FEATURES_PATH}")

    with open(METRICAS_PATH, "wb") as f:
        import pickle
        pickle.dump(metricas, f)
    log(f"✓ Métricas salvas em: {METRICAS_PATH}")

    with open(HYPERPARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(hiperparams, f, indent=2, ensure_ascii=False)
    log(f"✓ Hiperparâmetros salvos em: {HYPERPARAMS_PATH}")

    duracao_total = time.time() - inicio_total
    banner("✅ TREINAMENTO COMPLETO!", f"Tempo total: {duracao_total / 60:.1f} min (⏱ {_elapsed()})")
    log("💡 Agora você pode gerar os dados do site estático:")
    log("   Execute: python -m time_prevision.interfaces.cli.export_forecast --dias 30")


def _tamanho_arquivo(path):
    """Retorna tamanho do arquivo"""
    try:
        size = os.path.getsize(path)
        if size < 1024:
            return f"{size} bytes"
        elif size < 1024 ** 2:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / 1024 ** 2:.1f} MB"
    except Exception:
        return "?"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina modelo de previsão meteorológica")
    parser.add_argument("--retune", action="store_true",
                         help="Ignora cache de hiperparâmetros e busca do zero com Optuna")
    args = parser.parse_args()
    try:
        treinar_e_avaliar(retune=args.retune)
    except KeyboardInterrupt:
        log("\n❌ Treinamento interrompido pelo usuário")
    except Exception as e:
        log(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        raise
