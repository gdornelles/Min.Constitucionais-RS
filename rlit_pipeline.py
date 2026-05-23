# ============================================================
#  RLIT RS — Pipeline v4
#  Correções: dados faltantes, bounds Focus Dez/2023
#  PIB 2024: 1,52%  IPCA 2024: 3,90%
#  PIB 2025: 2,00%  IPCA 2025: 3,50%
#  Modelos: HWES, SARIMAX, AR, XGBoost, MLP, SES, MA+Tend.
# ============================================================

import pandas as pd
import numpy as np
import json, warnings, os, time
warnings.filterwarnings('ignore')

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import datetime as dt
import calendar
from dateutil.relativedelta import relativedelta

# ============================================================
#  PARÂMETROS MACRO — Relatório Focus Dez/2023
# ============================================================
PIB_2024   = 0.0152   # +1,52%
IPCA_2024  = 0.0390   # +3,90%
PIB_2025   = 0.0200   # +2,00%
IPCA_2025  = 0.0350   # +3,50%

# Crescimento nominal máximo por conta
# teto = (1+PIB)*(1+IPCA) - 1 + margem setorial
NOM_24 = (1+PIB_2024)*(1+IPCA_2024) - 1   # ~5,49%
NOM_25 = (1+PIB_2025)*(1+IPCA_2025) - 1   # ~5,57%

ACCOUNT_BOUNDS = {
    # (piso_anual, teto_anual)
    'IPTU':                            (-0.02, NOM_24+0.04),
    'ISS':                             (-0.05, NOM_24+0.06),
    'ITBI':                            (-0.15, NOM_24+0.10),
    'IRRF':                            (-0.05, NOM_24+0.05),
    'Cota-Parte do FPM':               ( 0.00, NOM_24+0.05),
    'Cota-Parte do ICMS':              (-0.05, NOM_24+0.05),
    'Cota-Parte do IPVA':              ( 0.00, NOM_24+0.04),
    'Cota-Parte do ITR':               (-0.20, NOM_24+0.15),
    'Transferências da LC nº 61/1989': (-0.10, NOM_24+0.08),
    'Transferencias LC 61_1989':       (-0.10, NOM_24+0.08),
    'Transferências do FUNDEB':        ( 0.00, NOM_24+0.05),
}
BOUNDS_25 = {k: (v[0], (1+PIB_2025)*(1+IPCA_2025)-1 + (v[1]-NOM_24))
             for k, v in ACCOUNT_BOUNDS.items()}
DEFAULT_B24 = (-0.05, NOM_24+0.04)
DEFAULT_B25 = (-0.05, NOM_25+0.04)
# ============================================================
ARQUIVO_BASE  = 'df2015-2023.xlsx'
ARQUIVO_SAIDA = 'data.json'
# ============================================================

col_to_month = {'<MR-11>':1,'<MR-10>':2,'<MR-9>':3,'<MR-8>':4,
                '<MR-7>':5,'<MR-6>':6,'<MR-5>':7,'<MR-4>':8,
                '<MR-3>':9,'<MR-2>':10,'<MR-1>':11,'<MR>':12}
FULL_DATES = [f"{yr}-{mo:02d}" for yr in range(2015,2024) for mo in range(1,13)]
N_OBS = 108

# ── Tratamento de séries com dados faltantes ──────────────────
def get_series(mun_df, conta=None):
    """
    Retorna série de 108 meses com tratamento de gaps:
    - Zeros por ausência de envio → interpolados linearmente
    - Anos inteiros zerados (município não enviou) → interpolados
      por crescimento médio dos anos adjacentes
    """
    sub = mun_df[mun_df['Conta']==conta] if conta else mun_df
    grp = sub.groupby(['Date','month'])['Valor'].sum().reset_index()
    ds  = grp.apply(lambda r: f"{int(r['Date'])}-{int(r['month']):02d}", axis=1)
    lkp = dict(zip(ds, grp['Valor']))
    raw = [float(lkp.get(d, np.nan)) for d in FULL_DATES]

    # Converte zeros que representam meses não enviados → NaN
    # Heurística: se um ano inteiro (12 meses) soma zero mas
    # anos adjacentes têm valores, é falha de envio
    arr = np.array(raw)
    for yr_idx in range(9):                       # 9 anos (2015-2023)
        sl = arr[yr_idx*12:(yr_idx+1)*12]
        if np.nansum(sl) == 0:                    # ano inteiro zerado
            arr[yr_idx*12:(yr_idx+1)*12] = np.nan # marca como faltante

    # Interpolação linear para NaNs internos
    s = pd.Series(arr)
    s = s.interpolate(method='linear', limit_direction='both')

    # Se ainda restar NaN nas bordas, preenche com média dos não-NaN
    if s.isna().any():
        mean_val = s.dropna().mean()
        s = s.fillna(mean_val if not np.isnan(mean_val) else 0.0)

    return s.tolist()

# ── Features para XGBoost e MLP ───────────────────────────────
def make_features(series_vals):
    df = pd.DataFrame({'v': series_vals})
    df['sma12'] = df['v'].rolling(12).mean()
    df['lag12'] = df['v'].shift(12)
    base = dt.datetime(2015,1,1)
    df['dord'] = [(base+relativedelta(months=i)).toordinal()
                  for i in range(len(series_vals))]
    return df.dropna()

def forecast_recursive(predict_fn, history, n=24):
    vals = list(history)
    base = dt.datetime(2015,1,1)
    preds = []
    for i in range(n):
        sma12 = float(np.mean(vals[-12:]))
        lag12 = float(vals[-12])
        next_d = base + relativedelta(months=len(vals))
        dord  = int(next_d.toordinal())
        p = max(0.0, float(predict_fn(np.array([[dord, sma12, lag12]]))))
        preds.append(p); vals.append(p)
    return preds

# ── Modelos ───────────────────────────────────────────────────
def ma_fc(y, n=24):
    if len(y) < 13: return [float(np.mean(y))]*n
    anual = np.mean(y[-12:])
    prev  = np.mean(y[-24:-12]) if len(y)>=24 else np.mean(y[:12])
    tr    = (anual - prev) / 12
    return [max(0, anual + tr*(i+1)) for i in range(n)]

def ses_fc(y, n=24):
    try:
        return [max(0,v) for v in
                SimpleExpSmoothing(y).fit(optimized=True).forecast(n)]
    except: return ma_fc(y, n)

def hwes_fc(y, n=24):
    try:
        m = ExponentialSmoothing(y, trend='add', seasonal='add',
                                 seasonal_periods=12,
                                 initialization_method='estimated')
        return [max(0,v) for v in m.fit(optimized=True,disp=False).forecast(n)]
    except:
        try:
            m = ExponentialSmoothing(y, trend='add', seasonal=None,
                                     initialization_method='estimated')
            return [max(0,v) for v in m.fit(optimized=True,disp=False).forecast(n)]
        except: return ma_fc(y, n)

def ar_fc(y, n=24):
    try:
        return [max(0,v) for v in AutoReg(y,lags=[12]).fit().forecast(n)]
    except: return hwes_fc(y, n)

def sarimax_fc(y, n=24):
    try:
        m = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12),
                    initialization='approximate_diffuse',
                    enforce_stationarity=False, enforce_invertibility=False)
        return [max(0,v) for v in m.fit(disp=False,maxiter=60).forecast(n)]
    except: return hwes_fc(y, n)

def xgb_fc(y, n=24):
    try:
        df = make_features(y)
        if len(df)<10: return ma_fc(y,n)
        X,Y = df[['dord','sma12','lag12']].values, df['v'].values
        mdl = xgb.XGBRegressor(max_depth=5,alpha=0.1,reg_lambda=0.1,
                                n_estimators=500,learning_rate=0.05,
                                subsample=0.8,random_state=42,verbosity=0)
        mdl.fit(X,Y)
        return forecast_recursive(mdl.predict, y, n)
    except: return ma_fc(y,n)

def mlp_fc(y, n=24):
    try:
        df = make_features(y)
        if len(df)<10: return ma_fc(y,n)
        X,Y = df[['dord','sma12','lag12']].values, df['v'].values
        mdl = MLPRegressor(hidden_layer_sizes=18,activation='identity',
                           learning_rate='adaptive',max_iter=1000,
                           alpha=0.01,random_state=42)
        mdl.fit(X,Y)
        return forecast_recursive(mdl.predict, y, n)
    except: return ma_fc(y,n)

MODELS = {'hwes':hwes_fc,'sarimax':sarimax_fc,'ar':ar_fc,
          'xgboost':xgb_fc,'mlp':mlp_fc,'ses':ses_fc,'ma_tend':ma_fc}

# ── Balizamento Focus Dez/2023 ────────────────────────────────
def apply_bounds(fc12, base_sum, conta, year=2024):
    bounds = (ACCOUNT_BOUNDS if year==2024 else BOUNDS_25).get(
        conta, DEFAULT_B24 if year==2024 else DEFAULT_B25)
    lo, hi = bounds
    base = max(base_sum, 1.0)
    lo_v, hi_v = base*(1+lo), base*(1+hi)
    fc_sum = max(sum(fc12), 1e-6)
    if fc_sum < lo_v:
        r = lo_v/fc_sum; fc12 = [v*r for v in fc12]
    elif fc_sum > hi_v:
        r = hi_v/fc_sum; fc12 = [v*r for v in fc12]
    return [max(0.0, v) for v in fc12]

# ── Seleção do melhor modelo ──────────────────────────────────
def best_fc(y, n=24, conta=''):
    y = [float(v) for v in y]
    obs12 = sum(y[-12:])

    # Série muito curta ou toda nula → fallback direto
    if len([v for v in y if v>0]) < 24:
        fc = ma_fc(y, n)
        fc[:12]  = apply_bounds(fc[:12],  obs12,       conta, 2024)
        fc[12:]  = apply_bounds(fc[12:],  sum(fc[:12]),conta, 2025)
        return [round(v,2) for v in fc], 'ma_tend', {}

    train, val = y[:-12], y[-12:]
    val_sum    = max(sum(val), 1.0)

    # MSE de cada modelo no holdout
    errors = {}
    for name, fn in MODELS.items():
        try:
            p = fn(list(train), 12)
            errors[name] = float(mean_squared_error(val, p))
        except: errors[name] = float('inf')

    # Tenta em ordem de menor erro; aceita o 1º que passe na
    # sanidade básica (não cai >70% nem explode >300%)
    best_name, best_raw = None, None
    for name, _ in sorted(errors.items(), key=lambda x: x[1]):
        if errors[name] == float('inf'): continue
        try:
            fc = MODELS[name](y, n)
            s24 = sum(fc[:12])
            if val_sum>0 and (s24 < val_sum*0.3 or s24 > val_sum*4.0):
                continue
            best_name, best_raw = name, fc
            break
        except: continue

    if best_name is None:
        best_name, best_raw = 'ma_tend', ma_fc(y, n)

    # Aplica balizamento Focus
    fc24 = apply_bounds(best_raw[:12], obs12,          conta, 2024)
    fc25 = apply_bounds(best_raw[12:], sum(fc24),      conta, 2025)
    fc   = [round(v,2) for v in fc24+fc25]
    return fc, best_name, {k:round(v,4) for k,v in errors.items()}

# ── Carregamento ──────────────────────────────────────────────
t0 = time.time()
print("="*60)
print("  RLIT RS — Pipeline v4  (Focus Dez/2023)")
print("="*60)
print(f"\n  PIB 2024 +{PIB_2024*100:.2f}%  IPCA 2024 +{IPCA_2024*100:.2f}%"
      f"  → nominal máx ~{NOM_24*100:.1f}%")
print(f"  PIB 2025 +{PIB_2025*100:.2f}%  IPCA 2025 +{IPCA_2025*100:.2f}%"
      f"  → nominal máx ~{NOM_25*100:.1f}%")
print(f"\n Lendo {ARQUIVO_BASE}...")

df = pd.read_excel(ARQUIVO_BASE)
print(f"  {len(df):,} linhas · {df['Cod.IBGE'].nunique()} municípios · {df['Date'].nunique()} anos")
df['month'] = df['Coluna'].map(col_to_month)

pop_latest = df.groupby('Cod.IBGE').apply(
    lambda g: int(g.loc[g['Date'].idxmax(),'População'])).to_dict()
nome_map   = df.groupby('Cod.IBGE')['Instituição'].first().to_dict()
ibge_list  = sorted(df['Cod.IBGE'].unique().tolist())
contas     = sorted(df['Conta'].unique().tolist())

meses = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']
obs_labels  = [f"{meses[int(d[5:])-1]}/{d[2:4]}" for d in FULL_DATES]
pred_labels = [f"{meses[mo-1]}/{str(yr)[2:]}"
               for yr in [2024,2025] for mo in range(1,13)]

# ── Processamento ─────────────────────────────────────────────
total = len(ibge_list); all_data = {}
print(f"\n Processando {total} municípios...")
print(f"  Modelos : HWES · SARIMAX · AR · XGBoost · MLP · SES · MA+Tend.")
print(f"  Gaps    : anos inteiros zerados → interpolação linear")
print(f"  Bounds  : variação anual limitada por Focus Dez/2023\n")

for i, ibge in enumerate(ibge_list):
    nome       = str(nome_map[ibge])
    nome_clean = nome.replace('Prefeitura Municipal de ','').replace(' - RS','').strip()
    pop        = pop_latest.get(ibge, 0)
    mun_df     = df[df['Cod.IBGE']==ibge]

    contas_data   = {}
    rlit_obs      = np.zeros(N_OBS)
    rlit_pred_sum = np.zeros(24)

    for conta in contas:
        series = get_series(mun_df, conta)   # com tratamento de gaps
        rlit_obs += np.array(series)
        fc, best, errs = best_fc(series, 24, conta)
        rlit_pred_sum += np.array(fc)
        contas_data[conta] = {
            'obs': [round(v,2) for v in series],
            'pred': fc, 'best_model': best, 'errors': errs
        }

    rs            = [round(v,2) for v in rlit_obs.tolist()]
    rfc,rbest,_   = best_fc(rs, 24, 'RLIT_TOTAL')

    all_data[str(ibge)] = {
        'nome': nome_clean, 'pop': pop, 'contas': contas_data,
        'rlit_obs':  rs,
        'rlit_pred': [round(v,2) for v in rlit_pred_sum.tolist()],
        'rlit_best': rbest
    }

    elapsed = time.time()-t0; done = i+1
    eta     = (elapsed/done)*(total-done)
    bar     = '#'*int(done/total*30)+'-'*(30-int(done/total*30))
    print(f"\r  [{bar}] {done}/{total}  "
          f"{elapsed/60:.0f}min · ETA {eta/60:.0f}min  ",
          end='', flush=True)

print(f"\n\n Salvando {ARQUIVO_SAIDA}...")
payload = {
    'municipios':  all_data,
    'obs_labels':  obs_labels,
    'pred_labels': pred_labels,
    'contas':      contas,
    'mr_date':     'Dezembro/2023',
    'pred_years':  '2024 e 2025',
    'macro': {'pib_2024':PIB_2024,'ipca_2024':IPCA_2024,
              'pib_2025':PIB_2025,'ipca_2025':IPCA_2025}
}
with open(ARQUIVO_SAIDA,'w',encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, separators=(',',':'))

sz = os.path.getsize(ARQUIVO_SAIDA)
print(f"  Arquivo : {ARQUIVO_SAIDA}  ({sz/1024/1024:.1f} MB)")
print(f"  Munis   : {len(all_data)}")
print(f"  Tempo   : {(time.time()-t0)/60:.1f} min")
print("\nPronto! Coloque data.json + index.html na pasta RLIT")
print("e abra com: python -m http.server 8080")
