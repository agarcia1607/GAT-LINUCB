"""
reports/recovery_analysis.py
Análisis de recuperación de crisis — 10 años
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import glob
from pathlib import Path

WEEKS = 52

# --- Métricas base ---
def sharpe(r): return np.sqrt(WEEKS)*r.mean()/r.std() if r.std()>0 else 0
def sortino(r):
    down = r[r<0].std()
    return np.sqrt(WEEKS)*r.mean()/down if down>0 else 0
def ann_return(r): return (1+r).prod()**(WEEKS/len(r))-1
def max_dd(r):
    cum=(1+r).cumprod()
    return ((cum-cum.cummax())/(cum.cummax()+1)).min()

# --- Cargar run más reciente ---
def load_latest_run():
    runs = sorted(glob.glob('artifacts/linucb/run_embeddings_*/'))
    for run in reversed(runs):
        try:
            df = pd.read_csv(f'{run}logs_linucb.csv')
            if len(df) > 100:
                print(f"Usando: {run}")
                return df, run
        except:
            continue
    raise FileNotFoundError("No runs found")

# --- Análisis de recuperación ---
def recovery_analysis(r, dates, label, threshold=-0.05):
    """
    Encuentra todos los períodos de drawdown > threshold
    y calcula tiempo de caída y recuperación.
    """
    cum = (1 + r).cumprod().values
    dates = pd.to_datetime(dates)

    peak = cum[0]
    peak_idx = 0
    in_drawdown = False
    drawdown_start = None
    trough_idx = None
    trough_val = None

    events = []

    for i in range(1, len(cum)):
        if cum[i] > peak:
            if in_drawdown and trough_idx is not None:
                # Recuperación completa
                dd_pct = (trough_val - peak) / peak
                weeks_down = trough_idx - peak_idx
                weeks_recovery = i - trough_idx
                events.append({
                    'label': label,
                    'peak_date': dates.iloc[peak_idx].date(),
                    'trough_date': dates.iloc[trough_idx].date(),
                    'recovery_date': dates.iloc[i].date(),
                    'drawdown': f"{dd_pct:.1%}",
                    'weeks_to_trough': weeks_down,
                    'weeks_to_recover': weeks_recovery,
                    'total_weeks': weeks_down + weeks_recovery,
                })
                in_drawdown = False
            peak = cum[i]
            peak_idx = i
        else:
            dd = (cum[i] - peak) / peak
            if dd < threshold:
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = i
                    trough_idx = i
                    trough_val = cum[i]
                elif cum[i] < trough_val:
                    trough_idx = i
                    trough_val = cum[i]

    return pd.DataFrame(events)


# --- Main ---
df, run_dir = load_latest_run()
r = df['reward_raw']
dates = df['date_t']

print(f"\nPeríodo: {dates.iloc[0]} → {dates.iloc[-1]}")
print(f"Semanas totales: {len(r)}")

# S&P 500 mismo período
start = pd.to_datetime(dates.iloc[0])
end   = pd.to_datetime(dates.iloc[-1])
sp = yf.download('^GSPC', start=start, end=end, interval='1wk', progress=False)
sp_r = sp['Close'].pct_change().dropna().squeeze()
sp_r = pd.Series(sp_r.values[:len(r)])
sp_dates = pd.Series(dates.values[:len(sp_r)])

# --- Métricas globales ---
print("\n=== Métricas Globales (10 años) ===")
print(f"{'':>20} | {'Ann.Ret':>8} | {'Sharpe':>6} | {'Sortino':>7} | {'MaxDD':>7}")
print('-'*58)
print(f"{'LinUCB+GAT':>20} | {ann_return(r):>8.1%} | {sharpe(r):>6.3f} | {sortino(r):>7.3f} | {max_dd(r):>7.1%}")
print(f"{'S&P 500':>20} | {ann_return(sp_r):>8.1%} | {sharpe(sp_r):>6.3f} | {sortino(sp_r):>7.3f} | {max_dd(sp_r):>7.1%}")

# --- Análisis por año ---
print("\n=== Rendimiento por Año ===")
print(f"{'Año':>6} | {'LinUCB Ann.Ret':>14} | {'S&P Ann.Ret':>11} | {'LinUCB Sharpe':>13}")
print('-'*55)
df['year'] = pd.to_datetime(df['date_t']).dt.year
for year, group in df.groupby('year'):
    ry = group['reward_raw']
    sp_year = sp_r.iloc[group.index - df.index[0]]
    if len(ry) > 4:
        print(f"{year:>6} | {ann_return(ry):>14.1%} | {ann_return(sp_year):>11.1%} | {sharpe(ry):>13.3f}")

# --- Análisis de recuperación ---
print("\n=== Análisis de Recuperación de Crisis ===")
events_linucb = recovery_analysis(r, pd.Series(dates), 'LinUCB+GAT', threshold=-0.10)
events_sp     = recovery_analysis(sp_r, pd.Series(dates), 'S&P 500', threshold=-0.10)

all_events = pd.concat([events_linucb, events_sp]).sort_values('peak_date')

if len(all_events) > 0:
    print(all_events.to_string(index=False))
else:
    print("No se encontraron drawdowns > 10%")

# --- Gráficas ---
Path('reports/figures').mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle('GAT-LINUCB — 10 Year Backtest (2015–2026)', fontsize=14)

# 1. Retorno acumulado
ax = axes[0]
ax.plot((1+r).cumprod().values, label='LinUCB+GAT', color='steelblue', linewidth=2)
ax.plot((1+sp_r).cumprod().values, label='S&P 500', color='gray', linestyle='--', linewidth=1.5)
# Marcar crisis
crisis_dates = ['2020-02-21', '2022-01-03']
for cd in crisis_dates:
    idx = (pd.to_datetime(dates) - pd.Timestamp(cd)).abs().argmin()
    ax.axvline(idx, color='red', linestyle=':', alpha=0.7)
ax.set_title('Cumulative Return — 10 Years')
ax.set_xlabel('Weeks')
ax.set_ylabel('Cumulative Return')
ax.legend()
ax.grid(alpha=0.3)

# 2. Drawdown
ax = axes[1]
cum_linucb = (1+r).cumprod()
dd_linucb = (cum_linucb - cum_linucb.cummax()) / cum_linucb.cummax()
cum_sp = (1+sp_r).cumprod()
dd_sp = (cum_sp - cum_sp.cummax()) / cum_sp.cummax()
ax.fill_between(range(len(dd_linucb)), dd_linucb.values, 0, alpha=0.4, color='steelblue', label='LinUCB+GAT')
ax.fill_between(range(len(dd_sp)), dd_sp.values, 0, alpha=0.4, color='gray', label='S&P 500')
for cd in crisis_dates:
    idx = (pd.to_datetime(dates) - pd.Timestamp(cd)).abs().argmin()
    ax.axvline(idx, color='red', linestyle=':', alpha=0.7)
ax.set_title('Drawdown — 10 Years')
ax.set_xlabel('Weeks')
ax.legend()
ax.grid(alpha=0.3)

# 3. Rolling Sharpe
ax = axes[2]
rs_linucb = r.rolling(12).apply(lambda x: np.sqrt(52)*x.mean()/x.std() if x.std()>0 else 0)
rs_sp = sp_r.rolling(12).apply(lambda x: np.sqrt(52)*x.mean()/x.std() if x.std()>0 else 0)
ax.plot(rs_linucb.values, color='steelblue', linewidth=1.5, label='LinUCB Rolling Sharpe')
ax.plot(rs_sp.values, color='gray', linestyle='--', linewidth=1, label='S&P 500 Rolling Sharpe')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
for cd in crisis_dates:
    idx = (pd.to_datetime(dates) - pd.Timestamp(cd)).abs().argmin()
    ax.axvline(idx, color='red', linestyle=':', alpha=0.7, label='Crisis' if cd == crisis_dates[0] else '')
ax.set_title('Rolling Sharpe Ratio (12-week window) — 10 Years')
ax.set_xlabel('Weeks')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/backtest_10y.png', dpi=150, bbox_inches='tight')
print("\nFigura guardada: reports/figures/backtest_10y.png")