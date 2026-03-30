import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import glob
from pathlib import Path

# --- Cargar runs ---
def load_latest_run(prefix):
    runs = sorted(glob.glob(f'artifacts/linucb/{prefix}*/'))
    for r in reversed(runs):
        csvs = {
            'linucb': f'{r}logs_linucb.csv',
            'greedy': f'{r}logs_greedy.csv',
            'random': f'{r}logs_random.csv',
        }
        try:
            dfs = {k: pd.read_csv(v) for k, v in csvs.items()}
            if len(list(dfs.values())[0]) > 10:
                print(f"Usando: {r}")
                return dfs
        except:
            continue
    return None

logs_emb = load_latest_run('run_embeddings_')

WEEKS = 52

def sharpe(r):
    return np.sqrt(WEEKS) * r.mean() / r.std() if r.std() > 0 else 0

def sortino(r):
    down = r[r < 0].std()
    return np.sqrt(WEEKS) * r.mean() / down if down > 0 else 0

def max_drawdown(cum):
    return ((cum - cum.cummax()) / (cum.cummax() + 1)).min()

def annual_return(r):
    return (1 + r).prod() ** (WEEKS / len(r)) - 1

def volatility(r):
    return r.std() * np.sqrt(WEEKS)

def calmar(r):
    ar = annual_return(r)
    cum = (1 + r).cumprod()
    md = abs(max_drawdown(cum))
    return ar / md if md > 0 else 0

def compute_metrics(df, label):
    r = df['reward_raw']
    cum = (1 + r).cumprod()
    return {
        'Policy': label,
        'Ann. Return': f"{annual_return(r):.1%}",
        'Sharpe':      f"{sharpe(r):.3f}",
        'Sortino':     f"{sortino(r):.3f}",
        'Max Drawdown':f"{max_drawdown(cum):.1%}",
        'Volatility':  f"{volatility(r):.1%}",
        'Calmar':      f"{calmar(r):.3f}",
    }

# --- S&P 500 benchmark ---
def get_sp500(start, end):
    sp = yf.download('^GSPC', start=start, end=end, interval='1wk', progress=False)
    sp = sp['Close'].pct_change().dropna().squeeze()
    sp.index = sp.index.tz_localize(None)
    return sp

dates = pd.to_datetime(logs_emb['linucb']['date_t'])
sp500 = get_sp500(dates.min(), dates.max())
sp500_df = pd.DataFrame({'reward_raw': sp500.values[:len(logs_emb['linucb'])]})

# --- Tabla global ---
rows = []
rows.append(compute_metrics(logs_emb['linucb'], 'LinUCB + Sharpe Reward'))
rows.append(compute_metrics(logs_emb['greedy'], 'Greedy'))
rows.append(compute_metrics(logs_emb['random'], 'Random'))
rows.append(compute_metrics(sp500_df,           'S&P 500 (benchmark)'))

df_metrics = pd.DataFrame(rows)
print("\n=== Global Financial Metrics ===")
print(df_metrics.to_string(index=False))

# --- Análisis por tercios ---
df = logs_emb['linucb']
T = len(df)
t1, t2 = T//3, 2*T//3
thirds = [('Early (exploration)', df[:t1]), ('Mid (convergence)', df[t1:t2]), ('Late (exploitation)', df[t2:])]

print("\n=== LinUCB Phase Analysis ===")
print(f"{'Phase':>22} | {'Ann.Ret':>8} | {'Sharpe':>6} | {'MaxDD':>7} | {'Uniq.Assets':>11} | {'Repeat':>6}")
print('-'*75)
for label, subset in thirds:
    r = subset['reward_raw']
    assets = subset['asset'].nunique()
    repeat = (subset['asset'] == subset['asset'].shift()).mean()
    cum = (1+r).cumprod()
    print(f"{label:>22} | {annual_return(r):>8.1%} | {sharpe(r):>6.3f} | {max_drawdown(cum):>7.1%} | {assets:>11} | {repeat:>6.1%}")

# --- Late phase vs S&P 500 ---
final = df[t2:]
start_f = pd.to_datetime(final['date_t'].iloc[0])
end_f   = pd.to_datetime(final['date_reward'].iloc[-1])
sp_final = get_sp500(start_f, end_f)
sp_final_df = pd.DataFrame({'reward_raw': sp_final.values[:len(final)]})

print("\n=== Converged Phase vs S&P 500 ===")
print(f"{'Policy':>25} | {'Ann.Ret':>8} | {'Sharpe':>6} | {'MaxDD':>7} | {'Volatility':>10}")
print('-'*70)
for r_data, label in [(final['reward_raw'], 'LinUCB (converged)'), (sp_final_df['reward_raw'], 'S&P 500')]:
    cum = (1+r_data).cumprod()
    print(f"{label:>25} | {annual_return(r_data):>8.1%} | {sharpe(r_data):>6.3f} | {max_drawdown(cum):>7.1%} | {volatility(r_data):>10.1%}")

# --- Gráficas ---
Path('reports/figures').mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GAT-LinUCB — Financial Evaluation (2023–2026)', fontsize=14)

# 1. Cumulative return vs S&P 500
ax = axes[0, 0]
r_linucb = logs_emb['linucb']['reward_raw']
ax.plot((1+r_linucb).cumprod().values, label='LinUCB+Sharpe', color='steelblue')
ax.plot((1+sp500_df['reward_raw']).cumprod().values, label='S&P 500', color='gray', linestyle='--')
ax.axvline(t1, color='orange', linestyle=':', alpha=0.7, label='Phase boundaries')
ax.axvline(t2, color='orange', linestyle=':', alpha=0.7)
ax.set_title('Cumulative Return vs S&P 500')
ax.set_xlabel('Weeks')
ax.legend()
ax.grid(alpha=0.3)

# 2. Rolling Sharpe
ax = axes[0, 1]
rs = r_linucb.rolling(12).apply(lambda x: np.sqrt(52)*x.mean()/x.std() if x.std()>0 else 0)
ax.plot(rs.values, color='steelblue', label='LinUCB Rolling Sharpe (12w)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.axhline(sharpe(sp500_df['reward_raw']), color='gray', linestyle=':', label='S&P 500 Sharpe')
ax.axvline(t1, color='orange', linestyle=':', alpha=0.7)
ax.axvline(t2, color='orange', linestyle=':', alpha=0.7)
ax.set_title('Rolling Sharpe Ratio (12-week window)')
ax.set_xlabel('Weeks')
ax.legend()
ax.grid(alpha=0.3)

# 3. Drawdown
ax = axes[1, 0]
cum = (1+r_linucb).cumprod()
dd = (cum - cum.cummax()) / (cum.cummax()+1)
ax.fill_between(range(len(dd)), dd.values, 0, alpha=0.4, color='steelblue', label='LinUCB')
ax.axvline(t1, color='orange', linestyle=':', alpha=0.7)
ax.axvline(t2, color='orange', linestyle=':', alpha=0.7)
ax.set_title('Drawdown')
ax.set_xlabel('Weeks')
ax.legend()
ax.grid(alpha=0.3)

# 4. Phase comparison bar chart
ax = axes[1, 1]
phase_labels = ['Early\n(exploration)', 'Mid\n(convergence)', 'Late\n(exploitation)']
phase_returns = [annual_return(df[:t1]['reward_raw'])*100,
                 annual_return(df[t1:t2]['reward_raw'])*100,
                 annual_return(df[t2:]['reward_raw'])*100]
colors = ['#d9534f', '#f0ad4e', '#5cb85c']
bars = ax.bar(phase_labels, phase_returns, color=colors, alpha=0.8)
ax.axhline(annual_return(sp500_df['reward_raw'])*100, color='gray',
           linestyle='--', label=f"S&P 500 ({annual_return(sp500_df['reward_raw']):.1%})")
ax.set_title('Ann. Return by Phase')
ax.set_ylabel('Annual Return (%)')
ax.legend()
ax.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, phase_returns):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('reports/figures/financial_evaluation.png', dpi=150, bbox_inches='tight')
print("\nFigura guardada: reports/figures/financial_evaluation.png")

df_metrics.to_csv('reports/financial_metrics.csv', index=False)
print("Métricas guardadas: reports/financial_metrics.csv")

# --- Análisis ventanas recientes ---
print("\n=== Recent Window Analysis (Sharpe reward, converged) ===")
print(f"{'Window':>12} | {'Ann.Ret':>8} | {'Sharpe':>6} | {'Sortino':>7} | {'MaxDD':>7} | {'Assets':>6}")
print('-'*58)

df_linucb = logs_emb['linucb']
for label, n in [('Last 8w', 8), ('Last 12w', 12), ('Last 26w', 26), ('Last 52w', 52)]:
    subset = df_linucb.tail(n)
    r = subset['reward_raw']
    assets = subset['asset'].nunique()
    down = r[r<0].std()
    so = np.sqrt(WEEKS)*r.mean()/down if down>0 else 0
    print(f"{label:>12} | {annual_return(r):>8.1%} | {sharpe(r):>6.3f} | {so:>7.3f} | {max_drawdown((1+r).cumprod()):>7.1%} | {assets:>6}")
