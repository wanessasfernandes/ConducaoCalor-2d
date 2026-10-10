import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from .solver import exact_solution
from .console import label

OUT = Path(__file__).resolve().parent.parent / 'resultados'

def plot_group(group, times, topic, filename, fields=False):
    """Linhas representam os casos, colunas representam os tempos."""
    
    rows, columns = len(group), len(times)
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(4 * columns, 3.4 * rows), layout='constrained')
    
    for row, (parameter, r) in enumerate(group.items()):
        for column, t in enumerate(times):
            ax = axes[row, column]
            ax.set_title(f'{parameter}, t={t:g} s', fontsize=10)
            
            if t not in r['snapshots']:
                ax.text(.5, .5, 'Instante indisponível', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_axis_off()
                continue
            
            T, x, y = r['snapshots'][t], r['x'], r['y']
            
            if fields:
                im = ax.pcolormesh(x, y, T.T, shading='auto', cmap='inferno')
                fig.colorbar(im, ax=ax, label='T (K)', shrink=.85)
                ax.set(xlabel='x (m)', ylabel='y (m)', aspect='equal')
            
            else:
                i = int(np.argmin(abs(x - .5)))
                j = int(np.argmin(abs(y - .5)))
                
                ax.plot(x, T[:, j], 'o-', ms=2, label='Numérico: y=0.5')
                ax.plot(y, T[i, :], 's:', ms=2, label='Numérico: x=0.5')
                ax.plot(x, exact_solution(x, .5, t), 'k--', label='Analítico: y=0.5')
                ax.plot(y, exact_solution(.5, y, t), '--', color='gray', label='Analítico: x=0.5')
                ax.set(xlabel='Coordenada (m)', ylabel='T (K)')
                ax.grid(alpha=.25)
                ax.legend(fontsize=7)
    
    fig.suptitle(topic, fontsize=15)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)


def plot_error_topics(topics):
    fig, axes = plt.subplots(1, len(topics), squeeze=False,
                             figsize=(6 * len(topics), 4.5), layout='constrained')
    
    for ax, (title, xlabel, groups) in zip(axes.flat, topics):
        for name, group in groups.items():
            values = sorted(group)
            ax.loglog(values, [group[v]['errors'][5000.] for v in values], 'o-', label=name)
        ax.set(title=title, xlabel=xlabel, ylabel='RMS em 5000 s (K)')
        ax.grid(which='both', alpha=.25)
        ax.legend()
    
    fig.savefig(OUT / 'erros_por_topico.png', dpi=160)
    plt.close(fig)

def plot_field(r, t, title_prefix, filename):
    if t not in r["snapshots"]:
        return
    
    arr = r["snapshots"][t]
    x, y = r["x"], r["y"]
    vmax = max(1e-12, float(np.nanmax(np.abs(arr))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(x, y, arr.T, shading="auto", cmap="inferno", vmin=min(0, float(np.min(arr))), vmax=vmax)
    fig.colorbar(im, ax=ax, label="Temperatura (K)")
    ax.set(xlabel="x (m)", ylabel="y (m)", title=f"{title_prefix}, t={t:g} s")
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)


def plot_center_profiles(r, t, filename):
    if t not in r["snapshots"]:
        return
    
    i = int(np.argmin(np.abs(r["x"] - 0.5)))
    j = int(np.argmin(np.abs(r["y"] - 0.5)))
    x, y, T = r["x"], r["y"], r["snapshots"][t]
    
    exact_x = exact_solution(x, np.full_like(x, 0.5), t)
    exact_y = exact_solution(np.full_like(y, 0.5), y, t)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, T[:, j], "o-", ms=3, label="Numérico")
    axes[0].plot(x, exact_x, "k--", label="Analítico")
    axes[0].set(xlabel="x (m)", ylabel="T (K)", title="Perfil em y=0.5 m")
    axes[1].plot(y, T[i, :], "o-", ms=3, label="Numérico")
    axes[1].plot(y, exact_y, "k--", label="Analítico")
    axes[1].set(xlabel="y (m)", ylabel="T (K)", title="Perfil em x=0.5 m")
    
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    
    fig.suptitle(f"{label(r['beta'])}, Δx={r['h']:g}, Δt={r['dt']:g}, t={t:g} s")
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)


def plot_error_curves(results, key, filename, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for name, group in results.items():
        vals = [group[k]["errors"][5000.0] for k in key]
        ax.loglog(key, vals, "o-", label=name)
    
    axis_label = "Δt (s)" if "tempo" in title else ("s (adimensional)" if title == "Influência de s" else "Δx (m)")
    
    ax.set(xlabel=axis_label, ylabel="ERMS em t=5000 s", title=title)
    ax.grid(True, which="both", alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160); plt.close(fig)


def plot_diagnostics(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for r in results:
        if r["beta"] == 0:
            axes[0].semilogy(r["history_times"], r["history_max"], label=f"s={r['s']:.3g}")
        
        else:
            axes[1].plot(np.arange(1, len(r["iterations"])+1)*r["dt"], r["iterations"],
                         label=f"β={r['beta']} Δt={r['dt']:g}")
    
    axes[0].set(xlabel="t (s)", ylabel="max |T| (K)", title="Estabilidade do explícito")
    axes[1].set(xlabel="t (s)", ylabel="Iterações", title="Convergência TDMA")
    
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    
    fig.tight_layout()
    fig.savefig(OUT / "estabilidade_iteracoes.png", dpi=160)
    plt.close(fig)
