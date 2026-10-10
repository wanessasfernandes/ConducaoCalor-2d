import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from calor.graficos import plt, OUT, plot_group, plot_error_topics, plot_diagnostics
from calor.solver import ALPHA, TIMES, simulate
from calor.console import label, print_result

def main():
    OUT.mkdir(exist_ok=True)
    comparison = []
    print("\n=== Comparação por s (Δx=Δy=0.05 m) ===")
    by_s = {}
    
    for beta in (0.0, 0.5, 1.0):
        by_s[label(beta)] = {}
        
        for s in (0.125, 0.25, 0.5):
            r = simulate(beta, 0.05, s * 0.05**2 / ALPHA)
            
            if r is None:
                return
            
            by_s[label(beta)][s] = r; print_result(r); comparison.append(r)
        
        group = {f's={s:g}': r for s, r in by_s[label(beta)].items()}
        for fields, kind in ((True, 'campos'), (False, 'perfis')):
            plot_group(group, TIMES, f'{label(beta)} — {kind}: comparação por s',
                       f'{kind}_s_beta_{beta:g}.png', fields=fields)
    
    print("\n=== Influência de Δt (Δx=Δy=0.05 m; métodos implícitos) ===")
    by_dt = {}
    
    for beta in (0.5, 1.0):
        by_dt[label(beta)] = {}
        
        for dt in (100.0, 50.0, 25.0, 12.5):
            times = tuple(t for t in TIMES if np.isclose(t / dt, round(t / dt)))
            r = simulate(beta, 0.05, dt, snapshots=times)
            
            if r is None:
                return
            
            by_dt[label(beta)][dt] = r
            print_result(r)
        
        group = {f'Δt={dt:g} s': r for dt, r in by_dt[label(beta)].items()}
        plot_group(group, (2500., 5000.), f'{label(beta)} — influência de Δt',
                   f'perfis_dt_beta_{beta:g}.png')

    print("\n=== Refinamento da malha (Δt=5 s) ===")
    by_h = {}
    
    for beta in (0.0, 0.5, 1.0):
        by_h[label(beta)] = {}
        
        for h in (0.10, 0.05, 0.025):
            r = simulate(beta, h, 5.0)
            
            if r is None:
                return
            
            by_h[label(beta)][h] = r
            print_result(r)
        
        group = {f'Δx=Δy={h:g} m': r for h, r in by_h[label(beta)].items()}
        plot_group(group, TIMES, f'{label(beta)} — refinamento da malha',
                   f'perfis_malha_beta_{beta:g}.png')
    
    plot_error_topics([
        ('Comparação por s', 's', by_s),
        ('Passo de tempo', 'Δt (s)', by_dt),
        ('Refinamento da malha', 'Δx (m)', by_h),
    ])

    # Um gráfico direto da comparação entre beta para o caso de referência.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for beta in (0.0, 0.5, 1.0):
        r = by_h[label(beta)][0.05]
        ax.plot(TIMES, [r["errors"][t] for t in TIMES], "o-", label=label(beta))
    
    ax.set(xlabel="t (s)", ylabel="ERMS (K)", title="Comparação dos esquemas: Δx=Δy=0.05 m, Δt=5 s")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "comparacao_metodos.png", dpi=160); plt.close(fig)
    plot_diagnostics(comparison)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    for name, group in by_dt.items():
        values = sorted(group)
        axes[0].plot(values, [group[v]['mean_iterations'] for v in values], 'o-', label=name)
        axes[1].plot(values, [group[v]['elapsed'] for v in values], 'o-', label=name)
    
    axes[0].set(xlabel='Δt (s)', ylabel='Iterações médias', title='Custo iterativo')
    axes[1].set(xlabel='Δt (s)', ylabel='Tempo de execução (s)', title='Custo computacional')
    
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    
    fig.tight_layout()
    fig.savefig(OUT / 'custo_computacional.png', dpi=160)
    plt.close(fig)
    
    print(f"\nGráficos salvos em: {OUT.resolve()}")


if __name__ == '__main__':
    main()
