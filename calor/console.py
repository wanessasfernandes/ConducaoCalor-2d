import numpy as np
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def label(beta):
    return {0.0: "Explícito",
            0.5: "Crank–Nicolson", 
            1.0: "Implícito"}[beta]


def print_result(r):
    if r.get('diverged', False):
        status = 'Divergência detectada'
    
    elif r['stable']:
        status = 'Estável'
    
    else:
        status = 'Fora do limite de estabilidade'

    print(f"\n{label(r['beta'])} | β = {r['beta']:g} | {status}")
    print('─' * 68)
    print(f"  Δx = Δy = {r['h']:g} m   |   Δt = {r['dt']:g} s   |   s = {r['s']:g}")
    
    iterations = (f"{r['mean_iterations']:.1f}" if r['beta'] > 0
                  else 'Não se aplica (sem TDMA)')
    
    print(f"  Iterações médias: {iterations}")
    print(f"  Tempo de execução: {r['elapsed']:.3f} s")
    print('\n  Erro RMS — todos os nós da malha')
    print(f"  {'Tempo (s)':>12}   {'RMS (K)':>16}")
    print(f"  {'─' * 12}   {'─' * 16}")
    
    for t, error in sorted(r['errors'].items()):
        value = f'{error:.6e}' if np.isfinite(error) else 'Indisponível'
        print(f"  {t:>12g}   {value:>16}")
