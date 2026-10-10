import time
import numpy as np
from .tdma import thomas

ALPHA = 1.0e-5
LX = LY = 1.0
T0 = 100.0
TFINAL = 5000.0
EPS = 1.0e-10
TIMES = (1250.0, 2500.0, 3750.0, 5000.0)


def exact_solution(x, y, t):
    return T0 * np.sin(np.pi * x / LX) * np.sin(np.pi * y / LY) * np.exp(
        -ALPHA * np.pi**2 * (1 / LX**2 + 1 / LY**2) * t
    )


def simulate(beta, h, dt, snapshots=TIMES, max_iter=100000):
    
    if beta not in (0, .5, 1) or h <= 0 or dt <= 0 or max_iter < 1:
        print("Erro: parâmetros inválidos. Simulação encerrada.")
        return None
    
    if not np.isclose(LX/h, round(LX/h)) or not np.isclose(LY/h, round(LY/h)):
        print("Erro: Δx deve dividir exatamente as dimensões da placa.")
        return None
    
    if any(not np.isclose(t/dt, round(t/dt), rtol=0, atol=1e-9) for t in snapshots):
        print("Erro: tempo de comparação fora da malha temporal.")
        return None
    
    nx = int(round(LX / h)) + 1
    ny = int(round(LY / h)) + 1
    x = np.linspace(0.0, LX, nx)
    y = np.linspace(0.0, LY, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    T = exact_solution(X, Y, 0.0)
    
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0.0
    
    sx = ALPHA * dt / h**2
    sy = sx
    stable = (sx + sy <= 0.5 + 1e-14) if beta == 0 else True
    nsteps = int(round(TFINAL / dt))
    
    if not np.isclose(nsteps * dt, TFINAL):
        print(f"Erro: Δt={dt:g} s não alcança exatamente o tempo final.")
        return None
    
    wanted = {int(round(t / dt)): t for t in snapshots}
    saved = {}
    iter_counts = []
    diverged = False
    history_times = [0.0]
    history_max = [float(np.max(np.abs(T)))]
    start = time.perf_counter()
    
    for step in range(1, nsteps + 1):
        
        if beta == 0.0:
            lap = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / h**2
            lap += (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / h**2
            new = T.copy()
            new[1:-1, 1:-1] = T[1:-1, 1:-1] + ALPHA * dt * lap
            T = new
            iterations = 0
        
        else:
            old = T
            current = old.copy()
            a_x = np.full(nx - 3, -beta * sx)
            b_x = np.full(nx - 2, 1 + 2 * beta * (sx + sy))
            c_x = np.full(nx - 3, -beta * sx)
            a_y = np.full(ny - 3, -beta * sy)
            b_y = np.full(ny - 2, 1 + 2 * beta * (sx + sy))
            c_y = np.full(ny - 3, -beta * sy)
            
            for iterations in range(1, max_iter + 1):
                intermediate = current.copy()
                explicit_lap = (
                    old[2:, 1:-1] - 2 * old[1:-1, 1:-1] + old[:-2, 1:-1]
                ) * sx
                explicit_lap += (
                    old[1:-1, 2:] - 2 * old[1:-1, 1:-1] + old[1:-1, :-2]
                ) * sy
                rhs = old[1:-1, 1:-1] + (1 - beta) * explicit_lap
                rhs += beta * sy * (current[1:-1, :-2] + current[1:-1, 2:])
                intermediate[1:-1, 1:-1] = thomas(a_x, b_x, c_x, rhs)
                updated = intermediate.copy()
                rhs = old[1:-1, 1:-1] + (1 - beta) * explicit_lap
                rhs += beta * sx * (intermediate[:-2, 1:-1] + intermediate[2:, 1:-1])
                updated[1:-1, 1:-1] = thomas(a_y, b_y, c_y, rhs.T).T
                updated[0, :] = updated[-1, :] = updated[:, 0] = updated[:, -1] = 0.0
                
                if np.max(np.abs(updated - current)) < EPS:
                    current = updated
                    break
                
                current = updated
            
            else:
                print(f"Erro: TDMA não convergiu no passo {step} após {max_iter} iterações.")
                return None
            
            T = current
            iter_counts.append(iterations)

        history_times.append(step * dt)
        history_max.append(float(np.max(np.abs(T))))
        
        if not np.all(np.isfinite(T)) or np.max(np.abs(T)) > 1.0e6:
            diverged = True
            
            if step in wanted:
                saved[wanted[step]] = T.copy()
            
            break
        
        if step in wanted:
            saved[wanted[step]] = T.copy()

    elapsed = time.perf_counter() - start
    errors = {}
    
    for t in snapshots:
        
        if t in saved:
            errors[t] = float(np.sqrt(np.mean((saved[t] - exact_solution(X, Y, t)) ** 2)))
        else:
            errors[t] = np.nan
    
    mean_iter = float(np.mean(iter_counts)) if iter_counts else 0.0
    
    return {
        "beta": beta, 
        "h": h, 
        "dt": dt, 
        "s": sx, 
        "x": x, 
        "y": y,
        "snapshots": saved, 
        "errors": errors, 
        "iterations": iter_counts,
        "mean_iterations": mean_iter, 
        "elapsed": elapsed,
        "history_times": history_times, 
        "history_max": history_max,
        "stable": stable, 
        "diverged": diverged,
    }
