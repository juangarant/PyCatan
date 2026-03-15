"""
Algoritmo genético para optimizar los parámetros de prueba1v2optimizado.

VERSIÓN CON GRÁFICAS Y ESTADÍSTICAS TIPO N-REINAS
"""

import multiprocessing
import random
import time
import csv
import itertools
import sys

import numpy as np
from deap import base, creator, tools, algorithms

# ---------------------------------------------------------------------------
#  Parámetros del GA
# ---------------------------------------------------------------------------
NGEN           = 15
POP_SIZE       = 12
CXPB           = 0.7
MUTPB          = 0.25
PERMS_PER_EVAL = 3     # 3 permutaciones × 4 posiciones = 12 partidas/ind
MAX_ROUNDS     = 150
GAME_TIMEOUT   = 45

# ---------------------------------------------------------------------------
#  Espacio de parámetros
# ---------------------------------------------------------------------------
PARAM_SPACE = [
    ("w_wood",               0.5,   3.0,   "float"),
    ("w_clay",               0.5,   3.0,   "float"),
    ("w_other",              0.3,   2.5,   "float"),
    ("harbor_bonus",         0.0,   1.0,   "float"),
    ("road_bias",            0.0,   1.0,   "float"),
    ("city_town_threshold",  1,     4,     "int"),
    ("min_prob_town",        0,     4,     "int"),
    ("min_prob_city",        1,     4,     "int"),
    ("trade_accept_ratio",   0.5,   1.5,   "float"),
    ("surplus_threshold",    3,     7,     "int"),
    ("thief_prob_threshold", 1,     4,     "int"),
    ("knight_eagerness",     0.3,   1.0,   "float"),
]
N_PARAMS = len(PARAM_SPACE)

OPTIMIZED_DEFAULTS = [
    1.5, 1.5, 1.0, 0.15, 0.7,
    2, 1, 2, 0.9, 5, 3, 0.85,
]

# ---------------------------------------------------------------------------
#  Utilidades
# ---------------------------------------------------------------------------

def _random_param(idx):
    name, lo, hi, ptype = PARAM_SPACE[idx]
    if ptype == "int":
        return float(random.randint(int(lo), int(hi)))
    return random.uniform(lo, hi)

def _clamp(value, idx):
    name, lo, hi, ptype = PARAM_SPACE[idx]
    v = max(lo, min(hi, value))
    return float(round(v)) if ptype == "int" else v

def individual_to_dict(ind):
    return {
        name: (int(round(ind[i])) if ptype == "int" else ind[i])
        for i, (name, lo, hi, ptype) in enumerate(PARAM_SPACE)
    }

# ---------------------------------------------------------------------------
#  Worker de partida
# ---------------------------------------------------------------------------

def _game_worker(opponent_classes, ga_position, ga_params, max_rounds, queue):
    try:
        from Agents.StevenJuanAgent import StevenJuanAgent
        from Managers.GameDirector import GameDirector
        
        class GAAgent(StevenJuanAgent):
            def __init__(self, agent_id):
                super().__init__(agent_id, **ga_params)
        
        agent_classes = list(opponent_classes)
        agent_classes.insert(ga_position, GAAgent)
        
        t_start = time.time()
        gd = GameDirector(agents=agent_classes, max_rounds=max_rounds, store_trace=True)
        trace = gd.game_start(print_outcome=False)
        elapsed = time.time() - t_start
        
        game_data = trace["game"]
        rounds_played = len(game_data)
        
        last_r = max(game_data, key=lambda r: int(r.split("_")[-1]))
        last_t = max(
            game_data[last_r],
            key=lambda t: int("".join(filter(str.isdigit, t.split("_")[-1])) or "0"),
        )
        vp = game_data[last_r][last_t]["end_turn"]["victory_points"]
        aid = f"J{ga_position}"
        pts = int(vp[aid])
        win = max(vp, key=lambda p: int(vp[p])) == aid
        
        queue.put(("ok", pts, int(win), rounds_played, elapsed))
        
    except Exception as e:
        queue.put(("error", str(e)))


def run_single_game(opponent_classes, ga_position, ga_params):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_game_worker,
        args=(opponent_classes, ga_position, ga_params, MAX_ROUNDS, queue),
        daemon=True
    )
    
    t0 = time.time()
    p.start()
    p.join(timeout=GAME_TIMEOUT)
    elapsed = time.time() - t0
    
    if p.is_alive():
        p.kill()
        p.join()
        return None
    
    if not queue.empty():
        result = queue.get_nowait()
        if result[0] == "ok":
            return (result[1], result[2], result[3], result[4])
        else:
            return None
    
    return None


# ---------------------------------------------------------------------------
#  Función de fitness
# ---------------------------------------------------------------------------

def evalAgent(individual):
    from Agents.RandomAgent import RandomAgent as ra
    from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
    from Agents.AlexPastorAgent import AlexPastorAgent as apa
    from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
    from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
    from Agents.CrabisaAgent import CrabisaAgent as ca
    from Agents.EdoAgent import EdoAgent as ea
    from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
    from Agents.SigmaAgent import SigmaAgent as sa
    from Agents.TristanAgent import TristanAgent as ta
    
    BENCHMARK_AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
    ALL_PERMUTATIONS = list(itertools.permutations(BENCHMARK_AGENTS, 3))
    
    params = individual_to_dict(individual)
    perms = random.sample(ALL_PERMUTATIONS, min(PERMS_PER_EVAL, len(ALL_PERMUTATIONS)))
    
    wins = 0
    pts = 0
    n_valid = 0
    n_timeout = 0
    total_rounds = 0
    total_games = len(perms) * 4
    
    for perm in perms:
        for pos in range(4):
            result = run_single_game(perm, pos, params)
            
            if result is not None:
                p_pts, win, rounds, elapsed = result
                pts += p_pts
                wins += win
                n_valid += 1
                total_rounds += rounds
                
                status = "W" if win else "."
                print(status, end="", flush=True)
            else:
                n_timeout += 1
                print("T", end="", flush=True)
    
    print(f" [{n_valid}/{total_games}]", end="", flush=True)
    
    if n_valid < 2:
        return (0.0,)
    
    winrate = wins / n_valid
    avg_pts = pts / n_valid
    avg_rounds = total_rounds / n_valid
    
    fitness = 10 * winrate + avg_pts
    
    if avg_rounds > 100:
        penalty = (avg_rounds - 100) * 0.02
        fitness -= penalty
    
    fitness = max(0.0, fitness)
    
    return (fitness,)


# ---------------------------------------------------------------------------
#  Operadores genéticos
# ---------------------------------------------------------------------------

def _cx_blend(ind1, ind2, alpha=0.5):
    for i in range(N_PARAMS):
        lo = min(ind1[i], ind2[i])
        hi = max(ind1[i], ind2[i])
        r = hi - lo
        ind1[i] = _clamp(random.uniform(lo - alpha * r, hi + alpha * r), i)
        ind2[i] = _clamp(random.uniform(lo - alpha * r, hi + alpha * r), i)
    return ind1, ind2

def _mut_gaussian(ind, mu=0, indpb=0.25):
    for i in range(N_PARAMS):
        if random.random() < indpb:
            _, lo, hi, _ = PARAM_SPACE[i]
            ind[i] = _clamp(ind[i] + random.gauss(mu, (hi - lo) * 0.15), i)
    return (ind,)


# ---------------------------------------------------------------------------
#  Configuración DEAP
# ---------------------------------------------------------------------------

if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [_random_param(i) for i in range(N_PARAMS)])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalAgent)
toolbox.register("mate", _cx_blend, alpha=0.5)
toolbox.register("mutate", _mut_gaussian, mu=0, indpb=0.25)
toolbox.register("select", tools.selTournament, tournsize=3)


# ---------------------------------------------------------------------------
#  Función para generar gráfica de convergencia
# ---------------------------------------------------------------------------

def plot_convergence(logbook, filename="GA_convergence.png"):
    """
    Genera gráfica de convergencia del GA.
    """
    try:
        import matplotlib.pyplot as plt
        
        gen = logbook.select("gen")
        avg = logbook.select("avg")
        max_ = logbook.select("max")
        min_ = logbook.select("min")
        std = logbook.select("std")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Línea de promedio con desviación estándar
        avg_arr = np.array(avg)
        std_arr = np.array(std)
        
        ax.fill_between(gen, avg_arr - std_arr, avg_arr + std_arr, 
                        alpha=0.2, color='blue', label='±1 Desv. Std.')
        ax.plot(gen, avg, 'b-', linewidth=2, label='Promedio')
        ax.plot(gen, max_, 'g--', linewidth=2, label='Máximo')
        ax.plot(gen, min_, 'r:', linewidth=1.5, label='Mínimo')
        
        # Marcar punto de convergencia (donde max se estabiliza)
        max_arr = np.array(max_)
        best_fitness = max(max_)
        convergence_gen = None
        
        for i, m in enumerate(max_arr):
            if m >= best_fitness * 0.98:  # 98% del mejor
                convergence_gen = gen[i]
                break
        
        if convergence_gen:
            ax.axvline(x=convergence_gen, color='purple', linestyle='--', 
                       alpha=0.7, label=f'Convergencia (Gen {convergence_gen})')
            ax.scatter([convergence_gen], [max_arr[convergence_gen-1]], 
                       color='purple', s=100, zorder=5)
        
        ax.set_xlabel('Generación', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Evolución del Algoritmo Genético - Agente Catan', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(gen))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Gráfica guardada en: {filename}")
        return True
        
    except ImportError:
        print("\n⚠️  matplotlib no instalado. Ejecuta: pip install matplotlib")
        return False


# ---------------------------------------------------------------------------
#  GA principal
# ---------------------------------------------------------------------------

def main():
    games_per_ind = PERMS_PER_EVAL * 4
    
    print("=" * 70)
    print(" ALGORITMO GENÉTICO - Optimización Agente Catan")
    print("=" * 70)
    print(f"  Generaciones: {NGEN}")
    print(f"  Población: {POP_SIZE}")
    print(f"  Partidas/individuo: {games_per_ind}")
    print(f"  Max rounds: {MAX_ROUNDS}")
    print(f"  Timeout: {GAME_TIMEOUT}s")
    print("=" * 70, flush=True)

    # Población
    pop = toolbox.population(n=POP_SIZE - 1)
    pop.append(creator.Individual([float(v) for v in OPTIMIZED_DEFAULTS]))

    hof = tools.HallOfFame(5, similar=np.array_equal)

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Logbook para guardar historial (como N-reinas)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]

    log_csv = "GA_log.csv"
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["gen", "nevals", "avg", "std", "min", "max"] + [p[0] for p in PARAM_SPACE]
        )

    t_total = time.time()
    best_global = 0.0

    for gen in range(NGEN):
        t0 = time.time()

        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        nevals = len(invalids)

        print(f"\n{'='*70}")
        print(f"GENERACIÓN {gen+1}/{NGEN}")
        print("=" * 70)

        for i, ind in enumerate(invalids, 1):
            t_ind = time.time()
            print(f"  Ind {i:2d}/{nevals}: ", end="", flush=True)
            
            ind.fitness.values = toolbox.evaluate(ind)
            
            print(f" fit={ind.fitness.values[0]:.2f} ({time.time()-t_ind:.0f}s)")

        # Compilar estadísticas
        record = stats.compile(offspring)
        hof.update(offspring)
        
        if hof[0].fitness.values[0] > best_global:
            best_global = hof[0].fitness.values[0]

        # Guardar en logbook (como N-reinas)
        logbook.record(gen=gen+1, nevals=nevals, **record)

        # Selección con elitismo
        elite = tools.selBest(offspring, 2)
        pop = toolbox.select(offspring, k=POP_SIZE - 3)
        pop += [toolbox.clone(e) for e in elite[:1]]
        
        # Diversidad
        nuevos = toolbox.population(n=2)
        pop += nuevos

        elapsed = time.time() - t0

        # Imprimir tabla de estadísticas (estilo N-reinas)
        print("\n" + "-" * 70)
        print(f"{'gen':>5} {'nevals':>8} {'avg':>10} {'std':>10} {'min':>10} {'max':>10}")
        print("-" * 70)
        print(f"{gen+1:>5} {nevals:>8} {record['avg']:>10.2f} {record['std']:>10.2f} "
              f"{record['min']:>10.2f} {record['max']:>10.2f}")
        print("-" * 70)
        print(f">>> Mejor global: {best_global:.2f} | Tiempo gen: {elapsed:.0f}s")

        # CSV
        best_gen = tools.selBest(offspring, 1)[0]
        best_p = individual_to_dict(best_gen)
        with open(log_csv, "a", newline="") as f:
            row = [gen + 1, nevals,
                   f"{record['avg']:.4f}", f"{record['std']:.4f}",
                   f"{record['min']:.4f}", f"{record['max']:.4f}"]
            row += [best_p[p[0]] for p in PARAM_SPACE]
            csv.writer(f).writerow(row)

    elapsed_total = time.time() - t_total

    # Tabla final completa (estilo N-reinas)
    print("\n" + "=" * 70)
    print(" RESUMEN DE EVOLUCIÓN")
    print("=" * 70)
    print(f"{'gen':>5} {'nevals':>8} {'avg':>10} {'std':>10} {'min':>10} {'max':>10}")
    print("-" * 70)
    for record in logbook:
        print(f"{record['gen']:>5} {record['nevals']:>8} {record['avg']:>10.2f} "
              f"{record['std']:>10.2f} {record['min']:>10.2f} {record['max']:>10.2f}")
    print("=" * 70)
    
    print(f"\nTIEMPO TOTAL: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

    # Generar gráfica
    plot_convergence(logbook)

    # Top 5
    print("\n" + "=" * 70)
    print(" TOP 5 MEJORES INDIVIDUOS")
    print("=" * 70)
    for rank, ind in enumerate(hof, 1):
        p = individual_to_dict(ind)
        print(f"\n  #{rank} Fitness: {ind.fitness.values[0]:.3f}")
        print(f"     Parámetros:")
        for k, v in p.items():
            if isinstance(v, float):
                print(f"       {k}: {v:.4f}")
            else:
                print(f"       {k}: {v}")

    # Guardar mejores parámetros
    best_p = individual_to_dict(hof[0])
    with open("GA_best_params.py", "w", encoding="utf-8") as f:
        f.write(f'# Mejor fitness: {hof[0].fitness.values[0]:.4f}\n')
        f.write(f'# Generaciones: {NGEN}, Población: {POP_SIZE}\n')
        f.write(f'# Tiempo total: {elapsed_total:.0f}s\n\n')
        f.write("OPTIMIZED_PARAMS = {\n")
        for name, val in best_p.items():
            if isinstance(val, float):
                f.write(f'    "{name}": {val:.4f},\n')
            else:
                f.write(f'    "{name}": {val},\n')
        f.write("}\n")
    print("\n✅ Parámetros guardados en: GA_best_params.py")

    return pop, hof, logbook


if __name__ == "__main__":
    multiprocessing.freeze_support()
    population, hof, logbook = main()