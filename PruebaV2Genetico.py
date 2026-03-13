"""
Algoritmo genético para optimizar los parámetros de prueba1v2.
Basado en el patrón del GA de N-Reinas.

Uso:
    python PruebaV2Genetico.py
"""

import random
import time
import csv
import itertools

import numpy as np
from deap import base, creator, tools, algorithms

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
from Agents.prueba1v2 import prueba1v2
from Managers.GameDirector import GameDirector

# ---------------------------------------------------------------------------
#  Parámetros del GA  (ajustables)
# ---------------------------------------------------------------------------
NGEN           = 30    # Generaciones
POP_SIZE       = 30    # Tamaño de la población
CXPB           = 0.7   # Probabilidad de cruce
MUTPB          = 0.3   # Probabilidad de mutación
PERMS_PER_EVAL = 5     # Permutaciones de oponentes por evaluación
                       # (× 4 posiciones = partidas por individuo)
MAX_ROUNDS     = 200   # Rondas máximas por partida

# ---------------------------------------------------------------------------
#  Espacio de parámetros de prueba1v2
# ---------------------------------------------------------------------------
PARAM_SPACE = [
    # nombre                 min    max    tipo
    ("w_wood",               0.5,   3.0,   "float"),
    ("w_clay",               0.5,   3.0,   "float"),
    ("w_other",              0.3,   2.5,   "float"),
    ("harbor_bonus",         0.0,   1.0,   "float"),
    ("road_bias",            0.1,   1.0,   "float"),
    ("city_town_threshold",  1,     5,     "int"),
    ("min_prob_town",        0,     5,     "int"),
    ("min_prob_city",        0,     5,     "int"),
    ("trade_accept_ratio",   0.3,   1.5,   "float"),
    ("surplus_threshold",    2,     8,     "int"),
    ("thief_prob_threshold", 1,     5,     "int"),
    ("knight_eagerness",     0.0,   1.0,   "float"),
]
N_PARAMS = len(PARAM_SPACE)

BENCHMARK_AGENTS   = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
ALL_PERMUTATIONS   = list(itertools.permutations(BENCHMARK_AGENTS, 3))

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
    d = {}
    for i, (name, lo, hi, ptype) in enumerate(PARAM_SPACE):
        d[name] = int(round(ind[i])) if ptype == "int" else ind[i]
    return d


def _make_agent_class(params):
    frozen = dict(params)

    class _Agente(prueba1v2):
        def __init__(self, agent_id):
            super().__init__(agent_id, **frozen)

    _Agente.__name__ = "prueba1v2_GA"
    return _Agente


# ---------------------------------------------------------------------------
#  Función de fitness  (secuencial, sin multiprocessing — igual que n-reinas)
# ---------------------------------------------------------------------------

def evalAgent(individual):
    """
    Juega PERMS_PER_EVAL permutaciones × 4 posiciones y devuelve el fitness.
    Fitness = tasa_victorias × 100 + media_puntos
    """
    params  = individual_to_dict(individual)
    cls     = _make_agent_class(params)
    n_perms = min(PERMS_PER_EVAL, len(ALL_PERMUTATIONS))
    perms   = random.sample(ALL_PERMUTATIONS, n_perms)

    wins = 0
    pts  = 0
    n    = 0

    for perm in perms:
        for pos in range(4):
            try:
                agents = list(perm)
                agents.insert(pos, cls)
                gd    = GameDirector(agents=agents, max_rounds=MAX_ROUNDS, store_trace=False)
                trace = gd.game_start(print_outcome=False)

                last_r = max(trace["game"], key=lambda r: int(r.split("_")[-1]))
                last_t = max(trace["game"][last_r], key=lambda t: int(t.split("_")[-1].lstrip("P")))
                vp     = trace["game"][last_r][last_t]["end_turn"]["victory_points"]

                aid    = f"J{pos}"
                pts   += int(vp[aid])
                wins  += 1 if max(vp, key=lambda p: int(vp[p])) == aid else 0
                n     += 1
            except BaseException as e:
                print(f"\n  [!] partida {pos} excepción {type(e).__name__}: {e}")
                n += 1  # cuenta la partida pero sin puntos

    if n == 0:
        return (0.0,)
    return (wins / n * 100 + pts / n,)


# ---------------------------------------------------------------------------
#  Operadores genéticos
# ---------------------------------------------------------------------------

def _cx_blend(ind1, ind2, alpha=0.5):
    for i in range(N_PARAMS):
        lo = min(ind1[i], ind2[i])
        hi = max(ind1[i], ind2[i])
        r  = hi - lo
        ind1[i] = _clamp(random.uniform(lo - alpha * r, hi + alpha * r), i)
        ind2[i] = _clamp(random.uniform(lo - alpha * r, hi + alpha * r), i)
    return ind1, ind2


def _mut_gaussian(ind, mu=0, indpb=0.3):
    for i in range(N_PARAMS):
        if random.random() < indpb:
            _, lo, hi, _ = PARAM_SPACE[i]
            ind[i] = _clamp(ind[i] + random.gauss(mu, (hi - lo) * 0.10), i)
    return (ind,)


# ---------------------------------------------------------------------------
#  Configuración DEAP  (a nivel de módulo, igual que n-reinas)
# ---------------------------------------------------------------------------

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [_random_param(i) for i in range(N_PARAMS)])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evalAgent)
toolbox.register("mate",       _cx_blend, alpha=0.5)
toolbox.register("mutate",     _mut_gaussian, mu=0, indpb=0.3)
toolbox.register("select",     tools.selTournament, tournsize=3)


# ---------------------------------------------------------------------------
#  GA principal  (estructura idéntica a n-reinas)
# ---------------------------------------------------------------------------

def main():
    games_per_ind = PERMS_PER_EVAL * 4
    print("=" * 65)
    print(" GA prueba1v2")
    print(f"  Generaciones={NGEN}  Población={POP_SIZE}  "
          f"Partidas/ind={games_per_ind}")
    print(f"  CXPB={CXPB}  MUTPB={MUTPB}  MAX_ROUNDS={MAX_ROUNDS}")
    print("=" * 65)

    # Población con semilla de valores por defecto
    pop = toolbox.population(n=POP_SIZE - 1)
    defaults = [1.5, 1.4, 1.0, 0.15, 0.70, 2, 2, 3, 0.8, 5, 3, 0.85]
    pop.append(creator.Individual([float(v) for v in defaults]))

    hof = tools.HallOfFame(5, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log_csv = "PruebaV2Genetico_log.csv"
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["gen", "avg", "std", "min", "max"] + [p[0] for p in PARAM_SPACE]
        )

    execution_time = time.time()

    for gen in range(NGEN):
        t0 = time.time()

        # Variación (igual que n-reinas)
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Solo evaluar individuos con fitness inválido (los que cambiaron)
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        total_partidas = len(invalids) * PERMS_PER_EVAL * 4
        print(f"\nGen {gen + 1}/{NGEN}: evaluando {len(invalids)} individuos "
              f"({total_partidas} partidas) ...")

        for i, ind in enumerate(invalids, 1):
            ind.fitness.values = toolbox.evaluate(ind)
            print(f"  [{i}/{len(invalids)}] fitness={ind.fitness.values[0]:.2f}", end="\r")

        print()

        # Selección + Hall of Fame
        hof.update(offspring)
        pop = toolbox.select(offspring, k=len(pop))

        # Estadísticas
        record = stats.compile(pop)
        elapsed = time.time() - t0
        print(f"Gen {gen + 1:3d}/{NGEN}:  "
              f"avg={record['avg'][0]:.2f}  "
              f"max={record['max'][0]:.2f}  "
              f"mejor_global={hof[0].fitness.values[0]:.2f}  "
              f"({elapsed:.0f}s)")

        # Log CSV
        best_p = individual_to_dict(hof[0])
        with open(log_csv, "a", newline="") as f:
            row = [gen + 1,
                   f"{record['avg'][0]:.4f}", f"{record['std'][0]:.4f}",
                   f"{record['min'][0]:.4f}", f"{record['max'][0]:.4f}"]
            row += [best_p[p[0]] for p in PARAM_SPACE]
            csv.writer(f).writerow(row)

    total = time.time() - execution_time
    print(f"\nTiempo total: {total:.1f}s")

    # Resultado final
    print("\n Top 5:")
    for rank, ind in enumerate(hof, 1):
        p = individual_to_dict(ind)
        print(f"  #{rank}  fitness={ind.fitness.values[0]:.2f}  {p}")

    # Guardar mejor individuo
    best_p = individual_to_dict(hof[0])
    with open("PruebaV2Genetico_params.py", "w", encoding="utf-8") as f:
        f.write(f'# Fitness: {hof[0].fitness.values[0]:.4f}\n')
        f.write("OPTIMIZED_PARAMS = {\n")
        for name, val in best_p.items():
            f.write(f'    "{name}": {val},\n')
        f.write("}\n")
    print("Parámetros guardados en PruebaV2Genetico_params.py")
    print(f"Log guardado en {log_csv}")

    return pop, hof


if __name__ == "__main__":
    population, hof = main()
