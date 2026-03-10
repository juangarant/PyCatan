"""
Algoritmo genético para optimizar los 12 parámetros del agente prueba1v2.

Cada individuo es un vector de 12 floats que representan los parámetros del
constructor de prueba1v2.  El fitness se evalúa simulando partidas de Catan
contra permutaciones de agentes estándar (pool de 10 agentes) midiendo la
tasa de victoria + puntos medios.

Uso:
    python optimizar_prueba1v2_genetico.py

Parámetros del GA (ajustables al inicio del script):
    NGEN              – Número de generaciones
    POP_SIZE          – Tamaño de la población
    CXPB              – Probabilidad de cruce
    MUTPB             – Probabilidad de mutación
    PERMS_PER_EVAL    – Permutaciones de oponentes muestreadas por evaluación
    MAX_ROUNDS        – Rondas máximas por partida
    WORKERS_PERCENT   – Porcentaje de CPUs a utilizar
"""

import os
import sys
import time
import random
import csv
import itertools
import traceback
import concurrent.futures
from copy import deepcopy

import numpy as np
from deap import base, creator, tools, algorithms

# ---------------------------------------------------------------------------
#  Imports del proyecto PyCatan
# ---------------------------------------------------------------------------
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
#  Pool de agentes estándar para entrenar
# ---------------------------------------------------------------------------
BENCHMARK_AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
ALL_PERMUTATIONS = list(itertools.permutations(BENCHMARK_AGENTS, 3))

# ---------------------------------------------------------------------------
#  Parámetros del algoritmo genético
# ---------------------------------------------------------------------------
NGEN = 30                # Generaciones
POP_SIZE = 40            # Tamaño de la población
CXPB = 0.6              # Probabilidad de cruce
MUTPB = 0.3             # Probabilidad de mutación
TOURNSIZE = 3            # Tamaño del torneo de selección
PERMS_PER_EVAL = 20      # Permutaciones de oponentes muestreadas por evaluación
                         # (cada una genera 4 partidas, una por posición)
MAX_ROUNDS = 200         # Rondas máximas por partida
WORKERS_PERCENT = 0.90   # Porcentaje de CPUs

# ---------------------------------------------------------------------------
#  Definición del espacio de parámetros
#
#  Cada entrada: (nombre, min, max, tipo)
#    tipo: 'float', 'int'
# ---------------------------------------------------------------------------
PARAM_SPACE = [
    # nombre                min    max    tipo
    ("w_wood",              0.5,   3.0,   "float"),
    ("w_clay",              0.5,   3.0,   "float"),
    ("w_other",             0.3,   2.5,   "float"),
    ("harbor_bonus",        0.0,   1.0,   "float"),
    ("road_bias",           0.1,   1.0,   "float"),
    ("city_town_threshold", 1,     5,     "int"),
    ("min_prob_town",       0,     5,     "int"),
    ("min_prob_city",       0,     5,     "int"),
    ("trade_accept_ratio",  0.3,   1.5,   "float"),
    ("surplus_threshold",   2,     8,     "int"),
    ("thief_prob_threshold", 1,    5,     "int"),
    ("knight_eagerness",    0.0,   1.0,   "float"),
]

N_PARAMS = len(PARAM_SPACE)

# ---------------------------------------------------------------------------
#  Funciones auxiliares
# ---------------------------------------------------------------------------

def random_param(idx):
    """Genera un valor aleatorio para el parámetro `idx` dentro de su rango."""
    name, lo, hi, ptype = PARAM_SPACE[idx]
    if ptype == "int":
        return float(random.randint(int(lo), int(hi)))
    return random.uniform(lo, hi)


def clamp_param(value, idx):
    """Restringe el valor al rango permitido."""
    name, lo, hi, ptype = PARAM_SPACE[idx]
    v = max(lo, min(hi, value))
    if ptype == "int":
        v = float(round(v))
    return v


def individual_to_dict(individual):
    """Convierte un individuo (lista de floats) a dict de kwargs."""
    d = {}
    for i, (name, lo, hi, ptype) in enumerate(PARAM_SPACE):
        v = individual[i]
        if ptype == "int":
            d[name] = int(round(v))
        else:
            d[name] = v
    return d


def crear_clase_agente(params_dict):
    """
    Crea una subclase de prueba1v2 con los parámetros fijados.
    Necesario porque GameDirector espera una clase (no instancia).
    """
    frozen = dict(params_dict)  # copia

    class AgenteOptimizado(prueba1v2):
        def __init__(self, agent_id):
            super().__init__(agent_id, **frozen)

    AgenteOptimizado.__name__ = "prueba1v2_Optimizado"
    return AgenteOptimizado


# ---------------------------------------------------------------------------
#  Simulación de una partida
# ---------------------------------------------------------------------------

def simulate_one_match(opponents, position, agent_class):
    """
    Ejecuta una partida con el agente en `position` (0-3) contra 3 oponentes.
    `opponents` es una lista de 3 clases de agente.
    Devuelve (victoria, puntos, puesto).
    """
    try:
        match_agents = list(opponents)
        match_agents.insert(position, agent_class)

        gd = GameDirector(agents=match_agents, max_rounds=MAX_ROUNDS, store_trace=False)
        trace = gd.game_start(print_outcome=False)

        last_round = max(trace["game"].keys(),
                         key=lambda r: int(r.split("_")[-1]))
        last_turn = max(trace["game"][last_round].keys(),
                        key=lambda t: int(t.split("_")[-1].lstrip("P")))
        vp = trace["game"][last_round][last_turn]["end_turn"]["victory_points"]

        agent_id = f"J{position}"
        points = int(vp[agent_id])
        winner = max(vp, key=lambda p: int(vp[p]))
        victory = 1 if winner == agent_id else 0

        ordenados = sorted(vp.items(), key=lambda x: int(x[1]), reverse=True)
        rank = 4
        for idx, (jug, _) in enumerate(ordenados, start=1):
            if jug == agent_id:
                rank = idx
                break

        return (victory, points, rank)
    except Exception:
        return (0, 0, 4)


# ---------------------------------------------------------------------------
#  Función de fitness  (wrapper para multiprocessing)
# ---------------------------------------------------------------------------

def _eval_single_game(args):
    """Wrapper para concurrent.futures: evalúa una partida."""
    opponents, position, params_dict = args
    agent_class = crear_clase_agente(params_dict)
    return simulate_one_match(opponents, position, agent_class)


def evalAgent(individual):
    """
    Evalúa un individuo jugando contra combinaciones de agentes estándar.
    Se muestrean PERMS_PER_EVAL permutaciones de 3 oponentes del pool de
    BENCHMARK_AGENTS, y en cada una se juega en las 4 posiciones.
    Total partidas = PERMS_PER_EVAL × 4.

    Fitness = ratio_victorias × 100 + media_puntos
    """
    params = individual_to_dict(individual)

    # Muestrear permutaciones (sin repetir si hay suficientes)
    n_perms = min(PERMS_PER_EVAL, len(ALL_PERMUTATIONS))
    sampled_perms = random.sample(ALL_PERMUTATIONS, n_perms)

    tasks = []
    for perm in sampled_perms:
        for pos in range(4):
            tasks.append((list(perm), pos, params))

    total_wins = 0
    total_points = 0
    total_rank = 0
    n_games = len(tasks)

    # Ejecutar las partidas en paralelo
    n_workers = max(1, int((os.cpu_count() or 1) * WORKERS_PERCENT))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_eval_single_game, tasks))

    for victory, points, rank in results:
        total_wins += victory
        total_points += points
        total_rank += rank

    win_rate = total_wins / n_games
    avg_points = total_points / n_games
    avg_rank = total_rank / n_games

    # Fitness compuesto: enfatiza victorias, bonifica puntos
    fitness = win_rate * 100 + avg_points
    return (fitness,)


# ---------------------------------------------------------------------------
#  Operadores genéticos personalizados
# ---------------------------------------------------------------------------

def cxBlendParam(ind1, ind2, alpha=0.5):
    """
    Cruce blend (BLX-α) respetando los rangos de cada parámetro.
    """
    for i in range(N_PARAMS):
        lo_val = min(ind1[i], ind2[i])
        hi_val = max(ind1[i], ind2[i])
        rng = hi_val - lo_val
        new1 = random.uniform(lo_val - alpha * rng, hi_val + alpha * rng)
        new2 = random.uniform(lo_val - alpha * rng, hi_val + alpha * rng)
        ind1[i] = clamp_param(new1, i)
        ind2[i] = clamp_param(new2, i)
    return ind1, ind2


def mutGaussianParam(individual, mu=0, indpb=0.3):
    """
    Mutación gaussiana adaptada: sigma = 10% del rango del parámetro.
    """
    for i in range(N_PARAMS):
        if random.random() < indpb:
            name, lo, hi, ptype = PARAM_SPACE[i]
            sigma = (hi - lo) * 0.10
            individual[i] = clamp_param(individual[i] + random.gauss(mu, sigma), i)
    return (individual,)


# ---------------------------------------------------------------------------
#  Configuración DEAP
# ---------------------------------------------------------------------------

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generación de individuos
def init_individual():
    return [random_param(i) for i in range(N_PARAMS)]

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores
toolbox.register("evaluate", evalAgent)
toolbox.register("mate", cxBlendParam, alpha=0.5)
toolbox.register("mutate", mutGaussianParam, mu=0, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)


# ---------------------------------------------------------------------------
#  Semilla del individuo con parámetros por defecto
# ---------------------------------------------------------------------------

def make_default_individual():
    """
    Crea un individuo con los parámetros por defecto de prueba1v2 para que
    el GA no parta de cero.
    """
    defaults = [1.5, 1.4, 1.0, 0.15, 0.70, 2, 2, 3, 0.8, 5, 3, 0.85]
    ind = creator.Individual([float(v) for v in defaults])
    return ind


# ---------------------------------------------------------------------------
#  Bucle principal del GA
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(" Optimización genética de parámetros de prueba1v2")
    print("=" * 70)
    print(f"  Generaciones:       {NGEN}")
    print(f"  Población:          {POP_SIZE}")
    games_per_ind = min(PERMS_PER_EVAL, len(ALL_PERMUTATIONS)) * 4
    print(f"  Permutaciones/eval: {PERMS_PER_EVAL} (de {len(ALL_PERMUTATIONS)} posibles)")
    print(f"  Partidas/individuo: {games_per_ind} ({PERMS_PER_EVAL} perms × 4 pos)")
    print(f"  Max rondas/partida: {MAX_ROUNDS}")
    print(f"  Oponentes: {[c.__name__ for c in BENCHMARK_AGENTS]}")
    print(f"  CXPB={CXPB}  MUTPB={MUTPB}  Torneo={TOURNSIZE}")
    print(f"  Espacio de parámetros ({N_PARAMS}):")
    for name, lo, hi, ptype in PARAM_SPACE:
        print(f"    {name:25s}  [{lo}, {hi}]  ({ptype})")
    print("=" * 70)

    # Crear población + inyectar individuo con valores por defecto
    pop = toolbox.population(n=POP_SIZE - 1)
    pop.append(make_default_individual())

    hof = tools.HallOfFame(5, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # CSV de log por generación
    log_csv = "optimizacion_genetica_log.csv"
    with open(log_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gen", "avg", "std", "min", "max",
                         "best_fitness"] + [p[0] for p in PARAM_SPACE])

    total_start = time.time()

    for gen in range(NGEN):
        gen_start = time.time()

        # --- Evaluar individuos sin fitness ---
        invalids = [ind for ind in pop if not ind.fitness.valid]
        print(f"\nGen {gen + 1}/{NGEN}: evaluando {len(invalids)} individuos "
              f"({len(invalids) * games_per_ind} partidas vs agentes estándar) ...")

        for i, ind in enumerate(invalids):
            ind.fitness.values = toolbox.evaluate(ind)
            elapsed = time.time() - gen_start
            print(f"  [{i + 1}/{len(invalids)}] fitness={ind.fitness.values[0]:.2f}  "
                  f"({elapsed:.0f}s)", end="\r")

        print()  # nueva línea

        # --- Hall of Fame ---
        hof.update(pop)

        # --- Estadísticas ---
        record = stats.compile(pop)
        best = hof[0]
        best_params = individual_to_dict(best)

        print(f"  Avg={record['avg']:.2f}  Std={record['std']:.2f}  "
              f"Min={record['min']:.2f}  Max={record['max']:.2f}")
        print(f"  Mejor global (fitness={best.fitness.values[0]:.2f}):")
        for name, val in best_params.items():
            print(f"    {name:25s} = {val}")

        # Guardar en CSV
        with open(log_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [gen + 1, f"{record['avg']:.4f}", f"{record['std']:.4f}",
                   f"{record['min']:.4f}", f"{record['max']:.4f}",
                   f"{best.fitness.values[0]:.4f}"]
            row += [best_params[p[0]] for p in PARAM_SPACE]
            writer.writerow(row)

        gen_elapsed = time.time() - gen_start
        print(f"  Tiempo generación: {gen_elapsed:.1f}s")

        # --- Selección + cruce + mutación para siguiente generación ---
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Cruce
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutación
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Elitismo: mantener al mejor de la generación anterior
        offspring[0] = toolbox.clone(hof[0])

        pop[:] = offspring

    # --- Resultado final ---
    total_elapsed = time.time() - total_start
    horas, resto = divmod(total_elapsed, 3600)
    minutos, segundos = divmod(resto, 60)

    print("\n" + "=" * 70)
    print(" OPTIMIZACIÓN COMPLETADA")
    print(f" Tiempo total: {int(horas)}h {int(minutos)}m {int(segundos)}s")
    print("=" * 70)

    print("\n Top 5 individuos:")
    for rank, ind in enumerate(hof, 1):
        params = individual_to_dict(ind)
        print(f"\n  #{rank}  fitness = {ind.fitness.values[0]:.2f}")
        for name, val in params.items():
            print(f"      {name:25s} = {val}")

    # --- Guardar mejor individuo como archivo Python ---
    best_params = individual_to_dict(hof[0])
    output_file = "prueba1v2_optimizado_params.py"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write("Parámetros optimizados para prueba1v2\n")
        f.write(f"Generados por algoritmo genético ({NGEN} gen, {POP_SIZE} pop, "
                f"{PERMS_PER_EVAL} perms/eval vs agentes estándar)\n")
        f.write(f"Fitness alcanzado: {hof[0].fitness.values[0]:.4f}\n")
        f.write('"""\n\n')
        f.write("OPTIMIZED_PARAMS = {\n")
        for name, val in best_params.items():
            if isinstance(val, int):
                f.write(f'    "{name}": {val},\n')
            else:
                f.write(f'    "{name}": {val:.6f},\n')
        f.write("}\n\n\n")
        f.write("# Para usar estos parámetros con el benchmark:\n")
        f.write("# En benchmark_vs_random.py, cambiar agentes_a_evaluar a:\n")
        f.write('#   ("Agents.prueba1v2.prueba1v2", OPTIMIZED_PARAMS)\n')

    print(f"\nMejores parámetros guardados en: {output_file}")
    print(f"Log por generación guardado en: {log_csv}")

    return pop, hof


if __name__ == "__main__":
    population, hof = main()
