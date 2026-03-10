import random
import time

import numpy as np
from deap import base, creator, tools, algorithms

# Parámetros del problema
N = 20  # Cambia este valor para diferentes tamaños del tablero

# Parámetros del algoritmo genético
CXPB, MUTPB, NGEN = 0.8, 0.2, 300

optimal_fitness = N * (N - 1) / 2
print(f"Fitness óptimo={optimal_fitness}")


# Definición de la función de fitness
# El fitness es mayor para soluciones con menos conflictos
def evalNQueens(individual):
    size = len(individual)
    # Calcula el número de pares de reinas en conflicto
    conflicts = 0
    for i in range(size):
        for j in range(i + 1, size):
            if individual[i] == individual[j]:
                conflicts += 1
            if individual[i] - individual[j] == i - j:
                conflicts += 1
            if individual[j] - individual[i] == i - j:
                conflicts += 1
    return optimal_fitness - conflicts,


# Configuración del algoritmo genético
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, 0, N - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalNQueens)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# Algoritmo genético
def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    execution_time = time.time()
    for gen in range(NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))

        # Hall of Fame actualizado
        hof.update(pop)

        # Imprimir las estadísticas para esta generación
        record = stats.compile(pop)
        print(
            f"Generación {gen + 1}: Promedio={record['avg']:.2f}, Desviación estándar={record['std']:.2f}, "
            f"Mínimo={record['min']:.2f}, Máximo={record['max']:.2f}")

        if hof[0].fitness.values[0] == optimal_fitness:
            print(f"\nSolución óptima encontrada en la generación {gen + 1}")
            break  # Sale del bucle si encuentra la solución óptima

    execution_time = time.time() - execution_time
    print(f"Tiempo de ejecución total={execution_time:.2f}s")
    return pop, stats, hof


if __name__ == "__main__":
    population, stats, hof = main()
    optimo = " óptimo"
    if hof[0].fitness.values[0] != optimal_fitness:
        optimo = " NO óptimo"
    print(f"Mejor individuo{optimo}:", hof[0], "Fitness:", hof[0].fitness.values[0])
