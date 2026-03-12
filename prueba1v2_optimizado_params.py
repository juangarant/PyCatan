"""
Parámetros optimizados para prueba1v2
Generados por algoritmo genético (30 gen, 20 pop, 5 perms/eval vs agentes estándar)
Fitness alcanzado: 110.3000
"""

OPTIMIZED_PARAMS = {
    "w_wood": 1.524308,
    "w_clay": 1.683878,
    "w_other": 0.726752,
    "harbor_bonus": 0.124196,
    "road_bias": 0.818645,
    "city_town_threshold": 1,
    "min_prob_town": 1,
    "min_prob_city": 2,
    "trade_accept_ratio": 1.187663,
    "surplus_threshold": 5,
    "thief_prob_threshold": 5,
    "knight_eagerness": 0.830529,
}


# Para usar estos parámetros con el benchmark:
# En benchmark_vs_random.py, cambiar agentes_a_evaluar a:
#   ("Agents.prueba1v2.prueba1v2", OPTIMIZED_PARAMS)
