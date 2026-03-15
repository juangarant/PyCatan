import random
from copy import copy

from Classes.Constants import (
    MaterialConstants,
    BuildConstants,
    HarborConstants,
    TerrainConstants,
    DevelopmentCardConstants,
)
from Classes.Board import Board
from Classes.DevelopmentCards import DevelopmentCardsHand, DevelopmentCard
from Classes.Hand import Hand
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface


class prueba1v2(AgentInterface):
    """
    Agente parametrizable con estrategia de expansión agresiva.
    Prioriza carreteras + pueblos para expandirse territorialmente,
    y solo upgradea a ciudades cuando tiene suficiente base.

    Parámetros (todos con valores por defecto):
        w_wood              – peso de la madera en el scoring inicial      (default 1.5)
        w_clay              – peso de la arcilla en el scoring inicial     (default 1.4)
        w_other             – peso de otros recursos en el scoring inicial (default 1.0)
        harbor_bonus        – bonus por nodo con puerto                    (default 0.15)
        road_bias           – probabilidad de construir carretera cuando
                              no hay pueblo válido de buena calidad        (default 0.70)
        city_town_threshold – nº mínimo de pueblos antes de construir
                              ciudades                                    (default 2)
        min_prob_town       – pips mínimos de terreno para construir
                              un pueblo en ese nodo                        (default 2)
        min_prob_city       – pips mínimos de terreno para mejorar
                              un pueblo a ciudad                           (default 3)
        trade_accept_ratio  – ratio mínimo recibo/doy para aceptar un
                              intercambio pasivo                           (default 0.8)
        surplus_threshold   – excedente de un recurso para cambiar 4:1
                              con el banco                                 (default 5)
        thief_prob_threshold– pips mínimos del terreno donde colocar
                              al ladrón                                    (default 3)
        knight_eagerness    – probabilidad [0-1] de jugar un caballero
                              al inicio del turno cuando el ladrón está
                              en terreno propio                            (default 0.85)
    """

    # ------------------------------------------------------------------ #
    #  Tabla de conversión probabilidad → pips (valor posicional)
    # ------------------------------------------------------------------ #
    PROB_TO_PIPS = {
        0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
        7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
    }

    # ------------------------------------------------------------------ #
    #  Constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        agent_id,
        w_wood=1.5,
        w_clay=1.4,
        w_other=1.0,
        harbor_bonus=0.15,
        road_bias=0.70,
        city_town_threshold=2,
        min_prob_town=2,
        min_prob_city=3,
        trade_accept_ratio=0.8,
        surplus_threshold=5,
        thief_prob_threshold=3,
        knight_eagerness=0.85,
    ):
        super().__init__(agent_id)
        # ----- parámetros -----
        self.w_wood = w_wood
        self.w_clay = w_clay
        self.w_other = w_other
        self.harbor_bonus = harbor_bonus
        self.road_bias = road_bias
        self.city_town_threshold = city_town_threshold
        self.min_prob_town = min_prob_town
        self.min_prob_city = min_prob_city
        self.trade_accept_ratio = trade_accept_ratio
        self.surplus_threshold = surplus_threshold
        self.thief_prob_threshold = thief_prob_threshold
        self.knight_eagerness = knight_eagerness

        # ----- estado interno -----
        self.town_count = 0       # pueblos propios actuales (no ciudades)
        self.city_count = 0       # ciudades propias
        self.year_of_plenty_mat1 = MaterialConstants.WOOD
        self.year_of_plenty_mat2 = MaterialConstants.CLAY

    # ================================================================== #
    #  Utilidades internas
    # ================================================================== #

    def _pips(self, probability):
        """Convierte la probabilidad del terreno (2-12) a pips (1-5)."""
        return self.PROB_TO_PIPS.get(probability, 0)

    def _resource_weight(self, terrain_type):
        """Devuelve el peso del recurso según su tipo."""
        if terrain_type == TerrainConstants.WOOD:
            return self.w_wood
        if terrain_type == TerrainConstants.CLAY:
            return self.w_clay
        if terrain_type == TerrainConstants.DESERT:
            return 0.0
        return self.w_other

    def _score_node(self, board, node_id):
        """
        Puntúa un nodo según:
          - suma de (pips * peso_recurso) de cada terreno adyacente
          - bonus si el nodo tiene puerto
        """
        score = 0.0
        for terrain_id in board.nodes[node_id]['contacting_terrain']:
            t = board.terrain[terrain_id]
            pips = self._pips(t['probability'])
            weight = self._resource_weight(t['terrain_type'])
            score += pips * weight
        # Bonus puerto
        if board.nodes[node_id]['harbor'] != HarborConstants.NONE:
            score += self.harbor_bonus
        return score

    def _best_node(self, board, node_ids):
        """Elige el nodo con mayor puntuación de una lista."""
        best_id = None
        best_score = -1.0
        for nid in node_ids:
            s = self._score_node(board, nid)
            if s > best_score:
                best_score = s
                best_id = nid
        return best_id, best_score

    def _max_pips_at_node(self, board, node_id):
        """Devuelve los pips máximos entre los terrenos adyacentes a un nodo."""
        mx = 0
        for tid in board.nodes[node_id]['contacting_terrain']:
            p = self._pips(board.terrain[tid]['probability'])
            if p > mx:
                mx = p
        return mx

    def _current_goal(self):
        """
        Devuelve 'city', 'town' o 'road' según la situación.
        Se usa para el descarte y el comercio orientado.
        """
        if self.town_count >= self.city_town_threshold:
            return 'city'
        # Si no tenemos ni madera ni arcilla básicas, priorizar carretera
        # para poder expandirnos antes de construir pueblo
        if self.town_count == 0:
            return 'road'
        return 'town'

    def _goal_materials(self, goal):
        """Materials que se necesitan para el objetivo."""
        if goal == 'city':
            return Materials(2, 3, 0, 0, 0)
        if goal == 'town':
            return Materials(1, 0, 1, 1, 1)
        # road
        return Materials(0, 0, 1, 1, 0)

    def _can_build_after_trade(self, hand_resources, offer_gives, offer_receives):
        """
        Simula aceptar una oferta y comprueba si se puede construir
        algo nuevo que antes no se podía.
        offer_gives = lo que el otro jugador da (nosotros recibimos)
        offer_receives = lo que el otro jugador pide (nosotros damos)
        """
        # recursos actuales
        cur = hand_resources
        # comprobar que tenemos suficiente para dar
        mat_names = ['cereal', 'mineral', 'clay', 'wood', 'wool']
        for i, name in enumerate(mat_names):
            if getattr(offer_receives, name) > getattr(cur, name):
                return False

        new_res = Materials(
            cur.cereal + offer_gives.cereal - offer_receives.cereal,
            cur.mineral + offer_gives.mineral - offer_receives.mineral,
            cur.clay + offer_gives.clay - offer_receives.clay,
            cur.wood + offer_gives.wood - offer_receives.wood,
            cur.wool + offer_gives.wool - offer_receives.wool,
        )

        builds = [BuildConstants.CITY, BuildConstants.TOWN, BuildConstants.ROAD, BuildConstants.CARD]
        for b in builds:
            if new_res.has_more(b) and not cur.has_more(b):
                return True
        return False

    def _count_own_towns_and_cities(self, board):
        """Actualiza los contadores de pueblos y ciudades propias."""
        towns = 0
        cities = 0
        for node in board.nodes:
            if node['player'] == self.id:
                if node['has_city']:
                    cities += 1
                else:
                    towns += 1
        self.town_count = towns
        self.city_count = cities

    def _thief_on_own_terrain(self, board):
        """True si el ladrón está en un terreno adyacente a un nodo propio."""
        for terrain in board.terrain:
            if terrain['has_thief']:
                for nid in terrain['contacting_nodes']:
                    if board.nodes[nid]['player'] == self.id:
                        return True
        return False

    # ================================================================== #
    #  on_game_start — colocación inicial
    # ================================================================== #
    def on_game_start(self, board_instance):
        self.board = board_instance
        possibilities = self.board.valid_starting_nodes()
        if not possibilities:
            return super().on_game_start(board_instance)

        chosen_node, _ = self._best_node(self.board, possibilities)
        if chosen_node is None:
            chosen_node = random.choice(possibilities)

        # Elegir la mejor carretera: la que apunte al nodo adyacente con mayor scoring
        adjacent = self.board.nodes[chosen_node]['adjacent']
        best_road = None
        best_road_score = -1.0
        for adj in adjacent:
            s = self._score_node(self.board, adj)
            # Bonus extra si el adyacente tiene puerto
            if self.board.nodes[adj]['harbor'] != HarborConstants.NONE:
                s += self.harbor_bonus
            if s > best_road_score:
                best_road_score = s
                best_road = adj
        if best_road is None:
            best_road = random.choice(adjacent)

        self.town_count += 1
        return chosen_node, best_road

    # ================================================================== #
    #  on_turn_start — jugar caballero si el ladrón molesta
    # ================================================================== #
    def on_turn_start(self):
        if not self.development_cards_hand.hand:
            return None

        if self._thief_on_own_terrain(self.board):
            # Buscar un caballero
            for i, card in enumerate(self.development_cards_hand.hand):
                if card.type == DevelopmentCardConstants.KNIGHT:
                    if random.random() < self.knight_eagerness:
                        return self.development_cards_hand.select_card(i)
        return None

    # ================================================================== #
    #  on_turn_end — jugar puntos de victoria si es posible
    # ================================================================== #
    def on_turn_end(self):
        if not self.development_cards_hand.hand:
            return None
        for i, card in enumerate(self.development_cards_hand.hand):
            if card.type == DevelopmentCardConstants.VICTORY_POINT:
                return self.development_cards_hand.select_card(i)
        return None

    # ================================================================== #
    #  on_trade_offer — recibir ofertas de otros jugadores
    # ================================================================== #
    def on_trade_offer(self, board_instance, offer=None, player_id=None):
        # offer.gives = lo que el otro jugador DA (nosotros recibimos)
        # offer.receives = lo que el otro jugador PIDE (nosotros damos)

        # Argumento centinela: evita el bug del objeto mutable compartido
        if offer is None:
            return False

        total_receive = sum(offer.gives)
        total_give = sum(offer.receives)

        # Evitar divisiones por cero: si no da nada, rechazar
        if total_receive == 0:
            return False

        # 1) Simulación: aceptar si podemos construir algo nuevo
        if self._can_build_after_trade(self.hand.resources, offer.gives, offer.receives):
            return True

        # 2) Ratio mínimo
        if total_give == 0:
            return True  # nos regalan algo
        ratio = total_receive / total_give
        if ratio >= self.trade_accept_ratio:
            # aceptar solo si no nos quedamos sin recursos para el objetivo actual
            goal = self._current_goal()
            goal_mats = self._goal_materials(goal)
            mat_names = ['cereal', 'mineral', 'clay', 'wood', 'wool']
            after = Materials(
                self.hand.resources.cereal + offer.gives.cereal - offer.receives.cereal,
                self.hand.resources.mineral + offer.gives.mineral - offer.receives.mineral,
                self.hand.resources.clay + offer.gives.clay - offer.receives.clay,
                self.hand.resources.wood + offer.gives.wood - offer.receives.wood,
                self.hand.resources.wool + offer.gives.wool - offer.receives.wool,
            )
            # No acepto si alguno queda negativo
            for name in mat_names:
                if getattr(after, name) < 0:
                    return False
            return True

        return False

    # ================================================================== #
    #  on_commerce_phase — comercio activo
    # ================================================================== #
    def on_commerce_phase(self):
        # 1) Intentar jugar carta de desarrollo (monopolio) si tiene
        if self.development_cards_hand.hand:
            for i, card in enumerate(self.development_cards_hand.hand):
                if card.type == DevelopmentCardConstants.MONOPOLY_EFFECT:
                    return self.development_cards_hand.select_card(i)

        # 2) Comercio 4:1 (o puerto) si excedente supera umbral
        mat_ids = [
            MaterialConstants.CEREAL,
            MaterialConstants.MINERAL,
            MaterialConstants.CLAY,
            MaterialConstants.WOOD,
            MaterialConstants.WOOL,
        ]
        mat_values = [
            self.hand.resources.cereal,
            self.hand.resources.mineral,
            self.hand.resources.clay,
            self.hand.resources.wood,
            self.hand.resources.wool,
        ]

        goal = self._current_goal()
        goal_mats = self._goal_materials(goal)
        goal_list = [goal_mats.cereal, goal_mats.mineral, goal_mats.clay, goal_mats.wood, goal_mats.wool]

        for idx in range(5):
            if mat_values[idx] >= self.surplus_threshold:
                # Buscar un recurso que necesitemos para el objetivo actual
                for j in range(5):
                    if j != idx and goal_list[j] > 0 and mat_values[j] < goal_list[j]:
                        return {'gives': mat_ids[idx], 'receives': mat_ids[j]}
                # Si no falta nada específico, cambiar por algo útil genérico
                for j in range(5):
                    if j != idx:
                        return {'gives': mat_ids[idx], 'receives': mat_ids[j]}

        # 3) Comercio activo con jugadores: pedir lo que falta para el objetivo
        needed = []
        have_extra = []
        for idx in range(5):
            diff = goal_list[idx] - mat_values[idx]
            if diff > 0:
                needed.append((idx, diff))
            elif mat_values[idx] > goal_list[idx]:
                have_extra.append((idx, mat_values[idx] - goal_list[idx]))

        if needed and have_extra:
            gives_list = [0, 0, 0, 0, 0]
            receives_list = [0, 0, 0, 0, 0]
            total_to_give = sum(d for _, d in needed)
            given = 0
            for idx, extra in have_extra:
                amount = min(extra, total_to_give - given)
                gives_list[idx] = amount
                given += amount
                if given >= total_to_give:
                    break
            for idx, diff in needed:
                receives_list[idx] = diff

            gives = Materials(*gives_list)
            receives = Materials(*receives_list)
            if not gives.is_empty() and not receives.is_empty():
                return TradeOffer(gives, receives)

        return None

    # ================================================================== #
    #  on_build_phase — expansión agresiva
    # ================================================================== #
    def on_build_phase(self, board_instance):
        self.board = board_instance
        self._count_own_towns_and_cities(self.board)

        # --- Intentar jugar carta de desarrollo útil ---
        if self.development_cards_hand.hand:
            for i, card in enumerate(self.development_cards_hand.hand):
                if card.type == DevelopmentCardConstants.YEAR_OF_PLENTY_EFFECT:
                    return self.development_cards_hand.select_card(i)
                if card.type == DevelopmentCardConstants.ROAD_BUILDING_EFFECT:
                    road_poss = self.board.valid_road_nodes(self.id)
                    if len(road_poss) > 1:
                        return self.development_cards_hand.select_card(i)

        # --- 1) Ciudad (si ya hemos alcanzado el umbral de pueblos) ---
        if (self.town_count >= self.city_town_threshold and
                self.hand.resources.has_more(BuildConstants.CITY)):
            valid_cities = self.board.valid_city_nodes(self.id)
            if valid_cities:
                # Elegir la ciudad con mejores pips
                best_city = None
                best_pips = -1
                for nid in valid_cities:
                    p = self._max_pips_at_node(self.board, nid)
                    if p > best_pips:
                        best_pips = p
                        best_city = nid
                if best_city is not None and best_pips >= self.min_prob_city:
                    self.town_count -= 1
                    self.city_count += 1
                    return {'building': BuildConstants.CITY, 'node_id': best_city}

        # --- 2) Pueblo ---
        if self.hand.resources.has_more(BuildConstants.TOWN):
            valid_towns = self.board.valid_town_nodes(self.id)
            if valid_towns:
                best_town, best_score = self._best_node(self.board, valid_towns)
                best_pips = self._max_pips_at_node(self.board, best_town) if best_town is not None else 0
                if best_town is not None and best_pips >= self.min_prob_town:
                    self.town_count += 1
                    return {'building': BuildConstants.TOWN, 'node_id': best_town}

        # --- 3) Carretera (expansión) ---
        if self.hand.resources.has_more(BuildConstants.ROAD):
            valid_roads = self.board.valid_road_nodes(self.id)
            if valid_roads:
                # Priorizar carreteras que lleguen a nodos con puerto
                harbor_roads = [
                    r for r in valid_roads
                    if self.board.nodes[r['finishing_node']]['harbor'] != HarborConstants.NONE
                ]
                if harbor_roads:
                    chosen = harbor_roads[0]
                    return {
                        'building': BuildConstants.ROAD,
                        'node_id': chosen['starting_node'],
                        'road_to': chosen['finishing_node'],
                    }

                # Priorizar carreteras hacia nodos con alta puntuación
                scored = []
                for r in valid_roads:
                    s = self._score_node(self.board, r['finishing_node'])
                    scored.append((s, r))
                scored.sort(key=lambda x: x[0], reverse=True)

                # Según road_bias, construir carretera o no
                if random.random() < self.road_bias:
                    chosen = scored[0][1]
                    return {
                        'building': BuildConstants.ROAD,
                        'node_id': chosen['starting_node'],
                        'road_to': chosen['finishing_node'],
                    }

        # --- 4) Carta de desarrollo ---
        if self.hand.resources.has_more(BuildConstants.CARD):
            return {'building': BuildConstants.CARD}

        return None

    # ================================================================== #
    #  on_moving_thief — cascada de prioridades
    # ================================================================== #
    def on_moving_thief(self):
        thief_terrain_id = -1
        candidates = []  # (priority, terrain_id, enemy_player)

        for terrain in self.board.terrain:
            if terrain['has_thief']:
                thief_terrain_id = terrain['id']
                continue

            pips = self._pips(terrain['probability'])
            if pips < self.thief_prob_threshold:
                continue

            # Comprobar nodos adyacentes
            has_own = False
            enemy_count = 0
            enemy_player = -1
            for nid in terrain['contacting_nodes']:
                if self.board.nodes[nid]['player'] == self.id:
                    has_own = True
                    break
                if self.board.nodes[nid]['player'] != -1:
                    enemy_count += 1
                    enemy_player = self.board.nodes[nid]['player']

            if has_own:
                continue  # Nunca colocar en terreno propio

            if enemy_count > 0:
                # prioridad = pips * 10 + enemy_count (mayor = mejor)
                priority = pips * 10 + enemy_count
                candidates.append((priority, terrain['id'], enemy_player))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, tid, enemy = candidates[0]
            return {'terrain': tid, 'player': enemy}

        # Fallback: dejar donde está (el GameManager lo moverá aleatorio)
        return {'terrain': thief_terrain_id, 'player': -1}

    # ================================================================== #
    #  on_having_more_than_7_materials — descarte contextual
    # ================================================================== #
    def on_having_more_than_7_materials_when_thief_is_called(self):
        goal = self._current_goal()
        goal_mats = self._goal_materials(goal)
        goal_list = [goal_mats.cereal, goal_mats.mineral, goal_mats.clay, goal_mats.wood, goal_mats.wool]

        # Nombres de atributos en el mismo orden que los índices 0-4
        mat_names = ['cereal', 'mineral', 'clay', 'wood', 'wool']

        # Descartar recursos que NO son necesarios para el objetivo actual
        while self.hand.get_total() > 7:
            discarded = False
            # Orden: descartar primero los menos útiles para el objetivo
            resource_order = sorted(range(5), key=lambda i: goal_list[i])
            for res_id in resource_order:
                # Acceso por atributo nombrado (Materials no soporta índice)
                current = getattr(self.hand.resources, mat_names[res_id])
                needed = goal_list[res_id]
                if current > needed:
                    self.hand.remove_material(res_id, 1)
                    discarded = True
                    break
            if not discarded:
                # Si todos son necesarios, quitar cualquiera que tengamos
                for res_id in resource_order:
                    if getattr(self.hand.resources, mat_names[res_id]) > 0:
                        self.hand.remove_material(res_id, 1)
                        discarded = True
                        break
                if not discarded:
                    # Mano vacía pero get_total() > 7: salir para evitar
                    # bucle infinito (no debería ocurrir, pero es seguro)
                    break
        return self.hand

    # ================================================================== #
    #  on_monopoly_card_use — elegir el recurso más escaso para nosotros
    # ================================================================== #
    def on_monopoly_card_use(self):
        goal = self._current_goal()
        goal_mats = self._goal_materials(goal)
        goal_list = [goal_mats.cereal, goal_mats.mineral, goal_mats.clay, goal_mats.wood, goal_mats.wool]
        hand_list = [
            self.hand.resources.cereal,
            self.hand.resources.mineral,
            self.hand.resources.clay,
            self.hand.resources.wood,
            self.hand.resources.wool,
        ]
        # Elegir el recurso con mayor déficit respecto al objetivo
        deficits = [goal_list[i] - hand_list[i] for i in range(5)]
        best = max(range(5), key=lambda i: deficits[i])
        return best

    # ================================================================== #
    #  on_road_building_card_use
    # ================================================================== #
    def on_road_building_card_use(self):
        valid_nodes = self.board.valid_road_nodes(self.id)
        if len(valid_nodes) > 1:
            # Elegir las 2 mejores carreteras por scoring del nodo destino
            scored = [(self._score_node(self.board, r['finishing_node']), r) for r in valid_nodes]
            scored.sort(key=lambda x: x[0], reverse=True)
            r1 = scored[0][1]
            r2 = scored[1][1]
            return {
                'node_id': r1['starting_node'],
                'road_to': r1['finishing_node'],
                'node_id_2': r2['starting_node'],
                'road_to_2': r2['finishing_node'],
            }
        elif len(valid_nodes) == 1:
            return {
                'node_id': valid_nodes[0]['starting_node'],
                'road_to': valid_nodes[0]['finishing_node'],
                'node_id_2': None,
                'road_to_2': None,
            }
        return None

    # ================================================================== #
    #  on_year_of_plenty_card_use — pedir lo que más falta para el objetivo
    # ================================================================== #
    def on_year_of_plenty_card_use(self):
        goal = self._current_goal()
        goal_mats = self._goal_materials(goal)
        goal_list = [goal_mats.cereal, goal_mats.mineral, goal_mats.clay, goal_mats.wood, goal_mats.wool]
        hand_list = [
            self.hand.resources.cereal,
            self.hand.resources.mineral,
            self.hand.resources.clay,
            self.hand.resources.wood,
            self.hand.resources.wool,
        ]
        deficits = [(goal_list[i] - hand_list[i], i) for i in range(5)]
        deficits.sort(key=lambda x: x[0], reverse=True)
        mat1 = deficits[0][1] if deficits[0][0] > 0 else random.randint(0, 4)
        mat2 = deficits[1][1] if deficits[1][0] > 0 else mat1
        return {'material': mat1, 'material_2': mat2}