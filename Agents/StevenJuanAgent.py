"""
prueba1v2optimizado - VERSIÓN CORREGIDA

Fixes aplicados:
  1. roads_built_this_turn ahora es atributo de instancia (no se resetea por llamada)
  2. build_actions contador para evitar loops infinitos
  3. Eliminados contadores manuales town_count/city_count - usa _count_own_towns_and_cities()
  4. trade_accept_ratio default más razonable (0.9)
  5. max_iters en descarte aumentado a 50
  6. on_moving_thief evita mover al mismo terreno
  7. on_road_building_card_use evita elegir la misma carretera dos veces
  8. Reset de contadores en on_turn_start
"""

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


class StevenJuanAgent(AgentInterface):
    """
    Agente parametrizable con estrategia de expansión agresiva.
    VERSIÓN CORREGIDA con límites de acciones y contadores arreglados.
    """

    PROB_TO_PIPS = {
        0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
        7: 0, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
    }

    def __init__(
        self,
        agent_id,
        w_wood=1.5,
        w_clay=1.5,
        w_other=1.0,
        harbor_bonus=0.15,
        road_bias=0.7,
        city_town_threshold=2,
        min_prob_town=1,
        min_prob_city=2,
        trade_accept_ratio=0.9,      # FIX: era 1.18, ahora más razonable
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
        # FIX: Contadores por turno como atributos de instancia
        self.roads_built_this_turn = 0
        self.build_actions_this_turn = 0
        self.MAX_BUILD_ACTIONS = 8  # FIX: límite para evitar loops
        
        self.year_of_plenty_mat1 = MaterialConstants.WOOD
        self.year_of_plenty_mat2 = MaterialConstants.CLAY

    # ================================================================== #
    #  Utilidades internas
    # ================================================================== #

    def _pips(self, probability):
        return self.PROB_TO_PIPS.get(probability, 0)

    def _resource_weight(self, terrain_type):
        if terrain_type == TerrainConstants.WOOD:
            return self.w_wood
        if terrain_type == TerrainConstants.CLAY:
            return self.w_clay
        if terrain_type == TerrainConstants.DESERT:
            return 0.0
        return self.w_other

    def _score_node(self, board, node_id):
        score = 0.0
        for terrain_id in board.nodes[node_id]['contacting_terrain']:
            t = board.terrain[terrain_id]
            pips = self._pips(t['probability'])
            weight = self._resource_weight(t['terrain_type'])
            score += pips * weight
        if board.nodes[node_id]['harbor'] != HarborConstants.NONE:
            score += self.harbor_bonus
        return score

    def _best_node(self, board, node_ids):
        best_id = None
        best_score = -1.0
        for nid in node_ids:
            s = self._score_node(board, nid)
            if s > best_score:
                best_score = s
                best_id = nid
        return best_id, best_score

    def _max_pips_at_node(self, board, node_id):
        mx = 0
        for tid in board.nodes[node_id]['contacting_terrain']:
            p = self._pips(board.terrain[tid]['probability'])
            if p > mx:
                mx = p
        return mx

    def _count_own_towns_and_cities(self, board):
        """Cuenta pueblos y ciudades desde el board (fuente de verdad)."""
        towns = 0
        cities = 0
        for node in board.nodes:
            if node['player'] == self.id:
                if node['has_city']:
                    cities += 1
                else:
                    towns += 1
        return towns, cities

    def _current_goal(self, town_count):
        """Devuelve 'city', 'town' o 'road' según la situación."""
        if town_count >= self.city_town_threshold:
            return 'city'
        return 'town'

    def _goal_materials(self, goal):
        if goal == 'city':
            return Materials(2, 3, 0, 0, 0)
        if goal == 'town':
            return Materials(1, 0, 1, 1, 1)
        return Materials(0, 0, 1, 1, 0)

    def _can_build_after_trade(self, hand_resources, offer_gives, offer_receives):
        cur = hand_resources
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

    def _thief_on_own_terrain(self, board):
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

        adjacent = self.board.nodes[chosen_node]['adjacent']
        best_road = None
        best_road_score = -1.0
        for adj in adjacent:
            s = self._score_node(self.board, adj)
            if self.board.nodes[adj]['harbor'] != HarborConstants.NONE:
                s += self.harbor_bonus
            if s > best_road_score:
                best_road_score = s
                best_road = adj
        if best_road is None:
            best_road = random.choice(adjacent)

        return chosen_node, best_road

    # ================================================================== #
    #  on_turn_start — FIX: resetear contadores de turno
    # ================================================================== #
    def on_turn_start(self):
        # FIX: Resetear contadores al inicio del turno
        self.roads_built_this_turn = 0
        self.build_actions_this_turn = 0
        
        if not self.development_cards_hand.hand:
            return None

        if self._thief_on_own_terrain(self.board):
            for i, card in enumerate(self.development_cards_hand.hand):
                if card.type == DevelopmentCardConstants.KNIGHT:
                    if random.random() < self.knight_eagerness:
                        return self.development_cards_hand.select_card(i)
        return None

    # ================================================================== #
    #  on_turn_end
    # ================================================================== #
    def on_turn_end(self):
        if not self.development_cards_hand.hand:
            return None
        for i, card in enumerate(self.development_cards_hand.hand):
            if card.type == DevelopmentCardConstants.VICTORY_POINT:
                return self.development_cards_hand.select_card(i)
        return None

    # ================================================================== #
    #  on_trade_offer
    # ================================================================== #
    def on_trade_offer(self, board_instance, offer=None, player_id=None):
        if offer is None:
            return False

        mat_names = ['cereal', 'mineral', 'clay', 'wood', 'wool']
        total_receive = sum(getattr(offer.gives, n) for n in mat_names)
        total_give = sum(getattr(offer.receives, n) for n in mat_names)

        if total_receive == 0:
            return False

        if self._can_build_after_trade(self.hand.resources, offer.gives, offer.receives):
            return True

        if total_give == 0:
            return True
        
        ratio = total_receive / total_give
        if ratio >= self.trade_accept_ratio:
            town_count, _ = self._count_own_towns_and_cities(board_instance)
            goal = self._current_goal(town_count)
            goal_mats = self._goal_materials(goal)
            after = Materials(
                self.hand.resources.cereal + offer.gives.cereal - offer.receives.cereal,
                self.hand.resources.mineral + offer.gives.mineral - offer.receives.mineral,
                self.hand.resources.clay + offer.gives.clay - offer.receives.clay,
                self.hand.resources.wood + offer.gives.wood - offer.receives.wood,
                self.hand.resources.wool + offer.gives.wool - offer.receives.wool,
            )
            for name in mat_names:
                if getattr(after, name) < 0:
                    return False
            return True

        return False

    # ================================================================== #
    #  on_commerce_phase
    # ================================================================== #
    def on_commerce_phase(self):
        if self.development_cards_hand.hand:
            for i, card in enumerate(self.development_cards_hand.hand):
                if card.type == DevelopmentCardConstants.MONOPOLY_EFFECT:
                    return self.development_cards_hand.select_card(i)

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

        town_count, _ = self._count_own_towns_and_cities(self.board)
        goal = self._current_goal(town_count)
        goal_mats = self._goal_materials(goal)
        goal_list = [goal_mats.cereal, goal_mats.mineral, goal_mats.clay, goal_mats.wood, goal_mats.wool]

        for idx in range(5):
            if mat_values[idx] >= self.surplus_threshold:
                for j in range(5):
                    if j != idx and goal_list[j] > 0 and mat_values[j] < goal_list[j]:
                        return {'gives': mat_ids[idx], 'receives': mat_ids[j]}
                for j in range(5):
                    if j != idx:
                        return {'gives': mat_ids[idx], 'receives': mat_ids[j]}

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
    #  on_build_phase — FIX: límite de acciones y contadores correctos
    # ================================================================== #
    def on_build_phase(self, board_instance):
        self.board = board_instance
        
        # FIX: Límite de acciones para evitar loops infinitos
        self.build_actions_this_turn += 1
        if self.build_actions_this_turn > self.MAX_BUILD_ACTIONS:
            return None
        
        # FIX: Usar _count_own_towns_and_cities() en lugar de contadores manuales
        town_count, city_count = self._count_own_towns_and_cities(self.board)

        # --- Intentar jugar carta de desarrollo útil ---
        if self.development_cards_hand.hand:
            for i, card in enumerate(self.development_cards_hand.hand):
                if card.type == DevelopmentCardConstants.YEAR_OF_PLENTY_EFFECT:
                    return self.development_cards_hand.select_card(i)
                if card.type == DevelopmentCardConstants.ROAD_BUILDING_EFFECT:
                    road_poss = self.board.valid_road_nodes(self.id)
                    if len(road_poss) >= 2:
                        return self.development_cards_hand.select_card(i)

        # --- 1) Ciudad ---
        if (town_count >= self.city_town_threshold and
                self.hand.resources.has_more(BuildConstants.CITY)):
            valid_cities = self.board.valid_city_nodes(self.id)
            if valid_cities:
                best_city = None
                best_pips = -1
                for nid in valid_cities:
                    p = self._max_pips_at_node(self.board, nid)
                    if p > best_pips:
                        best_pips = p
                        best_city = nid
                if best_city is not None and best_pips >= self.min_prob_city:
                    return {'building': BuildConstants.CITY, 'node_id': best_city}

        # --- 2) Pueblo ---
        if self.hand.resources.has_more(BuildConstants.TOWN):
            valid_towns = self.board.valid_town_nodes(self.id)
            if valid_towns:
                best_town, best_score = self._best_node(self.board, valid_towns)
                best_pips = self._max_pips_at_node(self.board, best_town) if best_town is not None else 0
                if best_town is not None and best_pips >= self.min_prob_town:
                    return {'building': BuildConstants.TOWN, 'node_id': best_town}

        # --- 3) Carretera (máx 2 por turno) ---
        # FIX: usar self.roads_built_this_turn (atributo de instancia)
        if self.roads_built_this_turn < 2 and self.hand.resources.has_more(BuildConstants.ROAD):
            valid_roads = self.board.valid_road_nodes(self.id)
            if valid_roads:
                harbor_roads = [
                    r for r in valid_roads
                    if self.board.nodes[r['finishing_node']]['harbor'] != HarborConstants.NONE
                ]
                if harbor_roads:
                    self.roads_built_this_turn += 1
                    chosen = harbor_roads[0]
                    return {
                        'building': BuildConstants.ROAD,
                        'node_id': chosen['starting_node'],
                        'road_to': chosen['finishing_node'],
                    }

                scored = []
                for r in valid_roads:
                    s = self._score_node(self.board, r['finishing_node'])
                    scored.append((s, r))
                scored.sort(key=lambda x: x[0], reverse=True)

                if random.random() < self.road_bias:
                    self.roads_built_this_turn += 1
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
    #  on_moving_thief — FIX: evitar mover al mismo terreno
    # ================================================================== #
    def on_moving_thief(self):
        current_thief_terrain = -1
        candidates = []

        for terrain in self.board.terrain:
            if terrain['has_thief']:
                current_thief_terrain = terrain['id']
                continue  # FIX: skip el terreno actual del ladrón

            pips = self._pips(terrain['probability'])
            if pips < self.thief_prob_threshold:
                continue

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
                continue

            if enemy_count > 0:
                priority = pips * 10 + enemy_count
                candidates.append((priority, terrain['id'], enemy_player))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, tid, enemy = candidates[0]
            return {'terrain': tid, 'player': enemy}

        # Fallback: si no hay candidatos válidos, buscar cualquier terreno válido
        for terrain in self.board.terrain:
            if terrain['id'] != current_thief_terrain:
                # Buscar un jugador enemigo en este terreno
                for nid in terrain['contacting_nodes']:
                    if self.board.nodes[nid]['player'] != -1 and self.board.nodes[nid]['player'] != self.id:
                        return {'terrain': terrain['id'], 'player': self.board.nodes[nid]['player']}
                # Si no hay enemigos, mover igual pero sin robar
                return {'terrain': terrain['id'], 'player': -1}
        
        return {'terrain': current_thief_terrain, 'player': -1}

    # ================================================================== #
    #  on_having_more_than_7_materials — FIX: max_iters aumentado
    # ================================================================== #
    def on_having_more_than_7_materials_when_thief_is_called(self):
        town_count, _ = self._count_own_towns_and_cities(self.board)
        goal = self._current_goal(town_count)
        goal_mats = self._goal_materials(goal)
        goal_list = [goal_mats.cereal, goal_mats.mineral, goal_mats.clay,
                     goal_mats.wood, goal_mats.wool]
        mat_names = ['cereal', 'mineral', 'clay', 'wood', 'wool']

        max_iters = 50   # FIX: aumentado de 20 a 50
        iters = 0
        while self.hand.get_total() > 7 and iters < max_iters:
            iters += 1
            discarded = False
            resource_order = sorted(range(5), key=lambda i: goal_list[i])
            for res_id in resource_order:
                current = getattr(self.hand.resources, mat_names[res_id])
                needed = goal_list[res_id]
                if current > needed:
                    self.hand.remove_material(res_id, 1)
                    discarded = True
                    break
            if not discarded:
                for res_id in resource_order:
                    if getattr(self.hand.resources, mat_names[res_id]) > 0:
                        self.hand.remove_material(res_id, 1)
                        discarded = True
                        break
                if not discarded:
                    break
        return self.hand

    # ================================================================== #
    #  on_monopoly_card_use
    # ================================================================== #
    def on_monopoly_card_use(self):
        town_count, _ = self._count_own_towns_and_cities(self.board)
        goal = self._current_goal(town_count)
        goal_mats = self._goal_materials(goal)
        goal_list = [goal_mats.cereal, goal_mats.mineral, goal_mats.clay, goal_mats.wood, goal_mats.wool]
        hand_list = [
            self.hand.resources.cereal,
            self.hand.resources.mineral,
            self.hand.resources.clay,
            self.hand.resources.wood,
            self.hand.resources.wool,
        ]
        deficits = [goal_list[i] - hand_list[i] for i in range(5)]
        best = max(range(5), key=lambda i: deficits[i])
        return best

    # ================================================================== #
    #  on_road_building_card_use — FIX: evitar elegir la misma carretera
    # ================================================================== #
    def on_road_building_card_use(self):
        valid_nodes = self.board.valid_road_nodes(self.id)
        if len(valid_nodes) >= 2:
            scored = [(self._score_node(self.board, r['finishing_node']), r) for r in valid_nodes]
            scored.sort(key=lambda x: x[0], reverse=True)
            r1 = scored[0][1]
            
            # FIX: Buscar segunda carretera que no sea la misma
            r2 = None
            for score, road in scored[1:]:
                # Evitar elegir la misma carretera (mismo par de nodos)
                if (road['starting_node'] != r1['starting_node'] or 
                    road['finishing_node'] != r1['finishing_node']):
                    r2 = road
                    break
            
            if r2 is None and len(scored) > 1:
                r2 = scored[1][1]  # Fallback
            
            if r2:
                return {
                    'node_id': r1['starting_node'],
                    'road_to': r1['finishing_node'],
                    'node_id_2': r2['starting_node'],
                    'road_to_2': r2['finishing_node'],
                }
            else:
                return {
                    'node_id': r1['starting_node'],
                    'road_to': r1['finishing_node'],
                    'node_id_2': None,
                    'road_to_2': None,
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
    #  on_year_of_plenty_card_use
    # ================================================================== #
    def on_year_of_plenty_card_use(self):
        town_count, _ = self._count_own_towns_and_cities(self.board)
        goal = self._current_goal(town_count)
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