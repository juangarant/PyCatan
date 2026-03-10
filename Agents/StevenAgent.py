import random

from Classes.Constants import *
from Classes.Materials import Materials
from Classes.TradeOffer import TradeOffer
from Interfaces.AgentInterface import AgentInterface


class GeneticAgent(AgentInterface):
    """
    Agente para Catan controlado por 10 parámetros evolucionables mediante GA.

    Parámetros:
        Colocación inicial:
            w_pips       – peso de producción bruta (pips de los números)
            w_diversity  – peso de diversidad de recursos
            w_port       – peso de acceso a puertos
            w_expand     – peso de potencial de expansión

        Prioridades de construcción (normalizadas, deben sumar 1):
            alpha_settlement – prioridad de asentamientos
            alpha_city       – prioridad de ciudades
            alpha_dev        – prioridad de cartas de desarrollo
            alpha_road       – prioridad de carreteras

        Ladrón y riesgo:
            w_robber_block   – agresividad al colocar el ladrón (pondera pips bloqueados)
            risk_hand_limit  – umbral de cartas en mano antes de empezar a gastar
    """

    PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    town_number = 0

    def __init__(self, agent_id,
                 w_pips=1.0, w_diversity=0.6, w_port=0.5, w_expand=0.4,
                 alpha_settlement=0.35, alpha_city=0.30, alpha_dev=0.20, alpha_road=0.15,
                 w_robber_block=1.0, risk_hand_limit=6):
        super().__init__(agent_id)

        self.w_pips      = w_pips
        self.w_diversity = w_diversity
        self.w_port      = w_port
        self.w_expand    = w_expand

        # Normalizar alphas para que sumen 1
        total = alpha_settlement + alpha_city + alpha_dev + alpha_road
        self.alpha_settlement = alpha_settlement / total
        self.alpha_city       = alpha_city       / total
        self.alpha_dev        = alpha_dev        / total
        self.alpha_road       = alpha_road       / total

        self.w_robber_block  = w_robber_block
        self.risk_hand_limit = risk_hand_limit

        # Recursos cubiertos por el primer asentamiento (para complementar en el segundo)
        self._first_settlement_resources = set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pips(self, probability):
        return self.PIPS.get(probability, 0)

    def _node_resources(self, node_id):
        """Devuelve el conjunto de tipos de recurso que produce un nodo."""
        return {
            self.board.terrain[t].get('resource')
            for t in self.board.nodes[node_id]['contacting_terrain']
            if self.board.terrain[t].get('resource') not in (None, 'desert')
        }

    def _score_node(self, node_id, covered_resources=None):
        """
        Puntúa una intersección con los 4 pesos del cromosoma.
        Si se pasa covered_resources, penaliza recursos ya cubiertos para
        fomentar diversidad entre el 1er y 2º asentamiento.
        """
        total_pips = sum(
            self._pips(self.board.terrain[t]['probability'])
            for t in self.board.nodes[node_id]['contacting_terrain']
        )
        resource_types = self._node_resources(node_id)
        diversity = len(resource_types)

        # Si ya tenemos recursos cubiertos, bonificamos los recursos nuevos
        if covered_resources:
            new_resources = resource_types - covered_resources
            diversity_bonus = len(new_resources) * self.w_diversity
        else:
            diversity_bonus = diversity * self.w_diversity

        port_bonus = 1 if self.board.nodes[node_id].get('harbor', HarborConstants.NONE) != HarborConstants.NONE else 0
        expand     = sum(1 for adj in self.board.nodes[node_id]['adjacent'] if self.board.nodes[adj]['player'] == -1)

        return (self.w_pips    * total_pips
                + diversity_bonus
                + self.w_port  * port_bonus
                + self.w_expand * expand)

    def _best_road_towards_value(self, node_id):
        """
        Elige la carretera desde node_id que apunte al nodo adyacente
        con mayor puntuación (libre y no ocupado).
        Fallback: adyacente aleatorio.
        """
        adjacent = self.board.nodes[node_id]['adjacent']
        free_adj = [n for n in adjacent if self.board.nodes[n]['player'] == -1]

        if not free_adj:
            return adjacent[random.randint(0, len(adjacent) - 1)]

        # Puntúa cada nodo adyacente libre como destino de carretera
        return max(free_adj, key=lambda n: sum(
            self._pips(self.board.terrain[t]['probability'])
            for t in self.board.nodes[n]['contacting_terrain']
        ))

    # ------------------------------------------------------------------
    # on_trade_offer
    # ------------------------------------------------------------------

    def on_trade_offer(self, board_instance, offer=TradeOffer(), player_id=int):
        return offer.gives.has_more(offer.receives)

    # ------------------------------------------------------------------
    # on_turn_start
    # ------------------------------------------------------------------

    def on_turn_start(self):
        knight_cards = self.development_cards_hand.find_card_by_effect(DevelopmentCardConstants.KNIGHT_EFFECT)
        return knight_cards[0] if len(knight_cards) > 0 else None

    # ------------------------------------------------------------------
    # on_having_more_than_7_materials_when_thief_is_called
    # ------------------------------------------------------------------

    def on_having_more_than_7_materials_when_thief_is_called(self):
        # Si puede hacer ciudad, conserva trigo y mineral, descarta el resto
        if self.hand.resources.has_more(BuildConstants.CITY):
            discard_order = [
                MaterialConstants.WOOL,
                MaterialConstants.CLAY,
                MaterialConstants.WOOD,
            ]
            # Solo descarta sobrantes de cereal/mineral si hay demasiados
            while self.hand.get_total() > 7:
                discarded = False
                for mat in discard_order:
                    if self.hand.get_total() <= 7:
                        break
                    if self.hand.resources.get_from_id(mat) > 0:
                        self.hand.remove_material(mat, 1)
                        discarded = True
                # Sobrantes de cereal (>2) y mineral (>3) al final
                if not discarded or self.hand.get_total() > 7:
                    if self.hand.resources.cereal > 2:
                        self.hand.remove_material(MaterialConstants.CEREAL, 1)
                    elif self.hand.resources.mineral > 3:
                        self.hand.remove_material(MaterialConstants.MINERAL, 1)
                    else:
                        break  # No hay nada más que descartar sin perjudicarnos
        else:
            # Descarta en orden de menor utilidad según plan activo
            discard_order = [
                MaterialConstants.WOOL,
                MaterialConstants.CLAY,
                MaterialConstants.WOOD,
                MaterialConstants.CEREAL,
                MaterialConstants.MINERAL,
            ]
            while self.hand.get_total() > 7:
                discarded = False
                for mat in discard_order:
                    if self.hand.get_total() <= 7:
                        break
                    if self.hand.resources.get_from_id(mat) > 0:
                        self.hand.remove_material(mat, 1)
                        discarded = True
                        break
                if not discarded:
                    break  # Evita bucle infinito si no se puede descartar nada
        return self.hand

    # ------------------------------------------------------------------
    # on_moving_thief
    # ------------------------------------------------------------------

    def on_moving_thief(self):
        best_score            = -1
        best_result           = None
        terrain_with_thief_id = -1

        for terrain in self.board.terrain:
            if terrain['has_thief']:
                terrain_with_thief_id = terrain['id']
                continue

            nodes   = self.board.__get_contacting_nodes__(terrain['id'])
            has_own = any(self.board.nodes[n]['player'] == self.id for n in nodes)
            if has_own:
                continue

            enemy = next((self.board.nodes[n]['player'] for n in nodes
                          if self.board.nodes[n]['player'] not in (-1, self.id)), -1)
            if enemy == -1:
                continue

            score = self.w_robber_block * self._pips(terrain['probability'])
            if score > best_score:
                best_score  = score
                best_result = {'terrain': terrain['id'], 'player': enemy}

        return best_result if best_result else {'terrain': terrain_with_thief_id, 'player': -1}

    # ------------------------------------------------------------------
    # on_turn_end
    # ------------------------------------------------------------------

    def on_turn_end(self):
        if len(self.development_cards_hand.hand):
            for i in range(0, len(self.development_cards_hand.hand)):
                if self.development_cards_hand.hand[i].type == DevelopmentCardConstants.VICTORY_POINT:
                    return self.development_cards_hand.select_card(i)
        return None

    # ------------------------------------------------------------------
    # on_commerce_phase
    # ------------------------------------------------------------------

    def on_commerce_phase(self):
        # Si puede construir ciudad directamente, no comercia
        if self.town_number >= 1 and self.hand.resources.has_more(BuildConstants.CITY):
            return None

        # Pide para ciudad si tiene pueblos
        if self.town_number >= 1:
            cereal_hand  = self.hand.resources.cereal
            mineral_hand = self.hand.resources.mineral
            wood_hand    = self.hand.resources.wood
            clay_hand    = self.hand.resources.clay
            wool_hand    = self.hand.resources.wool

            total_needed = max(0, 2 - cereal_hand) + max(0, 3 - mineral_hand)

            if total_needed <= 0:
                return None

            surplus = wood_hand + clay_hand + wool_hand

            if surplus > total_needed:
                materials_to_give = [0, 0, 0, 0, 0]
                for i in range(0, total_needed):
                    order = [MaterialConstants.CLAY, MaterialConstants.WOOD, MaterialConstants.WOOL]
                    random.shuffle(order)
                    for mat in order:
                        if self.hand.resources.get_from_id(mat) > 0:
                            self.hand.remove_material(mat, 1)
                            materials_to_give[mat] += 1
                            break
                gives = Materials(materials_to_give[0], materials_to_give[1], materials_to_give[2],
                                  materials_to_give[3], materials_to_give[4])
            else:
                gives = Materials(0, 0, clay_hand, wood_hand, wool_hand)

            receives = Materials(2, 3, 0, 0, 0)

        # Si no tiene pueblos, pide para pueblo
        else:
            if self.hand.resources.has_more(Materials(1, 0, 1, 1, 1)):
                return None

            materials_to_receive = [0, 0, 0, 0, 0]
            materials_to_give    = [0, 0, 0, 0, 0]
            number_of_materials_received = 0

            materials_to_receive[0] = 1 - self.hand.resources.cereal
            materials_to_receive[1] = 0 - self.hand.resources.mineral
            materials_to_receive[2] = 1 - self.hand.resources.clay
            materials_to_receive[3] = 1 - self.hand.resources.wood
            materials_to_receive[4] = 1 - self.hand.resources.wool

            for i in range(0, len(materials_to_receive)):
                if materials_to_receive[i] <= 0:
                    materials_to_receive[i] = 0
                else:
                    number_of_materials_received += 1

            for j in range(0, number_of_materials_received):
                order = [MaterialConstants.CEREAL, MaterialConstants.MINERAL, MaterialConstants.CLAY,
                         MaterialConstants.WOOD, MaterialConstants.WOOL]
                random.shuffle(order)
                for mat in order:
                    if self.hand.resources.get_from_id(mat) > 1 or mat == MaterialConstants.MINERAL:
                        self.hand.remove_material(mat, 1)
                        materials_to_give[mat] += 1
                        break

            gives    = Materials(materials_to_give[0], materials_to_give[1], materials_to_give[2],
                                 materials_to_give[3], materials_to_give[4])
            receives = Materials(materials_to_receive[0], materials_to_receive[1], materials_to_receive[2],
                                 materials_to_receive[3], materials_to_receive[4])

        return TradeOffer(gives, receives)

    # ------------------------------------------------------------------
    # on_build_phase
    # ------------------------------------------------------------------

    def on_build_phase(self, board_instance):
        self.board = board_instance

        # Cartas de desarrollo activas
        if len(self.development_cards_hand.hand):
            for i in range(0, len(self.development_cards_hand.hand)):
                road_possibilities = self.board.valid_road_nodes(self.id)
                if (self.development_cards_hand.hand[i].effect == DevelopmentCardConstants.YEAR_OF_PLENTY_EFFECT or
                        (self.development_cards_hand.hand[i].effect == DevelopmentCardConstants.ROAD_BUILDING_EFFECT and
                         len(road_possibilities) > 1)):
                    return self.development_cards_hand.select_card(i)

        # Construir según prioridad de alphas
        actions = sorted([
            ('city',       self.alpha_city       if self.town_number > 0 else 0),
            ('settlement', self.alpha_settlement),
            ('road',       self.alpha_road),
            ('dev',        self.alpha_dev),
        ], key=lambda x: x[1], reverse=True)

        for action, _ in actions:
            result = self._try_build(action)
            if result is not None:
                return result

        return None

    def _try_build(self, action):
        if action == 'city':
            if self.hand.resources.has_more(BuildConstants.CITY) and self.town_number > 0:
                possibilities = self.board.valid_city_nodes(self.id)
                for node_id in possibilities:
                    for terrain_piece_id in self.board.nodes[node_id]['contacting_terrain']:
                        if self.board.terrain[terrain_piece_id]['probability'] in (5, 6, 8, 9):
                            self.town_number -= 1
                            return {'building': BuildConstants.CITY, 'node_id': node_id}

        elif action == 'settlement':
            if self.hand.resources.has_more(BuildConstants.TOWN):
                possibilities = self.board.valid_town_nodes(self.id)
                for node_id in possibilities:
                    for terrain_piece_id in self.board.nodes[node_id]['contacting_terrain']:
                        if self.board.terrain[terrain_piece_id]['probability'] in (4, 5, 6, 8, 9, 10):
                            self.town_number += 1
                            return {'building': BuildConstants.TOWN, 'node_id': node_id}

        elif action == 'road':
            if self.hand.resources.has_more(BuildConstants.ROAD):
                possibilities = self.board.valid_road_nodes(self.id)
                # Priorizar rutas hacia puertos
                for road_obj in possibilities:
                    if (self.board.is_coastal_node(road_obj['finishing_node']) and
                            self.board.nodes[road_obj['finishing_node']]['harbor'] != HarborConstants.NONE):
                        return {'building': BuildConstants.ROAD,
                                'node_id': road_obj['starting_node'],
                                'road_to': road_obj['finishing_node']}
                # Carretera aleatoria ponderada por alpha_road (≈60% con valores por defecto)
                will_build = random.randint(0, 2)
                if will_build and len(possibilities):
                    road_node = random.randint(0, len(possibilities) - 1)
                    return {'building': BuildConstants.ROAD,
                            'node_id': possibilities[road_node]['starting_node'],
                            'road_to': possibilities[road_node]['finishing_node']}

        elif action == 'dev':
            if self.hand.resources.has_more(BuildConstants.CARD):
                return {'building': BuildConstants.CARD}

        return None

    # ------------------------------------------------------------------
    # on_game_start  [MEJORADO: 2 asentamientos complementarios + carretera dirigida]
    # ------------------------------------------------------------------

    def on_game_start(self, board_instance):
        self.board = board_instance
        possibilities = self.board.valid_starting_nodes()

        if self.town_number == 0:
            # ── Primer asentamiento: máxima puntuación bruta ──
            chosen_node_id = max(possibilities, key=lambda n: self._score_node(n))
            # Guardamos los recursos que cubre para complementar en el segundo
            self._first_settlement_resources = self._node_resources(chosen_node_id)
        else:
            # ── Segundo asentamiento: complementa al primero ──
            # _score_node bonifica recursos nuevos que no tenga el primer asentamiento
            chosen_node_id = max(
                possibilities,
                key=lambda n: self._score_node(n, covered_resources=self._first_settlement_resources)
            )

        self.town_number += 1

        # ── Carretera dirigida: apunta al nodo adyacente de mayor valor ──
        chosen_road_to_id = self._best_road_towards_value(chosen_node_id)

        return chosen_node_id, chosen_road_to_id

    # ------------------------------------------------------------------
    # on_monopoly_card_use
    # ------------------------------------------------------------------

    def on_monopoly_card_use(self):
        if self.town_number >= 1:
            cereal_need  = max(0, 2 - self.hand.resources.cereal)
            mineral_need = max(0, 3 - self.hand.resources.mineral)
            return MaterialConstants.CEREAL if cereal_need >= mineral_need else MaterialConstants.MINERAL
        return MaterialConstants.CEREAL

    # ------------------------------------------------------------------
    # on_road_building_card_use  [FIX: eliminado while True]
    # ------------------------------------------------------------------

    def on_road_building_card_use(self):
        valid_nodes = self.board.valid_road_nodes(self.id)
        if len(valid_nodes) > 1:
            # FIX: sample garantiza índices distintos sin riesgo de bucle infinito
            r1, r2 = random.sample(range(len(valid_nodes)), 2)
            return {'node_id':   valid_nodes[r1]['starting_node'],
                    'road_to':   valid_nodes[r1]['finishing_node'],
                    'node_id_2': valid_nodes[r2]['starting_node'],
                    'road_to_2': valid_nodes[r2]['finishing_node']}
        elif len(valid_nodes) == 1:
            return {'node_id':   valid_nodes[0]['starting_node'],
                    'road_to':   valid_nodes[0]['finishing_node'],
                    'node_id_2': None,
                    'road_to_2': None}
        return None

    # ------------------------------------------------------------------
    # on_year_of_plenty_card_use  [FIX: dinámico según mano real]
    # ------------------------------------------------------------------

    def on_year_of_plenty_card_use(self):
        """
        Pide los 2 recursos de los que más nos falte para la construcción
        prioritaria según los alphas, mirando la mano real en cada momento.
        """
        # Determinar objetivo prioritario según alphas
        if self.alpha_city > self.alpha_settlement and self.town_number > 0:
            # Objetivo: ciudad (coste: 2 cereal + 3 mineral)
            shortage = [
                (MaterialConstants.CEREAL,  max(0, 2 - self.hand.resources.cereal)),
                (MaterialConstants.MINERAL, max(0, 3 - self.hand.resources.mineral)),
                (MaterialConstants.WOOL,    0),
                (MaterialConstants.CLAY,    0),
                (MaterialConstants.WOOD,    0),
            ]
        else:
            # Objetivo: pueblo (coste: 1 cereal + 1 clay + 1 wood + 1 wool)
            shortage = [
                (MaterialConstants.CEREAL, max(0, 1 - self.hand.resources.cereal)),
                (MaterialConstants.CLAY,   max(0, 1 - self.hand.resources.clay)),
                (MaterialConstants.WOOD,   max(0, 1 - self.hand.resources.wood)),
                (MaterialConstants.WOOL,   max(0, 1 - self.hand.resources.wool)),
                (MaterialConstants.MINERAL, 0),
            ]

        # Ordenar por escasez descendente y coger los 2 más necesarios
        shortage.sort(key=lambda x: x[1], reverse=True)

        mat1 = shortage[0][0]
        mat2 = shortage[1][0]

        # Si los 2 primeros tienen escasez 0, pide los más genéricamente útiles
        if shortage[0][1] == 0:
            mat1, mat2 = MaterialConstants.CEREAL, MaterialConstants.MINERAL

        return {'material': mat1, 'material_2': mat2}