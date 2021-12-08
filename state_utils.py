from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from catanatron_experimental.my_player import MyPlayer
from catanatron_server.utils import open_link
from catanatron.models.enums import Resource, BuildingType
from catanatron.models.map import NUM_NODES, NUM_EDGES, NUM_TILES, Port, Tile, Water
from catanatron import state_functions
import catanatron
import pydoc
import numpy as np
import torch

import constants


def visualize_map(map_tiles):
    """
    Visualizes a Catan board map.
    """
    map = list()
    for i in range(7):
        row = list()
        for j in range(7):
            row.append('WATER')
        map.append(row)
    for (coord, tile) in map_tiles.items():
        axial_coord = catanatron.models.coordinate_system.cube_to_offset(coord)
        axial_coord = tuple(int(x) + 3 for x in axial_coord)
        map[axial_coord[0]][axial_coord[1]] = 'earth'
    return map


def extract_board_state(game, playable_actions, agent_color=constants.AGENT_COLOR):
    """
    Returns a board state from the given game.

    NOTE: Edge serialization is going to be
    sorted version of IDs (node_id_1 * 100 + node_id_2)
    """
    map = game.state.board.map
    board_state = {
        # --- The above are immutable ---
        'grid_robber_loc': np.zeros((7, 7)),

        # --- What do we need? Edges ---
        # Road colors for each edge (one-hot)
        # Alteratively, we can train a 1v1 bot and do {1, 0, -1} indicators
        'road_colors': np.zeros(NUM_EDGES),

        # --- What do we need? Vertices ---
        # Settlements and cities for each vertex (one-hot)
        # Alternatively, train a 1v1 bot and do {1, 0, -1} indicators
        'settlement_locs': np.zeros(NUM_NODES),
        'city_locs': np.zeros(NUM_NODES),

        # --- What do we need? Game state ---
        # Vector of resources and how many of each kind you have
        # > Ore, wheat, sheep, brick, wood
        'player_resources': np.zeros(len(Resource)),
        'opponent_resources': np.zeros(len(Resource)),

        # --- Action mask ---
        'action_mask': np.zeros(constants.TOTAL_DQN_ACTIONS)
    }

    # --- Create action mask ---
    for action in playable_actions:
        if action not in constants.ACTIONS_TO_INDICES:
            raise RuntimeError(f'Ruh roh, {action} not in ACTIONS_TO_INDICES!')
        board_state['action_mask'][constants.ACTIONS_TO_INDICES[action]] = 1

    # --- Throw in the immutables (TODO(ryancao): Do we even need this?) ---
    # board_state.update(constants.IMMUTABLE_BOARD_STATE)

    # --- Robber ---
    robber_coords = catanatron.models.coordinate_system.cube_to_offset(game.state.board.robber_coordinate)
    board_state['grid_robber_loc'][robber_coords] = 1
    
    # --- Grab edge info ---
    for (edge, road) in game.state.board.roads.items():
        # --- Edges are always (small_node_id, large_node_id) order ---
        edge = (min(edge), max(edge))
        if edge in constants.EDGES_TO_INDICES:
            edge_idx = constants.EDGES_TO_INDICES[edge]
        else:
            print(f'\n\nRuh roh: {edge} not contained within the dict! This is very bad!\n\n')
        # --- NOTE: There should only be your roads and those of your opponents ---
        # --- NOTE: Agent's bot is ALWAYS blue ---
        board_state['road_colors'][edge_idx] = 1 if road == agent_color else -1
    
    # --- Grab node info ---
    for (node, (building_color, building_type)) in game.state.board.buildings.items():

        # --- Buildings ---
        value = 1 if building_color == agent_color else -1
        if building_type == BuildingType.SETTLEMENT:
            board_state['settlement_locs'][node] = value
        elif building_type == BuildingType.CITY:
            board_state['city_locs'][node] = value
    
    # --- Grab resource info for agent + opponent ---
    agent_key = state_functions.player_key(game.state, agent_color)
    board_state['player_resources'][constants.RESOURCES_TO_IDX[Resource.WOOD]] = \
        game.state.player_state[f"{agent_key}_WOOD_IN_HAND"]
    board_state['player_resources'][constants.RESOURCES_TO_IDX[Resource.BRICK]] = \
        game.state.player_state[f"{agent_key}_BRICK_IN_HAND"]
    board_state['player_resources'][constants.RESOURCES_TO_IDX[Resource.SHEEP]] = \
        game.state.player_state[f"{agent_key}_SHEEP_IN_HAND"]
    board_state['player_resources'][constants.RESOURCES_TO_IDX[Resource.WHEAT]] = \
        game.state.player_state[f"{agent_key}_WHEAT_IN_HAND"]
    board_state['player_resources'][constants.RESOURCES_TO_IDX[Resource.ORE]] = \
        game.state.player_state[f"{agent_key}_ORE_IN_HAND"]

    opponent_key = state_functions.player_key(game.state, agent_color)
    board_state['opponent_resources'][constants.RESOURCES_TO_IDX[Resource.WOOD]] = \
        game.state.player_state[f"{opponent_key}_WOOD_IN_HAND"]
    board_state['opponent_resources'][constants.RESOURCES_TO_IDX[Resource.BRICK]] = \
        game.state.player_state[f"{opponent_key}_BRICK_IN_HAND"]
    board_state['opponent_resources'][constants.RESOURCES_TO_IDX[Resource.SHEEP]] = \
        game.state.player_state[f"{opponent_key}_SHEEP_IN_HAND"]
    board_state['opponent_resources'][constants.RESOURCES_TO_IDX[Resource.WHEAT]] = \
        game.state.player_state[f"{opponent_key}_WHEAT_IN_HAND"]
    board_state['opponent_resources'][constants.RESOURCES_TO_IDX[Resource.ORE]] = \
        game.state.player_state[f"{opponent_key}_ORE_IN_HAND"]

    # --- Adding channel dim and converting to tensor ---
    for (key, value) in board_state.items():
        board_state[key] = torch.from_numpy(np.expand_dims(value, axis=0))

    return board_state