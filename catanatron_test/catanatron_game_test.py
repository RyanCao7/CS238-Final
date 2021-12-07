from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color, HumanPlayer
from catanatron_experimental.my_player import MyPlayer
from catanatron_server.utils import open_link
from catanatron.models.enums import Resource, BuildingType
from catanatron.models.map import NUM_NODES, NUM_EDGES, NUM_TILES, Port, Tile, Water
from catanatron.models.board import get_edges
from catanatron import state_functions
import catanatron
import pydoc
import numpy as np

# --- Our agent's color ---
AGENT_COLOR = Color.BLUE

# --- 1v1 opponent color ---
OPPONENT_COLOR = Color.RED

game_attrs = [
    'copy',
    'execute', 
    'id', 
    'play', 
    'play_tick', 
    'seed', 
    'state', 
    'winning_color'
]

game_state_attrs = [
    'actions',
    'board',
    'buildings_by_color',
    'color_to_index',
    'colors',
    'copy',
    'current_player',
    'current_player_index',
    'current_prompt',
    'current_turn_index',
    'development_deck',
    'free_roads_available',
    'is_discarding',
    'is_initial_build_phase',
    'is_moving_knight',
    'is_road_building',
    'num_turns',
    'playable_actions',
    'player_state',
    'players',
    'resource_deck'
]

game_state_board_attrs = [
    'bfs_walk', 
    'board_buildable_ids', 
    'build_city', 
    'build_road', 
    'build_settlement', 
    'buildable_edges', 
    'buildable_node_ids', 
    'buildings', 
    'connected_components', 
    'continuous_roads_by_player', 
    'copy', 
    'find_connected_components', 
    'get_edge_color', 
    'get_node_color', 
    'get_player_port_resources', 
    'is_enemy_node', 
    'is_enemy_road', 
    'map', 
    'road_color', 
    'road_length', 
    'road_lengths', 
    'roads', 
    'robber_coordinate'
]


def generate_documentation(game):
    # --- For game attrs ---
    print(dir(game))
    for game_attr in game_attrs:
        print('\n' + ('-' * 30))
        print(game_attr)
        attr = getattr(game, game_attr)
        print(attr)

        attr_type = type(attr).__name__
        if attr_type == 'method':
            print(pydoc.render_doc(attr))

    # --- For game state ---
    print(dir(game.state))
    for game_state_attr in game_state_attrs:
        print('\n' + ('-' * 30))
        print(game_state_attr)
        attr = getattr(game.state, game_state_attr)
        print(attr)

        attr_type = type(attr).__name__
        if attr_type == 'method':
            print(pydoc.render_doc(attr))

    # --- For game.state.board ---
    print(dir(game.state.board))
    for game_state_board_attr in game_state_board_attrs:
        print('\n' + ('-' * 30))
        print(game_state_board_attr)
        attr = getattr(game.state.board, game_state_board_attr)
        print(attr)

        attr_type = type(attr).__name__
        if attr_type == 'method':
            print(pydoc.render_doc(attr))


def visualize_map(map_tiles):
    """
    Visualizes a Catan board map
    """
    map = list()
    for i in range(7):
        row = list()
        for j in range(7):
            row.append('NO')
        map.append(row)
    for (coord, tile) in map_tiles.items():
        axial_coord = catanatron.models.coordinate_system.cube_to_offset(coord)
        axial_coord = tuple(int(x) + 3 for x in axial_coord)
        map[axial_coord[0]][axial_coord[1]] = 'ye'
    return map


def get_edge_mapping():
    """
    Maps from edge coordinate tuples to index and vice versa.

    TODO: Compute this once and cache it!
    """
    edges_to_indices = dict()
    indices_to_edges = dict()
    all_edges_sorted = sorted(get_edges(), key=lambda x: x[0] * 100 + x[1])
    for (idx, edge) in enumerate(all_edges_sorted):
        edge = (min(edge), max(edge))
        edges_to_indices[edge] = idx
        indices_to_edges[idx] = edge
    return edges_to_indices, indices_to_edges


# --- For resource ordering within resource vector ---
RESOURCES_TO_IDX = {
    Resource.WOOD: 0,
    Resource.BRICK: 1,
    Resource.SHEEP: 2,
    Resource.WHEAT: 3,
    Resource.ORE: 4
}
IDX_TO_RESOURCES = [
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
]


def extract_board_state(game):
    """
    Returns a board state from the given game.

    NOTE: Edge serialization is going to be
    sorted version of IDs (node_id_1 * 100 + node_id_2)
    """
    map = game.state.board.map
    state = {
        # --- What do we need? Center hexagons ---
        # Dice numbers
        # One-hot for each resource type:
        # > Ore, wheat, sheep, brick, wood, desert
        'grid_dice_nums': np.zeros((7, 7)),
        'grid_wood_locs': np.zeros((7, 7)),
        'grid_brick_locs': np.zeros((7, 7)),
        'grid_sheep_locs': np.zeros((7, 7)),
        'grid_wheat_locs': np.zeros((7, 7)),
        'grid_ore_locs': np.zeros((7, 7)),
        'grid_desert_locs': np.zeros((7, 7)),

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
    }

    # --- Robber ---
    state['grid_robber_loc'] = catanatron.models.coordinate_system.cube_to_offset(game.state.board.robber_coordinate)

    # --- Grab resource types (this can be cached, or not used at all) ---
    for (coord, tile) in map.tiles.items():
        offset_coord = catanatron.models.coordinate_system.cube_to_offset(coord)
        offset_coord = tuple(int(x) + 3 for x in offset_coord) # --- [-3, 3] to [0, 6] ---

        if type(tile) == Port:
            pass # We are not doing trading for now

        elif type(tile) == Tile:
            # --- Dice numbers ---
            state['grid_dice_nums'][offset_coord[0]][offset_coord[1]] = tile.number

            # --- Resources ---
            if tile.resource == Resource.WOOD:
                state['grid_wood_locs'][offset_coord[0]][offset_coord[1]] = 1
            elif tile.resource == Resource.BRICK:
                state['grid_brick_locs'][offset_coord[0]][offset_coord[1]] = 1
            elif tile.resource == Resource.SHEEP:
                state['grid_sheep_locs'][offset_coord[0]][offset_coord[1]] = 1
            elif tile.resource == Resource.WHEAT:
                state['grid_wheat_locs'][offset_coord[0]][offset_coord[1]] = 1
            elif tile.resource == Resource.ORE:
                state['grid_ore_locs'][offset_coord[0]][offset_coord[1]] = 1

        elif type(tile) == Water:
            pass # Nothing useful here

        else:
            print(f'No such tile type: {tile} with type {type(tile)}')
    
    # --- Grab edge info ---
    edges_to_indices, indices_to_edges = get_edge_mapping()
    for (edge, road) in game.state.board.roads.items():
        edge = (min(edge), max(edge))
        if edge in edges_to_indices:
            edge_idx = edges_to_indices[edge]
        else:
            print(f'Ruh roh: {edge} not contained within the dict!')
        # --- NOTE: There should only be your roads and those of your opponents ---
        # --- NOTE: Agent's bot is ALWAYS blue ---
        state['road_colors'][edge_idx] = 1 if road == AGENT_COLOR else -1
    
    # --- Grab node info ---
    for (node, (building_color, building_type)) in game.state.board.buildings.items():

        # --- Buildings ---
        value = 1 if building_color == AGENT_COLOR else -1
        if building_type == BuildingType.SETTLEMENT:
            state['settlement_locs'][node] = value
        elif building_type == BuildingType.CITY:
            state['city_locs'][node] = value
    
    # --- Grab resource info for agent + opponent ---
    agent_key = state_functions.player_key(game.state, AGENT_COLOR)
    state['player_resources'][RESOURCES_TO_IDX[Resource.WOOD]] = game.state.player_state[f"{agent_key}_WOOD_IN_HAND"]
    state['player_resources'][RESOURCES_TO_IDX[Resource.BRICK]] = game.state.player_state[f"{agent_key}_BRICK_IN_HAND"]
    state['player_resources'][RESOURCES_TO_IDX[Resource.SHEEP]] = game.state.player_state[f"{agent_key}_SHEEP_IN_HAND"]
    state['player_resources'][RESOURCES_TO_IDX[Resource.WHEAT]] = game.state.player_state[f"{agent_key}_WHEAT_IN_HAND"]
    state['player_resources'][RESOURCES_TO_IDX[Resource.ORE]] = game.state.player_state[f"{agent_key}_ORE_IN_HAND"]

    opponent_key = state_functions.player_key(game.state, AGENT_COLOR)
    state['opponent_resources'][RESOURCES_TO_IDX[Resource.WOOD]] = game.state.player_state[f"{opponent_key}_WOOD_IN_HAND"]
    state['opponent_resources'][RESOURCES_TO_IDX[Resource.BRICK]] = game.state.player_state[f"{opponent_key}_BRICK_IN_HAND"]
    state['opponent_resources'][RESOURCES_TO_IDX[Resource.SHEEP]] = game.state.player_state[f"{opponent_key}_SHEEP_IN_HAND"]
    state['opponent_resources'][RESOURCES_TO_IDX[Resource.WHEAT]] = game.state.player_state[f"{opponent_key}_WHEAT_IN_HAND"]
    state['opponent_resources'][RESOURCES_TO_IDX[Resource.ORE]] = game.state.player_state[f"{opponent_key}_ORE_IN_HAND"]

    return state


def main():
    # Play a simple 4v4 game. Edit MyPlayer with your logic!
    players = [
        HumanPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players)

    # --- Document ---
    # generate_documentation(game)

    # --- Play through a game, step by step ---
    counter = 0
    while game.winning_color() is None:
        game.play_tick(action_callbacks=[], decide_fn=None)
        map = game.state.board.map
        print('\n' + ('-' * 30))
        # print(dir(map), '\n')
        # print(map.coordinate_system, '\n')
        # print(map.tiles, '\n')
        # visualized_map = visualize_map(map.tiles)
        # for row in visualized_map:
        #     print(row)
        # for (coord, tile) in map.tiles.items():
        #     axial_coord = catanatron.models.coordinate_system.cube_to_axial(coord)
        #     offset_coord = catanatron.models.coordinate_system.cube_to_offset(coord)
        #     print(coord, axial_coord, offset_coord, tile)
        # print(map.resource_tiles, '\n')

        # roads = game.state.board.roads
        # buildings = game.state.board.buildings
        # for (edge, road) in roads.items():
        #     print(road)
        #     print(type(road))
        #     print(dir(road))
        # for (node, building) in buildings.items():
        #     print(building)
        #     print(type(building))
        #     print(dir(building))
        # if len(roads) > 4 and len(buildings) > 4:
        #     break

        game_state = extract_board_state(game)
        print(game_state['player_resources'])
        print(game_state['settlement_locs'])

        counter += 1
        print(f'Counter: {counter}')
        # if counter == 1:
        #     break


if __name__ == '__main__':
    main()