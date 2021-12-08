from catanatron.game import Game
from catanatron.models.coordinate_system import cube_to_offset
from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES, NUM_EDGES, NUM_TILES
from catanatron.models.enums import Action, ActionType, BuildingType, Resource
from catanatron.models.player import RandomPlayer, Color, HumanPlayer
import torch
import models
import os

# --- train_dqn.py ---
ROOT_MODEL_SAVE_DIR = 'models'
ROOT_VIZ_SAVE_DIR = 'visuals'
def get_model_save_dir(model_type, model_name):
    return os.path.join(ROOT_MODEL_SAVE_DIR, model_type, model_name)
def get_viz_save_dir(model_type, model_name):
    return os.path.join(ROOT_VIZ_SAVE_DIR, model_type, model_name)

# --- opts.py ---
REPLAY_BUFFER_CAPACITY = 20000
DEFAULT_DQN_LR = 1e-3
DEFAULT_DQN_NUM_STEPS = 1000000 # Roughly 1000 to 5000 games
DEFAULT_DQN_OPTIM = 'adam'
TRAIN_EVERY_NUM_TIMESTEPS = 5
UPDATE_DQN_EVERY_NUM_TIMESTEPS = 500
DEFAULT_BATCH_SIZE = 16
GAMMA = 0.99
EPSILON = 0.5
DQN_MODEL_TYPES = {
    'Catan_Feedforward_DQN': models.Catan_Feedforward_DQN
}
DEFAULT_DQN_MODEL = 'Catan_Feedforward_DQN'

# --- models.py ---
DEFAULT_BOARD_SIZE = 7
BUY_DEVELOPMENT_CARD_ACTIONS = 2
PLAY_DEVELOPMENT_CARD_ACTIONS = 2
# Knight, road, era of plenty, monopoly, (VP)
PLAYABLE_DEV_CARDS = [
    ActionType.PLAY_KNIGHT_CARD,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
    ActionType.PLAY_ROAD_BUILDING
]

# --- Device ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Our agent's color ---
AGENT_COLOR = Color.BLUE

# --- 1v1 opponent color ---
OPPONENT_COLOR = Color.RED

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


def get_total_num_tiles():
    """
    Returns total number of tiles within the map (including water)
    """

    # --- Dummy setup ---
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players)
    map = game.state.board.map
    return len(map.tiles)

# --- Cached ---
TOTAL_NUM_TILES = get_total_num_tiles()

# Settlements, cities, roads, robber moving, play dev card, buy dev card, and end turn.
TOTAL_DQN_ACTIONS = NUM_NODES * 2 + NUM_EDGES + TOTAL_NUM_TILES + len(PLAYABLE_DEV_CARDS) + 1 + 1

def edge_hash(edge):
    """
    Hashes a given (n_1, n_2) edge.
    """
    return edge[0] * 100 + edge[1]


def cubical_coord_hash(cubical_coord):
    """
    Returns the "hash value" of a given cubical coord.

    This does NOT produce the same ordering as converting to offset, then hashing!
    """
    all_positive_coord = list(x + int(DEFAULT_BOARD_SIZE / 2) for x in cubical_coord)
    return all_positive_coord[0] * 100 + all_positive_coord[1] * 10 + all_positive_coord[2]


def get_edge_mapping():
    """
    Maps from edge coordinate tuples to index and vice versa.
    """
    edges_to_indices = dict()
    indices_to_edges = dict()
    all_edges_sorted = sorted(get_edges(), key=edge_hash)
    for (idx, edge) in enumerate(all_edges_sorted):
        edge = (min(edge), max(edge))
        edges_to_indices[edge] = idx
        indices_to_edges[idx] = edge
    return edges_to_indices, indices_to_edges

# --- Cached ---
EDGES_TO_INDICES, INDICES_TO_EDGES = get_edge_mapping()

def get_action_mapping(agent_color=AGENT_COLOR):
    """
    Maps from actions to action indices and vice versa.

    Note: Action = namedtuple("Action", ["color", "action_type", "value"])
    """
    # --- Dummy setup ---
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players)
    map = game.state.board.map

    actions_to_indices = dict()
    indices_to_actions = dict()
    all_nodes = list(range(NUM_NODES))
    all_edges_sorted = sorted(get_edges(), key=edge_hash)
    all_tiles_sorted = sorted(list(map.tiles.keys()), key=cubical_coord_hash)

    # --- First, all building actions. Settlements, then cities. ---
    idx = 0
    for node in all_nodes:
        build_settlement_action = Action(agent_color, ActionType.BUILD_SETTLEMENT, node)
        actions_to_indices[build_settlement_action] = idx
        indices_to_actions[idx] = build_settlement_action
        idx += 1
    for node in all_nodes:
        build_city_action = Action(agent_color, ActionType.BUILD_CITY, node)
        actions_to_indices[build_city_action] = idx
        indices_to_actions[idx] = build_city_action
        idx += 1

    # --- Next, all roads. ---
    for idx_delta in range(len(INDICES_TO_EDGES)):
        edge = INDICES_TO_EDGES[idx_delta]
        build_road_action = Action(agent_color, ActionType.BUILD_ROAD, edge)
        actions_to_indices[build_road_action] = idx
        indices_to_actions[idx] = build_road_action
        idx += 1

    # --- Next, all robber movement actions. ---
    for tile_loc in all_tiles_sorted:
        move_robber_action = Action(agent_color, ActionType.MOVE_ROBBER, tile_loc)
        actions_to_indices[move_robber_action] = idx
        indices_to_actions[idx] = move_robber_action
        idx += 1

    # --- Play development cards ---
    for card in PLAYABLE_DEV_CARDS:
        # --- TODO(ryancao): Randomly choose a resource for monopoly and year of plenty! ---
        play_card_action = Action(agent_color, card, None)
        actions_to_indices[play_card_action] = idx
        indices_to_actions[idx] = play_card_action
        idx += 1

    # --- Buy development card ---
    buy_card_action = Action(agent_color, ActionType.BUY_DEVELOPMENT_CARD, None)
    actions_to_indices[buy_card_action] = idx
    indices_to_actions[idx] = buy_card_action
    idx += 1

    # --- End turn ---
    end_turn_action = Action(agent_color, ActionType.END_TURN, None)
    actions_to_indices[end_turn_action] = idx
    indices_to_actions[idx] = end_turn_action
    idx += 1

    return actions_to_indices, indices_to_actions

# --- Cached ---
ACTIONS_TO_INDICES, INDICES_TO_ACTIONS = get_action_mapping()
# for idx in range(len(INDICES_TO_ACTIONS)):
#     print(idx, INDICES_TO_ACTIONS[idx])

def get_immutable_board_state():
    """
    Returns the immutable component of the game board.
    """

    # --- Dummy setup ---
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game = Game(players)
    map = game.state.board.map

    board_state = {
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
    }

    # --- Grab resource types (this can be cached, or not used at all) ---
    for (coord, tile) in map.tiles.items():
        offset_coord = cube_to_offset(coord)
        offset_coord = tuple(int(x) + 3 for x in offset_coord) # --- [-3, 3] to [0, 6] ---

        if type(tile) == Port:
            pass # We are not doing trading for now

        elif type(tile) == Tile:
            # --- Dice numbers ---
            board_state['grid_dice_nums'][offset_coord[0]][offset_coord[1]] = tile.number

            # --- Resources ---
            if tile.resource == Resource.WOOD:
                board_state['grid_wood_locs'][offset_coord[0]][offset_coord[1]] = 1
            elif tile.resource == Resource.BRICK:
                board_state['grid_brick_locs'][offset_coord[0]][offset_coord[1]] = 1
            elif tile.resource == Resource.SHEEP:
                board_state['grid_sheep_locs'][offset_coord[0]][offset_coord[1]] = 1
            elif tile.resource == Resource.WHEAT:
                board_state['grid_wheat_locs'][offset_coord[0]][offset_coord[1]] = 1
            elif tile.resource == Resource.ORE:
                board_state['grid_ore_locs'][offset_coord[0]][offset_coord[1]] = 1

        elif type(tile) == Water:
            pass # Nothing useful here

        else:
            print(f'No such tile type: {tile} with type {type(tile)}')

    return board_state

    # --- Cached ---
    IMMUTABLE_BOARD_STATE = get_immutable_board_state()