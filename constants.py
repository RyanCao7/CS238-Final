from catanatron.models.board import get_edges
from catanatron.models.map import NUM_NODES, NUM_EDGES, NUM_TILES
import torch
import models

# --- opts.py ---
REPLAY_BUFFER_CAPACITY = 20000
DEFAULT_DQN_LR = 1e-3
DEFAULT_DQN_EPISODES = 10000
DEFAULT_DQN_OPTIM = 'adam'
TRAIN_EVERY_NUM_TIMESTEPS = 100
UPDATE_DQN_EVERY_NUM_TIMESTEPS = 500
DQN_MODEL_TYPES = {
    'Catan_Feedforward_DQN': models.Catan_Feedforward_DQN
}
DEFAULT_DQN_MODEL = 'Catan_Feedforward_DQN'

# --- models.py ---
DEFAULT_BOARD_SIZE = 7
BUY_DEVELOPMENT_CARD_ACTIONS = 2
PLAY_DEVELOPMENT_CARD_ACTIONS = 2
# Settlements, cities, roads, robber moving, and (None)
TOTAL_DQN_ACTIONS = NUM_NODES * 2 + NUM_EDGES + NUM_TILES + 1

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

# --- Cached ---
EDGES_TO_INDICES, INDICES_TO_EDGES = get_edge_mapping()

# --- Cached ---
IMMUTABLE_BOARD_STATE = get_immutable_board_state()

# --- Device ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_edge_mapping():
    """
    Maps from edge coordinate tuples to index and vice versa.
    """
    edges_to_indices = dict()
    indices_to_edges = dict()
    all_edges_sorted = sorted(get_edges(), key=lambda x: x[0] * 100 + x[1])
    for (idx, edge) in enumerate(all_edges_sorted):
        edge = (min(edge), max(edge))
        edges_to_indices[edge] = idx
        indices_to_edges[idx] = edge
    return edges_to_indices, indices_to_edges

def get_immutable_board_state():
    """
    Returns the immutable component of the game board.
    """

    # --- Dummy setup ---
    players = [
        MyPlayer(Color.RED),
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
        offset_coord = catanatron.models.coordinate_system.cube_to_offset(coord)
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