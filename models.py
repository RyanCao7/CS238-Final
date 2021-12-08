import torch
import torch.nn as nn
import torch.nn.functional as F
from catanatron.models.map import NUM_NODES, NUM_EDGES, NUM_TILES
from catanatron.models.enums import Resource

import constants

NUM_RESOURCES = len(Resource)

class Catan_Feedforward_DQN(nn.Module):
    """
    Multi-input feedforward DQN. Very simple.
    """

    def conv2d_size_out(self, size, kernel_size = 3, stride = 1):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def __init__(self, board_size, total_num_actions):
        super(Catan_Feedforward_DQN, self).__init__()

        # --- Robber repr is very straightforward ---
        self.grid_robber_head_1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.grid_robber_head_bn = nn.BatchNorm2d(8)

        # --- Roads will need more processing ---
        self.road_color_head_1 = nn.Linear(NUM_EDGES, NUM_EDGES * 2)
        self.road_color_head_2 = nn.Linear(NUM_EDGES * 2, NUM_EDGES * 2)
        self.road_color_head_bn_1 = nn.BatchNorm1d(1)
        self.road_color_head_bn_2 = nn.BatchNorm1d(1)

        # --- Settlements and cities ---
        self.settlements_head = nn.Linear(NUM_NODES, NUM_NODES)
        self.cities_head = nn.Linear(NUM_NODES, NUM_NODES)
        self.settlements_cities_head = nn.Linear(NUM_NODES * 2, NUM_NODES * 2)
        self.settlements_head_bn = nn.BatchNorm1d(1)
        self.cities_head_bn = nn.BatchNorm1d(1)
        self.settlements_cities_head_bn = nn.BatchNorm1d(1)

        # --- Resources ---
        self.player_resources_head = nn.Linear(NUM_RESOURCES, NUM_RESOURCES * 4)
        self.opponent_resources_head = nn.Linear(NUM_RESOURCES, NUM_RESOURCES * 4)
        self.combined_resources_head = nn.Linear(NUM_RESOURCES * 8, NUM_RESOURCES * 8)
        self.player_resources_head_bn = nn.BatchNorm1d(1)
        self.opponent_resources_head_bn = nn.BatchNorm1d(1)
        self.combined_resources_head_bn = nn.BatchNorm1d(1)

        # --- Combining ---
        combined_dim = (self.conv2d_size_out(board_size, kernel_size=3, stride=1) ** 2) * 8
        combined_dim += NUM_EDGES * 2 + NUM_NODES * 2 + NUM_RESOURCES * 8
        intermediate_dim = NUM_EDGES * 2 + NUM_NODES * 2 + NUM_RESOURCES * 8
        self.combined_head = nn.Linear(combined_dim, intermediate_dim)
        self.combined_head_bn = nn.BatchNorm1d(1)
        self.output_head = nn.Linear(intermediate_dim, total_num_actions)


    def forward(self, batched_board_state):
        """
        Assumption: batched_board_state is a dict of pre-batched state variables.
        """
        # --- Inputs ---
        robber_input = batched_board_state['grid_robber_loc'].to(constants.DEVICE)
        road_input = batched_board_state['road_colors'].to(constants.DEVICE)
        settlement_input = batched_board_state['settlement_locs'].to(constants.DEVICE)
        city_input = batched_board_state['city_locs'].to(constants.DEVICE)
        player_resources_input = batched_board_state['player_resources'].to(constants.DEVICE)
        opponent_resources_input = batched_board_state['opponent_resources'].to(constants.DEVICE)

        # --- Robber ---
        robber_input = F.relu(self.grid_robber_head_bn(self.grid_robber_head_1(robber_input)))
        # --- Flatten ---
        robber_input = robber_input.view(robber_input.size(0), 1, -1)

        # --- Roads ---
        road_input = F.relu(self.road_color_head_bn_1(self.road_color_head_1(road_input)))
        road_input = F.relu(self.road_color_head_bn_2(self.road_color_head_2(road_input)))

        # --- Settlements and cities ---
        settlement_input = F.relu(self.settlements_head_bn(self.settlements_head(settlement_input)))
        city_input = F.relu(self.cities_head_bn(self.cities_head(city_input)))
        settlement_city_input = torch.cat([settlement_input, city_input], axis=-1)
        settlement_city_input = F.relu(self.settlements_cities_head_bn(self.settlements_cities_head(settlement_city_input)))

        # --- Resources ---
        player_resources_input = F.relu(self.player_resources_head_bn(self.player_resources_head(player_resources_input)))
        opponent_resources_input = F.relu(self.opponent_resources_head_bn(self.opponent_resources_head(opponent_resources_input)))
        combined_resources_input = torch.cat([player_resources_input, opponent_resources_input], axis=-1)

        # --- Combining ---
        combined_input = torch.cat([robber_input, road_input, settlement_city_input, combined_resources_input], axis=-1)
        combined_input = F.relu(self.combined_head_bn(self.combined_head(combined_input)))

        # --- Output ---
        output = self.output_head(combined_input)
        return output