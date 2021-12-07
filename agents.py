from catanatron.models.player import Player
from state_utils import extract_board_state
import numpy as np
import torch

import constants
import models

class DQN_Agent(Player):
    """
    Simple DQN agent which takes advantage of the DQN network.
    """

    def __init__(self, color, is_bot=True):
        self.color = color
        self.is_bot = is_bot
        self.dqn = models.Catan_Feedforward_DQN(board_size, total_num_actions)

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.
        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """
        game_state_dict = extract_board_state(game)

        # --- Adding dummy "batch" dim and moving to GPU ---
        for (key, value) in game_state_dict.items():
            game_state_dict[key] = torch.unsqueeze(value, dim=0).to(constants.DEVICE)
        
        # --- Computing forward pass ---


    def reset_state(self):
        """Hook for resetting state between games"""
        pass