from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType
from state_utils import extract_board_state
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import constants
import models
import replay_buffer

class DQN_Agent(Player):
    """
    Simple DQN agent which takes advantage of the DQN network.
    """

    def __init__(self, args, color=constants.AGENT_COLOR, is_bot=True):
        """
        args is from opts.get_train_dqn_args()
        """
        super(DQN_Agent, self).__init__(color, is_bot)
        self.args = args

        print('--> Setting up policy/target models...')
        if args.model_type not in constants.DQN_MODEL_TYPES:
            raise RuntimeError(f'Error: {args.model_type} is not one of {list(constants.DQN_MODEL_TYPES.keys())}')
        self.policy_dqn = constants.DQN_MODEL_TYPES[args.model_type](constants.DEFAULT_BOARD_SIZE, constants.TOTAL_DQN_ACTIONS).to(constants.DEVICE)
        self.target_dqn = constants.DQN_MODEL_TYPES[args.model_type](constants.DEFAULT_BOARD_SIZE, constants.TOTAL_DQN_ACTIONS).to(constants.DEVICE)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()
        print('Done!\n')

        print('--> Setting up optimizer/criterion...')
        self.optim = optim.Adam(self.policy_dqn.parameters(), lr=args.lr)
        self.criterion = nn.SmoothL1Loss()
        print('Done!\n')

        print('--> Setting up replay buffer...')
        self.replay_buffer = replay_buffer.ReplayMemory(constants.REPLAY_BUFFER_CAPACITY)
        print('Done!\n')

        print('--> Setting up train stats...')
        self.train_stats = {
            # --- Note that `losses` is every t steps, while `returns` and `VPs` is every episode ---
            'losses': list(),
            'returns': list(),
            'VPs': list(),
        }
        print('Done!\n')

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.
        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """

        # --- Edge case: roll action ---
        if playable_actions[0] == (self.color, ActionType.ROLL, None):
            print('Rolling')
            return playable_actions[0]

        game_state_dict = extract_board_state(game, playable_actions, agent_color=self.color)

        # --- Adding dummy "batch" dim and moving to GPU ---
        for (key, value) in game_state_dict.items():
            game_state_dict[key] = torch.unsqueeze(value, dim=0).to(constants.DEVICE)
            # print(key)
            # print(game_state_dict[key].shape)
            # print('-' * 30)

        # --- Computing forward pass ---
        with torch.no_grad():
            action_vector = self.policy_dqn(game_state_dict)
            # print('-' * 30)
            # print(action_vector)
            # print(action_vector.shape)
            action_vector = action_vector * game_state_dict['action_mask']
            # print('-' * 15)
            # print(game_state_dict['action_mask'])
            # print(action_vector)
            # print(action_vector.shape)
            # print()

        return random.choice(playable_actions)

    def save_stats(self):
        save_path = constants.get_model_save_dir(self.args.model_type, self.args.model_name)
        save_path = os.path.join(save_path, 'train_stats.json')
        print(f'--> Saving train stats to {save_path}...')
        with open(save_path, 'w') as f:
            json.dump(self.train_stats, f)
        print('Done!\n')
    
    def save_model(self, timestep, num_episodes):
        save_path = constants.get_model_save_dir(self.args.model_type, self.args.model_name)
        save_path = os.path.join(save_path, 'dqn_model_timestep_{}.pth')
        print(f'--> Saving model DQN weights to {save_path}...')
        with open(save_path, 'wb') as f:
            torch.save(self.policy_dqn.state_dict(), f)
        print('Done!\n')

    def plot_stats(self, timestep, num_episodes):
        pass # TODO(ryancao)!

    def reset_state(self):
        """Hook for resetting state between games"""
        pass