from collections import deque
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType
from state_utils import extract_board_state
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
        self.loss_cache = deque([], maxlen=1000)
        self.train_stats = {
            # --- Note that `losses` is every t steps, while `returns` and `VPs` is every episode ---
            'smoothed_losses': list(),
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
            return playable_actions[0]

        # --- Edge case: discard action ---
        # TODO(ryancao): Actually let the Q-network learn this set of actions too
        if playable_actions[0][1] == ActionType.DISCARD:
            return random.choice(playable_actions)

        # --- Filter all maritime trade actions ---
        playable_actions = list(filter(lambda x: x[1] != ActionType.MARITIME_TRADE, playable_actions))

        # --- Fix other actions to be generic ---
        robber_stealing_potential_colors = dict()
        year_of_plenty_potential_options = dict()
        play_monopoly_potential_options = dict()
        for idx in range(len(playable_actions)):
            action = playable_actions[idx]

            # --- Fix robber-moving action ---
            # --- Repr: (Color, ActionType.MOVE_ROBBER, (coord, color, None))
            # ---   --> (Color, ActionType.MOVE_ROBBER, coord)
            # --- We also keep track of a mapping from the latter to all initial actions (with colors)
            if action[1] == ActionType.MOVE_ROBBER:
                simplified_robber_action = Action(action[0], action[1], (action[2][0]))
                playable_actions[idx] = simplified_robber_action
                if simplified_robber_action not in robber_stealing_potential_colors:
                    robber_stealing_potential_colors[simplified_robber_action] = list()
                robber_stealing_potential_colors[simplified_robber_action].append(action)

            # --- Fix year-of-plenty action ---
            if action[1] == ActionType.PLAY_YEAR_OF_PLENTY:
                simplified_year_of_plenty_action = Action(action[0], action[1], None)
                playable_actions[idx] = simplified_year_of_plenty_action
                if simplified_year_of_plenty_action not in year_of_plenty_potential_options:
                    year_of_plenty_potential_options[simplified_year_of_plenty_action] = list()
                year_of_plenty_potential_options[simplified_year_of_plenty_action].append(action)
            
            # --- Fix monnopoly action ---
            if action[1] == ActionType.PLAY_MONOPOLY:
                simplified_monopoly_action = Action(action[0], action[1], None)
                playable_actions[idx] = simplified_monopoly_action
                if simplified_monopoly_action not in play_monopoly_potential_options:
                    play_monopoly_potential_options[simplified_monopoly_action] = list()
                play_monopoly_potential_options[simplified_monopoly_action].append(action)

        # --- Get the actual game state ---
        game_state_dict = extract_board_state(game, playable_actions, agent_color=self.color)

        # --- Adding dummy "batch" dim and moving to GPU ---
        for (key, value) in game_state_dict.items():
            game_state_dict[key] = torch.unsqueeze(value, dim=0).to(constants.DEVICE)
            # print(key)
            # print(game_state_dict[key].shape)
            # print('-' * 30)

        # --- Computing forward pass ---
        with torch.no_grad():
            action_q_values = torch.squeeze(self.policy_dqn(game_state_dict)) # (223,)
            # --- Softmax, then multiply by mask. Technically not a probs distribution but that's okay. ---
            action_probs = torch.softmax(action_q_values, dim=0) * torch.squeeze(game_state_dict['action_mask'])
            action_idx = torch.argmax(action_probs).item()
            action = constants.INDICES_TO_ACTIONS[action_idx]

        # --- Finally, determine the action (sampling from options if needed) ---
        # action = random.choice(playable_actions)
        if action in robber_stealing_potential_colors:
            action = random.choice(robber_stealing_potential_colors[action])
        elif action in year_of_plenty_potential_options:
            action = random.choice(year_of_plenty_potential_options[action])
        elif action in play_monopoly_potential_options:
            action = random.choice(play_monopoly_potential_options[action])
        return action

    def save_stats(self):
        save_path = constants.get_model_save_dir(self.args.model_type, self.args.model_name)
        save_path = os.path.join(save_path, 'train_stats.json')
        print(f'--> Saving train stats to {save_path}...')
        with open(save_path, 'w') as f:
            json.dump(self.train_stats, f)
        print('Done!\n')

    def save_model(self, timestep, num_episodes):
        save_path = constants.get_model_save_dir(self.args.model_type, self.args.model_name)
        save_path = os.path.join(save_path, f'dqn_model_timestep_{timestep}.pth')
        print(f'--> Saving model DQN weights to {save_path}...')
        with open(save_path, 'wb') as f:
            torch.save(self.policy_dqn.state_dict(), f)
        print('Done!\n')

    def plot_stats(self, timestep, num_episodes):
        pass # TODO(ryancao)!

    def reset_state(self):
        """Hook for resetting state between games"""
        pass