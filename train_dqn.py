import os
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color, HumanPlayer

import agents
import constants
import models
import opts
import state_utils
import replay_buffer

def train_dqn_agent(dqn_agent, args):
    """
    Trains the given DQN agent with the given args.
    """
    print(('-' * 15) + ' BEGIN TRAINING ' + ('-' * 15))

    # --- Create environment: 1v1 setting for now ---
    players = [
        dqn_agent,
        RandomPlayer(Color.RED),
    ]
    game = Game(players)
    num_episodes = 0

    # --- Main data collection loop ---
    for timestep in range(args.train_num_steps):

        # --- Play the game by a single tick ---
        game.play_tick(action_callbacks=[], decide_fn=None)

        # --- Run one iteration of model optimization ---
        if timestep % args.train_every_num_timesteps == 0:
            dqn_agent.optimize_one_step()

        # --- See if we need to log or visualize ---
        if timestep % args.print_every == 0:
            dqn_agent.print_stats(timestep)
        if timestep % args.save_every == 0:
            dqn_agent.save_stats()
            dqn_agent.save_model(timestep, num_episodes)
            dqn_agent.plot_stats(timestep, num_episodes)

        # --- End of an episode; log and reset the game ---
        if game.winning_color() is not None:
            num_episodes += 1
            dqn_agent_num_vps = game.state.player_state['P0_VICTORY_POINTS']
            dqn_agent.train_stats['VPs'].append(dqn_agent_num_vps)
            game = Game(players)

    return dqn_agent


def main():
    # --- Args ---
    args = opts.get_train_dqn_args()
    print('\n' + '-' * 30 + ' Args ' + '-' * 30)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()

    # --- Model and viz save dir ---
    model_save_dir = constants.get_model_save_dir(args.model_type, args.model_name)
    viz_save_dir = constants.get_viz_save_dir(args.model_type, args.model_name)
    if os.path.isdir(model_save_dir):
        raise RuntimeError(f'Error: {model_save_dir} already exists! Exiting...')
    elif os.path.isdir(viz_save_dir):
        raise RuntimeError(f'Error: {viz_save_dir} already exists! Exiting...')
    else:
        print(f'--> Creating directory {model_save_dir}...')
        os.makedirs(model_save_dir)
        print(f'--> Creating directory {viz_save_dir}...')
        os.makedirs(viz_save_dir)
    print('Done!\n')

    # --- Setup agent ---
    print('----> Setting up DQN agent...\n')
    dqn_agent = agents.DQN_Agent(args)
    print('Done!\n')

    # --- Train ---
    dqn_agent = train_dqn_agent(dqn_agent, args)

    # --- Save model ---
    model_save_path = os.path.join(model_save_dir, 'final_model.pth')
    print(f'Done training! Saving model to {model_save_path}...')
    torch.save(model.state_dict(), model_save_path)

    # --- Plot final round of loss/iou metrics ---
    # viz_path = constants.get_classification_viz_save_dir(args.model_type, args.model_name)
    # viz_utils.plot_losses_ious(train_losses, train_ious, viz_path, prefix='train')
    # viz_utils.plot_losses_ious(val_losses, val_ious, viz_path, prefix='val')

    # --- Do a final train stats save ---
    # save_train_stats(train_losses, train_ious, val_losses, val_ious, args)


if __name__ == '__main__':
    main()