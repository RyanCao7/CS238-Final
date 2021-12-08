import argparse
import constants

def get_train_dqn_args():
    """
    Args for `train_dqn.py`.
    """
    parser = argparse.ArgumentParser()

    # --- Replay Buffer ---
    parser.add_argument('--replay-buffer-len', type=int, 
                        help=f'Replay buffer max capacity (default: {constants.REPLAY_BUFFER_CAPACITY})', 
                        default=constants.REPLAY_BUFFER_CAPACITY)

    # --- Model ---
    parser.add_argument('--model-type', type=str,
                        help=f'Model type. Choices are {constants.DQN_MODEL_TYPES} (default: {constants.DEFAULT_DQN_MODEL}).',
                        default=constants.DEFAULT_DQN_MODEL)

    # --- Hyperparams ---
    parser.add_argument('--lr', type=float, default=constants.DEFAULT_DQN_LR)
    parser.add_argument('--train-num-steps', type=int, default=constants.DEFAULT_DQN_NUM_STEPS)
    parser.add_argument('--optimizer', type=str, default=constants.DEFAULT_DQN_OPTIM)
    parser.add_argument('--train-batch-size', type=int, default=constants.DEFAULT_BATCH_SIZE)
    parser.add_argument('--gamma', type=float, default=constants.GAMMA)
    parser.add_argument('--epsilon', type=float, default=constants.EPSILON)
    parser.add_argument('--train-every-num-timesteps', type=int, 
                        default=constants.TRAIN_EVERY_NUM_TIMESTEPS,
                        help=f'Train every t timesteps (default: {constants.TRAIN_EVERY_NUM_TIMESTEPS}).')
    parser.add_argument('--update-target-dqn-every-num-timesteps', type=int,
                        default=constants.UPDATE_DQN_EVERY_NUM_TIMESTEPS,
                        help=f'Update target DQN weights every t timesteps. (default: {constants.UPDATE_DQN_EVERY_NUM_TIMESTEPS}).')

    # --- Save dir ---
    parser.add_argument('--model-name', type=str, required=True, help='Where to save model weights')
    
    # --- Other ---
    parser.add_argument('--eval-every', type=int, default=1000, help='Eval every n timesteps.')
    parser.add_argument('--save-every', type=int, default=3000, help='Save model every n timesteps.')
    parser.add_argument('--print-every', type=int, default=500, help='Print every n timesteps.')
    
    args = parser.parse_args()
    return args