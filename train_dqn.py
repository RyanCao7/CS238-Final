import constants
import models
import agents
import state_utils
import replay_buffer


def main():
    # --- Args ---
    args = opts.get_classify_train_args()
    print('\n' + '-' * 30 + ' Args ' + '-' * 30)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print()
    
    # --- Model and viz save dir ---
    model_save_dir = constants.get_classification_model_save_dir(args.model_type, args.model_name)
    viz_save_dir = constants.get_classification_viz_save_dir(args.model_type, args.model_name)
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

    # --- Setup dataset ---
    print('--> Setting up dataset...')
    train_dataset = dataset.CXR_Classification_Dataset(mode='train')
    val_dataset = dataset.CXR_Classification_Dataset(mode='val')
    print('Done!\n')

    # --- Dataloaders ---
    print('--> Setting up dataloaders...')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4)
    print('Done!\n')

    # --- Setup model ---
    # TODO(ryancao): Actually pull the ResNet model! ---
    print('--> Setting up model...')
    model = models.get_model(args.model_type)
    torch.cuda.set_device(constants.GPU)
    model = model.cuda(constants.GPU)
    print('Done!\n')

    # --- Optimizer ---
    print('--> Setting up optimizer/criterion...')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # --- Loss fn ---
    pos_weight = torch.Tensor(constants.get_indexes_to_weights())
    print(f'Here are the weights: {pos_weight}')
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda(constants.GPU)
    print('Done!\n')

    # --- Train ---
    train_losses, train_ious, val_losses, val_ious =\
        train(args, model, train_dataloader, val_dataloader, criterion, opt)

    # --- Save model ---
    model_save_path = os.path.join(model_save_dir, 'final_model.pth')
    print(f'Done training! Saving model to {model_save_path}...')
    torch.save(model.state_dict(), model_save_path)
    
    # --- Plot final round of loss/iou metrics ---
    viz_path = constants.get_classification_viz_save_dir(args.model_type, args.model_name)
    viz_utils.plot_losses_ious(train_losses, train_ious, viz_path, prefix='train')
    viz_utils.plot_losses_ious(val_losses, val_ious, viz_path, prefix='val')
    
    # --- Do a final train stats save ---
    save_train_stats(train_losses, train_ious, val_losses, val_ious, args)


if __name__ == '__main__':
    main()