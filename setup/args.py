import argparse


def initialize_args():
    parser = argparse.ArgumentParser(description='SpaceTime arguments')
    
    # Model
    parser.add_argument('--model', type=str, default='hedgehog')
    parser.add_argument('--embedding_config', type=str, default='embedding/repeat')
    parser.add_argument('--preprocess_config', type=str, default='preprocess/default')
    parser.add_argument('--encoder_config', type=str, default='encoder/default')
    parser.add_argument('--decoder_config', type=str, default='decoder/default')
    parser.add_argument('--output_config', type=str, default='output/default')
    
    # Model config arguments
    parser.add_argument('--n_blocks', type=int, default=None)  # Only update encoder blocks
    parser.add_argument('--n_kernels', type=int, default=None)
    parser.add_argument('--n_heads', type=int, default=None)
    
    parser.add_argument('--model_dim', type=int, default=None)
    parser.add_argument('--input_dim', type=int, default=1, 
                        help='Input dimensions. Updated based on dataset.')
    parser.add_argument('--kernel_dim', type=int, default=None)
    # parser.add_argument('--head_dim', type=int, default=None)
    parser.add_argument('--activation', type=str, choices=['gelu', 'relu'])
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--layernorm', action='store_true', default=None)

    # Kernel-specific args
    parser.add_argument('--kernel_init', type=str, default=None)
    # Skip Connections
    parser.add_argument('--skip_conv', action='store_true', default=None)
    parser.add_argument('--skip_kernel', action='store_true', default=None)
    # MLP
    parser.add_argument('--mlp_n_layers', type=int, default=None)
    parser.add_argument('--mlp_n_activations', type=int, default=None)
    parser.add_argument('--mlp_preactivation', type=int, default=None)
    parser.add_argument('--skip_mlp', action='store_true', default=None)
    
    # Data
    parser.add_argument('--dataset', type=str, default='etth1')
    parser.add_argument('--dataset_type', type=str, default='')
    parser.add_argument('--variant', type=int, default=None)
    parser.add_argument('--trainer', type=str, default='default')
    parser.add_argument('--loader', type=str, default='default')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='./data')
    ## Informer-specific
    parser.add_argument('--no_scale', type=bool, action='store_true', default=False)
    parser.add_argument('--inverse', type=bool, action='store_true', default=False)
    
    # Prediction Task
    parser.add_argument('--lag', type=float, default=1, 
                        help="Number of samples included in input. If 0, then can change?") 
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--loss', type=str, default='rmse',
                        choices=['rmse', 'mse', 'mae', 'cross_entropy'])
    
    # Training
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='timm_cosine',
                        choices=['none', 'plateau', 'timm_cosine'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--early_stopping_epochs', type=int, default=50)
    parser.add_argument('--val_metric', type=str, default='loss')
    
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.0)
    
    # Saving + logging
    parser.add_argument('--log_epoch', type=int, default=5)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_entity', type=str, default='mzhang')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    # Misc.
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--no_pin_memory', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--replicate', type=int, default=1)  # hack for waterbirds
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args