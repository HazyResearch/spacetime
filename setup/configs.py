"""
Load default configs
"""
from os.path import join
from omegaconf import OmegaConf

# For loading data
from dataloaders import get_data_module

# -----------------------
# Everything except model
# -----------------------
def load_main_config(args, config_dir='./configs'):
    configs = {'dataset':   get_dataset_config(args, config_dir),
               'loader':    get_dataloader_config(args, config_dir),
               'optimizer': get_optimizer_config(args, config_dir),
               'scheduler': get_scheduler_config(args, config_dir)}
    return configs

# ---------------
# SpaceTime model
# ---------------
def load_model_config(config, config_dir='./configs/model', args=None):
    for k in ['embedding_config', 'preprocess_config', 'encoder_config', 
              'decoder_config', 'output_config']:
        _config = OmegaConf.load(join(config_dir, f'{config[k]}.yaml'))
        if k == 'encoder_config' or k == 'decoder_config':
            for ix, block_config in enumerate(_config['blocks']):
                # Load SSM kernel configs
                c_path = join(config_dir, f"{block_config['kernel_config']}.yaml")
                block_config['conv'] = OmegaConf.load(c_path)
                # Load MLP configs
                c_path = join(config_dir, f"{block_config['mlp_config']}.yaml")
                block_config['mlp'] = OmegaConf.load(c_path)
        config[k] = _config
        
    config = update_embedding_config_from_args(config, args)
    config = update_block_config_from_args(config, args)
    config.output_config.mlp = update_output_config_from_args(
        config.output_config.mlp, args)
    config.output_config.input_dim = config.output_config.mlp.kwargs.input_dim
    config.output_config.output_dim = config.output_config.mlp.kwargs.output_dim        
    return config

# ----
# Data
# ----
def get_dataset_config(args, config_dir='./configs'):
    get_data_module(args)  # Initialize args.dataset_type
    fpath = join(config_dir, 'datasets', args.dataset_type, f'{args.dataset}.yaml')
    config = OmegaConf.load(fpath)
    config = update_dataset_config_from_args(config, args)
    return config


def get_dataloader_config(args, config_dir='./configs'):
    get_data_module(args)  # Initialize args.dataset_type
    fpath = join(config_dir, 'loader', f'{args.dataset_type}.yaml')
    config = OmegaConf.load(fpath)
    
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.pin_memory = not args.no_pin_memory
    return config

# ---------
# Optimizer
# ---------
def get_optimizer_config(args, config_dir='./configs'):
    config = OmegaConf.load(
        join(config_dir, 'optimizer', f'{args.optimizer}.yaml'))
    if args.lr is not None:
        config.lr = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.optimizer == 'sgd' and args.momentum is not None:
        config.momentum = args.momentum
    return config


def get_scheduler_config(args, config_dir='./configs'):
    config = OmegaConf.load(
        join(config_dir, 'scheduler', f'{args.scheduler}.yaml'))
    if (config.scheduler._name_ == 'plateau' and args.val_metric == 'acc'):
        config.scheduler.mode = 'max'
    return config


# ---------------------------------
# Update configs from argparse args
# ---------------------------------

# Data + Task
def update_dataset_config_from_args(config, args):
    if args.dataset_type == 'informer':
        config.size = [args.lag, args.horizon, args.horizon]
        config.variant = args.variant
        config.scale = not args.no_scale
        config.inverse = args.inverse
    else:
        pass
    return config

# Model
def update_embedding_config_from_args(config, args):
    if args.model_dim is not None:
        config.embedding_config['embedding_dim'] = args.model_dim
        
    if args.n_heads is not None:
        config.embedding_config['n_heads'] = args.n_heads
    elif config.embedding_config['n_heads'] is not None:
        args.n_heads = config.embedding_config['n_heads']
    else:
        args.n_heads = args.model_dim // (args.input_dim * args.n_kernels)
        config.embedding_config['n_heads'] = args.n_heads
    
    if args.n_kernels is not None:
        config.embedding_config['n_kernels'] = args.n_kernels
    elif config.embedding_config['n_kernels'] is not None:
        args.n_kernels = config.embedding_config['n_kernels']
    else:
        args.n_kernels = args.model_dim // (args.input_dim * args.n_heads)
        config.embedding_config['n_kernels'] = args.n_kernels
        
    args.model_dim = args.n_heads * args.n_kernels
        
    return config


def update_block_config_from_args(config, args):
    # Update encoder
    # For now args only update encoder
    # if args.n_encoder_blocks is not None:
    #     # Copy first block n_blocks times, because last one may do pooling
    #     n_blocks = args.n_encoder_blocks - 1  
    #     last_block = config.encoder_config['blocks'][-1]
    #     config.encoder_config['blocks'] = [config.encoder_config['blocks']] * n_blocks
    #     config.encoder_config['blocks'].append(last_block)
    # else:
    #     args.n_encoder_blocks = len(config.encoder_config['blocks'])
    
    # Update decoder
    # For now args only update decoder
    encoder_block = config.encoder_config['blocks'][0]
    decoder_block.kernel = update_kernel_config_from_args(decoder_block.kernel, args)
    decoder_block.mlp = update_mlp_config_from_args(decoder_block.mlp, args)
    
    n_blocks = len(config.decoder_config['blocks'])
    if args.n_blocks is not None:
        n_blocks = args.n_decoder_blocks
    else:
        args.n_blocks = len(config.decoder_config['blocks']) + 1
    config.decoder_config['blocks'] = [decoder_block] * n_blocks
    return config


def _update_head_dim(config, args):
    if args.head_dim is None:
        if 'head_dim' in config.keys():
            head_dim = config['head_dim']
        elif 'head_dim' in config.kwargs:
            head_dim = config.kwargs['head_dim']
        else:
            head_dim = None
    else:
        head_dim = args.head_dim
    return head_dim


def _update_n_heads(config, args):
    if args.n_heads is None:
        if 'n_heads' in config.kwargs:
            n_heads = config.kwargs['n_heads']
        else:
            n_heads = None
    else:
        n_heads = args.n_heads
    return n_heads


def update_conv_attn_config_from_args(config, args):
    head_dim = _update_head_dim(config, args)
    # n_heads  = _update_n_heads(config, args)
    kwargs = {
        'input_dim': args.embedding_dim,
        'head_dim': head_dim,
        # 'model_dim': args.n_kernels * n_heads * head_dim,
        'output_dim': args.embedding_dim,
        'bidirectional': args.bidirectional,
        'norm_qk': args.norm_qk,
        'scale_qk': args.scale_qk,
        'softmax_qk': args.softmax_qk,
        'affine_qk': args.affine_qk,
        'context_len': args.context_len,
        'dropout': args.dropout,
        'layernorm': args.layernorm,
        'skip_connection': args.skip_conv,
        # 'linear_bias': false,
    }
    for k, v in kwargs.items():
        if v is not None:
            try:
                config[k] = v
            except Exception as e:
                print(e)
                # pass
        else:
            if k in config:
                setattr(args, k, config[k])
#             try:
#                 setattr(args, k, config[k])
#             except Exception as e:
#                 print(e)
                
    return config


def update_conv_config_from_args(config, args):
    head_dim = _update_head_dim(config, args)
    n_heads  = _update_n_heads(config, args)
    kwargs = {
        # 'input_dim': args.n_kernels * args.n_heads * head_dim,
        # 'model_dim': args.n_kernels * n_heads * head_dim,
        # 'output_dim': args.n_kernels * n_heads * head_dim,
        # 'n_heads': args.n_heads,
        'bidirectional': args.bidirectional,
        'norm_qk': args.norm_qk,
        'scale_qk': args.scale_qk,
        'softmax_qk': args.softmax_qk,
        'context_len': args.context_len,
        'dropout': args.dropout,
        'layernorm': args.layernorm,
        'skip_connection': args.skip_conv,
        # 'linear_bias': false,
    }
    for k, v in kwargs.items():
        if v is not None:
            try:
                config[k] = v
            except Exception as e:
                print(e)
                breakpoint()
        else:
            try:
                setattr(args, k, config[k])
            except Exception as e:
                print(e)
                breakpoint()
    return config
    
    
def update_kernel_config_from_args(config, args):
    head_dim = _update_head_dim(config, args)
    # n_heads  = _update_n_heads(config, args)

    if config.method == 'shift' or config.method == 'multihead_shift':
        kwargs = {
            # 'input_dim': args.n_kernels * n_heads * head_dim,
            'input_dim': args.embedding_dim,
            'n_kernels': args.n_kernels,
            'kernel_dim': args.kernel_dim,
            # 'n_heads': args.n_heads,
            'head_dim': args.head_dim,
            # kernel_weights: None,
            # kernel_train: True,
            'dilations': args.dilations,
            'kernel_init': args.kernel_init,
            'skip_connection': args.skip_kernel,
        }
        for k, v in kwargs.items():
            if v is not None:
                config.kwargs[k] = v
            else:
                try:
                    setattr(args, k, config.kwargs[k])
                except Exception as e:
                    print(e)
                    assert k not in config.kwargs
    return config


def update_mlp_config_from_args(config, args, input_dims=True, output_dims=True):
    head_dim = _update_head_dim(config, args)
    # n_heads  = _update_n_heads(config, args)
        
    if config.method == 'mlp':
        kwargs = {
            # 'input_dim': args.n_kernels * n_heads * head_dim if input_dims else args.embedding_dim,
            'input_dim': args.embedding_dim,
            # 'output_dim': args.n_kernels * n_heads * head_dim if output_dims else None,
            'output_dim': args.embedding_dim if output_dims else None,
            'activation': args.activation,
            'dropout': args.dropout,
            'layernorm': args.layernorm,
            'n_layers': args.mlp_n_layers,
            'n_activations': args.mlp_n_activations,
            'pre_activation': args.mlp_preactivation,
            'skip_connection': args.skip_mlp,
        }
        
        for k, v in kwargs.items():
            if v is not None:
                # if not input_dims and k == 'input_dim':
                #     pass
                if not input_dims and k == 'skip_connection':
                    pass
                elif not output_dims and k == 'skip_connection':
                    pass
                else:
                    config.kwargs[k] = v
            elif input_dims and output_dims:
                setattr(args, k, config.kwargs[k])
    return config


def update_encoder_config_from_args(config, args):
    if config.n_blocks > 0:
        config = update_block_config_from_args(config, args)        
    return config


def update_decoder_config_from_args(config, args):
    if config.n_blocks > 0:
        config = update_block_config_from_args(config, args)        
    return config


def update_output_config_from_args(config, args):
    # Some things not handled yet
    if config.method == 'mlp':
        config = update_mlp_config_from_args(config, args, 
                                             output_dims=False)
        if 'embedding' not in args.output_config:
            config.kwargs.output_dim = args.vocab_size
    else:
        raise NotImplementedError
    return config

