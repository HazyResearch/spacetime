"""
Load and update model configs
"""
from os.path import join
from omegaconf import OmegaConf


# SpaceTime model
def load_model_config(config, config_dir='./configs/model', args=None):
    for k in ['embedding_config', 'encoder_config', 
              'decoder_config', 'output_config']:
        _config = OmegaConf.load(join(config_dir, f'{config[k]}.yaml'))
        if k == 'encoder_config' or k == 'decoder_config':
            for ix, block_config in enumerate(_config['blocks']):
                # Load preprocess kernel configs
                c_path = join(config_dir, f"{block_config['pre_config']}.yaml")
                block_config['pre_config'] = OmegaConf.load(c_path)
                # Load SSM kernel configs
                c_path = join(config_dir, f"{block_config['ssm_config']}.yaml")
                block_config['ssm_config'] = OmegaConf.load(c_path)
                # Load MLP configs
                c_path = join(config_dir, f"{block_config['mlp_config']}.yaml")
                block_config['mlp_config'] = OmegaConf.load(c_path)
        config[k] = _config
        
    config = update_embedding_config_from_args(config, args)
    config = update_block_config_from_args(config, args)
    config.output_config = update_output_config_from_args(config.output_config, args)
    config.output_config.input_dim = config.output_config.kwargs.input_dim
    config.output_config.output_dim = config.output_config.kwargs.output_dim        
    return config


def update_embedding_config_from_args(config, args):
    if args.model_dim is not None:
        config.embedding_config['kwargs']['embedding_dim'] = args.model_dim
        
    if args.n_heads is not None:
        config.embedding_config['kwargs']['n_heads'] = args.n_heads
    elif config.embedding_config['kwargs']['n_heads'] is not None:
        args.n_heads = config.embedding_config['kwargs']['n_heads']
    else:
        args.n_heads = args.model_dim // (args.input_dim * args.n_kernels)
        config.embedding_config['kwargs']['n_heads'] = args.n_heads
    
    if args.n_kernels is not None:
        config.embedding_config['kwargs']['n_kernels'] = args.n_kernels
    elif config.embedding_config['kwargs']['n_kernels'] is not None:
        args.n_kernels = config.embedding_config['kwargs']['n_kernels']
    else:
        args.n_kernels = args.model_dim // (args.input_dim * args.n_heads)
        config.embedding_config['kwargs']['n_kernels'] = args.n_kernels
        
    if args.model_dim is None:
        args.model_dim = args.n_heads * args.n_kernels

    return config


def update_block_config_from_args(config, args):
    # Update encoder only
    # - Update both SSM and MLP configs, and also total number of blocks
    # - For blocks, preserve first (which may be slightly special due to preprocessing)
    #   then add (args.n_blocks - 1) copies of an updated block
    
    # Update first block preprocess_config
    # config.encoder_config['blocks'][0].pre_config = update_preprocess_config_from_args(
    #     config.encoder_config['blocks'][0].pre_config, args)
    encoder_block = config.encoder_config['blocks'][0]
    encoder_block.pre_config = update_preprocess_config_from_args(encoder_block.pre_config, args)
    encoder_block.ssm_config = update_ssm_config_from_args(encoder_block.ssm_config, args)
    encoder_block.mlp_config = update_mlp_config_from_args(encoder_block.mlp_config, args)
    
    # Update remaining blocks
    encoder_block = config.encoder_config['blocks'][-1]
    if encoder_block.pre_config.kwargs is not None:
        encoder_block.pre_config = update_preprocess_config_from_args(encoder_block.pre_config, args)
    encoder_block.ssm_config = update_ssm_config_from_args(encoder_block.ssm_config, args)
    encoder_block.mlp_config = update_mlp_config_from_args(encoder_block.mlp_config, args)
    n_blocks = len(config.encoder_config['blocks'])
    if args.n_blocks is not None:
        n_blocks = args.n_blocks
    else:
        args.n_blocks = len(config.encoder_config['blocks']) + 1  # 1 decoder block for now
    config.encoder_config['blocks'] = ([config.encoder_config['blocks'][0]] + 
                                       [encoder_block] * (n_blocks - 1))
    return config


def update_preprocess_config_from_args(config, args):
    model_dim = (args.model_dim if args.model_dim is not None 
                 else config.kwargs.model_dim)
    kwargs = {
        'model_dim': model_dim,
        'kernel_repeat': model_dim // (config.kwargs.n_kernels * 
                                       config.kwargs.head_dim)
    }
    if 'max_avg_window' in config.kwargs:
        kwargs['max_avg_window'] = args.lag
        
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


def update_ssm_config_from_args(config, args):
    if config.method == 'companion' or config.method == 'shift':
        kwargs = get_companion_ssm_kwargs_from_args(config, args)
    else:
        raise NotImplementedError('Still need to implement non-companion SSM')
        
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
    
    
def get_companion_ssm_kwargs_from_args(config, args):
    n_heads, head_dim = update_n_heads(config, args)
    n_kernels = update_n_kernels(config, args, n_heads)
    kwargs = {
        'model_dim': args.model_dim,
        'n_kernels': n_kernels,
        'kernel_dim': args.kernel_dim,
        'n_heads': n_heads,
        'head_dim': head_dim,
        'kernel_init': args.kernel_init,
        'skip_connection': args.skip_ssm,
    }
    return kwargs


def update_n_heads(config, args):
    model_dim = (args.model_dim if args.model_dim is not None 
                 else config.kwargs.model_dim)
    head_dim = args.input_dim if config.kwargs.head_dim != 1 else 1
    if config.kwargs.n_heads == 1:
        n_heads = 1
    elif args.n_heads is not None:
        n_heads = args.n_heads
    elif args.n_heads is None:
        n_heads = config.kwargs.n_heads
    return n_heads, head_dim
    # if args.n_heads is None:
    #     if 'n_heads' in config.kwargs:
    #         n_heads = config.kwargs['n_heads']
    #     else:
    #         n_heads = None
    # else:
    #     n_heads = args.n_heads
    # return n_heads


def update_n_kernels(config, args, n_heads):
    model_dim = (args.model_dim if args.model_dim is not None 
                 else config.kwargs.model_dim)

    if n_heads == 1:
        n_kernels = model_dim
    elif args.n_kernels is not None:
        n_kernels = args.n_kernels
    elif args.n_kernels is None:
        n_kernels = model_dim // (config.kwargs.head_dim * n_heads)
        
    assert model_dim % (n_kernels * config.kwargs.head_dim * n_heads) == 0
    return n_kernels


def update_mlp_config_from_args(config, args, input_dims=True, output_dims=True):
    if config.method == 'mlp':
        kwargs = {
            'input_dim': args.model_dim,
            'output_dim': args.model_dim if output_dims else None,
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
                if not input_dims and k == 'skip_connection':
                    pass
                elif not output_dims and k == 'skip_connection':
                    pass
                else:
                    config.kwargs[k] = v
            elif input_dims and output_dims:
                setattr(args, k, config.kwargs[k])
    return config


def update_output_config_from_args(config, args):
    if config.method == 'mlp':
        config = update_mlp_config_from_args(config, args, 
                                             output_dims=False)
    return config
