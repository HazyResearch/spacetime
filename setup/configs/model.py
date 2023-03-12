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
    
    n_heads, head_dim = update_n_heads(config.embedding_config, args)
        
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
        
    if args.input_dim != 1:
        embedding_dim = (n_heads * head_dim * args.n_kernels)
        config.embedding_config['kwargs']['input_dim'] = args.input_dim
        
    elif args.model_dim is not None:
        embedding_dim = args.model_dim
        
    elif args.model_dim is None:
        args.model_dim = args.n_heads * args.n_kernels * head_dim
        embedding_dim = args.model_dim
        
    config.embedding_config['kwargs']['embedding_dim'] = embedding_dim
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
    if args.input_dim > 1:
        _config = encoder_block.pre_config
        _config.kwargs.head_dim = args.input_dim
        _config.kwargs.model_dim = (args.input_dim * _config.kwargs.n_heads *
                                    _config.kwargs.n_kernels * _config.kwargs.kernel_repeat)

    encoder_block.pre_config = update_preprocess_config_from_args(encoder_block.pre_config, args)
    if args.replicate != 2 or args.input_dim > 1:
        _config = encoder_block.ssm_config
        _config.kwargs.head_dim = args.input_dim
        _config.kwargs.model_dim = (args.input_dim * _config.kwargs.n_heads *
                                    _config.kwargs.n_kernels * _config.kwargs.kernel_repeat)
    encoder_block.mlp_config = update_mlp_config_from_args(encoder_block.mlp_config, args,
                                                           input_dim=_config.kwargs.model_dim if args.input_dim > 1 else None)
    
    # update_mlp_config_from_args(config, args, 
    #                             input_dims=True, output_dims=True, 
    #                             input_dim=None, output_dim=None)
    
    
    # encoder_block = config.encoder_config['blocks'][1]
    # if args.input_dim > 1:
    #     _config = encoder_block.pre_config
    #     _config.kwargs.head_dim = args.input_dim
    #     _config.kwargs.model_dim = (args.input_dim * _config.kwargs.n_kernels * 
    #                                 _config.kwargs.kernel_repeat)
    # encoder_block.pre_config = update_preprocess_config_from_args(encoder_block.pre_config, args)
    # if args.replicate != 2:
    #     encoder_block.ssm_config = update_ssm_config_from_args(encoder_block.ssm_config, args)
    # encoder_block.mlp_config = update_mlp_config_from_args(encoder_block.mlp_config, args)
    
    # Update remaining blocks
    encoder_block = config.encoder_config['blocks'][-1]
    if encoder_block.pre_config.kwargs is not None:
        encoder_block.pre_config = update_preprocess_config_from_args(encoder_block.pre_config, args)
    encoder_block.ssm_config = update_ssm_config_from_args(encoder_block.ssm_config, args)
    encoder_block.mlp_config = update_mlp_config_from_args(encoder_block.mlp_config, args)
    n_blocks = len(config.encoder_config['blocks'])
    if args.n_blocks is not None:
        n_blocks = args.n_blocks - 1
    else:
        args.n_blocks = len(config.encoder_config['blocks']) + 1  # 1 decoder block for now
    config.encoder_config['blocks'] = ([config.encoder_config['blocks'][0]] + 
                                       [encoder_block] * (n_blocks - 1))
    
    # Update decoder block
    config.decoder_config.blocks[0] = update_decoder_block(config.decoder_config.blocks[0], args)
    return config


def update_decoder_block(decoder_block, args):
    decoder_block.ssm_config.kwargs.lag = args.lag
    decoder_block.ssm_config.kwargs.horizon = args.horizon
    
    n_heads, head_dim = update_n_heads(decoder_block.ssm_config, args)
    n_kernels = update_n_kernels(decoder_block.ssm_config, args, n_heads)
    
    if args.model_dim is not None:
        decoder_block.ssm_config.kwargs.model_dim = args.model_dim 
    decoder_block.ssm_config.kwargs.n_kernels = n_kernels
    decoder_block.ssm_config.kwargs.n_heads = n_heads
    decoder_block.ssm_config.kwargs.head_dim = head_dim
    if args.norm_order is not None:
        decoder_block.ssm_config.kwargs.norm_order = args.norm_order
    return decoder_block


def update_preprocess_config_from_args(config, args):
    # if args.input_dim > 1:
    #     config.kwargs.head_dim = args.input_dim
    #     model_dim = (args.input_dim * config.kwargs.n_kernels * 
    #                  config.kwargs.kernel_repeat)
    if args.model_dim is not None:
        model_dim = args.model_dim
    else:
        model_dim = config.kwargs.model_dim
    
    kwargs = {
        'model_dim': model_dim,
        'head_dim': args.input_dim,
        'kernel_repeat': model_dim // (config.kwargs.n_kernels * 
                                       config.kwargs.head_dim),
        'seed': args.seed
    }
    # if 'max_avg_window' in config.kwargs:
    #     kwargs['max_avg_window'] = args.lag
        
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
    if 'companion' in config.method or 'shift' in config.method:
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
        'norm_order': args.norm_order
    }
    return kwargs


def update_n_heads(config, args):
    # if args.input_dim > 1:
    #     head_dim = args.input_dim
    if 'head_dim' in config.kwargs:
        head_dim = args.input_dim if config.kwargs.head_dim != 1 else 1
    else:
        head_dim = args.input_dim
        
    if config.kwargs.n_heads == 1:
        n_heads = 1
    elif args.n_heads is not None:
        n_heads = args.n_heads
    elif args.n_heads is None:
        n_heads = config.kwargs.n_heads
    return n_heads, head_dim


def update_n_kernels(config, args, n_heads):
    model_dim = (args.model_dim if args.model_dim is not None 
                 else config.kwargs.model_dim)
    
    # if args.input_dim > 1:
    #     config.kwargs.head_dim = args.input_dim

    if n_heads == 1:
        n_kernels = model_dim // config.kwargs.head_dim
    elif args.n_kernels is not None:
        n_kernels = args.n_kernels
    elif args.n_kernels is None:
        n_kernels = model_dim // (config.kwargs.head_dim * n_heads)
        
    try:
        assert model_dim % (n_kernels * config.kwargs.head_dim * n_heads) == 0
    except Exception as e:
        print(e)
        print(f'model_dim:', model_dim)
        print(f'n_kernels:', n_kernels)
        print(f'config.kwargs.head_dim:', config.kwargs.head_dim)
        print(f'n_heads:', n_heads)
        breakpoint()
        raise e
    return n_kernels


def update_mlp_config_from_args(config, args, 
                                input_dims=True, output_dims=True, 
                                input_dim=None, output_dim=None):
    # Logic for handling input and output dimension update
    if input_dims and input_dim is None:
        input_dim = args.model_dim
    elif not input_dims:
        input_dim = None
        
    if output_dims and output_dim is None:
        output_dim = args.model_dim
    elif not output_dims:
        output_dim = None
        
    if config.method == 'mlp':
        kwargs = {
            'input_dim': input_dim,
            'output_dim': output_dim,
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


def update_output_config_from_args(config, args, update_output_dim=False, 
                                   output_dim=None):
    if config.method == 'mlp':
        config = update_mlp_config_from_args(config, args, input_dims=True,
                                             output_dims=update_output_dim,
                                             input_dim=args.model_dim,
                                             output_dim=output_dim)
    return config
