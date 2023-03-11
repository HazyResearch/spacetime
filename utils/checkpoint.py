import copy


def update_args_from_checkpoint_name(args, fname):
    _args = copy.deepcopy(args)
    fname = fname.replace('=no-', '=normal-').replace('=xa-', '=xavier-').replace('.pth', '').replace('=tc-', '=timm_cosine-').replace('=ir-', '=informer_rmse-')
    all_args = []
    for f in fname.split('-')[2:]:
        k, v = f.split('=')

        if k in all_args:
            k += '_'
        all_args.append(k)
        try:
            v = arg_type[k](v)
        except Exception as e:
            print(k, v, e)

        if v != 'None':
            if k in ['ec', 'pc', 'ec_', 'dc', 'oc']:
                v = set_config_arg(v, arg_map[k])
            setattr(_args, arg_map[k], v)
        else:
            setattr(_args, arg_map[k], None)
    return _args


def set_config_arg(config_name, arg_map_val):
    return f"{arg_map_val.split('_')[0]}/{config_name}"


arg_map = {'ns': 'n_shots',
           'ec': 'embedding_config',
           'pc': 'preprocess_config',
           'ec_': 'encoder_config',
           'dc': 'decoder_config',
           'oc': 'output_config',
           'nb': 'n_blocks',
           'nk': 'n_kernels',
           'nh': 'n_heads',
           'md': 'model_dim',
           'kd': 'kernel_dim',
           'ki': 'kernel_init',
           'no': 'norm_order',
           'la': 'lag',
           'ho': 'horizon',
           'dt': 'data_transform',
           'cw': 'criterion_weights',
           'lo': 'loss',
           'dr': 'dropout',
           'lr': 'lr',
           'op': 'optimizer',
           'sc': 'scheduler',
           'wd': 'weight_decay',
           'bs': 'batch_size',
           'vm': 'val_metric',
           'me': 'max_epochs',
           'ese': 'early_stopping_epochs',
           're': 'replicate',
           'se': 'seed'}


arg_type = {'ns': int,
            'ec': str,
            'pc': str,
            'ec_': str,
            'dc': str,
            'oc': str,
            'nb': int,
            'nk': int,
            'nh': int,
            'md': int,
            'kd': int,
            'hd': int,
            'ki': str,
            'no': int,
            'la': int,
            'ho': int,
            'dt': str,
            'cw': str,
            'lo': str,
            'dr': float,
            'la': int,  # bool
            'lr': float,
            'op': str,
            'sc': str,
            'wd': float,
            'bs': int,
            'vm': str,
            'me': int,
            'ese': int,
            're': int,
            'se': int}
