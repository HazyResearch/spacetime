"""
Logging utilities
"""
import rich.syntax
import rich.tree

from omegaconf import OmegaConf, DictConfig, ListConfig


def print_header(x, border='both'):
    print('-' * len(x))
    print(x)
    print('-' * len(x))
    
    
def print_args(args, return_dict=False, verbose=True):
    attributes = [a for a in dir(args) if a[0] != '_']
    arg_dict = {}  # switched to ewr
    if verbose: print('ARGPARSE ARGS')
    for ix, attr in enumerate(attributes):
        # fancy = '└──' if ix == len(attributes) - 1 else '├──'
        fancy = '└─' if ix == len(attributes) - 1 else '├─'
        if verbose: print(f'{fancy} {attr}: {getattr(args, attr)}')
        arg_dict[attr] = getattr(args, attr)
    if return_dict:
        return arg_dict
        
# Control how tqdm progress bar looks        
def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'
    
    
def print_config(config: DictConfig,
                 resolve: bool = True,
                 name: str = 'CONFIG') -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "bright"  # "dim"
    tree = rich.tree.Tree(name, style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        elif isinstance(config_section, ListConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    # with open("config_tree.txt", "w") as fp:
    #     rich.print(tree, file=fp)