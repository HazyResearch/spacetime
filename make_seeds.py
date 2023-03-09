import argparse


def main():
    parser = argparse.ArgumentParser(description='Make seeds')
    parser.add_argument('--script', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=5)
    
    args = parser.parse_args()
    
    seed = int(args.script.split('--seed ')[-1].split(' --')[0])
    
    all_scripts = []
    print(f'\nScripts:')
    print(f'--------')
    for _seed in range(args.num_seeds):
        print(args.script.replace(f'--seed {seed}', f'--seed {_seed}'))
    print('\n')
        
    
if __name__ == '__main__':
    main()
