# SpaceTime 🌌⏱️
Code for SpaceTime, a neural net for time series. Inspired by state-*space*s for *time* series modeling.

Cousin of S4, S4D, DSS, and H3. Descendent of LSSL. Expressive autoregressive modeling + fast flexible decoding (forecasting) ftw. 

Proposed in Effectively Modeling Time Series with Simple Discrete State Spaces, ICLR 2023. Paper links:     
* [ArXiv]()   
* [OpenReview](https://openreview.net/forum?id=2EpjkjzdCAa&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions))  

## Setup

We recommend creating a virtual environment with `conda`:  
```
conda env create -f environment.yml
conda activate spacetime
```

Data for the Informer benchmark is provided in `./dataloaders/data/informer/`. It can also be downloaded from [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset).

## Sample Commands  

### Informer

```
# ETTh1 720
python main.py --dataset etth1 --lag 336 --horizon 720 --embedding_config embedding/repeat --encoder_config encoder/default --decoder_config decoder/default --output_config output/default --lr 1e-3 --weight_decay 5e-4 --max_epochs 100 --data_transform mean --seed 0 --criterion_weights 1 1 1 --verbose

python main_spacecat_2.py --dataset etth1 --lag 720 --horizon 720 --criterion_weights 100 100 100 10 1 10 10 1 10 --n_blocks 4 --n_heads 128 --kernel_init xavier --normalize_c 1 --tie_weights --start_ix 0 --task_norm mean_lag --no_initial
```
