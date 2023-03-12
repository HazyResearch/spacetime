# SpaceTime 🌌⏱️
Code for SpaceTime, a neural net for time series. Inspired by state-**space**s for **time** series modeling.

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

Data for the Informer benchmark can be downloaded from [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset). The data exists as CSV files, whcih should be saved in the directory `./dataloaders/data/informer/`, e.g., `./dataloaders/data/informer/etth/ETTh1.csv`.

## Sample Commands  

### Informer

**ETTh1 720**  
```
python main.py --dataset etth1 --lag 336 --horizon 720 --embedding_config embedding/repeat --encoder_config encoder/default --decoder_config decoder/default --output_config output/default --n_blocks 1 --kernel_dim 64 --norm_order 1 --batch_size 50 --dropout 0.25 --lr 1e-3 --weight_decay 1e-4 --max_epochs 500 --data_transform mean --loss informer_rmse --val_metric informer_rmse --criterion_weights 1 1 1 --seed 0 
```

**ETTh2 720**  
```
python main.py --dataset etth2 --lag 336 --horizon 720 --embedding_config embedding/repeat --encoder_config encoder/default --decoder_config decoder/default --output_config output/default --n_blocks 1 --kernel_dim 64 --norm_order 1 --batch_size 50 --dropout 0.25 --lr 1e-3 --weight_decay 1e-4 --max_epochs 500 --data_transform mean --loss informer_rmse --val_metric informer_rmse --criterion_weights 1 1 1 --seed 0 
```

**ETTm1 720**  
```
python main.py --dataset ettm1 --lag 336 --horizon 720 --embedding_config embedding/repeat --encoder_config encoder/default --decoder_config decoder/default --output_config output/default --n_blocks 1 --kernel_dim 64 --norm_order 1 --batch_size 50 --dropout 0.25 --lr 1e-3 --weight_decay 1e-4 --max_epochs 500 --data_transform mean --loss informer_rmse --val_metric informer_rmse --criterion_weights 1 1 1 --seed 0 
```

**ETTm2 720**  
```
python main.py --dataset ettm2 --lag 336 --horizon 720 --embedding_config embedding/repeat --encoder_config encoder/default --decoder_config decoder/default --output_config output/default --n_blocks 1 --kernel_dim 64 --norm_order 1 --batch_size 50 --dropout 0.25 --lr 1e-3 --weight_decay 1e-4 --max_epochs 500 --data_transform mean --loss informer_rmse --val_metric informer_rmse --criterion_weights 1 1 1 --seed 0 
```
