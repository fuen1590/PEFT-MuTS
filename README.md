# PEFT-MuTS: A Multivariate Parameter-Efficient Fine-Tuning Framework for Remaining Useful Life Prediction based on Cross-domain Time Series Representation Model

This work presents a novel perspective based on cross-domain representation learning. By leveraging parameter-efficient fine-tuning (PEFT), it effectively addresses the few-shot Remaining Useful Life (RUL) prediction problem.
The original paper is currently under review.

## Datasets and Backbone weights
The sampled few-shot datasets and the backbone weights file can be downloaded from:🔗[Download Dataset and Backbone](https://yunpan.ustb.edu.cn/link/AA751D2BD16009481A93BEE755B735B795).

After downloading, move the `cmapss` and `bearing` folders to the `datasets/` directory, `model_ck.pt` to the `FEI_encResNet/` directory:
```
> PEFT-MuTS
  > datasets
    > bearing
      ...
    > cmapss
      ...
    - __init__.py
    - cmapss.py
      ...
  > FEI_encResNet
    - model_ck.pt
    ...
```

## Run
Install the required dependencies using requirements.txt, then start training and evaluation automatically by running:
```
python ./train/experiments.py
```
