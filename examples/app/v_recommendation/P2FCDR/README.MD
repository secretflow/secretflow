## 1 Introduction

This is the implementation of *[Win-Win: A Privacy-Preserving Federated Framework for Dual-Target Cross-Domain Recommendation]* in SecretFlow. *[Win-Win: A Privacy-Preserving Federated Framework for Dual-Target Cross-Domain Recommendation]* proposes P2FCDR, a novel federated cross-domain recommendation framework.

## 2 Train

To train P2FCDR, you can run the following command:

```bash
python -u main.py \
        --num_round 20 \
        --local_epoch 3 \
        --eval_interval 1 \
        --frac 1.0 \
        --batch_size 1024 \
        --log_dir log \
        --method FedP2FCDR \
        --lr 0.001 \
        --seed 42 
```