# Introduction
This folder contains the implementions of gradient inversion attacks and defense methods on horizontal Federal Learning.
- Attack
    
    [Surrogate model extension (SME): a fast and accurate weight update attack on federated learning](https://dl.acm.org/doi/abs/10.5555/3618408.3620229)


# Implemention
- Directory `SME_attack` contains the attack algorithm of SME:
    - SME attack: `gia_sme_torch.py`
    - Test of SME attack: `test_torch_gia_sme.py`

# Test
1. Test SME attack on MNIST dataset
    - `python test_torch_gia_sme.py`