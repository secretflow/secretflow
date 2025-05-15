# Introduction

This folder contains the implementions of backdoor attack on horizontal Federal Learning.

## Attack Method

[BACKDOOR FEDERATED LEARNING BY POISONING BACKDOOR-CRITICAL LAYERS](https://openreview.net/pdf?id=AJBGSVSTT2)

### Algorithm Description

Existing FL attack and defense methodologies typically focus on the whole model. None of them recognizes the existence of backdoor-critical (BC) layers—a small subset of layers that dominate the model vulnerabilities. Attacking the BC layers achieves equivalent effects as attacking the whole model but at a far smaller chance of being detected by state-of-the-art (SOTA) defenses. This paper proposes a general in-situ approach that identifies and verifies BC layers from the perspective of attackers. Based on the identified BC layers, the authors carefully craft a new backdoor attack methodology that adaptively seeks a fundamental balance between attacking effects and stealthiness under various defense strategies. Extensive experiments show that our BC layer-aware backdoor attacks can successfully backdoor FL under seven SOTA defenses with only 10% malicious clients and outperform latest backdoor attack methods.



# Implemention
  - `fl_model_bd.py`
  - `backdoor_BCL_torch.py`
  - Test of Backdoor Critical Layer Attack: `test_torch_backdoor.py`

# Test

1. Test Backdoor Critical Layer attack 
    - `pytest --env sim -n auto -v --capture=no tests/ml/nn/fl/attack/test_torch_backdoor.py`