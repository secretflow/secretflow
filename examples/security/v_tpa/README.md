# Introduction
These are the poisoning attacks and defenses of split learning, including:
- Attacks
	- Gradient Replacement Attack[^1,^2]
	- Replay Attack[^3]
 - Defenses
	 - CoAE[^2]
# Implementation
- Directory `custom_base` contains the attack algorithms
	- Gradient Replacement Attack: `gradreplace_sl_base.py`
	- Replay Attack: `replay_sl_base.py`
- Directory `attack` contains the trigger injection
- Directory `defense` contains the defense methods
	- CoAE: `coae.py`
- Directory `dataset` is the implementation of data loading
- Directory `logs` records the experimental results
- Directory `models` is the model repository
- Directory `test_model` contains the models used to test
- Directory `trained_model` is used to save the trained models
- Directory `data` keeps the data for split learning
- Directory `tools` contains some extra utilities may be used in the test
- Directory `env` is the test environment
	- `environment.yml`
- Directory `tips` records some issues have been solved
# Test
```
**Test Configure**
- Test environment: `secretflow:0.8.2b3`
- Dataset configure: the dataset configure is `dataset/dataset_config.py`
- Model configure: the model configure is `test_model/tf_model_config.py`
- Attack configure: the attack configure is `attack/attack_config.py`
- Basic configure of split learning: `config.py`
```


1. Test amplification coefficient of gradient replacement
	- `test_sl_model_gamma.py`
2. Test replay attack
	- `test_sl_model_replay.py`
3. Test different split positions
	- `test_sl_model_layer.py`
4. Test different numbers of participants
	- `test_sl_model_parties.py`
5. Test different target labels
	- `test_sl_model_targets.py`
6. Test different aggregation modes, e.g., average, sum, and concatenation
	- `test_sl_model_agg.py`
7. Test CoAE defense
	- `test_sl_model_coae.py`

# References
>[^1]: Liu, Yang, Zhihao Yi, and Tianjian Chen. "Backdoor attacks and defenses in feature-partitioned collaborative learning." _arXiv preprint arXiv:2007.03608_ (2020).
> [^2]:Liu, Yang, et al. "Batch label inference and replacement attacks in black-boxed vertical federated learning." _arXiv preprint arXiv:2112.05409_ (2021).
> [^3]:Qiu, Pengyu, et al. "Hijack Vertical Federated Learning Models with Adversarial Embedding." _arXiv preprint arXiv:2212.00322_ (2022).