# Introduction

This document describes the implementation and evaluation of a novel backdoor attack in horizontal federated learning, which leverages frequency domain injection to create stealthy triggers.

# Attack Method

FreqBackdoor from [Federated Learning Backdoor Attack Based on Frequency Domain Injection](https://www.mdpi.com/1099-4300/26/2/164/pdf)

The proposed method, often referred to as **FreqBackdoor**, uses frequency domain injection for backdoor attacks. The key idea is to mix the amplitude spectra of the trigger image with that of the clean image, while keeping the phase spectrum unchanged. This way, the low-level frequency information of the trigger is injected invisibly into the clean image. Figure 1 gives the frequency-domain poisoning sample generation process.

<img width="874" alt="image" src="https://github.com/user-attachments/assets/3cea383c-44f8-477c-8573-dae3b9827ca6" />


# Main Steps

1. **Fourier Transformation:**  
   Both a clean image and a trigger image are transformed into the frequency domain using a Fourier transform. This process separates each image into an amplitude spectrum and a phase spectrum.

2. **Amplitude Mixing:**  
   A new amplitude spectrum is created by linearly mixing the amplitude spectrum of the clean image with that of the trigger image. A binary mask is used to control which parts of the amplitude spectrum are modified, ensuring that only selected low-frequency regions are affected.

3. **Image Reconstruction:**  
   The modified amplitude spectrum is then combined with the original phase spectrum of the clean image. This merged information is transformed back into the spatial domain using the inverse Fourier transform, resulting in a poisoned image that looks very similar to the original but contains the injected trigger.

4. **Injection into Federated Learning:**  
   The poisoned samples produced through this process are inserted into the training set of a malicious client. During federated training, the model learns to maintain high accuracy on clean samples while being manipulated to misclassify the poisoned samples as the attacker-specified target label.

---
# Implementation

  - `fl_model_freqbd.py`
  - `freqbackdoor_fl_torch.py`
  - Test of Model replacement backdoor attack: `test_torch_freqbackdoor.py`

# Test

1. Test **FreqBackdoor** on CIFAR-10 dataset
   
```bash
pytest --env sim -n auto -v --capture=no tests/ml/nn/fl/attack/test_torch_freqbackdoor.py

```

# Test results

After 50 epochs of training on CIFAR-10 (backdoor attack start at 30 epochs, poison_rate is 0.1), accuracy is 0.4907 and ASR is 0.9995.
![image](https://github.com/user-attachments/assets/939e9762-941d-4702-ae90-69355ed3eb4e)

