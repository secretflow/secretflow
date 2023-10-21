# SplitGuard Detection
1. Methon
   
   By studying FSHA attacks, Erdogan et al. proposed SplitGuard, a method that can detect whether a split learning client is the target of a training hijacking attack. The starting point of SplitGuard is that if a client's local model is learning the intended classification task, then it should behave in a significantly different way when the task is reversed (i.e. when success on the original task means failure on the new task).

   [论文] https://arxiv.org/abs/2108.09052

2. Code file structure
    ``` python
    root:SplitGuard/

        slmodel_splitguard.py #Calculation implementation of SG score during training process
        mnist_sl_torch_splitguard.ipynb #test sample on mnist
        fashionmnist_sl_torch_splitguard.ipynb #test sample on fashion-mnist
        cifar10_sl_torch_splitguard.ipynb #test sample on cifar10
        cifar100_sl_torch_splitguard.ipynb #test sample on cifar100
    ```
