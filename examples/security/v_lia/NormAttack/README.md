# Norm Attack

1. 原理
   
   针对Naive架构的SplitNN的二分类任务，通过对返回的梯度进行分析，从而推断真实标签。

   [论文] https://arxiv.org/abs/2102.08504

2. 代码文件结构
    ``` python
    root:NormAttack/

    attack/
        labelleakage/     
            normattack.py  # Norm Attack的实现函数
    collaborative/  # SplitNN构建函数
    manager/  # SplitNN构建函数
    split_learning.ipynb  # Norm Attack的pipeline

    ```

3. 注意事项

    `attack/labelleakage/normattack.py`文件中包含了torch实现的原版Norm Attack和SecretFlow版的Norm Attack.
    
    Norm Attack原版是针对Naive SplitNN实现的，但此处做了修改，是针对两个client，一个server的SplitNN.

    此处SecretFlow版的Norm Attack没有效果（torch版本的有效果），但是pipeline仍然可以参考。