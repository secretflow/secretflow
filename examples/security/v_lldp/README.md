<!--
 * @Descripttion: 
 * @version: 
 * @Author: syrbloom
 * @LastEditors: bloom0705
 * @E-mail: 809127446@qq.com
-->

# Layer-wise LDP

## Introduction

This folder contains the examples of split learning on Yahoo Answers & Criteo and the implementation of Layer-wise LDP(LLDP).

A layer-wise LDP for the SL system, dubbed LLDP, which disturbs various layers of a `sl_model` according to clients’ self-assigned privacy budgets.

## Implementation

``` python
    root：Layer-wise LDP/

    src/

        LLDP-bankmarketing.ipynb # demonstrate how the proposed LLDP scheme outperforms the embeddingdp and labeldp
        LLDP-Yahoo/ # Yahoo Answers of LLDP
            datasets.py 
            LLDP-Yahoo.ipynb 
        databuilder/
            Criteo-databuilder.ipynb # Criteo's databuilder
            Yahoo-databuilder.ipynb # Yahoo Answers's databuilder
    requirments.txt
```

## Note

If you want to run the LLDP-Yahoo.pynb, please modify the `file_path` in `datasets.py`, and then place it with `dataset.py` in the directory `secretflow/utils/simulation`.

## Reference

[1] Q. Chen, et al., “LLDP: A Layer-wise Local Differential Privacy in Federated Learning,” 21st IEEE TrustCom, 2022  
