# PPML based on Crypto

![](https://badgen.net/badge/:update-to/:Apr-2023/red) ![](https://badgen.net/badge/:papers/:43/blue)


Privacy-preserving machine learning (PPML) based on Cryptographic techniques has been attracting attention in recent years. A lot papers provide matchine learning as a service (MLaaS) mainly in two ways:
- **Inference**: A company offers a trained ML model, and a customer is able to query a feature input to obtain the inference result.
- **Training**: Multiple companies work together to train a high accuracy model using their datasets.

The major concern is to keep the model (i.e., model parameters) and the input data (either training data or inference samples) secret.

Existing works can be divided into different categories from different aspects. The commonly-adopted criterion are as follows:
- **Computation Setting**: This denotes how many parties are involved in the computation (i.e., inference or training). Typically, there are 2PC, 3PC, 4PC and 4+ parties.
- **Security Model**: This depicts the behavior of the adversary. e.g., semi-honest (or malicious) and honest-majority (or dishonest-majority).
- **Capability**: This represents whether the work support inference or training.

An overview of existing works is illustrated in the table below.

|     |      PPML      | Capability |          | Threat Model |                        |     Techniques    |
|:---:|:--------------:|:----------:|:--------:|:------------:|:----------------------:|:-----------------:|
|     |                |  Inference | Training |  Semi-honest |        Malicious       |                   |
| 2PC |       ABY      |  &#10004;  | &#10004; |   &#10004;   |                        |    OT & GC & SS   |
| 2PC |    SecureML    |  &#10004;  | &#10004; |   &#10004;   |                        |    HE & GC & SS   |
| 2PC |     MiniONN    |  &#10004;  |          |   &#10004;   |                        |    HE & GC & SS   |
| 2PC |     GAZELLE    |  &#10004;  |          |   &#10004;   |                        |    HE & GC & SS   |
| 2PC |      EzPC      |  &#10004;  |          |   &#10004;   |                        |      GC & SS      |
| 2PC |      XONN      |  &#10004;  |          |   &#10004;   |                        |      GC & SS      |
| 2PC |    QUOTIENT    |  &#10004;  | &#10004; |   &#10004;   |                        |    OT & GC & SS   |
| 2PC |      MP2ML     |  &#10004;  |          |   &#10004;   |                        |    HE & GC & SS   |
| 2PC |   CrypTFlow2   |  &#10004;  |          |   &#10004;   |                        |    HE & OT & SS   |
| 2PC |     Delphi     |  &#10004;  |          |   &#10004;   |                        |    HE & GC & SS   |
| 2PC |      GALA      |  &#10004;  |          |   &#10004;   |                        |      HE & GC      |
| 2PC |   QuantizedNN  |  &#10004;  |          |   &#10004;   |          Abort         |    HE & OT & SS   |
| 2PC |     GForce     |  &#10004;  |          |   &#10004;   |                        |      HE & SS      |
| 2PC |     ABY 2.0    |  &#10004;  | &#10004; |   &#10004;   |                        |    OT & GC & SS   |
| 2PC |      MUSE      |  &#10004;  |          |   &#10004;   |    Malicious clients   |    HE & GC & SS   |
| 2PC |      SIRNN     |  &#10004;  |          |   &#10004;   |                        |      SS & OT      |
| 2PC |    SecFloat    |  &#10004;  | &#10004; |   &#10004;   |                        |      SS & OT      |
| 2PC |     Cheetah    |  &#10004;  |          |   &#10004;   |                        |    HE & SS & OT   |
| 2PC |    PRNNInfer   |  &#10004;  |          |   &#10004;   |                        |         HE        |
| 2PC |     AriaNN     |  &#10004;  | &#10004; |   &#10004;   |                        |      FSS & SS     |
| 2PC |      Pika      |  &#10004;  | &#10004; |   &#10004;   |                        |        FSS        |
| 2PC |     Fusion     |  &#10004;  |          |   &#10004;   |    Malicious servers   |      SS & ZKP     |
| 2PC |      SIMC      |  &#10004;  |          |   &#10004;   |    Malicious clients   | SS & HE & OT & GC |
| 2PC |    Squirrel    |  &#10004;  | &#10004; |   &#10004;   |                        |    SS & HE & OT   |
|     |                |            |          |              |                        |                   |
| 3PC |    Chameleon   |  &#10004;  |          |   &#10004;   |                        |      GC & SS      |
| 3PC |      ABY3      |  &#10004;  | &#10004; |   &#10004;   |                        |      GC & SS      |
| 3PC |      ASTRA     |  &#10004;  | &#10004; |   &#10004;   |          Abort         |         SS        |
| 3PC |    SecureNN    |  &#10004;  | &#10004; |   &#10004;   |                        |         SS        |
| 3PC |      BLAZE     |  &#10004;  | &#10004; |   &#10004;   |        Fairness        |         SS        |
| 3PC |   QuantizedNN  |  &#10004;  |          |   &#10004;   |          Abort         |         SS        |
| 3PC |    CrypTFlow   |  &#10004;  |          |   &#10004;   |                        |         SS        |
| 3PC |      SWIFT     |  &#10004;  |          |   &#10004;   |           GOD          |         SS        |
| 3PC |     Falcon     |  &#10004;  | &#10004; |   &#10004;   |          Abort         |         SS        |
| 3PC |    CryptGPU    |  &#10004;  | &#10004; |   &#10004;   |                        |         SS        |
| 3PC | SecQuantizedNN |  &#10004;  | &#10004; |   &#10004;   |                        |         SS        |
| 3PC |     Piranha    |  &#10004;  | &#10004; |   &#10004;   |                        |         SS        |
| 3PC |      pMPL      |  &#10004;  | &#10004; |   &#10004;   | GOD (privileged party) |         SS        |
| 3PC |       PEA      |  &#10004;  | &#10004; |   &#10004;   |                        |      SS & DP      |
| 3PC |      LLAMA     |  &#10004;  |          |   &#10004;   |                        |      FSS & SS     |
|     |                |            |          |              |                        |                   |
| 4PC |      FLASH     |  &#10004;  | &#10004; |   &#10004;   |           GOD          |         SS        |
| 4PC |      SWIFT     |  &#10004;  | &#10004; |   &#10004;   |           GOD          |         SS        |
| 4PC |     Trident    |  &#10004;  | &#10004; |   &#10004;   |        Fairness        |      GC & SS      |
| 4PC | Fantastic Four |  &#10004;  | &#10004; |   &#10004;   |           GOD          |         SS        |
| 4PC |     Tetrad     |  &#10004;  | &#10004; |   &#10004;   |           GOD          |      GC & SS      |
|     |                |            |          |              |                        |                   |

> Note: one paper may be included in several categories (e.g. a paper that supports training naturally supports inference).

## Table of Contents
- [Survey](#survey)
- [2PC](#two-party-computation-2pc)
  * [Inference](#2pc-infer)
  * [Training](#2pc-train)
- [3PC](#three-party-computation-3pc)
  * [Inference](#3pc-infer)
  * [Training](#3pc-train)
- [4PC](#four-party-computation-4pc)
  * [Training](#4pc-train)

## Survey
- Cryptographic Primitives in Privacy-Preserving Machine Learning: A Survey.
    *H. Qin, D. He, Q. Feng, M. K. Khan, M. Luo and K. -K. R. Choo*
    IEEE Transactions on Knowledge and Data Engineering, [eprint](https://ieeexplore.ieee.org/document/10269692) 


## Two-party Computation (2PC)

### <a id='2pc-infer'>Secure Inference</a>
- Fusion: Efficient and Secure Inference Resilient to Malicious Servers.
    *Caiqin Dong, Jian Weng, Jia-Nan Liu, Yue Zhang, Yao Tong, Anjia Yang, Yudan Cheng, Shun Hu*
    NDSS 2023, [eprint](https://www.ndss-symposium.org/ndss-paper/fusion-efficient-and-secure-inference-resilient-to-malicious-servers/)
- SIMC: ML Inference Secure Against Malicious Clients at Semi-Honest Cost.
    *Nishanth Chandran, Divya Gupta, Sai Lakshmi Bhavana Obbattu, Akash Shah*
    USENIX 2022, [eprint](https://eprint.iacr.org/2021/1538)
- Private and Reliable Neural Network Inference.
    *Nikola Jovanovic, Marc Fischer, Samuel Steffen, Martin T. Vechev*
    CCS 2022, [eprint](https://doi.org/10.48550/arXiv.2210.15614)
- Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference.
    *Zhicong Huang, Wen-jie Lu, Cheng Hong, Jiansheng Ding*
    USENIX 2022, [eprint](https://eprint.iacr.org/2022/207)
- SIRNN: A Math Library for Secure RNN Inference.
    *Deevashwer Rathee, Mayank Rathee, Rahul Kranti Kiran Goli, Divya Gupta, Rahul Sharma, Nishanth Chandran, Aseem Rastogi*
    S&P 2021, [eprint](https://eprint.iacr.org/2021/459)
- Muse: Secure Inference Resilient to Malicious Clients.
    *Ryan Lehmkuhl, Pratyush Mishra, Akshayaram Srinivasan, Raluca Ada Popa*
    S&P 2021, [eprint](https://eprint.iacr.org/2021/1040)
- GForce: GPU-Friendly Oblivious and Rapid Neural Network Inference.
    *Lucien K. L. Ng, Sherman S. M. Chow*
    USENIX 2021, [eprint](https://www.usenix.org/conference/usenixsecurity21/presentation/ng)
- GALA: Greedy ComputAtion for Linear Algebra in Privacy-Preserved Neural Networks.
    *Qiao Zhang, Chunsheng Xin, Hongyi Wu*
    NDSS 2021, [eprint](https://arxiv.org/abs/2105.01827)
- Delphi: A Cryptographic Inference Service for Neural Networks.
    *Pratyush Mishra, Ryan Lehmkuhl, Akshayaram Srinivasan, Wenting Zheng, Raluca Ada Popa*
    USENIX 2020, [eprint](https://eprint.iacr.org/2020/050)
- CrypTFlow2: Practical 2-Party Secure Inference.
    *Deevashwer Rathee, Mayank Rathee, Nishant Kumar, Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma*
    CCS 2020, [eprint](https://eprint.iacr.org/2020/1002)
- MP2ML: A mixed-protocol machine learning framework for private inference.
    *Fabian Boemer, Rosario Cammarota, Daniel Demmler, Thomas Schneider, Hossein Yalame*
    ARES 2020, [eprint](https://eprint.iacr.org/2020/721)
- XONN: XNOR-based Oblivious Deep Neural Network Inference.
    *M. Sadegh Riazi, Mohammad Samragh, Hao Chen, Kim Laine, Kristin E. Lauter, Farinaz Koushanfar*
    USENIX 2019, [eprint](https://eprint.iacr.org/2019/171)
- EzPC: Programmable and efficient secure two-party computation for machine learning.
    *Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma, Shardul Tripathi*
    EuroS&P 2019, [eprint](https://doi.org/10.1109/EuroSP.2019.00043)
- GAZELLE: A Low Latency Framework for Secure Neural Network Inference.
    *Chiraag Juvekar, Vinod Vaikuntanathan, Anantha P. Chandrakasan*
    USENIX 2018, [eprint](http://eprint.iacr.org/2018/073)
- Oblivious Neural Network Predictions via MiniONN Transformations.
    *Jian Liu, Mika Juuti, Yao Lu, N. Asokan*
    CCS 2017, [eprint](http://eprint.iacr.org/2017/452)

### <a id='2pc-train'>Secure Training</a>
- Squirrel: A Scalable Secure Two-Party Computation Framework for Training Gradient Boosting Decision Tree.
    *Wen-jie Lu, Zhicong Huang, Qizhi Zhang, Yuchen Wang, Cheng Hong*
    USENIX 2023, [eprint](https://eprint.iacr.org/2023/527.pdf)
- AriaNN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing.
    *Théo Ryffel, Pierre Tholoniat, David Pointcheval, Francis R. Bach*
    PETS 2022, [eprint](https://arxiv.org/abs/2006.04593)
- Pika: Secure Computation using Function Secret Sharing over Rings.
    *Sameer Wagh*
    PETS 2022, [eprint](https://eprint.iacr.org/2022/826)
- SecFloat: Accurate Floating-Point meets Secure 2-Party Computation.
    *Deevashwer Rathee, Anwesh Bhattacharya, Rahul Sharma, Divya Gupta, Nishanth Chandran, Aseem Rastogi*
    S&P 2022, [eprint](https://eprint.iacr.org/2022/322)
- Piranha: A GPU Platform for Secure Computation.
    *Jean-Luc Watson, Sameer Wagh, Raluca Ada Popa*
    USENIX 2022, [eprint](https://eprint.iacr.org/2022/892)
- ABY2.0: Improved Mixed-Protocol Secure Two-Party Computation.
    *Arpita Patra, Thomas Schneider, Ajith Suresh, Hossein Yalame*
    USENIX 2021, [eprint](https://eprint.iacr.org/2020/1225)
- QUOTIENT: Two-party secure neural network training and prediction.
    *Nitin Agrawal, Ali Shahin Shamsabadi, Matt J. Kusner, Adrià Gascón*
    CCS 2019, [eprint](http://arxiv.org/abs/1907.03372)
- SecureML: A System for Scalable Privacy-Preserving Machine Learning.
    *Payman Mohassel, Yupeng Zhang*
    S&P 2017, [eprint](http://eprint.iacr.org/2017/396)
- ABY - A Framework for Efficient Mixed-Protocol Secure Two-Party Computation.
    *Daniel Demmler, Thomas Schneider, Michael Zohner*
    NDSS 2015, [eprint](https://www.ndss-symposium.org/ndss2015/aby---framework-efficient-mixed-protocol-secure-two-party-computation)


## Three-party Computation (3PC)
### <a id='3pc-infer'>Secure Inference</a>
- LLAMA: A Low Latency Math Library for Secure Inference.
    *Kanav Gupta, Deepak Kumaraswamy, Nishanth Chandran, Divya Gupta*
    PETS 2022, [eprint](https://eprint.iacr.org/2022/793)
- CrypTFlow: Secure TensorFlow Inference.
    *Nishant Kumar, Mayank Rathee, Nishanth Chandran, Divya Gupta, Aseem Rastogi, Rahul Sharma*
    S&P 2020, [eprint](http://arxiv.org/abs/1909.07814)
- Secure evaluation of quantized neural networks.
    *Anders P. K. Dalskov, Daniel Escudero, Marcel Keller*
    PETS 2020, [eprint](https://eprint.iacr.org/2019/131)
- Chameleon: a hybrid secure computation framework for machine learning applications.
    *M. Sadegh Riazi, Christian Weinert, Oleksandr Tkachenko, Ebrahim M. Songhori, Thomas Schneider, Farinaz Koushanfar*
    AsiaCCS 2018, [eprint](http://arxiv.org/abs/1801.03239)

### <a id='3pc-train'>Secure Training</a>
- Private, Efficient, and Accurate: Protecting Models Trained by Multi-party Learning with Differential Privacy
    *Wenqiang Ruan, Mingxin Xu, Wenjing Fang, Li Wang, Lei Wang, Weili Han*
    S&P 2023, [eprint](https://doi.org/10.48550/arXiv.2208.08662)
- Efficient decision tree training with new data structure for secure multi-party computation.
    *Koki Hamada, Dai Ikarashi, Ryo Kikuchi, Koji Chida*
    PETS 2023, [eprint](https://arxiv.org/abs/2112.12906)
- Multi-Party Replicated Secret Sharing over a Ring with Applications to Privacy-Preserving Machine Learning.
    *Alessandro N. Baccarini, Marina Blanton, Chen Yuan*
    PETS 2023, [eprint](https://eprint.iacr.org/2020/1577)
- Convolutions in Overdrive: Maliciously Secure Convolutions for MPC.
    *Marc Rivinius, Pascal Reisert, Sebastian Hasler, Ralf Küsters*
    PETS 2023, [eprint](https://eprint.iacr.org/2023/359)
- pMPL: A Robust Multi-Party Learning Framework with a Privileged Party.
    *Lushan Song, Jiaxuan Wang, Zhexuan Wang, Xinyu Tu, Guopeng Lin, Wenqiang Ruan, Haoqi Wu, Weili Han*
    CCS 2022, [eprint](https://doi.org/10.48550/arXiv.2210.00486)
- Piranha: A GPU Platform for Secure Computation.
    *Jean-Luc Watson, Sameer Wagh, Raluca Ada Popa*
    USENIX 2022, [eprint](https://eprint.iacr.org/2022/892)
- Secure Quantized Training for Deep Learning.
    *Marcel Keller, Ke Sun*
    ICML 2022, [eprint](https://eprint.iacr.org/2022/933)
- SWIFT: super-fast and robust privacy-preserving machine learning.
    *Nishat Koti, Mahak Pancholi, Arpita Patra, Ajith Suresh*
    USENIX 2021, [eprint](https://eprint.iacr.org/2020/592)
- CryptGPU: Fast Privacy-Preserving Machine Learning on the GPU.
    *Sijun Tan, Brian Knott, Yuan Tian, David J. Wu*
    S&P 2021, [eprint](https://eprint.iacr.org/2021/533)
- Falcon: Honest-Majority Maliciously Secure Framework for Private Deep Learning.
    *Sameer Wagh, Shruti Tople, Fabrice Benhamouda, Eyal Kushilevitz, Prateek Mittal, Tal Rabin*
    PETS 2021, [eprint](https://arxiv.org/abs/2004.02229)
- BLAZE: Blazing Fast Privacy-Preserving Machine Learning.
    *Arpita Patra, Ajith Suresh*
    NDSS 2020, [eprint](https://eprint.iacr.org/2020/042)
- SecureNN: 3-Party Secure Computation for Neural Network Training.
    *Sameer Wagh, Divya Gupta, Nishanth Chandran*
    PETS 2019, [eprint](https://eprint.iacr.org/2018/442)
- ASTRA: High Throughput 3PC over Rings with Application to Secure Prediction.
    *Harsh Chaudhari, Ashish Choudhury, Arpita Patra, Ajith Suresh*
    CCSW 2019, [eprint](https://eprint.iacr.org/2019/429)
- ABY3: A Mixed Protocol Framework for Machine Learning.
    *Payman Mohassel, Peter Rindal*
    CCS 2018, [eprint](https://eprint.iacr.org/2018/403)

## Four-party Computation (4PC)
### <a id='4pc-train'>Secure Training</a>
- Tetrad: Actively Secure 4PC for Secure Training and Inference.
    *Nishat Koti, Arpita Patra, Rahul Rachuri, Ajith Suresh*
    NDSS 2022, [eprint](https://eprint.iacr.org/2021/755)
- Fantastic Four: Honest-Majority Four-Party Secure Computation With Malicious Security.
    *Anders P. K. Dalskov, Daniel Escudero, Marcel Keller*
    USENIX 2021, [eprint](https://eprint.iacr.org/2020/1330)
- Trident: Efficient 4PC Framework for Privacy Preserving Machine Learning.
    *Harsh Chaudhari, Rahul Rachuri, Ajith Suresh*
    NDSS 2020, [eprint](http://arxiv.org/abs/1912.02631)
- SWIFT: super-fast and robust privacy-preserving machine learning.
    *Nishat Koti, Mahak Pancholi, Arpita Patra, Ajith Suresh*
    USENIX 2021, [eprint](https://eprint.iacr.org/2020/592)
- FLASH: Fast and Robust Framework for Privacy-preserving Machine Learning.
    *Megha Byali, Harsh Chaudhari, Arpita Patra, Ajith Suresh*
    PETS 2020, [eprint](https://eprint.iacr.org/2019/1365)
