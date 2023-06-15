# Secure Multi-Party Computation (MPC)

![](https://badgen.net/badge/:update-to/:Apr-2023/red) ![](https://badgen.net/badge/:papers/:70/blue)

> "The design of scure protocols that implement arbitrarily desired functionalities is a major part of mordern cryptography."
> -- Foundation of Cryptography, Volumn 2, Oded Goldreich.

MPC has evolved from a theoretical curiosity in the 1980s to a tool for building real systems today. Over the past decade, MPC has been one of the most active research areas in both theoretical and applied cryptography. In the following, we try to show the newest and interesting advances in mpc (both theory & applicaiton), and also the infulential papers in history.

Note: one paper may be included in several categories (e.g. a paper may introduce a new protocol for both OT and VOLE, we decide to include it in both categories).

## Table of Contents

- [Secure Multi-Party Computation (MPC)](#secure-multi-party-computation-mpc)
  - [Table of Contents](#table-of-contents)
  - [Offline Techniques](#offline-techniques)
    - [Oblivious transfer](#oblivious-transfer)
    - [vector Oblivious Linear Evaluation](#vector-oblivious-linear-evaluation)
    - [Pseudorandom-Correlation Generator](#pseudorandom-correlation-generator)
    - [Preprocessing](#preprocessing)
  - [Online Techniques](#online-techniques)
    - [Semi-Honest Secret Sharing](#semi-honest-secret-sharing)
    - [Malicious Secret Sharing](#malicious-secret-sharing)

  - [Distance Calculation](#distance-calculation)
  - [Secure Multi-party Computation,MPC](#secure-multi-party-computationmpc)
  - [FHE-based MPC](#fhe-based-mpc)


    - [Function Secret Sharing](#function-secret-sharing)


## Offline Techniques

### Oblivious transfer

- Endemic Oblivious Transfer via Random Oracles, Revisited
  *Zhelei Zhou, Bingsheng Zhang, Hong-Sheng Zhou, Kui Ren*
  EuroCrypt 2023, [eprint](https://eprint.iacr.org/2022/1525), ZZZR23

- SoftSpokenOT: Quieter OT Extension from Small-Field Silent VOLE in the Minicrypt Model
  *Lawrence Roy*
  Crypto 2022, [eprint](https://eprint.iacr.org/2022/192), Roy22

- Silver: Silent VOLE and Oblivious Transfer from Hardness of Decoding Structured LDPC Codes
  *Geoffroy Couteau, Peter Rindal, Srinivasan Raghuraman*
  Crypto 2021, [eprint](https://eprint.iacr.org/2021/1150), CRR21

- The Rise of Paillier: Homomorphic Secret Sharing and Public-Key Silent OT
  *Claudio Orlandi, Peter Scholl, Sophia Yakoubov*
  EuroCrypt 2021, [eprint](https://eprint.iacr.org/2021/262), OSY21

- Batching Base Oblivious Transfers
  *Ian McQuoid, Mike Rosulek, Lawrence Roy*
  AsiaCrypt 2021, [eprint](https://eprint.iacr.org/2021/682), MRR21

- Ferret: Fast Extension for Correlated OT with Small Communication
  *Kang Yang, Chenkai Weng, Xiao Lan, Jiang Zhang, Xiao Wang*
  CCS 2020, [eprint](https://eprint.iacr.org/2020/924.pdf), YWLZ+20

- Efficient and Round-Optimal Oblivious Transfer and Commitment with Adaptive Security
  *Ran Canetti, Pratik Sarkar, Xiao Wang*
  AsiaCrypt 2020, [eprint](https://eprint.iacr.org/2020/545), CSW20

- Efficient Two-Round OT Extension and Silent Non-Interactive Secure Computation
  *Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai, Lisa Kohl, Peter Rindal, Peter Scholl*
  CCS 2019, [eprint](https://eprint.iacr.org/2019/1159), BCGI+19 (with Peter Rindal)

- Endemic Oblivious Transfer
  *Daniel Masny, Peter Rindal*
  CCS 2019, [eprint](https://eprint.iacr.org/2019/706), MR19

- Efficient Pseudorandom Correlation Generators: Silent OT Extension and More
  *Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai, Lisa Kohl, Peter Scholl*
  Crypto 2019, [eprint](https://eprint.iacr.org/2019/448), BCGI+19 (without Peter Rindal)

- Equational Security Proofs of Oblivious Transfer Protocols
  *Baiyu Li, Daniele Micciancio*
  PKC 2018, [eprint](https://eprint.iacr.org/2016/624), LM18

- Actively Secure 1-out-of-N OT Extension with Application to Private Set Intersection
  *Michele Orrù, Emmanuela Orsini, Peter Scholl*
  CT-RSA 2017, [eprint](https://eprint.iacr.org/2016/933), OOS17

- Actively Secure OT Extension with Optimal Overhead
  *Marcel Keller, Emmanuela Orsini, Peter Scholl*
  Crypto 2015, [eprint](https://eprint.iacr.org/2015/546), KOS15

- The Simplest Protocol for Oblivious Transfer
  *Tung Chou, Claudio Orlandi*
  LatinCrypt 2015, [eprint](https://eprint.iacr.org/2015/267), CO15

- More Efficient Oblivious Transfer and Extensions for Faster Secure Computation
  *Gilad Asharov, Yehuda Lindell, Thomas Schneider, Michael Zohner*
  CCS 2013, [eprint](https://eprint.iacr.org/2013/552), ALSZ13

- A Framework for Efficient and Composable Oblivious Transfer
  *Chris Peikert, Vinod Vaikuntanathan, Brent Waters*
  Crypto 2008, [eprint](https://eprint.iacr.org/2007/348), PVW08

- Extending Oblivious Transfers Efficiently
  *Yuval Ishai, Joe Kilian, Kobbi Nissim, Erez Petrank*
  Crypto 2003, [eprint](https://www.iacr.org/archive/crypto2003/27290145/27290145.pdf), IKNP03

- Oblivious Transfer and Polynomial Evaluation
  *Moni Naor, Benny Pinkas*
  STOC 1999, [eprint](https://dl.acm.org/doi/pdf/10.1145/301250.301312), NP99

### vector Oblivious Linear Evaluation

- Actively Secure Arithmetic Computation and VOLE with Constant Computational Overhead
  *Benny Applebaum, Niv Konstantini*
  EuroCrypt 2023, [eprint](https://eprint.iacr.org/2023/270), AK23

- Two-Round Oblivious Linear Evaluation from Learning with Errors
  *Pedro Branco, Nico Do ̈ttling, Paulo Mateus*
  PKC 2022, [eprint](https://eprint.iacr.org/2020/635), BDM22

- Correlated Pseudorandomness from Expand-Accumulate Codes
  *Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai, Lisa Kohl, Nicolas Resch, Peter Scholl*
  Crypto 2022, [eprint](https://eprint.iacr.org/2022/1014), BCG+22

- Two-Round Oblivious Linear Evaluation from Learning with Errors
  *Pedro Branco, Nico Döttling, Paulo Mateus*
  PKC 2022, [eprint](https://eprint.iacr.org/2020/635), BDM22

- Silver: Silent VOLE and Oblivious Transfer from Hardness of Decoding Structured LDPC Codes
  *Geoffroy Couteau, Peter Rindal, Srinivasan Raghuraman*
  Crypto 2021, [eprint](https://eprint.iacr.org/2021/1150), CRR21

- Efficient Protocols for Oblivious Linear Function Evaluation from Ring-LWE
  *Carsten Baum, Daniel Escudero, Alberto Pedrouzo-Ulloa, Peter Scholl, Juan Ramón Troncoso-Pastoriza*
  SCN 2020, [eprint](https://eprint.iacr.org/2020/970), BEPS+20

- Distributed vector-OLE: Improved constructions and implementation
  *Phillipp Schoppmann, Adrià Gascón, Leonie Reichert, Mariana Raykova*
  CCS 2019, [eprint](https://eprint.iacr.org/2019/1084), SGRR19

- Compressing vector OLE
  *Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai*
  CCS 2018, [eprint](https://eprint.iacr.org/2019/273), BCGI18

- Secure Arithmetic Computation with Constant Computational Overhead
  *Benny Applebaum, Ivan Damgård, Yuval Ishai, Michael Nielsen, Lior Zichron*
  Crypto 2017, [eprint](https://eprint.iacr.org/2017/617), ADI+17

- Maliciously secure oblivious linear function evaluation with constant overhead
  *Satrajit Ghosh, Jesper Buus Nielsen, Tobias Nilges*
  AsiaCrypt 2017, [eprint](https://eprint.iacr.org/2017/409), GNN17

- TinyOLE: Efficient actively secure two-party computation from oblivious linear function evaluation, 2017,
  *Nico Döttling, Satrajit Ghosh, Jesper Buus Nielsen, Tobias Nilges, Roberto Trifiletti*
  CCS 2017, [eprint](https://eprint.iacr.org/2017/790), DGNN+17

- Oblivious Transfer and Polynomial Evaluation
  *Moni Naor, Benny Pinkas*
  STOC 1999, [eprint](https://dl.acm.org/doi/pdf/10.1145/301250.301312), NP99

### Pseudorandom-Correlation Generator

- Correlated Pseudorandomness from Expand-Accumulate Codes
  *Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai, Lisa Kohl, Nicolas Resch, Peter Scholl*
  Crypto 2022, [eprint](https://eprint.iacr.org/2022/1014), BCG+22

### Preprocessing

- Improved primitives for MPC over mixed arithmetic-binary circuits
  *Daniel Escudero, Satrajit Ghosh, Marcel Keller ,Rahul Rachuri, Peter Scholl*
  Crypto 2020, [eprint](https://eprint.iacr.org/2020/338.pdf)

- MArBled Circuits: Mixing Arithmetic and Boolean Circuits with Active Security
  *Dragos Rotaru, Tim Wood*
  IndoCrypt 2019, [eprint](https://eprint.iacr.org/2019/207)

- High-Throughput Secure Three-Party Computation for Malicious Adversaries and an Honest Majority
  *Jun Furukawa, Yehuda Lindell, Ariel Nof, Or Weistein*
  EuroCrypt 2017, [eprint](https://eprint.iacr.org/2016/944.pdf), FLNW17

## Online Techniques

### Semi-Honest Secret Sharing

- Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference
  *Zhicong Huang, Wen-jie Lu, Cheng Hong, Jiansheng Ding*
  USENIX Security 2022, [eprint](https://eprint.iacr.org/2022/207), HLHD22

- ABY2.0: Improved Mixed-Protocol Secure Two-Party Computation
  *Arpita Patra, Thomas Schneider, Ajith Suresh, Hossein Yalame*
  Usenix Security 2021, [eprint](https://eprint.iacr.org/2020/1225), PSSY21

- ABY - A Framework for Efficient Mixed-Protocol Secure Two-Party Computation
  *Daniel Demmler, Thomas Schneider, Michael Zohner*
  NDSS 2017, [eprint](https://www.ndss-symposium.org/wp-content/uploads/2017/09/08_2_1.pdf), DSZ17

- Secure Computation with Fixed-Point Numbers
  *Octavian Catrina, Amitabh Saxena*
  FC 2010, [eprint](https://ifca.ai/pub/fc10/31_47.pdf), CS10

- The Round Complexity of Secure Protocols
  *Donald Beaver, Silvio Micali, Phillip Rogaway*
  STOC 1990, [eprint](http://web.cs.ucdavis.edu/~rogaway/papers/bmr90), BMR90

- Completeness Theorems for Non-Cryptographic Fault Tolerant Distributed Computation
  *Michael Ben-Or, Shafi Goldwasser, Avi Wigderson*
  STOC 1988, [eprint](https://dl.acm.org/doi/10.1145/62212.62213), BGW88

- How to play any mental game?
  *Oded Goldreich, Silvio Micali, Avi Wigderson*
  STOC 1987, [eprint](https://dl.acm.org/doi/10.1145/28395.28420), GMW87

- How to generate and exchange secrets?
  *Andrew Chi-Chih Yao*
  FOCS 1986, [eprint](https://ieeexplore.ieee.org/document/4568207), Yao86

### Malicious Secret Sharing

- MHz2k: MPC from HE over Z2k with New Packing, Simpler Reshare, and Better ZKP
  *Jung Hee Cheon, Dongwoo Kim, and Keewoo Lee*
  Crypto 2021, [eprint](https://eprint.iacr.org/2021/1383), CKL21

- High-Performance Multi-party Computation for Binary Circuits Based on Oblivious Transfer
  *Sai Sheshank Burra, Enrique Larraia, Jesper Buus Nielsen, Peter Sebastian Nordholt, Claudio Orlandi, Emmanuela Orsini, Peter Scholl, Nigel P. Smart*
  JCryptpo 2021, [eprint](https://eprint.iacr.org/2015/472), BLNN+21

- An Efficient Passive-to-Active Compiler for Honest-Majority MPC over Rings
  *Mark Absponel, Anders Dalskov, Daniel Escudero, Ariel Nof*
  ACNS 2021, [eprint](https://eprint.iacr.org/2019/1298)

- Fantastic Four: Honest-Majority Four-Party Secure Computation With Malicious Security
  *Anders Dalskov, Daniel Escudero, and Marcel Keller*
  Usenix Security 2021, [eprint](https://eprint.iacr.org/2020/1330)

- Overdrive2k: Efficient Secure MPC over Z2k from Somewhat Homomorphic Encryption
  *Emmanuela Orsini, Nigel P. Smart, Frederik Vercauteren*
  CT-RSA 2020, [eprint](https://eprint.iacr.org/2019/153), OSV20

- MonZ2k: Fast Maliciously Secure Two Party Computation on Z2k
  *Dario Catalano, Mario Di Raimondo, Dario Fiore, and Irene Giacomelli*
  PKC 2020, [eprint](https://eprint.iacr.org/2019/211), CRFG20

- MP-SPDZ: A Versatile Framework for Multi-Party Computation
  *Marcel Keller*
  CCS 2020, [eprint](https://eprint.iacr.org/2020/521), Kel20

- Covert Security with Public Verifiability: Faster, Leaner, and Simpler
  *Cheng Hong, Jonathan Katz, Vladimir Kolesnikov, Wen-jie Lu, Xiao Wang*
  EuroCrypt 2019, [eprint](https://eprint.iacr.org/2018/1108), HKKL+19

- Two-Thirds Honest-Majority MPC for Malicious Adversaries at Almost the Cost of Semi-Honest
  *Jun Furukawa, Yehuda Lindell*
  CCS 2019, [eprint](https://eprint.iacr.org/2019/658), FL19

- Communication Lower Bounds for Statistically Secure MPC, With or Without Preprocessing
  *Ivan Damgård, Kasper Green Larsen, Jesper Buus Nielsen*
  Crypto 2019, [eprint](https://eprint.iacr.org/2019/220), DLN19

- New Primitives for Actively-Secure MPC over Rings with Applications to Private Machine Learning
  *Ivan Damgård, Daniel Escudero, Tore Frederiksen, Marcel Keller, Peter Scholl, Nikolaj Volgushev*
  SP 2019, [eprint](https://eprint.iacr.org/2019/599), DEFK+19

- Adaptively Secure MPC with Sublinear Communication Complexity
  *Ran Cohen, Abhi Shelat, Daniel Wichs*
  Crypto 2019, [eprint](https://eprint.iacr.org/2018/1161), CSW19

- ABY3: A Mixed Protocol Framework for Machine Learning
  *Payman Mohassel, Peter Rindal*
  CCS 2018, [eprint](https://eprint.iacr.org/2018/403), MR18

- Fast Large-Scale Honest-Majority MPC for Malicious Adversaries
  *Koji Chida, Daniel Genkin, Koki Hamada, Dai Ikarashi, Ryo Kikuchi, Yehuda Lindell, Ariel Nof*
  Crypto 2018, [eprint](https://eprint.iacr.org/2018/570)

- SPD$\mathbb {Z}_{2^k}$: Efficient MPC mod $2^k$ for Dishonest Majority
  *Ronald Cramer, Ivan Damgård, Daniel Escudero, Peter Scholl, Chaoping Xing*
  Crypto 2018, [eprint](https://eprint.iacr.org/2018/482.pdf), Spdz2k

- Optimized Honest-Majority MPC for Malicious Adversaries — Breaking the 1 Billion-Gate Per Second Barrier
  *Toshinori Araki, Assi Barak, Jun Furukawa, Tamar Lichter, Yehuda Lindell, Ariel Nof, Kazuma Ohara, Adi Watzman, Or Weinstein*
  S&P 2017, [eprint](https://www.ieee-security.org/TC/SP2017/papers/96.pdf)

- Zero-Knowledge Proofs on Secret-Shared Data via Fully Linear PCPs
  *Dan Boneh, Elette Boyle, Henry Corrigan-Gibbs, Niv Gilboa, Yuval Ishai*
  Crypto 2019, [eprint](https://eprint.iacr.org/2019/188.pdf), BBCG+19

- Practical Fully Secure Three-Party Computation via Sublinear Distributed Zero-Knowledge Proofs
  *Elette Boyle, Niv Gilboa, Yuval Ishai, and Ariel Nof*
  CCS 2019, [eprint](https://eprint.iacr.org/2019/1390), BGIN19



## Distance Calculation

- Faster Privacy-Preserving Location Proximity Schemes  
  *Kimmo Järvinen, Ágnes Kiss, Thomas Schneider, Oleksandr Tkachenko, Zheng Yang*  
  CANS 2018, [eprint](https://link.springer.com/chapter/10.1007/978-3-030-00434-7_1), JKST+19  

- Faster privacy-preserving location proximity schemes for circles and polygons  
  *Kimmo Järvinen, Ágnes Kiss, Thomas Schneider, Oleksandr Tkachenko, Zheng Yang*  
  CANS 2018, [eprint](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ifs.2019.0125), JKST+19  

- Universal Location Referencing and Homomorphic Evaluation of Geospatial Query  
  *Asma Aloufi, Peizhao Hu, Hang Liu, Sherman S. M. Chow*  
  IACR Cryptol. ePrint Arch. 2019, [eprint](https://eprint.iacr.org/2019/820), AHLC19  
  

- Geosocial Query with User-Controlled Privacy  
  *Peizhao Hu, Sherman S. M. Chow, Asma Aloufi*  
  IACR Cryptol. ePrint Arch. 2018, [eprint](https://eprint.iacr.org/2018/304), HCA18  
  

- "Are we close?" – Secure Proximity Computation in Geosocial Networks  
  *Peizhao Hu, Tamalika Mukherjee, Alagu Valliappan and Stanisław Radziszowski*  
  [eprint](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3df1e190b007459e1babf9204821c5f340b6da91)

- Private Computation on Encrypted Genomic Data  
  *Kristin Lauter, Adriana López-Alt, Michael Naehrig*  
  LATINCRYPT 2014, [eprint](https://link.springer.com/chapter/10.1007/978-3-319-16295-9_1), LLN14  



## Secure Multi-party Computation,MPC

- A Gentle Introduction to Yao’s Garbled Circuits  
  *Sophia Yakoubov*  
  [eprint](https://web.mit.edu/sonka89/www/papers/2017ygc.pdf),Y17  

- A Proof of Security of Yao’s Protocol for Two-Party Computation  
  *Yehuda Lindell, Benny Pinkas*  
  Journal of Cryptology volume 22, pages161–188 (2009), [eprint](https://eprint.iacr.org/2004/175.pdf), YP09  
  

- Blinder – MPC Based Scalable and Robust Anonymous Committed Broadcast  
  *Ittai Abraham, Benny Pinkas, Avishay Yanai*  
  CCS 20, [eprint](https://eprint.iacr.org/2020/248.pdf), APY20    

## FHE-based MPC

- On-the-Fly Multiparty Computation on the Cloud via Multikey Fully Homomorphic Encryption  
  *Adriana Lopez-Alt, Eran Tromer, Vinod Vaikuntanathan*  
  STOC 2012, [eprint](https://eprint.iacr.org/2013/094.pdf), LTV12    
 

- Constant-Round MPC with Fairness and Guarantee of Output Delivery  
  *S. Dov Gordon, Feng-Hao Liu, Elaine Shi*  
  CRYPTO 2015, [eprint](https://eprint.iacr.org/2015/371.pdf), DLS15    

- Two Round Multiparty Computation via Multi-Key FHE  
  *Pratyay Mukherjee, Daniel Wichs*  
  EUROCRYPT 2016, [eprint](https://eprint.iacr.org/2015/345.pdf), MW16    

- Towards Round-Optimal Secure Multiparty Computations: Multikey FHE without a CRS  
  *Eunkyung Kim, Hyang-Sook Lee, Jeongeun Park*  
  ACISP2018, [eprint](https://eprint.iacr.org/2018/1156.pdf), KLP18    

- Four Round Secure Computation without Setup  
  *Zvika Brakerski, Shai Halevi, Antigoni Polychroniadou*  
  TCC 2017, [eprint](https://eprint.iacr.org/2017/386.pdf), BHP17   

- On the Security of Multikey Homomorphic Encryption  
  *Hyang-Sook Lee, Jeongeun Park*  
  IMA CC 2019, [eprint](https://eprint.iacr.org/2019/1082.pdf), LP19    
  

- Multi-Key Homomophic Encryption from TFHE  
  *Hao Chen, Ilaria Chillotti, Yongsoo Song*  
  ASIACRYPT 2019, [eprint](https://eprint.iacr.org/2019/116.pdf), CCS19    

- Multi-key Fully-Homomorphic Encryption in the Plain Model  
  *Prabhanjan Ananth, Abhishek Jain, ZhengZhong Jin, Giulio Malavolta*  
  TCC 2020, [eprint](https://eprint.iacr.org/2020/180.pdf), AJJM20    
  
- Multiparty Reusable Non-Interactive Secure Computation from LWE  
  *Fabrice Benhamouda, Aayush Jain, Ilan Komargodski, Huijia Lin*  
  EUROCRYPT 2021, [eprint](https://eprint.iacr.org/2021/378.pdf), BJKL21    

- Threshold Cryptosystems From Threshold Fully Homomorphic Encryption  
  *Dan Boneh, Rosario Gennaro, Steven Goldfeder, Aayush Jain, Sam Kim, Peter M. R. Rasmussen, Amit Sahai*  
  CRYPTO 2018, [eprint](https://eprint.iacr.org/2017/956.pdf), BGGJ+18  

- Faster Non-interactive Verifiable Computing  
  *Pascal Lafourcade, Gael Marcadet, Léo Robert*  
  IACR 2021, [eprint](https://eprint.iacr.org/2022/646), LMR21    

- Efficient Threshold FHE with Application to Real-Time Systems  
  *Siddhartha Chowdhury, Sayani Sinha, Animesh Singh, Shubham Mishra, Chandan Chaudhary, Sikhar Patranabis, Pratyay Mukherjee, Ayantika Chatterjee, Debdeep Mukhopadhyay*  
  IACR 2022, [eprint](https://eprint.iacr.org/2022/1625.pdf), CSMC+22    

- Circuit-Private Multi-key FHE  
  *Wutichai Chongchitmate, Rafail Ostrovsky*  
  PKC 2017, [eprint](https://link.springer.com/chapter/10.1007/978-3-662-54388-7_9), CO17  

- Single Secret Leader Election  
  *Dan Boneh, Saba Eskandarian, Lucjan Hanzlik, Nicola Greco*  
  AFT '20, [eprint](https://eprint.iacr.org/2020/025.pdf), BEHG20    
  
- Simple Threshold (Fully Homomorphic) Encryption From LWE With Polynomial Modulus  
  *Katharina Boudgoust, Peter Scholl*  
  IACR 2023, [eprint](https://eprint.iacr.org/2023/016.pdf), BS23    

- Plug-and-play sanitization for TFHE  
  *Florian Bourse, Malika Izabachène*  
  IACR 2022, [eprint](https://eprint.iacr.org/2022/1438.pdf), BI22    

- Round-Optimal Secure Multi-party Computation  
  *Shai Halevi, Carmit Hazay, Antigoni Polychroniadou, Muthuramakrishnan Venkitasubramaniam*  
  CRYPTO 2018, [eprint](https://link.springer.com/article/10.1007/s00145-021-09382-3), HHPV18    
  

- Interaction-Preserving Compilers for Secure Computation  
  *Nico Döttling, Vipul Goyal, Giulio Malavolta, Justin Raizes*  
  ITCS 2022, [eprint](https://eprint.iacr.org/2021/1503.pdf), DGMR22    
  

- RMC-PVC: A Multi-Client Reusable Verifiable Computation Protocol (Long version)  
  *Pascal Lafourcade, Gael Marcadet, Léo Robert*  
  SAC 23, [eprint](https://eprint.iacr.org/2022/1748.pdf), LMR23    
  
- Round-Optimal Secure Multi-party Computation  
  *Shai Halevi, Carmit Hazay, Antigoni Polychroniadou, Muthuramakrishnan Venkitasubramaniam*  
  CRYPTO 2018, [eprint](https://link.springer.com/article/10.1007/s00145-021-09382-3), HHPV18    

- Interaction-Preserving Compilers for Secure Computation  
  *Nico Döttling, Vipul Goyal, Giulio Malavolta, Justin Raizes*  
  ITCS 2022, [eprint](https://eprint.iacr.org/2021/1503.pdf), DGMR22    
  
=======
 ### Function Secret Sharing
- Programmable Distributed Point Functions
  *Elette Boyle, Niv Gilboa, Yuval Ishai, Victor I. Kolobov*
  Crypto 2022, [eprint](https://eprint.iacr.org/2022/1060.pdf), BGIK22

- Lightweight, Maliciously Secure Verifiable Function Secret Sharing
  *Leo de Castro and Antigoni Polychroniadou*
  EuroCrypt 2022, [eprint](https://eprint.iacr.org/2021/580.pdf), CP22

- Lightweight Techniques for Private Heavy Hitters
  *Dan Boneh, Elette Boyle, Henry Corrigan-Gibbs, Niv Gilboa, Yuval Ishai*
  SP 2021, [eprint](https://arxiv.org/abs/2012.14884), BBGG+21

- Function Secret Sharing for PSI-CA : With Applications to Private Contact Tracing
  *Samuel Dittmer, Yuval Ishai, Steve Lu, Rafail Ostrovsky, Mohamed Elsabagh, Nikolaos Kiourtis, Brian Schulte, Angelos Stavrou*
  Unpublished 2021, [eprint](https://eprint.iacr.org/2020/1599), DILO+21

- Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation
  *Elette Boyle, Nishanth Chandran, Niv Gilboa, Divya Gupta, Yuval Ishai, Nishant Kumar, Mayank Rathee*
  EuroCrypt 2021, [eprint](https://eprint.iacr.org/2020/1392), BCGI+21

- Correlated Pseudorandom Functions from Variable-Density LPN
  *Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai, Lisa Kohl, Peter Scholl*
  FOCS 2020, [eprint](https://eprint.iacr.org/2020/1417), BCGI+20

- Secure Computation with Preprocessing via Function Secret Sharing
  *Elette Boyle, Niv Gilboa, Yuval Ishai*
  TCC 2019, [eprint](https://eprint.iacr.org/2019/1095), BGI19

- Efficient Two-Round OT Extension and Silent Non-Interactive Secure Computation
  *Elette Boyle, Geoffroy Couteau, Niv Gilboa, Yuval Ishai, Lisa Kohl, Peter Rindal, Peter Scholl*
  CCS 2019, [eprint](https://eprint.iacr.org/2019/1159), BCGI+19

- Function secret sharing: Improvements and extensions
  *Elette Boyle, Niv Gilboa, Yuval Ishai*
  CCS 2016, [eprint](https://eprint.iacr.org/2018/707), BGI16

- Function Secret Sharing
  *Elette Boyle, Niv Gilboa, Yuval Ishai*
  EuroCrypt 2015, [eprint](https://www.iacr.org/archive/eurocrypt2015/90560300/90560300.pdf), BGI15

- Distributed Point Functions and Their Applications
  *Niv Gilboa, Yuval Ishai*
  EuroCrypt 2014, [eprint](https://www.iacr.org/archive/eurocrypt2014/84410245/84410245.pdf), GI14

