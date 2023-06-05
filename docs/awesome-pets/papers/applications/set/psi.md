# Private Set Intersection (PSI)

![](https://badgen.net/badge/:update-to/:Mar-2023/red) ![](https://badgen.net/badge/:papers/:23/blue) 

Private set intersection (PSI) is a special case of multiparty computation, in which each party has a set of items and the goal is to learn the intersection of those sets while revealing nothing else about those sets.

Note: one paper may be included in several categories (e.g. a paper may introduce a new protocol for both OT and VOLE, we decide to include it in both categories).

## Table of Contents

- [PKC-based PSI](#pkc-based-psi)
- [OT-based PSI](#ot-based-psi)
- [VOLE-based PSI](#vole-based-psi)
- [Other Variants](#other-variants)

## PKC-based PSI

- Improved Private Set Intersection for Sets with Small Entries  
  *S. Dov Gordon, Carmit Hazay, Phi Hung Le*  
  PKC 2023, [eprint](https://eprint.iacr.org/2022/334)
  
- Compact and Malicious Private Set Intersection for Small Sets  
  *Mike Rosulek, Ni Trieu*  
  CCS 2021, [eprint](https://eprint.iacr.org/2021/1159), RT21
  
- Private Matching for Compute  
  *Prasad Buddhavarapu, Andrew Knox, Payman Mohassel, Shubho Sengupta, Erik Taubeneck, Vlad Vlaskin*  
  Unpublished 2020, [eprint](https://eprint.iacr.org/2020/599)

- Scalable multi-party private set-intersection  
  *Carmit Hazay, Muthuramakrishnan Venkitasubramaniam*  
  PKC 2017, [eprint](https://eprint.iacr.org/2017/027)
  
- Linear-Complexity Private Set Intersection Protocols Secure in Malicious Model  
  *Emiliano De Cristofaro, Jihye Kim, Gene Tsudik*  
  AsiaCrypt 2010, [eprint](https://eprint.iacr.org/2010/469), CKT10

- Practical Private Set Intersection Protocols with Linear Computational and Bandwidth Complexity  
  *Emiliano De Cristofaro, Gene Tsudik*  
  Unpublished 2010, [eprint](https://eprint.iacr.org/2009/491), CT10

- Information Sharing Across Private Databases  
  *Rakesh Agrawal, Alexandre V. Evfimievski, Ramakrishnan Srikant*  
  SIGMOD 2003, [eprint](https://www.cs.cornell.edu/aevf/research/SIGMOD_2003.pdf), AES03
  
## OT-based PSI

- Circuit-PSI with Linear Complexity via Relaxed Batch OPPRF  
  *Nishanth Chandran, Divya Gupta, Akash Shah*  
  PETS 2022, [eprint](https://eprint.iacr.org/2021/034), CGS22
  
- Simple, Fast Malicious Multiparty Private Set Intersection  
  *Ofri Nevo, Ni Trieu, Avishay Yanai*  
  CCS 2021, [eprint](https://eprint.iacr.org/2021/1221), NTY21
  
- Private Set Operations from Oblivious Switching  
  *Gayathri Garimella, Payman Mohassel, Mike Rosulek, Saeed Sadeghian, Jaspal Singh*  
  PKC 2021, [eprint](https://eprint.iacr.org/2021/243), GMRS21
  
- PSI from PaXoS: Fast, Malicious Private Set Intersection  
  *Benny Pinkas, Mike Rosulek, Ni Trieu, Avishay Yanai*  
  EuroCrypt 2020, [eprint](https://eprint.iacr.org/2020/193), PaXoS
  
- Private Set Intersection in the Internet Setting From Lightweight Oblivious PRF  
  *Melissa Chase, Peihan Miao*  
  Crypto 2020, [eprint](https://eprint.iacr.org/2020/729), CM20

- SpOT-Light: Lightweight Private Set Intersection from Sparse OT Extension, 2019,   
  *Benny Pinkas, Mike Rosulek, Ni Trieu, Avishay Yanai*  
  Crypto 2019, [eprint](https://eprint.iacr.org/2019/634), PRTY19
  
- Malicious-Secure Private Set Intersection via Dual Execution  
  *Peter Rindal, Mike Rosulek*  
  CCS 2017, [eprint](https://eprint.iacr.org/2017/769), RR17b 

- Improved Private Set Intersection Against Malicious Adversaries  
  *Peter Rindal, Mike Rosulek*  
  EuroCrypt 2017, [eprint](https://eprint.iacr.org/2016/746), RR17a

- Efficient Batched Oblivious PRF with Applications to Private Set Intersection  
  *Vladimir Kolesnikov, Ranjit Kumaresan, Mike Rosulek, Ni Trieu*  
  CCS 2016, [eprint](https://eprint.iacr.org/2016/799), KKRT16

- Phasing: Private Set Intersection using Permutation-based Hashing  
  *Benny Pinkas, Thomas Schneider, Gil Segev, Michael Zohner*  
  Usenix Security 2015, [eprint](https://eprint.iacr.org/2015/634), PSSZ15
  
- Private Set Intersection: Are Garbled Circuits Better than Custom Protocols?
  *Yan Huang, David Evans, Jonathan Katz*  
  NDSS 2012, [eprint](https://www.cs.umd.edu/~jkatz/papers/psi.pdf), HEK12
  
## VOLE-based PSI

- Blazing Fast PSI from Improved OKVS and Subfield VOLE  
  *Peter Rindal, Srinivasan Raghuraman*  
  CCS 2022, [eprint](https://eprint.iacr.org/2022/320)

- Fully Secure PSI via MPC-in-the-Head  
  *S. Dov Gordon, Carmit Hazay, Phi Hung Le*  
  Pets 2022, [eprint](https://eprint.iacr.org/2022/379)

- PSI from Ring-OLE  
  *Wutichai Chongchitmate, Yuval Ishai, Steve Lu, Rafail Ostrovsky*  
  CCS 2022, [eprint](https://dl.acm.org/doi/abs/10.1145/3548606.3559378)
  
- Oblivious Key-Value Stores and Amplification for Private Set Intersection  
  *Gayathri Garimella, Benny Pinkas, Mike Rosulek, Ni Trieu, Avishay Yanai*  
  Crypto 2021, [eprint](https://eprint.iacr.org/2021/883), GPRT+21

- VOLE-PSI: Fast OPRF and Circuit-PSI from Vector-OLE  
  *Peter Rindal, Phillipp Schoppmann*  
  EuroCrypt 2021, [eprint](https://eprint.iacr.org/2021/266), RS21

## Other Variants

- Labeled PSI from Homomorphic Encryption with Reduced Computation and Communication  
  *Kelong Cong, Radames Cruz Moreno, Mariana Botelho da Gama, Wei Dai, Ilia Iliashenko, Kim Laine, Michael Rosenberg*  
  CCS 2021, [eprint](https://eprint.iacr.org/2021/1116), CMBD+21

- PIR-PSI: Scaling Private Contact Discovery  
  *Daniel Demmler, Peter Rindal, Mike Rosulek, Ni Trieu*  
  PETS 2018, [eprint](https://eprint.iacr.org/2018/579), DRRT18




