# Homomorphic Encryption (HE)

![](https://badgen.net/badge/:update-to/:June-2023/red) ![](https://badgen.net/badge/:papers/:52/blue) 

HE, especially FHE(Fully Homomorphic Encryption), does matter and are keys for now and future.

- [Homomorphic Encryption (HE)](#homomorphic-encryption-he)
  - [Survey](#survey)
  - [Partial HE(PHE)](#partial-hephe)
  - [FHE](#fhe)
    - [Classical(or Milestones)](#classicalor-milestones)
    - [Multi-key FHE](#multi-key-fhe)
  - [Impl. Efforts](#impl-efforts)
    - [Hardware-based Acceleration](#hardware-based-acceleration)
    - [Open-sourced libs](#open-sourced-libs)
  - [Applications](#applications)
  - [Standard Efforts](#standard-efforts)

## Survey

- Computing Blindfolded,New Developments in Fully Homomorphic Encryption  
  *V. Vinod*  
  FOCS 2011, [paper](https://people.csail.mit.edu/vinodv/FHE/FHE-focs-survey.pdf), Vin11

- Practical homomorphic encryption: A survey  
  *C. Moore, M. O’Neill, E. O’Sullivan, Y. Doroz, and B. Sunar*  
  ISCAS 2014, [paper](https://pure.qub.ac.uk/files/17845072/Practical_Homomorphic_Encryption_Survey_CameraReady.pdf), MOO+14

- A Guide to Fully Homomorphic Encryption  
  *F. Armknecht, C. Boyd, C. Carr, A. Jaschke, and C. A. Reuter*  
  2016, [paper](https://eprint.iacr.org/2015/1192.pdf), ACC+16

- Homomorphic Encryption  
  *H. Shai*  
  2017, [paper](https://shaih.github.io/pubs/he-chapter.pdf), Shai17

- A Survey on Fully Homomorphic Encryption: An Engineering Perspective  
  *P. Martins, L. Sousa, and A. Mariano*  
  ACM Comput. Surv. 2018, [paper](https://eprint.iacr.org/2022/1602.pdf), MSM18

- Fundamentals of Fully Homomorphic Encryption – A survey  
  *Z. Brakerski*  
  [paper](https://eccc.weizmann.ac.il/report/2018/125/download/), Bra18

- A Decade (or So) of Fully Homomorphic Encryption  
  *C. Gentry*  
  presented at the Eurocrypt2021 invited talk, [paper](https://eurocrypt.iacr.org/2021/slides/gentry.pdf)

## Partial HE(PHE)

- ⭐️⭐️⭐️ A method for obtaining digital signatures and public-key cryptosystems  
*R. L. Rivest, A. Shamir, and L. Adleman*  
Communications of the ACM, [paper](https://dl.acm.org/doi/pdf/10.1145/359340.359342), RSA78

- Probabilistic encryption & how to play mental poker keeping secret all partial information  
  *S. Goldwasser and S. Micali*  
  STOC 82, [paper](https://dl.acm.org/doi/10.1145/800070.802212), GM82

- A Public Key Cryptosystem and a Signature Scheme Based on Discrete Logarithms  
  *T. ElGamal*  
  CRYPTO 1984, [paper](https://link.springer.com/chapter/10.1007/3-540-39568-7_2), ElGamal84


- ⭐️⭐️⭐️ A new public-key cryptosystem as secure as factoring,” in Advances in Cryptology  
  *T. Okamoto and S. Uchiyama*  
  EUROCRYPT 1998, [paper](https://link.springer.com/chapter/10.1007/bfb0054135), OU98

- A new public key cryptosystem based on higher residues  
  *D. Naccache and J. Stern*  
  CCS 98, [paper](https://dl.acm.org/doi/10.1145/288090.288106), NS98

- ⭐️⭐️⭐️ Public-Key Cryptosystems Based on Composite Degree Residuosity Classes  
  *P. Paillier*  
  EUROCRYPT 1999, [paper](https://link.springer.com/chapter/10.1007/3-540-48910-X_16), Paillier99

- ⭐️⭐️ Why Textbook ElGamal and RSA Encryption Are Insecure?  
  *D. Boneh, A. Joux, and P. Q. Nguyen*  
  ASIACRYPT 2000, [paper](https://link.springer.com/chapter/10.1007/3-540-44448-3_3), BJN00

- Chosen-Ciphertext Security for Any One-Way Cryptosystem  
  *D. Pointcheval*  
  PKC 2000, [paper](https://link.springer.com/chapter/10.1007/978-3-540-46588-1_10), Poi00

- A Generalisation, a Simplification and Some Applications of Paillier's Probabilistic Public-Key System  
  *Ivan Damgård and Mads Jurik*  
  PKC 2001, [paper](https://link.springer.com/chapter/10.1007/3-540-44586-2_9), DJ01

- Elliptic Curve Paillier Schemes  
  *S. D. Galbraith*  
  J. Cryptology 2002, [paper](https://link.springer.com/article/10.1007/s00145-001-0015-6), Gal02

- Multi-bit Cryptosystems Based on Lattice Problems  
  *A. Kawachi, K. Tanaka, and K. Xagawa*  
  PKC 2007, [paper](https://link.springer.com/chapter/10.1007/978-3-540-71677-8_21),KTX07

- Optimized Paillier’s Cryptosystem with Fast Encryption and Decryption  
  *H. Ma, S. Han, and H. Lei*  
  ACSAC 21, [paper](https://doi.org/10.1145/3485832.3485842), MHL21


## FHE

### Classical(or Milestones)

- A fully homomorphic encryption scheme  
  *Gentry, Craig*  
  Stanford university 2009, [paper](https://www.proquest.com/openview/93369e65682e50979432340f1fdae44e/1?pq-origsite=gscholar&cbl=18750), Gentry09  

- Fully Homomorphic Encryption Using Ideal Lattices  
  *Gentry, Craig*  
  STOC 2009, [paper](https://www.cs.cmu.edu/~odonnell/hits09/gentry-homomorphic-encryption.pdf), Gentry09  

- A simple BGN-type cryptosystem from LWE  
  *Gentry, Craig, Shai Halevi, and Vinod Vaikuntanathan*  
  EUROCRYPT 2010, [paper](https://link.springer.com/chapter/10.1007/978-3-642-13190-5_26), GSV10    

- Fully homomorphic encryption from ring-LWE and security for key dependent messages  
  *Zvika Brakerski, Vinod Vaikuntanathan*  
  CRYPTO 2011, [paper](https://www.iacr.org/archive/crypto2011/68410501/68410501.pdf), BV11  

- (Leveled) fully homomorphic encryption without bootstrapping  
  *Zvika Brakerski, Craig Gentry, Vinod Vaikuntanathan  
  ITCS 2012, [paper](https://eprint.iacr.org/2011/277.pdf), BGV12  

- Fully Homomorphic Encryption without Modulus Switching from Classical GapSVP  
  *Zvika Brakerski*  
  CRYPTO 2012, [paper](https://eprint.iacr.org/2012/078.pdf), Brakerski12  

- Somewhat Practical Fully Homomorphic Encryption  
  *Junfeng Fan, Frederik Vercauteren*  
  eprint 2012, [paper](https://eprint.iacr.org/2012/144.pdf), FV12  

- Packed Ciphertexts in LWE-based Homomorphic Encryption  
  *Zvika Brakerski, Craig Gentry, Shai Halevi*  
  PKC 2013, [paper](https://eprint.iacr.org/2012/565.pdf), BGH13  

- Homomorphic Encryption from Learning with Errors: Conceptually-Simpler, Asymptotically-Faster, Attribute-Based  
  *Craig Gentry, Amit Sahai, Brent Waters*  
  CRYPTO 2013, [paper](https://eprint.iacr.org/2013/340.pdf), GSW13  

- Efficient Fully Homomorphic Encryption from (Standard) LWE  
  *Zvika Brakerski, Vinod Vaikuntanathan*  
  SIAM Journal on computing 2014, [paper](https://eprint.iacr.org/2011/344.pdf), BV14  

- FHEW: Bootstrapping Homomorphic Encryption in less than a second  
  *Léo Ducas, Daniele Micciancio*  
  EUROCRYPT 2015, [paper](https://eprint.iacr.org/2014/816.pdf), DM15  

- Faster Fully Homomorphic Encryption: Bootstrapping in less than 0.1 Seconds  
  *Ilaria Chillotti, Nicolas Gama, Mariya Georgieva, and Malika Izabachène*  
  ASIACRYPT 2016, [paper](https://eprint.iacr.org/2016/870.pdf), CGG+16  

- Homomorphic Encryption for Arithmetic of Approximate Numbers  
  *Jung Hee Cheon, Andrey Kim, Miran Kim, Yongsoo Song*  
  ASIACRYPT 2017, [paper](https://eprint.iacr.org/2016/421.pdf) , CKKS17  

- Threshold Cryptosystems from Threshold Fully Homomorphic Encryption  
  *Dan Boneh, Rosario Gennaro, Steven Goldfeder, Aayush Jain, Sam Kim, Peter M. R. Rasmussen, Amit Sahai*  
  CRYPTO 2018, [paper](https://eprint.iacr.org/2017/956.pdf), BGG+18  

- TFHE: Fast Fully Homomorphic Encryption Over the Torus  
  *Ilaria Chillotti, Nicolas Gama, Mariya Georgieva, Malika Izabachène*  
  Journal of Cryptology 2019, [paper](https://eprint.iacr.org/2018/421.pdf), BGG+2019  

- Bootstrapping fully homomorphic encryption over the integers in less than one second  
  *Hilder Vitor Lima Pereira*  
  PKC 2021, [paper](https://eprint.iacr.org/2020/995.pdf), Pereira21  

- Improved Programmable Bootstrapping with Larger Precision and Efficient Arithmetic Circuits for TFHE  
  *Ilaria Chillotti, Damien Ligier, Jean-Baptiste Orfila, Samuel Tap*  
  ASIACRYPT 2021, [paper](https://eprint.iacr.org/2021/729.pdf), CLO+21  

- Efficient FHEW Bootstrapping with Small Evaluation Keys, and Applications to Threshold Homomorphic Encryption    
  *Yongwoo Lee, Daniele Micciancio, Andrey Kim, Rakyong Choi, Maxim Deryabin, Jieun Eom, Donghoon Yoo*  
  EUROCRYPT 2023, [paper](https://eprint.iacr.org/2022/198.pdf), LMK+23  

### Multi-key FHE

- On-the-Fly Multiparty Computation on the Cloud via Multikey Fully Homomorphic Encryption  
  *Adriana Lopez-Alt, Eran Tromer, Vinod Vaikuntanathan*  
  STOC 2012, [paper](https://eprint.iacr.org/2013/094.pdf), LTV12  

- Multi-Identity and Multi-Key Leveled FHE from Learning with Errors  
  *Michael Clear, Ciarán McGoldrick*  
  CRYPTO 2015, [paper](https://eprint.iacr.org/2014/798.pdf), CM15  

- Lattice-Based Fully Dynamic Multi-key FHE with Short Ciphertexts  
  *Zvika Brakerski, Renen Perlman*  
  CRYPTO 2016, [paper](https://eprint.iacr.org/2016/339.pdf), BP16  

- Multi-Key FHE from LWE, Revisited  
  *Chris Peikert, Sina Shiehian*  
  TCC 2016, [paper](https://eprint.iacr.org/2016/196.pdf), PS16  

- Two Round Multiparty Computation via Multi-Key FHE  
  *Pratyay Mukherjee, Daniel Wichs*  
  EUROCRYPT 2016, [paper](https://eprint.iacr.org/2015/345.pdf), MW16  

- Efficient Multi-Key Homomorphic Encryption with Packed Ciphertexts with Application to Oblivious Neural Network Inference  
  *Hao Chen, Wei Dai, Miran Kim, Yongsoo Song*  
  CCS 2019, [paper](https://eprint.iacr.org/2019/524.pdf), CDKS19  

- Multi-Key Homomophic Encryption from TFHE  
  *Hao Chen, Ilaria Chillotti, Yongsoo Song*  
  ASIACRYPT 2019, [paper](https://eprint.iacr.org/2019/116.pdf), CCS19  

## Impl. Efforts

- Can homomorphic encryption be practical?  
  *M. Naehrig, K. Lauter, and V. Vaikuntanathan*  
  the 3rd ACM workshop on Cloud computing security workshop 2011, [paper](https://eprint.iacr.org/2011/405.pdf), NLV11
- A Comparison of the Homomorphic Encryption Schemes FV and YASHE  
   *T. Lepoint and M. Naehrig*  
   AFRICACRYPT 2014, [paper](https://eprint.iacr.org/2014/062.pdf), LN14
- Building an Efficient Lattice Gadget Toolkit: Subgaussian Sampling and More  
  *N. Genise, D. Micciancio, and Y. Polyakov*  
  EUROCRYPT 2019, [paper](https://eprint.iacr.org/2018/946.pdf), GMP19
- Simple Encrypted Arithmetic Library - SEAL v2.1  
  *Hao Chen, Kim Laine, Rachel Player*  
  FC 2017,[paper](https://eprint.iacr.org/2017/224.pdf), [version 2.3 by Kim Laine](https://www.microsoft.com/en-us/research/uploads/prod/2017/11/sealmanual-2-3-1.pdf), CLP17
- Faster Homomorphic Linear Transformations in HElib  
  *S. Halevi and V. Shoup*  
  CRYPTO 2018, [paper](https://eprint.iacr.org/2018/244), HS18
- OpenFHE: Open-Source Fully Homomorphic Encryption Library  
  *A. A. Badawi et al.*  
  WAHC 2022, [paper](https://eprint.iacr.org/2022/915), BBB+22

### Hardware-based Acceleration

TODO:

### Open-sourced libs

| Name                                                         | Description                                                                                             | Scheme                                                                           | Language    |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------- |
| [Secretflow/HEU](https://github.com/secretflow/heu)          | A high-performance homomorphic encryption algorithm library                                             | Paillier, OU, ElGamal, FHE(in developing)                                                            | C++, python |
| [OpenFHE](https://github.com/openfheorg/openfhe-development) | OpenFHE is an open-source FHE library that includes efficient implementations of all common FHE schemes | - BFV, BGV, CKKS, DM, CGGI, <br/> - Threshold FHE & Proxy Re-Encryption for BFV, BGV, CKKS | C++         |
| [microsoft/SEAL](https://github.com/microsoft/SEAL)          | an easy-to-use open-source homomorphic encryption library                                               | BFV, BGV, CKKS                                                                   | C++, C#     |

See more, https://github.com/jonaschn/awesome-he#libraries

## Applications

Here just list several inspirational and instructive applicaitons.

## Standard Efforts

- PSEC-3: Provably Secure Elliptic Curve Encryption Scheme  
  *T. Okamoto and D. Pointcheval*  
  Submission to IEEE P1363a, 2000, [paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4acdabff9b41622d0ee49ade2d0b4302e3727bf5), OP00; [a note by Rachel Shipsey](https://www.cosic.esat.kuleuven.be/nessie/reports/phase1/rhuwp3-008b.pdf),  

- Homomorphic Encryption Security Standard v1.1  
  *Martin Albrecht and Melissa Chase and Hao Chen and Jintai Ding and Shafi Goldwasser and Sergey Gorbunov and Shai Halevi and Jeffrey Hoffstein and Kim Laine and Kristin Lauter and Satya Lokam and Daniele Micciancio and Dustin Moody and Travis Morrison and Amit Sahai and Vinod Vaikuntanathan*  
  HomomorphicEncryption.org, [paper](http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf), [homepage](https://homomorphicencryption.org/standard/), ACC+18