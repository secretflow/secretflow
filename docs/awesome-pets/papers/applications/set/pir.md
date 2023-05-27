# Private Information Retrieval (PIR)

![](https://badgen.net/badge/:update-to/:Mar-2023/red) ![](https://badgen.net/badge/:papers/:33/blue) 

Private Information Retrival is a protocol that allows a client to retrieve an element of a database without the owner of that database being able to determine which element was selected. Note that the initial motivation of PIR is to let users to privately download a website from the public internet, hence the database in PIR is assumed to be **public** and **huge**, this should be the explanation why early researchers are trying to find a asymptotically better (sublinear) solution in stead of a concretely better one.


## Table of Contents

- [Single-Server PIR](#single-server-pir)
- [Multi-Server PIR](#multi-server-pir)
  
  
## Single-Server PIR

- SPG: Structure-Private Graph Database via SqueezePIR  
  *Ling Liang, Jilan Lin, Zheng Qu, Ishtiyaque Ahmad, Fengbin Tu, Trinabh Gupta, Yufei Ding, Yuan Xie*
  IEEE/ACM VLDB 2023, [eprint](https://www.vldb.org/pvldb/vol16/p1615-liang.pdf),
  
- One Server for the Price of Two: Simple and Fast Single-Server Private Information Retrieval  
  *Alexandra Henzinger, Matthew M. Hong, Henry Corrigan-Gibbs, Sarah Meiklejohn, Vinod Vaikuntanathan*  
  Usenix Security 2023, [eprint](https://eprint.iacr.org/2022/949), HHCM+23
  
- Limits of Preprocessing for Single-Server PIR  
  *Giuseppe Persiano, Kevin Yeo*  
  SIAM 2022, [eprint](https://eprint.iacr.org/2022/235), PY22

- Constant-weight PIR: Single-round Keyword PIR via Constant-weight Equality Operators  
  *Rasoul Akhavan Mahdavi, Florian Kerschbaum*  
  Usenix Security 2022, [eprint](https://www.usenix.org/conference/usenixsecurity22/presentation/mahdavi), MK22

- Single-Server Private Information Retrieval with Sublinear Amortized Time  
  *Henry Corrigan-Gibbs, Alexandra Henzinger, Dmitry Kogan*  
  EuroCrypt 2022, [eprint](https://eprint.iacr.org/2022/081), CHK22

- OnionPIR: Response Efficient Single-Server PIR  
  *Muhammad Haris Mughees, Hao Chen, Ling Ren*  
  CCS 2021, [eprint](https://eprint.iacr.org/2021/1081), MCR21

- Incremental Offline/Online PIR  
  *Yiping Ma, Ke Zhong, Tal Rabin, Sebastian Angel*  
  Unpublished, [eprint](https://eprint.iacr.org/2021/1438), MZRA

- Random-Index PIR and Applications  
  *Craig Gentry, Shai Halevi, Bernardo Magri, Jesper Buus Nielsen, Sophia Yakoubov*  
  TCC 2021, [eprint](https://eprint.iacr.org/2020/1248), GHMN+21

- On the Privacy of a Code-based Single-Server Computational PIR Scheme  
  *Sarah Bordage, Julien Lavauzelle*  
  Cryptogr. Commun. 2021, [eprint](https://eprint.iacr.org/2020/376), BL21

- Communication–Computation Trade-offs in PIR  
  *Asra Ali, Tancrède Lepoint, Sarvar Patel, Mariana Raykova, Phillipp Schoppmann, Karn Seth, Kevin Yeo*  
  Usenix Security 2021, [eprint](https://eprint.iacr.org/2019/1483), ALPR+21
 
- Private Anonymous Data Access  
  *Ariel Hamlin, Rafail Ostrovsky, Mor Weiss, Daniel Wichs*  
  EuroCrypt 2019, [eprint](https://eprint.iacr.org/2018/363), HOWW19

- Compressible FHE with Applications to PIR  
  *Craig Gentry, Shai Halevi*  
  TCC 2019, [eprint](https://eprint.iacr.org/2019/733), GH19

- Private Stateful Information Retrieval  
  *Sarvar Patel, Giuseppe Persiano, Kevin Yeo*  
  CCS 2018, [eprint](https://eprint.iacr.org/2018/1083.pdf), PPY18

- PIR with Compressed Queries and Amortized Query Processing  
  *Sebastian Angel, Hao Chen, Kim Laine, Srinath Setty*  
  SP 2018, [eprint](https://eprint.iacr.org/2017/1142.pdf), ACLS18

- XPIR : Private Information Retrieval for Everyone  
  *Carlos Aguilar-Melchor, Joris Barrier, Laurent Fousse, Marc-Olivier Killijian*  
  PETs 2106, [eprint](https://eprint.iacr.org/2014/1025.pdf), ABFK16

- Optimal Rate Private Information Retrieval from Homomorphic Encryption  
  *Aggelos Kiayias, Nikos Leonardos, Helger Lipmaa, Kateryna Pavlyk, Qiang Tang*  
  PETs 2015, [eprint](https://petsymposium.org/2015/papers/23_Kiayias.pdf), KLLP+15

- First CPIR Protocol with Data-Dependent Computation  
  *Helger Lipmaa*  
  ICISC 2012, [eprint](https://eprint.iacr.org/2009/395), Lip12

- On the Practicality of Private Information Retrieval  
  *Radu Sion, Bogdan Carbunar*  
  NDSS 2007, [eprint](https://www.ndss-symposium.org/ndss2007/practicality-private-information-retrieval/), SC07 

- Single-Database Private Information Retrieval with Constant Communication Rate  
  *Craig Gentry, Zulfikar Ramzan*  
  ICALP 2005, [eprint](https://link.springer.com/chapter/10.1007/11523468_65), GR05

- Single Database Private Information Retrieval with Logarithmic Communication  
  *Yan-Cheng Chang*  
  ISP 2004, [eprint](https://eprint.iacr.org/2004/036), Chang04 

- One-Way Trapdoor Permutations Are Sufficient for Non-trivial Single-Server Private Information Retrieval  
  *Eyal Kushilevitz, Rafail Ostrovsky*  
  EuroCrypt 2000, [eprint](https://www.iacr.org/archive/eurocrypt2000/1807/18070104-new.pdf), KO00

- Single Database Private Information Retrieval Implies Oblivious Transfer  
  *Giovanni Di Crescenzo, Tal Malkin, Rafail Ostrovsky*  
  EuroCrypt 2000, [eprint](https://www.iacr.org/archive/eurocrypt2000/1807/18070122-new.pdf), CMO00

- Computationally Private Information Retrieval with Polylogarithmic Communication  
  *Christian Cachin, Silvio Micali. Markus Stadler*  
  EuroCrypt 1999, [eprint](https://people.csail.mit.edu/silvio/Selected%20Scientific%20Papers/Private%20Information%20Retrieval/Computationally%20Private%20Information%20Retrieval%20with%20Polylogarithmic%20Communication.pdf), CMS99

- Replication is NOT Needed: SINGLE Database, Computationally-Private Information Retrieval  
  *Eyal Kushilevitz, Rafail Ostrovsky*  
  FOCS 1997, [eprint](https://doi.org/10.1109/SFCS.1997.646125), KO97


## Multi-Server PIR

- Private Information Retrieval with Sublinear Online Time  
  *Henry Corrigan-Gibbs, Dmitry Kogan*  
  EuroCrypt 2020, [eprint](https://eprint.iacr.org/2019/1075), CK20

- Private Information Retrieval is Graph Based Replication Systems  
  *Netanel Raviv, Itzhak Tamo*  
  ISIT 2018, [eprint](https://ieeexplore.ieee.org/document/8437311), RT18

- Towards Doubly Efficient Private Information Retrieval  
  *Ran Canetti, Justin Holmgren, Silas Richelson*  
  TCC 2017, [eprint](https://eprint.iacr.org/2017/568), CHR17

- Can We Access a Database Both Locally and Privately?  
  *Elette Boyle, Yuval Ishai, Rafael Pass, Mary Wootters*  
  TCC 2017, [eprint](https://eprint.iacr.org/2017/567.pdf), BIPW17

- Polynomial Batch Codes for Efficient IT-PIR  
  *Ryan Henry*  
  PETs 2016, [eprint](https://petsymposium.org/2016/files/papers/Polynomial_Batch_Codes_for_Efficient_IT-PIR.pdf), Ryan 2016

- 2-Server PIR with Subpolynomial Communication  
  *Zeev Dvir, Sivakakanth Gopi*  
  J. ACM 2015, [eprint](https://arxiv.org/abs/1407.6692), DG15

- Sublinear Scaling for Multi-Client Private Information Retrieval  
  *Wouter Lueks, Ian Goldberg*  
  FCDS 2015, [eprint](https://link.springer.com/chapter/10.1007/978-3-662-47854-7_10), LG15

- Practical PIR for Electronic Commerce  
  *Ryan Henry, Femi Olumofin, Ian Goldberg*  
  CCS 2011, [eprint](https://cacr.uwaterloo.ca/techreports/2011/cacr2011-04.pdf), HOG11 

- Reducing the Servers Computation in Private Information Retrieval: PIR with Preprocessing  
  *Amos Beimel, Yuval Ishai, Tal Malkin*  
  Crypto 2000, [eprint](https://www.iacr.org/archive/crypto2000/18800056/18800056.pdf), BIM00

- Private Information Retrieval  
  *Benny Chor, Oded Goldreich, Eyal Kushilevitz, Madhu Sudan*  
  FOCS 1995, [eprint](https://www.cs.umd.edu/~gasarch/TOPICS/pir/first.pdf), CGKS95






