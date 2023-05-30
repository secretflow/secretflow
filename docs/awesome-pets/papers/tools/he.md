# Homomorphic Encryption (HE)

![](https://badgen.net/badge/:update-to/:June-2023/red) ![](https://badgen.net/badge/:papers/:xx/blue) 

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

### Multi-key FHE

## Impl. Efforts

### Hardware-based Acceleration

### Open-sourced libs

## Applications

Here just list several inspirational and instructive applicaitons.

## Standard Efforts

- PSEC-3: Provably Secure Elliptic Curve Encryption Scheme
  *T. Okamoto and D. Pointcheval*
  Submission to IEEE P1363a, 2000, [paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4acdabff9b41622d0ee49ade2d0b4302e3727bf5), OP00; [a note by Rachel Shipsey](https://www.cosic.esat.kuleuven.be/nessie/reports/phase1/rhuwp3-008b.pdf), 

- Homomorphic Encryption Security Standard v1.1
  *Martin Albrecht and Melissa Chase and Hao Chen and Jintai Ding and Shafi Goldwasser and Sergey Gorbunov and Shai Halevi and Jeffrey Hoffstein and Kim Laine and Kristin Lauter and Satya Lokam and Daniele Micciancio and Dustin Moody and Travis Morrison and Amit Sahai and Vinod Vaikuntanathan*
  HomomorphicEncryption.org, [paper](http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf), [homepage](https://homomorphicencryption.org/standard/), ACC+18