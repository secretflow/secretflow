# Differential Privacy (DP)


![](https://badgen.net/badge/:papers/:22/blue) 


## Table of Contents

- [fundamental-principles](#fundamental-principles)
  * [definition-and-mechanism](#definition-and-mechanism)
  * [sensitivity](#sensitivity)
  * [accountant](#accountant)
  * [local-differential-privacy](#local-differential-privacy)


## Fundamental Principle

### Definition and Mechanism

- *Cynthia Dwork and Aaron Roth,*
  [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
  Foundations and Trends® in Theoretical Computer Science 9.3–4 (2014): 211-407.

- *Cynthia Dwork, Moni Naor, Omer Reingold, Guy N. Rothblum, and Salil P. Vadhan,*
  [On the complexity of differentially private data release: efficient algorithms and hardness results](https://dl.acm.org/doi/10.1145/1536414.1536467)
  Proceedings of the forty-first annual ACM symposium on Theory of computing. 2009.

- *Stanley L. Warner,* 
   [Randomized response: a survey technique for eliminating evasive answer bias](https://www.tandfonline.com/doi/abs/10.1080/01621459.1965.10480775)
   Journal of the American Statistical Association 60.309 (1965): 63-69.  


### Sensitivity
- *Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith, *
  [Smooth Sensitivity and Sampling in Private Data Analysis.](https://cs-people.bu.edu/ads22/pubs/NRS07/NRS07-full-draft-v1.pdf)
  Roceedings of the thirty-ninth annual ACM symposium on Theory of computing. 2007: 75-84
  
- Nissim, Kobbi, Sofya Raskhodnikova, and Adam Smith, 
  [Smooth sensitivity and sampling in private data analysis.](https://cs-people.bu.edu/ads22/pubs/NRS07/NRS07-full-draft-v1.pdf)
  Proceedings of the thirty-ninth annual ACM symposium on Theory of computing. 2007.

- Bun, Mark, Thomas Steinke, and Jonathan Ullman, 
  [Make up your mind: The price of online queries in differential privacy.](https://arxiv.org/pdf/1604.04618.pdf)
  Proceedings of the twenty-eighth annual ACM-SIAM symposium on discrete algorithms. Society for Industrial and Applied Mathematics, 2017.
  
- Feldman, Vitaly, and Thomas Steinke, 
  [Generalization for adaptively-chosen estimators via stable median](http://proceedings.mlr.press/v65/feldman17a/feldman17a.pdf) 
  Conference on learning theory. PMLR, 2017.
  
### Accountant

- Mironov, Ilya, 
  [Rényi differential privacy.](https://arxiv.org/pdf/1702.07476.pdf) 
  2017 IEEE 30th computer security foundations symposium (CSF). IEEE, 2017.

- Dwork, Cynthia, and Guy N. Rothblum, 
  [Concentrated differential privacy.](https://arxiv.org/pdf/1603.01887.pdf)
  
- Bun, Mark, and Thomas Steinke, 
  [Concentrated differential privacy: Simplifications, extensions, and lower bounds.](https://arxiv.org/pdf/1605.02065.pdf)
  Theory of Cryptography: 14th International Conference, TCC 2016-B, Beijing, China, October 31-November 3, 2016, Proceedings, Part I. Berlin, Heidelberg: Springer Berlin Heidelberg, 2016.

- Bun, Mark, et al, 
  [Composable and versatile privacy via truncated cdp.](https://projects.iq.harvard.edu/files/privacytools/files/bun_mark_composable_.pdf) 
  Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing. 2018.


### Local Differential Privacy
  
- Chen, Rui, et al. 
  [Differentially private transit data publication: a case study on the montreal transportation system.](https://dl.acm.org/doi/10.1145/2339530.2339564) 
  Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. 2012.
  
- S. P. Kasiviswanathan, H. K. Lee, K. Nissim, S. Raskhodnikova, and A. Smith, 
  [What can we learn privately?](https://arxiv.org/pdf/0803.0924.pdf),
  SIAM Journal on Computing, vol. 40, no. 3, pp. 793–826, 2011.

- B. Avent, A. Korolova, D. Zeber, T. Hovden, and B. Livshits,
  [BLENDER: Enabling local search with a hybrid differential privacy model](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/avent),
  USENIX Security Symposium, 2017, pp. 747–764

- C. Dwork, K. Kenthapadi, F. McSherry, I. Mironov, and M. Naor,
  [Our data, ourselves: Privacy via distributed noise generation](https://link.springer.com/chapter/10.1007/11761679_29),
  in Theory and Applications of Cryptographic Techniques, 2006, pp. 486–503

- M. S. Alvim, K. Chatzikokolakis, C. Palamidessi, and A. Pazii,
  [Metric-based local differential privacy for statistical applications](https://arxiv.org/pdf/1805.01456.pdf),

- K. Chatzikokolakis, M. E. Andrés, N. E. Bordenabe, and C. Palamidessi, 
  [Broadening the scope of differential privacy using metrics](https://link.springer.com/chapter/10.1007/978-3-642-39077-7_5), 
  in International Symposium on Privacy Enhancing Technologies Symposium, 2013, pp. 82–102

- M. E. Gursoy, A. Tamersoy, S. Truex, W. Wei, and L. Liu, 
  [Secure and utility-aware data collection with condensed local differential privacy](https://arxiv.org/pdf/1905.06361.pdf),
  IEEE Trans. on Dependable and Secure Comput., pp. 1–13, 2019

- Y. NIE, W. Yang, L. Huang, X. Xie, Z. Zhao, and S. Wang, 
 [A utility-optimized framework for personalized private histogram estimation](https://ieeexplore.ieee.org/document/8368271),
  IEEE Trans. Knowl. Data Eng., vol. 31, no. 4, pp. 655–669, 2019.

- T. Murakami and Y. Kawamoto, 
  [Utility-optimized local differential privacy mechanisms for distribution estimation](https://arxiv.org/pdf/1807.11317.pdf),
  in USENIX Security Symposium, 2019, pp. 1877–1894.

- X. Gu, M. Li, L. Xiong, and Y. Cao, 
  [Providing inputdiscriminative protection for local differential privacy](https://arxiv.org/pdf/1911.01402.pdf),
  in Proc.IEEE ICDE, 2020, pp. 505–516

- S. Takagi, Y. Cao, and M. Yoshikawa, 
  [POSTER: Data collection via local differential privacy with secret parameters](https://dl.acm.org/doi/10.1145/3320269.3405441),
  in Proc.ACM Asia CCS, 2020, p. 910–912

- Erlingsson Ú, Pihur V, Korolova A.
  [Rappor: Randomized aggregatable privacy-preserving ordinal response](https://dl.acm.org/doi/abs/10.1145/2660267.2660348)
  in Proc.ACM SIGSAC CCS, 2014, pp. 1054-1067.

- Wang T, Blocki J, Li N, et al.
  [Locally differentially private protocols for frequency estimation](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao)
  in USENIX Security Symposium, 2017, pp. 729-745.

- Fanti G, Pihur V, Erlingsson Ú
  [Building a RAPPOR with the unknown: Privacy-preserving learning of associations and data dictionaries](https://arxiv.org/abs/1503.01214)
  in arXiv preprint arXiv:1503.01214, 2015.

- Ren X, Yu C M, Yu W, et al. 
  [High-dimensional crowdsourced data distribution estimation with local privacy](https://ieeexplore.ieee.org/abstract/document/7876342),
  in Proc.IEEE International Conference on Computer and Information Technology, 2016, pp. 226-233.

- Zhang Z, Wang T, Li N, et al.
  [Calm: Consistent adaptive local marginal for marginal release under local differential privacy](https://dl.acm.org/doi/abs/10.1145/3243734.3243742),
  in Proc. ACM SIGSAC CCS, 2018, pp. 212-229

- Cormode G, Kulkarni T, Srivastava D.
  [Answering range queries under local differential privacy](https://dl.acm.org/doi/abs/10.14778/3339490.3339496)
  in VLDB Endowment, 2019, pp. 1126-1138

- Du L, Zhang Z, Bai S, et al.
  [AHEAD: adaptive hierarchical decomposition for range query under local differential privacy](https://dl.acm.org/doi/abs/10.1145/3460120.3485668)
  in Proc.ACM SIGSAC CCS, 2021, pp. 1266-1288

- Wang T, Ding B, Zhou J, et al.
  [Answering multi-dimensional analytical queries under local differential privacy](https://dl.acm.org/doi/abs/10.1145/3299869.3319891)
  in International Conference on Management of Data, 2019, pp. 159-176

- Yang J, Wang T, Li N, et al.
  [Answering multi-dimensional range queries under local differential privacy](https://arxiv.org/abs/2009.06538)
  in arXiv preprint arXiv:2009.06538, 2020

- Wang N, Xiao X, Yang Y, et al.
  [Collecting and analyzing multidimensional data with local differential privacy](https://ieeexplore.ieee.org/abstract/document/8731512)
  in International Conference on Data Engineering, 2019, pp. 638-649

- Ren X, Yu C M, Yu W, et al.
  [LoPub: high-dimensional crowdsourced data publication with local differential privacy](https://ieeexplore.ieee.org/abstract/document/8306916)
  in Proc.IEEE Transactions on Information Forensics and Security, 2018, pp. 2151-2166